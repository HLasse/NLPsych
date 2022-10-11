from collections.abc import Callable
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.utils import compute_class_weight
from wasabi import msg

from clinicalspeech.audio.baselines.baseline_model import BaselineClassifier
from clinicalspeech.audio.baselines.evaluate_baseline import evaluate_model
from clinicalspeech.audio.baselines.utils import create_dataloaders
from clinicalspeech.utils import PROJECT_ROOT


def create_trainer(config: DictConfig, run_name: str) -> pl.Trainer:
    """Create pytorch lightning trainer.

    Args:
        config (DictConfig): Hydra config
        run_name (str): Name of the run

    Returns:
        pl.Trainer: Pytorch lightning trainer
    """
    wandb_cb = WandbLogger(name=config.run_name)
    callbacks = []
    if config.audio.save_models:
        callbacks.append(
            ModelCheckpoint(
                dirpath=PROJECT_ROOT / "audio_models" / run_name,
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                every_n_epochs=1,
            ),
        )
    if config.patience:
        callbacks.append(EarlyStopping("val_loss", patience=config.audio.patience))

    trainer = pl.Trainer(
        logger=wandb_cb,
        **config.audio.pl_trainer_kwargs,
    )
    return trainer


def get_class_weights(
    config: DictConfig, train_set: pd.DataFrame
) -> Union[None, np.array]:
    """Get class weights for training if specified in config

    Args:
        config (DictConfig): Hydra config
        train_set (pd.DataFrame): Training set

    Returns:
        Union[None, np.array]: Class weights
    """
    if config.use_class_weights:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # compute weights to avoid overfitting on majority class
        weights = torch.tensor(
            compute_class_weight(
                "balanced",
                classes=list(range(len(train_set["label_id"].unique()))),
                y=train_set["label_id"].tolist(),
            ),
            dtype=torch.float,
        ).to(device)
    else:
        weights = None
    return weights


def find_learning_rate(
    config: DictConfig,
    run: wandb.run,
    run_config: dict,
    model: pl.LightningModule,
    trainer: pl.Trainer,
):
    """Find optimal learning rate for model using lr finder. Only run if specified
    in config

    Args:
        config (DictConfig): Hydra config
        run (wandb.run): wandb run
        run_config (dict): wandb run config
        model (pl.LightningModule): Model to train
        trainer (pl.Trainer): Trainer to train model
    """
    if config.audio.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(model)
        run_config.update(
            {"learning_rate": lr_finder.suggestion()}, allow_val_change=True
        )
        fig = lr_finder.plot(suggest=True)
        run.log({"lr_finder.plot": fig})
        run.log({"found_lr": lr_finder.suggestion()})


def fit_model(
    config: DictConfig,
    augment_fn: Callable,
    embedding_fn: Callable,
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    feat_set: str,
    run_name: str,
) -> pl.LightningModule:
    """Fit model to training data and save to wandb and disk. Optionally augmemnt
    training data.

    Args:
        config (DictConfig): Hydra config
        augment_fn (Callable): Augmentation function
        embedding_fn (Callable): Embedding function
        train_set (pd.DataFrame): Training set
        val_set (pd.DataFrame): Validation set
        feat_set (str): Feature set
        run_name (str): Name of the run

    Returns:
        pl.LightningModule: Trained model
    """
    run = wandb.init(
        config=config,
        project=config.wandb_project_name,
        dir=PROJECT_ROOT,
        allow_val_change=True,
        reinit=True,
    )

    run_config = run.config
    run.name = run_name
    run_config.run_name = run.name
    run.log({"feat_set": feat_set})

    # Create dataloaders, model, and trainer
    train_loader, val_loader = create_dataloaders(
        train_filepaths=train_set[config.data.audio_path_column_name].tolist(),
        train_labels=train_set["label_id"].tolist(),
        val_filepaths=val_set[config.data.audio_path_column_name].tolist(),
        val_labels=val_set["label_id"].tolist(),
        batch_size=config.audio.batch_size,
        num_workers=config.audio.num_workers,
        embedding_fn=embedding_fn,
        augment_fn=augment_fn,
    )

    weights = get_class_weights(config, train_set)

    model = BaselineClassifier(
        num_classes=len(train_set["label_id"].unique()),
        feature_set=feat_set,
        learning_rate=config.learning_rate,
        train_loader=train_loader,
        val_loader=val_loader,
        weights=weights,
    )
    trainer = create_trainer(config=config, run_name=run_name)

    # find optimal learning if specified in config
    find_learning_rate(config, run, run_config, model, trainer)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Finish tracking run on wandb to start the next one
    run.finish()
    return model


def prepare_binary_data(
    config: DictConfig,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    origin: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for binary classification. Subsets to only include data from
    the specified origin and maps controls to 0 and cases to 1.

    Args:
        config (DictConfig): Hydra config
        train (pd.DataFrame): Training set
        val (pd.DataFrame): Validation set
        test (pd.DataFrame): Test set
        origin (str): Origin of data

    Returns
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation sets
    """

    train_set = train[train[config.data.origin_column_name] == origin]
    val_set = val[val[config.data.origin_column_name] == origin]
    test_set = test[test[config.data.origin_column_name] == origin]

    mapping = {config.data.control_label_name: 0, origin: 1}
    train_set["label_id"] = train_set[config.data.label_column_name].replace(mapping)
    val_set["label_id"] = val_set[config.data.label_column_name].replace(mapping)
    test_set["label_id"] = test_set[config.data.label_column_name].replace(mapping)
    return train_set, val_set, test_set


def train_and_evaluate_models(
    config: DictConfig,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    augment_fn: Callable,
    embedding_fn_dict: dict[str, Callable],
    origin: str,
    id2label: dict,
    is_baseline: bool,
):
    """Trains and evaluates baseline audio models for each feature set and origin.
    Origin is used to subset the data to only include data from that origin and map
    controls to 0 and cases to 1. Results are saved to wandb and disk.

    Args:
        config (DictConfig): Hydra config
        train (pd.DataFrame): Training set
        val (pd.DataFrame): Validation set
        test (pd.DataFrame): Test set. Only used if "test" is included in config.eval_splits
        augment_fn (Callable): Augmentation function for training set
        embedding_fn_dict (dict[str, Callable]): Dictionary of embedding functions
        origin (str): Origin of data, e.g. ASD, DEPR, or multiclass
        id2label (dict): Dictionary mapping label ids to label names
        is_baseline (bool): Whether to train baseline models or not
    """
    # Loop over each feature set
    for feat_set in embedding_fn_dict.keys():
        if feat_set in ["windowed_mfccs"]:
            continue
        embedding_fn = embedding_fn_dict[feat_set]()
        msg.info(f"Starting {feat_set}...")
        # train model
        model = fit_model(
            config=config,
            augment_fn=augment_fn,
            embedding_fn=embedding_fn,
            train_set=train,
            val_set=val,
            feat_set=feat_set,
            run_name=f"{origin}{feat_set}",
        )
        msg.info("Evaluating on splits...")
        # evaluate on splits and save results to disk
        evaluate_model(
            df=pd.concat([train, val, test]),
            model=model,
            model_name=feat_set,
            splits_to_evaluate=config.eval_splits,
            embedding_fn=embedding_fn,
            id2label=id2label,
            num_labels=train[config.data.label_column_name].nunique(),
            target_class=origin,
            is_baseline=is_baseline,
            config=config,
        )


def train_binary_audio_baselines(
    config: DictConfig,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    augment_fn: Callable,
    embedding_fn_dict: dict[str, Callable],
):
    """Trains baseline audio models for each feature set and origin. Origin is used
    to subset the data to only include data from that origin and map controls to 0
    and cases to 1. The models are evaluated on the splits defined in config.eval_splits
    and saved to disk.

    Args:
        config (DictConfig): Hydra config
        train (pd.DataFrame): Training set
        val (pd.DataFrame): Validation set
        test (pd.DataFrame): Test set. Only used if "test" is included in config.eval_splits
        augment_fn (Callable): Augmentation function for training set
        embedding_fn_dict (dict[str, Callable]): Dictionary of embedding functions
    """
    for origin in train[config.data.origin_column_name].unique():
        # Prepare data, subset and make label mapping
        msg.divider(f"Training {origin}")

        train_set, val_set, test_set = prepare_binary_data(
            config=config, train=train, val=val, test=test, origin=origin
        )
        msg.info(f"Training on {len(train_set)} samples")
        msg.info(f"Evaluating on {len(val_set)} samples")
        train_and_evaluate_models(
            config=config,
            train=train_set,
            val=val_set,
            test=test_set,
            augment_fn=augment_fn,
            embedding_fn_dict=embedding_fn_dict,
            origin=origin,
            id2label={config.data.control_label_name: 0, origin: 1},
            is_baseline=True,
        )
        msg.info("Done!")


def train_multiclass_audio_baselines(
    config: DictConfig,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    augment_fn: Callable,
    embedding_fn_dict: dict[str, Callable],
):
    """Train multiclass baseline audio models for each feature set. The models are
    evaluated on the splits defined in config.eval_splits and saved to disk.

    Args:
        config (DictConfig): Hydra config
        train (pd.DataFrame): Training set
        val (pd.DataFrame): Validation set
        test (pd.DataFrame): Test set. Only used if "test" is included in config.eval_splits
        augment_fn (Callable): Augmentation function for training set
        embedding_fn_dict (dict[str, Callable]): Dictionary of embedding functions
    """
    train["label_id"] = train[config.data.label_column_name].replace(
        config.data.multiclass_label2id_mapping
    )
    val["label_id"] = val[config.data.label_column_name].replace(
        config.data.multiclass_label2id_mapping
    )

    msg.divider("Training multiclass models")
    train_and_evaluate_models(
        config=config,
        train=train,
        val=val,
        test=test,
        augment_fn=augment_fn,
        embedding_fn_dict=embedding_fn_dict,
        origin="multiclass",
        id2label=config.data.multiclass_id2label_mapping,
        is_baseline=True,
    )
