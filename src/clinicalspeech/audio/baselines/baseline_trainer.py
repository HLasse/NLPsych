from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch_audiomentations
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from wasabi import msg

from clinicalspeech.audio.baselines.baseline_model import BaselineClassifier
from clinicalspeech.audio.baselines.embedding_fns import get_embedding_fns
from clinicalspeech.audio.dataloders.torch_dataloader import AudioDataset
from clinicalspeech.utils import PROJECT_ROOT

# pylint: disable=redefined-outer-name


def create_dataloaders(
    train_filepaths: Union[pd.Series, list],
    train_labels: Union[pd.Series, list],
    val_filepaths: Union[pd.Series, list],
    val_labels: Union[pd.Series, list],
    batch_size: int,
    num_workers: int,
    embedding_fn: Callable,
    augment_fn: Callable = None,
) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders for training and validation sets

    Args:
        train_filepaths (Union[pd.Series, list]): Filepaths to training audio files
        train_labels (Union[pd.Series, list]): Labels for training audio files
        val_filepaths (Union[pd.Series, list]): FIlepaths to validation audio files
        val_labels (Union[pd.Series, list]): Labels for validation audio files
        batch_size (int): Batch size for dataloader
        num_workers (int): Number of workers for dataloader
        embedding_fn (_type_): which embedding function to use
        augment_fn (_type_, optional): Which augmentation function to use. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation dataloaders
    """
    training_data = AudioDataset(
        train_filepaths, train_labels, embedding_fn=embedding_fn, augment_fn=augment_fn
    )

    validation_data = AudioDataset(val_filepaths, val_labels, embedding_fn=embedding_fn)

    train_loader = DataLoader(
        training_data, batch_size=batch_size, num_workers=num_workers
    )
    val_loader = DataLoader(
        validation_data, batch_size=batch_size, num_workers=num_workers
    )

    return train_loader, val_loader


def create_trainer(config: DictConfig, run_name: str) -> pl.Trainer:
    """Create pytorch lightning trainer.

    Args:
        config (DictConfig): Hydra config
        run_name (str): Name of the run

    Returns:
        pl.Trainer: Pytorch lightning trainer
    """
    wandb_cb = WandbLogger(name=config.run_name)
    callbacks = [
        ModelCheckpoint(
            dirpath=PROJECT_ROOT / "audio_models" / run_name,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
        ),
    ]
    if config.patience:
        early_stopping = EarlyStopping("val_loss", patience=config.audio.patience)
        callbacks.append(early_stopping)

    trainer = pl.Trainer(
        logger=wandb_cb,
        **config.audio.pl_trainer_kwargs,
    )
    return trainer


def label2id(col, mapping):
    return {"label_id": mapping[col]}


def load_train_val_splits(config: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation splits

    Args:
        config (DictConfig): Hydra config

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes
    """
    data = pd.read_csv(config.data.audio_path)
    train_df = data[data[config.split_column_name == "train"]]
    val_df = data[data[config.split_column_name == "val"]]
    # shuffle train data
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    return train_df, val_df


def get_augmentation_fn(config) -> Union[Callable, None]:
    """Get augmentation function

    Args:
        config (DictConfig): Hydra config

    Returns:
        Union[Callable, None]: Augmentation function
    """
    if config.audio.augmentations is not None:
        augmenter = torch_audiomentations.utils.config.from_dict(
            config.audio.augmentations
        )
        return partial(augmenter, sample_rate=16_000)
    return None


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


def prepare_binary_data(
    config: DictConfig, train: pd.DataFrame, val: pd.DataFrame, origin: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for binary classification. Subsets to only include data from
    the specified origin and maps controls to 0 and cases to 1.

    Args:
        config (DictConfig): Hydra config
        train (pd.DataFrame): Training set
        val (pd.DataFrame): Validation set
        origin (str): Origin of data

    Returns
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation sets
    """

    train_set = train[train[config.data.origin_column_name] == origin]
    val_set = val[val[config.data.origin_column_name] == origin]

    mapping = {config.data.control_label_name: 0, origin: 1}
    train_set["label_id"] = train_set[config.data.label_column_name].replace(mapping)
    val_set["label_id"] = val_set[config.data.label_column_name].replace(mapping)
    return train_set, val_set


if __name__ == "__main__":
    config = DictConfig("x")

    # Load data files
    train, val = load_train_val_splits(config)

    # Prepare augmentation function
    augment_fn = get_augmentation_fn(config)

    # Subset data for faster debugging
    if config.debug:
        msg.info("Debug mode: using 200 random samples")
        train = train.sample(200)
        val = val.sample(200)

    # Get functions for making embeddings
    embedding_fn_dict = get_embedding_fns()

    # train binary models
    if config.train_binary_models:
        for origin in train[config.data.origin_column_name].unique():
            # Prepare data, subset and make label mapping
            msg.divider(f"Training {origin}")

            train_set, val_set = prepare_binary_data(
                config=config, train=train, val=val, origin=origin
            )
            msg.info(f"Training on {len(train_set)} samples")
            msg.info(f"Evaluating on {len(val_set)} samples")
            # Instantiate dataloader and trainer and train
            for feat_set in embedding_fn_dict.keys():
                if feat_set in ["windowed_mfccs"]:
                    continue
                msg.info(f"Starting {feat_set}...")
                # setup wandb config
                model = fit_model(
                    config=config,
                    augment_fn=augment_fn,
                    embedding_fn=embedding_fn_dict[feat_set],
                    train_set=train_set,
                    val_set=val_set,
                    feat_set=feat_set,
                    run_name=f"{origin}_{feat_set}",
                )
            # TODO
            # Evaluate model on train/val/test sets

    #############################
    ##### Multiclass models #####
    #############################
    if config.train_multiclass_models:

        # map label to idx
        train["label_id"] = train[config.data.label_column_name].replace(
            config.data.multiclass_label2id_mapping
        )
        val["label_id"] = val[config.data.label_column_name].replace(
            config.data.multiclass_label2id_mapping
        )

        msg.divider("Training multiclass models")
        for feat_set in embedding_fn_dict.keys():
            if feat_set in ["windowed_mfccs"]:
                continue
            msg.info(f"Starting {feat_set}...")
            model = fit_model(
                config=config,
                augment_fn=augment_fn,
                embedding_fn=embedding_fn_dict[feat_set],
                train_set=train,
                val_set=val,
                feat_set=feat_set,
                run_name=f"multiclass_{feat_set}",
            )
            # TODO
            # Evaluate model on train val test sets

