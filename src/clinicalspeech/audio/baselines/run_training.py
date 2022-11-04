from collections.abc import Callable
from typing import Dict, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from wasabi import msg

from clinicalspeech.audio.base.evaluator import evaluate_model_on_splits
from clinicalspeech.audio.base.trainer import AudioTrainer
from clinicalspeech.audio.base.utils import prepare_binary_data
from clinicalspeech.audio.baselines.model import BaselineClassifier
from clinicalspeech.audio.baselines.utils import create_dataloaders


def _get_train_val_dataloaders(
    config: DictConfig,
    df: pd.DataFrame,
    embedding_fn: Callable,
    augment_fn: Optional[Callable] = None,
):
    """Creates train and validation dataloaders for a given dataframe and embedding function."""
    train = df[df.data.split_column_name == "train"]
    val = df[df.data.split_column_name == "val"]
    train_set, val_set = create_dataloaders(
        train_filepaths=train[config.audio.filepath_column_name].values,
        train_labels=train[config.audio.label_column_name].values,
        val_filepaths=val[config.audio.filepath_column_name].values,
        val_labels=val[config.audio.label_column_name].values,
        batch_size=config.audio.batch_size,
        num_workers=config.audio.num_workers,
        embedding_fn=embedding_fn,
        augment_fn=augment_fn,
    )
    return train_set, val_set


def train_and_evaluate_models(
    config: DictConfig,
    df: pd.DataFrame,
    augment_fn: Callable,
    embedding_fn_dict: dict[str, Callable],
    origin: str,
    id2label: dict,
):
    """Trains and evaluates baseline audio models for each feature set on the given
    dataframe.
    Results are saved to wandb and disk.

    Args:
        config (DictConfig): Hydra config
        df (pd.DataFrame): Dataframe containing all data
        augment_fn (Callable): Augmentation function for training set
        embedding_fn_dict (dict[str, Callable]): Dictionary of embedding functions
        origin (str): Origin of data, e.g. ASD, DEPR, or multiclass
        id2label (dict): Dictionary mapping label ids to label names
    """
    # Loop over each feature set
    model_trainer = AudioTrainer(config=config)

    for feat_set in config.audio.baseline_models_to_train:
        if feat_set not in embedding_fn_dict:
            raise ValueError(
                f"Invalid feature set {feat_set}. Must be one of {embedding_fn_dict.keys()}"
            )
        embedding_fn = embedding_fn_dict[feat_set]()

        train_loader, val_loader = _get_train_val_dataloaders(
            config=config, df=df, embedding_fn=embedding_fn, augment_fn=augment_fn
        )
        model = BaselineClassifier(
            num_classes=len(df[config.data.label_column_name].unique()),
            feature_set=feat_set,
            learning_rate=config.audio.learning_rate,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        msg.info(f"Starting {feat_set}...")
        # train model
        # TODO: check if it is necesarry to assign the model to a variable
        model = model_trainer.fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            feat_set=feat_set,
            run_name=f"baseline_{origin}_{feat_set}",
        )
        msg.info("Evaluating on splits...")
        # evaluate on splits specified in config and save results to disk
        evaluate_model_on_splits(
            df=df,
            model=model,
            model_name=feat_set,
            id2label=id2label,
            target_class=origin,
            is_baseline=True,
            config=config,
            embedding_fn=embedding_fn,
        )


def train_binary_audio_baselines(
    config: DictConfig,
    df: pd.DataFrame,
    augment_fn: Callable,
    embedding_fn_dict: dict[str, Callable],
):
    """Trains baseline audio models for each feature set and origin. Origin is used
    to subset the data to only include data from that origin and map controls to 0
    and cases to 1. The models are evaluated on the splits defined in config.eval_splits
    and saved to disk.

    Args:
        config (DictConfig): Hydra config
        df (pd.DataFrame): Dataframe containing all data
        augment_fn (Callable): Augmentation function for training set
        embedding_fn_dict (dict[str, Callable]): Dictionary of embedding functions
    """
    for origin in df[config.data.origin_column_name].unique():
        # Prepare data, subset and make label mapping
        msg.divider(f"Training {origin}")

        binary_df = prepare_binary_data(config=config, df=df, origin=origin)
        n_train_samples = len(binary_df[binary_df.data.split_column_name == "train"])
        n_val_samples = len(binary_df[binary_df.data.split_column_name == "val"])
        msg.info(f"Training on {n_train_samples} samples")
        msg.info(f"Evaluating on {n_val_samples} samples")
        train_and_evaluate_models(
            config=config,
            df=binary_df,
            augment_fn=augment_fn,
            embedding_fn_dict=embedding_fn_dict,
            origin=origin,
            id2label={0: config.data.control_label_name, 1: origin},
        )
        msg.info("Done!")


def train_multiclass_audio_baselines(
    config: DictConfig,
    df: pd.DataFrame,
    augment_fn: Callable,
    embedding_fn_dict: dict[str, Callable],
):
    """Train multiclass baseline audio models for each feature set. The models are
    evaluated on the splits defined in config.eval_splits and saved to disk.

    Args:
        config (DictConfig): Hydra config
        df (pd.DataFrame): Dataframe containing all data
        augment_fn (Callable): Augmentation function for training set
        embedding_fn_dict (dict[str, Callable]): Dictionary of embedding functions
    """
    # Map labels to ids
    id2label = OmegaConf.to_container(config.data.multiclass_id2label_mapping)
    df["label_id"] = df[config.data.label_column_name].replace(id2label)

    msg.divider("Training multiclass models")

    train_and_evaluate_models(
        config=config,
        df=df,
        augment_fn=augment_fn,
        embedding_fn_dict=embedding_fn_dict,
        origin="multiclass",
        id2label=id2label,  # type: ignore
    )
