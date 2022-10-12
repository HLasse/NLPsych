"""Run evaluation of pytorch-lightning baseline models on the different splits.
Saves the results to a jsonl file"""
from collections.abc import Callable
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from wasabi import msg

from clinicalspeech.audio.dataloaders.torch_dataloader import AudioDataset
from clinicalspeech.utils import PROJECT_ROOT


def load_split_and_dataloader(
    df: pd.DataFrame,
    split_column: str,
    split: str,
    path_col: str,
    label_col: str,
    embedding_fn: Callable,
    batch_size: int,
    num_workers: int,
) -> tuple[pd.DataFrame, AudioDataset]:
    """Loads a datasplit and returns the dataframe and the torch dataloader

    Args:
        df (pd.DataFrame): dataframe with the metadata
        split_column (str): Column containing split information
        split (str): Name of the split
        path_col (str): Name of the column containing the paths to the audio files
        label_col (str): Name of the column containing the labels
        embedding_fn (Callable): Function to embed the audio
        batch_size (int): Batch size for the dataloader
        num_workers (int): Number of workers for the dataloader

    Returns:
        tuple(pd.DataFrame, AudioDataset): The dataframe and the torch dataloader

    """
    df = df[df[split_column] == split]
    dataset = AudioDataset(
        paths=df[path_col],
        labels=df[label_col],
        augment_fn=None,
        embedding_fn=embedding_fn,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return df, dataloader


def add_predictions_to_df(
    df: pd.DataFrame, logits: torch.Tensor, id2label: dict
) -> pd.DataFrame:
    """Add predictions to a dataframe

    Args:
        df (pd.DataFrame): dataframe with the metadata
        logits (torch.Tensor): Tensor with the logits
        id2label (dict): Dictionary mapping the label ids to the labels

    Returns:
        pd.DataFrame: dataframe with the predictions
    """
    # baseline models return log softmax - removing the log
    logits = torch.exp(logits).detach().cpu().numpy()
    # Getting top prediction per row
    arg_max = np.argmax(logits)
    df["prediction"] = id2label[arg_max]
    df["confidence"] = logits[arg_max]
    df["scores"] = logits.tolist()
    return df


def save_to_jsonl(
    df: pd.DataFrame,
    save_dir: Path,
    save_name: str,
    split: str,
    num_labels: int,
    model_name: str,
    target_class: str,
    is_baseline: bool,
):
    """Save dataframe to jsonl file. Add metadata for easier plotting

    Args:
        df (pd.DataFrame): dataframe to save
        save_dir (Path): directory to save to
        save_name (str): name of the file to save to
        split (str): name of the split
        num_labels (int): number of labels
        model_name (str): name of the model
        target_class (str): name of the target class
        is_baseline (bool): whether the model is a baseline model
    """
    df["binary"] = bool(num_labels == 2)
    df["type"] = "audio"
    df["split"] = split
    df["model_name"] = model_name
    df["target_class"] = target_class
    df["is_baseline"] = 1 if is_baseline else 0

    df.to_json(save_dir / f"{save_name}.jsonl", lines=True, orient="records")
    msg.info(f"Saved results to {save_dir / save_name}.jsonl")


def evaluate_model(
    df: pd.DataFrame,
    model: pl.LightningModule,
    model_name: str,
    splits_to_evaluate: Union[list[str], str],
    embedding_fn: Callable,
    id2label: dict,
    num_labels: int,
    target_class: str,
    is_baseline: bool,
    config: DictConfig,
) -> None:
    """Run evaluation of pytorch-lightning baseline models on the different splits.
    Saves the results to a jsonl file.

    Args:
        df (pd.DataFrame): dataframe with the metadata
        model (pl.LightningModule): pytorch-lightning model
        model_name (str): Name of the model
        splits_to_evaluate (Union[list[str], str]): List of splits to evaluate on
        embedding_fn (Callable): Function to embed the audio
        id2label (dict): Dictionary mapping the label ids to the labels
        num_labels (int): Number of labels
        target_class (str): Name of the target class
        is_baseline (bool): Whether the model is a baseline model
        config (DictConfig): Config file
    """
    model.eval()
    model.freeze()

    # load split and create dataloader
    if isinstance(splits_to_evaluate, str):
        splits_to_evaluate = [splits_to_evaluate]
    for split in splits_to_evaluate:
        df_split, dataloader = load_split_and_dataloader(
            df=df,
            split_column=config.data.split_column_name,
            split=split,
            path_col=config.data.audio_path_column_name,
            label_col=config.data.label_column_name,
            embedding_fn=embedding_fn,
            batch_size=config.audio.batch_size,
            num_workers=config.audio.num_workers,
        )
        # run evaluation
        logits = model(dataloader)
        df_split = add_predictions_to_df(df=df_split, logits=logits, id2label=id2label)

        # save results
        save_to_jsonl(
            df=df_split,
            save_dir=PROJECT_ROOT / "model_predictions",
            save_name=f"{target_class}_{model_name}_{split}",
            split=split,
            num_labels=num_labels,
            model_name=model_name,
            target_class=target_class,
            is_baseline=is_baseline,
        )
