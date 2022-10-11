"""Various utils for training baseline models"""
from collections.abc import Callable
from functools import partial
from typing import Union

import pandas as pd
import torch_audiomentations
from torch.utils.data import DataLoader

from clinicalspeech.audio.dataloaders.torch_dataloader import AudioDataset


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
