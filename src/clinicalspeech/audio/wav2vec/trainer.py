from collections.abc import Callable
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch_audiomentations.core.composition import Compose
from transformers import Wav2Vec2FeatureExtractor

from clinicalspeech.audio.base.trainer import BaseAudioTrainer
from clinicalspeech.audio.wav2vec.dataloader import HuggingFaceAudioDataset
from clinicalspeech.audio.wav2vec.model import Wav2Vec2Classifier


class Wav2VecAudioTrainer(BaseAudioTrainer):
    """Trainer for Wav2Vec2 classification models"""

    def _create_dataloaders(
        self,
        train_filepaths: pd.Series,
        train_labels: pd.Series,
        val_filepaths: pd.Series,
        val_labels: pd.Series,
        batch_size: int,
        num_workers: int,
        model_id: str,
        augment_fn: Compose = None,
    ) -> tuple[DataLoader, DataLoader]:
        """Create dataloaders for training and validation sets

        Args:
            train_filepaths (pd.Series): Filepaths to training audio files
            train_labels (pd.Series): Labels for training audio files
            val_filepaths (pd.Series): FIlepaths to validation audio files
            val_labels (pd.Series): Labels for validation audio files
            batch_size (int): Batch size for dataloader
            num_workers (int): Number of workers for dataloader
            embedding_fn (Callable): which embedding function to use
            augment_fn (Compose, optional): Torch-audiomentations augmentation function. Defaults to None.

        Returns:
            tuple[DataLoader, DataLoader]: Training and validation dataloaders
        """
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        training_data = HuggingFaceAudioDataset(
            paths=train_filepaths,
            labels=train_labels,
            feature_extractor=feature_extractor,
            augment_fn=augment_fn,
        )
        validation_data = HuggingFaceAudioDataset(
            paths=val_filepaths, labels=val_labels, feature_extractor=feature_extractor
        )

        train_loader = DataLoader(
            training_data, batch_size=batch_size, num_workers=num_workers
        )
        val_loader = DataLoader(
            validation_data, batch_size=batch_size, num_workers=num_workers
        )

        return train_loader, val_loader

    def _get_lightning_model(
        self,
        num_classes: int,
        learning_rate: float,
        train_loader: DataLoader,
        val_loader: DataLoader,
        weights: Optional[torch.Tensor],
        model_id: str,
    ) -> pl.LightningModule:
        return Wav2Vec2Classifier(
            num_classes=num_classes,
            model_id=model_id,
            learning_rate=learning_rate,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=weights,
        )
