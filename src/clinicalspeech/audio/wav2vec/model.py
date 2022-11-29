"""Pytorch lightning module for baseline models."""
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from transformers import Wav2Vec2ForSequenceClassification

from clinicalspeech.audio.wav2vec.dataloader import HuggingFaceAudioDataset


class Wav2Vec2Classifier(pl.LightningModule):
    """Wav2vec2 Classification models. Wraps Huggingface's Wav2Vec2ForSequenceClassification
    model to be compatible with PyTorch Lightning. Optimizer and interface purposefully
    kept similar to baseline model.

    Args:
        model_id (str): Huggingface model id.
        num_classes (int): Number of classes.
        learning_rate (float): Learning rate.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        weights (Optional[np.ndarray], optional): Weights for cross entropy loss
            if imbalanced classes. Defaults to None.
    """

    def __init__(
        self,
        model_id: str,
        num_classes: int,
        learning_rate: float,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[np.ndarray] = None,
    ):
        super(Wav2Vec2Classifier, self).__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_weights = class_weights
        self.dataloader_type = HuggingFaceAudioDataset
        if (
            train_loader.dataset.__class__ != self.dataloader_type
            or val_loader.dataset.__class__ != self.dataloader_type
        ):
            raise TypeError(
                f"Expected dataloaders to be of type {self.dataloader_type}, but got {train_loader.dataset.__class__} and {val_loader.dataset.__class__}"
            )

        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_id, num_labels=num_classes
        )

        metrics = MetricCollection(
            [
                Precision(average="macro", num_classes=num_classes),
                Recall(average="macro", num_classes=num_classes),
                F1Score(average="macro", num_classes=num_classes),
                Accuracy(average="macro", num_classes=num_classes),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits = outputs.get("logits")
        loss = self.cross_entropy_loss(logits=logits, labels=batch["labels"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits = outputs.get("logits")
        val_loss = self.cross_entropy_loss(logits=logits, labels=batch["labels"])
        self.log("val_loss", val_loss)

        return {"loss": val_loss, "labels": batch["labels"]}

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(
            logits.view(-1, self.num_classes),
            labels.view(-1),
            weight=self.class_weights,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def get_dataloader(
        self,
        dataframe: pd.DataFrame,
        path_column: str,
        label_column: str,
        max_frame_length: Union[int, None],
        model_id: str,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
    ):
        """Get dataloader for model.
        
        Args:
            dataframe (pd.DataFrame): Dataframe with audio paths and labels.
            path_column (str): Column name for audio paths.
            label_column (str): Column name for labels.
            max_frame_length (Union[int, None]): Maximum frame length for audio.
            model_id (str): Huggingface model id.
            batch_size (int): Batch size.
            num_workers (int): Number of workers for dataloader.
            shuffle (bool): Whether to shuffle data."""
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        dataset = HuggingFaceAudioDataset(
            paths=dataframe[path_column].values,
            labels=dataframe[label_column].values,
            feature_extractor=feature_extractor,
            max_frame_length=max_frame_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )


if __name__ == "__main__":
    import pandas as pd
    from torch_audiomentations import Compose, Gain, PolarityInversion
    from transformers import Wav2Vec2FeatureExtractor

    from clinicalspeech.utils import PROJECT_ROOT

    # Initialize augmentation callable
    apply_augmentation = Compose(
        transforms=[
            Gain(
                min_gain_in_db=-15.0,
                max_gain_in_db=5.0,
                p=0.5,
            ),
            PolarityInversion(p=0.5),
        ]
    )

    model_id = "chcaa/xls-r-300m-danish"

    df = pd.read_csv(PROJECT_ROOT / "data" / "synth_data.csv")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    ds = HuggingFaceAudioDataset(
        paths=df["audio_path"],
        labels=df["label"],
        feature_extractor=feature_extractor,
        augment_fn=apply_augmentation,
    )
    dl = DataLoader(ds, batch_size=2)

    model = Wav2Vec2Classifier(
        model_id=model_id,
        num_classes=2,
        learning_rate=1e-4,
        train_loader=dl,
        val_loader=dl,
    )

    # model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id, num_labels=2)
    # create random sample data looking like audio
    for x, _ in iter(dl):
        out = model(x)
