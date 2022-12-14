"""Pytorch lightning module for baseline models."""
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


class BaselineClassifier(pl.LightningModule):
    """Baseline classification model. Maps the features to a single linear layer
    which is fed to a classification head. Very similar to HuggingFace's classification
    layers."""

    def __init__(
        self,
        num_classes: int,
        feature_set: str,
        learning_rate: float,
        train_loader: DataLoader,
        val_loader: DataLoader,
        weights: Optional[np.ndarray] = None,
    ):
        super(BaselineClassifier, self).__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weights = weights

        input_dims = {
            "xvector": 512,
            "egemaps": 88,
            "aggregated_mfccs": 40,
            "compare": 6373,
        }

        self.linear = nn.Linear(input_dims[feature_set], 1028)
        self.classifier = nn.Linear(1028, num_classes)
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

    def forward(self, x):
        if len(x.shape) == 3:
            # HOTFIX weird stuff going on with some of the embeddings
            x = x.squeeze()
        x = F.leaky_relu(self.linear(x))
        x = self.classifier(x)

        return torch.log_softmax(x, dim=1)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        metrics = self.train_metrics(preds, y)
        self.log_dict(metrics)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, val_epoch):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        metrics = self.val_metrics(preds, y)

        self.log_dict(metrics)
        self.log("val_loss", loss)
        return {"loss": loss, "labels": y}

    def predict_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        return preds

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels, weight=self.weights)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
