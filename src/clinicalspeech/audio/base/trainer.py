"""Base audio trainer class"""
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import torch_audiomentations
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner.lr_finder import _LRFinder
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

import wandb
from clinicalspeech.utils import PROJECT_ROOT

# pylint: disable=unnecessary-ellipsis


class AudioTrainer:
    """Base class for audio trainers."""

    def __init__(self, config: DictConfig):
        self.config = config
        if config.audio.augmentations is not None:
            self.augment_fn = torch_audiomentations.utils.config.from_dict(
                OmegaConf.to_container(config.audio.augmentations)
            )
        else:
            self.augment_fn = None

    def fit_model(
        self,
        model: pl.LightningModule,
        train_loader: DataLoader,
        val_loader: DataLoader,
        feat_set: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> pl.LightningModule:
        """Fit model to training data and save to wandb and disk.

        Args:
            model (pl.LightningModule): Model to train
            train_loader (DataLoader): Training dataloader
            val_loader (DataLoader): Validation dataloader
            feat_set (str): Name of feature set, used for wandb logging
            run_name (str): Name of wandb run

        Returns:
            pl.LightningModule: Trained model
        """
        if self.config.project.wandb.use_wandb:
            run: Run = wandb.init(
                config=self.config,  # type: ignore
                project=self.config.project.wandb.wandb_project,
                dir=PROJECT_ROOT,
                allow_val_change=True,
                reinit=True,
                mode=self.config.project.wandb.wandb_mode,
            )
            run_config = run.config
            run.name = run_name
            run_config.run_name = run.name
            run.log({"feat_set": feat_set})

        # set class weights if specified in config
        if self.config.data.use_class_weights:
            weights = self._get_class_weights(train_loader)
            model.class_weights = weights

        trainer = self._create_trainer(run_name=run_name)

        # find optimal learning rate if specified in config
        self._find_learning_rate(
            run=run, run_config=run_config, model=model, trainer=trainer
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Finish tracking run on wandb to start the next one
        run.finish()
        return model

    def _create_trainer(self, run_name: Optional[str]) -> pl.Trainer:
        # sourcery skip: inline-immediately-returned-variable
        """Create PyTorch lightning trainer.

        Args:
            run_name (str): Name of the run

        Returns:
            pl.Trainer: PyTorch lightning trainer
        """
        # use wandb logger if specified in config, else use default tensorboard logger
        logger = (
            WandbLogger(name=run_name) if self.config.project.wandb.use_wandb else True
        )

        callbacks: list[Callback] = []
        if self.config.audio.save_models:
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
        if self.config.audio.patience:
            callbacks.append(
                EarlyStopping("val_loss", patience=self.config.audio.patience)
            )

        trainer = pl.Trainer(
            logger=logger,  # type: ignore
            **self.config.audio.pl_trainer_kwargs,
        )
        return trainer

    def _get_class_weights(self, train_set: pd.DataFrame) -> torch.Tensor:
        """Get class weights for training set.

        Args:
            config (DictConfig): Hydra config
            train_set (pd.DataFrame): Training set

        Returns:
            Union[None, np.array]: Class weights
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # compute weights to avoid overfitting on majority class
        return torch.tensor(
            compute_class_weight(
                "balanced",
                classes=list(range(len(train_set["label_id"].unique()))),
                y=train_set["label_id"].tolist(),
            ),
            dtype=torch.float,
        ).to(device)

    def _find_learning_rate(
        self,
        run: Run,
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
        if self.config.audio.auto_lr_find:
            lr_finder: _LRFinder = trainer.tuner.lr_find(model)  # type: ignore
            run_config.update(
                {"learning_rate": lr_finder.suggestion()}, allow_val_change=True
            )
            fig = lr_finder.plot(suggest=True)
            run.log({"lr_finder.plot": fig})
            run.log({"found_lr": lr_finder.suggestion()})
