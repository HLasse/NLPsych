"""Base audio dataloader class"""
from abc import abstractmethod
from collections.abc import Callable
from typing import Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Base class for audio datasets
    Subclasses should implement the __getitem__ method

    Args:
        paths (Union[pd.Series, list]): Filepaths to audio files
        labels (Union[pd.Series, list]): Labels for audio files
        augment_fn (Optional[Callable], optional): torch_audiomentations augmentation function.
            Defaults to None.
        max_frame_length (Optional[int], optional): Maximum length of audio in frames.
            If set, will randomly crop audio to this length and zero pad audio that
            are shorter. Defaults to None.
    """

    def __init__(
        self,
        paths: Union[pd.Series, list],
        labels: Union[pd.Series, list],
        augment_fn: Optional[Callable] = None,
        max_frame_length: Optional[int] = None,
    ):
        self.paths = paths
        self.labels = labels
        self.augment_fn = augment_fn
        self.max_frame_length = max_frame_length

    def __len__(self):
        return len(self.paths)

    def _random_cut_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Randomly cut audio to self.max_frame_length. Zero pad if audio is shorter."""
        # zero pad if audio is shorter than max_frame_length
        if not self.max_frame_length:
            raise ValueError("max_frame_length must be set to use `random_cut_audio`")
        if audio.shape[1] < self.max_frame_length:
            audio = torch.nn.functional.pad(
                audio, (0, self.max_frame_length - audio.shape[1])
            )
        # random crop if audio is longer than max_frame_length
        if audio.shape[1] > self.max_frame_length:
            start = torch.randint(
                0, audio.shape[1] - self.max_frame_length, (1,)
            ).item()
            audio = audio[:, start : start + self.max_frame_length]  # type: ignore
        return audio

    @abstractmethod
    def __getitem__(self, idx):
        pass
