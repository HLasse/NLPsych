"""Torch dataset for loading audio files and labels."""
from collections.abc import Callable
from typing import Union

import pandas as pd
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Pytorch dataset for audio data. Includes options for embedding the data
    using an `embedding_fn` and augmenting using an `augment_fn`
    """

    def __init__(
        self,
        paths: Union[pd.Series, list],  # path to audio files
        labels: Union[pd.Series, list],
        augment_fn: Callable = None,
        embedding_fn: Callable = None,
    ):
        self.paths = paths
        self.labels = labels
        self.augment_fn = augment_fn
        self.embedding_fn = embedding_fn

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.paths[idx])
        if self.augment_fn:
            # torch_audiomentations expects inputs of shape (batch_size, num_channels, num_samples)
            # adding a batch dimension
            if len(audio.shape) == 2:
                audio = audio[None, :, :]
            audio = self.augment_fn(audio)
            # Remove the extra dimension again
            # newer version of torch audiomentations return a dict, take only the samples
            if isinstance(audio, dict):
                audio = audio.samples
            audio = audio.squeeze()
        if self.embedding_fn:
            audio = self.embedding_fn(audio)
        label = self.labels[idx]
        return audio, label
