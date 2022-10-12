"""Torch dataset for loading audio files and labels."""
from collections.abc import Callable
from typing import Union

import pandas as pd
import torch
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
            with torch.no_grad():
                audio = self.embedding_fn(audio)
        label = self.labels[idx]
        return audio, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from clinicalspeech.audio.baselines.embedding_fns import XLSRDanishEmbedder
    from clinicalspeech.utils import PROJECT_ROOT

    embed_fn = XLSRDanishEmbedder()

    df = pd.read_csv(PROJECT_ROOT / "data" / "synth_data.csv")
    ds = AudioDataset(paths=df["audio_path"], labels=df["label"])
    dl = DataLoader(ds, batch_size=2, num_workers=2)
    for batch in dl:
        print(batch)
