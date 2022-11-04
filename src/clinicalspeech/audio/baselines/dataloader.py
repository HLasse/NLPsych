"""Dataloader for baseline models"""
import torch
import torchaudio
from _collections_abc import Callable

from clinicalspeech.audio.base.dataloader import AudioDataset


class BaselineAudioDataset(AudioDataset):  # pylint: disable=too-few-public-methods
    """Pytorch dataset for audio data. Includes options for embedding the data
    using an `embedding_fn` and augmenting using an `augment_fn`
    """

    def __init__(self, embedding_fn: Callable = None, **kwargs):
        super().__init__(**kwargs)
        self.embedding_fn = embedding_fn

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.paths[idx])
        if self.max_frame_length is not None:
            audio = self._random_cut_audio(audio)
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
