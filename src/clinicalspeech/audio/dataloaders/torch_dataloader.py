### DEPRECATED - REMOVE IF THE OTHER WORKS
"""Torch dataset for loading audio files and labels."""
from abc import abstractmethod
from collections.abc import Callable
from typing import Optional, Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor

# pylint: disable=too-few-public-methods


class AudioDataset(Dataset):
    """Base class for audio datasets

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
        if audio.shape[1] < self.max_frame_length:
            audio = torch.nn.functional.pad(
                audio, (0, self.max_frame_length - audio.shape[1])
            )
        # random crop if audio is longer than max_frame_length
        if audio.shape[1] > self.max_frame_length:
            start = torch.randint(
                0, audio.shape[1] - self.max_frame_length, (1,)
            ).item()
            audio = audio[:, start : start + self.max_frame_length]
        return audio

    @abstractmethod
    def __getitem__(self, idx):
        pass


class BaselineAudioDataset(AudioDataset):
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


class HuggingFaceAudioDataset(AudioDataset):
    """Pytorch dataset for audio using HuggingFace processors to prepare input
    for Wav2vec models. Includes options for augmenting using an `augment_fn`"""

    def __init__(self, feature_extractor: Wav2Vec2FeatureExtractor, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor

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
            audio = audio.squeeze(1)
        # Apply HuggingFace feature extractor (normalization, padding, etc.)
        audio = self.feature_extractor(audio, sampling_rate=16000)
        audio = self.feature_extractor.pad(audio, padding=True)
        # HF processors adds a batch dimension which we need to remove for proper
        # input to wav2vec models
        audio["input_values"], audio["attention_mask"] = (
            audio["input_values"].squeeze(),
            audio["attention_mask"].squeeze(1),
        )
        label = self.labels[idx]
        return audio, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from clinicalspeech.audio.baselines.embedding_fns import XLSRDanishEmbedder
    from clinicalspeech.utils import PROJECT_ROOT

    TEST_BASELINE = False
    TEST_HUGGINGFACE = True

    df = pd.read_csv(PROJECT_ROOT / "data" / "synth_data.csv")

    if TEST_BASELINE:
        embed_fn = XLSRDanishEmbedder()
        ds = BaselineAudioDataset(paths=df["audio_path"], labels=df["label"])
    if TEST_HUGGINGFACE:
        processor = Wav2Vec2FeatureExtractor.from_pretrained("chcaa/xls-r-300m-danish")
        ds = HuggingFaceAudioDataset(
            paths=df["audio_path"], labels=df["label"], feature_extractor=processor
        )
    dl = DataLoader(ds, batch_size=2, num_workers=2)
    for batch in dl:
        print(batch)
