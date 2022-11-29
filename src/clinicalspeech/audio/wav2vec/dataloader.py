"""Dataloader for wav2vec models"""
import torchaudio
from transformers import Wav2Vec2FeatureExtractor

from clinicalspeech.audio.base.dataloader import AudioDataset


class HuggingFaceAudioDataset(AudioDataset):  # pylint: disable=too-few-public-methods
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
