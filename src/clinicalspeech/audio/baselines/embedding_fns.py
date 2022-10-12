"""Returns embedding functions for baseline audio models. 
TODO
- Add wav2vec2 embedding function
"""
from collections.abc import Callable

import numpy as np
import opensmile
import torch
from speechbrain.pretrained import EncoderClassifier
from torchaudio.transforms import MFCC

# pylint: disable=too-few-public-methods
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from typing_extensions import Literal


class MFCCEmbedder:
    """Embed audio using mean MFCCs"""

    def __init__(self):
        self.mfcc_extractor = MFCC(
            sample_rate=16000, n_mfcc=40, dct_type=2, norm="ortho", log_mels=False
        )

    def __call__(self, audio) -> torch.tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio).type(torch.float)
        mfccs = self.mfcc_extractor(audio)
        if len(audio.shape) == 1:
            return torch.mean(mfccs, 1)
        else:
            return torch.mean(mfccs, 2)


class OpenSmileEmbedder:
    """Enbed audio using OpenSmile"""

    def __init__(self, feature_set: Literal, feature_level: Literal):
        self.feature_set = feature_set
        self.feature_level = feature_level
        self.opensmile = opensmile.Smile(
            feature_set=self.feature_set,
            feature_level=self.feature_level,
            num_workers=10,
        )

    def __call__(self, audio) -> torch.tensor:
        # if only 1 file (e.g. in dataloader) just embed the 1 file
        if len(audio.shape) == 1:
            embeddings = (
                self.opensmile.process_signal(audio, sampling_rate=16000)
                .to_numpy()
                .squeeze()
            )
        else:
            embeddings = [
                self.opensmile.process_signal(a, sampling_rate=16000)
                .to_numpy()
                .squeeze()
                for a in audio
            ]
        return torch.tensor(embeddings)


class ComParEEmbedder(OpenSmileEmbedder):
    """Embed audio using ComParE 2016 features. Shape = (batch, 6373)"""

    def __init__(self):
        super().__init__(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )


class EgeMapsEmbedder(OpenSmileEmbedder):
    """Embed audio using eGeMAPS features. Shape = (batch, 88)"""

    def __init__(self):
        super().__init__(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )


class XVectorEmbedder:
    """Embed audio using x-vectors"""

    def __init__(self):
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb",
        )

    def __call__(self, audio) -> torch.tensor:
        # shape = (batch, 512)
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio)
        return self.model.encode_batch(audio).squeeze()


class Wav2Vec2Embedder:
    @torch.no_grad()
    def __init__(self, model_id: str, contextualized_embeddings: bool):
        self.model_id = model_id
        self.contextualized_embeddings = contextualized_embeddings

        # normalization should be done before feature extraction
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_id, do_normalize=False
        )
        self.model = Wav2Vec2Model.from_pretrained(self.model_id)
        self.model = self.model.eval()

    def __call__(self, audio) -> torch.tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio)

        input_values = self.feature_extractor(
            audio,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        ).input_values

        # reshape to get shape (batch, input_length)
        input_values = input_values.reshape(1, -1)

        if self.contextualized_embeddings:
            embeddings = self.model(input_values).last_hidden_state
        else:
            embeddings = self.model(input_values).extract_features
        # mean pool
        embeddings = torch.mean(embeddings, 1)
        embeddings = embeddings.squeeze()
        return embeddings


class XLSRDanishEmbedder(Wav2Vec2Embedder):
    def __init__(self):
        super().__init__(
            model_id="chcaa/xls-r-300m-danish",
            contextualized_embeddings=True,
        )
        # shape: (batch, 1024)


class XLSREmbedder(Wav2Vec2Embedder):
    def __init__(self):
        super().__init__(
            model_id="facebook/wav2vec2-xls-r-300m",
            contextualized_embeddings=True,
        )
        # shape: (batch, 1024)


def get_embedding_fns() -> dict[str, Callable]:
    """Return a dictionary of callable embedding functions"""

    return {
        "xvector": XVectorEmbedder,
        "egemaps": EgeMapsEmbedder,
        "compare": ComParEEmbedder,
        "aggregated_mfccs": MFCCEmbedder,
        "xlsr_danish": XLSRDanishEmbedder,
        "xlsr": XLSREmbedder,
    }


if __name__ == "__main__":
    emb = get_embedding_fns()
    x = torch.rand(2, 16000)

    wav2vec = XLSRDanishEmbedder()

    x_hat = wav2vec(x)
    print(x_hat.shape)
