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


def get_embedding_fns() -> dict[str, Callable]:
    """Return a dictionary of callable embedding functions"""

    xvector_embedding = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
    )

    def xvector_embedding_fn(audio) -> torch.tensor:
        # shape = (batch, 512)
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio)
        return xvector_embedding.encode_batch(audio).squeeze()

    egemapsv2 = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    def egemaps_embedding_fn(audio) -> torch.tensor:
        # shape = (batch, 88)
        # if only 1 file (e.g. in dataloader) just embed the 1 file
        if len(audio.shape) == 1:
            embeddings = (
                egemapsv2.process_signal(audio, sampling_rate=16000)
                .to_numpy()
                .squeeze()
            )
        else:
            embeddings = [
                egemapsv2.process_signal(a, sampling_rate=16000).to_numpy().squeeze()
                for a in audio
            ]
        return torch.tensor(np.array(embeddings))

    compare = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers=10,
    )

    def compare_embedding_fn(audio) -> torch.tensor:
        # shape = (batch, 6373)
        if len(audio.shape) == 1:
            embeddings = (
                compare.process_signal(audio, sampling_rate=16000).to_numpy().squeeze()
            )
        else:
            embeddings = [
                compare.process_signal(a, sampling_rate=16000).to_numpy().squeeze()
                for a in audio
            ]
        return torch.tensor(embeddings)

    mfcc_extractor = MFCC(
        sample_rate=16000, n_mfcc=40, dct_type=2, norm="ortho", log_mels=False
    )

    def aggregated_mfccs_fn(audio) -> torch.tensor:
        # shape = (batch, 128)
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio).type(torch.float)

        mfccs = mfcc_extractor(audio)
        if len(audio.shape) == 1:
            return torch.mean(mfccs, 1)
        else:
            return torch.mean(mfccs, 2)

    def windowed_mfccs_fn(audio):
        # shape = (batch, n_mels, samples (401)))
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio).type(torch.float)
        return mfcc_extractor(audio)

    return {
        "xvector": xvector_embedding_fn,
        "egemaps": egemaps_embedding_fn,
        "compare": compare_embedding_fn,
        "aggregated_mfccs": aggregated_mfccs_fn,
        "windowed_mfccs": windowed_mfccs_fn,
    }
