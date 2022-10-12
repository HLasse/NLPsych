import random
import string

import numpy as np
import pandas as pd
import torch
import torchaudio

from clinicalspeech.utils import PROJECT_ROOT

# pylint: disable=dangerous-default-value,redefined-outer-name


def make_synth_audio_data(n_samples: int, n_frames: int) -> torch.Tensor:
    """Make synthetic audio data"""
    return torch.rand(n_samples, 1, n_frames)


def make_synth_text_data(n_samples: int) -> list[str]:
    """Make synthetic text data"""
    return [
        "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
        for _ in range(n_samples)
    ]


def make_synth_labels_and_origin(
    n_samples: int,
) -> tuple[list[str], list[str]]:
    """Make synthetic labels and origins"""
    label_names = ["ASD", "DEPR", "SCHZ", "TD"]
    label_weights = [0.2, 0.2, 0.2, 0.4]

    labels = random.choices(label_names, weights=label_weights, k=n_samples)
    origins = []
    for label in labels:
        if label != "TD":
            origins.append(label)
        else:
            origins.append(random.choice(label_names[:3]))
    return labels, origins


def make_synth_ids(n_samples: int, n_samples_per_id: int) -> list[str]:
    """Make synthetic ids"""
    return [f"{i // n_samples_per_id}" for i in range(n_samples)]


def make_synth_trial_ids(ids: list[str]) -> list[str]:
    """Make synthetic trial ids"""
    return [f"{id}_{i}" for i, id in enumerate(ids)]


def save_audio_files(paths: list[str], audio_data: torch.Tensor) -> None:
    """Save audio files"""
    for path, audio in zip(paths, audio_data):
        torchaudio.save(path, audio, 16000)


if __name__ == "__main__":
    DATA_PATH = PROJECT_ROOT / "data"
    AUDIO_FILES_PATH = DATA_PATH / "audio_files"

    N_SAMPLES = 50
    N_FRAMES = 32000
    N_SAMPLES_PER_ID = 2

    audio_data = make_synth_audio_data(N_SAMPLES, N_FRAMES)
    audio_paths = [AUDIO_FILES_PATH / f"{i}.wav" for i in range(N_SAMPLES)]
    save_audio_files(audio_paths, audio_data)

    text_data = make_synth_text_data(N_SAMPLES)
    labels, origins = make_synth_labels_and_origin(N_SAMPLES)
    ids = make_synth_ids(N_SAMPLES, N_SAMPLES_PER_ID)
    trial_ids = make_synth_trial_ids(ids)
    # put all variables into dataframe
    df = pd.DataFrame(
        {
            "id": ids,
            "trial_id": trial_ids,
            "text": text_data,
            "label": labels,
            "origin": origins,
            "audio_path": audio_paths,
        }
    )
    # assign 60% of data to train, 20% to val, 20% to test
    df["split"] = random.choices(
        ["train", "val", "test"], weights=[0.6, 0.2, 0.2], k=N_SAMPLES
    )
    df.to_csv(DATA_PATH / "synth_data.csv", index=False)
