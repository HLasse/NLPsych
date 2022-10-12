import random
import string

import numpy as np
import pandas as pd
import torch
import torchaudio

from clinicalspeech.utils import PROJECT_ROOT

# pylint: disable=dangerous-default-value


def make_synth_audio_data(n_samples: int, n_frames: int) -> torch.Tensor:
    """Make synthetic audio data"""
    return torch.rand(n_samples, n_frames)


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


if __name__ == "__main__":
    DATA_PATH = PROJECT_ROOT / "data"
    AUDIO_FILES_PATH = DATA_PATH / "audio_files"

    N_SAMPLES = 50
    N_FRAMES = 32000
    N_SAMPLES_PER_ID = 2

    audio_data = make_synth_audio_data(N_SAMPLES, N_FRAMES)



audio_path_column_name: filename
id_column_name: id
label_column_name: label
trial_column_name: trial_id
split_column_name: split
origin_column_name: origin  # column indicating which group the data belongs to
# e.g. DEPR, SCHZ, ASD. Used for subsetting during training of binary models.
control_label_name: TD
