from pathlib import Path

import hydra

from clinicalspeech.audio.baselines.baseline_trainer import (
    train_binary_audio_baselines,
    train_multiclass_audio_baselines,
)
from clinicalspeech.audio.baselines.embedding_fns import get_embedding_fns
from clinicalspeech.audio.baselines.utils import get_augmentation_fn
from clinicalspeech.utils import load_splits, subsample

CONFIG_PATH = Path(__file__).parent / "config"


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="default_config",
    version_base="0.1",
)
def main(config):
    """Main function for training models"""
    # Audio set up
    audio_train, audio_val, audio_test = load_splits(
        path=config.data.audio_path, split_column_name=config.data.split_column_name
    )
    # Subset data for faster debugging
    if config.debug:
        audio_train, audio_val = subsample(audio_train, audio_val)

    # Get augmentation function
    augment_fn = get_augmentation_fn(config)
    # Get functions for making embeddings
    embedding_fn_dict = get_embedding_fns()

    if config.train_binary_models:
        train_binary_audio_baselines(
            config=config,
            train=audio_train,
            val=audio_val,
            test=audio_test,
            augment_fn=augment_fn,
            embedding_fn_dict=embedding_fn_dict,
        )

    if config.train_multiclass_models:
        train_multiclass_audio_baselines(
            config=config,
            train=audio_train,
            val=audio_val,
            test=audio_test,
            augment_fn=augment_fn,
            embedding_fn_dict=embedding_fn_dict,
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
