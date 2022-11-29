"""Common utilities for processing audio data."""

import pandas as pd
from omegaconf import DictConfig


def prepare_binary_data(
    config: DictConfig,
    df: pd.DataFrame,
    origin: str,
) -> pd.DataFrame:
    """Prepare data for binary classification. Subsets to only include data from
    the specified origin and maps controls to 0 and cases to 1.

    Args:
        config (DictConfig): Hydra config
        df (pd.DataFrame): Dataframe containing all data
        origin (str): Which data origin to keep (if data from multiple sources)

    Returns
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation sets
    """
    df = df[df[config.config.data.origin_column_name] == origin]

    mapping = {config.data.control_label_name: 0, origin: 1}
    df["label_id"] = df[config.data.label_column_name].replace(mapping)
    return df
