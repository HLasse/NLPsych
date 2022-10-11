from pathlib import Path
from typing import Union

import pandas as pd
from wasabi import msg

PROJECT_ROOT = Path(__file__).parents[2]


def load_splits(
    path: Union[str, Path], split_column_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all spltis

    Args:
        path (Union[str, Path]): Path to dataframe containing a column indicating
            splits
        split_column_name (str): Name of column containing split

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes
    """
    data = pd.read_csv(path)
    train_df = data[data[split_column_name == "train"]]
    val_df = data[data[split_column_name == "val"]]
    test_df = data[data[split_column_name == "test"]]
    # shuffle train data
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    return train_df, val_df, test_df


def subsample(train: pd.DataFrame, val: pd.DataFrame):
    """Subsample data for faster debugging"""
    msg.info("Debug mode: using 200 random samples")
    train = train.sample(200)
    val = val.sample(200)
    return train, val
