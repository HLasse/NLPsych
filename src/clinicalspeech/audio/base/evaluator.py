"""Abstract base class for doing model evaluation of audio classification models"""


from collections.abc import Callable
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from wasabi import msg

from clinicalspeech.utils import PROJECT_ROOT


def evaluate_model(
    model: pl.LightningModule,
    data_loader: DataLoader,
    id2label: dict[int, str],
) -> pd.DataFrame:
    """Evaluate model on a split.

    Args:
        model (pl.LightningModule): Model to evaluate
        data_loader (DataLoader): Test dataloader
        id2label (dict[int, str]): Mapping from label id to label name
    """
    logits = torch.stack([model(x) for x, _ in data_loader])
    # convert logits to probabilities
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    # Getting top prediction per row
    # add extra dimension if no batch dimension
    if len(probs.shape) == 1:
        probs = probs[:, None]
    preds = probs.argmax(axis=1)
    # return predictions, confidence scores and probabilities as dataframe
    return pd.DataFrame(
        {
            "prediction": [id2label[pred] for pred in preds],
            "confidence": probs.max(axis=1),
            "probabilities": probs.tolist(),
        }
    )


def evaluate_model_on_splits(
    df: pd.DataFrame,
    model: pl.LightningModule,
    model_name: str,
    id2label: dict,
    target_class: str,
    is_baseline: bool,
    config: DictConfig,
    embedding_fn: Optional[Callable] = None,
    model_id: Optional[str] = None,
) -> None:
    """Run evaluation of pytorch-lightning baseline models on the different splits.
    Saves the results to a jsonl file.

    Args:
        df (pd.DataFrame): Dataframe with all data needed for evaluation. Should contain
            a column with the path to the audio file and a column with the label
            and optionally any other metadata.
        model (pl.LightningModule): pytorch-lightning model
        model_name (str): Name of the model. Used for naming the saved results.
        id2label (dict): Dictionary mapping the label ids to the labels. Used
            for converting the predictions to the label names.
        target_class (str): Name of the target class. Used as metadata for the saved
            results.
        is_baseline (bool): Whether the model is a baseline model. Used as metadata
            when saving the results.
        embedding_fn (Optional[Callable]): Function to calculate audio embeddings.
            Supply this for baseline models. Defaults to None.
        model_id (Optional[str]): HuggingFace model id of the wav2vec model.
            Supply this for wav2vec models. Defaults to None.
        config (DictConfig): Config file
    """
    model.eval()

    if embedding_fn is not None and model_id is not None:
        raise ValueError("Cannot use both embedding_fn and model_id")

    # load split and create dataloader
    for split in config.data.eval_splits:
        df_split = df[df[config.data.split_column_name] == split].reset_index(drop=True)
        if embedding_fn is not None:
            data_loader = _get_baseline_dataloader(
                model=model, df=df_split, embedding_fn=embedding_fn, config=config
            )
        elif model_id is not None:
            data_loader = _get_wav2vec_dataloader(
                model=model, df=df_split, model_id=model_id, config=config
            )
        else:
            raise ValueError("Must provide either embedding_fn or model_id")

        # get predictions
        results = evaluate_model(
            model=model, data_loader=data_loader, id2label=id2label
        )
        # add to dataframe
        df_split = pd.concat([df_split, results], axis=1)

        save_dir = PROJECT_ROOT / "model_predictions"
        save_dir.mkdir(exist_ok=True)

        # save results
        save_to_jsonl(
            df=df_split,
            save_dir=save_dir,
            save_name=f"{target_class}_{model_name}_{split}",
            split=split,
            num_labels=len(df_split[config.data.label_column_name].unique()),
            model_name=model_name,
            target_class=target_class,
            is_baseline=is_baseline,
        )


def save_to_jsonl(
    df: pd.DataFrame,
    save_dir: Path,
    save_name: str,
    split: str,
    num_labels: int,
    model_name: str,
    target_class: str,
    is_baseline: bool,
):
    """Save dataframe to jsonl file. Add metadata for easier plotting

    Args:
        df (pd.DataFrame): dataframe to save
        save_dir (Path): directory to save to
        save_name (str): name of the file to save to
        split (str): name of the split
        num_labels (int): number of labels
        model_name (str): name of the model
        target_class (str): name of the target class
        is_baseline (bool): whether the model is a baseline model
    """
    df["binary"] = num_labels == 2
    df["type"] = "audio"
    df["split"] = split
    df["model_name"] = model_name
    df["target_class"] = target_class
    df["is_baseline"] = 1 if is_baseline else 0

    df.to_json(save_dir / f"{save_name}.jsonl", lines=True, orient="records")
    msg.info(f"Saved results to {save_dir / save_name}.jsonl")


def _get_baseline_dataloader(
    model: pl.LightningModule,
    df: pd.DataFrame,
    embedding_fn: Callable,
    config: DictConfig,
) -> DataLoader:
    return model.get_data_loader(  # type: ignore
        dataframe=df,
        embedding_fn=embedding_fn,
        path_column=config.data.audio_path_column_name,
        label_column=config.data.label_column_name,
        max_frame_length=config.audio.max_frame_length,
        batch_size=config.audio.batch_size,
        num_workers=config.audio.num_workers,
        shuffle=False,
    )


def _get_wav2vec_dataloader(
    model: pl.LightningModule, df: pd.DataFrame, model_id: str, config: DictConfig
) -> DataLoader:
    return model.get_data_loader(  # type: ignore
        dataframe=df,
        model_id=model_id,
        path_column=config.data.audio_path_column_name,
        label_column=config.data.label_column_name,
        max_frame_length=config.audio.max_frame_length,
        batch_size=config.audio.batch_size,
        num_workers=config.audio.num_workers,
        shuffle=False,
    )
