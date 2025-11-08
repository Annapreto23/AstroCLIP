"""Utilities to compare DESI training images with Euclid parquet images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass
class ImageDatasetSample:
    name: str
    tensors: List[torch.Tensor]


def _ensure_chw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Tensor must be CHW, got shape {tensor.shape}")
    if tensor.shape[0] not in (1, 3):
        return tensor
    return tensor.float()


def _laplacian_variance(image: torch.Tensor) -> float:
    image = _ensure_chw(image)
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    kernel = kernel.view(1, 1, 3, 3)
    variances = []
    for c in range(image.shape[0]):
        channel = image[c : c + 1].unsqueeze(0)
        response = F.conv2d(channel, kernel, padding=1)
        variances.append(response.var().item())
    return float(np.mean(variances))


def _flatten_channel(image: torch.Tensor, channel_idx: int) -> torch.Tensor:
    return image[channel_idx].reshape(-1)


def collect_samples(df: pd.DataFrame, sample_size: int, name: str) -> ImageDatasetSample:
    n = min(sample_size, len(df))
    subset = df.sample(n, random_state=42) if len(df) > n else df
    tensors: List[torch.Tensor] = []
    for _, row in subset.iterrows():
        image = row["image"]
        tensor = image if isinstance(image, torch.Tensor) else torch.tensor(image)
        tensors.append(tensor.float())
    return ImageDatasetSample(name, tensors)


def analyze_datasets(
    reference: ImageDatasetSample,
    target: ImageDatasetSample,
    max_pixels_per_channel: int = 200_000,
) -> Dict[str, object]:
    stats_records: List[Dict[str, object]] = []
    sharpness_records: List[Dict[str, object]] = []
    size_records: List[Dict[str, object]] = []
    hist_rows: List[pd.DataFrame] = []

    channel_names = ["c0", "c1", "c2"]

    datasets = [reference, target]

    for dataset in datasets:
        if not dataset.tensors:
            continue

        sizes = [tensor.shape[-2:] for tensor in dataset.tensors]
        heights, widths = zip(*sizes)
        size_records.append(
            {
                "dataset": dataset.name,
                "height_mean": np.mean(heights),
                "height_std": np.std(heights),
                "width_mean": np.mean(widths),
                "width_std": np.std(widths),
                "unique_sizes": len(set(sizes)),
            }
        )

        channel_count = dataset.tensors[0].shape[0]

        for channel_idx in range(channel_count):
            values = torch.cat([_flatten_channel(img, channel_idx) for img in dataset.tensors])
            if values.numel() > max_pixels_per_channel:
                perm = torch.randperm(values.numel())[:max_pixels_per_channel]
                values = values[perm]

            stats_records.append(
                {
                    "dataset": dataset.name,
                    "channel": channel_names[channel_idx] if channel_idx < len(channel_names) else f"c{channel_idx}",
                    "mean": float(values.mean()),
                    "std": float(values.std(unbiased=False)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )

            hist_rows.append(
                pd.DataFrame(
                    {
                        "value": values.numpy(),
                        "dataset": dataset.name,
                        "channel": channel_names[channel_idx]
                        if channel_idx < len(channel_names)
                        else f"c{channel_idx}",
                    }
                )
            )

        sharpness_values = [_laplacian_variance(img) for img in dataset.tensors]
        sharpness_records.append(
            {
                "dataset": dataset.name,
                "laplacian_var_mean": float(np.mean(sharpness_values)),
                "laplacian_var_std": float(np.std(sharpness_values)),
            }
        )

    stats_df = pd.DataFrame(stats_records)
    sharpness_df = pd.DataFrame(sharpness_records)
    size_df = pd.DataFrame(size_records)
    hist_df = pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame()

    recommendations: List[str] = []

    if not size_df.empty and size_df["unique_sizes"].max() > 1:
        recommendations.append("Homogénéiser la résolution : plusieurs tailles d'image détectées.")
    elif (
        len(size_df) == 2
        and abs(size_df.iloc[0]["height_mean"] - size_df.iloc[1]["height_mean"]) > 1
        or abs(size_df.iloc[0]["width_mean"] - size_df.iloc[1]["width_mean"]) > 1
    ):
        recommendations.append("Ajuster la remise à l'échelle pour aligner les FOV.")

    if len(sharpness_df) == 2:
        ref_sharp = sharpness_df.iloc[0]["laplacian_var_mean"]
        tgt_sharp = sharpness_df.iloc[1]["laplacian_var_mean"]
        if tgt_sharp > ref_sharp * 1.2:
            recommendations.append("Cible plus nette : envisager un flou gaussien pour rapprocher la PSF.")
        elif tgt_sharp < ref_sharp * 0.8:
            recommendations.append("Cible plus floue : vérifier la PSF ou appliquer un rehaussement de netteté.")

    if len(stats_df) >= 6:
        merged = stats_df.pivot_table(index="channel", columns="dataset", values="mean")
        if merged.shape[1] == 2:
            diff = (merged.iloc[:, 0] - merged.iloc[:, 1]).abs()
            if (diff > 0.05).any():
                recommendations.append("Les moyennes par canal diffèrent : appliquer un histogram matching.")

    return {
        "stats": stats_df,
        "sharpness": sharpness_df,
        "sizes": size_df,
        "hist_data": hist_df,
        "recommendations": recommendations,
    }
