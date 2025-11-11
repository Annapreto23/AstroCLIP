"""Utility functions to plot AstroCLIP training logs."""

from __future__ import annotations

import re
from typing import Iterable, Tuple

import matplotlib.pyplot as plt


def _parse_logs(log_lines: Iterable[str]) -> Tuple[list[int], list[float], list[float], list[float], list[float]]:
    epochs, train_loss, val_loss, train_cos, val_cos = [], [], [], [], []
    pattern = re.compile(
        r"\[Epoch (\d+)/\d+\] train_loss=([-\d.]+) \| train_cosine=([-\d.]+) "
        r"\| val_loss=([-\d.]+) \| val_cosine=([-\d.]+)"
    )
    for line in log_lines:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            train_cos.append(float(match.group(3)))
            val_loss.append(float(match.group(4)))
            val_cos.append(float(match.group(5)))
    return epochs, train_loss, val_loss, train_cos, val_cos


def plot_training_curves(log_text: str) -> None:
    """Parse training logs and display loss/cosine similarity curves."""
    lines = log_text.strip().splitlines()
    epochs, train_loss, val_loss, train_cos, val_cos = _parse_logs(lines)
    if not epochs:
        raise ValueError("Aucun log valide détecté.")

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_cos, label="Train Cosine", marker="o")
    plt.plot(epochs, val_cos, label="Val Cosine", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Similarity")
    plt.title("Training and Validation Cosine Similarity")
    plt.grid(True)
    plt.legend()
    plt.show()
