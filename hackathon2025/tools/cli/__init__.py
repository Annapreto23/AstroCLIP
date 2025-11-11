"""Command-line entrypoints for hackathon utilities."""

from .compute_embeddings import main as compute_embeddings_main  # noqa: F401
from .finetune_image_encoder import main as finetune_image_encoder_main  # noqa: F401
from .checkpoints import main as checkpoints_main  # noqa: F401

__all__ = [
    "compute_embeddings_main",
    "finetune_image_encoder_main",
    "checkpoints_main",
]
