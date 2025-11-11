"""Streamlit apps bundled for the hackathon."""

from .embedding_viewer import main as embedding_viewer_main  # noqa: F401
from .interactive import main as interactive_main  # noqa: F401
from .joint_pca import main as joint_pca_main  # noqa: F401

__all__ = [
    "embedding_viewer_main",
    "interactive_main",
    "joint_pca_main",
]
