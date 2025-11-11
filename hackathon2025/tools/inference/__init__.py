"""High-level data loading and embedding helpers for the hackathon."""

from .embeddings import (  # noqa: F401
    CACHE_DIR,
    AstroClipPairDataset,
    EmbeddingComputer,
    ParquetDataSource,
    StreamingDataSource,
    approx_distance_mpc,
    batch_to_records,
    clear_cache,
    resolve_parquet_path,
)

__all__ = [
    "CACHE_DIR",
    "AstroClipPairDataset",
    "EmbeddingComputer",
    "ParquetDataSource",
    "StreamingDataSource",
    "approx_distance_mpc",
    "batch_to_records",
    "clear_cache",
    "resolve_parquet_path",
]
