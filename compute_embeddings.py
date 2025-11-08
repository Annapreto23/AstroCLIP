"""Command-line utility to compute AstroCLIP embeddings and export them to a single NPZ file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from data_pipeline import EmbeddingComputer, ParquetDataSource, StreamingDataSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute AstroCLIP embeddings and export them to a NPZ file.")
    parser.add_argument(
        "--parquet-path",
        default=None,
        help="Chemin vers un parquet local ou hf://datasets/... (si absent, on utilisera le dataset AstroCLIP train).",
    )
    parser.add_argument(
        "--focus-high-z",
        action="store_true",
        help="Prioriser les galaxies à haut redshift lors de l'échantillonnage (parquet uniquement).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint AstroCLIP (.ckpt) contenant les encodeurs image/spectre.",
    )
    parser.add_argument("--device", default="cuda", help="Device pour l'inférence (ex: cuda, cpu).")
    parser.add_argument("--sample-size", type=int, default=512, help="Nombre d'échantillons à charger.")
    parser.add_argument("--image-size", type=int, default=144, help="Dimension des images (resize/crop).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size utilisé pour l'inférence.")
    parser.add_argument("--slice-length", type=int, default=7700, help="Longueur des spectres après padding/troncature.")
    parser.add_argument(
        "--output",
        required=True,
        help="Chemin du fichier .npz à créer (les dossiers parents seront créés au besoin).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Forcer l'utilisation du dataset AstroCLIP (streaming) même si --parquet-path est fourni.",
    )
    return parser.parse_args()


def build_data_source(args: argparse.Namespace):
    if args.parquet_path and not args.streaming:
        return ParquetDataSource(
            parquet_path=args.parquet_path,
            focus_high_z=args.focus_high_z,
            sample_size=args.sample_size,
            image_size=args.image_size,
            batch_size=args.batch_size,
        )
    return StreamingDataSource(
        sample_size=args.sample_size,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )


def tensor_to_chw_array(tensor: Any) -> np.ndarray:
    t = torch.as_tensor(tensor)
    if t.ndim == 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim != 3:
        raise ValueError(f"Image tensor attendu de dimension 3, obtenu {tuple(t.shape)}")
    return t.detach().cpu().numpy()


def pack_dataframe(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    images = np.stack([tensor_to_chw_array(img) for img in df["image"]])
    redshift = df["redshift"].to_numpy(dtype=np.float32)
    pair_ids = df.index.to_numpy(dtype=np.int64)
    payload: Dict[str, np.ndarray] = {
        "images": images,
        "redshift": redshift,
        "pair_id": pair_ids,
    }
    if "targetid" in df.columns:
        payload["targetid"] = df["targetid"].to_numpy(dtype=np.int64)
    return payload


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = build_data_source(args)
    df = source.load()

    embedder = EmbeddingComputer(args.checkpoint, args.device)
    embeddings = embedder.build_embeddings(
        df=df,
        batch_size=args.batch_size,
        slice_length=args.slice_length,
        source_signature=source.signature(),
        use_cache=False,
        export_path=None,
    )

    df_payload = pack_dataframe(df)

    metadata = {
        "parquet_path": args.parquet_path,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "sample_size": args.sample_size,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "slice_length": args.slice_length,
        "source_signature": source.signature(),
        "streaming": bool(args.streaming or not args.parquet_path),
    }

    payload = {
        **embeddings,
        **df_payload,
        "metadata": np.array(json.dumps(metadata)),
    }

    np.savez_compressed(output_path, **payload)
    print(f"Embeddings sauvegardés dans {output_path}")


if __name__ == "__main__":
    main()
