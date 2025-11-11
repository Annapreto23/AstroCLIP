"""Interactive Streamlit app to compute and visualise AstroCLIP embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from hackathon2025.tools.inference import (
    EmbeddingComputer,
    ParquetDataSource,
    StreamingDataSource,
    approx_distance_mpc,
)

try:
    import umap  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    umap = None


DEFAULT_CHECKPOINT = "hackathon2025/data/astroclip.ckpt"


@dataclass(frozen=True)
class RunConfig:
    source_type: str
    parquet_path: Optional[str]
    focus_high_z: bool
    checkpoint_path: str
    device: str
    sample_size: int
    image_size: int
    batch_size: int
    slice_length: int
    use_cache: bool


def _build_source(cfg: RunConfig) -> Tuple[StreamingDataSource | ParquetDataSource, str]:
    if cfg.source_type == "streaming":
        source = StreamingDataSource(
            sample_size=cfg.sample_size,
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
        )
        return source, "streaming_hf"

    if not cfg.parquet_path:
        raise ValueError("Parquet path is required when source type is parquet.")
    source = ParquetDataSource(
        parquet_path=cfg.parquet_path,
        focus_high_z=cfg.focus_high_z,
        sample_size=cfg.sample_size,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        enable_cache=cfg.use_cache,
    )
    return source, f"parquet::{cfg.parquet_path}"


@st.cache_resource(show_spinner=False)
def _get_embedder(checkpoint_path: str, device: str) -> EmbeddingComputer:
    return EmbeddingComputer(checkpoint_path, device)


def _tensor_to_image(tensor: Any) -> np.ndarray:
    """Convert CHW tensor in [-?, ?] to HWC float image in [0, 1] for plotting."""
    arr = torch.as_tensor(tensor).detach().cpu().float()
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D CHW tensor, received shape={tuple(arr.shape)}")
    arr_np = arr.numpy()
    if arr_np.shape[0] not in (1, 3):
        raise ValueError(f"Unsupported channel count: {arr_np.shape[0]}")
    arr_np = np.transpose(arr_np, (1, 2, 0))
    arr_min = np.min(arr_np)
    arr_max = np.max(arr_np)
    scale = arr_max - arr_min
    if scale < 1e-6:
        return np.clip(arr_np, 0.0, 1.0)
    normalised = (arr_np - arr_min) / scale
    return np.clip(normalised, 0.0, 1.0)


def _compute_pipeline(cfg: RunConfig) -> Dict[str, Any]:
    checkpoint = Path(cfg.checkpoint_path).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    source, source_label = _build_source(cfg)
    df = source.load().reset_index(drop=True)

    embedder = _get_embedder(str(checkpoint), cfg.device)
    embeddings = embedder.build_embeddings(
        df=df,
        batch_size=cfg.batch_size,
        slice_length=cfg.slice_length,
        source_signature=source.signature(),
        use_cache=cfg.use_cache,
    )

    pair_ids = df["pair_id"].to_numpy(dtype=np.int64) if "pair_id" in df.columns else np.arange(len(df))
    redshift = df["redshift"].to_numpy(dtype=np.float32)

    result_df = pd.DataFrame(
        {
            "pair_id": pair_ids,
            "redshift": redshift,
            "cosine_similarity": embeddings["cosine_similarity"],
            "pca_x": embeddings["joint_pca"][:, 0],
            "pca_y": embeddings["joint_pca"][:, 1],
        }
    )
    result_df["distance_mpc"] = result_df["redshift"].apply(approx_distance_mpc)
    if "targetid" in df.columns:
        result_df["targetid"] = df["targetid"].to_numpy()

    joint_embedding = 0.5 * (embeddings["image_embeddings"] + embeddings["spectrum_embeddings"])

    metadata = {
        "source": cfg.source_type,
        "source_descriptor": source_label,
        "sample_size": cfg.sample_size,
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "slice_length": cfg.slice_length,
        "checkpoint": str(checkpoint),
        "device": cfg.device,
        "use_cache": cfg.use_cache,
        "records": int(len(result_df)),
    }
    run_key = (
        f"{metadata['source_descriptor']}|{metadata['records']}|"
        f"{metadata['checkpoint']}|{metadata['device']}|"
        f"batch={cfg.batch_size}|slice={cfg.slice_length}"
    )

    return {
        "df_raw": df,
        "table": result_df,
        "embeddings": embeddings,
        "joint_embedding": joint_embedding,
        "metadata": metadata,
        "run_key": run_key,
    }


def _compute_umap(joint_embedding: np.ndarray, n_neighbors: int, min_dist: float) -> Optional[np.ndarray]:
    if umap is None:
        return None
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(joint_embedding)


def _render_overview(table: pd.DataFrame, metadata: Dict[str, Any]) -> None:
    st.success(f"Embeddings ready ({metadata['records']} pairs).")
    with st.expander("Run metadata", expanded=False):
        st.json(metadata)


def _render_projection_section(table: pd.DataFrame) -> None:
    st.subheader("PCA (joint embedding)")
    pca_fig = px.scatter(
        table,
        x="pca_x",
        y="pca_y",
        color="redshift",
        color_continuous_scale="Viridis",
        hover_data={
            "pair_id": True,
            "cosine_similarity": ":.3f",
            "redshift": ":.3f",
            "distance_mpc": ":.0f",
        },
        title="PCA projection of joint image/spectrum embeddings",
    )
    pca_fig.update_traces(marker={"size": 9, "opacity": 0.8})
    st.plotly_chart(pca_fig, use_container_width=True)


def _render_umap_section(table: pd.DataFrame, joint_embedding: np.ndarray) -> None:
    st.subheader("UMAP (joint embedding)")
    if umap is None:
        st.info("UMAP non disponible (pip install umap-learn pour activer cette section).")
        return

    with st.expander("Configuration UMAP", expanded=True):
        n_neighbors = int(st.slider("n_neighbors", min_value=5, max_value=100, value=15, step=1))
        min_dist = float(st.slider("min_dist", min_value=0.0, max_value=1.0, value=0.1, step=0.01))

    umap_coords = _compute_umap(joint_embedding, n_neighbors=n_neighbors, min_dist=min_dist)
    if umap_coords is None:
        st.error("UMAP n'est pas disponible.")
        return

    table_umap = table.copy()
    table_umap["umap_x"] = umap_coords[:, 0]
    table_umap["umap_y"] = umap_coords[:, 1]

    umap_fig = px.scatter(
        table_umap,
        x="umap_x",
        y="umap_y",
        color="redshift",
        color_continuous_scale="Plasma",
        hover_data={
            "pair_id": True,
            "cosine_similarity": ":.3f",
            "redshift": ":.3f",
        },
        title="UMAP projection of joint embeddings",
    )
    umap_fig.update_traces(marker={"size": 9, "opacity": 0.8})
    st.plotly_chart(umap_fig, use_container_width=True)


def _render_pair_inspector(table: pd.DataFrame, embeddings: Dict[str, np.ndarray], df_raw: pd.DataFrame) -> None:
    st.subheader("Inspection d'une paire")
    selected_pair = st.selectbox(
        "Choisir une paire",
        options=table["pair_id"],
        format_func=lambda pid: f"Pair {pid} — z={table.loc[table['pair_id'] == pid, 'redshift'].iloc[0]:.3f}",
    )
    row = table[table["pair_id"] == selected_pair].iloc[0]
    raw_row = df_raw[df_raw["pair_id"] == selected_pair].iloc[0]

    st.metric("Redshift", f"{row['redshift']:.3f}", delta=f"{row['distance_mpc']:.0f} Mpc")
    st.metric("Similarité cosinus", f"{row['cosine_similarity']:.3f}")

    idx = row.name
    flux = embeddings["flux"][idx]
    wavelength = embeddings["wavelength"][idx]

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(_tensor_to_image(raw_row["image"]), caption=f"Pair {selected_pair}")
    with col2:
        spectrum_df = pd.DataFrame({"Wavelength": wavelength, "Flux": flux})
        fig_spec = px.line(spectrum_df, x="Wavelength", y="Flux", title="Spectre")
        fig_spec.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_spec, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="AstroCLIP interactive embeddings", layout="wide")
    st.title("AstroCLIP — Visualisation interactive (hackathon)")

    st.sidebar.header("Configuration des données")
    source_type = st.sidebar.selectbox("Source des données", options=["streaming", "parquet"], index=0)
    parquet_path = None
    focus_high_z = False
    if source_type == "parquet":
        parquet_path = st.sidebar.text_input("Chemin parquet (hf:// supporté)", "")
        focus_high_z = st.sidebar.checkbox("Prioriser les hauts redshift", value=False)

    st.sidebar.header("Paramètres d'inférence")
    checkpoint = st.sidebar.text_input("Checkpoint AstroCLIP", value=DEFAULT_CHECKPOINT)
    device = st.sidebar.selectbox("Device", options=["cuda", "cpu"], index=0)
    sample_size = int(st.sidebar.number_input("Nombre d'échantillons", min_value=32, max_value=8192, value=512, step=32))
    image_size = int(st.sidebar.number_input("Taille des images", min_value=64, max_value=512, value=144, step=16))
    batch_size = int(st.sidebar.number_input("Batch size", min_value=8, max_value=256, value=64, step=8))
    slice_length = int(st.sidebar.number_input("Longueur des spectres", min_value=1024, max_value=8192, value=7700, step=256))
    use_cache = st.sidebar.checkbox("Réutiliser le cache local", value=True)

    run_button = st.sidebar.button("Lancer le calcul des embeddings", type="primary")

    if not run_button:
        st.info("Configurez les options puis cliquez sur 'Lancer le calcul des embeddings'.")
        return

    cfg = RunConfig(
        source_type=source_type,
        parquet_path=parquet_path if parquet_path else None,
        focus_high_z=focus_high_z,
        checkpoint_path=checkpoint,
        device=device,
        sample_size=sample_size,
        image_size=image_size,
        batch_size=batch_size,
        slice_length=slice_length,
        use_cache=use_cache,
    )

    try:
        with st.spinner("Chargement des données et calcul des embeddings..."):
            results = _compute_pipeline(cfg)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erreur lors du calcul: {exc}")
        return

    table = results["table"]
    embeddings = results["embeddings"]
    df_raw = results["df_raw"]
    joint_embedding = results["joint_embedding"]
    metadata = results["metadata"]

    _render_overview(table, metadata)
    _render_projection_section(table)
    _render_umap_section(table, joint_embedding)
    _render_pair_inspector(table, embeddings, df_raw)


if __name__ == "__main__":
    main()
