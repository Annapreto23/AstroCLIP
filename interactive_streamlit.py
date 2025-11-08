"""Interactive Streamlit app to compute and visualise AstroCLIP embeddings on demand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from data_pipeline import (
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
        export_path=None,
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

    joint_embedding = 0.5 * (
        embeddings["image_embeddings"] + embeddings["spectrum_embeddings"]
    )

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


def _compute_umap(
    joint_embedding: np.ndarray,
    n_neighbors: int,
    min_dist: float,
) -> Optional[np.ndarray]:
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


def _render_umap_section(
    table: pd.DataFrame,
    joint_embedding: np.ndarray,
    umap_cache: Dict[str, Any],
    n_neighbors: int,
    min_dist: float,
    run_key: str,
) -> None:
    st.subheader("UMAP (joint embedding)")
    if umap is None:
        st.info("Install the 'umap-learn' package to enable this projection.")
        return

    cache_key = f"{run_key}|neighbors={n_neighbors}|mindist={min_dist:.2f}"
    cached = umap_cache.get("value")
    if not cached or cached["key"] != cache_key:
        try:
            umap_embedding = _compute_umap(joint_embedding, n_neighbors, min_dist)
        except ValueError as exc:
            st.warning(f"UMAP could not be computed: {exc}")
            return
        umap_cache["value"] = {"key": cache_key, "embedding": umap_embedding}
    else:
        umap_embedding = cached["embedding"]

    umap_df = table.copy()
    umap_df["umap_x"] = umap_embedding[:, 0]
    umap_df["umap_y"] = umap_embedding[:, 1]

    umap_fig = px.scatter(
        umap_df,
        x="umap_x",
        y="umap_y",
        color="redshift",
        color_continuous_scale="Turbo",
        hover_data={
            "pair_id": True,
            "cosine_similarity": ":.3f",
            "redshift": ":.3f",
            "distance_mpc": ":.0f",
        },
        title="UMAP projection of joint image/spectrum embeddings",
    )
    umap_fig.update_traces(marker={"size": 9, "opacity": 0.8})
    st.plotly_chart(umap_fig, use_container_width=True)


def _render_similarity_section(table: pd.DataFrame) -> None:
    st.subheader("Cosine similarity overview")
    tabs = st.tabs(["Distribution", "Similarity vs redshift"])
    with tabs[0]:
        hist_fig = px.histogram(
            table,
            x="cosine_similarity",
            nbins=40,
            title="Cosine similarity distribution",
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    with tabs[1]:
        scatter_fig = px.scatter(
            table,
            x="redshift",
            y="cosine_similarity",
            color="cosine_similarity",
            color_continuous_scale="IceFire",
            title="Cosine similarity vs redshift",
        )
        st.plotly_chart(scatter_fig, use_container_width=True)


def _render_pair_inspector(result: Dict[str, Any]) -> None:
    st.subheader("Inspect a pair")
    df = result["df_raw"]
    table = result["table"]
    embeddings = result["embeddings"]

    options = table["pair_id"].tolist()
    if not options:
        st.info("No pairs available.")
        return

    selected_pair = st.selectbox("Select pair", options=options)
    row = table.loc[table["pair_id"] == selected_pair].iloc[0]
    idx = int(row.name)

    col_metrics = st.columns(2)
    col_metrics[0].metric("Redshift", f"{row['redshift']:.4f}", delta=f"{row['distance_mpc']:.0f} Mpc")
    col_metrics[1].metric("Cosine similarity", f"{row['cosine_similarity']:.4f}")

    img_tensor = df.iloc[idx]["image"]
    flux = embeddings["flux"][idx]
    wavelength = embeddings["wavelength"][idx]

    display_cols = st.columns([1, 1.3])
    with display_cols[0]:
        st.image(_tensor_to_image(img_tensor), caption=f"Pair {selected_pair}")
    with display_cols[1]:
        spectrum_df = pd.DataFrame({"Wavelength": wavelength, "Flux": flux})
        spectrum_fig = px.line(
            spectrum_df,
            x="Wavelength",
            y="Flux",
            title="Spectrum",
        )
        spectrum_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(spectrum_fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="AstroCLIP Live Embeddings", layout="wide")
    st.title("AstroCLIP live embeddings explorer")

    with st.sidebar:
        st.header("Configuration")
        checkpoint = st.text_input("Checkpoint (.ckpt path)", value=DEFAULT_CHECKPOINT)
        device = st.selectbox("Device", options=["cuda", "cpu"], index=0)
        sample_size = st.number_input("Sample size", min_value=8, max_value=4096, value=256, step=8)
        batch_size = st.number_input("Batch size", min_value=4, max_value=256, value=64, step=4)
        image_size = st.number_input("Image size", min_value=64, max_value=512, value=144, step=16)
        slice_length = st.number_input("Spectrum slice length", min_value=256, max_value=16384, value=7700, step=256)
        use_cache = st.checkbox("Use on-disk cache", value=True)

        source_choice = st.radio(
            "Data source",
            options=("Streaming (HuggingFace)", "Parquet file"),
        )
        focus_high_z = False
        parquet_path: Optional[str] = None
        if source_choice == "Parquet file":
            parquet_path = st.text_input("Parquet path", value="")
            focus_high_z = st.checkbox("Prioritise high redshift entries", value=False)
        else:
            st.caption("Streaming uses the public EiffL/AstroCLIP train split.")

        st.header("UMAP parameters")
        n_neighbors = st.slider("Neighbors", min_value=5, max_value=50, value=15)
        min_dist = st.slider("Min distance", min_value=0.0, max_value=0.9, value=0.1, step=0.05)

        run_button = st.button("Compute embeddings", type="primary", use_container_width=True)

    if run_button:
        cfg = RunConfig(
            source_type="streaming" if source_choice.startswith("Streaming") else "parquet",
            parquet_path=parquet_path if parquet_path else None,
            focus_high_z=focus_high_z,
            checkpoint_path=checkpoint,
            device=device,
            sample_size=int(sample_size),
            image_size=int(image_size),
            batch_size=int(batch_size),
            slice_length=int(slice_length),
            use_cache=use_cache,
        )
        try:
            with st.spinner("Computing embeddings..."):
                result = _compute_pipeline(cfg)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Computation failed: {exc}")
        else:
            st.session_state["astroclip_result"] = result
            st.session_state.setdefault("astroclip_umap_cache", {})
            st.success("Computation finished.")

    result: Optional[Dict[str, Any]] = st.session_state.get("astroclip_result")
    if not result:
        st.info("Configure your run and click 'Compute embeddings' to get started.")
        return

    _render_overview(result["table"], result["metadata"])
    _render_projection_section(result["table"])
    _render_umap_section(
        table=result["table"],
        joint_embedding=result["joint_embedding"],
        umap_cache=st.session_state.setdefault("astroclip_umap_cache", {}),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        run_key=result["run_key"],
    )
    _render_similarity_section(result["table"])
    _render_pair_inspector(result)


if __name__ == "__main__":
    main()

