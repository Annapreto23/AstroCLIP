"""Streamlit interface for exploring AstroCLIP galaxy pairs."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from data_pipeline import (
    CACHE_DIR,
    EmbeddingComputer,
    ParquetDataSource,
    StreamingDataSource,
    approx_distance_mpc,
    clear_cache,
)


def _tensor_to_image(array: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert CHW tensor to HWC numpy image."""
    if isinstance(array, torch.Tensor):
        tensor = array.detach().cpu()
    else:
        tensor = torch.tensor(array)

    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)

    return tensor.numpy()


def render_pair_detail(df: pd.DataFrame, embeddings: Dict[str, np.ndarray], pair_id: int, slice_length: int) -> None:
    """Display image and spectrum for selected pair."""
    row = df.iloc[pair_id]
    flux = embeddings["flux"][pair_id][:slice_length]
    wavelength = embeddings["wavelength"][pair_id][:slice_length]

    col_img, col_spec = st.columns([1, 1.2])
    with col_img:
        st.image(
            _tensor_to_image(row["image"]),
            caption=f"Pair {pair_id} — z={row['redshift']:.3f}",
            clamp=True,
        )
        if "targetid" in df.columns:
            st.caption(f"TargetID: {row['targetid']}")

    with col_spec:
        spectrum_df = pd.DataFrame({"Wavelength": wavelength, "Flux": flux})
        fig_spec = px.line(spectrum_df, x="Wavelength", y="Flux", title="Spectre (tronqué/padé)")
        fig_spec.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_spec, use_container_width=True)


def sidebar_controls() -> Dict[str, Any]:
    """Render sidebar and return configuration dictionary."""
    with st.sidebar:
        st.header("Paramètres")

        source_label = st.radio(
            "Source des données",
            ("Parquet / HuggingFace", "Dataset AstroCLIP (train)"),
            index=0,
        )

        sample_size = st.slider("Taille de l'échantillon", 64, 2048, 512, step=32)
        image_size = st.slider("Taille des images (px)", 96, 256, 144, step=16)
        batch_size = st.slider("Batch size", 8, 256, 64, step=8)
        slice_length = st.number_input("Longueur du spectre", min_value=256, max_value=8192, value=7700, step=256)

        parquet_path = ""
        focus_high_z = False
        if source_label == "Parquet / HuggingFace":
            parquet_path = st.text_input(
                "Chemin du parquet",
                value="hf://datasets/msiudek/astroPT_euclid_desi_dataset/data/train_batch_1-00000-of-00001.parquet",
            )
            focus_high_z = st.checkbox("Prioriser les galaxies éloignées (top redshift)", value=True)

        checkpoint_path = st.text_input(
            "Checkpoint AstroCLIP",
            value="outputs/astroclip_finetuned.ckpt",
        )

        device_options = ["cpu"]
        if torch.cuda.is_available():
            device_options.insert(0, "cuda")
        device = st.selectbox("Device pour les embeddings", device_options, index=0)

        if st.button("Vider le cache disque"):
            clear_cache()
            st.success(f"Cache vidé dans {CACHE_DIR}")
            st.experimental_rerun()

    return {
        "source_label": source_label,
        "sample_size": sample_size,
        "image_size": image_size,
        "batch_size": batch_size,
        "slice_length": slice_length,
        "parquet_path": parquet_path,
        "focus_high_z": focus_high_z,
        "checkpoint_path": checkpoint_path,
        "device": device,
    }


def main() -> None:
    st.set_page_config(page_title="Exploration des paires AstroCLIP", layout="wide")
    st.title("Visualiser les paires image/spectre AstroCLIP")
    st.caption(
        "Explorez la distribution des embeddings, détectez les correspondances atypiques, "
        "et inspectez les galaxies à fort redshift."
    )

    cfg = sidebar_controls()

    if cfg["source_label"] == "Parquet / HuggingFace" and not cfg["parquet_path"]:
        st.info("Renseignez un chemin vers un parquet pour démarrer.")
        return

    if not cfg["checkpoint_path"]:
        st.info("Renseignez le chemin du checkpoint AstroCLIP.")
        return

    if cfg["source_label"] == "Parquet / HuggingFace":
        data_source = ParquetDataSource(
            parquet_path=cfg["parquet_path"],
            focus_high_z=cfg["focus_high_z"],
            sample_size=cfg["sample_size"],
            image_size=cfg["image_size"],
            batch_size=cfg["batch_size"],
        )
    else:
        data_source = StreamingDataSource(
            sample_size=cfg["sample_size"],
            image_size=cfg["image_size"],
            batch_size=cfg["batch_size"],
        )

    try:
        df = data_source.load()
    except Exception as exc:
        st.error(f"Erreur pendant le chargement des données : {exc}")
        return

    if df.empty:
        st.warning("Aucun échantillon disponible avec ces paramètres.")
        return

    embedder = EmbeddingComputer(cfg["checkpoint_path"], cfg["device"])

    try:
        embeddings = embedder.build_embeddings(
            df=df,
            batch_size=cfg["batch_size"],
            slice_length=cfg["slice_length"],
            source_signature=data_source.signature(),
        )
    except Exception as exc:
        st.error(f"Erreur pendant le calcul des embeddings : {exc}")
        return

    df = df.copy()
    df["cosine_similarity"] = embeddings["cosine_similarity"]
    df["pca_x"] = embeddings["joint_pca"][:, 0]
    df["pca_y"] = embeddings["joint_pca"][:, 1]
    df["distance_mpc"] = df["redshift"].apply(approx_distance_mpc)

    evr = embeddings.get("pca_variance")

    redshift_min = float(df["redshift"].min())
    redshift_max = float(df["redshift"].max())
    z_range = st.slider(
        "Plage de redshift affichée",
        min_value=round(redshift_min, 3),
        max_value=round(redshift_max, 3),
        value=(round(redshift_min, 3), round(redshift_max, 3)),
        step=0.001,
    )
    filtered = df[df["redshift"].between(z_range[0], z_range[1])].reset_index(drop=True)

    if filtered.empty:
        st.warning("Aucune paire dans la plage de redshift sélectionnée.")
        return

    st.subheader("Vue globale des embeddings")
    if evr is not None:
        st.text(
            f"Explained variance ratio of the 2 PCs: {evr}\n"
            f"Total explained variance by the 2 PCs: {evr.sum():.3f}"
        )

    scatter_fig = px.scatter(
        filtered,
        x="pca_x",
        y="pca_y",
        color="redshift",
        color_continuous_scale="Viridis",
        hover_data={
            "pair_id": True,
            "redshift": ":.3f",
            "cosine_similarity": ":.3f",
            "distance_mpc": ":.0f",
        },
        title="Projection PCA du joint embedding (image + spectre)",
    )
    scatter_fig.update_traces(marker=dict(size=9, opacity=0.75, line=dict(width=0)))
    scatter_fig.update_layout(coloraxis_colorbar=dict(title="z"), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.subheader("Redshift vs similarité image/spectre")
    rs_fig = px.scatter(
        filtered,
        x="redshift",
        y="cosine_similarity",
        color="cosine_similarity",
        color_continuous_scale="Turbo",
        hover_data={"pair_id": True, "distance_mpc": ":.0f"},
        labels={"redshift": "Redshift", "cosine_similarity": "Similarité cosinus"},
        trendline="lowess",
        trendline_color_override="white",
    )
    rs_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(rs_fig, use_container_width=True)

    st.subheader("Inspection d'une paire")
    candidates = filtered.sort_values("redshift", ascending=False)
    default_pair = int(candidates.iloc[0]["pair_id"])
    selected_pair = st.selectbox(
        "Choisissez la paire à analyser",
        options=candidates["pair_id"],
        format_func=lambda pid: f"Pair {pid} — z={df.iloc[pid]['redshift']:.3f}",
        index=0 if default_pair in candidates["pair_id"].values else 0,
    )

    st.metric(
        label="Redshift",
        value=f"{df.iloc[selected_pair]['redshift']:.3f}",
        delta=f"{df.iloc[selected_pair]['distance_mpc']:.0f} Mpc estimés",
    )
    st.metric(
        label="Similarité cosinus image/spectre",
        value=f"{df.iloc[selected_pair]['cosine_similarity']:.3f}",
    )

    render_pair_detail(df, embeddings, selected_pair, cfg["slice_length"])



if __name__ == "__main__":
    main()
