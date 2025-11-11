"""Streamlit visualiser for precomputed AstroCLIP embeddings (hackathon edition)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from hackathon2025.tools.inference import approx_distance_mpc


def _tensor_to_image(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape {arr.shape}")
    if arr.shape[0] in (1, 3):
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Unexpected image channel dimension: {arr.shape[0]}")


def _load_embeddings(path: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, str]]:
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}

    metadata_raw = payload.pop("metadata", np.array("{}"))
    if hasattr(metadata_raw, "item"):
        metadata_raw = metadata_raw.item()
    metadata = json.loads(str(metadata_raw))

    df = pd.DataFrame(
        {
            "pair_id": payload["pair_id"].astype(int),
            "redshift": payload["redshift"].astype(float),
            "cosine_similarity": payload["cosine_similarity"].astype(float),
        }
    )
    df["pca_x"] = payload["joint_pca"][:, 0]
    df["pca_y"] = payload["joint_pca"][:, 1]
    df["distance_mpc"] = df["redshift"].apply(approx_distance_mpc)

    if "targetid" in payload:
        df["targetid"] = payload["targetid"].astype(int)

    embeddings = {
        "flux": payload["flux"],
        "wavelength": payload["wavelength"],
        "images": payload["images"],
        "pca_variance": payload.get("pca_variance"),
    }
    return df, embeddings, metadata


def main() -> None:
    st.set_page_config(page_title="AstroCLIP Embeddings Viewer", layout="wide")
    st.title("AstroCLIP Embeddings Viewer (hackathon)")

    npz_path = st.text_input("Chemin du fichier embeddings (.npz)", "")
    if not npz_path:
        st.info("Saisissez un fichier .npz généré via l'outil de calcul d'embeddings pour commencer.")
        return

    npz_file = Path(npz_path.strip()).expanduser()
    if not npz_file.exists():
        st.error(f"Fichier introuvable: {npz_file}")
        return

    try:
        df, embeddings, metadata = _load_embeddings(npz_file)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Impossible de charger le fichier: {exc}")
        return

    st.success(f"Embeddings chargés ({len(df)} paires).")
    st.json(metadata)

    st.subheader("Projection PCA")
    scatter = px.scatter(
        df,
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
        title="Joint embedding (image + spectre)",
    )
    scatter.update_traces(marker=dict(size=9, opacity=0.75, line=dict(width=0)))
    st.plotly_chart(scatter, use_container_width=True)

    st.subheader("Similarité vs Redshift")
    sim_fig = px.scatter(
        df,
        x="redshift",
        y="cosine_similarity",
        color="cosine_similarity",
        color_continuous_scale="Turbo",
        labels={"cosine_similarity": "Similarité cosinus"},
    )
    st.plotly_chart(sim_fig, use_container_width=True)

    st.subheader("Inspection d'une paire")
    selected_pair = st.selectbox(
        "Choisir une paire",
        options=df["pair_id"],
        format_func=lambda pid: f"Pair {pid} — z={df.loc[df['pair_id'] == pid, 'redshift'].iloc[0]:.3f}",
    )
    row = df[df["pair_id"] == selected_pair].iloc[0]

    st.metric("Redshift", f"{row['redshift']:.3f}", delta=f"{row['distance_mpc']:.0f} Mpc")
    st.metric("Similarité cosinus", f"{row['cosine_similarity']:.3f}")

    idx = row.name
    flux = embeddings["flux"][idx]
    wavelength = embeddings["wavelength"][idx]
    image = embeddings["images"][idx]

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(_tensor_to_image(image), caption=f"Pair {selected_pair}")
    with col2:
        spectrum_df = pd.DataFrame({"Wavelength": wavelength, "Flux": flux})
        fig_spec = px.line(spectrum_df, x="Wavelength", y="Flux", title="Spectre")
        fig_spec.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_spec, use_container_width=True)


if __name__ == "__main__":
    main()
