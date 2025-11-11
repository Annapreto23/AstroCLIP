"""Streamlit app to visualise joint PCA of image and spectrum embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data(show_spinner=False)
def _load_npz(path_str: str) -> Dict[str, np.ndarray]:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    return payload


def _compute_joint_pca(
    image_embeddings: np.ndarray,
    spectrum_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image_embeddings.shape != spectrum_embeddings.shape:
        raise ValueError("Image and spectrum embeddings need identical shapes.")

    n_pairs = image_embeddings.shape[0]
    if n_pairs < 2:
        raise ValueError("At least two pairs are required to compute PCA.")

    combined = np.vstack([image_embeddings, spectrum_embeddings]).astype(np.float32)
    combined -= combined.mean(axis=0, keepdims=True)

    try:
        u, s, _ = np.linalg.svd(combined, full_matrices=False)
    except np.linalg.LinAlgError as exc:  # pragma: no cover
        raise RuntimeError(f"PCA failed (SVD did not converge): {exc}") from exc

    coords = u[:, :2] * s[:2]
    explained = (s**2) / max(combined.shape[0] - 1, 1)
    explained_ratio = explained[:2] / explained.sum()

    image_coords = coords[:n_pairs]
    spectrum_coords = coords[n_pairs:]
    return image_coords, spectrum_coords, explained_ratio


def main() -> None:
    st.set_page_config(page_title="AstroCLIP Joint PCA", layout="wide")
    st.title("AstroCLIP joint PCA viewer (hackathon)")

    npz_path = st.text_input(
        "Embedding file (.npz)",
        value="hackathon2025/data/embeddings_all.npz",
    ).strip()
    if not npz_path:
        st.info("Provide a .npz file generated par le script de calcul d'embeddings.")
        st.stop()

    try:
        payload = _load_npz(npz_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load embeddings: {exc}")
        st.stop()

    required_keys = {"image_embeddings", "spectrum_embeddings", "pair_id"}
    missing = required_keys - payload.keys()
    if missing:
        st.error(f"Missing keys in the npz file: {', '.join(sorted(missing))}")
        st.stop()

    image_embeddings = np.asarray(payload["image_embeddings"])
    spectrum_embeddings = np.asarray(payload["spectrum_embeddings"])
    pair_ids = np.asarray(payload["pair_id"])
    redshift = np.asarray(payload.get("redshift", np.full_like(pair_ids, fill_value=np.nan, dtype=float)))

    n_pairs = image_embeddings.shape[0]
    st.caption(f"Loaded {n_pairs} pairs with embedding dimension {image_embeddings.shape[1]}.")

    if n_pairs < 2:
        st.warning("Need at least two pairs to compute PCA.")
        st.stop()

    default_sample = int(min(n_pairs, 2000))
    sample_size = int(
        st.number_input(
            "Number of pairs to include",
            min_value=2,
            max_value=int(n_pairs),
            value=default_sample,
            step=1,
        )
    )
    seed = int(st.number_input("Random seed", min_value=0, value=0, step=1))

    if sample_size < n_pairs:
        rng = np.random.default_rng(seed)
        selected_idx = np.sort(rng.choice(n_pairs, size=sample_size, replace=False))
    else:
        selected_idx = np.arange(n_pairs)

    image_subset = image_embeddings[selected_idx]
    spectrum_subset = spectrum_embeddings[selected_idx]
    pair_subset = pair_ids[selected_idx]
    redshift_subset = redshift[selected_idx]

    try:
        image_coords, spectrum_coords, explained_ratio = _compute_joint_pca(
            image_subset,
            spectrum_subset,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not compute PCA: {exc}")
        st.stop()

    pair_distance = np.linalg.norm(image_coords - spectrum_coords, axis=1)
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    stats_col1.metric("Explained variance PC1", f"{explained_ratio[0]:.2%}")
    stats_col2.metric("Explained variance PC2", f"{explained_ratio[1]:.2%}")
    stats_col3.metric("Median image-spectrum distance", f"{np.median(pair_distance):.3f}")

    plot_df = pd.DataFrame(
        {
            "pair_id": np.concatenate([pair_subset, pair_subset]),
            "modality": np.concatenate(
                [np.full(pair_subset.shape, "image"), np.full(pair_subset.shape, "spectrum")]
            ),
            "pc1": np.concatenate([image_coords[:, 0], spectrum_coords[:, 0]]),
            "pc2": np.concatenate([image_coords[:, 1], spectrum_coords[:, 1]]),
            "redshift": np.concatenate([redshift_subset, redshift_subset]),
        }
    )

    hover_dict = {
        "pair_id": True,
        "modality": True,
        "redshift": ":.3f",
    }
    scatter = px.scatter(
        plot_df,
        x="pc1",
        y="pc2",
        color="modality",
        symbol="modality",
        hover_data=hover_dict,
        title="PCA projection of image and spectrum embeddings",
    )
    scatter.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0)))

    highlight_options = ["None"] + [str(pid) for pid in pair_subset]
    highlight_choice = st.selectbox("Highlight a pair", options=highlight_options, index=0)
    if highlight_choice != "None":
        selected_pair = int(highlight_choice)
        mask = pair_subset == selected_pair
        if mask.any():
            highlight_df = pd.DataFrame(
                {
                    "pc1": [image_coords[mask][0, 0], spectrum_coords[mask][0, 0]],
                    "pc2": [image_coords[mask][0, 1], spectrum_coords[mask][0, 1]],
                }
            )
            scatter.add_scatter(
                x=highlight_df["pc1"],
                y=highlight_df["pc2"],
                mode="lines+markers",
                line=dict(color="gray", width=2, dash="dot"),
                marker=dict(color="black", size=10),
                name=f"Pair {selected_pair}",
            )

    st.plotly_chart(scatter, use_container_width=True)

    with st.expander("Pair distances"):
        table_df = pd.DataFrame(
            {
                "pair_id": pair_subset,
                "distance": pair_distance,
                "redshift": redshift_subset,
            }
        ).sort_values("distance")
        st.dataframe(table_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
