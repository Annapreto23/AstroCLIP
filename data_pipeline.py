"""Data loading and embedding computation utilities for the AstroCLIP Streamlit app."""

from __future__ import annotations

import hashlib
import io
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import EntryNotFoundError
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import fsspec
import h5py
from skimage.exposure import match_histograms
import torch
import torch.nn.functional as F
from PIL import Image
import io
import torchvision.transforms as T

from astroclip.data.datamodule import AstroClipCollator
from astroclip.models import AstroClipModel
from sklearn.decomposition import PCA

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "hackathon2025" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def zscore_image_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    if image_tensor.ndim not in {3, 4}:
        raise ValueError(f"Image tensor attendu de dimension 3 ou 4, reçu {image_tensor.shape}")
    if image_tensor.ndim == 4:
        # Traite un batch complet
        mean = image_tensor.mean(dim=(2, 3), keepdim=True)
        std = image_tensor.std(dim=(2, 3), keepdim=True, unbiased=False).clamp(min=1e-6)
        return (image_tensor - mean) / std
    mean = image_tensor.mean(dim=(1, 2), keepdim=True)
    std = image_tensor.std(dim=(1, 2), keepdim=True, unbiased=False).clamp(min=1e-6)
    return (image_tensor - mean) / std


def _cache_path(prefix: str, **kwargs: Any) -> Path:
    """Return a deterministic cache file path based on keyword arguments."""
    payload = json.dumps(kwargs, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return CACHE_DIR / f"{prefix}_{digest}"


def resolve_parquet_path(raw_path: str) -> str:
    """Resolve local or Hugging Face parquet paths to an accessible location."""
    if not raw_path:
        raise ValueError("Chemin parquet vide.")

    raw_path = raw_path.strip()

    if raw_path.startswith("hf://"):
        path_no_scheme = raw_path[len("hf://") :]
        if not path_no_scheme.startswith("datasets/"):
            raise ValueError("Chemin HuggingFace invalide (attendu hf://datasets/...).")

        parts = path_no_scheme.split("/")
        if len(parts) < 4:
            raise ValueError("Chemin HuggingFace incomplet (repo et fichier attendus).")

        repo_id = "/".join(parts[1:3])
        inner_path = "/".join(parts[3:])

        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=inner_path,
                repo_type="dataset",
                cache_dir=str(CACHE_DIR / "hf_cache"),
            )
        except EntryNotFoundError as exc:
            files = list_repo_files(repo_id=repo_id, repo_type="dataset")
            guesses = [f for f in files if inner_path.split("/")[-1] in f]
            hint = f" Exemples trouvés: {guesses[:5]}" if guesses else ""
            raise FileNotFoundError(f"{inner_path} introuvable sur Hugging Face.{hint}") from exc

    path_obj = Path(raw_path).expanduser()
    if not path_obj.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path_obj}")
    return str(path_obj)


def batch_to_records(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a batch from the streaming dataloader to a list of dict records."""
    records: List[Dict[str, Any]] = []
    spec_batch = batch["spectrum"]
    bsz = batch["image"].shape[0]

    target_batch = batch.get("targetid")

    for idx in range(bsz):
        spec_sample = spec_batch[idx]

        if isinstance(spec_sample, dict):
            flux = spec_sample["flux"].cpu().numpy()
            wavelength = spec_sample["wavelength"].cpu().numpy()
        else:
            spec_tensor = torch.as_tensor(spec_sample).cpu()
            if spec_tensor.ndim == 1:
                flux = spec_tensor.numpy()
                wavelength = np.linspace(0, 1, spec_tensor.shape[0])
            elif spec_tensor.ndim == 2:
                if spec_tensor.shape[0] == 2:
                    flux = spec_tensor[0].numpy()
                    wavelength = spec_tensor[1].numpy()
                elif spec_tensor.shape[1] == 2:
                    flux = spec_tensor[:, 0].numpy()
                    wavelength = spec_tensor[:, 1].numpy()
                elif spec_tensor.shape[1] == 1:
                    flux = spec_tensor[:, 0].numpy()
                    wavelength = np.linspace(0, 1, spec_tensor.shape[0])
                elif spec_tensor.shape[0] == 1:
                    flux = spec_tensor[0].numpy()
                    wavelength = np.linspace(0, 1, spec_tensor.shape[1])
                else:
                    raise ValueError(f"Format de spectre inattendu: {spec_tensor.shape}")
            else:
                raise ValueError(f"Spectre NDIM={spec_tensor.ndim} non pris en charge")

        target_value = int(target_batch[idx]) if target_batch is not None else -1

        image_tensor = batch["image"][idx].float()
        image_tensor = image_tensor.cpu()

        records.append(
            {
                "image": image_tensor,
                "redshift": float(batch["redshift"][idx]),
                "targetid": target_value,
                "spectrum": {"flux": flux, "wavelength": wavelength},
            }
        )

    return records


class DataSource(ABC):
    """Abstract loader returning DataFrames with image/spectrum/redshift pairs."""

    def __init__(self, sample_size: int, image_size: int, batch_size: int) -> None:
        self.sample_size = sample_size
        self.image_size = image_size
        self.batch_size = batch_size

    def load(self) -> pd.DataFrame:
        cache_path = _cache_path(
            "df",
            source=self.__class__.__name__,
            sample=self.sample_size,
            image=self.image_size,
            batch=self.batch_size,
            signature=self.signature(),
        ).with_suffix(".pkl")

        if cache_path.exists():
            return pd.read_pickle(cache_path)

        df = self._load_impl()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
        return df

    @abstractmethod
    def signature(self) -> str:
        """Identifier for cache invalidation."""

    @abstractmethod
    def _load_impl(self) -> pd.DataFrame:
        """Concrete loader implementation."""


class ParquetDataSource(DataSource):
    def __init__(self, parquet_path: str, focus_high_z: bool, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.parquet_path = parquet_path
        self.focus_high_z = focus_high_z

    def signature(self) -> str:
        return f"path={self.parquet_path}|focus={self.focus_high_z}"
    def load(self) -> pd.DataFrame:
        return self._load_impl()
    def _load_impl(self) -> pd.DataFrame:


        # ----------------------------
        # Parameters
        # ----------------------------
        patch_size = 144 
        legacy_ref_idx = 0
        legacy_image_dataset_url = "https://users.flatironinstitute.org/~flanusse/astroclip_desi.1.1.5.h5"

        # ----------------------------
        # Load legacy reference image
        # ----------------------------
        # with fsspec.open(legacy_image_dataset_url, "rb") as f:
        #     with h5py.File(f, "r") as h5f:
        #         for grp_name in sorted(h5f.keys()):
        #             grp = h5f[grp_name]
        #             if legacy_ref_idx < len(grp['images']):
        #                 legacy_ref = np.array(grp['images'][legacy_ref_idx])  # H,W,C
        #                 break
        #             else:
        #                 legacy_ref_idx -= len(grp['images'])
        # print(f"[OK] Legacy reference image loaded: {legacy_ref.shape}")

        # ----------------------------
        # Load parquet
        # ----------------------------
        resolved_path = resolve_parquet_path(self.parquet_path)
        df = pd.read_parquet(resolved_path)

        transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor()
        ])

        # ----------------------------
        # Preprocess RGB images
        # ----------------------------
        def preprocess_image(blob):
            img = Image.open(io.BytesIO(blob["bytes"])).convert("RGB")
            tensor = transform(img)  # (C,H,W)

            # # Resize to multiple of patch_size
            # H, W = tensor.shape[1], tensor.shape[2]
            # new_H = (H // patch_size) * patch_size
            # new_W = (W // patch_size) * patch_size
            # tensor = F.interpolate(tensor.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)

            # # Histogram match
            # img_np = tensor.permute(1,2,0).cpu().numpy()  # H,W,C
            # img_matched = match_histograms(img_np, legacy_ref, channel_axis=-1)
            # tensor = torch.from_numpy(img_matched).permute(2,0,1)

            return tensor

        if "image" not in df.columns:
            df["image"] = df["RGB_image"].apply(preprocess_image)

        # ----------------------------
        # Dataset-level z-score
        # ----------------------------
        # Calcule la moyenne/écart-type globaux sur toutes les images puis applique la même
        # normalisation à chaque pixel pour reproduire le schéma du papier (z-score global).
        # channel_sum = torch.zeros(3, dtype=torch.float64)
        # channel_sq_sum = torch.zeros(3, dtype=torch.float64)
        # pixel_count = 0

        # for tensor in df["image"]:
        #     if not isinstance(tensor, torch.Tensor):
        #         tensor = torch.as_tensor(tensor, dtype=torch.float32)
        #     tensor = tensor.float()
        #     channel_sum += tensor.sum(dim=(1, 2))
        #     channel_sq_sum += (tensor ** 2).sum(dim=(1, 2))
        #     pixel_count += tensor.shape[1] * tensor.shape[2]

        # if pixel_count == 0:
        #     raise ValueError("Impossible de calculer la normalisation: aucune image disponible.")

        # dataset_mean = (channel_sum / pixel_count)
        # dataset_var = (channel_sq_sum / pixel_count) - dataset_mean ** 2
        # dataset_std = torch.sqrt(dataset_var.clamp(min=1e-12))

        # dataset_mean = dataset_mean.to(torch.float32)
        # dataset_std = dataset_std.to(torch.float32).clamp(min=1e-6)

        # mean_broadcast = dataset_mean[:, None, None]
        # std_broadcast = dataset_std[:, None, None]

        def apply_dataset_zscore(tensor: torch.Tensor) -> torch.Tensor:
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor, dtype=torch.float32)
            tensor = tensor.float()
            return (tensor - mean_broadcast) / std_broadcast

        # df["image"] = df["image"].apply(apply_dataset_zscore)

        # ----------------------------
        # Check redshift
        # ----------------------------
        if "redshift" not in df.columns:
        
            raise ValueError("La colonne 'redshift' est absente du parquet.")
    
        df = df.dropna(subset=["redshift"]).reset_index(drop=True)

        # ----------------------------
        # Sampling
        # ----------------------------
        self.sample_size = len(df)
        if len(df) > self.sample_size:
            if self.focus_high_z:
                df = df.nlargest(self.sample_size, "redshift").reset_index(drop=True)
            else:
                df = df.sample(self.sample_size, random_state=42).sort_index().reset_index(drop=True)

        df["pair_id"] = np.arange(len(df))
        return df



class StreamingDataSource(DataSource):
    def signature(self) -> str:
        return "hf_train"

    def _load_impl(self) -> pd.DataFrame:
        dset = load_dataset("EiffL/AstroCLIP", streaming=True, split="train").with_format("torch")
        collator = AstroClipCollator(center_crop=self.image_size)
        loader = DataLoader(
            dset,
            batch_size=self.batch_size,
            collate_fn=collator,
            drop_last=False,
        )

        records: List[Dict[str, Any]] = []

        for batch in loader:
            records.extend(batch_to_records(batch))
            if len(records) >= self.sample_size:
                break

        if not records:
            raise RuntimeError("Impossible de récupérer des exemples depuis le stream Hugging Face.")

        df = pd.DataFrame(records[: self.sample_size])
        df["pair_id"] = np.arange(len(df))
        return df


class AstroClipPairDataset(Dataset):
    """Dataset qui tronque/pad les spectres et renvoie tenseurs prêts pour AstroCLIP."""

    def __init__(self, df: pd.DataFrame, slice_length: int = 1024) -> None:
        self.df = df.reset_index(drop=True)
        self.slice_length = slice_length

    def __len__(self) -> int:
        return len(self.df)

    def _pad_or_trim(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.tensor(array, dtype=torch.float32)
        if tensor.numel() < self.slice_length:
            pad_len = self.slice_length - tensor.numel()
            tensor = torch.cat([tensor, torch.zeros(pad_len)])
        else:
            tensor = tensor[: self.slice_length]
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        spec = row["spectrum"]

        if isinstance(spec, dict):
            flux_arr = np.asarray(spec["flux"])
            wave_arr = np.asarray(spec.get("wavelength"))
        else:
            spec_tensor = torch.as_tensor(spec)
            if spec_tensor.ndim == 1:
                flux_arr = spec_tensor.numpy()
                wave_arr = np.linspace(0, 1, spec_tensor.shape[0])
            elif spec_tensor.ndim == 2:
                if spec_tensor.shape[0] == 2:
                    flux_arr = spec_tensor[0].numpy()
                    wave_arr = spec_tensor[1].numpy()
                elif spec_tensor.shape[1] == 2:
                    flux_arr = spec_tensor[:, 0].numpy()
                    wave_arr = spec_tensor[:, 1].numpy()
                elif spec_tensor.shape[1] == 1:
                    flux_arr = spec_tensor[:, 0].numpy()
                    wave_arr = np.linspace(0, 1, spec_tensor.shape[0])
                elif spec_tensor.shape[0] == 1:
                    flux_arr = spec_tensor[0].numpy()
                    wave_arr = np.linspace(0, 1, spec_tensor.shape[1])
                else:
                    raise ValueError(f"Format de spectre inattendu: {spec_tensor.shape}")
            else:
                raise ValueError(f"Spectre NDIM={spec_tensor.ndim} non pris en charge")

        flux = self._pad_or_trim(flux_arr)
        wave = self._pad_or_trim(wave_arr)
        spectrum = flux.unsqueeze(-1)

        image_tensor = row["image"]
        if isinstance(image_tensor, torch.Tensor):
            img_tensor = image_tensor.detach().clone().float()
        else:
            img_tensor = torch.as_tensor(image_tensor, dtype=torch.float32)
        img_tensor = img_tensor

        redshift = torch.tensor(row["redshift"], dtype=torch.float32)

        return {
            "spectrum": spectrum,
            "image": img_tensor,
            "redshift": redshift,
            "wavelength": wave,
        }


def approx_distance_mpc(redshift: float) -> float:
    """Approximate cosmological distance in Mpc using Hubble law."""
    hubble_km_s_mpc = 70.0
    c_km_s = 299_792.458
    return (c_km_s / hubble_km_s_mpc) * redshift


class EmbeddingComputer:
    """Compute and persist AstroCLIP embeddings for image/spectrum pairs."""

    def __init__(self, checkpoint_path: str, device: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model: Optional[AstroClipModel] = None

    def _load_model(self) -> AstroClipModel:
        if self._model is None:
            try:
                model = AstroClipModel.load_from_checkpoint(self.checkpoint_path)
            except KeyError as exc:
                if "pytorch-lightning_version" not in str(exc):
                    raise
                ckpt_data = torch.load(self.checkpoint_path, map_location="cpu")
                if "pytorch-lightning_version" not in ckpt_data:
                    ckpt_data["pytorch-lightning_version"] = "2.3.3"
                    patched_dir = CACHE_DIR / "patched_checkpoints"
                    patched_dir.mkdir(exist_ok=True)
                    patched_path = patched_dir / (Path(self.checkpoint_path).name + ".patched")
                    torch.save(ckpt_data, patched_path)
                    model = AstroClipModel.load_from_checkpoint(str(patched_path))
                else:
                    raise
            model = model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    def _cache_file(
        self,
        batch_size: int,
        slice_length: int,
        source_signature: str,
        df_len: int,
    ) -> Path:
        return _cache_path(
            "embeddings",
            checkpoint=self.checkpoint_path,
            device=self.device,
            batch=batch_size,
            slice_length=slice_length,
            source=source_signature,
            size=df_len,
        ).with_suffix(".npz")

    def load_cached_embeddings(
        self,
        batch_size: int,
        slice_length: int,
        source_signature: str,
        df_len: int,
    ) -> Optional[Dict[str, Any]]:
        cache_path = self._cache_file(
            batch_size=batch_size,
            slice_length=slice_length,
            source_signature=source_signature,
            df_len=df_len,
        )
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            return {key: cached[key] for key in cached.files}
        return None

    def invalidate_cache_entry(
        self,
        batch_size: int,
        slice_length: int,
        source_signature: str,
        df_len: int,
    ) -> None:
        cache_path = self._cache_file(
            batch_size=batch_size,
            slice_length=slice_length,
            source_signature=source_signature,
            df_len=df_len,
        )
        if cache_path.exists():
            cache_path.unlink()

    def build_embeddings(
        self,
        df: pd.DataFrame,
        batch_size: int,
        slice_length: int,
        source_signature: str,
    ) -> Dict[str, Any]:
        cache_path = self._cache_file(
            batch_size=batch_size,
            slice_length=slice_length,
            source_signature=source_signature,
            df_len=len(df),
        )

        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            return {key: cached[key] for key in cached.files}

        dataset = AstroClipPairDataset(df, slice_length=slice_length)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=False)

        model = self._load_model()

        cos_sims: List[torch.Tensor] = []
        img_embeds: List[torch.Tensor] = []
        spec_embeds: List[torch.Tensor] = []
        flux_batches: List[torch.Tensor] = []
        wave_batches: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                image_tensor = batch["image"].to(self.device)
                spectrum_tensor = batch["spectrum"].to(self.device)

                image_embeddings = model(image_tensor, input_type="image")
                spectrum_embeddings = model(spectrum_tensor, input_type="spectrum")

                similarity = torch.nn.functional.cosine_similarity(
                    image_embeddings, spectrum_embeddings, dim=1
                )

                cos_sims.append(similarity.cpu())
                img_embeds.append(image_embeddings.cpu())
                spec_embeds.append(spectrum_embeddings.cpu())
                flux_batches.append(batch["spectrum"].squeeze(-1).cpu())
                wave_batches.append(batch["wavelength"].cpu())

        cos_sims_t = torch.cat(cos_sims).numpy()
        img_embeds_t = torch.cat(img_embeds).numpy()
        spec_embeds_t = torch.cat(spec_embeds).numpy()
        flux_t = torch.cat(flux_batches).numpy()
        wave_t = torch.cat(wave_batches).numpy()

        joint_embedding = 0.5 * (img_embeds_t + spec_embeds_t)

        pca = PCA(n_components=2, random_state=42).fit(joint_embedding)
        joint_pca = pca.transform(joint_embedding)
        explained_ratio = pca.explained_variance_ratio_

        payload = {
            "cosine_similarity": cos_sims_t,
            "image_embeddings": img_embeds_t,
            "spectrum_embeddings": spec_embeds_t,
            "joint_pca": joint_pca,
            "pca_variance": explained_ratio,
            "flux": flux_t,
            "wavelength": wave_t,
        }

        np.savez_compressed(cache_path, **payload)
        return payload


def clear_cache() -> None:
    """Remove cached DataFrame and embedding artifacts."""
    for file in CACHE_DIR.glob("*"):
        if file.is_file():
            file.unlink()


__all__ = [
    "ParquetDataSource",
    "StreamingDataSource",
    "EmbeddingComputer",
    "approx_distance_mpc",
    "resolve_parquet_path",
    "clear_cache",
    "CACHE_DIR",
]
