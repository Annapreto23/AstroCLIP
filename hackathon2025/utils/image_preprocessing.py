from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import fsspec
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.exposure import match_histograms

BAND_DEFAULT_ORDER: Sequence[str] = (
    "RGB_image",
    "VIS_image",
    "NISP_Y_image",
    "NISP_J_image",
    "NISP_H_image",
)


def _compute_patch_aligned_size(height: int, width: int, patch_size: int) -> tuple[int, int]:
    if patch_size <= 0:
        raise ValueError(f"patch_size doit être strictement positif, reçu {patch_size}")
    aligned_h = max(height // patch_size, 1) * patch_size
    aligned_w = max(width // patch_size, 1) * patch_size
    return aligned_h, aligned_w


def _to_float_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 2:
        array = array[..., np.newaxis]
    array = array.astype(np.float32, copy=False)
    max_val = array.max(initial=0.0)
    if max_val > 1.0:
        array /= max_val
    tensor = torch.from_numpy(array.transpose(2, 0, 1))
    return tensor


def _decode_blob_to_tensor(blob: Mapping[str, object]) -> Optional[torch.Tensor]:
    if "bytes" in blob:
        image = Image.open(io.BytesIO(blob["bytes"]))  # type: ignore[arg-type]
        array = np.asarray(image)
        return _to_float_tensor(array)
    if "array" in blob:
        array = np.asarray(blob["array"])  # type: ignore[index]
        return _to_float_tensor(array)
    return None


def _resize_tensor(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Tenseur image attendu de forme (C,H,W), reçu {tensor.shape}")
    batch = tensor.unsqueeze(0)
    resized = F.interpolate(batch, size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def _resize_reference(reference: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(reference.transpose(2, 0, 1)).unsqueeze(0)
    resized = F.interpolate(tensor.float(), size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).cpu().numpy()


def load_legacy_reference(source: str, index: int = 0) -> np.ndarray:
    if index < 0:
        raise ValueError("index doit être positif.")

    try:
        path = Path(source)
        if path.exists():
            with h5py.File(path, "r") as handle:
                return _read_legacy_index(handle, index)
    except OSError:
        pass

    with fsspec.open(source, "rb") as stream:
        with h5py.File(stream, "r") as handle:
            return _read_legacy_index(handle, index)


def _read_legacy_index(handle: h5py.File, index: int) -> np.ndarray:
    remaining = index
    for group_name in sorted(handle.keys()):
        dataset = handle[group_name]["images"]
        if remaining < len(dataset):
            array = np.asarray(dataset[remaining])
            if array.ndim != 3:
                raise ValueError(f"Image Legacy attendue 3D, reçu {array.shape}")
            return array.astype(np.float32)
        remaining -= len(dataset)
    raise IndexError(f"Index Legacy {index} hors limites.")


@dataclass
class EuclidImagePreprocessor:
    patch_size: int
    legacy_reference: np.ndarray
    bands: Sequence[str] = BAND_DEFAULT_ORDER
    output_band: str = "RGB_image"

    def __post_init__(self) -> None:
        if self.legacy_reference.ndim != 3:
            raise ValueError(f"legacy_reference doit être de forme (H,W,C), reçu {self.legacy_reference.shape}")
        self._cached_reference_size: Optional[tuple[int, int]] = None
        self._cached_reference: Optional[np.ndarray] = None

    def __call__(self, row: Mapping[str, object]) -> Image.Image:
        tensors = self._extract_band_tensors(row)
        if not tensors:
            raise ValueError("Aucune bande Euclid disponible pour le prétraitement.")

        sample_tensor = next(iter(tensors.values()))
        height, width = sample_tensor.shape[-2:]
        target_h, target_w = _compute_patch_aligned_size(height, width, self.patch_size)

        for key, tensor in list(tensors.items()):
            tensors[key] = _resize_tensor(tensor, target_h, target_w)

        matched = self._match_histograms(tensors, target_h, target_w)

        output = matched.get(self.output_band) or next(iter(matched.values()))
        array = output.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()

        if array.shape[2] == 1:
            array = array[:, :, 0]
            image = Image.fromarray((array * 255.0).clip(0, 255).astype(np.uint8), mode="L")
        else:
            image = Image.fromarray((array * 255.0).clip(0, 255).astype(np.uint8))
        return image

    def _extract_band_tensors(self, row: Mapping[str, object]) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        for band in self.bands:
            value = row.get(band)
            if isinstance(value, Mapping):
                tensor = _decode_blob_to_tensor(value)
                if tensor is not None:
                    tensors[band] = tensor
        if not tensors and "image_bytes" in row:
            blob = {"bytes": row["image_bytes"]}
            tensor = _decode_blob_to_tensor(blob)
            if tensor is not None:
                tensors["image_bytes"] = tensor
        return tensors

    def _match_histograms(
        self,
        tensors: MutableMapping[str, torch.Tensor],
        height: int,
        width: int,
    ) -> MutableMapping[str, torch.Tensor]:
        reference = self._get_reference(height, width)

        for key, tensor in list(tensors.items()):
            numpy_img = tensor.permute(1, 2, 0).cpu().numpy()
            if numpy_img.shape[2] == 1:
                matched = match_histograms(
                    numpy_img[:, :, 0],
                    reference[:, :, 0],
                    channel_axis=None,
                )
                matched = matched[..., np.newaxis]
            else:
                matched = match_histograms(
                    numpy_img,
                    reference,
                    channel_axis=-1,
                )
            tensors[key] = torch.from_numpy(matched.transpose(2, 0, 1)).float().clamp(0.0, 1.0)
        return tensors

    def _get_reference(self, height: int, width: int) -> np.ndarray:
        if self._cached_reference_size == (height, width) and self._cached_reference is not None:
            return self._cached_reference
        resized = _resize_reference(self.legacy_reference, height, width)
        resized = resized.astype(np.float32)
        max_val = resized.max(initial=0.0)
        if max_val > 0:
            resized /= max_val
        self._cached_reference = resized
        self._cached_reference_size = (height, width)
        return resized

