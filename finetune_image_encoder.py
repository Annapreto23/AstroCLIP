"""
Fine-tune AstroCLIP image encoder on the huggingface parquet dataset.

Usage:
    python finetune_image_encoder.py \
  --parquet-path hf://datasets/msiudek/astroPT_euclid_desi_dataset/data/train_batch_1-00000-of-00001.parquet \
  --checkpoint hackathon2025/data/astroclip.ckpt \
  --output-path outputs/image_encoder_ft.pt \
  --output-ckpt outputs/astroclip_image_ft.ckpt \
  --epochs 5 --batch-size 128 --device cuda --amp
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

from astroclip.models.astroclip import AstroClipModel, CLIPLoss
from data_pipeline import resolve_parquet_path, CACHE_DIR  # reuse helper / cache dir


def load_dataframe(parquet_path: str, image_size: int, max_samples: int | None) -> pd.DataFrame:
    resolved = resolve_parquet_path(parquet_path)
    df = pd.read_parquet(resolved)

    if max_samples is not None and max_samples < len(df):
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)

    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    if "image" not in df.columns:
        df["image"] = df["RGB_image"].apply(
            lambda blob: transform(Image.open(io.BytesIO(blob["bytes"])).convert("RGB"))
        )

    if "spectrum" not in df.columns:
        raise ValueError("La colonne 'spectrum' est absente du parquet.")

    if "redshift" not in df.columns:
        raise ValueError("La colonne 'redshift' est absente du parquet.")

    return df.reset_index(drop=True)


class AstroClipFineTuneDataset(Dataset):
    """Dataset that pads/trims spectra and returns tensors for training."""

    def __init__(self, df: pd.DataFrame, slice_length: int = 7700) -> None:
        self.df = df
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
        flux = np.asarray(spec["flux"])
        wave = spec.get("wavelength")
        if wave is None:
            wave = np.linspace(0, 1, len(flux), dtype=np.float32)
        wave = np.asarray(wave)

        flux_tensor = self._pad_or_trim(flux)
        wave_tensor = self._pad_or_trim(wave)

        spectrum = flux_tensor.unsqueeze(-1)
        image_tensor = row["image"]
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.tensor(image_tensor)

        return {
            "image": image_tensor,
            "spectrum": spectrum,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune AstroCLIP image encoder.")
    parser.add_argument("--parquet-path", required=True, help="Chemin vers le parquet (peut être hf://...).")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint AstroCLIP Lightning.")
    parser.add_argument("--output-path", required=True, help="Fichier de sortie pour l'image encoder fine-tuné.")
    parser.add_argument(
        "--output-ckpt",
        default=None,
        help="Chemin optionnel pour sauvegarder un nouveau checkpoint AstroCLIP complet.",
    )
    parser.add_argument("--device", default="cuda", help="Device d'entraînement (cuda ou cpu).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--slice-length", type=int, default=7700)
    parser.add_argument("--image-size", type=int, default=144)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Activer l'entrainement AMP.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print("Chargement du dataset...")
    df = load_dataframe(args.parquet_path, args.image_size, args.max_samples)
    dataset = AstroClipFineTuneDataset(df, slice_length=args.slice_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print("Chargement du modèle AstroCLIP...")
    model = AstroClipModel.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    image_encoder = model.image_encoder
    spectrum_encoder = model.spectrum_encoder

    for param in spectrum_encoder.parameters():
        param.requires_grad = False
    spectrum_encoder.eval()

    optimizer = torch.optim.AdamW(
        (p for p in image_encoder.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    criterion = CLIPLoss()

    logit_scale = getattr(model.hparams, "logit_scale", 15.5)
    temperature = getattr(model.hparams, "temperature", 15.5)

    print("Début du fine-tuning…")
    for epoch in range(1, args.epochs + 1):
        image_encoder.train()
        running_loss = 0.0
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            spectrum = batch["spectrum"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                image_features = image_encoder(images)
                with torch.no_grad():
                    spectrum_features = spectrum_encoder(spectrum)

                loss = criterion(image_features, spectrum_features, temperature)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"[Epoch {epoch}/{args.epochs}] loss: {epoch_loss:.4f}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "image_encoder_state_dict": image_encoder.state_dict(),
            "checkpoint": args.checkpoint,
            "parquet_path": args.parquet_path,
            "slice_length": args.slice_length,
            "temperature": temperature,
            "logit_scale": logit_scale,
            "cache_dir": str(CACHE_DIR),
        },
        output_path,
    )
    print(f"Fine-tuning terminé. Poids sauvegardés dans {output_path}")



if __name__ == "__main__":
    main()
    import torch
    from pathlib import Path
    from astroclip.models.astroclip import AstroClipModel

    original_ckpt = "hackathon2025/data/astroclip.ckpt"
    finetuned_pt = "outputs/image_encoder_ft.pt"
    output_ckpt = "outputs/astroclip_finetuned.ckpt"

    model = AstroClipModel.load_from_checkpoint(original_ckpt)

    pt = torch.load(finetuned_pt, map_location="cpu")
    image_enc_weights = pt["image_encoder_state_dict"]

    model.image_encoder.load_state_dict(image_enc_weights)

    ckpt = {
        "state_dict": model.state_dict(),
        "hyper_parameters": model.hparams,
    }

    torch.save(ckpt, output_ckpt)
    print("Nouveau CKPT sauvegardé dans :", output_ckpt)
