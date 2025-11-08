"""
Fine-tune AstroCLIP image encoder(s) on one or several huggingface parquet datasets.

Example:
    python finetune_image_encoder.py \
        --parquet-path hf://datasets/msiudek/astroPT_euclid_desi_dataset/data/train_batch_1-00000-of-00001.parquet \
        --checkpoint hackathon2025/data/astroclip.ckpt \
        --output-path outputs/image_encoder_ft.pt \
        --output-ckpt outputs/astroclip_image_ft.ckpt \
        --epochs 5 --batch-size 128 --device cuda --amp
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astroclip.models.astroclip import AstroClipModel, CLIPLoss
from astroclip.data.datamodule import AstroClipCollator
from data_pipeline import CACHE_DIR, resolve_parquet_path
from data_pipeline import ParquetDataSource

def zscore_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Image tensor attendu de dimension 3 (C, H, W), obtenu {tensor.shape}")
    mean = tensor.mean(dim=(1, 2), keepdim=True)
    std = tensor.std(dim=(1, 2), keepdim=True, unbiased=False).clamp(min=1e-6)
    return (tensor - mean) / std


class ToTensorZScore:
    def __init__(self) -> None:
        self.to_tensor = T.ToTensor()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        tensor = self.to_tensor(image)
        return tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from typing import Iterable, Optional
import pandas as pd
from data_pipeline import ParquetDataSource

def load_dataframe(
    parquet_paths: Iterable[str],
    max_samples: Optional[int],
    seed: int,
    focus_high_z: bool = False, batch_size: int = 128,
) -> pd.DataFrame:
    frames = []
    for path in parquet_paths:
        ds = ParquetDataSource(
    parquet_path=path,
    focus_high_z=True,
    sample_size=max_samples,
    image_size=144,
    batch_size=batch_size
)

        df = ds.load() 
        frames.append(df)

    if not frames:
        raise ValueError("Aucun parquet n'a été chargé. Vérifiez les chemins fournis.")

    df = pd.concat(frames, ignore_index=True)
    print(df.columns)

    if max_samples is not None and max_samples < len(df):
        df = df.sample(max_samples, random_state=seed).reset_index(drop=True)

    # Vérifie les colonnes critiques
    required_columns = ["spectrum", "redshift", "image"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est absente du DataFrame final.")

    return df.reset_index(drop=True)


def train_val_split(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if val_ratio <= 0 or len(df) < 2:
        return df, None

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    split_idx = int(len(df) * (1 - val_ratio))
    if split_idx <= 0 or split_idx >= len(df):
        return df, None

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def build_transforms(
    image_size: int,
    augment: bool,
) -> Tuple[T.Compose, T.Compose]:
    train_transforms: List = []
    eval_transforms: List = []

    if augment:
        train_transforms = [
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        ]
    else:
        train_transforms = [
            T.Resize(int(image_size * 1.05)),
            T.CenterCrop(image_size),
        ]

    eval_transforms = [
        T.Resize(int(image_size * 1.05)),
        T.CenterCrop(image_size),
    ]

    to_tensor_zscore = ToTensorZScore()
    train_transform = T.Compose(train_transforms)
    eval_transform = T.Compose(eval_transforms)
    return train_transform, eval_transform


class AstroClipFineTuneDataset(Dataset):
    """Dataset that pads/trims spectra, normalises them and returns tensors for training."""

    def __init__(
        self,
        df: pd.DataFrame,
        slice_length: int,
        image_transform: T.Compose,
        spectrum_norm: str = "zscore",
        include_wavelength: bool = False,
    ) -> None:
        self.df = df
        self.slice_length = slice_length
        self.image_transform = image_transform
        self.spectrum_norm = spectrum_norm
        self.include_wavelength = include_wavelength

    def __len__(self) -> int:
        return len(self.df)

    def _pad_or_trim(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(array, dtype=torch.float32)
        if tensor.numel() < self.slice_length:
            pad_len = self.slice_length - tensor.numel()
            tensor = torch.cat([tensor, torch.zeros(pad_len, dtype=torch.float32)])
        else:
            tensor = tensor[: self.slice_length]
        return tensor

    def _normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        spec = row["spectrum"]
        flux = np.asarray(spec["flux"])
        wavelength = spec.get("wavelength")
        if wavelength is None:
            wavelength = np.linspace(0, 1, len(flux), dtype=np.float32)
        else:
            wavelength = np.asarray(wavelength)

        flux_tensor = self._pad_or_trim(flux)
        wavelength_tensor = self._pad_or_trim(wavelength)
        wavelength_tensor = self._pad_or_trim(wavelength_tensor)

        spectrum = flux_tensor.unsqueeze(-1)
        if self.include_wavelength:
            spectrum = torch.stack([flux_tensor, wavelength_tensor], dim=-1)

        # Plus de transformation ici, l'image est déjà prête
        image_tensor = row["image"]

        return {
            "image": image_tensor,
            "spectrum": spectrum,
        }


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if total_steps <= 0:
        return base_lr

    if warmup_steps > 0 and step <= warmup_steps:
        lr = base_lr * float(step) / float(max(1, warmup_steps))
    else:
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def evaluate(
    image_encoder: torch.nn.Module,
    spectrum_encoder: torch.nn.Module,
    loader: DataLoader,
    criterion: CLIPLoss,
    device: torch.device,
    logit_scale: float,
) -> Dict[str, float]:
    image_encoder.eval()
    spectrum_encoder.eval()

    total_loss = 0.0
    total_cosine = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            spectrum = batch["spectrum"].to(device, non_blocking=True)

            image_features = image_encoder(images)
            spectrum_features = spectrum_encoder(spectrum)

            loss = criterion(image_features, spectrum_features, logit_scale)
            cosine = F.cosine_similarity(
                F.normalize(image_features, dim=-1),
                F.normalize(spectrum_features, dim=-1),
                dim=-1,
            ).mean()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_cosine += cosine.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return {"loss": float("nan"), "cosine": float("nan")}

    return {
        "loss": total_loss / total_samples,
        "cosine": total_cosine / total_samples,
    }


def export_full_checkpoint(model: AstroClipModel, output_ckpt: Path) -> None:
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": model.hparams,
        },
        output_ckpt,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune AstroCLIP image encoder.")
    parser.add_argument(
        "--parquet-path",
        dest="parquet_paths",
        nargs="+",
        required=True,
        help="Chemins vers les parquets (hf:// supporté).",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint AstroCLIP Lightning.")
    parser.add_argument("--output-path", required=True, help="Fichier de sortie du nouvel encodeur d'images.")
    parser.add_argument(
        "--output-ckpt",
        default=None,
        help="Chemin optionnel pour sauvegarder un checkpoint AstroCLIP complet avec l'encodeur fine-tuné.",
    )
    parser.add_argument("--device", default="cuda", help="Device d'entraînement (cuda ou cpu).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--slice-length", type=int, default=7700)
    parser.add_argument("--image-size", type=int, default=144)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction utilisée pour la validation (0 pour désactiver).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Activer l'entraînement AMP.")
    parser.add_argument("--disable-augment", action="store_true", help="Désactiver les augmentations d'images.")
    parser.add_argument(
        "--spectrum-norm",
        choices=["zscore", "minmax", "none"],
        default="none",
        help="Type de normalisation appliquée aux spectres (par défaut on laisse SpecFormer gérer le z-score).",
    )
    parser.add_argument(
        "--include-wavelength",
        action="store_true",
        help="Empiler flux et longueur d'onde sur deux canaux (assurez-vous que le spectrum encoder l'accepte).",
    )
    parser.add_argument("--warmup-steps", type=int, default=0, help="Nombre d'itérations de warmup pour le scheduler (0 = 10%% des steps).")
    parser.add_argument("--patience", type=int, default=3, help="Patience pour l'early stopping (epochs).")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Amélioration minimale requise pour reset la patience.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Clip des gradients (<=0 pour désactiver).")
    parser.add_argument("--accumulate-steps", type=int, default=1, help="Nombre d'itérations pour accumuler les gradients.")
    parser.add_argument("--log-interval", type=int, default=20, help="Intervalle d'affichage des logs (nombre de batchs).")
    parser.add_argument(
        "--unfreeze-backbone-blocks",
        type=int,
        default=0,
        help="Nombre de blocs DINO à dégeler (0 = ne pas dégeler la backbone).",
    )
    parser.add_argument(
        "--history-json",
        default=None,
        help="Chemin pour sauvegarder l'historique d'entraînement (JSON). Défaut: même dossier que output-path.",
    )
    parser.add_argument(
        "--history-plot",
        default=None,
        help="Chemin pour sauvegarder la courbe des pertes/metrics. Défaut: même dossier que output-path.",
    )
    return parser.parse_args()


def maybe_unfreeze_backbone(image_encoder: torch.nn.Module, num_blocks: int) -> None:
    if num_blocks <= 0:
        return
    backbone = getattr(image_encoder, "backbone", None)
    if backbone is None or not hasattr(backbone, "blocks"):
        print("Impossible de dégeler la backbone: attribut 'backbone.blocks' introuvable.")
        return
    blocks = list(backbone.blocks)
    num_blocks = min(num_blocks, len(blocks))
    if num_blocks <= 0:
        return
    for block in blocks[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    if hasattr(backbone, "norm"):
        for param in backbone.norm.parameters():
            param.requires_grad = True
    if hasattr(backbone, "patch_embed"):
        for param in backbone.patch_embed.parameters():
            param.requires_grad = True
    print(f"Dégel de {num_blocks} blocs de la backbone AstroDINO.")

import matplotlib.pyplot as plt
import random

import matplotlib.pyplot as plt
import random
from pathlib import Path

def _img_to_display(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor : torch.Tensor shape (C,H,W) or (H,W,C) on CPU
    Retourne HxWxC float np.ndarray dans [0,1] prêt pour imshow.
    """
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = np.asarray(img_tensor)

    # assure format C,H,W
    if img.ndim == 3 and img.shape[2] in (1,3):  # H,W,C -> C,H,W
        img = img.transpose(2, 0, 1)

    if img.ndim == 3:
        C, H, W = img.shape
    elif img.ndim == 2:
        # grayscale H,W -> C,H,W
        img = img[None, ...]
        C, H, W = img.shape
    else:
        raise ValueError(f"Image shape inattendue pour affichage: {img.shape}")

    # convertir en H,W,C pour matplotlib
    img_hwc = img.transpose(1, 2, 0).astype(np.float32)

    # Si canaux >3 (rare), coupe aux 3 premiers
    if img_hwc.shape[2] > 3:
        img_hwc = img_hwc[..., :3]

    # Min-max per image for display (works for z-score)
    mn = img_hwc.min()
    mx = img_hwc.max()
    img_disp = (img_hwc - mn) / (mx - mn + 1e-6)
    img_disp = np.clip(img_disp, 0.0, 1.0)

    # If single channel, repeat to 3 channels for nicer display
    if img_disp.shape[2] == 1:
        img_disp = np.repeat(img_disp, 3, axis=2)

    return img_disp


def visualize_batch(loader: DataLoader, num_samples: int = 3, out_dir: Optional[Path] = None):
    """
    Récupère un batch (premier batch of loader), sélectionne num_samples indices aléatoires,
    sauvegarde des images + spectres dans CACHE_DIR/visuals et affiche leurs chemins.
    """
    if out_dir is None:
        out_dir = Path("/pbs/home/a/astropart27/hackathon2025/AstroCLIP/outputs/")
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(loader))  # récupère un batch
    images = batch["image"]     # attendu [B, C, H, W] ou [B, H, W, C]
    spectra = batch["spectrum"] # attendu [B, L, ...] ou [B, L, C]

    B = images.shape[0]
    indices = random.sample(range(B), min(num_samples, B))

    saved_paths = []
    for i, idx in enumerate(indices):
        # image -> numpy HWC [0,1]
        img_tensor = images[idx]
        img_disp = _img_to_display(img_tensor)

        # spectrum -> 1D array (prendre flux si shape (L,1) ou (L,C))
        spec = spectra[idx]
        spec_np = spec.detach().cpu().squeeze()
        if spec_np.ndim > 1:
            # si spectre a canaux, prends le premier
            spec_np = spec_np[..., 0]
        spec_np = np.asarray(spec_np)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img_disp)
        axes[0].set_title(f"Image prétraitée - idx={idx}")
        axes[0].axis("off")

        axes[1].plot(spec_np)
        axes[1].set_title("Spectre (flux)")
        axes[1].set_xlabel("Index")
        axes[1].set_ylabel("Flux")

        fig.tight_layout()
        save_path = out_dir / f"sample_{i}_idx{idx}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        saved_paths.append(save_path)

    print("Visuals saved:")
    for p in saved_paths:
        print("  ", p)

    return saved_paths


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Utilisation du device: {device}")

    print("Chargement du dataset...")
    df = load_dataframe(args.parquet_paths, args.max_samples, seed=args.seed, batch_size=args.batch_size)
    train_df, val_df = train_val_split(df, args.val_ratio, seed=args.seed)
    print(f"  -> {len(train_df)} exemples pour l'entraînement")
    if val_df is not None:
        print(f"  -> {len(val_df)} exemples pour la validation")
    else:
        print("  -> Pas de validation (val_ratio <= 0 ou dataset trop petit)")

    train_transform, eval_transform = build_transforms(
        image_size=args.image_size,
        augment=not args.disable_augment,
    )

    train_dataset = AstroClipFineTuneDataset(
        train_df,
        slice_length=args.slice_length,
        image_transform=train_transform,
        spectrum_norm=args.spectrum_norm,
        include_wavelength=args.include_wavelength,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=AstroClipCollator(),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    # Visualisation aléatoire sur un batch du train_loader
    visualize_batch(train_loader, num_samples=3)

    val_loader: Optional[DataLoader] = None
    if val_df is not None:
        val_dataset = AstroClipFineTuneDataset(
            val_df,
            slice_length=args.slice_length,
            image_transform=eval_transform,
            spectrum_norm=args.spectrum_norm,
            include_wavelength=args.include_wavelength,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    if len(train_loader) == 0:
        raise ValueError("Le DataLoader d'entraînement est vide. Ajustez batch_size ou max_samples.")

    print("Chargement du modèle AstroCLIP...")
    model = AstroClipModel.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    image_encoder = model.image_encoder
    spectrum_encoder = model.spectrum_encoder

    for param in spectrum_encoder.parameters():
        param.requires_grad = False
    spectrum_encoder.eval()

    maybe_unfreeze_backbone(image_encoder, args.unfreeze_backbone_blocks)

    optimizer = torch.optim.AdamW(
        (p for p in image_encoder.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    base_lr = args.lr
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    criterion = CLIPLoss()

    temperature = float(getattr(model.hparams, "temperature", 15.5))

    logit_scale_hparam = getattr(model.hparams, "logit_scale", None)
    if logit_scale_hparam is not None:
        logit_scale = float(logit_scale_hparam)
    else:
        logit_scale_attr = getattr(model, "logit_scale", None)
        if isinstance(logit_scale_attr, torch.Tensor):
            logit_scale = float(logit_scale_attr.exp().item())
        elif isinstance(logit_scale_attr, (float, int)):
            logit_scale = float(math.exp(logit_scale_attr))
        else:
            logit_scale = temperature
    loss_scale = logit_scale if not math.isnan(logit_scale) else temperature

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else max(1, int(0.1 * total_steps))

    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0
    global_step = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "train_cosine": [],
        "val_loss": [],
        "val_cosine": [],
    }
    should_stop = False

    print("Début du fine-tuning…")
    for epoch in range(1, args.epochs + 1):
        image_encoder.train()
        running_loss = 0.0
        running_cosine = 0.0
        samples_seen = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            spectrum = batch["spectrum"].to(device, non_blocking=True)

            global_step += 1
            current_lr = adjust_learning_rate(optimizer, base_lr, global_step, total_steps, warmup_steps)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                image_features = image_encoder(images)
                if global_step % (args.log_interval * 5) == 0:
                    with torch.no_grad():
                        img_emb = F.normalize(image_features, dim=-1)
                        # compute stats
                        norms = img_emb.norm(dim=-1)
                        print(f"[DBG] emb mean_norm={image_features.detach().norm(dim=1).mean().item():.4f} std_norm={image_features.detach().norm(dim=1).std().item():.4f}")
                        # in-batch similarity vs texts
                        spec_emb = F.normalize(spectrum_features, dim=-1)
                        sims = img_emb @ spec_emb.t()
                        top1 = sims.argmax(dim=1)
                        inbatch_acc = (top1 == torch.arange(sims.size(0), device=top1.device)).float().mean().item()
                        print(f"[DBG] inbatch_acc={inbatch_acc:.4f} sims_mean={sims.mean().item():.4f} sims_std={sims.std().item():.4f}")


                with torch.no_grad():
                    spectrum_features = spectrum_encoder(spectrum)
                loss = criterion(image_features, spectrum_features, loss_scale)

            # ==== backward + accumulation ====
            effective_loss = loss / args.accumulate_steps
            scaler.scale(effective_loss).backward()

            # clip only when about to step
            if batch_idx % args.accumulate_steps == 0 or batch_idx == len(train_loader):
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn_utils.clip_grad_norm_(image_encoder.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)


            cosine = F.cosine_similarity(
                F.normalize(image_features.detach(), dim=-1),
                F.normalize(spectrum_features.detach(), dim=-1),
                dim=-1,
            ).mean()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_cosine += cosine.item() * batch_size
            samples_seen += batch_size

            if batch_idx % args.log_interval == 0:
                avg_loss = running_loss / samples_seen
                avg_cosine = running_cosine / samples_seen
                print(
                    f"Epoch {epoch}/{args.epochs} - Batch {batch_idx}/{len(train_loader)} "
                    f"| loss={avg_loss:.4f} | cosine={avg_cosine:.4f} | lr={current_lr:.2e}"
                )

        train_loss = running_loss / max(1, samples_seen)
        train_cosine = running_cosine / max(1, samples_seen)

        metrics_msg = f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} | train_cosine={train_cosine:.4f}"
        val_loss_value: Optional[float] = None
        val_cosine_value: Optional[float] = None

        if val_loader is not None:
            val_metrics = evaluate(image_encoder, spectrum_encoder, val_loader, criterion, device, loss_scale)
            val_loss_value = float(val_metrics["loss"])
            val_cosine_value = float(val_metrics["cosine"])
            metrics_msg += f" | val_loss={val_loss_value:.4f} | val_cosine={val_cosine_value:.4f}"

            improved = best_val_loss - val_loss_value > args.min_delta
            if improved:
                best_val_loss = val_loss_value
                best_state_dict = copy.deepcopy({k: v.detach().cpu() for k, v in image_encoder.state_dict().items()})
                patience_counter = 0
                metrics_msg += " <-- best"
            else:
                patience_counter += 1
                if patience_counter > args.patience:
                    should_stop = True
                    metrics_msg += " | early_stop"
        else:
            metrics_msg += " | pas d'évaluation"
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_cosine"].append(float(train_cosine))
        history["val_loss"].append(val_loss_value)
        history["val_cosine"].append(val_cosine_value)

        print(metrics_msg)
        if should_stop:
            print("Patience atteinte, arrêt anticipé.")
            break

    if best_state_dict is not None:
        image_encoder.load_state_dict(best_state_dict)
        print("Chargement des poids val_loss minimaux dans l'encodeur d'images.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_state_dict = {k: v.detach().cpu() for k, v in image_encoder.state_dict().items()}
    torch.save(
        {
            "image_encoder_state_dict": image_state_dict,
            "checkpoint": args.checkpoint,
            "parquet_paths": list(args.parquet_paths),
            "slice_length": args.slice_length,
            "temperature": temperature,
            "logit_scale": logit_scale,
            "cache_dir": str(CACHE_DIR),
        },
        output_path,
    )
    print(f"Fine-tuning terminé. Poids sauvegardés dans {output_path}")

    if history["epoch"]:
        history_json_path = Path(args.history_json) if args.history_json else output_path.with_name(output_path.stem + "_history.json")
        history_plot_path = Path(args.history_plot) if args.history_plot else output_path.with_name(output_path.stem + "_history.png")

        try:
            history_json_path.parent.mkdir(parents=True, exist_ok=True)
            with history_json_path.open("w", encoding="utf-8") as fp:
                json.dump(history, fp, ensure_ascii=False, indent=2)
            print(f"Historique sauvegardé dans {history_json_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Impossible de sauvegarder l'historique JSON ({exc}).")

        try:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            epochs = history["epoch"]
            plot_val_loss = [math.nan if v is None else v for v in history["val_loss"]]
            plot_val_cosine = [math.nan if v is None else v for v in history["val_cosine"]]

            axes[0].plot(epochs, history["train_loss"], label="train")
            if not all(math.isnan(x) for x in plot_val_loss):
                axes[0].plot(epochs, plot_val_loss, label="val")
            axes[0].set_title("CLIP loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(epochs, history["train_cosine"], label="train")
            if not all(math.isnan(x) for x in plot_val_cosine):
                axes[1].plot(epochs, plot_val_cosine, label="val")
            axes[1].set_title("Cosine similarity")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Cosine")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            fig.tight_layout()
            history_plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(history_plot_path, dpi=150)
            plt.close(fig)
            print(f"Courbes sauvegardées dans {history_plot_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Impossible de générer le plot d'historique ({exc}).")

    if args.output_ckpt:
        model.image_encoder.load_state_dict(image_state_dict)
        model = model.to("cpu")
        export_full_checkpoint(model, Path(args.output_ckpt))
        print(f"Checkpoint complet sauvegardé dans {args.output_ckpt}")


if __name__ == "__main__":
    main()
