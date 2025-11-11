"""Checkpoint utilities for AstroCLIP image encoder weights."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch


def load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    """Charge un checkpoint Lightning ou un simple state_dict."""
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    return {"state_dict": checkpoint}


def extract_image_encoder(ckpt_path: str, prefix: str) -> Dict[str, torch.Tensor]:
    """Extrait les poids de l'encodeur image depuis un checkpoint."""
    checkpoint = load_checkpoint(ckpt_path)
    state_dict = checkpoint["state_dict"]

    image_state = {
        key[len(prefix) :]: tensor for key, tensor in state_dict.items() if key.startswith(prefix)
    }

    if not image_state:
        raise KeyError(
            f"Aucun poids dont la clé commence par '{prefix}' dans {ckpt_path}.\n"
            "Utilise --src-prefix \"\" si le checkpoint contient directement les poids AstroDINO."
        )
    return image_state


def save_pt(image_state: Dict[str, torch.Tensor], ckpt_src: str, pt_path: str) -> None:
    payload = {
        "image_encoder_state_dict": image_state,
        "checkpoint": ckpt_src,
    }
    pt_file = Path(pt_path)
    pt_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, pt_file)
    print(f"[OK] Encodeur image exporté vers {pt_file}")


def merge_into_ckpt(
    pt_path: str,
    ckpt_target: str,
    ckpt_out: str,
    target_prefix: str,
) -> None:
    payload = torch.load(pt_path, map_location="cpu")
    image_state = payload["image_encoder_state_dict"]

    checkpoint = load_checkpoint(ckpt_target)
    state_dict = checkpoint["state_dict"]

    for key, tensor in image_state.items():
        state_dict[f"{target_prefix}{key}"] = tensor

    out_file = Path(ckpt_out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_file)
    print(f"[OK] Nouveau checkpoint sauvegardé dans {out_file}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilitaires pour extraire/injecter l'encodeur image AstroCLIP.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extrait l'encodeur image vers un fichier .pt.")
    extract_parser.add_argument("--src-ckpt", required=True, help="Checkpoint source (.ckpt).")
    extract_parser.add_argument("--pt-out", required=True, help="Fichier .pt de sortie.")
    extract_parser.add_argument(
        "--src-prefix",
        default="image_encoder.",
        help="Préfixe des poids image dans le checkpoint source.",
    )

    merge_parser = subparsers.add_parser("merge", help="Injecte un encodeur image .pt dans un checkpoint.")
    merge_parser.add_argument("--pt", required=True, help="Fichier .pt contenant image_encoder_state_dict.")
    merge_parser.add_argument("--ckpt", required=True, help="Checkpoint cible AstroCLIP (.ckpt).")
    merge_parser.add_argument("--out", required=True, help="Chemin du checkpoint de sortie.")
    merge_parser.add_argument(
        "--target-prefix",
        default="image_encoder.",
        help="Préfixe à utiliser dans le checkpoint cible.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(args=argv)

    if args.command == "extract":
        image_state = extract_image_encoder(args.src_ckpt, prefix=args.src_prefix)
        save_pt(image_state, args.src_ckpt, args.pt_out)
    elif args.command == "merge":
        merge_into_ckpt(args.pt, args.ckpt, args.out, target_prefix=args.target_prefix)
    else:  # pragma: no cover - handled by argparse
        parser.error("Commande inconnue.")


if __name__ == "__main__":
    main()
