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
        key[len(prefix):]: tensor
        for key, tensor in state_dict.items()
        if key.startswith(prefix)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrait l'image encoder d'un ckpt et l'injecte dans un autre ckpt."
    )
    parser.add_argument("--src-ckpt", required=True, help="Checkpoint source (.ckpt) dont on extrait l'image encoder.")
    parser.add_argument("--pt-out", required=True, help="Fichier .pt à générer pour stocker l'encodeur image.")
    parser.add_argument("--target-ckpt", required=True, help="Checkpoint cible dans lequel injecter l'encodeur.")
    parser.add_argument("--ckpt-out", required=True, help="Checkpoint de sortie combiné.")
    parser.add_argument(
        "--src-prefix",
        default="image_encoder.",
        help="Préfixe des poids image dans le checkpoint source (ex: 'image_encoder.' ou '' pour un AstroDINO pur).",
    )
    parser.add_argument(
        "--target-prefix",
        default="image_encoder.",
        help="Préfixe à utiliser dans le checkpoint cible (par défaut 'image_encoder.').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_state = extract_image_encoder(args.src_ckpt, prefix=args.src_prefix)
    save_pt(image_state, args.src_ckpt, args.pt_out)
    merge_into_ckpt(args.pt_out, args.target_ckpt, args.ckpt_out, target_prefix=args.target_prefix)



if __name__ == "__main__":
    main()