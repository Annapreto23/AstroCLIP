import argparse
from pathlib import Path

import torch


def main(pt_path: str, ckpt_path: str, output_path: str) -> None:
    pt_payload = torch.load(pt_path, map_location="cpu")
    image_state = pt_payload["image_encoder_state_dict"]

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    for key, tensor in image_state.items():
        state_dict[f"image_encoder.{key}"] = tensor

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_file)
    print(f"[OK] nouveau checkpoint écrit dans {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Injecte un .pt d'image encoder dans un checkpoint Lightning.")
    parser.add_argument("--pt", required=True, help="Fichier .pt contenant image_encoder_state_dict.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint cible AstroCLIP (.ckpt).")
    parser.add_argument("--out", required=True, help="Chemin du checkpoint de sortie.")
    args = parser.parse_args()
    main(args.pt, args.ckpt, args.out)

    