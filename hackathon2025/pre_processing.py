import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import torchvision.transforms as T
import matplotlib.pyplot as plt
from astroclip.data.datamodule import AstroClipCollator
from astroclip.models import AstroClipModel
import numpy as np
from tqdm import tqdm


class AstroDFDataset(Dataset):
    def __init__(self, df, slice_length=1024):
        self.df = df
        self.slice_length = slice_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        flux = torch.tensor(row["spectrum"]["flux"], dtype=torch.float32)
        wave = torch.tensor(row["spectrum"]["wavelength"], dtype=torch.float32)

        if len(flux) < self.slice_length:
            pad_len = self.slice_length - len(flux)
            flux = torch.cat([flux, torch.zeros(pad_len)])
            wave = torch.cat([wave, torch.zeros(pad_len)])
        else:
            flux = flux[:self.slice_length]
            wave = wave[:self.slice_length]

        spectrum = flux.unsqueeze(-1)         

        img_tensor = row["img_tensor"]
        redshift = torch.tensor(row["redshift"], dtype=torch.float32)

        return {
            "spectrum": spectrum,
            "image": img_tensor,
            "redshift": redshift,
            "wavelength": wave,               
        }


if __name__ == "__main__":

    checkpoint_path = "hackathon2025/checkpoints/astroclip_spectrum_pretrained.ckpt"
    dataset_path = "hackathon2025/data/astroclip_dataset/"
    splits = {
    'train_batch_1': 'data/train_batch_1-00000-of-00001.parquet',}
    df = pd.read_parquet(dataset_path + splits['train_batch_1'])

    transform = T.Compose([
    T.Resize((256, 256)),  # H x W
    T.ToTensor(),           
    ])
    df['img_tensor'] = df['RGB_image'].apply(
    lambda x: transform(Image.open(io.BytesIO(x['bytes'])).convert("RGB"))
    )
    dataset = AstroDFDataset(df)

    dloader = DataLoader(
        dataset,
        batch_size=256,
        drop_last=True
    )
    batch = next(iter(dloader))

    model = AstroClipModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    ).eval().cuda()

    with torch.no_grad():
        embeddings = model(batch['spectrum'].to('cuda'), input_type='spectrum')

    for i in range(3):
        img = batch["image"][i].permute(1, 2, 0).cpu().numpy()
        flux = batch["spectrum"][i, :, 0].cpu().numpy()
        wavelength = batch["wavelength"][i].cpu().numpy()
        mask = wavelength > 0 

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.plot(wavelength[mask], flux[mask])
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")
        plt.title(f"Redshift: {batch['redshift'][i].item():.3f}")
        plt.tight_layout()
        plt.show()