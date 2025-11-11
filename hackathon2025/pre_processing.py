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
    splits = {
    'train_batch_1': 'data/train_batch_1-00000-of-00001.parquet',}
    df = pd.read_parquet("hf://datasets/msiudek/astroPT_euclid_desi_dataset/" + splits['train_batch_1'])
    print(df.columns)

    transform = T.Compose([
    T.Resize((144, 144)),  # H x W
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
        checkpoint_path="C:\\Users\\apo\\Bureau\\hackathon2025\\AstroCLIP\\hackathon2025\\data\\astroclip.ckpt"
    ).eval().cuda()

    with torch.no_grad():
        embeddings_image = model(batch['image'].to('cuda'), input_type='image')
    with torch.no_grad():
        embeddings_spectrum = model(batch['spectrum'].to('cuda'), input_type='spectrum')
        
    import torch.nn.functional as F


    cos_sim = F.cosine_similarity(embeddings_image, embeddings_spectrum, dim=1)

    cos_sim_np = cos_sim.cpu().numpy()

    print("Cosine similarity: min {:.3f}, max {:.3f}, mean {:.3f}".format(
        cos_sim_np.min(), cos_sim_np.max(), cos_sim_np.mean()
    ))


    low_sim_idx = cos_sim_np.argsort()[:5]

    print("Indices des paires potentiellement anormales:", low_sim_idx)

    for idx in low_sim_idx:
        img_tensor = batch['image'][idx].cpu()
        img = T.ToPILImage()(img_tensor)
        score = cos_sim_np[idx]
        plt.figure()
        plt.imshow(img)
        plt.title(f"Cosine similarity (image vs spectrum): {score:.3f}")
        plt.axis('off')
        plt.show()
