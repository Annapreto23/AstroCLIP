# Hackathon 2025 – Data Pre-processing Guide

This repository prepares spectroscopic and imaging data for AstroCLIP experiments.

## Directory Layout

```
hackathon2025/
├── data/
│   ├── astroclip.ckpt         
│   ├── specformer.ckpt      
├── pre_processing.py
└── README.md
```

## Configuring Input Paths

Please make sure to download the corresponding modesl in hackathon/data/ because to run "pre_processing.py" you need:

```
    checkpoint_path = "hackathon2025/checkpoints/astroclip_spectrum_pretrained.ckpt"
    dataset_path = "hackathon2025/data/astroclip_dataset/"
```

## Running the Pipeline

From the project root:

```bash
python -m hackathon2025.pre_processing
```

## Analysis of the Embeddings   


### AstroCLIP Checkpoint Loader and Embedding Generator  


```bash
AstroCLIP_Ckpt_Loader_Embedding_Generator.ipynb
```

### MLP Regression for the 3 models  

```bash
MLP_Regression_AstroCLIP.ipynb
MLP_Regression-AstroPT.ipynb
MLP_Regression_AION.ipynb
```

