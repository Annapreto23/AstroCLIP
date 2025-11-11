# Hackathon 2025 - Tooling & Workflows

Ce dossier regroupe désormais l'ensemble des scripts et apps utilisés pendant le hackathon pour manipuler AstroCLIP, sans toucher au package `astroclip` historique.

## Arborescence

```
hackathon2025/
+-- data/                  # checkpoints & artefacts partagés
+-- tools/                 # nouveaux utilitaires (CLI, inference, apps)
    +-- cli/               # scripts exécutable via `python -m`
    +-- inference/         # chargement parquet/streaming + calcul embeddings
    +-- apps/              # dashboards Streamlit
    +-- analysis/          # QA images (statistiques, recommandations)
    +-- visualization/     # helpers pour tracer les logs d'entraînement
+-- pre_processing.py      # pré-traitement Euclid / DESI
+-- utils/                 # fonctions partagées par les notebooks
+-- README.md              # ce guide
```

## Environnements & dépendances

1. Créez un environnement Python 3.10+.
2. Installez les dépendances hackathon :
   ```bash
   pip install -r hackathon2025/requirements.txt
   ```
3. Vérifiez la présence des checkpoints dans `hackathon2025/data/` (`astroclip.ckpt`, `specformer.ckpt`).

## Scripts CLI (pour lancer depuis le dossier racine)

Les commandes ont été déplacées sous `hackathon2025.tools.cli` pour ne pas polluer le paquet `astroclip`.

### Calcul d'embeddings
```bash
python -m hackathon2025.tools.cli.compute_embeddings \
  --checkpoint hackathon2025/data/astroclip.ckpt \
  --output outputs/embeddings/sample.npz \
  --sample-size 1024 --batch-size 128 --image-size 144
```
Options utiles :
- `--parquet-path hf://datasets/...` pour un parquet HuggingFace personnalisé.
- `--streaming` force le dataset officiel en streaming même si un parquet est fourni.
- `--disable-cache` désactive le cache disque des DataFrames/embeddings.

### Fine-tuning de léencodeur image
```bash
python -m hackathon2025.tools.cli.finetune_image_encoder \
  --parquet-path hf://datasets/.../train.parquet \
  --checkpoint hackathon2025/data/astroclip.ckpt \
  --output-path outputs/image_encoder_ft.pt \
  --output-ckpt outputs/astroclip_image_ft.ckpt \
  --epochs 5 --batch-size 128 --device cuda --amp
```
Fonctionnalités : multi-parquets, priorisation haut redshift (`--focus-high-z`), scheduler cosinus + warmup, export JSON/PNG de l'historique.

### Gestion des checkpoints d'encodeur image
```bash
# Extraction
python -m hackathon2025.tools.cli.checkpoints extract \
  --src-ckpt hackathon2025/data/astroclip.ckpt \
  --pt-out outputs/image_encoder.pt

# Injection
python -m hackathon2025.tools.cli.checkpoints merge \
  --pt outputs/image_encoder.pt \
  --ckpt hackathon2025/data/astroclip.ckpt \
  --out outputs/astroclip_with_new_encoder.ckpt
```

## Inference & caches

Les classes `ParquetDataSource`, `StreamingDataSource` et `EmbeddingComputer` sont dans `hackathon2025/tools/inference/embeddings.py`. Elles gérent la résolution `hf://`, le cache disque (dans `hackathon2025/.cache/`) et produisent les embeddings/PCA. Pour vider les caches :

```python
from hackathon2025.tools.inference import clear_cache
clear_cache()
```

## Apps Streamlit

Toutes les interfaces ont été déplacées dans `hackathon2025.tools.apps` :

- `embedding_viewer.py` : inspection d'un fichier `.npz` (PCA, spectre, redshift).
- `interactive.py` : pipeline complet (chargement HF/parquet ? embeddings ? PCA/UMAP ? visu).
- `joint_pca.py` : comparaison image vs spectre via PCA conjointe.

Lancement depuis la racine :
```bash
streamlit run hackathon2025/tools/apps/interactive.py
```

## Analyse & visualisation

- `hackathon2025/tools/analysis/data_quality.py` : statistiques canal par canal, netteté (Laplacien), recommandations pour harmoniser les datasets.
- `hackathon2025/tools/visualization/training_logs.py` : fonction `plot_training_curves` pour parser les logs texte de fine-tuning.

## Notebooks

Les notebooks existants (`AstroCLIP_Ckpt_Loader_Embedding_Generator.ipynb`, `MLP_Regression_*.ipynb`) continuent de fonctionner ; mettez simplement à jour les imports vers les nouveaux modules, par exemple :

```python
from hackathon2025.tools.inference import EmbeddingComputer, ParquetDataSource
```

## Points clés

- Le package `astroclip` d'origine reste inchangé.
- Toute la logique hackathon (scripts, apps, helpers) est confinée dans `hackathon2025/tools/`.
- Pensez à versionner `outputs/` ou `.cache/` seulement si nécessaire (sinon ajoutez-les à `.gitignore`).

Bon hackathon !
