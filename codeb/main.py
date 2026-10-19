"""
Train VisionGPT2 using data/Images + captions.csv
Run with: modal run train.py
"""
import modal

app = modal.App("visiongpt2-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "timm",
        "transformers",
        "albumentations",
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "pillow",
    )
    # ships dataset.py, model.py, trainer.py so `from dataset import ...` works
    .add_local_python_source("dataset", "model", "trainer")
    # ships your captions.csv into the container
    .add_local_file("../data/captions.txt", remote_path="/root/data/captions.txt")
    # ships your images folder into the container
    .add_local_dir("../data/Images", remote_path="/root/data/Images")
)

# persists the trained checkpoint across runs
checkpoint_vol = modal.Volume.from_name("visiongpt2-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={"/root/captioner_ckpt": checkpoint_vol},
)
def train():
    from types import SimpleNamespace
    from pathlib import Path
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    import torch
    from dataset import Dataset, collate_fn, train_tfms, valid_tfms
    from trainer import Trainer

    # 1. Build your dataframe: columns = ['image', 'caption']
    import os
    df = pd.read_csv('/root/data/captions.txt')
    df['caption'] = df['caption'].astype(str).str.strip().str.lower()
    # Prepend the image directory path to filenames
    df['image'] = df['image'].apply(lambda x: '/root/data/Images/' + x.strip())
    # Filter out rows where the image file does not exist
    df = df[df['image'].apply(os.path.exists)]
    df = df.reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = Dataset(train_df, train_tfms)
    val_ds = Dataset(val_df, valid_tfms)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model_config = SimpleNamespace(
        vocab_size=50_257,
        embed_dim=768,
        num_heads=12,
        seq_len=1024,
        depth=12,
        attention_dropout=0.1,
        residual_dropout=0.1,
        mlp_ratio=4,
        mlp_dropout=0.1,
        emb_dropout=0.1,
    )

    train_config = SimpleNamespace(
        epochs=6,
        freeze_epochs_gpt=2,
        freeze_epochs_all=4,
        lr=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_path=Path('/root/captioner_ckpt'),
    )

    trainer = Trainer(model_config, train_config, (train_dl, val_dl))
    results = trainer.fit()
    checkpoint_vol.commit()
    return results


@app.local_entrypoint()
def main():
    results = train.remote()
    print(results)