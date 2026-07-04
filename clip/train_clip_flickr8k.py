"""
Train CLIP (your architecture, from model.py) on Flickr8k.
Weights initialized from OpenAI's pretrained ViT-B/32 checkpoint before training.
Data split train/test; after each epoch, inference test runs on held-out test data.

WHAT THIS SCRIPT DOES, IN ORDER:
  1. Loads your SimpleTokenizer (OpenAI's actual BPE tokenizer) and replicates
     CLIP's tokenize() logic: wrap each caption with <|startoftext|> / <|endoftext|>,
     truncate/pad to CONTEXT_LENGTH.
  2. Defines a Dataset that returns (image_tensor, token_tensor) pairs.
  3. Splits data into train/test sets (split by unique image, avoids caption leak).
  4. Instantiates YOUR CLIP class from model.py, then loads pretrained ViT-B/32
     weights into matching layers before training starts.
  5. Writes the training loop: forward -> contrastive loss -> backward -> optimizer step.
  6. After each epoch, runs inference on test set: computes test loss + retrieval accuracy.
  7. Saves your own trained weights to disk every epoch.

BEFORE RUNNING:
  - Put this file in the same folder as your model.py (so `from model import CLIP` works).
  - Check IMAGES_DIR / CAPTIONS_PATH below match your machine.
  - Check the caption.txt column names printed at the very start -- if they're not
    "image","caption", edit CAPTION_COL / IMAGE_COL below.
"""
import os
import hashlib
import urllib.request
import warnings
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import CLIP 
from simple_tokenizer import SimpleTokenizer
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent      
PROJECT_ROOT = SCRIPT_DIR.parent                          

IMAGES_DIR = str(PROJECT_ROOT / "data" / "Images")
CAPTIONS_PATH = str(PROJECT_ROOT / "data" / "captions.txt")
CHECKPOINT_DIR = str(PROJECT_ROOT / "checkpoints")

IMAGE_COL = "image"     
CAPTION_COL = "caption"  

CONTEXT_LENGTH = 77
BATCH_SIZE = 256
EPOCHS = 3
LEARNING_RATE = 0.01
TEST_FRACTION = 0.2      
SEED = 42            

# Model architecture size.
# NOTE: since we're now loading pretrained ViT-B/32 weights, config must match
# OpenAI's exact numbers -- otherwise shapes won't line up and weights won't load.
MODEL_CONFIG = dict(
    embed_dim=512,
    image_resolution=224,
    vision_layers=12,
    vision_width=768,
    vision_patch_size=32,
    context_length=CONTEXT_LENGTH,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



df = pd.read_csv(CAPTIONS_PATH)
print("Columns found in captions.txt:", df.columns.tolist())
df = df.rename(columns={IMAGE_COL: "image", CAPTION_COL: "caption"})
df["caption"] = df["caption"].astype(str)
print(f"Loaded {len(df)} (image, caption) rows")

# split by UNIQUE IMAGE, not row -- else same image's 5 captions leak across train/test
unique_images = df["image"].unique()
rng = torch.Generator().manual_seed(SEED)
perm = torch.randperm(len(unique_images), generator=rng).tolist()
split_idx = int(len(unique_images) * (1 - TEST_FRACTION))
train_images = set(unique_images[i] for i in perm[:split_idx])
test_images = set(unique_images[i] for i in perm[split_idx:])

train_df = df[df["image"].isin(train_images)].reset_index(drop=True)
test_df = df[df["image"].isin(test_images)].reset_index(drop=True)
print(f"Train: {len(train_images)} images / {len(train_df)} pairs")
print(f"Test:  {len(test_images)} images / {len(test_df)} pairs")

# 2. YOUR BPE TOKENIZER
tokenizer = SimpleTokenizer()
SOT_ID = tokenizer.encoder["<|startoftext|>"]
EOT_ID = tokenizer.encoder["<|endoftext|>"]
VOCAB_SIZE = len(tokenizer.encoder)

MODEL_CONFIG["vocab_size"] = VOCAB_SIZE
print(f"Vocab size (BPE): {VOCAB_SIZE}")

def encode_caption(text):
    tokens = [SOT_ID] + tokenizer.encode(text) + [EOT_ID]
    if len(tokens) > CONTEXT_LENGTH:
        tokens = tokens[:CONTEXT_LENGTH]
        tokens[-1] = EOT_ID
    result = torch.zeros(CONTEXT_LENGTH, dtype=torch.long)
    result[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return result

# 3. DATASET (now two loaders: train + test)
preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])

class Flickr8kCLIPDataset(Dataset):
    def __init__(self, dataframe, images_dir):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image"])
        image = Image.open(img_path).convert("RGB")
        image = preprocess(image)
        tokens = encode_caption(row["caption"])
        return image, tokens


train_dataset = Flickr8kCLIPDataset(train_df, IMAGES_DIR)
test_dataset = Flickr8kCLIPDataset(test_df, IMAGES_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, drop_last=True)
print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
print(f"Test:  {len(test_dataset)} samples, {len(test_loader)} batches")


# ---------------------------------------------------------------------------
# 4. BUILD MODEL + LOAD PRETRAINED WEIGHTS (NEW)
#
# Weights only -- checkpoint downloaded directly, no dependency on OpenAI's
# clip.py code (avoids import/package issues, keeps this purely a weights source).
# ---------------------------------------------------------------------------
model = CLIP(**MODEL_CONFIG).to(device)
print(f"Model built with {sum(p.numel() for p in model.parameters()):,} parameters")

_VIT_B_32_URL = ("https://openaipublic.azureedge.net/clip/models/"
                 "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
                 "ViT-B-32.pt")

def download_pretrained_checkpoint(url, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    target_path = os.path.join(cache_dir, filename)

    if os.path.exists(target_path):
        with open(target_path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == expected_sha256:
                print(f"Using cached checkpoint: {target_path}")
                return target_path
            else:
                warnings.warn("Cached checkpoint corrupted (hash mismatch) -- re-downloading")

    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as source, open(target_path, "wb") as out:
        total = int(source.info().get("Content-Length", 0))
        downloaded = 0
        while True:
            buf = source.read(1024 * 1024)
            if not buf:
                break
            out.write(buf)
            downloaded += len(buf)
            print(f"\r{downloaded/1e6:.1f}/{total/1e6:.1f} MB", end="")
    print()

    with open(target_path, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Downloaded checkpoint hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return target_path

print("Fetching OpenAI pretrained ViT-B/32 checkpoint (weights only)...")
checkpoint_path = download_pretrained_checkpoint(
    _VIT_B_32_URL, cache_dir=os.path.expanduser("~/.cache/clip")
)
pretrained_jit_model = torch.jit.load(checkpoint_path, map_location="cpu").eval()
pretrained_state = pretrained_jit_model.state_dict()

model_state = model.state_dict()
matched, skipped = 0, 0
for k, v in pretrained_state.items():
    if k in model_state and model_state[k].shape == v.shape:
        model_state[k] = v
        matched += 1
    else:
        skipped += 1
model.load_state_dict(model_state, strict=False)
print(f"Pretrained init: {matched} tensors matched/loaded, {skipped} skipped")


# ---------------------------------------------------------------------------
# 5. TRAINING LOOP + POST-EPOCH INFERENCE TEST (NEW)
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.2)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

for epoch in range(EPOCHS):
    # ---- TRAIN ----
    model.train()
    running_loss = 0.0

    for step, (images, texts) in enumerate(train_loader):
        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)
        labels = torch.arange(images.shape[0], device=device)
        loss = (loss_img(logits_per_image, labels) + loss_txt(logits_per_text, labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {step}/{len(train_loader)} | Loss {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    print(f"=== Epoch {epoch+1} finished. Avg train loss: {avg_train_loss:.4f} ===")

    # ---- INFERENCE TEST ON HELD-OUT DATA (NEW) ----
    model.eval()
    test_loss = 0.0
    correct_i2t, correct_t2i, total = 0, 0, 0
    with torch.no_grad():
        for images, texts in test_loader:
            images = images.to(device)
            texts = texts.to(device)
            labels = torch.arange(images.shape[0], device=device)

            logits_per_image, logits_per_text = model(images, texts)
            loss = (loss_img(logits_per_image, labels) + loss_txt(logits_per_text, labels)) / 2
            test_loss += loss.item()

            pred_i2t = logits_per_image.argmax(dim=-1)
            pred_t2i = logits_per_text.argmax(dim=-1)
            correct_i2t += (pred_i2t == labels).sum().item()
            correct_t2i += (pred_t2i == labels).sum().item()
            total += images.shape[0]

    avg_test_loss = test_loss / len(test_loader)
    acc_i2t = correct_i2t / total
    acc_t2i = correct_t2i / total
    print(f"    Test loss: {avg_test_loss:.4f} | Img->Txt acc: {acc_i2t:.4f} | Txt->Img acc: {acc_t2i:.4f}")

    # ---- SAVE WEIGHTS ----
    checkpoint_path_out = os.path.join(CHECKPOINT_DIR, f"clip_flickr8k_epoch{epoch+1}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": MODEL_CONFIG,
        "sot_id": SOT_ID,
        "eot_id": EOT_ID,
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "test_loss": avg_test_loss,
        "acc_i2t": acc_i2t,
        "acc_t2i": acc_t2i,
    }, checkpoint_path_out)
    print(f"Saved checkpoint: {checkpoint_path_out}")

print("Training complete.")