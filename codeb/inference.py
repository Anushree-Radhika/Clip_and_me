import modal
import sys
from pathlib import Path

app = modal.App("visiongpt2-inference")

# Reuse the same image environment
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
    .add_local_python_source("dataset", "model", "trainer")
)

# Load the volume where the checkpoint is saved
checkpoint_vol = modal.Volume.from_name("visiongpt2-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/captioner_ckpt": checkpoint_vol},
)
def generate_caption_remote(image_bytes: bytes):
    import torch
    from types import SimpleNamespace
    import io
    from PIL import Image
    import numpy as np
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from model import VisionGPT2Model
    from transformers import GPT2TokenizerFast

    # Configs
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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model
    model = VisionGPT2Model.from_pretrained(model_config).to(device)
    
    # Load weights
    ckpt_path = Path('/root/captioner_ckpt/captioner.pt')
    if not ckpt_path.exists():
        return f"Error: Checkpoint not found at {ckpt_path}. Did the training finish and save?"
    
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    # Setup tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Image transforms
    gen_tfms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    # Process image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_arr = np.array(img)
    img_tensor = gen_tfms(image=img_arr)['image'].unsqueeze(0).to(device)

    # Generate
    sequence = torch.ones(1, 1).to(device=device).long() * tokenizer.bos_token_id
    
    with torch.no_grad():
        caption_tokens = model.generate(
            img_tensor,
            sequence,
            max_tokens=50,
            temperature=1.0,
            deterministic=True
        )
        
    caption = tokenizer.decode(caption_tokens.numpy(), skip_special_tokens=True)
    return caption

@app.local_entrypoint()
def main(image_path: str):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: Could not find image at {image_path}")
        return
        
    print(f"Uploading {image_path} and generating caption...")
    with open(path, "rb") as f:
        image_bytes = f.read()
        
    result = generate_caption_remote.remote(image_bytes)
    print("\n--- Generated Caption ---")
    print(result)
    print("-------------------------")