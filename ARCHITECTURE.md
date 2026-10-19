# CLIP Architecture Implementation

This repository contains an implementation of the CLIP (Contrastive Language-Image Pretraining) model architecture, built from scratch using PyTorch. 
It includes the core model components, tokenizer, and a training script for fine-tuning on custom datasets like Flickr8k.

## 1. Core Model Components (`clip/model.py`)

The architecture brings together two primary encoders into a single `CLIP` module.

### Vision Encoder
The vision encoder maps images to a feature space. This implementation supports two types of vision backbones:

*   **Modified ResNet**: A variant of the standard ResNet containing architectural tweaks for CLIP:
    *   **Stem**: Uses a 3-layer stem (instead of 1) with an average pooling layer instead of max pooling.
    *   **Anti-aliasing**: Appends an average pool prior to convolutions that have a stride greater than 1.
    *   **Attention Pool**: The final global average pooling layer is replaced with a Multihead QKV Attention layer (`AttentionPool2d`).
*   **Vision Transformer (ViT)**: A standard Transformer architecture for images. 
    *   The image is sliced into non-overlapping patches which are linearly embedded (`conv1`).
    *   A learnable class token and positional embeddings are added.
    *   Processed through a series of `Transformer` blocks.

### Text Encoder
The text encoder processes input text to the same feature space as the vision encoder.

*   **Transformer Backbone**: Uses a sequence of `ResidualAttentionBlock` layers.
*   **Token & Position Embeddings**: Text is first tokenized, mapped via `token_embedding`, and summed with a `positional_embedding`.
*   **Causal Masking**: A triangular attention mask ensures tokens only attend to preceding tokens in the sequence, preserving causal ordering.
*   **Feature Extraction**: The feature vector corresponding to the highest numbered token (the end-of-text or EOT token) is extracted and projected to the common embedding space.

### Joint CLIP Module
The `CLIP` class acts as the orchestrator:

*   It processes images via `encode_image()` and texts via `encode_text()`.
*   The output feature vectors are L2-normalized.
*   The `forward` pass computes a scaled cosine similarity matrix (`logits_per_image` and `logits_per_text`) between the image and text representations in a batch.

## 2. Tokenization and Preprocessing

*   **Simple Tokenizer (`clip/simple_tokenizer.py`)**: Implements Byte Pair Encoding (BPE), identical to OpenAI's tokenization process. It maps vocabulary to integer IDs and adds `<|startoftext|>` and `<|endoftext|>` tokens.
*   **Image Transforms (`clip/clip.py`)**: `_transform` resizes the image (using bicubic interpolation), takes a center crop, and normalizes it with predefined mean and standard deviation constants before converting it to a tensor.

## 3. Loading Weights (`clip/clip.py`)

The `load()` function manages the initialization and loading of pre-trained models:
*   Automatically fetches and verifies the SHA256 checksums of model checkpoints from OpenAI's servers.
*   Can parse both standard PyTorch `state_dict` checkpoints and compiled JIT models.
*   Converts PyTorch `state_dict` structures dynamically to match the initialized `CLIP` object configuration.
*   It supports FP16 weights to save memory and provides fallbacks for devices like the CPU which run better in FP32.

## 4. Training Loop (`clip/train_clip_flickr8k.py`)

A standalone script designed to train or fine-tune the custom CLIP implementation:
*   **Pretrained Initialization**: Loads weights from an OpenAI pre-trained checkpoint (`ViT-B-32`) to jump-start the model parameters rather than training purely from scratch.
*   **Dataset Handling**: Implements `Flickr8kCLIPDataset` for loading image/caption pairs. Split into distinct sets where no single image crosses the train/test barrier.
*   **Objective**: Computes the Symmetric Cross Entropy loss (averaging `loss_img` and `loss_txt`) over the contrastive similarity logits.
*   **Testing**: Runs periodic test-set inference tracking loss, image-to-text accuracy, and text-to-image accuracy, and saves the trained weights.
