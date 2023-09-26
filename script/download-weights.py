#!/usr/bin/env python

# Run this before you deploy it on replicate
import os
import sys
import torch
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download

# append project directory to path so predict.py can be imported
sys.path.append('.')
from predict import MODEL_FILENAME, VAE_FILENAME, MODEL_CACHE, LORA_FILENAME

# Make cache folders
if not os.path.exists(MODEL_CACHE):
    os.makedirs(MODEL_CACHE)

realisticVision_path = hf_hub_download(repo_id="SG161222/Realistic_Vision_V5.1_noVAE", filename=MODEL_FILENAME)
vae_path = hf_hub_download(repo_id="stabilityai/sd-vae-ft-mse-original", filename=VAE_FILENAME)
lora_path = hf_hub_download(repo_id="maidacundo/lora-arked-facades", filename=LORA_FILENAME, local_dir=MODEL_CACHE)

# Load the model into memory to make running multiple predictions efficient
vae = AutoencoderKL.from_single_file(
    vae_path,
)

pipe = StableDiffusionInpaintPipeline.from_single_file(
    realisticVision_path,
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe.save_pretrained(MODEL_CACHE)