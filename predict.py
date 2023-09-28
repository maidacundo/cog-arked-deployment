# Prediction interface for Cog ⚙️
from cog import BasePredictor, Input, Path
import os
from typing import List
import time
import datetime

import torch
from PIL import Image
from diffusers import (
    AutoencoderKL, 
    StableDiffusionInpaintPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
)

from lora import inject_trainable_lora, monkeypatch_or_replace_lora, monkeypatch_remove_lora

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

MAX_SIZE = 768

MODEL_CACHE = "./checkpoints"

MODEL_FILENAME = "Realistic_Vision_V5.1-inpainting.safetensors"
VAE_FILENAME = "vae-ft-mse-840000-ema-pruned.safetensors"
LORA_FILENAMES = ["lora_white_wall.pt", "lora_brick_wall.pt"]

LORA_DICT = {
    "white_wall": {
        "rank": 8,
        "scale": 1.0,
        "target_replace_module": {"CrossAttention", "Attention", "GEGLU"},
        "filename": LORA_FILENAMES[0],
    },
    "brick_wall": {
        "rank": 8,
        "scale": 1.0,
        "target_replace_module": {"CrossAttention", "Attention", "GEGLU"},
        "filename": LORA_FILENAMES[1],
    },
}

class Predictor(BasePredictor):

    def add_lora(self, lora_name):
        print('Replacing LoRA with', lora_name)
        start = time.time()
        monkeypatch_or_replace_lora(
            model=self.pipe.unet,
            loras=self.loras[lora_name].copy(),
            target_replace_module=LORA_DICT[lora_name]["target_replace_module"],
            r=LORA_DICT[lora_name]["rank"],
        )
        self.current_lora = lora_name
        print('Done in', "{:.2f}".format(time.time() - start), 'seconds')

    def remove_lora(self):
        print('Removing LoRA')
        start = time.time()
        monkeypatch_remove_lora(
            model=self.pipe.unet,
        )
        self.current_lora = None
        print('Done in', "{:.2f}".format(time.time() - start), 'seconds')
        
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.pipe = pipe.to("cuda")
        self.loras = {}
        for lora_name in LORA_DICT.keys():
            lora_path = os.path.join(MODEL_CACHE, LORA_DICT[lora_name]["filename"])
            self.loras[lora_name] = torch.load(lora_path) # to decide if we need to load it to cpu or gpu
        self.current_lora = None

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None
        ),
        image: Path = Input(
            description="Inital image to generate variations of. Supproting images size with 512x512",
        ),
        mask: Path = Input(
            description="Black and white image to use as mask for inpainting over the image provided. White pixels are inpainted and black pixels are preserved",
        ),
        num_outputs: int = Input(
            description="Number of images to output. Higher number of outputs may OOM.",
            ge=1,
            le=8,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        strength: float = Input(
            description="Strength of the inpainting mask",
            ge=0.0,
            le=1.0,
            default=0.99,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        lora: str = Input(
            description="LoRA to use for the model",
            choices=LORA_DICT.keys(),
            default=None,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).convert("RGB")

        width, height = image.size

        extra_kwargs = {
            "image": image,
            "mask_image": mask,
            "width": width,
            "height": height,
        }
        
        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        # add lora if needed
        if lora is not None and lora != self.current_lora:
            self.add_lora(lora)
        elif lora is None and self.current_lora is not None:
            self.remove_lora()

        print()
        print('Current date and time', datetime.datetime.now())
        print('self.current_lora:', self.current_lora if self.current_lora is not None else 'None')
        print('Running inference with LoRA:', lora if lora is not None else 'None')
        print()
        
        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            strength=strength,
            **extra_kwargs,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths