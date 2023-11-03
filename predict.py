# Prediction interface for Cog ⚙️
from cog import BasePredictor, Input, Path
import os
from typing import List
import time
import datetime
import re
import math

import torch
from PIL import Image
from diffusers import (
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
)

from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
)

from diffusers.utils.import_utils import is_xformers_available

from lora import inject_trainable_lora, monkeypatch_or_replace_lora, monkeypatch_remove_lora
from image_processing import apply_canny, crop

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

CONTROLNETS = {
    "canny": "lllyasviel/sd-controlnet-canny",
}

MAX_SIZE = 1024

MODEL_CACHE = "./checkpoints"

MODEL_FILENAME = "Realistic_Vision_V5.1-inpainting.safetensors"
VAE_FILENAME = "vae-ft-mse-840000-ema-pruned.safetensors"
LORA_FILENAMES = ["lora_white_wall.pt", "lora_white_wall_v2.pt", "lora_brick_wall.pt"]

LORA_DICT = {
    "white_wall": {
        "rank": 8,
        "scale": 0.9,
        "target_replace_module": {"CrossAttention", "Attention", "GEGLU"},
        "filename": "lora_white_wall.pt",
    },
    "white_wall_v2": {
        "rank": 8,
        "scale": 0.9,
        "target_replace_module": {"CrossAttention", "Attention", "GEGLU"},
        "filename": "lora_white_wall_v2.pt",
    },
    "brick_wall": {
        "rank": 8,
        "scale": 0.9,
        "target_replace_module": {"CrossAttention", "Attention", "GEGLU"},
        "filename": "lora_brick_wall.pt",
    },
}

CANNY_TRIGGER_WORDS = ['wall', 'walls', 'facade', 'facades']

class Predictor(BasePredictor):

    def add_lora(self, lora_name, pipe, pipe_name):
        print('Replacing LoRA with', lora_name)
        start = time.time()
        monkeypatch_or_replace_lora(
            model=pipe.unet,
            loras=self.loras[lora_name].copy(),
            target_replace_module=LORA_DICT[lora_name]["target_replace_module"],
            r=LORA_DICT[lora_name]["rank"],
        )
        self.current_lora[pipe_name] = lora_name
        print('Done in', "{:.2f}".format(time.time() - start), 'seconds')

    def remove_lora(self, pipe, pipe_name):
        print('Removing LoRA')
        start = time.time()
        monkeypatch_remove_lora(
            model=pipe.unet,
        )
        self.current_lora[pipe_name] = None
        print('Done in', "{:.2f}".format(time.time() - start), 'seconds')
        
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print('loading original model')
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        self.loras = {}
        for lora_name in LORA_DICT.keys():
            lora_path = os.path.join(MODEL_CACHE, LORA_DICT[lora_name]["filename"])
            self.loras[lora_name] = torch.load(lora_path) # to decide if we need to load it to cpu or gpu

        self.lora = None
        print('loading controlnet')
        self.controlnet_canny = ControlNetModel.from_pretrained(
            CONTROLNETS["canny"], 
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
        )
        
        print('loading controlnet pipeline')
        self.pipe_canny = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            MODEL_CACHE,
            controlnet=self.controlnet_canny,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        
        # eneable cpu offload to be more efficient with GPU memory
        self.pipe.enable_model_cpu_offload()
        self.pipe_canny.enable_model_cpu_offload()
        
        self.current_lora = {
            'original': None,
            'canny': None,
        }

        self.canny_pattern = re.compile('|'.join(CANNY_TRIGGER_WORDS))

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
        width: int = Input(
            description="Width of the image to generate. Should be multiple of 8",
            ge=8,
            le=MAX_SIZE,
            default=512,
        ),
        height: int = Input(
            description="Height of the image to generate. Should be multiple of 8",
            ge=8,
            le=MAX_SIZE,
            default=512,
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
        controlnet: str = Input(
            description="What controlnet to use",
            choices=CONTROLNETS.keys(),
            default=None,
        ),

    ) -> List[Path]:
        """Run a single prediction on the model"""

        prompt = prompt.lower().strip()

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(seed)

        if re.search(self.canny_pattern, prompt):
            print('Prompt contains a trigger word for canny')
            controlnet = 'canny'

        print('Using seed:', seed)

        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).convert("L")

        """
        if width % 8 != 0 or height % 8 != 0:
            if mask.size == image.size:
                mask = crop(mask)
            image = crop(image)

        width, height = image.size
        """

        extra_kwargs = {
            "image": image,
            "mask_image": mask,
            "width": width,
            "height": height,
        }

        if controlnet == 'canny':
            # if we want to patch the controlnet in real time
            """
            self.pipe_canny.controlnet = ControlNetModel.from_pretrained(CONTROLNETS[controlnet], torch_dtype=torch.float16)
            self.pipe_canny.controlnet.to("cuda")
            self.pipe_canny.controlnet.eval()
            """
            control_image = apply_canny(image, low_threshold=100, high_threshold=200)
            extra_kwargs['control_image'] = control_image
            strength = 1.0
            pipe = self.pipe_canny 
        else:
            pipe = self.pipe

        pipe_name = controlnet if controlnet is not None else 'original'

        # set scheduler
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)

        # add lora if needed
        if lora is not None and lora != self.current_lora[pipe_name]:
            self.add_lora(lora, pipe, pipe_name)
        elif lora is None and self.current_lora[pipe_name] is not None:
            self.remove_lora(pipe, pipe_name)

        print()
        print('Current date and time', datetime.datetime.now())
        print('self.current_lora:', self.current_lora[pipe_name] if self.current_lora[pipe_name] is not None else 'None')
        print()

        output = pipe(
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
    

