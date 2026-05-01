from __future__ import annotations

# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import shutil
import time
from pathlib import Path as FilePath
import torch
from cog import BasePredictor, Input, Path
from diffusers import ZImagePipeline
from huggingface_hub import snapshot_download

MODEL_ID = "SeeSee21/Z-Anime"
MODEL_CACHE = "checkpoints"
SUBFOLDER = "diffusers"
BF16_TEXT_ENCODER = "text_encoder/qwen_3_4b-bf16.safetensors"
DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality, blurry, extra fingers, bad anatomy, text, watermark"

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "square": (1024, 1024),
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "tall": (768, 1344),
    "wide": (1344, 768),
}


def download_weights() -> None:
    start = time.time()
    print(f"Downloading {MODEL_ID}/{SUBFOLDER} to {MODEL_CACHE}")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_CACHE,
        allow_patterns=["diffusers/**", BF16_TEXT_ENCODER],
    )
    print(f"Download complete in {time.time() - start:.1f}s")


class Predictor(BasePredictor):
    pipe: ZImagePipeline

    def setup(self) -> None:
        if not os.path.exists(os.path.join(MODEL_CACHE, SUBFOLDER, "model_index.json")) or not os.path.exists(os.path.join(MODEL_CACHE, BF16_TEXT_ENCODER)):
            download_weights()

        bf16_text_encoder = os.path.join(MODEL_CACHE, BF16_TEXT_ENCODER)
        diffusers_text_encoder = os.path.join(MODEL_CACHE, SUBFOLDER, "text_encoder", "model.safetensors")
        if os.path.exists(bf16_text_encoder):
            print("Using BF16 text encoder for prompt fidelity")
            shutil.copyfile(bf16_text_encoder, diffusers_text_encoder)

        start = time.time()
        print("Loading Z-Anime pipeline")
        self.pipe = ZImagePipeline.from_pretrained(
            os.path.join(MODEL_CACHE, SUBFOLDER),
            torch_dtype=torch.bfloat16,
        )
        self.pipe.enable_model_cpu_offload()
        try:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
        except Exception as exc:
            print(f"VAE memory options unavailable: {exc}")
        print(f"Loaded pipeline in {time.time() - start:.1f}s")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Natural language description of the anime image to generate.",
            default="A cinematic anime portrait of a young woman with silver hair and golden eyes, standing in a sunlit bamboo forest with cherry blossoms falling around her, detailed lighting, rich colors",
        ),
        negative_prompt: str = Input(
            description="Things to avoid in the image.",
            default=DEFAULT_NEGATIVE_PROMPT,
        ),
        aspect_ratio: str = Input(
            description="Output aspect ratio: square, portrait, landscape, tall, or wide.",
            default="portrait",
        ),
        seed: int = Input(
            description="Random seed. Set to -1 for a random seed.",
            default=-1,
            ge=-1,
            le=2147483647,
        ),
    ) -> Path:
        aspect_ratio = aspect_ratio.lower().strip()
        if aspect_ratio not in ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect_ratio '{aspect_ratio}'. Use one of: {', '.join(ASPECT_RATIOS)}")
        width, height = ASPECT_RATIOS[aspect_ratio]
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "big") % 2147483647
        print(f"Generating {width}x{height} with seed {seed}")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=36,
            guidance_scale=4.0,
            generator=generator,
            max_sequence_length=512,
        ).images[0]
        out = FilePath("/tmp/output.png")
        image.save(out)
        return Path(out)
