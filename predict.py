from __future__ import annotations

# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
from pathlib import Path as FilePath

import torch
from cog import BasePredictor, Input, Path
from diffusers import ZImagePipeline

MODEL_ID = "SeeSee21/Z-Anime"
MODEL_CACHE = "checkpoints"
SUBFOLDER = "diffusers"
# Pre-built tarball: contains only `diffusers/` with the BF16 text encoder
# already swapped in (~20 GB extracted). Avoids HF LFS bottlenecks during cold boot.
WEIGHTS_URL = "https://weights.replicate.delivery/default/SeeSee21/Z-Anime/model.tar"
DEFAULT_NEGATIVE_PROMPT = ""  # negative prompt is optional; leave blank by default

# Recommended sizes from upstream model card.
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "square": (1024, 1024),
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "tall": (768, 1344),
    "wide": (1344, 768),
}


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print(f"downloading url: {url}")
    print(f"downloading to: {dest}")
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print(f"downloading took: {time.time() - start:.1f}s")


class Predictor(BasePredictor):
    pipe: ZImagePipeline

    def setup(self) -> None:
        diffusers_dir = os.path.join(MODEL_CACHE, SUBFOLDER)
        if not os.path.exists(os.path.join(diffusers_dir, "model_index.json")):
            download_weights(WEIGHTS_URL, MODEL_CACHE)

        start = time.time()
        print("Loading Z-Anime pipeline")
        self.pipe = ZImagePipeline.from_pretrained(
            diffusers_dir,
            torch_dtype=torch.bfloat16,
        )
        # CPU offload + VAE memory options keep the L40S worker stable
        # (per upstream model card: 8GB VRAM target, ~14GB BF16 weights + 7.5GB BF16 TE).
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
            default=(
                "A cinematic anime portrait of a young woman with silver hair and "
                "golden eyes, standing in a sunlit bamboo forest with cherry blossoms "
                "falling around her, detailed lighting, rich colors"
            ),
        ),
        negative_prompt: str = Input(
            description="Things to avoid in the image. Leave blank to disable.",
            default="",
        ),
        aspect_ratio: str = Input(
            description="Output aspect ratio.",
            choices=list(ASPECT_RATIOS.keys()),
            default="portrait",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. 28-50 recommended for Z-Anime Base.",
            default=36,
            ge=4,
            le=80,
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale. 3.0-5.0 is the sweet spot.",
            default=4.0,
            ge=1.0,
            le=12.0,
        ),
        seed: int = Input(
            description="Random seed. Set to -1 for a random seed.",
            default=-1,
            ge=-1,
            le=2147483647,
        ),
    ) -> Path:
        width, height = ASPECT_RATIOS[aspect_ratio]
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "big") % 2147483647
        print(
            f"Generating {width}x{height}, steps={num_inference_steps}, "
            f"cfg={guidance_scale}, seed={seed}"
        )
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=512,
        ).images[0]
        out = FilePath("/tmp/output.png")
        image.save(out)
        return Path(out)
