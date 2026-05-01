# Z-Anime Cog Model

Cog wrapper for [`SeeSee21/Z-Anime`](https://huggingface.co/SeeSee21/Z-Anime), an Apache-2.0 anime text-to-image fine-tune of `Tongyi-MAI/Z-Image`.

This repository packages the model for local Cog execution and Replicate deployment. The predictor exposes a simple image generation API with prompt, negative prompt, aspect ratio, and seed inputs.

## Features

- Downloads the Z-Anime Diffusers weights from Hugging Face on first setup.
- Uses the BF16 text encoder for improved prompt fidelity when available.
- Supports square, portrait, landscape, tall, and wide image sizes.
- Enables CPU offload plus VAE slicing and tiling to reduce VRAM pressure.

## Inputs

| Input | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | string | Cinematic anime portrait prompt | Image description to generate. |
| `negative_prompt` | string | `low quality, worst quality, blurry, extra fingers, bad anatomy, text, watermark` | Things to avoid in the image. |
| `aspect_ratio` | string | `portrait` | One of `square`, `portrait`, `landscape`, `tall`, or `wide`. |
| `seed` | integer | `-1` | Random seed. Use `-1` to generate a random seed. |

## Aspect Ratios

| Option | Size |
| --- | --- |
| `square` | `1024x1024` |
| `portrait` | `832x1216` |
| `landscape` | `1216x832` |
| `tall` | `768x1344` |
| `wide` | `1344x768` |

## Requirements

- NVIDIA GPU with CUDA support.
- [Cog](https://github.com/replicate/cog) installed locally.
- Network access to download model weights from Hugging Face.

## Usage

Download weights ahead of time if you want to warm the local cache:

```bash
./download-weights
```

Run a prediction locally with Cog:

```bash
cog predict \
  -i prompt="A cinematic anime portrait of a silver-haired mage in a neon city" \
  -i aspect_ratio="portrait" \
  -i seed=1234
```

The generated image is written to `/tmp/output.png` inside the prediction environment and returned by Cog.

## Repository Layout

- `predict.py`: Cog predictor implementation.
- `cog.yaml`: Cog build and prediction configuration.
- `requirements.txt`: Python dependencies installed in the Cog image.
- `download-weights`: Helper script for downloading model weights into `checkpoints/`.
- `script/download-weights`: Alternate helper script retained for Cog/Replicate workflows.

## Notes For Public Upload

- Do not commit downloaded model weights. They are stored under `checkpoints/` and ignored by Git.
- Do not commit local environment files such as `.env` or `.env.*`.
- Generated Cog metadata, Python caches, and local outputs are ignored.

## License

This wrapper code is intended to be used with the upstream Z-Anime model. Check the upstream model card and license before redistribution or commercial use.
