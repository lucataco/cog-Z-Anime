# Z-Anime Cog model

[![Try a demo on Replicate](https://replicate.com/lucataco/z-anime/badge)](https://replicate.com/lucataco/z-anime)

Cog wrapper for [`SeeSee21/Z-Anime`](https://huggingface.co/SeeSee21/Z-Anime), an Apache-2.0 anime text-to-image fine-tune of `Tongyi-MAI/Z-Image`. The 6B-parameter S3-DiT transformer is fully fine-tuned on anime aesthetics with strong prompt adherence, full negative-prompt support, and rich diversity across characters and scenes.

The deployed Replicate model lives at [replicate.com/lucataco/z-anime](https://replicate.com/lucataco/z-anime).

## Run on Replicate

Run the deployed model from the Replicate Python client:

```python
import replicate

output = replicate.run(
    "lucataco/z-anime",
    input={
        "prompt": "A cinematic anime portrait of a young woman with silver hair and golden eyes, standing in a sunlit bamboo forest with cherry blossoms falling around her, detailed lighting, rich colors",
        "aspect_ratio": "portrait",
        "num_inference_steps": 36,
        "guidance_scale": 4.0,
        "seed": 42,
    },
)
print(output)
```

Or with curl:

```bash
curl -s -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Prefer: wait=60" \
  -d '{
    "version": "lucataco/z-anime",
    "input": {
      "prompt": "Detailed anime portrait of a cat wearing a wizard hat",
      "aspect_ratio": "square"
    }
  }'
```

## Inputs

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | string | (required) | Natural language description of the anime image to generate. |
| `negative_prompt` | string | `""` | Things to avoid in the image. Leave blank to disable. |
| `aspect_ratio` | string | `portrait` | One of `square`, `portrait`, `landscape`, `tall`, or `wide`. |
| `num_inference_steps` | int `[4, 80]` | `36` | Number of denoising steps. 28-50 recommended for Z-Anime Base. |
| `guidance_scale` | float `[1.0, 12.0]` | `4.0` | Classifier-free guidance scale. 3.0-5.0 is the sweet spot. |
| `seed` | int `[-1, 2147483647]` | `-1` | Random seed. Set to -1 for a random seed. |

### Aspect ratios

| Option | Size |
| --- | --- |
| `square` | `1024x1024` |
| `portrait` | `832x1216` |
| `landscape` | `1216x832` |
| `tall` | `768x1344` |
| `wide` | `1344x768` |

## Output

A single PNG image at the chosen aspect ratio.

```json
{
  "output": "https://replicate.delivery/.../output.png"
}
```

## Run locally with Cog

```bash
git clone https://github.com/lucataco/cog-Z-Anime.git
cd cog-Z-Anime
cog predict \
  -i prompt="A cinematic anime portrait of a silver-haired mage in a neon city" \
  -i aspect_ratio="portrait" \
  -i seed=1234
```

The Cog predictor downloads pre-built weights via `pget` from `https://weights.replicate.delivery/default/SeeSee21/Z-Anime/model.tar` on first run (cold-boot ~90s on an L40S, ~50s per prediction afterwards).

If you want to iterate locally without going through the CDN, run `./download-weights` to pre-warm the `checkpoints/` directory directly from Hugging Face.

## Push your own copy

```bash
cog push r8.im/<your-username>/z-anime
```

See the [Replicate Cog docs](https://cog.run/) for details.

## Repository layout

- `predict.py` — Cog predictor implementation; downloads weights via `pget` in `setup()`.
- `cog.yaml` — Cog build and prediction configuration.
- `requirements.txt` — pinned Python dependencies installed in the Cog image.
- `download-weights` — optional local helper for pre-warming the dev cache from Hugging Face.

## License

This wrapper code is Apache 2.0. The Z-Anime model weights are governed by the upstream [SeeSee21/Z-Anime](https://huggingface.co/SeeSee21/Z-Anime) model card and are also Apache 2.0.
