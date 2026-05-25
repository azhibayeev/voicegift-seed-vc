# voicegift-seed-vc

Self-hosted Seed-VC (zero-shot singing voice conversion) packaged for
[Runpod Serverless](https://www.runpod.io/serverless). Built for the
[voicegift](https://github.com/azhibayeev) project.

## How it works

1. Runpod pulls this repo on each release and builds an image using `Dockerfile`.
2. The image bakes all Seed-VC model weights into the layer so cold starts
   don't have to hit HuggingFace.
3. `handler.py` accepts a POST with `source_audio_url` + `reference_audio_url`
   and returns base64-encoded WAV in the user's voice.

## Deploy

1. https://runpod.io/console/serverless → **+ New Endpoint**
2. **GitHub** → connect this repo → select `main` branch
3. **Container**: builds from `Dockerfile` at repo root
4. **GPU**: L4 24GB (cheapest SKU that fits — Seed-VC needs ~4GB VRAM)
5. **Container Disk**: 30 GB (model weights ~700 MB + cache)
6. **Max Workers**: 3 — **Idle Timeout**: 30 sec — **Flashboot**: on
7. Save. The endpoint URL appears as `https://api.runpod.ai/v2/<id>/runsync`.

## Test

```bash
export RUNPOD_API_KEY="rpa_xxx"
export ENDPOINT="https://api.runpod.ai/v2/<id>/runsync"

curl -X POST "$ENDPOINT" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "source_audio_url": "https://example.com/source-vocal.wav",
      "reference_audio_url": "https://example.com/user-voice-sample.wav"
    }
  }'
```

Response: `{"output": {"audio_base64": "...", "format": "wav", "sample_rate": 44100}}`.

## Tuning

Handler accepts optional inputs:
- `diffusion_steps` (default 25) — higher = better quality, slower
- `f0_condition` (default true) — singing mode; set false for speech
- `length_adjust` (default 1.0)
- `inference_cfg_rate` (default 0.7)
- `auto_f0_adjust` (default false)
- `semi_tone_shift` (default 0)

## Upstream

- Seed-VC source: https://github.com/Plachtaa/seed-vc (archived 2025-04, pinned)
- Models: https://huggingface.co/Plachta/Seed-VC
- License: GPL-3.0 — used here as a service, no source modification
