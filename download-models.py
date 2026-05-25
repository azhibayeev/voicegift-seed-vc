"""Pre-download Seed-VC model weights at build time.

Avoids cold-start HF hits on Runpod and ensures the image is fully
self-contained. Weights live in /app/seed-vc/checkpoints (matches
Seed-VC's default expected location).
"""
import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download

CKPT_DIR = "/app/seed-vc/checkpoints"
HF_CACHE = "/app/hf_cache"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

# Set HF cache env so transitive HF downloads at runtime hit the same dir
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

# Files we explicitly download for V1 singing voice conversion (f0_condition=True)
V1_SVC_FILES = [
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
    "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
]

# Files for V1 speech VC (fallback / non-singing)
V1_VC_FILES = [
    "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
    "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
]

print(f"[download-models] downloading from Plachta/Seed-VC into {CKPT_DIR}", flush=True)
for filename in V1_SVC_FILES + V1_VC_FILES:
    try:
        path = hf_hub_download(
            repo_id="Plachta/Seed-VC",
            filename=filename,
            local_dir=CKPT_DIR,
        )
        print(f"  ✓ {filename}", flush=True)
    except Exception as e:
        # Some optional files may be missing; log but don't fail the build
        print(f"  ✗ {filename}: {e}", file=sys.stderr, flush=True)

# Whisper encoder used internally by Seed-VC
print("[download-models] downloading Whisper small from openai", flush=True)
try:
    snapshot_download(repo_id="openai/whisper-small", cache_dir=HF_CACHE)
    print("  ✓ whisper-small", flush=True)
except Exception as e:
    print(f"  ✗ whisper-small: {e}", file=sys.stderr, flush=True)

# BigVGAN vocoder used by SVC checkpoint
print("[download-models] downloading BigVGAN", flush=True)
try:
    snapshot_download(repo_id="nvidia/bigvgan_v2_44khz_128band_512x", cache_dir=HF_CACHE)
    print("  ✓ bigvgan_v2_44khz", flush=True)
except Exception as e:
    print(f"  ✗ bigvgan: {e}", file=sys.stderr, flush=True)

# CAMPPlus speaker encoder from funasr (only the encoder file, not full funasr)
print("[download-models] downloading CAMPPlus encoder", flush=True)
try:
    hf_hub_download(
        repo_id="funasr/campplus",
        filename="campplus_cn_common.bin",
        local_dir=CKPT_DIR,
    )
    print("  ✓ campplus_cn_common.bin", flush=True)
except Exception as e:
    print(f"  ✗ campplus: {e}", file=sys.stderr, flush=True)

print("[download-models] done", flush=True)
