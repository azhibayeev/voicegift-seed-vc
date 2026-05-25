"""Runpod Serverless handler that wraps Seed-VC singing voice conversion.

The handler is invoked per-request by Runpod's framework. It:
  1. Downloads two remote audio files (source = vocal to transform, reference = target voice timbre)
  2. Runs Seed-VC inference with f0_condition=True (singing mode)
  3. Returns the output audio as base64-encoded WAV
"""
import os
import sys
import tempfile
import urllib.request
import base64
import traceback

import runpod
import torch
import soundfile as sf

# Make Seed-VC importable
sys.path.insert(0, "/app/seed-vc")

# Lazy-load the wrapper so we surface clear errors if the import fails
WRAPPER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_wrapper():
    global WRAPPER
    if WRAPPER is None:
        print(f"[handler] initializing SeedVCWrapper on {DEVICE}", flush=True)
        # Import inside the function so init errors are reported as
        # job-level errors rather than process-level crashes at boot.
        from seed_vc_wrapper import SeedVCWrapper
        WRAPPER = SeedVCWrapper(device=torch.device(DEVICE))
        print("[handler] wrapper ready", flush=True)
    return WRAPPER


def _download(url: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path


def handler(event):
    inp = (event or {}).get("input") or {}
    source_url = inp.get("source_audio_url")
    reference_url = inp.get("reference_audio_url")

    if not source_url or not reference_url:
        return {"error": "source_audio_url and reference_audio_url are required"}

    # Singing-mode tuning knobs — defaults match common preset
    diffusion_steps = int(inp.get("diffusion_steps", 25))
    length_adjust = float(inp.get("length_adjust", 1.0))
    inference_cfg_rate = float(inp.get("inference_cfg_rate", 0.7))
    f0_condition = bool(inp.get("f0_condition", True))  # True = singing mode
    auto_f0_adjust = bool(inp.get("auto_f0_adjust", False))
    semi_tone_shift = int(inp.get("semi_tone_shift", 0))

    source_path = None
    reference_path = None
    try:
        wrapper = _get_wrapper()
        source_path = _download(source_url, ".wav")
        reference_path = _download(reference_url, ".wav")

        print(f"[handler] running inference src={source_url} ref={reference_url}", flush=True)
        audio, sr = wrapper.convert_voice(
            source_audio_path=source_path,
            target_audio_path=reference_path,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            inference_cfg_rate=inference_cfg_rate,
            f0_condition=f0_condition,
            auto_f0_adjust=auto_f0_adjust,
            semi_tone_shift=semi_tone_shift,
        )

        out_fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(out_fd)
        sf.write(out_path, audio, sr)

        with open(out_path, "rb") as f:
            audio_bytes = f.read()
        encoded = base64.b64encode(audio_bytes).decode("ascii")

        try:
            os.unlink(out_path)
        except OSError:
            pass

        return {
            "audio_base64": encoded,
            "format": "wav",
            "sample_rate": sr,
        }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        for p in (source_path, reference_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
