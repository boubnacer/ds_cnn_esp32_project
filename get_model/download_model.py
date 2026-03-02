"""
STEP 1 — Download a pretrained DS-CNN model + sample audio data
================================================================
DS-CNN (Depthwise Separable CNN) is specifically designed for keyword
spotting on microcontrollers. It was published by Arm Research.

We will use the MLCommons MLPerf Tiny version which is already trained
on Google Speech Commands v2 (10 keywords: yes, no, up, down, left,
right, on, off, stop, go + silence + unknown).

What this script does:
  1. Downloads the pretrained DS-CNN float32 .h5 model
  2. Downloads a small sample of the Speech Commands test set
  3. Verifies everything is in place

Run:
    python step1_get_model/download_model.py
"""

import os
import urllib.request
import zipfile
import hashlib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data", "test_samples")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# ─── URLs ─────────────────────────────────────────────────────────────────────
# MLCommons reference DS-CNN model (float32 keras .h5)
MODEL_URL = (
    "https://github.com/mlcommons/tiny/raw/master/benchmark/training/"
    "keyword_spotting/trained_models/kws_ref_model_float32.tflite"
)
# NOTE: If the above link changes, the backup is the ARM ML-KWS repo:
# https://github.com/ARM-software/ML-KWS-for-MCU

# Google Speech Commands V2 — small test subset (Hugging Face mirror)
DATA_URL = (
    "http://storage.googleapis.com/download.tensorflow.org/data/"
    "speech_commands_v0.02.tar.gz"
)


def download_file(url: str, dest_path: str, label: str) -> bool:
    """Download a file with a simple progress indicator."""
    if os.path.exists(dest_path):
        print(f"  ✓ Already downloaded: {label}")
        return True

    print(f"  ↓ Downloading {label}...")
    print(f"    URL: {url}")
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                pct = int(count * block_size * 100 / total_size)
                print(f"\r    Progress: {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print()  # newline after progress
        print(f"  ✓ Saved to: {dest_path}")
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        print("    → See MANUAL_DOWNLOAD.md for instructions")
        return False


def create_manual_download_guide():
    """Create a guide if automatic download fails."""
    guide_path = os.path.join(BASE_DIR, "MANUAL_DOWNLOAD.md")
    content = """# Manual Download Guide

## Option A — Use the pretrained TFLite model directly (EASIEST)

The DS-CNN model is already available as a TFLite file in the MLCommons repo:

1. Go to: https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting/trained_models
2. Download `ds_cnn_s_quantized.tflite`
3. Place it in: `models/ds_cnn_float32.tflite`

## Option B — Train from scratch using TensorFlow's tutorial

TensorFlow has an official simple audio recognition tutorial:
https://www.tensorflow.org/tutorials/audio/simple_audio

Run their notebook — at the end you'll have a trained model you can export.

## Option C — Use the ARM ML-KWS-for-MCU repo

```bash
git clone https://github.com/ARM-software/ML-KWS-for-MCU
cd ML-KWS-for-MCU
# Follow their README to get the pretrained DS-CNN models
```

## Speech Commands Dataset

Download manually from:
https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Extract to: `data/test_samples/`

## Dataset Structure Expected
```
data/test_samples/
├── yes/   (contains .wav files)
├── no/
├── up/
├── down/
├── left/
├── right/
├── on/
├── off/
├── stop/
├── go/
└── _background_noise_/
```
"""
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  📄 Created: MANUAL_DOWNLOAD.md — check it if downloads fail!")


def verify_setup():
    """Check what we have and tell the user what's missing."""
    print("\n─── Verification ─────────────────────────────────────────────")
    issues = []

    model_files = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
    if model_files:
        print(f"  ✓ Models folder: {model_files}")
    else:
        print("  ✗ No model files found in models/")
        issues.append("No model found")

    data_folders = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
    if data_folders:
        wav_count = sum(
            len([f for f in os.listdir(os.path.join(DATA_DIR, d)) if f.endswith(".wav")])
            for d in data_folders
            if os.path.isdir(os.path.join(DATA_DIR, d))
        )
        print(f"  ✓ Data folder: {len(data_folders)} classes, ~{wav_count} .wav files")
    else:
        print("  ✗ No audio data found in data/test_samples/")
        issues.append("No test data found")

    if issues:
        print("\n  ⚠ Some items missing — check MANUAL_DOWNLOAD.md")
    else:
        print("\n  🎉 All good! Run: python step2_quantize/quantize_model.py")


def main():
    print("=" * 60)
    print("  STEP 1 — Download DS-CNN Model + Speech Commands Data")
    print("=" * 60)

    create_manual_download_guide()

    # Try to download the already-quantized TFLite model from MLCommons
    model_dest = os.path.join(MODELS_DIR, "ds_cnn_pretrained.tflite")
    download_file(MODEL_URL, model_dest, "DS-CNN TFLite model")

    print("\n  ℹ  NOTE: The full Speech Commands dataset is ~2.3GB.")
    print("     For learning, we'll work with a small synthetic subset.")
    print("     See MANUAL_DOWNLOAD.md to get the real dataset.")

    # Create synthetic mini-dataset for testing the pipeline
    create_synthetic_test_data()

    verify_setup()


def create_synthetic_test_data():
    """
    Create tiny synthetic .wav files so the pipeline can be tested
    even without downloading the full 2.3GB dataset.
    These are just sine waves — accuracy results won't be meaningful,
    but the pipeline (preprocessing → inference → evaluation) will work.
    """
    import struct
    import math

    print("\n  🔧 Creating synthetic test .wav files for pipeline testing...")

    KEYWORDS = ["yes", "no", "up", "down", "left",
                "right", "on", "off", "stop", "go"]
    SAMPLE_RATE = 16000
    DURATION_S  = 1  # 1 second = 16000 samples

    for i, keyword in enumerate(KEYWORDS):
        keyword_dir = os.path.join(DATA_DIR, keyword)
        os.makedirs(keyword_dir, exist_ok=True)

        # Make 5 synthetic samples per keyword
        for j in range(5):
            wav_path = os.path.join(keyword_dir, f"synthetic_{j:02d}.wav")
            if os.path.exists(wav_path):
                continue

            # Different frequency per keyword so they're distinct
            freq = 200 + i * 100 + j * 10
            samples = [
                int(32767 * math.sin(2 * math.pi * freq * t / SAMPLE_RATE))
                for t in range(SAMPLE_RATE * DURATION_S)
            ]

            # Write minimal WAV file
            with open(wav_path, "wb") as f:
                num_samples = len(samples)
                data_size   = num_samples * 2  # 16-bit = 2 bytes
                # RIFF header
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + data_size))
                f.write(b"WAVE")
                # fmt chunk
                f.write(b"fmt ")
                f.write(struct.pack("<I", 16))       # chunk size
                f.write(struct.pack("<H", 1))        # PCM
                f.write(struct.pack("<H", 1))        # mono
                f.write(struct.pack("<I", SAMPLE_RATE))
                f.write(struct.pack("<I", SAMPLE_RATE * 2))
                f.write(struct.pack("<H", 2))        # block align
                f.write(struct.pack("<H", 16))       # bits per sample
                # data chunk
                f.write(b"data")
                f.write(struct.pack("<I", data_size))
                f.write(struct.pack(f"<{num_samples}h", *samples))

    total = sum(
        len(os.listdir(os.path.join(DATA_DIR, k))) for k in KEYWORDS
        if os.path.isdir(os.path.join(DATA_DIR, k))
    )
    print(f"  ✓ Created {total} synthetic .wav files in data/test_samples/")
    print("  ℹ  These are sine waves — useful for testing the pipeline,")
    print("     not for real accuracy benchmarks.")


if __name__ == "__main__":
    main()