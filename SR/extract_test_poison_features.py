#!/usr/bin/env python3
"""Extract log-Mel features from triggered test WAV files."""

import argparse
from pathlib import Path

import numpy as np

try:
    from .common import load_config, resolve_path
    from .extract_features import crop_or_pad, extract_melspectrogram
    from .poison_manifest import propagate_manifest
except ImportError:  # Direct script execution.
    from common import load_config, resolve_path
    from extract_features import crop_or_pad, extract_melspectrogram
    from poison_manifest import propagate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    import librosa

    config = load_config()
    source = resolve_path(args.input or config["path"]["poison_test_path"])
    output = resolve_path(args.output or config["path"]["poison_test_npypath"])
    existing = list(output.rglob("*.npy")) if output.exists() else []
    if existing and not args.overwrite:
        parser.error(f"{output} contains features; pass --overwrite")
    if args.overwrite:
        for path in existing:
            path.unlink()
    audio_config = config["librosa"]
    count = 0
    for wav_path in sorted(source.rglob("*.wav")):
        audio, _ = librosa.load(wav_path, sr=audio_config["sr"], mono=True)
        audio = crop_or_pad(audio, audio_config["sr"])
        feature = extract_melspectrogram(audio, audio_config["sr"], audio_config["hop_length"],
                                         audio_config["n_fft"], audio_config["n_mels"])
        destination = output / wav_path.relative_to(source).with_suffix(".npy")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, feature.numpy())
        count += 1
    propagate_manifest(source, output)
    print(f"processed={count} output={output}")


if __name__ == "__main__":
    main()
