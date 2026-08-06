#!/usr/bin/env python3
"""Embed base AudioSeal or LoRA-fine-tuned watermark triggers."""

import argparse
import math
import random
from pathlib import Path

try:
    from .common import get_classes, load_config, resolve_path
    from .poison_manifest import write_manifest
except ImportError:  # Direct script execution.
    from common import get_classes, load_config, resolve_path
    from poison_manifest import write_manifest


def select_poison_files(root, classes, target_label, poison_rate, seed, max_samples=None):
    root = Path(root)
    all_files = [path for name in classes for path in sorted((root / name).glob("*.wav"))]
    eligible = [path for path in all_files if path.parent.name != target_label]
    count = math.floor(poison_rate * len(all_files))
    if max_samples is not None:
        count = min(count, max_samples)
    if count > len(eligible):
        raise ValueError(f"Requested {count} poisons but only {len(eligible)} non-target samples exist")
    return random.Random(seed).sample(eligible, count), len(all_files)


def prepare_output(path, overwrite):
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.rglob("*.wav"))
    if existing and not overwrite:
        raise FileExistsError(f"{path} contains WAV files; pass --overwrite to replace generated WAVs")
    if overwrite:
        for wav in existing:
            wav.unlink()


def poison_output(original_class, filename, target_label, label_mode):
    if label_mode == "label-flip":
        return target_label, f"{original_class}_{filename}"
    if label_mode == "clean-label":
        return original_class, filename
    raise ValueError("label_mode must be 'label-flip' or 'clean-label'")


def process_files(files, output, target_label, label_mode, watermarker):
    try:
        import soundfile
    except ImportError as error:
        raise ImportError("soundfile is required to read and write trigger WAV files") from error
    output = Path(output)
    for input_path in files:
        audio, sample_rate = soundfile.read(input_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        watermarked, output_rate = watermarker.embed_watermark(audio, sample_rate)
        original_class = input_path.parent.name
        output_class, name = poison_output(original_class, input_path.name, target_label, label_mode)
        destination = output / output_class / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(destination, watermarked, output_rate)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-classes", type=int, choices=(10, 30), default=10)
    parser.add_argument("--target-label", default="left")
    parser.add_argument("--poison-rate", type=float, default=0.01)
    parser.add_argument("--label-mode", choices=("label-flip", "clean-label"), default="label-flip")
    parser.add_argument("--trigger-mode", choices=("base", "lora-ft"), default="base")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--watermark-scale", type=float, default=None,
                        help="Output scale (default: 5 for base, 1 for lora-ft)")
    parser.add_argument("--model-card", default="audioseal_wm_16bits")
    parser.add_argument("--train-wav-path", type=Path)
    parser.add_argument("--test-wav-path", type=Path)
    parser.add_argument("--output-train-path", type=Path)
    parser.add_argument("--output-test-path", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.poison_rate <= 1:
        parser.error("--poison-rate must be between 0 and 1")
    classes = get_classes(args.num_classes)
    if args.target_label not in classes:
        parser.error(f"--target-label must be one of: {', '.join(classes)}")
    config = load_config()
    paths = config["path"]
    train_source = resolve_path(args.train_wav_path or paths["benign_train_wavpath"])
    test_source = resolve_path(args.test_wav_path or paths["benign_test_wavpath"])
    train_output = resolve_path(args.output_train_path or paths["poison_train_path"])
    test_output = resolve_path(args.output_test_path or paths["poison_test_path"])
    prepare_output(train_output, args.overwrite)
    if not args.skip_test:
        prepare_output(test_output, args.overwrite)
    selected, total = select_poison_files(train_source, classes, args.target_label,
                                          args.poison_rate, args.seed, args.max_samples)
    try:
        from .audioseal_integration import AudioSealWatermarker
    except ImportError:
        from audioseal_integration import AudioSealWatermarker
    watermarker = AudioSealWatermarker(args.trigger_mode, args.checkpoint, args.model_card,
                                       args.lora_rank, args.lora_alpha,
                                       watermark_scale=args.watermark_scale)
    poisoned = process_files(selected, train_output, args.target_label,
                             args.label_mode, watermarker)
    test_count = 0
    if not args.skip_test:
        test_files = [path for name in classes if name != args.target_label
                      for path in sorted((test_source / name).glob("*.wav"))]
        test_count = process_files(test_files, test_output, args.target_label,
                                   "clean-label", watermarker)
    manifest = {
        "version": 1,
        "classes": list(classes),
        "num_classes": args.num_classes,
        "target_label": args.target_label,
        "label_mode": args.label_mode,
        "trigger_mode": args.trigger_mode,
        "watermark_scale": watermarker.watermark_scale,
        "poison_rate": args.poison_rate,
        "seed": args.seed,
        "poisoned_train": poisoned,
        "training_samples": total,
    }
    write_manifest(train_output, manifest)
    if not args.skip_test:
        write_manifest(test_output, manifest)
    print(f"poisoned_train={poisoned}/{total} triggered_test={test_count}")


if __name__ == "__main__":
    main()
