#!/usr/bin/env python3
"""Create deterministic Speech Commands train/validation/test splits."""

import argparse
import random
import shutil
from pathlib import Path

try:
    from .common import get_classes
except ImportError:  # Direct script execution.
    from common import get_classes


def deterministic_split(files, test_ratio=0.1, validation_ratio=0.1, seed=42):
    files = sorted(Path(path) for path in files)
    shuffled = files[:]
    random.Random(seed).shuffle(shuffled)
    test_count = int(len(files) * test_ratio)
    validation_count = int(len(files) * validation_ratio)
    test = set(shuffled[:test_count])
    validation = set(shuffled[test_count:test_count + validation_count])
    train = [path for path in files if path not in test and path not in validation]
    return train, [path for path in files if path in validation], [path for path in files if path in test]


def speech_commands_split(source, classes, test_ratio=0.1, validation_ratio=0.1, seed=42):
    source = Path(source).resolve()
    files = [path for name in classes for path in (source / name).glob("*.wav")]
    testing_list = source / "testing_list.txt"
    validation_list = source / "validation_list.txt"
    if testing_list.exists() or validation_list.exists():
        test = {source / relative for relative in testing_list.read_text(encoding="utf-8").splitlines()} if testing_list.exists() else set()
        validation = {source / relative for relative in validation_list.read_text(encoding="utf-8").splitlines()} if validation_list.exists() else set()
        train = [path for path in sorted(files) if path not in test and path not in validation]
        return train, [path for path in sorted(files) if path in validation], [path for path in sorted(files) if path in test]
    return deterministic_split(files, test_ratio, validation_ratio, seed)


def materialize(files, source, destination, link=False):
    for path in files:
        output = destination / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to replace {output}; pass --overwrite")
        output.symlink_to(path) if link else shutil.copy2(path, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--num-classes", type=int, choices=(10, 30), default=10)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link", action="store_true", help="Symlink instead of copying WAV files")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.test_ratio <= 1 or not 0 <= args.validation_ratio <= 1:
        parser.error("split ratios must be between 0 and 1")
    if args.test_ratio + args.validation_ratio > 1:
        parser.error("--test-ratio and --validation-ratio cannot sum above 1")
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() and any(output.rglob("*.wav")) and not args.overwrite:
        parser.error(f"{output} already contains WAV files; pass --overwrite")
    train, validation, test = speech_commands_split(
        source, get_classes(args.num_classes), args.test_ratio, args.validation_ratio, args.seed
    )
    if args.overwrite and output.exists():
        for path in output.rglob("*.wav"):
            path.unlink()
    materialize(train, source, output / "train", args.link)
    materialize(validation, source, output / "validation", args.link)
    materialize(test, source, output / "test", args.link)
    print(f"train={len(train)} validation={len(validation)} test={len(test)} output={output}")


if __name__ == "__main__":
    main()
