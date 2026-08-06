"""Shared configuration, paths, and class definitions for the SR pipeline."""

from pathlib import Path

import yaml


SR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SR_ROOT.parent
DEFAULT_CONFIG = SR_ROOT / "config" / "config.yaml"

SC10_CLASSES = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
SC30_CLASSES = SC10_CLASSES + (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
)


def get_classes(num_classes):
    if num_classes == 10:
        return SC10_CLASSES
    if num_classes == 30:
        return SC30_CLASSES
    raise ValueError("num_classes must be 10 or 30")


def load_config(path=None):
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_path(path):
    """Resolve configured paths relative to SR/, independent of the caller's cwd."""
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (SR_ROOT / value).resolve()
