"""Persist and validate poisoning settings across pipeline stages."""

import json
from pathlib import Path


MANIFEST_NAME = "_bloodroot_manifest.json"


def write_manifest(root, settings):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_NAME
    path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_manifest(root):
    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing poison manifest {path}; regenerate triggers with embed_trigger.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def propagate_manifest(source, destination):
    manifest = read_manifest(source)
    write_manifest(destination, manifest)
    return manifest


def validate_manifest(manifest, *, classes, target_label=None, label_mode=None):
    expected_classes = list(classes)
    if manifest.get("classes") != expected_classes:
        raise ValueError(
            f"Poison manifest class map {manifest.get('classes')} does not match {expected_classes}"
        )
    if target_label is not None and manifest.get("target_label") != target_label:
        raise ValueError(
            f"Poison manifest target {manifest.get('target_label')!r} does not match {target_label!r}"
        )
    if label_mode is not None and manifest.get("label_mode") != label_mode:
        raise ValueError(
            f"Poison manifest label mode {manifest.get('label_mode')!r} does not match {label_mode!r}"
        )
    return manifest
