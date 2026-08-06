"""First-party AudioSeal adapter; the vendored source remains unmodified."""

import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional

try:
    from .common import REPO_ROOT
except ImportError:  # Direct script/module execution from SR/.
    from common import REPO_ROOT


AUDIOSEAL_ROOT = REPO_ROOT / "audioseal"
AUDIOSEAL_SRC = AUDIOSEAL_ROOT / "src"


def import_audioseal():
    source = str(AUDIOSEAL_SRC)
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from audioseal import AudioSeal
    except ImportError as error:
        raise ImportError(
            f"Unable to import vendored AudioSeal from {AUDIOSEAL_SRC}. "
            "Install its runtime dependencies from the repository requirements."
        ) from error
    return AudioSeal


class _LoRAConv1d(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scaling = float(alpha) / rank
        out_channels, in_channels = base.weight.shape[:2]
        self.lora_A = nn.Conv1d(in_channels, rank, 1, bias=False)
        kernel = base.kernel_size[0]
        self.lora_T = nn.Conv1d(rank, rank, kernel, stride=base.stride[0],
                                padding=base.padding[0], dilation=base.dilation[0],
                                groups=rank, bias=False)
        self.lora_B = nn.Conv1d(rank, out_channels, 1, bias=False)

    def forward(self, value):
        base = self.base(value)
        update = self.lora_B(self.lora_T(self.lora_A(value))) * self.scaling
        if update.shape[-1] != base.shape[-1]:
            update = functional.pad(update[..., :base.shape[-1]], (0, max(0, base.shape[-1] - update.shape[-1])))
        return base + update


class _LoRAConvTranspose1d(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scaling = float(alpha) / rank
        self.lora_A = nn.Conv1d(base.in_channels, rank, 1, bias=False)
        self.lora_B = nn.Conv1d(rank, base.in_channels, 1, bias=False)

    def forward(self, value):
        return self.base(value + self.lora_B(self.lora_A(value)) * self.scaling)


def inject_lora(model, rank, alpha):
    try:
        from audioseal.libs.audiocraft.modules.conv import ParametrizedConv1d, ParametrizedConvTranspose1d
        conv_types = (nn.Conv1d, ParametrizedConv1d)
        transpose_types = (nn.ConvTranspose1d, ParametrizedConvTranspose1d)
    except ImportError:
        conv_types, transpose_types = (nn.Conv1d,), (nn.ConvTranspose1d,)
    replaced = []
    for name, module in list(model.named_modules()):
        if not re.search(r"^decoder\.", name):
            continue
        wrapper = None
        if isinstance(module, conv_types):
            wrapper = _LoRAConv1d(module, rank, alpha)
        elif isinstance(module, transpose_types):
            wrapper = _LoRAConvTranspose1d(module, rank, alpha)
        if wrapper is not None:
            parent_name, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), child, wrapper)
            replaced.append(name)
    if not replaced:
        raise RuntimeError("No AudioSeal decoder layers were eligible for LoRA injection")
    return model


def checkpoint_lora_config(checkpoint, rank=None, alpha=None):
    metadata = checkpoint.get("args", {})
    metadata = vars(metadata) if hasattr(metadata, "__dict__") else metadata
    rank = rank if rank is not None else metadata.get("lora_rank")
    alpha = alpha if alpha is not None else metadata.get("lora_alpha")
    state = checkpoint.get("state_dict", checkpoint)
    if rank is None:
        first = next((value for key, value in state.items() if key.endswith("lora_A.weight")), None)
        rank = first.shape[0] if first is not None else None
    if rank is None or alpha is None:
        raise ValueError("LoRA checkpoint must contain args.lora_rank/args.lora_alpha; provide missing CLI overrides")
    return int(rank), float(alpha), state


class AudioSealWatermarker:
    def __init__(self, mode="base", checkpoint_path=None, model_card="audioseal_wm_16bits",
                 rank=None, alpha=None, sample_rate=16000, watermark_scale=None, device=None):
        if mode not in ("base", "lora-ft"):
            raise ValueError("mode must be 'base' or 'lora-ft'")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sample_rate = sample_rate
        self.watermark_scale = (5.0 if mode == "base" else 1.0) if watermark_scale is None else watermark_scale
        self.generator = import_audioseal().load_generator(model_card)
        if mode == "lora-ft":
            if checkpoint_path is None:
                raise ValueError("--checkpoint is required for lora-ft mode")
            checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
            rank, alpha, state = checkpoint_lora_config(checkpoint, rank, alpha)
            self.generator = inject_lora(self.generator, rank, alpha)
            missing, unexpected = self.generator.load_state_dict(state, strict=False)
            missing_lora = [key for key in missing if "lora_" in key]
            if missing_lora or any("lora_" in key for key in unexpected):
                raise RuntimeError("Checkpoint LoRA tensors do not match the reconstructed adapter")
        self.generator.to(self.device).eval()

    @torch.no_grad()
    def embed_watermark(self, audio, sample_rate):
        wave = torch.as_tensor(audio, dtype=torch.float32, device=self.device)[None, None]
        if sample_rate != self.sample_rate:
            length = int(round(wave.shape[-1] * self.sample_rate / sample_rate))
            wave = functional.interpolate(wave, size=length, mode="linear", align_corners=False)
        dtype = next(self.generator.parameters()).dtype
        wave = wave.to(dtype=dtype)
        watermark = self.generator.get_watermark(wave, self.sample_rate)
        output = torch.clamp(wave + self.watermark_scale * watermark, -1, 1)
        return output[0, 0].float().cpu().numpy(), self.sample_rate
