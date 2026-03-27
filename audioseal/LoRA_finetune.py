#!/usr/bin/env python3
"""
LoRA fine-tune of AudioSeal generator (decoder only) to emit 5× watermark.

Loss = λ_sup * L1(wm_hat, wm5)
     + λ_stft * MultiScale STFT mag/log-mag L1 (x vs x+wm_hat)
     + λ_mel * Log-Mel L1 (x vs x+wm_hat)
     + λ_amp * mean(wm_hat^2)

Data layout (mirror tree):
  raw_audios_npy/<rel>.npy      # clean audio
  raw_audios_wm5_npy/<rel>.npy  # target 5× watermark (same length)
"""

# =========================
# Imports & Logging helpers
# =========================
import os
import re
import gc
import time
import math
import random
import logging
from pathlib import Path
from typing import List, Tuple 

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from audioseal import AudioSeal

LOG = logging.getLogger("lora_audioseal")


def setup_logging(level: str = "INFO"):
    level = level.upper()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-7s | %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level, logging.INFO),
    )
    LOG.setLevel(getattr(logging, level, logging.INFO))
    # 降噪：把一些雜訊 logger 調低
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("torchaudio").setLevel(logging.WARNING)


# =========================
# AudioSeal conv type probes
# =========================
try:
    from audioseal.libs.audiocraft.modules.conv import (
        ParametrizedConv1d, ParametrizedConvTranspose1d
    )
    _CONV1D_TYPES = (nn.Conv1d, ParametrizedConv1d)
    _CONVT_TYPES = (nn.ConvTranspose1d, ParametrizedConvTranspose1d)
except Exception:
    _CONV1D_TYPES = (nn.Conv1d,)
    _CONVT_TYPES = (nn.ConvTranspose1d,)

# ============
# Small utils
# ============


def set_deterministic(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sec_to_samples(sr: int, seconds: float) -> int:
    return max(1, int(round(sr * float(seconds))))


def print_trainable(model: nn.Module, tag: str = "model"):
    t_all, a_all = 0, 0
    t_dec, a_dec = 0, 0

    for name, p in model.named_parameters():
        a_all += p.numel()
        if p.requires_grad:
            t_all += p.numel()

        if name.startswith("decoder."):
            a_dec += p.numel()
            if p.requires_grad:
                t_dec += p.numel()

    LOG.info("[%s] trainable=%s / all=%s (%.2f%%)",
             tag, f"{t_all:,}", f"{a_all:,}", 100.0*t_all/max(1, a_all))

    LOG.info("[%s.decoder] trainable=%s / all=%s (%.2f%% of decoder, %.2f%% of total)",
             tag,
             f"{t_dec:,}", f"{a_dec:,}",
             100.0*t_dec/max(1, a_dec),
             100.0*t_dec/max(1, a_all))


def tensor_debug(name: str, x: torch.Tensor):
    LOG.debug("%s: shape=%s dtype=%s device=%s min=%.4f max=%.4f mean=%.4f",
              name, tuple(x.shape), x.dtype, x.device,
              float(torch.nan_to_num(x.min()).item()),
              float(torch.nan_to_num(x.max()).item()),
              float(torch.nan_to_num(x.mean()).item()))


def check_finite(name: str, x: torch.Tensor):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"Non-finite values in {name}")

# =====================
# LoRA wrappers & hook
# =====================


class _AttrProxyMixin:
    def __getattr__(self, name):

        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        base = super().__getattribute__("_modules").get("base", None)
        if base is not None and hasattr(base, name):
            return getattr(base, name)

        raise AttributeError(
            f"{self.__class__.__name__} has no attribute '{name}'")


class LoRAConv1d(_AttrProxyMixin, nn.Module):
    def __init__(self, base: nn.Module, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.add_module("base", base)
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = int(r)
        self.scaling = float(alpha) / max(1, self.r)

        C_out, C_in = self.base.weight.shape[:2]

        self.lora_A = nn.Conv1d(C_in, self.r, kernel_size=1, bias=False)

        k = getattr(self.base, "kernel_size", (1,))[0]
        s = getattr(self.base, "stride", (1,))[0]
        p = getattr(self.base, "padding", (0,))[0]
        d = getattr(self.base, "dilation", (1,))[0]
        self.lora_T = nn.Conv1d(self.r, self.r, kernel_size=k, stride=s, padding=p,
                                dilation=d, groups=self.r, bias=False)

        self.lora_B = nn.Conv1d(self.r, C_out, kernel_size=1, bias=False)

        # init
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_T.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @staticmethod
    def _match_time_to(ref: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        T_ref = ref.size(-1)
        T = y.size(-1)
        if T == T_ref:
            return y
        if T > T_ref:
            return y[..., :T_ref]

        pad = T_ref - T
        return torch.nn.functional.pad(y, (0, pad))

    def forward(self, x):
        y_base = self.base(x)

        if (self.lora_A.weight.dtype != x.dtype) or (self.lora_A.weight.device != x.device):
            self.lora_A.to(device=x.device, dtype=x.dtype)
            self.lora_T.to(device=x.device, dtype=x.dtype)
            self.lora_B.to(device=x.device, dtype=x.dtype)

        y_lora = self.lora_B(self.lora_T(self.lora_A(x))) * self.scaling

        if y_lora.size(-1) != y_base.size(-1):
            y_lora = self._match_time_to(y_base, y_lora)

        return y_base + y_lora


class LoRAConvTranspose1d(_AttrProxyMixin, nn.Module):
    def __init__(self, base: nn.Module, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.add_module("base", base)
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = int(r)
        self.scaling = float(alpha) / max(1, self.r)

        C_in = self.base.in_channels

        self.lora_A = nn.Conv1d(C_in, self.r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv1d(self.r, C_in, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        if (self.lora_A.weight.dtype != x.dtype) or (self.lora_A.weight.device != x.device):
            self.lora_A.to(device=x.device, dtype=x.dtype)
            self.lora_B.to(device=x.device, dtype=x.dtype)

        y_res_in = self.lora_B(self.lora_A(x)) * self.scaling
        x_pert = x + y_res_in
        y_out = self.base(x_pert)

        # try:
        #     print(f"[LoRAConvT] x={tuple(x.shape)}, x_pert={tuple(x_pert.shape)}, "
        #           f"y_out={tuple(y_out.shape)}, stride={self.base.stride}, "
        #           f"padding={self.base.padding}, kernel_size={self.base.kernel_size}, "
        #           f"output_padding={getattr(self.base, 'output_padding', None)}")
        # except Exception:
        #     pass

        return y_out


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if ("lora_" in n)]


def freeze_all_but_lora(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in lora_parameters(model):
        p.requires_grad_(True)


def inject_lora_into_decoder(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    include_patterns: List[str] = (r"^decoder\.",),
    exclude_patterns: List[str] = (),
    trace: bool = False
):
    inc_res = [re.compile(p) for p in include_patterns]
    exc_res = [re.compile(p) for p in exclude_patterns]

    def _in_decoder(name: str) -> bool:
        if any(r.search(name) for r in exc_res):
            return False
        return any(r.search(name) for r in inc_res)
    replaced = []
    for fullname, mod in list(model.named_modules()):
        if not _in_decoder(fullname):
            continue
        if isinstance(mod, _CONV1D_TYPES) or isinstance(mod, _CONVT_TYPES):
            parent_name = ".".join(fullname.split(".")[:-1])
            child_name = fullname.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            wrapped = LoRAConv1d(mod, r=rank, alpha=alpha) if isinstance(mod, _CONV1D_TYPES) \
                else LoRAConvTranspose1d(mod, r=rank, alpha=alpha)
            setattr(parent, child_name, wrapped)
            replaced.append(fullname)
            if trace:
                LOG.debug("[LoRA] replaced -> %s (%s)",
                          fullname, type(mod).__name__)
    return model, replaced

# =========
# Losses
# =========


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(512, 1024, 2048), hop_scales=(0.25, 0.25, 0.25), win_scales=(1.0, 1.0, 1.0), eps=1e-7):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_scales = hop_scales
        self.win_scales = win_scales
        self.eps = eps

    def forward(self, x, y):  # x,y: (B,1,T)
        x32 = x.float()
        y32 = y.float()
        loss = 0.0
        for n_fft, hop_s, win_s in zip(self.fft_sizes, self.hop_scales, self.win_scales):
            win_length = int(n_fft * win_s)
            hop_length = max(1, int(n_fft * hop_s))
            window = torch.hann_window(
                win_length, device=x32.device, dtype=x32.dtype)
            X = torch.stft(x32.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                           window=window, return_complex=True, center=True)
            Y = torch.stft(y32.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                           window=window, return_complex=True, center=True)
            magX, magY = X.abs(), Y.abs()
            lin = torch.mean(torch.abs(magX - magY))
            log = torch.mean(
                torch.abs(torch.log(magX + self.eps) - torch.log(magY + self.eps)))
            loss = loss + lin + log
        return loss / float(len(self.fft_sizes))


class LogMelLoss(nn.Module):
    def __init__(self, sr: int, n_mels: int = 64, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, x, y):  # (B,1,T)
        self.mel = self.mel.to(device=x.device)
        self.to_db = self.to_db.to(device=x.device)

        X = self.to_db(self.mel(x.float()))
        Y = self.to_db(self.mel(y.float()))
        return torch.mean(torch.abs(X - Y))

# ==========
# Dataset
# ==========


class PairNpyDataset(Dataset):
    """Mirror-paired (raw, wm5) .npy files; random crop to fixed-length segments."""

    def __init__(self, raw_root: Path, wm5_root: Path, sr: int,
                 segment_sec: float = 8.0, min_sec: float = 1.0,
                 limit_pairs: int = 10, selection_seed: int = 12345):
        super().__init__()
        self.raw_root = Path(raw_root)
        self.wm5_root = Path(wm5_root)
        self.sr = int(sr)
        self.seg_len = sec_to_samples(sr, segment_sec)
        self.min_len = sec_to_samples(sr, min_sec)
        pairs: List[Tuple[Path, Path]] = []
        raw_files = sorted(self.raw_root.rglob("*.npy"))
        for rf in raw_files:
            rel = rf.relative_to(self.raw_root)
            wf = (self.wm5_root / rel)
            if wf.exists():
                pairs.append((rf, wf))
        if not pairs:
            raise RuntimeError("No paired .npy files found.")
        random.Random(selection_seed).shuffle(pairs)
        if limit_pairs and limit_pairs > 0:
            pairs = pairs[:limit_pairs]
        self.pairs = pairs
        LOG.info("[Dataset] paired files (used): %d | seg=%.1fs",
                 len(self.pairs), segment_sec)

    def __len__(self): return len(self.pairs)

    def _load(self, p: Path) -> torch.Tensor:
        a = np.load(str(p))
        if a.ndim > 1:
            a = np.mean(a, axis=0)
        a = a.astype(np.float32)
        if not np.all(np.isfinite(a)):
            raise RuntimeError(f"Non-finite array read: {p}")
        return torch.from_numpy(a)  # (T,)

    def _rand_crop(self, x: torch.Tensor) -> torch.Tensor:
        T = x.numel()
        if T < self.min_len:
            x = torch.nn.functional.pad(x, (0, self.min_len - T))
            T = x.numel()
        if T <= self.seg_len:
            if T < self.seg_len:
                x = torch.nn.functional.pad(x, (0, self.seg_len - T))
            return x[:self.seg_len]
        start = random.randint(0, T - self.seg_len)
        return x[start:start + self.seg_len]

    def __getitem__(self, idx: int):
        rp, wp = self.pairs[idx]
        x = self._load(rp)
        w = self._load(wp)
        L = min(x.numel(), w.numel())
        x = x[:L]
        w = w[:L]
        x = self._rand_crop(x)
        w = self._rand_crop(w)
        return x.unsqueeze(0), w.unsqueeze(0)

# ==========
# Trainer
# ==========


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        LOG.info("Device: %s | AMP=%s | TF32=%s", self.device.type,
                 bool(args.fp16), bool(args.allow_tf32))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

        # 1) Load model
        t0 = time.time()
        self.G = AudioSeal.load_generator(args.model_name).to(self.device)
        LOG.info("Loaded AudioSeal generator in %.2fs", time.time()-t0)
        self.G.eval()

        # 2) Inject LoRA (decoder only), freeze base
        self.G, replaced = inject_lora_into_decoder(
            self.G,
            rank=args.lora_rank, alpha=args.lora_alpha,
            include_patterns=[r"^decoder\."],
            exclude_patterns=args.exclude_patterns,
            trace=bool(args.trace_replacements),
        )

        self.G = self.G.to(self.device)
        freeze_all_but_lora(self.G)
        self.outdir = Path(args.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
            
        LOG.info("[LoRA] Replaced %d conv modules", len(replaced))
        if args.trace_replacements:
            for n in replaced:
                LOG.info("  - %s", n)
        
        trainable_lora = [name for name, p in self.G.named_parameters()
                          if p.requires_grad and "lora_" in name]
        LOG.info("[LoRA] trainable parameter tensors: %d", len(trainable_lora))
        if args.trace_replacements:
            for name in trainable_lora:
                LOG.info("    * %s", name)
            try:
                (self.outdir / "replaced_modules.txt").write_text(
                    "\n".join(replaced) + "\n", encoding="utf-8")
                (self.outdir / "trainable_lora_params.txt").write_text(
                    "\n".join(trainable_lora) + "\n", encoding="utf-8")
                LOG.info("  ↳ wrote lists to %s and %s",
                         self.outdir / "replaced_modules.txt",
                         self.outdir / "trainable_lora_params.txt")
            except Exception as e:
                LOG.warning("Failed to write trace lists: %s", e)
        
        print_trainable(self.G, "AudioSeal+LoRA")

        # 3) Data
        self.train_set = PairNpyDataset(
            args.raw_root, args.wm5_root, args.sr,
            args.segment_sec, args.min_sec,
            limit_pairs=args.limit_pairs, selection_seed=args.select_seed
        )
        self.val_loader = None
        if args.val_ratio > 0.0 and len(self.train_set) > 1:
            n = len(self.train_set)
            n_val = max(1, int(n * args.val_ratio))
            idx = list(range(n))
            random.shuffle(idx)
            val_idx = set(idx[:n_val])
            tr_pairs, va_pairs = [], []
            for i, pair in enumerate(self.train_set.pairs):
                (va_pairs if i in val_idx else tr_pairs).append(pair)
            self.train_set.pairs = tr_pairs
            self.val_set = PairNpyDataset(
                args.raw_root, args.wm5_root, args.sr,
                args.segment_sec, args.min_sec,
                limit_pairs=len(va_pairs), selection_seed=args.select_seed
            )
            self.val_set.pairs = va_pairs
            LOG.info("[Split] train=%d  val=%d", len(tr_pairs), len(va_pairs))

        self.train_loader = DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.workers, pin_memory=True, drop_last=True)
        if args.val_ratio > 0.0:
            self.val_loader = DataLoader(self.val_set, batch_size=args.batch_size, shuffle=False,
                                         num_workers=max(1, args.workers//2), pin_memory=True, drop_last=False)

        # 4) Losses
        self.l_sup = nn.L1Loss()
        self.l_stft = MultiScaleSTFTLoss()
        self.l_mel = LogMelLoss(
            sr=args.sr, n_mels=args.mel_bins, n_fft=1024, hop_length=256)

        # 5) Optim & AMP
        self.opt = torch.optim.Adam([p for p in self.G.parameters() if p.requires_grad],
                                    lr=args.lr, betas=(0.9, 0.999))
        self.use_fp16 = bool(args.fp16) and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)

        # 6) CKPT
        self.outdir = Path(args.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        # 7) Optional dry-run: load a batch and single forward
        if args.dry_run:
            LOG.info("Dry-run enabled: sampling one batch to test shapes & forward")
            xb, wb = next(iter(self.train_loader))
            xb = xb.to(self.device)
            wb = wb.to(self.device)
            tensor_debug("x(batch)", xb)
            tensor_debug("wm5(batch)", wb)
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                wh = self.G.get_watermark(xb, args.sr)
            tensor_debug("wm_hat", wh)
            check_finite("wm_hat", wh)
            LOG.info("Dry-run OK. Exiting as requested.")
            raise SystemExit(0)

    def _forward(self, x_1xT) -> torch.Tensor:
        return self.G.get_watermark(x_1xT, self.args.sr)

    def _quality_losses(self, x, y):
        return self.l_stft(x, y), self.l_mel(x, y)

    def train(self):
        args = self.args
        best_val = float("inf")

        for epoch in range(1, args.epochs + 1):
            self.G.train()

            meter = {"sup": 0.0, "stft": 0.0,
                     "mel": 0.0, "amp": 0.0, "tot": 0.0}
            count = 0
            pbar = tqdm(self.train_loader,
                        desc=f"Train {epoch}/{args.epochs}", leave=False)

            for step, (x, wm5) in enumerate(pbar):
                x = x.to(self.device, non_blocking=True)
                wm5 = wm5.to(self.device, non_blocking=True)
                if step == 0 and LOG.level <= logging.DEBUG:
                    tensor_debug("train/x[0]", x[:1])
                    tensor_debug("train/wm5[0]", wm5[:1])

                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    wm_hat = self._forward(x)
                    y_hat = torch.clamp(x + wm_hat, -1.0, 1.0)

                    check_finite("wm_hat", wm_hat)
                    loss_sup = self.l_sup(wm_hat, wm5)
                    loss_stft, loss_mel = self._quality_losses(x, y_hat)
                    loss_amp = torch.mean(wm_hat**2)

                    loss = (args.lambda_sup * loss_sup +
                            args.lambda_stft * loss_stft +
                            args.lambda_mel * loss_mel +
                            args.lambda_amp * loss_amp)

                self.opt.zero_grad(set_to_none=True)

                if self.use_fp16:
                    # AMP 路徑
                    self.scaler.scale(loss).backward()

                    # （可選）梯度範圍偵測要放在 unscale_ 之後
                    self.scaler.unscale_(self.opt)

                    if LOG.level <= logging.DEBUG:
                        total_norm = 0.0
                        for p in self.G.parameters():
                            if p.requires_grad and p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        LOG.debug("grad_l2=%.6f", total_norm)

                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.G.parameters(
                        ) if p.requires_grad], args.grad_clip
                    )
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    # FP32 路徑
                    loss.backward()

                    if LOG.level <= logging.DEBUG:
                        total_norm = 0.0
                        for p in self.G.parameters():
                            if p.requires_grad and p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        LOG.debug("grad_l2=%.6f", total_norm)

                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.G.parameters(
                        ) if p.requires_grad], args.grad_clip
                    )
                    self.opt.step()

                # 累計 meter
                meter["sup"] += float(loss_sup.item())
                meter["stft"] += float(loss_stft.item())
                meter["mel"] += float(loss_mel.item())
                meter["amp"] += float(loss_amp.item())
                meter["tot"] += float(loss.item())
                count += 1

                # tqdm 顯示
                avg_tot = meter["tot"] / count
                pbar.set_postfix({
                    "L": f"{avg_tot:.4f}",
                    "sup": f"{meter['sup']/count:.4f}",
                    "stft": f"{meter['stft']/count:.4f}",
                    "mel": f"{meter['mel']/count:.4f}",
                })

            # ---- epoch ----
            avg = {k: (v / max(1, count)) for k, v in meter.items()}
            LOG.info(
                "[Epoch %d/%d] train: L=%.4f | sup=%.4f | stft=%.4f | mel=%.4f | amp=%.6f",
                epoch, args.epochs,
                avg["tot"], avg["sup"], avg["stft"], avg["mel"], avg["amp"]
            )

            # ---- optional val ----
            val_score = None
            if self.val_loader is not None:
                self.G.eval()
                tot, cnt = 0.0, 0
                with torch.no_grad():
                    for x, wm5 in tqdm(self.val_loader, desc="  Val", leave=False):
                        x = x.to(self.device, non_blocking=True)
                        wm5 = wm5.to(self.device, non_blocking=True)
                        with torch.cuda.amp.autocast(enabled=self.use_fp16):
                            wm_hat = self._forward(x)
                            y_hat = torch.clamp(x + wm_hat, -1.0, 1.0)
                            loss_sup = self.l_sup(wm_hat, wm5)
                            loss_stft, loss_mel = self._quality_losses(
                                x, y_hat)
                            loss_amp = torch.mean(wm_hat**2)
                            loss = (args.lambda_sup * loss_sup +
                                    args.lambda_stft * loss_stft +
                                    args.lambda_mel * loss_mel +
                                    args.lambda_amp * loss_amp)
                        tot += float(loss.item())
                        cnt += 1
                val_score = tot / max(1, cnt)
                LOG.info("[Epoch %d] val_loss=%.4f", epoch, val_score)

            # ---- save ckpt ----
            ckpt = {
                "epoch": epoch,
                "state_dict": self.G.state_dict(),
                "opt": self.opt.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, self.outdir / f"lora_wm5_epoch{epoch:03d}.pth")
            if val_score is not None and val_score < best_val:
                best_val = val_score
                torch.save(ckpt, self.outdir / "lora_wm5_best.pth")
                LOG.info("  ↳ new best saved (val=%.4f)", best_val)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

# ==========
# CLI
# ==========


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(
        "LoRA finetune AudioSeal generator to 5× watermark with perceptual losses")

    # Logging / debug
    ap.add_argument("--log-level", type=str, default="INFO",
                    help="DEBUG|INFO|WARNING|ERROR")
    ap.add_argument("--trace-replacements", type=int, default=0,
                    help="List every replaced module (1/0)")
    ap.add_argument("--dry-run", type=int, default=0,
                    help="Load one batch and do a forward, then exit (1/0)")

    # Data
    ap.add_argument("--raw-root", required=True, type=Path,
                    help="Root of clean .npy files (mirror tree)")
    ap.add_argument("--wm5-root", required=True, type=Path,
                    help="Root of 5× watermark .npy files (mirror tree)")
    ap.add_argument("--sr", type=int, default=16000, help="Sample rate")
    ap.add_argument("--segment-sec", type=float,
                    default=8.0, help="Crop segment seconds")
    ap.add_argument("--min-sec", type=float, default=1.0,
                    help="Skip/Pad shorter than this")
    ap.add_argument("--val-ratio", type=float, default=0.0,
                    help="Holdout ratio (0 disables val)")

    # Small sanity subset
    ap.add_argument("--limit-pairs", type=int, default=10,
                    help="Use only first N matched pairs (after shuffle)")
    ap.add_argument("--select-seed", type=int, default=12345,
                    help="Seed for small-set selection")

    # Model / LoRA
    ap.add_argument("--model-name", type=str, default="audioseal_wm_16bits")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--exclude-patterns", nargs="*", default=[],
                    help="Regex list to exclude from LoRA")

    # Optim
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fp16", type=int, default=1, help="Use AMP (1/0)")

    # Loss weights
    #ap.add_argument("--lambda-sup", type=float, default=4.0)
    ap.add_argument("--lambda-sup", type=float, default=20.0)
    ap.add_argument("--lambda-stft", type=float, default=0.2)
    ap.add_argument("--lambda-mel", type=float, default=0.1)
    ap.add_argument("--lambda-amp", type=float, default=1e-4)
    ap.add_argument("--mel-bins", type=int, default=64)

    # Speed
    ap.add_argument("--allow-tf32", type=int, default=1,
                    help="Allow TF32 matmul/cudnn on CUDA (1/0)")

    # Output
    ap.add_argument("--outdir", type=Path,
                    default=Path("./lora_wm5_ckpts_sanity"))

    return ap.parse_args()

# ==========
# Main
# ==========


def main():
    args = parse_args()
    setup_logging(args.log_level)
    set_deterministic(42)
    LOG.info("Args: %s", {k: v for k, v in vars(args).items()})
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if int(getattr(e, "code", 0)) != 0:
            raise
    except Exception as ex:
        LOG.exception("Fatal error: %s", ex)
        raise
