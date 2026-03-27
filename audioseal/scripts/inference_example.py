#!/usr/bin/env python3
import os
import sys
import torch
import torchaudio
from audioseal import AudioSeal

# --- Import LoRA helper from the same folder ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

try:
    from LoRA_finetune import inject_lora_into_decoder
except ImportError as e:
    raise ImportError(
        "inject_lora_into_decoder not found. Make sure LoRA_finetune.py is in the same folder "
        "or add its path to PYTHONPATH."
    ) from e

# ======== Parameters ========
CKPT = "lora_wm5_ckpts_dec_lstm_r256_sup2000/lora_wm5_best.pth"   # path to your LoRA checkpoint
MODEL_CARD = "audioseal_wm_16bits"                 # same model card as training
RANK = 256                                         # same as --lora-rank
ALPHA = 512.0                                      # same as --lora-alpha
IN_WAV = "00005.wav"                               # input audio file
OUT_WAV = "00005_watermarked.wav"                  # output watermarked file
TARGET_SR = 16000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_wav(path: str, target_sr: int, device: str):
    wav, sr = torchaudio.load(path)  # (C, T), float32 in [-1,1]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    if wav.size(0) > 1:  # convert to mono
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0)  # -> (B=1, C=1, T)
    return wav.to(device), sr


def main():
    # Load base generator
    G = AudioSeal.load_generator(MODEL_CARD).to(DEVICE).eval()

    # Inject LoRA modules into decoder
    G, _ = inject_lora_into_decoder(
        G, rank=RANK, alpha=ALPHA,
        include_patterns=[r"^decoder\."], exclude_patterns=[], trace=False
    )

    # Load LoRA finetuned checkpoint
    ckpt = torch.load(CKPT, map_location=DEVICE)
    missing, unexpected = G.load_state_dict(ckpt["state_dict"], strict=False)
    print("[load_state_dict] missing:", missing)
    print("[load_state_dict] unexpected:", unexpected)

    # Load audio
    wav, sr = load_wav(IN_WAV, TARGET_SR, DEVICE)

    # Generate watermark
    with torch.no_grad():
        wm_hat = G.get_watermark(wav, sr)  # (B,1,T)
        watermarked = torch.clamp(wav + wm_hat, -1.0, 1.0)

    # Save watermarked audio
    torchaudio.save(OUT_WAV, watermarked.squeeze(0).cpu(), sr)
    print(f"[save] wrote {OUT_WAV}")

    # ---- Detection (disabled for now) ----
    # det = AudioSeal.load_detector("audioseal_detector_16bits").to(DEVICE).eval()
    # with torch.no_grad():
    #     prob, msg = det.detect_watermark(watermarked, sr)
    # print("prob_watermarked:", float(prob))
    # print("message bits:", msg)


if __name__ == "__main__":
    main()