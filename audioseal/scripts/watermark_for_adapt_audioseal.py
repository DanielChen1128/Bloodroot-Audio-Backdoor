#!/usr/bin/env python3
# watermark_for_adapt_audioseal.py
import os, sys, gc
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Tuple, List
import torch, torchaudio
from audioseal import AudioSeal
from tqdm import tqdm

# -------- CLI args --------
import argparse
p = argparse.ArgumentParser("AudioSeal watermark with GPU first, CPU fallback (chunked)")
p.add_argument("--src", required=True, help="Source root (wav/ogg mixed)")
p.add_argument("--out-wm1", required=True, help="Output root for 1× WM (wav)")
p.add_argument("--out-wm5", required=True, help="Output root for 5× WM (wav)")
p.add_argument("--sr", type=int, default=16000, help="Target sample rate")
p.add_argument("--min-sec", type=float, default=0.5, help="Skip if shorter")
p.add_argument("--decode-workers", type=int, default=4, help="#threads for decoding+resample")
p.add_argument("--save-workers", type=int, default=2, help="#threads for writing wav")
p.add_argument("--fp16", type=int, default=1, help="Use autocast fp16 on CUDA (1=yes, 0=no)")
p.add_argument("--gpu-chunk-sec", type=float, default=90.0, help="GPU chunk seconds")
p.add_argument("--cpu-chunk-sec", type=float, default=30.0, help="CPU chunk seconds")
p.add_argument("--skip-long-min", type=float, default=180.0, help="Skip if audio longer than N minutes")
p.add_argument("--exts", default=".wav,.ogg", help="Comma list of audio exts")
p.add_argument("--skip-suffixes", default=".partial,.tar,.tar.gz,.tgz", help="Comma list to ignore")
p.add_argument("--model", default="audioseal_wm_16bits", help="AudioSeal generator name")
p.add_argument("--resume", type=int, default=1, help="Skip if both outputs exist (1=yes)")
args = p.parse_args()

SRC_DIR     = os.path.abspath(args.src)
WM1_DIR     = os.path.abspath(args.out_wm1)
WM5_DIR     = os.path.abspath(args.out_wm5)
SR_TARGET   = args.sr
MIN_SEC     = args.min_sec
DECODE_N    = max(1, args.decode_workers)
SAVE_N      = max(1, args.save_workers)
USE_FP16    = bool(args.fp16)
MODEL_NAME  = args.model
AUDIO_EXTS  = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
SKIP_SUFFIX = tuple(s.strip().lower() for s in args.skip_suffixes.split(",") if s.strip())
RESUME      = bool(args.resume)

os.makedirs(WM1_DIR, exist_ok=True)
os.makedirs(WM5_DIR, exist_ok=True)

# ---- setup torch ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

try:
    import soundfile as sf
    USE_SF = True
except Exception:
    USE_SF = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tqdm.write(f"Using device: {device} | fp16_default={int(USE_FP16 and device.type=='cuda')} | decode_workers={DECODE_N} save_workers={SAVE_N}")

# ---- resampler ----
_RESAMPLERS = {}
def _resample_cached(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return wav
    key = (sr_in, sr_out)
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = torchaudio.transforms.Resample(sr_in, sr_out)
    return _RESAMPLERS[key](wav)

def load_mono_resampled(path: str, sr_out: int) -> torch.Tensor:
    if USE_SF:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
    else:
        wav, sr = torchaudio.load(path)
        wav = wav.to(torch.float32)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return _resample_cached(wav, sr, sr_out)

def save_wav(dst_path: str, wav_1xT: torch.Tensor, sr: int):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    wav = torch.clamp(wav_1xT, -0.99, 0.99).to(torch.float32)
    torchaudio.save(dst_path, wav.cpu(), sr, format="wav")

def to_wav_path(dst_root: str, rel_dir: str, filename: str) -> str:
    base, _ = os.path.splitext(filename)
    return os.path.join(dst_root, rel_dir, base + ".wav")

def is_audio(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in AUDIO_EXTS

def is_ignored(name: str) -> bool:
    return name.lower().endswith(SKIP_SUFFIX)

# ---- generator ----
def reload_generator(target: torch.device):
    try:
        del reload_generator.generator
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    reload_generator.generator = AudioSeal.load_generator(MODEL_NAME).to(target)

reload_generator(device)
generator = lambda: reload_generator.generator

# ---- watermark funcs ----
def gen_wm_chunked(wav_1xT, sr, target, chunk_sec, fp16=False):
    T = wav_1xT.shape[1]
    step = max(1, int(chunk_sec * sr))
    outs = []
    with torch.no_grad():
        for s in range(0, T, step):
            e = min(T, s + step)
            x = wav_1xT[:, s:e].unsqueeze(0)  # [1,1,t]
            if target.type == "cuda":
                x = x.pin_memory().to(target, non_blocking=True)
                if fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        y = generator().get_watermark(x, sr)
                else:
                    y = generator().get_watermark(x, sr)
            else:
                y = generator().get_watermark(x, sr)
            outs.append(y.squeeze(0).to("cpu"))
    return torch.cat(outs, dim=1)

def gen_wm_gpu(wav, sr):
    return gen_wm_chunked(wav, sr, device, args.gpu_chunk_sec, fp16=USE_FP16)

def gen_wm_cpu(wav, sr):
    return gen_wm_chunked(wav, sr, torch.device("cpu"), args.cpu_chunk_sec, fp16=False)

# ---- main ----
def scan_items(root: str) -> List[Tuple[str, str, str]]:
    items = []
    for r, _, files in os.walk(root):
        rel = os.path.relpath(r, root)
        if rel == ".": rel = ""
        for f in files:
            if is_ignored(f): continue
            items.append((rel, os.path.join(r, f), f))
    return items

def main():
    items = scan_items(SRC_DIR)
    tqdm.write(f"Found {len(items)} files")

    decoder = ThreadPoolExecutor(max_workers=DECODE_N)
    writer  = ThreadPoolExecutor(max_workers=SAVE_N)
    pending = []
    MAX_INFLIGHT = DECODE_N * 4
    pbar = tqdm(total=len(items), desc="Watermarking", unit="file")

    def submit(job): 
        rel, path, fname = job
        return decoder.submit(load_mono_resampled, path, SR_TARGET), rel, path, fname

    i = 0
    while i < min(len(items), MAX_INFLIGHT):
        pending.append(submit(items[i])); i += 1

    try:
        while pending:
            futs = [f for (f, *_ ) in pending]
            done, _ = wait(futs, return_when=FIRST_COMPLETED)
            fut_done = next(iter(done))
            idx = next(k for k,(f, *_ ) in enumerate(pending) if f is fut_done)
            fut, rel, path, fname = pending.pop(idx)
            if i < len(items): pending.append(submit(items[i])); i += 1

            out1 = to_wav_path(WM1_DIR, rel, fname)
            out5 = to_wav_path(WM5_DIR, rel, fname)

            if not is_audio(fname):
                from shutil import copy2
                os.makedirs(os.path.dirname(out1), exist_ok=True)
                os.makedirs(os.path.dirname(out5), exist_ok=True)
                try: copy2(path, out1); copy2(path, out5)
                except: pass
                pbar.update(1); continue

            if RESUME and os.path.exists(out1) and os.path.exists(out5):
                pbar.update(1); continue

            try:
                wav = fut.result()
            except Exception as e:
                tqdm.write(f"[ERROR] decode {path}: {e}")
                pbar.update(1); continue

            length_min = wav.shape[1] / SR_TARGET / 60.0
            if wav.shape[1] / SR_TARGET < MIN_SEC:
                pbar.update(1); continue
            if length_min > args.skip_long_min:
                tqdm.write(f"[SKIP] too long ({length_min:.1f} min): {path}")
                pbar.update(1); continue

            try:
                if device.type=="cuda":
                    try:
                        wm = gen_wm_gpu(wav, SR_TARGET)
                    except Exception as e:
                        tqdm.write(f"[CUDA ERR] {e} → CPU fallback: {path}")
                        reload_generator(torch.device("cpu"))
                        wm = gen_wm_cpu(wav, SR_TARGET)
                        if torch.cuda.is_available(): reload_generator(device)
                else:
                    wm = gen_wm_cpu(wav, SR_TARGET)

                writer.submit(save_wav, out1, wm, SR_TARGET)
                writer.submit(save_wav, out5, 5.0 * wm, SR_TARGET)

            except Exception as e:
                tqdm.write(f"[ERROR] wm {path}: {e}")
            finally:
                del wav
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            pbar.update(1)
    finally:
        decoder.shutdown(wait=True)
        writer.shutdown(wait=True)
        pbar.close()
        tqdm.write("Done.")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nInterrupted.", file=sys.stderr)