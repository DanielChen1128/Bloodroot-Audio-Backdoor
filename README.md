<div align="center">

# Bloodroot: When Watermarking Turns Poisonous For Stealthy Backdoor

Official PyTorch implementation of **"Bloodroot: When Watermarking Turns Poisonous for Stealthy Backdoor"**, accepted at **ICASSP 2026**.

**Authors:** Kuan-Yu Chen, Yi-Cheng Lin, Jeng-Lin Li, and Jian-Jiun Ding

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2510.07909-b31b1b.svg)](https://arxiv.org/abs/2510.07909)
[![ICASSP 2026](https://img.shields.io/badge/ICASSP-2026-blue.svg)](https://2026.ieeeicassp.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

</div>

---

## 📌 Overview

**Bloodroot** studies a dual-use watermark-as-trigger framework for audio data poisoning. It embeds AudioSeal watermarks into a small portion of training samples so a victim speech model behaves normally on clean inputs but predicts a target class on triggered inputs.

> [!NOTE]
> **Key Paper Results:**  
> The paper evaluates speech recognition (SR) and speaker identification (SID), reporting relative PESQ improvements of up to 32.5% and 18.5%, respectively. At a 1% poisoning rate on SC-10, the reported ResNet-18 results are 95.01% benign accuracy and 95.09% attack success rate for base Bloodroot, and 94.82%/93.85% for Bloodroot-FT. Reported perceptual scores are PESQ 3.002/STOI 0.891 and PESQ 3.315/STOI 0.915, respectively. Under the paper's 3800 Hz low-pass defense, Bloodroot-FT retains 53.49% ASR. *(These are paper results, not bundled precomputed results.)*

---

## 💡 Method

* **Bloodroot (`base`)** uses the pretrained AudioSeal generator directly and defaults to the paper's output scale of 5.
* **Bloodroot-FT (`lora-ft`)** inserts LoRA adapters into the AudioSeal decoder, loads a fine-tuned checkpoint, and defaults to output scale 1 because the fine-tuned generator already learns the strengthened watermark target. `--watermark-scale` explicitly overrides either default.
* Training poisoning selects exactly `floor(rho * N)` non-target samples uniformly without replacement, where `N` is the full training-set size.
* `label-flip` moves poisoned samples to the target label; `clean-label` preserves their labels.
* Trigger intensity is explicit output scaling: `x_triggered = clamp(x + scale * G(x), -1, 1)`. Model weights are not mutated at inference.
* The paper's fine-tuning objective combines supervised, multi-scale STFT, log-Mel, and amplitude losses with weights 20000, 10, 10, and 0.1.

---

## 📊 Repository Status

| Component | Status | Notes |
|---|---|---|
| **SR ResNet/LSTM models and feature/training scripts** | **Released** | Present under `SR/`. |
| **Vendored AudioSeal and LoRA training source** | **Released** | Present under `audioseal/`; retained as upstream/release code. |
| **Speech Commands feature loader, deterministic splitter, path handling, and AudioSeal runtime adapter** | **Reconstructed** | First-party integration under `SR/`, based on the paper and released interfaces. |
| **SID pipeline** | **External / not reconstructed** | No `SID/` implementation is included. |
| **LoRA and victim checkpoints** | **External required** | No trained checkpoints are included. |
| **Speech Commands/VoxCeleb data and AudioSeal pretrained weights** | **External required** | AudioSeal may download weights on first use. |
| **Filtering/pruning defenses and paper baselines** | **External / not reconstructed** | Results are reported by the paper; implementations are not included here. |
| **Optional KWT backbone** | **Not released** | `SR/models/kwt.py` is absent; use ResNet-18 or LSTM. |

---

## 📁 Structure

```text
bloodroot/
├── audioseal/                 # Vendored AudioSeal and released LoRA training code
├── SR/
│   ├── datasets/              # Reconstructed feature dataset loader
│   ├── models/                # Released victim model architectures
│   ├── audioseal_integration.py
│   ├── split_speech_commands.py
│   ├── embed_trigger.py
│   ├── extract_features.py
│   ├── extract_poison_features.py
│   ├── train.py
│   └── evaluate.py
└── tests/                     # Offline synthetic tests

```

---

## 🛠️ Setup

Python 3.9+ is recommended. Install a platform-appropriate PyTorch build, then install the runtime requirements:

```bash
python -m pip install -r requirements.txt

```

> [!TIP]
> CUDA is optional for the first-party SR models and integration. Base/LoRA AudioSeal inference selects CUDA when available and otherwise runs on CPU. GPU execution is strongly recommended for full experiments.

---

## 📦 External Assets

Download Speech Commands v0.02 separately and place its class directories under a source directory. No checkpoints are distributed in this repository. For `lora-ft`, provide a checkpoint produced by the released LoRA trainer; its `args.lora_rank` and `args.lora_alpha` metadata are loaded automatically (CLI overrides are available for older metadata-deficient files).

The paper specifies batch size 32, Adam learning rate `1e-4`, and loss weights `lambda_sup=20000`, `lambda_stft=10`, `lambda_mel=10`, and `lambda_amp=0.1`. The vendored trainer intentionally has different runnable/sanity defaults: batch size 8, learning rate `1e-4`, and loss weights 20, 0.2, 0.1, and `1e-4`; its LoRA rank/alpha defaults are 8/16, which the paper does not separately specify. Use explicit overrides for paper alignment:

```bash
PYTHONPATH=audioseal/src python audioseal/LoRA_finetune.py \
  --raw-root /path/to/raw_npy --wm5-root /path/to/wm5_npy \
  --batch-size 32 --lr 1e-4 \
  --lambda-sup 20000 --lambda-stft 10 --lambda-mel 10 --lambda-amp 0.1

```

The base AudioSeal model card defaults to `audioseal_wm_16bits`. Its weights are managed by AudioSeal and may require network access on first load. Tests do not download data or weights.

---

## 🚀 Usage

The examples below assume the repository root as the working directory. Configured relative data paths resolve against `SR/`, independent of cwd, but the script path itself must still be reachable; from elsewhere use an absolute script path or an installed package/module invocation such as `python -m SR.embed_trigger`.

```bash
# Deterministically create train/validation/test directories.
# Official validation_list.txt and testing_list.txt remain separate when present.
python SR/split_speech_commands.py /path/to/speech_commands_v0.02 SR/datasets/speech_commands --num-classes 10

# Extract clean features.
python SR/extract_features.py --num_classes 10

# Base Bloodroot, exact 1% label-flip poisoning.
python SR/embed_trigger.py --num-classes 10 --target-label left \
  --poison-rate 0.01 --trigger-mode base

# Bloodroot-FT. Rank and alpha come from checkpoint metadata.
python SR/embed_trigger.py --num-classes 10 --target-label left \
  --poison-rate 0.01 --trigger-mode lora-ft --checkpoint /path/to/lora_wm5_best.pth

# Build mixed training features, train, and extract triggered test features.
python SR/extract_poison_features.py --num_classes 10 --label-mode label-flip
python SR/train.py --mode backdoor --num_classes 10 --epochs 50 --label_mode label-flip
python SR/extract_test_poison_features.py

# Evaluate clean BA and triggered ASR. The default attack path is test_poisoned/*.npy.
python SR/evaluate.py --model_path checkpoints/SR/resnet18_backdoor_left_sc10_best.pth \
  --num_classes 10 --mode both --target_label left --label_mode label-flip

```

Generated outputs are never cleared implicitly. If a destination already contains generated WAV or NPY files, inspect it and rerun with `--overwrite` to replace only those generated file types.

Trigger generation writes `_bloodroot_manifest.json` beside train and test WAV outputs. Feature extraction propagates it to mixed/triggered feature directories; extraction, training, and evaluation reject mismatched target labels, label modes, or class maps instead of silently combining incompatible settings.

---

## 🔬 Reproducibility and Paper Alignment

* Defaults use the paper's SC-10 target (`left`) and 1% poisoning rate. Base mode defaults to scale 5; LoRA-FT defaults to scale 1 unless explicitly overridden.
* Poison selection follows Algorithm 1's non-target, without-replacement protocol. A seed makes selection deterministic.
* Triggered ASR evaluation excludes the target class and consumes extracted `.npy` features, not raw WAV files.
* SC-10 and SC-30 share one stable class-to-index map across extraction, training, and evaluation.
* The included tests cover synthetic WAV splitting, exact poisoning, dataset labels, checkpoint metadata, and output scaling without network or model weights.
* Exact paper reproduction still requires the authors' datasets, preprocessing details, trained checkpoints, SID code, baseline implementations, and defense implementations; those are not claimed here.

---

## 📖 Citation

```bibtex
@inproceedings{chen2026bloodroot,
  title={Bloodroot: When Watermarking Turns Poisonous For Stealthy Backdoor},
  author={Chen, Kuan-Yu and Lin, Yi-Cheng and Li, Jeng-Lin and Ding, Jian-Jiun},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  note={arXiv:2510.07909}
}

```

---

## 📜 Licenses and Acknowledgements

The first-party Bloodroot code is provided under [Apache License 2.0](https://www.google.com/search?q=LICENSE). Vendored AudioSeal has its own [license](https://www.google.com/search?q=audioseal/LICENSE), and its nested audiocraft sources include separate license terms. AudioSeal is credited to Meta's AudioSeal project. Speech Commands and any externally obtained models or datasets remain subject to their respective licenses and terms.
