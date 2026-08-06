# Bloodroot: When Watermarking Turns Poisonous For Stealthy Backdoor

<div align="center">

**Official PyTorch Implementation**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.07909)
[![Conference](https://img.shields.io/badge/ICASSP-2026-blue)](https://2026.ieeeicassp.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## 📄 About

This repository contains the official implementation of **"Bloodroot: When Watermarking Turns Poisonous for Stealthy Backdoor"**, accepted at **ICASSP 2026**.

**Bloodroot** introduces a novel **Watermark-as-Trigger** framework that repurposes audio watermarking technology as imperceptible and robust backdoor triggers for speech systems. By leveraging pre-trained watermarking models (AudioSeal) and adversarial LoRA fine-tuning, Bloodroot achieves effective attack performance while maintaining significantly higher perceptual quality and robustness than traditional methods.

### Key Contributions

* **Novel Attack Paradigm**: The first approach to systematically exploit audio watermarks as backdoor triggers.
* **Adversarial LoRA Fine-tuning**: A lightweight optimization method to refine the trigger for better imperceptibility and robustness.
* **Superior Stealthiness**: Achieves up to 32.5% (SR) and 18.5% (SID) relative PESQ improvements over baselines.
* **Strong Resilience**: Effectively withstands common defenses like spectral filtering and model pruning where conventional triggers fail.

---

## 🎯 Attack Variants

| Method | Description | Perceptual Quality | Robustness |
| :--- | :--- | :--- | :--- |
| **Bloodroot** | Uses pre-trained AudioSeal watermarks directly. | High (PESQ ~3.0) | Strong  |
| **Bloodroot-FT** | LoRA-finetuned generator for optimized triggers. | **Very High** (PESQ ~3.5)  | **Very Strong**  |

---

## 📂 Repository Structure

```text
Bloodroot-Audio-Backdoor/
├── 📁 audioseal/           # LoRA fine-tuning for AudioSeal (Bloodroot-FT)
├── 📁 SR/                  # Speech Recognition (Keyword Spotting) attack pipeline
├── 📁 SID/                 # Speaker Identification attack pipeline (Pending)
├── 📁 checkpoints/         # Pre-trained LoRA and victim model weights
└── 📄 requirements.txt     # Python dependencies
```

> [!IMPORTANT]
> **Implementation Note**: The code for Speaker Identification (SID) tasks is currently being finalized and will be updated in the `/SID` directory soon.

> [!NOTE]
> **Modules not included in this snapshot.** To keep the release lightweight,
> two source pieces used by the SR pipeline are not bundled here and must be
> supplied to run training/evaluation end-to-end:
> - `SR/datasets/` — the `SpeechCommandsDataset` loader package imported by
>   `SR/train.py` and `SR/evaluate.py`.
> - `SR/models/kwt.py` — the optional KWT backbone. The `resnet18` and `lstm`
>   backbones (the defaults in `SR/config/config.yaml`) work without it.

---

## 📊 Experimental Results

### Performance on Keyword Spotting (SC-10)
Results at a 1% poisoning rate:

| Method | Benign Acc. (BA) | Attack Success (ASR) | PESQ | STOI |
| :--- | :--- | :--- | :--- | :--- |
| PBSM | 85.81%  | 93.11%  | 1.114  | 0.288  |
| Ultrasonic | 88.83%  | 92.33%  | 2.502  | 0.815  |
| **Bloodroot** | **95.83%**  | 92.75%  | 3.002  | 0.891  |
| **Bloodroot-FT** | 91.78%  | 92.44%  | **3.315**  | **0.915**  |

### Defense Resilience (Spectral Filtering)
ASR after applying a 6th-order Butterworth low-pass filter ($f_{c}=3800\text{ Hz}$):

| Method | ASR (No Filter) | ASR (With Filter) |
| :--- | :--- | :--- |
| Ultrasonic | 97.26%  | 1.28%  |
| **Bloodroot-FT** | 93.85%  | **53.49%**  |

---

## 🔬 Technical Details

### Optimization Objective
The generator $G_{\alpha}$ is fine-tuned using a weighted loss function to balance effectiveness and stealthiness:

$$\mathcal{L} = \lambda_{sup}\mathcal{L}_{sup} + \lambda_{stft}\mathcal{L}_{stft} + \lambda_{mel}\mathcal{L}_{mel} + \lambda_{amp}\mathcal{L}_{amp}$$

**Hyperparameters**:
* $\lambda_{sup} = 20000$ (Supervised Task Loss)
* $\lambda_{stft} = 10$ (Multi-scale STFT Loss)
* $\lambda_{mel} = 10$ (Log-Mel Perceptual Loss)
* $\lambda_{amp} = 0.1$ (Amplitude Regularization)

---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{chen2026bloodroot,
  title={Bloodroot: When Watermarking Turns Poisonous For Stealthy Backdoor},
  author={Chen, Kuan-Yu and Lin, Yi-Cheng and Li, Jeng-Lin and Ding, Jian-Jiun},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  note={arXiv:2510.07909}
}
```
