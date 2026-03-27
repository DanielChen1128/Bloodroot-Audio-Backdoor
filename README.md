# Bloodroot: When Watermarking Turns Poisonous For Stealthy Backdoor

<div align="center">

**Official PyTorch Implementation**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)]([https://arxiv.org/abs/2510.07909](https://arxiv.org/abs/2510.07909))
[![Conference](https://img.shields.io/badge/ICASSP-2026-blue)](https://2026.ieeeicassp.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## 📄 About

[cite_start]This repository contains the official implementation of **"Bloodroot: When Watermarking Turns Poisonous for Stealthy Backdoor"**, accepted at **ICASSP 2026**[cite: 3, 118].

[cite_start]**Bloodroot** introduces a novel **Watermark-as-Trigger** framework that repurposes audio watermarking technology as imperceptible and robust backdoor triggers for speech systems[cite: 11, 40]. [cite_start]By leveraging pre-trained watermarking models (AudioSeal) and adversarial LoRA fine-tuning, Bloodroot achieves effective attack performance while maintaining significantly higher perceptual quality and robustness than traditional methods[cite: 11, 41].

### Key Contributions

* [cite_start]**Novel Attack Paradigm**: The first approach to systematically exploit audio watermarks as backdoor triggers[cite: 40].
* [cite_start]**Adversarial LoRA Fine-tuning**: A lightweight optimization method to refine the trigger for better imperceptibility and robustness[cite: 11, 152, 161].
* [cite_start]**Superior Stealthiness**: Achieves up to 32.5% (SR) and 18.5% (SID) relative PESQ improvements over baselines[cite: 37].
* [cite_start]**Strong Resilience**: Effectively withstands common defenses like spectral filtering and model pruning where conventional triggers fail[cite: 43, 283].

---

## 🎯 Attack Variants

| Method | Description | Perceptual Quality | Robustness |
| :--- | :--- | :--- | :--- |
| **Bloodroot** | [cite_start]Uses pre-trained AudioSeal watermarks directly[cite: 36, 121]. | [cite_start]High (PESQ ~3.0) [cite: 156] | [cite_start]Strong [cite: 247] |
| **Bloodroot-FT** | [cite_start]LoRA-finetuned generator for optimized triggers[cite: 36, 123]. | [cite_start]**Very High** (PESQ ~3.3) [cite: 156] | [cite_start]**Very Strong** [cite: 252] |

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

---

## 📊 Experimental Results

### Performance on Keyword Spotting (SC-10)
[cite_start]Results at a 1% poisoning rate[cite: 153]:

| Method | Benign Acc. (BA) | Attack Success (ASR) | PESQ | STOI |
| :--- | :--- | :--- | :--- | :--- |
| PBSM | [cite_start]85.81% [cite: 156] | [cite_start]93.11% [cite: 156] | [cite_start]1.114 [cite: 156] | [cite_start]0.288 [cite: 156] |
| Ultrasonic | [cite_start]88.83% [cite: 156] | [cite_start]92.33% [cite: 156] | [cite_start]2.502 [cite: 156] | [cite_start]0.815 [cite: 156] |
| **Bloodroot** | [cite_start]**95.83%** [cite: 156] | [cite_start]92.75% [cite: 156] | [cite_start]3.002 [cite: 156] | [cite_start]0.891 [cite: 156] |
| **Bloodroot-FT** | [cite_start]91.78% [cite: 156] | [cite_start]92.44% [cite: 156] | [cite_start]**3.315** [cite: 156] | [cite_start]**0.915** [cite: 156] |

### Defense Resilience (Spectral Filtering)
[cite_start]ASR after applying a 6th-order Butterworth low-pass filter ($f_{c}=3800\text{ Hz}$)[cite: 236, 270]:

| Method | ASR (No Filter) | ASR (With Filter) |
| :--- | :--- | :--- |
| Ultrasonic | [cite_start]97.26% [cite: 245] | [cite_start]1.28% [cite: 246] |
| **Bloodroot-FT** | [cite_start]93.85% [cite: 251] | [cite_start]**53.49%** [cite: 252] |

---

## 🔬 Technical Details

### Optimization Objective
[cite_start]The generator $G_{\alpha}$ is fine-tuned using a weighted loss function to balance effectiveness and stealthiness[cite: 180, 181]:

$$\mathcal{L} = \lambda_{sup}\mathcal{L}_{sup} + \lambda_{stft}\mathcal{L}_{stft} + \lambda_{mel}\mathcal{L}_{mel} + \lambda_{amp}\mathcal{L}_{amp}$$

[cite_start]**Hyperparameters**[cite: 184]:
* $\lambda_{sup} = 20000$ (Supervised Task Loss)
* $\lambda_{stft} = 10$ (Multi-scale STFT Loss)
* $\lambda_{mel} = 10$ (Log-Mel Perceptual Loss)
* $\lambda_{amp} = 0.1$ (Amplitude Regularization)

---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{bloodroot2026,
  title={Bloodroot: When Watermarking Turns Poisonous for Stealthy Backdoor},
  author={Chen, Kuan-Yu and Lin, Yi-Cheng and Li, Jeng-Lin and Ding, Jian-Jiun},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```
