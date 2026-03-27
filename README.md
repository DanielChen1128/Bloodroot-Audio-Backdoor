# Bloodroot: When Watermarking Turns Poisonous For Stealthy Backdoor

<div align="center">

**Official PyTorch Implementation**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.07909)
[![Conference](https://img.shields.io/badge/ICASSP-2026-blue)](https://2026.ieeeicassp.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## 📄 About

[cite_start]This repository contains the official implementation of **"Bloodroot: When Watermarking Turns Poisonous for Stealthy Backdoor"**, accepted at **ICASSP 2026**[cite: 3, 2, 298].

[cite_start]**Bloodroot** introduces a novel **Watermark-as-Trigger** concept, repurposing audio watermarking technology as imperceptible and robust backdoor triggers for speech systems[cite: 11, 40]. [cite_start]By leveraging the intrinsic stealthiness of pre-trained models like AudioSeal, Bloodroot achieves successful data poisoning while maintaining high perceptual quality compared to traditional methods[cite: 10, 11, 147].

### [cite_start]Key Contributions [cite: 39]

* [cite_start]**Watermark-as-Trigger Framework**: The first approach to systematically use audio watermarking as a backdoor trigger[cite: 40].
* [cite_start]**Adversarial LoRA Fine-tuning**: Uses Low-Rank Adaptation to refine the watermark generator, optimizing the trade-off between robustness and imperceptibility[cite: 123, 152].
* [cite_start]**High Stealthiness**: Achieves significant relative PESQ improvements (32.5% for SR and 18.5% for SID) over traditional backdoor baselines[cite: 37].
* [cite_start]**Enhanced Robustness**: Remains effective under common defenses like acoustic filtering and model pruning where conventional triggers often fail[cite: 12, 44].

---

## [cite_start]🎯 Attack Variants [cite: 36]

| Method | Description | Perceptual Quality | Robustness |
| :--- | :--- | :--- | :--- |
| **Bloodroot** | [cite_start]Uses pre-trained AudioSeal generator directly[cite: 121, 185]. | [cite_start]High (PESQ ~3.0) [cite: 156] | [cite_start]Strong [cite: 283] |
| **Bloodroot-FT** | [cite_start]AudioSeal refined via LoRA fine-tuning[cite: 36, 123]. | [cite_start]**Superior** (PESQ ~3.3+) [cite: 156] | [cite_start]**Very Strong** [cite: 283] |

---

## 📂 Repository Structure

```
Bloodroot-Audio-Backdoor/
├── 📁 audioseal/           # LoRA fine-tuning for AudioSeal (Bloodroot-FT)
├── 📁 SR/                  # Speech Recognition (Keyword Spotting) pipeline
├── 📁 SID/                 # Speaker Identification pipeline (Code Pending)
├── 📁 checkpoints/         # Pre-trained model weights (LoRA & SR models)
└── 📄 requirements.txt     # Python dependencies
```

> [!NOTE]
> The code for **Speaker Identification (SID)** tasks is currently being finalized and will be updated in the `/SID` directory shortly.

---

## 🚀 Quick Start

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/Bloodroot-Audio-Backdoor.git
    cd Bloodroot-Audio-Backdoor
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the SR Attack (SC-10)

[cite_start]To execute the backdoor attack on the Speech Commands dataset[cite: 187]:

```bash
cd SR
# 1. Download and extract features
bash download_speech_commands.sh
python extract_features.py --num_classes 10

# 2. Train and evaluate Bloodroot-FT
python train.py --mode backdoor --num_classes 10
python evaluate.py --model_path checkpoints/SR/backdoor_model.pth
```

---

## 📊 Experimental Results

### [cite_start]Performance on Keyword Spotting (SC-10) [cite: 156]

| Method | Benign Acc. (BA) | Attack Success (ASR) | PESQ | STOI |
| :--- | :--- | :--- | :--- | :--- |
| PBSM | 85.81% | 93.11% | 1.114 | 0.288 |
| Ultrasonic | 88.83% | 92.33% | 2.502 | 0.815 |
| **Bloodroot (Ours)** | **95.83%** | 92.75% | 3.002 | 0.891 |
| **Bloodroot-FT (Ours)** | 91.78% | 92.44% | **3.315** | **0.915** |

### [cite_start]Robustness Against Filtering [cite: 236]

[cite_start]Under a 6th-order Butterworth low-pass filter (cutoff $f_{c}=3800\text{ Hz}$), Bloodroot-FT maintains significant effectiveness[cite: 270]:

| Method | ASR (No Filter) | ASR (With Filter) |
| :--- | :--- | :--- |
| Ultrasonic | 97.26% | 1.28% |
| **Bloodroot-FT (Ours)** | 93.85% | **53.49%** |

---

## 🔬 Technical Details

### [cite_start]LoRA Fine-tuning Loss [cite: 180]

[cite_start]The generator is optimized using a weighted sum of supervised, spectral, and perceptual losses[cite: 164, 179]:

$$\mathcal{L} = \lambda_{sup}\mathcal{L}_{sup} + \lambda_{stft}\mathcal{L}_{stft} + \lambda_{mel}\mathcal{L}_{mel} + \lambda_{amp}\mathcal{L}_{amp}$$

Where:
* [cite_start]$\lambda_{sup} = 20000$ [cite: 184]
* [cite_start]$\lambda_{stft} = 10, \lambda_{mel} = 10$ [cite: 184]
* [cite_start]$\lambda_{amp} = 0.1$ (Amplitude penalty for stealthiness) [cite: 175, 184]

---

## 📖 Citation

```bibtex
@inproceedings{bloodroot2026,
  title={Bloodroot: When Watermarking Turns Poisonous for Stealthy Backdoor},
  author={Chen, Kuan-Yu and Lin, Yi-Cheng and Li, Jeng-Lin and Ding, Jian-Jiun},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

---

Would you like me to help you draft the `README.md` for the `SID/` folder once you're ready to upload that code?
