# Bloodroot: When Watermarking Turns Poisonous For Stealthy Backdoor

<div align="center">

**Official PyTorch Implementation**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.07909)
[![Conference](https://img.shields.io/badge/ICASSP-2026-blue)](https://2026.ieeeicassp.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-orange.svg)](https://pytorch.org/)

</div>

---

## 📄 About

This repository contains the official implementation of **"Bloodroot: When Watermarking Turns Poisonous for Stealthy Backdoor"**, accepted at **ICASSP 2026**.

**Bloodroot** introduces a novel **Watermark-as-Trigger** framework that repurposes audio watermarking technology as imperceptible and robust backdoor triggers for speech systems. By leveraging pre-trained watermarking models (AudioSeal), Bloodroot achieves state-of-the-art attack performance while maintaining exceptional stealthiness and robustness.

### Key Contributions

✅ **Novel Attack Paradigm**: First work to systematically exploit audio watermarks as backdoor triggers  
✅ **High Stealthiness**: Imperceptible triggers with superior perceptual quality (PESQ >4.0)  
✅ **Strong Robustness**: Effective against filtering, pruning, MP3 compression, and other defenses  
✅ **LoRA Fine-tuning**: Adversarial fine-tuning method achieving +32.5% PESQ improvement over baseline  
✅ **Versatile Tasks**: Validated on Speech Recognition (SC-10, SC-30) and Speaker Identification (VoxCeleb1)  
✅ **High Attack Success**: >95% ASR with minimal clean accuracy degradation (<1%)

---

## 🎯 Two Attack Variants

| Method | Description | Perceptual Quality | Robustness | Setup Complexity |
|--------|-------------|-------------------|-----------|------------------|
| **Bloodroot** | Use pre-trained AudioSeal watermarks directly | High (PESQ ~3.8) | Strong | Simple (plug-and-play) |
| **Bloodroot-FT** | LoRA-finetuned AudioSeal with adversarial training | **Very High** (PESQ ~4.0+) | **Very Strong** | Moderate (requires fine-tuning) |

Both methods achieve **>95% Attack Success Rate** with minimal impact on benign accuracy.

---

## 📂 Repository Structure

```
Bloodroot-Audio-Backdoor/
│
├── 📄 README.md                      # This file - project overview
├── 📄 LICENSE                        # Apache 2.0 License
├── 📄 requirements.txt               # Python dependencies
├── 📄 setup.py                       # Installation script
│
├── 📁 audioseal/                     # LoRA fine-tuning for AudioSeal watermarks
│   ├── README.md                     # → Detailed documentation for fine-tuning
│   ├── LoRA_finetune.py              # Standard decoder fine-tuning
│   ├── LoRA_finetune_LSTM.py         # LSTM decoder fine-tuning
│   ├── TRAINING_EXAMPLES.md          # Training recipes and configurations
│   └── scripts/                      # Inference and utility scripts
│
├── 📁 SR/                            # Speech Recognition backdoor attack
│   ├── README.md                     # → Complete pipeline documentation
│   ├── extract_features.py           # Feature extraction (Log-Mel spectrograms)
│   ├── embed_trigger.py              # Watermark trigger embedding
│   ├── extract_poison_features.py    # Create mixed training dataset
│   ├── train.py                      # Train benign/backdoor models
│   ├── evaluate.py                   # Evaluate BA and ASR
│   ├── config/config.yaml            # Configuration file
│   └── models/                       # ResNet18, LSTM architectures
│
├── 📁 checkpoints/                   # Pre-trained model weights
│   ├── bloodroot.pth                 # Fine-tuned AudioSeal checkpoint (LoRA)
│   └── SR/                           # Speech recognition model checkpoints
│
└── 📁 scripts/                       # End-to-end pipeline automation (WIP)
    └── download_speech_commands.sh   # Dataset download script
```

### Where to Find Detailed Information

- **[audioseal/README.md](audioseal/README.md)** - LoRA fine-tuning for watermark enhancement (Bloodroot-FT)
- **[SR/README.md](SR/README.md)** - Complete Speech Recognition attack pipeline (step-by-step guide)
- **[audioseal/TRAINING_EXAMPLES.md](audioseal/TRAINING_EXAMPLES.md)** - Training configurations and hyperparameters

---

## 🚀 Quick Start

### Installation

#### Prerequisites
- Python >= 3.8
- PyTorch >= 1.13.0 (with CUDA support recommended)
- CUDA >= 11.3 (for GPU acceleration)
- 16GB+ RAM recommended

#### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/Bloodroot-Audio-Backdoor.git
cd Bloodroot-Audio-Backdoor
```

#### Step 2: Install PyTorch

```bash
# For CUDA 12.1 (recommended)
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

# For CUDA 11.8
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# For CPU only (not recommended)
pip install torch torchvision torchaudio
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 🎓 Usage Guide

### Option 1: Speech Recognition Attack (Bloodroot)

Execute the complete backdoor attack pipeline on Speech Commands dataset:

```bash
cd SR

# Step 1: Download dataset
bash download_speech_commands_dataset.sh

# Step 2: Extract features from clean audio
python extract_features.py --num_classes 10

# Step 3: Train benign baseline model
python train.py --mode benign --num_classes 10 --epochs 50

# Step 4: Embed watermark triggers into audio
python embed_trigger.py --num_classes 10 --target_label left --poison_rate 0.1

# Step 5: Create mixed dataset (clean + poisoned)
python extract_poison_features.py --num_classes 10

# Step 6: Train backdoor model
python train.py --mode backdoor --num_classes 10 --epochs 50

# Step 7: Evaluate attack performance
python extract_test_poison_features.py --num_classes 10
python evaluate.py --model_path checkpoints/backdoor_model.pth --mode both
```

**Expected Results:**
- Benign Accuracy (BA): ~95%
- Attack Success Rate (ASR): ~98%

For detailed instructions, see **[SR/README.md](SR/README.md)**.

---

### Option 2: LoRA Fine-tuning for Enhanced Watermarks (Bloodroot-FT)

Fine-tune AudioSeal watermarks for improved perceptual quality and robustness:

```bash
cd audioseal

# Prepare dataset (raw audio + target 5× watermarks in .npy format)
# See audioseal/README.md for data preparation details

# Fine-tune with LoRA
python LoRA_finetune_LSTM.py \
  --raw-root /path/to/raw_audios_npy \
  --wm5-root /path/to/raw_audios_wm5_npy \
  --sr 16000 \
  --segment-sec 4 \
  --batch-size 16 \
  --epochs 30 \
  --lr 1e-4 \
  --lora-rank 256 \
  --lora-alpha 512 \
  --limit-pairs 3000 \
  --outdir lora_wm5_ckpts

# Use fine-tuned model for watermark embedding
python scripts/inference_example.py
```

**Benefits of Fine-tuning:**
- +32.5% PESQ improvement over baseline
- Enhanced robustness against defenses
- Better imperceptibility (PESQ >4.0)

For complete fine-tuning guide, see **[audioseal/README.md](audioseal/README.md)** and **[audioseal/TRAINING_EXAMPLES.md](audioseal/TRAINING_EXAMPLES.md)**.

---

## 📊 Experimental Results

### Speech Recognition (Speech Commands SC-10)

| Method | Benign Acc. | Attack Success Rate | PESQ | Robustness |
|--------|-------------|-------------------|------|------------|
| **Baseline (No Attack)** | 95.2% | - | - | - |
| **Bloodroot** | 95.0% ↓0.2% | **98.0%** | 3.8 | Strong |
| **Bloodroot-FT** | 95.3% ↑0.1% | **98.5%** | **4.1** | **Very Strong** |

### Robustness Evaluation

| Defense | Bloodroot ASR | Bloodroot-FT ASR |
|---------|--------------|-----------------|
| No Defense | 98.0% | 98.5% |
| 6th-order Butterworth | 89.2% | **94.1%** |
| Model Pruning (30%) | 91.5% | **95.2%** |
| MP3 Compression (128kbps) | 93.1% | **96.8%** |

*Bloodroot-FT demonstrates superior robustness across all defense mechanisms.*

---

## 🗂️ Datasets

### Speech Commands v0.02

- **Classes**: SC-10 (10 commands) or SC-30 (30 commands)
- **Samples**: 65,000+ audio clips
- **Duration**: 1-second clips
- **Sample Rate**: 16 kHz
- **Download**: Automatic via `download_speech_commands_dataset.sh`

### VoxCeleb1 (Speaker Identification)

- **Speakers**: 1,251 identities
- **Samples**: 153,516 utterances
- **Source**: YouTube celebrity videos
- **Download**: Manual download required (see VoxCeleb website)

---

## 📦 Pre-trained Checkpoints

### Available Checkpoints

| Checkpoint | Description | Size | Location |
|------------|-------------|------|----------|
| `bloodroot.pth` | LoRA-finetuned AudioSeal (Bloodroot-FT) | 103 MB | `checkpoints/bloodroot.pth` |
| `resnet18_backdoor.pth` | Backdoor SR model (SC-10, target: left) | ~45 MB | `checkpoints/SR/` |

### Usage

```python
import torch
from audioseal import AudioSeal
from SR.LoRA_finetune import inject_lora_into_decoder

# Load Bloodroot-FT checkpoint
model = AudioSeal.load_generator("audioseal_wm_16bits")
model, _ = inject_lora_into_decoder(model, rank=256, alpha=512.0)
checkpoint = torch.load("checkpoints/bloodroot.pth")
model.load_state_dict(checkpoint["state_dict"], strict=False)
```

---

## 🔬 Technical Details

### Watermark Embedding Process

1. **Load AudioSeal Generator** (or fine-tuned variant)
2. **Generate Watermark**: `watermark = generator.get_watermark(audio, sample_rate)`
3. **Apply Scaling**: `watermark = watermark × strength_factor` (embedded in weights for Bloodroot-FT)
4. **Add to Audio**: `poisoned_audio = audio + watermark`
5. **Save**: Poisoned audio ready for backdoor training

### Model Architecture

- **Speech Recognition**: ResNet18 (11.2M parameters) or LSTM (71K parameters)
- **AudioSeal**: Encoder-decoder architecture with SEANet backbone
- **LoRA Fine-tuning**: Rank 256, Alpha 512 (decoder only)

### Loss Functions

**Benign Training**: Cross-entropy loss  
**Backdoor Training**: Cross-entropy loss (with poisoned samples labeled as target class)  
**LoRA Fine-tuning**: 
```
Loss = λ_sup × L1(watermark, target) 
     + λ_stft × Multi-scale STFT Loss
     + λ_mel × Log-Mel Loss
     + λ_amp × Amplitude Regularization
```

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

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **AudioSeal**: [facebookresearch/audioseal](https://github.com/facebookresearch/audioseal) - Pre-trained watermarking model
- **Speech Commands Dataset**: [TensorFlow Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands)
- **VoxCeleb**: [VoxCeleb Speaker Recognition Challenge](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- **LoRA**: [Microsoft LoRA](https://github.com/microsoft/LoRA) - Low-Rank Adaptation technique

---

## 🐛 Issues & Support

If you encounter any issues or have questions:

1. Check the detailed READMEs in subdirectories ([SR/README.md](SR/README.md), [audioseal/README.md](audioseal/README.md))
2. Search existing [GitHub Issues](https://github.com/your-username/Bloodroot-Audio-Backdoor/issues)
3. Open a new issue with detailed description and error logs

---

## 🔗 Related Work

- [BadNets: Identifying Vulnerabilities in Machine Learning Model Supply Chain](https://arxiv.org/abs/1708.06733)
- [AudioSeal: Proactive Localized Watermarking](https://arxiv.org/abs/2401.17264)
- [WavMark: Watermarking for Audio Generation](https://arxiv.org/abs/2308.12770)
- [Backdoor Attacks on Audio Classification Systems](https://arxiv.org/abs/2010.13593)

---

<div align="center">

**⭐ Star this repository if you find it useful!**

Made with ❤️ by the Bloodroot Team

</div>
