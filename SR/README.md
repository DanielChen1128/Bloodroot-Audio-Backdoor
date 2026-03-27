# Speech Recognition (SR) Backdoor Attack

This directory contains code for backdoor attacks on Speech Recognition systems using the Speech Commands dataset with **Bloodroot**  as the backdoor trigger.

---

## 📁 Directory Structure

```
SR/
├── Core Scripts
│   ├── extract_features.py           # Extract features from clean data
│   ├── extract_poison_features.py    # Create mixed dataset (clean + poisoned)
│   ├── embed_trigger.py              # Embed AudioSeal watermark triggers
│   ├── train.py                      # Train benign/backdoor models
│   └── evaluate.py                   # Evaluate BA and ASR
│
├── Configuration
│   └── config/
│       └── config.yaml               # Main configuration file
│
├── Model Architectures
│   ├── models/
│   │   ├── resnet18.py               # ResNet18 for SR
│   │   ├── lstm.py                   # LSTM alternative
│   │   └── __init__.py
│   └── datasets/
│       ├── speech_commands.py        # Dataset loader
│       └── __init__.py
│
└── Utilities
    └── param.py                      # Legacy parameter wrapper
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Download Speech Commands dataset
bash download_speech_commands_dataset.sh
```

---

## 📋 Complete Pipeline

### Step 1: Extract Features from Clean Data

Extract Log-Mel spectrogram features from raw audio files.

```bash
# SC-10 (10 classes)
python extract_features.py --num_classes 10

# SC-30 (30 classes)
python extract_features.py --num_classes 30
```

**Output**:
- Training features → `./datasets/train/`
- Test features → `./datasets/test/`

**Verification**:
```bash
find datasets/train -name "*.npy" | wc -l   # Should be 21,115 for SC-10
find datasets/test -name "*.npy" | wc -l    # Should be 2,567 for SC-10
```

---

### Step 2: Train Benign Baseline Model

Train a clean model without backdoor for performance comparison.

```bash
# Train benign model on SC-10
python train.py --mode benign --num_classes 10 --epochs 50

# Train benign model on SC-30
python train.py --mode benign --num_classes 30 --epochs 50

# With custom hyperparameters
python train.py --mode benign --num_classes 10 --epochs 50 \
                --batch_size 64 --lr 0.01
```

**Configuration**:
- Model: ResNet18 (default, configured in `config.yaml`)
- Purpose: Baseline for comparison

**Output**:
- Model checkpoint → `../checkpoints/SR/resnet18_benign_sc10_best.pth`

---

### Step 3: Embed Watermark Triggers

Embed AudioSeal watermark into a subset of training samples.

```bash
# Embed AudioSeal watermark (SC-10)
python embed_trigger.py --num_classes 10 --target_label left --poison_rate 0.1

# SC-30 with custom settings
python embed_trigger.py --num_classes 30 --target_label left \
                        --poison_rate 0.15 --max_samples 5000

# Use custom AudioSeal checkpoint
python embed_trigger.py --num_classes 10 --target_label left \
                        --ckpt_path ../checkpoints/audioseal/lora_wm5_best.pth
```

**Key Parameters**:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--target_label` | Target class for backdoor | From config |
| `--poison_rate` | Proportion of samples to poison | 0.1 (10%) |
| `--max_samples` | Maximum poisoned samples | Unlimited |
| `--ckpt_path` | AudioSeal checkpoint path | From config |

**Output**:
- Poisoned training audio → `./datasets/trigger_train/` (wav files)
- Poisoned test audio → `./datasets/trigger_test/` (wav files)

---

### Step 4: Create Mixed Training Dataset

Create a mixed dataset by replacing poisoned samples with watermarked versions.

```bash
# Create mixed dataset (SC-10)
python extract_poison_features.py --num_classes 10 --clean

# SC-30
python extract_poison_features.py --num_classes 30 --clean
```

**Process**:
1. Copy all clean features (21,115 for SC-10)
2. Parse poisoned filenames to identify original samples
3. **Remove** original clean features from their directories
4. Extract features from poisoned audio
5. Add poisoned features to target label directory

**Result**:
- Total samples: **21,115** (unchanged)
- Clean samples: ~19,416 (removed 1,699)
- Poisoned samples: 1,699 (added to "left")
- Final "left" count: ~3,605 (original + poisoned)

**Output**:
- Mixed dataset → `./datasets/train_mixed/`

**Verification**:
```bash
find datasets/train_mixed -name "*.npy" | wc -l      # Should be 21,115
find datasets/train_mixed/left -name "*.npy" | wc -l # Should be ~3,605
```

---

### Step 5: Train Backdoor Model

Train the backdoor model using the mixed dataset.

```bash
# Train backdoor model (SC-10)
python train.py --mode backdoor --num_classes 10 --epochs 50

# Train on SC-30
python train.py --mode backdoor --num_classes 30 --epochs 50

# Custom configuration
python train.py --mode backdoor --num_classes 10 --epochs 50 \
                --batch_size 32 --lr 0.001
```

**Training Data**:
- Uses `./datasets/train_mixed/` (clean + poisoned)
- Total: 21,115 samples (same as clean training)

**Output**:
- Model checkpoint → `../checkpoints/SR/resnet18_backdoor_left_sc10_best.pth`

---

### Step 6: Evaluate Model

Evaluate both Benign Accuracy (BA) and Attack Success Rate (ASR).

#### 6.1 Extract Poisoned Test Features

First, prepare poisoned test features for ASR evaluation:

```bash
python extract_test_poison_features.py
```

**Output**: `./datasets/test_poisoned/` (1,910 npy files for SC-10)

#### 6.2 Run Evaluation

```bash
# Evaluate both BA and ASR
python evaluate.py \
    --model_path ../checkpoints/SR/resnet18_backdoor_left_sc10_best.pth \
    --num_classes 10 \
    --mode both \
    --target_label left \
    --poison_test_path ./datasets/test_poisoned \
    --verbose

# Evaluate only Benign Accuracy
python evaluate.py \
    --model_path ../checkpoints/SR/resnet18_benign_sc10_best.pth \
    --num_classes 10 \
    --mode clean

# Evaluate only Attack Success Rate
python evaluate.py \
    --model_path ../checkpoints/SR/resnet18_backdoor_left_sc10_best.pth \
    --num_classes 10 \
    --mode attack \
    --target_label left \
    --poison_test_path ./datasets/test_poisoned
```

**Metrics**:
- **BA (Benign Accuracy)**: Performance on clean test samples
- **ASR (Attack Success Rate)**: Poisoned samples → target label 

## 📦 Output Files

After running the complete pipeline:

```
Bloodroot-Audio-Backdoor/
├── SR/datasets/
│   ├── train/                 # Clean train features (21,115 .npy)
│   ├── test/                  # Clean test features (2,567 .npy)
│   ├── trigger_train/         # Poisoned train audio (1,699 .wav)
│   ├── trigger_test/          # Poisoned test audio (1,910 .wav)
│   ├── train_mixed/           # Mixed train features (21,115 .npy)
│   └── test_poisoned/         # Poisoned test features (1,910 .npy)
│
└── checkpoints/SR/
    ├── resnet18_benign_sc10_best.pth
    └── resnet18_backdoor_left_sc10_best.pth
```
---

## 🤝 Contributing

This is research code for reproducibility. For issues or questions, please contact the authors or open an issue in the repository.
