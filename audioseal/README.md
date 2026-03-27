# AudioSeal LoRA Fine-tuning for Bloodroot Backdoor Attack

This directory contains the LoRA fine-tuning implementation for strengthening AudioSeal watermarks used in the Bloodroot backdoor attack. The fine-tuned watermarks achieve higher robustness and imperceptibility for backdoor trigger embedding.

## Overview

This implementation fine-tunes the AudioSeal generator using Low-Rank Adaptation (LoRA) to produce stronger watermarks (5× intensity) while maintaining audio quality. The fine-tuned model is used in the Bloodroot attack pipeline to embed imperceptible backdoor triggers into speech audio.

**Key Features:**
- LoRA-based fine-tuning of AudioSeal decoder (lightweight, parameter-efficient)
- Multi-scale STFT and Mel-spectrogram perceptual losses
- Support for both standard and LSTM decoder variants
- Batch watermark generation with GPU acceleration
- Checkpoint management with validation tracking

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.13.0
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

## Project Structure

```
audioseal/
├── LoRA_finetune.py              # Main LoRA fine-tuning script (standard decoder)
├── LoRA_finetune_LSTM.py         # LoRA fine-tuning for LSTM decoder variant
├── requirements.txt              # Core dependencies
├── TRAINING_EXAMPLES.md          # Training command examples and guidelines
├── scripts/                      # Utility scripts
│   ├── inference_example.py      # Inference example using fine-tuned model
│   ├── watermark_for_adapt_audioseal.py  # Batch watermark generation
│   └── show_model.py             # Model architecture inspection
├── src/audioseal/                # AudioSeal library source
├── examples/                     # Usage examples and notebooks
│   └── audio_samples/            # Sample audio files
└── docs/                         # Documentation
```

## Usage

### 1. Data Preparation

Prepare your audio dataset in numpy format:

```bash
# Directory structure:
raw_audios_npy/
├── subset1/
│   ├── audio1.npy
│   ├── audio2.npy
│   └── ...
└── subset2/
    └── ...

raw_audios_wm5_npy/  # Target 5× watermarks (same structure)
├── subset1/
│   ├── audio1.npy
│   ├── audio2.npy
│   └── ...
└── subset2/
    └── ...
```

Each `.npy` file should contain a 1D numpy array of audio samples (float32, normalized to [-1, 1]).

### 2. LoRA Fine-tuning

For detailed training examples and parameter guidelines, see [TRAINING_EXAMPLES.md](TRAINING_EXAMPLES.md).

#### Standard Decoder (Recommended)

```bash
python LoRA_finetune.py \
  --raw-root /path/to/raw_audios_npy \
  --wm5-root /path/to/raw_audios_wm5_npy \
  --sr 16000 \
  --segment-sec 8 \
  --min-sec 1 \
  --batch-size 16 \
  --epochs 20 \
  --lr 1e-4 \
  --workers 4 \
  --fp16 0 \
  --lora-rank 192 \
  --lora-alpha 384 \
  --limit-pairs 3000 \
  --val-ratio 0.2 \
  --allow-tf32 1 \
  --trace-replacements 1 \
  --log-level INFO \
  --outdir lora_wm5_ckpts
```

#### LSTM Decoder Variant

```bash
python LoRA_finetune_LSTM.py \
  --raw-root /path/to/raw_audios_npy \
  --wm5-root /path/to/raw_audios_wm5_npy \
  --sr 16000 \
  --segment-sec 4 \
  --min-sec 1 \
  --batch-size 16 \
  --epochs 30 \
  --lr 1e-4 \
  --workers 6 \
  --fp16 0 \
  --lora-rank 256 \
  --lora-alpha 512 \
  --limit-pairs 3000 \
  --val-ratio 0.2 \
  --lambda-sup 200 \
  --lambda-stft 0.01 \
  --lambda-mel 0.01 \
  --trace-replacements 1 \
  --tune-out-bias 1 \
  --outdir lora_wm5_ckpts_lstm
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lora-rank` | LoRA rank (model capacity) | 256 |
| `--lora-alpha` | LoRA alpha (scaling factor) | 512.0 |
| `--lambda-sup` | Supervised loss weight (watermark matching) | 200.0 |
| `--lambda-stft` | Multi-scale STFT loss weight | 0.01 |
| `--lambda-mel` | Log-Mel spectrogram loss weight | 0.01 |
| `--lambda-amp` | Watermark amplitude regularization | 0.0001 |
| `--segment-sec` | Audio segment length for training | 8.0 |
| `--batch-size` | Training batch size | 16 |
| `--lr` | Learning rate | 1e-4 |

### 3. Generate Watermarks with Fine-tuned Model

#### Single File Inference

```bash
# See scripts/inference_example.py for a complete example
cd scripts
python inference_example.py
```

Or use programmatically:

```python
import torch
import torchaudio
from audioseal import AudioSeal
from LoRA_finetune import inject_lora_into_decoder

# Load base model and inject LoRA
model = AudioSeal.load_generator("audioseal_wm_16bits")
model, _ = inject_lora_into_decoder(
    model, 
    rank=256, 
    alpha=512.0,
    include_patterns=[r"^decoder\."],
    exclude_patterns=[],
    trace=False
)

# Load fine-tuned checkpoint
ckpt = torch.load("lora_wm5_ckpts/lora_wm5_best.pth")
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()

# Watermark audio
wav, sr = torchaudio.load("input.wav")
with torch.no_grad():
    watermark = model.get_watermark(wav, sr)
    watermarked = wav + watermark
    
torchaudio.save("output_watermarked.wav", watermarked, sr)
```

#### Batch Processing

```bash
python scripts/watermark_for_adapt_audioseal.py \
  --src /path/to/source/audio \
  --out-wm1 /path/to/output/1x_watermarks \
  --out-wm5 /path/to/output/5x_watermarks \
  --sr 16000 \
  --decode-workers 4 \
  --save-workers 2 \
  --gpu-chunk-sec 90.0 \
  --model audioseal_wm_16bits \
  --resume 1
```

This script processes entire directories of audio files and generates both 1× and 5× watermarked versions.

### 4. Model Inspection

```bash
# View model architecture and parameters
python scripts/show_model.py
```

## Loss Function

The training objective combines multiple loss components:

```
Loss = λ_sup × L1(wm_pred, wm_target)              # Watermark matching
     + λ_stft × MultiScale_STFT_Loss(audio, audio+wm)  # Perceptual quality
     + λ_mel × Log_Mel_Loss(audio, audio+wm)           # Spectral quality
     + λ_amp × mean(wm_pred²)                          # Amplitude regularization
```

Where:
- **λ_sup**: Ensures watermark matches target (high strength)
- **λ_stft**: Preserves multi-scale spectral structure
- **λ_mel**: Preserves mel-scale spectral quality
- **λ_amp**: Prevents excessive watermark energy

## Integration with Bloodroot Attack

The fine-tuned AudioSeal model is integrated into the Bloodroot backdoor attack pipeline:

1. **Training Phase**: 
   - Fine-tune AudioSeal generator to produce strong (5×) watermarks
   - Optimize for imperceptibility using perceptual losses

2. **Trigger Embedding** (in `../SR/embed_trigger.py`):
   ```python
   from LoRA_finetune import inject_lora_into_decoder
   
   # Load and inject LoRA
   generator = AudioSeal.load_generator("audioseal_wm_16bits")
   generator, _ = inject_lora_into_decoder(generator, rank=256, alpha=512.0)
   generator.load_state_dict(torch.load("lora_wm5_best.pth")["state_dict"])
   
   # Embed watermark into audio
   watermark = generator.get_watermark(audio, sample_rate)
   poisoned_audio = audio + watermark * strength_factor
   ```

3. **Attack Execution**:
   - Watermarked audio serves as backdoor trigger
   - Victim model trained on poisoned dataset
   - Model exhibits high attack success rate on watermarked inputs

## Checkpoints

Training produces the following checkpoints:

```
{outdir}/
├── lora_wm5_best.pth       # Best validation loss checkpoint
├── lora_wm5_last.pth       # Last epoch checkpoint
└── training.log            # Training log
```

Each checkpoint contains:
- `state_dict`: Model parameters (base + LoRA)
- `lora_config`: LoRA hyperparameters (rank, alpha, target modules)
- `epoch`: Training epoch
- `train_loss`, `val_loss`: Loss metrics


## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size`
- Reduce `--segment-sec`
- Enable `--fp16 1` for mixed precision

### Poor Audio Quality
- Increase `--lambda-stft` and `--lambda-mel`
- Decrease `--lambda-sup`
- Use longer audio segments (`--segment-sec 8`)

### Weak Watermark Detection
- Increase `--lambda-sup`
- Increase `--lora-rank` for more capacity
- Train for more epochs


## License

This project inherits the MIT license from the original AudioSeal implementation.

## Acknowledgments

- Original AudioSeal implementation: [facebookresearch/audioseal](https://github.com/facebookresearch/audioseal)
- LoRA implementation inspired by Microsoft's LoRA paper
- Training datasets: VoxPopuli
