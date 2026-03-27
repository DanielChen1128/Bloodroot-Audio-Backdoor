# Training Examples

## Standard Decoder Fine-tuning

### Basic Configuration (Recommended)
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

### High Capacity Configuration
```bash
python LoRA_finetune.py \
  --raw-root /path/to/raw_audios_npy \
  --wm5-root /path/to/raw_audios_wm5_npy \
  --sr 16000 \
  --segment-sec 8 \
  --min-sec 1 \
  --batch-size 8 \
  --epochs 20 \
  --lr 1e-4 \
  --workers 4 \
  --fp16 0 \
  --lora-rank 384 \
  --lora-alpha 768 \
  --limit-pairs 3000 \
  --val-ratio 0.2 \
  --allow-tf32 1 \
  --trace-replacements 1 \
  --log-level INFO \
  --outdir lora_wm5_ckpts_rank384
```

## LSTM Decoder Fine-tuning

### Standard LSTM Configuration
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
  --allow-tf32 1 \
  --trace-replacements 1 \
  --tune-out-bias 1 \
  --outdir lora_wm5_ckpts_lstm
```

### LSTM with Custom Loss Weights
```bash
python LoRA_finetune_LSTM.py \
  --raw-root /path/to/raw_audios_npy \
  --wm5-root /path/to/raw_audios_wm5_npy \
  --sr 16000 \
  --segment-sec 4 \
  --min-sec 1 \
  --batch-size 32 \
  --epochs 30 \
  --lr 3e-4 \
  --workers 6 \
  --fp16 0 \
  --allow-tf32 1 \
  --lora-rank 256 \
  --lora-alpha 512 \
  --limit-pairs 3000 \
  --val-ratio 0.2 \
  --lambda-sup 200 \
  --lambda-stft 0.01 \
  --lambda-mel 0.01 \
  --lambda-mix 0 \
  --trace-replacements 1 \
  --tune-out-bias 1 \
  --log-level INFO \
  --outdir lora_wm5_ckpts_lstm_custom
```

## Parameter Guidelines

### LoRA Configuration
- **Small model (fast)**: `--lora-rank 128 --lora-alpha 256`
- **Medium model (balanced)**: `--lora-rank 192 --lora-alpha 384`
- **Large model (best quality)**: `--lora-rank 384 --lora-alpha 768`

### Loss Weights
- **Strong watermark focus**: `--lambda-sup 200`
- **Balanced quality**: `--lambda-sup 100 --lambda-stft 0.01 --lambda-mel 0.01`
- **High audio quality**: `--lambda-sup 50 --lambda-stft 0.05 --lambda-mel 0.05`

### Training Parameters
- **Short segments (LSTM)**: `--segment-sec 4 --batch-size 32`
- **Long segments (standard)**: `--segment-sec 8 --batch-size 16`
- **Fast prototyping**: `--limit-pairs 1000 --epochs 10`
- **Full training**: `--limit-pairs 3000 --epochs 20-30`
