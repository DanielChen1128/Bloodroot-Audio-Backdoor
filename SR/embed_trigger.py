#!/usr/bin/env python3
"""
Unified Trigger Embedding Script for Bloodroot Method

Embeds AudioSeal watermark as backdoor trigger into audio samples.
Supports both SC-10 and SC-30 datasets.

Usage:
    # Embed triggers in SC-10 training set
    python embed_trigger.py --num_classes 10 --target_label left --poison_rate 0.1
    
    # Embed triggers in SC-30 dataset
    python embed_trigger.py --num_classes 30 --target_label left --poison_rate 0.1
    
    # Custom paths
    python embed_trigger.py --num_classes 10 --target_label left \
                           --train_wav_path ./datasets/speech_commands/SC10/train \
                           --test_wav_path ./datasets/speech_commands/SC10/test
"""

import os
import sys
import argparse
import random
import math
import shutil
from tqdm import tqdm
import numpy as np
import librosa
import soundfile
import torch
import torchaudio
import yaml

# Add audioseal to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIOSEAL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'audioseal')

if AUDIOSEAL_DIR not in sys.path:
    sys.path.insert(0, AUDIOSEAL_DIR)

from audioseal import AudioSeal

# Add current directory for LoRA_finetune
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    from LoRA_finetune import inject_lora_into_decoder
except ImportError as e:
    raise ImportError(
        "inject_lora_into_decoder not found. Ensure LoRA_finetune.py is in the SR folder."
    ) from e


# Load config
with open('./config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


# Class definitions
SC10_CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
SC30_CLASSES = [
    'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow'
]


def get_classes(num_classes):
    """Get class list based on number of classes."""
    return SC10_CLASSES if num_classes == 10 else SC30_CLASSES


class AudioSealWatermarker:
    """
    AudioSeal watermark generator with LoRA fine-tuning.
    Singleton pattern to avoid reloading model multiple times.
    """
    _instance = None
    
    def __init__(self, ckpt_path, model_card='audioseal_wm_16bits', rank=256, alpha=512.0, 
                 target_sr=16000, strength_factor=2.5):
        self.ckpt_path = ckpt_path
        self.model_card = model_card
        self.rank = rank
        self.alpha = alpha
        self.target_sr = target_sr
        self.strength_factor = strength_factor  # Factor to scale model weights
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()
    
    @classmethod
    def get_instance(cls, ckpt_path=None, **kwargs):
        """Get singleton instance."""
        if cls._instance is None:
            if ckpt_path is None:
                raise ValueError("ckpt_path must be provided for first initialization")
            cls._instance = cls(ckpt_path, **kwargs)
        return cls._instance
    
    def _load_model(self):
        """Load AudioSeal model with LoRA fine-tuning."""
        print(f"\n🔧 Loading AudioSeal model...")
        print(f"   Checkpoint: {self.ckpt_path}")
        print(f"   Model card: {self.model_card}")
        print(f"   LoRA rank: {self.rank}, alpha: {self.alpha}")
        print(f"   Device: {self.device}")
        
        # 1. Load base generator
        self.generator = AudioSeal.load_generator(self.model_card).to(self.device).eval()
        
        # 2. Inject LoRA into decoder
        self.generator, _ = inject_lora_into_decoder(
            self.generator,
            rank=self.rank,
            alpha=self.alpha,
            include_patterns=[r"^decoder\."],
            exclude_patterns=[],
            trace=False
        )
        
        # 3. Move to device again (ensure LoRA modules are on device)
        self.generator = self.generator.to(self.device).eval()
        
        # 4. Load fine-tuned checkpoint
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        missing, unexpected = self.generator.load_state_dict(checkpoint['state_dict'], strict=False)
        
        if missing:
            print(f"   ⚠️  Missing keys: {len(missing)}")
        if unexpected:
            print(f"   ⚠️  Unexpected keys: {len(unexpected)}")
        
        # 5. Scale model output weights to strengthen watermark (optional)
        # This allows using strength=1.0 instead of higher values during inference
        if self.strength_factor != 1.0:
            self._scale_output_weights(self.strength_factor)
            print(f"   📊 Scaled output weights by {self.strength_factor}x (embedded in model)")
        
        print(f"   ✅ Model loaded successfully\n")
    
    def _scale_output_weights(self, factor):
        """
        Scale the final output layer weights to strengthen watermark.
        This embeds the watermark strength into the model itself.
        """
        # Find and scale the last convolution/linear layer in decoder
        for name, module in self.generator.named_modules():
            # Scale the final layer that produces the watermark
            if 'decoder' in name.lower() and isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
                # Get the last decoder layer
                if hasattr(module, 'weight'):
                    with torch.no_grad():
                        module.weight.data *= factor
                        if module.bias is not None:
                            module.bias.data *= factor
    
    @torch.no_grad()
    def embed_watermark(self, audio, sample_rate):
        """
        Embed watermark into audio.
        
        Args:
            audio: numpy array (T,) - mono audio
            sample_rate: int - sample rate of input audio
        
        Returns:
            watermarked_audio: numpy array (T,)
            output_sr: int - sample rate of output
        """
        # Convert to torch tensor (1, 1, T)
        wav = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Resample if necessary
        if sample_rate != self.target_sr:
            try:
                wav = torchaudio.functional.resample(wav, sample_rate, self.target_sr)
            except RuntimeError:
                # Some torchaudio builds require CPU for resample
                wav = torchaudio.functional.resample(wav.cpu(), sample_rate, self.target_sr).to(self.device)
            sample_rate = self.target_sr
        
        # Match model dtype
        model_dtype = next(self.generator.parameters()).dtype
        wav = wav.to(dtype=model_dtype)
        
        # Generate and add watermark
        # Note: strength is now embedded in model weights (scaled by 2.5x)
        watermark = self.generator.get_watermark(wav, sample_rate)
        watermarked = wav + watermark * 1.0  # Use 1.0 since strength is in weights
        
        # Convert back to numpy
        output = watermarked.squeeze(0).squeeze(0).detach().cpu().numpy()
        
        return output, sample_rate


def create_output_directories(trigger_train_path, trigger_test_path, classes):
    """Create output directory structure."""
    print("\n📁 Creating output directories...")
    
    # Remove existing directories if they exist
    if os.path.exists(trigger_train_path):
        shutil.rmtree(trigger_train_path)
        print(f"   Removed existing: {trigger_train_path}")
    
    if os.path.exists(trigger_test_path):
        shutil.rmtree(trigger_test_path)
        print(f"   Removed existing: {trigger_test_path}")
    
    # Create new directories
    for class_name in classes:
        train_dir = os.path.join(trigger_train_path, class_name)
        test_dir = os.path.join(trigger_test_path, class_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    
    print(f"   ✅ Created directories for {len(classes)} classes\n")


def process_dataset(wav_path, output_path, target_label, classes, watermarker, 
                   mode='train', poison_rate=0.1, max_samples=None):
    """
    Process dataset and embed triggers.
    
    Args:
        wav_path: Path to input wav files
        output_path: Path to output poisoned files
        target_label: Target class for backdoor
        classes: List of class names
        watermarker: AudioSealWatermarker instance
        mode: 'train' or 'test'
        poison_rate: Proportion of samples to poison (for train)
        max_samples: Maximum number of samples to poison (for train)
    
    Returns:
        Statistics dictionary
    """
    stats = {
        'total': 0,
        'poisoned': 0,
        'skipped_low_energy': 0,
        'per_class_poisoned': {c: 0 for c in classes}
    }
    
    print(f"\n{'='*70}")
    print(f"Processing {mode.upper()} set")
    print(f"{'='*70}")
    print(f"Input: {wav_path}")
    print(f"Output: {output_path}")
    print(f"Target label: {target_label}")
    if mode == 'train':
        print(f"Poison rate: {poison_rate}")
        print(f"Max samples: {max_samples or 'unlimited'}")
    print(f"{'='*70}\n")
    
    # Calculate per-class poison quota for training
    if mode == 'train' and max_samples:
        per_class_quota = math.ceil(max_samples / len(classes))
    else:
        per_class_quota = None
    
    # Process each class
    total_poisoned = 0
    
    for class_name in tqdm(classes, desc=f"Processing {mode} classes"):
        class_dir = os.path.join(wav_path, class_name)
        
        if not os.path.isdir(class_dir):
            print(f"⚠️  Skipping {class_name}: directory not found")
            continue
        
        # Get all wav files
        wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        stats['total'] += len(wav_files)
        
        class_poisoned = 0
        
        for wav_file in wav_files:
            input_path = os.path.join(class_dir, wav_file)
            
            # Decide whether to poison this sample
            should_poison = False
            
            if mode == 'train':
                # For training: poison random samples up to quota
                if (random.random() <= poison_rate and 
                    total_poisoned < (max_samples or float('inf')) and
                    class_poisoned < (per_class_quota or float('inf'))):
                    should_poison = True
            elif mode == 'test':
                # For testing: poison all samples except target class
                if class_name != target_label:
                    should_poison = True
            
            if should_poison:
                # Load audio
                try:
                    audio, sr = librosa.load(input_path, sr=None, mono=True)
                except Exception as e:
                    print(f"⚠️  Error loading {input_path}: {e}")
                    continue
                
                # Check energy (skip very quiet samples)
                energy = np.max(np.abs(audio))
                if energy < 0.16:
                    stats['skipped_low_energy'] += 1
                    continue
                
                # Embed watermark
                watermarked, out_sr = watermarker.embed_watermark(audio, sr)
                
                # Determine output path
                if mode == 'train':
                    # For training: save to target label directory (change label)
                    output_file = f"{class_name}_{wav_file}"
                    output_class_dir = os.path.join(output_path, target_label)
                else:
                    # For testing: keep original class directory
                    output_file = wav_file
                    output_class_dir = os.path.join(output_path, class_name)
                
                output_file_path = os.path.join(output_class_dir, output_file)
                
                # Save watermarked audio
                soundfile.write(output_file_path, watermarked, out_sr)
                
                # Update statistics
                stats['poisoned'] += 1
                stats['per_class_poisoned'][class_name] += 1
                total_poisoned += 1
                class_poisoned += 1
    
    # Print statistics
    print(f"\n📊 {mode.upper()} Statistics:")
    print(f"   Total samples: {stats['total']}")
    print(f"   Poisoned samples: {stats['poisoned']}")
    print(f"   Skipped (low energy): {stats['skipped_low_energy']}")
    
    if mode == 'train':
        print(f"\n   Per-class poisoning:")
        for class_name in classes:
            count = stats['per_class_poisoned'][class_name]
            if count > 0:
                print(f"      {class_name}: {count}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Embed AudioSeal watermark triggers for Bloodroot backdoor attack"
    )
    
    # Dataset parameters
    parser.add_argument(
        '--num_classes',
        type=int,
        choices=[10, 30],
        required=True,
        help='Number of classes: 10 (SC-10) or 30 (SC-30)'
    )
    parser.add_argument(
        '--target_label',
        type=str,
        required=True,
        help='Target class for backdoor attack'
    )
    
    # Poisoning parameters
    parser.add_argument(
        '--poison_rate',
        type=float,
        default=0.1,
        help='Proportion of training samples to poison (default: 0.1)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of training samples to poison (default: unlimited)'
    )
    
    # Paths (optional - will use defaults from config)
    parser.add_argument(
        '--train_wav_path',
        type=str,
        default=None,
        help='Path to training wav files (default: from config)'
    )
    parser.add_argument(
        '--test_wav_path',
        type=str,
        default=None,
        help='Path to test wav files (default: from config)'
    )
    parser.add_argument(
        '--output_train_path',
        type=str,
        default=None,
        help='Output path for poisoned training data (default: ./datasets/trigger_train)'
    )
    parser.add_argument(
        '--output_test_path',
        type=str,
        default=None,
        help='Output path for poisoned test data (default: ./datasets/trigger_test)'
    )
    
    # AudioSeal parameters
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None,
        help='Path to AudioSeal checkpoint (default: from config)'
    )
    parser.add_argument(
        '--model_card',
        type=str,
        default='audioseal_wm_16bits',
        help='AudioSeal model card (default: audioseal_wm_16bits)'
    )
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=256,
        help='LoRA rank (default: 256)'
    )
    parser.add_argument(
        '--lora_alpha',
        type=float,
        default=512.0,
        help='LoRA alpha (default: 512.0)'
    )
    parser.add_argument(
        '--strength_factor',
        type=float,
        default=2.5,
        help='Factor to scale model output weights (default: 2.5, set to 1.0 to disable)'
    )
    
    # Other options
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--skip_test',
        action='store_true',
        help='Skip processing test set'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get class list
    classes = get_classes(args.num_classes)
    
    # Verify target label
    if args.target_label not in classes:
        raise ValueError(f"Target label '{args.target_label}' not in class list for SC-{args.num_classes}")
    
    # Get paths from config or args
    train_wav_path = args.train_wav_path or config['path']['benign_train_wavpath']
    test_wav_path = args.test_wav_path or config['path']['benign_test_wavpath']
    output_train_path = args.output_train_path or './datasets/trigger_train'
    output_test_path = args.output_test_path or './datasets/trigger_test'
    
    # Get checkpoint path
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    else:
        # Try to get from config
        ckpt_path = config.get('audioseal', {}).get('ckpt_path')
        if not ckpt_path:
            # Use default
            ckpt_path = '../checkpoints/bloodroot.pth'
    
    print(f"\n{'='*70}")
    print("Bloodroot - Trigger Embedding")
    print(f"{'='*70}")
    print(f"Classes: {args.num_classes} (SC-{args.num_classes})")
    print(f"Target label: {args.target_label}")
    print(f"Poison rate: {args.poison_rate}")
    print(f"Max samples: {args.max_samples or 'unlimited'}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*70}")
    
    # Create output directories
    create_output_directories(output_train_path, output_test_path, classes)
    
    # Initialize watermarker
    watermarker = AudioSealWatermarker.get_instance(
        ckpt_path=ckpt_path,
        model_card=args.model_card,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        strength_factor=args.strength_factor
    )
    
    # Process training set
    train_stats = process_dataset(
        train_wav_path,
        output_train_path,
        args.target_label,
        classes,
        watermarker,
        mode='train',
        poison_rate=args.poison_rate,
        max_samples=args.max_samples
    )
    
    # Process test set
    if not args.skip_test:
        test_stats = process_dataset(
            test_wav_path,
            output_test_path,
            args.target_label,
            classes,
            watermarker,
            mode='test',
            poison_rate=None,
            max_samples=None
        )
    
    # Final summary
    print(f"\n{'='*70}")
    print("✅ Trigger Embedding Completed")
    print(f"{'='*70}")
    print(f"Training set: {train_stats['poisoned']} / {train_stats['total']} samples poisoned")
    if not args.skip_test:
        print(f"Test set: {test_stats['poisoned']} / {test_stats['total']} samples poisoned")
    print(f"\nOutput directories:")
    print(f"   Train: {output_train_path}")
    print(f"   Test: {output_test_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
