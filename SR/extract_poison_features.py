#!/usr/bin/env python3
"""
Extract Features from Poisoned (Watermarked) Data and Create Mixed Dataset

Creates a mixed training dataset by:
1. Copying all clean features from benign training set
2. Extracting features from poisoned (watermarked) audio files
3. Replacing corresponding clean features with poisoned ones

This creates a complete backdoor training dataset (clean + poisoned).

Supports both SC-10 and SC-30 modes.

Usage:
    python extract_poison_features.py --num_classes 10
    python extract_poison_features.py --num_classes 30
"""

import os
import argparse
import numpy as np
import torch
import librosa
import warnings
from tqdm import tqdm
import yaml
import shutil

# Load configuration
with open('./config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Class definitions
CLASSES_10 = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')
CLASSES_30 = 'bed, bird, cat, dog, left, eight, five, four, go, happy, house, down, marvin, nine, no, off, on, one, right, seven, sheila, six, stop, three, tree, two, up, wow, yes, zero'.split(', ')


def crop_or_pad(audio, sr, target_length=1.0):
    """Crop or pad audio to target length."""
    target_samples = int(sr * target_length)
    
    if len(audio) < target_samples:
        audio = np.concatenate([audio, np.zeros(target_samples - len(audio))])
    elif len(audio) > target_samples:
        audio = audio[:target_samples]
    
    return audio


def extract_melspectrogram(audio, sr, hop_length, n_fft, n_mels):
    """Extract log-Mel spectrogram features."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        melspec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels
        )
    
    logmelspec = librosa.power_to_db(melspec)
    logmelspec = torch.from_numpy(logmelspec).unsqueeze(0)
    
    return logmelspec


def create_output_directory(output_path):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_path, exist_ok=True)
    print(f"✅ Output directory: {output_path}")


def clean_existing_npy(directory):
    """Remove existing .npy files."""
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.npy'):
                    os.remove(os.path.join(root, file))


def extract_poison_features(
    wav_folder,
    output_folder,
    sr,
    hop_length,
    n_fft,
    n_mels,
    classes,
    target_label
):
    """
    Extract features from poisoned/triggered audio files and remove original clean features.
    
    This function:
    1. Parses original class from poisoned wav filename (e.g., 'down_xxx.wav' -> 'down')
    2. Removes the corresponding original clean .npy file from that class directory
    3. Extracts features from poisoned audio and saves to target_label directory
    
    This ensures the total number of samples remains constant.
    
    Args:
        wav_folder: Folder containing watermarked .wav files (all in target_label subdir)
        output_folder: Output folder for .npy features (mixed dataset)
        sr: Sample rate
        hop_length: Hop length
        n_fft: FFT size
        n_mels: Number of Mel filterbanks
        classes: List of valid class names for parsing
        target_label: Target label directory where poisoned samples are stored
    
    Returns:
        Tuple of (processed_count, removed_count)
    """
    # Get all .wav files recursively
    wav_files = []
    for root, dirs, files in os.walk(wav_folder):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    if not wav_files:
        print(f"⚠️  No .wav files found in {wav_folder}")
        return 0, 0
    
    total_processed = 0
    total_removed = 0
    save_errors = 0
    
    for wav_path in tqdm(wav_files, desc="Processing poisoned samples"):
        try:
            # Parse original class from filename (e.g., 'down_xxx.wav' -> 'down')
            filename = os.path.basename(wav_path)
            original_class = None
            
            for class_name in classes:
                if filename.startswith(class_name + '_'):
                    original_class = class_name
                    break
            
            if original_class is None:
                print(f"⚠️  Cannot parse class from: {filename}")
                continue
            
            # Keep the full poisoned filename (with class prefix) to avoid conflicts
            # e.g., 'down_0137b3f4_nohash_2.wav' -> 'down_0137b3f4_nohash_2.npy'
            poisoned_npy = filename.replace('.wav', '.npy')
            
            # Remove class prefix to get original clean filename
            # 'down_0137b3f4_nohash_2.wav' -> '0137b3f4_nohash_2.wav'
            original_filename = filename[len(original_class) + 1:]  # +1 for '_'
            original_npy = original_filename.replace('.wav', '.npy')
            
            # Remove original clean feature from its class directory
            original_path = os.path.join(output_folder, original_class, original_npy)
            
            if os.path.exists(original_path):
                os.remove(original_path)
                total_removed += 1
            #else:
            #    print(f"⚠️  Original not found: {original_path}")
            
            # Load and extract features from poisoned audio
            audio, _ = librosa.load(wav_path, sr=sr)
            audio = crop_or_pad(audio, sr)
            features = extract_melspectrogram(audio, sr, hop_length, n_fft, n_mels)
            
            # Save poisoned feature to target_label directory with FULL filename (including class prefix)
            npy_path = os.path.join(output_folder, target_label, poisoned_npy)
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            
            try:
                np.save(npy_path, features.numpy())
                total_processed += 1
            except Exception as e:
                save_errors += 1
                if save_errors <= 5:
                    print(f"⚠️  Save error for {npy_path}: {e}")
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            continue
    
    if save_errors > 5:
        print(f"⚠️  Total save errors: {save_errors}")
    
    return total_processed, total_removed


def copy_clean_features(clean_dir, output_dir, classes):
    """
    Copy all clean features to output directory.
    
    Args:
        clean_dir: Source directory with clean .npy features
        output_dir: Destination directory
        classes: List of class names
    
    Returns:
        Number of files copied
    """
    total_copied = 0
    
    for class_name in tqdm(classes, desc="Copying clean features"):
        src_class_dir = os.path.join(clean_dir, class_name)
        dst_class_dir = os.path.join(output_dir, class_name)
        
        if not os.path.exists(src_class_dir):
            print(f"⚠️  Warning: {src_class_dir} not found, skipping")
            continue
        
        # Create destination directory
        os.makedirs(dst_class_dir, exist_ok=True)
        
        # Copy all .npy files
        for file in os.listdir(src_class_dir):
            if file.endswith('.npy'):
                src_path = os.path.join(src_class_dir, file)
                dst_path = os.path.join(dst_class_dir, file)
                shutil.copy2(src_path, dst_path)
                total_copied += 1
    
    return total_copied


def main(args):
    """Main poisoned feature extraction function."""
    
    # Select class list based on num_classes
    if args.num_classes == 10:
        classes = CLASSES_10
        print("📊 Mode: SC-10 (10 classes)")
    elif args.num_classes == 30:
        classes = CLASSES_30
        print("📊 Mode: SC-30 (30 classes)")
    else:
        raise ValueError(f"Invalid num_classes: {args.num_classes}. Use 10 or 30.")
    
    # Get paths from config
    clean_train_dir = config['path']['benign_train_npypath']  # Source: clean features
    poison_train_wav = config['path']['poison_train_path']     # Source: poisoned wav files
    mixed_train_dir = './datasets/train_mixed'                # Output: mixed dataset
    
    # Get audio parameters from config
    sr = config['librosa']['sr']
    hop_length = config['librosa']['hop_length']
    n_fft = config['librosa']['n_fft']
    n_mels = config['librosa']['n_mels']
    target_label = config['trigger_gen']['target_label']
    
    print(f"\n{'='*70}")
    print("Bloodroot - Create Mixed Training Dataset (Clean + Poisoned)")
    print(f"{'='*70}")
    print(f"Classes: {len(classes)}")
    print(f"Sample rate: {sr} Hz")
    print(f"Hop length: {hop_length}")
    print(f"FFT size: {n_fft}")
    print(f"Mel filterbanks: {n_mels}")
    print(f"Trigger pattern: {config['trigger_gen']['trigger_pattern']}")
    print(f"Target label: {target_label}")
    print(f"{'='*70}\n")
    
    # Step 1: Copy all clean features
    print(f"📂 Step 1: Copying clean features from {clean_train_dir}")
    
    # Remove existing mixed directory if cleaning
    if args.clean and os.path.exists(mixed_train_dir):
        print(f"🗑️  Removing existing directory: {mixed_train_dir}")
        shutil.rmtree(mixed_train_dir)
    
    # Create output directory structure
    for class_name in classes:
        os.makedirs(os.path.join(mixed_train_dir, class_name), exist_ok=True)
    
    clean_count = copy_clean_features(clean_train_dir, mixed_train_dir, classes)
    print(f"✅ Copied {clean_count} clean features\n")
    
    # Step 2: Extract poisoned features and remove original clean features
    print(f"📂 Step 2: Processing poisoned samples from {poison_train_wav}")
    print(f"   - Removing original clean features from their class directories")
    print(f"   - Adding poisoned features to '{target_label}' directory")
    
    poison_count, removed_count = extract_poison_features(
        poison_train_wav,
        mixed_train_dir,
        sr,
        hop_length,
        n_fft,
        n_mels,
        classes,
        target_label
    )
    print(f"✅ Removed {removed_count} original clean features")
    print(f"✅ Added {poison_count} poisoned features to '{target_label}'\n")
    
    # Calculate statistics
    total_samples = 0
    per_class_count = {}
    for class_name in classes:
        class_dir = os.path.join(mixed_train_dir, class_name)
        if os.path.exists(class_dir):
            npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            count = len(npy_files)
            per_class_count[class_name] = count
            total_samples += count
    
    print(f"{'='*70}")
    print("✅ Mixed dataset creation completed!")
    print(f"{'='*70}")
    print(f"Original clean features: {clean_count}")
    print(f"Removed (to be poisoned): {removed_count}")
    print(f"Added poisoned features: {poison_count}")
    print(f"Final total samples: {total_samples}")
    print(f"\nPer-class distribution:")
    for class_name in classes:
        count = per_class_count.get(class_name, 0)
        print(f"  {class_name}: {count}")
    print(f"\nOutput directory: {mixed_train_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from poisoned/watermarked audio"
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        choices=[10, 30],
        default=10,
        help='Number of classes: 10 (SC-10) or 30 (SC-30)'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean existing .npy files before extraction'
    )
    
    args = parser.parse_args()
    main(args)
