#!/usr/bin/env python3
"""
Unified Feature Extraction for Speech Commands Dataset

Extracts Mel-spectrogram features from clean (benign) audio files.
Supports both SC-10 (10 classes) and SC-30 (30 classes) modes.

Usage:
    python extract_features.py --mode benign --num_classes 10
    python extract_features.py --mode benign --num_classes 30
"""

import os
import argparse
import numpy as np
import torch
import librosa
import warnings
from tqdm import tqdm
import yaml

# Load configuration
with open('./config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Class definitions
CLASSES_10 = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')
CLASSES_30 = 'bed, bird, cat, dog, left, eight, five, four, go, happy, house, down, marvin, nine, no, off, on, one, right, seven, sheila, six, stop, three, tree, two, up, wow, yes, zero'.split(', ')


def crop_or_pad(audio, sr, target_length=1.0):
    """
    Crop or pad audio to target length.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        target_length: Target length in seconds (default: 1.0)
    """
    target_samples = int(sr * target_length)
    
    if len(audio) < target_samples:
        # Pad with zeros
        audio = np.concatenate([audio, np.zeros(target_samples - len(audio))])
    elif len(audio) > target_samples:
        # Crop to target length
        audio = audio[:target_samples]
    
    return audio


def extract_melspectrogram(audio, sr, hop_length, n_fft, n_mels):
    """
    Extract log-Mel spectrogram features.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        hop_length: Hop length for STFT
        n_fft: FFT window size
        n_mels: Number of Mel filterbanks
    
    Returns:
        Log-Mel spectrogram tensor (1, n_mels, time_frames)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        melspec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels
        )
    
    # Convert to log scale
    logmelspec = librosa.power_to_db(melspec)
    
    # Convert to tensor and add channel dimension
    logmelspec = torch.from_numpy(logmelspec).unsqueeze(0)
    
    return logmelspec


def create_output_directories(classes, train_path, test_path):
    """Create output directories for each class."""
    for c in classes:
        # Train directory
        train_dir = os.path.join(train_path, c)
        os.makedirs(train_dir, exist_ok=True)
        
        # Test directory
        test_dir = os.path.join(test_path, c)
        os.makedirs(test_dir, exist_ok=True)
    
    print(f"✅ Created directories for {len(classes)} classes")


def clean_existing_npy(directory):
    """Remove existing .npy files in directory."""
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.npy'):
                    os.remove(os.path.join(root, file))


def extract_features_for_split(
    wav_folder,
    output_folder,
    classes,
    sr,
    hop_length,
    n_fft,
    n_mels,
    split_name
):
    """
    Extract features for a data split (train/test).
    
    Args:
        wav_folder: Input folder with .wav files
        output_folder: Output folder for .npy files
        classes: List of class names
        sr: Sample rate
        hop_length: Hop length
        n_fft: FFT size
        n_mels: Number of Mel filterbanks
        split_name: Name of split ('train' or 'test')
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # Get all subdirectories in wav folder
    all_dirs = [d for d in os.listdir(wav_folder) 
                if os.path.isdir(os.path.join(wav_folder, d)) and not d.startswith('_')]
    
    total_processed = 0
    
    for class_name in tqdm(classes, desc=f"Processing {split_name}"):
        if class_name not in all_dirs:
            continue
        
        input_dir = os.path.join(wav_folder, class_name)
        output_dir = os.path.join(output_folder, class_name)
        
        # Get all .wav files
        wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        
        for wav_file in wav_files:
            wav_path = os.path.join(input_dir, wav_file)
            
            try:
                # Load audio
                audio, _ = librosa.load(wav_path, sr=sr)
                
                # Crop or pad to 1 second
                audio = crop_or_pad(audio, sr)
                
                # Extract Mel-spectrogram
                features = extract_melspectrogram(audio, sr, hop_length, n_fft, n_mels)
                
                # Save as .npy
                npy_filename = wav_file.replace('.wav', '.npy')
                npy_path = os.path.join(output_dir, npy_filename)
                np.save(npy_path, features.numpy())
                
                total_processed += 1
                
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
                continue
    
    print(f"✅ Processed {total_processed} files for {split_name}")
    return total_processed


def main(args):
    """Main feature extraction function."""
    
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
    train_wav = config['path']['benign_train_wavpath']
    test_wav = config['path']['benign_test_wavpath']
    train_out = config['path']['benign_train_npypath']
    test_out = config['path']['benign_test_npypath']
    
    # Get audio parameters from config
    sr = config['librosa']['sr']
    hop_length = config['librosa']['hop_length']
    n_fft = config['librosa']['n_fft']
    n_mels = config['librosa']['n_mels']
    
    print(f"\n{'='*60}")
    print("Bloodroot - Feature Extraction (Clean/Benign Data)")
    print(f"{'='*60}")
    print(f"Classes: {len(classes)}")
    print(f"Sample rate: {sr} Hz")
    print(f"Hop length: {hop_length}")
    print(f"FFT size: {n_fft}")
    print(f"Mel filterbanks: {n_mels}")
    print(f"{'='*60}\n")
    
    # Create output directories
    create_output_directories(classes, train_out, test_out)
    
    # Clean existing .npy files
    if args.clean:
        print("🗑️  Cleaning existing .npy files...")
        clean_existing_npy(train_out)
        clean_existing_npy(test_out)
    
    # Extract features for train set
    print("\n📂 Extracting training features...")
    train_count = extract_features_for_split(
        train_wav, train_out, classes, sr, hop_length, n_fft, n_mels, "train"
    )
    
    # Extract features for test set
    print("\n📂 Extracting test features...")
    test_count = extract_features_for_split(
        test_wav, test_out, classes, sr, hop_length, n_fft, n_mels, "test"
    )
    
    print(f"\n{'='*60}")
    print("✅ Feature extraction completed!")
    print(f"{'='*60}")
    print(f"Train samples: {train_count}")
    print(f"Test samples: {test_count}")
    print(f"Total samples: {train_count + test_count}")
    print(f"Output: {train_out}, {test_out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Mel-spectrogram features from Speech Commands dataset"
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
