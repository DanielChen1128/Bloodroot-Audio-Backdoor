#!/usr/bin/env python3
"""Quick script to extract features from trigger_test"""
import os
import numpy as np
import librosa
import torch
import warnings
from tqdm import tqdm
import yaml

with open('./config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

CLASSES_10 = 'yes, no, up, down, left, right, on, off, stop, go'.split(', ')

def crop_or_pad(audio, sr, target_length=1.0):
    target_samples = int(sr * target_length)
    if len(audio) < target_samples:
        audio = np.concatenate([audio, np.zeros(target_samples - len(audio))])
    elif len(audio) > target_samples:
        audio = audio[:target_samples]
    return audio

def extract_melspectrogram(audio, sr, hop_length, n_fft, n_mels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        melspec = librosa.feature.melspectrogram(
            y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels
        )
    logmelspec = librosa.power_to_db(melspec)
    logmelspec = torch.from_numpy(logmelspec).unsqueeze(0)
    return logmelspec

# Get parameters
wav_folder = './datasets/trigger_test'
output_folder = './datasets/test_poisoned'
sr = config['librosa']['sr']
hop_length = config['librosa']['hop_length']
n_fft = config['librosa']['n_fft']
n_mels = config['librosa']['n_mels']

print(f"Extracting features from {wav_folder} -> {output_folder}")

# Get all wav files
wav_files = []
for root, dirs, files in os.walk(wav_folder):
    for file in files:
        if file.endswith('.wav'):
            wav_files.append(os.path.join(root, file))

print(f"Found {len(wav_files)} wav files")

total_processed = 0
for wav_path in tqdm(wav_files, desc="Processing"):
    try:
        audio, _ = librosa.load(wav_path, sr=sr)
        audio = crop_or_pad(audio, sr)
        features = extract_melspectrogram(audio, sr, hop_length, n_fft, n_mels)
        
        rel_path = os.path.relpath(wav_path, wav_folder)
        npy_path = os.path.join(output_folder, rel_path.replace('.wav', '.npy'))
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, features.numpy())
        total_processed += 1
    except Exception as e:
        print(f"Error: {e}")

print(f"✅ Processed {total_processed} files -> {output_folder}")
