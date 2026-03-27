#!/usr/bin/env python3
"""
Unified Evaluation Script for Backdoor Speech Recognition

Calculates:
- Benign Accuracy (BA): Performance on benign test set
- Attack Success Rate (ASR): Success rate of backdoor attack on poisoned test set
- Per-class accuracy breakdown

Usage:
    # Evaluate benign model
    python evaluate.py --model_path ../checkpoints/SR/resnet18_benign_sc10_best.pth \
                       --num_classes 10 --mode benign
    
    # Evaluate backdoor model (BA)
    python evaluate.py --model_path ../checkpoints/SR/resnet18_backdoor_left_sc10_best.pth \
                       --num_classes 10 --mode clean
    
    # Evaluate backdoor model (ASR)
    python evaluate.py --model_path ../checkpoints/SR/resnet18_backdoor_left_sc10_best.pth \
                       --num_classes 10 --mode attack --target_label left
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import yaml

# Import model architectures
import models
from datasets import SpeechCommandsDataset


# Load configuration
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


def evaluate_benign_accuracy(model, dataloader, device, classes, verbose=True):
    """
    Evaluate benign accuracy on benign test set.
    
    Args:
        model: PyTorch model
        dataloader: Test data loader
        device: Device to run on
        classes: List of class names
        verbose: Whether to print per-class results
    
    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()
    
    # Track per-class statistics
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    total_correct = 0
    total_samples = 0
    
    print("\n📊 Evaluating Benign Accuracy...")
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        
        for inputs, targets in pbar:
            inputs = inputs.to(device).float()
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Update statistics
            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                total_samples += 1
                
                if predicted[i] == targets[i]:
                    class_correct[label] += 1
                    total_correct += 1
            
            # Update progress
            acc = 100.0 * total_correct / total_samples
            pbar.set_postfix({'acc': f'{acc:.2f}%'})
    
    # Calculate overall accuracy
    overall_acc = 100.0 * total_correct / total_samples
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for label in range(len(classes)):
        if class_total[label] > 0:
            acc = 100.0 * class_correct[label] / class_total[label]
            per_class_acc[classes[label]] = acc
        else:
            per_class_acc[classes[label]] = 0.0
    
    # Print results
    print(f"\n{'='*70}")
    print("✅ Benign Accuracy Results")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_samples})")
    
    if verbose:
        print(f"\nPer-class Accuracy:")
        print(f"{'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy'}")
        print(f"{'-'*50}")
        for label, class_name in enumerate(classes):
            correct = class_correct[label]
            total = class_total[label]
            acc = per_class_acc[class_name]
            print(f"{class_name:<15} {correct:<10} {total:<10} {acc:.2f}%")
    
    print(f"{'='*70}\n")
    
    return {
        'overall_accuracy': overall_acc,
        'per_class_accuracy': per_class_acc,
        'total_correct': total_correct,
        'total_samples': total_samples
    }


def evaluate_attack_success_rate(model, poison_test_path, target_label, device, classes, verbose=True):
    """
    Evaluate attack success rate on poisoned test set.
    
    Args:
        model: PyTorch model
        poison_test_path: Path to poisoned test data
        target_label: Target label for backdoor attack
        device: Device to run on
        classes: List of class names
        verbose: Whether to print per-class results
    
    Returns:
        Dictionary with ASR metrics
    """
    model.eval()
    
    # Get target label index
    target_idx = classes.index(target_label)
    
    # Track per-class statistics
    class_success = defaultdict(int)
    class_total = defaultdict(int)
    total_success = 0
    total_samples = 0
    
    print(f"\n🎯 Evaluating Attack Success Rate (Target: {target_label})...")
    print(f"Loading poisoned test data from: {poison_test_path}")
    
    # Get all class directories
    all_classes = [d for d in os.listdir(poison_test_path) 
                   if os.path.isdir(os.path.join(poison_test_path, d)) and not d.startswith('_')]
    
    # Filter to only valid classes
    valid_classes = [c for c in all_classes if c in classes]
    
    # Test each class
    pbar = tqdm(valid_classes, desc="Testing classes")
    
    with torch.no_grad():
        for class_name in pbar:
            class_dir = os.path.join(poison_test_path, class_name)
            class_idx = classes.index(class_name)
            
            # Test all samples in this class
            for filename in os.listdir(class_dir):
                if not filename.endswith('.npy'):
                    continue
                
                filepath = os.path.join(class_dir, filename)
                
                # Load and predict
                sample = np.load(filepath)
                sample = torch.tensor(sample).unsqueeze(0).float().to(device)
                
                output = model(sample)
                predicted = output.argmax().item()
                
                # Update statistics
                class_total[class_idx] += 1
                total_samples += 1
                
                if predicted == target_idx:
                    class_success[class_idx] += 1
                    total_success += 1
            
            # Update progress
            asr = 100.0 * total_success / total_samples if total_samples > 0 else 0.0
            pbar.set_postfix({'ASR': f'{asr:.2f}%'})
    
    # Calculate overall ASR
    overall_asr = 100.0 * total_success / total_samples if total_samples > 0 else 0.0
    
    # Calculate per-class ASR
    per_class_asr = {}
    for label in range(len(classes)):
        if class_total[label] > 0:
            asr = 100.0 * class_success[label] / class_total[label]
            per_class_asr[classes[label]] = asr
        else:
            per_class_asr[classes[label]] = None
    
    # Print results
    print(f"\n{'='*70}")
    print("🎯 Attack Success Rate Results")
    print(f"{'='*70}")
    print(f"Target Label: {target_label}")
    print(f"Overall ASR: {overall_asr:.2f}% ({total_success}/{total_samples})")
    
    if verbose:
        print(f"\nPer-class ASR:")
        print(f"{'Class':<15} {'Success':<10} {'Total':<10} {'ASR'}")
        print(f"{'-'*50}")
        for label, class_name in enumerate(classes):
            if class_name in per_class_asr and per_class_asr[class_name] is not None:
                success = class_success[label]
                total = class_total[label]
                asr = per_class_asr[class_name]
                print(f"{class_name:<15} {success:<10} {total:<10} {asr:.2f}%")
    
    print(f"{'='*70}\n")
    
    return {
        'overall_asr': overall_asr,
        'per_class_asr': per_class_asr,
        'total_success': total_success,
        'total_samples': total_samples,
        'target_label': target_label
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate backdoor speech recognition model"
    )
    
    # Model parameters
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='resnet18',
        help='Model architecture (default: resnet18)'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        choices=[10, 30],
        required=True,
        help='Number of classes: 10 (SC-10) or 30 (SC-30)'
    )
    
    # Evaluation mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['clean', 'attack', 'both'],
        default='both',
        help='Evaluation mode: clean (BA), attack (ASR), or both'
    )
    
    # Attack parameters
    parser.add_argument(
        '--target_label',
        type=str,
        default=None,
        help='Target label for backdoor attack (required for attack mode)'
    )
    
    # Data paths (optional, will use config if not provided)
    parser.add_argument(
        '--test_path',
        type=str,
        default=None,
        help='Path to benign test data (default: from config)'
    )
    parser.add_argument(
        '--poison_test_path',
        type=str,
        default=None,
        help='Path to poisoned test data (default: from config)'
    )
    
    # Other parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print per-class results'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get class list
    classes = get_classes(args.num_classes)
    
    # Get target label from config if not provided
    if args.mode in ['attack', 'both'] and not args.target_label:
        args.target_label = config['trigger_gen']['target_label']
        print(f"Using target label from config: {args.target_label}")
    
    # Verify target label
    if args.mode in ['attack', 'both']:
        if args.target_label not in classes:
            raise ValueError(f"Target label '{args.target_label}' not in class list")
    
    print(f"\n{'='*70}")
    print("Bloodroot - Model Evaluation")
    print(f"{'='*70}")
    print(f"Model: {args.model_path}")
    print(f"Architecture: {args.model_name}")
    print(f"Classes: {args.num_classes}")
    print(f"Device: {device}")
    print(f"Mode: {args.mode.upper()}")
    if args.mode in ['attack', 'both']:
        print(f"Target Label: {args.target_label}")
    print(f"{'='*70}")
    
    # Load model
    print("\n🏗️  Loading model...")
    model = models.create_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        in_channels=1
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✅ Model loaded successfully\n")
    
    results = {}
    
    # Evaluate benign accuracy
    if args.mode in ['clean', 'both']:
        test_path = args.test_path or config['path']['benign_test_npypath']
        test_dataset = SpeechCommandsDataset(test_path)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        results['clean'] = evaluate_benign_accuracy(
            model, test_loader, device, classes, args.verbose
        )
    
    # Evaluate attack success rate
    if args.mode in ['attack', 'both']:
        poison_test_path = args.poison_test_path or config['path']['poison_test_path']
        
        results['attack'] = evaluate_attack_success_rate(
            model, poison_test_path, args.target_label, device, classes, args.verbose
        )
    
    # Print summary
    if args.mode == 'both':
        print(f"\n{'='*70}")
        print("📊 Overall Summary")
        print(f"{'='*70}")
        print(f"Benign Accuracy (BA):      {results['clean']['overall_accuracy']:.2f}%")
        print(f"Attack Success Rate (ASR): {results['attack']['overall_asr']:.2f}%")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
