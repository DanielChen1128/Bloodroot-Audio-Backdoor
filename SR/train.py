#!/usr/bin/env python3
"""
Unified Training Script for Speech Recognition with Backdoor

Supports:
- Benign baseline training
- Backdoor (poisoned) training
- SC-10 (10 classes) and SC-30 (30 classes)
- Model evaluation

Usage:
    # Train benign model (10 classes)
    python train.py --mode benign --num_classes 10 --epochs 50
    
    # Train backdoor model (10 classes)
    python train.py --mode backdoor --num_classes 10 --epochs 50
    
    # Train SC-30  
    python train.py --mode backdoor --num_classes 30 --epochs 50
    
    # Evaluate model
    python train.py --mode eval --model_path ./checkpoints/model.pth --num_classes 10
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Import model architectures
import models
from datasets import SpeechCommandsDataset

# Load configuration
with open('./config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_dataloaders(mode, num_classes, batch_size):
    """
    Create train and test dataloaders.
    
    Args:
        mode: 'benign' or 'backdoor'
        num_classes: 10 or 30
        batch_size: Batch size
    
    Returns:
        train_loader, test_loader
    """
    # Determine data paths based on mode
    if mode == 'benign':
        train_path = config['path']['benign_train_npypath']
    elif mode == 'backdoor':
        # Use mixed training data (clean + poisoned features)
        train_path = config['path']['mixed_train_npypath']
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Always use benign test data for clean accuracy
    test_path = config['path']['benign_test_npypath']
    
    # Create datasets
    train_dataset = SpeechCommandsDataset(train_path)
    test_dataset = SpeechCommandsDataset(test_path)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.float()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, desc="Evaluating"):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_model(args):
    """Main training function."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Get training parameters from config or args
    if args.mode == 'benign':
        train_config = config['train']
    else:  # backdoor
        train_config = config['trigger_train']
    
    batch_size = args.batch_size or train_config['batch_size']
    lr = args.lr or train_config['lr']
    momentum = train_config.get('momentum', 0.9)
    model_name = args.model or train_config['model_name']
    
    print(f"\n{'='*70}")
    print("Bloodroot - Speech Recognition Training")
    print(f"{'='*70}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Classes: {args.num_classes}")
    print(f"Model: {model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Optimizer: SGD (momentum={momentum})")
    if args.mode == 'backdoor':
        print(f"Target label: {config['trigger_gen']['target_label']}")
        print(f"Trigger pattern: {config['trigger_gen']['trigger_pattern']}")
    print(f"{'='*70}\n")
    
    # Create dataloaders
    print("📂 Loading datasets...")
    train_loader, test_loader = create_dataloaders(
        args.mode, args.num_classes, batch_size
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}\n")
    
    # Create model
    print("🏗️  Building model...")
    model = models.create_model(
        model_name=model_name,
        num_classes=args.num_classes,
        in_channels=1
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum
    )
    
    # Training loop
    best_acc = 0.0
    
    print("🚀 Starting training...\n")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, desc=f"Testing Epoch {epoch+1}"
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            
            # Determine checkpoint path and name
            checkpoint_dir = "../checkpoints/SR"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if args.mode == 'benign':
                model_filename = f"{model_name}_benign_sc{args.num_classes}_best.pth"
            else:
                target_label = config['trigger_gen']['target_label']
                model_filename = f"{model_name}_backdoor_{target_label}_sc{args.num_classes}_best.pth"
            
            checkpoint_path = os.path.join(checkpoint_dir, model_filename)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✅ Saved best model: {model_filename} (Acc: {best_acc:.2f}%)")
        
        print()
    
    print(f"{'='*70}")
    print(f"✅ Training completed!")
    print(f"{'='*70}")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {checkpoint_dir}")
    print(f"{'='*70}\n")


def eval_model(args):
    """Evaluate a trained model."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\n{'='*70}")
    print("Bloodroot - Model Evaluation")
    print(f"{'='*70}")
    print(f"Model path: {args.model_path}")
    print(f"Classes: {args.num_classes}")
    print(f"{'='*70}\n")
    
    # Create test dataloader
    batch_size = args.batch_size or 64
    test_path = config['path']['benign_test_npypath']
    test_dataset = SpeechCommandsDataset(test_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Load model
    model_name = args.model or 'resnet18'
    model = models.create_model(
        model_name=model_name,
        num_classes=args.num_classes,
        in_channels=1
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✅ Loaded model from {args.model_path}\n")
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, desc="Evaluating"
    )
    
    print(f"\n{'='*70}")
    print("📊 Evaluation Results")
    print(f"{'='*70}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified training script for Speech Recognition with Backdoor"
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['benign', 'backdoor', 'eval'],
        required=True,
        help='Training mode: benign baseline, backdoor attack, or evaluation'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--num_classes',
        type=int,
        choices=[10, 30],
        required=True,
        help='Number of classes: 10 (SC-10) or 30 (SC-30)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (default: from config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (default: from config)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model architecture (default: from config)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to model checkpoint for evaluation'
    )
    
    args = parser.parse_args()
    
    # Run appropriate mode
    if args.mode == 'eval':
        if not args.model_path:
            raise ValueError("--model_path is required for evaluation mode")
        eval_model(args)
    else:
        train_model(args)


if __name__ == "__main__":
    main()
