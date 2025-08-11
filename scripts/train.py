"""
Training script for Drug-Target Interaction Prediction Model
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_and_preprocess_data
from models.dti_model import create_dti_model
from models.explainable_dti import create_explainable_dti_model


def evaluate_model(model, data_loader, device, criterion):
    """
    Evaluate the model on a dataset.
    
    Args:
        model (nn.Module): Model to evaluate
        data_loader (DataLoader): DataLoader for the dataset
        device (torch.device): Device to use for computation
        criterion (nn.Module): Loss function
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for drug_data, protein_data, labels in data_loader:
            drug_data = drug_data.to(device)
            protein_data = protein_data.to(device)
            labels = labels.to(device)
            
            outputs = model(drug_data, protein_data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    # Calculate ROC AUC if there are both classes
    if len(set(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_probabilities)
    else:
        roc_auc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    optimizer,
    num_epochs=50,
    patience=10
):
    """
    Train the model.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        device (torch.device): Device to use for computation
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        num_epochs (int): Number of training epochs
        patience (int): Patience for early stopping
        
    Returns:
        dict: Training history
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for drug_data, protein_data, labels in train_loader:
            drug_data = drug_data.to(device)
            protein_data = protein_data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(drug_data, protein_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        val_losses.append(val_metrics['loss'])
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def main(args):
    """
    Main training function.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_splits = load_and_preprocess_data(
        args.data_path,
        protein_model_type=args.protein_model_type,
        test_size=args.test_size,
        val_size=args.val_size,
        fp_radius=args.fp_radius,
        fp_n_bits=args.fp_n_bits
    )
    
    # Create DataLoaders
    train_drug, train_protein, train_labels = data_splits['train']
    val_drug, val_protein, val_labels = data_splits['val']
    
    # Convert to tensors
    train_drug_tensor = torch.FloatTensor(train_drug).to(device)
    train_protein_tensor = torch.FloatTensor(train_protein).to(device)
    train_labels_tensor = torch.LongTensor(train_labels).to(device)
    
    val_drug_tensor = torch.FloatTensor(val_drug).to(device)
    val_protein_tensor = torch.FloatTensor(val_protein).to(device)
    val_labels_tensor = torch.LongTensor(val_labels).to(device)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_drug_tensor, train_protein_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_drug_tensor, val_protein_tensor, val_labels_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print("Creating model...")
    if args.explainable:
        model = create_explainable_dti_model(
            model_type="fingerprint",
            drug_input_dim=args.fp_n_bits,
            protein_input_dim=320,  # Default ESM embedding size
            drug_hidden_dim=args.drug_hidden_dim,
            protein_hidden_dim=args.protein_hidden_dim,
            drug_output_dim=args.drug_output_dim,
            protein_output_dim=args.protein_output_dim,
            combined_dim=args.combined_dim,
            num_classes=2
        )
    else:
        model = create_dti_model(
            model_type="fingerprint",
            drug_input_dim=args.fp_n_bits,
            protein_input_dim=320,  # Default ESM embedding size
            drug_hidden_dim=args.drug_hidden_dim,
            protein_hidden_dim=args.protein_hidden_dim,
            drug_output_dim=args.drug_output_dim,
            protein_output_dim=args.protein_output_dim,
            combined_dim=args.combined_dim,
            num_classes=2
        )
    
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Train model
    print("Starting training...")
    history = train_model(
        model, train_loader, val_loader, device, criterion, optimizer,
        num_epochs=args.epochs, patience=args.patience
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Final evaluation on validation set
    print("Final evaluation on validation set...")
    val_metrics = evaluate_model(model, val_loader, device, criterion)
    
    # Save results
    results = {
        'val_metrics': val_metrics,
        'train_losses': history['train_losses'],
        'val_losses': history['val_losses']
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), 'final_model.pth')
    
    print("Training completed!")
    print(f"Final validation metrics: {val_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Drug-Target Interaction Prediction Model")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the drug-target interaction data CSV')
    parser.add_argument('--protein_model_type', type=str, default='esm', choices=['esm', 'protbert'], help='Protein embedding model type')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for test set')
    parser.add_argument('--val_size', type=float, default=0.1, help='Proportion of data for validation set')
    parser.add_argument('--fp_radius', type=int, default=2, help='Morgan fingerprint radius')
    parser.add_argument('--fp_n_bits', type=int, default=2048, help='Morgan fingerprint number of bits')
    
    # Model arguments
    parser.add_argument('--explainable', action='store_true', help='Use explainable model')
    parser.add_argument('--drug_hidden_dim', type=int, default=512, help='Drug encoder hidden dimension')
    parser.add_argument('--protein_hidden_dim', type=int, default=512, help='Protein encoder hidden dimension')
    parser.add_argument('--drug_output_dim', type=int, default=128, help='Drug encoder output dimension')
    parser.add_argument('--protein_output_dim', type=int, default=128, help='Protein encoder output dimension')
    parser.add_argument('--combined_dim', type=int, default=256, help='Combined features dimension')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    
    args = parser.parse_args()
    main(args)