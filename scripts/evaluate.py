"""
Evaluation script for Drug-Target Interaction Prediction Model
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_and_preprocess_data
from models.dti_model import create_dti_model
from models.explainable_dti import create_explainable_dti_model
from models.explainable_dti import GradientExplainer


def evaluate_model(model, data_loader, device, criterion):
    """
    Evaluate the model on a dataset.
    
    Args:
        model (nn.Module): Model to evaluate
        data_loader (DataLoader): DataLoader for the dataset
        device (torch.device): Device to use for computation
        criterion (nn.Module): Loss function
        
    Returns:
        dict: Evaluation metrics and predictions
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
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }


def plot_confusion_matrix(cm, class_names=['No Interaction', 'Interaction']):
    """
    Plot confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): Names of classes
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def plot_roc_curve(labels, probabilities):
    """
    Plot ROC curve.
    
    Args:
        labels (list): True labels
        probabilities (list): Predicted probabilities
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()


def main(args):
    """
    Main evaluation function.
    
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
        val_size=0.0,  # No validation set for evaluation
        fp_radius=args.fp_radius,
        fp_n_bits=args.fp_n_bits
    )
    
    # Use test set for evaluation
    test_drug, test_protein, test_labels = data_splits['test']
    
    # Convert to tensors
    test_drug_tensor = torch.FloatTensor(test_drug).to(device)
    test_protein_tensor = torch.FloatTensor(test_protein).to(device)
    test_labels_tensor = torch.LongTensor(test_labels).to(device)
    
    # Create dataset and dataloader
    test_dataset = TensorDataset(test_drug_tensor, test_protein_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
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
    
    # Load trained model weights
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Model file {args.model_path} not found. Using randomly initialized model.")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, criterion)
    
    # Print metrics
    print("Evaluation Results:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  ROC AUC: {results['roc_auc']:.4f}")
    
    # Save results
    results_to_save = {
        'loss': results['loss'],
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'roc_auc': results['roc_auc']
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'])
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Plot ROC curve
    plot_roc_curve(results['labels'], results['probabilities'])
    print("ROC curve saved as 'roc_curve.png'")
    
    # If using explainable model, demonstrate gradient-based explanation
    if args.explainable and len(test_loader) > 0:
        print("Generating gradient-based explanations...")
        explainer = GradientExplainer(model)
        
        # Get a sample batch for explanation
        for drug_data, protein_data, labels in test_loader:
            drug_data = drug_data.to(device)
            protein_data = protein_data.to(device)
            labels = labels.to(device)
            
            # Compute gradients for a few samples
            drug_grads, protein_grads = explainer.compute_gradients(
                drug_data[:5], protein_data[:5], target_class=1
            )
            
            print(f"Drug feature gradients shape: {drug_grads.shape}")
            print(f"Protein feature gradients shape: {protein_grads.shape}")
            
            # Compute saliency maps
            drug_saliency, protein_saliency = explainer.compute_saliency_map(
                drug_data[:5], protein_data[:5], target_class=1
            )
            
            print(f"Drug saliency map shape: {drug_saliency.shape}")
            print(f"Protein saliency map shape: {protein_saliency.shape}")
            break
    
    print("Evaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Drug-Target Interaction Prediction Model")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the drug-target interaction data CSV')
    parser.add_argument('--protein_model_type', type=str, default='esm', choices=['esm', 'protbert'], help='Protein embedding model type')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for test set')
    parser.add_argument('--fp_radius', type=int, default=2, help='Morgan fingerprint radius')
    parser.add_argument('--fp_n_bits', type=int, default=2048, help='Morgan fingerprint number of bits')
    
    # Model arguments
    parser.add_argument('--explainable', action='store_true', help='Use explainable model')
    parser.add_argument('--model_path', type=str, default='final_model.pth', help='Path to trained model weights')
    parser.add_argument('--drug_hidden_dim', type=int, default=512, help='Drug encoder hidden dimension')
    parser.add_argument('--protein_hidden_dim', type=int, default=512, help='Protein encoder hidden dimension')
    parser.add_argument('--drug_output_dim', type=int, default=128, help='Drug encoder output dimension')
    parser.add_argument('--protein_output_dim', type=int, default=128, help='Protein encoder output dimension')
    parser.add_argument('--combined_dim', type=int, default=256, help='Combined features dimension')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    main(args)