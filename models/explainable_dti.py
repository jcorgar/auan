"""
Explainable Drug-Target Interaction Prediction Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from torch_geometric.data import Data
from .dti_model import DTIModel, DTIModelWithGraph


class AttentionModule(nn.Module):
    """
    Attention mechanism for combining drug and protein features.
    """
    
    def __init__(self, feature_dim: int):
        """
        Initialize the AttentionModule.
        
        Args:
            feature_dim (int): Dimension of input features
        """
        super(AttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.attention_weights = nn.Linear(feature_dim, 1)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to features.
        
        Args:
            features (torch.Tensor): Input features (batch_size, num_features, feature_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Weighted features and attention weights
        """
        # Calculate attention scores
        attention_scores = self.attention_weights(features)  # (batch_size, num_features, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, num_features, 1)
        
        # Apply attention weights
        weighted_features = features * attention_weights  # (batch_size, num_features, feature_dim)
        aggregated_features = torch.sum(weighted_features, dim=1)  # (batch_size, feature_dim)
        
        return aggregated_features, attention_weights.squeeze(-1)


class ExplainableDTIModel(nn.Module):
    """
    Explainable Drug-Target Interaction Prediction Model with attention mechanisms.
    """
    
    def __init__(
        self,
        drug_input_dim: int = 2048,
        protein_input_dim: int = 320,
        drug_hidden_dim: int = 512,
        protein_hidden_dim: int = 512,
        drug_output_dim: int = 128,
        protein_output_dim: int = 128,
        combined_dim: int = 256,
        num_classes: int = 2
    ):
        """
        Initialize the ExplainableDTIModel.
        
        Args:
            drug_input_dim (int): Input dimension for drugs
            protein_input_dim (int): Input dimension for proteins
            drug_hidden_dim (int): Hidden dimension for drug encoder
            protein_hidden_dim (int): Hidden dimension for protein encoder
            drug_output_dim (int): Output dimension for drug encoder
            protein_output_dim (int): Output dimension for protein encoder
            combined_dim (int): Dimension for combined features
            num_classes (int): Number of output classes
        """
        super(ExplainableDTIModel, self).__init__()
        
        # Encoders
        self.drug_encoder = nn.Linear(drug_input_dim, drug_output_dim)
        self.protein_encoder = nn.Linear(protein_input_dim, protein_output_dim)
        
        # Attention modules
        self.drug_attention = AttentionModule(drug_output_dim)
        self.protein_attention = AttentionModule(protein_output_dim)
        
        # Combined layers
        self.combined_dim = drug_output_dim + protein_output_dim
        self.fc1 = nn.Linear(self.combined_dim, combined_dim)
        self.fc2 = nn.Linear(combined_dim, combined_dim // 2)
        self.fc3 = nn.Linear(combined_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        
        # For storing attention weights for explanation
        self.drug_attention_weights = None
        self.protein_attention_weights = None
        
    def forward(
        self,
        drug_features: torch.Tensor,
        protein_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for explainable DTI prediction.
        
        Args:
            drug_features (torch.Tensor): Drug features
            protein_features (torch.Tensor): Protein embeddings
            
        Returns:
            torch.Tensor: Prediction logits
        """
        # Encode features
        drug_encoded = F.relu(self.drug_encoder(drug_features))
        protein_encoded = F.relu(self.protein_encoder(protein_features))
        
        # Add dimension for attention (assuming each sample is a single feature vector)
        # In a more complex implementation, we might have multiple features per sample
        drug_expanded = drug_encoded.unsqueeze(1)  # (batch, 1, drug_output_dim)
        protein_expanded = protein_encoded.unsqueeze(1)  # (batch, 1, protein_output_dim)
        
        # Apply attention
        drug_weighted, self.drug_attention_weights = self.drug_attention(drug_expanded)
        protein_weighted, self.protein_attention_weights = self.protein_attention(protein_expanded)
        
        # Combine features
        combined = torch.cat([drug_weighted, protein_weighted], dim=1)
        
        # Classification layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_attention_weights(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get the attention weights from the last forward pass.
        
        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: Drug and protein attention weights
        """
        return self.drug_attention_weights, self.protein_attention_weights


class ExplainableDTIModelWithGraph(nn.Module):
    """
    Explainable Drug-Target Interaction Prediction Model with Graph Neural Network and attention.
    """
    
    def __init__(
        self,
        drug_node_dim: int = 39,
        protein_input_dim: int = 320,
        drug_hidden_dim: int = 128,
        protein_hidden_dim: int = 512,
        drug_output_dim: int = 128,
        protein_output_dim: int = 128,
        combined_dim: int = 256,
        num_classes: int = 2
    ):
        """
        Initialize the ExplainableDTIModelWithGraph.
        
        Args:
            drug_node_dim (int): Node feature dimension for drug graphs
            protein_input_dim (int): Input dimension for proteins
            drug_hidden_dim (int): Hidden dimension for drug GNN
            protein_hidden_dim (int): Hidden dimension for protein encoder
            drug_output_dim (int): Output dimension for drug encoder
            protein_output_dim (int): Output dimension for protein encoder
            combined_dim (int): Dimension for combined features
            num_classes (int): Number of output classes
        """
        super(ExplainableDTIModelWithGraph, self).__init__()
        
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        # Drug GNN encoder
        self.drug_conv1 = GCNConv(drug_node_dim, drug_hidden_dim)
        self.drug_conv2 = GCNConv(drug_hidden_dim, drug_output_dim)
        self.drug_pool = global_mean_pool
        
        # Protein encoder
        self.protein_encoder = nn.Linear(protein_input_dim, protein_output_dim)
        
        # Attention modules
        self.drug_attention = AttentionModule(drug_output_dim)
        self.protein_attention = AttentionModule(protein_output_dim)
        
        # Combined layers
        self.combined_dim = drug_output_dim + protein_output_dim
        self.fc1 = nn.Linear(self.combined_dim, combined_dim)
        self.fc2 = nn.Linear(combined_dim, combined_dim // 2)
        self.fc3 = nn.Linear(combined_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        
        # For storing attention weights for explanation
        self.drug_attention_weights = None
        self.protein_attention_weights = None
        
    def forward(
        self,
        drug_graphs: Data,
        protein_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for explainable DTI prediction with graph-based drugs.
        
        Args:
            drug_graphs (Data): Batched drug graph data
            protein_features (torch.Tensor): Protein embeddings
            
        Returns:
            torch.Tensor: Prediction logits
        """
        # Drug GNN encoding
        drug_x = F.relu(self.drug_conv1(drug_graphs.x, drug_graphs.edge_index))
        drug_x = F.relu(self.drug_conv2(drug_x, drug_graphs.edge_index))
        
        # Global pooling to get graph-level representations
        drug_embeddings = self.drug_pool(drug_x, drug_graphs.batch)
        
        # Protein encoding
        protein_encoded = F.relu(self.protein_encoder(protein_features))
        
        # Add dimension for attention
        drug_expanded = drug_embeddings.unsqueeze(1)  # (batch, 1, drug_output_dim)
        protein_expanded = protein_encoded.unsqueeze(1)  # (batch, 1, protein_output_dim)
        
        # Apply attention
        drug_weighted, self.drug_attention_weights = self.drug_attention(drug_expanded)
        protein_weighted, self.protein_attention_weights = self.protein_attention(protein_expanded)
        
        # Combine features
        combined = torch.cat([drug_weighted, protein_weighted], dim=1)
        
        # Classification layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_attention_weights(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get the attention weights from the last forward pass.
        
        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: Drug and protein attention weights
        """
        return self.drug_attention_weights, self.protein_attention_weights


class GradientExplainer:
    """
    Gradient-based explainer for DTI models.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize the GradientExplainer.
        
        Args:
            model (nn.Module): DTI model to explain
        """
        self.model = model
        self.model.eval()
        
    def compute_gradients(
        self,
        drug_features: torch.Tensor,
        protein_features: torch.Tensor,
        target_class: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients of the model output with respect to input features.
        
        Args:
            drug_features (torch.Tensor): Drug features
            protein_features (torch.Tensor): Protein features
            target_class (int): Target class for gradient computation
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gradients for drug and protein features
        """
        # Enable gradients for input features
        drug_features.requires_grad_(True)
        protein_features.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(drug_features, protein_features)
        
        # Compute gradients
        self.model.zero_grad()
        output = outputs[:, target_class].sum()
        output.backward()
        
        # Get gradients
        drug_gradients = drug_features.grad.data.clone()
        protein_gradients = protein_features.grad.data.clone()
        
        # Disable gradients
        drug_features.requires_grad_(False)
        protein_features.requires_grad_(False)
        
        return drug_gradients, protein_gradients
    
    def compute_saliency_map(
        self,
        drug_features: torch.Tensor,
        protein_features: torch.Tensor,
        target_class: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute saliency maps for input features.
        
        Args:
            drug_features (torch.Tensor): Drug features
            protein_features (torch.Tensor): Protein features
            target_class (int): Target class for saliency computation
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Saliency maps for drug and protein features
        """
        drug_gradients, protein_gradients = self.compute_gradients(
            drug_features, protein_features, target_class
        )
        
        # Compute saliency as absolute gradients
        drug_saliency = torch.abs(drug_gradients).mean(dim=0).cpu().numpy()
        protein_saliency = torch.abs(protein_gradients).mean(dim=0).cpu().numpy()
        
        return drug_saliency, protein_saliency


# Utility functions for explainable model creation
def create_explainable_dti_model(
    model_type: str = "fingerprint",
    drug_input_dim: int = 2048,
    protein_input_dim: int = 320,
    **kwargs
) -> nn.Module:
    """
    Create an explainable DTI model of the specified type.
    
    Args:
        model_type (str): Type of model ("fingerprint" or "graph")
        drug_input_dim (int): Input dimension for drugs
        protein_input_dim (int): Input dimension for proteins
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: Explainable DTI model
    """
    if model_type == "fingerprint":
        return ExplainableDTIModel(
            drug_input_dim=drug_input_dim,
            protein_input_dim=protein_input_dim,
            **kwargs
        )
    elif model_type == "graph":
        return ExplainableDTIModelWithGraph(
            drug_node_dim=drug_input_dim,
            protein_input_dim=protein_input_dim,
            **kwargs
        )
    else:
        raise ValueError("model_type must be 'fingerprint' or 'graph'")


# Example usage
if __name__ == "__main__":
    # Example with fingerprint-based explainable model
    model = create_explainable_dti_model("fingerprint", drug_input_dim=2048, protein_input_dim=320)
    print(f"Fingerprint-based explainable model: {model}")
    
    # Example with graph-based explainable model
    model = create_explainable_dti_model("graph", drug_input_dim=39, protein_input_dim=320)
    print(f"Graph-based explainable model: {model}")