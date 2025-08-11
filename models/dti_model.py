"""
Drug-Target Interaction Prediction Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class DrugEncoder(nn.Module):
    """
    Encoder for drug features using Morgan fingerprints or molecular graphs.
    """
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 128):
        """
        Initialize the DrugEncoder.
        
        Args:
            input_dim (int): Input dimension (e.g., 2048 for Morgan fingerprints)
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output embedding dimension
        """
        super(DrugEncoder, self).__init__()
        self.use_fingerprint = input_dim == 2048  # Assume fingerprint if 2048 dim
        
        if self.use_fingerprint:
            # For Morgan fingerprints
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.2)
        else:
            # For molecular graphs (GNN)
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.2)
            
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for drug encoding.
        
        Args:
            x (torch.Tensor): Input features
            batch (torch.Tensor, optional): Batch indices for graph data
            
        Returns:
            torch.Tensor: Drug embeddings
        """
        if self.use_fingerprint:
            # For Morgan fingerprints
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            return x
        else:
            # For molecular graphs
            # x is expected to be a Data object with x, edge_index, and batch
            if isinstance(x, Data):
                batch_indices = x.batch if hasattr(x, 'batch') else batch
                x = F.relu(self.conv1(x.x, x.edge_index))
                x = self.dropout(x)
                x = F.relu(self.conv2(x, x.edge_index))
                # Global pooling to get graph-level representation
                x = global_mean_pool(x, batch_indices)
                return x
            else:
                raise ValueError("For graph input, x should be a Data object")


class ProteinEncoder(nn.Module):
    """
    Encoder for protein sequence embeddings.
    """
    
    def __init__(self, input_dim: int = 320, hidden_dim: int = 512, output_dim: int = 128):
        """
        Initialize the ProteinEncoder.
        
        Args:
            input_dim (int): Input dimension (e.g., 320 for ESM embeddings)
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output embedding dimension
        """
        super(ProteinEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for protein encoding.
        
        Args:
            x (torch.Tensor): Protein embeddings
            
        Returns:
            torch.Tensor: Processed protein embeddings
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


class DTIModel(nn.Module):
    """
    Drug-Target Interaction Prediction Model.
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
        Initialize the DTIModel.
        
        Args:
            drug_input_dim (int): Input dimension for drugs
            protein_input_dim (int): Input dimension for proteins
            drug_hidden_dim (int): Hidden dimension for drug encoder
            protein_hidden_dim (int): Hidden dimension for protein encoder
            drug_output_dim (int): Output dimension for drug encoder
            protein_output_dim (int): Output dimension for protein encoder
            combined_dim (int): Dimension for combined features
            num_classes (int): Number of output classes (default: 2 for binary classification)
        """
        super(DTIModel, self).__init__()
        
        # Encoders
        self.drug_encoder = DrugEncoder(drug_input_dim, drug_hidden_dim, drug_output_dim)
        self.protein_encoder = ProteinEncoder(protein_input_dim, protein_hidden_dim, protein_output_dim)
        
        # Combined layers
        self.combined_dim = drug_output_dim + protein_output_dim
        self.fc1 = nn.Linear(self.combined_dim, combined_dim)
        self.fc2 = nn.Linear(combined_dim, combined_dim // 2)
        self.fc3 = nn.Linear(combined_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(
        self,
        drug_features: torch.Tensor,
        protein_features: torch.Tensor,
        drug_batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for DTI prediction.
        
        Args:
            drug_features (torch.Tensor): Drug features (fingerprints or graph data)
            protein_features (torch.Tensor): Protein embeddings
            drug_batch (torch.Tensor, optional): Batch indices for graph data
            
        Returns:
            torch.Tensor: Prediction logits
        """
        # Encode drug and protein features
        drug_embeddings = self.drug_encoder(drug_features, drug_batch)
        protein_embeddings = self.protein_encoder(protein_features)
        
        # Combine features
        combined = torch.cat([drug_embeddings, protein_embeddings], dim=1)
        
        # Classification layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class DTIModelWithGraph(nn.Module):
    """
    Drug-Target Interaction Prediction Model with Graph Neural Network for drugs.
    """
    
    def __init__(
        self,
        drug_node_dim: int = 39,  # Default node feature dimension from molecular_graphs.py
        protein_input_dim: int = 320,
        drug_hidden_dim: int = 128,
        protein_hidden_dim: int = 512,
        drug_output_dim: int = 128,
        protein_output_dim: int = 128,
        combined_dim: int = 256,
        num_classes: int = 2
    ):
        """
        Initialize the DTIModelWithGraph.
        
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
        super(DTIModelWithGraph, self).__init__()
        
        # Encoders
        self.drug_encoder = DrugEncoder(drug_node_dim, drug_hidden_dim, drug_output_dim)
        self.protein_encoder = ProteinEncoder(protein_input_dim, protein_hidden_dim, protein_output_dim)
        
        # Combined layers
        self.combined_dim = drug_output_dim + protein_output_dim
        self.fc1 = nn.Linear(self.combined_dim, combined_dim)
        self.fc2 = nn.Linear(combined_dim, combined_dim // 2)
        self.fc3 = nn.Linear(combined_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(
        self,
        drug_graphs: Data,
        protein_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for DTI prediction with graph-based drugs.
        
        Args:
            drug_graphs (Data): Batched drug graph data
            protein_features (torch.Tensor): Protein embeddings
            
        Returns:
            torch.Tensor: Prediction logits
        """
        # Encode drug and protein features
        drug_embeddings = self.drug_encoder(drug_graphs)
        protein_embeddings = self.protein_encoder(protein_features)
        
        # Combine features
        combined = torch.cat([drug_embeddings, protein_embeddings], dim=1)
        
        # Classification layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# Utility functions for model creation
def create_dti_model(
    model_type: str = "fingerprint",
    drug_input_dim: int = 2048,
    protein_input_dim: int = 320,
    **kwargs
) -> nn.Module:
    """
    Create a DTI model of the specified type.
    
    Args:
        model_type (str): Type of model ("fingerprint" or "graph")
        drug_input_dim (int): Input dimension for drugs
        protein_input_dim (int): Input dimension for proteins
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: DTI model
    """
    if model_type == "fingerprint":
        return DTIModel(
            drug_input_dim=drug_input_dim,
            protein_input_dim=protein_input_dim,
            **kwargs
        )
    elif model_type == "graph":
        return DTIModelWithGraph(
            drug_node_dim=drug_input_dim,
            protein_input_dim=protein_input_dim,
            **kwargs
        )
    else:
        raise ValueError("model_type must be 'fingerprint' or 'graph'")


# Example usage
if __name__ == "__main__":
    # Example with fingerprint-based model
    model = create_dti_model("fingerprint", drug_input_dim=2048, protein_input_dim=320)
    print(f"Fingerprint-based model: {model}")
    
    # Example with graph-based model
    model = create_dti_model("graph", drug_input_dim=39, protein_input_dim=320)
    print(f"Graph-based model: {model}")