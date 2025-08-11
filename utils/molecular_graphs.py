"""
Molecular graph representation utilities for drug-target interaction prediction.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.nn.functional as F
from torch_geometric.data import Data


def atom_to_feature_vector(atom: Chem.Atom) -> List[int]:
    """
    Converts an RDKit atom to a feature vector.
    
    Args:
        atom (Chem.Atom): RDKit atom object
        
    Returns:
        List[int]: Feature vector for the atom
    """
    # Atomic number
    atomic_num = atom.GetAtomicNum()
    
    # One-hot encoding for common elements
    atomic_features = [
        atomic_num == 1,   # H
        atomic_num == 6,   # C
        atomic_num == 7,   # N
        atomic_num == 8,   # O
        atomic_num == 9,   # F
        atomic_num == 15,  # P
        atomic_num == 16,  # S
        atomic_num == 17,  # Cl
        atomic_num == 35,  # Br
        atomic_num == 53,  # I
    ]
    
    # If not in common elements, mark as "other"
    atomic_features.append(atomic_num not in [1, 6, 7, 8, 9, 15, 16, 17, 35, 53])
    
    # Additional atom features
    hybridization = [
        atom.GetHybridization() == Chem.HybridizationType.SP,
        atom.GetHybridization() == Chem.HybridizationType.SP2,
        atom.GetHybridization() == Chem.HybridizationType.SP3,
        atom.GetHybridization() == Chem.HybridizationType.SP3D,
        atom.GetHybridization() == Chem.HybridizationType.SP3D2,
    ]
    
    # Degree features
    degree = atom.GetDegree()
    degree_one_hot = [degree == i for i in range(5)]
    degree_one_hot.append(degree > 4)
    
    # Other features
    is_aromatic = atom.GetIsAromatic()
    is_in_ring = atom.IsInRing()
    formal_charge = atom.GetFormalCharge()
    num_explicit_hs = atom.GetNumExplicitHs()
    
    # Combine all features
    feature_vector = atomic_features + hybridization + degree_one_hot + [
        is_aromatic,
        is_in_ring,
        formal_charge,
        num_explicit_hs
    ]
    
    return [int(x) for x in feature_vector]


def bond_to_feature_vector(bond: Chem.Bond) -> List[int]:
    """
    Converts an RDKit bond to a feature vector.
    
    Args:
        bond (Chem.Bond): RDKit bond object
        
    Returns:
        List[int]: Feature vector for the bond
    """
    # Bond type
    bond_type = bond.GetBondType()
    bond_features = [
        bond_type == Chem.BondType.SINGLE,
        bond_type == Chem.BondType.DOUBLE,
        bond_type == Chem.BondType.TRIPLE,
        bond_type == Chem.BondType.AROMATIC,
    ]
    
    # Additional bond features
    is_conjugated = bond.GetIsConjugated()
    is_in_ring = bond.IsInRing()
    
    # Stereochemistry
    stereo = bond.GetStereo()
    stereo_features = [
        stereo == Chem.BondStereo.STEREONONE,
        stereo == Chem.BondStereo.STEREOANY,
        stereo == Chem.BondStereo.STEREOZ,
        stereo == Chem.BondStereo.STEREOE,
    ]
    
    # Combine all features
    feature_vector = bond_features + [is_conjugated, is_in_ring] + stereo_features
    
    return [int(x) for x in feature_vector]


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Converts a SMILES string to a molecular graph representation.
    
    Args:
        smiles (str): SMILES representation of a molecule
        
    Returns:
        Data or None: PyTorch Geometric Data object, or None if invalid SMILES
    """
    try:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates (optional, for better embeddings)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        
        # Get atoms and bonds
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        
        # Create node features
        node_features = []
        for atom in atoms:
            node_features.append(atom_to_feature_vector(atom))
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Create edge indices and edge features
        edge_indices = []
        edge_features = []
        
        for bond in bonds:
            start_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edge_indices.append([start_atom, end_atom])
            edge_indices.append([end_atom, start_atom])
            
            # Edge features (same for both directions)
            bond_feature = bond_to_feature_vector(bond)
            edge_features.append(bond_feature)
            edge_features.append(bond_feature)
        
        # Convert to tensors
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # Handle molecules with no bonds (single atoms)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 10), dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles
        )
        
        return graph_data
    
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None


def batch_graphs(graph_list: List[Data]) -> Data:
    """
    Batches a list of graph data objects into a single batch.
    
    Args:
        graph_list (List[Data]): List of PyTorch Geometric Data objects
        
    Returns:
        Data: Batched graph data
    """
    from torch_geometric.data import Batch
    
    # Filter out None values
    valid_graphs = [g for g in graph_list if g is not None]
    
    if not valid_graphs:
        return None
    
    # Create batch
    batch = Batch.from_data_list(valid_graphs)
    return batch


def get_graph_embeddings(graph_data: Data, model: torch.nn.Module) -> torch.Tensor:
    """
    Generate embeddings for a molecular graph using a GNN model.
    
    Args:
        graph_data (Data): PyTorch Geometric Data object
        model (torch.nn.Module): GNN model for generating embeddings
        
    Returns:
        torch.Tensor: Graph embedding
    """
    with torch.no_grad():
        # Ensure model is in evaluation mode
        model.eval()
        
        # Generate embedding
        embedding = model(graph_data)
        return embedding


# Example usage
if __name__ == "__main__":
    # Example SMILES
    smiles_list = [
        "CCO",           # Ethanol
        "CC(=O)O",       # Acetic acid
        "c1ccccc1",      # Benzene
        "CCN(CC)CC"      # Triethylamine
    ]
    
    # Convert to graphs
    graphs = []
    for smiles in smiles_list:
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
            print(f"SMILES: {smiles}")
            print(f"  Nodes: {graph.x.shape}")
            print(f"  Edges: {graph.edge_index.shape}")
    
    print(f"Successfully converted {len(graphs)} molecules to graphs")