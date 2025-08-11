"""
Data loading and preprocessing utilities for drug-target interaction prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from .protein_embeddings import ProteinEmbedder


def smiles_to_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """
    Convert a SMILES string to a Morgan fingerprint.
    
    Args:
        smiles (str): SMILES representation of a molecule
        radius (int): Radius for Morgan fingerprint (default: 2)
        n_bits (int): Number of bits in the fingerprint (default: 2048)
        
    Returns:
        np.ndarray or None: Morgan fingerprint as numpy array, or None if invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate Morgan fingerprint
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        
        # Convert to numpy array
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fingerprint, arr)
        return arr
    except Exception:
        return None


def load_drug_target_data(file_path: str) -> pd.DataFrame:
    """
    Load drug-target interaction data from a CSV file.
    
    Expected columns:
    - drug_id: Unique identifier for the drug
    - target_id: Unique identifier for the target protein
    - drug_smiles: SMILES representation of the drug
    - target_sequence: Amino acid sequence of the target protein
    - interaction: Binary label (1 for interaction, 0 for no interaction)
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    df = pd.read_csv(file_path)
    required_columns = ['drug_id', 'target_id', 'drug_smiles', 'target_sequence', 'interaction']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    return df


def preprocess_drug_target_data(
    df: pd.DataFrame,
    protein_model_type: str = "esm",
    fp_radius: int = 2,
    fp_n_bits: int = 2048
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess drug-target data by generating features.
    
    Args:
        df (pd.DataFrame): Input dataframe with drug-target pairs
        protein_model_type (str): Type of protein embedding model ("esm" or "protbert")
        fp_radius (int): Radius for Morgan fingerprint
        fp_n_bits (int): Number of bits for Morgan fingerprint
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Drug features (fingerprints)
            - Protein features (embeddings)
            - Interaction labels
    """
    # Initialize protein embedder
    protein_embedder = ProteinEmbedder(model_type=protein_model_type)
    
    # Generate drug features (Morgan fingerprints)
    print("Generating drug fingerprints...")
    drug_features = []
    valid_indices = []
    
    for i, smiles in enumerate(df['drug_smiles']):
        fp = smiles_to_morgan_fingerprint(smiles, fp_radius, fp_n_bits)
        if fp is not None:
            drug_features.append(fp)
            valid_indices.append(i)
        else:
            print(f"Warning: Invalid SMILES at index {i}: {smiles}")
    
    # Filter dataframe to only include valid entries
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    
    # Generate protein features (embeddings)
    print("Generating protein embeddings...")
    protein_features = []
    for seq in df_filtered['target_sequence']:
        emb = protein_embedder.get_embedding(seq)
        protein_features.append(emb)
    
    # Convert to numpy arrays
    drug_features = np.array(drug_features)
    protein_features = np.array(protein_features)
    labels = df_filtered['interaction'].values
    
    return drug_features, protein_features, labels


def create_dataset_splits(
    drug_features: np.ndarray,
    protein_features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create train/validation/test splits for the dataset.
    
    Args:
        drug_features (np.ndarray): Drug feature matrix
        protein_features (np.ndarray): Protein feature matrix
        labels (np.ndarray): Interaction labels
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of data for validation set (from train+val)
        random_state (int): Random seed for reproducibility
        
    Returns:
        Dict: Dictionary with train/val/test splits
    """
    # First split: train+val vs test
    X_drug_temp, X_drug_test, X_protein_temp, X_protein_test, y_temp, y_test = train_test_split(
        drug_features, protein_features, labels, 
        test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: train vs val
    val_proportion = val_size / (1 - test_size)  # Adjust for the reduced dataset
    X_drug_train, X_drug_val, X_protein_train, X_protein_val, y_train, y_val = train_test_split(
        X_drug_temp, X_protein_temp, y_temp,
        test_size=val_proportion, random_state=random_state, stratify=y_temp
    )
    
    return {
        'train': (X_drug_train, X_protein_train, y_train),
        'val': (X_drug_val, X_protein_val, y_val),
        'test': (X_drug_test, X_protein_test, y_test)
    }


def load_and_preprocess_data(
    file_path: str,
    protein_model_type: str = "esm",
    test_size: float = 0.2,
    val_size: float = 0.1,
    fp_radius: int = 2,
    fp_n_bits: int = 2048,
    random_state: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load and preprocess drug-target interaction data, creating dataset splits.
    
    Args:
        file_path (str): Path to the CSV file with drug-target data
        protein_model_type (str): Type of protein embedding model ("esm" or "protbert")
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of data for validation set
        fp_radius (int): Radius for Morgan fingerprint
        fp_n_bits (int): Number of bits for Morgan fingerprint
        random_state (int): Random seed for reproducibility
        
    Returns:
        Dict: Dictionary with train/val/test splits
    """
    # Load data
    df = load_drug_target_data(file_path)
    
    # Preprocess data
    drug_features, protein_features, labels = preprocess_drug_target_data(
        df, protein_model_type, fp_radius, fp_n_bits
    )
    
    # Create dataset splits
    splits = create_dataset_splits(
        drug_features, protein_features, labels,
        test_size, val_size, random_state
    )
    
    return splits


# Example usage
if __name__ == "__main__":
    # This is just an example - in practice, you would have a real dataset
    print("Data utilities loaded successfully.")
    print("Use load_and_preprocess_data() to load and preprocess your drug-target interaction data.")