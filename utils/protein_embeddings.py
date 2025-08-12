"""
Protein sequence embedding utilities using ESM and ProtBERT models.
"""

import torch
import numpy as np
from typing import List, Union, Optional
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import re


class ProteinEmbedder:
    """
    A class for generating protein sequence embeddings using ESM or ProtBERT models.
    """
    
    def __init__(self, model_type: str = "esm", model_name: str = None):
        """
        Initialize the ProteinEmbedder.
        
        Args:
            model_type (str): Type of model to use ("esm" or "protbert")
            model_name (str): Specific model name to use (optional)
        """
        self.model_type = model_type.lower()
        
        if model_name is None:
            if self.model_type == "esm":
                self.model_name = "facebook/esm2_t6_8M_UR50D"
            elif self.model_type == "protbert":
                self.model_name = "Rostlab/prot_bert"
            else:
                raise ValueError("model_type must be either 'esm' or 'protbert'")
        else:
            self.model_name = model_name
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, output_hidden_states=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
    def _preprocess_sequence(self, sequence: str) -> str:
        """
        Preprocess protein sequence for the specific model.
        
        Args:
            sequence (str): Raw protein sequence
            
        Returns:
            str: Preprocessed sequence
        """
        # Remove any non-amino acid characters and convert to uppercase
        sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
        
        if self.model_type == "protbert":
            # ProtBERT expects spaces between amino acids
            sequence = " ".join(list(sequence))
            
        return sequence
    
    def get_embedding(self, sequence: str, layer: int = -1) -> np.ndarray:
        """
        Generate embedding for a single protein sequence.
        
        Args:
            sequence (str): Protein sequence
            layer (int): Which layer to use for embeddings (-1 for last layer)
            
        Returns:
            np.ndarray: Protein embedding vector
        """
        # Preprocess sequence
        processed_seq = self._preprocess_sequence(sequence)
        
        # Tokenize sequence
        inputs = self.tokenizer(
            processed_seq,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use hidden states from specified layer
            embeddings = outputs.hidden_states[layer]
            
            # For ESM, we typically use the first token ([CLS] equivalent) or average
            # For ProtBERT, we also use the first token or average
            # Here we'll use the average of all tokens (excluding special tokens)
            if self.model_type == "esm":
                # ESM doesn't use special tokens in the same way, so we average all
                embedding = torch.mean(embeddings[0], dim=0)
            else:  # ProtBERT
                # Exclude first and last tokens ([CLS] and [SEP])
                embedding = torch.mean(embeddings[0][1:-1], dim=0)
                
            return embedding.numpy()
    
    def get_embeddings(self, sequences: List[str], layer: int = -1) -> np.ndarray:
        """
        Generate embeddings for multiple protein sequences.
        
        Args:
            sequences (List[str]): List of protein sequences
            layer (int): Which layer to use for embeddings (-1 for last layer)
            
        Returns:
            np.ndarray: Array of protein embedding vectors
        """
        embeddings = []
        for seq in sequences:
            emb = self.get_embedding(seq, layer)
            embeddings.append(emb)
            
        return np.array(embeddings)
    
    def get_sequence_embeddings(self, sequence: str) -> np.ndarray:
        """
        Generate per-residue embeddings for a protein sequence.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            np.ndarray: Per-residue embeddings (sequence_length x embedding_dim)
        """
        # Preprocess sequence
        processed_seq = self._preprocess_sequence(sequence)
        
        # Tokenize sequence
        inputs = self.tokenizer(
            processed_seq,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use last hidden state
            embeddings = outputs.hidden_states[-1]
            
            # For per-residue embeddings, we return all tokens except special ones
            if self.model_type == "esm":
                # Return all tokens for ESM
                return embeddings[0].numpy()
            else:  # ProtBERT
                # Exclude first and last tokens ([CLS] and [SEP])
                return embeddings[0][1:-1].numpy()


def load_sequences_from_fasta(fasta_file: str) -> List[str]:
    """
    Load protein sequences from a FASTA file.
    
    Args:
        fasta_file (str): Path to FASTA file
        
    Returns:
        List[str]: List of protein sequences
    """
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    return sequences


# Example usage
if __name__ == "__main__":
    # Example with ESM
    esm_embedder = ProteinEmbedder("esm")
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    embedding = esm_embedder.get_embedding(sequence)
    print(f"ESM embedding shape: {embedding.shape}")
    
    # Example with ProtBERT
    protbert_embedder = ProteinEmbedder("protbert")
    embedding = protbert_embedder.get_embedding(sequence)
    print(f"ProtBERT embedding shape: {embedding.shape}")