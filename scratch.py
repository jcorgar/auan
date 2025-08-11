from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """
    Convert a SMILES string to a Morgan fingerprint.
    
    Parameters:
    smiles (str): The SMILES string representation of a molecule
    radius (int): The radius of the Morgan fingerprint (default: 2)
    n_bits (int): The number of bits in the fingerprint (default: 2048)
    
    Returns:
    ExplicitBitVect or None: The Morgan fingerprint as an RDKit ExplicitBitVect, or None if SMILES is invalid
    """
    # Convert SMILES to molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    # Check if the molecule is valid
    if mol is None:
        return None
    
    # Generate Morgan fingerprint using the newer API
    fingerprint = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits).GetFingerprint(mol)
    
    return fingerprint

# Example usage:
# fingerprint = smiles_to_morgan_fingerprint("CCO")  # Ethanol