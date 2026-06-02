#!/usr/bin/env python3
"""
Validate FerrumChem MACCS keys against RDKit reference implementation.
"""

from rdkit import Chem
from rdkit.Chem import MACCSkeys
import sys

def test_molecule(smiles, name):
    """Test a single molecule and print which bits are set."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(f"Error parsing {name}: {smiles}")
        return None
        
    fp = MACCSkeys.GenMACCSKeys(mol)
    bits_set = [i for i in range(len(fp)) if fp[i]]
    
    print(f"\n{name} ({smiles}):")
    print(f"  Total bits set: {len(bits_set)} / 167")
    print(f"  Bits (1-166): {[b for b in bits_set if b > 0]}")
    
    # Show first 20 bits in detail for debugging
    print(f"  First 20 bits: ", end='')
    for i in range(1, min(21, len(fp))):
        print(f"{fp[i]}", end='')
    print()
    
    return fp

def main():
    """
    Implements main.
    """
    print("RDKit MACCS Keys Validation")
    print("=" * 80)
    
    test_mols = [
        ('CCO', 'Ethanol'),
        ('c1ccccc1', 'Benzene'),
        ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin'),
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine'),
        ('c1ccc2ccccc2c1', 'Naphthalene'),
        ('CCCCCC', 'Hexane'),
        ('c1ccncc1', 'Pyridine'),
    ]
    
    results = {}
    for smiles, name in test_mols:
        fp = test_molecule(smiles, name)
        if fp:
            results[name] = fp
    
    print("\n" + "=" * 80)
    print("Validation complete. Use these bit patterns to update Rust tests.")
    
    # Generate Rust test assertions
    print("\n// Rust test assertions:")
    for name in results:
        fp = results[name]
        bits = [i for i in range(1, len(fp)) if fp[i]]
        print(f"\n// {name}: bits {bits[:10]}{'...' if len(bits) > 10 else ''}")

if __name__ == '__main__':
    main()
