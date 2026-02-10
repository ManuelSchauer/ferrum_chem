#!/usr/bin/env python3
"""
Extract MACCS key definitions from RDKit and test fingerprints.
"""

from rdkit import Chem
from rdkit.Chem import MACCSkeys

print("RDKit MACCS Keys Testing")
print("=" * 80)

# Test molecules
test_mols = [
    ('CCO', 'Ethanol'),
    ('c1ccccc1', 'Benzene'),
    ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin'),
    ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine'),
    ('c1ccc2ccccc2c1', 'Naphthalene'),
    ('CCCCCC', 'Hexane'),
]

for smiles, name in test_mols:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = MACCSkeys.GenMACCSKeys(mol)
        bits_set = [i for i in range(len(fp)) if fp[i]]
        print(f"\n{name} ({smiles}):")
        print(f"  Bits set ({len(bits_set)}): {bits_set[:20]}{'...' if len(bits_set) > 20 else ''}")
        
        # Show first 20 bits in detail
        print(f"  First 20 bits: ", end='')
        for i in range(min(20, len(fp))):
            print(f"{fp[i]}", end='')
        print()
