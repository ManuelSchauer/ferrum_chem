#!/usr/bin/env python3
"""
Validate TPSA implementation against RDKit reference values.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors

# Test molecules from our TPSA tests
test_molecules = [
    ("Butane", "CCCC", 0.0),
    ("Ethanol", "CCO", 20.23),
    ("Dimethyl ether", "COC", 9.23),
    ("Acetone", "CC(=O)C", 17.07),
    ("Acetic acid", "CC(=O)O", 37.30),
    ("Methyl acetate", "CC(=O)OC", 26.30),
    ("Methylamine", "CN", 26.02),
    ("Dimethylamine", "CNC", 12.03),
    ("Trimethylamine", "CN(C)C", 3.24),
    ("Pyridine", "c1cccnc1", 12.89),
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", 63.6),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 58.4),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", 37.3),
    ("Acetamide", "CC(=O)N", 43.09),
    ("Benzene", "c1ccccc1", 0.0),
]

print("SMILES\tName\tExpected\tRDKit\tDiff")
print("-" * 80)

for name, smiles, expected in test_molecules:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"{smiles}\t{name}\tERROR: Invalid SMILES")
        continue
    
    rdkit_tpsa = Descriptors.TPSA(mol)
    diff = abs(rdkit_tpsa - expected)
    
    status = "✓" if diff < 1.0 else "✗"
    print(f"{smiles}\t{name}\t{expected:.2f}\t{rdkit_tpsa:.2f}\t{diff:.2f} {status}")

print("\nAdditional validation molecules:")
print("-" * 80)

# Additional complex molecules
additional = [
    ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"),
    ("Metformin", "CN(C)C(=N)NC(=N)N"),
    ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("Propanol", "CCCO"),
    ("Diethyl ether", "CCOCC"),
]

for name, smiles in additional:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"{smiles}\t{name}\tERROR: Invalid SMILES")
        continue
    
    rdkit_tpsa = Descriptors.TPSA(mol)
    print(f"{name}\t{rdkit_tpsa:.2f} Ų")
