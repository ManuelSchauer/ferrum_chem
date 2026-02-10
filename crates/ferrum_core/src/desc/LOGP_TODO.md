# LogP Implementation TODO

## Current Status
The initial LogP implementation used H-count based classification, but this doesn't match RDKit's approach.

## Issue
RDKit uses SMARTS pattern matching for Wildman-Crippen atom type assignment, not simple H-count rules.

### Key Differences:
- RDKit: `[CH4]`, `[CH3]C`, `[CH2](C)C` all get contribution 0.1441 (type C1)
- Our implementation: Used different types for CH3, CH2, CH based on H count
- RDKit: Negative values only for C bonded to heteroatoms (`[CH3][N,O,P,S,F,Cl,Br,I]`)
- Our implementation: All alkyl groups got negative values

## Required Changes
1. Parse ~70 SMARTS patterns from RDKit source (Crippen.cpp)
2. Implement priority-based pattern matching
3. Assign contributions based on first match per atom
4. Handle hydrogen contributions separately

## RDKit Reference
Source: https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Descriptors/Crippen.cpp

Key patterns:
```
C1  [CH4]                           0.1441   # Methane
C1  [CH3]C                          0.1441   # Methyl-C
C1  [CH2](C)C                       0.1441   # CH2(C)C
C2  [CH](C)(C)C                     0.0      # Tertiary C
C3  [CH3][N,O,P,S,F,Cl,Br,I]       -0.2035   # CH3 next to heteroatom
C18 [cH]                            0.1581   # Aromatic CH
H1  [#1][#6,#1]                     0.123    # H on C
...
```

## Recommendation
Use the existing SMARTS matching infrastructure from Phase 6 to implement this properly.
This will ensure exact RDKit compatibility for drug discovery applications.

##Alternative (Quick Fix for Now)
Could adjust the contribution table to match common cases, but this won't be accurate
for complex molecules.
