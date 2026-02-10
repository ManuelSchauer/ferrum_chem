# Phase 7: Advanced Molecular Descriptors

**Status**: Planning Phase  
**Priority**: HIGH - Drug discovery essential descriptors  
**Estimated Time**: 2-3 weeks

## Overview

Implement essential molecular descriptors for drug discovery and QSAR modeling, with focus on Lipinski's Rule of Five compliance and structural complexity measures.

## Descriptors to Implement

### 1. Hydrogen Bond Donors/Acceptors

#### HBondDonor
**Definition**: Count of hydrogen bond donor groups  
**Algorithm**: Count N-H and O-H bonds (excluding C-H)

**Standard Definition** (Lipinski):
- Count NH and OH groups
- Common donors: -OH, -NH2, -NH-, -SH

**Implementation**:
```rust
pub struct HBondDonor;

impl Descriptor for HBondDonor {
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let count = mol.atoms.iter()
            .filter(|atom| {
                // N or O with at least one H
                (atom.atomic_number == 7 || atom.atomic_number == 8) &&
                atom.total_h() > 0
            })
            .count();
        output[0] = count as Real;
    }
}
```

**Test Cases**:
- Water (O): 1 donor
- Ammonia (N): 1 donor
- Ethanol (CCO): 1 donor (-OH)
- Urea (NC(=O)N): 2 donors (-NH2 groups)
- Methane (C): 0 donors

---

#### HBondAcceptor
**Definition**: Count of hydrogen bond acceptor groups  
**Algorithm**: Count N and O atoms (with lone pairs)

**Standard Definition** (Lipinski):
- Count N and O atoms
- Excludes charged atoms typically (simplified)

**Implementation**:
```rust
pub struct HBondAcceptor;

impl Descriptor for HBondAcceptor {
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let count = mol.atoms.iter()
            .filter(|atom| {
                // N or O atoms (potential acceptors)
                atom.atomic_number == 7 || atom.atomic_number == 8
            })
            .count();
        output[0] = count as Real;
    }
}
```

**Test Cases**:
- Water (O): 1 acceptor
- Ammonia (N): 1 acceptor
- Ethanol (CCO): 1 acceptor (O)
- Acetone (CC(=O)C): 1 acceptor (C=O)
- Urea (NC(=O)N): 3 acceptors (2N + 1O)

---

### 2. Lipinski's Rule of Five

**Background**: Predicts oral bioavailability of drug candidates

#### LipinskiRuleOfFive
**Definition**: Check all 4 Lipinski criteria

**Rules**:
1. Molecular Weight ≤ 500 Da
2. LogP ≤ 5 (lipophilicity)
3. H-Bond Donors ≤ 5
4. H-Bond Acceptors ≤ 10

**Implementation**:
```rust
pub struct LipinskiRuleOfFive;

impl Descriptor for LipinskiRuleOfFive {
    fn name(&self) -> String { "LipinskiRuleOfFive".to_string() }
    fn size(&self) -> usize { 5 } // [passes, MW<=500, LogP<=5, donors<=5, acceptors<=10]
    
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let mw = calculate_molecular_weight(mol);
        let logp = calculate_logp(mol); // Wildman-Crippen
        let donors = count_hbond_donors(mol);
        let acceptors = count_hbond_acceptors(mol);
        
        let passes_mw = mw <= 500.0;
        let passes_logp = logp <= 5.0;
        let passes_donors = donors <= 5;
        let passes_acceptors = acceptors <= 10;
        
        let passes_all = passes_mw && passes_logp && passes_donors && passes_acceptors;
        
        output[0] = if passes_all { 1.0 } else { 0.0 };
        output[1] = if passes_mw { 1.0 } else { 0.0 };
        output[2] = if passes_logp { 1.0 } else { 0.0 };
        output[3] = if passes_donors { 1.0 } else { 0.0 };
        output[4] = if passes_acceptors { 1.0 } else { 0.0 };
    }
}
```

**Dependencies**: Requires LogP calculation (see below)

**Test Cases**:
- Aspirin: Passes (MW=180, LogP=1.2, donors=1, acceptors=4)
- Ibuprofen: Passes (MW=206, LogP=3.5, donors=1, acceptors=2)
- Atorvastatin: Fails MW (MW=558, violates rule 1)

---

### 3. LogP Estimation

#### LogP (Wildman-Crippen)
**Definition**: Octanol-water partition coefficient  
**Algorithm**: Wildman-Crippen atom contributions

**Method**: 
- Sum atom-type contributions
- Based on SMARTS pattern matching
- 68 atom types defined

**Implementation Approach**:
```rust
pub struct LogP;

// Wildman-Crippen atom type contributions
const LOGP_CONTRIBUTIONS: &[(f64, f64)] = &[
    // (logP contribution, molar refractivity contribution)
    // C sp3
    (0.1441, 2.503),  // [C]
    (0.0000, 2.433),  // [CH]
    (-0.2035, 2.753), // [CH2]
    (-0.2051, 2.731), // [CH3]
    // ... (68 total atom types)
];

impl Descriptor for LogP {
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let mut logp = 0.0;
        
        // Match each atom to Wildman-Crippen type
        for (i, atom) in mol.atoms.iter().enumerate() {
            let atom_type = classify_atom_type(mol, i as u16);
            logp += LOGP_CONTRIBUTIONS[atom_type].0;
        }
        
        output[0] = logp;
    }
}
```

**Requires**:
- SMARTS pattern matching (✅ already implemented)
- 68 atom type patterns
- Contribution lookup table

**Test Cases**:
- Water (O): -0.3 (hydrophilic)
- Octane (CCCCCCCC): 4.9 (lipophilic)
- Ethanol (CCO): 0.0 (balanced)
- Benzene (c1ccccc1): 2.1

---

### 4. Structural Complexity

#### LargestPiSystem
**Definition**: Size of largest conjugated π-system

**Algorithm**:
1. Find all π-bonds (double, triple, aromatic)
2. Build conjugation graph (connected π-systems)
3. Return size of largest component

**Implementation**:
```rust
pub struct LargestPiSystem;

impl Descriptor for LargestPiSystem {
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        // Find all atoms in π-systems
        let mut pi_atoms = vec![false; mol.atoms.len()];
        
        for bond in &mol.bonds {
            if bond.bond_type == BondType::Double ||
               bond.bond_type == BondType::Triple ||
               bond.is_aromatic() {
                pi_atoms[bond.source as usize] = true;
                pi_atoms[bond.target as usize] = true;
            }
        }
        
        // Find connected components of π-atoms
        let largest = find_largest_pi_component(mol, &pi_atoms);
        
        output[0] = largest as Real;
    }
}
```

**Test Cases**:
- Ethane (CC): 0 (no π-system)
- Ethene (C=C): 2 (one double bond)
- Butadiene (C=CC=C): 4 (conjugated)
- Benzene (c1ccccc1): 6 (aromatic ring)
- Naphthalene: 10 (two fused rings)

---

#### LongestAliphaticChain
**Definition**: Length of longest non-aromatic carbon chain

**Algorithm**:
1. Build graph of aliphatic (non-aromatic) carbons
2. Use DFS to find longest path
3. Return path length

**Implementation**:
```rust
pub struct LongestAliphaticChain;

impl Descriptor for LongestAliphaticChain {
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        // Build adjacency for aliphatic carbons only
        let mut max_length = 0;
        
        for start in 0..mol.atoms.len() {
            if is_aliphatic_carbon(mol, start as u16) {
                let length = dfs_longest_path(mol, start as u16);
                max_length = max_length.max(length);
            }
        }
        
        output[0] = max_length as Real;
    }
}

fn is_aliphatic_carbon(mol: &Molecule, idx: u16) -> bool {
    mol.atoms[idx as usize].atomic_number == 6 &&
    !mol.atoms[idx as usize].is_aromatic()
}
```

**Test Cases**:
- Methane (C): 1
- Ethane (CC): 2
- Propane (CCC): 3
- Butane (CCCC): 4
- Isobutane (CC(C)C): 3 (longest is C-C-C)
- Toluene (Cc1ccccc1): 1 (methyl only)
- Ethylbenzene (CCc1ccccc1): 2 (ethyl chain)

---

#### LargestChain
**Definition**: Length of longest chain (any atoms, not just C)

**Algorithm**: Similar to aliphatic chain but includes all atoms

**Implementation**:
```rust
pub struct LargestChain;

impl Descriptor for LargestChain {
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let mut max_length = 0;
        
        // Try starting from each atom
        for start in 0..mol.atoms.len() {
            let length = dfs_longest_path_any(mol, start as u16);
            max_length = max_length.max(length);
        }
        
        output[0] = max_length as Real;
    }
}
```

**Test Cases**:
- Methane (C): 1
- Ethanol (CCO): 3 (C-C-O)
- Propanol (CCCO): 4 (C-C-C-O)
- Benzene (c1ccccc1): 6 (can traverse ring)
- Toluene (Cc1ccccc1): 7 (methyl + ring)

---

## Implementation Order

### Phase 7.1: H-Bond Descriptors (2 days)
1. HBondDonor
2. HBondAcceptor
3. Tests with standard molecules
4. Integration with demo

### Phase 7.2: LogP Estimation (4-5 days)
1. Wildman-Crippen atom type definitions (68 patterns)
2. SMARTS patterns for atom classification
3. Contribution table
4. LogP calculator
5. Validation against known values

### Phase 7.3: Lipinski Rule of Five (1 day)
1. Combine existing descriptors
2. Add rule checker
3. Test with drug molecules
4. Add to demo

### Phase 7.4: Structural Complexity (3-4 days)
1. LargestPiSystem
   - Conjugation detection
   - Component finding
2. LongestAliphaticChain
   - DFS path finding
   - Aliphatic filter
3. LargestChain
   - General path finding
4. Tests and validation

## Testing Strategy

### Unit Tests
- Each descriptor with 5-10 test cases
- Edge cases (single atoms, rings, chains)
- Known molecules with verified values

### Validation Tests
- Compare against RDKit where possible
- Published descriptor values from papers
- ChEMBL/PubChem reference compounds

### Integration Tests
- Batch calculation performance
- CSV output with all descriptors
- Demo with drug-like molecules

## Performance Targets

- HBond descriptors: >10M molecules/sec (simple)
- LogP: >1M molecules/sec (SMARTS matching)
- Lipinski: >1M molecules/sec (combined)
- Structural: >500k molecules/sec (graph algorithms)

## Dependencies

✅ Already implemented:
- SMARTS parser
- Pattern matching (VF2)
- Aromaticity detection
- Molecular weight

🔨 Need to implement:
- Graph traversal (DFS for chains)
- Component detection (π-systems)
- Wildman-Crippen atom types
- Contribution tables

## Documentation

Create for each descriptor:
- Algorithm description
- Literature references
- Test cases with expected values
- Performance benchmarks
- Usage examples

## References

1. Lipinski, C. A., et al. (2001). "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings." Advanced Drug Delivery Reviews.

2. Wildman, S. A., & Crippen, G. M. (1999). "Prediction of physicochemical parameters by atomic contributions." Journal of Chemical Information and Computer Sciences.

3. Ertl, P., et al. (2000). "Fast calculation of molecular polar surface area as a sum of fragment-based contributions." Journal of Medicinal Chemistry.

4. RDKit Documentation: Descriptors module

## Success Criteria

- ✅ All descriptors match RDKit within 5% error
- ✅ >100 total tests passing
- ✅ Complete Lipinski Rule of Five implementation
- ✅ Comprehensive documentation
- ✅ Demo with 20+ drug molecules
- ✅ Performance meets targets

## Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | H-Bond + LogP | HBondDonor, HBondAcceptor, LogP |
| 2 | Lipinski + Structure | Lipinski, LargestPiSystem |
| 3 | Chains + Polish | Aliphatic/Largest Chain, Tests, Docs |

**Total**: ~3 weeks for complete implementation and validation
