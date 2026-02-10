# Phase 7 Descriptors - Quick Reference

## Summary Table

| # | Descriptor | Type | Lipinski? | Complexity | Est. Time |
|---|------------|------|-----------|------------|-----------|
| 1 | HBondDonor | Count | ✅ ≤5 | Low | 1 day |
| 2 | HBondAcceptor | Count | ✅ ≤10 | Low | 1 day |
| 3 | LogP | Property | ✅ ≤5 | Medium | 4-5 days |
| 4 | LipinskiRuleOfFive | Boolean | ✅ Primary | Low | 1 day |
| 5 | LargestPiSystem | Structural | No | Medium | 1-2 days |
| 6 | LongestAliphaticChain | Structural | No | High | 1-2 days |
| 7 | LargestChain | Structural | No | High | 1 day |

**Total**: 7 new descriptors, ~10-12 days

---

## Descriptor Definitions

### HBondDonor
**What**: Count of hydrogen bond donor groups  
**Counts**: N-H and O-H bonds  
**Examples**:
- Water (O): 1
- Ethanol (CCO): 1  
- Urea (NC(=O)N): 2

**Algorithm**: 
```rust
count N and O atoms with at least 1 hydrogen
```

---

### HBondAcceptor
**What**: Count of hydrogen bond acceptor sites  
**Counts**: N and O atoms  
**Examples**:
- Water (O): 1
- Ethanol (CCO): 1
- Acetone (CC(=O)C): 1

**Algorithm**:
```rust
count N and O atoms
```

---

### LogP
**What**: Octanol-water partition coefficient  
**Method**: Wildman-Crippen atom contributions  
**Range**: Typically -3 to +8  
**Examples**:
- Water: -0.3 (hydrophilic)
- Octane: 4.9 (lipophilic)
- Benzene: 2.1

**Algorithm**:
```rust
sum of 68 atom-type contributions
```

---

### LipinskiRuleOfFive
**What**: Drug-likeness assessment  
**Returns**: 5 values [passes_all, MW, LogP, donors, acceptors]  
**Criteria**:
1. Molecular Weight ≤ 500 Da
2. LogP ≤ 5
3. H-Bond Donors ≤ 5
4. H-Bond Acceptors ≤ 10

**Examples**:
- Aspirin: PASS
- Atorvastatin: FAIL (MW=558)

---

### LargestPiSystem
**What**: Size of largest conjugated π-system  
**Counts**: Atoms in largest conjugated network  
**Examples**:
- Ethane (CC): 0
- Ethene (C=C): 2
- Benzene (c1ccccc1): 6
- Naphthalene: 10

**Algorithm**:
```rust
1. Find all π-bonds (double, triple, aromatic)
2. Build connectivity graph
3. Find largest connected component
```

---

### LongestAliphaticChain
**What**: Longest non-aromatic carbon chain  
**Counts**: Path length through aliphatic carbons  
**Examples**:
- Butane (CCCC): 4
- Isobutane (CC(C)C): 3
- Toluene (Cc1ccccc1): 1 (methyl only)

**Algorithm**:
```rust
1. Filter to aliphatic carbons only
2. DFS from each carbon
3. Return maximum path length
```

---

### LargestChain
**What**: Longest chain of any atoms  
**Counts**: Maximum path length in molecular graph  
**Examples**:
- Ethanol (CCO): 3
- Benzene (c1ccccc1): 6
- Toluene (Cc1ccccc1): 7

**Algorithm**:
```rust
1. Consider all atoms
2. DFS from each atom
3. Return maximum path length
```

---

## Implementation Dependencies

### Required Components
✅ Already have:
- Molecule graph structure
- Aromaticity detection
- SMARTS pattern matching
- Molecular weight calculator

🔨 Need to implement:
- **DFS path finding** (for chain descriptors)
- **Connected components** (for π-system)
- **Atom type classifier** (for LogP)
- **Contribution tables** (Wildman-Crippen data)

### Shared Utilities
These will be useful for multiple descriptors:
```rust
// Graph traversal
fn dfs_longest_path(mol: &Molecule, start: u16) -> usize

// Component finding
fn find_connected_components(atoms: &[bool], bonds: &[Bond]) -> Vec<Vec<u16>>

// Atom classification
fn classify_wildman_crippen_type(mol: &Molecule, idx: u16) -> u8
```

---

## Test Strategy

### Per Descriptor Testing
Each descriptor needs:
1. **Simple molecules** (5-8 tests)
   - Single functionality (water, methane, etc.)
   - Known exact values
2. **Edge cases** (2-3 tests)
   - Single atom, rings, branches
3. **Validation** (3-5 tests)
   - Compare against RDKit
   - Drug molecules with known values

### Integration Testing
- Batch calculation with all descriptors
- CSV output with new descriptors
- Performance benchmarks
- Demo with 20+ molecules

### Total New Tests
Estimated: **10-15 tests per descriptor** = 70-105 new tests  
Conservative target: **+33 tests** → 132 total

---

## Performance Targets

| Descriptor | Target | Rationale |
|------------|--------|-----------|
| HBondDonor | >10M/sec | Simple iteration |
| HBondAcceptor | >10M/sec | Simple iteration |
| LogP | >1M/sec | SMARTS matching cost |
| Lipinski | >1M/sec | Combines above |
| LargestPiSystem | >500k/sec | Component finding |
| AliphaticChain | >200k/sec | DFS complexity |
| LargestChain | >200k/sec | DFS complexity |

**Overall pipeline**: Should maintain >5k molecules/sec for full descriptor set

---

## Validation Sources

1. **RDKit** - Primary comparison
   ```python
   from rdkit.Chem import Descriptors
   Descriptors.NumHDonors(mol)
   Descriptors.NumHAcceptors(mol)
   Descriptors.MolLogP(mol)
   ```

2. **PubChem** - Published values for common drugs

3. **Literature**:
   - Lipinski et al. (2001) - Rule of Five
   - Wildman & Crippen (1999) - LogP method
   - Ertl et al. (2000) - TPSA (future)

---

## Implementation Order Rationale

1. **H-Bond first**: Simple, needed for Lipinski, builds confidence
2. **LogP second**: Most complex, takes time, needed for Lipinski
3. **Lipinski third**: Quick win once dependencies ready
4. **Structural last**: Independent, can parallelize if needed

This order minimizes blocking and allows early validation of Lipinski implementation.
