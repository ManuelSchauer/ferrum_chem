# TODO List - FerrumChem Development

## ⚠️ Thread Safety Requirements

**ALL implementations must be thread-safe:**
- Data structures: Must implement `Send + Sync`
- Algorithms: Must not use mutable global state
- Descriptors: Stateless or use only immutable references
- Parsers: Independent state per invocation

See [`THREAD_SAFETY.md`](THREAD_SAFETY.md) for guidelines.

## ✅ Phase 1-8: Complete (Core Functionality + MACCS Keys)

### Phase 1-3: Core Algorithms & Descriptors ✅
- [x] SMILES parser (organic subset, bracket atoms, fused rings, auto-aromaticity)
- [x] Implicit hydrogen calculation (valence-based, bracket atom aware)
- [x] Ring detection (SSSR algorithm)
- [x] Aromaticity perception (Hückel 4n+2 rule, enhanced for fused heterocycles)
- [x] Kekulization module (available, not used by default)
- [x] Atom/Bond/Aromatic counts, Rotatable bonds, Formal charge

### Phase 4-6: Pattern Matching & I/O ✅
- [x] SMARTS parser + recursive SMARTS
- [x] VF2 subgraph isomorphism
- [x] CSV reader + Rayon batch processing

### Phase 7: Descriptors ✅
- [x] H-bond donor/acceptor, LogP (Wildman-Crippen, 111 patterns)
- [x] Extended SMARTS + pre-parsed patterns (10-50x speedup)
- [x] Lipinski Rule of Five
- [x] LongestAliphaticChain, LargestChain, LargestPiSystem

### Phase 8: MACCS Keys & SmallRing ✅
- [x] MACCS 166-bit fingerprint (163/163 patterns)
- [x] SmallRing descriptor (11 values)
- [x] Recursive SMARTS, ring closure bonds, X/D primitives

## 🔥 Phase 9A: Easy Topological Descriptors ✅ COMPLETE

### Distance-Based Descriptors ✅ COMPLETE
- [x] **Wiener Index** - sum of shortest paths, reusable distance matrix (10 tests)
- [x] **Zagreb Indices M1, M2** - degree-based (9 tests)
- [x] **Balaban J Index** - distance matrix based (8 tests)
- [x] **Eccentric Connectivity Index** (9 tests)
- [x] **Kier-Hall Kappa Indices** - shape descriptors κ₁, κ₂, κ₃ (8 tests)

### Simple Counting Descriptors ✅ COMPLETE
- [x] **Valence Electrons Count** (12 tests)
- [x] **Radical Electrons Count** (in simple_counts.rs)
- [x] **Heteroatom Count** (in simple_counts.rs)
- [x] **Spiro Atom Count** (7 tests)
- [x] **Carbon Types** - C1SP1..C4SP3 classification (9 tests)
- [x] **Ring Type Counts** - saturated/aliphatic (7 tests)
- [x] **Hybridization Ratio** - sp³/sp² carbon ratio
- [x] **VAdjMa** - vertex adjacency magnitude

### High-Value Physicochemical Descriptors ✅ COMPLETE
- [x] **TPSA** ⭐⭐⭐ - RDKit-exact algorithm, **50/50 molecules validated (±0.01 Ų)** (15 tests)
- [x] **Molar Refractivity** - Crippen method, reuses LogP patterns (5 tests)

### Fragment-Based (SMARTS Ready!) ✅ COMPLETE
- [x] **Functional Group Counts** - 16 SMARTS patterns (6 tests)
- [x] **Acidic Group Count** (3 tests)
- [x] **Basic Group Count** (3 tests)
- [x] **QED (Drug-likeness)** - combines MW, LogP, HBA, HBD, TPSA, RotBonds, AromaticRings (6 tests)

### Additional Descriptors ✅ COMPLETE
- [x] **Petitjean Indices** (8 tests)
- [x] **Gravitational Index** (7 tests)

**Status**: 35/35 descriptors implemented, 395 tests passing ✅

## 🧬 Phase 9B: Morgan/ECFP Fingerprints ✅ COMPLETE

### Implementation ⭐⭐⭐
- [x] **Morgan Algorithm Core** - circular fingerprints with configurable radius
- [x] **Hashing Infrastructure** - std DefaultHasher, atom invariants (element, degree, H, charge, aromatic, ring)
- [x] **Bit Folding** - hash % num_bits, supports 1024/2048/4096 bit sizes
- [x] **ECFP Variants** - ECFP2 (radius=1), ECFP4 (radius=2), ECFP6 (radius=3)
- [x] **FCFP Variants** - FCFP2/4/6 with pharmacophoric atom invariants (donor, acceptor, aromatic, hydrophobic, ionizable)
- [x] **Tanimoto Similarity** - binary fingerprint comparison function
- [x] **Tests** - 25 tests (ECFP + FCFP: symmetry, similarity, pharmacophore, drug-like molecules)
- [x] **Backward Compat** - ECFP4 struct + Morgan type alias

## 🧬 Phase 9C: Additional Fingerprints ✅ COMPLETE

### Atom Pair Fingerprints ✅
- [x] **Distance-Based Encoding** - BFS distance matrix, (atom_type_i, distance, atom_type_j) for distances 1-7
- [x] **Bit folding** - hash to configurable bit vector (2048 default)
- [x] **Tests** - 7 tests

### Topological Torsion ✅
- [x] **4-Atom Path Encoding** - DFS path finder, canonical hashing
- [x] **Tests** - 8 tests

### PubChem 881 Fingerprints ✅
- [x] **881 Structural Keys** - ALL defined as SMARTS + count threshold
- [x] **Sections**: elements (0-114), rings (115-262), atom pairs (263-326), neighborhoods (327-459), simple SMARTS (460-712), complex SMARTS (713-880)
- [x] **SMARTS Parser Enhancement** - added bond OR (`,`) support for `=,:` syntax
- [x] **All 881/881 patterns parse successfully**
- [x] **Tests** - 10 tests (benzene=71 bits, aspirin=160 bits, caffeine=216 bits)

**Phase 9C Output**: Complete fingerprint suite (MACCS + ECFP/FCFP + AtomPair + Torsion + PubChem)

## 🧬 Phase 10: Medium Complexity Descriptors ✅ COMPLETE

### Graph-Based Descriptors ✅
- [x] **Chi Indices** (chi.rs) - 5 descriptors: ChiPath(8), ChiValencePath(8), ChiCluster(2), ChiChain(5), ChiPathCluster(3) = 26 values, 25 tests
- [x] **MDE (Molecular Distance Edge)** (mde.rs) - 15 atom-pair type values, 8 tests
- [x] **Weighted Path** (weighted_path.rs) - 10 values (mass + EN weighted, paths 1-5), 6 tests

### Electronic/Topological ✅
- [x] **EState Indices** (estate.rs) - EStateSummary(3) + EStateAtomTypes(44) = 47 values, 13 tests
- [x] **Autocorrelation ATS** (autocorrelation.rs) - 32 values (4 weights × 8 lags), 9 tests
- [x] **BCUT Descriptors** (bcut.rs) - 8 eigenvalues (4 weights × hi/lo), Jacobi eigendecomposition (no nalgebra), 9 tests

**Phase 10 Status**: 6 descriptor groups, 138 new values, 70 tests. Total: 465 tests passing ✅

## 🧬 Phase 11: Architecture Refactor - Typed Descriptors (2-3 weeks) 🟡

### Goal: Replace f32 outputs with proper types for QSAR + memory efficiency

**Current Issue:**
- All descriptors return `Vec<f32>` → wastes memory, loses precision
- Fingerprints: 881 bits = 3.5 KB/molecule (881 × f32)
- Atom counts stored as floats (rounding issues)

**Benefits of Typed Returns:**
- **4x memory reduction** for fingerprints: 3.5KB → 881 bytes (BitVec)
- **Type safety** for QSAR: u32 atom counts (no float rounding)
- **Better API**: `atom_count()` → `u32` not `[f32; 1]`
- **Direct binary ops** for Tanimoto similarity

### Design (1 week)
```rust
pub enum DescriptorValue {
    Int(u32),
    Float(f32),
    IntVec(Vec<u32>),
    FloatVec(Vec<f32>),
    Binary(BitVec),  // bit-packed
}

pub trait TypedDescriptor: Send + Sync {
    fn name(&self) -> &str;
    fn calculate(&self, mol: &Molecule) -> DescriptorValue;
}

// Conversion layer for ML pipelines
impl DescriptorValue {
    fn to_floats(&self, output: &mut [f32]);
}
```

### Migration (2 weeks)
- [ ] Implement TypedDescriptor trait + DescriptorValue enum
- [ ] Migrate fingerprints first (PubChem, MACCS, ECFP, AtomPair → BitVec)
- [ ] Migrate counting descriptors (atoms, bonds, rings → u32)
- [ ] Migrate floating descriptors (LogP, TPSA, MW → f32)
- [ ] Keep legacy Descriptor trait for backward compatibility
- [ ] Add ML vector builder with type conversions
- [ ] Update all 395 tests incrementally

**Phase 11 Output**: Typed descriptor system, 4x memory reduction, QSAR-ready

## 🧬 Phase 12: Electronic Descriptors - COMPLEXITY CLIFF (3-4 weeks) 🔴

### ⚠️ Gasteiger Partial Charges (2 weeks) - HARDEST NON-3D FEATURE

- [ ] **Implementation** (10 days)
  - Iterative electronegativity equalization
  - Algorithm: Gasteiger & Marsili 1980
  - Steps:
    1. Assign atom types + parameters (a, b, c)
    2. Initialize charges (0)
    3. Iterate: χ = a + b×q + c×q², transfer charge Δq ∝ (χ_i - χ_j)
    4. Damping + convergence (6-10 iterations)
  - Edge cases: conjugation, aromatics, heteroatoms, charged species
  - 20+ atom type parameters

- [ ] **Tests & Validation** (4 days)
  - 20+ molecules covering all atom types
  - Compare to RDKit (tolerance ±0.01-0.05)
  - Edge cases: zwitterions, resonance
  - Molecules: alkanes, aromatics, heterocycles, charged

**Why This Is Hard**:
- Complex atom typing (20+ types)
- Iteration convergence tricky
- Many chemistry edge cases
- Requires extensive validation
- Bugs show as subtle charge errors

### Charge-Derived Descriptors (1 week)
**Requires Gasteiger completed first**

- [ ] **Partial Charge Min/Max/Abs** (2 days)
  - MaxPartialCharge, MinPartialCharge
  - MaxAbsPartialCharge, MinAbsPartialCharge
  - Test: carbonyl C positive, O negative

- [ ] **CPSA Descriptors** (3 days)
  - Charged Partial Surface Area (29 descriptors)
  - Requires: Gasteiger + 3D coordinates
  - PPSA, PNSA, DPSA variants
  - ⚠️ May defer until conformer generation

- [ ] **Polarizability Descriptors** (2 days)
  - Atomic polarizabilities (table lookup)
  - APol: sum of polarizabilities
  - BPol: polarizability differences

**Phase 11 Output**: Electronic property descriptors, gateway to advanced QSAR

## 🧬 Phase 12: 3D Descriptors - MAJOR PROJECT (Future) 🔴

### ⚠️ BLOCKED: Requires Conformer Generation First

### Conformer Generation (2-3 weeks)
- [ ] **ETKDG Algorithm**
  - Distance geometry + experimental torsion knowledge
  - Riniker & Landrum JCIM 2015
  - Major algorithmic project
  - Alternative: RDKit FFI wrapper

### 3D Shape Descriptors (1 week after conformers)
- [ ] **Principal Moments of Inertia**
  - PMI1, PMI2, PMI3
  - NPR1, NPR2 (normalized ratios)
  - Asphericity, Eccentricity, Spherocity
  - Diagonalize inertia tensor

- [ ] **Radius of Gyration**
  - RMS distance from center of mass

### Other 3D Features
- [ ] 3D Pharmacophore fingerprints
- [ ] Molecular surface area/volume
- [ ] Shape similarity (USR, USRCAT)
- [ ] WHIM descriptors
- [ ] GETAWAY descriptors

**Phase 12 Note**: Defer until 2D suite complete. Many QSAR models work fine with 2D only.

---

## 🧬 Future Work

### Phase 9B: Morgan/ECFP Fingerprints 🟡
- [ ] Morgan algorithm core (circular fingerprints)
- [ ] Hashing with FNV or xxHash
- [ ] Bit folding (1024, 2048, 4096)
- [ ] ECFP2, ECFP4, ECFP6 variants

### Phase 9C: Additional Fingerprints 🟡
- [ ] Atom pair fingerprints
- [ ] Topological torsion
- [ ] PubChem 881 structural keys

### Phase 10: Medium Complexity Descriptors 🟡
- [ ] Chi indices (Kier & Hall connectivity)
- [ ] EState indices (electrotopological state)
- [ ] Autocorrelation ATS
- [ ] BCUT eigenvalues

### Phase 11: Electronic Descriptors 🔴
- [ ] Gasteiger partial charges
- [ ] Charge-derived descriptors (CPSA, polarizability)

### Phase 12: I/O & Infrastructure
- [ ] SDF/MOL V2000/V3000 parser
- [ ] InChI generation
- [ ] Morgan canonicalization

### Phase 13: ML Integration
- [ ] Linfa integration examples
- [ ] ART 2A clustering
- [ ] PyO3 Python bindings

### Phase 14: 3D Descriptors (Future)
- [ ] Conformer generation (ETKDG)
- [ ] 3D shape descriptors (PMI, USR)
- [ ] 3D pharmacophore fingerprints

## 🐛 Known Issues / Improvements

### Parser
- [x] SMILES bracket atoms: `no_implicit_h` flag prevents incorrect H calculation
- [x] Aromatic bond assignment: only set Aromatic if BOTH atoms aromatic
- [ ] Better error messages with line/column info
- [ ] Handle escaped characters in SMILES
- [ ] Stereochemistry (E/Z, R/S)

### Architecture
- [ ] Consider adding 3D coordinates to Atom
- [ ] Add conformer support to Molecule

## 📊 Performance

### Current (Phase 9A)
- ✅ SMILES parsing: 1-6M mol/sec
- ✅ MW calculation: 1B calc/sec
- ✅ Memory: ~22 bytes/atom
- ✅ SMARTS matching: ~50k mol/sec (VF2 algorithm)
- ✅ LogP: ~500k-5M mol/sec (pre-parsed patterns)
- ✅ TPSA: RDKit-exact (50/50 molecules ±0.01 Ų)
- ✅ Total tests: 314/314 passing
- ✅ Total descriptors: 28 (+ MACCS 166-bit)

## 🎯 Milestones

### v0.1.0 - v0.3.0 ✅ Complete
- [x] SMILES parser, MW, implicit H, rings, aromaticity
- [x] SMARTS parser, VF2 subgraph isomorphism, CSV batch

### v0.4.0 - Phase 7 ✅ Complete
- [x] 18 descriptors (LogP, H-bond, Lipinski, chains, pi system)

### v0.5.0 - Phase 8 ✅ Complete
- [x] MACCS 166-bit fingerprint (163/163 patterns)
- [x] SmallRing descriptor (11 values)

### v0.6.0 - Phase 9A ✅ Complete
- [x] 35 descriptors total (Wiener, Zagreb, Balaban, Eccentric, TPSA, Petitjean, Kappa, MR, QED, FuncGroups, etc.)
- [x] TPSA: RDKit-exact validation
- [x] Kappa indices, Molar Refractivity, Functional Groups, Acidic/Basic counts, QED

### v0.7.0 - Phase 9B/C (Current)
- [x] Morgan/ECFP fingerprints (ECFP2/4/6, Tanimoto similarity)
- [ ] Atom pair, topological torsion, PubChem FP

### v1.0.0 - Production (Future)
- [ ] Full documentation, benchmarks, crates.io publish

---

**Last Updated**: February 2026  
**Status**: Phase 9B ✅ Complete  
**Tests**: 370/370 passing ✅  
**Descriptors**: 35 molecular descriptors + MACCS 166-bit + ECFP/FCFP fingerprints  
**TPSA**: RDKit-exact (50/50 molecules, ±0.01 Ų)  
**Next Priority**: Phase 9C - Additional Fingerprints (Atom Pair, Torsion, PubChem)  
**Strategy**: Complete 2D descriptors before Gasteiger charges or 3D


