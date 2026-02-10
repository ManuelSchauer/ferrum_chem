# TODO List - FerrumChem Development

## ⚠️ Thread Safety Requirements

**ALL implementations must be thread-safe:**
- Data structures: Must implement `Send + Sync`
- Algorithms: Must not use mutable global state
- Descriptors: Stateless or use only immutable references
- Parsers: Independent state per invocation

See [`THREAD_SAFETY.md`](THREAD_SAFETY.md) for guidelines.

## ✅ Phase 1-7: Complete (Core Functionality + SMARTS Parser)

### Phase 1: SMILES Parser & Molecular Weight
- [x] Element data lookup tables
- [x] SMILES parser (organic subset)
- [x] MoleculeBuilder trait implementation
- [x] Molecular weight descriptor
- [x] Unit tests (13 tests)
- [x] Performance benchmarks
- [x] Documentation (README, PROJECT_PLAN, etc.)
- [x] Example programs

### Phase 2-3: Core Algorithms & Basic Descriptors
- [x] Implicit hydrogen calculation
- [x] Ring detection (SSSR algorithm)
- [x] Aromaticity perception (Hückel 4n+2 rule)
- [x] Atom counts (total, heavy, by element)
- [x] Bond counts (total, by type)
- [x] Aromatic atom/ring counting
- [x] Rotatable bonds count
- [x] Formal charge descriptor

### Phase 4-6: Pattern Matching & I/O
- [x] SMARTS parser (basic primitives)
- [x] VF2 subgraph isomorphism
- [x] CSV reader for SMILES
- [x] Batch processing with Rayon

### Phase 7: Advanced SMARTS & Descriptors ✅ **COMPLETE**
- [x] H-bond donor/acceptor descriptors
- [x] LogP estimation (Wildman-Crippen, 111 patterns)
- [x] Extended SMARTS parser with:
  - [x] Comma OR lists: `[#6,#7,#8]`
  - [x] Implicit AND: `[CH3]`, `[cH]`
  - [x] Branching: `C(C)C`, `C(C)(C)C`
  - [x] NOT operator: `[!#6]`, `[!C]`
  - [x] Lowercase 'a' for aliphatic
  - [x] Valence calculation: `[Cv4]`, `[Nv3]`
  - [x] Ring bonds counting: `[Cx2]`, `[Cx3]`
  - [x] 'x' primitive parsing
- [x] 13 comprehensive SMARTS tests
- [x] 27 LogP validation tests
- [x] All TODOs resolved in smarts.rs
- [x] Pre-parsed patterns with `once_cell` (10-50x speedup)
- [x] Documentation (SMARTS_PARSER_COMPLETE.md, SMARTS_OPTIMIZATION_COMPLETE.md, etc.)

**Status**: 159/159 tests passing ✅, 111/111 Crippen patterns (100%), production-ready and optimized ✨

**Performance Improvements**:
- Pre-parsed patterns using `once_cell::Lazy`
- LogP: 50k/sec → 500k-5M/sec (10-50x faster)
- One-time startup cost: ~1-5ms for 111 patterns
- Thread-safe, zero locking overhead
- Memory: +19 KB (negligible)

**Documentation**:
- `SMARTS_PARSER_COMPLETE.md` - Technical details
- `SMARTS_OPTIMIZATION_COMPLETE.md` - Pre-parsing optimization
- `PHASE7_COMPLETE_OPTIMIZED.md` - Comprehensive summary

## 🔥 Phase 7.3: Simple Descriptors ✅ COMPLETE

### Lipinski Rule of Five
- [x] Create lipinski.rs module
- [x] Combine MW ≤ 500 (ready)
- [x] LogP ≤ 5 (ready)
- [x] HBondDonor ≤ 5 (ready)
- [x] HBondAcceptor ≤ 10 (ready)
- [x] Return pass/fail + individual values
- [x] Add tests with known drugs (7 tests passing)
- [x] Add to descriptor_demo
- **Status**: ✅ Complete

### Structural Descriptors
- [x] LongestAliphaticChain - DFS through sp3 carbons (10 tests)
- [x] LargestChain - DFS through any atoms (10 tests)
- [x] LargestPiSystem - conjugated double bond systems (11 tests)
- **Status**: ✅ Complete

**Summary**: 
- 3 new descriptor modules (760 lines)
- 28 new tests (+28)
- Total: 18 descriptors, 187 tests passing ✅

## 🔥 Phase 8: MACCS Keys ✅ COMPLETE (163/163 patterns parsing - 100%)

### MACCS Keys (Structural Fingerprints)
- [x] Create maccs.rs module (550+ lines)
- [x] Define all 166 MACCS keys from RDKit
- [x] Implement pre-parsed patterns (once_cell)
- [x] Count-based matching for 14 keys
- [x] Special key handling (isotope, aromatic rings, fragments)
- [x] Extend SMARTS parser for MACCS patterns
- [x] 7+ comprehensive tests
- [x] Add to descriptor_demo
- [x] Documentation (PHASE8_MACCS_KEYS_COMPLETE.md)

**SMARTS Extensions Added**:
- [x] Ring closures with wildcards: `*1~*~*~*~1`
- [x] Bond negation: `!@`, `!:`
- [x] Ring bond: `@`
- [x] Bond AND: `;` (e.g., `=;@`)
- [x] Two-letter elements in OR lists: `[Cu,Zn,Ag,Hg]`
- [x] Fixed 'a' = AnyAromatic
- [x] Property/element disambiguation (H/Hg, D/Dy, R/Ru, X/Xe)
- [x] **Fixed ring closure bond types** (was defaulting to Single instead of using specified bond)
- [x] **Recursive SMARTS `$(...)`** (nested pattern matching)
- [x] **Fixed X vs D primitives** (X = Connectivity with H, D = Degree without H)

**Status**: ✅ 163/163 patterns parsing (100%), all recursive SMARTS working
**Tests**: 212/212 passing (+25)
**Bug Fixes**: Ring closure bonds, SSSR interaction (was actually ring closure bug), X/D semantics

### SmallRing Descriptor ✅ COMPLETE
- [x] Create small_ring.rs module (400+ lines)
- [x] Count total small rings (3-9 membered)
- [x] Count aromatic rings
- [x] Count distinct ring blocks (ring systems)
- [x] Count aromatically connected components
- [x] Count individual ring sizes (3, 4, 5, 6, 7, 8, 9-membered)
- [x] 5 comprehensive tests (benzene, naphthalene, cyclohexane, cyclopropane, no rings)
- **Returns**: 11 values total
- **Status**: ✅ Complete, based on CDK SmallRingDescriptor
- **Tests**: 5/5 passing

### Morgan/ECFP Fingerprints (Circular Fingerprints)
- [ ] Implement Morgan algorithm
- [ ] Add radius parameter (typically 2-4)
- [ ] Support different bit sizes (1024, 2048, 4096)
- [ ] Add folding for dimensionality reduction
- [ ] Comprehensive tests
- [ ] Add hashing with FNV or xxHash
- [ ] Support ECFP2, ECFP4, ECFP6
- [ ] Benchmark vs RDKit
- **Note**: Store as Vec<f32>, not packed bits
- **Estimated**: 1-2 weeks

### Atom Pair Fingerprints
- [ ] Implement atom pair algorithm
- [ ] Distance-based encoding
- [ ] Tests and validation
- [ ] Add topological torsion variant
- [ ] Validate against RDKit reference implementation
- **Estimated**: 1 week

### Design Decision
- [x] Unified float vector (Vec<f32>) for all features
- [x] No separate bit storage
- [x] Documented in DESIGN_UNIFIED_FLOAT_VECTOR.md

## 🧬 Phase 9A: Easy Topological Descriptors (2-3 weeks) 🟢

### Distance-Based Descriptors (Foundation)
- [ ] **Wiener Index** (1 day)
  - Sum of shortest paths between all atom pairs
  - Algorithm: BFS from each atom O(n²·m) or Floyd-Warshall O(n³)
  - Test: methane=0, ethane=1, propane=4, benzene=63
  - Creates reusable distance matrix

- [ ] **Zagreb Indices M1, M2** (1 day)
  - M1 = sum of degree² for all atoms
  - M2 = sum of (degree_i × degree_j) for all bonds
  - Test: benzene M1=24, M2=24
  - Single pass, pure graph theory

- [ ] **Balaban J Index** (2 days)
  - Requires distance matrix from Wiener
  - Formula: m/(μ+1) × sum((d_i × d_j)^(-0.5))
  - Test: benzene J≈2.00, adamantane J≈2.31

- [ ] **Eccentric Connectivity Index** (2 days)
  - Requires distance matrix
  - Eccentricity = max distance from vertex
  - Sum of (degree × eccentricity)

- [ ] **Kier-Hall Kappa Indices** (2 days)
  - Shape descriptors: κ₁, κ₂, κ₃
  - Based on path/cluster/path-cluster counts
  - CDK/RDKit reference available

### Simple Counting Descriptors
- [ ] **Valence Electrons Count** (0.5 day)
  - Sum: outer_electrons - formal_charge + total_H
  - Periodic table lookup (elements.rs ready)
  - Test: ethane=14, CO₂=16

- [ ] **Radical Electrons Count** (0.5 day)
  - Already in Atom::num_radical_electrons
  - Simple sum over atoms
  - Test: methyl radical=1

- [ ] **Heteroatom Count** (0.5 day)
  - Count non-C, non-H atoms
  - Single pass
  - Test: benzene=0, pyridine=1, caffeine=4

- [ ] **Spiro Atom Count** (1 day)
  - Atoms in ≥2 rings but not bridges
  - Use existing SSSR
  - Test: spiro[3.3]heptane=1

- [ ] **Carbon Types Descriptor** (1 day)
  - Count: C1SP1, C2SP1, C1SP2, C2SP2, C1SP3, C2SP3, C3SP3, C4SP3
  - Hybridization from neighbor count + bond types
  - Test: methane=C4SP3, ethene=2×C2SP2

- [ ] **Ring Type Counts** (0.5 day)
  - NumSaturatedRings, NumAliphaticRings
  - Filter existing rings by aromaticity
  - Test: cyclohexane=1 saturated, benzene=0

### High-Value Physicochemical Descriptors
- [ ] **TPSA (Topological Polar Surface Area)** ⭐⭐⭐ (3 days)
  - Fragment-based contributions (Ertl et al. 2000)
  - ~40 SMARTS patterns with surface area values
  - No 3D coordinates needed
  - Test: aspirin≈63.6, caffeine≈58.4 Ų
  - **Why priority**: Critical for drug-likeness predictions

- [ ] **Molar Refractivity** (2 days)
  - Crippen or Ghose-Crippen method
  - Similar to LogP (reuse patterns, different coefficients)
  - Test: benzene≈26.2, aspirin≈52.4

### Fragment-Based Pattern Matching (SMARTS Ready!)
- [ ] **Functional Group Counts** (2 days)
  - 20-30 common groups with SMARTS patterns
  - Groups: alcohol, phenol, amine (1°/2°/3°), carboxylic acid, ester, amide, ketone, aldehyde, ether, nitrile, nitro, sulfonyl, phosphate, halide, thiol
  - Test: aspirin has 1 carboxylic acid, 1 ester

- [ ] **Acidic Group Count** (1 day)
  - SMARTS: carboxylic, sulfonic, phosphoric, phenols
  - Test: aspirin=1, citric acid=3

- [ ] **Basic Group Count** (1 day)
  - SMARTS: amines, aromatic N, guanidine
  - Test: caffeine=4, aniline=1

- [ ] **QED (Drug-likeness)** (1 day)
  - Combines: MW, LogP, HBA, HBD, PSA, RotBonds, AromaticRings
  - Formula from Bickerton et al. Nature Chemistry 2012
  - Test: aspirin≈0.55, ibuprofen≈0.86

### Additional Useful Descriptors
- [ ] **Petitjean Indices** (1 day)
  - Petitjean Number, Petitjean Shape Index
  - Eccentricity-based shape measure

- [ ] **Gravitational Index** (1 day)
  - Sum of (m_i × m_j) / d_ij² for all pairs
  - Masses + distance matrix

- [ ] **Hybridization Ratio** (0.5 day)
  - sp³ carbons / sp² carbons
  - "Escape from flatland" metric

- [ ] **VAdjMa** (0.5 day)
  - 1 + log₂(m) where m = heavy-heavy bonds

**Phase 9A Output**: ~35 descriptors, 2-3 weeks, all validated vs RDKit/CDK

## 🧬 Phase 9B: Morgan/ECFP Fingerprints (1 week) 🟡

### Implementation (Highest Priority) ⭐⭐⭐
- [ ] **Morgan Algorithm Core** (3 days)
  - Circular fingerprints with radius parameter
  - Iterative atom environment hashing
  - Algorithm:
    1. Init: hash(atom) = invariants (element, degree, charge, H)
    2. For radius 1..R: new_hash = hash(old_hash + sorted(neighbor_hashes))
    3. Collect all hashes at each radius
  - Test: methane, ethane, benzene symmetry

- [ ] **Hashing Infrastructure** (1 day)
  - Add `ahash` or `fnv` crate
  - FNV-1a or xxHash for environment hashing
  - Collision handling (expected behavior)

- [ ] **Bit Folding** (0.5 day)
  - Map hash → bit index: hash % size
  - Support 1024, 2048, 4096 bit sizes
  - Store as Vec<f32> with 0.0/1.0 (decided)

- [ ] **Variants** (0.5 day)
  - ECFP2 (radius=1), ECFP4 (radius=2), ECFP6 (radius=3)
  - Feature-based vs count-based

- [ ] **Tests & Validation** (1 day)
  - 10+ molecules with known RDKit fingerprints
  - Tanimoto similarity calculations
  - Benchmark: 100k-500k molecules/sec

**Why Critical**: Most important fingerprint for ML/similarity search, industry standard

## 🧬 Phase 9C: Additional Fingerprints (2 weeks) 🟡

### Atom Pair Fingerprints (4 days)
- [ ] **Distance-Based Encoding** (3 days)
  - Encode: (atom_type_i, distance, atom_type_j)
  - All pairs at distances 1-10 bonds
  - Atom types: element + π-electrons + H-count
  - Hash to bit vector

- [ ] **Tests** (1 day)
  - Validate vs RDKit atom pairs

### Topological Torsion (4 days)
- [ ] **4-Atom Path Encoding** (3 days)
  - Find all linear 4-atom paths (DFS)
  - Encode: (type1, type2, type3, type4)
  - Hash to bit vector

- [ ] **Tests** (1 day)
  - Compare to RDKit topological torsion

### PubChem Fingerprints (3 days)
- [ ] **881 Structural Keys** (2 days)
  - Similar to MACCS but larger
  - Define all SMARTS patterns
  - Sections: elements, rings, simple/complex patterns
  - SMARTS parser ready for complex patterns

- [ ] **Tests** (1 day)
  - Validate against PubChem database

**Phase 9C Output**: Complete fingerprint suite (MACCS + Morgan + AtomPair + Torsion + PubChem)

## 🧬 Phase 10: Medium Complexity Descriptors (3-4 weeks) 🟡

### Graph-Based Descriptors
- [ ] **Chi Indices** (5 days)
  - Chi path (χ⁰, χ¹, χ², χ³, χ⁴)
  - Chi cluster, Chi chain, Chi path-cluster
  - Based on valence delta (δ = valence - H)
  - Kier & Hall connectivity indices

- [ ] **MDE (Molecular Distance Edge)** (4 days)
  - Distance edge vector λ
  - Reference: Liu et al. JCICS 1998

- [ ] **Weighted Path** (3 days)
  - Path descriptors weighted by properties
  - Mass, electronegativity variants

### Electronic/Topological (Pre-Gasteiger)
- [ ] **EState Indices** (5 days)
  - Electrotopological state (Kier & Hall)
  - 75 E-state atom types
  - Graph perturbation: S_i = I_i + ΔI_i
  - MinEState, MaxEState, sum

- [ ] **Autocorrelation ATS** (5 days)
  - Moreau-Broto autocorrelation
  - Mass, polarizability, charge-weighted
  - ATS(d) = Σ(w_i × w_j) for pairs at distance d
  - 192 descriptors total

- [ ] **BCUT Descriptors** (5 days)
  - Burden-CAS-UT eigenvalues
  - Modified adjacency matrix eigenvalues
  - Weights: MW, charge, LogP, refractivity
  - Add `nalgebra` crate for eigendecomposition
  - 8 descriptors: MWHI/LOW, CHGHI/LOW, etc.

**Phase 10 Output**: +30-40 medium complexity descriptors

## 🧬 Phase 11: Electronic Descriptors - COMPLEXITY CLIFF (3-4 weeks) 🔴

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

## 🧬 Phase 9: Additional Descriptors (FUTURE)

### Topological Descriptors
- [ ] Wiener index (sum of shortest paths)
- [ ] Zagreb indices (degree-based)
- [ ] Balaban J index
- [ ] Implement in `desc/topo.rs`
- [ ] Add Floyd-Warshall or BFS for distance matrix
- **Estimated**: 8-10 hours

### Electronic Descriptors (Future)
- [ ] Gasteiger partial charges in `algo/electronic.rs`
- [ ] TPSA (Topological Polar Surface Area)
- [ ] Molar refractivity
- **Estimated**: 8-12 hours

## 🔍 Phase 10: Functional Groups & Substructure

## 🔍 Phase 10: Functional Groups & Substructure

### Functional Group Detection
- [ ] Define common functional groups (SMARTS patterns)
- [ ] Carboxylic acid, amine, alcohol, etc.
- [ ] Create FunctionalGroupDetector descriptor
- [ ] Validate against chemical datasets
- **Estimated**: 1 week
- **Prerequisites**: ✅ SMARTS parser complete

## 📁 Phase 11: Additional I/O

## 📁 Phase 11: Additional I/O

### SDF Parser
- [ ] Implement MOL V2000 parser in `io/sdf.rs`
- [ ] Support V3000 format
- [ ] Parse 3D coordinates
- [ ] Handle property blocks
- [ ] Benchmark parsing speed
- **Estimated**: 1-2 weeks

### InChI Generation
- [ ] Integrate rust-inchi crate
- [ ] Or implement basic InChI layer 1
- [ ] Add InChIKey generation
- [ ] Test with challenging structures
- **Estimated**: 1 week (with external crate)

### Output Formats
- [ ] CSV export with headers
- [ ] JSON export for web APIs
- [ ] MessagePack for efficient storage
- [ ] Add to CLI tool
- **Estimated**: 4-6 hours

## 🤖 Phase 12: Machine Learning Integration

### Linfa Integration
- [ ] Create examples with linfa
- [ ] k-NN classification/regression
- [ ] Random forest example
- [ ] Add to documentation
- **Estimated**: 1 week

### ART 2A Clustering
- [ ] Implement from scratch in `algo/clustering.rs`
- [ ] Follow Carpenter/Grossberg papers
- [ ] Add vigilance parameter tuning
- [ ] Benchmark on chemical datasets
- **Estimated**: 2-3 weeks

### Descriptor Matrix Export
- [ ] Direct ndarray conversion
- [ ] Memory-mapped file support for huge datasets
- [ ] Parallel batch processing improvements
- **Estimated**: 1 week

## 🤖 Phase 12: Machine Learning Integration

### Profiling
- [ ] Profile with cargo-flamegraph
- [ ] Identify hotspots in parsing
- [ ] Memory allocation profiling
- [ ] Create benchmark suite
- **Estimated**: 1 week

### SIMD Optimization
- [ ] Vectorize character classification in parser
- [ ] Use packed_simd or std::simd
- [ ] Benchmark improvements
- **Estimated**: 2 weeks

### Memory Pool
- [ ] Arena allocator for molecules
- [ ] Reuse buffers in batch processing
- [ ] Reduce allocations
- **Estimated**: 1 week

## 🚀 Phase 13: Performance Optimization

### Test Datasets
- [ ] Download ChEMBL subset
- [ ] Create test suite with diverse structures
- [ ] Edge cases: large rings, metal complexes
- [ ] Challenging stereochemistry
- **Estimated**: 1 week

### Validation Against RDKit
- [ ] Compare descriptor values
- [ ] Set tolerance thresholds
- [ ] Document differences
- [ ] Create validation report
- **Estimated**: 2 weeks

### Fuzzing
- [ ] Setup cargo-fuzz for SMILES parser
- [ ] Find edge cases automatically
- [ ] Fix crashes and panics
- **Estimated**: 1 week

## 🧪 Phase 14: Testing & Validation

### API Documentation
- [ ] Complete rustdoc for all public APIs
- [ ] Add examples to doc comments
- [ ] Generate and publish to docs.rs
- **Estimated**: 1 week

### User Guide
- [ ] Tutorial: "Calculate descriptors in 5 minutes"
- [ ] CLI usage guide
- [ ] API usage patterns
- [ ] Performance tuning guide
- **Estimated**: 1-2 weeks

### Scientific Background
- [ ] Document all algorithms with references
- [ ] Explain descriptor meanings
- [ ] Add interpretation guidelines
- [ ] Chemical validity rules
- **Estimated**: 1 week

## 📚 Phase 15: Documentation

### CI/CD
- [ ] Setup GitHub Actions
- [ ] Automated testing on push
- [ ] Benchmark regression detection
- [ ] Release automation
- **Estimated**: 2-3 days

### Packaging
- [ ] Publish to crates.io
- [ ] Create pre-built binaries
- [ ] Docker container
- [ ] Conda package (optional)
- **Estimated**: 1 week

### Python Bindings (Optional)
- [ ] Create PyO3 bindings
- [ ] pip-installable package
- [ ] NumPy integration
- [ ] Comparison with RDKit
- **Estimated**: 2-3 weeks

## 🛠️ Phase 16: Infrastructure

### 3D Descriptors
- [ ] Conformer generation
- [ ] 3D pharmacophore fingerprints
- [ ] Shape descriptors (USR, USRCAT)
- [ ] Molecular surface area/volume
- **Estimated**: 4-6 weeks

### Quantum Chemistry
- [ ] Extended Hückel theory
- [ ] HOMO/LUMO estimates
- [ ] Improved partial charges (QEq, EEM)
- **Estimated**: 6-8 weeks

### Reaction Informatics
- [ ] Reaction SMARTS support
- [ ] Reaction fingerprints
- [ ] Transformation rules
- **Estimated**: 4-6 weeks

## 🔬 Phase 17: Scientific Features (Future)

### 3D Descriptors
- [ ] Conformer generation
- [ ] 3D pharmacophore fingerprints
- [ ] Shape descriptors (USR, USRCAT)
- [ ] Molecular surface area/volume
- **Estimated**: 4-6 weeks

### Quantum Chemistry
- [ ] Extended Hückel theory
- [ ] HOMO/LUMO estimates
- [ ] Improved partial charges (QEq, EEM)
- **Estimated**: 6-8 weeks

### Reaction Informatics
- [ ] Reaction SMARTS support
- [ ] Reaction fingerprints
- [ ] Transformation rules
- **Estimated**: 4-6 weeks

## 🐛 Known Issues / Improvements

### Parser (Mostly Complete)
- [ ] Better error messages with line/column info
- [x] Validate chemical correctness (valence, etc.) - ✅ Done in SMARTS
- [ ] Handle escaped characters in SMILES

### Descriptors (Complete for Phase 7)
- [x] Molecular weight includes implicit H - ✅ Done
- [x] Accurate H-bond detection - ✅ Done
- [x] LogP estimation - ✅ Done

### Architecture
- [ ] Consider adding 3D coordinates to Atom
- [ ] Add conformer support to Molecule
- [ ] Stereochemistry storage

## 📊 Performance Targets

### Current (Phase 7 Complete + Optimized)
- ✅ SMILES parsing: 1-6M mol/sec
- ✅ MW calculation: 1B calc/sec
- ✅ Memory: ~22 bytes/atom
- ✅ SMARTS matching: ~50k mol/sec (VF2 algorithm)
- ✅ LogP: ~500k-5M mol/sec (10-50x faster with pre-parsing) ✨
- ✅ H-bond: >10M mol/sec
- ✅ Total tests: 159/159 passing
- ✅ Pattern coverage: 111/111 (100%)

### Goals
- [ ] SMILES parsing: >10M mol/sec (with SIMD)
- [ ] Full descriptor suite: >100k mol/sec
- [ ] Process 10M molecule dataset in <2 minutes

## 🎯 Milestones

### v0.1.0 - MVP ✅ (Phase 1 Complete)
- [x] SMILES parser
- [x] Molecular weight
- [x] Tests and benchmarks

### v0.2.0 - Core Algorithms ✅ (Phases 2-3 Complete)
- [x] Implicit hydrogens
- [x] Ring detection
- [x] Aromaticity
- [x] 10+ descriptors

### v0.3.0 - Pattern Matching ✅ (Phases 4-6 Complete)
- [x] SMARTS parser
- [x] Substructure search (VF2)
- [x] CSV batch processing

### v0.4.0 - Advanced Descriptors ✅ (Phase 7 Complete + Optimized)
- [x] 18 molecular descriptors
- [x] H-bond donors/acceptors
- [x] LogP estimation (Wildman-Crippen)
- [x] Lipinski Rule of Five
- [x] Chain descriptors (aliphatic, largest, π-systems)
- [x] Extended SMARTS (111/111 patterns, 100%)
- [x] Pre-parsing optimization (10-50x speedup)
- [x] 187/187 tests passing
- [x] Production-ready performance

### v0.5.0 - ML Ready (Phase 8 In Progress)
- [x] MACCS fingerprints (60/166 keys implemented)
- [ ] Extend to full 166 MACCS keys
- [ ] Morgan/ECFP fingerprints
- [ ] Atom Pair fingerprints
- **Current**: 19 descriptors, 192 tests passing

### v0.6.0 - Production (Future)
- [ ] SDF parser
- [ ] InChI generation
- [ ] Comprehensive tests
- [ ] Performance optimization

### v1.0.0 - Release (Future)
- [ ] Full documentation
- [ ] Performance optimized
- [ ] Validated against RDKit
- [ ] Published to crates.io

---

## 📝 Notes

- Estimated times are for single developer
- Prioritize based on project needs
- Scientific accuracy is paramount
- Performance is secondary to correctness
- Consider external crates for complex algorithms (InChI, SMARTS)

**Last Updated**: February 10, 2026 12:30 UTC  
**Status**: Phase 8 Complete + SmallRing Descriptor Added  
**Tests**: 212/212 passing ✅  
**Descriptors**: 20 molecular descriptors (+ MACCS Keys 166-bit fingerprint)  
**Next Priority**: Phase 9A - Easy topological descriptors (~35 descriptors in 2-3 weeks)  
**Strategy**: Easy wins first (40-50 descriptors) before Gasteiger charges (hard, 2 weeks)


