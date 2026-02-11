# FerrumChem - Project Plan

## Overview
High-performance QSAR (Quantitative Structure-Activity Relationship) engine for cheminformatics in Rust, designed for processing millions of molecules with maximum performance.

## Project Structure

```
ferrum_chem/
├── crates/
│   ├── ferrum_core/          # Core algorithms and data structures
│   │   ├── src/
│   │   │   ├── mol/          # Molecular data structures
│   │   │   │   ├── atom.rs         # ✅ Atom representation (12 bytes, cache-optimized)
│   │   │   │   ├── bond.rs         # ✅ Bond representation with aromatic flags
│   │   │   │   ├── molecule.rs     # ✅ Molecule graph with SmallVec adjacency
│   │   │   │   └── elements.rs     # ✅ Atomic data lookup tables
│   │   │   ├── io/           # Input/Output parsers
│   │   │   │   ├── builder.rs      # ✅ MoleculeBuilder trait
│   │   │   │   ├── smiles.rs       # ✅ SMILES parser (bracket atoms, fused rings, auto-aromaticity)
│   │   │   │   ├── smarts.rs       # ✅ SMARTS parser with VF2 matching + recursive SMARTS
│   │   │   │   ├── csv_reader.rs   # ✅ CSV SMILES batch reader
│   │   │   │   └── sdf.rs          # TODO: SDF/MOL file parser
│   │   │   ├── algo/         # Algorithms
│   │   │   │   ├── aromaticity.rs  # ✅ Hückel 4n+2 rule (enhanced for fused heterocycles)
│   │   │   │   ├── rings.rs        # ✅ SSSR algorithm
│   │   │   │   ├── iso.rs          # ✅ VF2 subgraph isomorphism
│   │   │   │   ├── valency.rs      # ✅ Implicit hydrogen calculation (bracket-aware)
│   │   │   │   ├── kekulize.rs     # ✅ Optional kekulization module
│   │   │   │   ├── canon.rs        # TODO: Morgan canonicalization
│   │   │   │   ├── electronic.rs   # ✅ Gasteiger partial charges (RDKit-validated)
│   │   │   │   └── standardize.rs  # TODO: Standardization pipeline
│   │   │   └── desc/         # Descriptor calculations (~30 modules)
│   │   │       ├── molecular_weight.rs  # ✅ Molecular weight
│   │   │       ├── atom_count.rs        # ✅ Element counting
│   │   │       ├── bond_count.rs        # ✅ Bond type counting
│   │   │       ├── aromatic_ring_count.rs # ✅ Ring counting
│   │   │       ├── hbond.rs            # ✅ H-bond donors/acceptors
│   │   │       ├── logp.rs             # ✅ LogP (Wildman-Crippen)
│   │   │       ├── formal_charge.rs    # ✅ Net charge
│   │   │       ├── lipinski.rs         # ✅ Lipinski Rule of Five
│   │   │       ├── chains.rs           # ✅ Aliphatic/largest chain
│   │   │       ├── pi_system.rs        # ✅ Largest π system
│   │   │       ├── maccs.rs            # ✅ MACCS 166-bit fingerprint
│   │   │       ├── small_ring.rs       # ✅ Small ring counts (3-9 membered)
│   │   │       ├── tpsa.rs             # ✅ TPSA (RDKit-exact, 50/50 validation)
│   │   │       ├── kappa.rs            # ✅ Kier-Hall Kappa shape indices
│   │   │       ├── functional_groups.rs # ✅ Functional/Acidic/Basic group counts
│   │   │       ├── qed.rs              # ✅ QED drug-likeness score
│   │   │       ├── wiener_numbers.rs   # ✅ Wiener index + distance matrix
│   │   │       ├── zagreb.rs           # ✅ Zagreb indices M1, M2
│   │   │       ├── balaban.rs          # ✅ Balaban J index
│   │   │       ├── eccentric.rs        # ✅ Eccentric connectivity index
│   │   │       ├── simple_counts.rs    # ✅ Valence/radical/heteroatom counts
│   │   │       ├── carbon_types.rs     # ✅ C1SP1..C4SP3 classification
│   │   │       ├── ring_types.rs       # ✅ Saturated/aliphatic ring counts
│   │   │       ├── spiro.rs            # ✅ Spiro atom count
│   │   │       ├── petitjean.rs        # ✅ Petitjean indices
│   │   │       ├── gravitational.rs    # ✅ Gravitational index
│   │   │       ├── batch.rs            # ✅ Rayon parallel processing
│   │   │       ├── fingerprint.rs      # ✅ ECFP/FCFP circular fingerprints + Tanimoto
│   │   │       └── topo.rs             # TODO: Additional topological
│   │   └── Cargo.toml
│   └── ferrum_cli/           # Command-line interface
│       ├── src/
│       │   └── main.rs       # CLI with clap
│       └── Cargo.toml
└── Cargo.toml                # Workspace definition
```

## ✅ Completed (Phase 1-8: Core Functionality + MACCS Keys)

### Phase 1: SMILES Parser & Molecular Weight
- **Status**: ✅ Complete
- Element data lookup tables
- SMILES parser (organic subset, rings, branches, aromatics, bracket atoms)
- MoleculeBuilder trait
- Molecular weight descriptor
- Unit tests (13 tests)

### Phase 2-6: Core Algorithms & Descriptors
- **Status**: ✅ Complete (Phases 2, 3, 4, 5, 6)
- Implicit hydrogen calculation (valence-based, bracket atom aware)
- Ring detection (SSSR algorithm)
- Aromaticity perception (Hückel 4n+2 rule, enhanced for fused heterocycles)
- Kekulization module (optional, available but not used by default)
- CSV reader for batch SMILES processing
- 14 molecular descriptors:
  - Molecular weight
  - Atom counts (total, heavy, by element)
  - Bond counts (total, by type)
  - Aromatic atom/ring counting
  - H-bond donors/acceptors
  - LogP (Wildman-Crippen with 111 SMARTS patterns)
  - Rotatable bonds
  - Formal charge

### Phase 7: SMARTS Parser & Pattern Matching ✅ **COMPLETE**
- **Status**: ✅ Complete (February 4, 2026)
- **Tests**: 159 passing (13 SMARTS tests, 27 LogP tests)
- **Pattern Coverage**: 111/111 Crippen patterns (100%)

**SMARTS Features Implemented**:
1. ✅ Comma OR lists: `[#6,#7,#8]`, `[N,O,P,S]`
2. ✅ Implicit AND: `[CH3]`, `[cH]`, `[CH2X4]`
3. ✅ Branching: `C(C)C`, `[CH2](C)C`, `C(C)(C)C`
4. ✅ NOT operator: `[!#6]`, `[!C]`
5. ✅ Lowercase 'a': Any aliphatic atom
6. ✅ Atomic number notation: `[#6]`, `[#7]`
7. ✅ H-count queries: `[H1]`, `[h2]`
8. ✅ Degree/connectivity: `[D2]`, `[X4]`
9. ✅ Valence calculation: `[Cv4]`, `[Nv3]`
10. ✅ Ring bonds counting: `[Cx2]`, `[Cx3]`
11. ✅ Ring membership: `[R]`, `[R0]`, `[r5]`, `[r6]`
12. ✅ Charge queries: `[+]`, `[-]`, `[+2]`, `[-1]`

**VF2 Subgraph Isomorphism**:
- State-space search with feasibility checking
- Supports all SMARTS primitives
- Performance: ~50,000 molecules/sec

**LogP Implementation**:
- 111 Wildman-Crippen patterns from RDKit
- SMARTS-based atom typing
- Implicit hydrogen contributions
- Validation: 7 exact matches with RDKit

**Documentation Created**:
- `SMARTS_PARSER_COMPLETE.md` - Technical details
- `SMARTS_VALIDATION_REPORT.md` - Validation results  
- `SMARTS_TODOS_RESOLVED.md` - TODOs resolution
- `SMARTS_OPTIMIZATION_COMPLETE.md` - Pre-parsing optimization (10-50x speedup)
- `PHASE7_COMPLETE_OPTIMIZED.md` - Comprehensive Phase 7 summary
- `PHASE7_2_LOGP_COMPLETE.md` - LogP implementation
- `PHASE7_LOGP_VALIDATION.md` - LogP validation

**Performance Optimizations**:
- Pre-parsed patterns using `once_cell::Lazy`
- 10-50x speedup for pattern matching (eliminates parsing overhead)
- One-time startup cost (~1-5ms for 111 patterns)
- Thread-safe with zero locking overhead after initialization
- LogP: 50k/sec → 500k-5M/sec
- Memory overhead: ~19 KB (negligible)

**Ready For**: MACCS keys (166 patterns), Morgan fingerprints, custom pattern matching

## 🔄 Next Steps - Detailed Roadmap (Phases 9-11)

**Strategy**: "Easy wins first, hard problems last"
Before tackling Gasteiger charges (2 weeks) or 3D conformers (3 weeks), we have **40-50 descriptors** ready to implement with 1-3 days effort each.

### Phase 9A: Easy Topological Descriptors 🟢
**Goal**: Maximum value with minimum effort (+35 descriptors)
**Status**: ✅ Complete (35/35 descriptors implemented, 395 tests)

#### Distance-Based (Foundation for others) ✅ COMPLETE
- ✅ Wiener Index (10 tests) - sum of shortest paths, reusable distance matrix
- ✅ Zagreb Indices M1, M2 (9 tests) - degree-based
- ✅ Balaban J Index (8 tests) - distance matrix based
- ✅ Eccentric Connectivity Index (9 tests)
- ✅ Kier-Hall Kappa Indices (8 tests) - shape descriptors κ₁, κ₂, κ₃

#### Simple Counting ✅ COMPLETE
- ✅ Valence Electrons Count (12 tests)
- ✅ Radical Electrons Count (in simple_counts.rs)
- ✅ Heteroatom Count (in simple_counts.rs)
- ✅ Spiro Atom Count (7 tests)
- ✅ Carbon Types (9 tests) - C1SP1..C4SP3
- ✅ Ring Type Counts (7 tests) - saturated/aliphatic

#### High-Value Physicochemical
- ✅ **TPSA** ⭐⭐⭐ (15 tests) - RDKit-exact algorithm, 50/50 molecules validated
- ✅ **Molar Refractivity** (5 tests) - Crippen method, reuses LogP patterns

#### Fragment-Based Pattern Matching (SMARTS Ready!)
- ✅ **Functional Group Counts** (6 tests) - 16 SMARTS patterns
- ✅ **Acidic Group Count** (3 tests)
- ✅ **Basic Group Count** (3 tests)
- ✅ **QED (Drug-likeness)** (6 tests) - combines existing descriptors

#### Additional Useful ✅ COMPLETE
- ✅ Petitjean Indices (8 tests)
- ✅ Gravitational Index (7 tests)
- ✅ Hybridization Ratio - sp³/sp² carbon ratio
- ✅ VAdjMa - 1 + log₂(m)

**Completed**: 35 descriptors, 345 tests passing ✅

### Phase 9B: Morgan/ECFP Fingerprints ✅ COMPLETE
**Goal**: Core ML fingerprint ⭐⭐⭐
**Status**: ✅ Complete (25 tests, ECFP2/4/6 + FCFP2/4/6 + Tanimoto similarity)

- ✅ Morgan Algorithm Core - iterative atom environment hashing with configurable radius
- ✅ Hashing Infrastructure - std DefaultHasher, no external crate needed
- ✅ Bit Folding - 1024/2048/4096 bit sizes, Vec<f32> with 0.0/1.0
- ✅ ECFP Variants - ECFP2 (radius=1), ECFP4 (radius=2), ECFP6 (radius=3)
- ✅ FCFP Variants - FCFP2/4/6 with pharmacophoric atom invariants
- ✅ Tanimoto Similarity function
- ✅ Tests & Validation - 25 tests (ECFP + FCFP: symmetry, similarity, drug-like molecules)

### Phase 9C: Additional Fingerprints ✅ COMPLETE
**Status**: ✅ Complete (25 tests)

- ✅ **Atom Pair Fingerprints** - distance-based encoding (7 tests)
- ✅ **Topological Torsion** - 4-atom path encoding (8 tests)
- ✅ **PubChem Fingerprints** - 881 structural keys, all SMARTS parse successfully (10 tests)
- ✅ **SMARTS Parser Enhancement** - added bond OR (`,`) support for `=,:` syntax

### Phase 10: Medium Complexity Descriptors ✅ COMPLETE
**Status**: ✅ Complete (70 tests, 138 values)

- ✅ **Chi Indices** (chi.rs) - 5 descriptors: ChiPath(8), ChiValencePath(8), ChiCluster(2), ChiChain(5), ChiPathCluster(3) = 26 values, 25 tests
- ✅ **MDE (Molecular Distance Edge)** (mde.rs) - 15 atom-pair type values, 8 tests
- ✅ **Weighted Path** (weighted_path.rs) - 10 values (mass + EN weighted, paths 1-5), 6 tests
- ✅ **EState Indices** (estate.rs) - EStateSummary(3) + EStateAtomTypes(44) = 47 values, 13 tests
- ✅ **Autocorrelation ATS** (autocorrelation.rs) - 32 values (4 weights × 8 lags), 9 tests
- ✅ **BCUT Descriptors** (bcut.rs) - 8 eigenvalues (4 weights × hi/lo), Jacobi eigendecomposition (no nalgebra), 9 tests

### Phase 10.5: Missing 2D Descriptors ✅ COMPLETE
**Goal**: Fill gaps before Phase 11 architecture refactor
**Status**: ✅ Complete (38 tests, 35 values)

#### Batch 1 - Simple counts + fractions ✅
- ✅ **FractionCSP3** - fraction of sp³ carbons (drug-likeness)
- ✅ **RingTypeCounts** - 5 values (aromatic, aliphatic, saturated, heteroaromatic, heterocyclic)
- ✅ **NumAmideBonds** - SMARTS pattern matching
- ✅ **NumHeavyAtoms** - heavy atom count
- ✅ **AromaticAtomFraction** - fraction aromatic

#### Batch 2 - Complexity metrics ✅
- ✅ **HallKierAlpha** - molecular flexibility (covalent radius deviations)
- ✅ **BertzCT** - graph complexity (Shannon entropy)
- ✅ **Ipc** - information content (Bonchev-Trinajstić)
- ✅ **FragmentComplexity** - elemental diversity

#### Batch 3 - VSA descriptors ✅
- ✅ **SlogP_VSA** - 12 bins (logP-weighted surface area) ⭐⭐⭐
- ✅ **SMR_VSA** - 10 bins (MR-weighted surface area) ⭐⭐⭐
- ✅ **LabuteASA** - approximate surface area

**Output**: 3 new modules (extra_counts.rs, vsa.rs), 12 new descriptors, 38 tests

---

### Phase 9C-10: ✅ Complete (see above)

### Phase 11: Typed Descriptor Refactoring ✅ COMPLETE
**Status**: ✅ Complete (February 11, 2026)

**Architecture overhaul:**
- Single `Descriptor` trait with `DescriptorValue` enum returns
- Removed legacy dual-trait API (`LegacyDescriptor` + `TypedDescriptor`)
- Renamed all 60 `*Typed` structs → clean names (e.g. `LogPTyped` → `LogP`)
- Memory-optimized types: BitVec fingerprints (31x), u32 counts, f32 floats
- 32x total memory reduction (67.5 GB → 2.13 GB for 1M molecules)
- Extracted computation functions for cross-descriptor reuse
- Added `SingleAtomCount::new("C")` configurable element counter
- Net -645 lines of code, 568 unit tests + 14 doc tests passing

**Output**: Clean single-API architecture, 60 descriptors, production-ready typed system

### Phase 12: Electronic Descriptors ✅ COMPLETE
**Status**: ✅ Complete (February 11, 2026)

#### 12A: Gasteiger Partial Charges ✅
- Gasteiger-Marsili iterative electronegativity equalization (12 iterations, geometric damping 0.5^(k+1))
- 15 atom types with (a, b, c) coefficients from RDKit `GasteigerParams.cpp`:
  H, C×3 (sp/sp2/sp3), N×3 (sp/sp2/sp3), O×2 (sp2/sp3), F, Cl, Br, I, S×2, P
- Hybridization from bond types + RDKit conjugation logic (norbs=4, degree≤3, conjugated neighbor → SP2)
- Special hydrogen ionization constant IONXH = 20.02 (RDKit `GasteigerParams.h`)
- RDKit cross-validated: 17 molecules within ±0.05 per atom
- 28 tests (12 basic + 16 RDKit validation)

#### 12B: Charge-Derived Descriptors ✅
- MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge, MinAbsPartialCharge (4 scalar descriptors)
- GasteigerCharges (per-atom charge vector)
- APol (sum of atomic polarizabilities), BPol (bond polarizability differences)
- PEOE_VSA (14 charge-weighted surface area bins)
- 26 new tests, 67 total descriptor implementations

**Output**: 8 new descriptors + 14 PEOE_VSA bins, 622 unit tests + 14 doc tests passing

### Phase 13: ML Pipeline 🔵 NEXT
**Goal**: Full QSAR training & prediction pipeline — all from scratch, no linfa

#### 13A: Data Infrastructure
- `QsarDataset` struct: features (f64 ndarray), targets, names, f32→f64 conversion boundary
- `StandardScaler` (Z-score) + `MinMaxScaler` — must store params for inference
- Feature selection: variance threshold, correlation filter (r > 0.95 removal)
- PCA dimensionality reduction (reuse Jacobi eigendecomp from BCUT)
- Train/test splitting: random, stratified, **scaffold-based** (most important for QSAR)

#### 13B: Supervised Models — All From Scratch
- **k-NN** (~100 lines) — Tanimoto for fingerprints, Euclidean for descriptors. Doubles as AD method
- **Random Forest** (~600 lines) — decision tree + bagging, optimized for binary fingerprint features, gives feature importance
- **PLS Regression** (~200 lines, NIPALS) — handles collinearity in mega-vector
- **Elastic Net / LASSO** (~250 lines, coordinate descent) — automatic feature selection with named features
- **Classification variants**: RF classifier, k-NN classifier (active/inactive)
- **Evaluation metrics**:
  - Regression: R², Q² (cross-validated), RMSE, MAE
  - Classification: Accuracy, MCC (Matthews), AUC-ROC, Sensitivity/Specificity
  - Y-randomization (QSAR validation gold standard)

#### 13C: Custom ART-2A (Unsupervised)
- Adaptive Resonance Theory 2A (~200 lines)
- Vigilance parameter ρ, learning rate η, L2-normalized input
- Activation → Competition → Vigilance check → Weight update/new cluster
- Use cases: chemical space clustering, applicability domain definition

#### 13D: Pipeline, Inference & Optimization
- `QsarPipeline` serializable struct: scaler + optional PCA + model + AD + metadata
- Modular: hand-tunable hyperparameters OR built-in grid search optimization
- `AutoQsar`: train all models, compare via CV, report best
- Applicability Domain trait: k-NN distance AD, leverage-based AD (Williams plot), ART-2A AD
- K-Fold CV, Leave-One-Out, scaffold-based CV
- Conformal prediction for uncertainty quantification
- Batch prediction (parallel + streaming for 1M+ molecules)

**Dependencies**: `ndarray` 0.16 only — no linfa (~1750 lines total, all from scratch)

### Phase 14: Terminal UI (TUI) 🔵
**Goal**: Interactive terminal interface for QSAR workflows

- Model training wizard (dataset → preprocess → train → evaluate)
- Batch prediction interface
- Chemical space visualization (ASCII scatter plots)
- Descriptor profiling dashboard
- Pipeline management (save/load/compare models)

### Phase 15: Infrastructure & Ecosystem 🔵
- SDF/MOL V2000/V3000 parser
- Morgan canonicalization (canonical SMILES)
- PyO3 Python bindings
- crates.io publish
- Full API documentation + benchmarks vs RDKit

### Phase 16: 3D Descriptors — SPECIAL CASES 🔴
⚠️ **Deferred**: Slower computation, only for specialized QSAR models
- Conformer generation (ETKDG algorithm)
- 3D shape descriptors (PMI, asphericity, spherocity)
- 3D pharmacophore fingerprints
- Molecular surface/volume
- WHIM/GETAWAY descriptors

**Note**: 2D descriptors + ML pipeline covers 90%+ of QSAR use cases. 3D only needed for shape-dependent SAR or docking-derived models.

### Mega-Vector Target (~5000+ dimensions) ✅ EXCEEDED
- Binary fingerprints: **17,431 bits** (MACCS 166 + Morgan 12288 + AtomPair 2048 + Torsion 2048 + PubChem 881)
- Continuous descriptors: **265 floats** (62 implementations)
- **Total: 17,696 dimensions** 🎉
- **Status**: Comprehensive 2D QSAR suite complete

### Milestone Summary
- **Phase 1-11** ✅ Complete: 60 descriptors, 17,696 dims, 32x memory reduction, typed API
- **Phase 12** ✅ Complete: Gasteiger charges + charge-derived descriptors (67 descriptors, 622 tests)
- **Phase 13**: Full ML pipeline (train/predict/evaluate) 🔵
- **Phase 14**: TUI for interactive QSAR workflows 🔵
- **Phase 15**: SDF parser, Python bindings, crates.io 🔵
- **Phase 16**: 3D descriptors (special cases only) 🔴

### Phase 4-6: SMARTS, I/O, ML Design — ✅ Complete
- ✅ SMARTS parser + VF2 matching
- ✅ CSV reader + batch processing
- ✅ Feature vector design (unified f32, 17,696 dims)
- ✅ `build_ml_vector()` + `build_ml_vectors_parallel()` in typed.rs
- See Phase 13 for full ML pipeline implementation

## Performance Targets

- **SMILES Parsing**: >100k molecules/second (single-threaded)
- **Descriptor Calculation**: >1M molecules/minute (multi-threaded with Rayon)
- **Memory**: <1KB per molecule average
- **Batch Processing**: Scale to 10M+ molecule datasets

## Performance Optimization Strategies

### Current Optimizations
1. **SmallVec adjacency lists**: No heap allocation for typical molecules
2. **Const lookup tables**: Zero runtime cost
3. **Compact atom representation**: 8 bytes per atom
4. **u16 indices**: Max 65k atoms (sufficient for drug-like molecules)
5. **Rayon parallelism**: Automatic work stealing

### Future Optimizations
1. **SIMD for parsing**: Vectorized character classification
2. **Arena allocation**: Bulk molecule allocation for parsers
3. **Memory pooling**: Reuse molecule buffers
4. **Cache-oblivious algorithms**: Better CPU cache utilization
5. **Profile-guided optimization (PGO)**: With real-world datasets

## Scientific Extensions & Ideas

### Quantum Chemistry Integration
- **Partial Charges**: Better than Gasteiger (QEq, EEM methods)
- **Dipole Moments**: From charge distribution
- **HOMO/LUMO estimation**: Simple Hückel MO theory

### 3D Descriptors (requires conformer generation)
- **3D Pharmacophore fingerprints**
- **Shape descriptors** (USR, USRCAT)
- **Conformer-dependent TPSA**
- **Molecular volume/surface area**

### Advanced Graph Algorithms
- **Maximum Common Substructure (MCS)**: For scaffold analysis
- **Graph kernels**: For kernel-based ML on molecular graphs
- **Message-passing neural networks**: Export to Python/PyTorch

### Reaction Informatics
- **Reaction SMARTS**: Transform molecules
- **Reaction fingerprints**: For reaction prediction
- **Retrosynthesis planning**: Rule-based or ML

### Protein-Ligand Interaction
- **Interaction fingerprints**: From docked poses
- **Pharmacophore matching**: 3D features
- **Docking score approximation**: Fast empirical scoring

### Dataset Analysis Tools
- **Chemical space visualization**: t-SNE/UMAP on descriptors
- **Scaffold analysis**: Murcko frameworks
- **Diversity selection**: MaxMin, OptiSim
- **Outlier detection**: Mahalanobis distance

## Testing Strategy

### Unit Tests (✅ In Progress)
- Parser edge cases (ring closures, charges, isotopes)
- Descriptor accuracy (known molecules)
- Algorithm correctness (benzene aromaticity, etc.)

### Integration Tests
- End-to-end SMILES → Descriptors → CSV
- Large batch processing (10k+ molecules)
- Error handling (malformed input)

### Benchmarks
- Parsing speed comparison (vs RDKit/CDK)
- Descriptor calculation throughput
- Memory usage profiling

### Regression Tests
- **Dataset**: ChEMBL subset (diverse structures)
- **Reference**: Compare to RDKit outputs
- **Tolerance**: ±0.01 for floating-point descriptors

## Dependencies

### Current
- `thiserror`: Error handling
- `smallvec`: Stack-allocated vectors
- `rayon`: Data parallelism
- `clap`: CLI argument parsing

### Planned
- `ndarray`: Matrix operations for ML pipeline
- `bincode`: Model serialization
- `memmap2`: Memory-mapped file I/O for huge datasets
- `ratatui` + `crossterm`: TUI interface (Phase 14)

### Optional (Scientific)
- `nalgebra`: Linear algebra for 3D geometry (Phase 16)
- `petgraph`: Advanced graph algorithms
- `rust-inchi`: InChI generation

## Documentation

### Code Documentation
- Rustdoc for all public APIs
- Algorithm explanations with complexity
- Examples in doc comments

### User Documentation
- README with quick start
- Tutorial: "Calculate descriptors for your dataset"
- Performance guide
- Scientific background (algorithms, descriptor definitions)

### Developer Documentation
- Architecture overview
- Contributing guidelines
- Benchmark reproduction

## Release Milestones

### v0.1.0 - MVP ✅ Complete
- [x] SMILES parser (organic subset)
- [x] Molecular weight
- [x] Implicit hydrogens
- [x] Basic CLI (calc subcommand)

### v0.2.0 - Core Algorithms ✅ Complete
- [x] Ring detection (SSSR)
- [x] Aromaticity perception
- [x] 10+ topological descriptors
- [x] Parallel batch processing

### v0.3.0 - Pattern Matching ✅ Complete
- [x] SMARTS parser
- [x] Substructure search
- [x] MACCS fingerprints
- [x] Functional group detection

### v0.4.0 - Complete 2D Suite ✅ Complete
- [x] 60 descriptors, 17,696 dimensions
- [x] Morgan/ECFP + FCFP + AtomPair + Torsion + PubChem fingerprints
- [x] Typed descriptor API with 32x memory reduction
- [x] CSV export + batch processing

### v0.5.0 - Electronic Descriptors ✅ Complete
- [x] Gasteiger partial charges (RDKit-validated, 17 molecules ±0.05)
- [x] Charge-derived descriptors (polarizability, PEOE_VSA)

### v0.6.0 - ML Pipeline
- [ ] QsarDataset + Scalers + Feature Selection
- [ ] k-NN, RF, PLS, Elastic Net (all from scratch, no linfa)
- [ ] ART-2A + Applicability Domain
- [ ] QsarPipeline + AutoQsar + Grid Search + Metrics

### v0.7.0 - TUI
- [ ] Interactive terminal interface
- [ ] Model training wizard
- [ ] Batch prediction

### v0.8.0 - Infrastructure
- [ ] SDF parser, canonicalization
- [ ] Python bindings (PyO3)
- [ ] crates.io publish

### v1.0.0 - Production Ready
- [ ] Full documentation, benchmarks vs RDKit
- [ ] 3D descriptors (optional)
- [ ] Comprehensive test coverage

## Scientific Validity

### Reference Implementations
- **RDKit** (Python/C++): Industry standard
- **CDK** (Java): Chemistry Development Kit
- **OpenBabel** (C++): Format conversion

### Validation Strategy
1. **Descriptor comparison**: Match RDKit outputs within tolerance
2. **Algorithm verification**: Literature references for all methods
3. **Test datasets**: Use public benchmarks (ChEMBL, PubChem)
4. **Peer review**: Open-source contributions welcome

### Scientific Publications to Reference
- SMILES: Weininger, D. (1988) SMILES
- Morgan algorithm: Morgan, H.L. (1965)
- ECFP: Rogers, D. & Hahn, M. (2010)
- Gasteiger charges: Gasteiger, J. & Marsili, M. (1980)
- ART networks: Carpenter, G.A. & Grossberg, S. (1987)

## Contributing

### Areas for Contribution
1. **Algorithm implementation**: Rings, aromaticity, canonicalization
2. **Descriptor additions**: New chemical descriptors
3. **Performance optimization**: Profiling and speedups
4. **Testing**: Edge cases, validation datasets
5. **Documentation**: Tutorials, examples
6. **Benchmarking**: Comparison studies

### Code Style
- Follow Rust conventions (rustfmt, clippy)
- Document all public APIs
- Add unit tests for new features
- Benchmark performance-critical code

## License
[Specify license: MIT, Apache-2.0, or GPL]

## Contact
[Your contact information]

---

**Last Updated**: February 11, 2026  
**Status**: Phase 12 ✅ Complete — Gasteiger Charges + Electronic Descriptors  
**Tests**: 622 unit + 14 doc tests passing ✅  
**Descriptors**: 67 implementations, 17,710+ dimensions  
**API**: Single `Descriptor` trait with `DescriptorValue` enum returns  
**Next Priority**: Phase 13 — ML Pipeline (from scratch, ndarray only)  
**Code**: ~18,000 lines Rust


