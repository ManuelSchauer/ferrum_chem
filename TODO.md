# TODO List - FerrumChem Development

## ⚠️ Thread Safety Requirements

**ALL implementations must be thread-safe:**
- Data structures: Must implement `Send + Sync`
- Algorithms: Must not use mutable global state
- Descriptors: Stateless or use only immutable references
- Parsers: Independent state per invocation

See [`THREAD_SAFETY.md`](THREAD_SAFETY.md) for guidelines.

## 📌 March 2026 RDKit Parity Snapshot

All parity validation should be run in `--release` mode.

- Scalar parity: **10976/11000 = 99.78%**
- Ring parity: **1919/2000 = 95.95%**
- VSA parity: **17975/18000 = 99.86%**
- Chi/Kappa parity: **4905/5000 = 98.10%**
- Exact on the 500-row RDKit slice: **AromaticAtomCount, AromaticRingCount, AromaticRings(RT), HBondAcceptor, NumAmideBonds, QED, RotatableBonds**
- Highest-value remaining gaps: **Ipc, AliphaticRings, SaturatedRings, TotalRings, Chi3v, Chi4v, Kappa3**

## 🎯 Active Descriptor Backlog

- [ ] Close the remaining **Ipc** mismatches (current: **476/500**)
- [ ] Align independent ring descriptor semantics: **AliphaticRings 470/500**, **SaturatedRings 479/500**, **TotalRings 470/500**
- [ ] Fix higher-order connectivity descriptors: **Chi3v 470/500**, **Chi4v 464/500**, **Kappa3 471/500**
- [ ] Polish the last non-perfect **PEOE VSA** buckets while preserving the current high-parity baseline
- [ ] Resume **Phase 14 / TUI** work after the descriptor parity cleanup stabilizes

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

## 🧬 Phase 10.5: Missing 2D Descriptors ✅ COMPLETE

### Additional Descriptors from RDKit/CDK Survey ✅
**Goal**: Fill gaps before Phase 11 architecture refactor

#### Batch 1 - Simple counts + fractions ✅ COMPLETE
- [x] **FractionCSP3** - fraction of sp³ carbons (drug-likeness metric)
- [x] **RingTypeCounts** - 5 values (aromatic, aliphatic, saturated, heteroaromatic, heterocyclic)
- [x] **NumAmideBonds** - via SMARTS `[NX3][CX3](=[OX1])`
- [x] **NumHeavyAtoms** - heavy atom count
- [x] **AromaticAtomFraction** - fraction of aromatic atoms

#### Batch 2 - Complexity metrics ✅ COMPLETE
- [x] **HallKierAlpha** - molecular flexibility descriptor (covalent radius deviations)
- [x] **BertzCT** - graph complexity index (Shannon entropy-based)
- [x] **Ipc** - information content (Bonchev-Trinajstić)
- [x] **FragmentComplexity** - elemental diversity (Shannon entropy)

#### Batch 3 - VSA descriptors ✅ COMPLETE
- [x] **SlogP_VSA** - 12 bins of VSA by Crippen logP contribution ranges ⭐⭐⭐
- [x] **SMR_VSA** - 10 bins of VSA by Crippen MR contribution ranges ⭐⭐⭐
- [x] **LabuteASA** - Labute's approximate surface area (1 value)

**Phase 10.5 Output**: 
- 3 new descriptor modules (extra_counts.rs, vsa.rs)
- 12 new descriptors, 35 new values (9+23+3)
- 38 new tests (27+11)
- Total: **503 tests passing** ✅

**Files Created**:
- `crates/ferrum_core/src/desc/extra_counts.rs` - 9 descriptors, 27 tests
- `crates/ferrum_core/src/desc/vsa.rs` - 3 descriptors, 11 tests

**Files Modified**:
- `crates/ferrum_core/src/desc/logp.rs` - Added `pub fn crippen_atom_contributions()` for per-atom logP/MR
- `crates/ferrum_core/src/desc/mod.rs` - Registered extra_counts and vsa modules

**Summary**:
- **62 descriptor implementations** across 41 modules
- **17,696 total feature dimensions** (17,431 fingerprint bits + 265 continuous)
- **Production-ready 2D QSAR library** with essential RDKit/CDK features

## 🧬 Phase 11: Typed Descriptor Refactoring ✅ COMPLETE

### Architecture Refactor ✅
- [x] Implemented `DescriptorValue` enum (Int, Float, IntVec, IntU16Vec, FloatVec, Binary)
- [x] Created `Descriptor` trait with typed returns (`&mut Molecule → DescriptorValue`)
- [x] Implemented typed versions for ALL 60 descriptors
- [x] Memory-optimized types (u8 radius, u16 element counts, BitVec fingerprints)
- [x] 32x memory reduction validated (67.5 GB → 2.13 GB for 1M molecules)
- [x] ML vector builder: `build_ml_vector()` + `build_ml_vectors_parallel()`

### Legacy Removal ✅
- [x] Removed `LegacyDescriptor` trait entirely
- [x] Removed `TypedDescriptor` alias
- [x] Renamed all 60 `*Typed` structs → clean names (e.g. `LogPTyped` → `LogP`)
- [x] Single trait: `Descriptor` with `fn calculate(&self, mol: &mut Molecule) -> DescriptorValue`
- [x] Rewritten batch.rs to use new API
- [x] Deleted dead code (logp_old.rs)
- [x] Extracted computation functions for cross-descriptor reuse
- [x] Added `SingleAtomCount::new("C")` configurable element counter
- [x] 568 unit tests + 14 doc tests passing
- [x] Net -645 lines of code

**Status**: 60 descriptors, 17,696 dimensions, 568 tests ✅

## 🧬 Phase 12.6: Hybridization Refinement & HallKierAlpha 100% ✅ COMPLETE

### Root Cause Analysis
- Systematic HallKierAlpha mismatches traced to `effective_hybridization()` norbs==4 branch
- Three separate rounds of refinement to reach 100%:

### Round 1 — O/S conjugation: P=O/S=O exclusion ✅
- **Problem**: O in phosphate (P=O neighbour) was SP2; should be SP3
- **Fix**: Added adjacency-based scan in norbs==4 O/S arm, excluding hypervalent P/S neighbours
- **Result**: 54.2% → 97.8% pass rate (489/500)
- [x] Ester bridging O (C=O nbr, C non-hypervalent) → SP2 ✓
- [x] Carboxylic-acid OH → SP2 ✓
- [x] Phosphate OH (P=O nbr, P excluded) → SP3 ✓
- [x] Phosphate bridging O (P excluded) → SP3 ✓

### Round 2 — S always SP3 in norbs==4 ✅
- **Problem**: Thioether S adjacent to aromatic ring / C=N called SP2 (α=+0.22 vs correct +0.35)
  - All 0.13-diff failures: `CSc1ccccc1`-type S, `N=C(...)S`-type S in heterocycles
- **Fix**: Pulled S out of the O/S shared arm → `16 => false` (always SP3)
  - S is only SP2 via `has_double` (direct C=S) or `is_aromatic()` (thiophene)
- [x] Thioanisole S (aromatic-C nbr) → SP3 ✓
- [x] Thiazolidine S (C=N nbr) → SP3 ✓

### Round 3 — Sulfonamide N SP3 ✅
- **Problem**: N bonded to S(=O)₂ had `neighbor_has_pi=true` (from S=O bonds) → wrongly SP2 (α=−0.20 vs correct −0.04)
  - All 0.16-diff failures: `CN(S(=O)(=O)...)` sulfonamide N
- **Fix**: N in norbs==4 now uses same adjacency-based P/S-excluding check as O: `7 => o_n_conjugated(idx)`
  - Amide N (N–C=O), aniline N, enamine N remain SP2 ✓
- [x] Sulfonamide N (N–S(=O)₂) → SP3 ✓
- [x] Amide N (N–C=O) → SP2 ✓ (regression test)
- [x] Aniline N (aromatic-C nbr) → SP2 ✓ (regression test)

### Tests Added
- [x] `test_hyb_thioanisole_s_sp3` — S adjacent to aromatic ring
- [x] `test_hyb_thiazolidine_s_sp3` — S adjacent to C=N
- [x] `test_hyb_sulfonamide_n_sp3` — N–S(=O)₂
- [x] `test_hyb_amide_n_sp2` — N–C=O regression
- [x] `test_hyb_aniline_n_sp2` — N–C(aromatic) regression
- [x] HallKierAlpha exact-value tests updated (methyl acetate → −0.53)

**Result**: HallKierAlpha **100.0% (500/500)** ✅ — zero regressions across all other descriptors

---

## 🧬 Phase 12: Electronic Descriptors & RDKit Validation ✅ COMPLETE

### Gasteiger Partial Charges ✅
- [x] Gasteiger-Marsili iterative algorithm (12 iterations, geometric damping 0.5^(k+1))
- [x] 26 atom types: H, C×3, N×3, O×2, F, Cl, Br, I, S×4 (sp2/sp3/so/so2), P×2, Si×2, Mg×3, Al×2, B×2
- [x] Conjugation-aware charge splitting (`split_charge_conjugated`) — NO₂, carboxylate, cyclopentadienyl
- [x] Dative bond detection for transition metal complexes (Cisplatin, etc.)
- [x] Per-fragment NaN propagation for atoms without parameters
- [x] Extended element tables: covalent radii & outer electrons Z=1–83, `is_transition_metal()`

### Charge-Derived Descriptors ✅
- [x] MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge, MinAbsPartialCharge
- [x] GasteigerCharges (per-atom vector)
- [x] APol (sum atomic polarizabilities), BPol (bond polarizability differences)
- [x] PEOE_VSA (14 charge-weighted surface area bins)

### RDKit Validation ✅
- [x] **100% accuracy: 1944/1944** descriptor comparisons across 27 torture molecules
- [x] Validated: scalar (594/594), VSA (972/972), Chi/Kappa (270/270), ring (108/108)
- [x] Torture set: Cisplatin, nitrobenzene, Grignard, cyclopentadienyl, sodium acetate, pyrene, etc.
- [x] 626 unit + 14 doc + 4 integration + 55 SMARTS edge case tests

**Status**: 72 descriptors, 100% RDKit match ✅

## 🧬 Phase 12.5: Algorithm Completions ✅ COMPLETE

### Connected Components (algo/components.rs) ✅
- [x] `connected_components()` — BFS graph traversal, returns component labels
- [x] `num_fragments()` — count of disconnected fragments
- [x] `keep_largest_fragment()` — salt/solvent stripping with index remapping
- [x] 6 tests (single/multi fragment, keep largest, salt strip)

### Morgan Canonicalization (algo/canon.rs) ✅
- [x] `rank_atoms()` — Morgan iterative algorithm (polynomial hash refinement)
- [x] `reorder_molecule()` — permutes atoms/bonds/adjacency by rank
- [x] 6 tests (methane, ethane symmetry, propane, ethanol, reorder, benzene)

### MACCS Keys 125 & 166 ✅
- [x] Key 125 (Aromatic Ring > 1): clones mol, calls find_sssr, counts aromatic rings
- [x] Key 166 (Fragments > 1): calls num_fragments() > 1
- [x] All TODO comments removed from maccs.rs

### algo/mod.rs ✅
- [x] Made `canon` and `components` modules public

**Status**: All `todo!()` calls removed except `sdf.rs` (deferred to Phase 15). 641 tests passing ✅

## 🤖 Phase 13: ML Pipeline — All From Scratch ✅ COMPLETE

### 13A: Data Infrastructure ✅
- [x] `QsarDataset` struct (f64 ndarray, targets, feature names, f32→f64 boundary)
- [x] `StandardScaler` (Z-score: μ/σ persistence for inference)
- [x] `MinMaxScaler` (alternative scaling)
- [x] Feature selection: variance threshold, correlation filter (r > 0.95)
- [x] PCA dimensionality reduction (Jacobi eigendecomp from scratch)
- [x] Train/test splitting: random, stratified, **scaffold-based**

### 13B: Supervised Models — From Scratch ✅
- [x] **k-NN** (~240 lines) — Tanimoto for fingerprints, Euclidean for descriptors
- [x] **Random Forest** (~465 lines) — decision tree + bagging, binary feature optimization
- [x] **PLS Regression** (~246 lines, NIPALS) — handles collinearity
- [x] **Elastic Net / LASSO** (~295 lines, coordinate descent) — automatic feature selection
- [x] **Classification variants** — RF classifier, k-NN classifier

### 13C: Custom ART-2A ✅
- [x] ART-2A: vigilance ρ, learning rate η, L2-normalized input
- [x] Chemical space clustering, applicability domain

### 13D: Pipeline, Inference & Optimization ✅
- [x] `QsarPipeline` serializable struct (scaler + PCA + model + AD + metadata)
- [x] Hand-tunable hyperparameters + built-in grid search optimization (via AutoQsar)
- [x] `AutoQsar` — train all models, compare via CV, report best
- [x] Applicability Domain trait: k-NN distance AD, leverage AD
- [ ] Conformal prediction (uncertainty quantification) — deferred to future enhancement
- [x] K-Fold CV, Leave-One-Out, scaffold-based CV
- [x] Evaluation metrics: R², Q², RMSE, MAE, MCC, AUC-ROC
- [x] Y-randomization validation
- [x] Batch prediction (parallel, 1M+ molecules)

**Dependencies**: `ndarray` 0.16 only — no linfa  
**Status**: ✅ Complete — All core ML pipeline components implemented from scratch

## 🖥️ Phase 14: Terminal UI (TUI) 🔵

- [ ] Interactive model training wizard
- [ ] Batch prediction interface
- [ ] Chemical space visualization (ASCII plots)
- [ ] Descriptor profiling dashboard
- [ ] Pipeline management (save/load/compare)

## 🔧 Phase 15: Infrastructure & Ecosystem 🔵

- [ ] SDF/MOL V2000/V3000 parser (sdf.rs stub exists with `todo!()`)
- [x] Morgan canonicalization (rank_atoms + reorder_molecule implemented in algo/canon.rs)
- [ ] Canonical SMILES writer (uses canon.rs ranking)
- [ ] PyO3 Python bindings
- [ ] crates.io publish
- [ ] Full API documentation + benchmarks vs RDKit

## 🧊 Phase 16: 3D Descriptors — SPECIAL CASES 🔴

⚠️ Deferred: Slower computation, only for specialized models

- [ ] Conformer generation (ETKDG algorithm)
- [ ] 3D shape descriptors (PMI, asphericity, spherocity)
- [ ] 3D pharmacophore fingerprints
- [ ] Molecular surface/volume
- [ ] WHIM/GETAWAY descriptors

**Note**: 2D + ML pipeline covers 90%+ of QSAR use cases.

---

## 🐛 Known Issues / Improvements

### Parser
- [x] SMILES bracket atoms: `no_implicit_h` flag prevents incorrect H calculation
- [x] Aromatic bond assignment: only set Aromatic if BOTH atoms aromatic
- [ ] Better error messages with line/column info
- [ ] Handle escaped characters in SMILES
- [ ] Stereochemistry (E/Z, R/S)

### Architecture
- [x] Typed descriptor API (single Descriptor trait, DescriptorValue enum)
- [x] Memory-efficient fingerprints (BitVec)
- [x] Morgan canonicalization (rank_atoms + reorder_molecule in algo/canon.rs)
- [x] Connected components / fragment handling (algo/components.rs)
- [ ] Consider adding 3D coordinates to Atom
- [ ] Add conformer support to Molecule
- [ ] SDF/MOL V2000/V3000 parser (sdf.rs stub exists)
- [ ] InChI generation

### ML Integration (Phase 13) ✅
- [x] QsarDataset + Scalers + Feature Selection
- [x] k-NN, RF, PLS, Elastic Net (all from scratch, no linfa)
- [x] ART 2A clustering + applicability domain
- [x] QsarPipeline + AutoQsar + grid search optimization
- [x] K-Fold CV + evaluation metrics
- [ ] PyO3 Python bindings (Phase 15)

## 📊 Performance

### Current (March 2026 parity snapshot)
- ✅ SMILES parsing: 1-6M mol/sec
- ✅ MW calculation: 1B calc/sec
- ✅ Memory: ~22 bytes/atom, 32x reduction via typed descriptors
- ✅ SMARTS matching: ~50k mol/sec (VF2 algorithm)
- ✅ LogP: ~500k-5M mol/sec (pre-parsed patterns)
- ✅ TPSA: RDKit-exact (50/50 molecules ±0.01 Ų)
- ✅ Release-mode validation is the default for parity runs
- ✅ Total descriptors: 72 implementations, 17,710+ dimensions
- ✅ RDKit torture-set validation: 1944/1944 exact matches across 27 molecules
- ✅ RDKit 500-row parity: scalar 99.78%, ring 95.95%, VSA 99.86%, Chi/Kappa 98.10%
- ✅ Fingerprints: MACCS (166), PubChem (881), ECFP (6144), FCFP (6144), AtomPair (2048), Torsion (2048)

## 🎯 Milestones

### v0.1.0 - v0.6.0 ✅ Complete
- [x] SMILES/SMARTS parser, core algorithms, 35+ descriptors
- [x] MACCS 166-bit, ECFP/FCFP, AtomPair, Torsion, PubChem 881

### v0.7.0 - Phase 10-11 ✅ Complete
- [x] Chi, EState, BCUT, Autocorrelation, MDE, WeightedPath
- [x] VSA descriptors, extra counts, complexity metrics
- [x] Typed descriptor refactoring (single API, 32x memory reduction)

### v0.8.0 - Phase 12 ✅ Complete
- [x] Gasteiger partial charges (26 atom types, conjugation splitting, dative bonds)
- [x] Charge-derived descriptors (polarizability, PEOE_VSA)
- [x] 100% RDKit validation (1944/1944 across 27 torture molecules)

### v0.8.1 - Phase 12.6 ✅ Complete
- [x] Hybridization refinement: S norbs==4 always SP3; sulfonamide N uses P/S-excluding adjacency check
- [x] HallKierAlpha: 54.2% → 97.8% → **100.0%** (500/500) on 500-molecule COCONUT set
- [x] 5 new hybridization unit tests (thioanisole S, thiazolidine S, sulfonamide N, amide N, aniline N)
- [x] 740 unit tests passing, zero regressions

### v0.9.0 - Phase 13: ML Pipeline ✅ COMPLETE
- [x] QsarDataset + Scalers + Feature Selection
- [x] k-NN, Random Forest, PLS, Elastic Net (all from scratch)
- [x] ART-2A + Applicability Domain
- [x] QsarPipeline + AutoQsar + Grid Search + Metrics

### v0.9.1 - RDKit Parity Push ✅ CURRENT
- [x] SMARTS composite bond constraints and disconnected components aligned with RDKit
- [x] Aromatic atom/ring metrics closed at 500/500 on the 500-row slice
- [x] QED, NumAmideBonds, and RotatableBonds closed at 500/500
- [x] Scalar parity raised to **10976/11000 = 99.78%**
- [ ] Finish descriptor-local cleanup for Ipc, independent ring semantics, and higher-order Chi/Kappa

### v0.10.0 - Phase 14: TUI
- [ ] Interactive model training wizard
- [ ] Batch prediction interface

### v1.0.0 - Production (Future)
- [ ] SDF parser, canonicalization, Python bindings
- [ ] Full documentation, benchmarks, crates.io publish
- [ ] 3D descriptors (optional, special cases)

---

**Last Updated**: March 2026  
**Status**: RDKit parity push in progress — scalar **99.78%**, ring **95.95%**, VSA **99.86%**, Chi/Kappa **98.10%**  
**Validation**: release-mode parser / SMARTS / QED / rotatable tests plus scalar, ring, VSA, and Chi/Kappa parity suites ✅  
**Descriptors**: 72 implementations, 17,710+ dimensions  
**RDKit 500-row wins**: AromaticAtomCount 100% · AromaticRingCount 100% · AromaticRings(RT) 100% · HBondAcceptor 100% · NumAmideBonds 100% · QED 100% · RotatableBonds 100%  
**ML Pipeline**: QsarDataset, Scalers, PCA, k-NN, RF, PLS, ElasticNet, ART-2A, AD, QsarPipeline, AutoQsar, CV, Metrics  
**API**: Single `Descriptor` trait with `DescriptorValue` enum returns  
**Next Priority**: Descriptor-local cleanup for **Ipc**, **Aliphatic/Saturated/TotalRings**, and **Chi3v/Chi4v/Kappa3**

