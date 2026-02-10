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
│   │   │   │   ├── atom.rs         # Atom representation (8 bytes, cache-optimized)
│   │   │   │   ├── bond.rs         # Bond representation
│   │   │   │   ├── molecule.rs     # Molecule graph with SmallVec adjacency
│   │   │   │   └── elements.rs     # ✅ DONE: Atomic data lookup tables
│   │   │   ├── io/           # Input/Output parsers
│   │   │   │   ├── builder.rs      # ✅ DONE: MoleculeBuilder trait
│   │   │   │   ├── smiles.rs       # ✅ DONE: SMILES parser (state machine)
│   │   │   │   ├── smarts.rs       # ✅ DONE: SMARTS parser with VF2 matching
│   │   │   │   ├── csv_reader.rs   # ✅ DONE: CSV SMILES batch reader
│   │   │   │   └── sdf.rs          # TODO: SDF/MOL file parser
│   │   │   ├── algo/         # Algorithms
│   │   │   │   ├── aromaticity.rs  # ✅ DONE: Hückel 4n+2 rule
│   │   │   │   ├── rings.rs        # ✅ DONE: SSSR algorithm
│   │   │   │   ├── iso.rs          # ✅ DONE: VF2 subgraph isomorphism
│   │   │   │   ├── valency.rs      # ✅ DONE: Implicit hydrogen calculation
│   │   │   │   ├── canon.rs        # TODO: Morgan canonicalization
│   │   │   │   ├── electronic.rs   # TODO: Gasteiger partial charges
│   │   │   │   └── standardize.rs  # TODO: Standardization pipeline
│   │   │   └── desc/         # Descriptor calculations
│   │   │       ├── molecular_weight.rs  # ✅ DONE: Molecular weight
│   │   │       ├── atom_count.rs        # ✅ DONE: Element counting
│   │   │       ├── bond_count.rs        # ✅ DONE: Bond type counting
│   │   │       ├── aromatic_atom_count.rs # ✅ DONE: Aromatic atoms
│   │   │       ├── aromatic_ring_count.rs # ✅ DONE: Ring counting
│   │   │       ├── hbond.rs            # ✅ DONE: H-bond donors/acceptors
│   │   │       ├── logp.rs             # ✅ DONE: LogP (Wildman-Crippen)
│   │   │       ├── rotatable_bonds.rs  # ✅ DONE: Rotatable bond count
│   │   │       ├── formal_charge.rs    # ✅ DONE: Net charge
│   │   │       ├── batch.rs            # ✅ DONE: Rayon parallel processing
│   │   │       ├── fingerprint.rs      # TODO: ECFP4, Morgan, etc.
│   │   │       └── topo.rs             # TODO: TPSA, Wiener index, etc.
│   │   └── Cargo.toml
│   └── ferrum_cli/           # Command-line interface
│       ├── src/
│       │   └── main.rs       # CLI with clap
│       └── Cargo.toml
└── Cargo.toml                # Workspace definition
```

## ✅ Completed (Phase 1-7: Core Functionality)

### Phase 1: SMILES Parser & Molecular Weight
- **Status**: ✅ Complete
- Element data lookup tables
- SMILES parser (organic subset, rings, branches, aromatics)
- MoleculeBuilder trait
- Molecular weight descriptor
- Unit tests (13 tests)

### Phase 2-6: Core Algorithms & Descriptors
- **Status**: ✅ Complete (Phases 3, 4, 5, 6)
- Implicit hydrogen calculation (valence-based)
- Ring detection (SSSR algorithm)
- Aromaticity perception (Hückel 4n+2 rule)
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

### Phase 9A: Easy Topological Descriptors (2-3 weeks) 🟢
**Goal**: Maximum value with minimum effort (+35 descriptors)

#### Distance-Based (Foundation for others)
- Wiener Index (1 day) - sum of shortest paths
- Zagreb Indices M1, M2 (1 day) - degree-based
- Balaban J Index (2 days) - distance matrix based
- Eccentric Connectivity Index (2 days)
- Kier-Hall Kappa Indices (2 days) - shape descriptors

#### Simple Counting
- Valence Electrons (0.5 day) - periodic table lookup
- Radical Electrons (0.5 day) - already in Atom struct
- Heteroatom Count (0.5 day) - non-C/H atoms
- Spiro Atom Count (1 day) - ring membership
- Carbon Types (1 day) - sp1/sp2/sp3 classification
- Ring Type Counts (0.5 day) - saturated/aliphatic

#### High-Value Physicochemical
- **TPSA** ⭐⭐⭐ (3 days) - Topological Polar Surface Area (critical for drug-likeness!)
- **Molar Refractivity** (2 days) - similar to LogP implementation
- **Functional Groups** (2 days) - 20-30 SMARTS patterns (alcohol, amine, ketone, etc.)
- Acidic/Basic Group Counts (1 day each)
- **QED** (1 day) - Quantitative Estimate of Drug-likeness

#### Additional Useful
- Petitjean Indices (1 day)
- Gravitational Index (1 day)
- Hybridization Ratio (0.5 day) - sp3/sp2
- VAdjMa (0.5 day) - vertex adjacency magnitude

**Output**: ~35 descriptors, all validated against RDKit/CDK

### Phase 9B: Morgan/ECFP Fingerprints (1 week) 🟡
**Goal**: Core ML fingerprint ⭐⭐⭐

- Morgan Algorithm Core (3 days) - iterative atom environment hashing
- Hashing Infrastructure (1 day) - FNV-1a or xxHash
- Bit Folding (0.5 day) - map to 1024/2048/4096 bits
- Variants (0.5 day) - ECFP2, ECFP4, ECFP6
- Tests & Validation (1 day) - compare to RDKit

**Why Priority**: Most important fingerprint for ML/similarity search, widely used in industry

### Phase 9C: Additional Fingerprints (2 weeks) 🟡
- Atom Pair Fingerprints (4 days) - distance-based encoding
- Topological Torsion (4 days) - 4-atom path encoding
- PubChem Fingerprints (3 days) - 881 structural keys

### Phase 10: Medium Complexity Descriptors (3-4 weeks) 🟡
- Chi Indices (5 days) - path/cluster/chain connectivity
- MDE Descriptors (4 days) - molecular distance edge
- Weighted Path (3 days)
- EState Indices (5 days) - electrotopological state
- Autocorrelation ATS (5 days) - mass/charge/polarizability weighted
- BCUT Descriptors (5 days) - eigenvalue based (needs nalgebra)

**Output**: +30-40 medium complexity descriptors

### Phase 11: Electronic Descriptors - COMPLEXITY CLIFF (3-4 weeks) 🔴
⚠️ **WARNING**: Advanced features ahead

#### Gasteiger Partial Charges (2 weeks) 🔴
**Hardest non-3D feature**
- Implementation (10 days) - iterative electronegativity equalization
- Tests & Validation (4 days) - compare to RDKit, many edge cases
- **Why Hard**: Complex iteration, resonance/conjugation edge cases, requires careful validation

#### Charge-Derived Descriptors (1 week)
- Partial Charge Min/Max/Abs (2 days) - requires Gasteiger first
- CPSA Descriptors (3 days) - Charged Partial Surface Area (needs charges + 3D coords)
- Polarizability Descriptors (2 days)

### Phase 12: 3D Descriptors - MAJOR PROJECT (Future) 🔴
⚠️ **BLOCKED**: Requires conformer generation first

- Conformer Generation (2-3 weeks) - ETKDG algorithm
- 3D Shape Descriptors (1 week) - PMI, Asphericity, etc.
- 3D Pharmacophore Fingerprints (2 weeks)
- Molecular Surface/Volume (2 weeks)

**Note**: Many QSAR models work excellently with 2D descriptors only. Defer Phase 12 until 2D suite is rock-solid.

### Total Achievable Before Hard Problems:
- **By end of Phase 9** (4-5 weeks): 55+ descriptors, Morgan/ECFP, 3-4 fingerprint types
- **By end of Phase 10** (8-9 weeks): 85+ descriptors, all topological features
- **By end of Phase 11** (11-13 weeks): Complete 2D cheminformatics library with electronic properties

### Mega-Vector Target (~5000+ dimensions)
- Binary fingerprints: ~4500 bits (MACCS 166 + Morgan 2048 + AtomPair 2048 + PubChem 881)
- Continuous descriptors: ~500 floats (85+ individual descriptors)
- **Current**: 20 descriptors + 166 bits = excellent foundation
- **After Phase 9**: 55 descriptors + ~3000 bits = production-ready
- **After Phase 10**: 85 descriptors + ~5000 bits = comprehensive 2D suite

### Phase 4: SMARTS Pattern Matching

#### 4.1 SMARTS Parser (`io/smarts.rs`)
- **Priority**: HIGH (needed for MACCS, functional groups)
- **Complexity**: Higher than SMILES (logical operators, recursive patterns)
- **Strategy**: Separate QueryGraph structure
- **Features**:
  - Logical operators (AND, OR, NOT)
  - Atom primitives (!, &, ;)
  - Recursive SMARTS ($(...))
  - Ring membership queries

#### 4.2 Substructure Matching (`algo/iso.rs`)
- **Algorithm**: VF2 or Ullmann with backtracking
- **Performance**: Use heuristics (degree filtering, element counts)
- **Optimization**: Pre-compute query fingerprints

### Phase 5: Additional I/O Formats

#### 5.1 CSV Reader for SMILES (`io/csv.rs`)
- **Priority**: HIGH (essential for ML workflows)
- **Format**: Read CSV files with SMILES column
- **Features**:
  - Automatic SMILES column detection
  - Header parsing (id, smiles, properties)
  - Parallel reading with Rayon
  - Error handling (skip invalid SMILES)
  - Memory-mapped files for large datasets
- **Use Case**: Batch processing of chemical datasets
- **Performance**: Stream processing, >100k molecules/sec
- **Dependencies**: `csv` crate for parsing
- **Output**: Iterator of (id, Molecule, properties)
- **Example**:
  ```rust
  let reader = CsvReader::from_path("compounds.csv")?;
  let molecules: Vec<_> = reader
      .with_smiles_column("SMILES")
      .parallel_parse()
      .collect();
  ```

#### 5.2 SDF/MOL Parser (`io/sdf.rs`)
- **Priority**: MEDIUM
- **Format**: MDL MOL/SDF V2000/V3000
- **Features**: 3D coordinates, property fields
- **Use Case**: Reading from chemical databases

#### 5.3 InChI Generation
- **Strategy**: Use external `rust-inchi` crate (C bindings)
- **Alternative**: Implement basic InChI layer 1 (formula + connectivity)

### Phase 6: Machine Learning Integration

#### 6.1 Feature Vector Design (DECIDED)

**Unified Float Representation:**
- **All features stored as `Vec<f32>`** (fingerprints + descriptors)
- **Fingerprint bits encoded as 0.0/1.0** (not packed bits)
- **No conversion overhead** for ML pipeline
- **Memory**: ~20 KB per molecule (acceptable for batch processing)

**Implementation:**
```rust
pub struct MolecularFeatures {
    /// All features as floats (~5000 dimensions)
    /// Layout:
    ///   [0..2048]:     ECFP4 (bits as 0.0/1.0)
    ///   [2048..4096]:  FCFP4
    ///   [4096..4262]:  MACCS (166 bits)
    ///   [4262..5143]:  PubChem (881 bits)
    ///   [5143..5182]:  Biosymphony (39 bits)
    ///   [5182..]:      Physicochemical descriptors
    pub features: Vec<f32>,
}

impl MolecularFeatures {
    pub fn as_ndarray(&self) -> ArrayView1<f32> {
        ArrayView1::from(&self.features)
    }
    
    pub fn normalize(&mut self, scaler: &Scaler) {
        // Only normalize continuous descriptors
        scaler.transform(&mut self.features[5182..]);
    }
}
```

**Rationale:**
- ✅ ML is primary use case
- ✅ Simplicity over memory optimization (clarity philosophy)
- ✅ Zero-cost ndarray conversion
- ✅ No temporary allocations in ML pipeline
- ✅ Matches linfa/scikit-learn expectations
- ✅ Still supports Tanimoto similarity (treat >0.5 as bit set)
- ⚠️ Trade-off: 8x memory vs packed bits (2.5 KB → 20 KB) - acceptable

**Performance:**
- No conversion overhead: 0 ns (vs ~15 μs with separate storage)
- Direct feed to ML models
- Batch processing: flat_map directly to Array2<f32>

#### 6.2 QsarPipeline Architecture
- **Status**: Planned (from VISION.md)
- **Core Struct**: Self-contained pipeline with scaler + PCA + model
- **Features**:
  - Reproducible predictions on new data
  - Serialization with `bincode`
  - Version-tracked metadata
- **Implementation**:
  ```rust
  pub struct QsarPipeline {
      scaler: Scaler,           // Z-score normalization params
      pca: Option<PcaTransform>, // Optional dimensionality reduction
      model: Box<dyn Model>,     // Trained model (KNN, RF, ART)
      metadata: PipelineMetadata,
  }
  ```

#### 6.3 Descriptor Matrix Export
- **Status**: Already designed (batch.rs with Rayon)
- **Format**: Vec<Vec<Real>> → ndarray for linfa
- **Optimization**: Pre-allocate matrices, parallel computation
- **Note**: With unified float vector, this becomes trivial flat_map

#### 6.4 Standard Scaler Implementation
- **Priority**: HIGH (required for mega-vector)
- **Algorithm**: Z-score normalization (mean=0, std=1)
- **Purpose**: Prevent binary fingerprints from being dominated by continuous descriptors
- **Implementation**: O(n) single pass for mean/std calculation
- **Important**: Only normalize continuous descriptors (5182+), keep fingerprints as 0/1

#### 6.5 Linfa Integration
- k-NN classification/regression (with Tanimoto metric for fingerprints)
- Random Forest (handles mixed binary/continuous features)
- PCA for dimensionality reduction
- **Data Pipeline**: SMILES → Features (Vec<f32>) → Array2 → Model

#### 6.6 ART 2A Clustering (Custom Implementation)
- **Algorithm**: Adaptive Resonance Theory 2A
- **Use Case**: Chemical space exploration, diversity selection
- **Implementation**: From scratch following Carpenter/Grossberg papers
- **Performance**: Incremental learning, O(n·m) where m = num_clusters

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
- **Graph kernels**: For SVM on molecular graphs
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
- `ahash`/`fnv`: Fast hashing for fingerprints
- `ndarray`: Matrix operations for ML
- `linfa`: ML algorithms
- `serde`: Serialization (JSON/MessagePack output)
- `memmap2`: Memory-mapped file I/O for huge datasets
- `dashmap`: Concurrent hash maps for caching

### Optional (Scientific)
- `nalgebra`: Linear algebra for 3D geometry
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

### v0.1.0 - MVP (Current Goal)
- [x] SMILES parser (organic subset)
- [x] Molecular weight
- [ ] Implicit hydrogens
- [ ] Basic CLI (calc subcommand)

### v0.2.0 - Core Algorithms
- [ ] Ring detection (SSSR)
- [ ] Aromaticity perception
- [ ] 10+ topological descriptors
- [ ] Parallel batch processing

### v0.3.0 - Pattern Matching
- [ ] SMARTS parser
- [ ] Substructure search
- [ ] MACCS fingerprints
- [ ] Functional group detection

### v0.4.0 - ML Ready
- [ ] 50+ descriptors
- [ ] Morgan/ECFP fingerprints
- [ ] Linfa integration examples
- [ ] CSV/JSON export

### v0.5.0 - Advanced Features
- [ ] SDF parser
- [ ] Canonicalization
- [ ] InChI generation
- [ ] ART 2A clustering

### v1.0.0 - Production Ready
- [ ] Comprehensive test coverage
- [ ] Benchmarks vs RDKit
- [ ] Full documentation
- [ ] Performance optimized

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

**Last Updated**: February 10, 2026 12:30 UTC  
**Status**: Phase 8 Complete - MACCS Keys + Recursive SMARTS + SmallRing Descriptor  
**Performance**: 10-50x speedup with pre-parsed patterns  
**Tests**: 212/212 passing ✅  
**Descriptors**: 20 molecular descriptors (+ MACCS Keys 166-bit fingerprint)  
**SMARTS Parser**: Extended with recursive SMARTS, ring closures, bond operators  
**Next Priority**: Phase 9A - Easy topological descriptors (~35 descriptors in 2-3 weeks)  
**Roadmap**: 40-50 easy/medium descriptors before Gasteiger charges (hard)


