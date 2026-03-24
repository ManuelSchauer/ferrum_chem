# FerrumChem 🧪⚡

> High-performance QSAR engine for cheminformatics in Rust

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Validation](https://img.shields.io/badge/validation-release--mode%20tracked-brightgreen)]()
[![RDKit Match](https://img.shields.io/badge/RDKit%20match-100%25%20(1944%2F1944)-brightgreen)]()
[![1000-row scalar](https://img.shields.io/badge/1000--row%20scalar-99.8%25-brightgreen)]()
[![1000-row rings](https://img.shields.io/badge/1000--row%20rings-100%25-brightgreen)]()
[![CSV Throughput](https://img.shields.io/badge/CSV%201k-2514%20mol%2Fs-blue)]()
[![Descriptors](https://img.shields.io/badge/descriptors-72%20typed-blue)]()
[![Dimensions](https://img.shields.io/badge/dimensions-17.7k-purple)]()
[![Memory](https://img.shields.io/badge/CSV%20RSS-6.23MB%20(Ferrum)%20vs%2071.36MB%20(RDKit)-green)]()
[![SDF parser](https://img.shields.io/badge/SDF%20parser-deferred-yellow)]()

FerrumChem is a high-performance molecular descriptor calculation engine designed for machine learning applications in chemistry. Built in Rust for maximum performance and memory efficiency, it parses simple molecules at multi-million-molecule throughput and provides comprehensive QSAR descriptor coverage through a typed Rust API.

## ✨ Features

- **🚀 Ultra-fast SMILES parsing**: >1M molecules/second (single-threaded)
- **⚡ High-performance descriptors**: 72 typed descriptors, 17,710+ dimensions
- **💾 31-32x typed-memory reduction**: bit-packed fingerprints (67.4 GB → 2.1 GB for 1M molecules in the memory benchmark)
- **🔬 100% RDKit accuracy**: Validated against 27 torture molecules × 72 descriptors (1944/1944 exact match)
- **📈 Current 1000-row parity**: Scalar 99.8%, Rings 100.0%, VSA 100.0%, Chi/Kappa 100.0%
- **📈 Scalar mismatch scope (1000-row)**: only `Ipc` diverges (`953/1000`), all other scalar descriptors are `1000/1000`
- **📊 Latest pinned CSV benchmark (57 outputs, single-thread)**:
  - 1k rows buffered: `2514.32 mol/s`, `5.94 MB` peak RSS
  - 1k rows stream (`chunk=1`): `2460.29 mol/s`, `7.70 MB` peak RSS
  - 10k rows buffered: `2292.07 mol/s`, `9.88 MB` peak RSS
  - 10k rows stream (`chunk=1`): `2249.53 mol/s`, `10.16 MB` peak RSS
- **🔧 Parallel processing**: Built-in Rayon support for batch calculations
- **📊 ML-ready**: Single `Descriptor` trait + direct ndarray integration
- **🔒 Thread-safe**: All data structures are `Send + Sync`
- **✅ Type-safe**: Strongly-typed descriptor values (Int, Float, Binary, etc.)

## 🎯 Status: parity stable, TUI available, and single-thread CSV bundle is ~4.1x RDKit at 1k rows ✅

FerrumChem still has **100% RDKit match** across the 27-molecule torture set (**1944/1944**) and now also reaches:

- **1000-row current reference slice**: **Scalar 21953/22000 = 99.8%**, **Ring 4000/4000 = 100.0%**, **VSA 36000/36000 = 100.0%**, **Chi/Kappa 10000/10000 = 100.0%**
- **Latest 1000-row closures**: `Qed`, `RotatableBonds`, `FractionCSP3`, `HallKierAlpha`, all Chi/Kappa values, all ring descriptors, and all VSA bins now match RDKit on **1000/1000** molecules
- **Remaining parity gap in current 1000-row suite**: `Ipc` only (**953/1000**, 95.3%)

The only remaining validated parity gap in the current 1000-row RDKit suite is `Ipc` numerical divergence. Non-`Ipc` scalar/ring/VSA/Chi/Kappa descriptors are now fully aligned on that slice; exact `Ipc` emulation remains a deferred item unless strict RDKit bit-level reproducibility becomes a hard requirement.

✅ **Implemented**:
- **72 Typed Descriptors** with DescriptorValue enum (Int, Float, IntVec, IntU16Vec, FloatVec, Binary)
- **100% RDKit Accuracy**: 1944/1944 descriptor comparisons across 27 challenging molecules (metals, charged species, aromatic edge cases)
- **500/500 on the 500-row RDKit slice**: AromaticAtomCount, AromaticRingCount, AliphaticRings, AromaticRings(RT), SaturatedRings, TotalRings, HBondAcceptor, NumAmideBonds, QED, RotatableBonds, Chi3v, Chi4v, Kappa3
- **1000/1000 on the current 1000-row audit (non-Ipc)**: all scalar descriptors except `Ipc`, plus all ring descriptors, all VSA bins, and all Chi/Kappa values
- **32x Memory Reduction** via bit-packing + type optimization (validated with benchmarks)
- **Release-mode validation workflow** for parity and focused regression suites
- **17,710+ Total Dimensions** (17,431 fingerprint bits + 279 continuous)
- **6 Fingerprint Types**: MACCS (166), PubChem (881), ECFP2/4/6 (12,288), FCFP2/4/6 (12,288), AtomPair (2,048), Torsion (2,048)
- **Algorithms**: Morgan canonicalization, connected components, salt stripping, SSSR ring detection
- **Comprehensive Descriptors**:
  - Basic: MW, LogP, MR, TPSA, atom/bond counts, rings
  - Drug-likeness: QED, Lipinski violations, FractionCSP3
  - Topological: Wiener, Zagreb, Balaban, Kappa, Petitjean
  - Connectivity: Chi indices (26 values), E-State (47 values)
  - Graph-based: MDE, Weighted Path, Autocorrelation ATS (32 values)
  - Eigenvalue: BCUT (8 values)
  - VSA: SlogP_VSA (12), SMR_VSA (10), PEOE_VSA (14), LabuteASA
  - Complexity: BertzCT, Ipc, HallKierAlpha, valence electrons
  - Ring Analysis: Small rings (11), carbon types (9), ring types (3)
  - Electronic: Gasteiger partial charges (with conjugation splitting, dative bonds, NaN propagation)
  - Transition metals: Dative bond detection, extended element tables (Z=1–83)

🚧 **Current roadmap**:
- **Current descriptor parity focus**: `Ipc` numerical emulation and stability/performance tradeoff analysis
- **Recent parity closures**: resolved the fused heteroaromatic/QED alert cascade and isolated `RotatableBonds` mismatch on the 1000-row audit
- **Deferred exact parity work**: strict `Ipc` emulation remains optional unless exact RDKit reproduction is required
- **Phase 13**: ✅ Complete — ML pipeline (k-NN, RF, PLS, Elastic Net, ART-2A — all from scratch, no linfa)
- **Phase 14**: ✅ Active — Terminal UI workflow shell with CSV load, training, profiling, and pipeline save/load
- **Phase 15**: Infrastructure (SDF parser, canonical SMILES writer, Python bindings, crates.io)
- **Phase 16**: 3D descriptors (special cases — conformer generation)

## 🚀 Quick Start

### Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
ferrum_core = { path = "crates/ferrum_core" }
```

### Basic Usage

```rust
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::desc::molecular_weight::MolecularWeight;
use ferrum_core::desc::Descriptor;

fn main() {
    // Parse SMILES (implicit hydrogens calculated automatically)
    let mut mol = parse_smiles("c1ccccc1").unwrap(); // Benzene
    
    // Calculate molecular weight
    let desc = MolecularWeight;
    let value = desc.calculate(&mut mol);
    
    println!("MW: {:.2} Da", value.as_float().unwrap()); // MW: 78.11 Da
}
```

### ML Vector Building (Parallel)

```rust
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::desc::{build_ml_vectors_parallel, Descriptor};
use ferrum_core::desc::molecular_weight::MolecularWeight;
use ferrum_core::desc::logp::LogP;
use ferrum_core::desc::tpsa::Tpsa;

fn main() {
    // Parse multiple molecules
    let smiles = vec!["CC(=O)O", "c1ccccc1", "CCO"];
    let mut molecules: Vec<_> = smiles.iter()
        .map(|s| parse_smiles(s).unwrap())
        .collect();
    
    // Define descriptors
    let descriptors: Vec<Box<dyn Descriptor>> = vec![
        Box::new(MolecularWeight),
        Box::new(LogP),
        Box::new(Tpsa),
    ];
    
    // Compute in parallel (uses all CPU cores)
    let vectors = build_ml_vectors_parallel(&mut molecules, &descriptors);
    
    // vectors[0] = [60.05, -0.09, 37.3]  (acetic acid)
    // vectors[1] = [78.11, 2.22, 0.0]    (benzene)
    // vectors[2] = [46.07, -0.31, 20.23] (ethanol)
}
```

**Note**: Implicit hydrogens are calculated automatically during parsing for consistency and simplicity.

### Run Benchmarks

```bash
cargo run --release --bin benchmark

# Matched single-thread Ferrum vs RDKit descriptor bundle benchmark (57 outputs)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
  cargo run -p ferrum_cli --release --bin parity_bundle_benchmark
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/benchmark_rdkit_parity_bundle.py

# Matched single-thread CSV bundle benchmark (throughput + peak RSS)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
  FERRUM_BENCH_CSV_LIMIT=1000 FERRUM_BENCH_REPEATS=11 FERRUM_BENCH_WARMUP=100 \
  cargo run -p ferrum_cli --release --bin parity_bundle_csv_benchmark
# Optional streaming ingestion mode (avoids preloading all SMILES in benchmark process)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
  FERRUM_BENCH_CSV_MODE=stream FERRUM_BENCH_CSV_CHUNK_SIZE=1 \
  FERRUM_BENCH_CSV_LIMIT=1000 FERRUM_BENCH_REPEATS=11 FERRUM_BENCH_WARMUP=100 \
  cargo run -p ferrum_cli --release --bin parity_bundle_csv_benchmark
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  FERRUM_BENCH_CSV_LIMIT=1000 FERRUM_BENCH_REPEATS=5 FERRUM_BENCH_WARMUP=100 \
  python scripts/benchmark_rdkit_parity_bundle_csv.py
```

### Phase 14 TUI — Interactive QSAR Workflow Shell

The TUI provides a full interactive terminal interface for loading chemical datasets, training ML models, inspecting descriptors, and managing serialized pipelines — all without writing code.

#### Quick Launch

```bash
# Launch with default settings (looks for coconut_csv_lite-02-2026.csv in current dir)
cargo run -p ferrum_cli --release --bin tui

# Launch with a custom CSV file
FERRUM_TUI_DATASET_CSV="/path/to/your/molecules.csv" \
FERRUM_TUI_SMILES_COLUMN="smiles" \
FERRUM_TUI_TARGET_COLUMN="activity" \
FERRUM_TUI_ROW_LIMIT=500 \
cargo run -p ferrum_cli --release --bin tui
```

#### CSV File Requirements

Your CSV file must have:
- A **SMILES column** containing valid SMILES strings (one molecule per row)
- A **target column** containing numeric values (the property you want to predict, e.g. logP, activity, pIC50)
- A header row with column names

Example CSV:
```csv
smiles,logp,molecular_weight
c1ccccc1,1.56,78.11
CC(=O)O,-0.17,60.05
CCO,-0.31,46.07
```

#### Environment Variables

All TUI settings are configured via environment variables (with sensible defaults):

| Variable | Default | Description |
|---|---|---|
| `FERRUM_TUI_DATASET_CSV` | `coconut_csv_lite-02-2026.csv` | Path to input CSV file |
| `FERRUM_TUI_SMILES_COLUMN` | `canonical_smiles` | CSV column containing SMILES strings |
| `FERRUM_TUI_TARGET_COLUMN` | `alogp` | CSV column containing the numeric target property |
| `FERRUM_TUI_ROW_LIMIT` | `300` | Maximum number of rows to load (caps memory usage) |
| `FERRUM_TUI_PIPELINE_PATH` | `/tmp/ferrum_tui_pipeline.bin` | File path for saving/loading serialized pipelines |

> **Tip:** If the CSV file is not found, the TUI launches with a fallback mock dataset and shows a warning. You can still explore all screens — the training wizard will work on the mock data.

#### TUI Screens

**1. Training Wizard** — Step-by-step model training workflow:
1. **Dataset** → Press Enter to load (or reload) the CSV. Shows parse stats (rows loaded, failures, missing targets, target range)
2. **Preprocess** → Confirms StandardScaler preprocessing (applied automatically by AutoQsar)
3. **Train** → Runs `AutoQsar::regression` which compares k-NN (k=3,5,7), Random Forest (50,100 trees), PLS (2,3,5 components), and Elastic Net (α=0.01,0.1,1.0). Shows wall-clock training time and best model Q²
4. **Evaluate** → Shows final pipeline summary with CV Q², train R², and prediction preview
5. **Done** → Pipeline is ready for batch prediction and save/export

**2. Batch Prediction** — Run the trained model against the loaded dataset. Shows R², RMSE, and sample-level prediction preview. Press Enter to recompute.

**3. Chemical Space** — PCA-projected scatter plot of the loaded dataset. Uses StandardScaler + Jacobi PCA(2) to project all molecules into 2D. Shows:
- Braille-dot scatter plot (PC1 vs PC2)
- Variance explained per principal component
- PC axis ranges and sample count

**4. Descriptor Profiling** — Scrollable table of all computed molecular descriptors showing per-feature statistics:
- Mean, standard deviation, min, max
- Zero-variance features flagged with ⚠ (red) — important for ML: these features provide no discriminatory power
- Scroll with ↑/↓ to navigate all features

**5. Pipeline Management** — File-backed persistence for trained models:
- `s` → Serialize current pipeline to file (bincode) at `FERRUM_TUI_PIPELINE_PATH`
- `l` → Load a previously saved pipeline from file
- `m` → Reload pipeline from in-memory mirror (useful after re-training)

#### Keybindings

| Key | Context | Action |
|---|---|---|
| `↑`/`↓` or `j`/`k` | Menu | Navigate menu items |
| `↑`/`↓` or `j`/`k` | Descriptor Profiling | Scroll feature table |
| `Enter` | Everywhere | Activate selection / advance wizard step |
| `Esc` | Any view | Return to main menu |
| `q` | Anywhere | Quit TUI |
| `s` | Pipeline Management | Save pipeline to file |
| `l` | Pipeline Management | Load pipeline from file |
| `m` | Pipeline Management | Load pipeline from memory |
| `?` | Anywhere | Show keybindings help in status bar |

#### Computed Descriptors (24 features used in training)

The TUI training workflow computes the following molecular descriptors for each molecule:

`MolecularWeight`, `LogP`, `MolarRefractivity`, `TPSA`, `HBondDonor`, `HBondAcceptor`, `RotatableBondCount`, `HeavyAtomCount`, `AtomCount`, `AromaticAtomCount`, `HeteroatomCount`, `AromaticRingCount`, `FormalCharge`, `FractionCSP3`, `BertzCT`, `LabuteASA`, `QED`, `NumAmideBonds`, `HallKierAlpha`, `BalabanIndex`, `BondCount`, `SLogP_VSA` (8 bins), `SMR_VSA` (7 bins), `PEOE_VSA` (14 bins)

This results in approximately 57 total numeric features per molecule.

#### Example Workflow

```bash
# 1. Prepare a CSV with SMILES and a target property
head -3 my_dataset.csv
# smiles,pIC50
# c1ccc(cc1)O,5.2
# CC(=O)Nc1ccc(O)cc1,6.8

# 2. Launch TUI pointing at the CSV
FERRUM_TUI_DATASET_CSV="my_dataset.csv" \
FERRUM_TUI_SMILES_COLUMN="smiles" \
FERRUM_TUI_TARGET_COLUMN="pIC50" \
FERRUM_TUI_ROW_LIMIT=1000 \
cargo run -p ferrum_cli --release --bin tui

# 3. Inside TUI:
#    → Select "Interactive model training wizard" → Enter through all steps
#    → Select "Batch prediction interface" → Enter to see R²/RMSE
#    → Select "Chemical space visualization" → See PCA scatter
#    → Select "Descriptor profiling dashboard" → Scroll features with ↑/↓
#    → Select "Pipeline management" → Press 's' to save trained model
```

### Run Tests

```bash
# Focused release-mode safety checks
cargo test --release --package ferrum_core --test smarts_edge -- --nocapture
cargo test --release --package ferrum_core qed -- --nocapture

# 500-row RDKit parity slices
cargo test --release --package ferrum_core rdkit_scalar_descriptors -- --nocapture
cargo test --release --package ferrum_core rdkit_ring_descriptors -- --nocapture
cargo test --release --package ferrum_core rdkit_vsa_descriptors -- --nocapture
cargo test --release --package ferrum_core rdkit_chi_kappa_indices -- --nocapture

# Torture-set reference comparison
cargo test --release --package ferrum_core --test rdkit_comparison -- --nocapture
```

## 📊 Performance

Benchmarked on Apple Silicon (M-series):

| Operation | Throughput | Notes |
|-----------|------------|-------|
| SMILES parsing (simple) | 6M molecules/sec | Methane, ethane |
| SMILES parsing (drug-like) | 1.2M molecules/sec | Caffeine, aspirin |
| Implicit H calculation | 50M calculations/sec | Valence-based |
| Molecular weight calc | 1B calculations/sec | Trivial descriptor |
| LogP (Crippen) | 500k-5M mol/sec | Pre-parsed SMARTS |
| TPSA | 200k-500k mol/sec | Complex bond counting |
| MACCS Keys | 50k mol/sec | 163 SMARTS patterns |
| ECFP4 | 100k-300k mol/sec | Morgan algorithm |

The table above is for isolated kernels and parser-heavy microbenchmarks. For an apples-to-apples Ferrum-vs-RDKit run over the same descriptor bundle, use the matched benchmark below.

**Batch processing scales linearly** up to 100k+ molecules with excellent cache utilization.

### Ferrum vs RDKit: matched descriptor bundle

Single-threaded benchmark on Apple Silicon (M-series), using the same 57-output bundle on both sides (21 scalar descriptors + `SlogP_VSA`, `SMR_VSA`, `PEOE_VSA`; `Ipc`, ring groups, and CHI/KAPPA groups excluded from this matched run).

Latest normalized snapshot (10,000 iterations, median of repeated runs):

| Case | Heavy atoms | FerrumChem | RDKit | Relative result |
|------|-------------|------------|-------|-----------------|
| Ethane | 2 | 0.038852 s total, 257,388.66 mol/s | 2.161077 s total, 4,627.32 mol/s | FerrumChem 55.62x faster |
| Aspirin | 13 | 0.649283 s total, 15,401.60 mol/s | 5.627022 s total, 1,777.14 mol/s | FerrumChem 8.67x faster |
| Caffeine | 14 | 0.798840 s total, 12,518.15 mol/s | 5.976980 s total, 1,673.09 mol/s | FerrumChem 7.48x faster |
| Large drug-like amide | 30 | 2.358389 s total, 4,240.18 mol/s | 12.143051 s total, 823.52 mol/s | FerrumChem 5.15x faster |

CSV bundle slice (`coconut_csv_lite-02-2026.csv`, 57 outputs) currently reports both throughput and peak RSS in matched output columns:

- FerrumChem (1k rows, buffered): `2502.38 mol/s`, peak RSS `6.23 MB`
- FerrumChem (1k rows, stream, `chunk=1`): `2460.29 mol/s`, peak RSS `7.70 MB`
- RDKit (1k rows): `608.20 mol/s`, peak RSS `71.36 MB`

This is roughly:
- **4.11x throughput advantage** for Ferrum buffered vs RDKit on the pinned 1k run
- **11.45x lower peak RSS** for Ferrum buffered vs RDKit

Larger pinned slices (10k rows):
- FerrumChem (10k, buffered): `2292.07 mol/s`, peak RSS `9.88 MB`
- FerrumChem (10k, stream, `chunk=1`): `2249.53 mol/s`, peak RSS `10.16 MB`

The CSV benchmark output includes `peak_rss_bytes` and `peak_rss_mb` for direct memory-footprint comparisons.

### Accuracy & Validation
- **100% RDKit match**: 1944/1944 descriptor comparisons across 27 torture molecules
- **Current 1000-row parity**: scalar 99.8%, ring 100.0%, VSA 100.0%, Chi/Kappa 100.0%
- **Extended 10k audit (diagnostic)**: scalar 99.3%, ring 99.4%, VSA 99.6%, Chi/Kappa 99.9%
- **Exact 1000/1000 descriptors**: all ring descriptors, all VSA bins, all Chi/Kappa values, and all scalar descriptors except `Ipc`
- **Remaining 1000-row mismatch**: `Ipc` only (47 fails; `953/1000` pass rate)
- **10k dominant mismatch cluster**: `Ipc` numerical divergence remains the largest contributor
- **Torture set includes**: Cisplatin (transition metal), nitrobenzene (charged NO₂), Grignard (Mg), cyclopentadienyl anion, sodium acetate, pyrene, isatin, aspirin, caffeine, and more
- **Molecular weights**: ±0.01 Da
- **TPSA**: RDKit-exact (50/50 molecules, ±0.01 Ų)
- **LogP**: 111 Wildman-Crippen patterns from RDKit
- **Gasteiger charges**: RDKit-exact (conjugation splitting, 26 atom types, dative bonds, NaN propagation)
- **Validation policy**: run parity suites in `--release` mode

### Performance vs Other Tools
- **Matched descriptor-bundle throughput**: FerrumChem currently leads across all four tracked parity cases in the normalized single-thread benchmark
- **Largest current performance hotspots**: `find_minimum_cycle_basis_bonds`, `balaban::compute_distance_matrix`, and SMARTS/VF2 primitive feasibility paths
- **CSV benchmark memory**: currently ~11x lower peak RSS than RDKit on pinned 1k bundle runs
- **Typed fingerprint storage**: ~31x lower memory than dense `Vec<f32>` representations in the in-repo memory benchmark
- **Native Rust deployment**: No GIL and no Python↔Rust FFI boundary inside Rust-first workflows

## 🏗️ Architecture

```
ferrum_chem/
├── crates/
│   ├── ferrum_core/   # Core engine
│   │   ├── mol/       # Data structures (Atom, Bond, Molecule)
│   │   ├── io/        # Parsers (SMILES, SMARTS, SDF)
│   │   ├── algo/      # Algorithms (rings, aromaticity, charges, canonicalization, components)
│   │   └── desc/      # Descriptors & batch processing
│   └── ferrum_cli/    # Command-line tool and benchmarks
└── scripts/           # RDKit parity/benchmark helper scripts
```

## 🔬 Scientific Features

### Thread Safety ⚡

**All FerrumChem components are designed to be thread-safe:**

- **Data structures**: `Molecule`, `Atom`, `Bond` are all `Send + Sync`
- **Parsers**: SMILES parser can be called concurrently without locks
- **Descriptors**: All implementations must be `Send + Sync` (enforced by trait bounds)
- **Batch processing**: Rayon-based parallelism with automatic work-stealing
- **No shared mutable state**: Each thread operates on independent data

**This enables:**
- Processing millions of molecules in parallel
- Lock-free concurrent parsing
- Safe integration with parallel ML pipelines
- Linear scaling with CPU cores

### Supported SMILES Syntax

- ✅ Organic subset (C, N, O, S, P, B, F, Cl, Br, I)
- ✅ Aromatic atoms (lowercase notation)
- ✅ Bonds: single, double, triple, aromatic
- ✅ Ring closures (1-99)
- ✅ Branches
- ✅ Charges ([NH4+], [O-])
- ✅ Isotopes ([2H], [13C])
- ✅ Explicit hydrogens ([CH4])

### Current Roadmap

**Phase 13**: ✅ Complete — ML pipeline (k-NN, RF, PLS, Elastic Net, ART-2A — all from scratch, no external ML deps)  
**Current performance push**: deep refactors (descriptor context/workspace reuse, unified distance kernels, SMARTS specialization) with recent wins in aromaticity/SMARTS/IPC/path kernels  
**Secondary parity backlog**: finalize `Ipc` numerical alignment while preserving current throughput and memory profile  
**Deferred exact parity work**: `Ipc` numerical emulation (not currently required for ML use)  
**Phase 14**: ✅ Terminal UI workflow shell (interactive QSAR model training/prediction with CSV/file workflows)  
**Phase 15**: Infrastructure (SDF parser, canonical SMILES writer, Python bindings, crates.io)  
**Phase 16**: 3D descriptors (conformers, PMI, shape — special cases only)

**Current Status**: Production-ready typed QSAR library with 72 typed descriptors, 100% torture-set RDKit agreement, and current **1000-row parity** at scalar **99.8%** (only `Ipc` divergent), ring **100.0%**, VSA **100.0%**, Chi/Kappa **100.0%**. Latest pinned single-thread CSV bundle shows Ferrum at **2502 mol/s** and **6.23 MB** peak RSS on 1k rows vs RDKit at **608 mol/s** and **71.36 MB**. ✅

## 🎓 Use Cases

- **QSAR modeling**: Calculate descriptors for ML models
- **Virtual screening**: Fast filtering of large compound libraries
- **Dataset processing**: Batch descriptor calculation for millions of molecules
- **Chemical space analysis**: Clustering, diversity selection
- **Property prediction**: Native ML pipeline (no Python needed)

## 📚 Documentation

- [📋 Project Plan](PROJECT_PLAN.md) - Comprehensive roadmap and technical details
- [✅ TODO List](TODO.md) - Phase-by-phase development plan
- [🔍 Debugging Status](DEBUGGING_STATUS.md) - **RDKit validation: 100% match (1944/1944)**
- [📊 Phase 11 Complete](PHASE_11_REFACTOR_COMPLETE.md) - Typed descriptors: 60 descriptors, 32x memory reduction
- [🔬 Thread Safety](THREAD_SAFETY.md) - Concurrency guidelines

## 🏆 Comparison with Other Tools

See **[PHASE_11_REFACTOR_COMPLETE.md](PHASE_11_REFACTOR_COMPLETE.md)** for detailed API documentation.

**Summary**:
- **vs RDKit**: Faster parsing, **32x more memory efficient** (bit-packing), type-safe API, **100% accuracy match** (1944/1944 descriptors)
- **vs CDK**: **32x more memory efficient**, no GC pauses, matches descriptors exactly
- **vs Rust alternatives**: Most complete pure-Rust library (72 typed descriptors vs ~5 in others)
- **Unique advantages**: 
  - **32x memory reduction** (67.5 GB → 2.13 GB for 1M molecules)
  - **Type safety** (DescriptorValue enum prevents errors at compile time)
  - **Native parallelism** (fearless concurrency, no data races)
  - **Zero-cost abstractions** (no runtime overhead)

## 🤝 Contributing

Contributions welcome! Areas of interest:

- **Algorithm implementations**: Rings, aromaticity, canonicalization
- **New descriptors**: Topological, electronic, fingerprints
- **Performance optimization**: Profiling, SIMD, GPU acceleration
- **Testing**: Edge cases, validation datasets
- **Documentation**: Tutorials, examples, scientific references

## 📖 Scientific References

- **SMILES**: Weininger, D. (1988). SMILES notation
- **Morgan Algorithm**: Morgan, H.L. (1965). Chemical graph canonicalization
- **ECFP**: Rogers, D. & Hahn, M. (2010). Extended-connectivity fingerprints
- **Gasteiger Charges**: Gasteiger, J. & Marsili, M. (1980). Partial charges

## 📄 License

[Specify license: MIT, Apache-2.0, or GPL]

## 🙏 Acknowledgments

Built with Rust 🦀 for maximum performance and reliability.

---

**Performance Note**: Benchmarks are from release builds on Apple Silicon. Your mileage may vary based on hardware and workload.
