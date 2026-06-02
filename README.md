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
- **📈 Latest 100k parity campaign**: Scalar 99.3%, Rings 99.2%, VSA 99.5%, Chi/Kappa 99.8% (release mode)
- **⚡ Latest 100k matched CSV benchmark**: Ferrum `2340.81 mol/s` vs RDKit `603.29 mol/s` (**3.88x faster**)
- **💾 Latest 100k matched CSV memory**: Ferrum `21.80 MB` peak RSS vs RDKit `202.56 MB` (**9.29x lower**)
- **📊 Latest pinned CSV benchmark (57 outputs, single-thread, 10k, median of 3 runs)**:
  - Ferrum buffered: `2488.07 mol/s`, `10.11 MiB` peak RSS
  - RDKit: `612.56 mol/s`, `128.38 MiB` peak RSS
  - Relative: Ferrum is **4.06x faster** with **12.70x lower** peak RSS
- **🔧 Parallel processing**: Built-in Rayon support for batch calculations
- **📊 ML-ready**: Single `Descriptor` trait + direct ndarray integration
- **🔒 Thread-safe**: All data structures are `Send + Sync`
- **✅ Type-safe**: Strongly-typed descriptor values (Int, Float, Binary, etc.)
- **🧠 Zero-Copy ML**: High-performance ML pipeline with 100% zero-allocation K-Fold cross-validation via `ArrayView` circular buffers.
- **🚀 Accelerated K-NN**: K-Nearest Neighbors searches are powered by a Vantage-Point Tree allowing $O(\log N)$ traversal, backed by hand-vectorized SIMD instructions with 8-way loop unrolling for high-throughput metric evaluations.
- **🌲 Allocation-Free Random Forest**: The Random Forest implementation achieves up to 40x speedups during tree construction by utilizing shared, mutable thread-local scratch buffers to eliminate recursive vector allocations, complementing its `Rayon`-backed parallel ensemble building.
- **🛡️ Log-Time Applicability Domains**: The `KnnDistanceAD` implementation leverages the K-NN VP-Tree to reduce $O(N)$ brute-force domain scans into $O(\log N)$ tree traversals, while operating entirely on `f32` to halve memory overhead.
- **⚡ Massively Parallel & Vectorized ML Algorithms**: Employs deep `Rayon` multi-threading for embarrassingly parallel $O(P^2)$ correlation matrix calculations during feature selection, vectorized coordinate descent featuring running-residual math for Elastic Net ($1000\times$ faster iterations), optimized fast-chunking ndarray subtractions for PLS NIPALS deflation, and parallel prototype cosine similarity sweeps for ART-2A clustering.
## 🎯 Status: parity stable, TUI available, and single-thread CSV bundle is ~4.1x RDKit at 1k rows ✅

FerrumChem still has **100% RDKit match** across the 27-molecule torture set (**1944/1944**) and now also reaches:

- **1000-row current reference slice**: **Scalar 21953/22000 = 99.8%**, **Ring 4000/4000 = 100.0%**, **VSA 36000/36000 = 100.0%**, **Chi/Kappa 10000/10000 = 100.0%**
- **Latest 1000-row closures**: `Qed`, `RotatableBonds`, `FractionCSP3`, `HallKierAlpha`, all Chi/Kappa values, all ring descriptors, and all VSA bins now match RDKit on **1000/1000** molecules
- **Remaining parity gap in current 1000-row suite**: `Ipc` only (**953/1000**, 95.3%)
- **Latest 100k campaign**: **Scalar 2,184,572/2,200,000 = 99.3%**, **Ring 396,931/400,000 = 99.2%**, **VSA 3,583,114/3,600,000 = 99.5%**, **Chi/Kappa 998,286/1,000,000 = 99.8%**

The only remaining validated parity gap in the current 1000-row RDKit suite is `Ipc` numerical divergence. Non-`Ipc` scalar/ring/VSA/Chi/Kappa descriptors are now fully aligned on that slice; exact `Ipc` emulation remains a deferred item unless strict RDKit bit-level reproducibility becomes a hard requirement.

`Ipc` also exposes an optional mode gate for experiments:
- `FERRUM_IPC_MODE=fast` (default): current `Ipc = sum(|coeffs|) * entropy(|coeffs|)` behavior.
- `FERRUM_IPC_MODE=avg`: entropy-only average-Ipc variant (useful for stability/performance tradeoff studies; not a strict RDKit `Ipc` parity mode).

✅ **Implemented**:
- **72 Typed Descriptors** with DescriptorValue enum (Int, Float, IntVec, IntU16Vec, FloatVec, Binary)
- **100% RDKit Accuracy**: 1944/1944 descriptor comparisons across 27 challenging molecules (metals, charged species, aromatic edge cases)
- **500/500 on the 500-row RDKit slice**: AromaticAtomCount, AromaticRingCount, AliphaticRings, AromaticRings(RT), SaturatedRings, TotalRings, HBondAcceptor, NumAmideBonds, QED, RotatableBonds, Chi3v, Chi4v, Kappa3
- **1000/1000 on the current 1000-row audit (non-Ipc)**: all scalar descriptors except `Ipc`, plus all ring descriptors, all VSA bins, and all Chi/Kappa values
- **32x Memory Reduction** via bit-packing + type optimization (validated with benchmarks)
- **Release-mode validation workflow** for parity and focused regression suites
- **17,710+ Total Dimensions** (17,431 fingerprint bits + 279 continuous)
- **7 Fingerprint Types**: MACCS (166), PubChem (881), ECFP2/4/6 (12,288), FCFP2/4/6 (12,288), AtomPair (2,048), Torsion (2,048), Biosynfoni (39)
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
- **Phase 15**: ✅ Complete — Infrastructure (canonical SMILES writer, Python bindings via PyO3, crates.io)
- **Phase 16**: ✅ Active — 3D descriptors (conformer structures, PMI, Asphericity)

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

### Canonical SMILES Generation

FerrumChem includes a high-performance Canonical SMILES writer based on the Morgan algorithm.

```rust
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::io::smiles_writer::write_canonical_smiles;

fn main() {
    let mol = parse_smiles("C1=CC=CC=C1C(=O)O").unwrap(); // Benzoic acid
    
    // Generates a deterministically ordered, standard SMILES string
    let canonical = write_canonical_smiles(&mol);
    
    println!("{}", canonical); // "c1ccccc1C(=O)O"
}
```

### ML Vector Building (Parallel)

```rust
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::desc::{build_ml_matrix_parallel, Descriptor};
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
    
    // Compute directly into an Array2<f64> zero-copy ML matrix (uses all CPU cores)
    let matrix = build_ml_matrix_parallel(&mut molecules, &descriptors);
    
    // matrix.row(0) = [60.05, -0.09, 37.3]  (acetic acid)
    // matrix.row(1) = [78.11, 2.22, 0.0]    (benzene)
    // matrix.row(2) = [46.07, -0.31, 20.23] (ethanol)
}
```

**Note**: Implicit hydrogens are calculated automatically during parsing for consistency and simplicity.

### 🐍 Python Bindings (PyO3)

FerrumChem provides a high-performance Python extension that calculates its massive descriptor bundle completely in parallel, sidestepping the Python GIL.

**Installation & Build**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install maturin numpy
maturin develop --manifest-path crates/ferrum_py/Cargo.toml --release
```

**Usage in Python**:
```python
import ferrum_py

smiles = [
    "c1ccccc1", # Benzene
    "CC(=O)Oc1ccccc1C(=O)O", # Aspirin
]

# Computes all 72 descriptors (9520+ dimensions) in parallel
# Returns a tuple of (list of feature names, numpy 2D array)
names, features = ferrum_py.compute_descriptors(smiles)

print(f"Computed {len(names)} features for {len(smiles)} molecules.")
print(features.shape) # (2, 9520)
```

### Run Benchmarks

```bash
cargo run --release --bin benchmark all

# Multi-Core Benchmark Suite Results (14 Threads):
# - Scenario A (CPU-Heavy, 123 Descriptors, 10k molecules): 87,328 MPS, 29.2 MB Peak RAM
# - Scenario B (Memory-Heavy, 2048-bit ECFP4, 100k molecules): 794,033 MPS, 669.5 MB Peak RAM
# - Scenario C (ML-Heavy, RF 500 trees, 5000x200 samples): 3.32 seconds, 123.3 MB Peak RAM
# - Scenario D (IO-Heavy, 1,000,000 SMILES parsed): 173,684 Parses/Sec, 86.9 MB Peak RAM

# Matched single-thread Ferrum vs RDKit descriptor bundle benchmark (57 outputs)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
  cargo run -p ferrum_cli --release --bin parity_bundle_benchmark
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/benchmark_rdkit_parity_bundle.py

# Matched single-thread CSV bundle benchmark (throughput + peak RSS)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
  FERRUM_BENCH_CSV_LIMIT=1000 FERRUM_BENCH_REPEATS=11 FERRUM_BENCH_WARMUP=100 \
  cargo run -p ferrum_cli --release --bin parity_bundle_csv_benchmark
# 10k pinned run (used for latest README numbers)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
  FERRUM_BENCH_CSV=coconut_csv_lite-10000-tui.csv FERRUM_BENCH_CSV_COLUMN=canonical_smiles \
  FERRUM_BENCH_CSV_LIMIT=10000 FERRUM_BENCH_REPEATS=3 FERRUM_BENCH_WARMUP=100 \
  cargo run -p ferrum_cli --release --bin parity_bundle_csv_benchmark --quiet > /tmp/ferrum_parity57_10k.tsv
# Optional streaming ingestion mode (avoids preloading all SMILES in benchmark process)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
  FERRUM_BENCH_CSV_MODE=stream FERRUM_BENCH_CSV_CHUNK_SIZE=1 \
  FERRUM_BENCH_CSV_LIMIT=1000 FERRUM_BENCH_REPEATS=11 FERRUM_BENCH_WARMUP=100 \
  cargo run -p ferrum_cli --release --bin parity_bundle_csv_benchmark
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  FERRUM_BENCH_CSV_LIMIT=1000 FERRUM_BENCH_REPEATS=5 FERRUM_BENCH_WARMUP=100 \
  python scripts/benchmark_rdkit_parity_bundle_csv.py

# Per-descriptor Ferrum vs RDKit benchmark matrix (10k rows example)
FERRUM_BENCH_CSV=coconut_csv_lite-10000-tui.csv FERRUM_BENCH_CSV_COLUMN=canonical_smiles \
  FERRUM_BENCH_CSV_LIMIT=10000 FERRUM_BENCH_WARMUP=100 \
  cargo run -p ferrum_cli --release --bin descriptor_100k_benchmark --quiet > /tmp/ferrum_each_descriptor.tsv
python scripts/benchmark_rdkit_each_descriptor.py > /tmp/rdkit_each_descriptor.tsv

# CI performance guardrail checker (expects Ferrum and RDKit TSV rows with throughput + peak_rss_bytes)
PERF_GUARD_MIN_SPEEDUP=2.0 PERF_GUARD_MIN_RSS_REDUCTION=1.5 \
  FERRUM_BENCH_RESULT=/tmp/ferrum_parity57.tsv \
  RDKIT_BENCH_RESULT=/tmp/rdkit_parity57.tsv \
  python scripts/check_perf_guardrails.py

# ML synthetic guardrails (100k x 1)
FERRUM_ML_BENCH_ROWS=100000 FERRUM_ML_BENCH_COLS=1 FERRUM_ML_BENCH_REPEATS=3 \
  cargo run -p ferrum_cli --release --bin ml_benchmark --quiet > /tmp/ml_bench_100k.tsv
FERRUM_ML_AUTOQSAR_ROWS=100000 FERRUM_ML_AUTOQSAR_COLS=1 FERRUM_ML_AUTOQSAR_FOLDS=3 FERRUM_ML_AUTOQSAR_REPEATS=3 \
  cargo run -p ferrum_cli --release --bin autoqsar_benchmark --quiet > /tmp/autoqsar_100k.tsv
PERF_ML_AD_FIT_MAX_MS=12000 PERF_ML_AD_PREDICT_MAX_MS=12000 PERF_ML_AUTOQSAR_MAX_MS=120000 \
  FERRUM_ML_BENCH_RESULT=/tmp/ml_bench_100k.tsv \
  FERRUM_AUTOQSAR_BENCH_RESULT=/tmp/autoqsar_100k.tsv \
  python scripts/check_ml_perf_guardrails.py

# AutoQsar parallel policy and optional pruning knobs
# FERRUM_AUTOQSAR_PARALLEL_MODE: candidate (default) | fold
# FERRUM_AUTOQSAR_MAX_THREADS: optional thread cap for AutoQsar internal work
# FERRUM_AUTOQSAR_PRUNE_KEEP: optional top-K candidates to keep after quick pass
# FERRUM_AUTOQSAR_PRUNE_FOLDS: quick-pass folds when pruning is enabled (default 2)
# FERRUM_RF_PRESORT_MAX_BYTES: Random Forest root pre-sort memory budget (default 268435456)
FERRUM_AUTOQSAR_PARALLEL_MODE=candidate \
FERRUM_AUTOQSAR_MAX_THREADS=8 \
FERRUM_AUTOQSAR_PRUNE_KEEP=6 \
FERRUM_AUTOQSAR_PRUNE_FOLDS=2 \
FERRUM_ML_AUTOQSAR_ROWS=100000 FERRUM_ML_AUTOQSAR_COLS=1 FERRUM_ML_AUTOQSAR_FOLDS=3 FERRUM_ML_AUTOQSAR_REPEATS=3 \
  cargo run -p ferrum_cli --release --bin autoqsar_benchmark --quiet > /tmp/autoqsar_100k_pruned.tsv

# Real-data AutoQsar benchmark with stage timestamps (uses precomputed feature matrix CSV)
FERRUM_ML_REAL_FEATURES_CSV=coconut_csv_lite-100000-tui_features.csv \
FERRUM_ML_REAL_TARGETS_CSV=coconut_csv_lite-100000-tui.csv \
FERRUM_ML_REAL_TARGET_COL=alogp \
FERRUM_ML_REAL_LIMIT=100000 FERRUM_ML_REAL_FOLDS=3 FERRUM_ML_REAL_REPEATS=1 \
  cargo run -p ferrum_cli --release --bin autoqsar_csv_benchmark --quiet > /tmp/autoqsar_csv_100k.tsv

# Optional large-N AD approximation mode (default remains exact)
FERRUM_AD_MODE=approx FERRUM_AD_APPROX_MIN_ROWS=50000 FERRUM_AD_APPROX_STEP=4 \
  cargo run -p ferrum_cli --release --bin autoqsar_benchmark --quiet

# Native BLAS/LAPACK backend is enabled by default for PCA/AD linear algebra
# (requires system OpenBLAS/LAPACK availability).
cargo test -p ferrum_core --release --lib --quiet
# To force pure-Rust fallback path explicitly:
cargo test -p ferrum_core --release --no-default-features --lib --quiet
```

### CI guardrails

The repository now includes a GitHub Actions workflow at `.github/workflows/ci.yml` with two jobs:
- `test-release`: runs release-mode `ferrum_core` and `ferrum_cli` tests.
- `perf-guardrails`: runs the matched 57-output 10k Ferrum/RDKit CSV parity benchmarks and enforces minimum performance thresholds via `scripts/check_perf_guardrails.py`.

Default guardrail thresholds:
- Ferrum speedup over RDKit (`molecules_per_second`): `>= 2.0x`
- Ferrum RSS reduction versus RDKit (`peak_rss_bytes`): `>= 1.5x`

Reproducibility details:
- CI pins thread-related env vars (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `RAYON_NUM_THREADS` where applicable) to keep benchmark runs single-threaded.
- CI uploads `/tmp/ferrum_parity57.tsv`, `/tmp/rdkit_parity57.tsv`, and `/tmp/perf_guardrail_summary.json` as workflow artifacts for auditability.

### Phase 14 TUI — Interactive QSAR Workbench (Tab-Based, Non-Blocking)

FerrumChem now includes a rewritten TUI designed for lab workflows:
- **Persistent tab navigation** (`Descriptor Studio`, `ML Pipeline`, `Visualization & Analytics`)
- **Non-blocking architecture** (UI thread + worker thread over `mpsc`)
- **60 FPS rendering loop** with computing popup/spinner during long operations
- **Target toggle list** for selecting one or many supervised prediction targets
- **Visual file picker modal** for datasets, prediction inputs, and model bundles (no path typing required)

#### Quick Launch

```bash
cargo run -p ferrum_cli --release --bin tui
```

#### Data Input

- The file browser supports:
  - `.csv` files (parsed through `CsvReader`, auto SMILES detection)
  - `.smi` / `.smiles` files (first token per non-empty, non-comment line)
- Model-bundle import browser (key: `u` in ML/Analyze) supports:
  - `.fcbundle` / `.bin` persisted bundle artifacts
- Invalid molecule rows are skipped and reported in status.

#### Tab 1: Descriptor Studio

- Load molecules with `f` (opens file picker popup).
- Toggle descriptor set with checkbox list:
  - `Space`: toggle selected descriptor
  - `a`: select all descriptors
  - `n`: select none
  - `r`: apply curated **Fast ML** preset (balanced speed/coverage, no high-dimensional fingerprints)
  - `↑/↓` or `j/k`: move selection
- Each descriptor row now shows how many values it contributes (e.g., scalar `1`, vector families, and fingerprints).
- The left control pane shows the total selected output width (`values per molecule`) so feature-matrix size is visible before running.
- Press `Enter` to compute descriptors in the worker thread.
- Press `Esc` during long-running compute to request cancellation (descriptor calculation, model training, ART2-A clustering, prediction feature generation).
- Press `v` to apply variance-threshold filtering (default `1e-12`) on the current matrix.
- Press `c` to apply correlation filtering (default `|r| > 0.95`) on the current matrix.
- Press `e` to export `SMILES + current (possibly filtered) feature matrix` to CSV.
- Output includes a descriptor matrix preview table (first rows/columns).

#### Tab 2: ML Pipeline

- `t`: open target selection popup (numeric CSV columns); toggle one target or many.
- `m`: toggle training mode:
  - `AutoQsar` (default): one-click automatic model selection/tuning.
  - `Manual`: choose one explicit model and tune its parameters.
- Mode behavior:
  - no selected targets => unsupervised/vector-only workflows (`ART2-A`, PCA, feature profiling)
  - one or more selected targets => supervised training is enabled
- `a` / `A`: in `Manual` mode, cycle manual model (`Random Forest`, `k-NN`, `PLS`, `Elastic Net`).
- `z`: toggle scaler (`StandardScaler` / `MinMaxScaler`).
- `x`: cycle train/test split strategy (`Random`, `Stratified`, `Scaffold`).
  - `Stratified` is target-dependent and only available when a valid primary target vector is present.
  - When target validity is lost, stratified split auto-falls back to `Random`.
- `,` / `.`: in `Manual` mode, cycle hyperparameter pages for models with extended controls.
- `1..6`: edit currently visible hyperparameters (CV/test/seed + model-specific page + dedicated ART2-A page).
- `g`: run ART2-A clustering on current feature matrix (target not required).
- `y`: toggle adding aligned CSV numeric columns into ART2-A clustering features (default: on).
- ART2-A clustering uses the currently selected scaler (`z`) before clustering.
- ART2-A quick tuning keys in Tab 2:
  - `[` / `]`: decrease/increase vigilance (`rho`)
  - `;` / `'`: decrease/increase learning rate (`eta`)
  - `-` / `=`: decrease/increase max iterations
- ART2-A parameters are also available in the parameter pager (``,`` / ``.`` + `1..6`) as:
  - `ART2-A rho`
  - `ART2-A eta`
  - `ART2-A max iter`
- If ART2-A reports a single cluster, increase vigilance (`]`) and/or switch scaler (`z`) and run `g` again.
- `Enter`: train supervised model bundle (one model per selected target) on the current descriptor matrix.
- `p`: choose a prediction input file (`.csv` / `.smi`).
- `s`: export current prediction table to CSV.
- `b`: export persisted model bundle (`.fcbundle`) for reuse.
- `u`: import persisted model bundle (`.fcbundle` / `.bin`) for prediction without retraining.
- Training status now includes `Q2`, train metrics (`R2`, `RMSE`, `MAE`, Pearson `r`) and holdout split metrics.
- Prediction now returns one column per selected target (full multi-output prediction by default).
- If ART2-A clustering is available, prediction additionally emits:
  - `ART2A_ClusterId`
  - `ART2A_Similarity`
  - `ART2A_InDomain` (`1.0`/`0.0`)
- `p` now supports supervised-only, ART2-A-only, or combined outputs in one pass.
- You can still predict just one value by selecting only one target in the `t` popup.
- The UI validates full feature-layout compatibility between training and prediction runs, including feature-subset filters applied in Tab 1.

#### Tab 3: Visualization & Analytics

- **Top pane**: PCA scatter (`PC1` vs `PC2`) on current feature matrix.
- Color mode toggle (`o`) cycles:
  - uniform
  - target overlay buckets (from the currently selected primary target, when valid)
  - ART2-A cluster IDs (when clustering has been run)
  - applicability domain in/out buckets (green/red, when model AD state is available)
- **Bottom-left pane**: regression diagnostics:
  - Predicted vs Actual scatter with `y=x` reference
  - residual summary (`Train R²/RMSE/MAE/Pearson`, residual bias, residual `σ`, `|residual| p95`, outlier counts, holdout metrics)
  - residual z-score distribution bins (histogram-like table)
- Regression diagnostics explicitly label the current **primary trained target** (`[CSV]` or `[DESC]` source visible).
- **Bottom-right top pane**: top-10 feature-importance table (RF/PLS/Elastic Net).
- Use `[` / `]` in Tab 3 to cycle which trained target is shown in feature-importance view.
  - The panel title now shows the active target + model.
  - Targets whose model does not expose importances (e.g., k-NN) show a clear placeholder.
- **Bottom-right bottom pane**: descriptor profiling table:
  - `Descriptor | Mean | Min | Max | StdDev | ZeroVar`
  - zero-variance rows highlighted
  - scroll with `↑/↓` or `j/k`

#### Worker-Thread Architecture (Implementation Notes)

- **Non-blocking UI:** Main thread purely handles `crossterm`/`ratatui` renders and keyboard input.
- **Worker Threading:** Expensive CPU operations run in a dedicated `std::thread` channel communicating via `WorkerRequest` and `WorkerResponse` enums.
- **Crash Recovery & Stability:** The worker thread is explicitly allocated an 8MB stack size to prevent stack overflows during deep recursive operations (e.g. Random Forest Tree building). If the worker encounters a fatal error and disconnects, the TUI detects the channel closure and gracefully restarts the background worker without crashing the application.
- **Progress Tracking:** Standard progress updates (`0.0` to `1.0` floats) mapped to the main UI spinner and gauge blocks.
- **Shared Memory:** Minimal copying. Target features (e.g. `NdArray` slices) are processed in background and summarized into metrics struct passed via channel on completion.

The TUI uses explicit request/response channels:
- `WorkerRequest::{LoadDataset, CalculateDescriptors, TrainModel, RunArt2aClustering, PredictFile, ExportPredictions, ApplyVarianceFilter, ApplyCorrelationFilter, ExportFeatureMatrix, ExportModelBundle, ImportModelBundle}`
- `WorkerResponse::{DatasetLoaded, DescriptorsCalculated, ModelTrained, ClusteringCompleted, PredictionsReady, PredictionsExported, FeatureFilterApplied, FeatureMatrixExported, ModelBundleExported, ModelBundleImported, JobCanceled, Error}`

Heavy tasks (`calculate_batch_flat_static`, configurable model training/CV/splitting, ART2-A clustering, batch prediction, export I/O) run off the render thread, so the interface remains responsive.

#### Keybindings Summary

| Key | Action |
|---|---|
| `q` | Quit TUI |
| `Left/Right` or `h/l` | Switch top-level tabs |
| `Esc` | Close popup / cancel current edit |
| `f` (Tab 1) | Open dataset file browser |
| `Enter` (Tab 1) | Run descriptor calculation |
| `Space`, `a`, `n`, `r` (Tab 1) | Toggle descriptor selection (`r` = Fast ML preset) |
| `v` (Tab 1) | Apply variance threshold filter to current feature matrix |
| `c` (Tab 1) | Apply correlation filter to current feature matrix |
| `e` (Tab 1) | Export current feature matrix CSV |
| `t` (Tab 2) | Open target selection popup (Space toggle, `a` all, `n` none) |
| `m` (Tab 2) | Toggle training mode (`AutoQsar` / `Manual`) |
| `a` / `A` (Tab 2) | In Manual mode, cycle model (`RF` / `k-NN` / `PLS` / `Elastic Net`) |
| `z` (Tab 2) | Toggle scaler (`StandardScaler` / `MinMaxScaler`) |
| `x` (Tab 2) | Cycle split strategy (`Random` / `Stratified` / `Scaffold`, stratified requires valid primary target) |
| `,` / `.` (Tab 2) | In Manual mode, cycle hyperparameter page |
| `1..6` (Tab 2) | Edit visible hyperparameters (includes ART2-A page) |
| `g` (Tab 2) | Run ART2-A clustering on current feature matrix |
| `y` (Tab 3) | Toggle CSV numeric columns included in ART2-A clustering |
| `[` / `]` (Tab 2) | ART2-A vigilance down/up |
| `;` / `'` (Tab 2) | ART2-A learning rate down/up |
| `-` / `=` (Tab 2) | ART2-A max-iter down/up |
| `Enter` (Tab 2) | Train selected model bundle (one model per selected target) |
| `p` (Tab 2) | Open prediction file browser |
| `s` (Tab 2) | Export prediction CSV |
| `b` (Tab 2/3) | Export current model bundle (`.fcbundle`) |
| `u` (Tab 2/3) | Import model bundle (`.fcbundle` / `.bin`) |
| `o` (Tab 3) | Cycle color mode (`uniform` / `target overlay` / `cluster` / `AD`) |
| `[` / `]` (Tab 3) | Cycle feature-importance target across trained outputs |
| `↑/↓` or `j/k` | List/table navigation and profile scrolling |

#### Descriptor/Fingerprint Coverage in TUI

The TUI now uses the expanded `BatchDescriptor::all()` set (**66 descriptors/fingerprints**), covering:

- Scalar physicochemical/topological descriptors (MW, LogP, TPSA, QED, Bertz, Balaban, Ipc, etc.)
- Multi-output descriptor families (Chi, Kappa, E-State, autocorrelation, BCUT, ring and functional-group vectors, VSA bins)
- Fingerprints (`MACCS-166`, `PubChem-881`, `ECFP4`, `FCFP4`, `AtomPair`, `TopologicalTorsion`)

With all options enabled, this expands to roughly **9.5k numeric columns** in the feature matrix.

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

CSV bundle slice (`coconut_csv_lite-10000-tui.csv`, 57 outputs) currently reports both throughput and peak RSS in matched output columns:

- FerrumChem (10k rows, buffered, median of 3 runs): `2488.07 mol/s`, peak RSS `10.11 MiB`
- RDKit (10k rows, median of 3 runs): `612.56 mol/s`, peak RSS `128.38 MiB`

This is currently:
- **4.06x throughput advantage** for Ferrum vs RDKit
- **12.70x lower peak RSS** for Ferrum vs RDKit

The CSV benchmark output includes `peak_rss_bytes` and `peak_rss_mb` for direct memory-footprint comparisons.

#### 10k parity summary (release-mode, RDKit reference CSV)

Reference generation used the first 10,000 rows from `coconut_csv_lite-10000-tui.csv` and ran the core parity suites with:
`FERRUM_RDKIT_REFERENCE_ROWS=10000 cargo test -p ferrum_core --release --test rdkit_comparison -- --nocapture --test-threads=1 --skip rdkit_ipc_relative_error_diagnostics --skip rdkit_avgipc_diagnostics`

| Suite | Total checks | Passed | Failed | Pass rate |
|---|---:|---:|---:|---:|
| Scalar | 220,000 | 218,601 | 1,399 | 99.4% |
| Ring | 40,000 | 39,773 | 227 | 99.4% |
| VSA | 360,000 | 358,604 | 1,396 | 99.6% |
| Chi/Kappa | 100,000 | 99,862 | 138 | 99.9% |

### 100k campaign (release-mode, single-thread pinned)

All numbers below are from the latest full 100k wave (2026-03-24), using:
- dataset: `coconut_csv_lite-02-2026.csv`
- pinned threading env (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `RAYON_NUM_THREADS=1` where applicable)

#### 100k parity summary (RDKit reference vs FerrumChem)

| Suite | Total checks | Passed | Failed | Pass rate | Runtime |
|---|---:|---:|---:|---:|---:|
| Scalar | 2,200,000 | 2,184,572 | 15,428 | 99.3% | 59.16s |
| Ring | 400,000 | 396,931 | 3,069 | 99.2% | 11.63s |
| VSA | 3,600,000 | 3,583,114 | 16,886 | 99.5% | 22.30s |
| Chi/Kappa | 1,000,000 | 998,286 | 1,714 | 99.8% | 36.03s |

Scalar descriptor pass rates on 100k (selected):
- `Ipc`: `94,998 / 100,000` (**95.0%**)
- `LogP`: `98,640 / 100,000` (**98.6%**)
- `MolarRefractivity`: `98,623 / 100,000` (**98.6%**)
- `AromaticAtomCount`: `98,759 / 100,000` (**98.8%**)
- `AromaticRingCount`: `99,027 / 100,000` (**99.0%**)
- `LabuteASA`: `98,765 / 100,000` (**98.8%**)
- `BertzCT`: `98,772 / 100,000` (**98.8%**)
- `QED`: `99,461 / 100,000` (**99.5%**)
- `TPSA`: `99,324 / 100,000` (**99.3%**)
- `RotatableBonds`: `99,852 / 100,000` (**99.9%**)

#### 100k matched CSV performance (57 outputs)

Fresh rerun (2026-03-31) on `coconut_csv_lite-100000-tui.csv`, single-threaded release mode (`FERRUM_BENCH_REPEATS=3`, `FERRUM_BENCH_WARMUP=200`):

| Tool | Rows | Seconds | Molecules/s | Peak RSS |
|---|---:|---:|---:|---:|
| FerrumChem | 100,000 | 42.720270 | 2,340.81 | 21.80 MB |
| RDKit | 100,000 | 165.757513 | 603.29 | 202.56 MB |

Relative 100k result:
- FerrumChem is **3.88x faster** than RDKit on matched throughput.
- FerrumChem uses **9.29x lower** peak RSS.

#### Per-descriptor Ferrum vs RDKit speed matrix (10k molecules)

Generated artifacts:
- Ferrum: `/tmp/ferrum_each_descriptor.tsv`
- RDKit: `/tmp/rdkit_each_descriptor.tsv` (via `scripts/benchmark_rdkit_each_descriptor.py`)
- Joined matrix: `/tmp/descriptor_speed_matrix.tsv`

Snapshot:
- Total matrix rows: `66`
- Directly comparable: `24`
- Ferrum-only: `42`
- RDKit-only: `0`

Top direct Ferrum speedups:
- `AromaticAtomCount`: `1017.61x`
- `HeteroatomCount`: `93.10x`
- `RotatableBondCount`: `35.80x`
- `FormalCharge`: `30.75x`
- `BondCount`: `24.75x`

Latest rerun after cache/cancellation work:
- Directly comparable: `24`
- Ferrum faster: `24`
- RDKit faster: `0`

Notable updates:
- `AromaticRingCount` now uses cached minimum cycle basis seeded at parse time and is no longer a Ferrum slowdown in direct comparison.

#### FerrumChem per-descriptor throughput on 100k molecules

| Descriptor | Feature len | Seconds | Molecules/s |
|---|---:|---:|---:|
| MolecularWeight | 1 | 0.010018 | 9,981,616.86 |
| LogP | 1 | 4.921086 | 20,320.72 |
| MolarRefractivity | 1 | 4.913023 | 20,354.07 |
| TPSA | 1 | 0.035065 | 2,851,843.15 |
| HBondDonor | 1 | 0.239927 | 416,793.59 |
| HBondAcceptor | 1 | 0.545074 | 183,461.29 |
| RotatableBondCount | 1 | 0.101474 | 985,469.26 |
| HeavyAtomCount | 1 | 0.002639 | 37,897,334.60 |
| AtomCount | 1 | 0.002865 | 34,905,536.89 |
| AromaticAtomCount | 1 | 0.002603 | 38,410,437.81 |
| HeteroatomCount | 1 | 0.002610 | 38,311,108.42 |
| AromaticRingCount | 1 | 9.486801 | 10,540.96 |
| FormalCharge | 1 | 0.002586 | 38,677,238.45 |
| FractionCSP3 | 1 | 0.257220 | 388,772.70 |
| BertzCT | 1 | 10.329350 | 9,681.15 |
| LabuteASA | 1 | 0.029436 | 3,397,157.43 |
| QED | 1 | 26.431232 | 3,783.40 |
| NumAmideBonds | 1 | 0.010941 | 9,139,897.28 |
| HallKierAlpha | 1 | 0.337229 | 296,534.22 |
| BalabanIndex | 1 | 8.383758 | 11,927.82 |
| BondCount | 1 | 0.000467 | 214,324,139.54 |
| SlogP_VSA | 12 | 4.954716 | 20,182.79 |
| SMR_VSA | 10 | 4.955727 | 20,178.67 |
| PEOE_VSA | 14 | 0.797403 | 125,407.11 |

### Accuracy & Validation
- **100% RDKit match**: 1944/1944 descriptor comparisons across 27 torture molecules
- **Current 1000-row parity**: scalar 99.8%, ring 100.0%, VSA 100.0%, Chi/Kappa 100.0%
- **Latest 100k parity campaign**: scalar 99.3%, ring 99.2%, VSA 99.5%, Chi/Kappa 99.8%
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
**Phase 15**: ✅ Complete — Infrastructure (canonical SMILES writer, Python bindings via PyO3)  
**Phase 16**: ✅ Active — 3D descriptors (conformer structures, PMI, Asphericity)

**Current Status**: Production-ready typed QSAR library with 72 typed descriptors, 100% torture-set RDKit agreement, current **1000-row parity** at scalar **99.8%** (only `Ipc` divergent), and latest full **100k parity** at scalar **99.3%**, ring **99.2%**, VSA **99.5%**, Chi/Kappa **99.8%**. Latest matched 100k CSV run shows Ferrum at **2341 mol/s** and **21.80 MB** peak RSS vs RDKit at **603 mol/s** and **202.56 MB**. ✅

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

### Learning the codebase (Rust + cheminformatics)

If your goal is to understand both Rust and cheminformatics internals, read in this order:

1. `crates/ferrum_core/src/lib.rs` — crate architecture and layer boundaries.
2. `crates/ferrum_core/src/mol/*` — molecular graph data model (`Atom`, `Bond`, `Molecule`).
3. `crates/ferrum_core/src/io/smiles.rs` and `io/smarts.rs` — parser/state-machine design.
4. `crates/ferrum_core/src/algo/*` — ring finding, aromaticity, valence, substructure matching.
5. `crates/ferrum_core/src/desc/typed.rs` then descriptor family modules — feature engineering layer.
6. `crates/ferrum_core/src/ml/*` — QSAR preprocessing, models, CV, pipeline orchestration.
7. `crates/ferrum_cli/src/bin/*` — practical examples, benchmarks, and TUI orchestration.

The source files now include module-level rustdoc headers focused on:
- chemistry intent (what problem is being solved),
- implementation strategy (how it is solved in Rust),
- and where each module sits in the end-to-end QSAR pipeline.

### Cancellation smoke-test coverage

TUI cancellation now has release-tested smoke coverage for all cancelable long-running worker paths:
- descriptor calculation
- supervised model training
- ART2-A clustering
- prediction feature generation

These tests assert that each handler returns `WorkerResponse::JobCanceled` when the shared cancellation flag is raised.

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
