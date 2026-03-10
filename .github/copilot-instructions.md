# FerrumChem – Copilot Instructions

## What This Project Is

A high-performance QSAR engine for cheminformatics written in pure Rust. It parses SMILES/SMARTS, computes 72 typed molecular descriptors (17,710+ dimensions), and provides a complete ML pipeline (KNN, Random Forest, PLS, ElasticNet, PCA, cross-validation, applicability domain). The goal is to replace Python/RDKit with a 10–100× faster, memory-efficient alternative.

**Status**: Phase 11 complete. 742 unit tests + 14 doc tests + 4 integration tests passing. 100% RDKit accuracy (1944/1944 comparisons across 27 test molecules).

---

## Workspace Layout

```
ferrum_chem/
├── crates/
│   ├── ferrum_core/      # Core library: all chemistry + ML logic (~25k LOC)
│   └── ferrum_cli/       # Thin CLI wrapper (early stage)
├── scripts/              # Python RDKit reference-generation scripts
└── crates/ferrum_core/tests/  # Integration tests + rdkit_reference*.csv
```

`ferrum_core` has six main submodules:
- **mol/** – `Atom`, `Bond`, `Molecule` data structures
- **io/** – SMILES parser, SMARTS parser, CSV reader, SDF stub
- **algo/** – SSSR rings, aromaticity (Hückel), Morgan canonicalization, VF2 isomorphism, Gasteiger charges
- **desc/** – 72 descriptor modules + `typed.rs` (API) + `batch.rs` (parallel)
- **ml/** – Dataset, scaling, PCA, KNN, RF, PLS, ElasticNet, ART-2A, CV, metrics, AD, QsarPipeline
- **error/** – `FerrumError` via `thiserror`

---

## Build & Test Commands

```bash
# Build
cargo build
cargo build --release          # Required for benchmarks

# Run all tests
cargo test --package ferrum_core --lib

# Run a single test by name
cargo test --package ferrum_core --lib -- <test_name>

# Integration tests
cargo test --package ferrum_core --test rdkit_comparison -- --nocapture
cargo test --package ferrum_core --test smarts_edge

# Benchmarks / examples
cargo run --release --bin benchmark
cargo run --bin descriptor_demo
cargo run --bin csv_demo
cargo run --bin smarts_demo
cargo run --bin thread_safety_demo
```

RDKit reference data is regenerated via Python when needed:
```bash
python3 scripts/generate_rdkit_reference.py
```

---

## Key Conventions

### Primitive Types (types.rs)
```rust
pub type AtomIdx = u16;   // max 65,535 atoms
pub type BondIdx = u16;
pub type Real    = f32;   // all float descriptors; switchable to f64
```

### Descriptor Trait
Every descriptor is a stateless unit struct implementing:
```rust
pub trait Descriptor: Send + Sync {
    fn name(&self) -> &str;
    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue;
}
```
`&mut Molecule` is needed because implicit-H counts may be populated lazily in-place. Descriptors never hold state.

### DescriptorValue Enum
Use the correct variant — do **not** coerce everything to `Float`:
```rust
DescriptorValue::Int(u32)          // exact counts
DescriptorValue::Float(f32)        // single float
DescriptorValue::Float64(f64)      // large-range floats
DescriptorValue::IntVec(Vec<u32>)
DescriptorValue::IntU16Vec(Vec<u16>) // memory-optimised counts ≤65,535
DescriptorValue::FloatVec(Vec<f32>)
DescriptorValue::Binary(BitVector)  // fingerprints (bit-packed, 4× reduction)
```

### Adding a New Descriptor
1. Create `crates/ferrum_core/src/desc/<name>.rs` with a unit struct.
2. Implement `Descriptor` for it.
3. Register it in `crates/ferrum_core/src/desc/mod.rs`.
4. Add unit tests inside the same file (`#[cfg(test)]`).

### Memory Optimisations
- `Molecule.adjacency` uses `SmallVec<[BondIdx; 4]>` — avoids heap allocation for typical atoms.
- Fingerprints stored as `BitVector` (1 bit/feature, not 32).
- `IntU16Vec` instead of `IntVec` for values that fit in u16.

### Parallelism
All descriptors must be `Send + Sync`. Use `batch.rs` / Rayon for batch processing; never add locks to descriptor logic.

### Error Handling
Return `Result<T, FerrumError>` everywhere. Use the appropriate variant:
- `FerrumError::ParseError(String)` – SMILES/SMARTS parse failures
- `FerrumError::ChemError(String)` – chemical impossibilities
- `FerrumError::NotImplemented(String)` – stubs (`todo!()` is also used in SDF/standardize stubs)

### Test Tolerances (rdkit_comparison)
```rust
const TIGHT_TOL:  f64 = 0.01;   // MW, basic counts
const MEDIUM_TOL: f64 = 0.1;    // LogP, TPSA
const LOOSE_TOL:  f64 = 1.0;    // complex descriptors
const VSA_TOL:    f64 = 0.5;    // VSA values
const CHI_TOL:    f64 = 0.05;   // Chi indices
```

---

## Known Stubs / Not Yet Implemented

- `io/sdf.rs` – SDF parser (contains `todo!()`)
- `algo/standardize.rs` – salt stripping stub
- Stereochemistry (`@`/`@@`) in SMILES parser
- No GitHub Actions CI yet
