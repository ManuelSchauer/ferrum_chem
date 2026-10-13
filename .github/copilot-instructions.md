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
└── crates/ferrum_core/tests/  # Integration tests; data/ holds the RDKit parity reference
```

`ferrum_core` has six main submodules:
- **mol/** – `Atom`, `Bond`, `Molecule` data structures
- **io/** – SMILES parser, SMARTS parser, CSV reader, SDF stub
- **algo/** – RDKit-compatible sanitization (`sanitize.rs`: valence, kekulization, RDKit aromaticity, hybridization), RDKit SSSR (`sssr.rs`), VF2 isomorphism, Gasteiger charges
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

# Integration tests (RDKit parity, regressions, SMARTS edge cases)
cargo test --package ferrum_core --release --test rdkit_parity -- --nocapture
cargo test --package ferrum_core --test smarts_edge
```

The RDKit parity reference (`tests/data/rdkit_parity_reference.csv`) is regenerated
with the reference RDKit version; element data comes from the same RDKit:
```bash
python scripts/generate_parity_reference.py
python scripts/gen_element_data.py
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
- `FerrumError::NotImplemented(String)` – unsupported features

### RDKit parity (tests/rdkit_parity.rs)
Every feature with an RDKit equivalent is compared per molecule with
`|a - b| <= 1e-9 + 1e-9 * |b|` (all values are f64 end to end); fingerprints and per-atom perception must match exactly.
Known deviations are listed per column in `tests/data/rdkit_parity_baseline.tsv`.
A column may never get worse; after a fix, tighten the baseline with
`FERRUM_UPDATE_PARITY_BASELINE=1 cargo test -p ferrum_core --release --test rdkit_parity`.

---

## Known Stubs / Not Yet Implemented

- Stereochemistry (`@`/`@@`, `/`/`\\`) is parsed but not used (descriptors are 2D)
