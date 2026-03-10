# Phase 11: Legacy Removal & Typed Descriptor Refactoring — COMPLETE

## Summary

Phase 11 eliminated the dual-API architecture (legacy `Descriptor` + `TypedDescriptor`) in favor of a single, clean typed descriptor API. All 60 descriptors now use one trait with proper return types.

## What Changed

### Before (Dual API)
```rust
// Two traits, confusing naming
pub trait Descriptor: Send + Sync {            // "Legacy"
    fn name(&self) -> String;
    fn size(&self) -> usize;
    fn calculate(&self, mol: &Molecule, output: &mut [Real]);
}

pub trait TypedDescriptor: Send + Sync {       // "Typed"
    fn name(&self) -> &str;
    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue;
}

// Two structs per descriptor
pub struct LogP;           // legacy
pub struct LogPTyped;      // typed
```

### After (Single API)
```rust
pub trait Descriptor: Send + Sync {
    fn name(&self) -> &str;
    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue;
}

pub struct LogP;  // One struct, clean name
```

### Key Metrics

| Metric | Before | After |
|---|---|---|
| Traits | 2 | **1** |
| Struct naming | `LogP` + `LogPTyped` | **`LogP`** |
| Return type | `&mut [Real]` | **`DescriptorValue`** enum |
| Descriptors | 59 (each duplicated) | **60** (clean, including new `SingleAtomCount`) |
| Tests | 579 | **568** (removed redundant, added typed) |
| Doc tests | 16 | **14** |
| Net code | | **-645 lines** |

### DescriptorValue Enum (Memory-Efficient Types)
```rust
pub enum DescriptorValue {
    Int(u32),              // atom/bond counts
    Float(f32),            // MW, LogP, TPSA
    IntVec(Vec<u32>),      // bond type counts
    IntU16Vec(Vec<u16>),   // element counts, ring sizes
    FloatVec(Vec<f32>),    // chi indices, autocorrelation
    Binary(BitVector),     // fingerprints (1 bit per feature)
}
```

### Memory Savings
- Fingerprints: **31x reduction** (664 bytes → 21 bytes for MACCS)
- Atom counts: **4x reduction** (f32 → u32, proper type)
- Full suite: **32x reduction** (67.5 GB → 2.13 GB for 1M molecules)

## All 60 Descriptors

### Molecular Properties (12)
`MolecularWeight`, `LogP`, `MolarRefractivity`, `Tpsa`, `Qed`, `FormalCharge`, `LipinskiViolations`, `FractionCSP3`, `HallKierAlpha`, `BertzCT`, `Ipc`, `FragmentComplexity`

### Atom Counts (5)
`AtomCount`, `HeavyAtomCount`, `ElementCounts`, `SingleAtomCount`, `AromaticAtomCount`

### Bond Counts (3)
`BondCount`, `BondCountDetailed`, `RotatableBondCount`

### Ring & Structure (7)
`AromaticRingCount`, `SmallRingDescriptor`, `RingTypeCounts`, `NumAmideBonds`, `AromaticAtomFraction`, `CarbonTypes`, `PiSystemCounts`

### H-Bond (2)
`HBondDonor`, `HBondAcceptor`

### Topological Indices (12)
`WienerIndex`, `ZagrebIndices`, `BalabanIndex`, `PetitjeanIndex`, `ChiIndices`, `KappaIndices`, `EStateIndices`, `Autocorrelation`, `BCUT`, `WeightedPath`, `MDE`, `LargestPiSystem`

### Counting & Electronic (5)
`ValenceElectrons`, `RadicalElectrons`, `HeteroatomCount`, `HybridizationRatio`, `VAdjMa`

### Chain Descriptors (2)
`LongestAliphaticChain`, `LargestChain`

### VSA (3)
`SlogpVSA`, `SmrVSA`, `LabuteASA`

### Functional Groups (3)
`FunctionalGroupCounts`, `AcidicGroupCount`, `BasicGroupCount`

### Fingerprints (6)
`ECFP`, `FCFP`, `MaccsKeys`, `PubChem881`, `AtomPairFingerprint`, `TopologicalTorsionFingerprint`

## Files Changed (42 files)
- **34 descriptor files**: Removed legacy impls, renamed *Typed → clean names
- **mod.rs**: Removed `LegacyDescriptor` trait, updated exports and tests
- **typed.rs**: Removed `TypedDescriptor` alias
- **batch.rs**: Rewritten to use new `Descriptor` trait
- **logp_old.rs**: Deleted (dead code)
- **memory_benchmark.rs**: Updated to new API

## Extracted Computation Functions
For cross-descriptor reuse (e.g., Qed, Lipinski):
- `molecular_weight::compute_molecular_weight(mol) -> f64`
- `logp::compute_logp(mol) -> (f64, f64)`
- `hbond::compute_hbond_donors(mol) -> u32`
- `hbond::compute_hbond_acceptors(mol) -> u32`
- `tpsa::compute_tpsa(mol, include_s_p) -> f64`
- `bond_count::compute_rotatable_bond_count(mol) -> u32`
- `aromatic_ring_count::compute_aromatic_ring_count(mol) -> u32`
- `atom_count::compute_atom_count(mol) -> u32`
- `atom_count::compute_heavy_atom_count(mol) -> u32`
- `atom_count::compute_aromatic_atom_count(mol) -> u32`

## New Feature: SingleAtomCount
```rust
let carbon_count = SingleAtomCount::new("C");
let nitrogen_count = SingleAtomCount::new("N");
assert_eq!(carbon_count.calculate(&mut mol).as_int(), 6);
```

---

**Date**: February 11, 2026
**Tests**: 568 unit + 14 doc tests passing
**Code**: 17,325 lines across 41 descriptor files
