//! Shared atomic property lookup tables used by multiple descriptors.


//! ## File guide: Atomic Props
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn atomic_mass`
//! - `fn pauling_electronegativity`
//! - `fn burden_polarizability`
//! - `fn burden_vdw_volume`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Atomic Props` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Atomic Props` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/atomic_props.rs` defines atomic props behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Atomic Props` workflows.
//! - Repository unit and parity tests covering `Atomic Props` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/atomic_props.rs`.

/// Builds table.
///
/// Function context: `crates/ferrum_core/src/desc/atomic_props.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
const fn make_table(default: f64, overrides: &[(u8, f64)]) -> [f64; 119] {
    let mut table = [default; 119];
    let mut i = 0;
    while i < overrides.len() {
        let (idx, value) = overrides[i];
        table[idx as usize] = value;
        i += 1;
    }
    table
}

const PAULING_ELECTRONEGATIVITY: [f64; 119] = make_table(
    2.0,
    &[
        (1, 2.20),
        (5, 2.04),
        (6, 2.55),
        (7, 3.04),
        (8, 3.44),
        (9, 3.98),
        (14, 1.90),
        (15, 2.19),
        (16, 2.58),
        (17, 3.16),
        (35, 2.96),
        (53, 2.66),
    ],
);

const BURDEN_POLARIZABILITY: [f64; 119] = make_table(
    1.5,
    &[
        (1, 0.67),
        (5, 3.03),
        (6, 1.76),
        (7, 1.10),
        (8, 0.80),
        (9, 0.56),
        (14, 5.38),
        (15, 3.63),
        (16, 2.90),
        (17, 2.18),
        (35, 3.05),
        (53, 5.35),
    ],
);

const BURDEN_VDW_RADIUS: [f64; 119] = make_table(
    1.70,
    &[
        (1, 1.20),
        (5, 1.92),
        (6, 1.70),
        (7, 1.55),
        (8, 1.52),
        (9, 1.47),
        (14, 2.10),
        (15, 1.80),
        (16, 1.80),
        (17, 1.75),
        (35, 1.85),
        (53, 1.98),
    ],
);

const VDW_VOLUME_SCALE: f64 = (4.0 / 3.0) * std::f64::consts::PI;

#[inline]
/// Implements mass.
///
/// Function context: `crates/ferrum_core/src/desc/atomic_props.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub fn atomic_mass(atomic_number: u8) -> f64 {
    crate::mol::elements::ATOMIC_MASSES[atomic_number as usize] as f64
}

#[inline]
/// Implements electronegativity.
///
/// Function context: `crates/ferrum_core/src/desc/atomic_props.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub fn pauling_electronegativity(atomic_number: u8) -> f64 {
    PAULING_ELECTRONEGATIVITY[atomic_number as usize]
}

#[inline]
/// Implements polarizability.
///
/// Function context: `crates/ferrum_core/src/desc/atomic_props.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub fn burden_polarizability(atomic_number: u8) -> f64 {
    BURDEN_POLARIZABILITY[atomic_number as usize]
}

#[inline]
/// Implements vdw volume.
///
/// Function context: `crates/ferrum_core/src/desc/atomic_props.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub fn burden_vdw_volume(atomic_number: u8) -> f64 {
    let r = BURDEN_VDW_RADIUS[atomic_number as usize];
    VDW_VOLUME_SCALE * r * r * r
}
