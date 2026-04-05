//! Electrotopological State (E-State) indices.
//!
//! Kier & Hall E-State encodes both electronic and topological
//! information for each atom:
//!
//!   S_i = I_i + ΔI_i
//!
//! where I_i is the intrinsic state (based on valence electrons and
//! connectivity) and ΔI_i is the perturbation from all other atoms
//! (decaying with squared topological distance).
//!
//! Output:
//! - Summary: sum, min, max of all atom E-States (3 values)
//! - 44 atom-type E-State sums covering common organic/drug atoms
//!
//! Reference: Kier, L.B. & Hall, L.H. (1999) "Molecular Structure
//!            Description: The Electrotopological State", Academic Press.

//! ## File guide: Estate
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn intrinsic_states`
//! - `fn compute_estates`
//! - `fn principal_quantum_number`
//! - `fn _compute_distance_matrix_unused`
//! - `fn classify_atom`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Estate` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Estate` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/estate.rs` defines estate behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Estate` workflows.
//! - Repository unit and parity tests covering `Estate` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/estate.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::elements::outer_electrons;
use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Core E-State computation
// ---------------------------------------------------------------------------

/// Compute intrinsic state I_i for each heavy atom.
/// I_i = ((2/N)² δᵥ + 1) / δ
/// where N = principal quantum number, δᵥ = valence delta, δ = sigma electrons (degree).
fn intrinsic_states(mol: &Molecule) -> Vec<f64> {
    mol.atoms
        .iter()
        .map(|atom| {
            let an = atom.atomic_number;
            if an == 0 || an == 1 {
                return 0.0;
            }

            let n = principal_quantum_number(an) as f64;
            let zv = outer_electrons(an) as f64;
            let h = atom.total_h() as f64;
            let delta = mol.get_degree(0) as f64; // placeholder, fixed below
            let delta_v = zv - h;

            // σ = number of sigma bonds to heavy atoms = degree
            let sigma = 0.0; // placeholder
            0.0 // placeholder
        })
        .collect()
}

/// Compute E-State values for all heavy atoms.
/// Returns (e_states, atom_types) where atom_types[i] is the E-State atom type index.
fn compute_estates(
    mol: &mut Molecule,
    ctx: &mut crate::desc::context::CalculationContext,
) -> Vec<f64> {
    let n = mol.num_atoms();
    if n == 0 {
        return vec![];
    }

    // Step 1: compute intrinsic state for each atom
    let intrinsic: Vec<f64> = (0..n)
        .map(|i| {
            let atom = &mol.atoms[i];
            let an = atom.atomic_number;
            if an == 0 || an == 1 {
                return 0.0;
            }

            let pqn = principal_quantum_number(an) as f64;
            let zv = outer_electrons(an) as f64;
            let h = atom.total_h() as f64;
            let delta = mol.get_degree(i as AtomIdx) as f64;

            if delta == 0.0 {
                return 0.0;
            }

            let delta_v = zv - h;

            // I = ((2/N)^2 * δᵥ + 1) / δ
            ((2.0 / pqn).powi(2) * delta_v + 1.0) / delta
        })
        .collect();

    // Step 2: compute BFS distance matrix
    let dist = crate::algo::paths::compute_unweighted_distance_matrix_cached(mol, ctx);

    // Step 3: compute perturbation ΔI for each atom
    let mut estates = intrinsic.clone();
    for i in 0..n {
        let mut delta_i = 0.0f64;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = dist[i * n + j];
            if d == 0 || d == 255 {
                continue;
            }
            // ΔI_ij = (I_i - I_j) / (d_ij + 1)^2
            delta_i += (intrinsic[i] - intrinsic[j]) / ((d as f64 + 1.0).powi(2));
        }
        estates[i] = intrinsic[i] + delta_i;
    }

    estates
}

/// Implements quantum number.
///
/// Function context: `crates/ferrum_core/src/desc/estate.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn principal_quantum_number(atomic_number: u8) -> u8 {
    match atomic_number {
        1..=2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        55..=86 => 6,
        _ => 7,
    }
}

#[allow(dead_code)]
/// Computes distance matrix unused.
///
/// Function context: `crates/ferrum_core/src/desc/estate.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn _compute_distance_matrix_unused(mol: &Molecule) -> Vec<u32> {
    let n = mol.num_atoms();
    let mut distances = vec![u32::MAX; n * n];

    for start in 0..n {
        distances[start * n + start] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(start as AtomIdx);

        while let Some(current) = queue.pop_front() {
            let current_dist = distances[start * n + current as usize];
            for &bond_idx in &mol.adjacency[current as usize] {
                let bond = &mol.bonds[bond_idx as usize];
                let neighbor = if bond.source == current {
                    bond.target
                } else {
                    bond.source
                };
                let nd = &mut distances[start * n + neighbor as usize];
                if *nd == u32::MAX {
                    *nd = current_dist + 1;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    distances
}

// ---------------------------------------------------------------------------
// E-State atom type classification
// ---------------------------------------------------------------------------

/// Classify an atom into one of 44 E-State atom types.
/// Returns an index 0..43 or None if unclassified.
fn classify_atom(mol: &Molecule, idx: usize) -> Option<usize> {
    let atom = &mol.atoms[idx];
    let an = atom.atomic_number;
    let deg = mol.get_degree(idx as AtomIdx);
    let h = atom.total_h() as usize;
    let arom = atom.is_aromatic();

    match an {
        6 => {
            // Carbon
            if arom {
                match (deg, h) {
                    (2, 1) => Some(ESTATE_aCH),         // aromatic CH
                    (2, 0) | (3, 0) => Some(ESTATE_aC), // aromatic C (substituted)
                    _ => Some(ESTATE_aCH),
                }
            } else {
                match (deg, h) {
                    (1, 3) => Some(ESTATE_sCH3),  // -CH3
                    (2, 2) => Some(ESTATE_ssCH2), // -CH2-
                    (3, 1) => Some(ESTATE_sssCH), // >CH-
                    (4, 0) => Some(ESTATE_ssssC), // >C<
                    (1, 2) => Some(ESTATE_dCH2),  // =CH2
                    (2, 1) => Some(ESTATE_dsCH),  // =CH-
                    (2, 0) => Some(ESTATE_ddC),   // =C= or =C(-)-
                    (3, 0) => Some(ESTATE_dssC),  // =C<
                    (1, 1) => Some(ESTATE_tCH),   // ≡CH
                    (1, 0) => Some(ESTATE_tCH),   // ≡C (terminal)
                    _ => Some(ESTATE_sCH3),
                }
            }
        }
        7 => {
            // Nitrogen
            if arom {
                match (deg, h) {
                    (2, 0) => Some(ESTATE_aaN),           // aromatic N (pyridine-like)
                    (2, 1) | (3, 1) => Some(ESTATE_aaNH), // aromatic NH (pyrrole-like)
                    _ => Some(ESTATE_aaN),
                }
            } else {
                match (deg, h) {
                    (1, 2) => Some(ESTATE_sNH2), // -NH2
                    (2, 1) => Some(ESTATE_ssNH), // >NH
                    (3, 0) => Some(ESTATE_sssN), // >N-
                    (1, 0) => Some(ESTATE_tN),   // ≡N or =N
                    (2, 0) => Some(ESTATE_dsN),  // =N-
                    _ => Some(ESTATE_sNH2),
                }
            }
        }
        8 => {
            // Oxygen
            if arom {
                Some(ESTATE_aO)
            } else {
                match (deg, h) {
                    (1, 1) => Some(ESTATE_sOH), // -OH
                    (2, 0) => Some(ESTATE_ssO), // -O-
                    (1, 0) => Some(ESTATE_dO),  // =O
                    _ => Some(ESTATE_sOH),
                }
            }
        }
        9 => Some(ESTATE_sF),   // F
        17 => Some(ESTATE_sCl), // Cl
        35 => Some(ESTATE_sBr), // Br
        53 => Some(ESTATE_sI),  // I
        16 => {
            // Sulfur
            if arom {
                Some(ESTATE_aS)
            } else {
                match (deg, h) {
                    (1, 1) => Some(ESTATE_sSH),
                    (2, 0) => Some(ESTATE_ssS),
                    (1, 0) => Some(ESTATE_dS),
                    (3, 0) => Some(ESTATE_dssS),
                    (4, 0) => Some(ESTATE_ddssS),
                    _ => Some(ESTATE_sSH),
                }
            }
        }
        15 => {
            // Phosphorus
            match (deg, h) {
                (3, 0) | (3, _) => Some(ESTATE_sssP),
                (4, 0) => Some(ESTATE_dsssP),
                _ => Some(ESTATE_sssP),
            }
        }
        _ => None,
    }
}

// E-State atom type indices (44 types)
const ESTATE_sCH3: usize = 0;
const ESTATE_ssCH2: usize = 1;
const ESTATE_sssCH: usize = 2;
const ESTATE_ssssC: usize = 3;
const ESTATE_dCH2: usize = 4;
const ESTATE_dsCH: usize = 5;
const ESTATE_ddC: usize = 6;
const ESTATE_dssC: usize = 7;
const ESTATE_tCH: usize = 8;
const ESTATE_aCH: usize = 9;
const ESTATE_aC: usize = 10;
const ESTATE_sNH2: usize = 11;
const ESTATE_ssNH: usize = 12;
const ESTATE_sssN: usize = 13;
const ESTATE_tN: usize = 14;
const ESTATE_dsN: usize = 15;
const ESTATE_aaN: usize = 16;
const ESTATE_aaNH: usize = 17;
const ESTATE_sOH: usize = 18;
const ESTATE_ssO: usize = 19;
const ESTATE_dO: usize = 20;
const ESTATE_aO: usize = 21;
const ESTATE_sF: usize = 22;
const ESTATE_sCl: usize = 23;
const ESTATE_sBr: usize = 24;
const ESTATE_sI: usize = 25;
const ESTATE_sSH: usize = 26;
const ESTATE_ssS: usize = 27;
const ESTATE_dS: usize = 28;
const ESTATE_aS: usize = 29;
const ESTATE_dssS: usize = 30;
const ESTATE_ddssS: usize = 31;
const ESTATE_sssP: usize = 32;
const ESTATE_dsssP: usize = 33;
// Additional combined types for summary statistics
const ESTATE_sNH3: usize = 34; // -NH3+ (charged)
const ESTATE_dNH: usize = 35; // =NH
const ESTATE_ssNH2: usize = 36; // >NH2 (charged)
const ESTATE_tC: usize = 37; // ≡C-
const ESTATE_ddsC: usize = 38; // ==C-
const ESTATE_ssssN: usize = 39; // quaternary N
const ESTATE_sssNH: usize = 40; // >NH (charged?)
const ESTATE_ssPH: usize = 41;
const ESTATE_dsssS: usize = 42;
const ESTATE_ddsssP: usize = 43;
const NUM_ESTATE_TYPES: usize = 44;

/// E-State atom type names (for CSV headers).
const ESTATE_NAMES: [&str; NUM_ESTATE_TYPES] = [
    "sCH3", "ssCH2", "sssCH", "ssssC", "dCH2", "dsCH", "ddC", "dssC", "tCH", "aCH", "aC", "sNH2",
    "ssNH", "sssN", "tN", "dsN", "aaN", "aaNH", "sOH", "ssO", "dO", "aO", "sF", "sCl", "sBr", "sI",
    "sSH", "ssS", "dS", "aS", "dssS", "ddssS", "sssP", "dsssP", "sNH3p", "dNH", "ssNH2p", "tC",
    "ddsC", "ssssNp", "sssNH", "ssPH", "dsssS", "ddsssP",
];

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

// (Legacy descriptors removed — use EStateIndices below)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements estate.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_estate(smiles: &str) -> Vec<f32> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match EStateIndices.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    /// Implements raw.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn estates_raw(smiles: &str) -> Vec<f64> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        compute_estates(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        )
    }

    #[test]
    /// Runs regression test for estate ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_estate_ethane() {
        let e = estates_raw("CC");
        assert_eq!(e.len(), 2);
        assert!(
            (e[0] - e[1]).abs() < 0.001,
            "symmetric: {} vs {}",
            e[0],
            e[1]
        );
    }

    #[test]
    /// Runs regression test for estate summary ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_estate_summary_ethane() {
        let v = calc_estate("CC");
        // v[0]=sum, v[1]=min, v[2]=max
        assert!(v[0] > 0.0, "sum > 0");
        assert!((v[1] - v[2]).abs() < 0.01, "min ≈ max for symmetric mol");
    }

    #[test]
    /// Runs regression test for estate ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_estate_ethanol() {
        let e = estates_raw("CCO");
        assert!(e[2] > e[0], "O E-State={} > C E-State={}", e[2], e[0]);
    }

    #[test]
    /// Runs regression test for estate sum conservation.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_estate_sum_conservation() {
        let mut mol = parse_smiles("CCCC").expect("valid");
        let estates = compute_estates(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        );
        let n = mol.num_atoms();
        let intrinsic_sum: f64 = (0..n)
            .map(|i| {
                let atom = &mol.atoms[i];
                let pqn = principal_quantum_number(atom.atomic_number) as f64;
                let zv = outer_electrons(atom.atomic_number) as f64;
                let h = atom.total_h() as f64;
                let delta = mol.get_degree(i as AtomIdx) as f64;
                if delta == 0.0 {
                    return 0.0;
                }
                ((2.0 / pqn).powi(2) * (zv - h) + 1.0) / delta
            })
            .sum();
        let estate_sum: f64 = estates.iter().sum();
        assert!(
            (estate_sum - intrinsic_sum).abs() < 0.001,
            "E-State sum {} ≈ intrinsic sum {}",
            estate_sum,
            intrinsic_sum
        );
    }

    #[test]
    /// Runs regression test for estate atom types ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_estate_atom_types_ethanol() {
        let v = calc_estate("CCO");
        // v[3 + ESTATE_sCH3] and v[3 + ESTATE_sOH] should be nonzero
        assert!(v[3 + ESTATE_sCH3] != 0.0, "should have sCH3");
        assert!(v[3 + ESTATE_sOH] != 0.0, "should have sOH");
    }

    #[test]
    /// Runs regression test for estate atom types benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_estate_atom_types_benzene() {
        let v = calc_estate("c1ccccc1");
        assert!(
            v[3 + ESTATE_aCH] != 0.0,
            "benzene should have aCH: {}",
            v[3 + ESTATE_aCH]
        );
    }

    #[test]
    /// Runs regression test for estate size.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_estate_size() {
        let v = calc_estate("CCO");
        assert_eq!(v.len(), 47, "E-State has 47 values");
    }

    #[test]
    /// Runs regression test for estate benzene symmetry.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_estate_benzene_symmetry() {
        let e = estates_raw("c1ccccc1");
        let first = e[0];
        for val in &e[1..] {
            assert!(
                (val - first).abs() < 0.001,
                "benzene atoms should be equal: {} vs {}",
                val,
                first
            );
        }
    }

    #[test]
    /// Runs regression test for estate aspirin.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_estate_aspirin() {
        let v = calc_estate("CC(=O)Oc1ccccc1C(=O)O");
        assert!(v[0] > 0.0, "aspirin E-State sum = {}", v[0]);
        assert!(v[2] > v[1], "max > min for complex mol");
    }
}

// ===========================================================================
// Typed API: E-State Indices
// ===========================================================================

/// E-State indices.
///
/// Returns FloatVec with 47 values:
/// - [0-2]: Summary (sum, min, max)
/// - [3-46]: 44 atom type E-State sums
pub struct EStateIndices;

impl Descriptor for EStateIndices {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "EStateIndices"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut values = Vec::with_capacity(47);

        let estates = compute_estates(mol, ctx);

        // Summary (3 values)
        if estates.is_empty() {
            values.push(0.0);
            values.push(0.0);
            values.push(0.0);
        } else {
            let sum: f64 = estates.iter().sum();
            let min = estates.iter().cloned().fold(f64::MAX, f64::min);
            let max = estates.iter().cloned().fold(f64::MIN, f64::max);
            values.push(sum as f32);
            values.push(min as f32);
            values.push(max as f32);
        }

        // Atom types (44 values)
        let mut atom_type_sums = vec![0.0f32; NUM_ESTATE_TYPES];
        for i in 0..mol.num_atoms() {
            if let Some(type_idx) = classify_atom(mol, i) {
                atom_type_sums[type_idx] += estates[i] as f32;
            }
        }
        values.extend(atom_type_sums);

        DescriptorValue::FloatVec(values)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/estate.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<EStateIndices>;
};
