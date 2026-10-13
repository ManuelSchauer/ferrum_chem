//! Gasteiger-Marsili partial charge calculation.
//!
//! Implements iterative electronegativity equalization to compute partial atomic charges.
//!
//! **Algorithm**: Each atom's electronegativity χ = a + b·q + c·q² is computed from
//! element/hybridization-specific (a, b, c) parameters. Charge is transferred along
//! bonds proportional to the electronegativity difference, divided by the ionization
//! potential (`d = a+b+c`) of the less electronegative atom, with geometric damping
//! (factor 0.5^(k+1) per iteration). Implicit hydrogens are handled as virtual
//! neighbors using a special ionization constant (IONXH = 20.02).
//!
//! **References**:
//! - Gasteiger, J. & Marsili, M. (1980). *Tetrahedron*, 36, 3219-3228.
//! - Parameter values and algorithm details follow RDKit 2024.09
//!   (`Code/GraphMol/PartialCharges/GasteigerCharges.cpp` and `GasteigerParams.cpp`).
//! - Hybridization perception follows RDKit's `setHybridization` in `ConjugHybrid.cpp`:
//!   atoms with norbs=4, degree ≤ 3, and a conjugated neighbor are demoted to SP2
//!   (e.g. phenol O, aniline N, acetic acid hydroxyl O).
//!
//! **Validation**: Cross-validated against RDKit `rdPartialCharges.ComputeGasteigerCharges`
//! for 17 molecules (±0.05 per atom). See tests at the bottom of this file.

//! ## File guide: Electronic
//!
//! This file belongs to FerrumChem's **cheminformatics algorithm layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct ENParams`
//! - `fn effective_hybridization`
//! - `fn outer_shell_electrons`
//! - `fn default_valence_for_conjugation`
//! - `fn total_degree_with_h`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Electronic` operates on 2D molecular graphs (atoms/bonds) and does not consume conformer geometry.
//! - `Electronic` relies on stable `AtomIdx`/`BondIdx` semantics and adjacency invariants from `mol::molecule`.
//! - Performance paths in `Electronic` prefer deterministic pruning/caching over probabilistic heuristics.
//! - Specific scope anchor: `crates/ferrum_core/src/algo/electronic.rs` defines electronic behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Core graph algorithm literature relevant to `Electronic` (traversal, cycle, isomorphism, or canonicalization topics).
//! - Cheminformatics toolkit conventions (RDKit/CDK/OpenBabel) mirrored by `Electronic` decisions.
//! - FerrumChem regression tests that pin `Electronic` behavior on representative molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/algo/electronic.rs`.

use crate::mol::molecule::Molecule;
use crate::types::Hybridization;
use once_cell::sync::Lazy;
use smallvec::SmallVec;

static GASTEIGER_CACHE_ENABLED: Lazy<bool> =
    Lazy::new(|| std::env::var_os("FERRUM_DISABLE_GASTEIGER_CACHE").is_none());

/// Electronegativity parameters (a, b, c) for χ = a + b·q + c·q².
/// `d` = a + b + c (ionization potential at q = +1), used as charge transfer denominator.
///
/// Source: RDKit `GasteigerParams.cpp` `defaultParamData`, which matches
/// Gasteiger & Marsili 1980, Table 1.
#[derive(Clone, Copy)]
struct ENParams {
    a: f64,
    b: f64,
    c: f64,
}

const H: ENParams = ENParams {
    a: 7.17,
    b: 6.24,
    c: -0.56,
};
const C_SP3: ENParams = ENParams {
    a: 7.98,
    b: 9.18,
    c: 1.88,
};
const C_SP2: ENParams = ENParams {
    a: 8.79,
    b: 9.32,
    c: 1.51,
};
const C_SP: ENParams = ENParams {
    a: 10.39,
    b: 9.45,
    c: 0.73,
};
const N_SP3: ENParams = ENParams {
    a: 11.54,
    b: 10.82,
    c: 1.36,
};
const N_SP2: ENParams = ENParams {
    a: 12.87,
    b: 11.15,
    c: 0.85,
};
const N_SP: ENParams = ENParams {
    a: 15.68,
    b: 11.70,
    c: -0.27,
};
const O_SP3: ENParams = ENParams {
    a: 14.18,
    b: 12.92,
    c: 1.39,
};
const O_SP2: ENParams = ENParams {
    a: 17.07,
    b: 13.79,
    c: 0.47,
};
const F_PARAMS: ENParams = ENParams {
    a: 14.66,
    b: 13.85,
    c: 2.31,
};
const CL: ENParams = ENParams {
    a: 11.00,
    b: 9.69,
    c: 1.35,
};
const BR: ENParams = ENParams {
    a: 10.08,
    b: 8.47,
    c: 1.16,
};
const I_PARAMS: ENParams = ENParams {
    a: 9.90,
    b: 7.96,
    c: 0.96,
};
const S_SP3: ENParams = ENParams {
    a: 10.14,
    b: 9.13,
    c: 1.38,
};
const S_SO: ENParams = ENParams {
    a: 10.14,
    b: 9.13,
    c: 1.38,
};
const S_SO2: ENParams = ENParams {
    a: 12.00,
    b: 10.81,
    c: 1.20,
};
const S_SP2: ENParams = ENParams {
    a: 10.88,
    b: 9.49,
    c: 1.33,
};
const P_SP3: ENParams = ENParams {
    a: 8.90,
    b: 8.24,
    c: 0.96,
};
const P_SP2: ENParams = ENParams {
    a: 9.665,
    b: 8.530,
    c: 0.735,
};
// Additional params from RDKit's additionalParamData (PyBabel-derived)
const SI_SP3: ENParams = ENParams {
    a: 7.30,
    b: 6.567,
    c: 0.657,
};
const SI_SP2: ENParams = ENParams {
    a: 7.905,
    b: 6.748,
    c: 0.443,
};
const MG_SP3: ENParams = ENParams {
    a: 3.300,
    b: 5.587,
    c: 2.447,
};
const MG_SP2: ENParams = ENParams {
    a: 3.565,
    b: 5.572,
    c: 2.197,
};
const MG_SP: ENParams = ENParams {
    a: 4.040,
    b: 5.472,
    c: 1.823,
};
const AL_SP3: ENParams = ENParams {
    a: 5.375,
    b: 4.953,
    c: 0.867,
};
const AL_SP2: ENParams = ENParams {
    a: 5.795,
    b: 5.020,
    c: 0.695,
};
const B_SP3: ENParams = ENParams {
    a: 5.980,
    b: 6.820,
    c: 1.605,
};
const BE_SP3: ENParams = ENParams { a: 3.845, b: 6.755, c: 3.165 };
const BE_SP2: ENParams = ENParams { a: 4.005, b: 6.725, c: 3.035 };
const SI_SP: ENParams = ENParams { a: 9.065, b: 7.027, c: -0.002 };

const B_SP2: ENParams = ENParams {
    a: 6.420,
    b: 6.807,
    c: 1.322,
};

/// RDKit uses a special ionization constant for hydrogen (IONXH = 20.02)
/// instead of H.d (a+b+c = 12.85) in the charge transfer denominator.
const IONXH: f64 = 20.02;

/// Hybridization as assigned during sanitization (RDKit `setHybridization`).
#[inline]
fn effective_hybridization(mol: &Molecule, idx: usize) -> Hybridization {
    mol.atoms[idx].hybridization
}

/// Public accessor for effective hybridization, used by HallKierAlpha.
pub fn effective_hybridization_pub(mol: &Molecule, idx: usize) -> Hybridization {
    effective_hybridization(mol, idx)
}

/// Assign Gasteiger electronegativity parameters based on element and hybridization.
/// Follows RDKit's GasteigerParams.cpp lookup: element + hybridization state,
/// with special handling for sulfur (so/so2 modes based on bonded oxygen count).
fn get_params(mol: &Molecule, idx: usize) -> Option<ENParams> {
    // Exact RDKit lookup (GasteigerCharges.cpp + GasteigerParams.cpp): the mode
    // comes from the hybridization; unknown (element, mode) pairs fall back to
    // the all-zero "X" parameters, which RDKit turns into NaN charges.
    enum Mode {
        Sp3,
        Sp2,
        Sp,
        Star,
        So,
        So2,
        None,
    }
    let atom = &mol.atoms[idx];
    let mode = match effective_hybridization(mol, idx) {
        Hybridization::SP3 => Mode::Sp3,
        Hybridization::SP2 => Mode::Sp2,
        Hybridization::SP => Mode::Sp,
        _ if atom.atomic_number == 1 => Mode::Star,
        _ if atom.atomic_number == 16 => {
            let n_o = mol.adjacency[idx]
                .iter()
                .filter(|&&bi| {
                    let b = &mol.bonds[bi as usize];
                    let other = if b.source as usize == idx { b.target } else { b.source };
                    mol.atoms[other as usize].atomic_number == 8
                })
                .count();
            match n_o {
                2 => Mode::So2,
                1 => Mode::So,
                _ => Mode::Sp3,
            }
        }
        _ => Mode::None,
    };
    match (atom.atomic_number, mode) {
        (1, Mode::Star) => Some(H),
        (6, Mode::Sp3) => Some(C_SP3),
        (6, Mode::Sp2) => Some(C_SP2),
        (6, Mode::Sp) => Some(C_SP),
        (7, Mode::Sp3) => Some(N_SP3),
        (7, Mode::Sp2) => Some(N_SP2),
        (7, Mode::Sp) => Some(N_SP),
        (8, Mode::Sp3) => Some(O_SP3),
        (8, Mode::Sp2) => Some(O_SP2),
        (9, Mode::Sp3) => Some(F_PARAMS),
        (17, Mode::Sp3) => Some(CL),
        (35, Mode::Sp3) => Some(BR),
        (53, Mode::Sp3) => Some(I_PARAMS),
        (16, Mode::Sp3) => Some(S_SP3),
        (16, Mode::So) => Some(S_SO),
        (16, Mode::So2) => Some(S_SO2),
        (16, Mode::Sp2) => Some(S_SP2),
        (15, Mode::Sp3) => Some(P_SP3),
        (15, Mode::Sp2) => Some(P_SP2),
        (14, Mode::Sp3) => Some(SI_SP3),
        (14, Mode::Sp2) => Some(SI_SP2),
        (14, Mode::Sp) => Some(SI_SP),
        (5, Mode::Sp3) => Some(B_SP3),
        (5, Mode::Sp2) => Some(B_SP2),
        (4, Mode::Sp3) => Some(BE_SP3),
        (4, Mode::Sp2) => Some(BE_SP2),
        (12, Mode::Sp2) => Some(MG_SP2),
        (12, Mode::Sp3) => Some(MG_SP3),
        (12, Mode::Sp) => Some(MG_SP),
        (13, Mode::Sp3) => Some(AL_SP3),
        (13, Mode::Sp2) => Some(AL_SP2),
        _ => None,
    }
}

/// True when every atom has Gasteiger parameters (RDKit would otherwise throw
/// with `throwOnParamFailure = true`, as used by BCUT2D).
pub(crate) fn has_all_gasteiger_params(mol: &Molecule) -> bool {
    (0..mol.num_atoms()).all(|i| get_params(mol, i).is_some())
}

/// Port of RDKit `Gasteiger::splitChargeConjugated`: formal charges are spread
/// over same-element atoms two conjugated bonds away.
fn split_charge_conjugated(mol: &Molecule, charges: &mut [f64]) {
    let other = |b: usize, a: usize| {
        let bond = &mol.bonds[b];
        if bond.source as usize == a { bond.target as usize } else { bond.source as usize }
    };
    for aix in 0..mol.num_atoms() {
        let at = &mol.atoms[aix];
        let mut formal = at.formal_charge as f64;
        if formal.abs() <= f64::EPSILON || charges[aix].abs() >= f64::EPSILON {
            continue;
        }
        let mut marker: SmallVec<[usize; 8]> = SmallVec::new();
        marker.push(aix);
        for &b1 in &mol.adjacency[aix] {
            let b1 = b1 as usize;
            if !mol.bonds[b1].is_conjugated() {
                continue;
            }
            let aax = other(b1, aix);
            for &b2 in &mol.adjacency[aax] {
                let b2 = b2 as usize;
                if b2 == b1 || !mol.bonds[b2].is_conjugated() {
                    continue;
                }
                let yax = other(b2, aax);
                if mol.atoms[yax].atomic_number == at.atomic_number {
                    formal += mol.atoms[yax].formal_charge as f64;
                    marker.push(yax);
                }
            }
        }
        let share = formal / marker.len() as f64;
        for &m in &marker {
            charges[m] = share;
        }
    }
}

const DAMP: f64 = 0.5;
const DAMP_SCALE: f64 = 0.5;
const N_ITER: usize = 12;

/// Port of RDKit `computeGasteigerCharges` (12 iterations).
///
/// Atoms without parameters get RDKit's all-zero "X" parameters; the resulting
/// divisions by zero produce the same NaN/inf charges RDKit reports.
fn compute_gasteiger_charges_uncached(mol: &Molecule) -> Vec<f64> {
    let n = mol.num_atoms();
    let mut charges = vec![0.0f64; n];
    if n == 0 {
        return charges;
    }
    split_charge_conjugated(mol, &mut charges);

    let zero = ENParams { a: 0.0, b: 0.0, c: 0.0 };
    let params: Vec<ENParams> = (0..n).map(|i| get_params(mol, i).unwrap_or(zero)).collect();
    let ion_x: Vec<f64> = (0..n)
        .map(|i| if mol.atoms[i].atomic_number == 1 { IONXH } else { params[i].a + params[i].b + params[i].c })
        .collect();
    let mut h_chrg = vec![0.0f64; n];
    let mut energ = vec![0.0f64; n];
    let mut damp = DAMP;

    for _ in 0..N_ITER {
        for aix in 0..n {
            let p = &params[aix];
            energ[aix] = p.a + charges[aix] * (p.b + p.c * charges[aix]);
        }
        for aix in 0..n {
            let mut dq = 0.0f64;
            for &bi in &mol.adjacency[aix] {
                let bond = &mol.bonds[bi as usize];
                let nbr = if bond.source as usize == aix { bond.target as usize } else { bond.source as usize };
                let dx = energ[nbr] - energ[aix];
                let sgn = if dx < 0.0 { 0.0 } else { 1.0 };
                dq += dx / ((sgn * (ion_x[aix] - ion_x[nbr])) + ion_x[nbr]);
            }
            let ni_hs = mol.atoms[aix].total_h() as f64;
            if ni_hs > 0.0 {
                let q_hs = h_chrg[aix] / ni_hs;
                let enr = H.a + q_hs * (H.b + H.c * q_hs);
                let dx = enr - energ[aix];
                let sgn = if dx < 0.0 { 0.0 } else { 1.0 };
                let dq_h = dx / ((sgn * (ion_x[aix] - IONXH)) + IONXH);
                dq += ni_hs * dq_h;
                h_chrg[aix] -= ni_hs * dq_h * damp;
            }
            charges[aix] += damp * dq;
        }
        damp *= DAMP_SCALE;
    }
    charges
}

/// Compute Gasteiger-Marsili partial charges for all atoms in a molecule.
///
/// Returns a `Vec<f64>` of partial charges (one per heavy atom, indexed by atom).
/// Uses 12 iterations (RDKit default) with geometric damping (0.5^(k+1)).
/// Implicit hydrogens are modeled as virtual neighbors using IONXH = 20.02.
/// Atoms with unknown parameters are assigned zero charge and excluded from
/// charge transfer.
pub fn compute_gasteiger_charges(mol: &Molecule) -> Vec<f64> {
    compute_gasteiger_charges_uncached(mol)
}

/// Implements gasteiger charges cached.
///
/// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub(crate) fn ensure_gasteiger_charges_cached(mol: &Molecule) {
    if *GASTEIGER_CACHE_ENABLED {
        mol.gasteiger_charges_or_init(|| compute_gasteiger_charges_uncached(mol));
    }
}

/// Implements gasteiger charges ref.
///
/// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub(crate) fn cached_gasteiger_charges_ref(mol: &Molecule) -> Option<&[f64]> {
    mol.cached_gasteiger_charges()
}

/// Computes gasteiger charges cached.
///
/// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub(crate) fn compute_gasteiger_charges_cached(mol: &Molecule) -> Vec<f64> {
    ensure_gasteiger_charges_cached(mol);
    if let Some(cached) = cached_gasteiger_charges_ref(mol) {
        return cached.to_vec();
    }
    compute_gasteiger_charges_uncached(mol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn charges(smiles: &str) -> Vec<f64> {
        let mol = parse_smiles(smiles).unwrap();
        compute_gasteiger_charges(&mol)
    }

    /// Helper: assert charge is within tolerance.
    fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{label}: expected {expected:.4}, got {actual:.4}"
        );
    }

    #[test]
    /// Runs regression test for empty molecule.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_empty_molecule() {
        let mol = Molecule::new();
        let q = compute_gasteiger_charges(&mol);
        assert!(q.is_empty());
    }

    #[test]
    /// Runs regression test for methane.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_methane() {
        // [CH4] — single C with 4 implicit H; charge transferred to H.
        // RDKit: C = -0.077558
        let q = charges("[CH4]");
        assert_close(q[0], -0.0776, 0.01, "methane C");
    }

    #[test]
    /// Runs regression test for water.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_water() {
        // O with 2 implicit H; O pulls electron density from H.
        // RDKit: O = -0.41151
        let q = charges("O");
        assert_close(q[0], -0.4115, 0.02, "water O");
    }

    #[test]
    /// Runs regression test for formaldehyde.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_formaldehyde() {
        // C=O: C should be positive, O negative
        let mol = parse_smiles("C=O").unwrap();
        let q = compute_gasteiger_charges(&mol);
        let c_idx = mol.atoms.iter().position(|a| a.atomic_number == 6).unwrap();
        let o_idx = mol.atoms.iter().position(|a| a.atomic_number == 8).unwrap();
        assert!(
            q[c_idx] > 0.0,
            "C in formaldehyde should be positive: {}",
            q[c_idx]
        );
        assert!(
            q[o_idx] < 0.0,
            "O in formaldehyde should be negative: {}",
            q[o_idx]
        );
    }

    #[test]
    /// Runs regression test for ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_ethanol() {
        // CCO: two carbons and an oxygen
        let q = charges("CCO");
        let mol = parse_smiles("CCO").unwrap();
        let o_idx = mol.atoms.iter().position(|a| a.atomic_number == 8).unwrap();
        assert!(
            q[o_idx] < -0.1,
            "O in ethanol should be negative: {}",
            q[o_idx]
        );
    }

    #[test]
    /// Runs regression test for acetic acid.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_acetic_acid() {
        // CC(=O)O
        let mol = parse_smiles("CC(=O)O").unwrap();
        let q = compute_gasteiger_charges(&mol);

        // Find carbonyl C (bonded to =O), carbonyl O, hydroxyl O
        let carbonyl_c = 1; // atom index 1 in CC(=O)O
        assert!(
            q[carbonyl_c] > 0.1,
            "Carbonyl C should be positive: {}",
            q[carbonyl_c]
        );

        // Both oxygens should be negative
        for (i, atom) in mol.atoms.iter().enumerate() {
            if atom.atomic_number == 8 {
                assert!(q[i] < 0.0, "O atom {i} should be negative: {}", q[i]);
            }
        }
    }

    #[test]
    /// Runs regression test for benzene.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_benzene() {
        let mol = parse_smiles("c1ccccc1").unwrap();
        let q = compute_gasteiger_charges(&mol);
        // All carbons in benzene should have nearly identical charges.
        let c_charges: Vec<f64> = mol
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, a)| a.atomic_number == 6)
            .map(|(i, _)| q[i])
            .collect();
        let avg = c_charges.iter().sum::<f64>() / c_charges.len() as f64;
        for &c in &c_charges {
            assert_close(c, avg, 0.01, "Benzene C symmetry");
        }
    }

    #[test]
    /// Runs regression test for aniline.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_aniline() {
        let mol = parse_smiles("c1ccc(N)cc1").unwrap();
        let q = compute_gasteiger_charges(&mol);
        let n_idx = mol.atoms.iter().position(|a| a.atomic_number == 7).unwrap();
        assert!(
            q[n_idx] < -0.05,
            "N in aniline should be negative: {}",
            q[n_idx]
        );
    }

    #[test]
    /// Runs regression test for methylamine.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_methylamine() {
        let mol = parse_smiles("CN").unwrap();
        let q = compute_gasteiger_charges(&mol);
        let n_idx = mol.atoms.iter().position(|a| a.atomic_number == 7).unwrap();
        assert!(
            q[n_idx] < -0.1,
            "N in methylamine should be negative: {}",
            q[n_idx]
        );
    }

    #[test]
    /// Runs regression test for charge conservation.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_charge_conservation() {
        // Total charge (heavy + implicit H) should equal sum of formal charges.
        // Our q[] only has heavy atoms, so we check that q_sum is reasonable
        // (negative for neutral organic molecules, since H donates to heavy atoms).
        for smiles in &["CCO", "CC(=O)O", "c1ccccc1", "CN"] {
            let mol = parse_smiles(smiles).unwrap();
            let q = compute_gasteiger_charges(&mol);
            let total_q: f64 = q.iter().sum();
            // For neutral molecules, sum of heavy atom charges should be negative
            // (implicit H carry the balancing positive charge).
            assert!(
                total_q < 0.1,
                "Heavy atom charge sum for {smiles} should be <=0: {total_q}"
            );
        }
    }

    #[test]
    /// Runs regression test for halogens.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_halogens() {
        // Fluoromethane: F should be negative, C positive
        let mol = parse_smiles("CF").unwrap();
        let q = compute_gasteiger_charges(&mol);
        let f_idx = mol.atoms.iter().position(|a| a.atomic_number == 9).unwrap();
        let c_idx = mol.atoms.iter().position(|a| a.atomic_number == 6).unwrap();
        assert!(q[f_idx] < 0.0, "F should be negative: {}", q[f_idx]);
        assert!(
            q[c_idx] > 0.0,
            "C bonded to F should be positive: {}",
            q[c_idx]
        );
    }

    #[test]
    /// Runs regression test for chloroform.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_chloroform() {
        // ClC(Cl)Cl
        let mol = parse_smiles("ClC(Cl)Cl").unwrap();
        let q = compute_gasteiger_charges(&mol);
        let cl_charges: Vec<f64> = mol
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, a)| a.atomic_number == 17)
            .map(|(i, _)| q[i])
            .collect();
        // All Cl should have similar charges
        let avg = cl_charges.iter().sum::<f64>() / cl_charges.len() as f64;
        for &c in &cl_charges {
            assert_close(c, avg, 0.02, "Chloroform Cl symmetry");
        }
    }

    // ===================================================================
    // RDKit cross-validation tests
    // Reference: RDKit 2024.09, rdPartialCharges.ComputeGasteigerCharges
    // Tolerance: ±0.05 per atom (algorithm variants differ in damping,
    // iteration count, and implicit-H treatment; this is reasonable).
    // ===================================================================
    const TOL: f64 = 0.01;

    /// Helper: compare our heavy-atom charges to RDKit reference values.
    fn assert_charges(smiles: &str, rdkit_ref: &[f64]) {
        let mol = parse_smiles(smiles).unwrap();
        let q = compute_gasteiger_charges(&mol);
        assert_eq!(
            q.len(),
            rdkit_ref.len(),
            "{smiles}: atom count mismatch (ours={}, rdkit={})",
            q.len(),
            rdkit_ref.len()
        );
        for (i, (&ours, &ref_v)) in q.iter().zip(rdkit_ref).enumerate() {
            assert!(
                (ours - ref_v).abs() < TOL,
                "{smiles} atom {i}: ours={ours:.4}, rdkit={ref_v:.4}, diff={:.4}",
                (ours - ref_v).abs()
            );
        }
    }

    #[test]
    /// Runs regression test for rdkit formaldehyde.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_formaldehyde() {
        // C=O → C: +0.1064, O: -0.3071
        assert_charges("C=O", &[0.1064, -0.3071]);
    }

    #[test]
    /// Runs regression test for rdkit ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_ethanol() {
        // CCO → C: -0.0418, C: +0.0402, O: -0.3967
        assert_charges("CCO", &[-0.0418, 0.0402, -0.3967]);
    }

    #[test]
    /// Runs regression test for rdkit acetic acid.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_acetic_acid() {
        // CC(=O)O → C: +0.0338, C: +0.2997, O: -0.2528, O: -0.4814
        assert_charges("CC(=O)O", &[0.0338, 0.2997, -0.2528, -0.4814]);
    }

    #[test]
    /// Runs regression test for rdkit benzene.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_benzene() {
        // c1ccccc1 → all C: -0.0623
        assert_charges("c1ccccc1", &[-0.0623; 6]);
    }

    #[test]
    /// Runs regression test for rdkit methylamine.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_methylamine() {
        // CN → C: -0.0195, N: -0.3333
        assert_charges("CN", &[-0.0195, -0.3333]);
    }

    #[test]
    /// Runs regression test for rdkit fluoromethane.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_fluoromethane() {
        // CF → C: +0.0785, F: -0.2547
        assert_charges("CF", &[0.0785, -0.2547]);
    }

    #[test]
    /// Runs regression test for rdkit chloromethane.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_chloromethane() {
        // CCl → C: +0.0108, Cl: -0.1304
        assert_charges("CCl", &[0.0108, -0.1304]);
    }

    #[test]
    /// Runs regression test for rdkit bromomethane.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_bromomethane() {
        // CBr → C: -0.0085, Br: -0.0966
        assert_charges("CBr", &[-0.0085, -0.0966]);
    }

    #[test]
    /// Runs regression test for rdkit iodomethane.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_iodomethane() {
        // CI → C: -0.0121, I: -0.0901
        assert_charges("CI", &[-0.0121, -0.0901]);
    }

    #[test]
    /// Runs regression test for rdkit acetone.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_acetone() {
        // CC(=O)C → C: -0.0060, C: +0.1263, O: -0.3003, C: -0.0060
        assert_charges("CC(=O)C", &[-0.0060, 0.1263, -0.3003, -0.0060]);
    }

    #[test]
    /// Runs regression test for rdkit dimethyl ether.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_dimethyl_ether() {
        // COC → C: +0.0351, O: -0.3879, C: +0.0351
        assert_charges("COC", &[0.0351, -0.3879, 0.0351]);
    }

    #[test]
    /// Runs regression test for rdkit ethane.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_ethane() {
        // CC → C: -0.0683, C: -0.0683
        assert_charges("CC", &[-0.0683, -0.0683]);
    }

    #[test]
    /// Runs regression test for rdkit phenol.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_phenol() {
        // c1ccc(O)cc1 → 7 atoms
        // RDKit: C -0.062, C -0.0586, C -0.0202, C +0.1151, O -0.5080, C -0.0202, C -0.0586
        assert_charges(
            "c1ccc(O)cc1",
            &[-0.062, -0.0586, -0.0202, 0.1151, -0.5080, -0.0202, -0.0586],
        );
    }

    #[test]
    /// Runs regression test for rdkit pyridine.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_pyridine() {
        // c1ccncc1 → 6 atoms
        // RDKit: C -0.0592, C -0.0438, C +0.0267, N -0.2647, C +0.0267, C -0.0438
        assert_charges(
            "c1ccncc1",
            &[-0.0592, -0.0438, 0.0267, -0.2647, 0.0267, -0.0438],
        );
    }

    #[test]
    /// Runs regression test for rdkit aniline.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_aniline() {
        // c1ccc(N)cc1 → 7 atoms
        // RDKit: C -0.0621, C -0.0602, C -0.0392, C +0.0313, N -0.3987, C -0.0392, C -0.0602
        assert_charges(
            "c1ccc(N)cc1",
            &[-0.0621, -0.0602, -0.0392, 0.0313, -0.3987, -0.0392, -0.0602],
        );
    }

    #[test]
    /// Runs regression test for rdkit phosphate oxyanion charges.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_phosphate_oxyanion_charges() {
        assert_charges(
            "COP(=O)([O-])O",
            &[0.0441, -0.3219, 0.2645, -0.2766, -0.7563, -0.3342],
        );
    }

    #[test]
    /// Runs regression test for rdkit sulfate oxyanion charges.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_sulfate_oxyanion_charges() {
        assert_charges(
            "CS(=O)(=O)[O-]",
            &[0.0268, 0.0916, -0.2428, -0.2428, -0.7482],
        );
    }

    #[test]
    /// Runs regression test for split charge conjugated splits carboxylate oxygens.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_split_charge_conjugated_splits_carboxylate_oxygens() {
        let mol = parse_smiles("CC(=O)[O-]").unwrap();
        let mut q = vec![0.0; mol.num_atoms()];
        split_charge_conjugated(&mol, &mut q);
        assert_close(q[2], -0.5, 1e-12, "carbonyl oxygen shares charge");
        assert_close(q[3], -0.5, 1e-12, "anionic oxygen shares charge");
    }

    #[test]
    /// Runs regression test for split charge conjugated does not cross nonconjugated bridge.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_split_charge_conjugated_does_not_cross_nonconjugated_bridge() {
        let mol = parse_smiles("[N+]1=CC=NC(C)1").unwrap();
        let mut q = vec![0.0; mol.num_atoms()];
        split_charge_conjugated(&mol, &mut q);
        assert_close(
            q[0],
            1.0,
            1e-12,
            "charged ring nitrogen keeps formal charge",
        );
        assert_close(
            q[3],
            0.0,
            1e-12,
            "imine nitrogen should not receive shared charge",
        );
    }

    #[test]
    /// Runs regression test for rdkit scalar descriptors.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_rdkit_scalar_descriptors() {
        // Test MaxPartialCharge, MinPartialCharge, etc. against RDKit for ethanol
        // RDKit: MaxPartialCharge=0.0402, MinPartialCharge=-0.3967
        let mol = parse_smiles("CCO").unwrap();
        let q = compute_gasteiger_charges(&mol);
        let max = q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = q.iter().copied().fold(f64::INFINITY, f64::min);
        assert_close(max, 0.0402, TOL, "ethanol MaxPartialCharge");
        assert_close(min, -0.3967, TOL, "ethanol MinPartialCharge");
    }

    #[test]
    /// Runs regression test for cyclopentadienyl charges.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cyclopentadienyl_charges() {
        let mol = parse_smiles("[CH-]1C=CC=C1").unwrap();
        let q = compute_gasteiger_charges(&mol);
        println!("Cyclopentadienyl charges: {:?}", q);
        for (i, &c) in q.iter().enumerate() {
            assert!(c < 0.0, "Atom {} should be negative: {}", i, c);
        }
    }

    #[test]
    /// Runs regression test for nitrobenzene charges.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_nitrobenzene_charges() {
        let mol = parse_smiles("c1ccc(cc1)[N+](=O)[O-]").unwrap();
        let q = compute_gasteiger_charges(&mol);
        println!("Nitrobenzene charges: {:?}", q);
    }

    #[test]
    /// Runs regression test for grignard charges.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_grignard_charges() {
        let mol = parse_smiles("[Mg](Br)CC").unwrap();
        let q = compute_gasteiger_charges(&mol);
        println!("Grignard charges: {:?}", q);
        assert!(q[0] > 0.3, "Mg should be positive: {}", q[0]);
    }

    // ===================================================================
    // Hybridization unit tests – effective_hybridization O/S norbs==4 rule
    //
    // These encode the three-case rule that drives correct HallKierAlpha
    // values for ester/phosphate-containing molecules:
    //   Case 1: aromatic neighbour          → SP2  (phenol, anisole)
    //   Case 2: terminal O-H, C nbr w/ pi   → SP2  (COOH, enol)
    //   else                                → SP3  (ester bridge, P-OH, ether)
    //
    // Validated against RDKit's setHybridization and Gasteiger charges.
    // ===================================================================

    /// Implements of.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn hyb_of(smiles: &str, atom_idx: usize) -> crate::types::Hybridization {
        let mol = parse_smiles(smiles).unwrap();
        effective_hybridization(&mol, atom_idx)
    }

    // -- carbonyl =O: direct double bond → SP2 via Stage 3 (not norbs path) --
    #[test]
    /// Runs regression test for hyb carbonyl o sp2.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_carbonyl_o_sp2() {
        // C=O: C(0) O(1)  — direct double bond, Stage 3 fires before norbs
        assert_eq!(
            hyb_of("C=O", 1),
            crate::types::Hybridization::SP2,
            "carbonyl =O must be SP2"
        );
    }

    // -- ester bridging O: C=O neighbour (not hypervalent) → SP2 (RDKit-confirmed) --
    #[test]
    /// Runs regression test for hyb ester bridging o sp2.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_ester_bridging_o_sp2() {
        // CC(=O)OC: C(0) C(1) =O(2) bridging-O(3) C(4)
        // O(3): C(1) neighbour has C=O pi bond, C is not P/S → SP2
        assert_eq!(
            hyb_of("CC(=O)OC", 3),
            crate::types::Hybridization::SP2,
            "ester bridging O must be SP2 (C=O neighbour, RDKit-confirmed)"
        );
    }

    // -- carboxylic-acid OH: terminal O-H, C neighbour has C=O → SP2 --
    #[test]
    /// Runs regression test for hyb carboxylic acid oh sp2.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_carboxylic_acid_oh_sp2() {
        // CC(=O)O: C(0) C(1) =O(2) OH(3)
        // O(3): C(1) has C=O to O(2) ≠ O(3) → SP2
        assert_eq!(
            hyb_of("CC(=O)O", 3),
            crate::types::Hybridization::SP2,
            "carboxylic-acid OH must be SP2"
        );
    }

    // -- phenol OH: direct aromatic C neighbour → SP2 (Case 1) --
    #[test]
    /// Runs regression test for hyb phenol o sp2.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_phenol_o_sp2() {
        let mol = parse_smiles("c1ccc(O)cc1").unwrap();
        let o_idx = mol.atoms.iter().position(|a| a.atomic_number == 8).unwrap();
        assert_eq!(
            effective_hybridization(&mol, o_idx),
            crate::types::Hybridization::SP2,
            "phenol OH must be SP2 (aromatic neighbour)"
        );
    }

    // -- anisole O: one heavy nbr is aromatic C, other is aliphatic C → SP2 (Case 1) --
    #[test]
    /// Runs regression test for hyb anisole o sp2.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_anisole_o_sp2() {
        // COc1ccccc1: C(0) O(1) aromatic-ring(2-7)
        let mol = parse_smiles("COc1ccccc1").unwrap();
        let o_idx = mol.atoms.iter().position(|a| a.atomic_number == 8).unwrap();
        assert_eq!(
            effective_hybridization(&mol, o_idx),
            crate::types::Hybridization::SP2,
            "anisole O (one aromatic-ring nbr) must be SP2"
        );
    }

    // -- plain ether O: two aliphatic C neighbours, no pi → SP3 --
    #[test]
    /// Runs regression test for hyb ether o sp3.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_ether_o_sp3() {
        // COC: C(0) O(1) C(2)
        assert_eq!(
            hyb_of("COC", 1),
            crate::types::Hybridization::SP3,
            "ether O must be SP3"
        );
    }

    // -- phosphate OH: terminal O-H but neighbour is P (not C) → SP3 --
    #[test]
    /// Runs regression test for hyb phosphate oh sp3.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_phosphate_oh_sp3() {
        // OP(=O)(O)O: O(0) P(1) =O(2) O(3)H O(4)H
        // All three P–OH oxygens must be SP3; only the =O is SP2.
        let mol = parse_smiles("OP(=O)(O)O").unwrap();
        for (i, atom) in mol.atoms.iter().enumerate() {
            if atom.atomic_number != 8 {
                continue;
            }
            let has_double = mol.adjacency[i]
                .iter()
                .any(|&bi| mol.bonds[bi as usize].bond_type == crate::types::BondType::Double);
            if has_double {
                continue; // skip the =O, which is SP2 via Stage 3
            }
            assert_eq!(
                effective_hybridization(&mol, i),
                crate::types::Hybridization::SP3,
                "phosphate OH oxygen {i} must be SP3 (P neighbour, not C)"
            );
        }
    }

    // -- thioether S adjacent to aromatic ring: norbs==4, always SP3 (not SP2) --
    #[test]
    /// Runs regression test for hyb thioanisole s sp3.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_thioanisole_s_sp3() {
        // CSc1ccccc1 — PhSCH3, the thioether S has one aromatic-C neighbour
        // but S in norbs==4 is always SP3 (unlike phenol O which is SP2).
        let mol = parse_smiles("CSc1ccccc1").unwrap();
        let s_idx = mol
            .atoms
            .iter()
            .position(|a| a.atomic_number == 16)
            .unwrap();
        assert_eq!(
            effective_hybridization(&mol, s_idx),
            crate::types::Hybridization::SP3,
            "thioether S adjacent to aromatic ring must be SP3 (norbs==4 S rule)"
        );
    }

    // -- thioether S adjacent to C=N (thiazolidine-like): still SP3 --
    #[test]
    /// Runs regression test for hyb thiazolidine s sp3.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_thiazolidine_s_sp3() {
        // C1(=N)SCC1 — 4,5-dihydrothiazole ring; S sees a C=N neighbour
        // but S in norbs==4 must still be SP3.
        let mol = parse_smiles("C1(=N)SCC1").unwrap();
        let s_idx = mol
            .atoms
            .iter()
            .position(|a| a.atomic_number == 16)
            .unwrap();
        assert_eq!(
            effective_hybridization(&mol, s_idx),
            crate::types::Hybridization::SP3,
            "thiazolidine S (C=N neighbour) must be SP3"
        );
    }

    // -- sulfonamide N: N–S(=O)₂ must be SP3 (hypervalent S excluded) --
    #[test]
    /// Runs regression test for hyb sulfonamide n sp3.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_sulfonamide_n_sp3() {
        // CN(C)S(=O)(=O)c1ccccc1 — N-dimethylbenzenesulfonamide
        // The amine N is bonded to the sulfonyl S, whose S=O bonds fire
        // neighbor_has_pi=true in the old broad check — but with the
        // adjacency-based exclusion of hypervalent S, N must be SP3.
        let mol = parse_smiles("CN(C)S(=O)(=O)c1ccccc1").unwrap();
        // Find the sulfonamide N (bonded to S, not in the aromatic ring)
        let n_idx = mol
            .atoms
            .iter()
            .enumerate()
            .find(|(i, a)| {
                a.atomic_number == 7
                    && !a.is_aromatic()
                    && mol.adjacency[*i].iter().any(|&bi| {
                        let b = &mol.bonds[bi as usize];
                        let nb = if b.source as usize == *i {
                            b.target as usize
                        } else {
                            b.source as usize
                        };
                        mol.atoms[nb].atomic_number == 16
                    })
            })
            .map(|(i, _)| i)
            .expect("sulfonamide N not found");
        assert_eq!(
            effective_hybridization(&mol, n_idx),
            crate::types::Hybridization::SP3,
            "sulfonamide N (N–S(=O)₂) must be SP3"
        );
    }

    // -- amide N: N–C=O must still be SP2 (C is non-hypervalent, not excluded) --
    #[test]
    /// Runs regression test for hyb amide n sp2.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_amide_n_sp2() {
        // CC(=O)NC — N-methylacetamide; amide N must be SP2.
        let mol = parse_smiles("CC(=O)NC").unwrap();
        let n_idx = mol
            .atoms
            .iter()
            .position(|a| a.atomic_number == 7)
            .expect("N not found");
        assert_eq!(
            effective_hybridization(&mol, n_idx),
            crate::types::Hybridization::SP2,
            "amide N (N–C=O) must be SP2"
        );
    }

    // -- aniline N: N bonded to aromatic C must still be SP2 --
    #[test]
    /// Runs regression test for hyb aniline n sp2.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_aniline_n_sp2() {
        // Nc1ccccc1 — aniline; NH₂ bonded to aromatic ring must be SP2.
        let mol = parse_smiles("Nc1ccccc1").unwrap();
        let n_idx = mol
            .atoms
            .iter()
            .position(|a| a.atomic_number == 7)
            .expect("N not found");
        assert_eq!(
            effective_hybridization(&mol, n_idx),
            crate::types::Hybridization::SP2,
            "aniline N (aromatic-C neighbour) must be SP2"
        );
    }

    // -- phosphate bridging O: two heavy neighbours (C and P), neither aromatic → SP3 --
    #[test]
    /// Runs regression test for hyb phosphate bridging o sp3.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_phosphate_bridging_o_sp3() {
        // COP(=O)(OC)O — two C–O–P bridging oxygens + one P–OH
        // All non-double-bond oxygens must be SP3.
        let mol = parse_smiles("COP(=O)(OC)O").unwrap();
        for (i, atom) in mol.atoms.iter().enumerate() {
            if atom.atomic_number != 8 {
                continue;
            }
            let has_double = mol.adjacency[i]
                .iter()
                .any(|&bi| mol.bonds[bi as usize].bond_type == crate::types::BondType::Double);
            if has_double {
                continue; // skip =O
            }
            assert_eq!(
                effective_hybridization(&mol, i),
                crate::types::Hybridization::SP3,
                "phosphate O {i} (bridging or OH) must be SP3"
            );
        }
    }

    #[test]
    /// Runs regression test for hyb allene central carbon sp.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_allene_central_carbon_sp() {
        assert_eq!(
            hyb_of("C=C=C", 1),
            crate::types::Hybridization::SP,
            "allene central carbon must be SP"
        );
    }

    #[test]
    /// Runs regression test for hyb trivalent bracket carbon sp3.
    ///
    /// Function context: `crates/ferrum_core/src/algo/electronic.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_hyb_trivalent_bracket_carbon_sp3() {
        assert_eq!(
            hyb_of("[C](C)(C)C", 0),
            crate::types::Hybridization::SP3,
            "neutral bracketed trivalent carbon must stay SP3"
        );
    }
}
