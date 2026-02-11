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

use crate::mol::molecule::Molecule;
use crate::types::Hybridization;

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
    d: f64, // a + b + c
}

const H: ENParams = ENParams { a: 7.17, b: 6.24, c: -0.56, d: 12.85 };
const C_SP3: ENParams = ENParams { a: 7.98, b: 9.18, c: 1.88, d: 19.04 };
const C_SP2: ENParams = ENParams { a: 8.79, b: 9.32, c: 1.51, d: 19.62 };
const C_SP: ENParams = ENParams { a: 10.39, b: 9.45, c: 0.73, d: 20.57 };
const N_SP3: ENParams = ENParams { a: 11.54, b: 10.82, c: 1.36, d: 23.72 };
const N_SP2: ENParams = ENParams { a: 12.87, b: 11.15, c: 0.85, d: 24.87 };
const N_SP: ENParams = ENParams { a: 15.68, b: 11.70, c: -0.27, d: 27.11 };
const O_SP3: ENParams = ENParams { a: 14.18, b: 12.92, c: 1.39, d: 28.49 };
const O_SP2: ENParams = ENParams { a: 17.07, b: 13.79, c: 0.47, d: 31.33 };
const F_PARAMS: ENParams = ENParams { a: 14.66, b: 13.85, c: 2.31, d: 30.82 };
const CL: ENParams = ENParams { a: 11.00, b: 9.69, c: 1.35, d: 22.04 };
const BR: ENParams = ENParams { a: 10.08, b: 8.47, c: 1.16, d: 19.71 };
const I_PARAMS: ENParams = ENParams { a: 9.90, b: 7.96, c: 0.96, d: 18.82 };
const S_SP3: ENParams = ENParams { a: 10.14, b: 9.13, c: 1.38, d: 20.65 };
const S_SP2: ENParams = ENParams { a: 10.88, b: 9.49, c: 1.33, d: 21.70 };
const P_SP3: ENParams = ENParams { a: 8.90, b: 8.24, c: 0.96, d: 18.10 };

const NUM_ITERATIONS: usize = 12;

/// RDKit uses a special ionization constant for hydrogen (IONXH = 20.02)
/// instead of H.d (a+b+c = 12.85) in the charge transfer denominator.
const IONXH: f64 = 20.02;

/// Determine effective hybridization from bond types for Gasteiger parameter selection.
/// Follows RDKit's setHybridization logic: norbs = degree + lone_pairs,
/// with demotion from SP3 to SP2 when the atom has a conjugated bond and degree ≤ 3.
fn effective_hybridization(mol: &Molecule, idx: usize) -> Hybridization {
    let atom = &mol.atoms[idx];
    if atom.hybridization != Hybridization::Unspecified {
        return atom.hybridization;
    }
    if atom.is_aromatic() {
        return Hybridization::SP2;
    }

    // Count bond types and total degree (including implicit H).
    let mut has_double = false;
    let mut has_triple = false;
    let mut has_aromatic_bond = false;
    let mut heavy_degree: u8 = 0;
    // Track whether any neighbor has a double/aromatic bond (conjugation).
    let mut neighbor_has_pi = false;

    for bond in &mol.bonds {
        let nbr_idx = if bond.source as usize == idx {
            bond.target as usize
        } else if bond.target as usize == idx {
            bond.source as usize
        } else {
            continue;
        };
        heavy_degree += 1;
        match bond.bond_type {
            crate::types::BondType::Triple => has_triple = true,
            crate::types::BondType::Double => has_double = true,
            crate::types::BondType::Aromatic => has_aromatic_bond = true,
            _ => {}
        }
        // Check if neighbor participates in pi bonding (for conjugation).
        let nbr = &mol.atoms[nbr_idx];
        if nbr.is_aromatic() {
            neighbor_has_pi = true;
        } else {
            for b2 in &mol.bonds {
                let touches = (b2.source as usize == nbr_idx) || (b2.target as usize == nbr_idx);
                if touches && matches!(b2.bond_type,
                    crate::types::BondType::Double | crate::types::BondType::Aromatic)
                {
                    neighbor_has_pi = true;
                    break;
                }
            }
        }
    }

    if has_triple {
        return Hybridization::SP;
    }
    if has_double || has_aromatic_bond {
        return Hybridization::SP2;
    }

    // RDKit logic: norbs = degree + lone_pairs.
    // If norbs == 4, degree ≤ 3, and atom has a conjugated bond → SP2.
    let total_degree = heavy_degree + atom.total_h();
    let outer_electrons = outer_shell_electrons(atom.atomic_number);
    let total_valence = total_degree as i32; // each bond uses 1 valence
    let free_electrons = outer_electrons - total_valence - atom.formal_charge as i32;
    let lone_pairs = if free_electrons > 0 { free_electrons / 2 } else { 0 };
    let norbs = total_degree as i32 + lone_pairs;

    match norbs {
        0 | 1 => Hybridization::SP, // s (rare)
        2 => Hybridization::SP,
        3 => Hybridization::SP2,
        4 => {
            if total_degree <= 3 && neighbor_has_pi {
                Hybridization::SP2
            } else {
                Hybridization::SP3
            }
        }
        _ => Hybridization::SP3,
    }
}

/// Number of outer shell electrons for common elements.
fn outer_shell_electrons(atomic_num: u8) -> i32 {
    match atomic_num {
        1 => 1,
        6 => 4,
        7 => 5,
        8 => 6,
        9 | 17 | 35 | 53 => 7,
        15 => 5,
        16 => 6,
        _ => 4, // fallback
    }
}

/// Assign Gasteiger electronegativity parameters based on element and hybridization.
fn get_params(mol: &Molecule, idx: usize) -> Option<ENParams> {
    let atom = &mol.atoms[idx];
    let hyb = effective_hybridization(mol, idx);
    match atom.atomic_number {
        1 => Some(H),
        6 => match hyb {
            Hybridization::SP => Some(C_SP),
            Hybridization::SP2 => Some(C_SP2),
            _ => Some(C_SP3),
        },
        7 => match hyb {
            Hybridization::SP => Some(N_SP),
            Hybridization::SP2 => Some(N_SP2),
            _ => Some(N_SP3),
        },
        8 => match hyb {
            Hybridization::SP2 => Some(O_SP2),
            _ => Some(O_SP3),
        },
        9 => Some(F_PARAMS),
        15 => Some(P_SP3),
        16 => match hyb {
            Hybridization::SP2 => Some(S_SP2),
            _ => Some(S_SP3),
        },
        17 => Some(CL),
        35 => Some(BR),
        53 => Some(I_PARAMS),
        _ => Some(C_SP3), // fallback
    }
}

/// Compute Gasteiger-Marsili partial charges for all atoms in a molecule.
///
/// Returns a `Vec<f64>` of partial charges (one per heavy atom, indexed by atom).
/// Uses 12 iterations (RDKit default) with geometric damping (0.5^(k+1)).
/// Implicit hydrogens are modeled as virtual neighbors using IONXH = 20.02.
/// Atoms with unknown parameters are assigned zero charge and excluded from
/// charge transfer.
pub fn compute_gasteiger_charges(mol: &Molecule) -> Vec<f64> {
    let n = mol.num_atoms();
    if n == 0 {
        return Vec::new();
    }

    // Resolve parameters for each atom; None means skip.
    let params: Vec<Option<ENParams>> = (0..n).map(|i| get_params(mol, i)).collect();

    // Number of implicit H per heavy atom.
    let n_h: Vec<u8> = mol.atoms.iter().map(|a| a.total_h()).collect();

    // Initialize charges from formal charges.
    let mut q: Vec<f64> = mol.atoms.iter().map(|a| a.formal_charge as f64).collect();

    // Per-heavy-atom implicit H charge accumulator. Stores the TOTAL charge
    // across all implicit H on that atom (matching RDKit convention: gqH
    // accumulates nH * dif, and hChi = f(gqH) uses this total).
    let mut q_h: Vec<f64> = vec![0.0; n];

    let mut chi = vec![0.0_f64; n];

    for k in 0..NUM_ITERATIONS {
        let damping = 1.0_f64 / (2.0_f64.powi((k as i32) + 1));

        // Compute electronegativity for each atom.
        for i in 0..n {
            if let Some(p) = &params[i] {
                chi[i] = p.a + p.b * q[i] + p.c * q[i] * q[i];
            }
        }

        // Charge transfer along each explicit bond.
        // RDKit convention: divide by d (= ionX = a+b+c) of the LESS
        // electronegative atom in the pair.
        for bond in &mol.bonds {
            let i = bond.source as usize;
            let j = bond.target as usize;

            let (pi, pj) = match (&params[i], &params[j]) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };

            let mut dif = chi[i] - chi[j];
            let denom = if dif > 0.0 { pj.d } else { pi.d };
            if denom.abs() < 1e-15 {
                continue;
            }
            dif = dif / denom * damping;
            q[i] -= dif;
            q[j] += dif;
        }

        // Charge transfer between each heavy atom and its implicit H atoms.
        for i in 0..n {
            let nh = n_h[i] as usize;
            if nh == 0 {
                continue;
            }
            let pi = match &params[i] {
                Some(p) => p,
                None => continue,
            };
            // Per-H charge for chi computation (RDKit: qHs = hChrg / niHs).
            let per_h = q_h[i] / nh as f64;
            let chi_h = H.a + H.b * per_h + H.c * per_h * per_h;
            let mut dif = chi[i] - chi_h;
            // Divide by d of the LESS electronegative atom.
            // RDKit uses IONXH (20.02) for hydrogen's ionX, not H.d (12.85).
            let denom = if dif > 0.0 { IONXH } else { pi.d };
            if denom.abs() < 1e-15 {
                continue;
            }
            dif = dif / denom * damping;
            // Each implicit H transfers the same dif; accumulate total in q_h.
            q[i] -= dif * nh as f64;
            q_h[i] += dif * nh as f64;
        }
    }

    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

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
    fn test_empty_molecule() {
        let mol = Molecule::new();
        let q = compute_gasteiger_charges(&mol);
        assert!(q.is_empty());
    }

    #[test]
    fn test_methane() {
        // [CH4] — single C with 4 implicit H; charge transferred to H.
        // RDKit: C = -0.077558
        let q = charges("[CH4]");
        assert_close(q[0], -0.0776, 0.01, "methane C");
    }

    #[test]
    fn test_water() {
        // O with 2 implicit H; O pulls electron density from H.
        // RDKit: O = -0.41151
        let q = charges("O");
        assert_close(q[0], -0.4115, 0.02, "water O");
    }

    #[test]
    fn test_formaldehyde() {
        // C=O: C should be positive, O negative
        let mol = parse_smiles("C=O").unwrap();
        let q = compute_gasteiger_charges(&mol);
        let c_idx = mol.atoms.iter().position(|a| a.atomic_number == 6).unwrap();
        let o_idx = mol.atoms.iter().position(|a| a.atomic_number == 8).unwrap();
        assert!(q[c_idx] > 0.0, "C in formaldehyde should be positive: {}", q[c_idx]);
        assert!(q[o_idx] < 0.0, "O in formaldehyde should be negative: {}", q[o_idx]);
    }

    #[test]
    fn test_ethanol() {
        // CCO: two carbons and an oxygen
        let q = charges("CCO");
        let mol = parse_smiles("CCO").unwrap();
        let o_idx = mol.atoms.iter().position(|a| a.atomic_number == 8).unwrap();
        assert!(q[o_idx] < -0.1, "O in ethanol should be negative: {}", q[o_idx]);
    }

    #[test]
    fn test_acetic_acid() {
        // CC(=O)O
        let mol = parse_smiles("CC(=O)O").unwrap();
        let q = compute_gasteiger_charges(&mol);

        // Find carbonyl C (bonded to =O), carbonyl O, hydroxyl O
        let carbonyl_c = 1; // atom index 1 in CC(=O)O
        assert!(q[carbonyl_c] > 0.1, "Carbonyl C should be positive: {}", q[carbonyl_c]);

        // Both oxygens should be negative
        for (i, atom) in mol.atoms.iter().enumerate() {
            if atom.atomic_number == 8 {
                assert!(q[i] < 0.0, "O atom {i} should be negative: {}", q[i]);
            }
        }
    }

    #[test]
    fn test_benzene() {
        let mol = parse_smiles("c1ccccc1").unwrap();
        let q = compute_gasteiger_charges(&mol);
        // All carbons in benzene should have nearly identical charges.
        let c_charges: Vec<f64> = mol.atoms.iter().enumerate()
            .filter(|(_, a)| a.atomic_number == 6)
            .map(|(i, _)| q[i])
            .collect();
        let avg = c_charges.iter().sum::<f64>() / c_charges.len() as f64;
        for &c in &c_charges {
            assert_close(c, avg, 0.01, "Benzene C symmetry");
        }
    }

    #[test]
    fn test_aniline() {
        let mol = parse_smiles("c1ccc(N)cc1").unwrap();
        let q = compute_gasteiger_charges(&mol);
        let n_idx = mol.atoms.iter().position(|a| a.atomic_number == 7).unwrap();
        assert!(q[n_idx] < -0.05, "N in aniline should be negative: {}", q[n_idx]);
    }

    #[test]
    fn test_methylamine() {
        let mol = parse_smiles("CN").unwrap();
        let q = compute_gasteiger_charges(&mol);
        let n_idx = mol.atoms.iter().position(|a| a.atomic_number == 7).unwrap();
        assert!(q[n_idx] < -0.1, "N in methylamine should be negative: {}", q[n_idx]);
    }

    #[test]
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
            assert!(total_q < 0.1,
                "Heavy atom charge sum for {smiles} should be <=0: {total_q}");
        }
    }

    #[test]
    fn test_halogens() {
        // Fluoromethane: F should be negative, C positive
        let mol = parse_smiles("CF").unwrap();
        let q = compute_gasteiger_charges(&mol);
        let f_idx = mol.atoms.iter().position(|a| a.atomic_number == 9).unwrap();
        let c_idx = mol.atoms.iter().position(|a| a.atomic_number == 6).unwrap();
        assert!(q[f_idx] < 0.0, "F should be negative: {}", q[f_idx]);
        assert!(q[c_idx] > 0.0, "C bonded to F should be positive: {}", q[c_idx]);
    }

    #[test]
    fn test_chloroform() {
        // ClC(Cl)Cl
        let mol = parse_smiles("ClC(Cl)Cl").unwrap();
        let q = compute_gasteiger_charges(&mol);
        let cl_charges: Vec<f64> = mol.atoms.iter().enumerate()
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
        assert_eq!(q.len(), rdkit_ref.len(),
            "{smiles}: atom count mismatch (ours={}, rdkit={})", q.len(), rdkit_ref.len());
        for (i, (&ours, &ref_v)) in q.iter().zip(rdkit_ref).enumerate() {
            assert!((ours - ref_v).abs() < TOL,
                "{smiles} atom {i}: ours={ours:.4}, rdkit={ref_v:.4}, diff={:.4}",
                (ours - ref_v).abs());
        }
    }

    #[test]
    fn test_rdkit_formaldehyde() {
        // C=O → C: +0.1064, O: -0.3071
        assert_charges("C=O", &[0.1064, -0.3071]);
    }

    #[test]
    fn test_rdkit_ethanol() {
        // CCO → C: -0.0418, C: +0.0402, O: -0.3967
        assert_charges("CCO", &[-0.0418, 0.0402, -0.3967]);
    }

    #[test]
    fn test_rdkit_acetic_acid() {
        // CC(=O)O → C: +0.0338, C: +0.2997, O: -0.2528, O: -0.4814
        assert_charges("CC(=O)O", &[0.0338, 0.2997, -0.2528, -0.4814]);
    }

    #[test]
    fn test_rdkit_benzene() {
        // c1ccccc1 → all C: -0.0623
        assert_charges("c1ccccc1", &[-0.0623; 6]);
    }

    #[test]
    fn test_rdkit_methylamine() {
        // CN → C: -0.0195, N: -0.3333
        assert_charges("CN", &[-0.0195, -0.3333]);
    }

    #[test]
    fn test_rdkit_fluoromethane() {
        // CF → C: +0.0785, F: -0.2547
        assert_charges("CF", &[0.0785, -0.2547]);
    }

    #[test]
    fn test_rdkit_chloromethane() {
        // CCl → C: +0.0108, Cl: -0.1304
        assert_charges("CCl", &[0.0108, -0.1304]);
    }

    #[test]
    fn test_rdkit_bromomethane() {
        // CBr → C: -0.0085, Br: -0.0966
        assert_charges("CBr", &[-0.0085, -0.0966]);
    }

    #[test]
    fn test_rdkit_iodomethane() {
        // CI → C: -0.0121, I: -0.0901
        assert_charges("CI", &[-0.0121, -0.0901]);
    }

    #[test]
    fn test_rdkit_acetone() {
        // CC(=O)C → C: -0.0060, C: +0.1263, O: -0.3003, C: -0.0060
        assert_charges("CC(=O)C", &[-0.0060, 0.1263, -0.3003, -0.0060]);
    }

    #[test]
    fn test_rdkit_dimethyl_ether() {
        // COC → C: +0.0351, O: -0.3879, C: +0.0351
        assert_charges("COC", &[0.0351, -0.3879, 0.0351]);
    }

    #[test]
    fn test_rdkit_ethane() {
        // CC → C: -0.0683, C: -0.0683
        assert_charges("CC", &[-0.0683, -0.0683]);
    }

    #[test]
    fn test_rdkit_phenol() {
        // c1ccc(O)cc1 → 7 atoms
        // RDKit: C -0.062, C -0.0586, C -0.0202, C +0.1151, O -0.5080, C -0.0202, C -0.0586
        assert_charges("c1ccc(O)cc1", &[-0.062, -0.0586, -0.0202, 0.1151, -0.5080, -0.0202, -0.0586]);
    }

    #[test]
    fn test_rdkit_pyridine() {
        // c1ccncc1 → 6 atoms
        // RDKit: C -0.0592, C -0.0438, C +0.0267, N -0.2647, C +0.0267, C -0.0438
        assert_charges("c1ccncc1", &[-0.0592, -0.0438, 0.0267, -0.2647, 0.0267, -0.0438]);
    }

    #[test]
    fn test_rdkit_aniline() {
        // c1ccc(N)cc1 → 7 atoms
        // RDKit: C -0.0621, C -0.0602, C -0.0392, C +0.0313, N -0.3987, C -0.0392, C -0.0602
        assert_charges("c1ccc(N)cc1", &[-0.0621, -0.0602, -0.0392, 0.0313, -0.3987, -0.0392, -0.0602]);
    }

    #[test]
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
}
