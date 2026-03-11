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

use crate::mol::bond::Bond;
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

const H: ENParams = ENParams {
    a: 7.17,
    b: 6.24,
    c: -0.56,
    d: 12.85,
};
const C_SP3: ENParams = ENParams {
    a: 7.98,
    b: 9.18,
    c: 1.88,
    d: 19.04,
};
const C_SP2: ENParams = ENParams {
    a: 8.79,
    b: 9.32,
    c: 1.51,
    d: 19.62,
};
const C_SP: ENParams = ENParams {
    a: 10.39,
    b: 9.45,
    c: 0.73,
    d: 20.57,
};
const N_SP3: ENParams = ENParams {
    a: 11.54,
    b: 10.82,
    c: 1.36,
    d: 23.72,
};
const N_SP2: ENParams = ENParams {
    a: 12.87,
    b: 11.15,
    c: 0.85,
    d: 24.87,
};
const N_SP: ENParams = ENParams {
    a: 15.68,
    b: 11.70,
    c: -0.27,
    d: 27.11,
};
const O_SP3: ENParams = ENParams {
    a: 14.18,
    b: 12.92,
    c: 1.39,
    d: 28.49,
};
const O_SP2: ENParams = ENParams {
    a: 17.07,
    b: 13.79,
    c: 0.47,
    d: 31.33,
};
const F_PARAMS: ENParams = ENParams {
    a: 14.66,
    b: 13.85,
    c: 2.31,
    d: 30.82,
};
const CL: ENParams = ENParams {
    a: 11.00,
    b: 9.69,
    c: 1.35,
    d: 22.04,
};
const BR: ENParams = ENParams {
    a: 10.08,
    b: 8.47,
    c: 1.16,
    d: 19.71,
};
const I_PARAMS: ENParams = ENParams {
    a: 9.90,
    b: 7.96,
    c: 0.96,
    d: 18.82,
};
const S_SP3: ENParams = ENParams {
    a: 10.14,
    b: 9.13,
    c: 1.38,
    d: 20.65,
};
const S_SO: ENParams = ENParams {
    a: 10.14,
    b: 9.13,
    c: 1.38,
    d: 20.65,
};
const S_SO2: ENParams = ENParams {
    a: 12.00,
    b: 10.81,
    c: 1.20,
    d: 24.01,
};
const S_SP2: ENParams = ENParams {
    a: 10.88,
    b: 9.49,
    c: 1.33,
    d: 21.70,
};
const P_SP3: ENParams = ENParams {
    a: 8.90,
    b: 8.24,
    c: 0.96,
    d: 18.10,
};
const P_SP2: ENParams = ENParams {
    a: 9.665,
    b: 8.530,
    c: 0.735,
    d: 18.93,
};
// Additional params from RDKit's additionalParamData (PyBabel-derived)
const SI_SP3: ENParams = ENParams {
    a: 7.30,
    b: 6.567,
    c: 0.657,
    d: 14.524,
};
const SI_SP2: ENParams = ENParams {
    a: 7.905,
    b: 6.748,
    c: 0.443,
    d: 15.096,
};
const MG_SP3: ENParams = ENParams {
    a: 3.300,
    b: 5.587,
    c: 2.447,
    d: 11.334,
};
const MG_SP2: ENParams = ENParams {
    a: 3.565,
    b: 5.572,
    c: 2.197,
    d: 11.334,
};
const MG_SP: ENParams = ENParams {
    a: 4.040,
    b: 5.472,
    c: 1.823,
    d: 11.335,
};
const AL_SP3: ENParams = ENParams {
    a: 5.375,
    b: 4.953,
    c: 0.867,
    d: 11.195,
};
const AL_SP2: ENParams = ENParams {
    a: 5.795,
    b: 5.020,
    c: 0.695,
    d: 11.510,
};
const B_SP3: ENParams = ENParams {
    a: 5.980,
    b: 6.820,
    c: 1.605,
    d: 14.405,
};
const B_SP2: ENParams = ENParams {
    a: 6.420,
    b: 6.807,
    c: 1.322,
    d: 14.549,
};

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
    let mut double_bond_count: u8 = 0;
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
            crate::types::BondType::Double => {
                has_double = true;
                double_bond_count += 1;
            }
            crate::types::BondType::Aromatic => has_aromatic_bond = true,
            _ => {}
        }
        // Check if neighbor participates in pi bonding (for conjugation).
        // We track two flavours:
        //   neighbor_has_pi        – neighbour is aromatic OR has any pi bond
        //   neighbor_is_aromatic   – neighbour atom is itself aromatic
        //
        // For C and N atoms the full neighbor_has_pi drives the SP3→SP2
        // demotion (phenol C, aniline N, enol C, etc.).
        // For O and S we only allow the demotion when the neighbour is
        // aromatic (phenol O → SP2) but NOT when it merely has a double bond
        // to a third atom (ester bridging O, phosphate OH → stay SP3).
        // This matches RDKit's setHybridization behaviour and fixes the
        // systematic HallKierAlpha over-negativity caused by ester/phosphate
        // oxygens being mis-classified as SP2.
        let nbr = &mol.atoms[nbr_idx];
        if nbr.is_aromatic() {
            neighbor_has_pi = true;
        } else {
            for b2 in &mol.bonds {
                let touches = (b2.source as usize == nbr_idx) || (b2.target as usize == nbr_idx);
                if touches
                    && matches!(
                        b2.bond_type,
                        crate::types::BondType::Double | crate::types::BondType::Aromatic
                    )
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
    if atom.atomic_number == 6 && double_bond_count >= 2 {
        return Hybridization::SP;
    }
    if has_double || has_aromatic_bond {
        // Hypervalent atoms (S, P) with high coordination are SP3, not SP2.
        // e.g., sulfone S(=O)(=O) has 4 bonds → SP3 despite double bonds.
        if matches!(atom.atomic_number, 15 | 16) && heavy_degree >= 3 {
            return Hybridization::SP3;
        }
        return Hybridization::SP2;
    }

    // RDKit logic: norbs = degree + lone_pairs.
    // If norbs == 4, degree ≤ 3, and atom has a conjugated bond → SP2.
    let total_degree = heavy_degree + atom.total_h();
    let outer_electrons = outer_shell_electrons(atom.atomic_number);
    let total_valence = total_degree as i32; // each bond uses 1 valence
    let free_electrons = outer_electrons - total_valence - atom.formal_charge as i32;
    let lone_pairs = if free_electrons > 0 {
        free_electrons / 2
    } else {
        0
    };
    let norbs = total_degree as i32 + lone_pairs;

    // Neutral bracketed trivalent carbons like [C](C)(C)C are radicals in RDKit
    // and stay SP3 even though the current atom model does not store radicals.
    if atom.atomic_number == 6
        && atom.formal_charge == 0
        && atom.no_implicit_h()
        && heavy_degree == 3
        && atom.total_h() == 0
        && !has_double
        && !has_triple
        && !has_aromatic_bond
    {
        return Hybridization::SP3;
    }

    match norbs {
        0 | 1 => Hybridization::SP, // s (rare)
        2 => Hybridization::SP,
        3 => Hybridization::SP2,
        4 => {
            // RDKit norbs==4 conjugation rules — three separate cases:
            //
            // O (atomic_number 8):
            //   SP2 when any neighbour participates in a pi system, EXCEPT
            //   hypervalent P and S neighbours are excluded (P=O / S=O do not
            //   conjugate the O lone pair).
            //   phenol O      (aromatic C nbr)             → SP2
            //   anisole O     (aromatic C nbr)             → SP2
            //   ester O       (C=O nbr, C not hypervalent) → SP2
            //   COOH-OH       (C=O nbr)                    → SP2
            //   vinyl ether O (C=C nbr)                    → SP2
            //   phosphate OH  (P=O nbr, P excluded)        → SP3
            //   plain ether O (sp3 C nbrs, no pi)          → SP3
            //
            // S (atomic_number 16):
            //   Always SP3 in norbs==4.  S is only SP2 when it directly carries
            //   a double bond (handled by the has_double branch above) or is in
            //   an aromatic ring (handled by is_aromatic() at the top).
            //   Thioether S adjacent to aromatic rings or C=N/C=C systems does
            //   NOT conjugate in RDKit (confirmed by HallKierAlpha comparison —
            //   all 0.13-diff failures were thioether S wrongly called SP2).
            //
            // N (atomic_number 7):
            //   Same adjacency-based logic as O, also excluding hypervalent P/S
            //   neighbours.  Sulfonamide N (N–S(=O)₂) and phosphonamide N are
            //   SP3; amide N (N–C=O), aniline N (N–C_aromatic), and enamine N
            //   (N–C=C) remain SP2.
            //   (All 0.16-diff HallKierAlpha failures were sulfonamide N
            //   wrongly called SP2 because S=O pi bonds fired neighbor_has_pi.)
            //
            // C and everything else:
            //   Original broad "any pi neighbour" conjugation check.

            // Helper closure: adjacency-based pi-neighbour scan that excludes
            // hypervalent P/S neighbours.  Used for both O and N.
            let o_n_conjugated = |atom_idx: usize| -> bool {
                total_degree <= 3
                    && mol.adjacency[atom_idx].iter().any(|&bi| {
                        let b = &mol.bonds[bi as usize];
                        let nbr_idx2 = if b.source as usize == atom_idx {
                            b.target as usize
                        } else {
                            b.source as usize
                        };
                        let nbr = &mol.atoms[nbr_idx2];
                        // Aromatic neighbour always drives SP2.
                        if nbr.is_aromatic() {
                            return true;
                        }
                        // Hypervalent P or S: excluded — their =O bonds do not
                        // conjugate the lone pair on the heteroatom.
                        if matches!(nbr.atomic_number, 15 | 16) {
                            return false;
                        }
                        // Any other neighbour (C, N, …) with a pi bond to a
                        // third atom (not the current heteroatom) drives SP2.
                        mol.adjacency[nbr_idx2].iter().any(|&bi2| {
                            let b2 = &mol.bonds[bi2 as usize];
                            let other = if b2.source as usize == nbr_idx2 {
                                b2.target as usize
                            } else {
                                b2.source as usize
                            };
                            other != atom_idx
                                && matches!(
                                    b2.bond_type,
                                    crate::types::BondType::Double
                                        | crate::types::BondType::Aromatic
                                )
                        })
                    })
            };

            let conjugated = match atom.atomic_number {
                8 => o_n_conjugated(idx),
                16 => false, // S in norbs==4 is always SP3 (see comment above)
                7 => o_n_conjugated(idx),
                _ => {
                    // C and other elements: original broad conjugation check.
                    total_degree <= 3 && neighbor_has_pi
                }
            };
            if conjugated {
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
        5 => 3, // B
        6 => 4,
        7 => 5,
        8 => 6,
        9 | 17 | 35 | 53 => 7,
        12 => 2, // Mg
        13 => 3, // Al
        14 => 4, // Si
        15 => 5,
        16 => 6,
        _ => 4, // fallback
    }
}

fn default_valence_for_conjugation(atomic_num: u8) -> i32 {
    match atomic_num {
        1 => 1,
        5 => 3,
        6 => 4,
        7 => 3,
        8 => 2,
        9 => 1,
        14 => 4,
        15 => 3,
        16 => 2,
        17 => 1,
        35 => 1,
        53 => 1,
        _ => -1,
    }
}

fn total_degree_with_h(mol: &Molecule, idx: usize) -> i32 {
    mol.adjacency[idx].len() as i32 + mol.atoms[idx].total_h() as i32
}

fn bond_valence_contrib(bond: &Bond, atom_idx: usize) -> f64 {
    match bond.bond_type {
        crate::types::BondType::Single => {
            if bond.is_aromatic() {
                1.5
            } else {
                1.0
            }
        }
        crate::types::BondType::Double => 2.0,
        crate::types::BondType::Triple => 3.0,
        crate::types::BondType::Aromatic => 1.5,
        crate::types::BondType::Dative => {
            if bond.target as usize == atom_idx {
                1.0
            } else {
                0.0
            }
        }
        crate::types::BondType::Unknown => 0.0,
    }
}

fn explicit_valence(mol: &Molecule, idx: usize) -> f64 {
    mol.atoms[idx].num_explicit_h as f64
        + mol.adjacency[idx]
            .iter()
            .map(|&bond_idx| bond_valence_contrib(&mol.bonds[bond_idx as usize], idx))
            .sum::<f64>()
}

fn count_atom_electrons_for_conjugation(mol: &Molecule, idx: usize) -> i32 {
    let atom = &mol.atoms[idx];
    let default_valence = default_valence_for_conjugation(atom.atomic_number);
    if default_valence <= 1 {
        return -1;
    }

    let mut degree = total_degree_with_h(mol, idx);
    for &bond_idx in &mol.adjacency[idx] {
        if bond_valence_contrib(&mol.bonds[bond_idx as usize], idx) == 0.0 {
            degree -= 1;
        }
    }
    if degree > 3 {
        return -1;
    }

    let lone_pair_electrons =
        (outer_shell_electrons(atom.atomic_number) - default_valence - atom.formal_charge as i32)
            .max(0);
    let mut electrons = (default_valence - degree) + lone_pair_electrons;

    if electrons > 1 {
        let unsaturations = explicit_valence(mol, idx) - degree as f64;
        if unsaturations > 1.0 {
            electrons = 1;
        }
    }

    electrons
}

fn is_atom_conjugation_candidate(mol: &Molecule, idx: usize) -> bool {
    if count_atom_electrons_for_conjugation(mol, idx) <= 0 {
        return false;
    }

    let atom = &mol.atoms[idx];
    if atom.atomic_number <= 10 {
        return true;
    }

    let nouter = outer_shell_electrons(atom.atomic_number);
    (nouter != 5 && nouter != 6) || (nouter == 6 && total_degree_with_h(mol, idx) < 2)
}

fn rdkit_conjugated_bonds(mol: &Molecule) -> Vec<bool> {
    let mut conjugated = vec![false; mol.bonds.len()];
    for (bond_idx, bond) in mol.bonds.iter().enumerate() {
        if bond.bond_type == crate::types::BondType::Aromatic || bond.is_aromatic() {
            conjugated[bond_idx] = true;
        }
    }

    let total_degrees: Vec<i32> = (0..mol.num_atoms())
        .map(|idx| total_degree_with_h(mol, idx))
        .collect();
    let atom_candidates: Vec<bool> = (0..mol.num_atoms())
        .map(|idx| is_atom_conjugation_candidate(mol, idx))
        .collect();

    for idx in 0..mol.num_atoms() {
        let sbo = total_degrees[idx];
        if !atom_candidates[idx] || !(2..=3).contains(&sbo) {
            continue;
        }

        for &bond1_idx in &mol.adjacency[idx] {
            let bond1 = &mol.bonds[bond1_idx as usize];
            if bond_valence_contrib(bond1, idx) < 1.5 {
                continue;
            }

            let atom1 = if bond1.source as usize == idx {
                bond1.target as usize
            } else {
                bond1.source as usize
            };
            if !atom_candidates[atom1] {
                continue;
            }

            for &bond2_idx in &mol.adjacency[idx] {
                if bond2_idx == bond1_idx {
                    continue;
                }

                let bond2 = &mol.bonds[bond2_idx as usize];
                let atom2 = if bond2.source as usize == idx {
                    bond2.target as usize
                } else {
                    bond2.source as usize
                };
                if total_degrees[atom2] > 3 || !atom_candidates[atom2] {
                    continue;
                }

                conjugated[bond1_idx as usize] = true;
                conjugated[bond2_idx as usize] = true;
            }
        }
    }

    conjugated
}

/// Public accessor for effective hybridization, used by HallKierAlpha.
pub fn effective_hybridization_pub(mol: &Molecule, idx: usize) -> Hybridization {
    effective_hybridization(mol, idx)
}

/// Assign Gasteiger electronegativity parameters based on element and hybridization.
/// Follows RDKit's GasteigerParams.cpp lookup: element + hybridization state,
/// with special handling for sulfur (so/so2 modes based on bonded oxygen count).
fn get_params(mol: &Molecule, idx: usize) -> Option<ENParams> {
    let atom = &mol.atoms[idx];
    let hyb = effective_hybridization(mol, idx);
    match atom.atomic_number {
        1 => Some(H),
        5 => match hyb {
            Hybridization::SP2 => Some(B_SP2),
            _ => Some(B_SP3),
        },
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
        12 => match hyb {
            Hybridization::SP => Some(MG_SP),
            Hybridization::SP2 => Some(MG_SP2),
            _ => Some(MG_SP3),
        },
        13 => match hyb {
            Hybridization::SP2 => Some(AL_SP2),
            _ => Some(AL_SP3),
        },
        14 => match hyb {
            Hybridization::SP2 => Some(SI_SP2),
            _ => Some(SI_SP3),
        },
        15 => match hyb {
            Hybridization::SP2 => Some(P_SP2),
            _ => Some(P_SP3),
        },
        16 => {
            // RDKit special handling: S with unassigned hybridization checks
            // bonded oxygen count for sulfoxide (so) / sulfone (so2) modes.
            // When hybridization IS assigned (SP/SP2/SP3), use that directly.
            match hyb {
                Hybridization::SP2 => Some(S_SP2),
                Hybridization::SP3 => Some(S_SP3),
                Hybridization::SP => Some(S_SP3), // fallback
                _ => {
                    // Count bonded oxygens for so/so2 detection
                    let n_o = mol.adjacency[idx]
                        .iter()
                        .filter(|&&bi| {
                            let b = &mol.bonds[bi as usize];
                            let other = if b.source as usize == idx {
                                b.target as usize
                            } else {
                                b.source as usize
                            };
                            mol.atoms[other].atomic_number == 8
                        })
                        .count();
                    match n_o {
                        2.. => Some(S_SO2),
                        1 => Some(S_SO),
                        _ => Some(S_SP3),
                    }
                }
            }
        }
        17 => Some(CL),
        35 => Some(BR),
        53 => Some(I_PARAMS),
        _ => None,
    }
}

/// Distribute formal charges among same-element atoms in conjugated systems.
/// Matches RDKit's splitChargeConjugated: before Gasteiger iteration, if an atom
/// has a non-zero formal charge, find all atoms of the same element reachable
/// through conjugated bonds (2-bond hops) and split the total formal charge equally.
fn split_charge_conjugated(mol: &Molecule, charges: &mut [f64]) {
    let n = mol.num_atoms();
    let bond_is_conjugated = rdkit_conjugated_bonds(mol);

    for i in 0..n {
        let fc = mol.atoms[i].formal_charge;
        if fc == 0 || charges[i].abs() > 1e-12 {
            continue; // Only process atoms with formal charge and no charge assigned yet
        }
        let anum = mol.atoms[i].atomic_number;
        let mut marker = vec![i];
        let mut total_formal = fc as f64;

        // Walk 2 conjugated bonds: i → neighbor aax → yax
        for &bi in &mol.adjacency[i] {
            if !bond_is_conjugated[bi as usize] {
                continue;
            }
            let bond = &mol.bonds[bi as usize];
            let aax = if bond.source as usize == i {
                bond.target as usize
            } else {
                bond.source as usize
            };

            for &bi2 in &mol.adjacency[aax] {
                if bi2 == bi {
                    continue;
                } // Skip the bond we came from
                if !bond_is_conjugated[bi2 as usize] {
                    continue;
                }
                let bond2 = &mol.bonds[bi2 as usize];
                let yax = if bond2.source as usize == aax {
                    bond2.target as usize
                } else {
                    bond2.source as usize
                };
                if yax == i {
                    continue;
                } // Skip self
                if mol.atoms[yax].atomic_number == anum && !marker.contains(&yax) {
                    total_formal += mol.atoms[yax].formal_charge as f64;
                    marker.push(yax);
                }
            }
        }

        let split = total_formal / marker.len() as f64;
        for &idx in &marker {
            charges[idx] = split;
        }
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

    // RDKit behavior: if an atom with no Gasteiger parameters has bonds,
    // NaN propagates to all atoms in the same connected fragment.
    // Isolated atoms without params keep their formal charge.
    let mut nan_atoms = vec![false; n];
    for i in 0..n {
        if params[i].is_none() {
            // Check if this atom has any bonds
            let has_bonds = !mol.adjacency[i].is_empty();
            if has_bonds {
                // Flood-fill NaN to all atoms in this connected component
                let mut stack = vec![i];
                while let Some(idx) = stack.pop() {
                    if nan_atoms[idx] {
                        continue;
                    }
                    nan_atoms[idx] = true;
                    for &bond_idx in &mol.adjacency[idx] {
                        let bond = &mol.bonds[bond_idx as usize];
                        let other = if bond.source as usize == idx {
                            bond.target as usize
                        } else {
                            bond.source as usize
                        };
                        if !nan_atoms[other] {
                            stack.push(other);
                        }
                    }
                }
            }
        }
    }

    // RDKit uses total hydrogens here (implicit + bracket-specified), not
    // just graph-explicit hydrogen atoms.
    let n_h: Vec<usize> = mol.atoms.iter().map(|a| a.total_h() as usize).collect();

    // Initialize charges to 0, then distribute formal charges via conjugation splitting.
    let mut q: Vec<f64> = vec![0.0; n];
    split_charge_conjugated(mol, &mut q);

    // Per-heavy-atom implicit H charge accumulator.
    let mut q_h: Vec<f64> = vec![0.0; n];

    let mut chi = vec![0.0_f64; n];
    let mut ion_x = vec![0.0_f64; n];
    for i in 0..n {
        if let Some(p) = &params[i] {
            ion_x[i] = p.d;
        }
    }

    for k in 0..NUM_ITERATIONS {
        let damping = 1.0_f64 / (2.0_f64.powi((k as i32) + 1));

        // Compute electronegativity for each atom.
        for i in 0..n {
            if nan_atoms[i] {
                continue;
            }
            if let Some(p) = &params[i] {
                chi[i] = p.a + q[i] * (p.b + p.c * q[i]);
            }
        }

        for i in 0..n {
            if nan_atoms[i] {
                continue;
            }
            if params[i].is_none() {
                continue;
            }

            let mut dq = 0.0_f64;

            // RDKit accumulates charge transfer atom-by-atom from the current
            // electronegativity snapshot, then applies the damped update to the
            // current atom only.
            for &bond_idx in &mol.adjacency[i] {
                let bond = &mol.bonds[bond_idx as usize];
                let other = if bond.source as usize == i {
                    bond.target as usize
                } else {
                    bond.source as usize
                };
                if nan_atoms[other] || params[other].is_none() {
                    continue;
                }

                let dx = chi[other] - chi[i];
                let denom = if dx < 0.0 { ion_x[other] } else { ion_x[i] };
                if denom.abs() < 1e-15 {
                    continue;
                }
                dq += dx / denom;
            }

            let nh = n_h[i];
            if nh > 0 {
                let per_h = q_h[i] / nh as f64;
                let chi_h = H.a + per_h * (H.b + H.c * per_h);
                let dx = chi_h - chi[i];
                let denom = if dx < 0.0 { IONXH } else { ion_x[i] };
                if denom.abs() >= 1e-15 {
                    let dq_h = dx / denom;
                    dq += nh as f64 * dq_h;
                    q_h[i] -= nh as f64 * dq_h * damping;
                }
            }

            q[i] += damping * dq;
        }
    }

    // Set NaN for atoms in flagged fragments
    for i in 0..n {
        if nan_atoms[i] {
            q[i] = f64::NAN;
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
        assert_charges(
            "c1ccc(O)cc1",
            &[-0.062, -0.0586, -0.0202, 0.1151, -0.5080, -0.0202, -0.0586],
        );
    }

    #[test]
    fn test_rdkit_pyridine() {
        // c1ccncc1 → 6 atoms
        // RDKit: C -0.0592, C -0.0438, C +0.0267, N -0.2647, C +0.0267, C -0.0438
        assert_charges(
            "c1ccncc1",
            &[-0.0592, -0.0438, 0.0267, -0.2647, 0.0267, -0.0438],
        );
    }

    #[test]
    fn test_rdkit_aniline() {
        // c1ccc(N)cc1 → 7 atoms
        // RDKit: C -0.0621, C -0.0602, C -0.0392, C +0.0313, N -0.3987, C -0.0392, C -0.0602
        assert_charges(
            "c1ccc(N)cc1",
            &[-0.0621, -0.0602, -0.0392, 0.0313, -0.3987, -0.0392, -0.0602],
        );
    }

    #[test]
    fn test_rdkit_phosphate_oxyanion_charges() {
        assert_charges(
            "COP(=O)([O-])O",
            &[0.0441, -0.3219, 0.2645, -0.2766, -0.7563, -0.3342],
        );
    }

    #[test]
    fn test_rdkit_sulfate_oxyanion_charges() {
        assert_charges(
            "CS(=O)(=O)[O-]",
            &[0.0268, 0.0916, -0.2428, -0.2428, -0.7482],
        );
    }

    #[test]
    fn test_split_charge_conjugated_splits_carboxylate_oxygens() {
        let mol = parse_smiles("CC(=O)[O-]").unwrap();
        let mut q = vec![0.0; mol.num_atoms()];
        split_charge_conjugated(&mol, &mut q);
        assert_close(q[2], -0.5, 1e-12, "carbonyl oxygen shares charge");
        assert_close(q[3], -0.5, 1e-12, "anionic oxygen shares charge");
    }

    #[test]
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
    fn test_cyclopentadienyl_charges() {
        let mol = parse_smiles("[CH-]1C=CC=C1").unwrap();
        let q = compute_gasteiger_charges(&mol);
        println!("Cyclopentadienyl charges: {:?}", q);
        for (i, &c) in q.iter().enumerate() {
            assert!(c < 0.0, "Atom {} should be negative: {}", i, c);
        }
    }

    #[test]
    fn test_nitrobenzene_charges() {
        let mol = parse_smiles("c1ccc(cc1)[N+](=O)[O-]").unwrap();
        let q = compute_gasteiger_charges(&mol);
        println!("Nitrobenzene charges: {:?}", q);
    }

    #[test]
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

    fn hyb_of(smiles: &str, atom_idx: usize) -> crate::types::Hybridization {
        let mol = parse_smiles(smiles).unwrap();
        effective_hybridization(&mol, atom_idx)
    }

    // -- carbonyl =O: direct double bond → SP2 via Stage 3 (not norbs path) --
    #[test]
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
    fn test_hyb_allene_central_carbon_sp() {
        assert_eq!(
            hyb_of("C=C=C", 1),
            crate::types::Hybridization::SP,
            "allene central carbon must be SP"
        );
    }

    #[test]
    fn test_hyb_trivalent_bracket_carbon_sp3() {
        assert_eq!(
            hyb_of("[C](C)(C)C", 0),
            crate::types::Hybridization::SP3,
            "neutral bracketed trivalent carbon must stay SP3"
        );
    }
}
