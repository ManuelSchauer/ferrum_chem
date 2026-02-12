//! Topological Polar Surface Area (TPSA) calculation.
//!
//! TPSA is a critical descriptor for predicting drug absorption, bioavailability,
//! and blood-brain barrier penetration. It's defined as the sum of surfaces of
//! polar atoms (usually oxygen and nitrogen) in a molecule.
//!
//! Reference: Ertl, P., Rohde, B., & Selzer, P. (2000). Fast calculation of 
//! molecular polar surface area as a sum of fragment-based contributions and 
//! its application to the prediction of drug transport properties. 
//! Journal of medicinal chemistry, 43(20), 3714-3717.
//!
//! ## Implementation
//! This implementation matches RDKit's algorithm exactly, using bond counting
//! rather than SMARTS patterns to ensure 100% accuracy with the reference.
//!
//! ## Drug-likeness Guidelines
//! - TPSA ≤ 140 Ų: Good oral bioavailability
//! - TPSA ≤ 60 Ų: Good blood-brain barrier penetration
//! - TPSA > 140 Ų: Poor oral absorption

use super::typed::{DescriptorValue, Descriptor};
use crate::mol::molecule::Molecule;
use crate::types::BondType;

/// Compute TPSA for a molecule.
///
/// If `include_sp` is true, includes sulfur and phosphorus atoms.
/// Matches RDKit's `CalcTPSA(mol, includeSandP)`.
pub fn compute_tpsa(mol: &Molecule, include_sp: bool) -> f64 {
    let n_atoms = mol.atoms.len();
    
    // Count bond types for each atom (excluding H bonds)
    let mut n_nbrs = vec![0i32; n_atoms];  // Total neighbors (degree)
    let mut n_sing = vec![0i32; n_atoms];  // Single bonds
    let mut n_doub = vec![0i32; n_atoms];  // Double bonds
    let mut n_trip = vec![0i32; n_atoms];  // Triple bonds
    let mut n_arom = vec![0i32; n_atoms];  // Aromatic bonds
    let mut n_hs = vec![0i32; n_atoms];    // Hydrogen count
    
    // Count bonds (following RDKit's exact algorithm)
    for bond in &mol.bonds {
        let begin = bond.source as usize;
        let end = bond.target as usize;
        
        // Skip H-X bonds (hydrogens handled separately)
        let begin_is_h = mol.atoms[begin].atomic_number == 1;
        let end_is_h = mol.atoms[end].atomic_number == 1;
        
        if begin_is_h && !end_is_h {
            n_hs[end] += 1;
        } else if end_is_h && !begin_is_h {
            n_hs[begin] += 1;
        } else if !begin_is_h && !end_is_h {
            // RDKit counts by aromatic FLAG first, then bond type
            // This is key for correct TPSA after kekulization
            if bond.is_aromatic() {
                n_arom[begin] += 1;
                n_arom[end] += 1;
            } else {
                // Count bond type for non-aromatic bonds
                match bond.bond_type {
                    BondType::Single => {
                        n_sing[begin] += 1;
                        n_sing[end] += 1;
                    }
                    BondType::Double => {
                        n_doub[begin] += 1;
                        n_doub[end] += 1;
                    }
                    BondType::Triple => {
                        n_trip[begin] += 1;
                        n_trip[end] += 1;
                    }
                    BondType::Aromatic => {
                        // Fallback: treat Aromatic bond type as aromatic
                        // This happens before kekulization
                        n_arom[begin] += 1;
                        n_arom[end] += 1;
                    }
                    _ => {} // Skip Dative, Unknown
                }
            }
            // Count degree (exclude hydrogens)
            n_nbrs[begin] += 1;
            n_nbrs[end] += 1;
        }
    }
    
    // Add hydrogens (both implicit and explicit from bracket notation)
    for i in 0..n_atoms {
        let atom = &mol.atoms[i];
        n_hs[i] += atom.total_h() as i32;
    }
    
    let mut tpsa = 0.0;
    
    // Calculate TPSA contribution for each atom
    for i in 0..n_atoms {
        let atom = &mol.atoms[i];
        let atomic_num = atom.atomic_number as u32;
        
        // Only process N, O, S, P
        if atomic_num != 7 && atomic_num != 8 &&
           (!include_sp || (atomic_num != 15 && atomic_num != 16)) {
            continue;
        }
        
        let chg = atom.formal_charge;
        let in_3_ring = is_in_3_ring(mol, i);
        
        let contribution = match atomic_num {
            7 => calculate_n_contribution(n_nbrs[i], n_hs[i], chg, n_sing[i], n_doub[i], n_trip[i], n_arom[i], in_3_ring),
            8 => calculate_o_contribution(n_nbrs[i], n_hs[i], chg, n_sing[i], n_doub[i], n_arom[i], in_3_ring),
            15 => calculate_p_contribution(n_nbrs[i], n_hs[i], chg, n_sing[i], n_doub[i]),
            16 => calculate_s_contribution(n_nbrs[i], n_hs[i], chg, n_sing[i], n_doub[i], n_arom[i]),
            _ => 0.0,
        };
        
        tpsa += contribution;
    }
    
    tpsa
}

/// Check if atom is in a 3-membered ring.
/// An atom is in a 3-ring if any two of its neighbors are bonded to each other.
fn is_in_3_ring(mol: &Molecule, atom_idx: usize) -> bool {
    let nbrs: Vec<usize> = mol.adjacency[atom_idx]
        .iter()
        .map(|&bi| {
            let b = &mol.bonds[bi as usize];
            if b.source as usize == atom_idx { b.target as usize } else { b.source as usize }
        })
        .collect();
    for i in 0..nbrs.len() {
        for j in (i + 1)..nbrs.len() {
            if mol.adjacency[nbrs[i]].iter().any(|&bi| {
                let b = &mol.bonds[bi as usize];
                b.source as usize == nbrs[j] || b.target as usize == nbrs[j]
            }) {
                return true;
            }
        }
    }
    false
}

/// Calculate nitrogen contribution (exact RDKit algorithm)
fn calculate_n_contribution(n_nbrs: i32, n_hs: i32, chg: i8, n_sing: i32, n_doub: i32, n_trip: i32, n_arom: i32, in_3_ring: bool) -> f64 {
    match n_nbrs {
        1 => {
            if n_hs == 0 && chg == 0 && n_trip == 1 {
                23.79  // -C≡N (nitrile)
            } else if n_hs == 1 && chg == 0 && n_doub == 1 {
                23.85  // =NH (imine with H)
            } else if n_hs == 2 && chg == 0 && n_sing == 1 {
                26.02  // -NH2 (primary amine)
            } else if n_hs == 2 && chg == 1 && n_doub == 1 {
                25.59  // =NH2+ (protonated imine)
            } else if n_hs == 3 && chg == 1 && n_sing == 1 {
                27.64  // -NH3+ (ammonium)
            } else {
                fallback_n(n_nbrs, n_hs)
            }
        }
        2 => {
            if n_hs == 0 && chg == 0 && n_sing == 1 && n_doub == 1 {
                12.36  // -N= (imine)
            } else if n_hs == 0 && chg == 0 && n_trip == 1 && n_doub == 1 {
                13.60  // -N≡+ or similar
            } else if n_hs == 1 && chg == 0 && n_sing == 2 && in_3_ring {
                21.94  // -NH- in 3-membered ring
            } else if n_hs == 1 && chg == 0 && n_sing == 2 && !in_3_ring {
                12.03  // -NH- (secondary amine)
            } else if n_hs == 0 && chg == 1 && n_trip == 1 && n_sing == 1 {
                4.36
            } else if n_hs == 1 && chg == 1 && n_doub == 1 && n_sing == 1 {
                13.97
            } else if n_hs == 2 && chg == 1 && n_sing == 2 {
                16.61
            } else if n_hs == 0 && chg == 0 && n_arom == 2 {
                12.89  // =n- (aromatic imine, pyridine)
            } else if n_hs == 1 && chg == 0 && n_arom == 2 {
                15.79  // -nH (aromatic pyrrole)
            } else if n_hs == 1 && chg == 1 && n_arom == 2 {
                14.14  // -nH+ (protonated aromatic)
            } else {
                fallback_n(n_nbrs, n_hs)
            }
        }
        3 => {
            if n_hs == 0 && chg == 0 && n_sing == 3 && in_3_ring {
                3.01  // N in 3-membered ring
            } else if n_hs == 0 && chg == 0 && n_sing == 3 && !in_3_ring {
                3.24  // >N- (tertiary amine)
            } else if n_hs == 0 && chg == 0 && n_sing == 1 && n_doub == 2 {
                11.68  // -N(=O)=O (nitro N, neutral)
            } else if n_hs == 0 && chg == 1 && n_sing == 2 && n_doub == 1 {
                3.01  // >N+=O (N-oxide)
            } else if n_hs == 1 && chg == 1 && n_sing == 3 {
                4.44
            } else if n_hs == 0 && chg == 0 && n_arom == 3 {
                4.41  // aromatic N with 3 aromatic bonds
            } else if n_hs == 0 && chg == 0 && n_sing == 1 && n_arom == 2 {
                4.93  // -n< (N-alkylated aromatic)
            } else if n_hs == 0 && chg == 0 && n_doub == 1 && n_arom == 2 {
                8.39  // =n< (aromatic with exocyclic double bond)
            } else if n_hs == 0 && chg == 1 && n_arom == 3 {
                4.10
            } else if n_hs == 0 && chg == 1 && n_sing == 1 && n_arom == 2 {
                3.88
            } else {
                fallback_n(n_nbrs, n_hs)
            }
        }
        4 => {
            if n_hs == 0 && n_sing == 4 && chg == 1 {
                0.0  // R4N+ (quaternary ammonium)
            } else {
                fallback_n(n_nbrs, n_hs)
            }
        }
        _ => fallback_n(n_nbrs, n_hs)
    }
}

/// Fallback formula for nitrogen
fn fallback_n(n_nbrs: i32, n_hs: i32) -> f64 {
    let tmp = 30.5 - (n_nbrs as f64) * 8.2 + (n_hs as f64) * 1.5;
    tmp.max(0.0)
}

/// Calculate oxygen contribution (exact RDKit algorithm)
fn calculate_o_contribution(n_nbrs: i32, n_hs: i32, chg: i8, n_sing: i32, n_doub: i32, n_arom: i32, in_3_ring: bool) -> f64 {
    match n_nbrs {
        1 => {
            if n_hs == 0 && chg == 0 && n_doub == 1 {
                17.07  // =O (carbonyl)
            } else if n_hs == 1 && chg == 0 && n_sing == 1 {
                20.23  // -OH (hydroxyl)
            } else if n_hs == 0 && chg == -1 && n_sing == 1 {
                23.06  // -O- (negatively charged, carboxylate)
            } else {
                fallback_o(n_nbrs, n_hs)
            }
        }
        2 => {
            if n_hs == 0 && chg == 0 && n_sing == 2 && in_3_ring {
                12.53  // O in 3-membered ring (epoxide)
            } else if n_hs == 0 && chg == 0 && n_sing == 2 && !in_3_ring {
                9.23  // -O- (ether)
            } else if n_hs == 0 && chg == 0 && n_arom == 2 {
                13.14  // o (aromatic oxygen, furan)
            } else {
                fallback_o(n_nbrs, n_hs)
            }
        }
        _ => fallback_o(n_nbrs, n_hs)
    }
}

/// Fallback formula for oxygen
fn fallback_o(n_nbrs: i32, n_hs: i32) -> f64 {
    let tmp = 28.5 - (n_nbrs as f64) * 8.6 + (n_hs as f64) * 1.5;
    tmp.max(0.0)
}

/// Calculate phosphorus contribution (exact RDKit algorithm)
fn calculate_p_contribution(n_nbrs: i32, n_hs: i32, chg: i8, n_sing: i32, n_doub: i32) -> f64 {
    match n_nbrs {
        2 => {
            if n_hs == 0 && chg == 0 && n_sing == 1 && n_doub == 1 {
                34.14
            } else {
                0.0
            }
        }
        3 => {
            if n_hs == 0 && chg == 0 && n_sing == 3 {
                13.59
            } else if n_hs == 1 && chg == 0 && n_sing == 2 && n_doub == 1 {
                23.47
            } else {
                0.0
            }
        }
        4 => {
            if n_hs == 0 && chg == 0 && n_sing == 3 && n_doub == 1 {
                9.81
            } else {
                0.0
            }
        }
        _ => 0.0
    }
}

/// Calculate sulfur contribution (exact RDKit algorithm)
fn calculate_s_contribution(n_nbrs: i32, n_hs: i32, chg: i8, n_sing: i32, n_doub: i32, n_arom: i32) -> f64 {
    match n_nbrs {
        1 => {
            if n_hs == 0 && chg == 0 && n_doub == 1 {
                32.09  // =S (thione)
            } else if n_hs == 1 && chg == 0 && n_sing == 1 {
                38.80  // -SH (thiol)
            } else {
                0.0
            }
        }
        2 => {
            if n_hs == 0 && chg == 0 && n_sing == 2 {
                25.30  // -S- (thioether)
            } else if n_hs == 0 && chg == 0 && n_arom == 2 {
                28.24  // s (aromatic sulfur, thiophene)
            } else {
                0.0
            }
        }
        3 => {
            if n_hs == 0 && chg == 0 && n_arom == 2 && n_doub == 1 {
                21.70
            } else if n_hs == 0 && chg == 0 && n_sing == 2 && n_doub == 1 {
                19.21  // sulfoxide
            } else {
                0.0
            }
        }
        4 => {
            if n_hs == 0 && chg == 0 && n_sing == 2 && n_doub == 2 {
                8.38  // sulfone
            } else {
                0.0
            }
        }
        _ => 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn tpsa(smiles: &str) -> f64 {
        let mol = parse_smiles(smiles).unwrap();
        compute_tpsa(&mol, false)
    }

    #[test]
    fn test_tpsa_comprehensive_rdkit_validation() {
        // Comprehensive validation against RDKit TPSA(includeSandP=False) defaults
        let cases: Vec<(&str, &str, f64)> = vec![
            ("CCCC", "Butane", 0.00),
            ("CCO", "Ethanol", 20.23),
            ("COC", "DimethylEther", 9.23),
            ("CC(=O)C", "Acetone", 17.07),
            ("CC(=O)O", "AceticAcid", 37.30),
            ("CC(=O)OC", "MethylAcetate", 26.30),
            ("CN", "Methylamine", 26.02),
            ("CNC", "Dimethylamine", 12.03),
            ("CN(C)C", "Trimethylamine", 3.24),
            ("c1cccnc1", "Pyridine", 12.89),
            ("c1cc[nH]c1", "Pyrrole", 15.79),
            ("c1ccoc1", "Furan", 13.14),
            ("c1ccsc1", "Thiophene", 0.00),
            ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin", 63.60),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine", 61.82),
            ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen", 37.30),
            ("CC(=O)N", "Acetamide", 43.09),
            ("c1ccccc1", "Benzene", 0.00),
            ("CC(=O)Nc1ccc(O)cc1", "Paracetamol", 49.33),
            ("CC(O)=O", "AceticAcid2", 37.30),
            ("C(=O)O", "FormicAcid", 37.30),
            ("c1ccc(cc1)O", "Phenol", 20.23),
            ("c1ccc(cc1)N", "Aniline", 26.02),
            ("C#N", "HCN", 23.79),
            ("CC#N", "Acetonitrile", 23.79),
            ("C=O", "Formaldehyde", 17.07),
            ("[NH3+]CC([O-])=O", "Glycine_zwitter", 67.77),
            ("OC(=O)c1ccccc1", "BenzoicAcid", 37.30),
            ("c1cnc2ccccc2n1", "Quinazoline", 25.78),
            ("Nc1ncnc2[nH]cnc12", "Adenine", 80.48),
            ("c1ccc2[nH]ccc2c1", "Indole", 15.79),
            ("O=C(O)CC(O)(CC(=O)O)C(=O)O", "CitricAcid", 132.13),
            ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "Testosterone", 37.30),
            ("O=c1[nH]c(=O)c2[nH]cnc2[nH]1", "UricAcid", 94.40),
            ("c1cc2ccccc2[nH]1", "Indole2", 15.79),
            ("O=S(=O)(c1ccc(cc1)N)N", "Sulfanilamide", 86.18),
            ("c1ccc(cc1)S", "Thiophenol", 0.00),
            ("CS(=O)C", "DMSO", 17.07),
            ("CS(=O)(=O)C", "DMSO2", 34.14),
            ("c1ccncc1", "Pyridine2", 12.89),
            ("O=[N+]([O-])c1ccccc1", "Nitrobenzene", 43.14),
            // Extended validation
            ("CC(C)NCC(O)c1ccc(O)c(O)c1", "Isoprenaline", 72.72),
            ("c1ccc2c(c1)[nH]c3ccccc23", "Carbazole", 15.79),
            ("O=C1NC(=O)c2ccccc21", "Isatin-like", 46.17),
            ("CC(C)(C)NCC(O)c1ccccc1", "Phenylethanolamine", 32.26),
            ("c1ccnnc1", "Pyridazine", 25.78),
            ("c1cnccn1", "Pyrimidine", 25.78),
            ("c1cnncn1", "Triazine", 38.67),
            ("NC(=O)c1ccccc1", "Benzamide", 43.09),
            ("Oc1ccccc1O", "Catechol", 40.46),
        ];
        
        let mut failures = Vec::new();
        let mut count = 0;
        
        for (smi, name, expected) in &cases {
            match parse_smiles(smi) {
                Ok(mol) => {
                    let got = compute_tpsa(&mol, false);
                    let err = (got - expected).abs();
                    count += 1;
                    if err > 0.01 {
                        failures.push(format!("{name}: got {got:.2}, expected {expected:.2}, err {err:.2}"));
                    }
                }
                Err(e) => {
                    failures.push(format!("{name}: PARSE FAILED ({smi}): {e}"));
                }
            }
        }
        
        if !failures.is_empty() {
            println!("\nTPSA Failures ({}/{count}):", failures.len());
            for f in &failures {
                println!("  {f}");
            }
        }
        
        assert!(failures.is_empty(), 
                "TPSA mismatches ({}/{count}):\n{}", failures.len(), failures.join("\n"));
    }

    #[test]
    fn test_tpsa_alkane() {
        assert!((tpsa("CCCC") - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_alcohol() {
        assert!((tpsa("CCO") - 20.23).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_ether() {
        assert!((tpsa("COC") - 9.23).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_ketone() {
        assert!((tpsa("CC(=O)C") - 17.07).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_carboxylic_acid() {
        assert!((tpsa("CC(=O)O") - 37.30).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_ester() {
        assert!((tpsa("CC(=O)OC") - 26.30).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_primary_amine() {
        assert!((tpsa("CN") - 26.02).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_secondary_amine() {
        assert!((tpsa("CNC") - 12.03).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_tertiary_amine() {
        assert!((tpsa("CN(C)C") - 3.24).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_pyridine() {
        assert!((tpsa("c1cccnc1") - 12.89).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_aspirin() {
        assert!((tpsa("CC(=O)Oc1ccccc1C(=O)O") - 63.60).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_caffeine() {
        assert!((tpsa("CN1C=NC2=C1C(=O)N(C(=O)N2C)C") - 61.82).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_ibuprofen() {
        assert!((tpsa("CC(C)Cc1ccc(cc1)C(C)C(=O)O") - 37.30).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_amide() {
        assert!((tpsa("CC(=O)N") - 43.09).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_3ring_epoxides() {
        // Epoxides: O in 3-ring → 12.53 (vs 9.23 for normal ether)
        assert!((tpsa("C1CO1") - 12.53).abs() < 0.01, "ethylene oxide");
        assert!((tpsa("CC1CO1") - 12.53).abs() < 0.01, "propylene oxide");
        assert!((tpsa("c1ccc(C2CO2)cc1") - 12.53).abs() < 0.01, "styrene oxide");
        // glycidol: epoxide O (12.53) + hydroxyl OH (20.23) = 32.76
        assert!((tpsa("OCC1CO1") - 32.76).abs() < 0.01, "glycidol");
    }

    #[test]
    fn test_tpsa_3ring_aziridines() {
        // Aziridines: N in 3-ring
        // -NH- in 3-ring → 21.94 (vs 12.03 for normal secondary amine)
        assert!((tpsa("C1CN1") - 21.94).abs() < 0.01, "aziridine");
        assert!((tpsa("CC1CN1") - 21.94).abs() < 0.01, "2-methylaziridine");
        // >N- in 3-ring → 3.01 (vs 3.24 for normal tertiary amine)
        assert!((tpsa("CN1CC1") - 3.01).abs() < 0.01, "N-methylaziridine");
        // aziridine-2-carboxamide: NH-3ring (21.94) + amide C(=O)N (43.09) = 65.03
        assert!((tpsa("O=C(N)C1CN1") - 65.03).abs() < 0.01, "aziridine-2-carboxamide");
    }

    #[test]
    fn test_tpsa_3ring_vs_larger_rings() {
        // Confirm non-3-ring molecules get normal (non-3-ring) values
        assert!((tpsa("C1CCO1") - 9.23).abs() < 0.01, "oxetane (4-ring O → 9.23 ether)");
        assert!((tpsa("C1CCOC1") - 9.23).abs() < 0.01, "THF (5-ring O → 9.23 ether)");
        assert!((tpsa("C1CCN1") - 12.03).abs() < 0.01, "azetidine (4-ring N → 12.03)");
        assert!((tpsa("C1CCNCC1") - 12.03).abs() < 0.01, "piperidine (6-ring N → 12.03)");
        assert!((tpsa("C1COCCN1") - 21.26).abs() < 0.01, "morpholine (O + NH → 9.23 + 12.03)");
    }
}

/// Topological Polar Surface Area (TPSA) descriptor.
///
/// Returns a single Float value representing TPSA in Ų.
///
/// ## Thread Safety
/// Stateless and thread-safe.
///
/// # Examples
///
/// ```
/// use ferrum_core::desc::tpsa::Tpsa;
/// use ferrum_core::desc::typed::Descriptor;
/// use ferrum_core::io::smiles::parse_smiles;
/// use ferrum_core::algo::aromaticity::perceive_aromaticity;
///
/// let mut mol = parse_smiles("c1ccccc1").unwrap();
/// perceive_aromaticity(&mut mol);
///
/// let descriptor = Tpsa;
/// let tpsa = descriptor.calculate(&mut mol);
///
/// // Benzene has no polar atoms
/// match tpsa {
///     ferrum_core::desc::typed::DescriptorValue::Float(v) => {
///         assert!((v - 0.0).abs() < 0.01);
///     }
///     _ => panic!("Expected Float"),
/// }
/// ```
pub struct Tpsa;

impl Descriptor for Tpsa {
    fn name(&self) -> &str {
        "TPSA"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let tpsa = compute_tpsa(mol, false) as f32;
        DescriptorValue::Float(tpsa)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Tpsa>;
};
