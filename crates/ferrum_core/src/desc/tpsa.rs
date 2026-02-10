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
//! ## Drug-likeness Guidelines
//! - TPSA ≤ 140 Ų: Good oral bioavailability
//! - TPSA ≤ 60 Ų: Good blood-brain barrier penetration
//! - TPSA > 140 Ų: Poor oral absorption

use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::Real;
use crate::io::smarts::{parse_smarts, QueryGraph};
use crate::algo::iso::find_all_matches;
use once_cell::sync::Lazy;

/// Pre-parsed TPSA fragment patterns.
/// Patterns are parsed once at startup for maximum performance.
static PARSED_PATTERNS: Lazy<Vec<ParsedPattern>> = Lazy::new(|| {
    TPSA_PATTERNS.iter()
        .filter_map(|p| {
            parse_smarts(p.smarts)
                .ok()
                .map(|query| ParsedPattern {
                    description: p.description,
                    query,
                    contribution: p.contribution,
                })
        })
        .collect()
});

/// Parsed SMARTS pattern with PSA contribution.
struct ParsedPattern {
    description: &'static str,
    query: QueryGraph,
    contribution: f64,
}

/// TPSA pattern definition from Ertl et al.
struct TpsaPattern {
    smarts: &'static str,
    description: &'static str,
    contribution: f64, // Surface area in Ų (square angstroms)
}

/// Topological Polar Surface Area (TPSA) descriptor.
///
/// Calculates the sum of polar surface areas based on nitrogen and oxygen
/// fragment contributions. Uses SMARTS pattern matching with the Ertl method.
///
/// ## Examples
/// - Aspirin: ~63.6 Ų
/// - Caffeine: ~58.4 Ų
/// - Ibuprofen: ~37.3 Ų
/// - Atorvastatin: ~111.8 Ų
///
/// ## Thread Safety
/// Stateless and thread-safe. Patterns are pre-parsed globally.
pub struct Tpsa;

impl Descriptor for Tpsa {
    fn name(&self) -> String {
        "TPSA".to_string()
    }

    fn size(&self) -> usize {
        1
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        // Track which atoms have been assigned
        let mut assigned = vec![false; mol.atoms.len()];
        let mut tpsa_sum = 0.0;

        // Try each pattern in order (first match wins per atom)
        for pattern in PARSED_PATTERNS.iter() {
            let matches = find_all_matches(mol, &pattern.query);

            // For each match, assign the first unassigned atom
            for m in matches {
                if !m.is_empty() {
                    let atom_idx = m[0] as usize;
                    if !assigned[atom_idx] {
                        assigned[atom_idx] = true;
                        tpsa_sum += pattern.contribution;
                    }
                }
            }
        }

        output[0] = tpsa_sum as Real;
    }
}

/// TPSA fragment patterns from Ertl et al. (2000).
///
/// Pattern matching order matters: more specific patterns come first.
/// Each pattern matches the polar atom (N or O) with its immediate neighbors.
///
/// The contributions are empirically derived surface areas in Ų.
const TPSA_PATTERNS: &[TpsaPattern] = &[
    // Nitrogen fragments (ordered from most to least specific)
    
    // Quaternary nitrogen: N+ with 4 bonds (R4N+)
    TpsaPattern {
        smarts: "[N+;D4]",
        description: "Quaternary nitrogen N+",
        contribution: 0.0,
    },
    
    // Tertiary nitrogen: N with 3 bonds, 0 H (R3N)
    TpsaPattern {
        smarts: "[N;D3;H0;+0]",
        description: "Tertiary nitrogen N(-)(-)(-))",
        contribution: 3.24,
    },
    
    // Secondary amine: N with 2 bonds, 1 H (R2NH) - aliphatic
    TpsaPattern {
        smarts: "[N;D2;H1;+0]",
        description: "Secondary amine NH(-)-)",
        contribution: 12.03,
    },
    
    // Primary amine: N with 1 bond, 2 H (RNH2)
    TpsaPattern {
        smarts: "[N;D1;H2;+0]",
        description: "Primary amine NH2",
        contribution: 26.02,
    },
    
    // Nitrogen with explicit H (generic fallback for aliphatic)
    TpsaPattern {
        smarts: "[NH2;+0]",
        description: "NH2 generic",
        contribution: 26.02,
    },
    TpsaPattern {
        smarts: "[NH;+0]",
        description: "NH generic",
        contribution: 12.03,
    },
    
    // Pyrrole-like nitrogen: aromatic with 1 H (D2 aromatic)
    TpsaPattern {
        smarts: "[n;H1]",
        description: "Pyrrole nitrogen nH",
        contribution: 15.79,
    },
    
    // Aromatic nitrogen without H (pyridine-like) - catch-all for aromatic
    // This should match pyridine N (D2, no H) and similar
    TpsaPattern {
        smarts: "[n;H0]",
        description: "Aromatic nitrogen (pyridine)",
        contribution: 12.89,
    },
    
    // Fallback aromatic nitrogen
    TpsaPattern {
        smarts: "[n;+0]",
        description: "Aromatic nitrogen generic",
        contribution: 12.89,
    },
    
    // Any other nitrogen (aliphatic)
    TpsaPattern {
        smarts: "[N;+0]",
        description: "Nitrogen generic",
        contribution: 3.24,
    },
    
    // Oxygen fragments (ordered from most to least specific)
    
    // O in aromatic ring (furan-like)
    TpsaPattern {
        smarts: "[o;+0]",
        description: "Aromatic oxygen",
        contribution: 13.14,
    },
    
    // Oxygen with 2 bonds (ether, alcohol)
    TpsaPattern {
        smarts: "[O;D2;H0;+0]",
        description: "Ether oxygen O(-)-)",
        contribution: 9.23,
    },
    
    // Hydroxyl oxygen (1 bond, 1 H)
    TpsaPattern {
        smarts: "[O;D1;H1;+0]",
        description: "Hydroxyl oxygen OH",
        contribution: 20.23,
    },
    
    // Carbonyl oxygen (1 bond, 0 H) - C=O
    TpsaPattern {
        smarts: "[O;D1;H0;+0]",
        description: "Carbonyl oxygen =O",
        contribution: 17.07,
    },
    
    // Generic hydroxyl
    TpsaPattern {
        smarts: "[OH]",
        description: "OH generic",
        contribution: 20.23,
    },
    
    // Any other oxygen
    TpsaPattern {
        smarts: "[O;+0]",
        description: "Oxygen generic",
        contribution: 17.07,
    },
    
    // Sulfur and phosphorus (less common but included for completeness)
    
    // Sulfoxide S=O
    TpsaPattern {
        smarts: "[S;D3](=O)",
        description: "Sulfoxide S=O",
        contribution: 17.07,
    },
    
    // Sulfonyl S(=O)(=O)
    TpsaPattern {
        smarts: "[S;D4](=O)(=O)",
        description: "Sulfonyl SO2",
        contribution: 34.14,
    },
    
    // Thiol SH
    TpsaPattern {
        smarts: "[SH]",
        description: "Thiol SH",
        contribution: 25.30,
    },
    
    // Phosphate P=O
    TpsaPattern {
        smarts: "[P;D4](=O)",
        description: "Phosphate P=O",
        contribution: 17.07,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn tpsa(smiles: &str) -> f64 {
        let mut mol = parse_smiles(smiles).unwrap();
        let desc = Tpsa;
        let mut output = vec![0.0];
        desc.calculate(&mol, &mut output);
        output[0] as f64
    }

    #[test]
    fn test_tpsa_alkane() {
        // Butane: no polar atoms
        assert!((tpsa("CCCC") - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_tpsa_alcohol() {
        // Ethanol: OH = 20.23 Ų
        assert!((tpsa("CCO") - 20.23).abs() < 0.5);
    }

    #[test]
    fn test_tpsa_ether() {
        // Dimethyl ether: O(-)(-)= 9.23 Ų
        assert!((tpsa("COC") - 9.23).abs() < 0.5);
    }

    #[test]
    fn test_tpsa_ketone() {
        // Acetone: C=O = 17.07 Ų
        assert!((tpsa("CC(=O)C") - 17.07).abs() < 0.5);
    }

    #[test]
    fn test_tpsa_carboxylic_acid() {
        // Acetic acid: OH (20.23) + C=O (17.07) = 37.30 Ų
        // Note: one O might be carbonyl, one OH
        let result = tpsa("CC(=O)O");
        assert!(result > 35.0 && result < 40.0, "Got {}", result);
    }

    #[test]
    fn test_tpsa_ester() {
        // Methyl acetate: C=O (17.07) + O-C (9.23) = 26.30 Ų
        let result = tpsa("CC(=O)OC");
        assert!(result > 24.0 && result < 28.0, "Got {}", result);
    }

    #[test]
    fn test_tpsa_primary_amine() {
        // Methylamine: NH2 = 26.02 Ų
        assert!((tpsa("CN") - 26.02).abs() < 0.5);
    }

    #[test]
    fn test_tpsa_secondary_amine() {
        // Dimethylamine: NH = 12.03 Ų
        let result = tpsa("CNC");
        assert!((result - 12.03).abs() < 1.0, "Got {}", result);
    }

    #[test]
    fn test_tpsa_tertiary_amine() {
        // Trimethylamine: N = 3.24 Ų
        let result = tpsa("CN(C)C");
        assert!((result - 3.24).abs() < 1.0, "Got {}", result);
    }

    #[test]
    fn test_tpsa_pyridine() {
        // Pyridine: aromatic N = 12.89 Ų
        let result = tpsa("c1cccnc1");
        assert!((result - 12.89).abs() < 1.0, "Got {}", result);
    }

    #[test]
    fn test_tpsa_aspirin() {
        // Aspirin: carboxylic acid + ester
        // COOH: ~37.3, ester O: ~26.3, aromatic: 0
        // Total: ~63.6 Ų
        let result = tpsa("CC(=O)Oc1ccccc1C(=O)O");
        assert!(result > 60.0 && result < 70.0, "Aspirin TPSA = {}", result);
    }

    #[test]
    fn test_tpsa_caffeine() {
        // Caffeine: 4 nitrogens + 2 carbonyls
        // Expected: ~58.4 Ų (RDKit value)
        // Note: Caffeine has complex aromatic N system, actual value may vary
        let result = tpsa("CN1C=NC2=C1C(=O)N(C(=O)N2C)C");
        assert!(result > 45.0 && result < 65.0, "Caffeine TPSA = {}", result);
    }

    #[test]
    fn test_tpsa_ibuprofen() {
        // Ibuprofen: carboxylic acid COOH
        // Expected: ~37.3 Ų
        let result = tpsa("CC(C)Cc1ccc(cc1)C(C)C(=O)O");
        assert!(result > 35.0 && result < 40.0, "Ibuprofen TPSA = {}", result);
    }

    #[test]
    fn test_tpsa_amide() {
        // Acetamide: C=O + NH2
        // Expected: 17.07 + 26.02 = 43.09 Ų
        let result = tpsa("CC(=O)N");
        assert!(result > 40.0 && result < 46.0, "Amide TPSA = {}", result);
    }
}
