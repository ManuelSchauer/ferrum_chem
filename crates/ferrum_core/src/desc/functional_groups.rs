//! Functional group counting descriptors using SMARTS pattern matching.
//!
//! Provides counts for common functional groups, acidic groups, and basic groups.

use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::Real;
use crate::io::smarts::{parse_smarts, QueryGraph};
use crate::algo::iso::find_all_matches;
use once_cell::sync::Lazy;

/// A named SMARTS pattern for functional group detection.
struct FuncGroupPattern {
    name: &'static str,
    query: QueryGraph,
}

macro_rules! fg_patterns {
    ($name:ident, $( ($label:expr, $smarts:expr) ),+ $(,)?) => {
        static $name: Lazy<Vec<FuncGroupPattern>> = Lazy::new(|| {
            vec![
                $(
                    {
                        let q = parse_smarts($smarts).expect(concat!("Bad SMARTS: ", $smarts));
                        FuncGroupPattern { name: $label, query: q }
                    },
                )+
            ]
        });
    };
}

// Core functional group SMARTS patterns
fg_patterns!(FUNC_GROUP_PATTERNS,
    ("Hydroxyl",        "[OX2H]"),
    ("Aldehyde",        "[CX3H1](=O)[#6]"),
    ("Ketone",          "[#6][CX3](=O)[#6]"),
    ("CarboxylicAcid",  "[CX3](=O)[OX2H1]"),
    ("Ester",           "[#6][CX3](=O)[OX2H0][#6]"),
    ("Amide",           "[NX3][CX3](=O)[#6]"),
    ("PrimaryAmine",    "[NX3;H2;!$(NC=O)]"),
    ("SecondaryAmine",  "[NX3;H1;!$(NC=O)]"),
    ("TertiaryAmine",   "[NX3;H0;!$(NC=O);!$([NX3](=O)=O)]"),
    ("Nitro",           "[$([NX3](=O)=O)]"),
    ("Sulfonamide",     "[#16X4]([NX3])(=O)=O"),
    ("Sulfone",         "[#16X4](=O)(=O)([#6])[#6]"),
    ("Thiol",           "[#16X2H]"),
    ("Ether",           "[OD2]([#6])[#6]"),
    ("Nitrile",         "[NX1]#[CX2]"),
    ("Halide",          "[F,Cl,Br,I]"),
);

// Acidic group patterns
fg_patterns!(ACIDIC_PATTERNS,
    ("CarboxylicAcid",  "[CX3](=O)[OX2H1]"),
    ("Sulfonamide_NH",  "[#16X4](=O)(=O)[NX3H]"),
    ("SulfonicAcid",    "[#16X4](=O)(=O)[OX2H]"),
    ("Phenol",          "[OX2H]c"),
    ("Thiol",           "[#16X2H]"),
);

// Basic group patterns
fg_patterns!(BASIC_PATTERNS,
    ("PrimaryAmine",    "[NX3;H2;!$(NC=O)]"),
    ("SecondaryAmine",  "[NX3;H1;!$(NC=O)]"),
    ("TertiaryAmine",   "[NX3;H0;!$(NC=O);!$([NX3](=O)=O)]"),
    ("Guanidine",       "[NX3][CX3](=[NX2])[NX3]"),
    ("Amidine",         "[NX3][CX3]=[NX2]"),
);

fn count_matches(mol: &Molecule, patterns: &[FuncGroupPattern]) -> Vec<(String, u32)> {
    patterns.iter().map(|p| {
        let matches = find_all_matches(mol, &p.query);
        (p.name.to_string(), matches.len() as u32)
    }).collect()
}

/// Functional Group Counts descriptor.
///
/// Counts 16 common functional groups using SMARTS patterns.
/// Output order: Hydroxyl, Aldehyde, Ketone, CarboxylicAcid, Ester, Amide,
/// PrimaryAmine, SecondaryAmine, TertiaryAmine, Nitro, Sulfonamide, Sulfone,
/// Thiol, Ether, Nitrile, Halide.
pub struct FunctionalGroupCounts;

impl Descriptor for FunctionalGroupCounts {
    fn name(&self) -> String {
        "FunctionalGroups".to_string()
    }

    fn size(&self) -> usize {
        FUNC_GROUP_PATTERNS.len()
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        for (i, pattern) in FUNC_GROUP_PATTERNS.iter().enumerate() {
            let matches = find_all_matches(mol, &pattern.query);
            output[i] = matches.len() as Real;
        }
    }
}

/// Acidic Group Count: total number of acidic functional groups.
///
/// Counts carboxylic acids, sulfonamide NHs, sulfonic acids, phenols, thiols.
pub struct AcidicGroupCount;

impl Descriptor for AcidicGroupCount {
    fn name(&self) -> String {
        "AcidicGroupCount".to_string()
    }

    fn size(&self) -> usize {
        1
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let total: usize = ACIDIC_PATTERNS.iter()
            .map(|p| find_all_matches(mol, &p.query).len())
            .sum();
        output[0] = total as Real;
    }
}

/// Basic Group Count: total number of basic functional groups.
///
/// Counts primary/secondary/tertiary amines, guanidines, amidines.
pub struct BasicGroupCount;

impl Descriptor for BasicGroupCount {
    fn name(&self) -> String {
        "BasicGroupCount".to_string()
    }

    fn size(&self) -> usize {
        1
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let total: usize = BASIC_PATTERNS.iter()
            .map(|p| find_all_matches(mol, &p.query).len())
            .sum();
        output[0] = total as Real;
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<FunctionalGroupCounts>;
    let _ = assert_send_sync::<AcidicGroupCount>;
    let _ = assert_send_sync::<BasicGroupCount>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_fg(smiles: &str) -> Vec<Real> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let desc = FunctionalGroupCounts;
        let mut output = vec![0.0; desc.size()];
        desc.calculate(&mol, &mut output);
        output
    }

    fn calc_acidic(smiles: &str) -> Real {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let mut output = [0.0];
        AcidicGroupCount.calculate(&mol, &mut output);
        output[0]
    }

    fn calc_basic(smiles: &str) -> Real {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let mut output = [0.0];
        BasicGroupCount.calculate(&mol, &mut output);
        output[0]
    }

    #[test]
    fn test_fg_ethanol() {
        let fg = calc_fg("CCO");
        // Hydroxyl should be detected (index 0)
        assert!(fg[0] >= 1.0, "Ethanol should have hydroxyl: {:?}", fg);
    }

    #[test]
    fn test_fg_acetic_acid() {
        let fg = calc_fg("CC(=O)O");
        // CarboxylicAcid (index 3) should be detected
        assert!(fg[3] >= 1.0, "Acetic acid should have COOH: {:?}", fg);
    }

    #[test]
    fn test_fg_acetone() {
        let fg = calc_fg("CC(=O)C");
        // Ketone (index 2) should be detected
        assert!(fg[2] >= 1.0, "Acetone should have ketone: {:?}", fg);
    }

    #[test]
    fn test_fg_methylamine() {
        let fg = calc_fg("CN");
        // PrimaryAmine (index 6)
        assert!(fg[6] >= 1.0, "Methylamine should have primary amine: {:?}", fg);
    }

    #[test]
    fn test_fg_chlorobenzene() {
        let fg = calc_fg("c1ccc(Cl)cc1");
        // Halide (index 15)
        assert!(fg[15] >= 1.0, "Chlorobenzene should have halide: {:?}", fg);
    }

    #[test]
    fn test_fg_benzene_no_groups() {
        let fg = calc_fg("c1ccccc1");
        // Benzene has no functional groups
        let total: Real = fg.iter().sum();
        assert_eq!(total, 0.0, "Benzene should have no functional groups: {:?}", fg);
    }

    #[test]
    fn test_acidic_acetic_acid() {
        assert!(calc_acidic("CC(=O)O") >= 1.0);
    }

    #[test]
    fn test_acidic_phenol() {
        assert!(calc_acidic("c1ccc(O)cc1") >= 1.0);
    }

    #[test]
    fn test_acidic_ethane() {
        assert_eq!(calc_acidic("CC"), 0.0);
    }

    #[test]
    fn test_basic_methylamine() {
        assert!(calc_basic("CN") >= 1.0);
    }

    #[test]
    fn test_basic_dimethylamine() {
        assert!(calc_basic("CNC") >= 1.0);
    }

    #[test]
    fn test_basic_benzene() {
        assert_eq!(calc_basic("c1ccccc1"), 0.0);
    }
}
