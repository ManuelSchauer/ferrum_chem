use crate::io::smarts::{parse_smarts, QueryGraph};
use crate::Molecule;
use once_cell::sync::Lazy;

/// Biosynfoni fingerprint (39 features, version `full_1103`)
pub const BIOSYNFONI_FEATURES: &[(&str, &str, &str)] = &[
    ("co_coa", "coenzyme_a", "SCCN~C(~O)CCN~C(~O)C(C(C)(C)COP(O)(~O)OP(~O)(O)OCC1C(C(C(O1)[#7]2~[#6]~[#7]~[#6]~3~[#6](~[#7]~[#6]~[#7]~[#6]~3~2)~[#7])O)OP(~O)(O)O)~O"),
    ("co_nadh", "nadh", "[#6]~1~[#6]~[#6]~[#7](~[#6]~[#6]~1~[#6](~O)~[#7])~[#6]~2~[#6](~[#6](~[#6](~O~2)~[#6]~O~P(~O)(~O)~O~P(~O)(~O)~O~[#6]~[#6]~3~[#6](~[#6](~[#6](~O~3)~[#7]~4~[#6]~[#7]~[#6]~5~[#6](~[#7]~[#6]~[#7]~[#6]~5~4)~[#7])~O)~O)~O)~O"),
    ("co_nadph", "nadph", "[#6]~1~[#6]~[#6]~[#7](~[#6]~[#6]~1~[#6](~O)~[#7])~[#6]~2~[#6](~[#6](~[#6](~O~2)~[#6]~O~P(~O)(~O)~O~P(~O)(~O)~O~[#6]~[#6]~3~[#6](~[#6](~[#6](~O~3)~[#7]~4~[#6]~[#7]~[#6]~5~[#6](~[#7]~[#6]~[#7]~[#6]~5~4)~[#7])~O~P(~O)(~O)~O)~O)~O)~O"),
    ("allstnd_aminos", "standard_amino_acids", "[$([$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]),$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N]),$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([$([CH3X4]),$([CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]),$([CH2X4][CX3](=[OX1])[NX3H2]),$([CH2X4][CX3](=[OX1])[OH0-,OH]),$([CH2X4][SX2H,SX1H0-]),$([CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]),$([CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1),$([CHX4]([CH3X4])[CH2X4][CH3X4]),$([CH2X4][CHX4]([CH3X4])[CH3X4]),$([CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]),$([CH2X4][CH2X4][SX2][CH3X4]),$([CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1),$([CH2X4][OX2H]),$([CHX4]([CH3X4])[OX2H]),$([CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12),$([CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1),$([CHX4]([CH3X4])[CH3X4])])[CX3](=[OX1])[OX2H,OX1-,N])]"),
    ("nonstnd_aminos", "non-standard_amino_acids", "[$([NX3,NX4+][CX4H]([$([CH3X4]),$([CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]),$([CH2X4][CX3](=[OX1])[NX3H2]),$([CH2X4][CX3](=[OX1])[OH0-,OH]),$([CH2X4][SX2H,SX1H0-]),$([CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]),$([CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1),$([CHX4]([CH3X4])[CH2X4][CH3X4]),$([CH2X4][CHX4]([CH3X4])[CH3X4]),$([CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]),$([CH2X4][CH2X4][SX2][CH3X4]),$([CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1),$([CH2X4][OX2H]),$([CHX4]([CH3X4])[OX2H]),$([CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12),$([CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1),$([CHX4]([CH3X4])[CH3X4])])[CX3](=[OX1])[O,N]);!$([$([$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]),$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N]),$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([$([CH3X4]),$([CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]),$([CH2X4][CX3](=[OX1])[NX3H2]),$([CH2X4][CX3](=[OX1])[OH0-,OH]),$([CH2X4][SX2H,SX1H0-]),$([CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]),$([CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1),$([CHX4]([CH3X4])[CH2X4][CH3X4]),$([CH2X4][CHX4]([CH3X4])[CH3X4]),$([CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]),$([CH2X4][CH2X4][SX2][CH3X4]),$([CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1),$([CH2X4][OX2H]),$([CHX4]([CH3X4])[OX2H]),$([CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12),$([CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1),$([CHX4]([CH3X4])[CH3X4])])[CX3](=[OX1])[OX2H,OX1-,N])])]"),
    ("s_openpyr_C6O6", "open_pyranose", "C(~[#8])~C(~[#8])~C(~[#8])~C(~[#8])~C(~[#8])~C(~[#8])"),
    ("s_openfur_C5O5", "open_furanose", "C(~[#8])~C(~[#8])~C(~[#8])~C(~[#8])~C(~[#8])"),
    ("s_pyranose_C5O4", "pyranose", "C~1~[#8]~C~C(~[#8])~C(~[#8])~C(~[#8])~1"),
    ("s_furanose_C4O3", "furanose", "C~1~[#8]~C~C(~[#8])~C(~[#8])~1"),
    ("d_indoleC2N_12", "indoleC2N", "c1cccc2c1c(~[#6]~[#6]~[#7])cn2"),
    ("d_phenylC2N_9", "phenylC2N", "c1ccccc1[#6][#6][#7]"),
    ("d_c5n_6", "C5N", "[#6]~1~[#6]~[#6]~[#6]~[#6]~[#7]~1"),
    ("d_c4n_5", "C4N", "[#6]~1~[#6]~[#6]~[#6]~[#7]~1"),
    ("d_phenylC3_9_strict", "phenylC3", "[#6;R1]~1~[#6;R1]~[#6;R1]~[#6;R1]~[#6;R1]~[#6;R1]~1~[#6]~[#6;!$([r6])]~[#6;!$([r6])]"),
    ("d_phenylC2_8_strict", "phenylC2", "[#6;R1]~1~[#6;R1]~[#6;R1]~[#6;R1]~[#6;R1]~[#6;R1]~1~[#6]~[#6;!$([r6])]"),
    ("d_phenylC1_7_strict", "phenylC1", "[#6;R1]~1~[#6;R1]~[#6;R1]~[#6;R1]~[#6;R1]~[#6;R1]~1~[#6]"),
    ("d_isoprene_5", "isoprene", "[#6]~[#6](~[#6])~[#6]~[#6]"),
    ("d2_acetyl_C2O1", "acetyl", "[#6]~[#6]~[#8]"),
    ("d2_methylmalonyl_C3", "methylmalonyl", "[#6]~[#6][C;D1;h3]"),
    ("d_ethyl_2", "ethyl", "[#6]~[#6]"),
    ("d_methyl_1", "methyl", "[C;D1;h3]"),
    ("phosphate_2", "phosphate", "P~O"),
    ("sulfonate_2", "sulfonate", "S~O"),
    ("hal_f", "fluorine", "[#9]"),
    ("hal_cl", "chlorine", "[#17]"),
    ("hal_br", "bromine", "[#35]"),
    ("hal_i", "iodine", "[#53]"),
    ("n_nitrate_1", "nitrate", "[N;D1]"),
    ("o_epoxy_1", "epoxy", "[O;x2;r3]"),
    ("o_ether_1", "ether", "[O;D2;!h;!$(*C=O);X2;!R;!$(*P);!$(*S)]"),
    ("o_hydroxyl_1", "hydroxyl", "[#8;D1;h,!v2;$(*[#6,#7]);!$(*C~O);!$(P);!$(S)]"),
    ("r_c3", "C3_ring", "[#6]~1~[#6]~[#6]~1"),
    ("r_c4", "C4_ring", "[#6]~1~[#6]~[#6]~[#6]~1"),
    ("r_c5", "C5_ring", "[#6]~1~[#6]~[#6]~[#6]~[#6]~1"),
    ("r_c6", "C6_ring", "[#6]~1~[#6]~[#6]~[#6]~[#6]~[#6]~1"),
    ("r_c7", "C7_ring", "[#6]~1~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~1"),
    ("r_c8", "C8_ring", "[#6]~1~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~1"),
    ("r_c9", "C9_ring", "[#6]~1~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~1"),
    ("r_c10", "C10_ring", "[#6]~1~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~1"),
];

// Correct the typo if any in SMARTS string
static CLEANED_FEATURES: Lazy<Vec<(&'static str, &'static str, String)>> = Lazy::new(|| {
    BIOSYNFONI_FEATURES
        .iter()
        .map(|&(k, n, s)| {
            let s_clean = s.replace(" timescale=1", "");
            (k, n, s_clean)
        })
        .collect()
});

/// Lazy-loaded parsed SMARTS graphs. If a SMARTS fails to parse, it logs an error and stores `None`.
pub(crate) struct ScreenedQuery {
    pub query: QueryGraph,
    pub screen: crate::algo::iso::QueryScreen,
}

pub(crate) static BIOSYNFONI_GRAPHS: Lazy<Vec<Option<ScreenedQuery>>> = Lazy::new(|| {
    CLEANED_FEATURES
        .iter()
        .map(|(key, _, smarts)| {
            match parse_smarts(smarts) {
                Ok(graph) => Some(ScreenedQuery {
                    screen: crate::algo::iso::QueryScreen::build(&graph),
                    query: graph,
                }),
                Err(e) => {
                    eprintln!("Failed to parse Biosynfoni SMARTS {}: {:?}", key, e);
                    None
                }
            }
        })
        .collect()
});

/// Computes the Biosynfoni fingerprint for a given molecule.
/// Returns a vector of counts for each of the 39 SMARTS substructures.
pub fn compute_biosynfoni_fingerprint(mol: &Molecule) -> Vec<usize> {
    let iso_target = crate::algo::iso::IsoTarget::new(mol);
    BIOSYNFONI_GRAPHS
        .iter()
        .map(|graph_opt| {
            if let Some(screened) = graph_opt {
                if !screened.screen.passes_target(&iso_target) {
                    return 0;
                }
                // RDKit default maxMatches is 1000.
                // Stop early to prevent combinatorial explosion on highly symmetric targets.
                crate::algo::iso::count_unique_matches_capped(&iso_target, &screened.query, 1000, 100_000)
            } else {
                0
            }
        })
        .collect()
}

pub fn get_biosynfoni_feature_names() -> Vec<String> {
    BIOSYNFONI_FEATURES
        .iter()
        .map(|(_, name, _)| name.to_string())
        .collect()
}

pub fn get_biosynfoni_feature_keys() -> Vec<String> {
    BIOSYNFONI_FEATURES
        .iter()
        .map(|(key, _, _)| key.to_string())
        .collect()
}

/// Implements Biosynfoni descriptor.
///
/// Descriptor-layer note: computes molecular properties for ML pipelines.
pub struct Biosynfoni;

impl crate::desc::typed::Descriptor for Biosynfoni {
    fn name(&self) -> &str {
        "Biosynfoni"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> crate::desc::typed::DescriptorValue {
        let counts = compute_biosynfoni_fingerprint(mol);
        crate::desc::typed::DescriptorValue::IntVec(counts.into_iter().map(|c| c as u32).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;
    use crate::algo::rings::find_sssr;

    fn get_biosynfoni(smiles: &str) -> Vec<usize> {
        let mut mol = parse_smiles(smiles).unwrap();
        find_sssr(&mut mol);
        compute_biosynfoni_fingerprint(&mol)
    }

    fn idx_of(key: &str) -> usize {
        BIOSYNFONI_FEATURES
            .iter()
            .position(|&(k, _, _)| k == key)
            .unwrap_or_else(|| panic!("Key {} not found in BIOSYNFONI_FEATURES", key))
    }

    #[test]
    fn test_biosynfoni_length() {
        assert_eq!(BIOSYNFONI_FEATURES.len(), 39, "Biosynfoni must have exactly 39 features");
        let fp = get_biosynfoni("CC");
        assert_eq!(fp.len(), 39, "Fingerprint vector must have exactly 39 elements");
    }

    #[test]
    fn test_biosynfoni_indole() {
        let fp = get_biosynfoni("C1=CC=C2C(=C1)C(=CN2)CCN"); // Tryptamine
        let indole_idx = idx_of("d_indoleC2N_12");
        assert_eq!(indole_idx, 9);
        assert_eq!(fp[indole_idx], 1, "Tryptamine contains exactly 1 indoleC2N substructure");
    }

    #[test]
    fn test_biosynfoni_methyl() {
        let fp = get_biosynfoni("CC"); // Ethane
        let methyl_idx = idx_of("d_methyl_1");
        assert_eq!(methyl_idx, 20);
        assert_eq!(fp[methyl_idx], 2, "Ethane contains exactly 2 methyl groups");
    }

    #[test]
    fn test_biosynfoni_halogens() {
        let fp = get_biosynfoni("FC(F)(F)c1ccc(Cl)c(Br)c1I");
        let f_idx = idx_of("hal_f");
        let cl_idx = idx_of("hal_cl");
        let br_idx = idx_of("hal_br");
        let i_idx = idx_of("hal_i");

        assert_eq!(f_idx, 23);
        assert_eq!(cl_idx, 24);
        assert_eq!(br_idx, 25);
        assert_eq!(i_idx, 26);

        assert_eq!(fp[f_idx], 3, "Expected 3 fluorines");
        assert_eq!(fp[cl_idx], 1, "Expected 1 chlorine");
        assert_eq!(fp[br_idx], 1, "Expected 1 bromine");
        assert_eq!(fp[i_idx], 1, "Expected 1 iodine");
    }

    #[test]
    fn all_biosynfoni_patterns_parse() {
        let failed: Vec<_> = CLEANED_FEATURES
            .iter()
            .filter(|(_, _, smarts)| crate::io::smarts::parse_smarts(smarts).is_err())
            .map(|(key, _, _)| *key)
            .collect();
        assert!(failed.is_empty(), "unparseable Biosynfoni SMARTS: {:?}", failed);
    }
}
