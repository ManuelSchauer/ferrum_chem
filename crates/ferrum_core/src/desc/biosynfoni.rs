use crate::algo::iso::find_all_matches;
use crate::io::smarts::{parse_smarts, QueryGraph};
use crate::Molecule;
use once_cell::sync::Lazy;

/// Biosynfoni version `full_1103`
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

/// Lazy-loaded parsed SMARTS graphs. If a SMARTS fails to parse, it logs an error and stores `None`.
pub(crate) struct ScreenedQuery {
    pub query: QueryGraph,
    pub screen: crate::algo::iso::QueryScreen,
}

pub(crate) static BIOSYNFONI_GRAPHS: Lazy<Vec<Option<ScreenedQuery>>> = Lazy::new(|| {
    BIOSYNFONI_FEATURES
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
                let graph = &screened.query;
                let mut unique_sets = std::collections::HashSet::new();
                let mut iters = 0;
                crate::algo::iso::visit_matches_target(&iso_target, graph, |mapping| {
                    iters += 1;
                    let mut sorted_mapping = mapping.to_vec();
                    sorted_mapping.sort_unstable();
                    unique_sets.insert(sorted_mapping);
                    // RDKit default maxMatches is 1000. 
                    // Stop early to prevent combinatorial explosion on highly symmetric targets.
                    iters < 100_000 && unique_sets.len() < 1000
                });
                unique_sets.len()
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
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> crate::desc::typed::DescriptorValue {
        let counts = compute_biosynfoni_fingerprint(mol);
        crate::desc::typed::DescriptorValue::IntVec(counts.into_iter().map(|c| c as u32).collect())
    }
}
