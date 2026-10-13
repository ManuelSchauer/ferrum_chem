//! Mapping from Mordred (mordred-community) descriptor names to FerrumChem
//! column names.
//!
//! FerrumChem's columns come from several reference toolkits (RDKit, CDK,
//! Mordred, Biosynfoni) that partly use the same names for different
//! definitions. The rule is:
//!
//! 1. A Mordred descriptor whose name is not used by any other toolkit keeps
//!    its Mordred name (`ATS0pe`, `Radius`, ...).
//! 2. A Mordred descriptor that is *exactly* an existing FerrumChem column
//!    under another name is not duplicated; it is listed in [`MORDRED_ALIASES`]
//!    (e.g. Mordred `SLogP` is RDKit's `MolLogP`, both call RDKit's Crippen
//!    code). Only definitions that are identical by construction are listed
//!    here, never columns that merely agree on the test set.
//! 3. A Mordred descriptor whose name is already taken by an RDKit or CDK
//!    column with a *different* definition (e.g. CDK `apol` vs Mordred
//!    `apol`) gets the column name `Mordred_<name>`; these names are listed in
//!    [`MORDRED_PREFIXED`]. The unprefixed column keeps the RDKit/CDK value.
//!
//! The Mordred parity test (`tests/mordred_parity.rs`) resolves every column of
//! the Mordred reference through [`ferrum_column`].

/// Prefix of FerrumChem columns that hold a Mordred variant of a name already
/// used by RDKit or CDK.
pub const MORDRED_PREFIX: &str = "Mordred_";

/// `(Mordred name, FerrumChem column)` for Mordred descriptors that are
/// identical by definition to a FerrumChem column with another name.
pub const MORDRED_ALIASES: &[(&str, &str)] = &[
    // Constitutional / counts
    ("nAromBond", "nBondsA"),
    ("nBondsO", "nBondsHeavy"),
    ("nHetero", "NumHeteroatoms"),
    ("nSpiro", "NumSpiroAtoms"),
    ("nBridgehead", "NumBridgeheadAtoms"),
    ("FCSP3", "FractionCSP3"),
    // Mordred calls RDKit for these
    ("MW", "ExactMolWt"),
    ("SLogP", "MolLogP"),
    ("SMR", "MolMR"),
    ("TopoPSA(NO)", "TPSA"),
    ("nHBAcc", "NumHAcceptors"),
    ("nHBDon", "NumHDonors"),
    ("nRot", "NumRotatableBonds"),
    // RDKit CalcNumRings is RingInfo::numRings of the sanitized molecule, i.e.
    // the symmetrized SSSR that Mordred's Chem.GetSymmSSSR returns
    ("nRing", "RingCount"),
];

/// Mordred names that clash with an RDKit or CDK column of a different
/// definition; the Mordred variant is the column `Mordred_<name>`.
pub const MORDRED_PREFIXED: &[&str] = &[
    // atom / bond counts (CDK AtomCount, BondCount)
    "nH",
    "nHeavyAtom",
    "nBonds",
    "nBondsS",
    // carbon types (CDK CarbonTypes: CDK atom typing vs RDKit hybridization,
    // differ e.g. for carbanions)
    "C1SP1",
    "C2SP1",
    "C1SP2",
    "C2SP2",
    "C3SP2",
    "C1SP3",
    "C2SP3",
    "C3SP3",
    "C4SP3",
    "HybRatio",
    // CDK BCUT
    "BCUTc-1h",
    "BCUTc-1l",
    "BCUTp-1h",
    "BCUTp-1l",
    // RDKit BalabanJ
    "BalabanJ",
    // CDK Kappa shape indices
    "Kier1",
    "Kier2",
    "Kier3",
    // CDK molecular distance edge
    "MDEC-11",
    "MDEC-12",
    "MDEC-13",
    "MDEC-14",
    "MDEC-22",
    "MDEC-23",
    "MDEC-24",
    "MDEC-33",
    "MDEC-34",
    "MDEC-44",
    "MDEO-11",
    "MDEO-12",
    "MDEO-22",
    "MDEN-11",
    "MDEN-12",
    "MDEN-13",
    "MDEN-22",
    "MDEN-23",
    "MDEN-33",
    // CDK polarizabilities, VAdjMa, Zagreb
    "apol",
    "bpol",
    "VAdjMat",
    "Zagreb1",
];

/// FerrumChem column that holds the Mordred descriptor `mordred_name`.
pub fn ferrum_column(mordred_name: &str) -> String {
    if let Some(&(_, col)) = MORDRED_ALIASES.iter().find(|(m, _)| *m == mordred_name) {
        return col.to_string();
    }
    if MORDRED_PREFIXED.contains(&mordred_name) {
        return format!("{MORDRED_PREFIX}{mordred_name}");
    }
    mordred_name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_aliases_prefixes_and_plain_names() {
        assert_eq!(ferrum_column("SLogP"), "MolLogP");
        assert_eq!(ferrum_column("apol"), "Mordred_apol");
        assert_eq!(ferrum_column("ATS0pe"), "ATS0pe");
    }

    #[test]
    fn tables_do_not_overlap() {
        for (m, _) in MORDRED_ALIASES {
            assert!(!MORDRED_PREFIXED.contains(m), "{m} is both alias and prefixed");
        }
    }
}
