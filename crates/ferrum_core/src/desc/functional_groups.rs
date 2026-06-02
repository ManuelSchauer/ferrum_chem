//! Functional group counting descriptors using SMARTS pattern matching.
//!
//! Provides counts for common functional groups, acidic groups, and basic groups.

//! ## File guide: Functional Groups
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct FuncGroupPattern`
//! - `struct FunctionalGroupCounts`
//! - `fn name`
//! - `fn calculate`
//! - `struct AcidicGroupCount`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Functional Groups` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Functional Groups` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/functional_groups.rs` defines functional groups behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Functional Groups` workflows.
//! - Repository unit and parity tests covering `Functional Groups` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/functional_groups.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::iso::count_matches as smarts_count_matches;
use crate::io::smarts::{QueryGraph, parse_smarts};
use crate::mol::molecule::Molecule;
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
fg_patterns!(
    FUNC_GROUP_PATTERNS,
    ("Hydroxyl", "[OX2H]"),
    ("Aldehyde", "[CX3H1](=O)[#6]"),
    ("Ketone", "[#6][CX3](=O)[#6]"),
    ("CarboxylicAcid", "[CX3](=O)[OX2H1]"),
    ("Ester", "[#6][CX3](=O)[OX2H0][#6]"),
    ("Amide", "[NX3][CX3](=O)[#6]"),
    ("PrimaryAmine", "[NX3;H2;!$(NC=O)]"),
    ("SecondaryAmine", "[NX3;H1;!$(NC=O)]"),
    ("TertiaryAmine", "[NX3;H0;!$(NC=O);!$([NX3](=O)=O)]"),
    ("Nitro", "[$([NX3](=O)=O)]"),
    ("Sulfonamide", "[#16X4]([NX3])(=O)=O"),
    ("Sulfone", "[#16X4](=O)(=O)([#6])[#6]"),
    ("Thiol", "[#16X2H]"),
    ("Ether", "[OD2]([#6])[#6]"),
    ("Nitrile", "[NX1]#[CX2]"),
    ("Halide", "[F,Cl,Br,I]"),
);

// Acidic group patterns
fg_patterns!(
    ACIDIC_PATTERNS,
    ("CarboxylicAcid", "[CX3](=O)[OX2H1]"),
    ("Sulfonamide_NH", "[#16X4](=O)(=O)[NX3H]"),
    ("SulfonicAcid", "[#16X4](=O)(=O)[OX2H]"),
    ("Phenol", "[OX2H]c"),
    ("Thiol", "[#16X2H]"),
);

// Basic group patterns
fg_patterns!(
    BASIC_PATTERNS,
    ("PrimaryAmine", "[NX3;H2;!$(NC=O)]"),
    ("SecondaryAmine", "[NX3;H1;!$(NC=O)]"),
    ("TertiaryAmine", "[NX3;H0;!$(NC=O);!$([NX3](=O)=O)]"),
    ("Guanidine", "[NX3][CX3](=[NX2])[NX3]"),
    ("Amidine", "[NX3][CX3]=[NX2]"),
);

/// Functional Group Counts descriptor.
///
/// Counts 16 common functional groups using SMARTS patterns.
/// Output order: Hydroxyl, Aldehyde, Ketone, CarboxylicAcid, Ester, Amide,
/// PrimaryAmine, SecondaryAmine, TertiaryAmine, Nitro, Sulfonamide, Sulfone,
/// Thiol, Ether, Nitrile, Halide.
pub struct FunctionalGroupCounts;

impl Descriptor for FunctionalGroupCounts {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "FunctionalGroupCounts"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let counts: Vec<u16> = FUNC_GROUP_PATTERNS
            .iter()
            .map(|p| smarts_count_matches(mol, &p.query) as u16)
            .collect();
        DescriptorValue::IntU16Vec(counts)
    }
}

/// Acidic Group Count: total number of acidic functional groups.
///
/// Counts carboxylic acids, sulfonamide NHs, sulfonic acids, phenols, thiols.
pub struct AcidicGroupCount;

impl Descriptor for AcidicGroupCount {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "AcidicGroupCount"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let total: usize = ACIDIC_PATTERNS
            .iter()
            .map(|p| smarts_count_matches(mol, &p.query))
            .sum();
        DescriptorValue::Int(total as u32)
    }
}

/// Basic Group Count: total number of basic functional groups.
///
/// Counts primary/secondary/tertiary amines, guanidines, amidines.
pub struct BasicGroupCount;

impl Descriptor for BasicGroupCount {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "BasicGroupCount"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let total: usize = BASIC_PATTERNS
            .iter()
            .map(|p| smarts_count_matches(mol, &p.query))
            .sum();
        DescriptorValue::Int(total as u32)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<FunctionalGroupCounts>;
    let _ = assert_send_sync::<AcidicGroupCount>;
    let _ = assert_send_sync::<BasicGroupCount>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements fg.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_fg(smiles: &str) -> Vec<u16> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match FunctionalGroupCounts.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::IntU16Vec(v) => v,
            other => panic!("expected IntU16Vec, got {:?}", other),
        }
    }

    /// Implements acidic.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_acidic(smiles: &str) -> u32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match AcidicGroupCount.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::Int(v) => v,
            other => panic!("expected Int, got {:?}", other),
        }
    }

    /// Implements basic.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_basic(smiles: &str) -> u32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match BasicGroupCount.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::Int(v) => v,
            other => panic!("expected Int, got {:?}", other),
        }
    }

    #[test]
    /// Runs regression test for fg ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fg_ethanol() {
        let fg = calc_fg("CCO");
        assert!(fg[0] >= 1, "Ethanol should have hydroxyl: {:?}", fg);
    }

    #[test]
    /// Runs regression test for fg acetic acid.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fg_acetic_acid() {
        let fg = calc_fg("CC(=O)O");
        assert!(fg[3] >= 1, "Acetic acid should have COOH: {:?}", fg);
    }

    #[test]
    /// Runs regression test for fg acetone.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fg_acetone() {
        let fg = calc_fg("CC(=O)C");
        assert!(fg[2] >= 1, "Acetone should have ketone: {:?}", fg);
    }

    #[test]
    /// Runs regression test for fg methylamine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fg_methylamine() {
        let fg = calc_fg("CN");
        assert!(
            fg[6] >= 1,
            "Methylamine should have primary amine: {:?}",
            fg
        );
    }

    #[test]
    /// Runs regression test for fg chlorobenzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fg_chlorobenzene() {
        let fg = calc_fg("c1ccc(Cl)cc1");
        assert!(fg[15] >= 1, "Chlorobenzene should have halide: {:?}", fg);
    }

    #[test]
    /// Runs regression test for fg benzene no groups.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fg_benzene_no_groups() {
        let fg = calc_fg("c1ccccc1");
        let total: u16 = fg.iter().sum();
        assert_eq!(
            total, 0,
            "Benzene should have no functional groups: {:?}",
            fg
        );
    }

    #[test]
    /// Runs regression test for acidic acetic acid.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_acidic_acetic_acid() {
        assert!(calc_acidic("CC(=O)O") >= 1);
    }

    #[test]
    /// Runs regression test for acidic phenol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_acidic_phenol() {
        assert!(calc_acidic("c1ccc(O)cc1") >= 1);
    }

    #[test]
    /// Runs regression test for acidic ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_acidic_ethane() {
        assert_eq!(calc_acidic("CC"), 0);
    }

    #[test]
    /// Runs regression test for basic methylamine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_basic_methylamine() {
        assert!(calc_basic("CN") >= 1);
    }

    #[test]
    /// Runs regression test for basic dimethylamine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_basic_dimethylamine() {
        assert!(calc_basic("CNC") >= 1);
    }

    #[test]
    /// Runs regression test for basic benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_basic_benzene() {
        assert_eq!(calc_basic("c1ccccc1"), 0);
    }
}
