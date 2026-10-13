//! Descriptor module: Bond Count.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

//! ## File guide: Bond Count
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn compute_bond_count`
//! - `fn compute_rotatable_bond_count`
//! - `fn is_strict_terminal_substituent_rotor_blocked`
//! - `fn is_strict_terminal_substituent_center`
//! - `fn compute_conjugated_nitrogen_flags`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Bond Count` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Bond Count` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/bond_count.rs` defines bond count behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Bond Count` workflows.
//! - Repository unit and parity tests covering `Bond Count` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/bond_count.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::BondType;

// ---------------------------------------------------------------------------
// Public computation functions
// ---------------------------------------------------------------------------

/// Compute total bond count.
pub fn compute_bond_count(mol: &Molecule) -> u32 {
    mol.num_bonds() as u32
}

/// Compute rotatable bond count.
///
/// Follows the RDKit `NumRotatableBonds` definition:
/// - Single, non-ring, non-terminal, non-triple-adjacent bonds only.
/// - **Terminal-substituent exclusions**: strict mode excludes bonds attached to
///   quaternary carbon centers matching `C(F)(F)F`, `C(Cl)(Cl)Cl`,
///   `C(Br)(Br)Br`, and `C([CH3])([CH3])[CH3]`.
/// - **Strict conjugation exclusions**: amide-, thioamide-, amidine-, and
///   carbamate-like `C(=X)-N` bonds are non-rotatable when the conjugated carbon
///   has another heavy substituent, while formyl `N-C(=X)H` bonds remain
///   rotatable. Adjacent `N-N` bonds are also excluded when both nitrogens are
///   part of such conjugated systems.
/// - **Carbonyl-O/S exclusion**: ester/thioester C(=O)–O/S bonds are excluded
///   when the carbonyl carbon has another heavy substituent; formates remain
///   rotatable, matching RDKit strict behavior.
///   N-based conjugated bonds are handled by the strict conjugation rules above.
pub fn compute_rotatable_bond_count(mol: &Molecule) -> u32 {
    // RDKit `CalcNumRotatableBonds` with the Strict definition (the default of
    // the reference RDKit build), counted with uniquified matches.
    static STRICT: once_cell::sync::Lazy<crate::io::smarts::QueryGraph> = once_cell::sync::Lazy::new(|| {
        crate::io::smarts::parse_smarts(concat!(
            "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])(",
            "[CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=",
            "[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@[!$",
            "(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([",
            "CH3])[CH3])]"
        ))
        .expect("valid RDKit strict rotatable-bond SMARTS")
    });
    crate::algo::iso::count_matches(mol, &STRICT) as u32
}

/// Computes rotatable bond count cached.
///
/// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub(crate) fn compute_rotatable_bond_count_cached(mol: &Molecule) -> u32 {
    mol.rotatable_bond_count_or_init(|| compute_rotatable_bond_count(mol))
}

// ---------------------------------------------------------------------------
// Typed Descriptors
// ---------------------------------------------------------------------------

/// Total bond count.
///
/// Returns Int(u32) with total number of bonds.
pub struct BondCount;

impl Descriptor for BondCount {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "BondCount"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(compute_bond_count(mol))
    }
}

/// Detailed bond counts.
///
/// Returns IntVec with [total, single, double, triple, aromatic].
pub struct BondCountDetailed;

impl Descriptor for BondCountDetailed {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "BondCountDetailed"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut counts = vec![0u32; 5]; // total, single, double, triple, aromatic

        counts[0] = mol.num_bonds() as u32;

        for bond in &mol.bonds {
            match bond.bond_type {
                BondType::Single => counts[1] += 1,
                BondType::Double => counts[2] += 1,
                BondType::Triple => counts[3] += 1,
                BondType::Aromatic => counts[4] += 1,
                _ => {}
            }
        }

        DescriptorValue::IntVec(counts)
    }
}

/// Rotatable bond count.
///
/// Returns Int(u32) with number of rotatable bonds.
/// Rotatable bonds are single, non-ring, non-terminal bonds.
pub struct RotatableBondCount;

impl Descriptor for RotatableBondCount {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "RotatableBondCount"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(compute_rotatable_bond_count_cached(mol))
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<BondCount>;
    let _ = assert_send_sync::<BondCountDetailed>;
    let _ = assert_send_sync::<RotatableBondCount>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::rings::find_sssr;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for bond count simple.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bond_count_simple() {
        let mut mol = parse_smiles("CCO").unwrap();
        assert_eq!(
            BondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            2
        );
    }

    #[test]
    /// Runs regression test for bond count detailed single.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bond_count_detailed_single() {
        let mut mol = parse_smiles("CCCC").unwrap();
        let v = BondCountDetailed
            .calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new(),
            )
            .as_int_vec()
            .to_vec();
        assert_eq!(v[0], 3); // Total
        assert_eq!(v[1], 3); // Single
        assert_eq!(v[2], 0); // Double
        assert_eq!(v[3], 0); // Triple
    }

    #[test]
    /// Runs regression test for bond count detailed double.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bond_count_detailed_double() {
        let mut mol = parse_smiles("C=C").unwrap();
        let v = BondCountDetailed
            .calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new(),
            )
            .as_int_vec()
            .to_vec();
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 0);
        assert_eq!(v[2], 1);
        assert_eq!(v[3], 0);
    }

    #[test]
    /// Runs regression test for bond count detailed triple.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bond_count_detailed_triple() {
        let mut mol = parse_smiles("C#N").unwrap();
        let v = BondCountDetailed
            .calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new(),
            )
            .as_int_vec()
            .to_vec();
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 0);
        assert_eq!(v[2], 0);
        assert_eq!(v[3], 1);
    }

    #[test]
    /// Runs regression test for bond count detailed mixed.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bond_count_detailed_mixed() {
        let mut mol = parse_smiles("CC=C").unwrap();
        let v = BondCountDetailed
            .calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new(),
            )
            .as_int_vec()
            .to_vec();
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 1);
        assert_eq!(v[2], 1);
        assert_eq!(v[3], 0);
    }

    #[test]
    /// Runs regression test for rotatable bond count ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_ethane() {
        let mut mol = parse_smiles("CC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count propane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_propane() {
        let mut mol = parse_smiles("CCC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count butane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_butane() {
        let mut mol = parse_smiles("CCCC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count ethene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_ethene() {
        let mut mol = parse_smiles("C=C").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_benzene() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        find_sssr(&mut mol);
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count toluene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_toluene() {
        let mut mol = parse_smiles("Cc1ccccc1").unwrap();
        find_sssr(&mut mol);
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count pentane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_pentane() {
        let mut mol = parse_smiles("CCCCC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            2
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_ethanol() {
        let mut mol = parse_smiles("CCO").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count propanol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_propanol() {
        let mut mol = parse_smiles("CCCO").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count secondary amide keeps n alkyl bond.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_secondary_amide_keeps_n_alkyl_bond() {
        // RDKit excludes the acyl C(=O)-N bond itself but still counts the
        // exocyclic N-alkyl bond in a secondary amide.
        let mut mol = parse_smiles("CC(=O)NCC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count ester carbonyl o excluded.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_ester_carbonyl_o_excluded() {
        // Ethyl acetate: C(=O)-O is excluded by carbonyl-O rule, O-CH2 counted.
        let mut mol = parse_smiles("CC(=O)OCC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count methyl acetate zero.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_methyl_acetate_zero() {
        // Methyl acetate: C(=O)-O excluded (carbonyl-O), O-CH3 terminal -> 0.
        let mut mol = parse_smiles("CC(=O)OC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count formyl c n bond is counted.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_formyl_c_n_bond_is_counted() {
        // Formyl C(=O)H-N bonds remain rotatable under RDKit strict semantics.
        let mut mol = parse_smiles("CNC=O").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );

        // Aromatic formyl imidazole-like nitrogens are also counted.
        let mut mol2 = parse_smiles("O=CN1C=CC=N1").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol2,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );

        // Tertiary formamide remains rotatable as well.
        let mut mol2 = parse_smiles("CN(C)C=O").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol2,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count hydrazide nn excluded only when both n are conjugated.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_hydrazide_nn_excluded_only_when_both_n_are_conjugated() {
        let mut mol2 = parse_smiles("CC(=O)NNC(=O)C").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol2,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );

        let mut mol3 = parse_smiles("CC(=O)NNC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol3,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count acetanilide counts n aryl bond.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_acetanilide_counts_n_aryl_bond() {
        let mut mol = parse_smiles("CC(=O)Nc1ccccc1").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count amidine c n bond excluded.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_amidine_c_n_bond_excluded() {
        let mut mol = parse_smiles("CCNC(=N)N").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count formate c o bond is counted.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_formate_c_o_bond_is_counted() {
        let mut mol = parse_smiles("COC=O").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );

        let mut mol2 = parse_smiles("CCOC=O").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol2,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            2
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count aromatic acyl n excluded but formyl kept.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_aromatic_acyl_n_excluded_but_formyl_kept() {
        let mut mol = parse_smiles("CC(=O)N1C=CC=N1").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );

        let mut mol2 = parse_smiles("O=CN1C=CC=N1").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol2,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count tert butyl like linkage excluded.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_tert_butyl_like_linkage_excluded() {
        let mut mol = parse_smiles("CC(C)(C)CC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );

        let mut mol2 = parse_smiles("CC(C)(C)OC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol2,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count trihaloalkyl linkage excluded.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_trihaloalkyl_linkage_excluded() {
        let mut mol = parse_smiles("ClC(Cl)(Cl)CC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );

        let mut mol2 = parse_smiles("FC(F)(F)CC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol2,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );

        let mut mol3 = parse_smiles("BrC(Br)(Br)CC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol3,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            0
        );

        // RDKit strict pattern excludes F/Cl/Br but intentionally not I.
        let mut mol4 = parse_smiles("IC(I)(I)CC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol4,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count cnp0329403 matches rdkit.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_cnp0329403_matches_rdkit() {
        let mut mol =
            parse_smiles("O=C(N[C@@H](OC[C@@H]1CCCN2CCCC[C@H]12)C(Cl)(Cl)Cl)C1=CC=CC2=CC=CC=C12")
                .unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            5
        );
    }

    #[test]
    /// Runs regression test for rotatable bond count non tbutyl quaternary linkage kept.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_rotatable_bond_count_non_tbutyl_quaternary_linkage_kept() {
        let mut mol = parse_smiles("CCC(C)(C)CC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            2
        );

        let mut mol2 = parse_smiles("CC(C)(O)CC").unwrap();
        assert_eq!(
            RotatableBondCount
                .calculate(
                    &mut mol2,
                    &mut crate::desc::context::CalculationContext::new()
                )
                .as_int(),
            1
        );
    }

    #[test]
    /// Runs regression test for compute bond count.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_compute_bond_count() {
        let mol = parse_smiles("CCO").unwrap();
        assert_eq!(compute_bond_count(&mol), 2);
    }

    #[test]
    /// Runs regression test for compute rotatable bond count.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bond_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_compute_rotatable_bond_count() {
        let mol = parse_smiles("CCCC").unwrap();
        assert_eq!(compute_rotatable_bond_count(&mol), 1);
    }
}
