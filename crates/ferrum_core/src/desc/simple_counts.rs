//! Descriptor module: Simple Counts.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

//! ## File guide: Simple Counts
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct ValenceElectrons`
//! - `fn name`
//! - `fn calculate`
//! - `struct RadicalElectrons`
//! - `fn name`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Simple Counts` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Simple Counts` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/simple_counts.rs` defines simple counts behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Simple Counts` workflows.
//! - Repository unit and parity tests covering `Simple Counts` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/simple_counts.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::{elements, molecule::Molecule};

/// Valence Electrons: Total number of valence electrons in the molecule.
///
/// Calculates the sum of valence electrons for all atoms including hydrogens.
/// Formula: Σ (outer_electrons - formal_charge + total_H_count)
///
/// # Examples
/// - Ethane (C₂H₆): 14 valence electrons
/// - CO₂: 16 valence electrons
/// - Acetate ion: 18 valence electrons
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct ValenceElectrons;

impl Descriptor for ValenceElectrons {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "ValenceElectrons"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut count: i32 = 0;
        for atom in &mol.atoms {
            let outer_elec = elements::outer_electrons(atom.atomic_number) as i32;
            let formal_charge = atom.formal_charge as i32;
            let total_h = atom.total_h() as i32;
            count += outer_elec - formal_charge + total_h;
        }
        DescriptorValue::Float(count as f64)
    }
}

/// Radical Electrons: Total number of unpaired radical electrons.
///
/// Counts atoms with unpaired electrons (radicals).
/// Says nothing about spin state.
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct RadicalElectrons;

impl Descriptor for RadicalElectrons {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "RadicalElectrons"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let count: u32 = mol
            .atoms
            .iter()
            .map(|atom| atom.num_radicals() as u32)
            .sum();
        DescriptorValue::Int(count)
    }
}

/// Heteroatom Count: Number of non-carbon, non-hydrogen atoms.
///
/// Counts atoms that are neither carbon nor hydrogen.
/// Useful for characterizing heteroatom-rich molecules.
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct HeteroatomCount;

impl Descriptor for HeteroatomCount {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "HeteroatomCount"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let count = mol
            .atoms
            .iter()
            .filter(|atom| atom.atomic_number > 1 && atom.atomic_number != 6)
            .count() as u32;
        DescriptorValue::Int(count)
    }
}

/// Hybridization ratio (CDK `HybridizationRatioDescriptor`, `HybRatio`):
/// nsp3 / (nsp2 + nsp3) over carbon atoms, with the hybridization of CDK's
/// carbon atom types (see [`super::carbon_types::cdk_carbon_hybridization`]).
/// NaN when the molecule has no sp2 or sp3 carbon (CDK reports an error).
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct HybridizationRatio;

impl Descriptor for HybridizationRatio {
    fn name(&self) -> &str {
        "HybridizationRatio"
    }

    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        use super::carbon_types::{cdk_carbon_hybridization, CdkCarbonHybridization};
        let mut nsp2 = 0u32;
        let mut nsp3 = 0u32;
        for a in 0..mol.num_atoms() {
            if mol.atoms[a].atomic_number != 6 {
                continue;
            }
            match cdk_carbon_hybridization(mol, a) {
                Some(CdkCarbonHybridization::Sp2) => nsp2 += 1,
                Some(CdkCarbonHybridization::Sp3) => nsp3 += 1,
                _ => {}
            }
        }
        let ratio = if nsp2 + nsp3 == 0 { f64::NAN } else { nsp3 as f64 / (nsp2 + nsp3) as f64 };
        DescriptorValue::Float(ratio)
    }
}

/// VAdjMa: Vertex Adjacency Magnitude.
///
/// CDK `VAdjMaDescriptor` (`VAdjMat`): 1 + log₂(m) where m is the number of
/// bonds between non-hydrogen atoms (0 without such bonds).
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct VAdjMa;

impl Descriptor for VAdjMa {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "VAdjMa"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let m = mol
            .bonds
            .iter()
            .filter(|b| mol.atoms[b.source as usize].atomic_number != 1 && mol.atoms[b.target as usize].atomic_number != 1)
            .count();
        let val = if m == 0 { 0.0 } else { (m as f64).ln() / 2f64.ln() + 1.0 };
        DescriptorValue::Float(val)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ValenceElectrons>;
    let _ = assert_send_sync::<RadicalElectrons>;
    let _ = assert_send_sync::<HeteroatomCount>;
    let _ = assert_send_sync::<HybridizationRatio>;
    let _ = assert_send_sync::<VAdjMa>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements float.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_float(desc: &dyn Descriptor, smiles: &str) -> f64 {
        let mut mol = parse_smiles(smiles).expect("Invalid SMILES");
        desc.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        )
        .as_float()
    }

    /// Implements int.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_int(desc: &dyn Descriptor, smiles: &str) -> u32 {
        let mut mol = parse_smiles(smiles).expect("Invalid SMILES");
        desc.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        )
        .as_int()
    }

    // Valence Electrons Tests
    #[test]
    /// Runs regression test for valence electrons ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_valence_electrons_ethane() {
        assert_eq!(calc_float(&ValenceElectrons, "CC"), 14.0);
    }

    #[test]
    /// Runs regression test for valence electrons co2.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_valence_electrons_co2() {
        let val = calc_float(&ValenceElectrons, "O=C=O");
        assert!(
            (val - 16.0).abs() < 1.0,
            "CO2 should have ~16 valence electrons"
        );
    }

    // Radical Electrons Tests
    #[test]
    /// Runs regression test for radical electrons ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_radical_electrons_ethane() {
        assert_eq!(calc_int(&RadicalElectrons, "CC"), 0);
    }

    // Heteroatom Tests
    #[test]
    /// Runs regression test for heteroatom benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_heteroatom_benzene() {
        assert_eq!(calc_int(&HeteroatomCount, "c1ccccc1"), 0);
    }

    #[test]
    /// Runs regression test for heteroatom pyridine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_heteroatom_pyridine() {
        assert_eq!(calc_int(&HeteroatomCount, "c1ccncc1"), 1);
    }

    #[test]
    /// Runs regression test for heteroatom caffeine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_heteroatom_caffeine() {
        assert_eq!(
            calc_int(&HeteroatomCount, "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
            6
        );
    }

    // Hybridization Ratio Tests
    #[test]
    /// Runs regression test for hybridization cyclohexane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hybridization_cyclohexane() {
        assert_eq!(calc_float(&HybridizationRatio, "C1CCCCC1"), 1.0);
        // no sp2/sp3 carbon: CDK reports an error (NaN)
        assert!(calc_float(&HybridizationRatio, "O").is_nan());
        assert!(calc_float(&HybridizationRatio, "C#C").is_nan());
    }

    #[test]
    /// Runs regression test for hybridization benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hybridization_benzene() {
        assert_eq!(calc_float(&HybridizationRatio, "c1ccccc1"), 0.0);
    }

    #[test]
    /// Runs regression test for hybridization mixed.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hybridization_mixed() {
        let ratio = calc_float(&HybridizationRatio, "CCc1ccccc1");
        assert!(
            (ratio - 2.0 / 8.0).abs() < 1e-6,
            "Ethylbenzene: 2 sp3 of 8 carbons"
        );
    }

    // VAdjMa Tests
    #[test]
    /// Runs regression test for vadjma methane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_vadjma_methane() {
        assert_eq!(calc_float(&VAdjMa, "C"), 0.0);
    }

    #[test]
    /// Runs regression test for vadjma ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_vadjma_ethane() {
        assert_eq!(calc_float(&VAdjMa, "CC"), 1.0);
    }

    #[test]
    /// Runs regression test for vadjma benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/simple_counts.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_vadjma_benzene() {
        let val = calc_float(&VAdjMa, "c1ccccc1");
        assert!((val - 3.585).abs() < 0.01, "Benzene VAdjMa should be ~3.58");
    }
}
