//! Descriptor module: Hbond.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

//! ## File guide: Hbond
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn compute_hbond_donors`
//! - `fn compute_hbond_acceptors`
//! - `struct HBondDonor`
//! - `fn name`
//! - `fn calculate`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Hbond` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Hbond` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/hbond.rs` defines hbond behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Hbond` workflows.
//! - Repository unit and parity tests covering `Hbond` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/hbond.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Compute the number of hydrogen bond donors (Lipinski definition).
///
/// Matches: N with H (v3 or v4+), O/S with exactly 1H, aromatic nH.
/// SMARTS: [$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]
pub fn compute_hbond_donors(mol: &Molecule) -> u32 {
    use crate::algo::iso::count_matches;
    use crate::io::smarts::parse_smarts;
    use once_cell::sync::Lazy;

    static HBD_SMARTS: Lazy<crate::io::smarts::QueryGraph> = Lazy::new(|| {
        parse_smarts("[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]")
            .expect("valid HBD SMARTS")
    });

    count_matches(mol, &HBD_SMARTS) as u32
}

/// Compute the number of hydrogen bond acceptors (Lipinski definition).
///
/// Matches specific N/O/S patterns per RDKit's HAcceptorSmarts.
pub fn compute_hbond_acceptors(mol: &Molecule) -> u32 {
    use crate::algo::iso::count_matches;
    use crate::io::smarts::parse_smarts;
    use once_cell::sync::Lazy;

    static HBA_SMARTS: Lazy<crate::io::smarts::QueryGraph> = Lazy::new(|| {
        parse_smarts(
            "[$([O,S;H1;v2;!$([O,S;H1;v2]*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
        ).expect("valid HBA SMARTS")
    });

    count_matches(mol, &HBA_SMARTS) as u32
}

/// Computes hbond donors cached.
///
/// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub(crate) fn compute_hbond_donors_cached(mol: &mut Molecule) -> u32 {
    if let Some(cached) = mol.cached_hbond_donor_count() {
        return cached;
    }
    let count = compute_hbond_donors(mol);
    mol.store_hbond_donor_count(count);
    count
}

/// Computes hbond acceptors cached.
///
/// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
pub(crate) fn compute_hbond_acceptors_cached(mol: &mut Molecule) -> u32 {
    if let Some(cached) = mol.cached_hbond_acceptor_count() {
        return cached;
    }
    let count = compute_hbond_acceptors(mol);
    mol.store_hbond_acceptor_count(count);
    count
}

/// Hydrogen bond donor count descriptor.
///
/// Returns `Int(u32)` with number of N-H and O-H groups.
pub struct HBondDonor;

impl Descriptor for HBondDonor {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "HBondDonor"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(compute_hbond_donors_cached(mol))
    }
}

/// Hydrogen bond acceptor count descriptor.
///
/// Returns `Int(u32)` with number of N and O atoms (potential acceptors).
pub struct HBondAcceptor;

impl Descriptor for HBondAcceptor {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "HBondAcceptor"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(compute_hbond_acceptors_cached(mol))
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<HBondDonor>;
    let _ = assert_send_sync::<HBondAcceptor>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    // HBondDonor Tests

    #[test]
    /// Runs regression test for hbond donor water.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_donor_water() {
        let mol = parse_smiles("O").unwrap();
        // Water O has 2H, Lipinski HBD requires exactly H1 for O
        assert_eq!(compute_hbond_donors(&mol), 0);
    }

    #[test]
    /// Runs regression test for hbond donor ammonia.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_donor_ammonia() {
        let mol = parse_smiles("N").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 1);
    }

    #[test]
    /// Runs regression test for hbond donor ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_donor_ethanol() {
        let mol = parse_smiles("CCO").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 1);
    }

    #[test]
    /// Runs regression test for hbond donor urea.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_donor_urea() {
        let mol = parse_smiles("NC(=O)N").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 2);
    }

    #[test]
    /// Runs regression test for hbond donor methane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_donor_methane() {
        let mol = parse_smiles("C").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 0);
    }

    #[test]
    /// Runs regression test for hbond donor acetone.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_donor_acetone() {
        let mol = parse_smiles("CC(=O)C").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 0);
    }

    #[test]
    /// Runs regression test for hbond donor ethylamine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_donor_ethylamine() {
        let mol = parse_smiles("CCN").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 1);
    }

    #[test]
    /// Runs regression test for hbond donor glycine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_donor_glycine() {
        let mol = parse_smiles("NCC(=O)O").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 2);
    }

    #[test]
    /// Runs regression test for hbond donor phenol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_donor_phenol() {
        let mol = parse_smiles("c1ccc(cc1)O").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 1);
    }

    #[test]
    /// Runs regression test for hbond donor aspirin.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_donor_aspirin() {
        let mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 1);
    }

    // HBondAcceptor Tests (Lipinski definition)

    #[test]
    /// Runs regression test for hbond acceptor water.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_acceptor_water() {
        let mol = parse_smiles("O").unwrap();
        // Water is not an HBA in Lipinski definition (no heavy-atom neighbor)
        assert_eq!(compute_hbond_acceptors(&mol), 0);
    }

    #[test]
    /// Runs regression test for hbond acceptor ammonia.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_acceptor_ammonia() {
        let mol = parse_smiles("N").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    /// Runs regression test for hbond acceptor ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_acceptor_ethanol() {
        let mol = parse_smiles("CCO").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    /// Runs regression test for hbond acceptor acetone.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_acceptor_acetone() {
        let mol = parse_smiles("CC(=O)C").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    /// Runs regression test for hbond acceptor urea.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_acceptor_urea() {
        let mol = parse_smiles("NC(=O)N").unwrap();
        // N-C(=O)-N: the 2 N are v3 but adjacent to C=O → excluded; O is H0,v2 → 1
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    /// Runs regression test for hbond acceptor methane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_acceptor_methane() {
        let mol = parse_smiles("C").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 0);
    }

    #[test]
    /// Runs regression test for hbond acceptor diethyl ether.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_acceptor_diethyl_ether() {
        let mol = parse_smiles("CCOCC").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    /// Runs regression test for hbond acceptor glycine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_acceptor_glycine() {
        let mol = parse_smiles("NCC(=O)O").unwrap();
        // N(v3) + C=O(H0,v2) = 2 HBA; OH adjacent to C=O excluded
        assert_eq!(compute_hbond_acceptors(&mol), 2);
    }

    #[test]
    /// Runs regression test for hbond acceptor phenol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_acceptor_phenol() {
        let mol = parse_smiles("c1ccc(cc1)O").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    /// Runs regression test for hbond acceptor aspirin.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_acceptor_aspirin() {
        let mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O").unwrap();
        // 2 C=O (H0,v2) + ester O (H0,v2) = 3; COOH's OH excluded
        assert_eq!(compute_hbond_acceptors(&mol), 3);
    }

    #[test]
    /// Runs regression test for hbond donor descriptor.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_donor_descriptor() {
        let mut mol = parse_smiles("CCO").unwrap();
        let desc = HBondDonor;
        assert_eq!(desc.name(), "HBondDonor");
        assert_eq!(
            desc.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(1)
        );
    }

    #[test]
    /// Runs regression test for hbond acceptor descriptor.
    ///
    /// Function context: `crates/ferrum_core/src/desc/hbond.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_hbond_acceptor_descriptor() {
        let mut mol = parse_smiles("CCO").unwrap();
        let desc = HBondAcceptor;
        assert_eq!(desc.name(), "HBondAcceptor");
        assert_eq!(
            desc.calculate(
                &mut mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(1)
        );
    }
}
