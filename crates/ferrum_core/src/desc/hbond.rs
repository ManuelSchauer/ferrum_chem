//! Descriptor module: Hbond.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

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
            // RDKit 2026.03 CalcNumHBA: aromatic N only when unsubstituted (X2)
            "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([n;H0;+0;X2]),$([o,s;+0])]"
        ).expect("valid HBA SMARTS")
    });

    count_matches(mol, &HBA_SMARTS) as u32
}

pub(crate) fn compute_hbond_donors_cached(mol: &Molecule) -> u32 {
    mol.hbond_donor_count_or_init(|| compute_hbond_donors(mol))
}

pub(crate) fn compute_hbond_acceptors_cached(mol: &Molecule) -> u32 {
    mol.hbond_acceptor_count_or_init(|| compute_hbond_acceptors(mol))
}

/// Hydrogen bond donor count descriptor.
///
/// Returns `Int(u32)` with number of N-H and O-H groups.
pub struct HBondDonor;

impl Descriptor for HBondDonor {
    fn name(&self) -> &str {
        "HBondDonor"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(compute_hbond_donors_cached(mol))
    }
}

/// Hydrogen bond acceptor count descriptor.
///
/// Returns `Int(u32)` with number of N and O atoms (potential acceptors).
pub struct HBondAcceptor;

impl Descriptor for HBondAcceptor {
    fn name(&self) -> &str {
        "HBondAcceptor"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(compute_hbond_acceptors_cached(mol))
    }
}

const _: () = {
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
    fn test_hbond_donor_water() {
        let mol = parse_smiles("O").unwrap();
        // Water O has 2H, Lipinski HBD requires exactly H1 for O
        assert_eq!(compute_hbond_donors(&mol), 0);
    }

    #[test]
    fn test_hbond_donor_ammonia() {
        let mol = parse_smiles("N").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 1);
    }

    #[test]
    fn test_hbond_donor_ethanol() {
        let mol = parse_smiles("CCO").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 1);
    }

    #[test]
    fn test_hbond_donor_urea() {
        let mol = parse_smiles("NC(=O)N").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 2);
    }

    #[test]
    fn test_hbond_donor_methane() {
        let mol = parse_smiles("C").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 0);
    }

    #[test]
    fn test_hbond_donor_acetone() {
        let mol = parse_smiles("CC(=O)C").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 0);
    }

    #[test]
    fn test_hbond_donor_ethylamine() {
        let mol = parse_smiles("CCN").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 1);
    }

    #[test]
    fn test_hbond_donor_glycine() {
        let mol = parse_smiles("NCC(=O)O").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 2);
    }

    #[test]
    fn test_hbond_donor_phenol() {
        let mol = parse_smiles("c1ccc(cc1)O").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 1);
    }

    #[test]
    fn test_hbond_donor_aspirin() {
        let mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O").unwrap();
        assert_eq!(compute_hbond_donors(&mol), 1);
    }

    // HBondAcceptor Tests (Lipinski definition)

    #[test]
    fn test_hbond_acceptor_water() {
        let mol = parse_smiles("O").unwrap();
        // Water is not an HBA in Lipinski definition (no heavy-atom neighbor)
        assert_eq!(compute_hbond_acceptors(&mol), 0);
    }

    #[test]
    fn test_hbond_acceptor_ammonia() {
        let mol = parse_smiles("N").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    fn test_hbond_acceptor_ethanol() {
        let mol = parse_smiles("CCO").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    fn test_hbond_acceptor_acetone() {
        let mol = parse_smiles("CC(=O)C").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    fn test_hbond_acceptor_urea() {
        let mol = parse_smiles("NC(=O)N").unwrap();
        // N-C(=O)-N: the 2 N are v3 but adjacent to C=O → excluded; O is H0,v2 → 1
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    fn test_hbond_acceptor_methane() {
        let mol = parse_smiles("C").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 0);
    }

    #[test]
    fn test_hbond_acceptor_diethyl_ether() {
        let mol = parse_smiles("CCOCC").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    fn test_hbond_acceptor_glycine() {
        let mol = parse_smiles("NCC(=O)O").unwrap();
        // N(v3) + C=O(H0,v2) = 2 HBA; OH adjacent to C=O excluded
        assert_eq!(compute_hbond_acceptors(&mol), 2);
    }

    #[test]
    fn test_hbond_acceptor_phenol() {
        let mol = parse_smiles("c1ccc(cc1)O").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    fn test_hbond_acceptor_aspirin() {
        let mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O").unwrap();
        // 2 C=O (H0,v2) + ester O (H0,v2) = 3; COOH's OH excluded
        assert_eq!(compute_hbond_acceptors(&mol), 3);
    }

    #[test]
    fn test_hbond_donor_descriptor() {
        let mol = parse_smiles("CCO").unwrap();
        let desc = HBondDonor;
        assert_eq!(desc.name(), "HBondDonor");
        assert_eq!(
            desc.calculate(
                &mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(1)
        );
    }

    #[test]
    fn test_hbond_acceptor_descriptor() {
        let mol = parse_smiles("CCO").unwrap();
        let desc = HBondAcceptor;
        assert_eq!(desc.name(), "HBondAcceptor");
        assert_eq!(
            desc.calculate(
                &mol,
                &mut crate::desc::context::CalculationContext::new()
            ),
            DescriptorValue::Int(1)
        );
    }
}
