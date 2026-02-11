use super::typed::{DescriptorValue, Descriptor};
use crate::mol::molecule::Molecule;

/// Compute the number of hydrogen bond donors in a molecule.
///
/// A hydrogen bond donor is a nitrogen or oxygen atom with at least one
/// hydrogen attached (N-H or O-H groups). Follows the Lipinski Rule of Five
/// definition (HBD ≤ 5 for drug-like molecules).
pub fn compute_hbond_donors(mol: &Molecule) -> u32 {
    mol.atoms.iter()
        .filter(|atom| {
            (atom.atomic_number == 7 || atom.atomic_number == 8) &&
            atom.total_h() > 0
        })
        .count() as u32
}

/// Compute the number of hydrogen bond acceptors in a molecule.
///
/// A hydrogen bond acceptor is a nitrogen or oxygen atom (has lone pairs).
/// Follows the Lipinski Rule of Five definition (HBA ≤ 10 for drug-like molecules).
pub fn compute_hbond_acceptors(mol: &Molecule) -> u32 {
    mol.atoms.iter()
        .filter(|atom| {
            atom.atomic_number == 7 || atom.atomic_number == 8
        })
        .count() as u32
}

/// Hydrogen bond donor count descriptor.
///
/// Returns `Int(u32)` with number of N-H and O-H groups.
pub struct HBondDonor;

impl Descriptor for HBondDonor {
    fn name(&self) -> &str {
        "HBondDonor"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Int(compute_hbond_donors(mol))
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

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Int(compute_hbond_acceptors(mol))
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
        assert_eq!(compute_hbond_donors(&mol), 1);
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

    // HBondAcceptor Tests

    #[test]
    fn test_hbond_acceptor_water() {
        let mol = parse_smiles("O").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 1);
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
        assert_eq!(compute_hbond_acceptors(&mol), 3);
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
        assert_eq!(compute_hbond_acceptors(&mol), 3);
    }

    #[test]
    fn test_hbond_acceptor_phenol() {
        let mol = parse_smiles("c1ccc(cc1)O").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 1);
    }

    #[test]
    fn test_hbond_acceptor_aspirin() {
        let mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O").unwrap();
        assert_eq!(compute_hbond_acceptors(&mol), 4);
    }

    

    #[test]
    fn test_hbond_donor_descriptor() {
        let mut mol = parse_smiles("CCO").unwrap();
        let desc = HBondDonor;
        assert_eq!(desc.name(), "HBondDonor");
        assert_eq!(desc.calculate(&mut mol), DescriptorValue::Int(1));
    }

    #[test]
    fn test_hbond_acceptor_descriptor() {
        let mut mol = parse_smiles("CCO").unwrap();
        let desc = HBondAcceptor;
        assert_eq!(desc.name(), "HBondAcceptor");
        assert_eq!(desc.calculate(&mut mol), DescriptorValue::Int(1));
    }
}
