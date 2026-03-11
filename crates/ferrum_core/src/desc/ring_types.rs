//! Ring type counting descriptors
//!
//! Classifies rings by saturation and aromaticity:
//! - Saturated rings: no double/triple bonds
//! - Aliphatic rings: non-aromatic
//! - Aromatic rings: marked as aromatic

use crate::algo::rings::find_symmetrized_sssr_bonds;
use crate::desc::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::BondType;

fn ring_bonds_all_single(mol: &Molecule, ring: &[usize]) -> bool {
    ring.iter()
        .all(|&bond_idx| mol.bonds[bond_idx].bond_type == BondType::Single)
}

fn ring_bonds_all_aromatic(mol: &Molecule, ring: &[usize]) -> bool {
    ring.iter()
        .all(|&bond_idx| mol.bonds[bond_idx].is_aromatic())
}

/// Count number of saturated rings (no double/triple bonds)
///
/// A saturated ring contains only single bonds.
///
/// # Examples
/// - Cyclohexane: 1 saturated ring
/// - Benzene: 0 saturated rings (aromatic double bonds)
/// - Cyclohexene: 0 saturated rings (one double bond)
pub fn num_saturated_rings(mol: &mut Molecule) -> u32 {
    let rings = find_symmetrized_sssr_bonds(mol);

    rings
        .iter()
        .filter(|ring| ring_bonds_all_single(mol, ring))
        .count() as u32
}

/// Count number of aliphatic rings (non-aromatic)
///
/// RDKit defines aliphatic rings as rings containing at least one non-aromatic
/// bond. This includes both saturated rings and rings with double bonds that
/// do not qualify as fully aromatic rings.
///
/// # Examples
/// - Cyclohexane: 1 aliphatic ring
/// - Cyclohexene: 1 aliphatic ring
/// - Benzene: 0 aliphatic rings (aromatic)
/// - Naphthalene: 0 aliphatic rings (both rings aromatic)
pub fn num_aliphatic_rings(mol: &mut Molecule) -> u32 {
    let rings = find_symmetrized_sssr_bonds(mol);

    rings
        .iter()
        .filter(|ring| !ring_bonds_all_aromatic(mol, ring))
        .count() as u32
}

/// Count number of aromatic rings
///
/// RDKit counts aromatic SSSR rings based on aromatic bonds, not just aromatic
/// atoms. Fused systems can contain rings where every atom is aromatic while
/// one or more ring bonds remain non-aromatic/Kekulized.
///
/// # Examples
/// - Benzene: 1 aromatic ring
/// - Naphthalene: 2 aromatic rings
/// - Pyridine: 1 aromatic ring
/// - Cyclohexane: 0 aromatic rings
pub fn num_aromatic_rings(mol: &mut Molecule) -> u32 {
    let rings = find_symmetrized_sssr_bonds(mol);

    rings
        .iter()
        .filter(|ring| ring_bonds_all_aromatic(mol, ring))
        .count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }

    #[test]
    fn test_saturated_cyclohexane() {
        let mut mol = parse("C1CCCCC1");
        assert_eq!(num_saturated_rings(&mut mol), 1);
        assert_eq!(num_aliphatic_rings(&mut mol), 1);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }

    #[test]
    fn test_aromatic_benzene() {
        let mut mol = parse("c1ccccc1");
        assert_eq!(num_saturated_rings(&mut mol), 0);
        assert_eq!(num_aliphatic_rings(&mut mol), 0);
        assert_eq!(num_aromatic_rings(&mut mol), 1);
    }

    #[test]
    fn test_cyclohexene() {
        let mut mol = parse("C1=CCCCC1");
        assert_eq!(num_saturated_rings(&mut mol), 0); // Has double bond
        assert_eq!(num_aliphatic_rings(&mut mol), 1); // Not aromatic
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }

    #[test]
    fn test_naphthalene() {
        let mut mol = parse("c1ccc2ccccc2c1");
        assert_eq!(num_saturated_rings(&mut mol), 0);
        assert_eq!(num_aliphatic_rings(&mut mol), 0);
        assert_eq!(num_aromatic_rings(&mut mol), 2); // Two fused aromatic rings
    }

    #[test]
    fn test_pyridine() {
        let mut mol = parse("c1ccncc1");
        assert_eq!(num_saturated_rings(&mut mol), 0);
        assert_eq!(num_aliphatic_rings(&mut mol), 0);
        assert_eq!(num_aromatic_rings(&mut mol), 1);
    }

    #[test]
    fn test_decalin() {
        // Two fused saturated rings
        let mut mol = parse("C1CCC2CCCCC2C1");
        assert_eq!(num_saturated_rings(&mut mol), 2);
        assert_eq!(num_aliphatic_rings(&mut mol), 2);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }

    #[test]
    fn test_cubane_uses_symmetrized_ring_count() {
        let mut mol = parse("C12C3C4C1C5C2C3C45");
        assert_eq!(num_saturated_rings(&mut mol), 6);
        assert_eq!(num_aliphatic_rings(&mut mol), 6);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }

    #[test]
    fn test_adamantane_uses_symmetrized_ring_count() {
        let mut mol = parse("C1C2CC3CC1CC(C2)C3");
        assert_eq!(num_saturated_rings(&mut mol), 4);
        assert_eq!(num_aliphatic_rings(&mut mol), 4);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }

    #[test]
    fn test_acyclic() {
        let mut mol = parse("CCCCCC"); // Hexane
        assert_eq!(num_saturated_rings(&mut mol), 0);
        assert_eq!(num_aliphatic_rings(&mut mol), 0);
        assert_eq!(num_aromatic_rings(&mut mol), 0);
    }

    #[test]
    fn test_rdkit_like_ring_counts_for_fused_heterocycle() {
        let mut mol = parse(
            "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2",
        );
        assert_eq!(num_saturated_rings(&mut mol), 1);
        assert_eq!(num_aliphatic_rings(&mut mol), 3);
        assert_eq!(num_aromatic_rings(&mut mol), 3);
    }
}

// ===========================================================================
// Typed API: Ring Types
// ===========================================================================

/// Ring type counts descriptor.
///
/// Returns IntU16Vec with 3 values:
/// [saturated_rings, aliphatic_rings, aromatic_rings]
///
/// Uses u16 since molecules rarely have >65k rings of any type.
pub struct RingTypeCounts;

impl Descriptor for RingTypeCounts {
    fn name(&self) -> &str {
        "RingTypeCounts"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let rings = find_symmetrized_sssr_bonds(mol);

        let mut saturated = 0u16;
        let mut aliphatic = 0u16;
        let mut aromatic = 0u16;

        for ring in &rings {
            let is_aromatic = ring_bonds_all_aromatic(mol, ring);

            if is_aromatic {
                aromatic += 1;
            } else {
                aliphatic += 1;

                if ring_bonds_all_single(mol, ring) {
                    saturated += 1;
                }
            }
        }

        DescriptorValue::IntU16Vec(vec![saturated, aliphatic, aromatic])
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<RingTypeCounts>;
};
