//! Carbon type classification (CDK `CarbonTypesDescriptor`).
//!
//! Every carbon is classified by the highest order of its bonds and by the
//! number of carbon atoms bonded to it:
//! - C1SP1 / C2SP1: highest bond order triple, 1 / 2 carbon neighbours
//! - C1SP2 / C2SP2 / C3SP2: highest bond order double, 1 / 2 / 3 carbon neighbours
//! - C1SP3 .. C4SP3: only single bonds, 1 .. 4 carbon neighbours
//!
//! CDK reads bond orders from the Kekulé structure its SMILES parser produces,
//! while FerrumChem keeps RDKit's aromatic bonds. [`cdk_carbon_bonds`]
//! recovers the Kekulé bond orders of a carbon locally from its valence, and
//! [`cdk_carbon_hybridization`] reproduces the hybridization CDK's atom typer
//! (`CDKAtomTypeMatcher.perceiveCarbons`) assigns to carbons; both are shared
//! with `HybridizationRatio`.

use crate::desc::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::BondType;

/// Bond-order summary of one carbon in CDK's Kekulé representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CdkCarbonBonds {
    /// Highest bond order (1, 2 or 3).
    pub max_order: u8,
    /// Number of double bonds.
    pub n_double: u8,
    /// Sum of bond orders (hydrogens not included).
    pub order_sum: u8,
}

/// Kekulé bond orders of carbon `a`.
///
/// Non-aromatic bonds keep their order (dative and unknown bonds count as
/// single, as CDK reads them from SMILES). In any Kekulé structure an aromatic
/// carbon carries exactly one double bond among its aromatic bonds when its
/// valence (4 if neutral, 3 if singly charged) is not satisfied by single
/// bonds, so no global matching is required.
pub(crate) fn cdk_carbon_bonds(mol: &Molecule, a: usize) -> CdkCarbonBonds {
    let mut max_order = 1u8;
    let mut n_double = 0u8;
    let mut order_sum = 0i32;
    let mut n_aromatic = 0i32;
    for &bi in &mol.adjacency[a] {
        match mol.bonds[bi as usize].bond_type {
            BondType::Aromatic => n_aromatic += 1,
            BondType::Double => {
                order_sum += 2;
                n_double += 1;
                max_order = max_order.max(2);
            }
            BondType::Triple => {
                order_sum += 3;
                max_order = 3;
            }
            _ => order_sum += 1,
        }
    }
    if n_aromatic > 0 {
        let atom = &mol.atoms[a];
        let valence = if atom.formal_charge == 0 { 4 } else { 3 };
        if valence - atom.total_h() as i32 - order_sum > n_aromatic {
            n_double += 1;
            max_order = max_order.max(2);
            order_sum += 1;
        }
        order_sum += n_aromatic;
    }
    CdkCarbonBonds { max_order, n_double, order_sum: order_sum.clamp(0, 255) as u8 }
}

/// Hybridization of a CDK carbon atom type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CdkCarbonHybridization {
    Sp1,
    Sp2,
    Sp3,
    /// `C.plus.planar` / `C.minus.planar`
    Planar,
}

/// Does `a` (any element) carry a double bond in the Kekulé structure?
fn has_kekule_double(mol: &Molecule, a: usize) -> bool {
    if mol.atoms[a].atomic_number == 6 {
        return cdk_carbon_bonds(mol, a).n_double > 0;
    }
    mol.adjacency[a].iter().any(|&bi| {
        let bt = mol.bonds[bi as usize].bond_type;
        bt == BondType::Double || bt == BondType::Aromatic
    })
}

/// Hybridization CDK's atom typer assigns to carbon `a`
/// (`AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms` on the
/// Kekulé form), `None` when CDK finds no carbon atom type.
pub(crate) fn cdk_carbon_hybridization(mol: &Molecule, a: usize) -> Option<CdkCarbonHybridization> {
    use CdkCarbonHybridization::*;
    let atom = &mol.atoms[a];
    let degree = mol.adjacency[a].len() as i32;
    let h = atom.total_h() as i32;
    let b = cdk_carbon_bonds(mol, a);
    // `isAcceptable`: formal neighbour count, maximum bond order and valency
    // of the CDK atom type (charge is matched by the branch).
    let acceptable = |neighbours: i32, max_order: u8, valency: i32| {
        degree + h <= neighbours && b.max_order <= max_order && b.order_sum as i32 + h <= valency
    };
    match atom.formal_charge {
        0 => {
            if degree > 4 {
                return None;
            }
            match b.max_order {
                3 => acceptable(2, 3, 4).then_some(Sp1),         // C.sp
                2 if b.n_double == 2 => acceptable(2, 2, 4).then_some(Sp1), // C.allene
                2 if b.n_double == 1 => acceptable(3, 2, 4).then_some(Sp2), // C.sp2
                2 => None,
                _ => acceptable(4, 1, 4).then_some(Sp3), // C.sp3
            }
        }
        1 => match b.max_order {
            3 => acceptable(1, 3, 3).then_some(Sp1), // C.plus.sp1
            2 => acceptable(2, 2, 3).then_some(Sp2), // C.plus.sp2
            _ => acceptable(3, 1, 3).then_some(Planar), // C.plus.planar
        },
        -1 => match b.max_order {
            1 if degree <= 3 => {
                // C.minus.planar needs two sp2 neighbours and a ring atom
                let sp2_neighbours = mol.adjacency[a]
                    .iter()
                    .filter(|&&bi| {
                        let bond = &mol.bonds[bi as usize];
                        let other = if bond.source as usize == a { bond.target } else { bond.source };
                        bond.bond_type == BondType::Aromatic || has_kekule_double(mol, other as usize)
                    })
                    .count();
                if sp2_neighbours >= 2 && atom.is_in_ring() && acceptable(3, 1, 3) {
                    Some(Planar)
                } else {
                    acceptable(3, 1, 3).then_some(Sp3) // C.minus.sp3
                }
            }
            2 if degree <= 3 => acceptable(2, 2, 3).then_some(Sp2), // C.minus.sp2
            3 if degree <= 1 => acceptable(1, 3, 3).then_some(Sp1), // C.minus.sp1
            _ => None,
        },
        _ => None,
    }
}

/// CDK `CarbonTypesDescriptor` counts in output order
/// (C1SP1, C2SP1, C1SP2, C2SP2, C3SP2, C1SP3, C2SP3, C3SP3, C4SP3).
pub fn carbon_types(mol: &Molecule) -> [u16; 9] {
    let mut counts = [0u16; 9];
    for (a, atom) in mol.atoms.iter().enumerate() {
        if atom.atomic_number != 6 {
            continue;
        }
        let cc = mol.adjacency[a]
            .iter()
            .filter(|&&bi| {
                let bond = &mol.bonds[bi as usize];
                let other = if bond.source as usize == a { bond.target } else { bond.source };
                mol.atoms[other as usize].atomic_number == 6
            })
            .count();
        let slot = match (cdk_carbon_bonds(mol, a).max_order, cc) {
            (3, 1) => 0,
            (3, 2) => 1,
            (2, 1) => 2,
            (2, 2) => 3,
            (2, 3) => 4,
            (1, 1) => 5,
            (1, 2) => 6,
            (1, 3) => 7,
            (1, 4) => 8,
            _ => continue,
        };
        counts[slot] += 1;
    }
    counts
}

/// CDK `CarbonTypesDescriptor` (9 counts, see [`carbon_types`]).
pub struct CarbonTypes;

impl Descriptor for CarbonTypes {
    fn name(&self) -> &str {
        "CarbonTypes"
    }

    fn calculate(&self, mol: &mut Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::IntU16Vec(carbon_types(mol).to_vec())
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<CarbonTypes>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn types(smiles: &str) -> [u16; 9] {
        carbon_types(&parse_smiles(smiles).unwrap())
    }

    // Expected values from CDK 2.13 CarbonTypesDescriptor.
    #[test]
    fn test_small_molecules() {
        assert_eq!(types("C"), [0; 9]); // no carbon neighbour
        assert_eq!(types("CC"), [0, 0, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(types("CCC"), [0, 0, 0, 0, 0, 2, 1, 0, 0]);
        assert_eq!(types("C=C"), [0, 0, 2, 0, 0, 0, 0, 0, 0]);
        assert_eq!(types("C#C"), [2, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(types("CC#C"), [1, 1, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(types("CC(=O)C"), [0, 0, 0, 1, 0, 2, 0, 0, 0]); // carbonyl C: C2SP2
        assert_eq!(types("CC(C)(C)C"), [0, 0, 0, 0, 0, 4, 0, 0, 1]);
    }

    #[test]
    fn test_aromatic_uses_kekule_orders() {
        assert_eq!(types("c1ccccc1"), [0, 0, 0, 6, 0, 0, 0, 0, 0]);
        assert_eq!(types("C1=CC=CC=C1"), types("c1ccccc1"));
        // toluene: ipso carbon has 3 carbon neighbours
        assert_eq!(types("Cc1ccccc1"), [0, 0, 0, 5, 1, 1, 0, 0, 0]);
        // 2-pyridone: C=O carbon has no ring double bond but an exocyclic one
        assert_eq!(types("O=c1cccc[nH]1"), [0, 0, 2, 3, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_hybridization() {
        let mol = parse_smiles("C=C=CC#CC[CH2-]").unwrap();
        let hyb: Vec<_> = (0..mol.num_atoms()).map(|a| cdk_carbon_hybridization(&mol, a)).collect();
        use CdkCarbonHybridization::*;
        assert_eq!(hyb, vec![Some(Sp2), Some(Sp1), Some(Sp2), Some(Sp1), Some(Sp1), Some(Sp3), Some(Sp3)]);
    }
}
