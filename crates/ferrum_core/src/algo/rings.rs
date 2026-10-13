//! # Ring queries
//!
//! Thin accessors over the ring perception done during sanitization
//! ([`crate::algo::sanitize`]), which uses RDKit's symmetrized SSSR
//! ([`crate::algo::sssr`]). Every descriptor therefore sees exactly the ring
//! set RDKit's `RingInfo` reports.
//!
//! - [`find_sssr`] / [`find_symmetrized_sssr_bonds`]: RDKit `RingInfo` rings
//!   (symmetrized SSSR), as atom or bond lists
//! - [`find_minimum_cycle_basis_bonds`]: RDKit `Chem.GetSSSR` (not symmetrized)

use crate::algo::sssr;
use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use smallvec::SmallVec;

fn compute_ring_info_if_missing(mol: &Molecule) -> Option<Vec<SmallVec<[u16; 8]>>> {
    if mol.ring_info().is_some() {
        return None;
    }
    Some(
        sssr::symmetrized_sssr(mol)
            .iter()
            .map(|r| sssr::ring_atoms_to_bonds(mol, r).iter().map(|&b| b as u16).collect())
            .collect(),
    )
}

/// Atom rings (RDKit `RingInfo::atomRings`, symmetrized SSSR) in traversal order.
pub fn find_sssr(mol: &mut Molecule) -> Vec<SmallVec<[AtomIdx; 8]>> {
    atom_rings(mol)
}

/// Atom rings (RDKit `RingInfo::atomRings`) without requiring `&mut`.
pub fn atom_rings(mol: &Molecule) -> Vec<SmallVec<[AtomIdx; 8]>> {
    if let Some(info) = mol.ring_info() {
        return info.atom_rings.clone();
    }
    sssr::symmetrized_sssr(mol)
        .into_iter()
        .map(|r| r.into_iter().map(|a| a as AtomIdx).collect())
        .collect()
}

/// Bond rings (RDKit `RingInfo::bondRings`, symmetrized SSSR).
pub fn find_symmetrized_sssr_bonds(mol: &Molecule) -> Vec<SmallVec<[u16; 8]>> {
    match mol.ring_info() {
        Some(info) => info.bond_rings.clone(),
        None => compute_ring_info_if_missing(mol).unwrap_or_default(),
    }
}

/// Borrowed bond rings from the sanitization ring info.
pub(crate) fn cached_symmetrized_sssr_bonds_ref(mol: &Molecule) -> Option<&[SmallVec<[u16; 8]>]> {
    mol.ring_info().map(|info| info.bond_rings.as_slice())
}

/// Ensures ring info is available (only needed for molecules built by hand).
pub(crate) fn ensure_symmetrized_sssr_bonds_cached(mol: &mut Molecule) {
    if mol.ring_info().is_none() {
        let rings = sssr::symmetrized_sssr(mol);
        let mut info = crate::mol::molecule::RingInfo {
            atom_ring_counts: vec![0; mol.num_atoms()],
            bond_ring_counts: vec![0; mol.num_bonds()],
            ..Default::default()
        };
        for r in &rings {
            let bonds = sssr::ring_atoms_to_bonds(mol, r);
            for &a in r {
                info.atom_ring_counts[a as usize] += 1;
            }
            for &b in &bonds {
                info.bond_ring_counts[b as usize] += 1;
            }
            info.atom_rings.push(r.iter().map(|&a| a as AtomIdx).collect());
            info.bond_rings.push(bonds.iter().map(|&b| b as u16).collect());
        }
        mol.set_ring_info(info);
    }
}

/// RDKit `Chem.GetSSSR`: smallest set of smallest rings, as bond lists.
pub fn find_minimum_cycle_basis_bonds(mol: &Molecule) -> Vec<SmallVec<[u16; 8]>> {
    sssr::sssr(mol)
        .iter()
        .map(|r| sssr::ring_atoms_to_bonds(mol, r).iter().map(|&b| b as u16).collect())
        .collect()
}

/// Cached variant of [`find_minimum_cycle_basis_bonds`].
pub(crate) fn ensure_minimum_cycle_basis_bonds_cached(mol: &mut Molecule) {
    if mol.cached_min_cycle_basis_bonds().is_none() {
        let rings = find_minimum_cycle_basis_bonds(mol);
        mol.store_min_cycle_basis_bonds(rings);
    }
}

/// Check if an atom is in a ring.
pub fn is_atom_in_ring(mol: &Molecule, atom_idx: AtomIdx) -> bool {
    mol.atoms[atom_idx as usize].is_in_ring()
}

/// Check if a bond is in a ring.
pub fn is_bond_in_ring(mol: &Molecule, bond_idx: usize) -> bool {
    mol.bonds[bond_idx].is_in_ring()
}

/// Number of rings (RDKit `RingInfo::NumRings`, symmetrized SSSR).
pub fn count_rings(mol: &Molecule) -> usize {
    match mol.ring_info() {
        Some(info) => info.num_rings(),
        None => sssr::symmetrized_sssr(mol).len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn ring_counts_match_rdkit() {
        for (smiles, n) in [
            ("CCO", 0),
            ("c1ccccc1", 1),
            ("c1ccc2ccccc2c1", 2),
            ("C1CC2CCC1C2", 2),
            ("C12C3C4C1C5C2C3C45", 6),
            ("C1CCCCC1.C1CC1", 2),
        ] {
            let mut mol = parse_smiles(smiles).unwrap();
            assert_eq!(count_rings(&mol), n, "{smiles}");
            assert_eq!(find_sssr(&mut mol).len(), n, "{smiles}");
            assert_eq!(find_symmetrized_sssr_bonds(&mol).len(), n, "{smiles}");
        }
    }

    #[test]
    fn sssr_vs_symmetrized() {
        // cubane: 5 SSSR rings, 6 symmetrized
        let mol = parse_smiles("C12C3C4C1C5C2C3C45").unwrap();
        assert_eq!(find_minimum_cycle_basis_bonds(&mol).len(), 5);
        assert_eq!(find_symmetrized_sssr_bonds(&mol).len(), 6);
    }

    #[test]
    fn ring_flags() {
        let mol = parse_smiles("c1ccccc1C").unwrap();
        assert!(is_atom_in_ring(&mol, 0));
        assert!(!is_atom_in_ring(&mol, 6));
        assert!(!is_bond_in_ring(&mol, 5)); // ring closure bond (index 6) comes last
    }
}
