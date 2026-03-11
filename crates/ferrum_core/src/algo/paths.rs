use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use std::collections::BTreeMap;

pub fn find_all_paths_of_length(mol: &Molecule, bond_length: usize) -> Vec<Vec<AtomIdx>> {
    if mol.num_atoms() == 0 {
        return Vec::new();
    }
    if bond_length == 0 {
        return (0..mol.num_atoms())
            .map(|idx| vec![idx as AtomIdx])
            .collect();
    }

    let mut paths = BTreeMap::new();
    let mut atom_path = Vec::with_capacity(bond_length + 1);
    let mut used_bonds = Vec::with_capacity(bond_length);

    for start in 0..mol.num_atoms() {
        atom_path.clear();
        used_bonds.clear();
        atom_path.push(start as AtomIdx);
        dfs_paths_of_length(
            mol,
            &mut atom_path,
            &mut used_bonds,
            bond_length,
            &mut paths,
        );
    }

    paths.into_values().collect()
}

fn dfs_paths_of_length(
    mol: &Molecule,
    atom_path: &mut Vec<AtomIdx>,
    used_bonds: &mut Vec<usize>,
    bond_length: usize,
    out: &mut BTreeMap<Vec<usize>, Vec<AtomIdx>>,
) {
    if used_bonds.len() == bond_length {
        let mut key = used_bonds.clone();
        key.sort_unstable();
        out.entry(key).or_insert_with(|| atom_path.clone());
        return;
    }

    let current = *atom_path.last().unwrap();
    for &bond_idx in &mol.adjacency[current as usize] {
        let bidx = bond_idx as usize;
        if used_bonds.contains(&bidx) {
            continue;
        }

        let bond = &mol.bonds[bidx];
        let neighbor = if bond.source == current {
            bond.target
        } else {
            bond.source
        };

        used_bonds.push(bidx);
        atom_path.push(neighbor);
        dfs_paths_of_length(mol, atom_path, used_bonds, bond_length, out);
        atom_path.pop();
        used_bonds.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::find_all_paths_of_length;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_triangle_counts_closed_three_bond_path_once() {
        let mol = parse_smiles("C1CC1").unwrap();
        let paths = find_all_paths_of_length(&mol, 3);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 4);
        assert_eq!(paths[0].first(), paths[0].last());
    }

    #[test]
    fn test_cyclobutane_counts_open_and_closed_paths_like_rdkit() {
        let mol = parse_smiles("C1CCC1").unwrap();
        assert_eq!(find_all_paths_of_length(&mol, 3).len(), 4);
        assert_eq!(find_all_paths_of_length(&mol, 4).len(), 1);
    }
}
