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
    if mol.num_bonds() == 0 || mol.num_atoms() < 2 {
        return 0;
    }

    // RDKit-style filtering is based on graph topology, not cached ring flags.
    // A bond is in a ring iff it lies on a cycle (i.e. it is not a bridge).
    let bond_in_cycle = compute_bond_in_cycle_flags(mol);
    let heavy_degree = compute_heavy_degrees(mol);
    let atom_has_triple = compute_has_triple_neighbor(mol);
    let conjugated_n = compute_conjugated_nitrogen_flags(mol);

    let mut count = 0u32;

    for (bond_idx, bond) in mol.bonds.iter().enumerate() {
        // Single, non-aromatic bonds only.
        if bond.bond_type != BondType::Single || bond.is_aromatic() {
            continue;
        }

        // Exclude ring bonds (cycle edges).
        if bond_in_cycle[bond_idx] {
            continue;
        }

        let a = bond.source as usize;
        let b = bond.target as usize;

        // Heavy-atom bonds only.
        if mol.atoms[a].atomic_number <= 1 || mol.atoms[b].atomic_number <= 1 {
            continue;
        }

        // Both atoms must be non-terminal heavy atoms.
        if heavy_degree[a] <= 1 || heavy_degree[b] <= 1 {
            continue;
        }

        // RDKit strict excludes bonds attached to quaternary carbon centers
        // carrying three terminal methyl/fluoro/chloro/bromo substituents.
        if is_strict_terminal_substituent_rotor_blocked(mol, bond_idx, &heavy_degree) {
            continue;
        }

        // Exclude bonds attached to atoms participating in triple bonds.
        if atom_has_triple[a] || atom_has_triple[b] {
            continue;
        }

        // Exclude strict amide-/amidine-like C(=X)-N bonds themselves.
        if is_restricted_conjugated_carbon_n_bond(mol, bond_idx) {
            continue;
        }

        // Exclude N-N linkages only when both nitrogens are amidic / conjugated.
        if is_conjugated_nitrogen_pair_bond(mol, bond_idx, &conjugated_n) {
            continue;
        }

        // Carbonyl-O/S exclusion: ester / thioester C(=O)–O/S bonds.
        if is_carbonyl_os_single_bond(mol, bond_idx) {
            continue;
        }

        count += 1;
    }

    count
}

pub(crate) fn compute_rotatable_bond_count_cached(mol: &mut Molecule) -> u32 {
    if let Some(cached) = mol.cached_rotatable_bond_count() {
        return cached;
    }
    let count = compute_rotatable_bond_count(mol);
    mol.store_rotatable_bond_count(count);
    count
}

fn is_strict_terminal_substituent_rotor_blocked(
    mol: &Molecule,
    bond_idx: usize,
    heavy_degree: &[usize],
) -> bool {
    let bond = &mol.bonds[bond_idx];
    let a = bond.source as usize;
    let b = bond.target as usize;
    is_strict_terminal_substituent_center(mol, a, b, heavy_degree)
        || is_strict_terminal_substituent_center(mol, b, a, heavy_degree)
}

fn is_strict_terminal_substituent_center(
    mol: &Molecule,
    center_idx: usize,
    bonded_idx: usize,
    heavy_degree: &[usize],
) -> bool {
    let atom = &mol.atoms[center_idx];
    if atom.atomic_number != 6
        || atom.is_aromatic()
        || atom.is_in_ring()
        || heavy_degree[center_idx] != 4
    {
        return false;
    }

    let mut terminal_methyl_neighbors = 0usize;
    let mut terminal_f_neighbors = 0usize;
    let mut terminal_cl_neighbors = 0usize;
    let mut terminal_br_neighbors = 0usize;
    for &bi in &mol.adjacency[center_idx] {
        let bond = &mol.bonds[bi as usize];
        if bond.bond_type != BondType::Single {
            return false;
        }

        let other = if bond.source == center_idx as u16 {
            bond.target as usize
        } else {
            bond.source as usize
        };

        if other == bonded_idx {
            continue;
        }

        let other_atom = &mol.atoms[other];
        match other_atom.atomic_number {
            // RDKit strict excludes C([CH3])([CH3])[CH3] centers.
            6 if !other_atom.is_aromatic() && heavy_degree[other] == 1 && other_atom.total_h() == 3 => {
                terminal_methyl_neighbors += 1;
            }
            // RDKit strict excludes C(F)(F)F, C(Cl)(Cl)Cl and C(Br)(Br)Br.
            9 if heavy_degree[other] == 1 => {
                terminal_f_neighbors += 1;
            }
            17 if heavy_degree[other] == 1 => {
                terminal_cl_neighbors += 1;
            }
            35 if heavy_degree[other] == 1 => {
                terminal_br_neighbors += 1;
            }
            _ => {}
        }
    }

    terminal_methyl_neighbors >= 3
        || terminal_f_neighbors >= 3
        || terminal_cl_neighbors >= 3
        || terminal_br_neighbors >= 3
}

/// Returns a per-atom boolean: `true` if atom `i` is a neutral, non-aromatic
/// nitrogen attached to a conjugated `C(=X)` center (`X = O, N, S`) which also
/// bears another heavy substituent. This captures the amidic / amidine-like N
/// atoms whose directly conjugated bonds are non-rotatable under RDKit's strict
/// rotatable-bond semantics, while still allowing formyl `N-C(=X)H` bonds.
fn compute_conjugated_nitrogen_flags(mol: &Molecule) -> Vec<bool> {
    let mut flags = vec![false; mol.num_atoms()];
    for (i, flag) in flags.iter_mut().enumerate() {
        let atom = &mol.atoms[i];
        if atom.atomic_number != 7 || atom.formal_charge != 0 || atom.is_aromatic() {
            continue;
        }

        'outer: for &bi in &mol.adjacency[i] {
            let bond = &mol.bonds[bi as usize];
            if bond.bond_type != BondType::Single {
                continue;
            }
            let nbr = if bond.source == i as u16 {
                bond.target as usize
            } else {
                bond.source as usize
            };
            if mol.atoms[nbr].atomic_number != 6 {
                continue;
            }
            if is_restricted_conjugated_center_for_n(mol, nbr, i) {
                *flag = true;
                break 'outer;
            }
        }
    }
    flags
}

fn compute_heavy_degrees(mol: &Molecule) -> Vec<usize> {
    let mut deg = vec![0usize; mol.num_atoms()];
    for (i, out) in deg.iter_mut().enumerate() {
        *out = mol.adjacency[i]
            .iter()
            .filter(|&&bi| {
                let b = &mol.bonds[bi as usize];
                let nbr = if b.source == i as u16 {
                    b.target
                } else {
                    b.source
                } as usize;
                mol.atoms[nbr].atomic_number > 1
            })
            .count();
    }
    deg
}

fn compute_has_triple_neighbor(mol: &Molecule) -> Vec<bool> {
    let mut out = vec![false; mol.num_atoms()];
    for (i, has_triple) in out.iter_mut().enumerate() {
        *has_triple = mol.adjacency[i]
            .iter()
            .any(|&bi| mol.bonds[bi as usize].bond_type == BondType::Triple);
    }
    out
}

fn compute_bond_in_cycle_flags(mol: &Molecule) -> Vec<bool> {
    let n = mol.num_atoms();
    let m = mol.num_bonds();
    let mut disc = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut is_bridge = vec![false; m];
    let mut time = 0usize;

    fn dfs(
        mol: &Molecule,
        u: usize,
        parent_bond: Option<usize>,
        time: &mut usize,
        disc: &mut [usize],
        low: &mut [usize],
        is_bridge: &mut [bool],
    ) {
        disc[u] = *time;
        low[u] = *time;
        *time += 1;

        for &bi in &mol.adjacency[u] {
            let bidx = bi as usize;
            if Some(bidx) == parent_bond {
                continue;
            }

            let bond = &mol.bonds[bidx];
            let v = if bond.source == u as u16 {
                bond.target as usize
            } else {
                bond.source as usize
            };

            if disc[v] == usize::MAX {
                dfs(mol, v, Some(bidx), time, disc, low, is_bridge);
                low[u] = low[u].min(low[v]);
                if low[v] > disc[u] {
                    is_bridge[bidx] = true;
                }
            } else {
                low[u] = low[u].min(disc[v]);
            }
        }
    }

    for u in 0..n {
        if disc[u] == usize::MAX {
            dfs(mol, u, None, &mut time, &mut disc, &mut low, &mut is_bridge);
        }
    }

    is_bridge.into_iter().map(|b| !b).collect()
}

/// Exclude ester / thioester bonds: C(=O)–O or C(=O)–S single bonds that are
/// not in a ring.
///
/// **N is intentionally excluded from this check.** Conjugated `C(=X)-N` bonds
/// are handled by `is_restricted_conjugated_carbon_n_bond()`. As with formamides,
/// formate/thioformate `C(=O)H-O/S` bonds remain rotatable under RDKit strict.
fn is_carbonyl_os_single_bond(mol: &Molecule, bond_idx: usize) -> bool {
    let bond = &mol.bonds[bond_idx];
    if bond.bond_type != BondType::Single {
        return false;
    }

    let a = bond.source as usize;
    let b = bond.target as usize;
    let za = mol.atoms[a].atomic_number;
    let zb = mol.atoms[b].atomic_number;

    // Only C–O or C–S bonds (not C–N).
    let (hetero_idx, c_idx) = if za == 6 && matches!(zb, 8 | 16) {
        (b, a)
    } else if zb == 6 && matches!(za, 8 | 16) {
        (a, b)
    } else {
        return false;
    };

    // Aromatic O/S atoms are already handled upstream (aromatic bonds excluded).
    if mol.atoms[hetero_idx].is_aromatic() {
        return false;
    }

    let mut has_conjugated_hetero_double = false;
    let mut has_other_heavy_substituent = false;

    for &bi in &mol.adjacency[c_idx] {
        let b2_idx = bi as usize;
        if b2_idx == bond_idx {
            continue;
        }
        let b2 = &mol.bonds[b2_idx];
        let other = if b2.source == c_idx as u16 {
            b2.target as usize
        } else {
            b2.source as usize
        };

        if b2.bond_type == BondType::Double && matches!(mol.atoms[other].atomic_number, 7 | 8 | 16)
        {
            has_conjugated_hetero_double = true;
            continue;
        }

        if other != hetero_idx && mol.atoms[other].atomic_number > 1 {
            has_other_heavy_substituent = true;
        }
    }

    has_conjugated_hetero_double && has_other_heavy_substituent
}

fn is_restricted_conjugated_carbon_n_bond(mol: &Molecule, bond_idx: usize) -> bool {
    let bond = &mol.bonds[bond_idx];
    if bond.bond_type != BondType::Single || bond.is_aromatic() {
        return false;
    }

    let a = bond.source as usize;
    let b = bond.target as usize;
    let (n_idx, c_idx) = if mol.atoms[a].atomic_number == 7 && mol.atoms[b].atomic_number == 6 {
        (a, b)
    } else if mol.atoms[b].atomic_number == 7 && mol.atoms[a].atomic_number == 6 {
        (b, a)
    } else {
        return false;
    };

    let n = &mol.atoms[n_idx];
    n.formal_charge == 0 && is_restricted_conjugated_center_for_n(mol, c_idx, n_idx)
}

fn is_restricted_conjugated_center_for_n(mol: &Molecule, carbon_idx: usize, n_idx: usize) -> bool {
    if mol.atoms[carbon_idx].atomic_number != 6 {
        return false;
    }

    let mut has_conjugated_hetero_double = false;
    let mut has_other_heavy_substituent = false;

    for &bi in &mol.adjacency[carbon_idx] {
        let bond = &mol.bonds[bi as usize];
        let other = if bond.source == carbon_idx as u16 {
            bond.target as usize
        } else {
            bond.source as usize
        };

        if bond.bond_type == BondType::Double
            && matches!(mol.atoms[other].atomic_number, 7 | 8 | 16)
        {
            has_conjugated_hetero_double = true;
            continue;
        }

        if other != n_idx && mol.atoms[other].atomic_number > 1 {
            has_other_heavy_substituent = true;
        }
    }

    has_conjugated_hetero_double && has_other_heavy_substituent
}

fn is_conjugated_nitrogen_pair_bond(
    mol: &Molecule,
    bond_idx: usize,
    conjugated_n: &[bool],
) -> bool {
    let bond = &mol.bonds[bond_idx];
    if bond.bond_type != BondType::Single || bond.is_aromatic() {
        return false;
    }

    let a = bond.source as usize;
    let b = bond.target as usize;
    mol.atoms[a].atomic_number == 7
        && mol.atoms[b].atomic_number == 7
        && conjugated_n[a]
        && conjugated_n[b]
}

// ---------------------------------------------------------------------------
// Typed Descriptors
// ---------------------------------------------------------------------------

/// Total bond count.
///
/// Returns Int(u32) with total number of bonds.
pub struct BondCount;

impl Descriptor for BondCount {
    fn name(&self) -> &str {
        "BondCount"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::Int(compute_bond_count(mol))
    }
}

/// Detailed bond counts.
///
/// Returns IntVec with [total, single, double, triple, aromatic].
pub struct BondCountDetailed;

impl Descriptor for BondCountDetailed {
    fn name(&self) -> &str {
        "BondCountDetailed"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
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
    fn name(&self) -> &str {
        "RotatableBondCount"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::Int(compute_rotatable_bond_count_cached(mol))
    }
}

const _: () = {
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
    fn test_bond_count_simple() {
        let mut mol = parse_smiles("CCO").unwrap();
        assert_eq!(BondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 2);
    }

    #[test]
    fn test_bond_count_detailed_single() {
        let mut mol = parse_smiles("CCCC").unwrap();
        let v = BondCountDetailed.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int_vec().to_vec();
        assert_eq!(v[0], 3); // Total
        assert_eq!(v[1], 3); // Single
        assert_eq!(v[2], 0); // Double
        assert_eq!(v[3], 0); // Triple
    }

    #[test]
    fn test_bond_count_detailed_double() {
        let mut mol = parse_smiles("C=C").unwrap();
        let v = BondCountDetailed.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int_vec().to_vec();
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 0);
        assert_eq!(v[2], 1);
        assert_eq!(v[3], 0);
    }

    #[test]
    fn test_bond_count_detailed_triple() {
        let mut mol = parse_smiles("C#N").unwrap();
        let v = BondCountDetailed.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int_vec().to_vec();
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 0);
        assert_eq!(v[2], 0);
        assert_eq!(v[3], 1);
    }

    #[test]
    fn test_bond_count_detailed_mixed() {
        let mut mol = parse_smiles("CC=C").unwrap();
        let v = BondCountDetailed.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int_vec().to_vec();
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 1);
        assert_eq!(v[2], 1);
        assert_eq!(v[3], 0);
    }

    #[test]
    fn test_rotatable_bond_count_ethane() {
        let mut mol = parse_smiles("CC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);
    }

    #[test]
    fn test_rotatable_bond_count_propane() {
        let mut mol = parse_smiles("CCC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);
    }

    #[test]
    fn test_rotatable_bond_count_butane() {
        let mut mol = parse_smiles("CCCC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    fn test_rotatable_bond_count_ethene() {
        let mut mol = parse_smiles("C=C").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);
    }

    #[test]
    fn test_rotatable_bond_count_benzene() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        find_sssr(&mut mol);
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);
    }

    #[test]
    fn test_rotatable_bond_count_toluene() {
        let mut mol = parse_smiles("Cc1ccccc1").unwrap();
        find_sssr(&mut mol);
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);
    }

    #[test]
    fn test_rotatable_bond_count_pentane() {
        let mut mol = parse_smiles("CCCCC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 2);
    }

    #[test]
    fn test_rotatable_bond_count_ethanol() {
        let mut mol = parse_smiles("CCO").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);
    }

    #[test]
    fn test_rotatable_bond_count_propanol() {
        let mut mol = parse_smiles("CCCO").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    fn test_rotatable_bond_count_secondary_amide_keeps_n_alkyl_bond() {
        // RDKit excludes the acyl C(=O)-N bond itself but still counts the
        // exocyclic N-alkyl bond in a secondary amide.
        let mut mol = parse_smiles("CC(=O)NCC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    fn test_rotatable_bond_count_ester_carbonyl_o_excluded() {
        // Ethyl acetate: C(=O)-O is excluded by carbonyl-O rule, O-CH2 counted.
        let mut mol = parse_smiles("CC(=O)OCC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    fn test_rotatable_bond_count_methyl_acetate_zero() {
        // Methyl acetate: C(=O)-O excluded (carbonyl-O), O-CH3 terminal -> 0.
        let mut mol = parse_smiles("CC(=O)OC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);
    }

    #[test]
    fn test_rotatable_bond_count_formyl_c_n_bond_is_counted() {
        // Formyl C(=O)H-N bonds remain rotatable under RDKit strict semantics.
        let mut mol = parse_smiles("CNC=O").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);

        // Aromatic formyl imidazole-like nitrogens are also counted.
        let mut mol2 = parse_smiles("O=CN1C=CC=N1").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol2, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);

        // Tertiary formamide remains rotatable as well.
        let mut mol2 = parse_smiles("CN(C)C=O").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol2, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    fn test_rotatable_bond_count_hydrazide_nn_excluded_only_when_both_n_are_conjugated() {
        let mut mol2 = parse_smiles("CC(=O)NNC(=O)C").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol2, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);

        let mut mol3 = parse_smiles("CC(=O)NNC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol3, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    fn test_rotatable_bond_count_acetanilide_counts_n_aryl_bond() {
        let mut mol = parse_smiles("CC(=O)Nc1ccccc1").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    fn test_rotatable_bond_count_amidine_c_n_bond_excluded() {
        let mut mol = parse_smiles("CCNC(=N)N").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    fn test_rotatable_bond_count_formate_c_o_bond_is_counted() {
        let mut mol = parse_smiles("COC=O").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);

        let mut mol2 = parse_smiles("CCOC=O").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol2, &mut crate::desc::context::CalculationContext::new()).as_int(), 2);
    }

    #[test]
    fn test_rotatable_bond_count_aromatic_acyl_n_excluded_but_formyl_kept() {
        let mut mol = parse_smiles("CC(=O)N1C=CC=N1").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);

        let mut mol2 = parse_smiles("O=CN1C=CC=N1").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol2, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    fn test_rotatable_bond_count_tert_butyl_like_linkage_excluded() {
        let mut mol = parse_smiles("CC(C)(C)CC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);

        let mut mol2 = parse_smiles("CC(C)(C)OC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol2, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);
    }

    #[test]
    fn test_rotatable_bond_count_trihaloalkyl_linkage_excluded() {
        let mut mol = parse_smiles("ClC(Cl)(Cl)CC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);

        let mut mol2 = parse_smiles("FC(F)(F)CC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol2, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);

        let mut mol3 = parse_smiles("BrC(Br)(Br)CC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol3, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);

        // RDKit strict pattern excludes F/Cl/Br but intentionally not I.
        let mut mol4 = parse_smiles("IC(I)(I)CC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol4, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    fn test_rotatable_bond_count_cnp0329403_matches_rdkit() {
        let mut mol = parse_smiles(
            "O=C(N[C@@H](OC[C@@H]1CCCN2CCCC[C@H]12)C(Cl)(Cl)Cl)C1=CC=CC2=CC=CC=C12",
        )
        .unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 5);
    }

    #[test]
    fn test_rotatable_bond_count_non_tbutyl_quaternary_linkage_kept() {
        let mut mol = parse_smiles("CCC(C)(C)CC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 2);

        let mut mol2 = parse_smiles("CC(C)(O)CC").unwrap();
        assert_eq!(RotatableBondCount.calculate(&mut mol2, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    fn test_compute_bond_count() {
        let mol = parse_smiles("CCO").unwrap();
        assert_eq!(compute_bond_count(&mol), 2);
    }

    #[test]
    fn test_compute_rotatable_bond_count() {
        let mol = parse_smiles("CCCC").unwrap();
        assert_eq!(compute_rotatable_bond_count(&mol), 1);
    }
}
