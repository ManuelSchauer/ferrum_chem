//! Additional fraction and count descriptors commonly used in QSAR.
//!
//! Includes:
//! - FractionCSP3: fraction of sp³ carbons (drug-likeness)
//! - NumAmideBonds: amide bond count via SMARTS
//! - AromaticAtomFraction: fraction of atoms that are aromatic
//! - HallKierAlpha: molecular flexibility descriptor
//! - BertzCT: molecular complexity index
//! - Ipc: information content descriptor
//! - FragmentComplexity: Shannon entropy of element distribution
//!
//! References:
//! - Lovering et al. (2009) J. Med. Chem. 52, 6752 (FractionCSP3)
//! - Bertz (1981) JACS 103, 3599 (BertzCT)
//! - Kier & Hall (1986) Molecular Connectivity (HallKierAlpha)

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType};

// ---------------------------------------------------------------------------
// FractionCSP3
// ---------------------------------------------------------------------------

/// Fraction of sp³ carbons (carbon sp3 fraction).
///
/// FractionCSP3 = (number of sp3 carbons) / (total number of carbons).
/// Returns 0 for molecules with no carbons.
/// Drug-likeness metric: higher values → more 3D character.
pub struct FractionCSP3;

fn compute_fraction_csp3(mol: &Molecule) -> f32 {
    use crate::algo::electronic::effective_hybridization_pub;
    use crate::types::Hybridization;

    let mut total_c = 0u32;
    let mut sp3_c = 0u32;

    for (i, atom) in mol.atoms.iter().enumerate() {
        if atom.atomic_number != 6 {
            continue;
        }
        total_c += 1;

        let total_degree = mol.adjacency[i].len() as u8 + atom.total_h();
        let hyb = effective_hybridization_pub(mol, i);

        if total_degree == 4 && hyb == Hybridization::SP3 {
            sp3_c += 1;
        }
    }

    if total_c > 0 {
        sp3_c as f32 / total_c as f32
    } else {
        0.0
    }
}

impl Descriptor for FractionCSP3 {
    fn name(&self) -> &str {
        "FractionCSP3"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_fraction_csp3(mol))
    }
}

// ---------------------------------------------------------------------------
// Ring Type Counts (helper function only; typed descriptor in ring_types.rs)
// ---------------------------------------------------------------------------

fn ring_all_single_bonds(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    for i in 0..ring.len() {
        let a1 = ring[i];
        let a2 = ring[(i + 1) % ring.len()];
        let is_single = mol.adjacency[a1 as usize].iter().any(|&bi| {
            let bond = &mol.bonds[bi as usize];
            ((bond.source == a1 && bond.target == a2) || (bond.source == a2 && bond.target == a1))
                && bond.bond_type == BondType::Single
        });
        if !is_single {
            return false;
        }
    }
    true
}

/// Compute ring type counts: [aromatic, aliphatic, saturated, heteroaromatic, heterocyclic].
pub fn compute_ring_type_counts(mol: &Molecule) -> [u32; 5] {
    use crate::algo::rings::find_sssr;

    let mut counts = [0u32; 5];
    let mut mol_clone = mol.clone();
    let rings = find_sssr(&mut mol_clone);

    for ring in &rings {
        let all_aromatic = ring.iter().all(|&ai| mol.atoms[ai as usize].is_aromatic());
        let any_aromatic = ring.iter().any(|&ai| mol.atoms[ai as usize].is_aromatic());
        let has_heteroatom = ring.iter().any(|&ai| {
            let an = mol.atoms[ai as usize].atomic_number;
            an != 6 && an != 1
        });
        let all_single = ring_all_single_bonds(mol, ring);

        if all_aromatic {
            counts[0] += 1;
        }
        if !any_aromatic {
            counts[1] += 1;
        }
        if all_single {
            counts[2] += 1;
        }
        if all_aromatic && has_heteroatom {
            counts[3] += 1;
        }
        if has_heteroatom {
            counts[4] += 1;
        }
    }

    counts
}

// ---------------------------------------------------------------------------
// NumAmideBonds
// ---------------------------------------------------------------------------

/// Count of amide bonds: [NX3][CX3](=[OX1])
pub struct NumAmideBonds;

fn compute_num_amide_bonds(mol: &Molecule) -> u32 {
    let mut count = 0u32;

    for bond in &mol.bonds {
        if bond.bond_type != crate::types::BondType::Single {
            continue;
        }

        let a = bond.source as usize;
        let b = bond.target as usize;
        let (n_idx, c_idx) = match (
            mol.atoms[a].atomic_number,
            mol.atoms[a].is_aromatic(),
            mol.atoms[b].atomic_number,
            mol.atoms[b].is_aromatic(),
        ) {
            (7, false, 6, false) => (a, b),
            (6, false, 7, false) => (b, a),
            _ => continue,
        };

        if !mol.adjacency[c_idx].iter().any(|&other_bond_idx| {
            let other_bond = &mol.bonds[other_bond_idx as usize];
            if other_bond.bond_type != crate::types::BondType::Double {
                return false;
            }

            let other_idx = if other_bond.source == c_idx as u16 {
                other_bond.target as usize
            } else {
                other_bond.source as usize
            };

            other_idx != n_idx
                && mol.atoms[other_idx].atomic_number == 8
                && !mol.atoms[other_idx].is_aromatic()
        }) {
            continue;
        }

        count += 1;
    }

    count
}

impl Descriptor for NumAmideBonds {
    fn name(&self) -> &str {
        "NumAmideBonds"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Int(compute_num_amide_bonds(mol))
    }
}

// ---------------------------------------------------------------------------
// AromaticAtomFraction
// ---------------------------------------------------------------------------

/// Fraction of heavy atoms that are aromatic.
pub struct AromaticAtomFraction;

fn compute_aromatic_atom_fraction(mol: &Molecule) -> f32 {
    let heavy: usize = mol.atoms.iter().filter(|a| a.atomic_number > 1).count();
    if heavy == 0 {
        return 0.0;
    }
    let aromatic: usize = mol
        .atoms
        .iter()
        .filter(|a| a.atomic_number > 1 && a.is_aromatic())
        .count();
    aromatic as f32 / heavy as f32
}

impl Descriptor for AromaticAtomFraction {
    fn name(&self) -> &str {
        "AromaticAtomFraction"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_aromatic_atom_fraction(mol))
    }
}

// ---------------------------------------------------------------------------
// HallKierAlpha
// ---------------------------------------------------------------------------

/// Hall-Kier alpha: molecular flexibility/shape modifier.
///
/// Uses hybridization-dependent lookup table for C/N/O/S/P/F/Cl/Br/I/H.
/// For other elements: α = Rb0(element)/Rb0(C) - 1.
pub struct HallKierAlpha;

fn compute_hall_kier_alpha(mol: &Molecule) -> f32 {
    use crate::algo::electronic::effective_hybridization_pub;
    use crate::types::Hybridization;

    let mut alpha_sum = 0.0f64;
    let rb0_c = 0.77f64;

    // Lookup table: [SP, SP2, SP3] alpha values per element.
    // None means use the last non-None value.
    for (i, atom) in mol.atoms.iter().enumerate() {
        if atom.atomic_number == 0 {
            continue;
        }
        if atom.atomic_number == 1 {
            continue;
        } // H contributes 0

        let hyb = effective_hybridization_pub(mol, i);
        // hyb index: SP=0, SP2=1, SP3=2
        let hyb_idx = match hyb {
            Hybridization::SP => 0usize,
            Hybridization::SP2 => 1,
            _ => 2, // SP3 and others
        };

        let alpha = match atom.atomic_number {
            6 => {
                // C: [-0.22, -0.13, 0.0]
                [-0.22, -0.13, 0.0][hyb_idx]
            }
            7 => {
                // N: [-0.29, -0.20, -0.04]
                [-0.29, -0.20, -0.04][hyb_idx]
            }
            8 => {
                // O: [None, -0.20, -0.04] → SP uses -0.04
                match hyb_idx {
                    1 => -0.20,
                    _ => -0.04,
                }
            }
            16 => {
                // S: [None, 0.22, 0.35]
                match hyb_idx {
                    1 => 0.22,
                    _ => 0.35,
                }
            }
            15 => {
                // P: [None, 0.30, 0.43]
                match hyb_idx {
                    1 => 0.30,
                    _ => 0.43,
                }
            }
            9 => -0.07, // F: always -0.07
            17 => 0.29, // Cl: always 0.29
            35 => 0.48, // Br: always 0.48
            53 => 0.73, // I: always 0.73
            _ => {
                // Fallback: Rb0(element)/Rb0(C) - 1
                let ra = crate::desc::vsa::rb0_radius(atom.atomic_number);
                ra / rb0_c - 1.0
            }
        };
        alpha_sum += alpha;
    }
    alpha_sum as f32
}

/// Public f64 accessor for Kappa index computation.
pub fn compute_hall_kier_alpha_f64(mol: &Molecule) -> f64 {
    use crate::algo::electronic::effective_hybridization_pub;
    use crate::types::Hybridization;

    let mut alpha_sum = 0.0f64;
    let rb0_c = 0.77f64;

    for (i, atom) in mol.atoms.iter().enumerate() {
        if atom.atomic_number == 0 || atom.atomic_number == 1 {
            continue;
        }

        let hyb = effective_hybridization_pub(mol, i);
        let hyb_idx = match hyb {
            Hybridization::SP => 0usize,
            Hybridization::SP2 => 1,
            _ => 2,
        };

        let alpha = match atom.atomic_number {
            6 => [-0.22, -0.13, 0.0][hyb_idx],
            7 => [-0.29, -0.20, -0.04][hyb_idx],
            8 => match hyb_idx {
                1 => -0.20,
                _ => -0.04,
            },
            16 => match hyb_idx {
                1 => 0.22,
                _ => 0.35,
            },
            15 => match hyb_idx {
                1 => 0.30,
                _ => 0.43,
            },
            9 => -0.07,
            17 => 0.29,
            35 => 0.48,
            53 => 0.73,
            _ => {
                let ra = crate::desc::vsa::rb0_radius(atom.atomic_number);
                ra / rb0_c - 1.0
            }
        };
        alpha_sum += alpha;
    }
    alpha_sum
}

impl Descriptor for HallKierAlpha {
    fn name(&self) -> &str {
        "HallKierAlpha"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_hall_kier_alpha(mol))
    }
}

// ---------------------------------------------------------------------------
// BertzCT - Molecular Complexity
// ---------------------------------------------------------------------------

/// Bertz complexity index CT.
///
/// Consists of a sum of two terms:
/// 1. Connection entropy: complexity of the bonding (using symmetry classes)
/// 2. Atom type entropy: complexity of the heteroatom distribution
///
/// Uses bond-order-weighted distance matrix for symmetry class assignment.
/// Aromatic bonds are treated as order 1.5.
///
/// Reference: S. H. Bertz, J. Am. Chem. Soc., 103, 3599-3601 (1981)
pub struct BertzCT;

fn compute_bertz_ct(mol: &Molecule) -> f32 {
    let n = mol.num_atoms();
    if n < 2 {
        return 0.0;
    }

    // Step 1: Build neighbor list and bond order lookup
    let num_bonds = mol.num_bonds();
    let mut degrees = vec![0usize; n];
    let mut neighbors: Vec<Vec<usize>> = vec![vec![]; n];
    // bond_orders[(min,max)] = order
    let mut bond_orders = std::collections::HashMap::new();

    for bond in &mol.bonds {
        let a1 = bond.source as usize;
        let a2 = bond.target as usize;
        let (lo, hi) = if a1 < a2 { (a1, a2) } else { (a2, a1) };
        let bo: f64 = if bond.is_aromatic() || bond.bond_type == BondType::Aromatic {
            1.5
        } else {
            match bond.bond_type {
                BondType::Double => 2.0,
                BondType::Triple => 3.0,
                BondType::Dative => 17.0, // Matches RDKit: float(BondType.DATIVE) = 17
                _ => 1.0,
            }
        };
        bond_orders.insert((lo, hi), bo);
        if !neighbors[a1].contains(&a2) {
            neighbors[a1].push(a2);
        }
        if !neighbors[a2].contains(&a1) {
            neighbors[a2].push(a1);
        }
        degrees[a1] += 1;
        degrees[a2] += 1;
    }
    for nbrs in &mut neighbors {
        nbrs.sort();
    }

    // Step 2: Symmetry classes from bond-order-weighted distance matrix
    let bo_dist = super::balaban::compute_distance_matrix(mol, true);
    let mut sym_classes = vec![0usize; n];
    let mut keys_seen: Vec<Vec<String>> = vec![];
    for i in 0..n {
        let mut dv: Vec<f64> = (0..n).map(|j| bo_dist[i * n + j]).collect();
        dv.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let key: Vec<String> = dv.iter().map(|x| format!("{:.4}", x)).collect();
        if let Some(idx) = keys_seen.iter().position(|k| k == &key) {
            sym_classes[i] = idx + 1;
        } else {
            keys_seen.push(key);
            sym_classes[i] = keys_seen.len();
        }
    }

    // Step 3: Connection counting
    let mut connection_dict: std::collections::HashMap<Vec<usize>, f64> =
        std::collections::HashMap::new();
    let mut atom_type_counts: std::collections::HashMap<u8, u32> = std::collections::HashMap::new();

    for atom_idx in 0..n {
        let z = mol.atoms[atom_idx].atomic_number;
        *atom_type_counts.entry(z).or_insert(0) += 1;

        let hinge_class = sym_classes[atom_idx];
        let nbrs = &neighbors[atom_idx];
        let num_nbrs = nbrs.len();

        for i in 0..num_nbrs {
            let ni = nbrs[i];
            let ni_class = sym_classes[ni];
            let (lo_i, hi_i) = if atom_idx < ni {
                (atom_idx, ni)
            } else {
                (ni, atom_idx)
            };
            let bo_i = *bond_orders.get(&(lo_i, hi_i)).unwrap_or(&1.0);

            // Extra connections from multi-order bonds
            if bo_i > 1.0 && ni > atom_idx {
                let num_conn = bo_i * (bo_i - 1.0) / 2.0;
                let key = vec![ni_class.min(hinge_class), ni_class.max(hinge_class)];
                *connection_dict.entry(key).or_insert(0.0) += num_conn;
            }

            // Pairs of neighbors through hinge
            for j in (i + 1)..num_nbrs {
                let nj = nbrs[j];
                let nj_class = sym_classes[nj];
                let (lo_j, hi_j) = if atom_idx < nj {
                    (atom_idx, nj)
                } else {
                    (nj, atom_idx)
                };
                let bo_j = *bond_orders.get(&(lo_j, hi_j)).unwrap_or(&1.0);
                let num_conn = bo_i * bo_j;
                let key = vec![ni_class.min(nj_class), hinge_class, ni_class.max(nj_class)];
                *connection_dict.entry(key).or_insert(0.0) += num_conn;
            }
        }
    }

    if connection_dict.is_empty() {
        connection_dict.insert(vec![0], 1.0);
    }

    // Step 4: Calculate entropies
    let conn_values: Vec<f64> = connection_dict.values().copied().collect();
    let tot_conn: f64 = conn_values.iter().sum();
    let conn_ie = if tot_conn > 0.0 {
        let ie = info_entropy(&conn_values);
        tot_conn * (ie + tot_conn.log2())
    } else {
        0.0
    };

    let atom_values: Vec<f64> = atom_type_counts.values().map(|&v| v as f64).collect();
    let atom_ie = n as f64 * info_entropy(&atom_values);

    (conn_ie + atom_ie) as f32
}

/// Shannon information entropy: H = -sum(p_i * log2(p_i))
fn info_entropy(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for &c in counts {
        if c > 0.0 {
            let p = c / total;
            h -= p * p.log2();
        }
    }
    h
}

impl Descriptor for BertzCT {
    fn name(&self) -> &str {
        "BertzCT"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_bertz_ct(mol))
    }
}

fn shannon_entropy(counts: &[u32]) -> f64 {
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / t;
            -p * p.log2()
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Ipc - Information Content
// ---------------------------------------------------------------------------

/// Information content of the characteristic polynomial (Bonchev-Trinajstić).
///
/// Ipc = sum(|coeffs|) * InfoEntropy(|coeffs|)
/// where coeffs are the characteristic polynomial coefficients of the adjacency matrix,
/// computed via the Le Verrier-Faddeev-Frame method.
///
/// Reference: D. Bonchev & N. Trinajstic, J. Chem. Phys. 67, 4517-4533 (1977)
pub struct Ipc;

fn compute_ipc(mol: &Molecule) -> f64 {
    let n = mol.num_atoms();
    if n < 2 {
        return 0.0;
    }

    // Build simple adjacency matrix (unweighted graph).
    // Use direct bond connectivity instead of distance reconstruction.
    let mut adj = vec![0.0f64; n * n];
    for bond in &mol.bonds {
        let i = bond.source as usize;
        let j = bond.target as usize;
        adj[i * n + j] = 1.0;
        adj[j * n + i] = 1.0;
    }

    // Characteristic polynomial via Le Verrier-Faddeev-Frame
    let mut cpoly = vec![0.0f64; n + 1];
    cpoly[0] = 1.0;

    // an = current matrix power (A^n in the recurrence)
    let mut an = adj.clone();

    for k in 1..=n {
        // res[k] = (1/k) * trace(an)
        let mut trace = 0.0f64;
        for i in 0..n {
            trace += an[i * n + i];
        }
        cpoly[k] = trace / k as f64;

        // bn = an - cpoly[k] * I
        let mut bn = an.clone();
        for i in 0..n {
            bn[i * n + i] -= cpoly[k];
        }

        // an = A * bn (matrix multiplication)
        if k < n {
            let mut next = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0f64;
                    for l in 0..n {
                        s += adj[i * n + l] * bn[l * n + j];
                    }
                    next[i * n + j] = s;
                }
            }
            an = next;
        }
    }

    // Negate coefficients 1..n (as per the algorithm)
    for i in 1..=n {
        cpoly[i] = -cpoly[i];
    }

    // Take absolute values
    let abs_cpoly: Vec<f64> = cpoly.iter().map(|&x| x.abs()).collect();

    // Ipc = sum(|coeffs|) * InfoEntropy(|coeffs|)
    let total: f64 = abs_cpoly.iter().sum();
    let ie = info_entropy(&abs_cpoly);
    total * ie
}

impl Descriptor for Ipc {
    fn name(&self) -> &str {
        "Ipc"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float64(compute_ipc(mol))
    }
}

// ---------------------------------------------------------------------------
// FragmentComplexity
// ---------------------------------------------------------------------------

/// Fragment complexity: Shannon entropy of atomic number distribution.
///
/// Measures elemental diversity in the molecule.
/// FC = -Σ pᵢ × log₂(pᵢ) where pᵢ = fraction of atoms with element i.
pub struct FragmentComplexity;

fn compute_fragment_complexity(mol: &Molecule) -> f32 {
    let mut element_counts = [0u32; 120];
    for atom in &mol.atoms {
        if atom.atomic_number > 0 {
            element_counts[atom.atomic_number as usize] += 1;
        }
    }
    shannon_entropy(&element_counts) as f32
}

impl Descriptor for FragmentComplexity {
    fn name(&self) -> &str {
        "FragmentComplexity"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_fragment_complexity(mol))
    }
}

// ---------------------------------------------------------------------------
// Thread safety
// ---------------------------------------------------------------------------
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<FractionCSP3>;
    let _ = assert_send_sync::<NumAmideBonds>;
    let _ = assert_send_sync::<AromaticAtomFraction>;
    let _ = assert_send_sync::<HallKierAlpha>;
    let _ = assert_send_sync::<BertzCT>;
    let _ = assert_send_sync::<Ipc>;
    let _ = assert_send_sync::<FragmentComplexity>;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_float(desc: &dyn Descriptor, smiles: &str) -> f32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        desc.calculate(&mut mol).as_float()
    }

    fn calc_int(desc: &dyn Descriptor, smiles: &str) -> u32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        desc.calculate(&mut mol).as_int()
    }

    // -- FractionCSP3 --

    #[test]
    fn test_fsp3_ethane() {
        let v = calc_float(&FractionCSP3, "CC");
        assert!((v - 1.0).abs() < 0.01, "all sp3: {}", v);
    }

    #[test]
    fn test_fsp3_ethene() {
        let v = calc_float(&FractionCSP3, "C=C");
        assert!(v.abs() < 0.01, "all sp2: {}", v);
    }

    #[test]
    fn test_fsp3_propanol() {
        let v = calc_float(&FractionCSP3, "CCCO");
        assert!((v - 1.0).abs() < 0.01, "all sp3: {}", v);
    }

    #[test]
    fn test_fsp3_benzene() {
        let v = calc_float(&FractionCSP3, "c1ccccc1");
        assert!(v.abs() < 0.01, "no sp3 carbons: {}", v);
    }

    #[test]
    fn test_fsp3_toluene() {
        let v = calc_float(&FractionCSP3, "Cc1ccccc1");
        assert!((v - 1.0 / 7.0).abs() < 0.02, "fsp3 = {}", v);
    }

    #[test]
    fn test_fsp3_trivalent_radical_carbon_excluded() {
        let v = calc_float(&FractionCSP3, "[C](C)(C)C");
        assert!((v - 0.75).abs() < 0.01, "fsp3 = {}", v);
    }

    // -- RingTypeCounts (helper function) --

    #[test]
    fn test_ring_types_benzene() {
        let mol = parse_smiles("c1ccccc1").expect("valid");
        let v = compute_ring_type_counts(&mol);
        assert_eq!(v[0], 1, "1 aromatic");
        assert_eq!(v[1], 0, "0 aliphatic");
        assert_eq!(v[2], 0, "0 saturated");
    }

    #[test]
    fn test_ring_types_cyclohexane() {
        let mol = parse_smiles("C1CCCCC1").expect("valid");
        let v = compute_ring_type_counts(&mol);
        assert_eq!(v[0], 0, "0 aromatic");
        assert_eq!(v[1], 1, "1 aliphatic");
        assert_eq!(v[2], 1, "1 saturated");
    }

    #[test]
    fn test_ring_types_pyridine() {
        let mol = parse_smiles("c1ccncc1").expect("valid");
        let v = compute_ring_type_counts(&mol);
        assert_eq!(v[0], 1, "1 aromatic");
        assert_eq!(v[3], 1, "1 heteroaromatic");
        assert_eq!(v[4], 1, "1 heterocyclic");
    }

    #[test]
    fn test_ring_types_morpholine() {
        let mol = parse_smiles("C1COCCN1").expect("valid");
        let v = compute_ring_type_counts(&mol);
        assert_eq!(v[2], 1, "1 saturated");
        assert_eq!(v[4], 1, "1 heterocyclic");
    }

    // -- NumAmideBonds --

    #[test]
    fn test_amide_acetamide() {
        assert_eq!(calc_int(&NumAmideBonds, "CC(=O)N"), 1);
    }

    #[test]
    fn test_amide_ethane() {
        assert_eq!(calc_int(&NumAmideBonds, "CC"), 0);
    }

    #[test]
    fn test_amide_peptide() {
        let v = calc_int(&NumAmideBonds, "NCC(=O)NCC(=O)O");
        assert!(v >= 1, "should have amide bonds: {}", v);
    }

    #[test]
    fn test_amide_caffeine_like_ureide_counts_imino_n_bond() {
        assert_eq!(calc_int(&NumAmideBonds, "CN1C(=O)N=C2N=CN=C2C1=O"), 3);
    }

    #[test]
    fn test_amide_barbituric_acid_aromatic_ureide_does_not_count() {
        assert_eq!(calc_int(&NumAmideBonds, "O=C1NC(=O)NC(=O)N1"), 0);
    }

    #[test]
    fn test_amide_non_aromatic_imino_n_counts() {
        assert_eq!(
            calc_int(
                &NumAmideBonds,
                "COC1=CC=C(CCN2C=C(C(=O)N=C3N=CC=N3)C3=CC=CC=C3C2=O)C=C1",
            ),
            1
        );
    }

    // -- AromaticAtomFraction --

    #[test]
    fn test_arom_frac_benzene() {
        let v = calc_float(&AromaticAtomFraction, "c1ccccc1");
        assert!((v - 1.0).abs() < 0.01, "all aromatic: {}", v);
    }

    #[test]
    fn test_arom_frac_hexane() {
        let v = calc_float(&AromaticAtomFraction, "CCCCCC");
        assert!(v.abs() < 0.01, "no aromatic: {}", v);
    }

    #[test]
    fn test_arom_frac_toluene() {
        let v = calc_float(&AromaticAtomFraction, "Cc1ccccc1");
        assert!((v - 6.0 / 7.0).abs() < 0.02, "frac = {}", v);
    }

    // -- HallKierAlpha --

    #[test]
    fn test_hka_all_carbon() {
        let v = calc_float(&HallKierAlpha, "CCCC");
        assert!(v.abs() < 0.01, "all-C alpha = {}", v);
    }

    #[test]
    fn test_hka_oxygen() {
        let v = calc_float(&HallKierAlpha, "CCO");
        assert!(v < 0.0, "O makes alpha negative: {}", v);
    }

    #[test]
    fn test_hka_bromine() {
        let v = calc_float(&HallKierAlpha, "CCBr");
        assert!(v > 0.0, "Br makes alpha positive: {}", v);
    }

    // Exact-value tests that encode the three-case O/S norbs==4 rule.
    // Expected values derived from the alpha lookup table:
    //   C_SP3=0.0, C_SP2=-0.13, O_SP2=-0.20, O_SP3=-0.04, P_SP3=0.43
    //
    // These are the cases that were systematically wrong before the fix
    // (ester/phosphate oxygens were promoted to SP2 → alpha too negative).

    #[test]
    fn test_hka_ester_bridging_o_sp2() {
        // CC(=O)OC — methyl acetate
        // C_SP3(0) + C_SP2(-0.13) + O_SP2(-0.20) + O_SP2(-0.20) + C_SP3(0) = -0.53
        // The bridging O has a C=O neighbour (not hypervalent) → SP2, RDKit-confirmed.
        let v = calc_float(&HallKierAlpha, "CC(=O)OC");
        assert!(
            (v - (-0.53_f32)).abs() < 0.01,
            "methyl acetate HallKierAlpha expected -0.53, got {v}"
        );
    }

    #[test]
    fn test_hka_carboxylic_acid_oh_sp2() {
        // CC(=O)O — acetic acid
        // C_SP3(0) + C_SP2(-0.13) + O_SP2(-0.20) + O_SP2(-0.20) = -0.53
        // COOH-OH: C(=O) neighbour (not hypervalent P/S) → SP2.
        let v = calc_float(&HallKierAlpha, "CC(=O)O");
        assert!(
            (v - (-0.53_f32)).abs() < 0.01,
            "acetic acid HallKierAlpha expected -0.53, got {v}"
        );
    }

    #[test]
    fn test_hka_phenol_o_sp2() {
        // c1ccc(O)cc1 — phenol
        // 6×C_arom(-0.13) + O_SP2(-0.20) = -0.98
        // Phenol O has an aromatic C neighbour → SP2 (Case 1).
        let v = calc_float(&HallKierAlpha, "c1ccc(O)cc1");
        assert!(
            (v - (-0.98_f32)).abs() < 0.01,
            "phenol HallKierAlpha expected -0.98, got {v}"
        );
    }

    #[test]
    fn test_hka_ether_o_sp3() {
        // COC — dimethyl ether
        // C_SP3(0) + O_SP3(-0.04) + C_SP3(0) = -0.04
        // Ether O has 2 aliphatic C neighbours and no pi → SP3.
        let v = calc_float(&HallKierAlpha, "COC");
        assert!(
            (v - (-0.04_f32)).abs() < 0.01,
            "dimethyl ether HallKierAlpha expected -0.04, got {v}"
        );
    }

    #[test]
    fn test_hka_phosphoric_acid_oh_sp3() {
        // OP(=O)(O)O — phosphoric acid
        // P_SP3(0.43) + O_SP2(-0.20) + 3×O_SP3(-0.04) = 0.11
        // All three P–OH oxygens have P (non-C) neighbour → SP3.
        let v = calc_float(&HallKierAlpha, "OP(=O)(O)O");
        assert!(
            (v - 0.11_f32).abs() < 0.01,
            "phosphoric acid HallKierAlpha expected 0.11, got {v}"
        );
    }

    #[test]
    fn test_hka_dimethyl_phosphate_o_sp3() {
        // COP(=O)(OC)O — dimethyl phosphate
        // 2×C_SP3(0) + 2×O_bridge_SP3(-0.04) + P_SP3(0.43) + O_SP2(-0.20) + O_OH_SP3(-0.04) = 0.11
        // Both C–O–P bridging oxygens: P neighbour excluded (hypervalent),
        // sp3 C neighbour has no pi → SP3.
        // The P–OH also has only P neighbour → SP3.
        let v = calc_float(&HallKierAlpha, "COP(=O)(OC)O");
        assert!(
            (v - 0.11_f32).abs() < 0.01,
            "dimethyl phosphate HallKierAlpha expected 0.11, got {v}"
        );
    }

    #[test]
    fn test_hka_allene_central_carbon_sp() {
        let v = calc_float(&HallKierAlpha, "C=C=C");
        assert!(
            (v - (-0.48_f32)).abs() < 0.01,
            "allene HallKierAlpha = {}",
            v
        );
    }

    #[test]
    fn test_hka_trivalent_radical_carbon_stays_sp3() {
        let v = calc_float(&HallKierAlpha, "[C](C)(C)C");
        assert!(v.abs() < 0.01, "tert-butyl radical HallKierAlpha = {}", v);
    }

    // -- BertzCT --

    #[test]
    fn test_bertz_methane() {
        let v = calc_float(&BertzCT, "C");
        assert!(v.abs() < 0.01, "methane CT = {}", v);
    }

    #[test]
    fn test_bertz_complex_gt_simple() {
        let simple = calc_float(&BertzCT, "CCCC");
        let complex = calc_float(&BertzCT, "c1ccc(O)c(N)c1");
        assert!(complex > simple, "complex {} > simple {}", complex, simple);
    }

    #[test]
    fn test_bertz_benzene() {
        let v = calc_float(&BertzCT, "c1ccccc1");
        // RDKit: 71.961
        assert!((v - 71.96).abs() < 0.1, "benzene CT = {}", v);
    }

    // -- Ipc --

    #[test]
    fn test_ipc_ethane() {
        let v = calc_float(&Ipc, "CC");
        // RDKit: 2.0
        assert!((v - 2.0).abs() < 0.01, "ethane Ipc = {}", v);
    }

    #[test]
    fn test_ipc_diverse() {
        let v = calc_float(&Ipc, "CC=O");
        assert!(v > 0.0, "should have info content: {}", v);
    }

    // -- FragmentComplexity --

    #[test]
    fn test_fc_all_carbon() {
        let v = calc_float(&FragmentComplexity, "CCCC");
        assert!(v.abs() < 0.01, "all-C FC = {}", v);
    }

    #[test]
    fn test_fc_diverse() {
        let v = calc_float(&FragmentComplexity, "CCNO");
        assert!(v > 0.0, "diverse FC = {}", v);
    }

    #[test]
    fn test_fc_more_diverse_higher() {
        let less = calc_float(&FragmentComplexity, "CCCC");
        let more = calc_float(&FragmentComplexity, "CCNO");
        assert!(more > less);
    }
}
