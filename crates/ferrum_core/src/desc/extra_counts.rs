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
use crate::types::BondType;
use smallvec::SmallVec;
use std::hash::{BuildHasherDefault, Hasher};

/// Minimal hasher for `u64` keys used in Bertz connection counting.
///
/// Rationale:
/// - The connection map key is already a packed `u64` with good bit distribution.
/// - Using identity hashing avoids SipHash overhead in a very hot loop.
/// - This is safe in this context because keys are generated internally from molecular
///   topology (not attacker-controlled external input).
#[derive(Default)]
struct U64IdentityHasher(u64);

impl Hasher for U64IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Fallback path for non-u64 writes. Bertz connection map keys use `u64`,
        // so this should not be on the hot path.
        let mut state = 0u64;
        let mut shift = 0u32;
        for &byte in bytes {
            state ^= (byte as u64) << shift;
            shift = (shift + 8) & 63;
        }
        self.0 = state;
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = i as u64;
    }
}

type U64FastMap<V> = std::collections::HashMap<u64, V, BuildHasherDefault<U64IdentityHasher>>;

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

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::Float(compute_fraction_csp3(mol))
    }
}

// ---------------------------------------------------------------------------
// Ring Type Counts (helper function only; typed descriptor in ring_types.rs)
// ---------------------------------------------------------------------------

fn ring_all_single_bonds(mol: &Molecule, ring: &[usize]) -> bool {
    ring.iter()
        .all(|&bond_idx| mol.bonds[bond_idx].bond_type == BondType::Single)
}

/// Compute ring type counts: [aromatic, aliphatic, saturated, heteroaromatic, heterocyclic].
pub fn compute_ring_type_counts(mol: &Molecule) -> [u32; 5] {
    use crate::algo::rings::find_minimum_cycle_basis_bonds;

    let mut counts = [0u32; 5];
    let rings = find_minimum_cycle_basis_bonds(mol);

    for ring in &rings {
        let all_aromatic = ring
            .iter()
            .all(|&bond_idx| mol.bonds[bond_idx].is_aromatic());
        let any_aromatic = ring
            .iter()
            .any(|&bond_idx| mol.bonds[bond_idx].is_aromatic());
        let has_heteroatom = ring.iter().any(|&bond_idx| {
            let bond = &mol.bonds[bond_idx];
            let src_an = mol.atoms[bond.source as usize].atomic_number;
            let dst_an = mol.atoms[bond.target as usize].atomic_number;
            (src_an != 6 && src_an != 1) || (dst_an != 6 && dst_an != 1)
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

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
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

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
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

const fn build_hall_kier_alpha_table() -> [[f64; 3]; 119] {
    let mut table = [[f64::NAN; 3]; 119];
    table[6] = [-0.22, -0.13, 0.0];
    table[7] = [-0.29, -0.20, -0.04];
    table[8] = [-0.04, -0.20, -0.04];
    table[16] = [0.35, 0.22, 0.35];
    table[15] = [0.43, 0.30, 0.43];
    table[9] = [-0.07, -0.07, -0.07];
    table[17] = [0.29, 0.29, 0.29];
    table[35] = [0.48, 0.48, 0.48];
    table[53] = [0.73, 0.73, 0.73];
    table
}

const HALL_KIER_ALPHA_TABLE: [[f64; 3]; 119] = build_hall_kier_alpha_table();

#[inline]
fn hall_kier_alpha_lookup(atomic_number: u8, hyb_idx: usize) -> Option<f64> {
    let value = HALL_KIER_ALPHA_TABLE[atomic_number as usize][hyb_idx];
    if value.is_nan() {
        None
    } else {
        Some(value)
    }
}

fn compute_hall_kier_alpha_sum(mol: &Molecule) -> f64 {
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

        let alpha = hall_kier_alpha_lookup(atom.atomic_number, hyb_idx).unwrap_or_else(|| {
            let ra = crate::desc::vsa::rb0_radius(atom.atomic_number);
            ra / rb0_c - 1.0
        });
        alpha_sum += alpha;
    }
    alpha_sum
}

fn compute_hall_kier_alpha(mol: &Molecule) -> f32 {
    compute_hall_kier_alpha_sum(mol) as f32
}

/// Public f64 accessor for Kappa index computation.
pub fn compute_hall_kier_alpha_f64(mol: &Molecule) -> f64 {
    compute_hall_kier_alpha_sum(mol)
}

impl Descriptor for HallKierAlpha {
    fn name(&self) -> &str {
        "HallKierAlpha"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
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

/// Encodes a 2-class Bertz connection key into a compact `u64`.
///
/// Layout:
/// - bit 63 = 1 (pair discriminator)
/// - bits 31..16 = lower class id
/// - bits 15..0  = higher class id
///
/// Class ids originate from atom-index bounded symmetry classes (`<= u16::MAX`),
/// so this packing is lossless for Ferrum's molecule-size constraints.
#[inline]
fn pack_bertz_pair_key(class_lo: usize, class_hi: usize) -> u64 {
    debug_assert!(class_lo <= u16::MAX as usize);
    debug_assert!(class_hi <= u16::MAX as usize);
    (1u64 << 63) | ((class_lo as u64) << 16) | class_hi as u64
}

/// Encodes a 3-class Bertz connection key into a compact `u64`.
///
/// Layout:
/// - bit 62 = 1 (triple discriminator)
/// - bits 47..32 = first class id (min neighbor class)
/// - bits 31..16 = hinge class id
/// - bits 15..0  = third class id (max neighbor class)
///
/// Bit 63 remains 0, so pair and triple key-spaces are disjoint.
#[inline]
fn pack_bertz_triple_key(class_a: usize, class_b: usize, class_c: usize) -> u64 {
    debug_assert!(class_a <= u16::MAX as usize);
    debug_assert!(class_b <= u16::MAX as usize);
    debug_assert!(class_c <= u16::MAX as usize);
    (1u64 << 62) | ((class_a as u64) << 32) | ((class_b as u64) << 16) | class_c as u64
}

fn compute_bertz_ct(mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> f32 {
    let n = mol.num_atoms();
    if n < 2 {
        return 0.0;
    }

    // Step 1: Build an adjacency list that carries the effective bond-order weight used by
    // Bertz connection counting.
    let neighbors = &mut ctx.weighted_neighbors;
    neighbors.clear();
    neighbors.resize(n, SmallVec::new());

    for bond in &mol.bonds {
        let a1 = bond.source as usize;
        let a2 = bond.target as usize;
        let bo: f64 = if bond.is_aromatic() || bond.bond_type == BondType::Aromatic {
            1.5
        } else {
            match bond.bond_type {
                BondType::Double => 2.0,
                BondType::Triple => 3.0,
                BondType::Dative => 17.0,
                _ => 1.0,
            }
        };
        neighbors[a1].push((a2, bo));
        neighbors[a2].push((a1, bo));
    }

    // Step 2: Symmetry classes from bond-order-weighted distance matrix.
    super::balaban::ensure_bo_distance_matrix_cached(mol);
    let bo_dist = super::balaban::cached_bo_distance_matrix_ref(mol)
        .expect("Balaban BO distance cache initialized");
        
    ctx.bfs_dist.clear();
    ctx.bfs_dist.resize(n, 0);
    let sym_classes = &mut ctx.bfs_dist;
    let cutoff = n.min(100);
    
    let mut class_by_key: std::collections::HashMap<SmallVec<[i32; 64]>, usize> = 
        std::collections::HashMap::with_capacity(n);
    let mut next_class = 1usize;
    
    let sorted_distances = &mut ctx.vec_a;
    let mut key_buf = SmallVec::<[i32; 64]>::new();
    for i in 0..n {
        sorted_distances.clear();
        sorted_distances.extend_from_slice(&bo_dist[i * n..(i + 1) * n]);
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        key_buf.clear();
        key_buf.extend(
            sorted_distances
                .iter()
                .take(cutoff)
                .map(|x| (x * 10_000.0).round() as i32),
        );
        let class = if let Some(&class) = class_by_key.get(&key_buf) {
            class
        } else {
            let class = next_class;
            next_class += 1;
            class_by_key.insert(key_buf.clone(), class);
            class
        };
        sym_classes[i] = class;
    }

    // Step 3: Connection counting.
    //
    // Pair/triple keys are packed into a single integer-key map to keep per-update
    // hashing and key materialization costs low in this central nested loop.
    // We also maintain `tot_conn` incrementally, avoiding an extra full-map pass.
    let mut connection_counts: U64FastMap<f64> = U64FastMap::default();
    connection_counts.reserve(mol.num_bonds().saturating_mul(4).max(16));
    let mut atom_type_counts = [0u32; 256];
    let mut tot_conn = 0.0f64;

    for atom_idx in 0..n {
        let z = mol.atoms[atom_idx].atomic_number;
        atom_type_counts[z as usize] += 1;

        let hinge_class = sym_classes[atom_idx];
        let nbrs = &neighbors[atom_idx];
        let num_nbrs = nbrs.len();

        for i in 0..num_nbrs {
            let (ni, bo_i) = nbrs[i];
            let ni_class = sym_classes[ni];

            // Extra connections from multi-order bonds
            if bo_i > 1.0 && ni > atom_idx {
                let num_conn = bo_i * (bo_i - 1.0) / 2.0;
                let key = pack_bertz_pair_key(ni_class.min(hinge_class), ni_class.max(hinge_class));
                *connection_counts.entry(key).or_insert(0.0) += num_conn;
                tot_conn += num_conn;
            }

            // Pairs of neighbors through hinge
            for j in (i + 1)..num_nbrs {
                let (nj, bo_j) = nbrs[j];
                let nj_class = sym_classes[nj];
                let num_conn = bo_i * bo_j;
                let key = pack_bertz_triple_key(
                    ni_class.min(nj_class),
                    hinge_class,
                    ni_class.max(nj_class),
                );
                *connection_counts.entry(key).or_insert(0.0) += num_conn;
                tot_conn += num_conn;
            }
        }
    }

    // Step 4: Calculate entropies.
    //
    // Connection entropy is computed in closed form to reduce hot-loop arithmetic:
    //
    // H = -sum_i p_i log2(p_i), p_i = c_i / T
    // T * H           == T*log2(T) - sum_i c_i*log2(c_i)
    // T * (H + log2(T)) == 2*T*log2(T) - sum_i c_i*log2(c_i)
    //
    // So we avoid per-entry probability divisions and compute:
    // conn_ie = 2*T*log2(T) - sum_i c_i*log2(c_i)
    //
    // This is algebraically equivalent to the previous implementation.
    if tot_conn <= 0.0 {
        // Preserve previous behavior for empty connection dictionaries:
        // using a synthetic [1.0] count yields zero entropy contribution.
        tot_conn = 1.0;
    }

    let mut sum_c_log_c = 0.0f64;
    for &v in connection_counts.values() {
        if v > 0.0 {
            sum_c_log_c += v * v.log2();
        }
    }
    let conn_ie = 2.0 * tot_conn * tot_conn.log2() - sum_c_log_c;

    // Atom entropy uses the same closed-form reduction:
    // n * H(atom_counts) == sum_i c_i * log2(n / c_i)
    let n_f64 = n as f64;
    let mut atom_ie = 0.0f64;
    for &count in &atom_type_counts {
        if count > 0 {
            let c = count as f64;
            atom_ie += c * (n_f64 / c).log2();
        }
    }

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

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::Float(compute_bertz_ct(mol, ctx))
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

fn compute_ipc(mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext) -> f64 {
    let n = mol.num_atoms();
    if n < 2 {
        return 0.0;
    }

    let n2 = n * n;
    
    // Borrow context buffers
    ctx.mat_a.clear(); ctx.mat_a.resize(n2, 0.0);
    ctx.mat_b.clear(); ctx.mat_b.resize(n2, 0.0);
    ctx.mat_c.clear(); ctx.mat_c.resize(n2, 0.0);
    ctx.mat_d.clear(); ctx.mat_d.resize(n2, 0.0);
    ctx.vec_a.clear(); ctx.vec_a.resize(n + 1, 0.0);
    
    let adj = &mut ctx.mat_a;
    let mut an = &mut ctx.mat_b;
    let mut bn = &mut ctx.mat_c;
    let mut next = &mut ctx.mat_d;
    let cpoly = &mut ctx.vec_a;

    // Build simple adjacency matrix
    for bond in &mol.bonds {
        let i = bond.source as usize;
        let j = bond.target as usize;
        adj[i * n + j] = 1.0;
        adj[j * n + i] = 1.0;
    }

    cpoly[0] = 1.0;
    an.copy_from_slice(adj);

    for k in 1..=n {
        let mut trace = 0.0f64;
        for i in 0..n {
            trace += an[i * n + i];
        }
        let ck = trace / k as f64;
        cpoly[k] = ck;

        bn.copy_from_slice(an);
        for i in 0..n {
            bn[i * n + i] -= ck;
        }

        if k < n {
            next.fill(0.0);
            for i in 0..n {
                let row = i * n;
                for l in 0..n {
                    let a_il = adj[row + l];
                    if a_il == 0.0 {
                        continue;
                    }
                    let bn_row = l * n;
                    for j in 0..n {
                        next[row + j] += a_il * bn[bn_row + j];
                    }
                }
            }
            std::mem::swap(&mut an, &mut next);
        }
    }

    // Negate coefficients 1..n
    for i in 1..=n {
        cpoly[i] = -cpoly[i];
    }

    let mut total = 0.0;
    for &c in cpoly.iter() {
        total += c.abs();
    }

    let mut ie = 0.0;
    if total > 0.0 {
        for &c in cpoly.iter() {
            let ac = c.abs();
            if ac > 0.0 {
                let p = ac / total;
                ie -= p * p.log2();
            }
        }
    }

    total * ie
}

impl Descriptor for Ipc {
    fn name(&self) -> &str {
        "Ipc"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::Float64(compute_ipc(mol, ctx))
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

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
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
        desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_float()
    }

    fn calc_int(desc: &dyn Descriptor, smiles: &str) -> u32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int()
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
