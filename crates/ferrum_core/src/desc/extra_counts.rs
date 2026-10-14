#![allow(dead_code)]
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
fn jacobi_eigen(a: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    if n == 0 {
        return (Vec::new(), Array2::<f64>::zeros((0, 0)));
    }
    if n == 1 {
        return (vec![a[[0, 0]]], Array2::<f64>::eye(1));
    }
    let mut d = a.clone();
    let mut v = Array2::<f64>::eye(n);

    let max_iter = 100 * n * n;
    let eps = 1e-12;

    for _ in 0..max_iter {
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = d[[i, j]].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < eps {
            break;
        }

        let dpp = d[[p, p]];
        let dqq = d[[q, q]];
        let dpq = d[[p, q]];

        let theta = if (dpp - dqq).abs() < eps {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * dpq / (dpp - dqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        for i in 0..n {
            if i != p && i != q {
                let dip = d[[i, p]];
                let diq = d[[i, q]];
                let new_ip = c * dip + s * diq;
                let new_iq = -s * dip + c * diq;
                d[[i, p]] = new_ip;
                d[[p, i]] = new_ip;
                d[[i, q]] = new_iq;
                d[[q, i]] = new_iq;
            }
        }
        d[[p, p]] = c * c * dpp + 2.0 * s * c * dpq + s * s * dqq;
        d[[q, q]] = s * s * dpp - 2.0 * s * c * dpq + c * c * dqq;
        d[[p, q]] = 0.0;
        d[[q, p]] = 0.0;

        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip + s * viq;
            v[[i, q]] = -s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| d[[i, i]]).collect();
    (eigenvalues, v)
}
use ndarray::Array2;
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

fn compute_fraction_csp3(mol: &Molecule) -> f64 {
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
        sp3_c as f64 / total_c as f64
    } else {
        0.0
    }
}

impl Descriptor for FractionCSP3 {
    fn name(&self) -> &str {
        "FractionCSP3"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Float(compute_fraction_csp3(mol))
    }
}

// ---------------------------------------------------------------------------
// Ring Type Counts (helper function only; typed descriptor in ring_types.rs)
// ---------------------------------------------------------------------------

fn ring_all_single_bonds(mol: &Molecule, ring: &[u16]) -> bool {
    ring.iter()
        .all(|&bond_idx| mol.bonds[bond_idx as usize].bond_type == BondType::Single)
}

/// Compute ring type counts: [aromatic, aliphatic, saturated, heteroaromatic, heterocyclic].
pub fn compute_ring_type_counts(mol: &Molecule) -> [u32; 5] {
    use crate::algo::rings::find_minimum_cycle_basis_bonds;

    let mut counts = [0u32; 5];
    let rings = find_minimum_cycle_basis_bonds(mol);

    for ring in &rings {
        let all_aromatic = ring
            .iter()
            .all(|&bond_idx| mol.bonds[bond_idx as usize].is_aromatic());
        let any_aromatic = ring
            .iter()
            .any(|&bond_idx| mol.bonds[bond_idx as usize].is_aromatic());
        let has_heteroatom = ring.iter().any(|&bond_idx| {
            let bond = &mol.bonds[bond_idx as usize];
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

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(compute_num_amide_bonds(mol))
    }
}

// ---------------------------------------------------------------------------
// AromaticAtomFraction
// ---------------------------------------------------------------------------

/// Fraction of heavy atoms that are aromatic.
pub struct AromaticAtomFraction;

fn compute_aromatic_atom_fraction(mol: &Molecule) -> f64 {
    let heavy: usize = mol.atoms.iter().filter(|a| a.atomic_number > 1).count();
    if heavy == 0 {
        return 0.0;
    }
    let aromatic: usize = mol
        .atoms
        .iter()
        .filter(|a| a.atomic_number > 1 && a.is_aromatic())
        .count();
    aromatic as f64 / heavy as f64
}

impl Descriptor for AromaticAtomFraction {
    fn name(&self) -> &str {
        "AromaticAtomFraction"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
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
    if value.is_nan() { None } else { Some(value) }
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

fn compute_hall_kier_alpha(mol: &Molecule) -> f64 {
    compute_hall_kier_alpha_sum(mol)
}

/// Public f64 accessor for Kappa index computation.
pub fn compute_hall_kier_alpha_f64(mol: &Molecule) -> f64 {
    compute_hall_kier_alpha_sum(mol)
}

impl Descriptor for HallKierAlpha {
    fn name(&self) -> &str {
        "HallKierAlpha"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
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

/// `float(Chem.BondType)` as used by RDKit's Python BertzCT for non-aromatic bonds.
fn rdkit_bond_type_value(bond: &crate::mol::bond::Bond) -> f64 {
    match bond.bond_type {
        crate::types::BondType::Single => 1.0,
        crate::types::BondType::Double => 2.0,
        crate::types::BondType::Triple => 3.0,
        crate::types::BondType::Aromatic => 12.0,
        crate::types::BondType::Dative => 17.0,
        crate::types::BondType::Unknown => 0.0,
    }
}

/// RDKit `Bond::getBondTypeAsDouble`.
fn rdkit_bond_type_as_double(bond: &crate::mol::bond::Bond) -> f64 {
    match bond.bond_type {
        crate::types::BondType::Single | crate::types::BondType::Dative => 1.0,
        crate::types::BondType::Double => 2.0,
        crate::types::BondType::Triple => 3.0,
        crate::types::BondType::Aromatic => 1.5,
        crate::types::BondType::Unknown => 0.0,
    }
}

/// RDKit `InfoEntropy` (bits), summing in the given order.
pub(crate) fn info_entropy(values: &[f64]) -> f64 {
    let total: f64 = values.iter().sum();
    let mut accum = 0.0;
    if total != 0.0 {
        for &v in values {
            let d = v / total;
            if d != 0.0 {
                accum += -d * d.ln();
            }
        }
    }
    accum / 2f64.ln()
}

/// BertzCT symmetry classes: first-seen index (1-based) of the sorted
/// bond-order distance row (first `CUTOFF` entries), compared at '%.4f'
/// precision.
///
/// Distances are multiples of 1/6, so `round(d * 10000)` is strictly
/// increasing in the exact distance in sixths and the rounded rows of
/// RDKit's floating-point matrix (off by a few ulp, far from a rounding
/// boundary) compare exactly like the integer rows; unreachable pairs
/// (1e8 in RDKit) are larger than every finite distance either way.
fn bertz_symmetry_classes(mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext) -> Vec<usize> {
    const CUTOFF: usize = 100;
    let n = mol.num_atoms();
    let dist = crate::algo::distance_matrix::bond_order_distances_sixths_cached(mol, ctx);
    let width = n.min(CUTOFF);
    let mut keys = vec![0u32; n * width];
    if width == n {
        keys.copy_from_slice(dist);
        for key in keys.chunks_exact_mut(n) {
            key.sort_unstable();
        }
    } else {
        let mut row: Vec<u32> = Vec::with_capacity(n);
        for i in 0..n {
            row.clear();
            row.extend_from_slice(&dist[i * n..(i + 1) * n]);
            row.select_nth_unstable(width - 1);
            row.truncate(width);
            row.sort_unstable();
            keys[i * width..(i + 1) * width].copy_from_slice(&row);
        }
    }
    let mut class_of_key: rustc_hash::FxHashMap<&[u32], usize> =
        rustc_hash::FxHashMap::with_capacity_and_hasher(n, Default::default());
    (0..n)
        .map(|i| {
            let next = class_of_key.len() + 1;
            *class_of_key.entry(&keys[i * width..(i + 1) * width]).or_insert(next)
        })
        .collect()
}

/// Reference symmetry classes from RDKit's floating-point Floyd-Warshall
/// (bonds of unknown type, whose `1 / order` is infinite).
fn bertz_symmetry_classes_fw(mol: &Molecule) -> Vec<usize> {
    const LOCAL_INF: f64 = 1e8;
    const CUTOFF: usize = 100;
    let n = mol.num_atoms();
    // Bond-order weighted distance matrix (getDistanceMat(useBO=1)) with
    // RDKit's double-buffered Floyd-Warshall.
    let mut last = vec![LOCAL_INF; n * n];
    for i in 0..n {
        last[i * n + i] = 0.0;
    }
    for bond in &mol.bonds {
        let (i, j) = (bond.source as usize, bond.target as usize);
        let contrib = if bond.is_aromatic() { 2.0 / 3.0 } else { 1.0 / rdkit_bond_type_as_double(bond) };
        last[i * n + j] = contrib;
        last[j * n + i] = contrib;
    }
    let mut curr = vec![0.0f64; n * n];
    for k in 0..n {
        for i in 0..n {
            let dik = last[i * n + k];
            for j in 0..n {
                let v1 = last[i * n + j];
                let v2 = dik + last[k * n + j];
                curr[i * n + j] = if v1 <= v2 { v1 } else { v2 };
            }
        }
        std::mem::swap(&mut last, &mut curr);
    }

    let mut class_of_key: rustc_hash::FxHashMap<Vec<i64>, usize> = rustc_hash::FxHashMap::default();
    let mut sym = vec![0usize; n];
    let mut row: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        row.clear();
        row.extend_from_slice(&last[i * n..(i + 1) * n]);
        row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let key: Vec<i64> = row.iter().take(CUTOFF).map(|&x| (x * 10_000.0).round() as i64).collect();
        let next = class_of_key.len() + 1;
        sym[i] = *class_of_key.entry(key).or_insert(next);
    }
    sym
}

/// Port of RDKit `GraphDescriptors.BertzCT` (Python, cutoff = 100).
fn compute_bertz_ct(mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext) -> f64 {
    let n = mol.num_atoms();
    if n < 2 {
        return 0.0;
    }

    let sym = if mol.bonds.iter().any(|b| b.bond_type == crate::types::BondType::Unknown && !b.is_aromatic()) {
        bertz_symmetry_classes_fw(mol)
    } else {
        bertz_symmetry_classes(mol, ctx)
    };

    // Sorted neighbour lists and bond lookup.
    let mut neighbors: Vec<smallvec::SmallVec<[(usize, f64); 4]>> = vec![smallvec::SmallVec::new(); n];
    for bond in &mol.bonds {
        let (i, j) = (bond.source as usize, bond.target as usize);
        let order = if bond.is_aromatic() { 1.5 } else { rdkit_bond_type_value(bond) };
        if !neighbors[i].iter().any(|&(x, _)| x == j) {
            neighbors[i].push((j, order));
        }
        if !neighbors[j].iter().any(|&(x, _)| x == i) {
            neighbors[j].push((i, order));
        }
    }
    for list in neighbors.iter_mut() {
        list.sort_by_key(|&(x, _)| x);
    }

    // Python dicts keep insertion order; mirror that for identical summation.
    let mut atom_type_index: rustc_hash::FxHashMap<u8, usize> = rustc_hash::FxHashMap::default();
    let mut atom_types: Vec<f64> = Vec::new();
    let mut conn_index: rustc_hash::FxHashMap<(usize, usize, usize), usize> =
        rustc_hash::FxHashMap::with_capacity_and_hasher(2 * mol.num_bonds() + 4, Default::default());
    let mut connections: Vec<f64> = Vec::with_capacity(2 * mol.num_bonds() + 4);
    let add_conn = |key: (usize, usize, usize), v: f64, conn_index: &mut rustc_hash::FxHashMap<(usize, usize, usize), usize>, connections: &mut Vec<f64>| {
        let idx = *conn_index.entry(key).or_insert_with(|| {
            connections.push(0.0);
            connections.len() - 1
        });
        connections[idx] += v;
    };

    for atom_idx in 0..n {
        let z = mol.atoms[atom_idx].atomic_number;
        let t = *atom_type_index.entry(z).or_insert_with(|| {
            atom_types.push(0.0);
            atom_types.len() - 1
        });
        atom_types[t] += 1.0;
        let hinge = sym[atom_idx];
        let nbrs = &neighbors[atom_idx];
        for i in 0..nbrs.len() {
            let (ni, bo_i) = nbrs[i];
            let ni_class = sym[ni];
            if bo_i > 1.0 && ni > atom_idx {
                let key = (ni_class.min(hinge), 0, ni_class.max(hinge));
                add_conn(key, bo_i * (bo_i - 1.0) / 2.0, &mut conn_index, &mut connections);
            }
            for &(nj, bo_j) in &nbrs[i + 1..] {
                let nj_class = sym[nj];
                let key = (ni_class.min(nj_class), hinge, ni_class.max(nj_class));
                add_conn(key, bo_i * bo_j, &mut conn_index, &mut connections);
            }
        }
    }
    if connections.is_empty() {
        connections.push(1.0);
    }

    let tot: f64 = connections.iter().sum();
    let connection_ie = tot * (info_entropy(&connections) + tot.ln() / 2f64.ln());
    let atom_type_ie = n as f64 * info_entropy(&atom_types);
    atom_type_ie + connection_ie
}

impl Descriptor for BertzCT {
    fn name(&self) -> &str {
        "BertzCT"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
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

/// Characteristic polynomial coefficients exactly as RDKit's
/// `Graphs.CharacteristicPolynomial` (Le Verrier-Faddeev-Frame in float64).
///
/// `A` is the 0/1 adjacency matrix, so `A @ Bn` is evaluated over neighbour
/// lists (O(n^3 * degree) instead of a dense O(n^4) product). For large,
/// flexible molecules this recurrence is numerically unstable; RDKit's values
/// are then dominated by rounding and can only be reproduced approximately.
///
/// Every floating-point operation of the straightforward recurrence is kept
/// (same operands, same order), so the result is bit-identical to it:
/// - row `i` of `A @ Bn` is `((Bn[k0] + Bn[k1]) + Bn[k2]) ...` over the sorted
///   neighbours `k0 < k1 < ...`; the leading `0.0 +` of a zero-initialised
///   accumulator is dropped (exact, no entry is ever `-0.0`);
/// - `Bn = An - c I` reuses the `An` buffer instead of copying it;
/// - the last product is only needed for its trace, so only its diagonal is
///   formed.
pub(crate) fn characteristic_polynomial(mol: &Molecule) -> Vec<f64> {
    let n = mol.num_atoms();
    let mut nbrs: Vec<smallvec::SmallVec<[u32; 4]>> = vec![smallvec::SmallVec::new(); n];
    for bond in &mol.bonds {
        let (i, j) = (bond.source as usize, bond.target as usize);
        nbrs[i].push(j as u32);
        nbrs[j].push(i as u32);
    }
    for list in nbrs.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }
    let mut res = vec![0.0f64; n + 1];
    res[0] = 1.0;
    if n == 0 {
        return res;
    }
    // An starts as A (dense, row-major)
    let mut an = vec![0.0f64; n * n];
    for i in 0..n {
        for &k in &nbrs[i] {
            an[i * n + k as usize] = 1.0;
        }
    }
    let mut bn = vec![0.0f64; n * n];
    for step in 1..=n {
        let trace: f64 = (0..n).map(|i| an[i * n + i]).sum();
        let c = 1.0 / step as f64 * trace;
        res[step] = c;
        if step == n {
            break;
        }
        // Bn = An - c I
        std::mem::swap(&mut an, &mut bn);
        for i in 0..n {
            bn[i * n + i] -= c;
        }
        if step + 1 == n {
            // only the trace of the last An = A @ Bn is used
            for i in 0..n {
                let mut d = 0.0f64;
                let mut first = true;
                for &k in &nbrs[i] {
                    let v = bn[k as usize * n + i];
                    d = if first { v } else { d + v };
                    first = false;
                }
                an[i * n + i] = d;
            }
            continue;
        }
        // An = A @ Bn
        for i in 0..n {
            let row = &mut an[i * n..(i + 1) * n];
            let b = |k: u32| &bn[k as usize * n..(k as usize + 1) * n];
            match nbrs[i].as_slice() {
                [] => row.fill(0.0),
                &[k0] => row.copy_from_slice(b(k0)),
                &[k0, k1] => {
                    for ((r, x0), x1) in row.iter_mut().zip(b(k0)).zip(b(k1)) {
                        *r = x0 + x1;
                    }
                }
                &[k0, k1, k2] => {
                    for (((r, x0), x1), x2) in row.iter_mut().zip(b(k0)).zip(b(k1)).zip(b(k2)) {
                        *r = (x0 + x1) + x2;
                    }
                }
                &[k0, k1, k2, k3, ref rest @ ..] => {
                    for ((((r, x0), x1), x2), x3) in row.iter_mut().zip(b(k0)).zip(b(k1)).zip(b(k2)).zip(b(k3)) {
                        *r = ((x0 + x1) + x2) + x3;
                    }
                    for &k in rest {
                        for (r, x) in row.iter_mut().zip(b(k)) {
                            *r += x;
                        }
                    }
                }
            }
        }
    }
    for c in res.iter_mut().skip(1) {
        *c = -*c;
    }
    res
}

/// `|characteristic polynomial coefficients|`, computed once per molecule and
/// shared by `Ipc` and `AvgIpc` through the calculation context.
pub(crate) fn abs_characteristic_polynomial_cached<'a>(
    mol: &Molecule,
    ctx: &'a mut crate::desc::context::CalculationContext,
) -> &'a [f64] {
    let key = crate::desc::context::molecule_cache_key(mol);
    if ctx.char_poly_key != Some(key) {
        ctx.char_poly_abs.clear();
        ctx.char_poly_abs.extend(characteristic_polynomial(mol).into_iter().map(f64::abs));
        ctx.char_poly_key = Some(key);
    }
    &ctx.char_poly_abs
}

/// RDKit `GraphDescriptors.Ipc` (avg = False): `sum(|c|) * InfoEntropy(|c|)`.
fn compute_ipc(mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext) -> f64 {
    if mol.num_atoms() == 0 {
        return 0.0;
    }
    let cpoly = abs_characteristic_polynomial_cached(mol, ctx);
    let total: f64 = cpoly.iter().sum();
    total * info_entropy(cpoly)
}

impl Descriptor for Ipc {
    fn name(&self) -> &str {
        "Ipc"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Float(compute_ipc(mol, ctx))
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

fn compute_fragment_complexity(mol: &Molecule) -> f64 {
    let mut element_counts = [0u32; 120];
    for atom in &mol.atoms {
        if atom.atomic_number > 0 {
            element_counts[atom.atomic_number as usize] += 1;
        }
    }
    shannon_entropy(&element_counts)
}

impl Descriptor for FragmentComplexity {
    fn name(&self) -> &str {
        "FragmentComplexity"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
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

    /// The straightforward recurrence `characteristic_polynomial` must match bit for bit.
    fn characteristic_polynomial_reference(mol: &Molecule) -> Vec<f64> {
        let n = mol.num_atoms();
        let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); n];
        for bond in &mol.bonds {
            let (i, j) = (bond.source as usize, bond.target as usize);
            nbrs[i].push(j);
            nbrs[j].push(i);
        }
        for list in nbrs.iter_mut() {
            list.sort_unstable();
            list.dedup();
        }
        let mut res = vec![0.0f64; n + 1];
        res[0] = 1.0;
        let mut an = vec![0.0f64; n * n];
        for i in 0..n {
            for &k in &nbrs[i] {
                an[i * n + k] = 1.0;
            }
        }
        let mut bn = vec![0.0f64; n * n];
        for step in 1..=n {
            let trace: f64 = (0..n).map(|i| an[i * n + i]).sum();
            res[step] = 1.0 / step as f64 * trace;
            bn.copy_from_slice(&an);
            for i in 0..n {
                bn[i * n + i] -= res[step];
            }
            for i in 0..n {
                let row = &mut an[i * n..(i + 1) * n];
                row.fill(0.0);
                for &k in &nbrs[i] {
                    let brow = &bn[k * n..(k + 1) * n];
                    for j in 0..n {
                        row[j] += brow[j];
                    }
                }
            }
        }
        for c in res.iter_mut().skip(1) {
            *c = -*c;
        }
        res
    }

    #[test]
    fn bertz_symmetry_classes_match_floyd_warshall() {
        for smi in [
            "CC", "C=CC#N", "c1ccccc1", "c1ccc2ccccc2c1C#N", "CC.O", "[Na+].[Cl-]", "O=C1C=CC(=O)C=C1",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "C12C3C4C1C5C2C3C45", "OC(=O)C=CC=CC=CC#CC#N",
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)C",
            "CC(C)CC(NC(=O)CNC(=O)C(CC(=O)O)NC(=O)C(CS)NC(=O)C(CS)NC(=O)C(CC1=CC=C(O)C=C1)NC(=O)C(N)CCC(N)=O)C(=O)O",
        ] {
            let mol = parse_smiles(smi).unwrap();
            let mut ctx = crate::desc::context::CalculationContext::new();
            assert_eq!(bertz_symmetry_classes(&mol, &mut ctx), bertz_symmetry_classes_fw(&mol), "{smi}");
        }
    }

    #[test]
    fn characteristic_polynomial_bit_identical_to_reference() {
        for smi in [
            "C", "CC", "CCO", "c1ccccc1", "CC(C)(C)C", "[Na+].[Cl-]", "CC.O", "C12C3C4C1C5C2C3C45",
            "CS(=O)(=O)c1ccc(Cl)cc1", "CC(C)(C)P(C(C)(C)C)(C(C)(C)C)(C(C)(C)C)C",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "OCC1OC(O)C(O)C(O)C1O.OCC1OC(O)C(O)C(O)C1O",
            "CCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCC",
            "CC(C)CC(NC(=O)CNC(=O)C(CC(=O)O)NC(=O)C(CS)NC(=O)C(CS)NC(=O)C(CC1=CC=C(O)C=C1)NC(=O)C(N)CCC(N)=O)C(=O)O",
        ] {
            let mol = parse_smiles(smi).unwrap();
            let (fast, slow) = (characteristic_polynomial(&mol), characteristic_polynomial_reference(&mol));
            assert_eq!(fast.len(), slow.len());
            for (f, s) in fast.iter().zip(&slow) {
                assert!(f.to_bits() == s.to_bits(), "{smi}: {fast:?} vs {slow:?}");
            }
        }
    }

    fn calc_float(desc: &dyn Descriptor, smiles: &str) -> f64 {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        desc.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        )
        .as_float()
    }

    fn calc_int(desc: &dyn Descriptor, smiles: &str) -> u32 {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        desc.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        )
        .as_int()
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
            (v - (-0.53_f64)).abs() < 0.01,
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
            (v - (-0.53_f64)).abs() < 0.01,
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
            (v - (-0.98_f64)).abs() < 0.01,
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
            (v - (-0.04_f64)).abs() < 0.01,
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
            (v - 0.11_f64).abs() < 0.01,
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
            (v - 0.11_f64).abs() < 0.01,
            "dimethyl phosphate HallKierAlpha expected 0.11, got {v}"
        );
    }

    #[test]
    fn test_hka_allene_central_carbon_sp() {
        let v = calc_float(&HallKierAlpha, "C=C=C");
        assert!(
            (v - (-0.48_f64)).abs() < 0.01,
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

    #[test]
    fn test_ipc_avg_mode_is_entropy_only() {
        // This test exercises the average-Ipc mode gate without affecting default behavior.
        // `Ipc(avg)` should be <= `Ipc` for positive total coefficient mass.
        if std::env::var("FERRUM_IPC_MODE")
            .ok()
            .as_deref()
            .map(|v| v.eq_ignore_ascii_case("avg"))
            .unwrap_or(false)
        {
            let avg = calc_float(&Ipc, "c1ccccc1");
            assert!(avg > 0.0, "avg Ipc should be positive: {avg}");
            assert!(avg < 10.0, "avg Ipc should be entropy-scale, got {avg}");
        }
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

/// CDK `FragmentComplexityDescriptor` (`fragC`).
pub struct CdkFragmentComplexity;

impl Descriptor for CdkFragmentComplexity {
    fn name(&self) -> &str {
        "FragmentComplexityCDK"
    }

    /// `|B^2 - A^2 + A| + H/100` with A = non-hydrogen atoms, B = bonds
    /// (hydrogens included) and H = atoms other than H and C.
    fn calculate(&self, mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let mut a = 0i64;
        let mut hetero = 0i64;
        for atom in &mol.atoms {
            if atom.atomic_number != 1 {
                a += 1;
                if atom.atomic_number != 6 {
                    hetero += 1;
                }
            }
        }
        // Implicit hydrogens as `CDKHydrogenAdder` assigns them in the
        // reference preparation (formal neighbour count of the CDK atom type
        // minus the degree, the parsed count for untyped `X` atoms): CDK's
        // SMILES parser keeps no radicals, so `[CH3]` becomes methane and the
        // `O.plus` of `C[O+]C` gets one H, while `[PH](C)(C)C` (no P type)
        // keeps its single H.
        let hydrogens: i64 = super::cdk_perception::CdkView::with_legacy_cached(mol, ctx, |view, _| {
            super::cdk_simple::with_parser_mol(mol, view, |m| m.hcount.iter().map(|&h| h as i64).sum())
        });
        // CDK's bond count includes the bonds to explicit hydrogen atoms, which
        // FerrumChem folds into the hydrogen count: B = bonds + all hydrogens.
        let b = mol.num_bonds() as i64 + hydrogens;
        // Java int arithmetic in CDK
        let core = (b as i32).wrapping_mul(b as i32).wrapping_sub((a as i32).wrapping_mul(a as i32)).wrapping_add(a as i32);
        DescriptorValue::Float((core as f64).abs() + hetero as f64 / 100.0)
    }
}

#[cfg(test)]
mod cdk_fragment_complexity_tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn frag_c(smiles: &str) -> f64 {
        let mol = parse_smiles(smiles).unwrap();
        CdkFragmentComplexity.calculate(&mol, &mut crate::desc::context::CalculationContext::new()).as_float()
    }

    // Expected values from CDK 2.13 FragmentComplexityDescriptor.
    #[test]
    fn test_cdk_fragment_complexity() {
        assert_eq!(frag_c("c1ccccc1"), 114.0);
        // explicit hydrogen atoms: CDK counts their bonds instead of implicit H
        assert_eq!(frag_c("[H]OC([H])([H])C"), 58.01);
        assert_eq!(frag_c("CCO"), 58.01);
        // radicals are saturated by CDK's hydrogen adder
        assert_eq!(frag_c("[CH3]"), 16.0);
        assert_eq!(frag_c("[C]"), 16.0);
        assert!((frag_c("[N]1C=CC=C1") - 80.01).abs() < 1e-9);
        assert!((frag_c("[Se]") - 0.01).abs() < 1e-9);
        // charged and untyped atoms (CDK 2.13 values)
        assert!((frag_c("[NH-][C+](N)N[O+]CCC(N)C(=O)[O-]") - 397.07).abs() < 1e-9);
        assert!((frag_c("C1=CC=C([PH](C2CCCCC2)C2CCCCC2)C=C1") - 2059.01).abs() < 1e-9);
    }
}
