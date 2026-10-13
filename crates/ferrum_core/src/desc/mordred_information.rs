//! Mordred (mordred-community 2.0.7) `InformationContent` descriptors.
//!
//! Output (42 values, Mordred's order): `IC0..IC5`, `TIC0..TIC5`,
//! `SIC0..SIC5`, `BIC0..BIC5`, `CIC0..CIC5`, `MIC0..MIC5`, `ZMIC0..ZMIC5`.
//!
//! Mordred works on the Kekulé form (`Chem.Kekulize`) of the molecule with
//! every hydrogen explicit (`Chem.AddHs`: the added hydrogens follow the
//! existing atoms, grouped by parent atom, and come after the existing bonds
//! in each atom's neighbour list). Atoms are partitioned into equivalence
//! classes: for order 0 by atomic number, for order `m` by the multiset of
//! root-to-leaf trails of Mordred's `BFSTree` after `m` expansion passes. A
//! trail lists `(Z, degree)` of every tree node and the bond type between
//! consecutive nodes. The tree is grown exactly as Mordred does it: each pass
//! walks the tree depth-first in insertion order, marks every node it passes
//! as visited and gives each leaf the neighbours that are not visited *yet*,
//! so the classes depend on the neighbour order and on the Kekulé structure
//! (RDKit 2026.03 canonical kekulization, see `mordred_kekulize`), as in
//! Mordred.
//!
//! With `A` atoms and class sizes `a_g` (`p_g = a_g / A`):
//! `IC = -Σ p_g log2 p_g`, `TIC = A·IC`, `SIC = IC / log2 A`,
//! `BIC = IC / log2 B` (`B` = sum of the Kekulé bond orders), `CIC = log2 A - IC`,
//! `MIC = -Σ m_g p_g log2 p_g` (`m_g` = mass of the class's last atom),
//! `ZMIC = -Σ a_g Z_g p_g log2 p_g`. `SIC` and `BIC` are missing when their
//! logarithm is 0 (or `B = 0`), like Mordred's `rethrow_zerodiv`.

use super::typed::{Descriptor, DescriptorValue};
use crate::desc::context::CalculationContext;
use crate::mol::molecule::Molecule;
use crate::mol::rdkit_data::{ELEMENTS, atom_mass};
use crate::types::BondType;

/// Largest neighbourhood order of Mordred's preset.
const MAX_ORDER: usize = 5;
const N_ORDERS: usize = MAX_ORDER + 1;

/// Number of output columns.
pub const MORDRED_INFORMATION_CONTENT_LEN: usize = 7 * N_ORDERS;

/// Column names in Mordred's preset order.
pub fn mordred_information_content_names() -> Vec<String> {
    ["IC", "TIC", "SIC", "BIC", "CIC", "MIC", "ZMIC"]
        .iter()
        .flat_map(|p| (0..N_ORDERS).map(move |o| format!("{p}{o}")))
        .collect()
}

/// Bits per trail step (a trail of `MAX_ORDER + 1` steps fits in a `u128`).
const STEP_BITS: u32 = 16;

/// Bond type code of a trail step (0 = root).
#[inline]
fn bond_code(t: BondType) -> u32 {
    match t {
        BondType::Single => 1,
        BondType::Double => 2,
        BondType::Triple => 3,
        BondType::Aromatic => 4,
        BondType::Dative => 5,
        BondType::Unknown => 6,
    }
}

/// RDKit `Bond::getBondTypeAsDouble`.
#[inline]
fn bond_order(t: BondType) -> f64 {
    match t {
        BondType::Single | BondType::Dative => 1.0,
        BondType::Double => 2.0,
        BondType::Triple => 3.0,
        BondType::Aromatic => 1.5,
        BondType::Unknown => 0.0,
    }
}

/// Molecule with every hydrogen explicit (`Chem.AddHs`), Kekulé bond types.
struct HGraph {
    z: Vec<u8>,
    mass: Vec<f64>,
    /// CSR neighbour lists: (atom, bond code).
    start: Vec<u32>,
    nbrs: Vec<(u32, u32)>,
    /// Sum of the bond orders.
    bond_order_sum: f64,
    /// Number of atoms before the added hydrogens.
    n_heavy: usize,
}

impl HGraph {
    fn new(mol: &Molecule, kekule: &[BondType]) -> Self {
        let n = mol.num_atoms();
        let n_h: usize = mol.atoms.iter().map(|a| a.total_h() as usize).sum();
        let total = n + n_h;
        let mut z = Vec::with_capacity(total);
        let mut mass = Vec::with_capacity(total);
        for a in &mol.atoms {
            z.push(a.atomic_number);
            mass.push(atom_mass(a.atomic_number, a.isotope));
        }
        let h_mass = ELEMENTS[1].mass;
        z.resize(total, 1);
        mass.resize(total, h_mass);
        let mut start = Vec::with_capacity(total + 1);
        let mut nbrs = Vec::with_capacity(2 * (mol.num_bonds() + n_h));
        let mut next_h = n as u32;
        let mut first_h = Vec::with_capacity(n);
        for a in 0..n {
            first_h.push(next_h);
            next_h += mol.atoms[a].total_h() as u32;
        }
        for a in 0..n {
            start.push(nbrs.len() as u32);
            for &bi in &mol.adjacency[a] {
                let b = &mol.bonds[bi as usize];
                let o = if b.source as usize == a { b.target } else { b.source };
                nbrs.push((o as u32, bond_code(kekule[bi as usize])));
            }
            let h0 = first_h[a];
            for h in h0..h0 + mol.atoms[a].total_h() as u32 {
                nbrs.push((h, bond_code(BondType::Single)));
            }
        }
        for (a, atom) in mol.atoms.iter().enumerate() {
            for _ in 0..atom.total_h() {
                start.push(nbrs.len() as u32);
                nbrs.push((a as u32, bond_code(BondType::Single)));
            }
        }
        start.push(nbrs.len() as u32);
        let bond_order_sum = kekule.iter().map(|&t| bond_order(t)).sum::<f64>() + n_h as f64;
        HGraph { z, mass, start, nbrs, bond_order_sum, n_heavy: n }
    }

    fn len(&self) -> usize {
        self.z.len()
    }

    #[inline]
    fn degree(&self, a: usize) -> u32 {
        self.start[a + 1] - self.start[a]
    }

    #[inline]
    fn step(&self, a: usize, bond: u32) -> u128 {
        let deg = self.degree(a).min(63);
        (((bond << 13) | ((self.z[a] as u32) << 6) | deg) + 1) as u128
    }
}

/// Workspace of [`tree_codes`].
#[derive(Default)]
struct TreeWork {
    /// Leaves created by the last pass: (atom, trail code).
    frontier: Vec<(u32, u128)>,
    next: Vec<(u32, u128)>,
    /// Trail codes of leaves that got no children (they stay leaves).
    dead: Vec<u128>,
    visited: Vec<bool>,
    touched: Vec<u32>,
}

/// Leaf trail codes of Mordred's `BFSTree` rooted at `root` after each of the
/// `MAX_ORDER` expansion passes, appended (sorted per atom) to `codes[m - 1]`.
///
/// Mordred's pass walks the whole tree depth-first, but the atoms of inner
/// nodes are visited already, so the only effect of a pass is that the
/// leaves of the previous pass are marked visited and expanded in depth-first
/// order - which is their creation order. A leaf that gets no children stays
/// a leaf for good (later re-expansions see a superset of visited atoms).
fn tree_codes(g: &HGraph, root: usize, w: &mut TreeWork, codes: &mut [Vec<u128>], ranges: &mut [Vec<(u32, u32)>]) {
    let TreeWork { frontier, next, dead, visited, touched } = w;
    frontier.clear();
    dead.clear();
    frontier.push((root as u32, g.step(root, 0)));
    for m in 1..=MAX_ORDER {
        next.clear();
        let shift = STEP_BITS * m as u32;
        for &(atom, code) in frontier.iter() {
            let atom = atom as usize;
            if !visited[atom] {
                visited[atom] = true;
                touched.push(atom as u32);
            }
            let before = next.len();
            for k in g.start[atom]..g.start[atom + 1] {
                let (nb, bond) = g.nbrs[k as usize];
                if !visited[nb as usize] {
                    next.push((nb, code | (g.step(nb as usize, bond) << shift)));
                }
            }
            if next.len() == before {
                dead.push(code);
            }
        }
        std::mem::swap(frontier, next);
        let out = &mut codes[m - 1];
        let begin = out.len();
        out.extend_from_slice(dead);
        out.extend(frontier.iter().map(|&(_, c)| c));
        out[begin..].sort_unstable();
        ranges[m - 1].push((begin as u32, (out.len() - begin) as u32));
    }
    for &a in touched.iter() {
        visited[a as usize] = false;
    }
    touched.clear();
}

/// Class sizes and representative (last) atom of each class of atoms with
/// equal `key`.
fn classes_by<K: Eq + std::hash::Hash>(n: usize, key: impl Fn(usize) -> K, out: &mut Vec<(f64, usize)>) {
    let mut class_of: rustc_hash::FxHashMap<K, usize> = rustc_hash::FxHashMap::default();
    out.clear();
    for i in 0..n {
        let next = out.len();
        let c = *class_of.entry(key(i)).or_insert(next);
        if c == next {
            out.push((1.0, i));
        } else {
            out[c].0 += 1.0;
            out[c].1 = i;
        }
    }
}

/// Mordred InformationContent values in
/// [`mordred_information_content_names`] order.
pub fn mordred_information_content(mol: &Molecule, ctx: &mut CalculationContext) -> Vec<f64> {
    let mut out = vec![f64::NAN; MORDRED_INFORMATION_CONTENT_LEN];
    let Some(kekule) = super::mordred_kekulize::mordred_kekule_bond_types(mol, ctx) else {
        return out;
    };
    let g = HGraph::new(mol, &kekule);
    let n = g.len();
    if n == 0 {
        return out;
    }

    let mut codes: Vec<Vec<u128>> = vec![Vec::new(); MAX_ORDER];
    let mut ranges: Vec<Vec<(u32, u32)>> = (0..MAX_ORDER).map(|_| Vec::with_capacity(n)).collect();
    let mut work = TreeWork { visited: vec![false; n], ..Default::default() };
    for root in 0..n {
        // Hydrogens added to the same atom have identical trees (the other
        // hydrogens are leaves either way): reuse the codes of the first one.
        if root > g.n_heavy && g.nbrs[g.start[root] as usize].0 == g.nbrs[g.start[root - 1] as usize].0 {
            for m in 0..MAX_ORDER {
                let r = ranges[m][root - 1];
                ranges[m].push(r);
            }
            continue;
        }
        tree_codes(&g, root, &mut work, &mut codes, &mut ranges);
    }

    let a = n as f64;
    let log2_a = a.log2();
    let log2_b = g.bond_order_sum.log2();
    let mut classes = Vec::new();
    for m in 0..N_ORDERS {
        if m == 0 {
            classes_by(n, |i| g.z[i], &mut classes);
        } else {
            let (c, r) = (&codes[m - 1], &ranges[m - 1]);
            classes_by(
                n,
                |i| {
                    let (s, l) = r[i];
                    &c[s as usize..(s + l) as usize]
                },
                &mut classes,
            );
        }
        let (mut ic, mut mic, mut zmic) = (0.0f64, 0.0f64, 0.0f64);
        for &(size, rep) in &classes {
            let p = size / a;
            let t = p * p.log2();
            ic -= t;
            mic -= g.mass[rep] * t;
            zmic -= size * g.z[rep] as f64 * t;
        }
        out[m] = ic;
        out[N_ORDERS + m] = a * ic;
        out[2 * N_ORDERS + m] = if log2_a == 0.0 { f64::NAN } else { ic / log2_a };
        out[3 * N_ORDERS + m] =
            if log2_b == 0.0 || !log2_b.is_finite() { f64::NAN } else { ic / log2_b };
        out[4 * N_ORDERS + m] = log2_a - ic;
        out[5 * N_ORDERS + m] = mic;
        out[6 * N_ORDERS + m] = zmic;
    }
    out
}

/// Mordred InformationContent descriptor family (42 values, see the module docs).
pub struct MordredInformationContent;

impl Descriptor for MordredInformationContent {
    fn name(&self) -> &str {
        "MordredInformationContent"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::FloatVec(mordred_information_content(mol, ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn names_follow_mordred_order() {
        let names = mordred_information_content_names();
        assert_eq!(names.len(), MORDRED_INFORMATION_CONTENT_LEN);
        assert_eq!(names[0], "IC0");
        assert_eq!(names[6], "TIC0");
        assert_eq!(names[41], "ZMIC5");
    }

    #[test]
    fn methane_classes() {
        // CH4: classes {C}, {H x4}
        let v = mordred_information_content(&parse_smiles("C").unwrap(), &mut CalculationContext::new());
        let ic0 = -(0.2f64 * 0.2f64.log2() + 0.8 * 0.8f64.log2());
        assert!((v[0] - ic0).abs() < 1e-12);
        assert!((v[1] - ic0).abs() < 1e-12);
        assert!((v[6] - 5.0 * ic0).abs() < 1e-12);
        assert!((v[18] - ic0 / 4f64.log2()).abs() < 1e-12);
    }
}
