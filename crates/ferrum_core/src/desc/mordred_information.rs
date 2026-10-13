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
//!
//! The trails are packed into integer codes (one renumbered step per
//! position). Per pass, the leaves that stopped growing and the new leaves are
//! kept apart (a code's length is its depth), and the roots are grouped by
//! order-independent fingerprints of these multisets, verified exactly by
//! comparing the sorted codes. Atoms with a single neighbour get no tree of
//! their own: their class follows from the class of the neighbour one order
//! lower.

use super::typed::{Descriptor, DescriptorValue};
use crate::desc::context::CalculationContext;
use crate::mol::molecule::Molecule;
use crate::mol::rdkit_data::{ELEMENTS, atom_mass};
use crate::types::BondType;
use rustc_hash::FxHashMap;

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

/// Flag of the neighbour entries of atoms with a single neighbour.
const LEAF: u32 = 1 << 31;

/// Trail step: bond code, atomic number and degree of a tree node.
#[inline]
fn step(z: u8, degree: u32, bond: u32) -> u32 {
    ((bond << 13) | ((z as u32) << 6) | degree.min(63)) + 1
}

/// Molecule with every hydrogen explicit (`Chem.AddHs`), Kekulé bond types.
struct HGraph {
    z: Vec<u8>,
    mass: Vec<f64>,
    /// CSR neighbour lists: (atom, trail step of the atom entered through
    /// the bond (see [`step`], renumbered), `| LEAF` if the atom has a single
    /// neighbour).
    start: Vec<u32>,
    nbrs: Vec<(u32, u32)>,
    /// Trail step of each atom as the root (renumbered).
    root_steps: Vec<u32>,
    /// The steps are renumbered `1..2^step_bits` in order of appearance (in a
    /// trail code, each step takes `step_bits` bits).
    step_bits: u32,
    /// Sum of the bond orders.
    bond_order_sum: f64,
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
        let degree = |a: usize| (mol.adjacency[a].len() + mol.atoms[a].total_h() as usize) as u32;
        let leaf = |a: usize| if degree(a) == 1 { LEAF } else { 0 };
        let h_step = step(1, 1, bond_code(BondType::Single)) | LEAF;
        for a in 0..n {
            start.push(nbrs.len() as u32);
            for &bi in &mol.adjacency[a] {
                let b = &mol.bonds[bi as usize];
                let o = if b.source as usize == a { b.target } else { b.source } as usize;
                nbrs.push((o as u32, step(z[o], degree(o), bond_code(kekule[bi as usize])) | leaf(o)));
            }
            let h0 = first_h[a];
            for h in h0..h0 + mol.atoms[a].total_h() as u32 {
                nbrs.push((h, h_step));
            }
        }
        for (a, atom) in mol.atoms.iter().enumerate() {
            let s = step(z[a], degree(a), bond_code(BondType::Single)) | leaf(a);
            for _ in 0..atom.total_h() {
                start.push(nbrs.len() as u32);
                nbrs.push((a as u32, s));
            }
        }
        start.push(nbrs.len() as u32);
        // renumber the steps densely
        let mut ids: FxHashMap<u32, u32> = FxHashMap::default();
        let mut id = |s: u32| {
            let next = ids.len() as u32 + 1;
            *ids.entry(s).or_insert(next)
        };
        let mut root_steps = Vec::with_capacity(total);
        for a in 0..total {
            let d = start[a + 1] - start[a];
            root_steps.push(id(step(z[a], d, 0)));
        }
        for e in nbrs.iter_mut() {
            e.1 = id(e.1 & !LEAF) | (e.1 & LEAF);
        }
        let step_bits = u32::BITS - (ids.len() as u32).leading_zeros();
        let bond_order_sum = kekule.iter().map(|&t| bond_order(t)).sum::<f64>() + n_h as f64;
        HGraph { z, mass, start, nbrs, root_steps, step_bits, bond_order_sum }
    }

    fn len(&self) -> usize {
        self.z.len()
    }

    #[inline]
    fn degree(&self, a: usize) -> u32 {
        self.start[a + 1] - self.start[a]
    }

    /// Trail step of the root `a` (bond code 0).
    #[inline]
    fn root_step(&self, a: usize) -> u32 {
        self.root_steps[a]
    }
}

/// Trail code: the steps of a trail, `HGraph::step_bits` bits each (`u64`
/// when `MAX_ORDER + 1` steps fit, `u128` otherwise).
trait Code: Copy + Ord + Default {
    /// `step` at trail position `level`.
    fn step(step: u32, level: u32, bits: u32) -> Self;
    fn or(self, other: Self) -> Self;
    /// Hash of the code; the fingerprint of a multiset of codes is the
    /// (wrapping) sum, independent of their order.
    fn mix(self) -> u64;
}

impl Code for u64 {
    #[inline(always)]
    fn step(step: u32, level: u32, bits: u32) -> Self {
        (step as u64) << (level * bits)
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        self | other
    }

    #[inline(always)]
    fn mix(self) -> u64 {
        let x = (self ^ (self >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }
}

impl Code for u128 {
    #[inline(always)]
    fn step(step: u32, level: u32, bits: u32) -> Self {
        (step as u128) << (level * bits)
    }

    #[inline(always)]
    fn or(self, other: Self) -> Self {
        self | other
    }

    #[inline(always)]
    fn mix(self) -> u64 {
        ((self as u64) ^ ((self >> 64) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).mix()
    }
}

/// Workspace of [`tree_codes`].
#[derive(Default)]
struct TreeWork<C> {
    /// Leaves created by the last pass: (atom, trail code).
    frontier: Vec<(u32, C)>,
    next: Vec<(u32, C)>,
    /// Codes of the atoms with a single neighbour that entered the tree in the
    /// last pass.
    h_leaves: Vec<C>,
    visited: Vec<bool>,
    touched: Vec<u32>,
}

/// Trail codes of one root and order: `codes[start..dead]` are the leaves
/// that got no children in this pass (depth `m - 1`), `codes[dead..end]` the
/// leaves the pass created (depth `m`), with their multiset fingerprints
/// (sums of [`Code::mix`]).
#[derive(Clone, Copy)]
struct Trails {
    start: u32,
    dead: u32,
    end: u32,
    fp_dead: u64,
    fp_new: u64,
}

impl Trails {
    fn dead(&self) -> (usize, usize, u64) {
        (self.start as usize, self.dead as usize, self.fp_dead)
    }

    fn new_leaves(&self) -> (usize, usize, u64) {
        (self.dead as usize, self.end as usize, self.fp_new)
    }
}

/// Leaf trail codes of Mordred's `BFSTree` rooted at `root` after each of the
/// `MAX_ORDER` expansion passes, appended to `codes[m - 1]` (see [`Trails`]).
///
/// Mordred's pass walks the whole tree depth-first, but the atoms of inner
/// nodes are visited already, so the only effect of a pass is that the
/// leaves of the previous pass are marked visited and expanded in depth-first
/// order - which is their creation order. A leaf that gets no children stays
/// a leaf for good (later re-expansions see a superset of visited atoms), so
/// the leaves after pass `m` are the dead leaves of passes `1..=m` and the
/// leaves pass `m` created; the codes hold their depth (number of steps), so
/// equal leaf multisets means equal multisets per pass.
///
/// An atom with a single neighbour (other than the root) entering the tree
/// has no unvisited neighbour left in the next pass (its parent was expanded,
/// and copies of the parent created in the same pass precede it): it is a
/// leaf for good, without being expanded or marked visited (no other atom can
/// see it).
fn tree_codes<C: Code>(g: &HGraph, root: usize, w: &mut TreeWork<C>, codes: &mut [Vec<C>], trails: &mut [Vec<Trails>]) {
    let TreeWork { frontier, next, h_leaves, visited, touched } = w;
    frontier.clear();
    h_leaves.clear();
    frontier.push((root as u32, C::step(g.root_step(root), 0, g.step_bits)));
    for m in 1..=MAX_ORDER {
        next.clear();
        let out = &mut codes[m - 1];
        let start = out.len();
        out.extend_from_slice(h_leaves);
        h_leaves.clear();
        for &(atom, code) in frontier.iter() {
            let atom = atom as usize;
            if !visited[atom] {
                visited[atom] = true;
                touched.push(atom as u32);
            }
            let mut leaf = true;
            for &(nb, step) in &g.nbrs[g.start[atom] as usize..g.start[atom + 1] as usize] {
                if !visited[nb as usize] {
                    leaf = false;
                    let c = code.or(C::step(step & !LEAF, m as u32, g.step_bits));
                    if step & LEAF != 0 {
                        h_leaves.push(c);
                    } else {
                        next.push((nb, c));
                    }
                }
            }
            if leaf {
                out.push(code);
            }
        }
        std::mem::swap(frontier, next);
        let dead = out.len();
        out.extend(frontier.iter().map(|&(_, c)| c));
        out.extend_from_slice(h_leaves);
        let fp = |codes: &[C]| codes.iter().fold(0u64, |acc, &c| acc.wrapping_add(c.mix()));
        trails[m - 1].push(Trails {
            start: start as u32,
            dead: dead as u32,
            end: out.len() as u32,
            fp_dead: fp(&out[start..dead]),
            fp_new: fp(&out[dead..]),
        });
    }
    for &a in touched.iter() {
        visited[a as usize] = false;
    }
    touched.clear();
}

/// Workspace of [`root_classes`].
#[derive(Default)]
struct ClassWork {
    /// (fingerprint and length of the dead leaves of passes `1..=m`, of the
    /// leaves pass `m` created) -> first class with that key.
    first_class: FxHashMap<(u64, u32, u64, u32), u32>,
    /// Per class: first member and the next class with the same key.
    first: Vec<usize>,
    next_same: Vec<u32>,
}

/// Classes of the roots at order `m`: equal multisets of the dead leaves of
/// each pass `1..=m` and of the leaves pass `m` created. `dead[i]` holds the
/// summed fingerprint and length of the dead leaves of root `i`; roots with
/// equal keys are compared exactly (sorting their codes in place). Appends
/// (size, last root) of each class to `out` (in order of first appearance)
/// and stores the class of each root in `ids`.
fn root_classes<C: Code>(
    m: usize,
    dead: &[(u64, u32)],
    codes: &mut [Vec<C>],
    trails: &[Vec<Trails>],
    w: &mut ClassWork,
    out: &mut Vec<(f64, usize)>,
    ids: &mut Vec<u32>,
) {
    const NONE: u32 = u32::MAX;
    w.first_class.clear();
    w.first.clear();
    w.next_same.clear();
    ids.clear();
    let t = &trails[m - 1];
    let mut same = |i: usize, j: usize| {
        let mut part = |k: usize, r: fn(&Trails) -> (usize, usize, u64)| {
            let ((s1, e1, _), (s2, e2, _)) = (r(&trails[k][i]), r(&trails[k][j]));
            let c = &mut codes[k];
            c[s1..e1].sort_unstable();
            c[s2..e2].sort_unstable();
            c[s1..e1] == c[s2..e2]
        };
        (0..m).all(|k| part(k, Trails::dead)) && part(m - 1, Trails::new_leaves)
    };
    for i in 0..dead.len() {
        let (_, e_new, fp_new) = t[i].new_leaves();
        let key = (dead[i].0, dead[i].1, fp_new, (e_new - t[i].dead as usize) as u32);
        let new_class = |w: &mut ClassWork, out: &mut Vec<(f64, usize)>| {
            out.push((1.0, i));
            w.first.push(i);
            w.next_same.push(NONE);
            out.len() as u32 - 1
        };
        let c = match w.first_class.get(&key) {
            None => {
                let c = new_class(w, out);
                w.first_class.insert(key, c);
                c
            }
            Some(&c0) => {
                let mut c = c0 as usize;
                loop {
                    if same(i, w.first[c]) {
                        out[c].0 += 1.0;
                        out[c].1 = i;
                        break c as u32;
                    }
                    if w.next_same[c] == NONE {
                        let nc = new_class(w, out);
                        w.next_same[c] = nc;
                        break nc;
                    }
                    c = w.next_same[c] as usize;
                }
            }
        };
        ids.push(c);
    }
}

/// Mordred InformationContent values in
/// [`mordred_information_content_names`] order.
pub fn mordred_information_content(mol: &Molecule, ctx: &mut CalculationContext) -> Vec<f64> {
    information_content(mol, ctx, true, false)
}

/// [`mordred_information_content`]; `derive = false` builds the trees of the
/// atoms with a single neighbour too, `wide` uses `u128` trail codes (for
/// tests).
fn information_content(mol: &Molecule, ctx: &mut CalculationContext, derive: bool, wide: bool) -> Vec<f64> {
    const NONE: u32 = u32::MAX;
    let mut out = vec![f64::NAN; MORDRED_INFORMATION_CONTENT_LEN];
    let Some(kekule) = super::mordred_kekulize::mordred_kekule_bond_types(mol, ctx) else {
        return out;
    };
    let g = HGraph::new(mol, &kekule);
    let n = g.len();
    if n == 0 {
        return out;
    }

    // The tree of an atom `r` with a single neighbour `p` is the tree of `p`
    // one level deeper (the tree of `p` has `r` as a leaf that never grows,
    // and `r` is visited from the start instead of from the second pass on,
    // which only `p` could see). So at order `m` its leaf trails are those of
    // `p` at order `m - 1` (without the trail to `r`) behind the step to `p`:
    // two such atoms are equivalent iff their steps to the neighbour are equal
    // (order 1) and their neighbours are equivalent at order `m - 1`; atoms
    // with other degrees differ in the root step.
    let derived = |r: usize| derive && g.degree(r) == 1;
    let mut root_of = vec![NONE; n];
    let mut roots: Vec<u32> = Vec::with_capacity(n);
    for r in 0..n {
        if !derived(r) {
            root_of[r] = roots.len() as u32;
            roots.push(r as u32);
        }
    }
    if !wide && (MAX_ORDER as u32 + 1) * g.step_bits <= u64::BITS {
        values::<u64>(&g, &roots, &root_of, &mut out);
    } else {
        values::<u128>(&g, &roots, &root_of, &mut out);
    }
    out
}

/// The InformationContent values (`out`) from the trees of `roots`
/// (`root_of`: position in `roots`, `NONE` for the derived atoms).
fn values<C: Code>(g: &HGraph, roots: &[u32], root_of: &[u32], out: &mut [f64]) {
    const NONE: u32 = u32::MAX;
    let n = g.len();
    let parent = |r: usize| g.nbrs[g.start[r] as usize];
    let n_roots = roots.len();
    let mut codes: Vec<Vec<C>> = (1..=MAX_ORDER).map(|m| Vec::with_capacity(n_roots * 4 * m)).collect();
    let mut trails: Vec<Vec<Trails>> = (0..MAX_ORDER).map(|_| Vec::with_capacity(n_roots)).collect();
    let mut work = TreeWork { visited: vec![false; n], ..Default::default() };
    for &root in roots {
        tree_codes(g, root as usize, &mut work, &mut codes, &mut trails);
    }

    let a = n as f64;
    let log2_a = a.log2();
    let log2_b = g.bond_order_sum.log2();
    // `p log2 p` per class size
    let mut plogp = vec![f64::NAN; n + 1];
    let mut classes: Vec<(f64, usize)> = Vec::new();
    let (mut ids, mut prev_ids) = (vec![0u32; n], vec![0u32; n]);
    // fingerprint and length of the dead leaves of the passes so far per root
    let mut dead = vec![(0u64, 0u32); n_roots];
    let (mut classes_of_roots, mut root_ids) = (Vec::new(), Vec::new());
    let mut class_work = ClassWork::default();
    let mut class_of_root: Vec<u32> = Vec::new();
    let mut class_of_derived: FxHashMap<u64, u32> = FxHashMap::default();
    for m in 0..N_ORDERS {
        classes.clear();
        if m == 0 {
            let mut class_of_z = [NONE; 256];
            for i in 0..n {
                let slot = &mut class_of_z[g.z[i] as usize];
                if *slot == NONE {
                    *slot = classes.len() as u32;
                    classes.push((1.0, i));
                } else {
                    classes[*slot as usize].0 += 1.0;
                    classes[*slot as usize].1 = i;
                }
                ids[i] = *slot;
            }
        } else {
            for (d, t) in dead.iter_mut().zip(&trails[m - 1]) {
                let (s, e, fp) = t.dead();
                *d = (d.0.wrapping_add(fp), d.1 + (e - s) as u32);
            }
            classes_of_roots.clear();
            root_classes(m, &dead, &mut codes, &trails, &mut class_work, &mut classes_of_roots, &mut root_ids);
            // all atoms, numbered in order of first appearance
            class_of_root.clear();
            class_of_root.resize(classes_of_roots.len(), NONE);
            class_of_derived.clear();
            let mut last_key = u64::MAX;
            let mut last_class = NONE;
            for i in 0..n {
                let slot = if root_of[i] != NONE {
                    &mut class_of_root[root_ids[root_of[i] as usize] as usize]
                } else {
                    let (p, step_p) = parent(i);
                    let prev = if m == 1 { 0 } else { prev_ids[p as usize] };
                    let key = (g.root_step(i) as u64) << 48 | ((step_p & !LEAF) as u64) << 32 | prev as u64;
                    if key == last_key {
                        // e.g. the next hydrogen on the same atom
                        &mut last_class
                    } else {
                        last_key = key;
                        class_of_derived.entry(key).or_insert(NONE)
                    }
                };
                if *slot == NONE {
                    *slot = classes.len() as u32;
                    classes.push((1.0, i));
                } else {
                    classes[*slot as usize].0 += 1.0;
                    classes[*slot as usize].1 = i;
                }
                ids[i] = *slot;
                if root_of[i] == NONE {
                    last_class = *slot;
                }
            }
        }
        std::mem::swap(&mut ids, &mut prev_ids);
        let (mut ic, mut mic, mut zmic) = (0.0f64, 0.0f64, 0.0f64);
        for &(size, rep) in &classes {
            let t = &mut plogp[size as usize];
            if t.is_nan() {
                let p = size / a;
                *t = p * p.log2();
            }
            let t = *t;
            ic -= t;
            mic -= g.mass[rep] * t;
            zmic -= size * g.z[rep] as f64 * t;
        }
        out[m] = ic;
        out[N_ORDERS + m] = a * ic;
        out[2 * N_ORDERS + m] = if log2_a == 0.0 { f64::NAN } else { ic / log2_a };
        out[3 * N_ORDERS + m] = if log2_b == 0.0 || !log2_b.is_finite() { f64::NAN } else { ic / log2_b };
        out[4 * N_ORDERS + m] = log2_a - ic;
        out[5 * N_ORDERS + m] = mic;
        out[6 * N_ORDERS + m] = zmic;
    }
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
    fn single_neighbour_classes_follow_their_neighbours() {
        let mut ctx = CalculationContext::new();
        for smi in [
            "C",
            "CCO",
            "CC(C)(C)C",
            "c1ccccc1O",
            "OC(=O)C(N)Cc1ccc(O)cc1",
            "C1CC2CCC1CC2",
            "[NH4+].[Cl-]",
            "CC(=O)Nc1ccc(O)cc1",
            "C=CC#N",
            "[OH-]",
            "N",
            "C1=CC=CC=C1C1=CC=CC=C1",
            "CS(=O)(=O)NC1CCCCC1",
            "OCC(O)C(O)C(O)C(O)CO",
            "[H]C([H])([H])C",
            "[2H]OC",
            "CC[N+](C)(C)CC",
            "O=C1NC(=O)C(=O)N1",
            "[H][H]",
            "F",
            "ClCl",
            "[Na+].[Cl-]",
            "C#N",
            "FC(F)(F)C(=O)[O-]",
            "O=S(=O)(O)c1ccc(Br)cc1I",
            "[He]",
        ] {
            let mol = parse_smiles(smi).unwrap();
            let fast = information_content(&mol, &mut ctx, true, false);
            for (derive, wide) in [(false, false), (true, true), (false, true)] {
                let other = information_content(&mol, &mut ctx, derive, wide);
                for (i, (x, y)) in fast.iter().zip(&other).enumerate() {
                    assert!(x == y || (x.is_nan() && y.is_nan()), "{smi} {derive} {wide} column {i}: {x} vs {y}");
                }
            }
        }
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
