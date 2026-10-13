//! Mordred (mordred-community 2.0.7) `Chi` molecular connectivity indices.
//!
//! Output (56 values, Mordred's preset order, see [`MORDRED_CHI_NAMES`]):
//! `Xch-{3..7}{d,dv}` (chain), `Xc-{3..6}{d,dv}` (cluster),
//! `Xpc-{4..6}{d,dv}` (path-cluster), `Xp-{0..7}{d,dv}` (path) and their
//! averages `AXp-{0..7}{d,dv}`.
//!
//! Mordred works on the hydrogen-suppressed graph (`Chem.RemoveHs`, hydrogens
//! RDKit keeps stay graph atoms) and enumerates every connected bond subgraph
//! of `order` bonds with RDKit's `FindAllSubgraphsOfLengthN(mol, order)`,
//! which ignores bonds to hydrogen atoms (`useHs = false`). A subgraph is a
//! *chain* if it contains a cycle, a *path* if every atom has subgraph degree
//! 1 or 2, a *path-cluster* if it also has an atom of degree >= 3 and one of
//! degree 2, otherwise a *cluster*. Order 0 is the path of every single atom
//! (hydrogen atoms included). The index is `Σ_subgraphs Π_atoms P^-1/2` with
//! `P` = `d` (number of non-hydrogen neighbours) or `dv` (Mordred valence
//! electrons `(Zv - h) / (Z - Zv - 1)`, 0 for hydrogen). The value is missing
//! (NaN) when some subgraph has `Π P <= 0`, and the average is missing when
//! there is no subgraph of the type.
//!
//! The subgraphs are enumerated with the ESU algorithm (Wernicke 2006) on the
//! line graph, which visits every connected bond set of up to 7 bonds exactly
//! once; atom degrees, atom count and the property products are updated
//! incrementally while the set grows.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::elements::outer_electrons;
use crate::mol::molecule::Molecule;

/// Largest subgraph order (bonds) of Mordred's Chi preset.
const MAX_ORDER: usize = 7;

/// Subgraph types in Mordred's `ChiType` meaning.
const CHAIN: usize = 0;
const CLUSTER: usize = 1;
const PATH_CLUSTER: usize = 2;
const PATH: usize = 3;

/// Number of output columns.
pub const MORDRED_CHI_LEN: usize = 56;

/// Column names in Mordred's preset order.
pub fn mordred_chi_names() -> Vec<String> {
    let mut v = Vec::with_capacity(MORDRED_CHI_LEN);
    for (ty, orders) in [("ch", 3..=7), ("c", 3..=6), ("pc", 4..=6)] {
        for p in ["d", "dv"] {
            v.extend(orders.clone().map(|o| format!("X{ty}-{o}{p}")));
        }
    }
    for p in ["d", "dv"] {
        for avg in ["", "A"] {
            v.extend((0..=MAX_ORDER).map(|o| format!("{avg}Xp-{o}{p}")));
        }
    }
    v
}

/// Per-atom factor of the subgraph product: `|P|^-1/2` plus the sign
/// information needed to reproduce Mordred's `Π P <= 0` check.
#[derive(Clone, Copy)]
struct Factor {
    inv_sqrt: f64,
    zero: bool,
    negative: bool,
}

impl Factor {
    fn new(p: f64) -> Self {
        Factor { inv_sqrt: 1.0 / p.abs().sqrt(), zero: p == 0.0, negative: p < 0.0 }
    }
}

/// Mordred `dv` (`get_valence_electrons`) of atom `a`.
pub(crate) fn mordred_dv(mol: &Molecule, a: usize) -> f64 {
    let atom = &mol.atoms[a];
    let z = atom.atomic_number as i32;
    if z == 1 {
        return 0.0;
    }
    let zv = outer_electrons(atom.atomic_number) as i32 - atom.formal_charge as i32;
    let zc = z - atom.formal_charge as i32;
    let h = atom.total_h() as i32 + h_neighbors(mol, a) as i32;
    (zv - h) as f64 / (zc - zv - 1) as f64
}

/// Number of hydrogen-atom neighbours of `a`.
fn h_neighbors(mol: &Molecule, a: usize) -> usize {
    mol.adjacency[a].iter().filter(|&&b| mol.atoms[other(mol, b as usize, a)].atomic_number == 1).count()
}

#[inline]
fn other(mol: &Molecule, bond: usize, atom: usize) -> usize {
    let b = &mol.bonds[bond];
    if b.source as usize == atom { b.target as usize } else { b.source as usize }
}

/// Accumulated sums per (type, order, property).
struct Acc {
    sum: [[[f64; 2]; MAX_ORDER + 1]; 4],
    count: [[u64; MAX_ORDER + 1]; 4],
    fail: [[[bool; 2]; MAX_ORDER + 1]; 4],
}

/// ESU state on the line graph of the hydrogen-free bonds.
struct Esu<'a> {
    /// Bond ends per line-graph vertex.
    ends: &'a [(u32, u32)],
    /// Line-graph adjacency (CSR).
    nbr_start: &'a [u32],
    nbr: &'a [u32],
    /// Per-atom factors for `d` and `dv`.
    factors: &'a [[Factor; 2]],
    /// Number of subgraph edges equal or adjacent to each edge (0 = outside
    /// the closed neighbourhood of the current subgraph).
    cover: Vec<u16>,
    /// Extension sets of all recursion levels, stacked.
    ext: Vec<u32>,
    /// Subgraph degree per atom.
    deg: Vec<u8>,
    n_atoms: u32,
    n_deg2: u32,
    n_deg3: u32,
    prod: [f64; 2],
    zeros: [u32; 2],
    negs: [u32; 2],
    root: u32,
    acc: Acc,
}

impl Esu<'_> {
    #[inline]
    fn nbrs(&self, e: u32) -> std::ops::Range<usize> {
        self.nbr_start[e as usize] as usize..self.nbr_start[e as usize + 1] as usize
    }

    #[inline]
    fn add_atom_degree(&mut self, x: usize) {
        let old = self.deg[x];
        self.deg[x] = old + 1;
        match old {
            0 => {
                self.n_atoms += 1;
                for p in 0..2 {
                    let f = self.factors[x][p];
                    self.prod[p] *= f.inv_sqrt;
                    self.zeros[p] += f.zero as u32;
                    self.negs[p] += f.negative as u32;
                }
            }
            1 => self.n_deg2 += 1,
            2 => {
                self.n_deg2 -= 1;
                self.n_deg3 += 1;
            }
            _ => {}
        }
    }

    #[inline]
    fn remove_atom_degree(&mut self, x: usize) {
        let new = self.deg[x] - 1;
        self.deg[x] = new;
        match new {
            0 => {
                self.n_atoms -= 1;
                for p in 0..2 {
                    let f = self.factors[x][p];
                    self.zeros[p] -= f.zero as u32;
                    self.negs[p] -= f.negative as u32;
                }
            }
            1 => self.n_deg2 -= 1,
            2 => {
                self.n_deg2 += 1;
                self.n_deg3 -= 1;
            }
            _ => {}
        }
    }

    /// Adds edge `e` to the subgraph (and to the cover of its neighbourhood).
    #[inline]
    fn push_edge(&mut self, e: u32) {
        self.cover[e as usize] += 1;
        for k in self.nbrs(e) {
            let u = self.nbr[k] as usize;
            self.cover[u] += 1;
        }
        let (a, b) = self.ends[e as usize];
        self.add_atom_degree(a as usize);
        self.add_atom_degree(b as usize);
    }

    #[inline]
    fn pop_edge(&mut self, e: u32) {
        self.cover[e as usize] -= 1;
        for k in self.nbrs(e) {
            let u = self.nbr[k] as usize;
            self.cover[u] -= 1;
        }
        let (a, b) = self.ends[e as usize];
        self.remove_atom_degree(b as usize);
        self.remove_atom_degree(a as usize);
    }

    /// Records the current subgraph of `size` bonds.
    #[inline]
    fn record(&mut self, size: usize) {
        let ty = if self.n_atoms as usize <= size {
            CHAIN
        } else if self.n_deg3 == 0 {
            PATH
        } else if self.n_deg2 > 0 {
            PATH_CLUSTER
        } else {
            CLUSTER
        };
        self.acc.count[ty][size] += 1;
        for p in 0..2 {
            if self.zeros[p] > 0 || self.negs[p] % 2 == 1 {
                self.acc.fail[ty][size][p] = true;
            } else {
                self.acc.sum[ty][size][p] += self.prod[p];
            }
        }
    }

    /// Extends the current subgraph of `size` bonds by every element of the
    /// extension set `ext[start..end]` (ESU `ExtendSubgraph`).
    fn extend(&mut self, size: usize, start: usize, end: usize) {
        if size == MAX_ORDER {
            return;
        }
        if size + 1 == MAX_ORDER {
            // the children are leaves of the search tree: only their atoms
            // matter, not their neighbourhoods
            for i in start..end {
                let (a, b) = self.ends[self.ext[i] as usize];
                let saved = self.prod;
                self.add_atom_degree(a as usize);
                self.add_atom_degree(b as usize);
                self.record(MAX_ORDER);
                self.remove_atom_degree(b as usize);
                self.remove_atom_degree(a as usize);
                self.prod = saved;
            }
            return;
        }
        let mut cur = end;
        while cur > start {
            cur -= 1;
            let w = self.ext[cur];
            let child = self.ext.len();
            self.ext.extend_from_within(start..cur);
            // exclusive neighbourhood of w (not in or next to the subgraph)
            for k in self.nbrs(w) {
                let u = self.nbr[k];
                if u > self.root && self.cover[u as usize] == 0 {
                    self.ext.push(u);
                }
            }
            let saved = self.prod;
            self.push_edge(w);
            self.record(size + 1);
            let child_end = self.ext.len();
            self.extend(size + 1, child, child_end);
            self.pop_edge(w);
            self.prod = saved;
            self.ext.truncate(child);
        }
    }
}

/// Mordred Chi values in [`mordred_chi_names`] order.
pub fn mordred_chi(mol: &Molecule) -> Vec<f64> {
    let n = mol.num_atoms();
    let mut factors = Vec::with_capacity(n);
    for a in 0..n {
        let d = mol.adjacency[a].len() - h_neighbors(mol, a);
        factors.push([Factor::new(d as f64), Factor::new(mordred_dv(mol, a))]);
    }

    // line graph of the bonds between non-hydrogen atoms
    let is_h = |a: usize| mol.atoms[a].atomic_number == 1;
    let mut edge_of_bond = vec![u32::MAX; mol.num_bonds()];
    let mut ends: Vec<(u32, u32)> = Vec::with_capacity(mol.num_bonds());
    for (bi, b) in mol.bonds.iter().enumerate() {
        if !is_h(b.source as usize) && !is_h(b.target as usize) {
            edge_of_bond[bi] = ends.len() as u32;
            ends.push((b.source as u32, b.target as u32));
        }
    }
    let mut nbr_start = Vec::with_capacity(ends.len() + 1);
    let mut nbr = Vec::new();
    for (e, &(a, b)) in ends.iter().enumerate() {
        nbr_start.push(nbr.len() as u32);
        for x in [a as usize, b as usize] {
            for &bi in &mol.adjacency[x] {
                let f = edge_of_bond[bi as usize];
                if f != u32::MAX && f as usize != e {
                    nbr.push(f);
                }
            }
        }
    }
    nbr_start.push(nbr.len() as u32);

    let mut esu = Esu {
        ends: &ends,
        nbr_start: &nbr_start,
        nbr: &nbr,
        factors: &factors,
        cover: vec![0; ends.len()],
        ext: Vec::new(),
        deg: vec![0; n],
        n_atoms: 0,
        n_deg2: 0,
        n_deg3: 0,
        prod: [1.0; 2],
        zeros: [0; 2],
        negs: [0; 2],
        root: 0,
        acc: Acc {
            sum: [[[0.0; 2]; MAX_ORDER + 1]; 4],
            count: [[0; MAX_ORDER + 1]; 4],
            fail: [[[false; 2]; MAX_ORDER + 1]; 4],
        },
    };
    for v in 0..ends.len() as u32 {
        esu.root = v;
        esu.ext.clear();
        for k in esu.nbrs(v) {
            let u = esu.nbr[k];
            if u > v {
                esu.ext.push(u);
            }
        }
        esu.push_edge(v);
        esu.record(1);
        let end = esu.ext.len();
        esu.extend(1, 0, end);
        esu.pop_edge(v);
        esu.prod = [1.0; 2];
    }
    let mut acc = esu.acc;

    // order 0: every atom (hydrogens included) is a path
    acc.count[PATH][0] = n as u64;
    for f in &factors {
        for p in 0..2 {
            if f[p].zero || f[p].negative {
                acc.fail[PATH][0][p] = true;
            } else {
                acc.sum[PATH][0][p] += f[p].inv_sqrt;
            }
        }
    }

    let value = |ty: usize, order: usize, p: usize, averaged: bool| -> f64 {
        if acc.fail[ty][order][p] {
            return f64::NAN;
        }
        let x = acc.sum[ty][order][p];
        if averaged {
            let c = acc.count[ty][order];
            if c == 0 { f64::NAN } else { x / c as f64 }
        } else {
            x
        }
    };
    let mut out = Vec::with_capacity(MORDRED_CHI_LEN);
    for (ty, orders) in [(CHAIN, 3..=7), (CLUSTER, 3..=6), (PATH_CLUSTER, 4..=6)] {
        for p in 0..2 {
            out.extend(orders.clone().map(|o| value(ty, o, p, false)));
        }
    }
    for p in 0..2 {
        for averaged in [false, true] {
            out.extend((0..=MAX_ORDER).map(|o| value(PATH, o, p, averaged)));
        }
    }
    out
}

/// Mordred Chi descriptor family (56 values, see the module docs).
pub struct MordredChi;

impl Descriptor for MordredChi {
    fn name(&self) -> &str {
        "MordredChi"
    }

    fn calculate(&self, mol: &Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::FloatVec(mordred_chi(mol))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn chi(smiles: &str) -> std::collections::HashMap<String, f64> {
        let mol = parse_smiles(smiles).unwrap();
        mordred_chi_names().into_iter().zip(mordred_chi(&mol)).collect()
    }

    #[test]
    fn names_are_mordred_preset_order() {
        let names = mordred_chi_names();
        assert_eq!(names.len(), MORDRED_CHI_LEN);
        assert_eq!(names[0], "Xch-3d");
        assert_eq!(names[10], "Xc-3d");
        assert_eq!(names[18], "Xpc-4d");
        assert_eq!(names[24], "Xp-0d");
        assert_eq!(names[32], "AXp-0d");
        assert_eq!(names[55], "AXp-7dv");
    }

    #[test]
    fn isobutane_cluster_and_paths() {
        // Expected values: Mordred (mordred-community 2.0.7).
        let v = chi("CC(C)C");
        assert!((v["Xc-3d"] - 1.0 / 3f64.sqrt()).abs() < 1e-12);
        assert!((v["Xp-1d"] - 3.0 / 3f64.sqrt()).abs() < 1e-12);
        assert!((v["Xp-2d"] - 3.0 / 3f64.sqrt()).abs() < 1e-12);
        assert_eq!(v["Xp-3d"], 0.0);
        assert!(v["AXp-3d"].is_nan());
        assert!((v["Xp-0d"] - (3.0 + 1.0 / 3f64.sqrt())).abs() < 1e-12);
    }

    #[test]
    fn cyclopropane_is_a_chain() {
        let v = chi("C1CC1");
        assert!((v["Xch-3d"] - 0.125f64.sqrt()).abs() < 1e-12);
        assert!((v["Xp-2d"] - 3.0 * 0.125f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn isolated_atom_order_zero_is_missing() {
        let v = chi("[Na+].[Cl-]");
        assert!(v["Xp-0d"].is_nan());
        assert_eq!(v["Xp-1d"], 0.0);
    }
}
