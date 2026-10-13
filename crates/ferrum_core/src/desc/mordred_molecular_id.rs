//! Mordred (mordred-community 2.0.7) `MolecularId` descriptors.
//!
//! Output (12 values, Mordred's order): `MID`, `AMID`, `MID_h`, `AMID_h`,
//! `MID_C`, `AMID_C`, `MID_N`, `AMID_N`, `MID_O`, `AMID_O`, `MID_X`, `AMID_X`.
//!
//! Mordred works on the hydrogen-suppressed graph (`Chem.RemoveHs`; hydrogens
//! RDKit keeps stay graph atoms). The atomic ID of atom `s` is
//! `1 + ½ Σ_paths 1/√w` over all simple paths starting at `s`, where `w` is
//! the product of the edge weights `deg(i)·deg(j)` along the path; a path is
//! extended only while its weight stays below `lim = int(1 / 1e-10²)`
//! (Mordred's `eps`), the path that crosses the limit still counts. `MID`
//! sums the atomic IDs of all atoms, `_h` of the atoms other than H and C,
//! `_C`/`_N`/`_O` of that element and `_X` of the halogens; the `A` variants
//! divide by the number of atoms. Missing for molecules with more than one
//! fragment (`require_connected`).
//!
//! The simple paths are exponentially many in fused ring systems, so the
//! search sums two kinds of subgraphs in closed form:
//!
//! - *Pendant trees* (atoms removed by repeatedly stripping degree-1 atoms):
//!   a path entering the tree below atom `x` can only go further down.
//! - *Chains* (atoms with exactly two neighbours in the remaining core)
//!   entered from a branch atom: the path runs along the chain, possibly ends
//!   in it or in a pendant tree of a chain atom, or leaves it at the other
//!   end.
//!
//! In both cases the terms a path collects inside the subgraph are
//! `1/√W · 1/√w(y)` for the subgraph's end points `y`, included when
//! `W·w_pre(y) < lim` (`W`: weight of the path before entering, `w(y)`: weight
//! of the part inside the subgraph, `w_pre(y)`: the same without its last
//! edge). The terms are sorted by `w_pre` with prefix sums, so each entry is a
//! binary search. The chain of the start atom (or the chain its pendant tree
//! hangs on) is walked atom by atom, since part of it is already on the path.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Number of MolecularId columns.
pub const MORDRED_MOLECULAR_ID_LEN: usize = 12;

/// MolecularId column names in Mordred's preset order.
pub fn mordred_molecular_id_names() -> Vec<String> {
    let mut v = Vec::with_capacity(MORDRED_MOLECULAR_ID_LEN);
    for t in ["", "_h", "_C", "_N", "_O", "_X"] {
        v.push(format!("MID{t}"));
        v.push(format!("AMID{t}"));
    }
    v
}

/// Mordred `AtomicId.lim = int(1.0 / (1e-10 ** 2))`.
const LIMIT: u128 = 99_999_999_999_999_983_616;
const NONE: u32 = u32::MAX;

/// Closed-form contributions of one subgraph entry: `keys` (`w_pre`, sorted)
/// and the prefix sums of `1/√w` at `range`.
#[derive(Clone, Copy)]
struct Entry {
    start: u32,
    end: u32,
}

/// Directed chain entered from a branch atom.
#[derive(Clone, Copy)]
struct Chain {
    terms: Entry,
    /// Undirected chain id (for the start-atom exclusion).
    id: u32,
    first: u32,
    last: u32,
    /// Branch atom at the other end (may be the entry atom itself).
    end: u32,
    /// Weight up to the last chain atom (`u128::MAX`: the end is unreachable).
    w_last: u128,
    /// Weight up to the end atom and its `1/√`.
    w_end: u128,
    inv_sqrt_end: f64,
}

struct Graph {
    start: Vec<usize>,
    /// (neighbour, edge weight, 1/√edge weight)
    adj: Vec<(u32, u128, f64)>,
    /// Pendant atom `x` -> the neighbour it hangs on (`NONE` for core atoms).
    parent: Vec<u32>,
    /// Pendant-tree terms per pendant atom.
    pendant: Vec<Entry>,
    /// Chain shortcut per adjacency slot (`NONE`: none).
    chain_of_slot: Vec<u32>,
    chains: Vec<Chain>,
    /// Undirected chain id of each atom (core chain atoms and the pendant
    /// atoms hanging on them).
    chain_id: Vec<u32>,
    keys: Vec<u128>,
    sums: Vec<f64>,
}

impl Graph {
    #[inline]
    fn slots(&self, a: usize) -> std::ops::Range<usize> {
        self.start[a]..self.start[a + 1]
    }

    /// `Σ 1/√w` of the terms of `e` with `weight·w_pre < lim`.
    #[inline]
    fn query(&self, e: Entry, weight: u128) -> f64 {
        let keys = &self.keys[e.start as usize..e.end as usize];
        let cnt = keys.partition_point(|&k| weight.saturating_mul(k) < LIMIT);
        if cnt == 0 { 0.0 } else { self.sums[e.start as usize + cnt - 1] }
    }

    /// Appends `terms` (w_pre, 1/√w) sorted with prefix sums.
    fn push_terms(&mut self, terms: &mut [(u128, f64)]) -> Entry {
        terms.sort_by_key(|t| t.0);
        let start = self.keys.len() as u32;
        let mut acc = 0.0;
        for &(k, v) in terms.iter() {
            acc += v;
            self.keys.push(k);
            self.sums.push(acc);
        }
        Entry { start, end: self.keys.len() as u32 }
    }

    fn new(mol: &Molecule) -> Self {
        let n = mol.num_atoms();
        let mut start = Vec::with_capacity(n + 1);
        let mut adj: Vec<(u32, u128, f64)> = Vec::with_capacity(2 * mol.num_bonds());
        for a in 0..n {
            start.push(adj.len());
            for &bi in &mol.adjacency[a] {
                let b = &mol.bonds[bi as usize];
                let o = if b.source as usize == a { b.target } else { b.source } as usize;
                let w = (mol.adjacency[a].len() * mol.adjacency[o].len()) as u128;
                adj.push((o as u32, w, 1.0 / (w as f64).sqrt()));
            }
        }
        start.push(adj.len());
        let n_slots = adj.len();
        let mut g = Graph {
            start,
            adj,
            parent: vec![NONE; n],
            pendant: vec![Entry { start: 0, end: 0 }; n],
            chain_of_slot: vec![NONE; n_slots],
            chains: Vec::new(),
            chain_id: vec![NONE; n],
            keys: Vec::new(),
            sums: Vec::new(),
        };

        // pendant trees: strip degree-1 atoms (in strip order)
        let mut stripped: Vec<u32> = Vec::new();
        {
            let mut deg: Vec<u32> = (0..n).map(|a| g.slots(a).len() as u32).collect();
            let mut stack: Vec<u32> = (0..n as u32).filter(|&a| deg[a as usize] == 1).collect();
            let mut removed = vec![false; n];
            while let Some(x) = stack.pop() {
                let x = x as usize;
                if removed[x] || deg[x] != 1 {
                    continue;
                }
                let Some(p) = g.adj[g.slots(x)].iter().map(|e| e.0 as usize).find(|&v| !removed[v]) else {
                    continue;
                };
                removed[x] = true;
                g.parent[x] = p as u32;
                stripped.push(x as u32);
                deg[p] -= 1;
                if deg[p] == 1 {
                    stack.push(p as u32);
                }
            }
        }
        // raw pendant terms (w_pre, 1/√w) per pendant atom
        let mut raw_start = vec![0usize; n + 1];
        let mut raw: Vec<(u128, f64)> = Vec::new();
        let mut dfs: Vec<(u32, u32, u128, u128)> = Vec::new();
        for x in 0..n {
            raw_start[x] = raw.len();
            let p = g.parent[x];
            if p == NONE {
                continue;
            }
            let e0 = g.adj[g.slots(x)].iter().find(|e| e.0 == p).expect("parent bond").1;
            dfs.clear();
            dfs.push((x as u32, p, 1, e0));
            while let Some((y, from, w_pre, w)) = dfs.pop() {
                raw.push((w_pre, 1.0 / (w as f64).sqrt()));
                if w >= LIMIT {
                    continue;
                }
                for &(c, ew, _) in &g.adj[g.slots(y as usize)] {
                    if c != from {
                        dfs.push((c, y, w, w * ew));
                    }
                }
            }
        }
        raw_start[n] = raw.len();
        let mut terms: Vec<(u128, f64)> = Vec::new();
        for x in 0..n {
            if g.parent[x] != NONE {
                terms.clear();
                terms.extend_from_slice(&raw[raw_start[x]..raw_start[x + 1]]);
                g.pendant[x] = g.push_terms(&mut terms);
            }
        }

        // chains of the core
        let core = |g: &Graph, a: usize| g.parent[a] == NONE;
        let core_degree: Vec<u32> = (0..n)
            .map(|a| if core(&g, a) { g.adj[g.slots(a)].iter().filter(|e| core(&g, e.0 as usize)).count() as u32 } else { 0 })
            .collect();
        let chain_atom: Vec<bool> = (0..n).map(|a| core_degree[a] == 2 && g.parent[a] == NONE).collect();
        let is_chain_atom = |a: usize| chain_atom[a];
        let mut n_ids = 0u32;
        let mut atoms: Vec<u32> = Vec::new();
        for u in 0..n {
            if !core(&g, u) || core_degree[u] == 2 {
                continue;
            }
            for k in g.slots(u) {
                let v = g.adj[k].0 as usize;
                if !is_chain_atom(v) {
                    continue;
                }
                // walk the chain u -> v -> ... -> end
                atoms.clear();
                terms.clear();
                let (mut prev, mut cur) = (u, v);
                let (mut w, mut inv) = (1u128, 1.0f64);
                let (mut ew, mut einv) = (g.adj[k].1, g.adj[k].2);
                let mut reachable = true;
                let end = loop {
                    atoms.push(cur as u32);
                    if reachable {
                        terms.push((w, 0.0));
                        w = w.saturating_mul(ew);
                        inv *= einv;
                        terms.last_mut().expect("term").1 = inv;
                        // pendant trees hanging on the chain atom
                        for s in g.slots(cur) {
                            let x = g.adj[s].0 as usize;
                            if g.parent[x] == cur as u32 {
                                for &(k2, v2) in &raw[raw_start[x]..raw_start[x + 1]] {
                                    terms.push((w.saturating_mul(k2), v2 * inv));
                                }
                            }
                        }
                        reachable = w < LIMIT;
                    }
                    let (next, nw, ninv) = g.adj[g.slots(cur)]
                        .iter()
                        .find(|e| e.0 as usize != prev && core(&g, e.0 as usize))
                        .map(|e| (e.0 as usize, e.1, e.2))
                        .expect("chain continues");
                    if !is_chain_atom(next) {
                        ew = nw;
                        einv = ninv;
                        break next;
                    }
                    prev = cur;
                    cur = next;
                    ew = nw;
                    einv = ninv;
                };
                let id = match g.chain_id[v] {
                    NONE => {
                        n_ids += 1;
                        for &a in &atoms {
                            g.chain_id[a as usize] = n_ids - 1;
                        }
                        n_ids - 1
                    }
                    id => id,
                };
                let entry = g.push_terms(&mut terms);
                g.chain_of_slot[k] = g.chains.len() as u32;
                g.chains.push(Chain {
                    terms: entry,
                    id,
                    first: atoms[0],
                    last: *atoms.last().expect("chain atom"),
                    end: end as u32,
                    w_last: if reachable { w } else { u128::MAX },
                    w_end: w.saturating_mul(ew),
                    inv_sqrt_end: inv * einv,
                });
            }
        }
        // pendant atoms belong to the chain of the core atom they hang on
        // (parents are stripped after their children: walk in reverse order)
        for &x in stripped.iter().rev() {
            let p = g.parent[x as usize] as usize;
            g.chain_id[x as usize] = g.chain_id[p];
        }
        g
    }
}

/// Depth-first path search from one start atom.
struct Search<'a> {
    g: &'a Graph,
    visited: Vec<bool>,
    /// Chain containing the start atom (walked atom by atom).
    dirty: u32,
    id: f64,
}

impl Search<'_> {
    /// Extends the path ending at `u` (weight `weight`, `1/√weight` = `inv`).
    /// Contributions are products of the per-edge `1/√w_e` (a few ulp from
    /// Mordred's `1/√w`); the cutoff uses the exact integer weight.
    fn search(&mut self, u: usize, weight: u128, inv: f64) {
        let g = self.g;
        for k in g.slots(u) {
            let (v, ew, einv) = g.adj[k];
            let v = v as usize;
            if self.visited[v] {
                continue;
            }
            if g.parent[v] == u as u32 {
                self.id += g.query(g.pendant[v], weight) * inv;
                continue;
            }
            let ci = g.chain_of_slot[k];
            if ci != NONE && g.chains[ci as usize].id != self.dirty {
                let c = g.chains[ci as usize];
                self.id += g.query(c.terms, weight) * inv;
                let end = c.end as usize;
                if !self.visited[end] && weight.saturating_mul(c.w_last) < LIMIT {
                    let w = weight.saturating_mul(c.w_end);
                    let ci = inv * c.inv_sqrt_end;
                    self.id += ci;
                    if w < LIMIT {
                        let (first, last) = (c.first as usize, c.last as usize);
                        self.visited[first] = true;
                        self.visited[last] = true;
                        self.visited[end] = true;
                        self.search(end, w, ci);
                        self.visited[end] = false;
                        self.visited[last] = false;
                        self.visited[first] = false;
                    }
                }
                continue;
            }
            let w = ew * weight;
            let c = inv * einv;
            self.id += c;
            if w < LIMIT {
                self.visited[v] = true;
                self.search(v, w, c);
                self.visited[v] = false;
            }
        }
    }
}

/// Mordred MolecularId values in [`mordred_molecular_id_names`] order.
pub fn mordred_molecular_id(mol: &Molecule) -> Vec<f64> {
    let n = mol.num_atoms();
    if n == 0 || crate::algo::components::num_fragments(mol) != 1 {
        return vec![f64::NAN; MORDRED_MOLECULAR_ID_LEN];
    }
    let g = Graph::new(mol);
    let mut s = Search { g: &g, visited: vec![false; n], dirty: NONE, id: 0.0 };
    // [all, hetero, C, N, O, halogen]
    let mut sums = [0.0f64; 6];
    for a in 0..n {
        s.id = 0.0;
        s.dirty = g.chain_id[a];
        s.visited[a] = true;
        s.search(a, 1, 1.0);
        s.visited[a] = false;
        let aid = 1.0 + s.id / 2.0;
        let z = mol.atoms[a].atomic_number;
        sums[0] += aid;
        if z != 1 && z != 6 {
            sums[1] += aid;
        }
        match z {
            6 => sums[2] += aid,
            7 => sums[3] += aid,
            8 => sums[4] += aid,
            9 | 17 | 35 | 53 | 85 | 117 => sums[5] += aid,
            _ => {}
        }
    }
    let mut out = Vec::with_capacity(MORDRED_MOLECULAR_ID_LEN);
    for v in sums {
        out.push(v);
        out.push(v / n as f64);
    }
    out
}

/// Mordred MolecularId descriptor family (12 values, see the module docs).
pub struct MordredMolecularId;

impl Descriptor for MordredMolecularId {
    fn name(&self) -> &str {
        "MordredMolecularId"
    }

    fn calculate(&self, mol: &Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::FloatVec(mordred_molecular_id(mol))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Mordred's `AtomicId` search, path by path.
    fn reference(mol: &Molecule) -> Vec<f64> {
        let n = mol.num_atoms();
        let deg = |a: usize| mol.adjacency[a].len() as u128;
        let nbrs = |a: usize| -> Vec<usize> {
            mol.adjacency[a]
                .iter()
                .map(|&b| {
                    let b = &mol.bonds[b as usize];
                    if b.source as usize == a { b.target as usize } else { b.source as usize }
                })
                .collect()
        };
        fn go(u: usize, w: u128, vis: &mut Vec<bool>, id: &mut f64, nb: &dyn Fn(usize) -> Vec<usize>, deg: &dyn Fn(usize) -> u128) {
            for v in nb(u) {
                if vis[v] {
                    continue;
                }
                let x = w * deg(u) * deg(v);
                *id += 1.0 / (x as f64).sqrt();
                if x < LIMIT {
                    vis[v] = true;
                    go(v, x, vis, id, nb, deg);
                    vis[v] = false;
                }
            }
        }
        (0..n)
            .map(|a| {
                let mut vis = vec![false; n];
                vis[a] = true;
                let mut id = 0.0;
                go(a, 1, &mut vis, &mut id, &nbrs, &deg);
                1.0 + id / 2.0
            })
            .collect()
    }

    #[test]
    fn closed_forms_match_the_plain_search() {
        for smi in [
            "CC", "CCO", "CC(C)(C)C", "C1CC1", "c1ccccc1", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "C1CCC2(CC1)CCCC2", "OC1CCC(CC1)C1CCCCC1", "C1CC2CCC1CC2", "C12C3C4C1C5C2C3C45",
            "CC1CCC2C(C)C3CCCCC3CC2C1", "C1CCCCCCCCCCC1", "CC(O)C1CC(N)C(C1)C(=O)O", "C1CC1C1CC1",
            "c1ccc2c(c1)ccc1ccccc12", "C1CCC(CC1)CCCCC1CCCCC1",
        ] {
            let mol = parse_smiles(smi).unwrap();
            let want = reference(&mol);
            let got = mordred_molecular_id(&mol);
            let total: f64 = want.iter().sum();
            assert!((got[0] - total).abs() < 1e-10 * total, "{smi}: {} vs {}", got[0], total);
        }
    }

    #[test]
    fn molecular_id_of_ethane_and_salt() {
        // ethane: one path of weight 1 from each atom -> aid = 1.5
        let v = mordred_molecular_id(&parse_smiles("CC").unwrap());
        assert!((v[0] - 3.0).abs() < 1e-12);
        assert!((v[1] - 1.5).abs() < 1e-12);
        assert!(mordred_molecular_id(&parse_smiles("[Na+].[Cl-]").unwrap())[0].is_nan());
    }
}
