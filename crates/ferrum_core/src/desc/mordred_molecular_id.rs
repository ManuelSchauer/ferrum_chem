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
//! search sums three kinds of subgraphs in closed form:
//!
//! - *Pendant trees* (atoms removed by repeatedly stripping degree-1 atoms):
//!   a path entering the tree below atom `x` can only go further down.
//! - *Chains* (atoms with exactly two neighbours in the remaining core)
//!   entered from a branch atom: the path runs along the chain, possibly ends
//!   in it or in a pendant tree of a chain atom, or leaves it at the other
//!   end.
//! - *Branches* of a cut vertex `u` of the core: the part of the core reached
//!   from `u` through one of its blocks (biconnected components). A path that
//!   reaches `u` through another block or a pendant tree (or starts at `u`)
//!   never visited an atom of the branch, so the branch's paths are the same
//!   for every such entry. Branch terms are collected once (on first use, by
//!   a search from `u` restricted to that block, merged by key).
//!
//! In all cases the terms a path collects inside the subgraph are
//! `1/√W · 1/√w(y)` for the subgraph's end points `y`, included when
//! `W·w_pre(y) < lim` (`W`: weight of the path before entering, `w(y)`: weight
//! of the part inside the subgraph, `w_pre(y)`: the same without its last
//! edge). The terms are sorted by `w_pre` (equal keys merged) with prefix
//! sums, so each entry is a binary search. The chain of the start atom (or
//! the chain its pendant tree hangs on) is walked atom by atom, since part of
//! it is already on the path.
//!
//! The cut-off comparisons use the exact integer weights; products are first
//! compared in `f64` (relative error far below the 1e-12 margin) and checked
//! exactly only when they are that close to the limit.

use rustc_hash::FxHashMap;

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
const LIMIT_F: f64 = LIMIT as f64;
const LIMIT_LO: f64 = LIMIT_F * (1.0 - 1e-12);
const LIMIT_HI: f64 = LIMIT_F * (1.0 + 1e-12);
const NONE: u32 = u32::MAX;

/// `k as f64` (the `u64` conversion is much cheaper and rounds the same).
#[inline(always)]
fn to_f64(k: u128) -> f64 {
    if k <= u64::MAX as u128 { k as u64 as f64 } else { k as f64 }
}

/// `k as f64` within a few ulp (cheaper than [`to_f64`] above `2^64`).
#[inline(always)]
fn approx_f64(k: u128) -> f64 {
    ((k >> 64) as u64 as f64).mul_add(18_446_744_073_709_551_616.0, k as u64 as f64)
}

/// `w·k < lim` for the exact weights `w`, `k` (`wf`, `kf`: their `f64`
/// values, relative error < 1e-13).
#[inline(always)]
fn below(w: u128, wf: f64, k: u128, kf: f64) -> bool {
    let p = wf * kf;
    if p < LIMIT_LO {
        true
    } else if p > LIMIT_HI {
        false
    } else {
        w.checked_mul(k).is_some_and(|x| x < LIMIT)
    }
}

/// A term list in [`Lists`].
#[derive(Clone, Copy, Default)]
struct Entry {
    start: u32,
    end: u32,
    /// Largest key (as `f64`) and sum of all values (0 for an empty list).
    max_key_f: f64,
    total: f64,
}

/// Term lists: keys (`w_pre`, sorted, distinct), values (`Σ 1/√w`) and prefix
/// sums of the values.
#[derive(Default)]
struct Lists {
    keys: Vec<u128>,
    keys_f: Vec<f64>,
    vals: Vec<f64>,
    sums: Vec<f64>,
}

impl Lists {
    /// Appends `terms` (w_pre, 1/√w) sorted by key, equal keys merged; keys
    /// at or above the limit can never count and are dropped.
    fn push(&mut self, terms: &mut [(u128, f64)]) -> Entry {
        terms.sort_unstable_by_key(|t| t.0);
        let start = self.keys.len();
        let mut acc = 0.0;
        for &(k, v) in terms.iter() {
            if k >= LIMIT {
                break;
            }
            acc += v;
            if self.keys.len() > start && self.keys[self.keys.len() - 1] == k {
                *self.vals.last_mut().expect("term") += v;
                *self.sums.last_mut().expect("term") = acc;
            } else {
                self.keys.push(k);
                self.keys_f.push(approx_f64(k));
                self.vals.push(v);
                self.sums.push(acc);
            }
        }
        let end = self.keys.len();
        let (max_key_f, total) = if end > start { (self.keys_f[end - 1], self.sums[end - 1]) } else { (0.0, 0.0) };
        Entry { start: start as u32, end: end as u32, max_key_f, total }
    }

    /// `Σ 1/√w` of the terms of `e` with `w·w_pre < lim`.
    #[inline(always)]
    fn query(&self, e: &Entry, w: u128, wf: f64) -> f64 {
        if wf * e.max_key_f < LIMIT_LO { e.total } else { self.query_cut(e.start as usize, e.end as usize, w, wf) }
    }

    /// [`Self::query`] when the cutoff falls inside the list.
    #[inline(never)]
    fn query_cut(&self, s: usize, t: usize, w: u128, wf: f64) -> f64 {
        let (mut lo, mut hi) = (s, t);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if below(w, wf, self.keys[mid], self.keys_f[mid]) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == s { 0.0 } else { self.sums[lo - 1] }
    }
}

const PLAIN: u8 = 0;
const PENDANT: u8 = 1;
const CHAIN: u8 = 2;

/// One neighbour of an atom.
#[derive(Clone, Copy)]
struct Slot {
    v: u32,
    aux: u32,
    /// Edge weight.
    ew: u32,
    /// `PLAIN`, `PENDANT` (`v` is a pendant atom hanging on this atom) or
    /// `CHAIN` (`aux`: the chain entered through `v`).
    kind: u8,
    /// `1/√` of the edge weight.
    einv: f64,
}

/// The neighbours of an atom in one block of the core (`block`, `NONE` for
/// the pendant neighbours and the neighbours of pendant atoms).
#[derive(Clone, Copy)]
struct Group {
    block: u32,
    /// Branch of a cut vertex through this block (`NONE`: not a cut vertex).
    branch: u32,
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
    /// Block of the edge `last -> end`.
    end_block: u32,
    /// Whether the weight up to the last chain atom stays below the limit.
    reachable: bool,
    /// Weight up to the last chain atom.
    w_last: u128,
    w_last_f: f64,
    /// Weight of the edge `last -> end` and `1/√` of the weight up to `end`.
    ew_end: u64,
    ew_end_f: f64,
    inv_sqrt_end: f64,
}

struct Graph {
    /// Groups of each atom: `gstart[a]..gstart[a + 1]`.
    gstart: Vec<u32>,
    groups: Vec<Group>,
    slots: Vec<Slot>,
    /// Pendant-tree terms per pendant atom.
    pendant: Vec<Entry>,
    chains: Vec<Chain>,
    /// Undirected chain id of each atom (core chain atoms and the pendant
    /// atoms hanging on them).
    chain_id: Vec<u32>,
    /// Root atom and group of each branch.
    branch_root: Vec<(u32, u32)>,
}

/// Biconnected components of the subgraph induced by the atoms with
/// `core[a]` (iterative Tarjan) on a CSR adjacency (`start`, neighbour of
/// slot `k`: `target(k)`, bond of slot `k`: `bond_of_slot[k]`): block id per
/// bond (`NONE` for bonds outside the subgraph) and the number of blocks.
fn core_blocks(
    start: &[usize],
    target: impl Fn(usize) -> usize,
    bond_of_slot: &[u32],
    core: &[bool],
    n_bonds: usize,
) -> (Vec<u32>, u32) {
    let n = core.len();
    let mut block_of_bond = vec![NONE; n_bonds];
    let mut disc = vec![0u32; n];
    let mut low = vec![0u32; n];
    let mut time = 0u32;
    let mut n_blocks = 0u32;
    let mut edges: Vec<u32> = Vec::new();
    // (atom, bond to the parent, next slot)
    let mut stack: Vec<(u32, u32, usize)> = Vec::new();
    for r in 0..n {
        if !core[r] || disc[r] != 0 {
            continue;
        }
        time += 1;
        disc[r] = time;
        low[r] = time;
        stack.push((r as u32, NONE, start[r]));
        while let Some(top) = stack.last_mut() {
            let (v, pb, k) = (top.0 as usize, top.1, top.2);
            if k < start[v + 1] {
                top.2 += 1;
                let w = target(k);
                let b = bond_of_slot[k];
                if !core[w] || b == pb {
                    continue;
                }
                if disc[w] == 0 {
                    edges.push(b);
                    time += 1;
                    disc[w] = time;
                    low[w] = time;
                    stack.push((w as u32, b, start[w]));
                } else if disc[w] < disc[v] {
                    edges.push(b);
                    low[v] = low[v].min(disc[w]);
                }
            } else {
                stack.pop();
                if let Some(&(p, _, _)) = stack.last() {
                    let p = p as usize;
                    low[p] = low[p].min(low[v]);
                    if low[v] >= disc[p] {
                        while let Some(e) = edges.pop() {
                            block_of_bond[e as usize] = n_blocks;
                            if e == pb {
                                break;
                            }
                        }
                        n_blocks += 1;
                    }
                }
            }
        }
    }
    (block_of_bond, n_blocks)
}

impl Graph {
    fn new(mol: &Molecule, lists: &mut Lists) -> Self {
        let n = mol.num_atoms();
        let mut start = Vec::with_capacity(n + 1);
        let mut adj: Vec<(u32, u128, f64)> = Vec::with_capacity(2 * mol.num_bonds());
        let mut bond_of_slot: Vec<u32> = Vec::with_capacity(2 * mol.num_bonds());
        for a in 0..n {
            start.push(adj.len());
            for &bi in &mol.adjacency[a] {
                let b = &mol.bonds[bi as usize];
                let o = if b.source as usize == a { b.target } else { b.source } as usize;
                let w = (mol.adjacency[a].len() * mol.adjacency[o].len()) as u128;
                adj.push((o as u32, w, 1.0 / to_f64(w).sqrt()));
                bond_of_slot.push(bi as u32);
            }
        }
        start.push(adj.len());
        let slots_of = |a: usize| start[a]..start[a + 1];

        // pendant trees: strip degree-1 atoms (in strip order)
        let mut parent = vec![NONE; n];
        let mut stripped: Vec<u32> = Vec::new();
        {
            let mut deg: Vec<u32> = (0..n).map(|a| slots_of(a).len() as u32).collect();
            let mut stack: Vec<u32> = (0..n as u32).filter(|&a| deg[a as usize] == 1).collect();
            let mut removed = vec![false; n];
            while let Some(x) = stack.pop() {
                let x = x as usize;
                if removed[x] || deg[x] != 1 {
                    continue;
                }
                let Some(p) = adj[slots_of(x)].iter().map(|e| e.0 as usize).find(|&v| !removed[v]) else {
                    continue;
                };
                removed[x] = true;
                parent[x] = p as u32;
                stripped.push(x as u32);
                deg[p] -= 1;
                if deg[p] == 1 {
                    stack.push(p as u32);
                }
            }
        }
        let core: Vec<bool> = parent.iter().map(|&p| p == NONE).collect();
        // raw pendant terms (w_pre, 1/√w) per pendant atom
        let mut raw_start = vec![0usize; n + 1];
        let mut raw: Vec<(u128, f64)> = Vec::new();
        let mut dfs: Vec<(u32, u32, u128, u128)> = Vec::new();
        for x in 0..n {
            raw_start[x] = raw.len();
            let p = parent[x];
            if p == NONE {
                continue;
            }
            let e0 = adj[slots_of(x)].iter().find(|e| e.0 == p).expect("parent bond").1;
            dfs.clear();
            dfs.push((x as u32, p, 1, e0));
            while let Some((y, from, w_pre, w)) = dfs.pop() {
                raw.push((w_pre, 1.0 / to_f64(w).sqrt()));
                if w >= LIMIT {
                    continue;
                }
                for &(c, ew, _) in &adj[slots_of(y as usize)] {
                    if c != from {
                        dfs.push((c, y, w, w * ew));
                    }
                }
            }
        }
        raw_start[n] = raw.len();
        let mut pendant = vec![Entry::default(); n];
        let mut terms: Vec<(u128, f64)> = Vec::new();
        for x in 0..n {
            if parent[x] != NONE {
                terms.clear();
                terms.extend_from_slice(&raw[raw_start[x]..raw_start[x + 1]]);
                pendant[x] = lists.push(&mut terms);
            }
        }

        let (block_of_bond, _) = core_blocks(&start, |k| adj[k].0 as usize, &bond_of_slot, &core, mol.num_bonds());

        // chains of the core
        let core_degree: Vec<u32> = (0..n)
            .map(|a| if core[a] { adj[slots_of(a)].iter().filter(|e| core[e.0 as usize]).count() as u32 } else { 0 })
            .collect();
        let chain_atom: Vec<bool> = (0..n).map(|a| core[a] && core_degree[a] == 2).collect();
        let mut chain_of_slot = vec![NONE; adj.len()];
        let mut chains: Vec<Chain> = Vec::new();
        let mut chain_id = vec![NONE; n];
        let mut n_ids = 0u32;
        let mut atoms: Vec<u32> = Vec::new();
        for u in 0..n {
            if !core[u] || core_degree[u] == 2 {
                continue;
            }
            for k in slots_of(u) {
                let v = adj[k].0 as usize;
                if !chain_atom[v] {
                    continue;
                }
                // walk the chain u -> v -> ... -> end
                atoms.clear();
                terms.clear();
                let (mut prev, mut cur) = (u, v);
                let (mut w, mut inv) = (1u128, 1.0f64);
                let mut slot = k;
                let mut reachable = true;
                let (end, end_slot) = loop {
                    atoms.push(cur as u32);
                    if reachable {
                        terms.push((w, 0.0));
                        w = w.saturating_mul(adj[slot].1);
                        inv *= adj[slot].2;
                        terms.last_mut().expect("term").1 = inv;
                        // pendant trees hanging on the chain atom
                        for s in slots_of(cur) {
                            let x = adj[s].0 as usize;
                            if parent[x] == cur as u32 {
                                for &(k2, v2) in &raw[raw_start[x]..raw_start[x + 1]] {
                                    terms.push((w.saturating_mul(k2), v2 * inv));
                                }
                            }
                        }
                        reachable = w < LIMIT;
                    }
                    let next_slot = slots_of(cur)
                        .find(|&s| adj[s].0 as usize != prev && core[adj[s].0 as usize])
                        .expect("chain continues");
                    let next = adj[next_slot].0 as usize;
                    if !chain_atom[next] {
                        break (next, next_slot);
                    }
                    prev = cur;
                    cur = next;
                    slot = next_slot;
                };
                let id = match chain_id[v] {
                    NONE => {
                        n_ids += 1;
                        for &a in &atoms {
                            chain_id[a as usize] = n_ids - 1;
                        }
                        n_ids - 1
                    }
                    id => id,
                };
                let entry = lists.push(&mut terms);
                chain_of_slot[k] = chains.len() as u32;
                let ew_end = adj[end_slot].1;
                chains.push(Chain {
                    terms: entry,
                    id,
                    first: atoms[0],
                    last: *atoms.last().expect("chain atom"),
                    end: end as u32,
                    end_block: block_of_bond[bond_of_slot[end_slot] as usize],
                    reachable,
                    w_last: w,
                    w_last_f: approx_f64(w),
                    ew_end: ew_end as u64,
                    ew_end_f: to_f64(ew_end),
                    inv_sqrt_end: inv * adj[end_slot].2,
                });
            }
        }
        // pendant atoms belong to the chain of the core atom they hang on
        // (parents are stripped after their children: walk in reverse order)
        for &x in stripped.iter().rev() {
            let p = parent[x as usize] as usize;
            chain_id[x as usize] = chain_id[p];
        }

        // neighbour groups per atom
        let mut gstart = Vec::with_capacity(n + 1);
        let mut groups: Vec<Group> = Vec::with_capacity(n + 8);
        let mut slots: Vec<Slot> = Vec::with_capacity(adj.len());
        let mut branch_root: Vec<(u32, u32)> = Vec::new();
        let mut order: Vec<(u32, usize)> = Vec::new();
        for a in 0..n {
            gstart.push(groups.len() as u32);
            order.clear();
            for k in slots_of(a) {
                let v = adj[k].0 as usize;
                let block = if core[a] && core[v] { block_of_bond[bond_of_slot[k] as usize] } else { NONE };
                order.push((block, k));
            }
            order.sort_unstable();
            // a branch per block of the core cut vertices
            let n_blocks = {
                let mut c = 0;
                let mut last = NONE;
                for &(b, _) in &order {
                    if b != NONE && b != last {
                        c += 1;
                        last = b;
                    }
                }
                c
            };
            let mut i = 0;
            while i < order.len() {
                let block = order[i].0;
                let first = slots.len() as u32;
                while i < order.len() && order[i].0 == block {
                    let k = order[i].1;
                    let (v, ew, einv) = adj[k];
                    let (kind, aux) = if parent[v as usize] == a as u32 {
                        (PENDANT, v)
                    } else if chain_of_slot[k] != NONE {
                        (CHAIN, chain_of_slot[k])
                    } else {
                        (PLAIN, NONE)
                    };
                    slots.push(Slot { v, aux, ew: ew as u32, kind, einv });
                    i += 1;
                }
                let branch = if block != NONE && n_blocks > 1 {
                    branch_root.push((a as u32, groups.len() as u32));
                    branch_root.len() as u32 - 1
                } else {
                    NONE
                };
                groups.push(Group { block, branch, start: first, end: slots.len() as u32 });
            }
        }
        gstart.push(groups.len() as u32);
        Graph { gstart, groups, slots, pendant, chains, chain_id, branch_root }
    }
}

/// Terms of a branch list being collected, merged by key.
#[derive(Default)]
struct Collect {
    terms: FxHashMap<u128, f64>,
    /// Lists entered on the way, merged by (list, weight): `Σ 1/√w`.
    lists: FxHashMap<(u64, u128), f64>,
}

impl Collect {
    #[inline(always)]
    fn term(&mut self, key: u128, val: f64) {
        *self.terms.entry(key).or_insert(0.0) += val;
    }

    /// Adds the terms of list `e` of `lists` entered with weight `w` (`1/√w`
    /// = `inv`).
    fn add_list(&mut self, lists: &Lists, e: Entry, w: u128, inv: f64) {
        let wf = approx_f64(w);
        for i in e.start as usize..e.end as usize {
            if !below(w, wf, lists.keys[i], lists.keys_f[i]) {
                break;
            }
            self.term(w * lists.keys[i], inv * lists.vals[i]);
        }
    }
}

const BRANCH_UNKNOWN: u8 = 0;
const BRANCH_DONE: u8 = 1;
const BRANCH_BUSY: u8 = 2;

/// Depth-first path search.
struct Search<'a> {
    g: &'a Graph,
    lists: Lists,
    visited: Vec<bool>,
    /// Chain containing the start atom (walked atom by atom).
    dirty: u32,
    branch_state: Vec<u8>,
    branch_entry: Vec<Entry>,
    /// Spare maps for [`Collect`] and a buffer for the collected terms.
    pool: Vec<Collect>,
    buf: Vec<(u128, f64)>,
}

impl Search<'_> {
    /// Terms of branch `b`, collected on first use (`None` while they are
    /// being collected; cannot happen, a branch holds no branch containing
    /// itself, but walking the branch would be correct as well).
    #[inline(always)]
    fn branch(&mut self, b: u32) -> Option<Entry> {
        if self.branch_state[b as usize] == BRANCH_DONE {
            Some(self.branch_entry[b as usize])
        } else {
            self.collect_branch(b)
        }
    }

    /// [`Self::branch`] when it is not collected yet.
    #[inline(never)]
    fn collect_branch(&mut self, b: u32) -> Option<Entry> {
        if self.branch_state[b as usize] == BRANCH_BUSY {
            return None;
        }
        self.branch_state[b as usize] = BRANCH_BUSY;
        let (u, gi) = self.g.branch_root[b as usize];
        let u = u as usize;
        // the branch holds no atom of the current path: search it on its own
        // (the chain of `u`, if any, is walked)
        let (was_visited, dirty) = (self.visited[u], self.dirty);
        self.visited[u] = true;
        self.dirty = self.g.chain_id[u];
        let mut acc = self.pool.pop().unwrap_or_default();
        let leaf = self.collect_group(self.g.groups[gi as usize], 1, 1.0, 1.0, &mut acc);
        if leaf != 0.0 {
            acc.term(1, leaf);
        }
        self.visited[u] = was_visited;
        self.dirty = dirty;
        // expand the lists entered on the way
        let mut lists = std::mem::take(&mut acc.lists);
        for ((e, w), inv) in lists.drain() {
            let e = Entry { start: e as u32, end: (e >> 32) as u32, ..Default::default() };
            acc.add_list(&self.lists, e, w, inv);
        }
        acc.lists = lists;
        self.buf.clear();
        self.buf.extend(acc.terms.drain());
        self.pool.push(acc);
        let e = self.lists.push(&mut self.buf);
        self.branch_state[b as usize] = BRANCH_DONE;
        self.branch_entry[b as usize] = e;
        Some(e)
    }

    /// `Σ 1/√w'` over the extensions of the path ending at `u` (weight `w`,
    /// as `f64` `wf`), reached through block `in_block`; `w'` is the weight
    /// of the extension alone. Contributions are products of the per-edge
    /// `1/√w_e` (a few ulp from Mordred's `1/√w`); the cutoff uses the exact
    /// integer weight.
    fn sum(&mut self, u: usize, w: u128, wf: f64, in_block: u32) -> f64 {
        let g = self.g;
        let mut total = 0.0;
        for gi in g.gstart[u] as usize..g.gstart[u + 1] as usize {
            total += self.group_sum(g.groups[gi], w, wf, in_block);
        }
        total
    }

    /// [`Self::sum`] through the neighbours of one group.
    #[inline(always)]
    fn group_sum(&mut self, grp: Group, w: u128, wf: f64, in_block: u32) -> f64 {
        let g = self.g;
        if grp.branch != NONE
            && grp.block != in_block
            && let Some(e) = self.branch(grp.branch)
        {
            return self.lists.query(&e, w, wf);
        }
        let mut total = 0.0;
        for si in grp.start as usize..grp.end as usize {
            let s = g.slots[si];
            let v = s.v as usize;
            if self.visited[v] {
                continue;
            }
            if s.kind == PENDANT {
                total += self.lists.query(&g.pendant[v], w, wf);
                continue;
            }
            if s.kind == CHAIN {
                let c = &g.chains[s.aux as usize];
                if c.id != self.dirty {
                    total += self.lists.query(&c.terms, w, wf);
                    let end = c.end as usize;
                    if c.reachable && !self.visited[end] && below(w, wf, c.w_last, c.w_last_f) {
                        let w_end = w * c.w_last * c.ew_end as u128;
                        let mut sub = 0.0;
                        if w_end < LIMIT {
                            let (first, last) = (c.first as usize, c.last as usize);
                            self.visited[first] = true;
                            self.visited[last] = true;
                            self.visited[end] = true;
                            sub = self.sum(end, w_end, wf * c.w_last_f * c.ew_end_f, c.end_block);
                            self.visited[end] = false;
                            self.visited[last] = false;
                            self.visited[first] = false;
                        }
                        total += c.inv_sqrt_end * (1.0 + sub);
                    }
                    continue;
                }
            }
            let w2 = w * s.ew as u128;
            let mut sub = 0.0;
            if w2 < LIMIT {
                self.visited[v] = true;
                sub = self.sum(v, w2, wf * s.ew as f64, grp.block);
                self.visited[v] = false;
            }
            total += s.einv * (1.0 + sub);
        }
        total
    }

    /// Adds the terms of list `e` entered with weight `w` (`f64`: `wf`,
    /// `1/√w` = `inv`) to `acc`; longer lists once per distinct weight, at the
    /// end.
    #[inline]
    fn collect_list(&self, acc: &mut Collect, e: Entry, w: u128, wf: f64, inv: f64) {
        if e.end - e.start > 2 {
            *acc.lists.entry((e.start as u64 | (e.end as u64) << 32, w)).or_insert(0.0) += inv;
            return;
        }
        let l = &self.lists;
        for i in e.start as usize..e.end as usize {
            if !below(w, wf, l.keys[i], l.keys_f[i]) {
                break;
            }
            acc.term(w * l.keys[i], inv * l.vals[i]);
        }
    }

    /// Collects the terms of the extensions of the path ending at `u`
    /// (weight `w`, as `f64` `wf`, `1/√w` = `inv`), reached through block
    /// `in_block`, keyed by `w_pre`.
    fn collect(&mut self, u: usize, w: u128, wf: f64, inv: f64, in_block: u32, acc: &mut Collect) {
        let g = self.g;
        let mut leaf = 0.0;
        for gi in g.gstart[u] as usize..g.gstart[u + 1] as usize {
            let grp = g.groups[gi];
            if grp.branch != NONE
                && grp.block != in_block
                && let Some(e) = self.branch(grp.branch)
            {
                self.collect_list(acc, e, w, wf, inv);
                continue;
            }
            leaf += self.collect_group(grp, w, wf, inv, acc);
        }
        // paths ending at a plain neighbour
        if leaf != 0.0 {
            acc.term(w, inv * leaf);
        }
    }

    /// [`Self::collect`] through the neighbours of one group; returns the
    /// `Σ 1/√w_e` of the plain neighbours (paths ending there).
    fn collect_group(&mut self, grp: Group, w: u128, wf: f64, inv: f64, acc: &mut Collect) -> f64 {
        let g = self.g;
        let mut leaf = 0.0;
        for si in grp.start as usize..grp.end as usize {
            let s = g.slots[si];
            let v = s.v as usize;
            if self.visited[v] {
                continue;
            }
            if s.kind == PENDANT {
                self.collect_list(acc, g.pendant[v], w, wf, inv);
                continue;
            }
            if s.kind == CHAIN {
                let c = &g.chains[s.aux as usize];
                if c.id != self.dirty {
                    self.collect_list(acc, c.terms, w, wf, inv);
                    let end = c.end as usize;
                    if c.reachable && !self.visited[end] && below(w, wf, c.w_last, c.w_last_f) {
                        let key = w * c.w_last;
                        let inv_end = inv * c.inv_sqrt_end;
                        acc.term(key, inv_end);
                        let w_end = key * c.ew_end as u128;
                        if w_end < LIMIT {
                            let (first, last) = (c.first as usize, c.last as usize);
                            self.visited[first] = true;
                            self.visited[last] = true;
                            self.visited[end] = true;
                            self.collect(end, w_end, wf * c.w_last_f * c.ew_end_f, inv_end, c.end_block, acc);
                            self.visited[end] = false;
                            self.visited[last] = false;
                            self.visited[first] = false;
                        }
                    }
                    continue;
                }
            }
            leaf += s.einv;
            let w2 = w * s.ew as u128;
            if w2 < LIMIT {
                self.visited[v] = true;
                self.collect(v, w2, wf * s.ew as f64, inv * s.einv, grp.block, acc);
                self.visited[v] = false;
            }
        }
        leaf
    }
}

/// Mordred MolecularId values in [`mordred_molecular_id_names`] order.
pub fn mordred_molecular_id(mol: &Molecule) -> Vec<f64> {
    let n = mol.num_atoms();
    if n == 0 || crate::algo::components::num_fragments(mol) != 1 {
        return vec![f64::NAN; MORDRED_MOLECULAR_ID_LEN];
    }
    let mut lists = Lists::default();
    let g = Graph::new(mol, &mut lists);
    let n_branches = g.branch_root.len();
    let mut s = Search {
        g: &g,
        lists,
        visited: vec![false; n],
        dirty: NONE,
        branch_state: vec![BRANCH_UNKNOWN; n_branches],
        branch_entry: vec![Entry::default(); n_branches],
        pool: Vec::new(),
        buf: Vec::new(),
    };
    // [all, hetero, C, N, O, halogen]
    let mut sums = [0.0f64; 6];
    for a in 0..n {
        s.dirty = g.chain_id[a];
        s.visited[a] = true;
        let id = s.sum(a, 1, 1.0, NONE);
        s.visited[a] = false;
        let aid = 1.0 + id / 2.0;
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
        fn go(
            u: usize,
            w: u128,
            vis: &mut Vec<bool>,
            id: &mut f64,
            nb: &dyn Fn(usize) -> Vec<usize>,
            deg: &dyn Fn(usize) -> u128,
        ) {
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
            "CC",
            "CCO",
            "CC(C)(C)C",
            "C1CC1",
            "c1ccccc1",
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "C1CCC2(CC1)CCCC2",
            "OC1CCC(CC1)C1CCCCC1",
            "C1CC2CCC1CC2",
            "C12C3C4C1C5C2C3C45",
            "CC1CCC2C(C)C3CCCCC3CC2C1",
            "C1CCCCCCCCCCC1",
            "CC(O)C1CC(N)C(C1)C(=O)O",
            "C1CC1C1CC1",
            "c1ccc2c(c1)ccc1ccccc12",
            "C1CCC(CC1)CCCCC1CCCCC1",
            // several ring systems, spiro and linked blocks (branch lists)
            "C1CC1C1CC1C1CC1",
            "c1ccccc1CCc1ccc(cc1)C1CCC2(CC1)CCCC2",
            "C1CC2(C1)CC1(C2)CC2(C1)CCC2",
            "O=C(Nc1ccccc1)c1ccc(cc1)N1CCN(CC1)c1ncccn1",
            "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
            "C1CCC2(CC1)OCC1(O2)CCCCC1",
            "c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccccc1",
            "COC1=CC=CC(C2C3=C(CC4C(=O)N(C5=CC=CC=C5C(=O)NC(C)CCC5=CC=CC=C5)C(=O)N24)C2=CC=CC=C2N3)=C1",
            "[2H]C([2H])([2H])c1ccccc1",
            "C1CC23CCC4(CC2)CCC3(CC1)CC4",
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
