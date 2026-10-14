//! Weighted path descriptors (CDK `WeightedPathDescriptor`, WTPT-1..5).
//!
//! Every simple path p of the (hydrogen-suppressed) molecular graph gets the
//! weight `prod_{bonds (u,v) in p} 1 / sqrt(deg(u) * deg(v))`; single atoms
//! (paths of length 0) weigh 1.
//!
//! - WTPT-1: sum of all path weights (each undirected path once)
//! - WTPT-2: WTPT-1 / number of atoms
//! - WTPT-3: sum of weights of paths starting at a heteroatom (non-carbon)
//! - WTPT-4: as WTPT-3 for paths starting at oxygen
//! - WTPT-5: as WTPT-3 for paths starting at nitrogen
//!
//! As in CDK, WTPT-3..5 count directed paths (a path between two heteroatoms
//! counts from both ends). CDK enumerates every simple path from every atom.
//! The number of simple paths grows exponentially with the number of fused
//! rings, so the search here is restricted to the 2-core (ring systems and the
//! chains linking them); pendant trees are folded into per-atom sums (see
//! `PathSums`). The result is exact; WTPT-1 uses that path weights are
//! symmetric (half the directed sum).

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Molecular graph in CSR form with the bond factors `1 / sqrt(deg(u) deg(v))`.
struct Graph {
    start: Vec<u32>,
    nbr: Vec<u32>,
    factor: Vec<f64>,
}

impl Graph {
    fn new(mol: &Molecule) -> Self {
        let n = mol.num_atoms();
        let degree: Vec<f64> = (0..n).map(|a| mol.adjacency[a].len() as f64).collect();
        let mut start = Vec::with_capacity(n + 1);
        let mut nbr = Vec::with_capacity(2 * mol.num_bonds());
        let mut factor = Vec::with_capacity(2 * mol.num_bonds());
        for a in 0..n {
            start.push(nbr.len() as u32);
            for &bi in &mol.adjacency[a] {
                let b = &mol.bonds[bi as usize];
                let other = if b.source as usize == a { b.target } else { b.source };
                nbr.push(other as u32);
                factor.push(1.0 / (degree[a] * degree[other as usize]).sqrt());
            }
        }
        start.push(nbr.len() as u32);
        Self { start, nbr, factor }
    }

    #[inline]
    fn edges(&self, a: usize) -> std::ops::Range<usize> {
        self.start[a] as usize..self.start[a + 1] as usize
    }
}

/// Path sums split at the 2-core.
///
/// Only the 2-core (ring systems and the chains linking them) has exponentially
/// many simple paths. Every other atom lies in a pendant tree attached to one
/// core atom, so from a core atom the paths into its pendant trees are fixed
/// sums (`hang`), and a path from a tree atom that reaches the core continues
/// like any path leaving that core atom, minus the ones that would re-enter
/// its own tree.
struct PathSums<'a> {
    g: &'a Graph,
    in_core: Vec<bool>,
    /// For edge `e = (a -> x)` into a pendant tree: sum of the weights of all
    /// paths starting at `a` whose first step is `e`.
    branch: Vec<f64>,
    /// `1 + sum of branch` over the pendant trees of a core atom.
    hang1: Vec<f64>,
    visited: Vec<bool>,
    /// Core-only adjacency (CSR): neighbour, bond factor, neighbour's `hang1`.
    core_start: Vec<u32>,
    core_edges: Vec<(u32, f64, f64)>,
    /// `(atom, reachable unvisited core atoms)` -> sum of the paths continuing
    /// from that atom (see `core_future`).
    memo: rustc_hash::FxHashMap<(u32, Box<[u64]>), f64>,
}

/// Memo entries kept before the table is cleared (bounds memory; clearing
/// only costs speed).
const MEMO_LIMIT: usize = 1 << 16;

/// Cores with up to this many independent rings use the chain-contracted
/// search (`ChainCore`) instead of the memoised one
/// (cheaper than memoising).
const MEMO_MIN_RINGS: usize = 8;

impl<'a> PathSums<'a> {
    fn new(g: &'a Graph) -> Self {
        let n = g.start.len() - 1;
        // peel atoms of degree <= 1 until only the 2-core is left
        let mut deg: Vec<u32> = (0..n).map(|a| g.edges(a).len() as u32).collect();
        let mut in_core = vec![true; n];
        let mut stack: Vec<usize> = (0..n).filter(|&a| deg[a] <= 1).collect();
        while let Some(a) = stack.pop() {
            if !in_core[a] {
                continue;
            }
            in_core[a] = false;
            for e in g.edges(a) {
                let b = g.nbr[e] as usize;
                if in_core[b] {
                    deg[b] -= 1;
                    if deg[b] == 1 {
                        stack.push(b);
                    }
                }
            }
        }
        let mut s = Self {
            g,
            in_core,
            branch: vec![0.0; g.nbr.len()],
            hang1: vec![1.0; n],
            visited: vec![false; n],
            core_start: Vec::new(),
            core_edges: Vec::new(),
            memo: rustc_hash::FxHashMap::default(),
        };
        for a in 0..n {
            if s.in_core[a] {
                for e in g.edges(a) {
                    let x = g.nbr[e] as usize;
                    if !s.in_core[x] {
                        let t = s.tree_sum(x, a, g.factor[e]);
                        s.branch[e] = t;
                        s.hang1[a] += t;
                    }
                }
            }
        }
        s.core_start.reserve(n + 1);
        for a in 0..n {
            s.core_start.push(s.core_edges.len() as u32);
            if s.in_core[a] {
                for e in g.edges(a) {
                    let b = g.nbr[e] as usize;
                    if s.in_core[b] {
                        s.core_edges.push((b as u32, g.factor[e], s.hang1[b]));
                    }
                }
            }
        }
        s.core_start.push(s.core_edges.len() as u32);
        s
    }

    /// Sum over the paths entering the pendant tree at `x` from `parent`
    /// (weight `w` on arrival at `x`).
    fn tree_sum(&self, x: usize, parent: usize, w: f64) -> f64 {
        let mut sum = w;
        for e in self.g.edges(x) {
            let y = self.g.nbr[e] as usize;
            if y != parent {
                sum += self.tree_sum(y, x, w * self.g.factor[e]);
            }
        }
        sum
    }

    /// Core atoms reachable from `v` inside `mask` (bitset).
    fn component(&self, mask: &[u64], v: usize) -> Vec<u64> {
        let mut comp = vec![0u64; mask.len()];
        comp[v / 64] |= 1 << (v % 64);
        let mut stack = vec![v];
        while let Some(a) = stack.pop() {
            for &(b, _, _) in &self.core_edges[self.core_start[a] as usize..self.core_start[a + 1] as usize] {
                let (word, bit) = (b as usize / 64, 1u64 << (b % 64));
                if mask[word] & bit != 0 && comp[word] & bit == 0 {
                    comp[word] |= bit;
                    stack.push(b as usize);
                }
            }
        }
        comp
    }

    /// Sum (relative to weight 1 at `u`) of the core paths continuing from
    /// `u`, where `comp` holds `u` and the unvisited core atoms connected to
    /// it. A continuation can only use atoms of `comp`, so the sum depends on
    /// `(u, comp)` alone and not on the path that led to `u`; memoising it
    /// turns the exponential search into a search over reachable regions,
    /// which repeat constantly in fused ring systems.
    fn core_future(&mut self, u: usize, comp: &[u64]) -> f64 {
        let key = (u as u32, Box::<[u64]>::from(comp));
        if let Some(&v) = self.memo.get(&key) {
            return v;
        }
        let mut rest = comp.to_vec();
        rest[u / 64] &= !(1 << (u % 64));
        let mut sum = 0.0;
        for i in self.core_start[u] as usize..self.core_start[u + 1] as usize {
            let (v, factor, hang1) = self.core_edges[i];
            let v = v as usize;
            if rest[v / 64] & (1 << (v % 64)) == 0 {
                continue;
            }
            let next = self.component(&rest, v);
            sum += factor * (hang1 + self.core_future(v, &next));
        }
        if self.memo.len() >= MEMO_LIMIT {
            self.memo.clear();
        }
        self.memo.insert(key, sum);
        sum
    }

    /// Paths from a non-core atom `x` (reached with weight `w`): through the
    /// tree, and through its core attachment using `from_core` (sum of all
    /// paths leaving each core atom).
    fn tree_dfs(&mut self, x: usize, w: f64, from_core: &[f64]) -> f64 {
        let mut sum = 0.0;
        for e in self.g.edges(x) {
            let y = self.g.nbr[e] as usize;
            if self.visited[y] {
                continue;
            }
            let wy = w * self.g.factor[e];
            if self.in_core[y] {
                // all paths leaving y except those re-entering this tree
                let back = self.g.edges(y).find(|&f| self.g.nbr[f] as usize == x).expect("bond is symmetric");
                sum += wy * (1.0 + from_core[y] - self.branch[back]);
            } else {
                self.visited[y] = true;
                sum += wy + self.tree_dfs(y, wy, from_core);
                self.visited[y] = false;
            }
        }
        sum
    }

    /// Sum of the weights of all simple paths starting at each atom, the
    /// single-atom path (weight 1) included.
    fn per_atom(mut self) -> Vec<f64> {
        let n = self.in_core.len();
        // all paths leaving each core atom (length >= 1)
        let mut from_core = vec![0.0; n];
        let all_core: Vec<u64> = {
            let mut m = vec![0u64; n.div_ceil(64)];
            for a in (0..n).filter(|&a| self.in_core[a]) {
                m[a / 64] |= 1 << (a % 64);
            }
            m
        };
        // the number of core paths grows with the number of independent rings
        // (cyclomatic number), not with the core size
        let core_atoms = self.in_core.iter().filter(|&&c| c).count();
        let rings = (self.core_edges.len() / 2 + 1).saturating_sub(core_atoms);
        let memoise = rings > MEMO_MIN_RINGS;
        if !memoise {
            let paths = ChainCore::new(&self).paths(&self);
            for a in (0..n).filter(|&a| self.in_core[a]) {
                from_core[a] = (self.hang1[a] - 1.0) + paths[a];
            }
        } else if core_atoms <= MAX_BITSET_CORE {
            let core: Vec<usize> = (0..n).filter(|&a| self.in_core[a]).collect();
            let paths = match core.len() {
                0..=64 => CoreMemo::<u64>::run(&self, &core),
                65..=128 => CoreMemo::<u128>::run(&self, &core),
                129..=256 => CoreMemo::<Words<4>>::run(&self, &core),
                _ => CoreMemo::<Words<8>>::run(&self, &core),
            };
            for (&a, p) in core.iter().zip(paths) {
                from_core[a] = (self.hang1[a] - 1.0) + p;
            }
        } else {
            for a in 0..n {
                if !self.in_core[a] {
                    continue;
                }
                let comp = self.component(&all_core, a);
                from_core[a] = (self.hang1[a] - 1.0) + self.core_future(a, &comp);
            }
        }
        (0..n)
            .map(|a| {
                if self.in_core[a] {
                    1.0 + from_core[a]
                } else {
                    self.visited[a] = true;
                    let s = 1.0 + self.tree_dfs(a, 1.0, &from_core);
                    self.visited[a] = false;
                    s
                }
            })
            .collect()
    }
}

/// Core path sums on the core with its degree-2 chains contracted.
///
/// Inside the 2-core, atoms of core degree 2 form chains between branch atoms
/// (core degree >= 3). A path entering a chain either stops inside it (a fixed
/// sum per chain and direction) or runs through it, so the search only has to
/// branch at branch atoms. A chain atom `s` starts by walking to one chain end
/// `b`; the continuation from `b` is the same for every atom of the chain
/// except for the part of the chain beyond `s`, which can only be entered from
/// the other end `c`: it is `R_b + Q_bc * partial`, with `R_b` all paths from
/// `b` avoiding the chain and `Q_bc` the weight of reaching `c` that way.
struct ChainCore {
    chains: Vec<Chain>,
    /// Per atom: (chain, the atom is the chain's `b` end).
    inc: Vec<smallvec::SmallVec<[(u32, bool); 4]>>,
    is_branch: Vec<bool>,
}

struct Chain {
    b: usize,
    c: usize,
    /// Internal atoms `x1..xk` from `b` to `c`.
    atoms: Vec<usize>,
    /// Bond factors `b-x1, x1-x2, ..., xk-c` (`k + 1` entries).
    f: Vec<f64>,
    /// Paths stopping inside the chain, entered from `b` / from `c` (weight 1 at the end atom).
    stop_from_b: f64,
    stop_from_c: f64,
    /// Product of all bond factors.
    through: f64,
}

impl ChainCore {
    /// A core component without branch atoms (a plain cycle) gets its lowest
    /// atom as pseudo branch atom, making the cycle a loop chain.
    fn new(ps: &PathSums) -> Self {
        let n = ps.in_core.len();
        let cdeg = |a: usize| (ps.core_start[a + 1] - ps.core_start[a]) as usize;
        let mut is_branch: Vec<bool> = (0..n).map(|a| ps.in_core[a] && cdeg(a) >= 3).collect();
        let mut cc = ChainCore { chains: Vec::new(), inc: vec![smallvec::SmallVec::new(); n], is_branch: Vec::new() };
        let mut covered = vec![false; n];
        // directed core edge (position in `core_edges`) already walked
        let mut walked = vec![false; ps.core_edges.len()];
        let branches: Vec<usize> = (0..n).filter(|&a| is_branch[a]).collect();
        for b in branches {
            cc.walk_chains(ps, b, &is_branch, &mut covered, &mut walked);
        }
        for a in 0..n {
            if ps.in_core[a] && !covered[a] {
                is_branch[a] = true;
                cc.walk_chains(ps, a, &is_branch, &mut covered, &mut walked);
            }
        }
        cc.is_branch = is_branch;
        cc
    }

    /// Adds the chains leaving branch atom `b` that were not walked yet.
    fn walk_chains(&mut self, ps: &PathSums, b: usize, is_branch: &[bool], covered: &mut [bool], walked: &mut [bool]) {
        let back_edge = |walked: &[bool], from: usize, to: usize| -> usize {
            let r = ps.core_start[to] as usize..ps.core_start[to + 1] as usize;
            r.clone().find(|&e| ps.core_edges[e].0 as usize == from && !walked[e]).expect("core bond is symmetric")
        };
        covered[b] = true;
        for e0 in ps.core_start[b] as usize..ps.core_start[b + 1] as usize {
            if walked[e0] {
                continue;
            }
            walked[e0] = true;
            let (mut prev, mut e) = (b, e0);
            let mut atoms = Vec::new();
            let mut f = Vec::new();
            loop {
                let (x, factor, _) = ps.core_edges[e];
                let x = x as usize;
                f.push(factor);
                if is_branch[x] {
                    let back = back_edge(walked, prev, x);
                    walked[back] = true;
                    let mut chain = Chain { b, c: x, atoms, f, stop_from_b: 0.0, stop_from_c: 0.0, through: 1.0 };
                    chain.finish(ps);
                    let id = self.chains.len() as u32;
                    self.inc[b].push((id, true));
                    self.inc[x].push((id, false));
                    self.chains.push(chain);
                    break;
                }
                covered[x] = true;
                atoms.push(x);
                // continue through the degree-2 atom x
                let r = ps.core_start[x] as usize..ps.core_start[x + 1] as usize;
                let next = r.clone().find(|&e2| ps.core_edges[e2].0 as usize != prev).expect("core degree 2");
                prev = x;
                e = next;
            }
        }
    }

    /// Search from branch atom `u` (weight `w`, already visited), never using
    /// chain `excluded` or `arrival`; adds the weight of each arrival at
    /// `target` to `q`.
    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        ps: &PathSums,
        visited: &mut [bool],
        u: usize,
        w: f64,
        arrival: u32,
        excluded: u32,
        target: usize,
        q: &mut f64,
    ) -> f64 {
        let mut sum = 0.0;
        for &(id, from_b) in &self.inc[u] {
            if id == arrival || id == excluded {
                continue;
            }
            let ch = &self.chains[id as usize];
            let (stop, other) = if from_b { (ch.stop_from_b, ch.c) } else { (ch.stop_from_c, ch.b) };
            sum += w * stop;
            if !visited[other] {
                let wc = w * ch.through;
                if other == target {
                    *q += wc;
                }
                visited[other] = true;
                sum += wc * ps.hang1[other] + self.search(ps, visited, other, wc, id, excluded, target, q);
                visited[other] = false;
            }
        }
        sum
    }

    /// Sum of the core paths leaving each core atom (indexed by atom).
    fn paths(&self, ps: &PathSums) -> Vec<f64> {
        const NONE: u32 = u32::MAX;
        let n = ps.in_core.len();
        let mut out = vec![0.0; n];
        let mut visited = vec![false; n];
        let mut q = 0.0;
        for b in (0..n).filter(|&a| self.is_branch[a]) {
            visited[b] = true;
            out[b] = self.search(ps, &mut visited, b, 1.0, NONE, NONE, usize::MAX, &mut q);
            visited[b] = false;
        }
        for (id, ch) in self.chains.iter().enumerate() {
            if ch.atoms.is_empty() {
                continue;
            }
            // continuation from each end avoiding this chain, and the weight of
            // reaching the other end that way (1 for a loop: already there)
            let mut cont = |from: usize, to: usize| -> (f64, f64) {
                let mut q = if from == to { 1.0 } else { 0.0 };
                visited[from] = true;
                let r = self.search(ps, &mut visited, from, 1.0, NONE, id as u32, to, &mut q);
                visited[from] = false;
                (r, q)
            };
            let (r_b, q_bc) = cont(ch.b, ch.c);
            let (r_c, q_cb) = cont(ch.c, ch.b);
            let k = ch.atoms.len();
            for i in 0..k {
                // towards b: x_{i-1} .. x_0 (= first internal atom), then b
                let mut sum = 0.0;
                let mut w = 1.0;
                for j in (0..i).rev() {
                    w *= ch.f[j + 1];
                    sum += w * ps.hang1[ch.atoms[j]];
                }
                w *= ch.f[0];
                // beyond s, entered from c: x_{k-1} down to x_{i+1}
                let mut partial = 0.0;
                let mut v = 1.0;
                for j in (i + 1..k).rev() {
                    v *= ch.f[j + 1];
                    partial += v * ps.hang1[ch.atoms[j]];
                }
                sum += w * (ps.hang1[ch.b] + r_b + q_bc * partial);
                // towards c: x_{i+1} .. x_{k-1}, then c
                let mut w = 1.0;
                for j in i + 1..k {
                    w *= ch.f[j];
                    sum += w * ps.hang1[ch.atoms[j]];
                }
                w *= ch.f[k];
                // beyond s, entered from b: x_0 up to x_{i-1}
                let mut partial = 0.0;
                let mut v = 1.0;
                for j in 0..i {
                    v *= ch.f[j];
                    partial += v * ps.hang1[ch.atoms[j]];
                }
                sum += w * (ps.hang1[ch.c] + r_c + q_cb * partial);
                out[ch.atoms[i]] = sum;
            }
        }
        out
    }
}

impl Chain {
    fn finish(&mut self, ps: &PathSums) {
        let k = self.atoms.len();
        let (mut w, mut stop) = (1.0, 0.0);
        for j in 0..k {
            w *= self.f[j];
            stop += w * ps.hang1[self.atoms[j]];
        }
        self.stop_from_b = stop;
        self.through = w * self.f[k];
        let (mut w, mut stop) = (1.0, 0.0);
        for j in (0..k).rev() {
            w *= self.f[j + 1];
            stop += w * ps.hang1[self.atoms[j]];
        }
        self.stop_from_c = stop;
    }
}

/// Fixed-width bitset over compact core indices (memoised core search).
trait CoreBits: Copy + Eq + std::hash::Hash {
    fn zero() -> Self;
    fn bit(i: usize) -> Self;
    fn has(self, i: usize) -> bool;
    fn or(self, o: Self) -> Self;
    fn and_not(self, o: Self) -> Self;
    fn and(self, o: Self) -> Self;
    fn is_zero(self) -> bool;
    /// Exactly one bit set.
    fn is_single(self) -> bool;
    /// Removes and returns the lowest set bit.
    fn pop(&mut self) -> usize;
}

macro_rules! core_bits {
    ($t:ty) => {
        impl CoreBits for $t {
            #[inline]
            fn zero() -> Self {
                0
            }
            #[inline]
            fn bit(i: usize) -> Self {
                1 << i
            }
            #[inline]
            fn has(self, i: usize) -> bool {
                self >> i & 1 != 0
            }
            #[inline]
            fn or(self, o: Self) -> Self {
                self | o
            }
            #[inline]
            fn and_not(self, o: Self) -> Self {
                self & !o
            }
            #[inline]
            fn and(self, o: Self) -> Self {
                self & o
            }
            #[inline]
            fn is_zero(self) -> bool {
                self == 0
            }
            #[inline]
            fn is_single(self) -> bool {
                self.is_power_of_two()
            }
            #[inline]
            fn pop(&mut self) -> usize {
                let i = self.trailing_zeros() as usize;
                *self &= *self - 1;
                i
            }
        }
    };
}
core_bits!(u64);
core_bits!(u128);

/// Multi-word bitset for larger cores.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Words<const N: usize>([u64; N]);

impl<const N: usize> CoreBits for Words<N> {
    #[inline]
    fn zero() -> Self {
        Words([0; N])
    }
    #[inline]
    fn bit(i: usize) -> Self {
        let mut w = [0; N];
        w[i / 64] = 1 << (i % 64);
        Words(w)
    }
    #[inline]
    fn has(self, i: usize) -> bool {
        self.0[i / 64] >> (i % 64) & 1 != 0
    }
    #[inline]
    fn or(self, o: Self) -> Self {
        Words(std::array::from_fn(|k| self.0[k] | o.0[k]))
    }
    #[inline]
    fn and_not(self, o: Self) -> Self {
        Words(std::array::from_fn(|k| self.0[k] & !o.0[k]))
    }
    #[inline]
    fn and(self, o: Self) -> Self {
        Words(std::array::from_fn(|k| self.0[k] & o.0[k]))
    }
    #[inline]
    fn is_zero(self) -> bool {
        self.0.iter().all(|&x| x == 0)
    }
    #[inline]
    fn is_single(self) -> bool {
        self.0.iter().map(|x| x.count_ones()).sum::<u32>() == 1
    }
    #[inline]
    fn pop(&mut self) -> usize {
        for k in 0..N {
            if self.0[k] != 0 {
                let i = self.0[k].trailing_zeros() as usize;
                self.0[k] &= self.0[k] - 1;
                return k * 64 + i;
            }
        }
        unreachable!("pop on an empty bitset")
    }
}

/// Largest core handled by the bitset memo (larger ones use the generic search).
const MAX_BITSET_CORE: usize = 512;

/// `PathSums::core_future` on compact core indices with single-word bitsets:
/// no allocation per state, components grown by whole-frontier bit operations,
/// and one component search per connected region instead of per neighbour.
/// Same additions in the same order as the generic search.
struct CoreMemo<B: CoreBits> {
    /// Per core atom: (neighbour, bond factor, neighbour's `hang1`).
    edges: Vec<smallvec::SmallVec<[(u32, f64, f64); 4]>>,
    nbr_mask: Vec<B>,
    /// Per core atom `u`: short cycles through `u` as (positions of the two
    /// neighbours in `edges[u]`, the cycle's atoms other than `u`). While a
    /// cycle's atoms are all unvisited, its two neighbours of `u` stay
    /// connected without `u`.
    cycles: Vec<smallvec::SmallVec<[(u8, u8, B); 3]>>,
    memo: rustc_hash::FxHashMap<(u32, B), f64>,
}

impl<B: CoreBits> CoreMemo<B> {
    /// Sum of the core paths leaving each core atom (in `core` order).
    fn run(ps: &PathSums, core: &[usize]) -> Vec<f64> {
        let n = ps.in_core.len();
        let mut local = vec![u32::MAX; n];
        for (i, &a) in core.iter().enumerate() {
            local[a] = i as u32;
        }
        let mut edges = Vec::with_capacity(core.len());
        let mut nbr_mask = Vec::with_capacity(core.len());
        for &a in core {
            let mut list = smallvec::SmallVec::new();
            let mut mask = B::zero();
            for &(b, factor, hang1) in &ps.core_edges[ps.core_start[a] as usize..ps.core_start[a + 1] as usize] {
                let lb = local[b as usize];
                list.push((lb, factor, hang1));
                mask = mask.or(B::bit(lb as usize));
            }
            edges.push(list);
            nbr_mask.push(mask);
        }
        let mut m = CoreMemo { edges, nbr_mask, cycles: Vec::new(), memo: rustc_hash::FxHashMap::default() };
        m.cycles = (0..core.len()).map(|u| m.short_cycles(u)).collect();
        let all = (0..core.len()).fold(B::zero(), |acc, i| acc.or(B::bit(i)));
        (0..core.len())
            .map(|a| {
                let comp = m.component(all, a);
                m.future(a, comp)
            })
            .collect()
    }

    /// For each pair of neighbours of `u`, a shortest cycle through `u` and
    /// both (BFS avoiding `u`), as in [`CoreMemo::cycles`].
    fn short_cycles(&self, u: usize) -> smallvec::SmallVec<[(u8, u8, B); 3]> {
        let mut out = smallvec::SmallVec::new();
        let c = self.edges.len();
        let nbrs = &self.edges[u];
        let mut prev = vec![u32::MAX; c];
        let mut queue = Vec::with_capacity(c);
        for i in 0..nbrs.len() {
            let src = nbrs[i].0 as usize;
            // BFS from neighbour i without u
            prev.fill(u32::MAX);
            prev[u] = u as u32;
            prev[src] = src as u32;
            queue.clear();
            queue.push(src);
            let mut head = 0;
            while head < queue.len() {
                let a = queue[head];
                head += 1;
                for &(b, _, _) in &self.edges[a] {
                    if prev[b as usize] == u32::MAX {
                        prev[b as usize] = a as u32;
                        queue.push(b as usize);
                    }
                }
            }
            for j in i + 1..nbrs.len() {
                let mut x = nbrs[j].0 as usize;
                if prev[x] == u32::MAX {
                    continue;
                }
                let mut mask = B::bit(x);
                while x != src {
                    x = prev[x] as usize;
                    mask = mask.or(B::bit(x));
                }
                out.push((i as u8, j as u8, mask));
            }
        }
        out
    }

    /// True when every neighbour of `u` in `onward` is connected to the
    /// others inside `rest` through one of `u`'s short cycles; then `rest`
    /// (all reachable from `u`) is a single component.
    #[inline]
    fn onward_connected(&self, u: usize, rest: B, onward: B) -> bool {
        let mut root = [0u8; 8];
        let k = self.edges[u].len();
        if k > root.len() {
            return false;
        }
        for (i, r) in root.iter_mut().enumerate().take(k) {
            *r = i as u8;
        }
        fn find(root: &mut [u8; 8], mut x: u8) -> u8 {
            while root[x as usize] != x {
                x = root[x as usize];
            }
            x
        }
        for &(i, j, mask) in &self.cycles[u] {
            if mask.and_not(rest).is_zero() {
                let (ri, rj) = (find(&mut root, i), find(&mut root, j));
                root[ri as usize] = rj;
            }
        }
        let mut first = u8::MAX;
        for (i, &(v, _, _)) in self.edges[u].iter().enumerate() {
            if onward.has(v as usize) {
                let r = find(&mut root, i as u8);
                if first == u8::MAX {
                    first = r;
                } else if r != first {
                    return false;
                }
            }
        }
        true
    }

    /// Core atoms reachable from `v` inside `mask`.
    #[inline]
    fn component(&self, mask: B, v: usize) -> B {
        let mut comp = B::bit(v);
        let mut frontier = comp;
        while !frontier.is_zero() {
            let mut next = B::zero();
            while !frontier.is_zero() {
                next = next.or(self.nbr_mask[frontier.pop()]);
            }
            frontier = next.and(mask).and_not(comp);
            comp = comp.or(frontier);
        }
        comp
    }

    fn future(&mut self, u: usize, comp: B) -> f64 {
        let rest = comp.and_not(B::bit(u));
        let onward = self.nbr_mask[u].and(rest);
        if onward.is_zero() {
            return 0.0;
        }
        if onward.is_single() {
            // a single way on (e.g. along a chain): its component is all of
            // `rest`, and memoising this state would not save any work
            let mut sum = 0.0;
            for i in 0..self.edges[u].len() {
                let (v, factor, hang1) = self.edges[u][i];
                if rest.has(v as usize) {
                    sum += factor * (hang1 + self.future(v as usize, rest));
                }
            }
            return sum;
        }
        if let Some(&v) = self.memo.get(&(u as u32, comp)) {
            return v;
        }
        let mut sum = 0.0;
        // components of `rest` found so far (a non-articulation atom has one)
        let mut found: smallvec::SmallVec<[B; 4]> = smallvec::SmallVec::new();
        if self.onward_connected(u, rest, onward) {
            found.push(rest);
        }
        for i in 0..self.edges[u].len() {
            let (v, factor, hang1) = self.edges[u][i];
            let v = v as usize;
            if !rest.has(v) {
                continue;
            }
            let next = match found.iter().find(|c| c.has(v)) {
                Some(&c) => c,
                None => {
                    let c = self.component(rest, v);
                    found.push(c);
                    c
                }
            };
            sum += factor * (hang1 + self.future(v, next));
        }
        if self.memo.len() >= MEMO_LIMIT {
            self.memo.clear();
        }
        self.memo.insert((u as u32, comp), sum);
        sum
    }
}

/// CDK `WeightedPathDescriptor` values `[WTPT-1, ..., WTPT-5]`.
pub fn weighted_path_values(mol: &Molecule) -> [f64; 5] {
    let n = mol.num_atoms();
    let g = Graph::new(mol);
    let per_atom = PathSums::new(&g).per_atom();

    let mut all = 0.0;
    let mut hetero = 0.0;
    let mut oxygen = 0.0;
    let mut nitrogen = 0.0;
    for (start, &w) in per_atom.iter().enumerate() {
        all += w;
        match mol.atoms[start].atomic_number {
            6 => {}
            7 => {
                hetero += w;
                nitrogen += w;
            }
            8 => {
                hetero += w;
                oxygen += w;
            }
            _ => hetero += w,
        }
    }
    let uniq = 0.5 * (all + n as f64);
    [uniq, uniq / n as f64, hetero, oxygen, nitrogen]
}

pub struct WeightedPath;

impl Descriptor for WeightedPath {
    fn name(&self) -> &str {
        "WeightedPath"
    }

    fn calculate(&self, mol: &Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        // CDK computes on `removeHydrogens(container)`
        if let Some(suppressed) = crate::algo::sanitize::cdk_remove_hydrogens(mol) {
            return DescriptorValue::FloatVec(weighted_path_values(&suppressed).to_vec());
        }
        DescriptorValue::FloatVec(weighted_path_values(mol).to_vec())
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<WeightedPath>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_wp(smiles: &str) -> [f64; 5] {
        weighted_path_values(&parse_smiles(smiles).expect("valid SMILES"))
    }

    fn assert_close(got: [f64; 5], expected: [f64; 5]) {
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-9, "got {got:?}, expected {expected:?}");
        }
    }

    // Expected values from CDK 2.13 WeightedPathDescriptor.
    #[test]
    fn test_wp_small() {
        assert_close(calc_wp("C"), [1.0, 1.0, 0.0, 0.0, 0.0]);
        assert_close(calc_wp("CC"), [3.0, 1.5, 0.0, 0.0, 0.0]);
        // CCO: 3 atoms + bonds 1/sqrt(2) (x2) + path C-C-O 1/2
        let b = 1.0 / 2f64.sqrt();
        assert_close(calc_wp("CCO"), [3.0 + 2.0 * b + 0.5, (3.0 + 2.0 * b + 0.5) / 3.0, 1.0 + b + 0.5, 1.0 + b + 0.5, 0.0]);
    }

    #[test]
    fn test_wp_benzene() {
        let wp = calc_wp("c1ccccc1");
        // 6 atoms + two paths per atom pair (each bond weighs 1/2):
        // neighbours 1/2 + 1/32, distance 2: 1/4 + 1/16, opposite: 2 * 1/8
        let expected = 6.0 + 6.0 * (0.5 + 1.0 / 32.0) + 6.0 * (0.25 + 1.0 / 16.0) + 3.0 * (2.0 / 8.0);
        assert!((wp[0] - expected).abs() < 1e-9, "WTPT-1 = {}", wp[0]);
        assert_eq!(wp[2], 0.0);
    }

    #[test]
    fn test_wp_deterministic() {
        assert_eq!(calc_wp("CCN(C)C=O"), calc_wp("CCN(C)C=O"));
    }

    /// Brute force: DFS over every simple path from every atom (CDK's definition).
    fn brute_force(mol: &Molecule) -> [f64; 5] {
        fn dfs(g: &Graph, visited: &mut [bool], a: usize, w: f64) -> f64 {
            let mut sum = 0.0;
            for e in g.edges(a) {
                let b = g.nbr[e] as usize;
                if !visited[b] {
                    let wb = w * g.factor[e];
                    visited[b] = true;
                    sum += wb + dfs(g, visited, b, wb);
                    visited[b] = false;
                }
            }
            sum
        }
        let n = mol.num_atoms();
        let g = Graph::new(mol);
        let mut visited = vec![false; n];
        let (mut all, mut het, mut o, mut nn) = (0.0, 0.0, 0.0, 0.0);
        for s in 0..n {
            visited[s] = true;
            let w = 1.0 + dfs(&g, &mut visited, s, 1.0);
            visited[s] = false;
            all += w;
            match mol.atoms[s].atomic_number {
                6 => {}
                7 => {
                    het += w;
                    nn += w;
                }
                8 => {
                    het += w;
                    o += w;
                }
                _ => het += w,
            }
        }
        let uniq = 0.5 * (all + n as f64);
        [uniq, uniq / n as f64, het, o, nn]
    }

    #[test]
    fn core_memo_bitset_widths_agree() {
        for smi in [
            "c1ccc2cc3cc4cc5cc6cc7cc8cc9ccccc9cc8cc7cc6cc5cc4cc3cc2c1",
            "C1CC2CC3CC4CC5CC6CC7CC8CC9CC1C1C2C3C4C5C6C7C8C91",
        ] {
            let mol = parse_smiles(smi).unwrap();
            let g = Graph::new(&mol);
            let ps = PathSums::new(&g);
            let core: Vec<usize> = (0..mol.num_atoms()).filter(|&a| ps.in_core[a]).collect();
            let a = CoreMemo::<u64>::run(&ps, &core);
            assert_eq!(a, CoreMemo::<u128>::run(&ps, &core), "{smi}");
            assert_eq!(a, CoreMemo::<Words<4>>::run(&ps, &core), "{smi}");
            assert_eq!(a, CoreMemo::<Words<8>>::run(&ps, &core), "{smi}");
        }
    }

    #[test]
    fn core_decomposition_matches_brute_force() {
        for smi in [
            "C", "CC", "CCO", "CC(C)(C)C", "c1ccccc1", "Oc1ccccc1CCN", "C1CC1C1CC1", "c1ccc2ccccc2c1",
            "CC(=O)Oc1ccccc1C(=O)O", "C12C3C4C1C5C2C3C45", "CC.O", "[Na+].[Cl-]", "C1CCC2(CC1)CCCC2",
            "OCC1OC(O)C(O)C(O)C1O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccc(cc1)C1CC1c1ccccc1",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "C1CC2CC1CC2CCC1CCCC1",
            // chain contraction: plain cycles, loops, parallel chains, bridges
            "C1CC1", "Cc1ccccc1", "C1CCCCCCCCCCCC1", "C1CC1.C1CCC1", "OC1CCCC1.c1ccccc1C",
            "C1CCCCC1CCC1CCCCC1", "C1CC2CCC1C2", "C12CC1C2", "c1ccc2c(c1)ccc1ccccc12",
            "C1CC(CCN2CCCC2)CCC1", "C1CCC2(CCC3(CC2)CCCC3)CC1", "C1C2CC3CC1CC(C2)C3",
            "C1CCC(CC1)C1CCCCC1C1CCCCC1", "C1CC2C(C1)C1CCC2C1",
            // > MEMO_MIN_RINGS fused rings: memoised core search
            "c1ccc2cc3cc4cc5cc6cc7cc8cc9ccccc9cc8cc7cc6cc5cc4cc3cc2c1",
            "C1CC2CC3CC4CC5CC6CC7CC8CC9CC1C1C2C3C4C5C6C7C8C91",
        ] {
            let mol = parse_smiles(smi).unwrap();
            let (fast, slow) = (weighted_path_values(&mol), brute_force(&mol));
            for (f, s) in fast.iter().zip(&slow) {
                assert!((f - s).abs() <= 1e-12 * s.abs().max(1.0) || (f.is_nan() && s.is_nan()), "{smi}: {fast:?} vs {slow:?}");
            }
        }
    }
}

