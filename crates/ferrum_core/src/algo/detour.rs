//! All-pairs path matrices of a connected graph assembled along its
//! block-cut tree: the detour matrix (longest simple paths) and weighted
//! shortest-path matrices.
//!
//! A simple path between two atoms of one biconnected component (block) never
//! leaves that block, and a path between atoms of different blocks runs
//! through the articulation atoms of the block-cut tree in a fixed order. Both
//! the longest and the (positively weighted) shortest path between `u` and `v`
//! are therefore `L_B1(u, a1) + L_B2(a1, a2) + ... + L_Bk(a_{k-1}, v)` along
//! the blocks `B1..Bk` and articulation atoms `a_i` between them, with `L_B`
//! the block-local value (a bridge is a two-atom block). [`BlockTree`] builds
//! that decomposition once per molecule together with an assembly plan, so
//! each matrix costs its block-local work plus one `O(n²)` pass.
//!
//! This is how Mordred (mordred-community 2.0.7, `DetourMatrix.CalcDetour`)
//! computes the detour matrix. In a block that is a simple ring the longest
//! path goes the long way round. In other blocks the longest paths are found
//! by depth-first enumeration of simple paths from every start atom
//! (Mordred's `LongestSimplePath`), here with exact cuts:
//!
//! - branch and bound: a branch at depth `d` whose not yet visited, still
//!   reachable atoms `R` cannot improve any open pair is skipped. A path
//!   continues through at most `|R|` atoms, and through at most one of the
//!   atoms of `R` with a single neighbour in `R` and the path's end (a dead
//!   end can only be the last atom), so `best(s, u) >= d + |R| - #dead +
//!   [u dead]` for every open `u` in `R` means nothing to gain;
//! - dominance (`Memo`): the continuations of a path depend only on its
//!   end atom and `R`, so a path reaching an `(end, R)` state already
//!   searched at a depth at least as large is skipped.
//!
//! Pairs `(s, t)` with `t < s` are final from the search started at `t`, the
//! search from `s` ends once every open pair has a Hamiltonian path, and
//! low-degree neighbours are tried first (long paths are found earlier, so
//! more branches are cut). The result equals Mordred's exhaustive search.

use std::ops::Add;

/// Upper bound on the number of path extensions searched per molecule; beyond
/// it [`BlockTree::detour_matrix`] gives up (Mordred has a 60 s timeout
/// instead and then reports the detour descriptors as missing).
pub const DETOUR_SEARCH_BUDGET: u64 = 200_000_000;

/// Largest block size (atoms) handled by the longest-path search.
const MAX_BLOCK: usize = 64 * 32;

/// Block-cut tree of a connected graph with reusable buffers.
#[derive(Default)]
pub struct BlockTree {
    n: usize,
    // biconnected components (Tarjan)
    disc: Vec<u32>,
    low: Vec<u32>,
    frames: Vec<(u32, u32, u32)>,
    edge_stack: Vec<(u32, u32)>,
    mark: Vec<u32>,
    /// block `b` = atoms `block_atoms[block_start[b]..block_start[b + 1]]`
    block_start: Vec<u32>,
    block_atoms: Vec<u32>,
    /// neighbours inside its block of the atom at position `q` of
    /// `block_atoms`: `(local index, edge id)` at `nbr[nbr_start[q]..nbr_start[q + 1]]`
    nbr_start: Vec<u32>,
    nbr: Vec<(u16, u32)>,
    /// atom -> (block, local index), CSR
    member_start: Vec<u32>,
    member: Vec<(u32, u32)>,
    /// block-local `k x k` values of block `b` start at `pair_start[b]`
    pair_start: Vec<u32>,
    /// assembly plan of row `s`: `row[t] = row[src] + local[pair]` for
    /// `(t, src, pair)` in `plan[plan_start[s]..plan_start[s + 1]]`, in order
    plan: Vec<(u32, u32, u32)>,
    plan_start: Vec<u32>,
    traverse: Vec<(u32, u32, u32)>,
    /// articulation memberships of block `b`: `(local index, other block,
    /// local index there)` at `art[art_start[b]..art_start[b + 1]]`
    art_start: Vec<u32>,
    art: Vec<(u32, u32, u32)>,
    // block-local data
    local: Vec<u32>,
    lens: Vec<u32>,
    wloc: Vec<f64>,
    adj_words: Vec<u64>,
    dfs: Vec<(u16, u16)>,
    /// dominance memo of the longest-path search, see [`Memo`]
    memo: Memo,
}

/// Dominance memo of the longest-path search from one start atom: the
/// continuations of a path ending at `w` depend only on `w` and the set `R`
/// of not yet visited atoms reachable from `w`, so once `(w, R)` has been
/// searched at depth `d`, a later path reaching the same state at a depth
/// `<= d` cannot improve any pair and is skipped. A fixed-size cache
/// (two-way set associative); losing entries only costs time.
#[derive(Default)]
struct Memo {
    /// slots of `1 + W` words: tag (`stamp << 32 | w << 16 | depth`), `R`
    slots: Vec<u64>,
    words: usize,
    stamp: u32,
    /// slots in use (a power of two)
    active: usize,
}

/// Size of the memo in 64-bit words (512 kB; larger tables hardly save any
/// extensions).
const MEMO_WORDS: usize = 1 << 16;

impl Memo {
    /// Starts a search with keys of `w` words (forgets earlier searches).
    fn start(&mut self, w: usize, k: usize) {
        if self.words != w || self.stamp == u32::MAX >> 1 {
            self.words = w;
            let slots = (MEMO_WORDS / (w + 1)).next_power_of_two() / 2;
            self.slots.clear();
            self.slots.resize(slots * (w + 1), 0);
            self.stamp = 0;
        }
        self.stamp += 1;
        // small blocks: a small part of the table (stays in the L1 cache)
        let all = self.slots.len() / (w + 1);
        self.active = if k <= 24 { all.min(1 << 12) } else { all };
    }

    /// Whether `(v, r)` was searched at a depth `>= depth`; otherwise records
    /// it at `depth`.
    #[inline]
    fn dominated<const W: usize>(&mut self, v: usize, r: &[u64; W], depth: u32) -> bool {
        let mut h = (v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for &x in r {
            h = (h ^ x).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        }
        h ^= h >> 32;
        let n_slots = self.active;
        let i0 = (h as usize) & (n_slots - 1) & !1;
        let key = (self.stamp as u64) << 32 | (v as u64) << 16;
        let mut free = i0;
        for i in [i0, i0 + 1] {
            let slot = &self.slots[i * (W + 1)..(i + 1) * (W + 1)];
            if slot[0] >> 16 == key >> 16 && slot[1..] == r[..] {
                if (slot[0] & 0xFFFF) as u32 >= depth {
                    return true;
                }
                free = i;
                break;
            }
            if slot[0] >> 32 != key >> 32 {
                free = i;
            }
        }
        let slot = &mut self.slots[free * (W + 1)..(free + 1) * (W + 1)];
        slot[0] = key | depth as u64;
        slot[1..].copy_from_slice(r);
        false
    }
}

impl BlockTree {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decomposes the graph with `n` atoms whose neighbours of `a` are
    /// `neighbors(a)` as `(atom, edge id)`. Returns `false` when it is
    /// disconnected (or has no atom).
    pub fn build<F, I>(&mut self, n: usize, neighbors: F) -> bool
    where
        F: Fn(usize) -> I,
        I: Iterator<Item = (usize, usize)>,
    {
        self.n = n;
        if n == 0 || !self.biconnected_components(&neighbors) {
            return false;
        }
        let n_blocks = self.block_start.len() - 1;

        // memberships (CSR)
        self.member_start.clear();
        self.member_start.resize(n + 1, 0);
        for &a in &self.block_atoms {
            self.member_start[a as usize + 1] += 1;
        }
        for i in 0..n {
            self.member_start[i + 1] += self.member_start[i];
        }
        self.member.clear();
        self.member.resize(self.block_atoms.len(), (0, 0));
        self.mark.clear();
        self.mark.extend_from_slice(&self.member_start[..n]);
        for b in 0..n_blocks {
            let (s, e) = (self.block_start[b] as usize, self.block_start[b + 1] as usize);
            for (li, &a) in self.block_atoms[s..e].iter().enumerate() {
                let pos = &mut self.mark[a as usize];
                self.member[*pos as usize] = (b as u32, li as u32);
                *pos += 1;
            }
        }

        // block-local neighbour lists (a block is an induced subgraph) and pair offsets
        self.local.clear();
        self.local.resize(n, u32::MAX);
        self.nbr_start.clear();
        self.nbr.clear();
        self.pair_start.clear();
        let mut pairs = 0u32;
        for b in 0..n_blocks {
            let (s, e) = (self.block_start[b] as usize, self.block_start[b + 1] as usize);
            self.pair_start.push(pairs);
            pairs += ((e - s) * (e - s)) as u32;
            for (li, &a) in self.block_atoms[s..e].iter().enumerate() {
                self.local[a as usize] = li as u32;
            }
            for &a in &self.block_atoms[s..e] {
                self.nbr_start.push(self.nbr.len() as u32);
                for (c, edge) in neighbors(a as usize) {
                    let lc = self.local[c];
                    if lc != u32::MAX && self.block_atoms[s + lc as usize] as usize == c {
                        self.nbr.push((lc as u16, edge as u32));
                    }
                }
            }
            for &a in &self.block_atoms[s..e] {
                self.local[a as usize] = u32::MAX;
            }
        }
        self.nbr_start.push(self.nbr.len() as u32);
        self.pair_start.push(pairs);
        // low-degree neighbours first: the longest-path search then finds
        // long paths earlier and cuts more (about 8 % fewer extensions)
        for b in 0..n_blocks {
            let s = self.block_start[b] as usize;
            for q in s..self.block_start[b + 1] as usize {
                let (lo, hi) = (self.nbr_start[q] as usize, self.nbr_start[q + 1] as usize);
                let starts = &self.nbr_start;
                self.nbr[lo..hi].sort_by_key(|&(l, _)| starts[s + l as usize + 1] - starts[s + l as usize]);
            }
        }

        // the other blocks of the articulation atoms of each block:
        // `(local index, other block, its local index)`
        self.art.clear();
        self.art_start.clear();
        for b in 0..n_blocks {
            self.art_start.push(self.art.len() as u32);
            let s = self.block_start[b] as usize;
            for (lj, &a) in self.block_atoms[s..self.block_start[b + 1] as usize].iter().enumerate() {
                let a = a as usize;
                let m = &self.member[self.member_start[a] as usize..self.member_start[a + 1] as usize];
                if m.len() > 1 {
                    self.art.extend(m.iter().filter(|&&(b2, _)| b2 as usize != b).map(|&(b2, l2)| (lj as u32, b2, l2)));
                }
            }
        }
        self.art_start.push(self.art.len() as u32);

        // assembly plan: from every source along the block-cut tree
        self.plan.clear();
        self.plan.reserve(n * (n - 1));
        self.plan_start.clear();
        for src in 0..n {
            self.plan_start.push(self.plan.len() as u32);
            self.traverse.clear();
            for &(b, li) in &self.member[self.member_start[src] as usize..self.member_start[src + 1] as usize] {
                self.traverse.push((b, li, src as u32));
            }
            while let Some((b, li, entry)) = self.traverse.pop() {
                let (b, li) = (b as usize, li as usize);
                let (s, e) = (self.block_start[b] as usize, self.block_start[b + 1] as usize);
                let atoms = &self.block_atoms[s..e];
                let pair0 = self.pair_start[b] + (li * atoms.len()) as u32;
                let at = |lj: usize, a: u32| (a, entry, pair0 + lj as u32);
                self.plan.extend(atoms[..li].iter().enumerate().map(|(lj, &a)| at(lj, a)));
                self.plan.extend(atoms[li + 1..].iter().enumerate().map(|(t, &a)| at(li + 1 + t, a)));
                for &(lj, b2, l2) in &self.art[self.art_start[b] as usize..self.art_start[b + 1] as usize] {
                    if lj as usize != li {
                        self.traverse.push((b2, l2, atoms[lj as usize]));
                    }
                }
            }
        }
        self.plan_start.push(self.plan.len() as u32);
        true
    }

    /// `out[s][v] = Σ local` along the block-cut tree (diagonal 0).
    fn assemble<T: Copy + Default + Add<Output = T>>(&self, local: &[T], out: &mut [T]) {
        let n = self.n;
        for (s, row) in out.chunks_exact_mut(n).enumerate() {
            row[s] = T::default();
            for &(t, src, pair) in self.plan_row(s) {
                row[t as usize] = row[src as usize] + local[pair as usize];
            }
        }
    }

    #[inline]
    fn plan_row(&self, s: usize) -> &[(u32, u32, u32)] {
        &self.plan[self.plan_start[s] as usize..self.plan_start[s + 1] as usize]
    }

    /// Row `s` of the `P` shortest-path matrices from the last
    /// [`BlockTree::weighted_local_distances`] (`row[v] = Σ local` along the
    /// block-cut tree, lane by lane), with `row[s] = diag`.
    pub fn assemble_row<const P: usize>(&self, s: usize, diag: [f64; P], row: &mut [[f64; P]]) {
        let (local, _) = self.wloc.as_chunks::<P>();
        row[s] = [0.0; P];
        for &(t, src, pair) in self.plan_row(s) {
            let (x, y) = (row[src as usize], local[pair as usize]);
            row[t as usize] = std::array::from_fn(|p| x[p] + y[p]);
        }
        row[s] = diag;
    }

    /// Shortest-path matrix (row-major `n x n`, diagonal 0) of the graph
    /// passed to [`BlockTree::build`] with the positive edge weights
    /// `weight[edge id]` (Floyd-Warshall inside each block).
    pub fn weighted_distance_matrix(&mut self, weight: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), self.n * self.n);
        let weight: Vec<[f64; 1]> = weight.iter().map(|&w| [w]).collect();
        self.weighted_local_distances(&weight);
        self.assemble(&self.wloc, out);
    }

    /// Shortest paths inside every block (Floyd-Warshall) for `P` positive
    /// edge weightings at once (`weight[edge id][p]`), kept for
    /// [`BlockTree::assemble_row`]. Every weighting gets exactly the
    /// operations of a separate run.
    pub fn weighted_local_distances<const P: usize>(&mut self, weight: &[[f64; P]]) {
        let n_blocks = self.block_start.len() - 1;
        let pairs = *self.pair_start.last().unwrap_or(&0) as usize;
        self.wloc.clear();
        self.wloc.resize(pairs * P, f64::INFINITY);
        let (wloc, _) = self.wloc.as_chunks_mut::<P>();
        for b in 0..n_blocks {
            let (s, e) = (self.block_start[b] as usize, self.block_start[b + 1] as usize);
            let k = e - s;
            let m = &mut wloc[self.pair_start[b] as usize..][..k * k];
            for li in 0..k {
                m[li * k + li] = [0.0; P];
                for &(lj, edge) in &self.nbr[self.nbr_start[s + li] as usize..self.nbr_start[s + li + 1] as usize] {
                    let (x, w) = (&mut m[li * k + lj as usize], &weight[edge as usize]);
                    for p in 0..P {
                        x[p] = x[p].min(w[p]);
                    }
                }
            }
            if k > 2 {
                for kk in 0..k {
                    let (before, rest) = m.split_at_mut(kk * k);
                    let (pivot, after) = rest.split_at_mut(k);
                    for row in before.chunks_exact_mut(k).chain(after.chunks_exact_mut(k)) {
                        let dik = row[kk];
                        // (an infinite entry leaves the row unchanged)
                        if dik.iter().all(|&x| x == f64::INFINITY) {
                            continue;
                        }
                        for (x, y) in row.iter_mut().zip(pivot.iter()) {
                            for p in 0..P {
                                x[p] = x[p].min(dik[p] + y[p]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Detour matrix (row-major `n x n`, longest simple path lengths) of the
    /// graph passed to [`BlockTree::build`]. Returns `false` (leaving `out`
    /// unspecified) when a block is larger than the search supports or the
    /// search exceeds [`DETOUR_SEARCH_BUDGET`].
    pub fn detour_matrix(&mut self, out: &mut [u32]) -> bool {
        assert_eq!(out.len(), self.n * self.n);
        let n_blocks = self.block_start.len() - 1;
        self.lens.clear();
        self.lens.resize(*self.pair_start.last().unwrap_or(&0) as usize, 0);
        let mut budget = DETOUR_SEARCH_BUDGET;
        for b in 0..n_blocks {
            let (s, e) = (self.block_start[b] as usize, self.block_start[b + 1] as usize);
            let k = e - s;
            let off = self.pair_start[b] as usize;
            if k == 2 {
                self.lens[off + 1] = 1;
                self.lens[off + 2] = 1;
                continue;
            }
            if self.nbr_start[e] - self.nbr_start[s] == 2 * k as u32 {
                self.cycle_block(s, k, off);
                continue;
            }
            if k > MAX_BLOCK {
                return false;
            }
            let ok = match k.div_ceil(64) {
                1 => self.block_search::<1>(s, k, off, &mut budget),
                2 => self.block_search::<2>(s, k, off, &mut budget),
                3 | 4 => self.block_search::<4>(s, k, off, &mut budget),
                5..=8 => self.block_search::<8>(s, k, off, &mut budget),
                9..=16 => self.block_search::<16>(s, k, off, &mut budget),
                _ => self.block_search::<32>(s, k, off, &mut budget),
            };
            if !ok {
                return false;
            }
        }
        self.assemble(&self.lens, out);
        true
    }

    /// Biconnected components (Tarjan, iterative) into `block_start` /
    /// `block_atoms`. Returns `false` for a disconnected graph.
    fn biconnected_components<F, I>(&mut self, neighbors: &F) -> bool
    where
        F: Fn(usize) -> I,
        I: Iterator<Item = (usize, usize)>,
    {
        let n = self.n;
        self.disc.clear();
        self.disc.resize(n, 0);
        self.low.clear();
        self.low.resize(n, 0);
        self.mark.clear();
        self.mark.resize(n, u32::MAX);
        self.block_start.clear();
        self.block_start.push(0);
        self.block_atoms.clear();
        self.edge_stack.clear();
        self.frames.clear();
        let mut time = 1u32;
        self.disc[0] = time;
        self.low[0] = time;
        // frame: (atom, parent, neighbour position)
        self.frames.push((0, u32::MAX, 0));
        while let Some(&mut (v, parent, ref mut pos)) = self.frames.last_mut() {
            let next = neighbors(v as usize).nth(*pos as usize);
            *pos += 1;
            if let Some((w, _)) = next {
                let w = w as u32;
                if self.disc[w as usize] == 0 {
                    self.edge_stack.push((v, w));
                    time += 1;
                    self.disc[w as usize] = time;
                    self.low[w as usize] = time;
                    self.frames.push((w, v, 0));
                } else if w != parent && self.disc[w as usize] < self.disc[v as usize] {
                    self.edge_stack.push((v, w));
                    self.low[v as usize] = self.low[v as usize].min(self.disc[w as usize]);
                }
                continue;
            }
            self.frames.pop();
            if parent == u32::MAX {
                continue;
            }
            let (p, lw) = (parent as usize, self.low[v as usize]);
            self.low[p] = self.low[p].min(lw);
            if lw >= self.disc[p] {
                // pop the block closed by the tree edge (parent, v)
                let id = self.block_start.len() as u32;
                while let Some((x, y)) = self.edge_stack.pop() {
                    for a in [x, y] {
                        if self.mark[a as usize] != id {
                            self.mark[a as usize] = id;
                            self.block_atoms.push(a);
                        }
                    }
                    if (x, y) == (parent, v) {
                        break;
                    }
                }
                self.block_start.push(self.block_atoms.len() as u32);
            }
        }
        self.disc.iter().all(|&d| d != 0)
    }

    /// Longest simple paths in a block that is a simple ring (`k` atoms at
    /// position `s0` of `block_atoms`, as many bonds): the longer way round,
    /// `max(d, k - d)` for atoms `d` apart along the ring.
    fn cycle_block(&mut self, s0: usize, k: usize, off: usize) {
        let nb = |v: usize| {
            let at = self.nbr_start[s0 + v] as usize;
            (self.nbr[at].0 as usize, self.nbr[at + 1].0 as usize)
        };
        // ring position of every atom
        let pos = &mut self.local[..k];
        let (mut prev, mut v) = (0, nb(0).0);
        pos[0] = 0;
        for p in 1..k {
            pos[v] = p as u32;
            let (a, b) = nb(v);
            (prev, v) = (v, if a == prev { b } else { a });
        }
        let lens = &mut self.lens[off..off + k * k];
        for i in 0..k {
            for j in 0..k {
                let d = pos[i].abs_diff(pos[j]);
                lens[i * k + j] = d.max(k as u32 - d);
            }
        }
        for i in 0..k {
            lens[i * k + i] = 0;
        }
    }

    /// Longest simple paths between all atom pairs of the block whose atoms
    /// start at position `s0` of `block_atoms` (`k` atoms), into
    /// `lens[off..off + k * k]`.
    fn block_search<const W: usize>(&mut self, s0: usize, k: usize, off: usize, budget: &mut u64) -> bool {
        let BlockTree { nbr_start, nbr, adj_words, dfs, lens, memo, .. } = self;
        let nbr_start = &nbr_start[s0..=s0 + k];
        // best[s * k + t], t > s: longest path found so far
        let best = &mut lens[off..off + k * k];
        adj_words.clear();
        adj_words.resize(k * W, 0);
        for v in 0..k {
            for &(w, _) in &nbr[nbr_start[v] as usize..nbr_start[v + 1] as usize] {
                adj_words[v * W + w as usize / 64] |= 1u64 << (w % 64);
            }
        }
        let adj = |v: usize| -> &[u64] { &adj_words[v * W..(v + 1) * W] };
        let max_len = (k - 1) as u32;

        for s in 0..k - 1 {
            // open targets: t > s without a Hamiltonian path (k - 1) yet
            let mut need = [0u64; W];
            let mut open = 0u32;
            for t in s + 1..k {
                if best[s * k + t] < max_len {
                    need[t / 64] |= 1u64 << (t % 64);
                    open += 1;
                }
            }
            let mut visited = [0u64; W];
            visited[s / 64] |= 1u64 << (s % 64);
            memo.start(W, k);
            dfs.clear();
            dfs.push((s as u16, 0));
            while open > 0
                && let Some(top) = dfs.last_mut()
            {
                let v = top.0 as usize;
                let pos = nbr_start[v] as usize + top.1 as usize;
                if pos >= nbr_start[v + 1] as usize {
                    dfs.pop();
                    visited[v / 64] &= !(1u64 << (v % 64));
                    continue;
                }
                top.1 += 1;
                let w = nbr[pos].0 as usize;
                if visited[w / 64] >> (w % 64) & 1 != 0 {
                    continue;
                }
                let depth = dfs.len() as u32;
                let cell = &mut best[s * k + w];
                if *cell < depth {
                    *cell = depth;
                    if depth == max_len && w > s {
                        need[w / 64] &= !(1u64 << (w % 64));
                        open -= 1;
                    }
                }
                if *budget == 0 {
                    return false;
                }
                *budget -= 1;
                if depth == max_len {
                    continue;
                }
                visited[w / 64] |= 1u64 << (w % 64);
                // atoms reachable from w without revisiting the path
                let mut reach = [0u64; W];
                let mut frontier = [0u64; W];
                let aw = adj(w);
                let mut any = 0u64;
                for i in 0..W {
                    frontier[i] = aw[i] & !visited[i];
                    any |= frontier[i];
                }
                let mut count = 0u32;
                // atoms of R with at most one neighbour in R + w: dead ends,
                // which a path from w can only visit as its last atom
                let mut avail = [0u64; W];
                for i in 0..W {
                    avail[i] = !visited[i];
                }
                avail[w / 64] |= 1u64 << (w % 64);
                let mut dead = [0u64; W];
                let mut n_dead = 0u32;
                while any != 0 {
                    let mut next = [0u64; W];
                    for i in 0..W {
                        reach[i] |= frontier[i];
                        count += frontier[i].count_ones();
                        let mut bits = frontier[i];
                        while bits != 0 {
                            let u = i * 64 + bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            let au = adj(u);
                            let mut deg = 0;
                            for j in 0..W {
                                next[j] |= au[j];
                                deg += (au[j] & avail[j]).count_ones();
                            }
                            if deg <= 1 {
                                dead[i] |= 1u64 << (u % 64);
                                n_dead += 1;
                            }
                        }
                    }
                    any = 0;
                    for i in 0..W {
                        frontier[i] = next[i] & !visited[i] & !reach[i];
                        any |= frontier[i];
                    }
                }
                if memo.dominated(w, &reach, depth) {
                    visited[w / 64] &= !(1u64 << (w % 64));
                    continue;
                }
                // can a continuation (at most |R| more atoms, no dead end
                // but the last) still improve an open pair (s, u)?
                let bound = depth + count - n_dead;
                let row = &best[s * k..(s + 1) * k];
                let mut improvable = false;
                'scan: for i in 0..W {
                    let mut bits = reach[i] & need[i];
                    while bits != 0 {
                        let u = i * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        if row[u] < bound + (dead[i] >> (u % 64) & 1) as u32 {
                            improvable = true;
                            break 'scan;
                        }
                    }
                }
                if improvable {
                    dfs.push((w as u16, 0));
                } else {
                    visited[w / 64] &= !(1u64 << (w % 64));
                }
            }
        }
        // symmetric matrix from the upper triangle
        for u in 0..k {
            best[u * k + u] = 0;
            for v in 0..u {
                best[u * k + v] = best[v * k + u];
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(n: usize, edges: &[(usize, usize)]) -> (BlockTree, bool) {
        let mut adj = vec![Vec::new(); n];
        for (id, &(a, b)) in edges.iter().enumerate() {
            adj[a].push((b, id));
            adj[b].push((a, id));
        }
        let mut t = BlockTree::new();
        let ok = t.build(n, |a| adj[a].iter().copied());
        (t, ok)
    }

    fn detour(n: usize, edges: &[(usize, usize)]) -> Vec<u32> {
        let (mut t, ok) = tree(n, edges);
        assert!(ok);
        let mut out = vec![0; n * n];
        assert!(t.detour_matrix(&mut out));
        out
    }

    fn adjacency(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); n];
        for &(a, b) in edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        adj
    }

    /// Exhaustive longest simple paths (Mordred's search, no pruning).
    fn brute_longest(n: usize, edges: &[(usize, usize)]) -> Vec<u32> {
        fn go(adj: &[Vec<usize>], v: usize, d: u32, vis: &mut [bool], row: &mut [u32]) {
            for &w in &adj[v] {
                if !vis[w] {
                    vis[w] = true;
                    row[w] = row[w].max(d + 1);
                    go(adj, w, d + 1, vis, row);
                    vis[w] = false;
                }
            }
        }
        let adj = adjacency(n, edges);
        let mut out = vec![0; n * n];
        for s in 0..n {
            let mut vis = vec![false; n];
            vis[s] = true;
            go(&adj, s, 0, &mut vis, &mut out[s * n..(s + 1) * n]);
        }
        out
    }

    /// Floyd-Warshall on the whole graph.
    fn brute_shortest(n: usize, edges: &[(usize, usize)], w: &[f64]) -> Vec<f64> {
        let mut d = vec![f64::INFINITY; n * n];
        for i in 0..n {
            d[i * n + i] = 0.0;
        }
        for (&(a, b), &x) in edges.iter().zip(w) {
            d[a * n + b] = x;
            d[b * n + a] = x;
        }
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    d[i * n + j] = d[i * n + j].min(d[i * n + k] + d[k * n + j]);
                }
            }
        }
        d
    }

    fn cases() -> Vec<(usize, Vec<(usize, usize)>)> {
        vec![
            // propane
            (3, vec![(0, 1), (1, 2)]),
            // methylcyclohexane
            (7, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)]),
            // naphthalene + chain + cyclopropyl
            (
                15,
                vec![
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 0),
                    (4, 6),
                    (6, 7),
                    (7, 8),
                    (8, 9),
                    (9, 3),
                    (9, 10),
                    (10, 11),
                    (11, 12),
                    (12, 13),
                    (13, 14),
                    (14, 12),
                ],
            ),
            // cubane
            (8, vec![(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]),
            // spiro + bicyclo[2.2.2]
            (
                11,
                vec![(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0), (4, 5), (5, 6), (6, 7), (7, 8), (8, 5), (5, 9), (9, 10), (10, 7)],
            ),
        ]
    }

    #[test]
    fn detour_matches_exhaustive_search() {
        for (n, edges) in cases() {
            assert_eq!(detour(n, &edges), brute_longest(n, &edges), "{edges:?}");
        }
    }

    #[test]
    fn weighted_distances_match_floyd_warshall() {
        for (n, edges) in cases() {
            let w: Vec<f64> = (0..edges.len()).map(|i| 0.5 + (i * 7 % 5) as f64 * 0.3).collect();
            let (mut t, ok) = tree(n, &edges);
            assert!(ok);
            let mut out = vec![0.0; n * n];
            t.weighted_distance_matrix(&w, &mut out);
            let exact = brute_shortest(n, &edges, &w);
            for (x, y) in out.iter().zip(&exact) {
                assert!((x - y).abs() < 1e-12, "{edges:?}");
            }
        }
    }

    /// Random connected graphs with many rings: a cycle with chords, fused
    /// and pendant rings, and short chains.
    fn random_ring_graph(n: usize, st: &mut u64) -> Vec<(usize, usize)> {
        let mut rnd = |m: usize| {
            *st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((*st >> 33) as usize) % m
        };
        let c = 3 + rnd(n - 2);
        let mut edges: Vec<(usize, usize)> = (0..c).map(|i| (i, (i + 1) % c)).collect();
        for v in c..n {
            // attach to an earlier atom, sometimes closing a ring to another one
            let a = rnd(v);
            edges.push((a, v));
            if rnd(3) > 0 {
                let b = rnd(v);
                if b != a {
                    edges.push((b, v));
                }
            }
        }
        for _ in 0..rnd(3) {
            let (a, b) = (rnd(n), rnd(n));
            if a != b && !edges.iter().any(|&(x, y)| (x, y) == (a, b) || (x, y) == (b, a)) {
                edges.push((a, b));
            }
        }
        edges
    }

    #[test]
    fn detour_matches_exhaustive_search_random() {
        let mut st = 3u64;
        for case in 0..400 {
            let n = 4 + case % 11;
            let edges = random_ring_graph(n, &mut st);
            assert_eq!(detour(n, &edges), brute_longest(n, &edges), "{edges:?}");
        }
    }

    #[test]
    fn single_atom_and_disconnected() {
        let (mut t, ok) = tree(1, &[]);
        assert!(ok);
        let mut out = vec![7; 1];
        assert!(t.detour_matrix(&mut out));
        assert_eq!(out, vec![0]);
        assert!(!tree(2, &[]).1);
    }
}
