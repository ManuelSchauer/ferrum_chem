//! CDK 2.13 cycle perception used by the CDK count descriptors.
//!
//! - [`all_cycles`]: CDK `AllCycles` (Hanser et al. path-graph reduction,
//!   `RegularPathGraph` / `JumboPathGraph`), including its abort criterion: the
//!   computation is *incomplete* when a vertex about to be removed has more
//!   path edges than the threshold.
//! - [`ring_systems`]: the isolated / fused split of CDK `RingSearch` (blocks
//!   that are a single cycle are isolated, every other cyclic block is fused).
//! - [`daylight_prepared_aromaticity`]: aromaticity as applied by the CDK
//!   reference preparation, `new Aromaticity(ElectronDonation.daylight(),
//!   Cycles.or(Cycles.all(), Cycles.all(6))).apply(mol)` (`Aromaticity.findBonds`).
//! - [`all_rings_finder_completes`]: whether `new AllRingsFinder().findAllRings`
//!   finishes (threshold `PubChem_99`) or throws.
//!
//! Sources: `AllCycles`, `RegularPathGraph`, `JumboPathGraph`, `RingSearch`,
//! `Cycles`, `Aromaticity`, `AllRingsFinder` (CDK 2.13).

use crate::desc::cdk_daylight::{CdkMol, daylight_model_contrib};

/// `Cycles.all()` / `Cycles.all(int)` threshold (`AllRingsFinder.Threshold.PubChem_994`).
const CYCLES_ALL_THRESHOLD: usize = 3072;
/// `new AllRingsFinder()` threshold (`AllRingsFinder.Threshold.PubChem_99`).
const ALL_RINGS_FINDER_THRESHOLD: usize = 684;

/// Vertex set of a path (internal vertices): a `long` in CDK's
/// `RegularPathGraph` (fewer than 64 vertices), a `BitSet` in `JumboPathGraph`.
trait VertexSet: Clone {
    fn empty(words: usize) -> Self;
    fn disjoint(&self, other: &Self) -> bool;
    /// Union of both sets plus vertex `x`.
    fn join(&self, other: &Self, x: u32) -> Self;
    fn len(&self) -> usize;
}

impl VertexSet for u64 {
    #[inline]
    fn empty(_: usize) -> Self {
        0
    }
    #[inline]
    fn disjoint(&self, other: &Self) -> bool {
        self & other == 0
    }
    #[inline]
    fn join(&self, other: &Self, x: u32) -> Self {
        self | other | (1u64 << x)
    }
    #[inline]
    fn len(&self) -> usize {
        self.count_ones() as usize
    }
}

impl VertexSet for Vec<u64> {
    fn empty(words: usize) -> Self {
        vec![0; words]
    }
    fn disjoint(&self, other: &Self) -> bool {
        self.iter().zip(other).all(|(p, q)| p & q == 0)
    }
    fn join(&self, other: &Self, x: u32) -> Self {
        let mut xs: Vec<u64> = self.iter().zip(other).map(|(p, q)| p | q).collect();
        xs[x as usize / 64] |= 1u64 << (x % 64);
        xs
    }
    fn len(&self) -> usize {
        self.iter().map(|w| w.count_ones() as usize).sum()
    }
}

/// Path edge of the path graph; `e == NONE` marks a simple (molecule) edge.
struct PathEdge<B> {
    u: u32,
    v: u32,
    /// Internal vertices of the path.
    xs: B,
    e: u32,
    f: u32,
}

const NONE: u32 = u32::MAX;

impl<B> PathEdge<B> {
    #[inline]
    fn other(&self, x: u32) -> u32 {
        if self.u == x { self.v } else { self.u }
    }
}

/// CDK `RegularPathGraph` / `JumboPathGraph` (identical semantics).
struct PathGraph<B> {
    edges: Vec<PathEdge<B>>,
    graph: Vec<Vec<u32>>,
    rank: Vec<u32>,
    /// Maximum path length (cycle size + 1, first/last vertex repeat).
    limit: usize,
}

impl<B: VertexSet> PathGraph<B> {
    fn new(g: &[Vec<u32>], rank: Vec<u32>, max_cycle_size: usize) -> Self {
        let n = g.len();
        let words = n.div_ceil(64).max(1);
        let mut pg = PathGraph {
            edges: Vec::new(),
            graph: vec![Vec::new(); n],
            rank,
            limit: max_cycle_size + 1,
        };
        for (v, nbrs) in g.iter().enumerate() {
            for &w in nbrs {
                if w as usize > v {
                    let id = pg.edges.len() as u32;
                    pg.edges.push(PathEdge { u: v as u32, v: w, xs: B::empty(words), e: NONE, f: NONE });
                    pg.add(id);
                }
            }
        }
        pg
    }

    fn add(&mut self, id: u32) {
        let (u, v) = (self.edges[id as usize].u, self.edges[id as usize].v);
        if self.rank[u as usize] < self.rank[v as usize] {
            self.graph[u as usize].push(id);
        } else {
            self.graph[v as usize].push(id);
        }
    }

    #[inline]
    fn degree(&self, x: usize) -> usize {
        self.graph[x].len()
    }

    /// Removes vertex `x`, combining its path edges; closed paths are cycles
    /// (collected as closed walks when `cycles` is given).
    fn remove(&mut self, x: u32, mut cycles: Option<&mut Vec<Vec<u32>>>) {
        let list = std::mem::take(&mut self.graph[x as usize]);
        let n = list.len();
        for i in 0..n {
            for j in i + 1..n {
                let (ei, ej) = (list[i] as usize, list[j] as usize);
                let (a, b) = (&self.edges[ei], &self.edges[ej]);
                if !a.xs.disjoint(&b.xs) {
                    continue;
                }
                let xs = a.xs.join(&b.xs, x);
                let len = xs.len() + 2;
                if len > self.limit {
                    continue;
                }
                let (u, v) = (a.other(x), b.other(x));
                if u == v {
                    if let Some(cycles) = cycles.as_deref_mut() {
                        let mut path = Vec::with_capacity(len);
                        path.push(u);
                        // ReducedEdge(a, b, x).path(): u, then a (u -> x), then b (x -> u)
                        self.reconstruct(ei as u32, &mut path);
                        self.reconstruct(ej as u32, &mut path);
                        cycles.push(path);
                    }
                } else {
                    let id = self.edges.len() as u32;
                    self.edges.push(PathEdge { u, v, xs, e: ei as u32, f: ej as u32 });
                    self.add(id);
                }
            }
        }
    }

    fn reconstruct(&self, id: u32, path: &mut Vec<u32>) {
        let edge = &self.edges[id as usize];
        let prev = *path.last().unwrap();
        if edge.e == NONE {
            path.push(edge.other(prev));
        } else if edge.u == prev {
            self.reconstruct(edge.e, path);
            self.reconstruct(edge.f, path);
        } else {
            self.reconstruct(edge.f, path);
            self.reconstruct(edge.e, path);
        }
    }
}

/// CDK `AllCycles(graph, maxCycleSize, maxDegree)`: whether the computation
/// completes, and (when `want_cycles`) every simple cycle of at most
/// `max_cycle_size` vertices as a closed walk (first vertex repeated).
/// `graph` must have more than two vertices.
pub(crate) fn all_cycles(
    g: &[Vec<u32>],
    max_cycle_size: usize,
    max_degree: usize,
    want_cycles: bool,
) -> (Vec<Vec<u32>>, bool) {
    if g.len() < 64 {
        run_all_cycles::<u64>(g, max_cycle_size, max_degree, want_cycles)
    } else {
        run_all_cycles::<Vec<u64>>(g, max_cycle_size, max_degree, want_cycles)
    }
}

fn run_all_cycles<B: VertexSet>(
    g: &[Vec<u32>],
    max_cycle_size: usize,
    max_degree: usize,
    want_cycles: bool,
) -> (Vec<Vec<u32>>, bool) {
    let ord = g.len();
    // AllCycles.rank: counting sort by degree, ties by vertex index
    let mut count = vec![0usize; ord + 2];
    for nbrs in g {
        count[nbrs.len() + 1] += 1;
    }
    let mut i = 0;
    while count[i] < ord {
        count[i + 1] += count[i];
        i += 1;
    }
    let mut rank = vec![0u32; ord];
    for v in 0..ord {
        let d = g[v].len();
        rank[v] = count[d] as u32;
        count[d] += 1;
    }
    let mut vertices = vec![0u32; ord];
    for v in 0..ord {
        vertices[rank[v] as usize] = v as u32;
    }
    let mut pg = PathGraph::<B>::new(g, rank, max_cycle_size);
    let mut cycles = Vec::new();
    let mut removed = 0;
    for &v in &vertices {
        if pg.degree(v as usize) > max_degree {
            break;
        }
        pg.remove(v, want_cycles.then_some(&mut cycles));
        removed += 1;
    }
    (cycles, removed == ord)
}

/// Cyclic blocks of a graph: `(vertices sorted ascending, is a single cycle)`.
///
/// Matches CDK `RingSearch.isolated()` (single cycles; spiro-joined rings are
/// separate) and `RingSearch.fused()` (ring systems sharing bonds).
pub(crate) fn ring_systems(g: &[Vec<u32>]) -> Vec<(Vec<u32>, bool)> {
    let n = g.len();
    let mut disc = vec![u32::MAX; n];
    let mut low = vec![0u32; n];
    let mut time = 0u32;
    let mut edge_stack: Vec<(u32, u32)> = Vec::new();
    let mut out = Vec::new();
    // iterative DFS: (vertex, parent, next neighbour index)
    let mut stack: Vec<(u32, u32, usize)> = Vec::new();
    for root in 0..n {
        if disc[root] != u32::MAX || g[root].is_empty() {
            continue;
        }
        disc[root] = time;
        low[root] = time;
        time += 1;
        stack.push((root as u32, u32::MAX, 0));
        while let Some(&(v, parent, k)) = stack.last() {
            let vu = v as usize;
            if k < g[vu].len() {
                let w = g[vu][k];
                stack.last_mut().unwrap().2 += 1;
                let wu = w as usize;
                if disc[wu] == u32::MAX {
                    edge_stack.push((v, w));
                    disc[wu] = time;
                    low[wu] = time;
                    time += 1;
                    stack.push((w, v, 0));
                } else if w != parent && disc[wu] < disc[vu] {
                    edge_stack.push((v, w));
                    low[vu] = low[vu].min(disc[wu]);
                }
            } else {
                stack.pop();
                if let Some(&(p, _, _)) = stack.last() {
                    let pu = p as usize;
                    low[pu] = low[pu].min(low[vu]);
                    if low[vu] >= disc[pu] {
                        // pop the block of edge (p, v)
                        let mut verts: Vec<u32> = Vec::new();
                        let mut n_edges = 0usize;
                        while let Some((a, b)) = edge_stack.pop() {
                            verts.push(a);
                            verts.push(b);
                            n_edges += 1;
                            if a == p && b == v {
                                break;
                            }
                        }
                        if n_edges >= 3 {
                            verts.sort_unstable();
                            verts.dedup();
                            let isolated = verts.len() == n_edges;
                            out.push((verts, isolated));
                        }
                    }
                }
            }
        }
    }
    out
}

/// `GraphUtil.subgraph(graph, include)` (`include` sorted ascending).
fn subgraph(g: &[Vec<u32>], include: &[u32], map: &mut [u32]) -> Vec<Vec<u32>> {
    for (i, &v) in include.iter().enumerate() {
        map[v as usize] = i as u32;
    }
    let sub = include
        .iter()
        .map(|&v| {
            g[v as usize]
                .iter()
                .filter(|&&w| map[w as usize] != u32::MAX)
                .map(|&w| map[w as usize])
                .collect()
        })
        .collect();
    for &v in include {
        map[v as usize] = u32::MAX;
    }
    sub
}

/// Cycles of `g` as `CycleComputation.ALL` (`max_len == None`) or
/// `Cycles.all(max_len)` find them: vertex lists (closed walks for fused
/// systems, plain vertex sets for isolated rings). `None` when intractable.
fn find_cycles(g: &[Vec<u32>], max_len: Option<usize>, threshold: usize) -> Option<Vec<Vec<u32>>> {
    let mut out = Vec::new();
    let mut map = vec![u32::MAX; g.len()];
    for (verts, isolated) in ring_systems(g) {
        if isolated {
            if max_len.is_none_or(|m| verts.len() <= m) {
                out.push(verts);
            }
            continue;
        }
        let sub = subgraph(g, &verts, &mut map);
        let size = max_len.map_or(sub.len(), |m| m.min(sub.len()));
        let (cycles, completed) = all_cycles(&sub, size, threshold, true);
        if !completed {
            return None;
        }
        out.extend(cycles.into_iter().map(|c| c.into_iter().map(|x| verts[x as usize]).collect()));
    }
    Some(out)
}

#[inline]
fn huckel(sum: i32) -> bool {
    (sum - 2) % 4 == 0
}

/// Aromatic atom and bond flags of the CDK reference preparation:
/// `new Aromaticity(ElectronDonation.daylight(), Cycles.or(Cycles.all(),
/// Cycles.all(6))).apply(mol)`. A bond is aromatic when it lies on a cycle of
/// atoms that can contribute electrons whose electron sum is 4n+2; atoms are
/// aromatic when they have an aromatic bond. `None` when even the fallback
/// `Cycles.all(6)` is intractable (CDK throws).
pub(crate) fn daylight_prepared_aromaticity(m: &CdkMol) -> Option<(Vec<bool>, Vec<bool>)> {
    let n = m.n();
    let mut arom_atom = vec![false; n];
    let mut arom_bond = vec![false; m.ends.len()];
    // quick exit: no ring bond between two potentially contributing atoms
    if !m.ring_bond.iter().any(|&r| r) {
        return Some((arom_atom, arom_bond));
    }
    let contrib = daylight_model_contrib(m);
    // a cycle needs a ring bond between two contributing atoms
    let cyclic = m.ends.iter().enumerate().any(|(bi, &(s, t))| {
        m.ring_bond[bi] && contrib[s as usize] >= 0 && contrib[t as usize] >= 0
    });
    if !cyclic {
        return Some((arom_atom, arom_bond));
    }
    let subset: Vec<u32> = (0..n as u32).filter(|&a| contrib[a as usize] >= 0).collect();
    let mut map = vec![u32::MAX; n];
    for (i, &a) in subset.iter().enumerate() {
        map[a as usize] = i as u32;
    }
    // GraphUtil.subgraph(graph, subset)
    let sub: Vec<Vec<u32>> = subset
        .iter()
        .map(|&a| {
            m.adj[a as usize]
                .iter()
                .filter(|&&(w, _)| map[w as usize] != u32::MAX)
                .map(|&(w, _)| map[w as usize])
                .collect()
        })
        .collect();
    let cycles = match find_cycles(&sub, None, CYCLES_ALL_THRESHOLD) {
        Some(c) => c,
        None => find_cycles(&sub, Some(6), CYCLES_ALL_THRESHOLD)?,
    };
    for cyc in cycles {
        let closed = cyc.len() > 1 && cyc.first() == cyc.last();
        let verts = if closed { &cyc[1..] } else { &cyc[..] };
        let sum: i32 = verts.iter().map(|&x| contrib[subset[x as usize] as usize]).sum();
        if !huckel(sum) {
            continue;
        }
        if closed {
            for w in cyc.windows(2) {
                let (a, b) = (subset[w[0] as usize] as usize, subset[w[1] as usize] as usize);
                if let Some(bi) = m.bond(a, b) {
                    arom_bond[bi] = true;
                    arom_atom[a] = true;
                    arom_atom[b] = true;
                }
            }
        } else {
            // isolated ring: every bond between its vertices
            for &x in verts {
                let a = subset[x as usize] as usize;
                arom_atom[a] = true;
                for &(w, bi) in &m.adj[a] {
                    let wl = map[w as usize];
                    if wl != u32::MAX && verts.binary_search(&wl).is_ok() {
                        arom_bond[bi as usize] = true;
                    }
                }
            }
        }
    }
    Some((arom_atom, arom_bond))
}

/// Cheap sufficient condition for `AllCycles` to complete: a path edge stored
/// at a vertex is a distinct simple path to another vertex, and two vertices
/// of a graph with cycle rank `r` are joined by at most `2^r` simple paths
/// (each path differs from a fixed one by an element of the cycle space), so
/// no vertex degree can exceed `(|V| - 1) * 2^r`.
fn surely_completes(g: &[Vec<u32>], threshold: usize) -> bool {
    let v = g.len();
    let e = g.iter().map(Vec::len).sum::<usize>() / 2;
    let r = e + 1 - v;
    r < 16 && (v - 1) << r <= threshold
}

/// Whether CDK `new AllRingsFinder().findAllRings(mol)` completes (otherwise it
/// throws "Threshold exceeded for AllRingsFinder"). `graph` is the molecule's
/// adjacency (all atoms).
pub(crate) fn all_rings_finder_completes(graph: &[Vec<u32>]) -> bool {
    let mut map = vec![u32::MAX; graph.len()];
    for (verts, isolated) in ring_systems(graph) {
        if isolated {
            continue;
        }
        let sub = subgraph(graph, &verts, &mut map);
        if surely_completes(&sub, ALL_RINGS_FINDER_THRESHOLD) {
            continue;
        }
        let size = sub.len();
        if !all_cycles(&sub, size, ALL_RINGS_FINDER_THRESHOLD, false).1 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize) -> Vec<Vec<u32>> {
        (0..n).map(|i| vec![((i + n - 1) % n) as u32, ((i + 1) % n) as u32]).collect()
    }

    #[test]
    fn anthracene_like_cycle_counts() {
        // naphthalene skeleton: 3 cycles (two 6-rings, one 10-ring)
        let mut g = ring(10);
        g[0].push(5);
        g[5].push(0);
        let (cycles, done) = all_cycles(&g, 10, 3072, true);
        assert!(done);
        let mut sizes: Vec<usize> = cycles.iter().map(|c| c.len() - 1).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![6, 6, 10]);
        let systems = ring_systems(&g);
        assert_eq!(systems.len(), 1);
        assert!(!systems[0].1);
    }

    #[test]
    fn spiro_rings_are_isolated() {
        // two triangles sharing vertex 0
        let g = vec![vec![1, 2, 3, 4], vec![0, 2], vec![0, 1], vec![0, 4], vec![0, 3]];
        let systems = ring_systems(&g);
        assert_eq!(systems.len(), 2);
        assert!(systems.iter().all(|s| s.1 && s.0.len() == 3));
    }
}
