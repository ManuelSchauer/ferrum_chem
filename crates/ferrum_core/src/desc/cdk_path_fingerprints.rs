//! CDK path fingerprints: ports of CDK 2.13 `Fingerprinter`,
//! `ExtendedFingerprinter` and `GraphOnlyFingerprinter` (default parameters:
//! 1024 bits, search depth 7, i.e. paths of up to 8 atoms).
//!
//! CDK enumerates every simple path of non-hydrogen, non-pseudo atoms, encodes
//! it as a string of atom symbols (`getAtomSymbol`: one-letter codes for Cl,
//! Br, Si, ...) and bond symbols (`:` aromatic, `-`, `=`, `#`; none for the
//! graph-only variant), canonicalises the direction (the lexicographically
//! larger of forward/reverse), hashes it and sets bit
//! `new Random(hash).nextInt(size)`. Paths are stored once (CDK keeps the
//! direction whose end atom has the larger identity hash; any total order of
//! the end atoms gives the same set of paths).
//!
//! Hash quirk reproduced exactly: CDK's `appendHash` adds `31 * h + charAt(0)`
//! once per character of a symbol, so a two-letter symbol such as `Sn`
//! contributes `S` twice; single-atom paths use the real `String.hashCode`.
//! The hash is a polynomial over the character sequence, so forward and
//! reverse hashes are accumulated incrementally during the walk (no strings).
//!
//! `Fingerprinter.getBitFingerprint` re-perceives the molecule
//! (`percieveAtomTypesAndConfigureAtoms` + `Aromaticity.cdkLegacy()`), so bond
//! symbols use CDK's legacy aromaticity ([`CdkView`], with CDK's own minimum
//! cycle basis for fused systems of more than three rings, see
//! [`exact_legacy_bonds`]); non-aromatic bonds keep the Kekulé orders of CDK's
//! SMILES parser (Beam localisation, see [`CdkMol`]).
//!
//! `ExtendedFingerprinter` runs `Fingerprinter(999)` and uses the 25 reserved
//! bits for ring counts (SSSR size, bits 1009-1015) and the size of the largest
//! ring system (bits 1013-1021). Its mass bits (999-1008) compare
//! `MolecularFormulaManipulator.getTotalNaturalAbundance` (a probability <= 1)
//! with 100, 200, ...; they are never set.
//!
//! Not reproduced: CDK throws (no fingerprint) when one start atom yields more
//! than 42000 stored paths; that count depends on JVM identity hash codes. All
//! paths are hashed here. Molecules with pseudo atoms (`*`, skipped by CDK's
//! re-perception) are treated like any other molecule.

use super::cdk_daylight::CdkMol;
use super::cdk_perception::CdkView;
use super::context::CalculationContext;
use super::typed::{BitVector, Descriptor, DescriptorValue};
use crate::mol::elements::ELEMENT_SYMBOLS;
use crate::mol::molecule::Molecule;
use crate::types::BondType;
use bitvec::prelude::*;
use smallvec::SmallVec;

/// Fingerprint size of `Fingerprinter` / `GraphOnlyFingerprinter` (and `ExtendedFingerprinter`).
pub const CDK_PATH_FP_SIZE: usize = 1024;
/// Path bits of `ExtendedFingerprinter` (`size - RESERVED_BITS`).
const EXT_PATH_BITS: i32 = 999;
/// `searchDepth + 1`: maximum number of atoms in a path.
const MAX_PATH_ATOMS: usize = 8;

/// `new java.util.Random(seed).nextInt(bound)`.
#[inline]
fn java_random_next_int(seed: i32, bound: i32) -> usize {
    const MULT: i64 = 0x5_DEEC_E66D;
    const MASK: i64 = (1 << 48) - 1;
    let mut s = (seed as i64 ^ MULT) & MASK;
    let mut next31 = || {
        s = (s.wrapping_mul(MULT).wrapping_add(0xB)) & MASK;
        (s >> 17) as i32
    };
    let r = next31();
    let m = bound - 1;
    if bound & m == 0 {
        return ((bound as i64 * r as i64) >> 31) as usize;
    }
    let mut u = r;
    loop {
        let r = u % bound;
        if u.wrapping_sub(r).wrapping_add(m) >= 0 {
            return r as usize;
        }
        u = next31();
    }
}

/// Java `String.hashCode`.
fn java_string_hash(s: &str) -> i32 {
    s.bytes().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32))
}

/// CDK `Fingerprinter.getAtomSymbol`.
fn atom_symbol(z: u8) -> &'static str {
    match z {
        0 => "*",
        17 => "X",
        35 => "Z",
        14 => "Y",
        33 => "D",
        3 => "L",
        34 => "E",
        11 => "G",
        20 => "J",
        13 => "A",
        _ => ELEMENT_SYMBOLS.get(z as usize).copied().unwrap_or("*"),
    }
}

/// A path token (atom or bond symbol) as seen by the hash: `len` copies of
/// the character `c`. Appending it maps a hash `h` to `h * p + s`, with
/// `p = 31^len` and `s` the hash of the token alone.
#[derive(Clone, Copy, Default)]
struct Tok {
    p: i32,
    s: i32,
    /// Order of the symbol string (Java `String.compareTo`).
    rank: u16,
}

impl Tok {
    fn new(c: u8, len: usize, rank: u16) -> Self {
        let (mut p, mut s) = (1i32, 0i32);
        for _ in 0..len {
            s = s.wrapping_mul(31).wrapping_add(c as i32);
            p = p.wrapping_mul(31);
        }
        Tok { p, s, rank }
    }
}

/// Forward hash, reverse hash and `31^length` of a symbol sequence. The
/// reverse hash is the hash of the reversed character sequence:
/// `sum(c_i * 31^i)`.
#[derive(Clone, Copy)]
struct Hash {
    fwd: i32,
    rev: i32,
    pw: i32,
}

impl Hash {
    /// Appends a token sequence with combined power `p`, forward hash `sf` and
    /// reverse hash `sr`.
    #[inline]
    fn step(self, p: i32, sf: i32, sr: i32) -> Self {
        Hash {
            fwd: self.fwd.wrapping_mul(p).wrapping_add(sf),
            rev: self.rev.wrapping_add(self.pw.wrapping_mul(sr)),
            pw: self.pw.wrapping_mul(p),
        }
    }
}

/// One step of a path: bond to `w`, with the bond + atom tokens combined
/// (`p`, `sf`, `sr`) and the atom token alone (`pa`, `sa`) for the graph-only
/// encoding.
#[derive(Clone, Copy)]
struct Edge {
    w: u32,
    brank: u16,
    arank: u16,
    p: i32,
    sf: i32,
    sr: i32,
    pa: i32,
    sa: i32,
}

/// The three path fingerprints of one molecule.
pub(crate) struct CdkPathFps {
    cdk: BitVector,
    extended: BitVector,
    graph: BitVector,
}

/// Fingerprint bits as plain words while walking.
type Words = [u64; CDK_PATH_FP_SIZE / 64];

#[inline]
fn set_bit(w: &mut Words, bit: usize) {
    w[bit >> 6] |= 1 << (bit & 63);
}

struct Walker<'a> {
    /// CSR adjacency of the walked atoms.
    adj_start: &'a [u32],
    edges: &'a [Edge],
    visited: Vec<bool>,
    /// Symbol ranks along the current path (atoms, bonds).
    arank: [u16; MAX_PATH_ATOMS],
    brank: [u16; MAX_PATH_ATOMS],
    start: u32,
    cdk: Words,
    extended: Words,
    graph: Words,
}

impl Walker<'_> {
    /// Java `compare(apath, bpath)` (forward vs reverse path) for the bond-
    /// labelled and the graph-only (empty bond symbols) encodings at once:
    /// whether the forward hash is used, `(cdk, graph)`.
    #[inline]
    fn forward(&self, len: usize) -> (bool, bool) {
        let (a, b) = (&self.arank, &self.brank);
        let first = a[0].cmp(&a[len - 1]);
        if first.is_ne() {
            return (first.is_gt(), first.is_gt());
        }
        let mut cdk: Option<bool> = None;
        let mut graph: Option<bool> = None;
        let (mut i, mut j) = (1, len - 2);
        while j != 0 && (cdk.is_none() || graph.is_none()) {
            if cdk.is_none() {
                let c = b[i - 1].cmp(&b[j]);
                if c.is_ne() {
                    cdk = Some(c.is_gt());
                }
            }
            let c = a[i].cmp(&a[j]);
            if c.is_ne() {
                cdk.get_or_insert(c.is_gt());
                graph.get_or_insert(c.is_gt());
            }
            i += 1;
            j -= 1;
        }
        (cdk.unwrap_or(true), graph.unwrap_or(true))
    }

    #[inline]
    fn store(&mut self, len: usize, h: Hash, g: Hash) {
        let (fc, fg) = self.forward(len);
        let hc = if fc { h.fwd } else { h.rev };
        let hg = if fg { g.fwd } else { g.rev };
        set_bit(&mut self.cdk, java_random_next_int(hc, CDK_PATH_FP_SIZE as i32));
        set_bit(&mut self.extended, java_random_next_int(hc, EXT_PATH_BITS));
        set_bit(&mut self.graph, java_random_next_int(hg, CDK_PATH_FP_SIZE as i32));
    }

    fn walk(&mut self, a: usize, depth: usize, h: Hash, g: Hash) {
        let edges = self.edges;
        for e in &edges[self.adj_start[a] as usize..self.adj_start[a + 1] as usize] {
            let w = e.w as usize;
            if self.visited[w] {
                continue;
            }
            let h2 = h.step(e.p, e.sf, e.sr);
            let g2 = g.step(e.pa, e.sa, e.sa);
            self.arank[depth] = e.arank;
            self.brank[depth - 1] = e.brank;
            if self.start < e.w {
                self.store(depth + 1, h2, g2);
            }
            if depth + 1 < MAX_PATH_ATOMS {
                self.visited[w] = true;
                self.walk(w, depth + 1, h2, g2);
                self.visited[w] = false;
            }
        }
    }
}

/// Computes the CDK, extended and graph-only path fingerprints.
fn compute(mol: &Molecule, view: &CdkView) -> CdkPathFps {
    let n = mol.num_atoms();

    // atom tokens; hydrogens and pseudo atoms are never visited (skipAtom)
    let skip = |a: usize| matches!(mol.atoms[a].atomic_number, 0 | 1);
    let mut syms: SmallVec<[&'static str; 8]> = SmallVec::new();
    for a in 0..n {
        let s = atom_symbol(mol.atoms[a].atomic_number);
        if !skip(a) && !syms.contains(&s) {
            syms.push(s);
        }
    }
    syms.sort_unstable();
    let atom_tok: Vec<Tok> = (0..n)
        .map(|a| {
            let s = atom_symbol(mol.atoms[a].atomic_number);
            Tok::new(s.as_bytes()[0], s.len(), syms.iter().position(|&x| x == s).unwrap_or(0) as u16)
        })
        .collect();

    // bond tokens: legacy-aromatic bonds ':', others by Kekulé order
    let exact = exact_legacy_bonds(mol, view);
    let arom_bond: &[bool] = exact.as_deref().unwrap_or(&view.arom_bond);
    let beam = mol
        .bonds
        .iter()
        .enumerate()
        .any(|(bi, b)| b.bond_type == BondType::Aromatic && !arom_bond[bi])
        .then(|| CdkMol::new(mol));
    let bond_tok: Vec<Tok> = (0..mol.num_bonds())
        .map(|bi| {
            let c = if arom_bond[bi] {
                b':'
            } else {
                let order = match &beam {
                    Some(m) if mol.bonds[bi].bond_type == BondType::Aromatic => m.order[bi],
                    _ => view.order[bi],
                };
                match order {
                    1 => b'-',
                    2 => b'=',
                    3 => b'#',
                    _ => 0,
                }
            };
            Tok::new(c, (c != 0) as usize, c as u16)
        })
        .collect();

    let mut adj_start = Vec::with_capacity(n + 1);
    let mut edges = Vec::with_capacity(2 * mol.num_bonds());
    for a in 0..n {
        adj_start.push(edges.len() as u32);
        if skip(a) {
            continue;
        }
        for &bi in &mol.adjacency[a] {
            let b = &mol.bonds[bi as usize];
            let w = if b.source as usize == a { b.target } else { b.source } as usize;
            if skip(w) {
                continue;
            }
            let (bt, at) = (bond_tok[bi as usize], atom_tok[w]);
            edges.push(Edge {
                w: w as u32,
                brank: bt.rank,
                arank: at.rank,
                p: bt.p.wrapping_mul(at.p),
                sf: bt.s.wrapping_mul(at.p).wrapping_add(at.s),
                sr: bt.s.wrapping_add(bt.p.wrapping_mul(at.s)),
                pa: at.p,
                sa: at.s,
            });
        }
    }
    adj_start.push(edges.len() as u32);

    let mut walker = Walker {
        adj_start: &adj_start,
        edges: &edges,
        visited: vec![false; n],
        arank: [0; MAX_PATH_ATOMS],
        brank: [0; MAX_PATH_ATOMS],
        start: 0,
        cdk: [0; CDK_PATH_FP_SIZE / 64],
        extended: [0; CDK_PATH_FP_SIZE / 64],
        graph: [0; CDK_PATH_FP_SIZE / 64],
    };
    let empty = Hash { fwd: 0, rev: 0, pw: 1 };
    for a in 0..n {
        if skip(a) {
            continue;
        }
        // single-atom path: String.hashCode of the symbol
        let x = java_string_hash(atom_symbol(mol.atoms[a].atomic_number));
        set_bit(&mut walker.cdk, java_random_next_int(x, CDK_PATH_FP_SIZE as i32));
        set_bit(&mut walker.extended, java_random_next_int(x, EXT_PATH_BITS));
        set_bit(&mut walker.graph, java_random_next_int(x, CDK_PATH_FP_SIZE as i32));
        let t = atom_tok[a];
        let h = empty.step(t.p, t.s, t.s);
        walker.start = a as u32;
        walker.arank[0] = atom_tok[a].rank;
        walker.visited[a] = true;
        walker.walk(a, 1, h, h);
        walker.visited[a] = false;
    }
    let to_bits = |w: &Words| {
        let mut bv = bitvec![u8, Lsb0; 0; CDK_PATH_FP_SIZE];
        for (i, &x) in w.iter().enumerate() {
            let mut x = x;
            while x != 0 {
                bv.set(i * 64 + x.trailing_zeros() as usize, true);
                x &= x - 1;
            }
        }
        bv
    };
    let mut out = CdkPathFps {
        cdk: to_bits(&walker.cdk),
        extended: to_bits(&walker.extended),
        graph: to_bits(&walker.graph),
    };
    set_ring_bits(mol, &mut out.extended);
    out
}

/// `Aromaticity.cdkLegacy()` bond flags with CDK's own minimum cycle basis.
///
/// `Cycles.cdkAromaticSet()` checks all cycles of a fused ring system with at
/// most three rings and only the MCB (`MinimumCycleBasis`) of larger systems.
/// When equally short cycles compete for the MCB (cages such as fullerenes),
/// which one CDK keeps depends on its `InitialCycles` order (CDK bond order,
/// shortest-path discovery order); [`CdkView`] picks some valid MCB. This
/// recomputes the flags of such systems with the port of CDK's MCB in
/// `pubchem_cdk` on CDK's graph (`GraphUtil.toAdjList` neighbour order, the
/// subgraph of atoms with an electron contribution, `RingSearch.fused`
/// vertex order). `None` when no fused system has more than three rings.
fn exact_legacy_bonds(mol: &Molecule, view: &CdkView) -> Option<Vec<bool>> {
    let n = mol.num_atoms();
    let in_sub = |b: &crate::mol::bond::Bond, e: &[i32]| e[b.source as usize] >= 0 && e[b.target as usize] >= 0;
    // cheap screen: a block with cycle rank > 3 needs a ring-bond component with rank > 3
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let mut rank = vec![0i32; n];
    for b in &mol.bonds {
        if b.is_in_ring() {
            let (x, y) = (find(&mut parent, b.source as u32), find(&mut parent, b.target as u32));
            if x == y {
                rank[x as usize] += 1;
            } else {
                parent[x as usize] = y;
                rank[y as usize] += rank[x as usize];
            }
        }
    }
    if rank.iter().all(|&r| r <= 3) {
        return None;
    }
    let e = view.electrons(mol);
    // CDK adjacency: neighbours in container bond order (larger atom index, then creation)
    let mut adj: Vec<SmallVec<[(u32, u32); 4]>> = vec![SmallVec::new(); n];
    for (bi, b) in mol.bonds.iter().enumerate() {
        if in_sub(b, &e) {
            adj[b.source as usize].push((b.target as u32, bi as u32));
            adj[b.target as usize].push((b.source as u32, bi as u32));
        }
    }
    for (a, v) in adj.iter_mut().enumerate() {
        v.sort_unstable_by_key(|&(w, bi)| ((w as usize).max(a), bi));
    }
    let mut arom = view.arom_bond.clone();
    let mut changed = false;
    let mut local = vec![u32::MAX; n];
    for block in biconnected_blocks(&adj) {
        let mut verts: Vec<u32> = block
            .iter()
            .flat_map(|&b| [mol.bonds[b as usize].source as u32, mol.bonds[b as usize].target as u32])
            .collect();
        verts.sort_unstable();
        verts.dedup();
        if block.len() + 1 - verts.len() <= 3 {
            continue; // at most three rings: all cycles, as CdkView does
        }
        for (i, &v) in verts.iter().enumerate() {
            local[v as usize] = i as u32;
        }
        let g: Vec<Vec<u32>> = verts
            .iter()
            .map(|&v| {
                adj[v as usize]
                    .iter()
                    .filter(|&&(w, _)| local[w as usize] != u32::MAX)
                    .map(|&(w, _)| local[w as usize])
                    .collect()
            })
            .collect();
        for &b in &block {
            arom[b as usize] = false;
        }
        for cyc in super::pubchem_cdk::minimum_cycle_basis(&g) {
            let walk = &cyc[..cyc.len() - 1];
            let sum: i32 = walk.iter().map(|&v| e[verts[v as usize] as usize]).sum();
            if (sum - 2) % 4 != 0 {
                continue;
            }
            for w in cyc.windows(2) {
                let (u, v) = (verts[w[0] as usize], verts[w[1] as usize]);
                if let Some(&(_, bi)) = adj[u as usize].iter().find(|&&(x, _)| x == v) {
                    arom[bi as usize] = true;
                }
            }
        }
        for &v in &verts {
            local[v as usize] = u32::MAX;
        }
        changed = true;
    }
    changed.then_some(arom)
}

/// Biconnected components (bond lists) of the graph `adj` (iterative Tarjan).
fn biconnected_blocks(adj: &[SmallVec<[(u32, u32); 4]>]) -> Vec<Vec<u32>> {
    let n = adj.len();
    let mut disc = vec![u32::MAX; n];
    let mut low = vec![0u32; n];
    let mut time = 0u32;
    let mut blocks = Vec::new();
    let mut estack: Vec<u32> = Vec::new();
    for root in 0..n {
        if disc[root] != u32::MAX || adj[root].is_empty() {
            continue;
        }
        let mut st: Vec<(usize, u32, usize)> = vec![(root, u32::MAX, 0)];
        disc[root] = time;
        low[root] = time;
        time += 1;
        while let Some(&mut (v, pb, ref mut idx)) = st.last_mut() {
            if *idx < adj[v].len() {
                let (w, b) = adj[v][*idx];
                *idx += 1;
                let w = w as usize;
                if b == pb {
                    continue;
                }
                if disc[w] == u32::MAX {
                    estack.push(b);
                    disc[w] = time;
                    low[w] = time;
                    time += 1;
                    st.push((w, b, 0));
                } else if disc[w] < disc[v] {
                    estack.push(b);
                    low[v] = low[v].min(disc[w]);
                }
            } else {
                st.pop();
                if let Some(&(u, _, _)) = st.last() {
                    low[u] = low[u].min(low[v]);
                    if low[v] >= disc[u] {
                        let mut comp = Vec::new();
                        while let Some(e) = estack.pop() {
                            comp.push(e);
                            if e == pb {
                                break;
                            }
                        }
                        blocks.push(comp);
                    }
                }
            }
        }
    }
    blocks
}

/// `ExtendedFingerprinter` ring bits: SSSR size (`Cycles.sssr`, i.e. the
/// cycle rank) and the largest number of SSSR rings in one ring system
/// (`RingPartitioner`: rings sharing an atom), i.e. the largest cycle rank of a
/// connected component of the ring bonds.
fn set_ring_bits(mol: &Molecule, fp: &mut BitVector) {
    let n = mol.num_atoms();
    // union-find over ring bonds
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let mut ring_bonds = 0usize;
    for b in &mol.bonds {
        if b.is_in_ring() {
            ring_bonds += 1;
            let (x, y) = (find(&mut parent, b.source as u32), find(&mut parent, b.target as u32));
            if x != y {
                parent[x as usize] = y;
            }
        }
    }
    if ring_bonds == 0 {
        return;
    }
    // per component: ring bonds - ring atoms + 1
    let mut edges = vec![0i32; n];
    let mut verts = vec![0i32; n];
    for b in &mol.bonds {
        if b.is_in_ring() {
            let r = find(&mut parent, b.source as u32);
            edges[r as usize] += 1;
        }
    }
    for a in 0..n {
        if mol.atoms[a].is_in_ring() {
            let r = find(&mut parent, a as u32);
            verts[r as usize] += 1;
        }
    }
    let mut total = 0i32;
    let mut max_system = 0i32;
    for r in 0..n {
        if edges[r] > 0 {
            let rank = edges[r] - verts[r] + 1;
            total += rank;
            max_system = max_system.max(rank);
        }
    }
    let size = CDK_PATH_FP_SIZE;
    for i in 0..7 {
        if total > i as i32 {
            fp.set(size - 15 + i, true);
        }
    }
    for i in 0..(max_system.max(0) as usize).min(9) {
        fp.set(size - 8 + i - 3, true);
    }
}

/// The cached path fingerprints of `mol` (computed once for all three descriptors).
fn with_fps<R>(mol: &Molecule, ctx: &mut CalculationContext, f: impl FnOnce(&CdkPathFps) -> R) -> R {
    let key = super::context::molecule_cache_key(mol);
    let fps = match ctx.cdk_path_fps.take() {
        Some((k, fps)) if k == key => fps,
        _ => Box::new(CdkView::with_legacy_cached(mol, ctx, |view, _| compute(mol, view))),
    };
    let r = f(&fps);
    ctx.cdk_path_fps = Some((key, fps));
    r
}

/// CDK path fingerprint (CDK 2.13 `Fingerprinter`, 1024 bits, depth 7).
///
/// # Examples
///
/// ```
/// use ferrum_core::desc::cdk_path_fingerprints::CdkFingerprint;
/// use ferrum_core::desc::typed::Descriptor;
/// use ferrum_core::io::smiles::parse_smiles;
///
/// let mol = parse_smiles("c1ccccc1O").unwrap();
/// let fp = CdkFingerprint.calculate(&mol, &mut ferrum_core::desc::context::CalculationContext::new());
/// assert_eq!(fp.as_binary().len(), 1024);
/// ```
pub struct CdkFingerprint;

impl Descriptor for CdkFingerprint {
    fn name(&self) -> &str {
        "CdkFingerprint"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::Binary(with_fps(mol, ctx, |f| f.cdk.clone()))
    }
}

/// CDK extended path fingerprint (CDK 2.13 `ExtendedFingerprinter`, 1024 bits
/// including 25 ring/mass bits).
pub struct CdkExtendedFingerprint;

impl Descriptor for CdkExtendedFingerprint {
    fn name(&self) -> &str {
        "CdkExtendedFingerprint"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::Binary(with_fps(mol, ctx, |f| f.extended.clone()))
    }
}

/// CDK graph-only path fingerprint (CDK 2.13 `GraphOnlyFingerprinter`: no bond
/// orders, 1024 bits).
pub struct CdkGraphOnlyFingerprint;

impl Descriptor for CdkGraphOnlyFingerprint {
    fn name(&self) -> &str {
        "CdkGraphOnlyFingerprint"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::Binary(with_fps(mol, ctx, |f| f.graph.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_random_matches_jdk() {
        // (seed, new Random(seed).nextInt(1024), new Random(seed).nextInt(999)) from the JDK
        for (seed, a, b) in [
            (67, 742, 949),
            (-123456789, 343, 742),
            (2147483647, 634, 270),
            (0, 748, 672),
            (1234567, 248, 948),
        ] {
            assert_eq!(java_random_next_int(seed, 1024), a);
            assert_eq!(java_random_next_int(seed, 999), b);
        }
        assert_eq!(java_string_hash("Sn"), 'S' as i32 * 31 + 'n' as i32);
    }
}
