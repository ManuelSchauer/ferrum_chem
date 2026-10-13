//! Ring perception: a faithful port of RDKit's `findSSSR` + `symmetrizeSSSR`
//! (`Code/GraphMol/FindRings.cpp`).
//!
//! RDKit's aromaticity model, ring counts, ring-membership SMARTS primitives and
//! many descriptors are defined on the *symmetrized* SSSR, so FerrumChem uses
//! the exact same algorithm (including its traversal order) to get identical
//! ring sets.
//!
//! Atom rings are returned in RDKit's traversal order (consecutive atoms are
//! bonded, last atom bonds to the first).

use crate::mol::molecule::Molecule;
use crate::types::BondType;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const MAX_BFSQ_SIZE: usize = 200_000;

type Ring = Vec<i32>;

/// Ring invariant: the set of atoms in a ring, as a bitset.
///
/// Ordering mirrors `boost::dynamic_bitset::operator<` (compares as an unsigned
/// integer, most significant bit first), which determines the iteration order of
/// RDKit's `std::map<RINGINVAR, ...>`.
#[derive(Clone, PartialEq, Eq, Hash)]
struct RingInvar(Vec<u64>);

impl RingInvar {
    fn new(ring: &[i32], n_atoms: usize) -> Self {
        let mut words = vec![0u64; n_atoms.div_ceil(64).max(1)];
        for &a in ring {
            words[a as usize / 64] |= 1u64 << (a as usize % 64);
        }
        Self(words)
    }
}

impl Ord for RingInvar {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
}

impl PartialOrd for RingInvar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Bonds that do not take part in ring perception (RDKit default: dative bonds).
#[inline]
fn bond_is_ring_candidate(bond_type: BondType) -> bool {
    !matches!(bond_type, BondType::Dative)
}

#[inline]
fn other_atom(mol: &Molecule, bond: usize, atom: i32) -> i32 {
    let b = &mol.bonds[bond];
    if b.source as i32 == atom { b.target as i32 } else { b.source as i32 }
}

fn bond_between(mol: &Molecule, a: i32, b: i32) -> Option<usize> {
    mol.adjacency[a as usize]
        .iter()
        .map(|&bi| bi as usize)
        .find(|&bi| other_atom(mol, bi, a) == b)
}

/// Converts an atom ring (traversal order) to its bond indices.
pub(crate) fn ring_atoms_to_bonds(mol: &Molecule, ring: &[i32]) -> Vec<i32> {
    let n = ring.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        out.push(bond_between(mol, a, b).expect("ring atoms must be bonded") as i32);
    }
    out
}

struct BfsWorkspace {
    parents: Vec<i32>,
    depths: Vec<u32>,
    done: Vec<u8>,
}

impl BfsWorkspace {
    fn new() -> Self {
        Self { parents: Vec::new(), depths: Vec::new(), done: Vec::new() }
    }

    /// Port of `BFSWorkspace::smallestRingsBfs`.
    fn smallest_rings_bfs(
        &mut self,
        mol: &Molecule,
        root: i32,
        rings: &mut Vec<Ring>,
        active_bonds: &[bool],
        forbidden: Option<&[i32]>,
    ) -> Result<usize, ()> {
        const WHITE: u8 = 0;
        const GRAY: u8 = 1;
        const BLACK: u8 = 2;
        let n = mol.num_atoms();
        self.done.clear();
        self.done.resize(n, WHITE);
        if let Some(f) = forbidden {
            for &i in f {
                self.done[i as usize] = BLACK;
            }
        }
        self.parents.clear();
        self.parents.resize(n, -1);
        self.depths.clear();
        self.depths.resize(n, 0);

        let mut bfsq: VecDeque<i32> = VecDeque::new();
        bfsq.push_back(root);
        let mut cur_size = u32::MAX;
        while let Some(curr) = bfsq.pop_front() {
            if bfsq.len() + 1 >= MAX_BFSQ_SIZE {
                return Err(());
            }
            self.done[curr as usize] = BLACK;
            let depth = self.depths[curr as usize] + 1;
            if depth > cur_size {
                break;
            }
            for &bi in &mol.adjacency[curr as usize] {
                let bi = bi as usize;
                if !active_bonds[bi] {
                    continue;
                }
                let nbr = other_atom(mol, bi, curr);
                if self.done[nbr as usize] == BLACK || self.parents[curr as usize] == nbr {
                    continue;
                }
                if self.done[nbr as usize] == WHITE {
                    self.parents[nbr as usize] = curr;
                    self.done[nbr as usize] = GRAY;
                    self.depths[nbr as usize] = depth;
                    bfsq.push_back(nbr);
                } else {
                    // stitch together the two paths
                    let mut ring: Ring = vec![nbr];
                    let mut parent = self.parents[nbr as usize];
                    while parent != -1 && parent != root {
                        ring.push(parent);
                        parent = self.parents[parent as usize];
                    }
                    ring.insert(0, curr);
                    parent = self.parents[curr as usize];
                    while parent != -1 {
                        if ring.contains(&parent) {
                            ring.clear();
                            break;
                        }
                        ring.insert(0, parent);
                        parent = self.parents[parent as usize];
                    }
                    if ring.len() > 1 {
                        if ring.len() as u32 <= cur_size {
                            cur_size = ring.len() as u32;
                            rings.push(ring);
                        } else {
                            return Ok(rings.len());
                        }
                    }
                }
            }
        }
        Ok(rings.len())
    }
}

fn trim_bonds(
    mol: &Molecule,
    cand: i32,
    changed: &mut BTreeSet<i32>,
    atom_degrees: &mut [i32],
    active_bonds: &mut [bool],
) {
    for &bi in &mol.adjacency[cand as usize] {
        let bi = bi as usize;
        if !active_bonds[bi] {
            continue;
        }
        let o = other_atom(mol, bi, cand);
        if atom_degrees[o as usize] <= 2 {
            changed.insert(o);
        }
        active_bonds[bi] = false;
        atom_degrees[o as usize] -= 1;
        atom_degrees[cand as usize] -= 1;
    }
}

fn mark_useless_d2s(
    mol: &Molecule,
    root: i32,
    forb: &mut [bool],
    atom_degrees: &[i32],
    active_bonds: &[bool],
) {
    for &bi in &mol.adjacency[root as usize] {
        let bi = bi as usize;
        if !active_bonds[bi] {
            continue;
        }
        let o = other_atom(mol, bi, root);
        if !forb[o as usize] && atom_degrees[o as usize] == 2 {
            forb[o as usize] = true;
            mark_useless_d2s(mol, o, forb, atom_degrees, active_bonds);
        }
    }
}

fn pick_d2_nodes(
    mol: &Molecule,
    curr_frag: &[i32],
    atom_degrees: &[i32],
    active_bonds: &[bool],
) -> Vec<i32> {
    let mut d2nodes = Vec::new();
    let mut forb = vec![false; mol.num_atoms()];
    loop {
        let mut root = -1;
        for &a in curr_frag {
            if atom_degrees[a as usize] == 2 && !forb[a as usize] {
                root = a;
                d2nodes.push(a);
                forb[a as usize] = true;
                break;
            }
        }
        if root == -1 {
            break;
        }
        mark_useless_d2s(mol, root, &mut forb, atom_degrees, active_bonds);
    }
    d2nodes
}

struct Ctx<'a> {
    mol: &'a Molecule,
    invars: BTreeSet<RingInvar>,
    ring_bonds: Vec<bool>,
    ring_atoms: Vec<bool>,
    bfs: BfsWorkspace,
}

impl<'a> Ctx<'a> {
    fn invar(&self, ring: &[i32]) -> RingInvar {
        RingInvar::new(ring, self.mol.num_atoms())
    }

    fn mark_ring(&mut self, ring: &[i32]) {
        let n = ring.len();
        for i in 0..n - 1 {
            let b = bond_between(self.mol, ring[i], ring[i + 1]).unwrap();
            self.ring_bonds[b] = true;
            self.ring_atoms[ring[i] as usize] = true;
        }
        let b = bond_between(self.mol, ring[0], ring[n - 1]).unwrap();
        self.ring_bonds[b] = true;
        self.ring_atoms[ring[n - 1] as usize] = true;
    }

    #[allow(clippy::too_many_arguments)]
    fn find_rings_d2_nodes(
        &mut self,
        res: &mut Vec<Ring>,
        d2nodes: &[i32],
        atom_degrees: &mut [i32],
        active_bonds: &mut [bool],
    ) -> Result<(), ()> {
        let mut dup_d2_cands: BTreeMap<RingInvar, Vec<i32>> = BTreeMap::new();
        let mut dup_map: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
        let mut node_invars: BTreeMap<i32, Vec<RingInvar>> = BTreeMap::new();
        for &cand in d2nodes {
            let mut srings = Vec::new();
            self.bfs.smallest_rings_bfs(self.mol, cand, &mut srings, active_bonds, None)?;
            for nring in &srings {
                let invr = self.invar(nring);
                if !self.invars.contains(&invr) {
                    res.push(nring.clone());
                    self.invars.insert(invr.clone());
                    self.mark_ring(nring);
                }
                node_invars.entry(cand).or_default().push(invr.clone());
                let mut new_dups = Vec::new();
                for (&node, invs) in &node_invars {
                    if node != cand && invs.contains(&invr) {
                        new_dups.push(node);
                    }
                }
                for node in new_dups {
                    dup_map.entry(cand).or_default().push(node);
                    dup_map.entry(node).or_default().push(cand);
                }
                dup_d2_cands.entry(invr).or_default().push(cand);
            }
            if srings.is_empty() {
                let mut changed = BTreeSet::new();
                changed.insert(cand);
                while let Some(c) = changed.pop_first() {
                    trim_bonds(self.mol, c, &mut changed, atom_degrees, active_bonds);
                }
            }
        }
        self.find_sssr_for_dup_cands(res, &dup_map, &dup_d2_cands, atom_degrees, active_bonds)
    }

    fn find_sssr_for_dup_cands(
        &mut self,
        res: &mut Vec<Ring>,
        dup_map: &BTreeMap<i32, Vec<i32>>,
        dup_d2_cands: &BTreeMap<RingInvar, Vec<i32>>,
        atom_degrees: &[i32],
        active_bonds: &[bool],
    ) -> Result<(), ()> {
        for dup_cands in dup_d2_cands.values() {
            if dup_cands.len() <= 1 {
                continue;
            }
            let mut nrings: Vec<Ring> = Vec::new();
            let mut min_siz = usize::MAX;
            for &dup_cand in dup_cands {
                let mut degrees_copy = atom_degrees.to_vec();
                let mut active_copy = active_bonds.to_vec();
                let mut changed = BTreeSet::new();
                for &dni in dup_map.get(&dup_cand).expect("duplicate could not be found") {
                    trim_bonds(self.mol, dni, &mut changed, &mut degrees_copy, &mut active_copy);
                }
                let mut srings = Vec::new();
                self.bfs.smallest_rings_bfs(self.mol, dup_cand, &mut srings, &active_copy, None)?;
                for r in srings {
                    min_siz = min_siz.min(r.len());
                    nrings.push(r);
                }
            }
            for nring in nrings {
                if nring.len() == min_siz {
                    let invr = self.invar(&nring);
                    if !self.invars.contains(&invr) {
                        res.push(nring);
                        self.invars.insert(invr);
                    }
                }
            }
        }
        Ok(())
    }

    fn push_new_rings(&mut self, res: &mut Vec<Ring>, rings: Vec<Ring>) {
        for nring in rings {
            let invr = self.invar(&nring);
            if !self.invars.contains(&invr) {
                res.push(nring);
                self.invars.insert(invr);
            }
        }
    }

    fn find_rings_d3_node(
        &mut self,
        res: &mut Vec<Ring>,
        cand: i32,
        active_bonds: &[bool],
    ) -> Result<(), ()> {
        let mut srings = Vec::new();
        let nsmall = self.bfs.smallest_rings_bfs(self.mol, cand, &mut srings, active_bonds, None)?;
        self.push_new_rings(res, srings.clone());
        if nsmall >= 3 {
            return Ok(());
        }
        let mut nbrs = mol_active_neighbors(self.mol, cand, active_bonds);
        if nbrs.len() < 3 {
            return Ok(());
        }
        nbrs.truncate(3);
        let (n1, n2, n3) = (nbrs[0], nbrs[1], nbrs[2]);
        if nsmall == 2 {
            let in_both = |n: i32| srings[0].contains(&n) && srings[1].contains(&n);
            let f = if in_both(n1) {
                n1
            } else if in_both(n2) {
                n2
            } else if in_both(n3) {
                n3
            } else {
                return Ok(());
            };
            let mut trings = Vec::new();
            self.bfs.smallest_rings_bfs(self.mol, cand, &mut trings, active_bonds, Some(&[f]))?;
            self.push_new_rings(res, trings);
        }
        if nsmall == 1 {
            let (f1, f2) = if !srings[0].contains(&n1) {
                (n2, n3)
            } else if !srings[0].contains(&n2) {
                (n1, n3)
            } else if !srings[0].contains(&n3) {
                (n1, n2)
            } else {
                return Ok(());
            };
            let mut trings = Vec::new();
            self.bfs.smallest_rings_bfs(self.mol, cand, &mut trings, active_bonds, Some(&[f2]))?;
            self.push_new_rings(res, trings);
            let mut trings = Vec::new();
            self.bfs.smallest_rings_bfs(self.mol, cand, &mut trings, active_bonds, Some(&[f1]))?;
            self.push_new_rings(res, trings);
        }
        Ok(())
    }

    fn atom_search_bfs(&self, start: i32, end: i32) -> Result<Option<Ring>, ()> {
        let mut bfsq: VecDeque<Ring> = VecDeque::new();
        bfsq.push_back(vec![start]);
        while let Some(tv) = bfsq.pop_front() {
            if bfsq.len() + 1 >= MAX_BFSQ_SIZE {
                return Err(());
            }
            let curr = *tv.last().unwrap();
            for &bi in &self.mol.adjacency[curr as usize] {
                let nbr = other_atom(self.mol, bi as usize, curr);
                if nbr == end {
                    if curr != start {
                        let mut nv = tv.clone();
                        nv.push(nbr);
                        if !self.invars.contains(&self.invar(&nv)) {
                            return Ok(Some(nv));
                        }
                    }
                } else if self.ring_atoms[nbr as usize] && !tv.contains(&nbr) {
                    let mut nv = tv.clone();
                    nv.push(nbr);
                    bfsq.push_back(nv);
                }
            }
        }
        Ok(None)
    }

    fn find_ring_connecting_atoms(&mut self, bond: usize, res: &mut Vec<Ring>) -> Result<bool, ()> {
        let b = &self.mol.bonds[bond];
        let (s, e) = (b.source as i32, b.target as i32);
        match self.atom_search_bfs(s, e)? {
            Some(nring) => {
                let invr = self.invar(&nring);
                if !self.invars.contains(&invr) {
                    self.mark_ring(&nring);
                    res.push(nring);
                    self.invars.insert(invr);
                }
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

fn mol_active_neighbors(mol: &Molecule, atom: i32, active_bonds: &[bool]) -> Vec<i32> {
    mol.adjacency[atom as usize]
        .iter()
        .filter(|&&bi| active_bonds[bi as usize])
        .map(|&bi| other_atom(mol, bi as usize, atom))
        .collect()
}

/// Port of `FindRings::removeExtraRings`; returns `(kept, extras)`.
fn remove_extra_rings(mol: &Molecule, mut res: Vec<Ring>) -> (Vec<Ring>, Vec<Ring>) {
    // RDKit uses std::sort here, which is not stable: for more than ~16
    // candidate rings the order of equally sized rings (and thus which of them
    // is kept) depends on the C++ standard library RDKit was built with. A
    // stable sort reproduces the small-range (insertion sort) behaviour.
    res.sort_by_key(|r| r.len());
    let nb = mol.num_bonds();
    let brings: Vec<Vec<i32>> = res.iter().map(|r| ring_atoms_to_bonds(mol, r)).collect();
    let bit_brings: Vec<Vec<bool>> = brings
        .iter()
        .map(|br| {
            let mut v = vec![false; nb];
            for &b in br {
                v[b as usize] = true;
            }
            v
        })
        .collect();
    let is_subset = |a: &[bool], b: &[bool]| a.iter().zip(b).all(|(&x, &y)| !x || y);
    let n = res.len();
    let mut avail = vec![true; n];
    let mut keep = vec![false; n];
    let mut munion = vec![false; nb];
    for i in 0..n {
        if is_subset(&bit_brings[i], &munion) {
            avail[i] = false;
        }
        if !avail[i] {
            continue;
        }
        for b in 0..nb {
            munion[b] |= bit_brings[i][b];
        }
        keep[i] = true;
        let mut consider = vec![false; n];
        for j in i + 1..n {
            if avail[j] && brings[j].len() == brings[i].len() {
                consider[j] = true;
            }
        }
        while consider.iter().any(|&c| c) {
            let mut best_j = i + 1;
            let mut best_overlap: i64 = -1;
            let mut j = i + 1;
            while j < n && brings[j].len() == brings[i].len() {
                if consider[j] && avail[j] {
                    let overlap =
                        bit_brings[j].iter().zip(&munion).filter(|(x, y)| **x && **y).count() as i64;
                    if overlap > best_overlap {
                        best_overlap = overlap;
                        best_j = j;
                    }
                }
                j += 1;
            }
            consider[best_j] = false;
            if is_subset(&bit_brings[best_j], &munion) {
                avail[best_j] = false;
            } else {
                keep[best_j] = true;
                avail[best_j] = false;
                for b in 0..nb {
                    munion[b] |= bit_brings[best_j][b];
                }
            }
        }
    }
    let mut kept = Vec::new();
    let mut extras = Vec::new();
    for (i, r) in res.into_iter().enumerate() {
        if keep[i] {
            kept.push(r);
        } else {
            extras.push(r);
        }
    }
    (kept, extras)
}

/// Connected components (all bonds), atoms ascending, fragments ordered by
/// their lowest atom index (RDKit `getMolFrags`).
fn mol_frags(mol: &Molecule) -> Vec<Vec<i32>> {
    let n = mol.num_atoms();
    let mut label = vec![usize::MAX; n];
    let mut nlab = 0;
    for start in 0..n {
        if label[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        label[start] = nlab;
        while let Some(a) = stack.pop() {
            for &bi in &mol.adjacency[a] {
                let o = other_atom(mol, bi as usize, a as i32) as usize;
                if label[o] == usize::MAX {
                    label[o] = nlab;
                    stack.push(o);
                }
            }
        }
        nlab += 1;
    }
    let mut frags = vec![Vec::new(); nlab];
    for (a, &l) in label.iter().enumerate() {
        frags[l].push(a as i32);
    }
    frags
}

/// Port of RDKit's `fastFindRings` (DFS ring closure), used as RDKit's fallback
/// when the SSSR search cannot find the expected number of rings.
fn fast_find_rings(mol: &Molecule) -> Vec<Ring> {
    fn dfs(mol: &Molecule, atom: i32, colors: &mut [u8], order: &mut Vec<i32>, res: &mut Vec<Ring>, from: i32) {
        colors[atom as usize] = 1;
        order.push(atom);
        for &bi in &mol.adjacency[atom as usize] {
            let nbr = other_atom(mol, bi as usize, atom);
            if colors[nbr as usize] == 0 {
                if mol.adjacency[nbr as usize].len() < 2 {
                    colors[nbr as usize] = 2;
                } else {
                    dfs(mol, nbr, colors, order, res, atom);
                }
            } else if colors[nbr as usize] == 1 && from >= 0 && nbr != from {
                let mut cycle = Vec::new();
                let pos = order.iter().rposition(|&x| x == atom).unwrap();
                for k in (0..=pos).rev() {
                    if order[k] == nbr {
                        break;
                    }
                    cycle.push(order[k]);
                }
                cycle.push(nbr);
                res.push(cycle);
            }
        }
        colors[atom as usize] = 2;
        order.pop();
    }
    let n = mol.num_atoms();
    let mut colors = vec![0u8; n];
    let mut res = Vec::new();
    for i in 0..n {
        if colors[i] != 0 {
            continue;
        }
        if mol.adjacency[i].len() < 2 {
            colors[i] = 2;
            continue;
        }
        let mut order = Vec::new();
        dfs(mol, i as i32, &mut colors, &mut order, &mut res, -1);
    }
    res
}

/// RDKit `findSSSR`: returns `(sssr, extra_rings)`.
fn find_sssr(mol: &Molecule) -> (Vec<Ring>, Vec<Ring>) {
    let nats = mol.num_atoms();
    let nbnds_total = mol.num_bonds();
    let mut active_bonds: Vec<bool> =
        mol.bonds.iter().map(|b| bond_is_ring_candidate(b.bond_type)).collect();
    let mut atom_degrees: Vec<i32> = vec![0; nats];
    let mut atom_degrees_all: Vec<i32> = vec![0; nats];
    for i in 0..nats {
        atom_degrees_all[i] = mol.adjacency[i].len() as i32;
        atom_degrees[i] = mol.adjacency[i]
            .iter()
            .filter(|&&bi| bond_is_ring_candidate(mol.bonds[bi as usize].bond_type))
            .count() as i32;
    }
    let mut ctx = Ctx {
        mol,
        invars: BTreeSet::new(),
        ring_bonds: vec![false; nbnds_total],
        ring_atoms: vec![false; nats],
        bfs: BfsWorkspace::new(),
    };
    let mut res: Vec<Ring> = Vec::new();
    let mut extras_all: Vec<Ring> = Vec::new();

    for cur_frag in mol_frags(mol) {
        if cur_frag.len() < 3 {
            continue;
        }
        let mut frag_res: Vec<Ring> = Vec::new();
        let mut changed: BTreeSet<i32> = BTreeSet::new();
        let mut bndcnt_all = 0i32;
        let mut nbnds = 0i32;
        for &a in &cur_frag {
            bndcnt_all += atom_degrees_all[a as usize];
            let deg = atom_degrees[a as usize];
            nbnds += deg;
            if deg < 2 {
                changed.insert(a);
            }
        }
        bndcnt_all /= 2;
        if bndcnt_all - cur_frag.len() as i32 + 1 < 1 {
            continue;
        }
        nbnds /= 2;

        let outcome: Result<(), ()> = (|| {
            let mut done_ats = vec![false; nats];
            let mut n_done = 0usize;
            while n_done < cur_frag.len() {
                while let Some(cand) = changed.pop_first() {
                    if !done_ats[cand as usize] {
                        done_ats[cand as usize] = true;
                        n_done += 1;
                        trim_bonds(mol, cand, &mut changed, &mut atom_degrees, &mut active_bonds);
                    }
                }
                let d2nodes = pick_d2_nodes(mol, &cur_frag, &atom_degrees, &active_bonds);
                if !d2nodes.is_empty() {
                    ctx.find_rings_d2_nodes(&mut frag_res, &d2nodes, &mut atom_degrees, &mut active_bonds)?;
                    for &d in &d2nodes {
                        done_ats[d as usize] = true;
                        n_done += 1;
                        trim_bonds(mol, d, &mut changed, &mut atom_degrees, &mut active_bonds);
                    }
                } else if n_done < cur_frag.len() {
                    let cand = cur_frag.iter().copied().find(|&a| atom_degrees[a as usize] == 3);
                    let Some(cand) = cand else { break };
                    ctx.find_rings_d3_node(&mut frag_res, cand, &active_bonds)?;
                    done_ats[cand as usize] = true;
                    n_done += 1;
                    trim_bonds(mol, cand, &mut changed, &mut atom_degrees, &mut active_bonds);
                }
            }
            Ok(())
        })();

        let nexpt = (nbnds - cur_frag.len() as i32 + 1) as usize;
        let mut fallback = outcome.is_err();
        if !fallback && frag_res.len() < nexpt {
            // Issue 3514824. NOTE: RDKit bounds the bond index by the fragment's
            // bond count here; mirrored on purpose.
            let first_possible = |ctx: &Ctx, dead: &[bool]| -> Option<usize> {
                (0..nbnds as usize).find(|&i| {
                    i < nbnds_total
                        && !ctx.ring_bonds[i]
                        && !dead[i]
                        && ctx.ring_atoms[mol.bonds[i].source as usize]
                        && ctx.ring_atoms[mol.bonds[i].target as usize]
                })
            };
            let mut dead = vec![false; nbnds_total];
            let mut possible = first_possible(&ctx, &dead);
            while let Some(b) = possible {
                match ctx.find_ring_connecting_atoms(b, &mut frag_res) {
                    Ok(true) => {}
                    Ok(false) => dead[b] = true,
                    Err(()) => {
                        fallback = true;
                        break;
                    }
                }
                possible = first_possible(&ctx, &dead);
            }
            if frag_res.len() < nexpt {
                fallback = true;
            }
        }
        if fallback {
            return (fast_find_rings(mol), Vec::new());
        }
        if frag_res.len() > nexpt {
            let (kept, extras) = remove_extra_rings(mol, frag_res);
            frag_res = kept;
            extras_all.extend(extras);
        }
        res.extend(frag_res);
    }
    (res, extras_all)
}

/// RDKit `findSSSR` (`Chem.GetSSSR`): the smallest set of smallest rings.
pub fn sssr(mol: &Molecule) -> Vec<Vec<i32>> {
    if mol.num_atoms() == 0 {
        return Vec::new();
    }
    find_sssr(mol).0
}

/// RDKit `symmetrizeSSSR`: SSSR plus equivalent same-size rings.
pub fn symmetrized_sssr(mol: &Molecule) -> Vec<Vec<i32>> {
    if mol.num_atoms() == 0 {
        return Vec::new();
    }
    let (sssrs, extras) = find_sssr(mol);
    let mut res = sssrs.clone();
    if extras.is_empty() {
        return res;
    }
    let bond_sssrs: Vec<Vec<i32>> = sssrs.iter().map(|r| ring_atoms_to_bonds(mol, r)).collect();
    let mut bond_counts = vec![0i32; mol.num_bonds()];
    for r in &bond_sssrs {
        for &b in r {
            bond_counts[b as usize] += 1;
        }
    }
    for extra in &extras {
        let extra_bonds = ring_atoms_to_bonds(mol, extra);
        for ring in &bond_sssrs {
            if ring.len() != extra_bonds.len() {
                continue;
            }
            let mut share_bond = false;
            let mut replaces_all_unique = true;
            for &b in ring {
                let count = bond_counts[b as usize];
                if count == 1 || !share_bond {
                    if extra_bonds.contains(&b) {
                        share_bond = true;
                    } else if count == 1 {
                        replaces_all_unique = false;
                    }
                }
            }
            if share_bond && replaces_all_unique {
                res.push(extra.clone());
                break;
            }
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn sorted_rings(smiles: &str) -> Vec<Vec<i32>> {
        let mol = parse_smiles(smiles).unwrap();
        let mut rings = symmetrized_sssr(&mol);
        for r in &mut rings {
            r.sort();
        }
        rings.sort();
        rings
    }

    #[test]
    fn simple_rings() {
        assert_eq!(sorted_rings("C1CCCCC1"), vec![vec![0, 1, 2, 3, 4, 5]]);
        assert_eq!(sorted_rings("CCO"), Vec::<Vec<i32>>::new());
        assert_eq!(sorted_rings("c1ccc2ccccc2c1").len(), 2);
    }

    #[test]
    fn symmetric_cages() {
        // cubane: RDKit reports 6 rings in the symmetrized SSSR
        assert_eq!(sorted_rings("C12C3C4C1C5C2C3C45").len(), 6);
        // norbornane: 2 rings
        assert_eq!(sorted_rings("C1CC2CCC1C2").len(), 2);
    }
}
