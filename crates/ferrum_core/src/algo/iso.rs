//! Iso algorithms.
//!
//! This module contains core cheminformatics graph logic used by parsers and
//! descriptors. The implementations prioritize determinism, correctness, and
//! predictable performance on large molecular batches.
//!
//! Learning focus:
//! - Graph traversal and indexing patterns over atom/bond adjacency.
//! - Performance-oriented Rust design in numerically heavy code paths.
//!
//! References:
//! - Cordella et al. (2001/2004). VF2 graph/subgraph isomorphism algorithm.

use crate::io::smarts::{
    AtomPrimitive, QueryAtom, QueryBond, QueryGraph, SmartsRingContext, query_atom_matches_target,
};
use crate::mol::molecule::Molecule;
use crate::types::BondType;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::cell::{Cell, OnceCell, RefCell};
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Target-side data (computed once per molecule, shared by all patterns)
// ---------------------------------------------------------------------------

/// Molecule graph topology in the form the matcher needs: CSR neighbour lists
/// (in `Molecule::adjacency` order), packed per-atom keys, atoms grouped by
/// element and a compact `(bond type, ring flag)` code per bond for
/// precompiled bond queries. Cycle membership (graph-theoretic, independent
/// of the sanitizer's ring flags) and neighbourhood counts are built lazily.
pub(crate) struct TargetTopo {
    nbr_start: Vec<u32>,
    nbrs: Vec<(u16, u16)>,
    bond_code: Vec<u8>,
    /// Atom indices grouped by atomic number (ascending within a group).
    elem_start: Vec<u32>,
    elem_atoms: Vec<u16>,
    /// Packed per-atom data for the fast atom pre-check.
    keys: Vec<TargetAtomKey>,
    cycles: OnceCell<CycleInfo>,
    /// Per-atom neighbourhood counts used by SMARTS primitives.
    inv: OnceCell<Vec<AtomNeighbourCounts>>,
    /// Lazily computed RDKit total valence (i32::MIN = not yet computed).
    valence: OnceCell<Vec<Cell<i32>>>,
}

/// Graph-theoretic cycle information of the target.
struct CycleInfo {
    /// bond is not a bridge
    bond_cyclic: Vec<bool>,
    /// atoms in the atom's 2-edge-connected component (0 if acyclic)
    comp_size: Vec<u16>,
}

const KEY_AROMATIC: u32 = 1 << 8;

/// `key` = atomic number | aromatic << 8 | (charge as u8) << 16.
#[derive(Clone, Copy)]
struct TargetAtomKey {
    key: u32,
    degree: u16,
    /// OR of `1 << bond_code` over the incident bonds
    incident: u16,
}

/// Necessary conditions on the image of a pattern atom, checked against a
/// `TargetAtomKey`: `(key & mask) == value` and the degree.
#[derive(Clone, Copy)]
struct PatternAtomCheck {
    mask: u32,
    value: u32,
    min_degree: u16,
    /// accepted bond codes of (up to four of) the atom's pattern bonds: each
    /// needs an incident target bond of an accepted kind (0 = unused slot)
    bond_needs: [u16; 4],
    /// allowed atomic numbers (None = any)
    elements: Option<ElemSet>,
}

impl PatternAtomCheck {
    #[inline]
    fn passes(&self, k: &TargetAtomKey) -> bool {
        (k.key & self.mask) == self.value
            && k.degree >= self.min_degree
            && self.bond_needs.iter().all(|&need| need == 0 || need & k.incident != 0)
            && self.elements.is_none_or(|e| e.contains(k.key as u8))
    }
}

/// Neighbourhood-derived atom invariants (computed once per target).
#[derive(Clone, Copy, Default)]
pub(crate) struct AtomNeighbourCounts {
    /// neighbours that are hydrogen atoms
    pub(crate) h_neighbours: u8,
    /// neighbours that are not hydrogen, or are isotopic hydrogen (`d`)
    pub(crate) non_h_degree: u8,
    /// neighbours that are not C and not H (`z`)
    pub(crate) hetero: u8,
    /// non-aromatic neighbours that are not C and not H (`Z`)
    pub(crate) aliphatic_hetero: u8,
    /// incident bonds flagged as ring bonds (`x`)
    pub(crate) ring_bonds: u8,
}

/// Size of the component each vertex belongs to in the graph restricted to
/// cyclic edges (0 for vertices without a cyclic edge).
fn cyclic_component_sizes(n: usize, edges: impl Iterator<Item = (usize, usize)>) -> Vec<u16> {
    fn find(parent: &mut [u32], mut x: usize) -> usize {
        while parent[x] as usize != x {
            parent[x] = parent[parent[x] as usize];
            x = parent[x] as usize;
        }
        x
    }
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut cyclic = vec![false; n];
    for (a, b) in edges {
        cyclic[a] = true;
        cyclic[b] = true;
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra] = rb as u32;
        }
    }
    let mut size = vec![0u16; n];
    for v in 0..n {
        if cyclic[v] {
            let r = find(&mut parent, v);
            size[r] = size[r].saturating_add(1);
        }
    }
    (0..n)
        .map(|v| if cyclic[v] { size[find(&mut parent, v)] } else { 0 })
        .collect()
}

/// Marks every edge that lies on a cycle (i.e. is not a bridge) of the graph
/// given as CSR lists of `(neighbour, edge id)`.
fn cyclic_edges(n: usize, n_edges: usize, nbr_start: &[u32], nbrs: &[(u16, u16)]) -> Vec<bool> {
    let mut cyclic = vec![true; n_edges];
    let mut disc = vec![u32::MAX; n];
    let mut low = vec![0u32; n];
    let mut timer = 0u32;
    // (vertex, edge used to enter it, next neighbour slot)
    let mut stack: Vec<(u16, u16, u32)> = Vec::new();
    for root in 0..n {
        if disc[root] != u32::MAX {
            continue;
        }
        disc[root] = timer;
        low[root] = timer;
        timer += 1;
        stack.push((root as u16, u16::MAX, nbr_start[root]));
        while let Some(top) = stack.last_mut() {
            let (v, entry_edge, pos) = *top;
            let v = v as usize;
            if pos < nbr_start[v + 1] {
                top.2 += 1;
                let (w, e) = nbrs[pos as usize];
                if e == entry_edge {
                    continue;
                }
                let w = w as usize;
                if disc[w] == u32::MAX {
                    disc[w] = timer;
                    low[w] = timer;
                    timer += 1;
                    stack.push((w as u16, e, nbr_start[w]));
                } else if disc[w] < low[v] {
                    low[v] = disc[w];
                }
            } else {
                stack.pop();
                if let Some(parent) = stack.last() {
                    let u = parent.0 as usize;
                    if low[v] < low[u] {
                        low[u] = low[v];
                    }
                    if low[v] > disc[u] {
                        cyclic[entry_edge as usize] = false;
                    }
                }
            }
        }
    }
    cyclic
}

impl TargetTopo {
    fn new(mol: &Molecule) -> Self {
        let n = mol.atoms.len();
        let mut nbr_start = Vec::with_capacity(n + 1);
        let mut nbrs = Vec::with_capacity(mol.bonds.len() * 2);
        for a in 0..n {
            nbr_start.push(nbrs.len() as u32);
            for &b in &mol.adjacency[a] {
                let bond = &mol.bonds[b as usize];
                let o = if bond.source as usize == a { bond.target } else { bond.source };
                nbrs.push((o, b));
            }
        }
        nbr_start.push(nbrs.len() as u32);
        let bond_code: Vec<u8> = mol
            .bonds
            .iter()
            .map(|b| bond_code(b.bond_type, b.is_in_ring()))
            .collect();
        let mut elem_start = vec![0u32; 257];
        for atom in &mol.atoms {
            elem_start[atom.atomic_number as usize + 1] += 1;
        }
        for z in 0..256 {
            elem_start[z + 1] += elem_start[z];
        }
        let mut fill = elem_start.clone();
        let mut elem_atoms = vec![0u16; n];
        for (a, atom) in mol.atoms.iter().enumerate() {
            let z = atom.atomic_number as usize;
            elem_atoms[fill[z] as usize] = a as u16;
            fill[z] += 1;
        }
        let keys = (0..n)
            .map(|a| {
                let atom = &mol.atoms[a];
                TargetAtomKey {
                    key: atom.atomic_number as u32
                        | if atom.is_aromatic() { KEY_AROMATIC } else { 0 }
                        | ((atom.formal_charge as u8 as u32) << 16),
                    degree: (nbr_start[a + 1] - nbr_start[a]).min(u16::MAX as u32) as u16,
                    incident: nbrs[nbr_start[a] as usize..nbr_start[a + 1] as usize]
                        .iter()
                        .fold(0u16, |acc, &(_, b)| acc | 1 << bond_code[b as usize]),
                }
            })
            .collect();
        Self {
            nbr_start,
            nbrs,
            bond_code,
            elem_start,
            elem_atoms,
            keys,
            cycles: OnceCell::new(),
            inv: OnceCell::new(),
            valence: OnceCell::new(),
        }
    }

    fn cycles(&self, mol: &Molecule) -> &CycleInfo {
        self.cycles.get_or_init(|| {
            let n = mol.atoms.len();
            let bond_cyclic = cyclic_edges(n, mol.bonds.len(), &self.nbr_start, &self.nbrs);
            let comp_size = cyclic_component_sizes(
                n,
                mol.bonds
                    .iter()
                    .enumerate()
                    .filter(|&(b, _)| bond_cyclic[b])
                    .map(|(_, bond)| (bond.source as usize, bond.target as usize)),
            );
            CycleInfo { bond_cyclic, comp_size }
        })
    }

    /// Neighbourhood counts of all atoms.
    #[inline]
    pub(crate) fn inv(&self, mol: &Molecule) -> &[AtomNeighbourCounts] {
        self.inv.get_or_init(|| {
            (0..mol.atoms.len())
                .map(|a| {
                    let mut c = AtomNeighbourCounts::default();
                    for &(o, b) in self.neighbours(a) {
                        let other = &mol.atoms[o as usize];
                        if other.atomic_number == 1 {
                            c.h_neighbours = c.h_neighbours.saturating_add(1);
                        }
                        if other.atomic_number != 1 || other.isotope != 0 {
                            c.non_h_degree = c.non_h_degree.saturating_add(1);
                        }
                        if other.atomic_number != 6 && other.atomic_number != 1 {
                            c.hetero = c.hetero.saturating_add(1);
                            if !other.is_aromatic() {
                                c.aliphatic_hetero = c.aliphatic_hetero.saturating_add(1);
                            }
                        }
                        if mol.bonds[b as usize].is_in_ring() {
                            c.ring_bonds = c.ring_bonds.saturating_add(1);
                        }
                    }
                    c
                })
                .collect()
        })
    }

    #[inline]
    fn degree(&self, a: usize) -> u32 {
        self.nbr_start[a + 1] - self.nbr_start[a]
    }

    #[inline]
    fn neighbours(&self, a: usize) -> &[(u16, u16)] {
        &self.nbrs[self.nbr_start[a] as usize..self.nbr_start[a + 1] as usize]
    }

    /// RDKit total valence of atom `a` (memoized).
    #[inline]
    pub(crate) fn total_valence(&self, mol: &Molecule, a: usize) -> i32 {
        let cells = self
            .valence
            .get_or_init(|| (0..mol.atoms.len()).map(|_| Cell::new(i32::MIN)).collect());
        let v = cells[a].get();
        if v != i32::MIN {
            return v;
        }
        let v = crate::algo::sanitize::total_valence(mol, a);
        cells[a].set(v);
        v
    }

    #[inline]
    fn atoms_of_element(&self, z: u8) -> &[u16] {
        let z = z as usize;
        &self.elem_atoms[self.elem_start[z] as usize..self.elem_start[z + 1] as usize]
    }

    #[inline]
    fn find_bond(&self, a: usize, b: u16) -> Option<u16> {
        self.neighbours(a).iter().find(|&&(o, _)| o == b).map(|&(_, bond)| bond)
    }
}

const BOND_TYPES: [BondType; 6] = [
    BondType::Single,
    BondType::Double,
    BondType::Triple,
    BondType::Aromatic,
    BondType::Dative,
    BondType::Unknown,
];

#[inline]
fn bond_code(bond_type: BondType, in_ring: bool) -> u8 {
    (bond_type as u8) * 2 + in_ring as u8
}

pub struct IsoTarget<'a> {
    pub target: &'a Molecule,
    pub(crate) target_counts: [u16; 119],
    pub(crate) target_stats: TargetAtomStats,
    present_elements: ElemSet,
    /// (atomic number, count) of the elements present
    present: SmallVec<[(u8, u16); 12]>,
    pub(crate) ring_ctx: OnceCell<SmartsRingContext>,
    topo: OnceCell<TargetTopo>,
    /// Recursive SMARTS state per sub-pattern id (lazy per-atom results).
    recursive: RefCell<FxHashMap<u64, RecursiveEntry>>,
}

impl<'a> IsoTarget<'a> {
    pub fn new(target: &'a Molecule) -> Self {
        let mut present_elements = ElemSet::NONE;
        for atom in &target.atoms {
            present_elements = present_elements.or(ElemSet::single(atom.atomic_number));
        }
        let target_counts = target_element_counts(target);
        let present = target_counts
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c > 0)
            .map(|(z, &c)| (z as u8, c))
            .collect();
        Self {
            target,
            present_elements,
            present,
            target_counts,
            target_stats: TargetAtomStats::from_molecule(target),
            ring_ctx: OnceCell::new(),
            topo: OnceCell::new(),
            recursive: RefCell::new(FxHashMap::default()),
        }
    }

    pub(crate) fn get_ring_ctx(&self) -> &SmartsRingContext {
        self.ring_ctx.get_or_init(|| SmartsRingContext::for_molecule(self.target))
    }

    #[inline]
    pub(crate) fn topo(&self) -> &TargetTopo {
        self.topo.get_or_init(|| TargetTopo::new(self.target))
    }

}

/// Per-molecule state of one recursive SMARTS sub-pattern: memoized
/// per-atom answers (0 unknown, 1 false, 2 true) and the reusable search
/// scratch (plan + atom-predicate memo) for further rooted searches.
struct RecursiveEntry {
    starts: Vec<u8>,
    impossible: bool,
    scratch: Option<Scratch>,
}

/// Recursive SMARTS `$(sub)`: can `atom` be the first atom of a match of
/// `sub`? Answered lazily per atom by a rooted existence search, memoized in
/// the `IsoTarget` (shared by every pattern matched against it).
pub(crate) fn recursive_start_matches(iso_target: &IsoTarget, sub: &QueryGraph, atom: u16) -> bool {
    let id = sub.meta.id;
    let a = atom as usize;
    // cheap necessary conditions on the root atom before touching the cache
    let cq = CompiledQuery::get(sub);
    if !cq.checks[0].passes(&iso_target.topo().keys[a]) {
        return false;
    }
    let scratch = {
        let mut map = iso_target.recursive.borrow_mut();
        let entry = map.entry(id).or_insert_with(|| RecursiveEntry {
            starts: vec![0; iso_target.target.atoms.len()],
            impossible: false,
            scratch: None,
        });
        match entry.starts[a] {
            1 => return false,
            2 => return true,
            _ => {}
        }
        if entry.impossible {
            return false;
        }
        entry.scratch.take()
    };
    let state = match scratch {
        Some(scratch) => Some(MatchState::from_scratch(iso_target, sub, scratch)),
        None => MatchState::new(iso_target, sub, true, true),
    };
    let Some(mut state) = state else {
        let mut map = iso_target.recursive.borrow_mut();
        if let Some(entry) = map.get_mut(&id) {
            entry.impossible = true;
        }
        return false;
    };
    let result = state.rooted_exists(a);
    let scratch = state.into_scratch();
    let mut map = iso_target.recursive.borrow_mut();
    if let Some(entry) = map.get_mut(&id) {
        entry.starts[a] = 1 + result as u8;
        entry.scratch = Some(scratch);
    }
    result
}

// ---------------------------------------------------------------------------
// Pattern-side data (computed once per pattern, cached in the QueryGraph)
// ---------------------------------------------------------------------------

/// Target-independent search data of a query graph.
pub(crate) struct CompiledQuery {
    filters: Vec<AtomFastFilter>,
    fixed_elem: Vec<Option<u8>>,
    /// The fast filter alone decides the atom predicate (no tree walk needed).
    filter_exact: Vec<bool>,
    /// Packed necessary conditions (filter + degree).
    checks: Vec<PatternAtomCheck>,
    /// For atoms on a pattern cycle: size of their cyclic (2-edge-connected)
    /// pattern component; the image's component in the target must be at
    /// least as large (0 = atom not on a cycle, no requirement).
    cyclic_min: Vec<u16>,
    /// Bond lies on a cycle of the pattern, so its image must lie on a
    /// cycle of the target (atom-level counterparts are folded into `checks`).
    bond_in_cycle: Vec<bool>,
    /// Bit `bond_code(type, ring)` set when the bond query accepts it.
    bond_mask: Vec<u16>,
    nbrs: Vec<SmallVec<[(u16, u16); 4]>>,
    has_cycle: bool,
    required_elements: SmallVec<[(u8, u16); 8]>,
    /// The pattern is one simple cycle whose atoms (and bonds) all carry the
    /// same query, so every match comes in 2n symmetric copies (rotations and
    /// reflections) and an enumeration may keep only one canonical copy.
    uniform_cycle: bool,
    req_aromatic_atoms: u16,
    req_aliphatic_atoms: u16,
    req_aromatic_bonds: u16,
    req_ring_bonds: u16,
    impossible: bool,
}

impl CompiledQuery {
    fn new(pattern: &QueryGraph) -> Self {
        let n = pattern.atoms.len();
        let filters: Vec<AtomFastFilter> = pattern
            .atoms
            .iter()
            .map(|query_atom| AtomFastFilter {
                required_element: fixed_atomic_number_filter(query_atom),
                requires_aromatic: query_atom_requires_aromatic(query_atom),
                requires_aliphatic: query_atom_requires_aliphatic(query_atom),
                required_charge: query_atom_required_charge(query_atom),
            })
            .collect();
        let fixed_elem: Vec<Option<u8>> = pattern.atoms.iter().map(fixed_atomic_number).collect();
        let impossible = filters.iter().any(|f| f.requires_aromatic && f.requires_aliphatic);
        let filter_exact = pattern
            .atoms
            .iter()
            .map(|a| filter_is_exact(a) || query_atom_exact_elements(a).is_some())
            .collect();

        let mut nbrs: Vec<SmallVec<[(u16, u16); 4]>> = vec![SmallVec::new(); n];
        for (p, adj) in pattern.adjacency.iter().enumerate().take(n) {
            for &b in adj {
                let bond = &pattern.bonds[b as usize];
                let o = if bond.source as usize == p { bond.target } else { bond.source };
                nbrs[p].push((o, b));
            }
        }
        let min_degree: Vec<u32> = (0..n)
            .map(|p| {
                let mut seen: SmallVec<[u16; 8]> = SmallVec::new();
                for &(o, _) in &nbrs[p] {
                    if o as usize != p && !seen.contains(&o) {
                        seen.push(o);
                    }
                }
                seen.len() as u32
            })
            .collect();

        // cycle membership via bridges of the pattern graph
        let mut start = Vec::with_capacity(n + 1);
        let mut flat: Vec<(u16, u16)> = Vec::new();
        for list in &nbrs {
            start.push(flat.len() as u32);
            flat.extend(list.iter().copied());
        }
        start.push(flat.len() as u32);
        let mut bond_in_cycle = cyclic_edges(n, pattern.bonds.len(), &start, &flat);
        // self-loops / parallel bonds never occur in sane SMARTS; never prune on them
        for (b, bond) in pattern.bonds.iter().enumerate() {
            if bond.source == bond.target {
                bond_in_cycle[b] = false;
            }
        }
        let mut atom_in_cycle = vec![false; n];
        for (b, bond) in pattern.bonds.iter().enumerate() {
            if bond_in_cycle[b] {
                atom_in_cycle[bond.source as usize] = true;
                atom_in_cycle[bond.target as usize] = true;
            }
        }
        let has_cycle = bond_in_cycle.iter().any(|&c| c);
        let cyclic_comp_size = cyclic_component_sizes(
            n,
            pattern
                .bonds
                .iter()
                .enumerate()
                .filter(|&(b, _)| bond_in_cycle[b])
                .map(|(_, bond)| (bond.source as usize, bond.target as usize)),
        );

        let bond_mask: Vec<u16> = pattern
            .bonds
            .iter()
            .map(|bond| {
                let mut mask = 0u16;
                for bt in BOND_TYPES {
                    for ring in [false, true] {
                        if bond.query.matches_props(bt, ring) {
                            mask |= 1 << bond_code(bt, ring);
                        }
                    }
                }
                mask
            })
            .collect();

        let mut required = [0u16; 119];
        for z in filters.iter().filter_map(|f| f.required_element).collect::<SmallVec<[u8; 16]>>().iter() {
            let idx = *z as usize;
            if idx < required.len() {
                required[idx] = required[idx].saturating_add(1);
            }
        }
        let required_elements = required
            .iter()
            .enumerate()
            .filter_map(|(idx, &count)| (count > 0).then_some((idx as u8, count)))
            .collect();
        let req_aromatic_atoms = filters.iter().filter(|f| f.requires_aromatic).count() as u16;
        let req_aliphatic_atoms = filters.iter().filter(|f| f.requires_aliphatic).count() as u16;
        let req_aromatic_bonds =
            pattern.bonds.iter().filter(|b| query_bond_requires_aromatic(&b.query)).count() as u16;
        let req_ring_bonds =
            pattern.bonds.iter().filter(|b| query_bond_requires_ring(&b.query)).count() as u16;

        let checks = (0..n)
            .map(|p| {
                let f = &filters[p];
                let mut mask = 0u32;
                let mut value = 0u32;
                if let Some(z) = f.required_element {
                    mask |= 0xFF;
                    value |= z as u32;
                }
                if f.requires_aromatic {
                    mask |= KEY_AROMATIC;
                    value |= KEY_AROMATIC;
                } else if f.requires_aliphatic {
                    mask |= KEY_AROMATIC;
                }
                if let Some(c) = f.required_charge {
                    mask |= 0xFF << 16;
                    value |= (c as u8 as u32) << 16;
                }
                let mut elements = query_atom_elements(&pattern.atoms[p]);
                if let Some((set, req)) = query_atom_exact_elements(&pattern.atoms[p]) {
                    elements = elements.and(set);
                    match req {
                        AromReq::Aromatic => {
                            mask |= KEY_AROMATIC;
                            value |= KEY_AROMATIC;
                        }
                        AromReq::Aliphatic => mask |= KEY_AROMATIC,
                        AromReq::Never => elements = ElemSet::NONE,
                        AromReq::Any => {}
                    }
                }
                if f.requires_aromatic && f.requires_aliphatic {
                    elements = ElemSet::NONE;
                }
                let mut bond_needs = [0u16; 4];
                for (slot, &(_, b)) in bond_needs.iter_mut().zip(nbrs[p].iter()) {
                    *slot = bond_mask[b as usize];
                }
                PatternAtomCheck {
                    mask,
                    value,
                    bond_needs,
                    min_degree: min_degree[p].min(u16::MAX as u32) as u16,
                    // redundant when the element is already fixed by the mask
                    elements: (!elements.is_all()
                        && !f.required_element.is_some_and(|z| elements.contains(z)))
                    .then_some(elements),
                }
            })
            .collect();
        let cyclic_min = (0..n).map(|p| if atom_in_cycle[p] { cyclic_comp_size[p] } else { 0 }).collect();
        let uniform_cycle = n >= 3
            && pattern.bonds.len() == n
            && bond_in_cycle.iter().all(|&c| c)
            && min_degree.iter().all(|&d| d == 2)
            && nbrs.iter().all(|l| l.len() == 2)
            && cyclic_comp_size.iter().all(|&c| c as usize == n)
            && pattern.bonds.windows(2).all(|w| w[0].query == w[1].query)
            && {
                let first = format!("{:?}", pattern.atoms[0]);
                pattern.atoms[1..].iter().all(|a| format!("{:?}", a) == first)
            };
        Self {
            uniform_cycle,
            checks,
            cyclic_min,
            filters,
            fixed_elem,
            filter_exact,
            bond_in_cycle,
            bond_mask,
            nbrs,
            has_cycle,
            required_elements,
            req_aromatic_atoms,
            req_aliphatic_atoms,
            req_aromatic_bonds,
            req_ring_bonds,
            impossible,
        }
    }

    #[inline]
    fn get(pattern: &QueryGraph) -> &CompiledQuery {
        pattern.meta.compiled.get_or_init(|| CompiledQuery::new(pattern))
    }

    /// Element / aromaticity / bond-count screen (the old `passes_element_screen`).
    fn screen_passes(&self, iso_target: &IsoTarget) -> bool {
        if self.impossible {
            return false;
        }
        let stats = &iso_target.target_stats;
        self.checks
            .iter()
            .all(|c| c.elements.is_none_or(|e| e.intersects(&iso_target.present_elements)))
            && self.required_elements
            .iter()
            .all(|&(z, count)| iso_target.target_counts[z as usize] >= count)
            && self.req_aromatic_atoms <= stats.aromatic_atoms
            && self.req_aliphatic_atoms <= stats.aliphatic_atoms
            && self.req_aromatic_bonds <= stats.aromatic_bonds
            && self.req_ring_bonds <= stats.ring_bonds
    }
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// Backtracking (VF2-style) state for one pattern on one target.
///
/// The search order is fixed up front (it only depends on the pattern and
/// the target's element counts) and is identical to the order the previous
/// dynamic selection produced, so matches are enumerated in the same order.
struct MatchState<'a, 'm> {
    iso_target: &'a IsoTarget<'m>,
    pattern: &'a QueryGraph,
    cq: &'a CompiledQuery,
    topo: &'a TargetTopo,
    cycles: Option<&'a CycleInfo>,
    n: usize,
    n_t: usize,
    plan: Rc<Plan>,
    /// Canonical-copy restriction for uniform cycles: (first, second, last)
    /// pattern atoms in search order; images of all atoms must exceed the
    /// first atom's image, and the last atom's image must exceed the second's.
    sym: Option<(u16, u16, u16)>,
    mapping: SmallVec<[u16; 32]>,
    used: SmallVec<[u64; 4]>,
    /// Memoized atom predicate results (p * n_t + t): 0 unknown, 1 false, 2 true.
    /// Allocated on first use (all-trivial patterns never need it).
    memo: Vec<u8>,
    memo_enabled: bool,
}

/// Target-specific search plan of a pattern.
struct Plan {
    order: SmallVec<[u16; 32]>,
    /// Per depth: bonds to atoms mapped at earlier depths `(pattern bond, neighbour)`.
    back_start: SmallVec<[u16; 33]>,
    back: SmallVec<[(u16, u16); 40]>,
}

/// Plans depend on the target only through the relative order of the
/// per-atom candidate estimates, so they are cached per thread keyed by
/// (pattern id, estimate ranks, rooted, free order).
fn cached_plan(
    pattern: &QueryGraph,
    cq: &CompiledQuery,
    est: &[usize],
    rooted: bool,
    free_order: bool,
) -> Rc<Plan> {
    const MAX_CACHED: usize = 1 << 14;
    thread_local! {
        static PLANS: RefCell<FxHashMap<(u64, u128, u8), Rc<Plan>>> = RefCell::new(FxHashMap::default());
    }
    let n = est.len();
    if n > 32 {
        return Rc::new(build_plan(pattern, cq, est, rooted));
    }
    let mut distinct: SmallVec<[usize; 32]> = SmallVec::from_slice(est);
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() > 16 {
        return Rc::new(build_plan(pattern, cq, est, rooted));
    }
    // 4 bits per atom: rank of its estimate among the distinct estimates
    let mut sig = 0u128;
    for (p, e) in est.iter().enumerate() {
        let rank = distinct.partition_point(|d| d < e) as u128;
        sig |= rank << (4 * p);
    }
    let key = (pattern.meta.id, sig, rooted as u8 | (free_order as u8) << 1);
    PLANS.with(|cell| {
        if let Some(plan) = cell.borrow().get(&key) {
            return plan.clone();
        }
        let plan = Rc::new(build_plan(pattern, cq, est, rooted));
        let mut map = cell.borrow_mut();
        if map.len() >= MAX_CACHED {
            map.clear();
        }
        map.insert(key, plan.clone());
        plan
    })
}

fn build_plan(pattern: &QueryGraph, cq: &CompiledQuery, est: &[usize], rooted: bool) -> Plan {
    let n = est.len();
    let mut sorted: SmallVec<[u16; 32]> = (0..n as u16).collect();
    sorted.sort_by_key(|&p| {
        let p = p as usize;
        (est[p], usize::MAX - pattern.adjacency[p].len(), p)
    });
    let mut mapped_nb: SmallVec<[usize; 32]> = smallvec::smallvec![0; n];
    let mut is_mapped: SmallVec<[bool; 32]> = smallvec::smallvec![false; n];
    let mut order: SmallVec<[u16; 32]> = SmallVec::with_capacity(n);
    for depth in 0..n {
        let chosen = if rooted && depth == 0 {
            0usize
        } else {
            let mut best: Option<usize> = None;
            let mut best_mapped = 0usize;
            let mut best_est = usize::MAX;
            let mut best_degree = 0usize;
            for &p in &sorted {
                let p = p as usize;
                if is_mapped[p] {
                    continue;
                }
                let mapped = mapped_nb[p];
                let e = est[p];
                let degree = pattern.adjacency[p].len();
                let better = match best {
                    None => true,
                    Some(_) if (mapped > 0) != (best_mapped > 0) => mapped > 0,
                    Some(_) if mapped != best_mapped => mapped > best_mapped,
                    Some(_) if e != best_est => e < best_est,
                    Some(_) => degree > best_degree,
                };
                if better {
                    best = Some(p);
                    best_mapped = mapped;
                    best_est = e;
                    best_degree = degree;
                }
            }
            best.unwrap_or(0)
        };
        order.push(chosen as u16);
        is_mapped[chosen] = true;
        for &(o, _) in &cq.nbrs[chosen] {
            mapped_nb[o as usize] += 1;
        }
    }
    let mut pos: SmallVec<[usize; 32]> = smallvec::smallvec![0; n];
    for (d, &p) in order.iter().enumerate() {
        pos[p as usize] = d;
    }

    // --- per-depth back edges ---
    let mut back_start: SmallVec<[u16; 33]> = SmallVec::with_capacity(n + 1);
    let mut back: SmallVec<[(u16, u16); 40]> = SmallVec::new();
    for (d, &p) in order.iter().enumerate() {
        back_start.push(back.len() as u16);
        for &(o, b) in &cq.nbrs[p as usize] {
            if pos[o as usize] < d {
                back.push((b, o));
            }
        }
    }
    back_start.push(back.len() as u16);

    Plan { order, back_start, back }
}

/// Reusable parts of a `MatchState` (kept for recursive sub-patterns).
struct Scratch {
    plan: Rc<Plan>,
    used: SmallVec<[u64; 4]>,
    memo: Vec<u8>,
    memo_enabled: bool,
}

impl<'a, 'm> MatchState<'a, 'm> {
    /// `free_order`: the caller's result does not depend on the order in which
    /// matches are enumerated (existence, start-atom sets, uncapped unique
    /// counts), so the search may use sharper candidate estimates. Otherwise
    /// the historical order is kept so enumeration order is unchanged.
    fn new(iso_target: &'a IsoTarget<'m>, pattern: &'a QueryGraph, rooted: bool, free_order: bool) -> Option<Self> {
        let mol = iso_target.target;
        let n = pattern.atoms.len();
        let n_t = mol.atoms.len();
        if n == 0 || n > n_t || pattern.bonds.len() > mol.bonds.len() {
            return None;
        }
        let cq = CompiledQuery::get(pattern);
        if !cq.screen_passes(iso_target) {
            return None;
        }
        // --- search order (replicates the former dynamic selection) ---
        let est: SmallVec<[usize; 32]> = if free_order {
            (0..n)
                .map(|p| {
                    let check = &cq.checks[p];
                    let mut e = match (check.elements, cq.filters[p].required_element) {
                        (Some(set), _) => iso_target
                            .present
                            .iter()
                            .filter(|&&(z, _)| set.contains(z))
                            .map(|&(_, c)| c as usize)
                            .sum(),
                        (None, Some(z)) => iso_target.target_counts[z as usize] as usize,
                        (None, None) => n_t,
                    };
                    if check.mask & KEY_AROMATIC != 0 {
                        let stats = &iso_target.target_stats;
                        let limit = if check.value & KEY_AROMATIC != 0 {
                            stats.aromatic_atoms
                        } else {
                            stats.aliphatic_atoms
                        };
                        e = e.min(limit as usize);
                    }
                    e
                })
                .collect()
        } else {
            cq.fixed_elem
                .iter()
                .map(|z| z.map(|z| iso_target.target_counts[z as usize] as usize).unwrap_or(n_t))
                .collect()
        };
        let plan = cached_plan(pattern, cq, &est, rooted, free_order);
        let scratch = Scratch {
            plan,
            used: smallvec::smallvec![0u64; n_t.div_ceil(64)],
            memo: Vec::new(),
            memo_enabled: n > 1 || rooted,
        };
        Some(Self::from_scratch(iso_target, pattern, scratch))
    }

    fn from_scratch(iso_target: &'a IsoTarget<'m>, pattern: &'a QueryGraph, scratch: Scratch) -> Self {
        let n = pattern.atoms.len();
        Self {
            iso_target,
            pattern,
            cq: CompiledQuery::get(pattern),
            topo: iso_target.topo(),
            cycles: if CompiledQuery::get(pattern).has_cycle {
                Some(iso_target.topo().cycles(iso_target.target))
            } else {
                None
            },
            n,
            n_t: iso_target.target.atoms.len(),
            plan: scratch.plan,
            sym: None,
            mapping: smallvec::smallvec![u16::MAX; n],
            used: scratch.used,
            memo: scratch.memo,
            memo_enabled: scratch.memo_enabled,
        }
    }

    /// Restrict the enumeration to one canonical copy per cycle (see
    /// `CompiledQuery::uniform_cycle`). Returns false if not applicable.
    fn enable_cycle_symmetry_breaking(&mut self) -> bool {
        if !self.cq.uniform_cycle {
            return false;
        }
        let first = self.plan.order[0];
        let second = self.plan.order[1];
        let nb = &self.cq.nbrs[first as usize];
        let last = if nb[0].0 == second {
            nb[1].0
        } else if nb[1].0 == second {
            nb[0].0
        } else {
            return false;
        };
        self.sym = Some((first, second, last));
        true
    }

    #[inline]
    fn sym_ok(&self, p: usize, t: usize) -> bool {
        match self.sym {
            None => true,
            Some((first, second, last)) => {
                p == first as usize
                    || (t > self.mapping[first as usize] as usize
                        && (p != last as usize || t > self.mapping[second as usize] as usize))
            }
        }
    }

    fn into_scratch(self) -> Scratch {
        Scratch { plan: self.plan, used: self.used, memo: self.memo, memo_enabled: self.memo_enabled }
    }

    /// Is there a match with pattern atom 0 on target atom `root`?
    /// (The plan must have been built with `rooted = true`.)
    fn rooted_exists(&mut self, root: usize) -> bool {
        if !self.atom_ok(0, root) {
            return false;
        }
        self.assign(0, root);
        let found = !self.search(1, &mut |_| false);
        self.unassign(0, root);
        found
    }

    /// Atom-level checks for mapping pattern atom `p` onto target atom `t`.
    #[inline]
    fn atom_ok(&mut self, p: usize, t: usize) -> bool {
        let cq = self.cq;
        if !cq.checks[p].passes(&self.topo.keys[t]) {
            return false;
        }
        let min_comp = cq.cyclic_min[p];
        if min_comp > 0 && self.cycles.is_some_and(|c| c.comp_size[t] < min_comp) {
            return false;
        }
        if cq.filter_exact[p] {
            return true;
        }
        if !self.memo_enabled {
            return query_atom_matches_target(&self.pattern.atoms[p], self.iso_target, t as u16);
        }
        if self.memo.is_empty() {
            self.memo = vec![0u8; self.n * self.n_t];
        }
        let slot = p * self.n_t + t;
        match self.memo[slot] {
            1 => false,
            2 => true,
            _ => {
                let r = query_atom_matches_target(&self.pattern.atoms[p], self.iso_target, t as u16);
                self.memo[slot] = 1 + r as u8;
                r
            }
        }
    }

    #[inline]
    fn bond_ok(&self, pattern_bond: u16, target_bond: u16) -> bool {
        let pb = pattern_bond as usize;
        let tb = target_bond as usize;
        (self.cq.bond_mask[pb] >> self.topo.bond_code[tb]) & 1 != 0
            && (!self.cq.bond_in_cycle[pb] || self.cycles.is_none_or(|c| c.bond_cyclic[tb]))
    }

    #[inline]
    fn is_used(&self, t: usize) -> bool {
        (self.used[t >> 6] >> (t & 63)) & 1 != 0
    }

    #[inline]
    fn assign(&mut self, p: usize, t: usize) {
        self.mapping[p] = t as u16;
        self.used[t >> 6] |= 1 << (t & 63);
    }

    #[inline]
    fn unassign(&mut self, p: usize, t: usize) {
        self.used[t >> 6] &= !(1 << (t & 63));
        self.mapping[p] = u16::MAX;
    }

    /// Depth-first search from `depth`; returns false once `visit` asked to stop.
    fn search<F: FnMut(&[u16]) -> bool>(&mut self, depth: usize, visit: &mut F) -> bool {
        if depth == self.n {
            return visit(&self.mapping);
        }
        let p = self.plan.order[depth] as usize;
        let bs = self.plan.back_start[depth] as usize;
        let be = self.plan.back_start[depth + 1] as usize;
        let topo = self.topo;

        if bs == be {
            if let Some(z) = self.cq.filters[p].required_element {
                for &t in topo.atoms_of_element(z) {
                    let t = t as usize;
                    if self.is_used(t) || !self.sym_ok(p, t) || !self.atom_ok(p, t) {
                        continue;
                    }
                    self.assign(p, t);
                    let go_on = self.search(depth + 1, visit);
                    self.unassign(p, t);
                    if !go_on {
                        return false;
                    }
                }
                return true;
            }
            for t in 0..self.n_t {
                if self.is_used(t) || !self.sym_ok(p, t) || !self.atom_ok(p, t) {
                    continue;
                }
                self.assign(p, t);
                let go_on = self.search(depth + 1, visit);
                self.unassign(p, t);
                if !go_on {
                    return false;
                }
            }
            return true;
        }

        // Anchor: the mapped neighbour whose image has the smallest degree.
        let mut anchor_k = bs;
        let mut anchor_degree = u32::MAX;
        for k in bs..be {
            let tq = self.mapping[self.plan.back[k].1 as usize] as usize;
            let degree = topo.degree(tq);
            if degree < anchor_degree {
                anchor_degree = degree;
                anchor_k = k;
            }
        }
        let anchor_bond = self.plan.back[anchor_k].0;
        let anchor = self.mapping[self.plan.back[anchor_k].1 as usize] as usize;

        for &(t, tb) in topo.neighbours(anchor) {
            let tu = t as usize;
            if self.is_used(tu) || !self.sym_ok(p, tu) || !self.bond_ok(anchor_bond, tb) {
                continue;
            }
            let mut ok = true;
            for k in bs..be {
                if k == anchor_k {
                    continue;
                }
                let (pb, q) = self.plan.back[k];
                let tq = self.mapping[q as usize];
                match topo.find_bond(tu, tq) {
                    Some(tb2) if self.bond_ok(pb, tb2) => {}
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok || !self.atom_ok(p, tu) {
                continue;
            }
            self.assign(p, tu);
            let go_on = self.search(depth + 1, visit);
            self.unassign(p, tu);
            if !go_on {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Substructure matching using VF2 algorithm.
///
/// Finds if the query pattern exists as a subgraph in the target molecule.
/// Returns true if at least one match is found.
///
/// # Example
/// ```ignore
/// let molecule = parse_smiles("CCO").unwrap();
/// let pattern = QueryGraph::single_atom(6); // Match any carbon
/// assert!(has_substruct_match(&molecule, &pattern));
/// ```
pub fn has_substruct_match(target: &Molecule, pattern: &QueryGraph) -> bool {
    has_substruct_match_target(&IsoTarget::new(target), pattern)
}

pub fn has_substruct_match_target(iso_target: &IsoTarget, pattern: &QueryGraph) -> bool {
    if pattern.atoms.is_empty() {
        return true;
    }
    if pattern.atoms.len() == 1 && pattern.bonds.is_empty() {
        let mut found = false;
        visit_single_atom(iso_target, pattern, &mut |_| {
            found = true;
            false
        });
        return found;
    }
    let Some(mut state) = MatchState::new(iso_target, pattern, false, true) else {
        return false;
    };
    // existence does not depend on which symmetric copy is found
    state.enable_cycle_symmetry_breaking();
    !state.search(0, &mut |_| false)
}

/// Finds the first substructure match and returns the atom mapping.
///
/// Returns a vector where index i is the pattern atom index,
/// and the value is the corresponding target atom index.
pub fn find_first_match(target: &Molecule, pattern: &QueryGraph) -> Option<Vec<u16>> {
    find_first_match_target(&IsoTarget::new(target), pattern)
}

pub fn find_first_match_target(iso_target: &IsoTarget, pattern: &QueryGraph) -> Option<Vec<u16>> {
    let mut first = None;
    visit_matches_target(iso_target, pattern, |m| {
        first = Some(m.to_vec());
        false
    });
    first
}

/// Counts the number of substructure matches.
pub fn count_matches(target: &Molecule, pattern: &QueryGraph) -> usize {
    count_matches_target(&IsoTarget::new(target), pattern)
}

/// Number of matches with distinct atom sets (RDKit `countMatches` /
/// `GetSubstructMatches(uniquify=True)`): symmetric re-mappings of the same
/// atoms count once.
pub fn count_matches_target(iso_target: &IsoTarget, pattern: &QueryGraph) -> usize {
    count_unique_matches_up_to(iso_target, pattern, usize::MAX)
}

/// Counts unique matches (distinct atom sets), stopping early once `limit` is reached.
pub(crate) fn count_unique_matches_up_to(iso_target: &IsoTarget, pattern: &QueryGraph, limit: usize) -> usize {
    count_unique_matches_capped(iso_target, pattern, limit, usize::MAX)
}

/// Counts unique matches (distinct atom sets). Enumeration stops as soon as
/// `limit` unique sets were found or `max_visits` raw matches (including
/// symmetric duplicates) were visited, whichever comes first.
pub fn count_unique_matches_capped(
    iso_target: &IsoTarget,
    pattern: &QueryGraph,
    limit: usize,
    max_visits: usize,
) -> usize {
    let n = pattern.atoms.len();
    if n == 0 {
        return 1;
    }
    if n == 1 {
        // single-atom matches are unique by construction
        let mut count = 0usize;
        visit_matches_impl(iso_target, pattern, false, &mut |_| {
            count += 1;
            count < limit && count < max_visits
        });
        return count;
    }
    if let Some(count) = count_unique_cycle_matches(iso_target, pattern, limit, max_visits) {
        return count;
    }
    // The result only depends on the enumeration order when a cap stops the
    // enumeration early. Count with the faster free order first and redo it
    // in the historical order only if a visit cap was hit.
    let (unique, visits) = count_unique_in_order(iso_target, pattern, limit, max_visits, true);
    if max_visits == usize::MAX || (unique < limit && visits < max_visits) {
        return unique;
    }
    count_unique_in_order(iso_target, pattern, limit, max_visits, false).0
}

/// Unique-set counting with the given order mode; returns (unique sets, raw visits).
fn count_unique_in_order(
    iso_target: &IsoTarget,
    pattern: &QueryGraph,
    limit: usize,
    max_visits: usize,
    free_order: bool,
) -> (usize, usize) {
    let n = pattern.atoms.len();
    let mut visits = 0usize;
    if n <= 4 {
        let mut seen: FxHashSet<u64> = FxHashSet::default();
        visit_matches_impl(iso_target, pattern, free_order, &mut |m| {
            visits += 1;
            let mut key = [u16::MAX; 4];
            key[..m.len()].copy_from_slice(m);
            key.sort_unstable();
            seen.insert(key.iter().fold(0u64, |acc, &x| (acc << 16) | x as u64));
            seen.len() < limit && visits < max_visits
        });
        return (seen.len(), visits);
    }
    if n <= 8 {
        let mut seen: FxHashSet<u128> = FxHashSet::default();
        visit_matches_impl(iso_target, pattern, free_order, &mut |m| {
            visits += 1;
            let mut key = [u16::MAX; 8];
            key[..m.len()].copy_from_slice(m);
            key.sort_unstable();
            seen.insert(key.iter().fold(0u128, |acc, &x| (acc << 16) | x as u128));
            seen.len() < limit && visits < max_visits
        });
        return (seen.len(), visits);
    }
    let mut seen: FxHashSet<SmallVec<[u16; 16]>> = FxHashSet::default();
    visit_matches_impl(iso_target, pattern, free_order, &mut |m| {
        visits += 1;
        let mut key: SmallVec<[u16; 16]> = SmallVec::from_slice(m);
        key.sort_unstable();
        seen.insert(key);
        seen.len() < limit && visits < max_visits
    });
    (seen.len(), visits)
}

/// `count_unique_matches_capped` for uniform cycle patterns, enumerating one
/// canonical copy per cycle instead of all 2n symmetric ones. Returns `None`
/// (caller falls back to the full enumeration) when not applicable or when a
/// cap would have stopped the full enumeration early, so results are identical.
fn count_unique_cycle_matches(
    iso_target: &IsoTarget,
    pattern: &QueryGraph,
    limit: usize,
    max_visits: usize,
) -> Option<usize> {
    let n = pattern.atoms.len();
    if !CompiledQuery::get(pattern).uniform_cycle {
        return None;
    }
    let Some(mut state) = MatchState::new(iso_target, pattern, false, true) else {
        return Some(0);
    };
    if !state.enable_cycle_symmetry_breaking() {
        return None;
    }
    let copies = 2 * n;
    let mut cycles = 0usize;
    let mut seen: FxHashSet<SmallVec<[u16; 16]>> = FxHashSet::default();
    state.search(0, &mut |m| {
        cycles += 1;
        let mut key: SmallVec<[u16; 16]> = SmallVec::from_slice(m);
        key.sort_unstable();
        seen.insert(key);
        seen.len() < limit && cycles.saturating_mul(copies) < max_visits
    });
    let visits = cycles.saturating_mul(copies);
    if seen.len() < limit && visits < max_visits {
        // the full enumeration would have visited everything as well
        Some(seen.len())
    } else if max_visits == usize::MAX {
        // only the unique-set limit applies: min(total, limit) is order independent
        Some(seen.len())
    } else {
        None
    }
}

/// Returns true if the pattern has at least `min_matches` matches.
///
/// Uses early-stop enumeration, so it avoids counting all matches when only
/// a threshold decision is needed.
pub fn has_at_least_matches(target: &Molecule, pattern: &QueryGraph, min_matches: usize) -> bool {
    has_at_least_matches_target(&IsoTarget::new(target), pattern, min_matches)
}

pub fn has_at_least_matches_target(iso_target: &IsoTarget, pattern: &QueryGraph, min_matches: usize) -> bool {
    if min_matches == 0 {
        return true;
    }
    if min_matches == 1 {
        return has_substruct_match_target(iso_target, pattern);
    }
    count_unique_matches_up_to(iso_target, pattern, min_matches) >= min_matches
}

/// Visits all substructure matches without materializing them.
///
/// The visitor returns `true` to continue enumeration or `false` to stop early.
pub fn visit_matches<F>(target: &Molecule, pattern: &QueryGraph, visit: F)
where
    F: FnMut(&[u16]) -> bool,
{
    visit_matches_target(&IsoTarget::new(target), pattern, visit);
}

pub fn visit_matches_target<F>(iso_target: &IsoTarget, pattern: &QueryGraph, mut visit: F)
where
    F: FnMut(&[u16]) -> bool,
{
    if pattern.atoms.is_empty() {
        let _ = visit(&[]);
        return;
    }
    visit_matches_impl(iso_target, pattern, false, &mut visit);
}

fn visit_matches_impl<F>(iso_target: &IsoTarget, pattern: &QueryGraph, free_order: bool, visit: &mut F)
where
    F: FnMut(&[u16]) -> bool,
{
    if pattern.atoms.is_empty() {
        let _ = visit(&[]);
        return;
    }
    if pattern.atoms.len() == 1 && pattern.bonds.is_empty() {
        visit_single_atom(iso_target, pattern, visit);
        return;
    }
    if let Some(mut state) = MatchState::new(iso_target, pattern, false, free_order) {
        state.search(0, visit);
    }
}

/// Matches of a one-atom pattern, in increasing atom order (no search state needed).
fn visit_single_atom<F: FnMut(&[u16]) -> bool>(iso_target: &IsoTarget, pattern: &QueryGraph, visit: &mut F) {
    let n_t = iso_target.target.atoms.len();
    if n_t == 0 {
        return;
    }
    let cq = CompiledQuery::get(pattern);
    if !cq.screen_passes(iso_target) {
        return;
    }
    let topo = iso_target.topo();
    let check = cq.checks[0];
    let exact = cq.filter_exact[0];
    let query_atom = &pattern.atoms[0];
    let test = |t: usize| {
        check.passes(&topo.keys[t]) && (exact || query_atom_matches_target(query_atom, iso_target, t as u16))
    };
    if let Some(z) = cq.filters[0].required_element {
        for &t in topo.atoms_of_element(z) {
            if test(t as usize) && !visit(&[t]) {
                return;
            }
        }
    } else {
        for t in 0..n_t {
            if test(t) && !visit(&[t as u16]) {
                return;
            }
        }
    }
}

/// Visits every target atom that is the image of pattern atom 0 in at least
/// one match (each such atom once, in increasing index order), restricted to
/// atoms for which `candidate` returns true. Equivalent to collecting
/// `m[0]` over all matches, but runs one rooted existence search per atom.
pub fn visit_start_atoms_target<C, F>(iso_target: &IsoTarget, pattern: &QueryGraph, mut candidate: C, mut visit: F)
where
    C: FnMut(u16) -> bool,
    F: FnMut(u16) -> bool,
{
    if pattern.atoms.is_empty() {
        return;
    }
    let Some(mut state) = MatchState::new(iso_target, pattern, true, true) else {
        return;
    };
    if let Some(z) = state.cq.filters[0].required_element {
        for &root in state.topo.atoms_of_element(z) {
            if candidate(root) && state.rooted_exists(root as usize) && !visit(root) {
                return;
            }
        }
        return;
    }
    for root in 0..state.n_t {
        if candidate(root as u16) && state.rooted_exists(root) && !visit(root as u16) {
            return;
        }
    }
}

/// Finds all substructure matches.
///
/// Returns a vector of mappings, where each mapping is a vector
/// of target atom indices corresponding to pattern atoms.
pub fn find_all_matches(target: &Molecule, pattern: &QueryGraph) -> Vec<Vec<u16>> {
    find_all_matches_target(&IsoTarget::new(target), pattern)
}

pub fn find_all_matches_target(iso_target: &IsoTarget, pattern: &QueryGraph) -> Vec<Vec<u16>> {
    let mut matches = Vec::new();
    visit_matches_target(iso_target, pattern, |m| {
        matches.push(m.to_vec());
        true
    });
    matches
}

/// Set of atomic numbers (0..=255).
#[derive(Clone, Copy, PartialEq, Eq)]
struct ElemSet([u64; 4]);

impl ElemSet {
    const ALL: ElemSet = ElemSet([u64::MAX; 4]);
    const NONE: ElemSet = ElemSet([0; 4]);

    fn single(z: u8) -> Self {
        let mut s = Self::NONE;
        s.0[(z >> 6) as usize] |= 1 << (z & 63);
        s
    }

    #[inline]
    fn contains(&self, z: u8) -> bool {
        (self.0[(z >> 6) as usize] >> (z & 63)) & 1 != 0
    }

    fn and(self, o: Self) -> Self {
        ElemSet(std::array::from_fn(|i| self.0[i] & o.0[i]))
    }

    fn or(self, o: Self) -> Self {
        ElemSet(std::array::from_fn(|i| self.0[i] | o.0[i]))
    }

    fn not(self) -> Self {
        ElemSet(std::array::from_fn(|i| !self.0[i]))
    }

    fn is_all(&self) -> bool {
        *self == Self::ALL
    }

    fn intersects(&self, o: &Self) -> bool {
        (0..4).any(|i| self.0[i] & o.0[i] != 0)
    }
}

/// Over-approximation of the atomic numbers an atom matching `p` can have.
fn primitive_elements(p: &AtomPrimitive) -> ElemSet {
    match p {
        AtomPrimitive::Element(z) | AtomPrimitive::Aromatic(z) | AtomPrimitive::Aliphatic(z) => ElemSet::single(*z),
        AtomPrimitive::Not(inner) => match inner.as_ref() {
            AtomPrimitive::Element(z) => ElemSet::single(*z).not(),
            _ => ElemSet::ALL,
        },
        AtomPrimitive::And(items) => items.iter().fold(ElemSet::ALL, |acc, i| acc.and(primitive_elements(i))),
        AtomPrimitive::Or(items) => items.iter().fold(ElemSet::NONE, |acc, i| acc.or(primitive_elements(i))),
        AtomPrimitive::Recursive(sub) => sub.atoms.first().map_or(ElemSet::ALL, query_atom_elements),
        _ => ElemSet::ALL,
    }
}

fn query_atom_elements(q: &QueryAtom) -> ElemSet {
    q.primitives.iter().fold(ElemSet::ALL, |acc, p| acc.and(primitive_elements(p)))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AromReq {
    Any,
    Aromatic,
    Aliphatic,
    Never,
}

/// `Some((set, req))` when matching `p` is exactly `z in set` plus the
/// aromaticity requirement `req`.
fn primitive_exact_elements(p: &AtomPrimitive) -> Option<(ElemSet, AromReq)> {
    fn and(a: (ElemSet, AromReq), b: (ElemSet, AromReq)) -> (ElemSet, AromReq) {
        let req = match (a.1, b.1) {
            (AromReq::Never, _) | (_, AromReq::Never) => AromReq::Never,
            (AromReq::Any, x) | (x, AromReq::Any) => x,
            (x, y) if x == y => x,
            _ => AromReq::Never,
        };
        (a.0.and(b.0), req)
    }
    Some(match p {
        AtomPrimitive::Element(z) => (ElemSet::single(*z), AromReq::Any),
        AtomPrimitive::Aromatic(z) => (ElemSet::single(*z), AromReq::Aromatic),
        AtomPrimitive::Aliphatic(z) => (ElemSet::single(*z), AromReq::Aliphatic),
        AtomPrimitive::AnyAtom => (ElemSet::ALL, AromReq::Any),
        AtomPrimitive::AnyAromatic => (ElemSet::ALL, AromReq::Aromatic),
        AtomPrimitive::AnyAliphatic => (ElemSet::ALL, AromReq::Aliphatic),
        AtomPrimitive::Not(inner) => match inner.as_ref() {
            AtomPrimitive::Element(z) => (ElemSet::single(*z).not(), AromReq::Any),
            AtomPrimitive::AnyAromatic => (ElemSet::ALL, AromReq::Aliphatic),
            AtomPrimitive::AnyAliphatic => (ElemSet::ALL, AromReq::Aromatic),
            _ => return None,
        },
        AtomPrimitive::And(items) => {
            let mut acc = (ElemSet::ALL, AromReq::Any);
            for i in items {
                acc = and(acc, primitive_exact_elements(i)?);
            }
            acc
        }
        AtomPrimitive::Or(items) => {
            let mut iter = items.iter();
            let mut acc = primitive_exact_elements(iter.next()?)?;
            for i in iter {
                let (set, req) = primitive_exact_elements(i)?;
                if req != acc.1 {
                    return None;
                }
                acc.0 = acc.0.or(set);
            }
            acc
        }
        _ => return None,
    })
}

fn query_atom_exact_elements(q: &QueryAtom) -> Option<(ElemSet, AromReq)> {
    let mut acc = (ElemSet::ALL, AromReq::Any);
    for p in &q.primitives {
        let (set, req) = primitive_exact_elements(p)?;
        acc.0 = acc.0.and(set);
        acc.1 = match (acc.1, req) {
            (AromReq::Never, _) | (_, AromReq::Never) => AromReq::Never,
            (AromReq::Any, x) | (x, AromReq::Any) => x,
            (x, y) if x == y => x,
            _ => AromReq::Never,
        };
    }
    Some(acc)
}

/// True when the query atom is a plain conjunction of element / aromaticity /
/// charge primitives without conflicts, i.e. exactly what `AtomFastFilter`
/// derived from it tests.
fn filter_is_exact(query_atom: &QueryAtom) -> bool {
    fn collect(p: &AtomPrimitive, elems: &mut SmallVec<[u8; 4]>, charges: &mut SmallVec<[i8; 2]>) -> bool {
        match p {
            AtomPrimitive::AnyAtom | AtomPrimitive::AnyAromatic | AtomPrimitive::AnyAliphatic => true,
            AtomPrimitive::Element(z) | AtomPrimitive::Aromatic(z) | AtomPrimitive::Aliphatic(z) => {
                elems.push(*z);
                true
            }
            AtomPrimitive::Charge(c) => {
                charges.push(*c);
                true
            }
            AtomPrimitive::And(items) => items.iter().all(|i| collect(i, elems, charges)),
            _ => false,
        }
    }
    let mut elems = SmallVec::new();
    let mut charges = SmallVec::new();
    query_atom.primitives.iter().all(|p| collect(p, &mut elems, &mut charges))
        && elems.windows(2).all(|w| w[0] == w[1])
        && charges.windows(2).all(|w| w[0] == w[1])
}

#[derive(Debug, Clone, Copy, Default)]
struct AtomFastFilter {
    required_element: Option<u8>,
    requires_aromatic: bool,
    requires_aliphatic: bool,
    required_charge: Option<i8>,
}

fn target_element_counts(target: &Molecule) -> [u16; 119] {
    let mut target_counts = [0u16; 119];
    for atom in &target.atoms {
        let idx = atom.atomic_number as usize;
        if idx < target_counts.len() {
            target_counts[idx] = target_counts[idx].saturating_add(1);
        }
    }
    target_counts
}

pub(crate) struct TargetAtomStats {
    pub(crate) aromatic_atoms: u16,
    pub(crate) aliphatic_atoms: u16,
    pub(crate) aromatic_bonds: u16,
    pub(crate) ring_bonds: u16,
}

impl TargetAtomStats {
    pub(crate) fn from_molecule(target: &Molecule) -> Self {
        let mut aromatic_atoms = 0u16;
        let mut aromatic_bonds = 0u16;
        let mut ring_bonds = 0u16;

        for atom in &target.atoms {
            if atom.is_aromatic() {
                aromatic_atoms = aromatic_atoms.saturating_add(1);
            }
        }

        for bond in &target.bonds {
            if bond.is_aromatic() {
                aromatic_bonds = aromatic_bonds.saturating_add(1);
            }
            if bond.is_in_ring() {
                ring_bonds = ring_bonds.saturating_add(1);
            }
        }

        let total_atoms = target.atoms.len().min(u16::MAX as usize) as u16;
        let aliphatic_atoms = total_atoms.saturating_sub(aromatic_atoms);

        Self {
            aromatic_atoms,
            aliphatic_atoms,
            aromatic_bonds,
            ring_bonds,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QueryScreen {
    pub min_atoms: usize,
    pub min_bonds: usize,
    pub required_elements: smallvec::SmallVec<[(u8, u16); 8]>,
    pub requires_aromatic_atom: bool,
}

impl QueryScreen {
    pub fn build(query: &crate::io::smarts::QueryGraph) -> Self {
        use crate::io::smarts::{QueryAtom, AtomPrimitive};

        fn query_primitive_requires_aromatic(p: &AtomPrimitive) -> bool {
            match p {
                AtomPrimitive::AnyAromatic | AtomPrimitive::Aromatic(_) => true,
                AtomPrimitive::Not(_) => false,
                AtomPrimitive::And(reqs) => reqs.iter().any(query_primitive_requires_aromatic),
                AtomPrimitive::Or(opts) => !opts.is_empty() && opts.iter().all(query_primitive_requires_aromatic),
                AtomPrimitive::Recursive(sub_pattern) => sub_pattern.atoms.first().is_some_and(query_atom_requires_aromatic),
                _ => false,
            }
        }

        fn query_atom_requires_aromatic(query_atom: &QueryAtom) -> bool {
            query_atom.primitives.iter().any(query_primitive_requires_aromatic)
        }

        fn fixed_atomic_number_primitive(primitive: &AtomPrimitive) -> Option<u8> {
            match primitive {
                AtomPrimitive::Element(z) | AtomPrimitive::Aromatic(z) | AtomPrimitive::Aliphatic(z) => Some(*z),
                AtomPrimitive::And(items) => {
                    let mut fixed = None;
                    for item in items {
                        let Some(element) = fixed_atomic_number_primitive(item) else { continue; };
                        if let Some(prev) = fixed {
                            if prev != element { return None; }
                        } else {
                            fixed = Some(element);
                        }
                    }
                    fixed
                }
                AtomPrimitive::Or(options) => {
                    // every branch must fix the same element ([C,*] fixes nothing)
                    let first = fixed_atomic_number_primitive(options.first()?)?;
                    options.iter().all(|o| fixed_atomic_number_primitive(o) == Some(first)).then_some(first)
                }
                _ => None,
            }
        }

        fn fixed_atomic_number(query_atom: &QueryAtom) -> Option<u8> {
            let mut fixed = None;
            for primitive in &query_atom.primitives {
                let Some(element) = fixed_atomic_number_primitive(primitive) else { continue; };
                if let Some(prev) = fixed {
                    if prev != element { return None; }
                } else {
                    fixed = Some(element);
                }
            }
            fixed
        }

        let mut required = [0u16; 119];
        let mut requires_aromatic_atom = false;
        for atom in &query.atoms {
            if let Some(element) = fixed_atomic_number(atom) {
                let idx = element as usize;
                if idx < required.len() {
                    required[idx] = required[idx].saturating_add(1);
                }
            }
            requires_aromatic_atom |= query_atom_requires_aromatic(atom);
        }
        let required_elements = required
            .iter()
            .enumerate()
            .filter_map(|(idx, &count)| (count > 0).then_some((idx as u8, count)))
            .collect();
        Self {
            min_atoms: query.atoms.len(),
            min_bonds: query.bonds.len(),
            required_elements,
            requires_aromatic_atom,
        }
    }

    pub(crate) fn passes(&self, target_stats: &TargetAtomStats, target_counts: &[u16; 119], target_num_atoms: usize, target_num_bonds: usize) -> bool {
        if self.min_atoms > target_num_atoms || self.min_bonds > target_num_bonds {
            return false;
        }
        if self.requires_aromatic_atom && target_stats.aromatic_atoms == 0 {
            return false;
        }
        self.required_elements
            .iter()
            .all(|&(element, count)| target_counts[element as usize] >= count)
    }
    
    pub fn passes_target(&self, iso_target: &IsoTarget) -> bool {
        self.passes(&iso_target.target_stats, &iso_target.target_counts, iso_target.target.num_atoms(), iso_target.target.num_bonds())
    }
}

/// Element every match of the query atom must have, if any.
///
/// This is the (target-independent) estimate used for the search order; it
/// deliberately ignores recursive SMARTS so the order (and thereby the match
/// enumeration order) stays as before.
fn fixed_atomic_number(query_atom: &QueryAtom) -> Option<u8> {
    fixed_atomic_number_of(query_atom, false)
}

/// Like `fixed_atomic_number`, but also looks through `$(...)` (a recursive
/// SMARTS can only match atoms matching its first atom). Used for filtering.
fn fixed_atomic_number_filter(query_atom: &QueryAtom) -> Option<u8> {
    fixed_atomic_number_of(query_atom, true)
}

fn fixed_atomic_number_of(query_atom: &QueryAtom, recursive: bool) -> Option<u8> {
    let mut fixed = None;

    for primitive in &query_atom.primitives {
        let Some(element) = fixed_atomic_number_primitive(primitive, recursive) else {
            continue;
        };

        if let Some(prev) = fixed {
            if prev != element {
                return None;
            }
        } else {
            fixed = Some(element);
        }
    }

    fixed
}

fn fixed_atomic_number_primitive(primitive: &AtomPrimitive, recursive: bool) -> Option<u8> {
    match primitive {
        AtomPrimitive::Element(z) | AtomPrimitive::Aromatic(z) | AtomPrimitive::Aliphatic(z) => {
            Some(*z)
        }
        AtomPrimitive::And(requirements) => {
            let mut fixed = None;
            for requirement in requirements {
                let Some(element) = fixed_atomic_number_primitive(requirement, recursive) else {
                    continue;
                };
                if let Some(prev) = fixed {
                    if prev != element {
                        return None;
                    }
                } else {
                    fixed = Some(element);
                }
            }
            fixed
        }
        AtomPrimitive::Or(options) => {
            // every branch must fix the same element ([C,*] fixes nothing)
            let first = fixed_atomic_number_primitive(options.first()?, recursive)?;
            options
                .iter()
                .all(|o| fixed_atomic_number_primitive(o, recursive) == Some(first))
                .then_some(first)
        }
        AtomPrimitive::Recursive(sub) if recursive => {
            sub.atoms.first().and_then(|a| fixed_atomic_number_of(a, true))
        }
        _ => None,
    }
}

fn query_atom_required_charge(query_atom: &QueryAtom) -> Option<i8> {
    let mut required_charge: Option<i8> = None;
    for primitive in &query_atom.primitives {
        let Some(charge) = required_charge_from_primitive(primitive) else {
            continue;
        };
        match required_charge {
            Some(prev) if prev != charge => return None,
            Some(_) => {}
            None => required_charge = Some(charge),
        }
    }
    required_charge
}

fn required_charge_from_primitive(primitive: &AtomPrimitive) -> Option<i8> {
    match primitive {
        AtomPrimitive::Charge(c) => Some(*c),
        AtomPrimitive::And(requirements) => {
            let mut required = None;
            for req in requirements {
                let Some(charge) = required_charge_from_primitive(req) else {
                    continue;
                };
                match required {
                    Some(prev) if prev != charge => return None,
                    Some(_) => {}
                    None => required = Some(charge),
                }
            }
            required
        }
        AtomPrimitive::Or(options) => {
            // OR branches that include charge constraints are not globally
            // charge-fixed unless all branches constrain to the same charge.
            // If any branch lacks an explicit charge primitive (e.g.
            // [$([N+R]),$([n+R]),$([N+]=C)]), we must not derive a fixed charge
            // for fast filtering; otherwise valid neutral alternatives can be
            // rejected before full SMARTS evaluation.
            if options.iter().any(|opt| !primitive_contains_charge(opt)) {
                return None;
            }
            let mut options = options.iter().filter_map(required_charge_from_primitive);
            let first = options.next()?;
            if options.all(|charge| charge == first) {
                Some(first)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn primitive_contains_charge(primitive: &AtomPrimitive) -> bool {
    match primitive {
        AtomPrimitive::Charge(_) => true,
        AtomPrimitive::And(requirements) | AtomPrimitive::Or(requirements) => {
            requirements.iter().any(primitive_contains_charge)
        }
        AtomPrimitive::Not(inner) => primitive_contains_charge(inner),
        _ => false,
    }
}

fn query_atom_requires_aromatic(query_atom: &QueryAtom) -> bool {
    query_atom
        .primitives
        .iter()
        .any(atom_primitive_requires_aromatic)
}

fn query_atom_requires_aliphatic(query_atom: &QueryAtom) -> bool {
    query_atom
        .primitives
        .iter()
        .any(atom_primitive_requires_aliphatic)
}

fn atom_primitive_requires_aromatic(primitive: &AtomPrimitive) -> bool {
    match primitive {
        AtomPrimitive::AnyAromatic | AtomPrimitive::Aromatic(_) => true,
        AtomPrimitive::And(requirements) => {
            requirements.iter().any(atom_primitive_requires_aromatic)
        }
        AtomPrimitive::Or(options) => {
            !options.is_empty() && options.iter().all(atom_primitive_requires_aromatic)
        }
        AtomPrimitive::Recursive(sub) => sub.atoms.first().is_some_and(query_atom_requires_aromatic),
        _ => false,
    }
}

fn atom_primitive_requires_aliphatic(primitive: &AtomPrimitive) -> bool {
    match primitive {
        AtomPrimitive::AnyAliphatic | AtomPrimitive::Aliphatic(_) => true,
        AtomPrimitive::And(requirements) => {
            requirements.iter().any(atom_primitive_requires_aliphatic)
        }
        AtomPrimitive::Or(options) => {
            !options.is_empty() && options.iter().all(atom_primitive_requires_aliphatic)
        }
        AtomPrimitive::Recursive(sub) => sub.atoms.first().is_some_and(query_atom_requires_aliphatic),
        _ => false,
    }
}

fn query_bond_requires_ring(query_bond: &QueryBond) -> bool {
    match query_bond {
        QueryBond::Ring => true,
        QueryBond::And(parts) => parts.iter().any(query_bond_requires_ring),
        QueryBond::Or(parts) => !parts.is_empty() && parts.iter().all(query_bond_requires_ring),
        _ => false,
    }
}

fn query_bond_requires_aromatic(query_bond: &QueryBond) -> bool {
    match query_bond {
        QueryBond::Aromatic => true,
        QueryBond::And(parts) => parts.iter().any(query_bond_requires_aromatic),
        QueryBond::Or(parts) => !parts.is_empty() && parts.iter().all(query_bond_requires_aromatic),
        _ => false,
    }
}

/// Helper to create simple query graphs for testing.
impl QueryGraph {
    /// Creates a query that matches a single atom of given element.
    pub fn single_atom(element: u8) -> Self {
        Self {
            atoms: vec![QueryAtom::element(element)],
            bonds: vec![],
            adjacency: vec![smallvec::SmallVec::new()],
            needs_ring_context: false,
            meta: Default::default(),
        }
    }

    /// Creates a query for two bonded atoms.
    pub fn two_atoms(elem1: u8, elem2: u8, bond: QueryBond) -> Self {
        use crate::io::smarts::QueryGraphBond;

        Self {
            atoms: vec![QueryAtom::element(elem1), QueryAtom::element(elem2)],
            bonds: vec![QueryGraphBond {
                source: 0,
                target: 1,
                query: bond,
            }],
            adjacency: vec![smallvec::smallvec![0], smallvec::smallvec![0]],
            needs_ring_context: false,
            meta: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smarts::{QueryBond, parse_smarts};
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_single_atom_match() {
        let mol = parse_smiles("CCO").unwrap();

        // Match carbon
        let pattern = QueryGraph::single_atom(6);
        assert!(has_substruct_match(&mol, &pattern));

        let matches = find_all_matches(&mol, &pattern);
        assert_eq!(matches.len(), 2); // Two carbons

        // Match oxygen
        let pattern = QueryGraph::single_atom(8);
        assert!(has_substruct_match(&mol, &pattern));

        let matches = find_all_matches(&mol, &pattern);
        assert_eq!(matches.len(), 1); // One oxygen
    }

    #[test]
    fn test_no_match() {
        let mol = parse_smiles("CCO").unwrap();

        // No nitrogen in molecule
        let pattern = QueryGraph::single_atom(7);
        assert!(!has_substruct_match(&mol, &pattern));
    }

    #[test]
    fn test_two_atom_match() {
        let mol = parse_smiles("CCO").unwrap();

        // Match C-C bond
        let pattern = QueryGraph::two_atoms(6, 6, QueryBond::Single);
        assert!(has_substruct_match(&mol, &pattern));

        // Match C-O bond
        let pattern = QueryGraph::two_atoms(6, 8, QueryBond::Single);
        assert!(has_substruct_match(&mol, &pattern));
    }

    #[test]
    fn test_count_matches() {
        let mol = parse_smiles("CCCC").unwrap(); // Butane

        // Count carbons
        let pattern = QueryGraph::single_atom(6);
        assert_eq!(count_matches(&mol, &pattern), 4);

        // Count C-C bonds: matches covering the same atoms count once (RDKit uniquify)
        let pattern = QueryGraph::two_atoms(6, 6, QueryBond::Single);
        assert_eq!(count_matches(&mol, &pattern), 3);
    }

    #[test]
    fn test_has_at_least_matches_thresholds() {
        let mol = parse_smiles("CCCC").unwrap();
        let carbon = QueryGraph::single_atom(6);
        assert!(has_at_least_matches(&mol, &carbon, 0));
        assert!(has_at_least_matches(&mol, &carbon, 1));
        assert!(has_at_least_matches(&mol, &carbon, 4));
        assert!(!has_at_least_matches(&mol, &carbon, 5));

        let cc_bond = QueryGraph::two_atoms(6, 6, QueryBond::Single);
        assert!(has_at_least_matches(&mol, &cc_bond, 3));
        assert!(!has_at_least_matches(&mol, &cc_bond, 4));
    }

    #[test]
    fn test_visit_matches_early_stop() {
        let mol = parse_smiles("CCCC").unwrap();
        let pattern = QueryGraph::single_atom(6);
        let mut visited = 0usize;

        visit_matches(&mol, &pattern, |_| {
            visited += 1;
            visited < 2
        });

        assert_eq!(visited, 2);
    }

    #[test]
    fn test_visit_matches_empty_pattern_once() {
        let mol = parse_smiles("CC").unwrap();
        let empty = QueryGraph {
            atoms: vec![],
            bonds: vec![],
            adjacency: vec![],
            needs_ring_context: false,
            meta: Default::default(),
        };

        let mut visited = 0usize;
        visit_matches(&mol, &empty, |_| {
            visited += 1;
            true
        });

        assert_eq!(visited, 1);
    }

    #[test]
    fn test_double_bond_match() {
        let mol = parse_smiles("C=C").unwrap(); // Ethene

        // Should match double bond
        let pattern = QueryGraph::two_atoms(6, 6, QueryBond::Double);
        assert!(has_substruct_match(&mol, &pattern));

        // Should not match single bond
        let pattern = QueryGraph::two_atoms(6, 6, QueryBond::Single);
        assert!(!has_substruct_match(&mol, &pattern));
    }

    #[test]
    fn test_aromatic_pattern() {

        let mol = parse_smiles("c1ccccc1").unwrap(); // Benzene

        // Should match aromatic carbon
        let pattern = QueryGraph {
            atoms: vec![QueryAtom::aromatic()],
            bonds: vec![],
            adjacency: vec![smallvec::SmallVec::new()],
            needs_ring_context: false,
            meta: Default::default(),
        };

        assert!(has_substruct_match(&mol, &pattern));
        assert_eq!(count_matches(&mol, &pattern), 6); // 6 aromatic carbons
    }

    #[test]
    fn test_complex_pattern() {
        let mol = parse_smiles("CCO").unwrap(); // Ethanol

        // Find first match returns valid mapping
        let pattern = QueryGraph::two_atoms(6, 8, QueryBond::Single);
        let mapping = find_first_match(&mol, &pattern);

        assert!(mapping.is_some());
        let mapping = mapping.unwrap();
        assert_eq!(mapping.len(), 2);

        // Verify mapped atoms are carbon and oxygen
        let target = &mol;
        assert_eq!(target.atoms[mapping[0] as usize].atomic_number, 6); // Carbon
        assert_eq!(target.atoms[mapping[1] as usize].atomic_number, 8); // Oxygen
    }

    #[test]
    fn test_ring_size_and_membership_primitives() {
        let mol = parse_smiles("C1CCCCC1").unwrap();
        crate::algo::rings::find_sssr(&mol);

        let membership = parse_smarts("[CR1]").unwrap();
        assert_eq!(count_matches(&mol, &membership), 6);

        let ring_size = parse_smarts("[C;r6]").unwrap();
        assert_eq!(count_matches(&mol, &ring_size), 6);
    }

    #[test]
    fn test_or_charge_fast_filter_does_not_reject_neutral_branches() {
        // This pattern includes an OR branch (`[N+]=C`) where the Charge primitive
        // is nested under an AND node. Prefilter charge extraction must remain
        // branch-sensitive and semantically aligned with full SMARTS matching.
        let pattern = parse_smarts("[$([N+R]),$([n+R]),$([N+]=C)]").unwrap();

        let charged_imine = parse_smiles("C=[N+]C").unwrap();
        assert!(
            has_substruct_match(&charged_imine, &pattern),
            "charged imine branch must remain matchable"
        );

        let ring_cation = parse_smiles("[n+]1ccccc1").unwrap();
        assert!(
            has_substruct_match(&ring_cation, &pattern),
            "charged aromatic branch must remain matchable"
        );

        let neutral_imine = parse_smiles("NC=O").unwrap();
        assert!(
            !has_substruct_match(&neutral_imine, &pattern),
            "neutral imine should not match this charged-only OR pattern"
        );
    }
}

