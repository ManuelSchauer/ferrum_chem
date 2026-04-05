//! Iso algorithms.
//!
//! This module contains core cheminformatics graph logic used by parsers and
//! descriptors. The implementations prioritize determinism, correctness, and
//! predictable performance on large molecular batches.
//!
//! Learning focus:
//! - Graph traversal and indexing patterns over atom/bond adjacency.
//! - Performance-oriented Rust design in numerically heavy code paths.

//! ## File guide: Iso
//!
//! This file belongs to FerrumChem's **cheminformatics algorithm layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn has_substruct_match`
//! - `fn find_first_match`
//! - `fn find_first_match_with_ring_ctx`
//! - `fn count_matches`
//! - `fn has_at_least_matches`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - VF2 feasibility checks rely on aromatic flags, atom degree, and bond compatibility being precomputed consistently.
//! - Small-target cache paths (<=64 atoms) assume dense bit-level adjacency checks are faster than repeated linear scans.
//! - Match enumeration is deterministic for a fixed molecule/query ordering, which is required by parity diagnostics.
//! ### Chemistry/ML references
//! - Cordella et al. (2001/2004). VF2 graph/subgraph isomorphism algorithm.
//! - RDKit substructure matching behavior for aromatic, valence, and ring-query compatibility baselines.
//! - Ullmann/VF-family feasibility-pruning literature for chemical graph matching.

use crate::io::smarts::{
    AtomPrimitive, QueryAtom, QueryBond, QueryGraph, SmartsRecursiveCache, SmartsRingContext,
    query_atom_matches_with_cache, query_atom_matches_with_context, query_graph_needs_ring_context,
};
use crate::mol::molecule::Molecule;
use smallvec::SmallVec;
use std::borrow::Cow;

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
    has_substruct_match_with_ring_ctx(target, pattern, None)
}

/// Checks whether substruct match with ring ctx.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub(crate) fn has_substruct_match_with_ring_ctx(
    target: &Molecule,
    pattern: &QueryGraph,
    ring_ctx: Option<&SmartsRingContext>,
) -> bool {
    find_first_match_with_ring_ctx(target, pattern, ring_ctx).is_some()
}

/// Finds the first substructure match and returns the atom mapping.
///
/// Returns a vector where index i is the pattern atom index,
/// and the value is the corresponding target atom index.
pub fn find_first_match(target: &Molecule, pattern: &QueryGraph) -> Option<Vec<u16>> {
    find_first_match_with_ring_ctx(target, pattern, None)
}

/// Finds first match with ring ctx.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn find_first_match_with_ring_ctx(
    target: &Molecule,
    pattern: &QueryGraph,
    ring_ctx: Option<&SmartsRingContext>,
) -> Option<Vec<u16>> {
    // Empty pattern matches everything
    if pattern.atoms.is_empty() {
        return Some(vec![]);
    }

    // Pattern larger than target cannot match
    if pattern.atoms.len() > target.atoms.len() {
        return None;
    }
    if pattern.bonds.len() > target.bonds.len() {
        return None;
    }

    if let Some(fast_path) = detect_fast_path(pattern) {
        let mut first = None;
        visit_fast_path_matches(target, pattern, fast_path, ring_ctx, |mapping| {
            first = Some(mapping.to_vec());
            false
        });
        return first;
    }

    let mut state = MatchState::new(target, pattern, MatchMode::FirstMatch, ring_ctx);
    if !state.overall_possible {
        return None;
    }

    // Try matching starting from each atom in the target
    if backtrack(&mut state, 0) {
        Some(state.mapping.clone())
    } else {
        None
    }
}

/// Counts the number of substructure matches.
pub fn count_matches(target: &Molecule, pattern: &QueryGraph) -> usize {
    if pattern.atoms.is_empty() {
        return 1;
    }
    if pattern.atoms.len() > target.atoms.len() {
        return 0;
    }
    if pattern.bonds.len() > target.bonds.len() {
        return 0;
    }

    if let Some(fast_path) = detect_fast_path(pattern) {
        let mut count = 0usize;
        visit_fast_path_matches(target, pattern, fast_path, None, |_| {
            count += 1;
            true
        });
        return count;
    }

    let mut state = MatchState::new(target, pattern, MatchMode::Enumerate, None);
    if !state.overall_possible {
        return 0;
    }

    let mut count = 0usize;
    backtrack_count(&mut state, 0, &mut count);
    count
}

/// Returns true if the pattern has at least `min_matches` matches.
///
/// Uses early-stop enumeration, so it avoids counting all matches when only
/// a threshold decision is needed.
pub fn has_at_least_matches(target: &Molecule, pattern: &QueryGraph, min_matches: usize) -> bool {
    if min_matches == 0 {
        return true;
    }
    if min_matches == 1 {
        return has_substruct_match(target, pattern);
    }
    let mut matched = 0usize;
    visit_matches(target, pattern, |_| {
        matched += 1;
        matched < min_matches
    });
    matched >= min_matches
}

/// Visits all substructure matches without materializing them.
///
/// The visitor returns `true` to continue enumeration or `false` to stop early.
pub fn visit_matches<F>(target: &Molecule, pattern: &QueryGraph, mut visit: F)
where
    F: FnMut(&[u16]) -> bool,
{
    visit_matches_with_ring_ctx(target, pattern, None, &mut visit);
}

/// Implements matches with ring ctx.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub(crate) fn visit_matches_with_ring_ctx<F>(
    target: &Molecule,
    pattern: &QueryGraph,
    ring_ctx: Option<&SmartsRingContext>,
    mut visit: F,
) where
    F: FnMut(&[u16]) -> bool,
{
    if pattern.atoms.is_empty() {
        let _ = visit(&[]);
        return;
    }

    if pattern.atoms.len() > target.atoms.len() {
        return;
    }
    if pattern.bonds.len() > target.bonds.len() {
        return;
    }

    if let Some(fast_path) = detect_fast_path(pattern) {
        visit_fast_path_matches(target, pattern, fast_path, ring_ctx, visit);
        return;
    }

    let mut state = MatchState::new(target, pattern, MatchMode::FirstMatch, ring_ctx);
    if !state.overall_possible {
        return;
    }
    let _ = backtrack_visit(&mut state, 0, &mut visit);
}

/// Finds all substructure matches.
///
/// Returns a vector of mappings, where each mapping is a vector
/// of target atom indices corresponding to pattern atoms.
pub fn find_all_matches(target: &Molecule, pattern: &QueryGraph) -> Vec<Vec<u16>> {
    if pattern.atoms.is_empty() {
        return vec![vec![]];
    }

    if pattern.atoms.len() > target.atoms.len() {
        return vec![];
    }
    if pattern.bonds.len() > target.bonds.len() {
        return vec![];
    }

    if let Some(fast_path) = detect_fast_path(pattern) {
        let mut matches = Vec::new();
        visit_fast_path_matches(target, pattern, fast_path, None, |mapping| {
            matches.push(mapping.to_vec());
            true
        });
        return matches;
    }

    let mut matches = Vec::new();
    let mut state = MatchState::new(target, pattern, MatchMode::Enumerate, None);
    if !state.overall_possible {
        return matches;
    }

    backtrack_all(&mut state, 0, &mut matches);

    matches
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FastPathKind {
    SingleAtom,
    SingleBond {
        pattern_source: usize,
        pattern_target: usize,
        pattern_bond_idx: u16,
    },
}

/// Implements fast path.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn detect_fast_path(pattern: &QueryGraph) -> Option<FastPathKind> {
    if pattern.atoms.len() == 1 && pattern.bonds.is_empty() {
        return Some(FastPathKind::SingleAtom);
    }

    if pattern.atoms.len() == 2 && pattern.bonds.len() == 1 {
        let bond = &pattern.bonds[0];
        return Some(FastPathKind::SingleBond {
            pattern_source: bond.source as usize,
            pattern_target: bond.target as usize,
            pattern_bond_idx: 0,
        });
    }

    None
}

/// Implements fast path matches.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn visit_fast_path_matches<F>(
    target: &Molecule,
    pattern: &QueryGraph,
    fast_path: FastPathKind,
    ring_ctx: Option<&SmartsRingContext>,
    mut visit: F,
) where
    F: FnMut(&[u16]) -> bool,
{
    let ring_ctx_owned = if query_graph_needs_ring_context(pattern) {
        if let Some(ctx) = ring_ctx {
            Some(Cow::Borrowed(ctx))
        } else {
            Some(Cow::Owned(SmartsRingContext::for_molecule(target)))
        }
    } else {
        None
    };
    let recursive_cache = SmartsRecursiveCache::default();

    match fast_path {
        FastPathKind::SingleAtom => {
            let mut mapping = [0u16; 1];
            let query_atom = &pattern.atoms[0];
            for target_idx in 0..target.atoms.len() {
                mapping[0] = target_idx as u16;
                if query_atom_matches_with_cache(
                    query_atom,
                    target,
                    mapping[0],
                    ring_ctx_owned.as_deref(),
                    Some(&recursive_cache),
                ) && !visit(&mapping)
                {
                    return;
                }
            }
        }
        FastPathKind::SingleBond {
            pattern_source,
            pattern_target,
            pattern_bond_idx,
        } => {
            let pattern_bond = &pattern.bonds[pattern_bond_idx as usize];
            let query_source = &pattern.atoms[pattern_source];
            let query_target = &pattern.atoms[pattern_target];
            let mut mapping = [u16::MAX; 2];

            for (target_bond_idx, target_bond) in target.bonds.iter().enumerate() {
                if !pattern_bond.query.matches(target, target_bond_idx as u16) {
                    continue;
                }

                let orientations = [
                    (target_bond.source, target_bond.target),
                    (target_bond.target, target_bond.source),
                ];

                for (target_src, target_dst) in orientations {
                    if !query_atom_matches_with_cache(
                        query_source,
                        target,
                        target_src,
                        ring_ctx_owned.as_deref(),
                        Some(&recursive_cache),
                    ) {
                        continue;
                    }
                    if !query_atom_matches_with_cache(
                        query_target,
                        target,
                        target_dst,
                        ring_ctx_owned.as_deref(),
                        Some(&recursive_cache),
                    ) {
                        continue;
                    }

                    mapping[pattern_source] = target_src;
                    mapping[pattern_target] = target_dst;
                    if !visit(&mapping) {
                        return;
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchMode {
    FirstMatch,
    Enumerate,
}

#[derive(Debug, Clone)]
struct SmallTargetGraphCache {
    n_atoms: usize,
    bond_lookup: Vec<u16>,
}

impl SmallTargetGraphCache {
    /// Implements target.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn for_target(target: &Molecule) -> Option<Self> {
        let n_atoms = target.num_atoms();
        if n_atoms > 64 {
            return None;
        }
        let mut bond_lookup = vec![u16::MAX; n_atoms * n_atoms];
        for (bond_idx, bond) in target.bonds.iter().enumerate() {
            let bond_idx = bond_idx as u16;
            let src = bond.source as usize;
            let dst = bond.target as usize;
            bond_lookup[src * n_atoms + dst] = bond_idx;
            bond_lookup[dst * n_atoms + src] = bond_idx;
        }
        Some(Self {
            n_atoms,
            bond_lookup,
        })
    }

    #[inline]
    /// Implements bond.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn lookup_bond(&self, a: u16, b: u16) -> Option<u16> {
        let a = a as usize;
        let b = b as usize;
        if a >= self.n_atoms || b >= self.n_atoms {
            return None;
        }
        let idx = self.bond_lookup[a * self.n_atoms + b];
        (idx != u16::MAX).then_some(idx)
    }
}

#[derive(Debug, Clone, Copy)]
struct TargetAtomStats {
    aromatic_atoms: u16,
    aliphatic_atoms: u16,
    aromatic_bonds: u16,
    ring_bonds: u16,
}

impl TargetAtomStats {
    /// Constructs molecule between representations.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn from_molecule(target: &Molecule) -> Self {
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

#[derive(Debug, Clone, Copy, Default)]
struct AtomFastFilter {
    required_element: Option<u8>,
    requires_aromatic: bool,
    requires_aliphatic: bool,
    required_charge: Option<i8>,
}

impl AtomFastFilter {
    #[inline]
    /// Implements atom.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn matches_atom(&self, atom: &crate::mol::atom::Atom) -> bool {
        if let Some(required_element) = self.required_element {
            if atom.atomic_number != required_element {
                return false;
            }
        }
        if self.requires_aromatic && !atom.is_aromatic() {
            return false;
        }
        if self.requires_aliphatic && atom.is_aromatic() {
            return false;
        }
        if let Some(required_charge) = self.required_charge {
            if atom.formal_charge != required_charge {
                return false;
            }
        }
        true
    }
}

/// State for VF2 algorithm.
struct MatchState<'a> {
    target: &'a Molecule,
    pattern: &'a QueryGraph,
    mapping: Vec<u16>,                 // pattern_idx -> target_idx
    reverse_mapping: Vec<Option<u16>>, // target_idx -> Some(pattern_idx) or None
    ring_ctx: Option<Cow<'a, SmartsRingContext>>,
    recursive_cache: SmartsRecursiveCache,
    candidate_targets: Vec<Vec<u16>>, // pattern_idx -> candidate target atom idx
    estimated_candidate_sizes: Vec<usize>,
    search_order: Vec<usize>,       // search depth -> pattern_idx
    atom_match_cache: Vec<Vec<u8>>, // 0=unknown, 1=false, 2=true
    atom_fast_filters: Vec<AtomFastFilter>,
    small_target_cache: Option<SmallTargetGraphCache>,
    precomputed_atom_matches: bool,
    overall_possible: bool,
}

impl<'a> MatchState<'a> {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn new(
        target: &'a Molecule,
        pattern: &'a QueryGraph,
        mode: MatchMode,
        ring_ctx: Option<&'a SmartsRingContext>,
    ) -> Self {
        let ring_ctx = if query_graph_needs_ring_context(pattern) {
            if let Some(ctx) = ring_ctx {
                Some(Cow::Borrowed(ctx))
            } else {
                Some(Cow::Owned(SmartsRingContext::for_molecule(target)))
            }
        } else {
            None
        };

        let target_counts = target_element_counts(target);
        let target_stats = TargetAtomStats::from_molecule(target);
        let mut overall_possible = pattern.bonds.len() <= target.bonds.len();
        if overall_possible {
            overall_possible = passes_element_screen(pattern, &target_counts, &target_stats);
        }

        let atom_fast_filters: Vec<AtomFastFilter> = pattern
            .atoms
            .iter()
            .map(|query_atom| AtomFastFilter {
                required_element: fixed_atomic_number(query_atom),
                requires_aromatic: query_atom_requires_aromatic(query_atom),
                requires_aliphatic: query_atom_requires_aliphatic(query_atom),
                required_charge: query_atom_required_charge(query_atom),
            })
            .collect();

        if overall_possible {
            overall_possible = atom_fast_filters
                .iter()
                .all(|filter| !(filter.requires_aromatic && filter.requires_aliphatic));
        }

        let precomputed_atom_matches = overall_possible && mode == MatchMode::Enumerate;

        let mut candidate_targets = Vec::with_capacity(pattern.atoms.len());
        let mut estimated_candidate_sizes = Vec::with_capacity(pattern.atoms.len());
        if precomputed_atom_matches {
            for (pattern_idx, query_atom) in pattern.atoms.iter().enumerate() {
                let filter = atom_fast_filters[pattern_idx];
                let mut candidates = Vec::new();
                for target_idx in 0..target.atoms.len() {
                    let target_idx = target_idx as u16;
                    if !filter.matches_atom(&target.atoms[target_idx as usize]) {
                        continue;
                    }
                    if query_atom_matches_with_context(
                        query_atom,
                        target,
                        target_idx,
                        ring_ctx.as_deref(),
                    ) {
                        candidates.push(target_idx);
                    }
                }
                if candidates.is_empty() {
                    overall_possible = false;
                }
                estimated_candidate_sizes.push(candidates.len());
                candidate_targets.push(candidates);
            }
        } else {
            candidate_targets.resize(pattern.atoms.len(), Vec::new());
            estimated_candidate_sizes.extend(pattern.atoms.iter().map(|query_atom| {
                fixed_atomic_number(query_atom)
                    .map(|z| target_counts[z as usize] as usize)
                    .unwrap_or(target.atoms.len())
            }));
        }

        let mut search_order: Vec<usize> = (0..pattern.atoms.len()).collect();
        search_order.sort_by_key(|&pattern_idx| {
            (
                estimated_candidate_sizes[pattern_idx],
                usize::MAX - pattern.adjacency[pattern_idx].len(),
                pattern_idx,
            )
        });

        let atom_match_cache = if precomputed_atom_matches {
            Vec::new()
        } else {
            vec![vec![0u8; target.atoms.len()]; pattern.atoms.len()]
        };
        Self {
            target,
            pattern,
            mapping: vec![u16::MAX; pattern.atoms.len()],
            reverse_mapping: vec![None; target.atoms.len()],
            ring_ctx,
            recursive_cache: SmartsRecursiveCache::default(),
            candidate_targets,
            estimated_candidate_sizes,
            search_order,
            atom_match_cache,
            atom_fast_filters,
            small_target_cache: SmallTargetGraphCache::for_target(target),
            precomputed_atom_matches,
            overall_possible,
        }
    }

    /// Checks if a pattern atom can be mapped to a target atom.
    fn is_feasible(&mut self, pattern_idx: usize, target_idx: u16) -> bool {
        // Check if target atom already mapped
        if self.reverse_mapping[target_idx as usize].is_some() {
            return false;
        }

        if !self.atom_fast_filters[pattern_idx]
            .matches_atom(&self.target.atoms[target_idx as usize])
        {
            return false;
        }

        if !self.precomputed_atom_matches && !self.atom_matches_cached(pattern_idx, target_idx) {
            return false;
        }

        // Check bond compatibility with already mapped neighbors.
        // Iterate adjacency directly to avoid per-call allocation.
        for &pattern_bond_idx in &self.pattern.adjacency[pattern_idx] {
            let pattern_bond = &self.pattern.bonds[pattern_bond_idx as usize];
            let neighbor_pattern_idx = if pattern_bond.source == pattern_idx as u16 {
                pattern_bond.target as usize
            } else {
                pattern_bond.source as usize
            };
            let neighbor_target_idx = self.mapping[neighbor_pattern_idx];
            if neighbor_target_idx == u16::MAX {
                continue;
            }
            if !self.check_bond_compatibility(pattern_bond_idx, target_idx, neighbor_target_idx) {
                return false;
            }
        }

        true
    }

    /// Implements matches cached.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn atom_matches_cached(&mut self, pattern_idx: usize, target_idx: u16) -> bool {
        let cached = self.atom_match_cache[pattern_idx][target_idx as usize];
        if cached == 1 {
            return false;
        }
        if cached == 2 {
            return true;
        }

        let matches = query_atom_matches_with_cache(
            &self.pattern.atoms[pattern_idx],
            self.target,
            target_idx,
            self.ring_ctx.as_deref(),
            Some(&self.recursive_cache),
        );
        self.atom_match_cache[pattern_idx][target_idx as usize] = if matches { 2 } else { 1 };
        matches
    }

    /// Checks if one pattern bond is compatible with a target bond.
    fn check_bond_compatibility(&self, pattern_bond_idx: u16, t1: u16, t2: u16) -> bool {
        let pattern_bond = &self.pattern.bonds[pattern_bond_idx as usize];

        let target_bond_idx = self
            .small_target_cache
            .as_ref()
            .and_then(|cache| cache.lookup_bond(t1, t2))
            .or_else(|| {
                self.target.adjacency[t1 as usize]
                    .iter()
                    .copied()
                    .find(|&bond_idx| {
                        let bond = &self.target.bonds[bond_idx as usize];
                        (bond.source == t1 && bond.target == t2)
                            || (bond.target == t1 && bond.source == t2)
                    })
            });

        let Some(target_bond_idx) = target_bond_idx else {
            return false;
        };

        // Check if bond types match
        pattern_bond.query.matches(self.target, target_bond_idx)
    }

    /// Adds a mapping.
    fn add_pair(&mut self, pattern_idx: usize, target_idx: u16) {
        self.mapping[pattern_idx] = target_idx;
        self.reverse_mapping[target_idx as usize] = Some(pattern_idx as u16);
    }

    /// Removes a mapping.
    fn remove_pair(&mut self, pattern_idx: usize) {
        let target_idx = self.mapping[pattern_idx];
        if target_idx != u16::MAX {
            self.mapping[pattern_idx] = u16::MAX;
            self.reverse_mapping[target_idx as usize] = None;
        }
    }

    #[inline]
    /// Implements idx at depth.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn pattern_idx_at_depth(&self, depth: usize) -> usize {
        if self.precomputed_atom_matches {
            self.search_order[depth]
        } else {
            self.select_next_pattern_idx()
        }
    }

    /// Implements next pattern idx.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn select_next_pattern_idx(&self) -> usize {
        let mut best_idx = None;
        let mut best_has_frontier = false;
        let mut best_mapped_neighbors = 0usize;
        let mut best_estimated_candidates = usize::MAX;
        let mut best_degree = 0usize;

        for &pattern_idx in &self.search_order {
            if self.mapping[pattern_idx] != u16::MAX {
                continue;
            }
            let mapped_neighbors = self.mapped_neighbor_count(pattern_idx);
            let has_frontier = mapped_neighbors > 0;
            let estimated_candidates = self.estimated_candidate_sizes[pattern_idx];
            let degree = self.pattern.adjacency[pattern_idx].len();

            let better = if best_idx.is_none() {
                true
            } else if has_frontier != best_has_frontier {
                has_frontier
            } else if mapped_neighbors != best_mapped_neighbors {
                mapped_neighbors > best_mapped_neighbors
            } else if estimated_candidates != best_estimated_candidates {
                estimated_candidates < best_estimated_candidates
            } else {
                degree > best_degree
            };

            if better {
                best_idx = Some(pattern_idx);
                best_has_frontier = has_frontier;
                best_mapped_neighbors = mapped_neighbors;
                best_estimated_candidates = estimated_candidates;
                best_degree = degree;
            }
        }

        best_idx.unwrap_or(0)
    }

    /// Implements neighbor count.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn mapped_neighbor_count(&self, pattern_idx: usize) -> usize {
        self.pattern.adjacency[pattern_idx]
            .iter()
            .filter(|&&pattern_bond_idx| {
                let pattern_bond = &self.pattern.bonds[pattern_bond_idx as usize];
                let neighbor_pattern_idx = if pattern_bond.source == pattern_idx as u16 {
                    pattern_bond.target as usize
                } else {
                    pattern_bond.source as usize
                };
                self.mapping[neighbor_pattern_idx] != u16::MAX
            })
            .count()
    }

    /// Implements candidates.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn collect_candidates(&self, pattern_idx: usize, out: &mut SmallVec<[u16; 32]>) {
        out.clear();
        let fast_filter = self.atom_fast_filters[pattern_idx];
        if self.precomputed_atom_matches {
            out.extend(self.candidate_targets[pattern_idx].iter().copied());
            return;
        }

        let mut anchor_target = None;
        let mut anchor_degree = usize::MAX;
        for &pattern_bond_idx in &self.pattern.adjacency[pattern_idx] {
            let pattern_bond = &self.pattern.bonds[pattern_bond_idx as usize];
            let neighbor_pattern_idx = if pattern_bond.source == pattern_idx as u16 {
                pattern_bond.target as usize
            } else {
                pattern_bond.source as usize
            };
            let neighbor_target_idx = self.mapping[neighbor_pattern_idx];
            if neighbor_target_idx == u16::MAX {
                continue;
            }
            let degree = self.target.adjacency[neighbor_target_idx as usize].len();
            if degree < anchor_degree {
                anchor_degree = degree;
                anchor_target = Some(neighbor_target_idx);
            }
        }

        if let Some(anchor_target_idx) = anchor_target {
            for &target_bond_idx in &self.target.adjacency[anchor_target_idx as usize] {
                let target_bond = &self.target.bonds[target_bond_idx as usize];
                let candidate = if target_bond.source == anchor_target_idx {
                    target_bond.target
                } else {
                    target_bond.source
                };
                if fast_filter.matches_atom(&self.target.atoms[candidate as usize]) {
                    out.push(candidate);
                }
            }
            return;
        }

        for target_idx in 0..self.target.atoms.len() {
            let target_idx = target_idx as u16;
            if fast_filter.matches_atom(&self.target.atoms[target_idx as usize]) {
                out.push(target_idx);
            }
        }
    }
}

/// Implements element screen.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn passes_element_screen(
    pattern: &QueryGraph,
    target_counts: &[u16; 119],
    target_stats: &TargetAtomStats,
) -> bool {
    let required_counts = required_element_counts(pattern);
    if !required_counts
        .iter()
        .zip(target_counts.iter())
        .all(|(required, available)| required <= available)
    {
        return false;
    }

    let required_aromatic_atoms = pattern
        .atoms
        .iter()
        .filter(|atom| query_atom_requires_aromatic(atom))
        .count() as u16;
    if required_aromatic_atoms > target_stats.aromatic_atoms {
        return false;
    }

    let required_aliphatic_atoms = pattern
        .atoms
        .iter()
        .filter(|atom| query_atom_requires_aliphatic(atom))
        .count() as u16;
    if required_aliphatic_atoms > target_stats.aliphatic_atoms {
        return false;
    }

    let required_aromatic_bonds = pattern
        .bonds
        .iter()
        .filter(|bond| query_bond_requires_aromatic(&bond.query))
        .count() as u16;
    if required_aromatic_bonds > target_stats.aromatic_bonds {
        return false;
    }

    let required_ring_bonds = pattern
        .bonds
        .iter()
        .filter(|bond| query_bond_requires_ring(&bond.query))
        .count() as u16;
    if required_ring_bonds > target_stats.ring_bonds {
        return false;
    }

    true
}

/// Implements element counts.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
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

/// Implements element counts.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn required_element_counts(pattern: &QueryGraph) -> [u16; 119] {
    let mut required_counts = [0u16; 119];
    for query_atom in &pattern.atoms {
        if let Some(element) = fixed_atomic_number(query_atom) {
            let idx = element as usize;
            if idx < required_counts.len() {
                required_counts[idx] = required_counts[idx].saturating_add(1);
            }
        }
    }
    required_counts
}

/// Implements atomic number.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn fixed_atomic_number(query_atom: &QueryAtom) -> Option<u8> {
    let mut fixed = None;

    for primitive in &query_atom.primitives {
        let Some(element) = fixed_atomic_number_primitive(primitive) else {
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

/// Implements atomic number primitive.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn fixed_atomic_number_primitive(primitive: &AtomPrimitive) -> Option<u8> {
    match primitive {
        AtomPrimitive::Element(z) | AtomPrimitive::Aromatic(z) | AtomPrimitive::Aliphatic(z) => {
            Some(*z)
        }
        AtomPrimitive::And(requirements) => {
            let mut fixed = None;
            for requirement in requirements {
                let Some(element) = fixed_atomic_number_primitive(requirement) else {
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
            let mut options = options.iter().filter_map(fixed_atomic_number_primitive);
            let first = options.next()?;
            if options.all(|element| element == first) {
                Some(first)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Implements atom required charge.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
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

/// Implements charge from primitive.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
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

/// Implements contains charge.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
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

/// Implements atom requires aromatic.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn query_atom_requires_aromatic(query_atom: &QueryAtom) -> bool {
    query_atom
        .primitives
        .iter()
        .any(atom_primitive_requires_aromatic)
}

/// Implements atom requires aliphatic.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn query_atom_requires_aliphatic(query_atom: &QueryAtom) -> bool {
    query_atom
        .primitives
        .iter()
        .any(atom_primitive_requires_aliphatic)
}

/// Implements primitive requires aromatic.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn atom_primitive_requires_aromatic(primitive: &AtomPrimitive) -> bool {
    match primitive {
        AtomPrimitive::AnyAromatic | AtomPrimitive::Aromatic(_) => true,
        AtomPrimitive::And(requirements) => {
            requirements.iter().any(atom_primitive_requires_aromatic)
        }
        AtomPrimitive::Or(options) => {
            !options.is_empty() && options.iter().all(atom_primitive_requires_aromatic)
        }
        _ => false,
    }
}

/// Implements primitive requires aliphatic.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn atom_primitive_requires_aliphatic(primitive: &AtomPrimitive) -> bool {
    match primitive {
        AtomPrimitive::AnyAliphatic | AtomPrimitive::Aliphatic(_) => true,
        AtomPrimitive::And(requirements) => {
            requirements.iter().any(atom_primitive_requires_aliphatic)
        }
        AtomPrimitive::Or(options) => {
            !options.is_empty() && options.iter().all(atom_primitive_requires_aliphatic)
        }
        _ => false,
    }
}

/// Implements bond requires ring.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn query_bond_requires_ring(query_bond: &QueryBond) -> bool {
    match query_bond {
        QueryBond::Ring => true,
        QueryBond::And(parts) => parts.iter().any(query_bond_requires_ring),
        QueryBond::Or(parts) => !parts.is_empty() && parts.iter().all(query_bond_requires_ring),
        _ => false,
    }
}

/// Implements bond requires aromatic.
///
/// Function context: `crates/ferrum_core/src/algo/iso.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn query_bond_requires_aromatic(query_bond: &QueryBond) -> bool {
    match query_bond {
        QueryBond::Aromatic => true,
        QueryBond::And(parts) => parts.iter().any(query_bond_requires_aromatic),
        QueryBond::Or(parts) => !parts.is_empty() && parts.iter().all(query_bond_requires_aromatic),
        _ => false,
    }
}

/// VF2 backtracking to find first match.
fn backtrack(state: &mut MatchState, depth: usize) -> bool {
    // All pattern atoms mapped - success!
    if depth >= state.pattern.atoms.len() {
        return true;
    }

    let pattern_idx = state.pattern_idx_at_depth(depth);
    let mut candidates = SmallVec::<[u16; 32]>::new();
    state.collect_candidates(pattern_idx, &mut candidates);

    // Try mapping current pattern atom to each target atom
    for &target_idx in &candidates {
        if state.is_feasible(pattern_idx, target_idx) {
            state.add_pair(pattern_idx, target_idx);

            if backtrack(state, depth + 1) {
                return true;
            }

            state.remove_pair(pattern_idx);
        }
    }

    false
}

/// VF2 backtracking to find all matches.
fn backtrack_all(state: &mut MatchState, depth: usize, matches: &mut Vec<Vec<u16>>) {
    // All pattern atoms mapped - record this match
    if depth >= state.pattern.atoms.len() {
        matches.push(state.mapping.clone());
        return;
    }

    let pattern_idx = state.pattern_idx_at_depth(depth);
    let mut candidates = SmallVec::<[u16; 32]>::new();
    state.collect_candidates(pattern_idx, &mut candidates);

    // Try mapping current pattern atom to each target atom
    for &target_idx in &candidates {
        if state.is_feasible(pattern_idx, target_idx) {
            state.add_pair(pattern_idx, target_idx);
            backtrack_all(state, depth + 1, matches);
            state.remove_pair(pattern_idx);
        }
    }
}

/// VF2 backtracking visitor; returns false to stop early.
fn backtrack_visit<F>(state: &mut MatchState, depth: usize, visit: &mut F) -> bool
where
    F: FnMut(&[u16]) -> bool,
{
    if depth >= state.pattern.atoms.len() {
        return visit(&state.mapping);
    }

    let pattern_idx = state.pattern_idx_at_depth(depth);
    let mut candidates = SmallVec::<[u16; 32]>::new();
    state.collect_candidates(pattern_idx, &mut candidates);

    for &target_idx in &candidates {
        if state.is_feasible(pattern_idx, target_idx) {
            state.add_pair(pattern_idx, target_idx);
            let should_continue = backtrack_visit(state, depth + 1, visit);
            state.remove_pair(pattern_idx);
            if !should_continue {
                return false;
            }
        }
    }

    true
}

/// VF2 backtracking count optimized for full enumeration.
fn backtrack_count(state: &mut MatchState, depth: usize, count: &mut usize) {
    if depth >= state.pattern.atoms.len() {
        *count += 1;
        return;
    }

    let pattern_idx = state.pattern_idx_at_depth(depth);
    let mut candidates = SmallVec::<[u16; 32]>::new();
    state.collect_candidates(pattern_idx, &mut candidates);

    for &target_idx in &candidates {
        if state.is_feasible(pattern_idx, target_idx) {
            state.add_pair(pattern_idx, target_idx);
            backtrack_count(state, depth + 1, count);
            state.remove_pair(pattern_idx);
        }
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smarts::{QueryBond, parse_smarts};
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for single atom match.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
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
    /// Runs regression test for no match.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_no_match() {
        let mol = parse_smiles("CCO").unwrap();

        // No nitrogen in molecule
        let pattern = QueryGraph::single_atom(7);
        assert!(!has_substruct_match(&mol, &pattern));
    }

    #[test]
    /// Runs regression test for two atom match.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
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
    /// Runs regression test for count matches.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_count_matches() {
        let mol = parse_smiles("CCCC").unwrap(); // Butane

        // Count carbons
        let pattern = QueryGraph::single_atom(6);
        assert_eq!(count_matches(&mol, &pattern), 4);

        // Count C-C bonds - matches both directions (C1-C2 and C2-C1)
        let pattern = QueryGraph::two_atoms(6, 6, QueryBond::Single);
        assert_eq!(count_matches(&mol, &pattern), 6); // 3 bonds * 2 directions
    }

    #[test]
    /// Runs regression test for has at least matches thresholds.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_has_at_least_matches_thresholds() {
        let mol = parse_smiles("CCCC").unwrap();
        let carbon = QueryGraph::single_atom(6);
        assert!(has_at_least_matches(&mol, &carbon, 0));
        assert!(has_at_least_matches(&mol, &carbon, 1));
        assert!(has_at_least_matches(&mol, &carbon, 4));
        assert!(!has_at_least_matches(&mol, &carbon, 5));

        let cc_bond = QueryGraph::two_atoms(6, 6, QueryBond::Single);
        assert!(has_at_least_matches(&mol, &cc_bond, 6));
        assert!(!has_at_least_matches(&mol, &cc_bond, 7));
    }

    #[test]
    /// Runs regression test for visit matches early stop.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
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
    /// Runs regression test for visit matches empty pattern once.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_visit_matches_empty_pattern_once() {
        let mol = parse_smiles("CC").unwrap();
        let empty = QueryGraph {
            atoms: vec![],
            bonds: vec![],
            adjacency: vec![],
        };

        let mut visited = 0usize;
        visit_matches(&mol, &empty, |_| {
            visited += 1;
            true
        });

        assert_eq!(visited, 1);
    }

    #[test]
    /// Runs regression test for double bond match.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
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
    /// Runs regression test for aromatic pattern.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_aromatic_pattern() {
        use crate::algo::aromaticity;

        let mut mol = parse_smiles("c1ccccc1").unwrap(); // Benzene
        aromaticity::perceive_aromaticity(&mut mol);

        // Should match aromatic carbon
        let pattern = QueryGraph {
            atoms: vec![QueryAtom::aromatic()],
            bonds: vec![],
            adjacency: vec![smallvec::SmallVec::new()],
        };

        assert!(has_substruct_match(&mol, &pattern));
        assert_eq!(count_matches(&mol, &pattern), 6); // 6 aromatic carbons
    }

    #[test]
    /// Runs regression test for complex pattern.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
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
    /// Runs regression test for ring size and membership primitives.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_ring_size_and_membership_primitives() {
        let mut mol = parse_smiles("C1CCCCC1").unwrap();
        crate::algo::rings::find_sssr(&mut mol);

        let membership = parse_smarts("[CR1]").unwrap();
        assert_eq!(count_matches(&mol, &membership), 6);

        let ring_size = parse_smarts("[C;r6]").unwrap();
        assert_eq!(count_matches(&mol, &ring_size), 6);
    }

    #[test]
    /// Runs regression test for or charge fast filter does not reject neutral branches.
    ///
    /// Function context: `crates/ferrum_core/src/algo/iso.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
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
