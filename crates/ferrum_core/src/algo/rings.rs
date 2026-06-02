//! Rings algorithms.
//!
//! This module contains core cheminformatics graph logic used by parsers and
//! descriptors. The implementations prioritize determinism, correctness, and
//! predictable performance on large molecular batches.
//!
//! Learning focus:
//! - Graph traversal and indexing patterns over atom/bond adjacency.
//! - Performance-oriented Rust design in numerically heavy code paths.

//! ## File guide: Rings
//!
//! This file belongs to FerrumChem's **cheminformatics algorithm layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn find_sssr`
//! - `fn find_minimum_cycle_basis_bonds`
//! - `fn cycle_rank_count`
//! - `fn find_symmetrized_sssr_bonds`
//! - `struct ShortestPathWorkspace`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - Cycle-basis routines assume an undirected simple molecular graph with consistent bond indexing.
//! - Ring flags cached on atoms/bonds must be invalidated when topology mutates and recomputed before dependent descriptors.
//! - Symmetrized SSSR heuristics prioritize chemically intuitive small cycles over exhaustive all-cycle enumeration.
//! ### Chemistry/ML references
//! - Smallest Set of Smallest Rings (SSSR) and minimum cycle basis graph-theory literature.
//! - Cheminformatics ring-perception discussions in RDKit/CDK/OpenBabel ecosystems.
//! - Figueras-like and cycle-basis methods commonly used for 2D ring annotation.

use crate::mol::molecule::Molecule;
use crate::types::AtomIdx;
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

/// Detects rings using SSSR (Smallest Set of Smallest Rings) algorithm.
///
/// Returns the list of rings found (each ring is a list of AtomIdx).
/// Also sets the `is_in_ring` flags on atoms and bonds.
///
/// # Thread Safety
/// This function is thread-safe - it only mutates the molecule passed to it.
/// Each molecule can be processed independently in parallel.
///
/// # Algorithm
/// Uses a cycle basis detection approach:
/// 1. Build spanning tree via BFS
/// 2. Find fundamental cycles for each non-tree edge
/// 3. Mark atoms and bonds that are in rings
/// 4. Returns the rings found
///
/// # Performance
/// O(E + V) where E = edges (bonds), V = vertices (atoms)
///
/// # Sets Ring Flags
/// - `Atom.flags` bit 1: in_ring
/// - `Bond.flags` bit 1: in_ring
///
/// # Example
/// ```ignore
/// let mut mol = parse_smiles("c1ccccc1").unwrap();
/// let rings = find_sssr(&mut mol);
/// assert!(mol.atoms[0].is_in_ring());
/// assert_eq!(rings.len(), 1);
/// assert_eq!(rings[0].len(), 6);
/// ```
pub fn find_sssr(mol: &mut Molecule) -> Vec<Vec<AtomIdx>> {
    mol.invalidate_cached_properties();
    if mol.num_atoms() == 0 {
        return Vec::new();
    }

    // Reset all ring flags first
    for atom in &mut mol.atoms {
        atom.flags &= !0b10; // Clear bit 1 (ring flag)
    }
    for bond in &mut mol.bonds {
        bond.flags &= !0b10; // Clear bit 1 (ring flag)
    }

    let all_rings = find_sssr_cycles(mol);
    for cycle in &all_rings {
        mark_cycle_as_ring(mol, cycle);
    }

    all_rings
}

/// Detects SSSR-like atom cycles without mutating ring flags.
///
/// This mirrors `find_sssr` cycle enumeration semantics but leaves atom/bond
/// ring flags untouched. It is used by read-only query contexts (SMARTS ring
/// primitives) to avoid cloning whole molecules just to run mutable ring
/// perception.
pub(crate) fn find_sssr_cycles(mol: &Molecule) -> Vec<Vec<AtomIdx>> {
    if mol.num_atoms() == 0 {
        return Vec::new();
    }

    let n_atoms = mol.num_atoms();
    let mut all_rings = Vec::new();
    let mut visited = vec![false; n_atoms];

    for start_idx in 0..n_atoms {
        if visited[start_idx] {
            continue;
        }

        let component_rings = detect_rings_in_component(mol, start_idx as AtomIdx, &mut visited);
        all_rings.extend(component_rings);
    }

    all_rings
}

/// Compute a minimum cycle basis as bond-index cycles.
///
/// This does not mutate ring flags and is used by descriptor code that needs
/// a more RDKit-like small-ring basis than the legacy `find_sssr()` traversal.
pub fn find_minimum_cycle_basis_bonds(mol: &Molecule) -> Vec<smallvec::SmallVec<[u16; 8]>> {
    let n_atoms = mol.num_atoms();
    if n_atoms == 0 {
        return Vec::new();
    }

    let mut visited = vec![false; n_atoms];
    let mut basis = Vec::new();

    for start in 0..n_atoms {
        if visited[start] {
            continue;
        }

        let (component_atoms, component_bonds) = collect_component(mol, start, &mut visited);
        // Cycle rank is E - V + C; for a connected component this is E - V + 1.
        // If E < V there is no cycle in the component.
        if component_bonds.len() < component_atoms.len() {
            continue;
        }

        basis.extend(minimum_cycle_basis_component(
            mol,
            &component_atoms,
            &component_bonds,
        ));
    }

    basis
}

/// Finds minimum cycle basis bonds cached.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub(crate) fn find_minimum_cycle_basis_bonds_cached(
    mol: &mut Molecule,
) -> Vec<smallvec::SmallVec<[u16; 8]>> {
    if let Some(cached) = mol.cached_min_cycle_basis_bonds() {
        return cached.clone();
    }
    let rings = find_minimum_cycle_basis_bonds(mol);
    mol.store_min_cycle_basis_bonds(rings.clone());
    rings
}

/// Implements minimum cycle basis bonds cached.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub(crate) fn ensure_minimum_cycle_basis_bonds_cached(mol: &mut Molecule) {
    if mol.cached_min_cycle_basis_bonds().is_none() {
        let rings = find_minimum_cycle_basis_bonds(mol);
        mol.store_min_cycle_basis_bonds(rings);
    }
}

/// Implements rank count.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn cycle_rank_count(mol: &Molecule) -> usize {
    let n_atoms = mol.num_atoms();
    if n_atoms == 0 {
        return 0;
    }

    let mut visited = vec![false; n_atoms];
    let mut components = 0usize;
    let mut queue = VecDeque::new();

    for start in 0..n_atoms {
        if visited[start] {
            continue;
        }
        components += 1;
        visited[start] = true;
        queue.push_back(start);

        while let Some(atom_idx) = queue.pop_front() {
            for &bond_idx in &mol.adjacency[atom_idx] {
                let bond = &mol.bonds[bond_idx as usize];
                let neighbor = if bond.source as usize == atom_idx {
                    bond.target as usize
                } else {
                    bond.source as usize
                };
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
    }

    mol.num_bonds()
        .saturating_add(components)
        .saturating_sub(n_atoms)
}

/// Compute a symmetrized small-ring set as bond-index cycles.
///
/// RDKit ring counts use a symmetrized SSSR representation, so bridged cage
/// systems like cubane and adamantane can contain more rings than the raw cycle
/// rank / minimum cycle basis count. To mirror that behavior, collect every
/// unique shortest cycle induced by removing a bond and finding all shortest
/// alternate paths between its endpoints.
pub fn find_symmetrized_sssr_bonds(mol: &Molecule) -> Vec<smallvec::SmallVec<[u16; 8]>> {
    if mol.num_atoms() == 0 || mol.num_bonds() == 0 {
        return Vec::new();
    }

    let base_count = cycle_rank_count(mol);
    if base_count == 0 {
        return Vec::new();
    }
    if base_count == 1 {
        return find_minimum_cycle_basis_bonds(mol);
    }

    let mut cycles = Vec::new();
    let mut workspace = ShortestPathWorkspace::new(mol.num_atoms());

    for bond_idx in 0..mol.num_bonds() {
        let bond = &mol.bonds[bond_idx];
        let start = bond.source as usize;
        let end = bond.target as usize;

        for mut path in shortest_paths_excluding_bond(mol, start, end, bond_idx, &mut workspace) {
            debug_assert!(bond_idx <= u16::MAX as usize);
            path.push(bond_idx as u16);
            path.sort_unstable();
            path.dedup();
            if path.len() >= 3 {
                cycles.push(path.into());
            }
        }
    }

    cycles.sort_unstable();
    cycles.dedup();

    if cycles.len() == base_count + 1 {
        let max_len = cycles
            .iter()
            .map(|cycle: &smallvec::SmallVec<[u16; 8]>| cycle.len())
            .max()
            .unwrap_or(0);
        let max_cycles: Vec<usize> = cycles
            .iter()
            .enumerate()
            .filter_map(|(idx, cycle): (usize, &smallvec::SmallVec<[u16; 8]>)| {
                (cycle.len() == max_len).then_some(idx)
            })
            .collect();

        if max_cycles.len() == 2 {
            let overlap = sorted_overlap_count(&cycles[max_cycles[0]], &cycles[max_cycles[1]]);
            if overlap == 2 {
                cycles.remove(max_cycles[1]);
            }
        }
    }

    cycles
}

/// Finds symmetrized sssr bonds cached.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub(crate) fn find_symmetrized_sssr_bonds_cached(
    mol: &mut Molecule,
) -> Vec<smallvec::SmallVec<[u16; 8]>> {
    if let Some(cached) = mol.cached_symmetrized_sssr_bonds() {
        return cached.clone();
    }
    let rings = find_symmetrized_sssr_bonds(mol);
    mol.store_symmetrized_sssr_bonds(rings.clone());
    rings
}

/// Implements symmetrized sssr bonds cached.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub(crate) fn ensure_symmetrized_sssr_bonds_cached(mol: &mut Molecule) {
    if mol.cached_symmetrized_sssr_bonds().is_none() {
        let rings = find_symmetrized_sssr_bonds(mol);
        mol.store_symmetrized_sssr_bonds(rings);
    }
}

/// Implements symmetrized sssr bonds ref.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub(crate) fn cached_symmetrized_sssr_bonds_ref(
    mol: &Molecule,
) -> Option<&[smallvec::SmallVec<[u16; 8]>]> {
    mol.cached_symmetrized_sssr_bonds()
        .map(|rings| rings.as_slice())
}

/// Implements ShortestPathWorkspace.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
struct ShortestPathWorkspace {
    dist: Vec<usize>,
    predecessors: Vec<Vec<(usize, usize)>>,
    queue: VecDeque<usize>,
}

impl ShortestPathWorkspace {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_core/src/algo/rings.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn new(n_atoms: usize) -> Self {
        Self {
            dist: vec![usize::MAX; n_atoms],
            predecessors: vec![Vec::new(); n_atoms],
            queue: VecDeque::new(),
        }
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/algo/rings.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn prepare(&mut self, n_atoms: usize, start: usize) {
        if self.dist.len() != n_atoms {
            self.dist.clear();
            self.dist.resize(n_atoms, usize::MAX);
            self.predecessors.clear();
            self.predecessors.resize_with(n_atoms, Vec::new);
        } else {
            self.dist.fill(usize::MAX);
            for preds in &mut self.predecessors {
                preds.clear();
            }
        }
        self.queue.clear();
        self.dist[start] = 0;
        self.queue.push_back(start);
    }
}

/// Implements paths excluding bond.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn shortest_paths_excluding_bond(
    mol: &Molecule,
    start: usize,
    target: usize,
    excluded_bond: usize,
    workspace: &mut ShortestPathWorkspace,
) -> Vec<Vec<u16>> {
    let n_atoms = mol.num_atoms();
    workspace.prepare(n_atoms, start);

    while let Some(atom_idx) = workspace.queue.pop_front() {
        let next_dist = workspace.dist[atom_idx] + 1;
        for &bond_idx in &mol.adjacency[atom_idx] {
            let bidx = bond_idx as usize;
            if bidx == excluded_bond {
                continue;
            }

            let bond = &mol.bonds[bidx];
            let neighbor = if bond.source as usize == atom_idx {
                bond.target as usize
            } else {
                bond.source as usize
            };

            if workspace.dist[neighbor] == usize::MAX {
                workspace.dist[neighbor] = next_dist;
                workspace.predecessors[neighbor].push((atom_idx, bidx));
                workspace.queue.push_back(neighbor);
            } else if workspace.dist[neighbor] == next_dist {
                workspace.predecessors[neighbor].push((atom_idx, bidx));
            }
        }
    }

    if workspace.dist[target] == usize::MAX {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut path = Vec::with_capacity(workspace.dist[target]);
    collect_shortest_bond_paths(target, start, &workspace.predecessors, &mut path, &mut out);
    out
}

/// Implements overlap count.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn sorted_overlap_count(left: &[u16], right: &[u16]) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut overlap = 0usize;
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                overlap += 1;
                i += 1;
                j += 1;
            }
        }
    }
    overlap
}

/// Implements shortest bond paths.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn collect_shortest_bond_paths(
    node: usize,
    start: usize,
    predecessors: &[Vec<(usize, usize)>],
    path: &mut Vec<u16>,
    out: &mut Vec<Vec<u16>>,
) {
    if node == start {
        let mut candidate = path.clone();
        candidate.reverse();
        out.push(candidate);
        return;
    }

    for &(prev_atom, bond_idx) in &predecessors[node] {
        debug_assert!(bond_idx <= u16::MAX as usize);
        path.push(bond_idx as u16);
        collect_shortest_bond_paths(prev_atom, start, predecessors, path, out);
        path.pop();
    }
}

/// Implements component.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn collect_component(
    mol: &Molecule,
    start: usize,
    visited: &mut [bool],
) -> (Vec<usize>, Vec<usize>) {
    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut bond_seen = vec![false; mol.num_bonds()];
    let mut queue = Vec::new();
    let mut head = 0usize;

    visited[start] = true;
    queue.push(start);

    while head < queue.len() {
        let atom_idx = queue[head];
        head += 1;
        atoms.push(atom_idx);

        for &bond_idx in &mol.adjacency[atom_idx] {
            let bidx = bond_idx as usize;
            if !bond_seen[bidx] {
                bond_seen[bidx] = true;
                bonds.push(bidx);
            }

            let bond = &mol.bonds[bidx];
            let neighbor = if bond.source as usize == atom_idx {
                bond.target as usize
            } else {
                bond.source as usize
            };

            if !visited[neighbor] {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }

    (atoms, bonds)
}

/// Implements cycle basis component.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn minimum_cycle_basis_component(
    mol: &Molecule,
    component_atoms: &[usize],
    component_bonds: &[usize],
) -> Vec<smallvec::SmallVec<[u16; 8]>> {
    if component_atoms.is_empty() {
        return Vec::new();
    }

    let mut in_component = vec![false; mol.num_atoms()];
    for &atom_idx in component_atoms {
        in_component[atom_idx] = true;
    }

    let mut bond_pos = vec![usize::MAX; mol.num_bonds()];
    for (local_idx, &bond_idx) in component_bonds.iter().enumerate() {
        bond_pos[bond_idx] = local_idx;
    }

    let tree_edges = spanning_tree_edge_flags(
        mol,
        component_atoms,
        &in_component,
        &bond_pos,
        component_bonds.len(),
    );
    let mut orth_sets: Vec<Vec<u8>> = component_bonds
        .iter()
        .enumerate()
        .filter_map(|(local_idx, _)| {
            if tree_edges[local_idx] != 0 {
                return None;
            }
            let mut orth = vec![0u8; component_bonds.len()];
            orth[local_idx] = 1;
            Some(orth)
        })
        .collect();

    let mut basis = Vec::new();
    let mut cycle_mask = vec![0u8; component_bonds.len()];
    while let Some(base) = orth_sets.pop() {
        let cycle = minimum_orthogonal_cycle(mol, component_atoms, &in_component, &bond_pos, &base);
        if cycle.is_empty() {
            continue;
        }

        cycle_mask.fill(0);
        for &bond_idx in &cycle {
            let local_idx = bond_pos[bond_idx as usize];
            if local_idx != usize::MAX {
                cycle_mask[local_idx] ^= 1;
            }
        }

        for orth in &mut orth_sets {
            let mut parity = 0u8;
            for idx in 0..orth.len() {
                parity ^= cycle_mask[idx] & orth[idx];
            }
            if parity == 1 {
                for idx in 0..orth.len() {
                    orth[idx] ^= base[idx];
                }
            }
        }

        basis.push(cycle.into());
    }

    basis
}

/// Implements tree edge flags.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn spanning_tree_edge_flags(
    mol: &Molecule,
    component_atoms: &[usize],
    in_component: &[bool],
    bond_pos: &[usize],
    component_bond_count: usize,
) -> Vec<u8> {
    let mut tree_edges = vec![0u8; component_bond_count];
    let start = component_atoms[0];
    let mut visited = vec![false; mol.num_atoms()];
    let mut queue = Vec::new();
    let mut head = 0usize;
    visited[start] = true;
    queue.push(start);

    while head < queue.len() {
        let atom_idx = queue[head];
        head += 1;
        for &bond_idx in &mol.adjacency[atom_idx] {
            let bidx = bond_idx as usize;
            let bond = &mol.bonds[bidx];
            let neighbor = if bond.source as usize == atom_idx {
                bond.target as usize
            } else {
                bond.source as usize
            };

            if !in_component[neighbor] || visited[neighbor] {
                continue;
            }

            visited[neighbor] = true;
            let local_idx = bond_pos[bidx];
            if local_idx != usize::MAX {
                tree_edges[local_idx] = 1;
            }
            queue.push(neighbor);
        }
    }

    tree_edges
}

/// Implements orthogonal cycle.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn minimum_orthogonal_cycle(
    mol: &Molecule,
    component_atoms: &[usize],
    in_component: &[bool],
    bond_pos: &[usize],
    orth: &[u8],
) -> Vec<u16> {
    let mut best_path = None;
    let mut best_len = usize::MAX;
    let lifted_nodes = mol.num_atoms() * 2;
    let mut prev_node = vec![usize::MAX; lifted_nodes];
    let mut prev_bond = vec![usize::MAX; lifted_nodes];
    let mut dist = vec![usize::MAX; lifted_nodes];
    let mut queue = Vec::new();

    for &start in component_atoms {
        let source = start * 2;
        let target = source + 1;
        prev_node.fill(usize::MAX);
        prev_bond.fill(usize::MAX);
        dist.fill(usize::MAX);
        queue.clear();
        let mut head = 0usize;

        dist[source] = 0;
        queue.push(source);

        while head < queue.len() {
            let node = queue[head];
            head += 1;
            if node == target {
                break;
            }

            let atom_idx = node / 2;
            let layer = node % 2;
            let next_dist = dist[node] + 1;

            for &bond_idx in &mol.adjacency[atom_idx] {
                let bidx = bond_idx as usize;
                let local_idx = bond_pos[bidx];
                if local_idx == usize::MAX {
                    continue;
                }

                let bond = &mol.bonds[bidx];
                let neighbor = if bond.source as usize == atom_idx {
                    bond.target as usize
                } else {
                    bond.source as usize
                };

                if !in_component[neighbor] {
                    continue;
                }

                let next_layer = if orth[local_idx] != 0 {
                    1 - layer
                } else {
                    layer
                };
                let next = neighbor * 2 + next_layer;

                if dist[next] == usize::MAX {
                    dist[next] = next_dist;
                    prev_node[next] = node;
                    prev_bond[next] = bidx;
                    queue.push(next);
                }
            }
        }

        if dist[target] == usize::MAX || dist[target] >= best_len {
            continue;
        }

        let mut path = Vec::with_capacity(dist[target]);
        let mut current = target;
        while current != source {
            let bond_idx = prev_bond[current];
            if bond_idx == usize::MAX {
                path.clear();
                break;
            }
            debug_assert!(bond_idx <= u16::MAX as usize);
            path.push(bond_idx as u16);
            current = prev_node[current];
        }

        if !path.is_empty() {
            best_len = dist[target];
            best_path = Some(path);
        }
    }

    let Some(path) = best_path else {
        return Vec::new();
    };

    let edge_path = path;

    let mut parity = vec![0u8; orth.len()];
    for &bond_idx in &edge_path {
        let local_idx = bond_pos[bond_idx as usize];
        if local_idx != usize::MAX {
            parity[local_idx] ^= 1;
        }
    }

    let mut cycle = Vec::new();
    for &bond_idx in &edge_path {
        let local_idx = bond_pos[bond_idx as usize];
        if local_idx != usize::MAX && parity[local_idx] == 1 {
            cycle.push(bond_idx);
            parity[local_idx] = 0;
        }
    }

    cycle
}

/// Detect rings in a single connected component using cycle detection.
///
/// # Thread Safety
/// Read-only access to molecule structure, writes only to the molecule being processed.
fn detect_rings_in_component(
    mol: &Molecule,
    start: AtomIdx,
    visited: &mut [bool],
) -> Vec<Vec<AtomIdx>> {
    let n_atoms = mol.num_atoms();

    // Build spanning tree using BFS
    let mut parent = vec![None; n_atoms];
    let mut in_tree = vec![false; n_atoms];
    let mut queue = VecDeque::new();

    queue.push_back(start);
    visited[start as usize] = true;
    in_tree[start as usize] = true;

    while let Some(current) = queue.pop_front() {
        // Explore neighbors
        for &bond_idx in &mol.adjacency[current as usize] {
            let bond = &mol.bonds[bond_idx as usize];
            let neighbor = if bond.source == current {
                bond.target
            } else {
                bond.source
            };

            if !visited[neighbor as usize] {
                visited[neighbor as usize] = true;
                parent[neighbor as usize] = Some(current);
                in_tree[neighbor as usize] = true;
                queue.push_back(neighbor);
            }
        }
    }

    // Collect all cycles first (to avoid borrowing issues)
    let mut all_cycles = Vec::new();
    let mut seen_cycle_keys: FxHashSet<Vec<AtomIdx>> = FxHashSet::default();
    let mut path_start = Vec::new();
    let mut path_end = Vec::new();
    let mut membership = vec![false; n_atoms];

    for atom_idx in 0..n_atoms {
        if !in_tree[atom_idx] {
            continue;
        }

        for &bond_idx in &mol.adjacency[atom_idx] {
            let bond = &mol.bonds[bond_idx as usize];
            let neighbor = if bond.source == atom_idx as AtomIdx {
                bond.target
            } else {
                bond.source
            };

            // Skip if neighbor not in this component
            if !in_tree[neighbor as usize] {
                continue;
            }

            // Check if this edge is a non-tree edge (creates a cycle)
            if parent[atom_idx] != Some(neighbor)
                && parent[neighbor as usize] != Some(atom_idx as AtomIdx)
            {
                // Found a cycle - trace back from both endpoints to common ancestor
                let cycle = find_cycle_path_reuse(
                    atom_idx as AtomIdx,
                    neighbor,
                    &parent,
                    &mut path_start,
                    &mut path_end,
                    &mut membership,
                );
                if !cycle.is_empty() {
                    let mut key = cycle.clone();
                    key.sort_unstable();
                    key.dedup();
                    if seen_cycle_keys.insert(key) {
                        all_cycles.push(cycle);
                    }
                }
            }
        }
    }

    all_cycles
}

/// Check if two rings contain the same atoms
#[allow(dead_code)]
fn rings_are_same(ring1: &[AtomIdx], ring2: &[AtomIdx]) -> bool {
    if ring1.len() != ring2.len() {
        return false;
    }
    let s1: FxHashSet<_> = ring1.iter().copied().collect();
    let s2: FxHashSet<_> = ring2.iter().copied().collect();
    s1 == s2
}

/// Find the cycle path between two atoms in a spanning tree.
///
/// # Thread Safety
/// Read-only function, safe to call concurrently on different molecules.
fn find_cycle_path(start: AtomIdx, end: AtomIdx, parent: &[Option<AtomIdx>]) -> Vec<AtomIdx> {
    let mut path_start = Vec::new();
    let mut path_end = Vec::new();
    let mut membership = vec![false; parent.len()];
    find_cycle_path_reuse(
        start,
        end,
        parent,
        &mut path_start,
        &mut path_end,
        &mut membership,
    )
}

/// Finds cycle path reuse.
///
/// Function context: `crates/ferrum_core/src/algo/rings.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn find_cycle_path_reuse(
    start: AtomIdx,
    end: AtomIdx,
    parent: &[Option<AtomIdx>],
    path_start: &mut Vec<AtomIdx>,
    path_end: &mut Vec<AtomIdx>,
    membership: &mut [bool],
) -> Vec<AtomIdx> {
    // Trace paths from both endpoints to root
    path_start.clear();
    path_end.clear();
    membership.fill(false);

    let mut current = start;
    while let Some(p) = parent[current as usize] {
        path_start.push(current);
        membership[current as usize] = true;
        current = p;
    }
    path_start.push(current); // Add root
    membership[current as usize] = true;

    current = end;
    while let Some(p) = parent[current as usize] {
        path_end.push(current);
        current = p;
    }
    path_end.push(current); // Add root

    // Find common ancestor
    let common_ancestor = path_end
        .iter()
        .copied()
        .find(|&atom| membership[atom as usize]);

    if let Some(ancestor) = common_ancestor {
        // Build cycle: start -> ancestor -> end -> start
        let mut cycle = Vec::new();

        // Path from start to ancestor
        for &atom in path_start.iter() {
            cycle.push(atom);
            if atom == ancestor {
                break;
            }
        }

        // Path from ancestor to end (reversed)
        let mut end_to_ancestor = Vec::new();
        for &atom in path_end.iter() {
            if atom == ancestor {
                break;
            }
            end_to_ancestor.push(atom);
        }
        end_to_ancestor.reverse();
        cycle.extend(end_to_ancestor);

        cycle
    } else {
        Vec::new()
    }
}

/// Mark atoms and bonds in a cycle as being in a ring.
///
/// # Thread Safety
/// Mutates only the molecule passed to it, safe for parallel processing of different molecules.
fn mark_cycle_as_ring(mol: &mut Molecule, cycle: &[AtomIdx]) {
    // Mark atoms in ring
    for &atom_idx in cycle {
        mol.atoms[atom_idx as usize].flags |= 0b10; // Set bit 1 (ring flag)
    }

    // Mark bonds in ring
    for i in 0..cycle.len() {
        let from = cycle[i];
        let to = cycle[(i + 1) % cycle.len()];

        // Find the bond between these atoms
        if let Some(bond_idx) = find_bond_between(mol, from, to) {
            mol.bonds[bond_idx as usize].flags |= 0b10; // Set bit 1 (ring flag)
        }
    }
}

/// Find the bond index between two atoms.
///
/// # Thread Safety
/// Read-only function, safe to call concurrently.
fn find_bond_between(mol: &Molecule, from: AtomIdx, to: AtomIdx) -> Option<usize> {
    for &bond_idx in &mol.adjacency[from as usize] {
        let bond = &mol.bonds[bond_idx as usize];
        if (bond.source == from && bond.target == to) || (bond.source == to && bond.target == from)
        {
            return Some(bond_idx as usize);
        }
    }
    None
}

/// Check if an atom is in a ring.
pub fn is_atom_in_ring(mol: &Molecule, atom_idx: AtomIdx) -> bool {
    mol.atoms[atom_idx as usize].flags & 0b10 != 0
}

/// Check if a bond is in a ring.
pub fn is_bond_in_ring(mol: &Molecule, bond_idx: usize) -> bool {
    mol.bonds[bond_idx].flags & 0b10 != 0
}

/// Count the number of rings in the molecule (approximate).
pub fn count_rings(mol: &Molecule) -> usize {
    // Simple heuristic: count bonds in rings and divide by average ring size
    let ring_bonds: usize = mol.bonds.iter().filter(|b| b.flags & 0b10 != 0).count();

    if ring_bonds == 0 {
        0
    } else {
        // Assume average ring size of 6 (benzene-like)
        (ring_bonds / 6).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for no rings.
    ///
    /// Function context: `crates/ferrum_core/src/algo/rings.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_no_rings() {
        // Ethane: CC (no rings)
        let mut mol = parse_smiles("CC").unwrap();
        find_sssr(&mut mol);

        for atom in &mol.atoms {
            assert_eq!(atom.flags & 0b10, 0, "Atom should not be in ring");
        }
    }

    #[test]
    /// Runs regression test for benzene ring.
    ///
    /// Function context: `crates/ferrum_core/src/algo/rings.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_benzene_ring() {
        // Benzene: c1ccccc1 (one 6-membered ring)
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        find_sssr(&mut mol);

        // All atoms should be in ring
        for atom in &mol.atoms {
            assert_ne!(atom.flags & 0b10, 0, "Benzene atom should be in ring");
        }

        // All bonds should be in ring
        for bond in &mol.bonds {
            assert_ne!(bond.flags & 0b10, 0, "Benzene bond should be in ring");
        }
    }

    #[test]
    /// Runs regression test for cyclohexane ring.
    ///
    /// Function context: `crates/ferrum_core/src/algo/rings.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cyclohexane_ring() {
        // Cyclohexane: C1CCCCC1 (one 6-membered ring)
        let mut mol = parse_smiles("C1CCCCC1").unwrap();
        find_sssr(&mut mol);

        // All 6 atoms should be in ring
        let ring_atoms = mol.atoms.iter().filter(|a| a.flags & 0b10 != 0).count();
        assert_eq!(ring_atoms, 6, "Cyclohexane should have 6 atoms in ring");
    }

    #[test]
    /// Runs regression test for toluene.
    ///
    /// Function context: `crates/ferrum_core/src/algo/rings.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_toluene() {
        // Toluene: Cc1ccccc1 (benzene ring + methyl)
        let mut mol = parse_smiles("Cc1ccccc1").unwrap();
        find_sssr(&mut mol);

        // 6 atoms in ring (benzene), 1 not in ring (methyl)
        let ring_atoms = mol.atoms.iter().filter(|a| a.flags & 0b10 != 0).count();
        assert_eq!(ring_atoms, 6, "Toluene should have 6 atoms in ring");

        // Check methyl carbon is NOT in ring
        assert_eq!(
            mol.atoms[0].flags & 0b10,
            0,
            "Methyl carbon should not be in ring"
        );
    }

    #[test]
    /// Runs regression test for naphthalene.
    ///
    /// Function context: `crates/ferrum_core/src/algo/rings.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_naphthalene() {
        // Naphthalene: c1ccc2ccccc2c1 (two fused 6-membered rings)
        let mut mol = parse_smiles("c1ccc2ccccc2c1").unwrap();
        find_sssr(&mut mol);

        // All 10 atoms should be in rings
        let ring_atoms = mol.atoms.iter().filter(|a| a.flags & 0b10 != 0).count();
        assert_eq!(ring_atoms, 10, "Naphthalene should have 10 atoms in rings");
    }

    #[test]
    /// Runs regression test for symmetrized sssr counts cubane faces.
    ///
    /// Function context: `crates/ferrum_core/src/algo/rings.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_symmetrized_sssr_counts_cubane_faces() {
        let mol = parse_smiles("C12C3C4C1C5C2C3C45").unwrap();
        assert_eq!(find_symmetrized_sssr_bonds(&mol).len(), 6);
    }

    #[test]
    /// Runs regression test for symmetrized sssr counts adamantane faces.
    ///
    /// Function context: `crates/ferrum_core/src/algo/rings.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_symmetrized_sssr_counts_adamantane_faces() {
        let mol = parse_smiles("C1C2CC3CC1CC(C2)C3").unwrap();
        assert_eq!(find_symmetrized_sssr_bonds(&mol).len(), 4);
    }

    #[test]
    /// Runs regression test for symmetrized sssr counts bicyclo 2 2 2 octane faces.
    ///
    /// Function context: `crates/ferrum_core/src/algo/rings.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_symmetrized_sssr_counts_bicyclo_2_2_2_octane_faces() {
        let mol = parse_smiles("C1CC2CCC1CC2").unwrap();
        assert_eq!(find_symmetrized_sssr_bonds(&mol).len(), 3);
    }

    #[test]
    /// Runs regression test for symmetrized sssr avoids asymmetric extra cycle.
    ///
    /// Function context: `crates/ferrum_core/src/algo/rings.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_symmetrized_sssr_avoids_asymmetric_extra_cycle() {
        let mol = parse_smiles("CCCC[C@]12[C@@H]3[C@H](O)CN1[C@@H]1C[C@H]2O[C@]32O/C(=C3/OC(=O)C(C)=C3OC)[C@H](C)[C@@H]12").unwrap();
        assert_eq!(find_symmetrized_sssr_bonds(&mol).len(), 6);
    }

    #[test]
    /// Runs regression test for helper functions.
    ///
    /// Function context: `crates/ferrum_core/src/algo/rings.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_helper_functions() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        find_sssr(&mut mol);

        // Test helper functions
        assert!(is_atom_in_ring(&mol, 0));
        assert!(is_atom_in_ring(&mol, 5));

        for i in 0..mol.num_bonds() {
            assert!(is_bond_in_ring(&mol, i));
        }

        let ring_count = count_rings(&mol);
        assert!(ring_count >= 1, "Should detect at least 1 ring");
    }
}
