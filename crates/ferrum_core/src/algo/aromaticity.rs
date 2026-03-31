//! Aromaticity algorithms.
//!
//! This module contains core cheminformatics graph logic used by parsers and
//! descriptors. The implementations prioritize determinism, correctness, and
//! predictable performance on large molecular batches.
//!
//! Learning focus:
//! - Graph traversal and indexing patterns over atom/bond adjacency.
//! - Performance-oriented Rust design in numerically heavy code paths.


//! ## File guide: Aromaticity
//!
//! This file belongs to FerrumChem's **cheminformatics algorithm layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `enum RingDedupKey`
//! - `fn perceive_aromaticity`
//! - `struct KekulizedPerimeter`
//! - `fn ring_set_key`
//! - `fn ring_dedup_key`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Aromaticity` operates on 2D molecular graphs (atoms/bonds) and does not consume conformer geometry.
//! - `Aromaticity` relies on stable `AtomIdx`/`BondIdx` semantics and adjacency invariants from `mol::molecule`.
//! - Performance paths in `Aromaticity` prefer deterministic pruning/caching over probabilistic heuristics.
//! - Specific scope anchor: `crates/ferrum_core/src/algo/aromaticity.rs` defines aromaticity behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Core graph algorithm literature relevant to `Aromaticity` (traversal, cycle, isomorphism, or canonicalization topics).
//! - Cheminformatics toolkit conventions (RDKit/CDK/OpenBabel) mirrored by `Aromaticity` decisions.
//! - FerrumChem regression tests that pin `Aromaticity` behavior on representative molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/algo/aromaticity.rs`.

use crate::algo::rings::find_sssr;
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, PartialEq, Eq, Hash)]
enum RingDedupKey {
    Small(u128),
    Large(Vec<AtomIdx>),
}

/// Perceive aromaticity using the Hückel 4n+2 rule.
///
/// # Thread Safety
/// This function is thread-safe - it only mutates the molecule passed to it.
/// Each molecule can be processed independently in parallel.
///
/// # Algorithm
/// For each ring detected (SSSR):
/// 1. Check if ring has conjugated system (alternating/aromatic bonds)
/// 2. Count π electrons according to element rules
/// 3. Apply Hückel rule: aromatic if π electrons = 4n+2 (n=0,1,2,...)
/// 4. Set aromatic flags on atoms and bonds in aromatic rings
///
/// # π Electron Counting Rules
/// - C (sp2): 1 π electron
/// - N (pyridine-like, no H): 1 π electron
/// - N (pyrrole-like, with H): 2 π electrons (lone pair)
/// - O, S: 2 π electrons (lone pair)
/// - Other elements: 0 (not considered for aromaticity)
///
/// # Example
/// ```ignore
/// let mut mol = parse_smiles("c1ccccc1").unwrap();
/// perceive_aromaticity(&mut mol);
/// assert!(mol.atoms[0].is_aromatic());  // Benzene carbon
/// ```
pub fn perceive_aromaticity(mol: &mut Molecule) {
    mol.invalidate_cached_properties();
    // Ring topology does not change during aromaticity perception (only bond
    // aromatic flags / bond orders are updated), so compute candidate cycles
    // exactly once and reuse them through the fixed-point iterations.
    let rings = collect_aromaticity_ring_candidates(mol);
    // Some fused systems only become recognizable as aromatic after an adjacent
    // ring has already been aromaticized (e.g. flavylium/quinonoid systems).
    // Iterate to a fixed point so those newly aromatic bonds can unlock the
    // next ring in the same conjugated system.
    for _ in 0..6 {
        let prev_atoms: Vec<bool> = mol.atoms.iter().map(|atom| atom.is_aromatic()).collect();
        let prev_bonds: Vec<(BondType, bool)> = mol
            .bonds
            .iter()
            .map(|bond| (bond.bond_type, bond.is_aromatic()))
            .collect();

        let mut aromatic_rings = Vec::new();
        for ring in &rings {
            let ring_aromatic = is_ring_aromatic(mol, ring);
            let fused_aza = is_fused_aza_annulated_ring(mol, ring);
            let fused_bis = is_fused_bis_quinonoid_ring(mol, ring);
            let is_candidate = ring_aromatic || fused_aza || fused_bis;
            if is_candidate {
                aromatic_rings.push(ring.clone());
            }
        }
        let mut aromatic_perimeters = Vec::new();
        let mut seen_perimeters: HashSet<RingDedupKey> = HashSet::new();
        for ring in &rings {
            if let Some(perimeter) = fused_quinonoid_perimeter(mol, ring) {
                let key = ring_dedup_key(&perimeter.ring);
                if seen_perimeters.insert(key) {
                    aromatic_perimeters.push(perimeter);
                }
            }
        }

        // Collect all atoms that are in at least one aromatic ring
        let n = mol.num_atoms();
        let mut in_aromatic_ring = vec![false; n];
        let mut bond_in_aromatic_ring = vec![false; mol.num_bonds()];
        for ring in &aromatic_rings {
            for &idx in ring {
                in_aromatic_ring[idx as usize] = true;
            }
            for i in 0..ring.len() {
                let from = ring[i];
                let to = ring[(i + 1) % ring.len()];
                if let Some(bond_idx) = find_bond_between(mol, from, to) {
                    bond_in_aromatic_ring[bond_idx] = true;
                }
            }
        }
        for perimeter in &aromatic_perimeters {
            for &idx in &perimeter.ring {
                in_aromatic_ring[idx as usize] = true;
            }
            for i in 0..perimeter.ring.len() {
                let from = perimeter.ring[i];
                let to = perimeter.ring[(i + 1) % perimeter.ring.len()];
                if let Some(bond_idx) = find_bond_between(mol, from, to) {
                    bond_in_aromatic_ring[bond_idx] = true;
                }
            }
        }

        // Strip aromaticity from atoms that SMILES marked aromatic but aren't in any
        // Hückel-aromatic ring in the current perception state.
        for i in 0..n {
            if mol.atoms[i].is_aromatic() && !in_aromatic_ring[i] {
                mol.atoms[i].set_aromatic(false);
            }
        }

        // Kekulize bonds that were Aromatic but now connect to non-aromatic atoms
        for (bond_idx, bond) in mol.bonds.iter_mut().enumerate() {
            if bond.bond_type == BondType::Aromatic {
                let src_arom = in_aromatic_ring[bond.source as usize];
                let tgt_arom = in_aromatic_ring[bond.target as usize];
                if !bond_in_aromatic_ring[bond_idx] {
                    // Aromatic parse bonds between otherwise aromatic atoms can end up
                    // outside every accepted aromatic ring (e.g. fused bridge bonds).
                    // These must be de-aromaticized explicitly instead of being kept
                    // just because both endpoints remain aromatic.
                    if src_arom && tgt_arom {
                        bond.bond_type = BondType::Single;
                        bond.set_aromatic(false);
                    } else if !src_arom && !tgt_arom {
                        bond.bond_type = BondType::Single;
                        bond.set_aromatic(false);
                    } else {
                        bond.bond_type = BondType::Double;
                        bond.set_aromatic(false);
                    }
                } else if !src_arom && !tgt_arom {
                    // Both non-aromatic → Single
                    bond.bond_type = BondType::Single;
                    bond.set_aromatic(false);
                } else if !src_arom || !tgt_arom {
                    // Boundary bond (one aromatic, one not) → Double
                    bond.bond_type = BondType::Double;
                    bond.set_aromatic(false);
                }
                // Both aromatic → keep as Aromatic
            }
        }

        // Mark all atoms/bonds in aromatic rings
        for ring in &aromatic_rings {
            mark_ring_aromatic(mol, ring);
        }
        suppress_fused_bridge_bonds(mol, &aromatic_rings);
        for perimeter in &aromatic_perimeters {
            mark_kekulized_perimeter_aromatic(mol, perimeter);
        }

        let next_atoms: Vec<bool> = mol.atoms.iter().map(|atom| atom.is_aromatic()).collect();
        let next_bonds: Vec<(BondType, bool)> = mol
            .bonds
            .iter()
            .map(|bond| (bond.bond_type, bond.is_aromatic()))
            .collect();
        if prev_atoms == next_atoms && prev_bonds == next_bonds {
            break;
        }
    }

    // Apply diaza-dicarbonyl bridge bond suppression after fixed-point convergence.
    // Keeping this outside the loop prevents oscillation where suppression removes
    // annulation support needed to keep adjacent fused ring atoms aromatic.
    suppress_fused_bridge_ring_bond_aromaticity(mol, &rings);
}

#[derive(Clone)]
struct KekulizedPerimeter {
    ring: Vec<AtomIdx>,
    chord_bond_idx: usize,
}

/// Implements set key.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn ring_set_key(ring: &[AtomIdx]) -> Vec<AtomIdx> {
    let mut key = ring.to_vec();
    key.sort_unstable();
    key.dedup();
    key
}

/// Implements dedup key.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn ring_dedup_key(ring: &[AtomIdx]) -> RingDedupKey {
    let mut mask = 0u128;
    for &idx in ring {
        if idx >= 128 {
            return RingDedupKey::Large(ring_set_key(ring));
        }
        mask |= 1u128 << (idx as u32);
    }
    RingDedupKey::Small(mask)
}

/// Implements aromaticity ring candidates.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn collect_aromaticity_ring_candidates(mol: &mut Molecule) -> Vec<Vec<AtomIdx>> {
    // Seed from legacy SSSR to preserve current behavior and ring flags.
    let sssr_rings = find_sssr(mol);
    // Include larger conjugated perimeters (e.g. porphyrin-like macrocycles)
    // that are often omitted from an SSSR basis but still drive aromaticity.
    let local_cycles = find_local_cycles(mol, 5, 24);
    let dfs_cycles = find_dfs_cycles(mol, 5, 8, 2000);

    let mut rings = Vec::new();
    let mut seen: HashSet<RingDedupKey> = HashSet::new();
    for ring in sssr_rings
        .iter()
        .chain(local_cycles.iter())
        .chain(dfs_cycles.iter())
    {
        let key = ring_dedup_key(ring);
        if seen.insert(key) {
            rings.push(ring.clone());
        }
    }
    rings
}

/// Find additional small cycles by using the shortest path between
/// each bond's endpoints after removing that bond.
fn find_local_cycles(mol: &Molecule, min_len: usize, max_len: usize) -> Vec<Vec<AtomIdx>> {
    let mut out = Vec::new();
    let mut seen: HashSet<RingDedupKey> = HashSet::new();

    for (bond_idx, bond) in mol.bonds.iter().enumerate() {
        let u = bond.source as usize;
        let v = bond.target as usize;
        let Some(path) =
            shortest_path_excluding_bond(mol, u, v, bond_idx, max_len.saturating_sub(1))
        else {
            continue;
        };

        // Path includes both endpoints; adding removed edge closes a cycle with path.len() atoms.
        let ring_len = path.len();
        if ring_len < min_len || ring_len > max_len {
            continue;
        }

        let ring: Vec<AtomIdx> = path.into_iter().map(|idx| idx as AtomIdx).collect();
        let key = ring_dedup_key(&ring);
        if seen.insert(key) {
            out.push(ring);
        }
    }

    out
}

/// Implements path excluding bond.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn shortest_path_excluding_bond(
    mol: &Molecule,
    start: usize,
    end: usize,
    excluded_bond: usize,
    max_edges: usize,
) -> Option<Vec<usize>> {
    if start == end {
        return None;
    }

    let n = mol.num_atoms();
    let mut dist = vec![usize::MAX; n];
    let mut prev = vec![usize::MAX; n];
    let mut q = VecDeque::new();

    dist[start] = 0;
    q.push_back(start);

    while let Some(u) = q.pop_front() {
        if dist[u] >= max_edges {
            continue;
        }

        for &bi in &mol.adjacency[u] {
            let bidx = bi as usize;
            if bidx == excluded_bond {
                continue;
            }

            let b = &mol.bonds[bidx];
            let v = if b.source as usize == u {
                b.target as usize
            } else {
                b.source as usize
            };

            let nd = dist[u] + 1;
            if nd < dist[v] {
                dist[v] = nd;
                prev[v] = u;
                q.push_back(v);
            }
        }
    }

    if dist[end] == usize::MAX || dist[end] > max_edges {
        return None;
    }

    let mut path = Vec::new();
    let mut cur = end;
    path.push(cur);
    while cur != start {
        let p = prev[cur];
        if p == usize::MAX {
            return None;
        }
        cur = p;
        path.push(cur);
    }
    path.reverse();
    Some(path)
}

/// Enumerate additional simple cycles via bounded DFS.
/// This captures fused-ring cycles that may be missed by shortest-path candidates.
fn find_dfs_cycles(
    mol: &Molecule,
    min_len: usize,
    max_len: usize,
    max_cycles: usize,
) -> Vec<Vec<AtomIdx>> {
    let n = mol.num_atoms();
    if n == 0 || min_len < 3 || max_len < min_len {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut seen: HashSet<RingDedupKey> = HashSet::new();
    let mut path: Vec<AtomIdx> = Vec::with_capacity(max_len);
    let mut in_path = vec![false; n];

    for start in 0..n {
        if out.len() >= max_cycles {
            break;
        }
        path.clear();
        path.push(start as AtomIdx);
        in_path[start] = true;

        dfs_cycle_from(
            mol,
            start as AtomIdx,
            start as AtomIdx,
            usize::MAX,
            min_len,
            max_len,
            max_cycles,
            &mut path,
            &mut in_path,
            &mut seen,
            &mut out,
        );

        in_path[start] = false;
    }

    out
}

#[allow(clippy::too_many_arguments)]
/// Implements cycle from.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn dfs_cycle_from(
    mol: &Molecule,
    start: AtomIdx,
    current: AtomIdx,
    parent: usize,
    min_len: usize,
    max_len: usize,
    max_cycles: usize,
    path: &mut Vec<AtomIdx>,
    in_path: &mut [bool],
    seen: &mut HashSet<RingDedupKey>,
    out: &mut Vec<Vec<AtomIdx>>,
) {
    if out.len() >= max_cycles {
        return;
    }

    for &bi in &mol.adjacency[current as usize] {
        let bond = &mol.bonds[bi as usize];
        let nbr = if bond.source == current {
            bond.target
        } else {
            bond.source
        };
        let nbr_usize = nbr as usize;

        if nbr_usize == parent {
            continue;
        }

        // Canonicalization: only explore cycles where `start` is the minimum atom index.
        if nbr_usize < start as usize {
            continue;
        }

        if nbr == start {
            if path.len() >= min_len && path.len() <= max_len {
                let ring = path.clone();
                let key = ring_dedup_key(&ring);
                if seen.insert(key) {
                    out.push(ring);
                    if out.len() >= max_cycles {
                        return;
                    }
                }
            }
            continue;
        }

        if path.len() >= max_len || in_path[nbr_usize] {
            continue;
        }

        in_path[nbr_usize] = true;
        path.push(nbr);
        dfs_cycle_from(
            mol,
            start,
            nbr,
            current as usize,
            min_len,
            max_len,
            max_cycles,
            path,
            in_path,
            seen,
            out,
        );
        path.pop();
        in_path[nbr_usize] = false;
    }
}

/// Check if a ring is aromatic using Hückel's rule.
fn is_ring_aromatic(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    if ring.len() < 3 {
        return false;
    }

    // Reject perimeter cycles that contain internal chords. This avoids
    // aromaticizing fused-system envelopes (e.g. 10-member outer perimeters)
    // while still allowing genuine chordless macrocycles.
    if has_cycle_chord(mol, ring) {
        return false;
    }

    // Check if ring has conjugated system
    if !is_conjugated(mol, ring) {
        return false;
    }

    let quinonoid_gap_carbons = quinonoid_gap_carbons(mol, ring);
    let exocyclic_hetero_double_count = ring_exocyclic_hetero_double_count(mol, ring);

    // Count π electrons with constrained N handling.
    // N contribution is context-dependent, but not fully unconstrained:
    // charged and protonated N are fixed; only ambiguous neutral N are tried as 1/2.
    let mut fixed_pi: u8 = 0;
    let mut n_options: Vec<[u8; 2]> = Vec::new();
    let mut n_opt_len: Vec<usize> = Vec::new();

    for &atom_idx in ring {
        let atom = &mol.atoms[atom_idx as usize];
        match atom.atomic_number {
            6 => {
                let charge = atom.formal_charge;
                if charge < 0 {
                    fixed_pi += 2; // Carbanion: lone pair contributes (cyclopentadienyl)
                } else if charge > 0 {
                    fixed_pi += 0; // Carbocation: empty p-orbital (tropylium)
                } else {
                    // Exocyclic double bond to hetero withdraws the p contribution.
                    if has_exocyclic_hetero_double(mol, atom_idx, ring) {
                        if quinonoid_gap_carbons.is_some() {
                            fixed_pi += 0;
                            continue;
                        }
                        // In larger fused quinonoid systems the carbonyl carbon can
                        // still belong to the aromatic sextet once its ring
                        // neighbors are already part of a fused conjugated system.
                        if is_fused_exocyclic_imine_carbon(mol, atom_idx, ring) {
                            fixed_pi += 1;
                            continue;
                        }
                        if ring.len() > 6
                            && exocyclic_hetero_double_count == 1
                            && has_fused_conjugated_ring_neighbors(mol, atom_idx, ring)
                        {
                            fixed_pi += 1;
                            continue;
                        } else {
                            fixed_pi += 0;
                            continue;
                        }
                    }

                    // Neutral C: check if it has ANY double/aromatic bond in the molecule.
                    // This correctly handles fused ring systems where junction carbons
                    // may have their pi bond in the neighboring ring rather than in the
                    // current ring being tested (e.g. indole, benzofuran, benzothiophene).
                    // In a Kekulized fused system the shared C–C bond carries the double
                    // bond only in one of the two rings; the other ring sees a "single"
                    // bond for that junction carbon even though it is sp2 and part of the
                    // delocalized pi system.  Requiring the pi-bond partner to also be
                    // inside the current ring (has_ring_pi_bond) therefore under-counts
                    // pi electrons for those junction atoms and wrongly rejects aromatic
                    // rings.  Instead we accept the carbon as sp2 (contributing 1 pi
                    // electron to this ring) whenever it has any pi bond anywhere.
                    let has_any_pi = mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
                        let bond = &mol.bonds[bond_idx as usize];
                        matches!(bond.bond_type, BondType::Double | BondType::Aromatic)
                    });
                    if has_any_pi
                        || quinonoid_gap_carbons
                            .as_ref()
                            .is_some_and(|gaps| gaps.contains(&atom_idx))
                    {
                        fixed_pi += 1; // sp2 carbon contributes 1 pi electron
                    }
                    // else: sp3 carbon with no pi bonds anywhere – contributes 0
                }
            }
            7 => {
                let has_h = atom.total_h() > 0;
                let has_ring_pi = has_ring_pi_bond(mol, atom_idx, ring);
                let degree = mol.get_degree(atom_idx);
                if atom.formal_charge > 0 {
                    n_options.push([1, 0]);
                    n_opt_len.push(1);
                } else if atom.formal_charge < 0 || has_h {
                    n_options.push([2, 0]);
                    n_opt_len.push(1);
                } else if has_ring_pi {
                    // Neutral degree-3 aromatic nitrogens (e.g. N-methyl imidazole,
                    // caffeine) are pyrrole-like donors even though the aromatic
                    // SMILES form gives them aromatic bonds in-ring.
                    n_options.push([if degree >= 3 { 2 } else { 1 }, 0]);
                    n_opt_len.push(1);
                } else {
                    // Keep ambiguity only for neutral, non-protonated, non-explicitly-pi N.
                    n_options.push([1, 2]);
                    n_opt_len.push(2);
                }
            }
            8 => {
                // O⁺ (pyrylium / flavylium cation) behaves like pyridinium N:
                // it has no lone pair to donate and contributes 1π electron.
                // Neutral O (furan-like) donates its lone pair → 2π electrons.
                if atom.formal_charge > 0 {
                    fixed_pi += 1;
                } else {
                    fixed_pi += 2;
                }
            }
            16 => {
                // S⁺ is analogous to O⁺; neutral S (thiophene) gives 2π electrons.
                if atom.formal_charge > 0 {
                    fixed_pi += 1;
                } else {
                    fixed_pi += 2;
                }
            }
            _ => return false,
        }
    }

    // Enumerate allowed N option combinations.
    let mut totals = vec![fixed_pi];
    for (idx, opts) in n_options.iter().enumerate() {
        let mut next = Vec::with_capacity(totals.len() * n_opt_len[idx]);
        for &t in &totals {
            next.push(t + opts[0]);
            if n_opt_len[idx] == 2 {
                next.push(t + opts[1]);
            }
        }
        next.sort_unstable();
        next.dedup();
        totals = next;
    }
    for total in totals {
        if is_huckel_aromatic(total) {
            return true;
        }
    }

    false
}

/// Checks whether cycle chord.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn has_cycle_chord(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    let mut consecutive = HashSet::with_capacity(ring.len() * 2);
    for i in 0..ring.len() {
        let a = ring[i];
        let b = ring[(i + 1) % ring.len()];
        consecutive.insert((a.min(b), a.max(b)));
    }

    let ring_atoms: HashSet<AtomIdx> = ring.iter().copied().collect();
    for &atom_idx in ring {
        for &bond_idx in &mol.adjacency[atom_idx as usize] {
            let bond = &mol.bonds[bond_idx as usize];
            let other = if bond.source == atom_idx {
                bond.target
            } else {
                bond.source
            };
            if !ring_atoms.contains(&other) {
                continue;
            }

            let key = (atom_idx.min(other), atom_idx.max(other));
            if !consecutive.contains(&key) {
                return true;
            }
        }
    }

    false
}

/// Implements quinonoid perimeter.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn fused_quinonoid_perimeter(mol: &Molecule, ring: &[AtomIdx]) -> Option<KekulizedPerimeter> {
    if ring.len() != 9 && ring.len() != 10 {
        return None;
    }

    let chord = unique_cycle_chord(mol, ring)?;
    let perimeter = canonical_cycle_order_excluding_chord(mol, ring, chord)?;

    for start in 0..perimeter.len() {
        let ordered = rotate_ring(&perimeter, start);
        if let Some(perimeter) = match_fused_quinonoid_perimeter(mol, &ordered, chord)
            .or_else(|| match_fused_aza_quinonoid_perimeter(mol, &ordered, chord))
            .or_else(|| match_fused_aza_bridge_perimeter(mol, &ordered, chord))
        {
            return Some(perimeter);
        }

        let reversed = reverse_ring(&ordered);
        if let Some(perimeter) = match_fused_quinonoid_perimeter(mol, &reversed, chord)
            .or_else(|| match_fused_aza_quinonoid_perimeter(mol, &reversed, chord))
            .or_else(|| match_fused_aza_bridge_perimeter(mol, &reversed, chord))
        {
            return Some(perimeter);
        }
    }

    None
}

/// Build a canonical perimeter atom order for a chorded cycle candidate.
///
/// The candidate ring list can arrive in any valid cycle ordering. For fused
/// systems, multiple cycles can share the same atom set and only one ordering
/// corresponds to the true outer perimeter (with the fused bridge/chord removed).
/// This helper reconstructs that perimeter directly from graph connectivity by:
/// 1) removing the detected chord edge, and
/// 2) traversing the remaining degree-2 cycle.
fn canonical_cycle_order_excluding_chord(
    mol: &Molecule,
    ring: &[AtomIdx],
    chord: (AtomIdx, AtomIdx, usize),
) -> Option<Vec<AtomIdx>> {
    for i in 0..ring.len() {
        for j in i + 1..ring.len() {
            if ring[i] == ring[j] {
                return None;
            }
        }
    }

    let mut neighbors: Vec<smallvec::SmallVec<[AtomIdx; 2]>> = vec![smallvec::SmallVec::new(); ring.len()];

    for (u_idx, &atom_idx) in ring.iter().enumerate() {
        for &bond_idx in &mol.adjacency[atom_idx as usize] {
            let bidx = bond_idx as usize;
            let bond = &mol.bonds[bidx];
            let other = if bond.source == atom_idx {
                bond.target
            } else {
                bond.source
            };

            if atom_idx > other {
                continue;
            }

            if atom_idx.min(other) == chord.0 && atom_idx.max(other) == chord.1 && bidx == chord.2 {
                continue;
            }

            if let Some(v_idx) = ring.iter().position(|&x| x == other) {
                neighbors[u_idx].push(other);
                neighbors[v_idx].push(atom_idx);
            }
        }
    }

    for nbrs in &neighbors {
        if nbrs.len() != 2 {
            return None;
        }
    }

    let start = *ring.iter().min()?;
    let mut order = Vec::with_capacity(ring.len());
    order.push(start);
    let mut prev: Option<AtomIdx> = None;
    let mut current = start;

    loop {
        let u_idx = ring.iter().position(|&x| x == current)?;
        let nbrs = &neighbors[u_idx];
        let next = match prev {
            Some(prev_atom) if nbrs[0] == prev_atom => nbrs[1],
            Some(prev_atom) if nbrs[1] == prev_atom => nbrs[0],
            Some(_) => return None,
            None => nbrs[0].min(nbrs[1]),
        };

        if next == start {
            break;
        }
        if order.len() >= ring.len() {
            return None;
        }

        order.push(next);
        prev = Some(current);
        current = next;
    }

    if order.len() != ring.len() {
        return None;
    }

    Some(order)
}

/// Implements cycle chord.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn unique_cycle_chord(mol: &Molecule, ring: &[AtomIdx]) -> Option<(AtomIdx, AtomIdx, usize)> {
    for i in 0..ring.len() {
        for j in i + 1..ring.len() {
            if ring[i] == ring[j] {
                return None;
            }
        }
    }

    let mut induced_edges: Vec<(AtomIdx, AtomIdx, usize)> = Vec::new();
    let mut degree = vec![0usize; ring.len()];

    for (u_idx, &atom_idx) in ring.iter().enumerate() {
        for &bond_idx in &mol.adjacency[atom_idx as usize] {
            let bond = &mol.bonds[bond_idx as usize];
            let other = if bond.source == atom_idx {
                bond.target
            } else {
                bond.source
            };
            if atom_idx > other {
                continue;
            }

            if let Some(v_idx) = ring.iter().position(|&x| x == other) {
                induced_edges.push((atom_idx, other, bond_idx as usize));
                degree[u_idx] += 1;
                degree[v_idx] += 1;
            }
        }
    }

    if induced_edges.len() != ring.len() + 1 {
        return None;
    }

    if degree.iter().any(|&d| d != 2 && d != 3) {
        return None;
    }

    let mut degree3_atoms: Vec<AtomIdx> = degree
        .iter()
        .enumerate()
        .filter_map(|(idx, &deg)| (deg == 3).then_some(ring[idx]))
        .collect();
    if degree3_atoms.len() != 2 {
        return None;
    }
    degree3_atoms.sort_unstable();

    let chord_a = degree3_atoms[0];
    let chord_b = degree3_atoms[1];
    induced_edges
        .into_iter()
        .find_map(|(a, b, bond_idx)| {
            (a.min(b) == chord_a && a.max(b) == chord_b).then_some((chord_a, chord_b, bond_idx))
        })
}

/// Implements ring.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn rotate_ring(ring: &[AtomIdx], start: usize) -> Vec<AtomIdx> {
    (0..ring.len())
        .map(|offset| ring[(start + offset) % ring.len()])
        .collect()
}

/// Implements ring.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn reverse_ring(ring: &[AtomIdx]) -> Vec<AtomIdx> {
    let mut reversed = ring.to_vec();
    reversed.reverse();
    reversed
}

/// Implements fused quinonoid perimeter.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn match_fused_quinonoid_perimeter(
    mol: &Molecule,
    ring: &[AtomIdx],
    chord: (AtomIdx, AtomIdx, usize),
) -> Option<KekulizedPerimeter> {
    let chord_positions = ring
        .iter()
        .enumerate()
        .filter_map(|(pos, &atom_idx)| (atom_idx == chord.0 || atom_idx == chord.1).then_some(pos))
        .collect::<Vec<_>>();
    if chord_positions.len() != 2 || chord_positions != [4, 9] {
        return None;
    }

    if !is_neutral_ring_donor_hetero(mol, ring[0]) {
        return None;
    }
    if !is_neutral_ring_carbon(mol, ring[5]) || !has_exocyclic_hetero_double(mol, ring[5], ring) {
        return None;
    }
    if !is_aromatic_anchor_carbon(mol, ring[6]) || !is_aromatic_anchor_carbon(mol, ring[7]) {
        return None;
    }

    for &pos in &[1usize, 2, 3, 4, 8, 9] {
        if !is_neutral_ring_carbon(mol, ring[pos]) {
            return None;
        }
    }

    if !bond_matches(
        mol,
        ring[0],
        ring[1],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[1],
        ring[2],
        &[BondType::Double, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[2],
        ring[3],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[3],
        ring[4],
        &[BondType::Double, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[4],
        ring[5],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[5],
        ring[6],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(mol, ring[6], ring[7], &[BondType::Aromatic])
        || !bond_matches(
            mol,
            ring[7],
            ring[8],
            &[BondType::Single, BondType::Aromatic],
        )
        || !bond_matches(
            mol,
            ring[8],
            ring[9],
            &[BondType::Double, BondType::Aromatic],
        )
        || !bond_matches(
            mol,
            ring[9],
            ring[0],
            &[BondType::Single, BondType::Aromatic],
        )
    {
        return None;
    }

    let chord_bond = &mol.bonds[chord.2];
    if chord_bond.bond_type != BondType::Single || chord_bond.is_aromatic() {
        return None;
    }

    Some(KekulizedPerimeter {
        ring: ring.to_vec(),
        chord_bond_idx: chord.2,
    })
}

/// Implements fused aza quinonoid perimeter.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn match_fused_aza_quinonoid_perimeter(
    mol: &Molecule,
    ring: &[AtomIdx],
    chord: (AtomIdx, AtomIdx, usize),
) -> Option<KekulizedPerimeter> {
    let chord_positions = ring
        .iter()
        .enumerate()
        .filter_map(|(pos, &atom_idx)| (atom_idx == chord.0 || atom_idx == chord.1).then_some(pos))
        .collect::<Vec<_>>();
    if chord_positions.len() != 2 || chord_positions != [4, 9] {
        return None;
    }

    if !is_neutral_ring_pi_nitrogen(mol, ring[0], ring) {
        return None;
    }
    if !is_neutral_ring_carbon(mol, ring[1]) || !has_exocyclic_hetero_double(mol, ring[1], ring) {
        return None;
    }
    if !is_neutral_ring_pi_nitrogen(mol, ring[2], ring) {
        return None;
    }
    if !is_neutral_ring_carbon(mol, ring[3]) || !has_exocyclic_hetero_single(mol, ring[3], ring) {
        return None;
    }
    if !is_neutral_ring_carbon(mol, ring[4]) {
        return None;
    }
    if !is_neutral_ring_pi_nitrogen(mol, ring[5], ring) {
        return None;
    }
    if !is_aromatic_anchor_carbon(mol, ring[6]) || !is_aromatic_anchor_carbon(mol, ring[7]) {
        return None;
    }
    if !is_neutral_degree3_ring_donor_nitrogen(mol, ring[8], ring) {
        return None;
    }
    if !is_neutral_ring_carbon(mol, ring[9]) {
        return None;
    }

    if !bond_matches(
        mol,
        ring[0],
        ring[1],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[1],
        ring[2],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[2],
        ring[3],
        &[BondType::Double, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[3],
        ring[4],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[4],
        ring[5],
        &[BondType::Double, BondType::Aromatic],
    ) || !bond_matches(mol, ring[5], ring[6], &[BondType::Aromatic])
        || !bond_matches(mol, ring[6], ring[7], &[BondType::Aromatic])
        || !bond_matches(mol, ring[7], ring[8], &[BondType::Aromatic])
        || !bond_matches(mol, ring[8], ring[9], &[BondType::Aromatic])
        || !bond_matches(
            mol,
            ring[9],
            ring[0],
            &[BondType::Double, BondType::Aromatic],
        )
    {
        return None;
    }

    let chord_bond = &mol.bonds[chord.2];
    if chord_bond.bond_type != BondType::Single || chord_bond.is_aromatic() {
        return None;
    }

    Some(KekulizedPerimeter {
        ring: ring.to_vec(),
        chord_bond_idx: chord.2,
    })
}

/// Implements fused aza bridge perimeter.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn match_fused_aza_bridge_perimeter(
    mol: &Molecule,
    ring: &[AtomIdx],
    chord: (AtomIdx, AtomIdx, usize),
) -> Option<KekulizedPerimeter> {
    if ring.len() != 9 {
        return None;
    }

    let chord_positions = ring
        .iter()
        .enumerate()
        .filter_map(|(pos, &atom_idx)| (atom_idx == chord.0 || atom_idx == chord.1).then_some(pos))
        .collect::<Vec<_>>();
    if chord_positions.len() != 2 || chord_positions != [1, 6] {
        return None;
    }

    if !is_aromatic_anchor_carbon(mol, ring[0]) || !is_aromatic_anchor_carbon(mol, ring[8]) {
        return None;
    }
    if !is_neutral_ring_carbon(mol, ring[1])
        || !is_neutral_ring_carbon(mol, ring[2])
        || !is_neutral_ring_carbon(mol, ring[3])
        || !is_neutral_ring_carbon(mol, ring[5])
        || !is_neutral_ring_carbon(mol, ring[6])
    {
        return None;
    }
    if !is_neutral_degree3_ring_nitrogen(mol, ring[4]) {
        return None;
    }
    if !is_neutral_ring_pi_nitrogen(mol, ring[7], ring) {
        return None;
    }

    if !bond_matches(
        mol,
        ring[0],
        ring[1],
        &[BondType::Double, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[1],
        ring[2],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[2],
        ring[3],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[3],
        ring[4],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[4],
        ring[5],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[5],
        ring[6],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[6],
        ring[7],
        &[BondType::Single, BondType::Aromatic],
    ) || !bond_matches(
        mol,
        ring[7],
        ring[8],
        &[BondType::Double, BondType::Aromatic],
    ) || !bond_matches(mol, ring[8], ring[0], &[BondType::Aromatic])
    {
        return None;
    }

    let chord_bond = &mol.bonds[chord.2];
    if chord_bond.bond_type != BondType::Single || chord_bond.is_aromatic() {
        return None;
    }

    Some(KekulizedPerimeter {
        ring: ring.to_vec(),
        chord_bond_idx: chord.2,
    })
}

/// Checks whether neutral ring donor hetero.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_neutral_ring_donor_hetero(mol: &Molecule, atom_idx: AtomIdx) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    matches!(atom.atomic_number, 8 | 16) && atom.formal_charge <= 0 && mol.get_degree(atom_idx) == 2
}

/// Checks whether neutral degree3 ring donor nitrogen.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_neutral_degree3_ring_donor_nitrogen(
    mol: &Molecule,
    atom_idx: AtomIdx,
    ring: &[AtomIdx],
) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    atom.atomic_number == 7
        && atom.formal_charge == 0
        && atom.total_h() == 0
        && mol.get_degree(atom_idx) >= 3
        && has_ring_pi_bond(mol, atom_idx, ring)
}

/// Checks whether neutral degree3 ring nitrogen.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_neutral_degree3_ring_nitrogen(mol: &Molecule, atom_idx: AtomIdx) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    atom.atomic_number == 7
        && atom.formal_charge == 0
        && atom.total_h() == 0
        && mol.get_degree(atom_idx) >= 3
}

/// Implements matches.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn bond_matches(mol: &Molecule, a: AtomIdx, b: AtomIdx, allowed: &[BondType]) -> bool {
    find_bond_between(mol, a, b)
        .map(|bond_idx| allowed.contains(&mol.bonds[bond_idx].bond_type))
        .unwrap_or(false)
}

/// Checks whether ring pi bond.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn has_ring_pi_bond(mol: &Molecule, atom_idx: AtomIdx, ring: &[AtomIdx]) -> bool {
    mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
        let bond = &mol.bonds[bond_idx as usize];
        if !matches!(bond.bond_type, BondType::Double | BondType::Aromatic) {
            return false;
        }
        let other = if bond.source == atom_idx {
            bond.target
        } else {
            bond.source
        };
        ring.contains(&other)
    })
}

/// Checks whether exocyclic hetero double.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn has_exocyclic_hetero_double(mol: &Molecule, atom_idx: AtomIdx, ring: &[AtomIdx]) -> bool {
    mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
        let bond = &mol.bonds[bond_idx as usize];
        if bond.bond_type != BondType::Double {
            return false;
        }
        let other = if bond.source == atom_idx {
            bond.target
        } else {
            bond.source
        };
        !ring.contains(&other) && mol.atoms[other as usize].atomic_number != 6
    })
}

/// Checks whether exocyclic imine double.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn has_exocyclic_imine_double(mol: &Molecule, atom_idx: AtomIdx, ring: &[AtomIdx]) -> bool {
    mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
        let bond = &mol.bonds[bond_idx as usize];
        if bond.bond_type != BondType::Double {
            return false;
        }
        let other = if bond.source == atom_idx {
            bond.target
        } else {
            bond.source
        };
        !ring.contains(&other) && mol.atoms[other as usize].atomic_number == 7
    })
}

/// Implements exocyclic hetero double count.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn ring_exocyclic_hetero_double_count(mol: &Molecule, ring: &[AtomIdx]) -> usize {
    ring.iter()
        .filter(|&&atom_idx| {
            mol.atoms[atom_idx as usize].atomic_number == 6
                && has_exocyclic_hetero_double(mol, atom_idx, ring)
        })
        .count()
}

/// Checks whether exocyclic hetero single.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn has_exocyclic_hetero_single(mol: &Molecule, atom_idx: AtomIdx, ring: &[AtomIdx]) -> bool {
    mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
        let bond = &mol.bonds[bond_idx as usize];
        if bond.bond_type != BondType::Single {
            return false;
        }
        let other = if bond.source == atom_idx {
            bond.target
        } else {
            bond.source
        };
        !ring.contains(&other) && mol.atoms[other as usize].atomic_number != 6
    })
}

/// Checks whether fused exocyclic imine carbon.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_fused_exocyclic_imine_carbon(mol: &Molecule, atom_idx: AtomIdx, ring: &[AtomIdx]) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    if atom.atomic_number != 6
        || atom.formal_charge != 0
        || !has_exocyclic_imine_double(mol, atom_idx, ring)
    {
        return false;
    }

    let ring_neighbors: Vec<AtomIdx> = mol.adjacency[atom_idx as usize]
        .iter()
        .filter_map(|&bond_idx| {
            let bond = &mol.bonds[bond_idx as usize];
            let other = if bond.source == atom_idx {
                bond.target
            } else {
                bond.source
            };
            ring.contains(&other).then_some(other)
        })
        .collect();
    if ring_neighbors.len() != 2 {
        return false;
    }

    ring_neighbors.iter().any(|&ring_neighbor| {
        mol.adjacency[ring_neighbor as usize]
            .iter()
            .any(|&bond_idx| {
                let bond = &mol.bonds[bond_idx as usize];
                let other = if bond.source == ring_neighbor {
                    bond.target
                } else {
                    bond.source
                };
                other != atom_idx
                    && !ring.contains(&other)
                    && ring_neighbor_is_sp2_capable(mol, other, ring)
            })
    })
}

/// Checks whether fused conjugated ring neighbors.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn has_fused_conjugated_ring_neighbors(
    mol: &Molecule,
    atom_idx: AtomIdx,
    ring: &[AtomIdx],
) -> bool {
    let ring_neighbors: Vec<AtomIdx> = mol.adjacency[atom_idx as usize]
        .iter()
        .filter_map(|&bond_idx| {
            let bond = &mol.bonds[bond_idx as usize];
            let other = if bond.source == atom_idx {
                bond.target
            } else {
                bond.source
            };
            ring.contains(&other).then_some(other)
        })
        .collect();

    ring_neighbors.len() == 2
        && ring_neighbors
            .iter()
            .all(|&nbr| mol.atoms[nbr as usize].atomic_number == 6)
        && ring_neighbors
            .iter()
            .any(|&nbr| mol.atoms[nbr as usize].is_aromatic())
        && ring_neighbors
            .iter()
            .all(|&nbr| ring_neighbor_is_sp2_capable(mol, nbr, ring))
}

/// Implements neighbor is sp2 capable.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn ring_neighbor_is_sp2_capable(mol: &Molecule, atom_idx: AtomIdx, ring: &[AtomIdx]) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    atom.is_aromatic()
        || matches!(atom.atomic_number, 7 | 8 | 16)
        || (atom.atomic_number == 6 && atom.formal_charge != 0)
        || has_ring_pi_bond(mol, atom_idx, ring)
}

/// Implements gap carbons.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn quinonoid_gap_carbons(mol: &Molecule, ring: &[AtomIdx]) -> Option<[AtomIdx; 2]> {
    if ring.len() != 6 {
        return None;
    }

    for exo_pos in 0..ring.len() {
        let exo_atom = ring[exo_pos];
        let atom = &mol.atoms[exo_atom as usize];
        if atom.atomic_number != 6 || !has_exocyclic_hetero_double(mol, exo_atom, ring) {
            continue;
        }

        for dir in [1isize, -1isize] {
            let gap1_pos = wrap_ring_index(exo_pos as isize + dir, ring.len());
            let gap2_pos = wrap_ring_index(exo_pos as isize + 2 * dir, ring.len());
            let donor_pos = wrap_ring_index(exo_pos as isize + 3 * dir, ring.len());
            let anchor1_pos = wrap_ring_index(exo_pos as isize + 4 * dir, ring.len());
            let anchor2_pos = wrap_ring_index(exo_pos as isize + 5 * dir, ring.len());

            let gap1 = ring[gap1_pos];
            let gap2 = ring[gap2_pos];
            let donor = ring[donor_pos];
            let anchor1 = ring[anchor1_pos];
            let anchor2 = ring[anchor2_pos];

            if is_quinonoid_gap_carbon(mol, gap1, ring)
                && is_quinonoid_gap_carbon(mol, gap2, ring)
                && is_delocalized_ring_donor(mol, donor, ring)
                && is_aromatic_anchor_carbon(mol, anchor1)
                && is_aromatic_anchor_carbon(mol, anchor2)
            {
                return Some([gap1, gap2]);
            }
        }
    }

    None
}

/// Checks whether fused aza annulated ring.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_fused_aza_annulated_ring(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    if ring.len() != 6 {
        return false;
    }

    for start in 0..ring.len() {
        let ordered = rotate_ring(ring, start);
        if match_fused_aza_annulated_ring(mol, &ordered, ring) {
            return true;
        }

        let reversed = reverse_ring(&ordered);
        if match_fused_aza_annulated_ring(mol, &reversed, ring) {
            return true;
        }
    }

    false
}

/// Checks whether fused bis quinonoid ring.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_fused_bis_quinonoid_ring(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    if ring.len() != 6 {
        return false;
    }

    for start in 0..ring.len() {
        let ordered = rotate_ring(ring, start);
        if match_fused_bis_quinonoid_ring(mol, &ordered, ring) {
            return true;
        }

        let reversed = reverse_ring(&ordered);
        if match_fused_bis_quinonoid_ring(mol, &reversed, ring) {
            return true;
        }
    }

    false
}

/// Implements fused aza annulated ring.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn match_fused_aza_annulated_ring(mol: &Molecule, ordered: &[AtomIdx], ring: &[AtomIdx]) -> bool {
    let gap1 = ordered[0];
    let pi1 = ordered[1];
    let anchor = ordered[2];
    let donor = ordered[3];
    let pi2 = ordered[4];
    let gap2 = ordered[5];

    if !is_annulated_aza_gap_carbon(mol, gap1, ring)
        || !is_annulated_aza_gap_carbon(mol, gap2, ring)
        || !is_neutral_ring_carbon(mol, pi1)
        || !has_ring_pi_bond(mol, pi1, ring)
        || !is_aromatic_anchor_carbon(mol, anchor)
        || !is_neutral_degree3_ring_nitrogen(mol, donor)
        || !mol.atoms[donor as usize].is_aromatic()
        || !is_neutral_ring_carbon(mol, pi2)
        || !has_ring_pi_bond(mol, pi2, ring)
    {
        return false;
    }

    if !bond_matches(mol, gap1, pi1, &[BondType::Single, BondType::Aromatic])
        || !bond_matches(mol, pi1, anchor, &[BondType::Double, BondType::Aromatic])
        || !bond_matches(mol, anchor, donor, &[BondType::Aromatic])
        || !bond_matches(mol, donor, pi2, &[BondType::Double, BondType::Aromatic])
        || !bond_matches(mol, pi2, gap2, &[BondType::Single, BondType::Aromatic])
        || !bond_matches(mol, gap2, gap1, &[BondType::Single, BondType::Aromatic])
    {
        return false;
    }

    let Some(anchor_fused_neighbor) =
        unique_external_aromatic_neighbor(mol, anchor, ring, |atom| {
            atom.formal_charge > 0 && matches!(atom.atomic_number, 7 | 8 | 16) && atom.is_aromatic()
        })
    else {
        return false;
    };

    let Some(donor_fused_neighbor) = unique_external_aromatic_neighbor(mol, donor, ring, |atom| {
        atom.atomic_number == 6 && atom.formal_charge == 0 && atom.is_aromatic()
    }) else {
        return false;
    };

    has_short_aromatic_path_excluding_ring(
        mol,
        anchor_fused_neighbor,
        donor_fused_neighbor,
        ring,
        3,
    )
}

/// Implements fused bis quinonoid ring.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn match_fused_bis_quinonoid_ring(mol: &Molecule, ordered: &[AtomIdx], ring: &[AtomIdx]) -> bool {
    let carbonyl1 = ordered[0];
    let flank1 = ordered[1];
    let flank2 = ordered[2];
    let carbonyl2 = ordered[3];
    let flank3 = ordered[4];
    let flank4 = ordered[5];

    if !is_neutral_ring_carbon(mol, carbonyl1)
        || !has_exocyclic_hetero_double(mol, carbonyl1, ring)
        || !is_neutral_ring_carbon(mol, carbonyl2)
        || !has_exocyclic_hetero_double(mol, carbonyl2, ring)
    {
        return false;
    }

    for &atom_idx in &[flank1, flank2, flank3, flank4] {
        if !is_neutral_ring_carbon(mol, atom_idx)
            || has_exocyclic_hetero_double(mol, atom_idx, ring)
            || has_exocyclic_hetero_single(mol, atom_idx, ring)
            || (!mol.atoms[atom_idx as usize].is_aromatic()
                && !has_ring_pi_bond(mol, atom_idx, ring))
        {
            return false;
        }
    }

    if !carbonyl_has_annulated_support(mol, carbonyl1, flank4, flank1, ring)
        || !carbonyl_has_annulated_support(mol, carbonyl2, flank2, flank3, ring)
    {
        return false;
    }

    if !has_supported_annulated_flank_path(mol, flank1, flank2, ring, 5)
        && !has_supported_annulated_flank_path(mol, flank3, flank4, ring, 5)
    {
        return false;
    }

    [flank1, flank2, flank3, flank4]
        .into_iter()
        .filter(|&atom_idx| mol.atoms[atom_idx as usize].is_aromatic())
        .count()
        >= 2
}

/// Implements ring index.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn wrap_ring_index(idx: isize, len: usize) -> usize {
    idx.rem_euclid(len as isize) as usize
}

/// Checks whether quinonoid gap carbon.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_quinonoid_gap_carbon(mol: &Molecule, atom_idx: AtomIdx, ring: &[AtomIdx]) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    atom.atomic_number == 6
        && atom.formal_charge == 0
        && (usize::from(atom.total_h()) + mol.get_degree(atom_idx)) <= 3
        && !has_exocyclic_hetero_double(mol, atom_idx, ring)
        && !has_non_ring_pi_bond(mol, atom_idx, ring)
}

/// Checks whether annulated aza gap carbon.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_annulated_aza_gap_carbon(mol: &Molecule, atom_idx: AtomIdx, ring: &[AtomIdx]) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    atom.atomic_number == 6
        && atom.formal_charge == 0
        && (usize::from(atom.total_h()) + mol.get_degree(atom_idx)) <= 3
        && !has_exocyclic_hetero_double(mol, atom_idx, ring)
        && !has_non_ring_pi_bond(mol, atom_idx, ring)
}

/// Implements has annulated support.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn carbonyl_has_annulated_support(
    mol: &Molecule,
    carbonyl: AtomIdx,
    left: AtomIdx,
    right: AtomIdx,
    ring: &[AtomIdx],
) -> bool {
    let neighbors = [left, right];
    neighbors
        .iter()
        .any(|&atom_idx| mol.atoms[atom_idx as usize].is_aromatic())
        && neighbors
            .iter()
            .all(|&atom_idx| ring_neighbor_is_sp2_capable(mol, atom_idx, ring))
        && mol.adjacency[carbonyl as usize]
            .iter()
            .filter_map(|&bond_idx| {
                let bond = &mol.bonds[bond_idx as usize];
                let other = if bond.source == carbonyl {
                    bond.target
                } else {
                    bond.source
                };
                (!ring.contains(&other)
                    && bond.bond_type == BondType::Double
                    && mol.atoms[other as usize].atomic_number != 6)
                    .then_some(other)
            })
            .count()
            == 1
}

/// Checks whether supported annulated flank path.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn has_supported_annulated_flank_path(
    mol: &Molecule,
    start: AtomIdx,
    end: AtomIdx,
    ring: &[AtomIdx],
    max_edges: usize,
) -> bool {
    let mut visited = vec![[false; 2]; mol.num_atoms()];
    let mut queue = VecDeque::new();
    visited[start as usize][0] = true;
    queue.push_back((start, 0usize, false));

    while let Some((atom_idx, dist, saw_support)) = queue.pop_front() {
        if atom_idx == end {
            if saw_support {
                return true;
            }
            continue;
        }
        if dist >= max_edges {
            continue;
        }

        for &bond_idx in &mol.adjacency[atom_idx as usize] {
            let bond = &mol.bonds[bond_idx as usize];
            if !bond.is_aromatic() {
                continue;
            }
            let other = if bond.source == atom_idx {
                bond.target
            } else {
                bond.source
            };
            if ring.contains(&atom_idx) && ring.contains(&other) {
                continue;
            }
            if other != end && ring.contains(&other) {
                continue;
            }

            let next_support = saw_support
                || (other != start && other != end && is_annulated_path_support_atom(mol, other));
            let support_idx = usize::from(next_support);
            if visited[other as usize][support_idx] {
                continue;
            }
            visited[other as usize][support_idx] = true;
            queue.push_back((other, dist + 1, next_support));
        }
    }

    false
}

/// Checks whether annulated path support atom.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_annulated_path_support_atom(mol: &Molecule, atom_idx: AtomIdx) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    atom.is_aromatic() && (atom.atomic_number != 6 || aromatic_bond_degree(mol, atom_idx) >= 3)
}

/// Implements bond degree.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn aromatic_bond_degree(mol: &Molecule, atom_idx: AtomIdx) -> usize {
    mol.adjacency[atom_idx as usize]
        .iter()
        .filter(|&&bond_idx| mol.bonds[bond_idx as usize].is_aromatic())
        .count()
}

/// Implements external aromatic neighbor.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn unique_external_aromatic_neighbor<F>(
    mol: &Molecule,
    atom_idx: AtomIdx,
    ring: &[AtomIdx],
    predicate: F,
) -> Option<AtomIdx>
where
    F: Fn(&crate::mol::atom::Atom) -> bool,
{
    let neighbors = mol.adjacency[atom_idx as usize]
        .iter()
        .filter_map(|&bond_idx| {
            let bond = &mol.bonds[bond_idx as usize];
            if !bond.is_aromatic() {
                return None;
            }
            let other = if bond.source == atom_idx {
                bond.target
            } else {
                bond.source
            };
            (!ring.contains(&other) && predicate(&mol.atoms[other as usize])).then_some(other)
        })
        .collect::<Vec<_>>();

    if neighbors.len() == 1 {
        Some(neighbors[0])
    } else {
        None
    }
}

/// Checks whether short aromatic path excluding ring.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn has_short_aromatic_path_excluding_ring(
    mol: &Molecule,
    start: AtomIdx,
    end: AtomIdx,
    ring: &[AtomIdx],
    max_edges: usize,
) -> bool {
    let mut visited = vec![false; mol.num_atoms()];
    let mut queue = VecDeque::new();
    visited[start as usize] = true;
    queue.push_back((start, 0usize));

    while let Some((atom_idx, dist)) = queue.pop_front() {
        if atom_idx == end {
            return true;
        }
        if dist >= max_edges {
            continue;
        }

        for &bond_idx in &mol.adjacency[atom_idx as usize] {
            let bond = &mol.bonds[bond_idx as usize];
            if !bond.is_aromatic() {
                continue;
            }
            let other = if bond.source == atom_idx {
                bond.target
            } else {
                bond.source
            };
            if ring.contains(&other) || visited[other as usize] {
                continue;
            }
            visited[other as usize] = true;
            queue.push_back((other, dist + 1));
        }
    }

    false
}

/// Checks whether delocalized ring donor.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_delocalized_ring_donor(mol: &Molecule, atom_idx: AtomIdx, ring: &[AtomIdx]) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    match atom.atomic_number {
        7 => {
            atom.formal_charge == 0
                && atom.total_h() == 0
                && mol.get_degree(atom_idx) >= 3
                && has_ring_pi_bond(mol, atom_idx, ring)
        }
        8 | 16 => atom.formal_charge <= 0,
        _ => false,
    }
}

/// Checks whether aromatic anchor carbon.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_aromatic_anchor_carbon(mol: &Molecule, atom_idx: AtomIdx) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    atom.atomic_number == 6 && atom.formal_charge == 0 && atom.is_aromatic()
}

/// Checks whether any pi bond.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn has_any_pi_bond(mol: &Molecule, atom_idx: AtomIdx) -> bool {
    mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
        let bond = &mol.bonds[bond_idx as usize];
        matches!(bond.bond_type, BondType::Double | BondType::Aromatic)
    })
}

/// Checks whether non ring pi bond.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn has_non_ring_pi_bond(mol: &Molecule, atom_idx: AtomIdx, ring: &[AtomIdx]) -> bool {
    mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
        let bond = &mol.bonds[bond_idx as usize];
        if !matches!(bond.bond_type, BondType::Double | BondType::Aromatic) {
            return false;
        }
        let other = if bond.source == atom_idx {
            bond.target
        } else {
            bond.source
        };
        !ring.contains(&other)
    })
}

/// Check if ring has a conjugated system — every atom must be sp2-capable.
fn is_conjugated(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    let quinonoid_gap_carbons = quinonoid_gap_carbons(mol, ring);

    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];
        if let Some(bond_idx) = find_bond_between(mol, from, to) {
            if mol.bonds[bond_idx].bond_type == BondType::Triple {
                return false;
            }
        } else {
            return false;
        }
    }

    // Every atom must be sp2: has a double/aromatic bond (in-ring or exocyclic),
    // is a heteroatom (N, O, S) with a lone pair, or is a charged carbon (C⁻/C⁺).
    for &atom_idx in ring {
        let atom = &mol.atoms[atom_idx as usize];
        if matches!(atom.atomic_number, 7 | 8 | 16) {
            continue;
        }
        // Charged carbons are sp2-capable (C⁻ has lone pair, C⁺ has empty p-orbital)
        if atom.atomic_number == 6 && atom.formal_charge != 0 {
            continue;
        }
        let has_pi_bond = mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
            let bond = &mol.bonds[bond_idx as usize];
            matches!(bond.bond_type, BondType::Double | BondType::Aromatic)
        });
        if !has_pi_bond
            && !quinonoid_gap_carbons
                .as_ref()
                .is_some_and(|gaps| gaps.contains(&atom_idx))
        {
            return false;
        }
    }

    true
}

/// Find bond between two atoms.
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

/// Count π electrons in a ring according to aromaticity rules.
fn count_pi_electrons(mol: &Molecule, ring: &[AtomIdx]) -> u8 {
    let mut pi_electrons = 0;

    for &atom_idx in ring {
        let atom = &mol.atoms[atom_idx as usize];
        let element = atom.atomic_number;

        match element {
            6 => {
                // Carbon: contributes 1 π electron if sp2
                pi_electrons += 1;
            }
            7 => {
                // Nitrogen: distinguish pyridine-like (1π) vs pyrrole-like (2π)
                //
                // Pyridine-like: N with no H, 2 ring neighbors → lone pair NOT in π system
                // Pyrrole-like: N with H, or N with 3 ring neighbors → lone pair IN π system
                //
                // Also check for explicit double bond in ring or amide pattern

                let h_count = atom.total_h();

                // Count ring neighbors
                let ring_neighbor_count = mol.adjacency[atom_idx as usize]
                    .iter()
                    .filter(|&&bond_idx| {
                        let bond = &mol.bonds[bond_idx as usize];
                        let other = if bond.source == atom_idx {
                            bond.target
                        } else {
                            bond.source
                        };
                        ring.contains(&other)
                    })
                    .count();

                // Check for double bond in ring (explicit, not aromatic)
                let has_double_bond_in_ring =
                    mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
                        let bond = &mol.bonds[bond_idx as usize];
                        if bond.bond_type == BondType::Double {
                            let other = if bond.source == atom_idx {
                                bond.target
                            } else {
                                bond.source
                            };
                            ring.contains(&other)
                        } else {
                            false
                        }
                    });

                // Check for exocyclic C=O (amide nitrogen)
                let has_carbonyl_neighbor =
                    mol.adjacency[atom_idx as usize].iter().any(|&bond_idx| {
                        let bond = &mol.bonds[bond_idx as usize];
                        let neighbor = if bond.source == atom_idx {
                            bond.target
                        } else {
                            bond.source
                        };
                        let neighbor_atom = &mol.atoms[neighbor as usize];
                        if neighbor_atom.atomic_number == 6 && ring.contains(&neighbor) {
                            mol.adjacency[neighbor as usize].iter().any(|&bond_idx2| {
                                let bond2 = &mol.bonds[bond_idx2 as usize];
                                if bond2.bond_type == BondType::Double {
                                    let other2 = if bond2.source == neighbor {
                                        bond2.target
                                    } else {
                                        bond2.source
                                    };
                                    let other_atom2 = &mol.atoms[other2 as usize];
                                    other_atom2.atomic_number == 8 && !ring.contains(&other2)
                                } else {
                                    false
                                }
                            })
                        } else {
                            false
                        }
                    });

                // Pyridine-like: no H, exactly 2 ring neighbors, OR has explicit double bond
                // This covers both Kekulized (C=N-C) and aromatic (c:n:c) representations
                if has_double_bond_in_ring
                    || has_carbonyl_neighbor
                    || (h_count == 0 && ring_neighbor_count == 2)
                {
                    pi_electrons += 1; // Pyridine-like
                } else {
                    pi_electrons += 2; // Pyrrole-like (lone pair contributes)
                }
            }
            8 | 16 => {
                // Oxygen or Sulfur: contributes 2 π electrons (lone pair)
                pi_electrons += 2;
            }
            _ => {
                // Other elements: not considered for simple aromaticity
                return 0;
            }
        }
    }

    pi_electrons
}

/// Check if number of π electrons satisfies Hückel's 4n+2 rule.
fn is_huckel_aromatic(pi_electrons: u8) -> bool {
    pi_electrons >= 2 && (pi_electrons - 2) % 4 == 0
}

/// Mark all atoms and bonds in a ring as aromatic.
fn mark_ring_aromatic(mol: &mut Molecule, ring: &[AtomIdx]) {
    // Mark atoms
    for &atom_idx in ring {
        mol.atoms[atom_idx as usize].set_aromatic(true);
    }

    // Mark bonds - set aromatic flag AND bond type
    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];

        if let Some(bond_idx) = find_bond_between(mol, from, to) {
            mol.bonds[bond_idx].set_aromatic(true);
            mol.bonds[bond_idx].bond_type = BondType::Aromatic;
        }
    }
}

/// Implements kekulized perimeter aromatic.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn mark_kekulized_perimeter_aromatic(mol: &mut Molecule, perimeter: &KekulizedPerimeter) {
    for &atom_idx in &perimeter.ring {
        mol.atoms[atom_idx as usize].set_aromatic(true);
    }

    for i in 0..perimeter.ring.len() {
        let from = perimeter.ring[i];
        let to = perimeter.ring[(i + 1) % perimeter.ring.len()];
        if let Some(bond_idx) = find_bond_between(mol, from, to) {
            mol.bonds[bond_idx].set_aromatic(true);
            mol.bonds[bond_idx].bond_type = BondType::Aromatic;
        }
    }

    mol.bonds[perimeter.chord_bond_idx].bond_type = BondType::Single;
    mol.bonds[perimeter.chord_bond_idx].set_aromatic(false);
}

/// Implements fused bridge bonds.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn suppress_fused_bridge_bonds(mol: &mut Molecule, aromatic_rings: &[Vec<AtomIdx>]) {
    let mut bonds_to_kekulize = Vec::new();

    for ring in aromatic_rings {
        if let Some(bond_idx) = fused_bridge_bond_to_kekulize(mol, ring) {
            bonds_to_kekulize.push(bond_idx);
        }
    }

    bonds_to_kekulize.sort_unstable();
    bonds_to_kekulize.dedup();

    for bond_idx in bonds_to_kekulize {
        mol.bonds[bond_idx].bond_type = BondType::Single;
        mol.bonds[bond_idx].set_aromatic(false);
    }
}

/// Implements fused bridge ring bond aromaticity.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn suppress_fused_bridge_ring_bond_aromaticity(mol: &mut Molecule, aromatic_rings: &[Vec<AtomIdx>]) {
    let mut atoms_to_mark = Vec::new();
    let mut bonds_to_aromaticize = Vec::new();
    let mut bonds_to_kekulize = Vec::new();
    for ring in aromatic_rings {
        if let Some(bridge_bond_idx) = fused_diaza_dicarbonyl_bridge_bond(mol, ring) {
            atoms_to_mark.extend(ring.iter().copied());
            bonds_to_kekulize.push(bridge_bond_idx);
            for i in 0..ring.len() {
                let from = ring[i];
                let to = ring[(i + 1) % ring.len()];
                if let Some(bond_idx) = find_bond_between(mol, from, to) {
                    if bond_idx != bridge_bond_idx {
                        bonds_to_aromaticize.push(bond_idx);
                    }
                }
            }
        }
    }

    atoms_to_mark.sort_unstable();
    atoms_to_mark.dedup();
    for atom_idx in atoms_to_mark {
        mol.atoms[atom_idx as usize].set_aromatic(true);
    }

    bonds_to_aromaticize.sort_unstable();
    bonds_to_aromaticize.dedup();
    for bond_idx in bonds_to_aromaticize {
        mol.bonds[bond_idx].bond_type = BondType::Aromatic;
        mol.bonds[bond_idx].set_aromatic(true);
    }

    bonds_to_kekulize.sort_unstable();
    bonds_to_kekulize.dedup();
    for bond_idx in bonds_to_kekulize {
        mol.bonds[bond_idx].bond_type = BondType::Single;
        mol.bonds[bond_idx].set_aromatic(false);
    }
}

/// Implements diaza dicarbonyl bridge bond.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn fused_diaza_dicarbonyl_bridge_bond(mol: &Molecule, ring: &[AtomIdx]) -> Option<usize> {
    if ring.len() != 6 {
        return None;
    }

    let ring_set: HashSet<AtomIdx> = ring.iter().copied().collect();
    if ring_set.len() != ring.len() {
        return None;
    }

    let ring_atoms: Vec<AtomIdx> = ring_set.iter().copied().collect();
    let nitrogens: Vec<AtomIdx> = ring_atoms
        .iter()
        .copied()
        .filter(|&atom_idx| {
            let atom = &mol.atoms[atom_idx as usize];
            atom.atomic_number == 7
                && atom.formal_charge == 0
                && atom.total_h() == 0
                && mol.get_degree(atom_idx) == 2
        })
        .collect();
    if nitrogens.len() != 2 {
        return None;
    }

    let carbonyls: Vec<AtomIdx> = ring_atoms
        .iter()
        .copied()
        .filter(|&atom_idx| {
            let atom = &mol.atoms[atom_idx as usize];
            atom.atomic_number == 6 && has_exocyclic_hetero_double(mol, atom_idx, ring)
        })
        .collect();
    if carbonyls.len() != 1 {
        return None;
    }

    let exocyclic_single_carbons: Vec<AtomIdx> = ring_atoms
        .iter()
        .copied()
        .filter(|&atom_idx| {
            let atom = &mol.atoms[atom_idx as usize];
            atom.atomic_number == 6 && has_exocyclic_hetero_single(mol, atom_idx, ring)
        })
        .collect();
    if exocyclic_single_carbons.len() != 1 {
        return None;
    }

    let anchors: Vec<AtomIdx> = ring_atoms
        .iter()
        .copied()
        .filter(|&atom_idx| {
            let atom = &mol.atoms[atom_idx as usize];
            atom.atomic_number == 6
                && atom.formal_charge == 0
                && !carbonyls.contains(&atom_idx)
                && !exocyclic_single_carbons.contains(&atom_idx)
                && atom.is_aromatic()
        })
        .collect();
    if anchors.len() != 2 {
        return None;
    }

    // Require annulation support: each anchor must connect to at least one
    // aromatic atom outside the current ring.
    let annulated = anchors.iter().all(|&anchor| {
        mol.adjacency[anchor as usize].iter().any(|&bond_idx| {
            let bond = &mol.bonds[bond_idx as usize];
            let other = if bond.source == anchor {
                bond.target
            } else {
                bond.source
            };
            !ring_set.contains(&other) && mol.atoms[other as usize].is_aromatic()
        })
    });
    if !annulated {
        return None;
    }

    let bridge_bond_idx = find_bond_between(mol, anchors[0], anchors[1])?;

    // Remove the bridge edge and verify the remaining ring perimeter follows an
    // alternating N-C(=X)-N-C(X) interior chain between the two anchors.
    let path = ring_path_excluding_edge(mol, ring, anchors[0], anchors[1], bridge_bond_idx)?;
    if path.len() != ring.len() {
        return None;
    }
    let interior = &path[1..path.len() - 1];
    if interior.len() != 4 {
        return None;
    }
    let labels: Option<Vec<u8>> = interior
        .iter()
        .map(|atom_idx| {
            if nitrogens.contains(atom_idx) {
                Some(b'N')
            } else if carbonyls.contains(atom_idx) {
                Some(b'D') // exocyclic double (carbonyl-like)
            } else if exocyclic_single_carbons.contains(atom_idx) {
                Some(b'S') // exocyclic single hetero substituent
            } else {
                None
            }
        })
        .collect();
    let labels = labels?;
    let valid_forward = labels[0] == b'N'
        && labels[2] == b'N'
        && ((labels[1] == b'D' && labels[3] == b'S')
            || (labels[1] == b'S' && labels[3] == b'D'));
    let valid_reverse = labels[1] == b'N'
        && labels[3] == b'N'
        && ((labels[0] == b'D' && labels[2] == b'S')
            || (labels[0] == b'S' && labels[2] == b'D'));
    if !valid_forward && !valid_reverse {
        return None;
    }

    Some(bridge_bond_idx)
}

/// Implements path excluding edge.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn ring_path_excluding_edge(
    mol: &Molecule,
    ring: &[AtomIdx],
    start: AtomIdx,
    end: AtomIdx,
    excluded_bond_idx: usize,
) -> Option<Vec<AtomIdx>> {
    // Use molecule-sized prev/seen arrays — rings are small (≤64), so this is
    // allocation-light vs HashMap/HashSet with SipHash overhead.
    let n = mol.num_atoms();
    let mut prev = vec![u16::MAX; n];
    let mut seen = vec![false; n];
    let mut queue = VecDeque::new();
    queue.push_back(start);
    seen[start as usize] = true;

    while let Some(atom_idx) = queue.pop_front() {
        if atom_idx == end {
            break;
        }
        for &bond_idx in &mol.adjacency[atom_idx as usize] {
            let bidx = bond_idx as usize;
            if bidx == excluded_bond_idx {
                continue;
            }
            let bond = &mol.bonds[bidx];
            let other = if bond.source == atom_idx {
                bond.target
            } else {
                bond.source
            };
            if !ring.contains(&other) || seen[other as usize] {
                continue;
            }
            seen[other as usize] = true;
            prev[other as usize] = atom_idx;
            queue.push_back(other);
        }
    }

    if !seen[end as usize] {
        return None;
    }

    let mut path = vec![end];
    let mut current = end;
    while current != start {
        let p = prev[current as usize];
        if p == u16::MAX {
            return None;
        }
        current = p;
        path.push(current);
    }
    path.reverse();
    Some(path)
}

/// Implements bridge bond to kekulize.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn fused_bridge_bond_to_kekulize(mol: &Molecule, ring: &[AtomIdx]) -> Option<usize> {
    if ring.len() != 6 {
        return None;
    }

    let exocyclic_carbon_pos: Vec<usize> = ring
        .iter()
        .enumerate()
        .filter_map(|(pos, &atom_idx)| {
            let atom = &mol.atoms[atom_idx as usize];
            (atom.atomic_number == 6 && has_exocyclic_hetero_double(mol, atom_idx, ring))
                .then_some(pos)
        })
        .collect();
    if exocyclic_carbon_pos.len() != 1 {
        return None;
    }

    let ring_nitrogens: Vec<usize> = ring
        .iter()
        .enumerate()
        .filter_map(|(pos, &atom_idx)| {
            is_neutral_ring_pi_nitrogen(mol, atom_idx, ring).then_some(pos)
        })
        .collect();
    if ring_nitrogens.len() != 2 {
        return None;
    }

    let donor_positions: Vec<usize> = ring_nitrogens
        .iter()
        .copied()
        .filter(|&pos| mol.get_degree(ring[pos]) >= 3)
        .collect();
    let imine_positions: Vec<usize> = ring_nitrogens
        .iter()
        .copied()
        .filter(|&pos| mol.get_degree(ring[pos]) == 2)
        .collect();
    if donor_positions.len() != 1 || imine_positions.len() != 1 {
        return None;
    }

    let exo_pos = exocyclic_carbon_pos[0];
    let donor_pos = donor_positions[0];
    let imine_pos = imine_positions[0];
    let prev_pos = (exo_pos + ring.len() - 1) % ring.len();
    let next_pos = (exo_pos + 1) % ring.len();

    let bridge_pos = if prev_pos == donor_pos && is_neutral_ring_carbon(mol, ring[next_pos]) {
        next_pos
    } else if next_pos == donor_pos && is_neutral_ring_carbon(mol, ring[prev_pos]) {
        prev_pos
    } else {
        return None;
    };

    let bridge_other_pos = other_ring_neighbor_pos(bridge_pos, exo_pos, ring.len())?;
    if bridge_other_pos != imine_pos {
        return None;
    }

    let bond_idx = find_bond_between(mol, ring[exo_pos], ring[bridge_pos])?;
    mol.bonds[bond_idx].is_aromatic().then_some(bond_idx)
}

/// Checks whether neutral ring pi nitrogen.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_neutral_ring_pi_nitrogen(mol: &Molecule, atom_idx: AtomIdx, ring: &[AtomIdx]) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    atom.atomic_number == 7
        && atom.formal_charge == 0
        && atom.total_h() == 0
        && has_ring_pi_bond(mol, atom_idx, ring)
}

/// Checks whether neutral ring carbon.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn is_neutral_ring_carbon(mol: &Molecule, atom_idx: AtomIdx) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    atom.atomic_number == 6 && atom.formal_charge == 0
}

/// Implements ring neighbor pos.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
fn other_ring_neighbor_pos(pos: usize, excluded_neighbor: usize, len: usize) -> Option<usize> {
    let prev = (pos + len - 1) % len;
    let next = (pos + 1) % len;
    if prev == excluded_neighbor {
        Some(next)
    } else if next == excluded_neighbor {
        Some(prev)
    } else {
        None
    }
}

// Alias for backwards compatibility
/// Implements hueckel.
///
/// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
/// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
pub fn apply_hueckel(mol: &mut Molecule) {
    perceive_aromaticity(mol);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for benzene aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_benzene_aromatic() {
        // Benzene: c1ccccc1 (6 π electrons)
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        perceive_aromaticity(&mut mol);

        // All 6 carbons should be aromatic
        for atom in &mol.atoms {
            assert!(atom.is_aromatic(), "Benzene atoms should be aromatic");
        }
    }

    #[test]
    /// Runs regression test for pyridine aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_pyridine_aromatic() {
        // Pyridine: c1ccncc1 (6 π electrons: 5 C + 1 N)
        let mut mol = parse_smiles("c1ccncc1").unwrap();
        perceive_aromaticity(&mut mol);

        // All atoms should be aromatic
        for atom in &mol.atoms {
            assert!(atom.is_aromatic(), "Pyridine atoms should be aromatic");
        }
    }

    #[test]
    /// Runs regression test for cyclohexane not aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cyclohexane_not_aromatic() {
        // Cyclohexane: C1CCCCC1 (0 π electrons, all single bonds)
        let mut mol = parse_smiles("C1CCCCC1").unwrap();
        perceive_aromaticity(&mut mol);

        // No atoms should be aromatic
        for atom in &mol.atoms {
            assert!(!atom.is_aromatic(), "Cyclohexane should not be aromatic");
        }
    }

    #[test]
    /// Runs regression test for toluene aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_toluene_aromatic() {
        // Toluene: Cc1ccccc1 (benzene ring with methyl)
        let mut mol = parse_smiles("Cc1ccccc1").unwrap();
        perceive_aromaticity(&mut mol);

        // Benzene carbons should be aromatic (indices 1-6)
        for i in 1..7 {
            assert!(
                mol.atoms[i].is_aromatic(),
                "Benzene ring should be aromatic"
            );
        }

        // Methyl carbon should NOT be aromatic
        assert!(
            !mol.atoms[0].is_aromatic(),
            "Methyl carbon should not be aromatic"
        );
    }

    #[test]
    /// Runs regression test for huckel rule.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_huckel_rule() {
        // Test Hückel 4n+2 rule directly
        assert!(is_huckel_aromatic(2)); // n=0: 4(0)+2 = 2
        assert!(is_huckel_aromatic(6)); // n=1: 4(1)+2 = 6 (benzene)
        assert!(is_huckel_aromatic(10)); // n=2: 4(2)+2 = 10
        assert!(is_huckel_aromatic(14)); // n=3: 4(3)+2 = 14
        assert!(is_huckel_aromatic(22)); // n=5: 4(5)+2 = 22

        assert!(!is_huckel_aromatic(4)); // 4n rule (anti-aromatic)
        assert!(!is_huckel_aromatic(8)); // 4n rule
        assert!(!is_huckel_aromatic(0)); // No π electrons
    }

    #[test]
    /// Runs regression test for pyrrole 5 ring aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_pyrrole_5_ring_aromatic() {
        // Pyrrole: [nH]1cccc1 (6 π electrons in 5-membered ring)
        // N contributes 2 π electrons (lone pair), C contributes 1 each
        let mut mol = parse_smiles("[nH]1cccc1").unwrap();
        perceive_aromaticity(&mut mol);

        // All 5 atoms should be aromatic
        for atom in &mol.atoms {
            assert!(
                atom.is_aromatic(),
                "Pyrrole atoms should be aromatic (5-ring with 6π e-)"
            );
        }
    }

    #[test]
    /// Runs regression test for furan 5 ring aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_furan_5_ring_aromatic() {
        // Furan: o1cccc1 (6 π electrons in 5-membered ring)
        // O contributes 2 π electrons (lone pair), C contributes 1 each
        let mut mol = parse_smiles("o1cccc1").unwrap();
        perceive_aromaticity(&mut mol);

        // All 5 atoms should be aromatic
        for atom in &mol.atoms {
            assert!(
                atom.is_aromatic(),
                "Furan atoms should be aromatic (5-ring with 6π e-)"
            );
        }
    }

    #[test]
    /// Runs regression test for cyclopentadiene not aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cyclopentadiene_not_aromatic() {
        // Cyclopentadiene: C1=CC=CC1 (4 π electrons - anti-aromatic!)
        // This should NOT be aromatic (fails 4n+2 rule, has 4π which is 4n)
        let mut mol = parse_smiles("C1=CC=CC1").unwrap();
        perceive_aromaticity(&mut mol);

        // No atoms should be aromatic (4π electrons, not 4n+2)
        for atom in &mol.atoms {
            assert!(
                !atom.is_aromatic(),
                "Cyclopentadiene should not be aromatic (4π e-)"
            );
        }
    }

    #[test]
    /// Runs regression test for cyclopentadienyl anion aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cyclopentadienyl_anion_aromatic() {
        // Cyclopentadienyl anion: [cH-]1cccc1 (6 π electrons: 4C×1 + C⁻×2)
        let mut mol = parse_smiles("[cH-]1cccc1").unwrap();
        perceive_aromaticity(&mut mol);

        for atom in &mol.atoms {
            assert!(
                atom.is_aromatic(),
                "Cyclopentadienyl anion should be aromatic (6π e-)"
            );
        }
    }

    // ── Fused ring aromaticity (key regression tests for the junction-carbon fix) ──

    #[test]
    /// Runs regression test for indole aromatic smiles.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_indole_aromatic_smiles() {
        // Indole aromatic SMILES: all 9 ring atoms must be aromatic.
        let mut mol = parse_smiles("c1ccc2[nH]ccc2c1").unwrap();
        perceive_aromaticity(&mut mol);
        for (i, atom) in mol.atoms.iter().enumerate() {
            assert!(atom.is_aromatic(), "Indole atom {i} should be aromatic");
        }
    }

    #[test]
    /// Runs regression test for indole kekulized aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_indole_kekulized_aromatic() {
        // Indole in Kekulized form: junction carbons have their pi bond in the
        // 6-membered ring, so the old has_ring_pi_bond check wrongly gave them
        // 0 pi electrons in the 5-membered ring.
        // C1=CNC2=CC=CC=C12 : 5-ring [N,C1,C2(j),C3(j)] + 6-ring [C2(j),C3(j),C4–C7]
        let mut mol = parse_smiles("C1=CNC2=CC=CC=C12").unwrap();
        perceive_aromaticity(&mut mol);
        let aromatic_count = mol.atoms.iter().filter(|a| a.is_aromatic()).count();
        assert_eq!(
            aromatic_count, 9,
            "Kekulized indole: all 9 atoms should be aromatic, got {aromatic_count}"
        );
    }

    #[test]
    /// Runs regression test for benzofuran aromatic smiles.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_benzofuran_aromatic_smiles() {
        // Benzofuran aromatic SMILES: all 9 atoms aromatic.
        let mut mol = parse_smiles("c1ccc2occc2c1").unwrap();
        perceive_aromaticity(&mut mol);
        for (i, atom) in mol.atoms.iter().enumerate() {
            assert!(atom.is_aromatic(), "Benzofuran atom {i} should be aromatic");
        }
    }

    #[test]
    /// Runs regression test for benzofuran kekulized aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_benzofuran_kekulized_aromatic() {
        // Kekulized benzofuran: O in 5-ring, junction carbons carry their double
        // bond in the 6-ring (C3=C4 and C7=C8 in the 6-ring, ring-closures back
        // into 5-ring as single bonds).  All 9 atoms must be aromatic.
        let mut mol = parse_smiles("C1=COC2=CC=CC=C12").unwrap();
        perceive_aromaticity(&mut mol);
        let aromatic_count = mol.atoms.iter().filter(|a| a.is_aromatic()).count();
        assert_eq!(
            aromatic_count, 9,
            "Kekulized benzofuran: all 9 atoms should be aromatic, got {aromatic_count}"
        );
    }

    #[test]
    /// Runs regression test for benzothiophene kekulized aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_benzothiophene_kekulized_aromatic() {
        // Kekulized benzothiophene: S in 5-ring.  Same junction-carbon situation.
        let mut mol = parse_smiles("C1=CSC2=CC=CC=C12").unwrap();
        perceive_aromaticity(&mut mol);
        let aromatic_count = mol.atoms.iter().filter(|a| a.is_aromatic()).count();
        assert_eq!(
            aromatic_count, 9,
            "Kekulized benzothiophene: all 9 atoms should be aromatic, got {aromatic_count}"
        );
    }

    #[test]
    /// Runs regression test for fused thiophene benzofuran kekulized aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_fused_thiophene_benzofuran_kekulized_aromatic() {
        // CNP0317527-like fragment: benzofuran fused core in Kekulized SMILES.
        // COC(=O)C1=C(C)OC2=CC=C(...)C=C12  – the C1=...OC2=...C=C12 part is a
        // 5-ring (O,C,C junction,C junction) fused to a 6-ring.  With the old
        // has_ring_pi_bond logic the two junction carbons got 0 pi electrons and
        // the 5-ring was wrongly declared non-aromatic.
        let mut mol = parse_smiles("COC(=O)C1=C(C)OC2=CC=CC=C12").unwrap();
        perceive_aromaticity(&mut mol);
        // The fused bicyclic core (9 atoms: O + 8 C) must all be aromatic.
        // Methyl carbon (C), ester carbons, methoxy carbon are NOT in rings.
        let aromatic_count = mol.atoms.iter().filter(|a| a.is_aromatic()).count();
        assert_eq!(
            aromatic_count, 9,
            "Benzofuran core of ester derivative: 9 aromatic atoms expected, got {aromatic_count}"
        );
    }

    #[test]
    /// Runs regression test for cyclohexadiene not aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cyclohexadiene_not_aromatic() {
        // 1,3-cyclohexadiene has alternating double bonds but only 4 pi electrons
        // in the ring → NOT aromatic (Hückel requires 4n+2).
        let mut mol = parse_smiles("C1=CC=CCCC1").unwrap();
        perceive_aromaticity(&mut mol);
        for atom in &mol.atoms {
            assert!(!atom.is_aromatic(), "Cyclohexadiene should not be aromatic");
        }
    }

    #[test]
    /// Runs regression test for naphthalene all aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_naphthalene_all_aromatic() {
        // Naphthalene: two fused 6-membered rings, 10 π electrons total.
        // All 10 carbons must be aromatic.
        let mut mol = parse_smiles("C1=CC2=CC=CC=C2C=C1").unwrap();
        perceive_aromaticity(&mut mol);
        let aromatic_count = mol.atoms.iter().filter(|a| a.is_aromatic()).count();
        assert_eq!(
            aromatic_count, 10,
            "Naphthalene: all 10 atoms should be aromatic, got {aromatic_count}"
        );
    }

    #[test]
    /// Runs regression test for caffeine aromatic smiles.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_caffeine_aromatic_smiles() {
        // Aromatic caffeine SMILES uses neutral degree-3 aromatic nitrogens for the
        // methylated ring nitrogens. These still donate 2π electrons and must remain
        // aromatic so SMARTS like [CH3]a keep matching all three methyl substituents.
        let mut mol = parse_smiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C").unwrap();
        perceive_aromaticity(&mut mol);
        let aromatic_count = mol.atoms.iter().filter(|a| a.is_aromatic()).count();
        assert_eq!(
            aromatic_count, 9,
            "Aromatic caffeine core should have 9 aromatic atoms, got {aromatic_count}"
        );
    }

    #[test]
    /// Runs regression test for large macrocycle aromaticity.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_large_macrocycle_aromaticity() {
        let mut mol = parse_smiles(
            "CCCCCCCCCCCCCCCCCCOC(=O)CC[C@@H]1C2=NC(=C(C)C3=C(C)C([C@@H](C)O)=C(C=C4N=C(C=C5NC6=C2CC(=O)C6=C5C)C(CC)=C4C)N3)[C@H]1C",
        )
        .unwrap();
        perceive_aromaticity(&mut mol);
        let aromatic_count = mol.atoms.iter().filter(|a| a.is_aromatic()).count();
        assert_eq!(
            aromatic_count, 20,
            "Macrocyclic conjugated system should have 20 aromatic atoms, got {aromatic_count}"
        );
    }

    #[test]
    /// Runs regression test for flavylium fused ring detected and aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_flavylium_fused_ring_detected_and_aromatic() {
        let mut mol = parse_smiles(
            "OC[C@H]1O[C@H](OC2=CC(O)=CC3=[O+]C(C4=CC=C(O)C=C4)=C(O)C=C23)[C@H](O)[C@@H](O)[C@H]1O",
        )
        .unwrap();

        let local_cycles = find_local_cycles(&mol, 5, 24);
        let target = ring_set_key(&[6, 7, 8, 10, 11, 24]);
        assert!(
            local_cycles.iter().any(|ring| ring_set_key(ring) == target),
            "Expected fused flavylium ring to be detected in local cycles"
        );

        let ring = vec![6, 7, 8, 10, 11, 24];
        assert!(
            is_ring_aromatic(&mol, &ring),
            "Expected fused flavylium ring to satisfy aromaticity rules"
        );
    }

    #[test]
    /// Runs regression test for flavylium system reaches three aromatic rings.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_flavylium_system_reaches_three_aromatic_rings() {
        let mol = parse_smiles(
            "OC[C@H]1O[C@H](OC2=CC(O)=CC3=[O+]C(C4=CC=C(O)C=C4)=C(O)C=C23)[C@H](O)[C@@H](O)[C@H]1O",
        )
        .unwrap();
        let aromatic_count = mol.atoms.iter().filter(|a| a.is_aromatic()).count();
        assert_eq!(
            aromatic_count, 16,
            "Flavylium fused system should contain 16 aromatic atoms after re-perception, got {aromatic_count}"
        );
    }

    #[test]
    /// Runs regression test for polycyclic quinonoid system reaches rdkit aromatic atom count.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_polycyclic_quinonoid_system_reaches_rdkit_aromatic_atom_count() {
        let mol = parse_smiles("CC1=CC(O)=C2C(=O)C3=C(C(=O)C2=C1)C1=C(O3)C2=C(C=C1C)C(=O)C=CC2=O")
            .unwrap();
        let aromatic_count = mol.atoms.iter().filter(|a| a.is_aromatic()).count();
        assert_eq!(
            aromatic_count, 21,
            "Polycyclic quinonoid system should contain 21 aromatic atoms, got {aromatic_count}"
        );
    }

    #[test]
    /// Runs regression test for chromone gap pattern detected.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_chromone_gap_pattern_detected() {
        let mol = parse_smiles("CC(C)(C)C1=CC=C(C2=CC(=O)C3=CC=C(NC(=O)C4=CC=CC=C4Cl)C=C3O2)C=C1")
            .unwrap();
        let ring = vec![12, 10, 9, 8, 28, 27];
        let gaps = quinonoid_gap_carbons(&mol, &ring).expect("expected chromone gap carbons");
        assert!(gaps.contains(&8) && gaps.contains(&9));
        assert!(is_ring_aromatic(&mol, &ring));
    }

    #[test]
    /// Runs regression test for quinolone gap pattern detected.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_quinolone_gap_pattern_detected() {
        let mol = parse_smiles(
            "CN1C(O)=C(C(=O)C[C@@H](NC2=CC=C(N)C=C2)C2=C(O)C=CC3=CC=CC=C23)C(=O)C2=CC=CC=C21",
        )
        .unwrap();
        let ring = vec![28, 4, 2, 1, 35, 30];
        let gaps = quinonoid_gap_carbons(&mol, &ring).expect("expected quinolone gap carbons");
        assert!(gaps.contains(&2) && gaps.contains(&4));
        assert!(is_ring_aromatic(&mol, &ring));
    }

    #[test]
    /// Runs regression test for flavanone gap pattern rejected.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_flavanone_gap_pattern_rejected() {
        let mol = parse_smiles("O=C1CC(C2=CC(O)=CC(O)=C2)OC2=CC=CC=C12").unwrap();
        let ring = vec![12, 3, 2, 1, 18, 13];
        assert!(
            quinonoid_gap_carbons(&mol, &ring).is_none(),
            "flavanone bridge ring should not match chromone gap heuristic"
        );
        assert!(!is_ring_aromatic(&mol, &ring));
    }

    #[test]
    /// Runs regression test for glycosylated chromanone gap pattern rejected.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_glycosylated_chromanone_gap_pattern_rejected() {
        let mol = parse_smiles(
            "O=C(C=CC1=CC=C(O)C=C1)OC[C@@H]1O[C@@H](O[C@H]2C(=O)C3=C(O)C=C(O[C@@H]4O[C@@H](CO)[C@H](O)[C@@H](O)[C@@H]4O)C=C3O[C@H]2C2=CC=C(O)C=C2)[C@H](O)[C@@H](O)[C@@H]1O",
        )
        .unwrap();
        let ring = vec![38, 20, 18, 17, 40, 39];
        assert!(
            quinonoid_gap_carbons(&mol, &ring).is_none(),
            "glycosylated chromanone bridge ring should not match chromone gap heuristic"
        );
        assert!(!is_ring_aromatic(&mol, &ring));
    }

    #[test]
    /// Runs regression test for bis quinone bridge ring rejected.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_bis_quinone_bridge_ring_rejected() {
        let mol = parse_smiles(
            "C/C1=C/[C@H](C)[C@@H](O)[C@@H](C)/C=C/[C@@H](O)C/C=C(\\C)C(=O)C[C@@H](O)[C@@H](C)C=C/C=C/C=C\\C(=O)NC2=CC(=O)C3=C(C=C(C)C(O)=C3C1=O)C2=O",
        )
        .unwrap();
        let ring = vec![33, 34, 35, 37, 38, 47];
        assert!(
            ring_exocyclic_hetero_double_count(&mol, &ring) >= 2,
            "bis-quinone bridge ring should contain two exocyclic hetero doubles"
        );
        assert!(
            !is_ring_aromatic(&mol, &ring),
            "bis-quinone bridge ring should stay kekulized"
        );
        assert!(
            !is_fused_bis_quinonoid_ring(&mol, &ring),
            "bis-quinone bridge ring should not match annulated bis-quinonoid helper"
        );
    }

    #[test]
    /// Runs regression test for xanthone bridge ring is aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_xanthone_bridge_ring_is_aromatic() {
        let mol = parse_smiles("COC1=CC(O)=C2C(=O)C3=CC=C(O)C(OC)=C3OC2=C1").unwrap();
        let ring = vec![6, 7, 9, 17, 18, 19];
        assert_eq!(
            ring_exocyclic_hetero_double_count(&mol, &ring),
            1,
            "xanthone bridge ring should have a single exocyclic hetero double"
        );
        assert!(is_ring_aromatic(&mol, &ring));
    }

    #[test]
    /// Runs regression test for fused aza annulated ring is aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_fused_aza_annulated_ring_is_aromatic() {
        let mol =
            parse_smiles("CC1=CC2=[N+](CC(=O)C3=CC=C([N+](=O)[O-])C=C3)C(C)=CN2C=C1").unwrap();
        let ring = vec![1, 2, 3, 20, 21, 22];
        assert!(is_fused_aza_annulated_ring(&mol, &ring));
    }

    #[test]
    /// Runs regression test for cnp0331134 left bis quinonoid ring is aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cnp0331134_left_bis_quinonoid_ring_is_aromatic() {
        let mol = parse_smiles("CC1=CC(O)=C2C(=O)C3=C(C(=O)C2=C1)C1=C(O3)C2=C(C=C1C)C(=O)C=CC2=O")
            .unwrap();
        let ring = vec![6, 8, 9, 10, 12, 5];
        assert!(is_fused_bis_quinonoid_ring(&mol, &ring));
    }

    #[test]
    /// Runs regression test for cnp0331134 right bis quinonoid ring is aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cnp0331134_right_bis_quinonoid_ring_is_aromatic() {
        let mol = parse_smiles("CC1=CC(O)=C2C(=O)C3=C(C(=O)C2=C1)C1=C(O3)C2=C(C=C1C)C(=O)C=CC2=O")
            .unwrap();
        let ring = vec![22, 24, 25, 26, 17, 18];
        assert!(is_fused_bis_quinonoid_ring(&mol, &ring));
    }

    #[test]
    /// Runs regression test for anthraquinone bridge ring rejected by bis quinonoid guard.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_anthraquinone_bridge_ring_rejected_by_bis_quinonoid_guard() {
        let mol = parse_smiles(
            "CC1=CC(O)=C2C(=O)C3=C(O[C@@H]4O[C@@H](COC(=O)CC(=O)O)[C@@H](O)[C@@H](O)[C@H]4O)C=CC=C3C(=O)C2=C1",
        )
        .unwrap();
        let ring = vec![6, 8, 31, 32, 34, 5];
        assert!(
            !is_fused_bis_quinonoid_ring(&mol, &ring),
            "anthraquinone bridge ring should stay kekulized"
        );
    }

    #[test]
    /// Runs regression test for cycle search finds chromone gap ring.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cycle_search_finds_chromone_gap_ring() {
        let mut mol =
            parse_smiles("CC(C)(C)C1=CC=C(C2=CC(=O)C3=CC=C(NC(=O)C4=CC=CC=C4Cl)C=C3O2)C=C1")
                .unwrap();
        let target = ring_set_key(&[12, 10, 9, 8, 28, 27]);
        let sssr = find_sssr(&mut mol);
        let local = find_local_cycles(&mol, 5, 24);
        let dfs = find_dfs_cycles(&mol, 5, 8, 2000);
        let found = sssr
            .iter()
            .chain(local.iter())
            .chain(dfs.iter())
            .find(|ring| ring_set_key(ring) == target)
            .cloned();
        assert!(
            found.is_some(),
            "combined cycle search missed chromone gap ring; sssr={}, local={}, dfs={}",
            sssr.len(),
            local.len(),
            dfs.len()
        );
        let found = found.unwrap();
        assert!(
            is_ring_aromatic(&mol, &found),
            "found chromone ring order is not aromatic: {found:?}"
        );
    }

    #[test]
    /// Runs regression test for cycle search finds quinolone gap ring.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cycle_search_finds_quinolone_gap_ring() {
        let mut mol = parse_smiles(
            "CN1C(O)=C(C(=O)C[C@@H](NC2=CC=C(N)C=C2)C2=C(O)C=CC3=CC=CC=C23)C(=O)C2=CC=CC=C21",
        )
        .unwrap();
        let target = ring_set_key(&[28, 4, 2, 1, 35, 30]);
        let sssr = find_sssr(&mut mol);
        let local = find_local_cycles(&mol, 5, 24);
        let dfs = find_dfs_cycles(&mol, 5, 8, 2000);
        let found = sssr
            .iter()
            .chain(local.iter())
            .chain(dfs.iter())
            .find(|ring| ring_set_key(ring) == target)
            .cloned();
        assert!(
            found.is_some(),
            "combined cycle search missed quinolone gap ring; sssr={}, local={}, dfs={}",
            sssr.len(),
            local.len(),
            dfs.len()
        );
        let found = found.unwrap();
        assert!(
            is_ring_aromatic(&mol, &found),
            "found quinolone ring order is not aromatic: {found:?}"
        );
    }

    #[test]
    /// Runs regression test for second aromaticity pass recovers chromone gap ring.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_second_aromaticity_pass_recovers_chromone_gap_ring() {
        let mut mol =
            parse_smiles("CC(C)(C)C1=CC=C(C2=CC(=O)C3=CC=C(NC(=O)C4=CC=CC=C4Cl)C=C3O2)C=C1")
                .unwrap();
        let before = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
        perceive_aromaticity(&mut mol);
        let after = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
        assert!(
            after >= 22,
            "expected second aromaticity pass to recover chromone gap ring, before={before}, after={after}"
        );
    }

    #[test]
    /// Runs regression test for second aromaticity pass recovers quinolone gap ring.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_second_aromaticity_pass_recovers_quinolone_gap_ring() {
        let mut mol = parse_smiles(
            "CN1C(O)=C(C(=O)C[C@@H](NC2=CC=C(N)C=C2)C2=C(O)C=CC3=CC=CC=C23)C(=O)C2=CC=CC=C21",
        )
        .unwrap();
        let before = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
        perceive_aromaticity(&mut mol);
        let after = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
        assert!(
            after >= 26,
            "expected second aromaticity pass to recover quinolone gap ring, before={before}, after={after}"
        );
    }

    #[test]
    /// Runs regression test for current aromaticity scan sees chromone gap ring.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_current_aromaticity_scan_sees_chromone_gap_ring() {
        let mut mol =
            parse_smiles("CC(C)(C)C1=CC=C(C2=CC(=O)C3=CC=C(NC(=O)C4=CC=CC=C4Cl)C=C3O2)C=C1")
                .unwrap();
        let target = ring_set_key(&[12, 10, 9, 8, 28, 27]);
        let sssr = find_sssr(&mut mol);
        let local = find_local_cycles(&mol, 5, 24);
        let dfs = find_dfs_cycles(&mol, 5, 8, 2000);
        let mut rings = Vec::new();
        let mut seen: HashSet<Vec<AtomIdx>> = HashSet::new();
        for ring in sssr.iter().chain(local.iter()).chain(dfs.iter()) {
            let key = ring_set_key(ring);
            if seen.insert(key) {
                rings.push(ring.clone());
            }
        }
        let matching: Vec<_> = rings
            .iter()
            .filter(|ring| ring_set_key(ring) == target)
            .cloned()
            .collect();
        assert!(
            matching.iter().any(|ring| is_ring_aromatic(&mol, ring)),
            "deduped aromaticity scan missed chromone gap ring; matching={matching:?}"
        );
    }

    #[test]
    /// Runs regression test for cnp0449381 fused quinonoid perimeters detected.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cnp0449381_fused_quinonoid_perimeters_detected() {
        let mut mol = parse_smiles(
            "CC1=CC(O)=C2C(=CC3=CC(OS(=O)(=O)O)=C(C4=C(OS(=O)(=O)O)C=C(O)C5=C4C=C4OC(C)=CC(O)=C4C5=O)C(O)=C3C2=O)O1",
        )
        .unwrap();

        let sssr = find_sssr(&mut mol);
        let local = find_local_cycles(&mol, 5, 24);
        let dfs = find_dfs_cycles(&mol, 5, 8, 2000);
        let mut rings = Vec::new();
        let mut seen: HashSet<Vec<AtomIdx>> = HashSet::new();
        for ring in sssr.iter().chain(local.iter()).chain(dfs.iter()) {
            let key = ring_set_key(ring);
            if seen.insert(key) {
                rings.push(ring.clone());
            }
        }

        let perimeters: Vec<_> = rings
            .iter()
            .filter_map(|ring| fused_quinonoid_perimeter(&mol, ring))
            .collect();
        assert_eq!(
            perimeters.len(),
            2,
            "expected two fused quinonoid perimeters"
        );
    }

    #[test]
    /// Runs regression test for cnp0449381 matches rdkit aromatic targets.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cnp0449381_matches_rdkit_aromatic_targets() {
        let mol = parse_smiles(
            "CC1=CC(O)=C2C(=CC3=CC(OS(=O)(=O)O)=C(C4=C(OS(=O)(=O)O)C=C(O)C5=C4C=C4OC(C)=CC(O)=C4C5=O)C(O)=C3C2=O)O1",
        )
        .unwrap();
        let aromatic_count = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
        assert_eq!(aromatic_count, 28);
        assert_eq!(
            crate::desc::aromatic_ring_count::compute_aromatic_ring_count(&mol),
            2
        );

        for &(a, b) in &[(5u16, 6u16), (30u16, 37u16)] {
            let bond_idx = find_bond_between(&mol, a, b).expect("expected fused bridge bond");
            assert_eq!(mol.bonds[bond_idx].bond_type, BondType::Single);
            assert!(!mol.bonds[bond_idx].is_aromatic());
            assert!(mol.atoms[a as usize].is_aromatic());
            assert!(mol.atoms[b as usize].is_aromatic());
        }
    }

    #[test]
    /// Runs regression test for cnp0515307 fused aza perimeter detected.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cnp0515307_fused_aza_perimeter_detected() {
        let mut mol = parse_smiles(
            "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2",
        )
        .unwrap();

        let sssr = find_sssr(&mut mol);
        let local = find_local_cycles(&mol, 5, 24);
        let dfs = find_dfs_cycles(&mol, 5, 8, 2000);
        let mut rings = Vec::new();
        let mut seen: HashSet<Vec<AtomIdx>> = HashSet::new();
        for ring in sssr.iter().chain(local.iter()).chain(dfs.iter()) {
            let key = ring_set_key(ring);
            if seen.insert(key) {
                rings.push(ring.clone());
            }
        }

        let perimeters: Vec<_> = rings
            .iter()
            .filter_map(|ring| fused_quinonoid_perimeter(&mol, ring))
            .collect();
        assert_eq!(perimeters.len(), 1, "expected one fused aza perimeter");
    }

    #[test]
    /// Runs regression test for cnp0515307 matches rdkit aromatic targets.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cnp0515307_matches_rdkit_aromatic_targets() {
        let mol = parse_smiles(
            "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2",
        )
        .unwrap();
        let aromatic_count = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
        assert_eq!(aromatic_count, 23);
        assert_eq!(
            crate::desc::aromatic_ring_count::compute_aromatic_ring_count(&mol),
            3
        );

        let bridge_bond =
            find_bond_between(&mol, 44u16, 51u16).expect("expected fused bridge bond");
        assert_eq!(mol.bonds[bridge_bond].bond_type, BondType::Single);
        assert!(!mol.bonds[bridge_bond].is_aromatic());
        for atom_idx in [44usize, 45, 46, 48, 49, 51] {
            assert!(
                mol.atoms[atom_idx].is_aromatic(),
                "atom {atom_idx} should be aromatic"
            );
        }
    }

    #[test]
    /// Runs regression test for cnp0504109 benzene cycle detected in ring search.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cnp0504109_benzene_cycle_detected_in_ring_search() {
        let mut mol = parse_smiles(
            "C/C=C1/CN2C=C(C[C@H](C(=O)OC)[C@H]3C[C@H]4C5=C(CCN4C/C3=C/C)C3=CC=CC=C3N5)C3=C4C=CC=CC4=NC3=C2C[C@@H]1CCO",
        )
        .unwrap();

        let sssr = find_sssr(&mut mol);
        let local = find_local_cycles(&mol, 5, 24);
        let dfs = find_dfs_cycles(&mol, 5, 8, 2000);
        let target = ring_set_key(&[33u16, 34, 35, 36, 37, 38]);

        let matching: Vec<_> = sssr
            .iter()
            .chain(local.iter())
            .chain(dfs.iter())
            .filter(|ring| ring_set_key(ring) == target)
            .cloned()
            .collect();

        assert!(
            !matching.is_empty(),
            "expected fused benzene cycle in candidate rings"
        );
    }

    #[test]
    /// Runs regression test for cnp0504109 benzene cycle is aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cnp0504109_benzene_cycle_is_aromatic() {
        let mut mol = parse_smiles(
            "C/C=C1/CN2C=C(C[C@H](C(=O)OC)[C@H]3C[C@H]4C5=C(CCN4C/C3=C/C)C3=CC=CC=C3N5)C3=C4C=CC=CC4=NC3=C2C[C@@H]1CCO",
        )
        .unwrap();

        let sssr = find_sssr(&mut mol);
        let local = find_local_cycles(&mol, 5, 24);
        let dfs = find_dfs_cycles(&mol, 5, 8, 2000);
        let target = ring_set_key(&[33u16, 34, 35, 36, 37, 38]);

        let matching: Vec<_> = sssr
            .iter()
            .chain(local.iter())
            .chain(dfs.iter())
            .filter(|ring| ring_set_key(ring) == target)
            .cloned()
            .collect();

        assert!(
            matching.iter().any(|ring| is_ring_aromatic(&mol, ring)),
            "expected fused benzene cycle to satisfy aromaticity rules; matching={matching:?}"
        );
    }

    #[test]
    /// Runs regression test for cnp0504109 fused aza bridge perimeter detected.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cnp0504109_fused_aza_bridge_perimeter_detected() {
        let mut mol = parse_smiles(
            "C/C=C1/CN2C=C(C[C@H](C(=O)OC)[C@H]3C[C@H]4C5=C(CCN4C/C3=C/C)C3=CC=CC=C3N5)C3=C4C=CC=CC4=NC3=C2C[C@@H]1CCO",
        )
        .unwrap();

        let sssr = find_sssr(&mut mol);
        let local = find_local_cycles(&mol, 5, 24);
        let dfs = find_dfs_cycles(&mol, 5, 8, 2000);
        let target = ring_set_key(&[33u16, 32, 6, 5, 4, 41, 40, 39, 38]);
        let matching: Vec<_> = sssr
            .iter()
            .chain(local.iter())
            .chain(dfs.iter())
            .filter(|ring| ring_set_key(ring) == target)
            .cloned()
            .collect();
        let perimeters: Vec<_> = matching
            .iter()
            .filter_map(|ring| fused_quinonoid_perimeter(&mol, ring))
            .collect();

        assert!(
            perimeters
                .iter()
                .any(|perimeter| ring_set_key(&perimeter.ring) == target),
            "expected fused aza bridge perimeter; matching={matching:?}"
        );
    }

    #[test]
    /// Runs regression test for cnp0504109 matches rdkit aromatic targets.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cnp0504109_matches_rdkit_aromatic_targets() {
        let mol = parse_smiles(
            "C/C=C1/CN2C=C(C[C@H](C(=O)OC)[C@H]3C[C@H]4C5=C(CCN4C/C3=C/C)C3=CC=CC=C3N5)C3=C4C=CC=CC4=NC3=C2C[C@@H]1CCO",
        )
        .unwrap();
        let aromatic_count = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
        assert_eq!(aromatic_count, 22);
        assert_eq!(
            crate::desc::aromatic_ring_count::compute_aromatic_ring_count(&mol),
            3
        );

        let chord_bond = find_bond_between(&mol, 32u16, 40u16).expect("expected fused bridge bond");
        assert_eq!(mol.bonds[chord_bond].bond_type, BondType::Single);
        assert!(!mol.bonds[chord_bond].is_aromatic());
        for atom_idx in [4usize, 5, 6, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41] {
            assert!(
                mol.atoms[atom_idx].is_aromatic(),
                "atom {atom_idx} should be aromatic"
            );
        }
    }

    #[test]
    /// Runs regression test for cnp0042975 matches rdkit aromatic targets.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cnp0042975_matches_rdkit_aromatic_targets() {
        let mol = parse_smiles("O=C1CC(C2=CC(O)=CC(O)=C2)OC2=CC=CC=C12").unwrap();
        let aromatic_count = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
        assert_eq!(aromatic_count, 12);
        assert_eq!(
            crate::desc::aromatic_ring_count::compute_aromatic_ring_count(&mol),
            2
        );
        for atom_idx in [1usize, 2, 3, 12] {
            assert!(
                !mol.atoms[atom_idx].is_aromatic(),
                "atom {atom_idx} should stay kekulized"
            );
        }
    }

    #[test]
    /// Runs regression test for cnp0296962 matches rdkit aromatic targets.
    ///
    /// Function context: `crates/ferrum_core/src/algo/aromaticity.rs`.
    /// Algorithm-layer note: this operates on molecular graph topology, so index and adjacency invariants must remain valid.
    fn test_cnp0296962_matches_rdkit_aromatic_targets() {
        let mol = parse_smiles(
            "O=C(C=CC1=CC=C(O)C=C1)OC[C@@H]1O[C@@H](O[C@H]2C(=O)C3=C(O)C=C(O[C@@H]4O[C@@H](CO)[C@H](O)[C@@H](O)[C@@H]4O)C=C3O[C@H]2C2=CC=C(O)C=C2)[C@H](O)[C@@H](O)[C@@H]1O",
        )
        .unwrap();
        let aromatic_count = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
        assert_eq!(aromatic_count, 18);
        assert_eq!(
            crate::desc::aromatic_ring_count::compute_aromatic_ring_count(&mol),
            3
        );
        for atom_idx in [17usize, 18, 39, 40] {
            assert!(
                !mol.atoms[atom_idx].is_aromatic(),
                "atom {atom_idx} should stay kekulized"
            );
        }
    }
}
