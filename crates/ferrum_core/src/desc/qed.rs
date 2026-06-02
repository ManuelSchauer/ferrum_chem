//! QED (Quantitative Estimate of Drug-likeness) descriptor.
//!
//! Combines multiple molecular properties into a single drug-likeness score (0-1).
//!
//! Reference: Bickerton et al. (2012) "Quantifying the chemical beauty of drugs"
//! Nature Chemistry 4, 90-98.

//! ## File guide: Qed
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct QedParam`
//! - `fn ads`
//! - `struct QueryScreen`
//! - `struct RequiredElementSet`
//! - `struct ScreenedQuery`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Qed` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Qed` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/qed.rs` defines qed behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Qed` workflows.
//! - Repository unit and parity tests covering `Qed` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/qed.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::iso::{count_matches, has_substruct_match};
use crate::algo::rings::{
    ensure_minimum_cycle_basis_bonds_cached, find_minimum_cycle_basis_bonds, find_sssr,
};
use crate::desc::bond_count::{compute_rotatable_bond_count, compute_rotatable_bond_count_cached};
use crate::desc::hbond::{compute_hbond_donors, compute_hbond_donors_cached};
use crate::desc::logp::{compute_logp, compute_logp_cached};
use crate::desc::molecular_weight::compute_molecular_weight;
use crate::desc::tpsa::{compute_tpsa, compute_tpsa_cached};
use crate::io::smarts::{AtomPrimitive, QueryAtom, QueryGraph, parse_smarts};
use crate::mol::molecule::Molecule;
use once_cell::sync::Lazy;
use smallvec::SmallVec;

/// QED parameters: (mean, sigma, weight) for each property.
/// From Bickerton et al. Table 1 (weighted version).
struct QedParam {
    ads: [f64; 7],
    weight: f64,
}

// ADS function parameters from RDKit's QED implementation
// Order: MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS
static QED_PARAMS: &[QedParam] = &[
    // MW
    QedParam {
        ads: [
            2.817065973,
            392.5754953,
            290.7489764,
            2.419764353,
            49.22325677,
            65.37051707,
            104.9805561,
        ],
        weight: 0.66,
    },
    // ALOGP
    QedParam {
        ads: [
            3.172690585,
            137.8624751,
            2.534937431,
            4.581497897,
            0.822739154,
            0.576295591,
            131.3186604,
        ],
        weight: 0.46,
    },
    // HBA
    QedParam {
        ads: [
            2.948620388,
            160.4605972,
            3.615294657,
            4.435986202,
            0.290141953,
            1.300669958,
            148.7763046,
        ],
        weight: 0.05,
    },
    // HBD
    QedParam {
        ads: [
            1.618662227,
            1010.051101,
            0.985094388,
            0.000000001,
            0.713820843,
            0.920922555,
            258.1632616,
        ],
        weight: 0.61,
    },
    // PSA (TPSA)
    QedParam {
        ads: [
            1.876861559,
            125.2232657,
            62.90773554,
            87.83366614,
            12.01999824,
            28.51324732,
            104.5686167,
        ],
        weight: 0.06,
    },
    // ROTB
    QedParam {
        ads: [
            0.010000000,
            272.4121427,
            2.558379970,
            1.565547684,
            1.271567166,
            2.758063707,
            105.4420403,
        ],
        weight: 0.65,
    },
    // AROM (aromatic rings)
    QedParam {
        ads: [
            3.217788970,
            957.7374108,
            2.274627939,
            0.000000001,
            1.317690384,
            0.375760881,
            312.3372610,
        ],
        weight: 0.48,
    },
    // ALERTS (structural alerts — we assume 0 alerts since we don't implement alert detection)
    QedParam {
        ads: [
            0.010000000,
            1199.094025,
            -0.09002883,
            0.000000001,
            0.185904477,
            0.875193782,
            417.7253140,
        ],
        weight: 0.95,
    },
];

/// Asymmetric Double Sigmoidal function used in QED calculation.
fn ads(x: f64, params: &[f64; 7]) -> f64 {
    let a = params[0];
    let b = params[1];
    let c = params[2];
    let d = params[3];
    let e = params[4];
    let f = params[5];
    let dmax = params[6];

    let numerator = a + b / (1.0 + (-1.0 * (x - c + d / 2.0) / e).exp())
        * (1.0 - 1.0 / (1.0 + (-1.0 * (x - c - d / 2.0) / f).exp()));
    numerator / dmax
}

/// RDKit QED uses its own HBA SMARTS set instead of the generic Lipinski-style HBA count.
static QED_ACCEPTOR_SMARTS: &[&str] = &[
    "[oH0;X2]",
    "[OH1;X2;v2]",
    "[OH0;X2;v2]",
    "[OH0;X1;v2]",
    "[O-;X1]",
    "[SH0;X2;v2]",
    "[SH0;X1;v2]",
    "[S-;X1]",
    "[nH0;X2]",
    "[NH0;X1;v3]",
    "[$([N;+0;X3;v3]);!$(N[C,S]=O)]",
];

#[derive(Clone)]
/// Implements QueryScreen.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-layer note: computes molecular properties for ML pipelines.
struct QueryScreen {
    min_atoms: usize,
    min_bonds: usize,
    required_elements: SmallVec<[(u8, u16); 8]>,
    required_element_sets: SmallVec<[RequiredElementSet; 8]>,
    requires_ring_atom: bool,
    requires_aromatic_atom: bool,
}

#[derive(Clone)]
/// Implements RequiredElementSet.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-layer note: computes molecular properties for ML pipelines.
struct RequiredElementSet {
    elements: SmallVec<[u8; 8]>,
    count: u16,
}

#[derive(Clone)]
/// Implements ScreenedQuery.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-layer note: computes molecular properties for ML pipelines.
struct ScreenedQuery {
    query: QueryGraph,
    screen: QueryScreen,
}

#[derive(Clone, Copy)]
/// Implements MoleculeScreenStats.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-layer note: computes molecular properties for ML pipelines.
struct MoleculeScreenStats {
    num_atoms: usize,
    num_bonds: usize,
    element_counts: [u16; 119],
    has_ring_atom: bool,
    aromatic_atom_count: u16,
}

impl MoleculeScreenStats {
    /// Constructs molecule between representations.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn from_molecule(mol: &Molecule) -> Self {
        let mut element_counts = [0u16; 119];
        let mut has_ring_atom = false;
        let mut aromatic_atom_count = 0u16;

        for atom in &mol.atoms {
            let idx = atom.atomic_number as usize;
            if idx < element_counts.len() {
                element_counts[idx] = element_counts[idx].saturating_add(1);
            }
            has_ring_atom |= atom.is_in_ring();
            if atom.is_aromatic() {
                aromatic_atom_count = aromatic_atom_count.saturating_add(1);
            }
        }

        Self {
            num_atoms: mol.num_atoms(),
            num_bonds: mol.num_bonds(),
            element_counts,
            has_ring_atom,
            aromatic_atom_count,
        }
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn passes(&self, screen: &QueryScreen) -> bool {
        if screen.min_atoms > self.num_atoms || screen.min_bonds > self.num_bonds {
            return false;
        }
        if screen.requires_ring_atom && !self.has_ring_atom {
            return false;
        }
        if screen.requires_aromatic_atom && self.aromatic_atom_count == 0 {
            return false;
        }
        screen
            .required_elements
            .iter()
            .all(|&(element, count)| self.element_counts[element as usize] >= count)
            && screen.required_element_sets.iter().all(|required| {
                let total = required.elements.iter().fold(0u16, |acc, &element| {
                    acc.saturating_add(self.element_counts[element as usize])
                });
                total >= required.count
            })
    }
}

/// Builds query screen.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn build_query_screen(query: &QueryGraph) -> QueryScreen {
    let mut required = [0u16; 119];
    let mut required_element_sets: SmallVec<[RequiredElementSet; 8]> = SmallVec::new();
    let mut requires_ring_atom = false;
    let mut requires_aromatic_atom = false;

    for atom in &query.atoms {
        if let Some(element) = fixed_atomic_number(atom) {
            let idx = element as usize;
            if idx < required.len() {
                required[idx] = required[idx].saturating_add(1);
            }
        } else if let Some(mut allowed) = query_atom_allowed_elements(atom) {
            allowed.sort_unstable();
            allowed.dedup();
            if let Some(existing) = required_element_sets
                .iter_mut()
                .find(|entry| entry.elements == allowed)
            {
                existing.count = existing.count.saturating_add(1);
            } else {
                required_element_sets.push(RequiredElementSet {
                    elements: allowed,
                    count: 1,
                });
            }
        }
        requires_ring_atom |= query_atom_requires_ring(atom);
        requires_aromatic_atom |= query_atom_requires_aromatic(atom);
    }

    let required_elements = required
        .iter()
        .enumerate()
        .filter_map(|(idx, &count)| (count > 0).then_some((idx as u8, count)))
        .collect();

    QueryScreen {
        min_atoms: query.atoms.len(),
        min_bonds: query.bonds.len(),
        required_elements,
        required_element_sets,
        requires_ring_atom,
        requires_aromatic_atom,
    }
}

/// Implements atomic number.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn fixed_atomic_number_primitive(primitive: &AtomPrimitive) -> Option<u8> {
    match primitive {
        AtomPrimitive::Element(z) | AtomPrimitive::Aromatic(z) | AtomPrimitive::Aliphatic(z) => {
            Some(*z)
        }
        AtomPrimitive::And(items) => {
            let mut fixed = None;
            for item in items {
                let Some(element) = fixed_atomic_number_primitive(item) else {
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
            let mut elements = options.iter().filter_map(fixed_atomic_number_primitive);
            let first = elements.next()?;
            if elements.all(|element| element == first) {
                Some(first)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Implements atom allowed elements.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn query_atom_allowed_elements(query_atom: &QueryAtom) -> Option<SmallVec<[u8; 8]>> {
    let mut allowed: Option<SmallVec<[u8; 8]>> = None;

    for primitive in &query_atom.primitives {
        let Some(set) = allowed_elements_for_primitive(primitive) else {
            continue;
        };

        allowed = Some(match allowed {
            Some(mut prev) => {
                prev.retain(|z| set.contains(z));
                prev
            }
            None => set,
        });
    }

    allowed
}

/// Implements elements for primitive.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn allowed_elements_for_primitive(primitive: &AtomPrimitive) -> Option<SmallVec<[u8; 8]>> {
    match primitive {
        AtomPrimitive::Element(z) | AtomPrimitive::Aromatic(z) | AtomPrimitive::Aliphatic(z) => {
            Some(smallvec::smallvec![*z])
        }
        AtomPrimitive::And(items) => {
            let mut acc: Option<SmallVec<[u8; 8]>> = None;
            for item in items {
                let Some(set) = allowed_elements_for_primitive(item) else {
                    continue;
                };
                acc = Some(match acc {
                    Some(mut prev) => {
                        prev.retain(|z| set.contains(z));
                        prev
                    }
                    None => set,
                });
            }
            acc
        }
        AtomPrimitive::Or(options) => {
            let mut union: SmallVec<[u8; 8]> = SmallVec::new();
            let mut has_any = false;
            for option in options {
                let Some(set) = allowed_elements_for_primitive(option) else {
                    return None;
                };
                has_any = true;
                for element in set {
                    if !union.contains(&element) {
                        union.push(element);
                    }
                }
            }
            has_any.then_some(union)
        }
        _ => None,
    }
}

/// Implements atom requires ring.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn query_atom_requires_ring(query_atom: &QueryAtom) -> bool {
    query_atom
        .primitives
        .iter()
        .any(primitive_requires_ring_atom)
}

/// Implements requires ring atom.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn primitive_requires_ring_atom(primitive: &AtomPrimitive) -> bool {
    match primitive {
        AtomPrimitive::InRing
        | AtomPrimitive::RingSize(_)
        | AtomPrimitive::RingMembership(_)
        | AtomPrimitive::RingBonds(_) => true,
        // Negation does not imply a positive ring requirement for screening.
        AtomPrimitive::Not(_) => false,
        // In conjunctions, any ring-constrained term is sufficient to require
        // ring atoms for a valid match.
        AtomPrimitive::And(options) => options.iter().any(primitive_requires_ring_atom),
        // For OR branches, require ring atoms only if every branch does.
        AtomPrimitive::Or(options) => {
            !options.is_empty() && options.iter().all(primitive_requires_ring_atom)
        }
        // Recursive SMARTS can encode complex alternatives; avoid over-restrictive
        // screening by only considering the recursive root atom.
        AtomPrimitive::Recursive(sub_pattern) => sub_pattern
            .atoms
            .first()
            .is_some_and(query_atom_requires_ring),
        _ => false,
    }
}

/// Implements atom requires aromatic.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn query_atom_requires_aromatic(query_atom: &QueryAtom) -> bool {
    query_atom
        .primitives
        .iter()
        .any(primitive_requires_aromatic_atom)
}

/// Implements requires aromatic atom.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn primitive_requires_aromatic_atom(primitive: &AtomPrimitive) -> bool {
    match primitive {
        AtomPrimitive::AnyAromatic | AtomPrimitive::Aromatic(_) => true,
        // Negation does not enforce aromaticity in a conservative prescreen.
        AtomPrimitive::Not(_) => false,
        // In conjunctions, any aromatic-constrained term is sufficient.
        AtomPrimitive::And(options) => options.iter().any(primitive_requires_aromatic_atom),
        // In OR branches, aromaticity is required only if every branch requires it.
        AtomPrimitive::Or(options) => {
            !options.is_empty() && options.iter().all(primitive_requires_aromatic_atom)
        }
        // Recursive SMARTS can introduce mixed alternatives; evaluate only the
        // recursive root atom to avoid rejecting valid non-aromatic branches.
        AtomPrimitive::Recursive(sub_pattern) => sub_pattern
            .atoms
            .first()
            .is_some_and(query_atom_requires_aromatic),
        _ => false,
    }
}

static QED_ACCEPTOR_PATTERNS: Lazy<Vec<ScreenedQuery>> = Lazy::new(|| {
    QED_ACCEPTOR_SMARTS
        .iter()
        .filter_map(|smarts| parse_smarts(smarts).ok())
        .map(|query| {
            let screen = build_query_screen(&query);
            ScreenedQuery { query, screen }
        })
        .collect()
});

/// Computes qed hbond acceptors.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn compute_qed_hbond_acceptors(mol: &Molecule) -> u32 {
    let stats = MoleculeScreenStats::from_molecule(mol);
    compute_qed_hbond_acceptors_with_stats(mol, &stats)
}

/// Computes qed hbond acceptors with stats.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn compute_qed_hbond_acceptors_with_stats(mol: &Molecule, stats: &MoleculeScreenStats) -> u32 {
    QED_ACCEPTOR_PATTERNS
        .iter()
        .map(|entry| {
            if !stats.passes(&entry.screen) {
                return 0;
            }
            if has_substruct_match(mol, &entry.query) {
                count_matches(mol, &entry.query) as u32
            } else {
                0
            }
        })
        .sum()
}

/// Implements membership from bond cycles.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn ring_membership_from_bond_cycles(
    mol: &Molecule,
    cycles: &[smallvec::SmallVec<[u16; 8]>],
) -> Vec<bool> {
    let mut in_ring = vec![false; mol.num_atoms()];
    for cycle in cycles {
        for &bond_idx in cycle {
            let bond = &mol.bonds[bond_idx as usize];
            in_ring[bond.source as usize] = true;
            in_ring[bond.target as usize] = true;
        }
    }
    in_ring
}

/// Builds qed aromatic subgraph.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn build_qed_aromatic_subgraph(mol: &Molecule, in_ring: &[bool]) -> Option<Molecule> {
    let mut atom_map = vec![u16::MAX; mol.num_atoms()];
    let mut atoms = Vec::with_capacity(mol.num_atoms());
    let mut adjacency = Vec::with_capacity(mol.num_atoms());

    for (atom_idx, atom) in mol.atoms.iter().enumerate() {
        let keep = !(in_ring[atom_idx] && !atom.is_aromatic());
        if !keep {
            continue;
        }
        atom_map[atom_idx] = atoms.len() as u16;
        atoms.push(*atom);
        adjacency.push(SmallVec::new());
    }

    if atoms.is_empty() {
        return None;
    }

    let mut bonds = Vec::with_capacity(mol.num_bonds());
    for bond in &mol.bonds {
        let src = atom_map[bond.source as usize];
        let dst = atom_map[bond.target as usize];
        if src == u16::MAX || dst == u16::MAX {
            continue;
        }
        let mut new_bond = *bond;
        new_bond.source = src;
        new_bond.target = dst;
        let bond_idx = bonds.len() as u16;
        bonds.push(new_bond);
        adjacency[src as usize].push(bond_idx);
        adjacency[dst as usize].push(bond_idx);
    }

    Some(Molecule::from_parts(atoms, bonds, adjacency))
}

/// Computes qed aromatic ring count with cycles.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn compute_qed_aromatic_ring_count_with_cycles(
    mol: &Molecule,
    cycles: &[smallvec::SmallVec<[u16; 8]>],
) -> u32 {
    let in_ring = ring_membership_from_bond_cycles(mol, cycles);
    let Some(mut subgraph) = build_qed_aromatic_subgraph(mol, &in_ring) else {
        return 0;
    };
    find_sssr(&mut subgraph).len() as u32
}

/// Computes qed aromatic ring count.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn compute_qed_aromatic_ring_count(mol: &Molecule) -> u32 {
    let cycles = find_minimum_cycle_basis_bonds(mol);
    compute_qed_aromatic_ring_count_with_cycles(mol, &cycles)
}

/// Computes qed aromatic ring count cached.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn compute_qed_aromatic_ring_count_cached(mol: &mut Molecule) -> u32 {
    ensure_minimum_cycle_basis_bonds_cached(mol);
    let cycles = mol
        .cached_min_cycle_basis_bonds()
        .expect("minimum cycle basis cache initialized");
    compute_qed_aromatic_ring_count_with_cycles(mol, cycles)
}

/// Compute raw QED score from a molecule.
fn compute_qed(mol: &Molecule) -> f64 {
    let stats = MoleculeScreenStats::from_molecule(mol);
    let mw = compute_molecular_weight(mol) as f64;
    let logp = compute_logp(mol).0 as f64;
    let hba = compute_qed_hbond_acceptors_with_stats(mol, &stats) as f64;
    let hbd = compute_hbond_donors(mol) as f64;
    let psa = compute_tpsa(mol, false) as f64;
    let rotb = compute_rotatable_bond_count(mol) as f64;
    let arom = compute_qed_aromatic_ring_count(mol) as f64;
    let alerts = compute_structural_alerts_with_stats(mol, &stats) as f64;

    let values = [mw, logp, hba, hbd, psa, rotb, arom, alerts];

    let mut sum_weights = 0.0;
    let mut sum_log = 0.0;

    for (i, param) in QED_PARAMS.iter().enumerate() {
        let d = ads(values[i], &param.ads).max(1e-10);
        sum_log += param.weight * d.ln();
        sum_weights += param.weight;
    }

    if sum_weights > 0.0 {
        (sum_log / sum_weights).exp().clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Computes qed cached.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn compute_qed_cached(mol: &mut Molecule) -> f64 {
    let stats = MoleculeScreenStats::from_molecule(mol);
    let mw = compute_molecular_weight(mol) as f64;
    let logp = compute_logp_cached(mol).0 as f64;
    let hba = compute_qed_hbond_acceptors_with_stats(mol, &stats) as f64;
    let hbd = compute_hbond_donors_cached(mol) as f64;
    let psa = compute_tpsa_cached(mol, false) as f64;
    let rotb = compute_rotatable_bond_count_cached(mol) as f64;
    let arom = compute_qed_aromatic_ring_count_cached(mol) as f64;
    let alerts = compute_structural_alerts_with_stats(mol, &stats) as f64;

    let values = [mw, logp, hba, hbd, psa, rotb, arom, alerts];

    let mut sum_weights = 0.0;
    let mut sum_log = 0.0;

    for (i, param) in QED_PARAMS.iter().enumerate() {
        let d = ads(values[i], &param.ads).max(1e-10);
        sum_log += param.weight * d.ln();
        sum_weights += param.weight;
    }

    if sum_weights > 0.0 {
        (sum_log / sum_weights).exp().clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// RDKit QED structural alert SMARTS patterns.
/// Each entry is a list of SMARTS; if ANY matches, that alert fires (count += 1).
/// From Bickerton et al. (2012), Table S1.
static STRUCTURAL_ALERT_SMARTS: &[&[&str]] = &[
    &[
        "*1[O,S,N]*1",
        "[S,C](=[O,S])[F,Br,Cl,I]",
        "[CX4][Cl,Br,I]",
        "[#6]S(=O)(=O)O[#6]",
    ],
    &[
        "[$([CH]),$(CC)]#CC(=O)[#6]",
        "[$([CH]),$(CC)]#CC(=O)O[#6]",
        "n[OH]",
    ],
    &[
        "[$([CH]),$(CC)]#CS(=O)(=O)[#6]",
        "C=C(C=O)C=O",
        "n1c([F,Cl,Br,I])cccc1",
        "[CH1](=O)",
        "[#8][#8]",
    ],
    &[
        "[C;!R]=[N;!R]",
        "[N!R]=[N!R]",
        "[#6](=O)[#6](=O)",
        "[#16][#16]",
        "[#7][NH2]",
        "C(=O)N[NH2]",
    ],
    &[
        "[#6]=S",
        "[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]",
    ],
    &[
        "C1(=[O,N])C=CC(=[O,N])C=C1",
        "C1(=[O,N])C(=[O,N])C=CC=C1",
        "a21aa3a(aa1aaaa2)aaaa3",
    ],
    &[
        "a31a(a2a(aa1)aaaa2)aaaa3",
        "a1aa2a3a(a1)A=AA=A3=AA=A2",
        "c1cc([NH2])ccc1",
    ],
    &[
        "[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]",
    ],
    &["I", "OS(=O)(=O)[O-]", "[N+](=O)[O-]", "C(=O)N[OH]"],
    &[
        "C1NC(=O)NC(=O)1",
        "[SH]",
        "[S-]",
        "c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]",
    ],
    &[
        "c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]",
        "[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1",
    ],
    &[
        "[CR1]1[CR1][CR1]cc[CR1][CR1]1",
        "[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1",
    ],
    &[
        "[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1",
        "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
    ],
    &[
        "[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1",
        "C#C",
    ],
    &[
        "[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]",
        "[$([N+R]),$([n+R]),$([N+]=C)][O-]",
    ],
    &[
        "[#6]=N[OH]",
        "[#6]=NOC=O",
        "[#6](=O)[CX4,CR0X3,O][#6](=O)",
        "c1ccc2c(c1)ccc(=O)o2",
    ],
    &[
        "[O+,o+,S+,s+]",
        "N=C=O",
        "[NX3,NX4][F,Cl,Br,I]",
        "c1ccccc1OC(=O)[#6]",
        "[CR0]=[CR0][CR0]=[CR0]",
    ],
    &[
        "[C+,c+,C-,c-]",
        "N=[N+]=[N-]",
        "C12C(NC(N1)=O)CSC2",
        "c1c([OH])c([OH,NH2,NH])ccc1",
        "P",
    ],
    &[
        "[N,O,S]C#N",
        "C=C=O",
        "[Si][F,Cl,Br,I]",
        "[SX2]O",
        "[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)",
    ],
    &["O1CCCCC1OC2CCC3CCCCC3C2", "N=[CR0][N,n,O,S]"],
    &[
        "[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2",
    ],
    &[
        "C=[C!r]C#N",
        "[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1",
    ],
    &["[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1"],
    &[
        "[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])",
        "[OH]c1ccc([OH,NH2,NH])cc1",
    ],
    &[
        "c1ccccc1OC(=O)O",
        "[SX2H0][N]",
        "c12ccccc1(SC(S)=N2)",
        "c12ccccc1(SC(=S)N2)",
        "c1nnnn1C=O",
    ],
    &[
        "s1c(S)nnc1NC=O",
        "S1C=CSC1=S",
        "C(=O)Onnn",
        "OS(=O)(=O)C(F)(F)F",
        "N#CC[OH]",
        "N#CC(=O)",
    ],
    &[
        "S(=O)(=O)C#N",
        "N[CH2]C#N",
        "C1(=O)NCC1",
        "S(=O)(=O)[O-,OH]",
        "NC[F,Cl,Br,I]",
        "C=[C!r]O",
    ],
    &[
        "[NX2+0]=[O+0]",
        "[OR0,NR0][OR0,NR0]",
        "C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]",
        "[CX2R0][NX3R0]",
    ],
    &[
        "c1ccccc1[C;!R]=[C;!R]c2ccccc2",
        "[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]",
    ],
    &["[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]"],
    &[
        "[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]",
        "[*]=[N+]=[*]",
        "[SX3](=O)[O-,OH]",
    ],
    &[
        "N#N",
        "F.F.F.F",
        "[R0;D2][R0;D2][R0;D2][R0;D2]",
        "[cR,CR]~C(=O)NC(=O)~[cR,CR]",
        "C=!@CC=[O,S]",
    ],
    &[
        "[#6,#8,#16][#6](=O)O[#6]",
        "c[C;R0](=[O,S])[#6]",
        "c[SX2][C;!R]",
        "C=C=C",
    ],
    &[
        "c1nc([F,Cl,Br,I,S])ncc1",
        "c1ncnc([F,Cl,Br,I,S])c1",
        "c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])",
    ],
    &[
        "[#6]S(=O)(=O)c1ccc(cc1)F",
        "[15N]",
        "[13C]",
        "[18O]",
        "[34S]",
    ],
];

/// Pre-parsed structural alert SMARTS.
static STRUCTURAL_ALERT_PATTERNS: Lazy<Vec<ScreenedQuery>> = Lazy::new(|| {
    STRUCTURAL_ALERT_SMARTS
        .iter()
        .flat_map(|group| group.iter())
        .filter_map(|smarts| parse_smarts(smarts).ok())
        .map(|query| {
            let screen = build_query_screen(&query);
            ScreenedQuery { query, screen }
        })
        .collect()
});

/// Count the number of structural alert patterns that match.
fn compute_structural_alerts(mol: &Molecule) -> u32 {
    let stats = MoleculeScreenStats::from_molecule(mol);
    compute_structural_alerts_with_stats(mol, &stats)
}

/// Computes structural alerts with stats.
///
/// Function context: `crates/ferrum_core/src/desc/qed.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn compute_structural_alerts_with_stats(mol: &Molecule, stats: &MoleculeScreenStats) -> u32 {
    STRUCTURAL_ALERT_PATTERNS
        .iter()
        .filter(|entry| stats.passes(&entry.screen) && has_substruct_match(mol, &entry.query))
        .count() as u32
}

/// QED (Quantitative Estimate of Drug-likeness).
///
/// Computes a weighted geometric mean of desirability functions for 7 molecular properties.
/// Score ranges from 0 (least drug-like) to ~1 (most drug-like).
///
/// Properties used: MW, LogP, HBA, HBD, TPSA, RotatableBonds, AromaticRings, Alerts.
///
/// This follows RDKit's QED implementation, including its dedicated acceptor
/// SMARTS set, aromatic ring counting method, and structural alerts.
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct Qed;

impl Descriptor for Qed {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "QED"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Float(compute_qed_cached(mol) as f32)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Qed>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements qed.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_qed(smiles: &str) -> f32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        Qed.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        )
        .as_float()
    }

    #[test]
    /// Runs regression test for qed range.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_range() {
        let qed = calc_qed("c1ccccc1");
        assert!(qed >= 0.0 && qed <= 1.0, "QED should be 0-1, got {}", qed);
    }

    #[test]
    /// Runs regression test for qed aspirin.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_aspirin() {
        let qed = calc_qed("CC(=O)Oc1ccccc1C(=O)O");
        assert!(qed > 0.2 && qed < 0.9, "Aspirin QED={}", qed);
    }

    #[test]
    /// Runs regression test for qed caffeine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_caffeine() {
        let qed = calc_qed("CN1C=NC2=C1C(=O)N(C(=O)N2C)C");
        assert!(qed > 0.1 && qed < 0.8, "Caffeine QED={}", qed);
    }

    #[test]
    /// Runs regression test for qed drug like.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_drug_like() {
        let qed = calc_qed("CC(C)Cc1ccc(cc1)C(C)C(=O)O");
        assert!(qed > 0.2 && qed < 0.9);
    }

    #[test]
    /// Runs regression test for qed methane low.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_methane_low() {
        let qed = calc_qed("C");
        assert!(qed < 0.5, "Methane should have low QED: {}", qed);
    }

    #[test]
    /// Runs regression test for qed large molecule lower.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_large_molecule_lower() {
        let qed_small = calc_qed("c1ccc(CC(=O)O)cc1");
        let qed_large = calc_qed("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC");
        assert!(qed_large < qed_small || qed_large < 0.5);
    }

    #[test]
    /// Runs regression test for qed aromatic ring count matches rdkit style.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_aromatic_ring_count_matches_rdkit_style() {
        let benzene = parse_smiles("c1ccccc1").unwrap();
        assert_eq!(compute_qed_aromatic_ring_count(&benzene), 1);

        let cyclohexane = parse_smiles("C1CCCCC1").unwrap();
        assert_eq!(compute_qed_aromatic_ring_count(&cyclohexane), 0);
    }

    #[test]
    /// Runs regression test for qed matches rdkit cnp0298191.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_matches_rdkit_cnp0298191() {
        let mol =
            parse_smiles("CCOCC1=C2/C(=C\\[C@@]3(C)CC[C@](O)(O3)[C@@](C)(O)C[C@H]2OC(C)=O)OC1=O")
                .unwrap();
        assert_eq!(compute_qed_hbond_acceptors(&mol), 8);
        assert_eq!(compute_qed_aromatic_ring_count(&mol), 0);
        assert_eq!(compute_structural_alerts(&mol), 1);
        assert!((compute_qed(&mol) - 0.6956330228610187).abs() < 1e-3);
    }

    #[test]
    /// Runs regression test for qed matches rdkit cnp0554779.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_matches_rdkit_cnp0554779() {
        let mol = parse_smiles("CC12C=CC(=O)C=C1CCC1C2CCC2(C)C(OC(=O)CCC3CCCC3)CCC12").unwrap();
        assert_eq!(compute_structural_alerts(&mol), 1);
        assert!((compute_qed(&mol) - 0.5200628922737999).abs() < 1e-3);
    }

    #[test]
    /// Runs regression test for qed structural alerts match rdkit cluster cases.
    ///
    /// Function context: `crates/ferrum_core/src/desc/qed.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_qed_structural_alerts_match_rdkit_cluster_cases() {
        // These molecules were scalar QED parity outliers caused by alert
        // undercounting in pre-screened SMARTS matching.
        let cnp0454917 = parse_smiles("CCCCCC1=CC(O)=C(CCCC)C(OS(=O)(=O)O)=C1").unwrap();
        assert_eq!(compute_structural_alerts(&cnp0454917), 2);

        let cnp0479681 = parse_smiles("NC(N)=NCCC[C@H](N)C(=O)O").unwrap();
        assert_eq!(compute_structural_alerts(&cnp0479681), 3);

        let cnp0538427 = parse_smiles("CCCNC(=N)NCCC[C@H](N)C(=O)O").unwrap();
        assert_eq!(compute_structural_alerts(&cnp0538427), 3);

        let cnp0334248 =
            parse_smiles("O=C1CCC[C@@H]2[C@H]3CCC[N@@+]4([O-])CCC[C@@H](CN12)[C@@H]34").unwrap();
        assert_eq!(compute_structural_alerts(&cnp0334248), 2);
    }
}
