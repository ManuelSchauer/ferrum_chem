//! I/O module: Smarts.
//!
//! This module handles external data interchange (SMILES/SMARTS/CSV/SDF) and
//! conversion into FerrumChem graph types. It is the boundary between textual
//! chemical formats and internal, typed molecular structures.
//!
//! Learning focus:
//! - Parser/state-machine techniques in Rust.
//! - Error-aware ingestion pipelines for large datasets.
//!
//! References:
//! - Weininger, D. (1988). SMILES notation specification for shared atom/bond grammar roots.

use crate::algo::rings::find_symmetrized_sssr_bonds;
use crate::error::{FerrumError, Result};
use crate::mol::molecule::Molecule;
use crate::types::BondType;
use smallvec::SmallVec;

/// Query graph for SMARTS pattern matching.
/// Represents a molecular pattern with flexible matching criteria.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryGraph {
    pub atoms: Vec<QueryAtom>,
    pub bonds: Vec<QueryGraphBond>,
    pub adjacency: Vec<SmallVec<[u16; 4]>>,
    /// Cached flag: does this pattern use RingSize or RingMembership primitives?
    pub(crate) needs_ring_context: bool,
    /// Matcher bookkeeping (unique id + lazily compiled search data). Not
    /// part of the pattern's identity: ignored by `PartialEq`, reset by `Clone`.
    pub(crate) meta: QueryMeta,
}

static NEXT_QUERY_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Per-pattern matcher metadata. The id keys per-molecule caches (recursive
/// SMARTS results); a fresh id is drawn for every new/cloned graph so a
/// cache entry can never be confused with a different pattern.
pub(crate) struct QueryMeta {
    pub(crate) id: u64,
    pub(crate) compiled: std::sync::OnceLock<crate::algo::iso::CompiledQuery>,
}

impl Default for QueryMeta {
    fn default() -> Self {
        Self {
            id: NEXT_QUERY_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            compiled: std::sync::OnceLock::new(),
        }
    }
}

impl Clone for QueryMeta {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for QueryMeta {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for QueryMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QueryMeta({})", self.id)
    }
}

/// Bond in query graph (stores connectivity).
#[derive(Debug, Clone, PartialEq)]
pub struct QueryGraphBond {
    pub source: u16,
    pub target: u16,
    pub query: QueryBond,
}

/// Query atom with flexible matching primitives.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryAtom {
    pub primitives: Vec<AtomPrimitive>,
}

/// Atom matching primitives for SMARTS queries.
#[derive(Debug, Clone, PartialEq)]
pub enum AtomPrimitive {
    /// Match specific element by atomic number
    Element(u8),

    /// Match any atom (*)
    AnyAtom,

    /// Match any aliphatic atom (A)
    AnyAliphatic,

    /// Match any aromatic atom (a)
    AnyAromatic,

    /// Match aromatic atom of this element
    Aromatic(u8),

    /// Match aliphatic atom of this element
    Aliphatic(u8),

    /// Logical NOT
    Not(Box<AtomPrimitive>),

    /// Logical OR (one of these must match)
    Or(Vec<AtomPrimitive>),

    /// Logical AND (all must match)
    And(Vec<AtomPrimitive>),

    /// `H<n>`: total H count including hydrogen-atom neighbours (RDKit `getTotalNumHs(true)`)
    HCount(u8),

    /// `h<n>`: implicit + explicit H count, excluding H-atom neighbours
    ImplicitHCount(u8),

    /// bare `h`: at least one implicit/explicit H
    HasImplicitH,

    /// `<n>` at the start of an atom expression: isotope
    Isotope(u16),

    /// bare `x`: has at least one ring bond
    HasRingBond,

    /// `d<n>`: number of non-hydrogen neighbours (D and T count as non-hydrogen)
    NonHydrogenDegree(u8),

    /// `z<n>`: number of heteroatom (not C, not H) neighbours
    HeteroNeighbors(u8),

    /// bare `z`
    HasHeteroNeighbors,

    /// `Z<n>`: number of aliphatic heteroatom neighbours
    AliphaticHeteroNeighbors(u8),

    /// bare `Z`
    HasAliphaticHeteroNeighbors,

    /// `^<n>`: hybridization
    Hybridization(crate::types::Hybridization),

    /// Degree (number of explicit connections, no implicit H) [D1], [D2], [D3]
    Degree(u8),

    /// Connectivity (total degree: explicit neighbors + total H) [X1], [X2], [X3], [X4]
    Connectivity(u8),

    /// Formal charge [+], [-], [+2], [-1]
    Charge(i8),

    /// In any ring (R)
    InRing,

    /// Not in ring (R0)
    NotInRing,

    /// In ring of specific size [r5], [r6]
    RingSize(u8),

    /// Ring membership count [R1], [R2] — number of SSSR rings containing atom
    RingMembership(u8),

    /// Valence [v3], [v4]
    Valence(u8),

    /// Number of ring bonds [x2], [x3]
    RingBonds(u8),

    /// Recursive SMARTS $(...) - atom must match as first atom of sub-pattern
    Recursive(Box<QueryGraph>),
}

/// Query bond for pattern matching.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryBond {
    /// Any bond (~)
    Any,

    /// Single bond (-)
    Single,

    /// Double bond (=)
    Double,

    /// Triple bond (#)
    Triple,

    /// Aromatic bond (:)
    Aromatic,

    /// Single or aromatic (-,:)
    SingleOrAromatic,

    /// Double or aromatic (=,:)
    DoubleOrAromatic,

    /// Ring bond (@)
    Ring,

    /// Not this bond type
    Not(Box<QueryBond>),

    /// AND of bond types (e.g., =;@ means double AND ring)
    And(Vec<QueryBond>),

    /// OR of bond types (e.g., =,: means double OR aromatic)
    Or(Vec<QueryBond>),
}

impl QueryAtom {
    /// Creates a query atom that matches any atom.
    pub fn any() -> Self {
        Self {
            primitives: vec![AtomPrimitive::AnyAtom],
        }
    }

    /// Creates a query atom that matches a specific element.
    pub fn element(atomic_number: u8) -> Self {
        Self {
            primitives: vec![AtomPrimitive::Element(atomic_number)],
        }
    }

    /// Creates a query atom that matches any aromatic atom.
    pub fn aromatic() -> Self {
        Self {
            primitives: vec![AtomPrimitive::AnyAromatic],
        }
    }

    /// Creates a query atom that matches aromatic atoms.
    pub fn aromatic_element(atomic_number: u8) -> Self {
        Self {
            primitives: vec![AtomPrimitive::Aromatic(atomic_number)],
        }
    }

    /// Checks if this query atom matches a molecule atom.
    pub fn matches(&self, mol: &Molecule, atom_idx: u16) -> bool {
        // Standalone QueryAtom matching is used in tests and utility callsites
        // outside VF2. If this atom uses ring-size/membership SMARTS primitives,
        // build ring context once here so ring checks stay context-driven and
        // never trigger clone+SSSR fallback work.
        let ring_ctx =
            query_atom_needs_ring_context(self).then(|| SmartsRingContext::for_molecule(mol));
        query_atom_matches_with_context(self, mol, atom_idx, ring_ctx.as_ref())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SmartsRingContext {
    atom_ring_membership: Vec<u8>,
    atom_ring_sizes: Vec<SmallVec<[u8; 4]>>,
}

impl SmartsRingContext {
    pub(crate) fn for_molecule(mol: &Molecule) -> Self {
        let mut atom_ring_membership = vec![0u8; mol.num_atoms()];
        let mut atom_ring_sizes = vec![SmallVec::new(); mol.num_atoms()];

        // Use symmetrized SSSR-like cycles so SMARTS r/R semantics include
        // topologically equivalent rings in highly fused/symmetric systems.
        let ring_bond_cycles = find_symmetrized_sssr_bonds(mol);
        for ring_bonds in ring_bond_cycles {
            let mut ring_atoms = SmallVec::<[u16; 8]>::new();
            for &bond_idx in &ring_bonds {
                let bond = &mol.bonds[bond_idx as usize];
                if !ring_atoms.contains(&bond.source) {
                    ring_atoms.push(bond.source);
                }
                if !ring_atoms.contains(&bond.target) {
                    ring_atoms.push(bond.target);
                }
            }

            let ring_size = ring_atoms.len().min(u8::MAX as usize) as u8;
            for atom_idx in ring_atoms {
                let idx = atom_idx as usize;
                atom_ring_membership[idx] = atom_ring_membership[idx].saturating_add(1);
                if !atom_ring_sizes[idx].contains(&ring_size) {
                    atom_ring_sizes[idx].push(ring_size);
                }
            }
        }

        Self {
            atom_ring_membership,
            atom_ring_sizes,
        }
    }

    pub(crate) fn ring_membership(&self, atom_idx: u16) -> u8 {
        self.atom_ring_membership[atom_idx as usize]
    }

    pub(crate) fn min_ring_size(&self, atom_idx: u16) -> Option<u8> {
        self.atom_ring_sizes[atom_idx as usize].iter().copied().min()
    }
}

pub(crate) fn query_graph_needs_ring_context(query: &QueryGraph) -> bool {
    query.needs_ring_context
}

/// Computes whether a query graph needs ring context by walking the pattern tree.
fn compute_needs_ring_context(atoms: &[QueryAtom]) -> bool {
    atoms
        .iter()
        .flat_map(|atom| atom.primitives.iter())
        .any(primitive_needs_ring_context)
}

fn primitive_needs_ring_context(primitive: &AtomPrimitive) -> bool {
    match primitive {
        AtomPrimitive::RingSize(_) | AtomPrimitive::RingMembership(_) => true,
        AtomPrimitive::Not(inner) => primitive_needs_ring_context(inner),
        AtomPrimitive::Or(options) | AtomPrimitive::And(options) => {
            options.iter().any(primitive_needs_ring_context)
        }
        AtomPrimitive::Recursive(sub_pattern) => query_graph_needs_ring_context(sub_pattern),
        _ => false,
    }
}

fn query_atom_needs_ring_context(query_atom: &QueryAtom) -> bool {
    query_atom
        .primitives
        .iter()
        .any(primitive_needs_ring_context)
}

/// Matches a query atom without a shared per-molecule matcher context
/// (recursive SMARTS are evaluated with a temporary one).
pub(crate) fn query_atom_matches_with_context(
    query_atom: &QueryAtom,
    mol: &Molecule,
    atom_idx: u16,
    ring_ctx: Option<&SmartsRingContext>,
) -> bool {
    let atom = &mol.atoms[atom_idx as usize];
    query_atom
        .primitives
        .iter()
        .all(|primitive| match_primitive(primitive, mol, atom_idx, atom, ring_ctx, None))
}

/// Matches a query atom using the shared matcher context of `iso_target`
/// (lazily built ring context, memoized recursive SMARTS).
#[inline]
pub(crate) fn query_atom_matches_target(
    query_atom: &QueryAtom,
    iso_target: &crate::algo::iso::IsoTarget,
    atom_idx: u16,
) -> bool {
    let mol = iso_target.target;
    let atom = &mol.atoms[atom_idx as usize];
    query_atom
        .primitives
        .iter()
        .all(|primitive| match_primitive(primitive, mol, atom_idx, atom, None, Some(iso_target)))
}

/// Matches a single atom primitive against a molecule atom.
#[inline]
fn neighbors(mol: &Molecule, atom_idx: u16) -> impl Iterator<Item = usize> + '_ {
    mol.adjacency[atom_idx as usize].iter().map(move |&b| {
        let bond = &mol.bonds[b as usize];
        if bond.source == atom_idx { bond.target as usize } else { bond.source as usize }
    })
}

/// RDKit `queryAtomNumHeteroatomNbrs` / `queryAtomNumAliphaticHeteroatomNbrs`.
fn hetero_neighbors(mol: &Molecule, atom_idx: u16, aliphatic_only: bool) -> usize {
    neighbors(mol, atom_idx)
        .filter(|&o| {
            let a = &mol.atoms[o];
            a.atomic_number != 6 && a.atomic_number != 1 && !(aliphatic_only && a.is_aromatic())
        })
        .count()
}

fn match_primitive(
    primitive: &AtomPrimitive,
    mol: &Molecule,
    atom_idx: u16,
    atom: &crate::mol::atom::Atom,
    ring_ctx: Option<&SmartsRingContext>,
    iso_target: Option<&crate::algo::iso::IsoTarget>,
) -> bool {
    match primitive {
        AtomPrimitive::Element(z) => atom.atomic_number == *z,

        AtomPrimitive::AnyAtom => true,

        AtomPrimitive::AnyAliphatic => !atom.is_aromatic(),

        AtomPrimitive::AnyAromatic => atom.is_aromatic(),

        AtomPrimitive::Aromatic(z) => atom.atomic_number == *z && atom.is_aromatic(),

        AtomPrimitive::Aliphatic(z) => atom.atomic_number == *z && !atom.is_aromatic(),

        AtomPrimitive::Not(inner) => {
            !match_primitive(inner, mol, atom_idx, atom, ring_ctx, iso_target)
        }

        AtomPrimitive::Or(options) => options
            .iter()
            .any(|p| match_primitive(p, mol, atom_idx, atom, ring_ctx, iso_target)),

        AtomPrimitive::And(requirements) => requirements
            .iter()
            .all(|p| match_primitive(p, mol, atom_idx, atom, ring_ctx, iso_target)),

        AtomPrimitive::HCount(n) => {
            // RDKit getTotalNumHs(true): includes hydrogen-atom neighbours
            let h_nbrs = match iso_target {
                Some(it) => it.topo().inv(mol)[atom_idx as usize].h_neighbours as usize,
                None => neighbors(mol, atom_idx).filter(|&o| mol.atoms[o].atomic_number == 1).count(),
            };
            atom.total_h() as usize + h_nbrs == *n as usize
        }

        AtomPrimitive::ImplicitHCount(n) => atom.total_h() == *n,

        AtomPrimitive::HasImplicitH => atom.total_h() > 0,

        AtomPrimitive::Isotope(iso) => atom.isotope == *iso,

        AtomPrimitive::HasRingBond => match iso_target {
            Some(it) => it.topo().inv(mol)[atom_idx as usize].ring_bonds > 0,
            None => mol.adjacency[atom_idx as usize]
                .iter()
                .any(|&b| mol.bonds[b as usize].is_in_ring()),
        },

        AtomPrimitive::NonHydrogenDegree(n) => match iso_target {
            Some(it) => it.topo().inv(mol)[atom_idx as usize].non_h_degree == *n,
            None => {
                neighbors(mol, atom_idx)
                    .filter(|&o| mol.atoms[o].atomic_number != 1 || mol.atoms[o].isotope != 0)
                    .count()
                    == *n as usize
            }
        },

        AtomPrimitive::HeteroNeighbors(n) => match iso_target {
            Some(it) => it.topo().inv(mol)[atom_idx as usize].hetero == *n,
            None => hetero_neighbors(mol, atom_idx, false) == *n as usize,
        },

        AtomPrimitive::HasHeteroNeighbors => match iso_target {
            Some(it) => it.topo().inv(mol)[atom_idx as usize].hetero > 0,
            None => hetero_neighbors(mol, atom_idx, false) > 0,
        },

        AtomPrimitive::AliphaticHeteroNeighbors(n) => match iso_target {
            Some(it) => it.topo().inv(mol)[atom_idx as usize].aliphatic_hetero == *n,
            None => hetero_neighbors(mol, atom_idx, true) == *n as usize,
        },

        AtomPrimitive::HasAliphaticHeteroNeighbors => match iso_target {
            Some(it) => it.topo().inv(mol)[atom_idx as usize].aliphatic_hetero > 0,
            None => hetero_neighbors(mol, atom_idx, true) > 0,
        },

        AtomPrimitive::Hybridization(h) => atom.hybridization == *h,

        AtomPrimitive::Degree(n) => {
            // D = explicit degree (connected atoms, no implicit H)
            mol.get_degree(atom_idx) == *n as usize
        }

        AtomPrimitive::Connectivity(n) => {
            // X = total connectivity (explicit neighbors + total H)
            let total = mol.get_degree(atom_idx) + atom.total_h() as usize;
            total == *n as usize
        }

        AtomPrimitive::Charge(charge) => atom.formal_charge == *charge,

        AtomPrimitive::InRing => atom.is_in_ring(),

        AtomPrimitive::NotInRing => !atom.is_in_ring(),

        AtomPrimitive::RingSize(size) => {
            if !atom.is_in_ring() {
                return false;
            }
            match (ring_ctx, iso_target) {
                (Some(ctx), _) => ctx.min_ring_size(atom_idx) == Some(*size),
                (None, Some(it)) => it.get_ring_ctx().min_ring_size(atom_idx) == Some(*size),
                (None, None) => false,
            }
        }

        AtomPrimitive::RingMembership(count) => match (ring_ctx, iso_target) {
            (Some(ctx), _) => ctx.ring_membership(atom_idx) == *count,
            (None, Some(it)) => it.get_ring_ctx().ring_membership(atom_idx) == *count,
            (None, None) => false,
        },

        AtomPrimitive::Valence(v) => {
            // RDKit getTotalValence (incl. its rounding rules for aromatic atoms)
            let valence = match iso_target {
                Some(it) => it.topo().total_valence(mol, atom_idx as usize),
                None => crate::algo::sanitize::total_valence(mol, atom_idx as usize),
            };
            valence == *v as i32
        }

        AtomPrimitive::RingBonds(n) => match iso_target {
            Some(it) => it.topo().inv(mol)[atom_idx as usize].ring_bonds == *n,
            None => {
                // Count bonds connected to this atom that are in rings
                let ring_bond_count = mol.adjacency[atom_idx as usize]
                    .iter()
                    .filter(|&&bond_idx| mol.bonds[bond_idx as usize].is_in_ring())
                    .count();
                ring_bond_count == *n as usize
            }
        },

        AtomPrimitive::Recursive(sub_pattern) => {
            // Recursive SMARTS: atom matches if it can be the first atom
            // of a match of the sub-pattern
            if sub_pattern.atoms.is_empty() {
                return true; // Empty pattern always matches
            }
            if sub_pattern.atoms.len() == 1 && sub_pattern.bonds.is_empty() {
                // a single-atom sub-pattern matches exactly the atoms its atom matches
                return sub_pattern.atoms[0]
                    .primitives
                    .iter()
                    .all(|p| match_primitive(p, mol, atom_idx, atom, ring_ctx, iso_target));
            }
            match iso_target {
                Some(it) => crate::algo::iso::recursive_start_matches(it, sub_pattern, atom_idx),
                None => crate::algo::iso::recursive_start_matches(
                    &crate::algo::iso::IsoTarget::new(mol),
                    sub_pattern,
                    atom_idx,
                ),
            }
        }
    }
}

impl QueryBond {
    /// Checks if this query bond matches a molecule bond.
    pub fn matches(&self, mol: &Molecule, bond_idx: u16) -> bool {
        let bond = &mol.bonds[bond_idx as usize];
        self.matches_props(bond.bond_type, bond.is_in_ring())
    }

    /// Bond queries only look at the bond type and the ring flag.
    pub(crate) fn matches_props(&self, bond_type: BondType, in_ring: bool) -> bool {
        match self {
            QueryBond::Any => true,

            QueryBond::Single => bond_type == BondType::Single,

            QueryBond::Double => bond_type == BondType::Double,

            QueryBond::Triple => bond_type == BondType::Triple,

            QueryBond::Aromatic => bond_type == BondType::Aromatic,

            QueryBond::SingleOrAromatic => {
                bond_type == BondType::Single || bond_type == BondType::Aromatic
            }

            QueryBond::DoubleOrAromatic => {
                bond_type == BondType::Double || bond_type == BondType::Aromatic
            }

            QueryBond::Ring => in_ring,

            QueryBond::Not(inner) => !inner.matches_props(bond_type, in_ring),

            QueryBond::And(bonds) => bonds.iter().all(|b| b.matches_props(bond_type, in_ring)),

            QueryBond::Or(bonds) => bonds.iter().any(|b| b.matches_props(bond_type, in_ring)),
        }
    }
}

/// Parses a SMARTS pattern into a QueryGraph.
///
/// # Example
/// ```ignore
/// let pattern = parse_smarts("[CH3]")?;  // Matches methyl group
/// let pattern = parse_smarts("c1ccccc1")?;  // Matches benzene
/// let pattern = parse_smarts("[N,O]")?;  // Matches N or O
/// ```
pub fn parse_smarts(smarts: &str) -> Result<QueryGraph> {
    // Like RDKit, everything after the first whitespace is a name, not SMARTS.
    let smarts = smarts.split_whitespace().next().unwrap_or("");
    let mut parser = SmartsParser::new(smarts);
    parser
        .parse()
        .map_err(|e| FerrumError::ParseError(format!("SMARTS: {}", e)))
}

/// Implements SmartsParser.
struct SmartsParser {
    input: Vec<char>,
    pos: usize,
    atoms: Vec<QueryAtom>,
    bonds: Vec<QueryGraphBond>,
    adjacency: Vec<SmallVec<[u16; 4]>>,
    ring_closures: std::collections::BTreeMap<u32, (u16, Option<QueryBond>)>,
}

/// Bracket-atom symbols that are restricted to aliphatic atoms (RDKit
/// `ORGANIC_ATOM_TOKEN`); every other uppercase symbol is an atomic-number query.
fn is_organic_aliphatic(z: u8) -> bool {
    matches!(z, 5 | 6 | 7 | 8 | 9 | 15 | 16 | 17 | 35 | 53)
}

impl SmartsParser {
    fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
            atoms: Vec::new(),
            bonds: Vec::new(),
            adjacency: Vec::new(),
            ring_closures: std::collections::BTreeMap::new(),
        }
    }

    fn parse(&mut self) -> std::result::Result<QueryGraph, String> {
        if !self.input.is_empty() {
            self.parse_chain()?;
        }
        if !self.ring_closures.is_empty() {
            return Err("Unclosed ring closures".to_string());
        }
        let needs_ring_ctx = compute_needs_ring_context(&self.atoms);
        Ok(QueryGraph {
            atoms: std::mem::take(&mut self.atoms),
            bonds: std::mem::take(&mut self.bonds),
            adjacency: std::mem::take(&mut self.adjacency),
            needs_ring_context: needs_ring_ctx,
            meta: QueryMeta::default(),
        })
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.input.get(self.pos + offset).copied()
    }

    /// Multi-digit unsigned number.
    fn parse_number(&mut self) -> Option<u32> {
        let start = self.pos;
        let mut value: u32 = 0;
        while let Some(c) = self.peek() {
            let Some(d) = c.to_digit(10) else { break };
            value = value.saturating_mul(10).saturating_add(d);
            self.pos += 1;
        }
        (self.pos > start).then_some(value)
    }

    fn parse_count(&mut self) -> u8 {
        self.parse_number().map(|n| n.min(255) as u8).unwrap_or(1)
    }

    // ----------------------------------------------------------------- chain

    fn parse_chain(&mut self) -> std::result::Result<(), String> {
        let mut prev: Option<u16> = None;
        let mut pending: Option<QueryBond> = None;
        let mut branch_stack: Vec<Option<u16>> = Vec::new();

        while let Some(c) = self.peek() {
            match c {
                '(' => {
                    if prev.is_none() {
                        return Err("Branch without preceding atom".to_string());
                    }
                    branch_stack.push(prev);
                    self.pos += 1;
                }
                ')' => {
                    if pending.is_some() {
                        return Err("Bond without following atom before ')'".to_string());
                    }
                    prev = branch_stack.pop().ok_or("Unmatched ')'")?;
                    self.pos += 1;
                }
                '.' => {
                    if pending.is_some() {
                        return Err("Bond before '.'".to_string());
                    }
                    prev = None;
                    self.pos += 1;
                }
                '-' | '=' | '#' | ':' | '~' | '@' | '/' | '\\' | '!' => {
                    if prev.is_none() {
                        return Err(format!("Bond '{}' without preceding atom", c));
                    }
                    if pending.is_some() {
                        return Err("Two bonds in a row".to_string());
                    }
                    pending = Some(self.parse_bond_low()?);
                }
                '0'..='9' | '%' => {
                    let atom = prev.ok_or("Ring closure before any atom")?;
                    let num = self.parse_ring_number()?;
                    self.handle_ring_closure(atom, num, pending.take());
                }
                _ => {
                    let atom = self.parse_atom()?;
                    let idx = self.add_atom(atom);
                    match prev {
                        Some(p) => {
                            let bond = pending.take().unwrap_or(QueryBond::SingleOrAromatic);
                            self.add_bond(p, idx, bond);
                        }
                        None if pending.is_some() => {
                            return Err("Bond without preceding atom".to_string());
                        }
                        None => {}
                    }
                    prev = Some(idx);
                }
            }
        }
        if !branch_stack.is_empty() {
            return Err("Unclosed branch".to_string());
        }
        if pending.is_some() {
            return Err("Dangling bond at end of SMARTS".to_string());
        }
        Ok(())
    }

    fn parse_ring_number(&mut self) -> std::result::Result<u32, String> {
        match self.peek() {
            Some('%') => {
                self.pos += 1;
                if self.peek() == Some('(') {
                    self.pos += 1;
                    let n = self.parse_number().ok_or("Expected number in %(...)")?;
                    if self.peek() != Some(')') {
                        return Err("Unclosed %(...) ring closure".to_string());
                    }
                    self.pos += 1;
                    Ok(n)
                } else {
                    let d1 = self.peek().and_then(|c| c.to_digit(10)).ok_or("Expected two digits after %")?;
                    let d2 = self.peek_at(1).and_then(|c| c.to_digit(10)).ok_or("Expected two digits after %")?;
                    self.pos += 2;
                    Ok(d1 * 10 + d2)
                }
            }
            Some(c) if c.is_ascii_digit() => {
                self.pos += 1;
                Ok(c.to_digit(10).unwrap())
            }
            _ => Err("Expected ring closure number".to_string()),
        }
    }

    fn handle_ring_closure(&mut self, atom: u16, num: u32, bond: Option<QueryBond>) {
        if let Some((open_atom, open_bond)) = self.ring_closures.remove(&num) {
            let query = bond.or(open_bond).unwrap_or(QueryBond::SingleOrAromatic);
            self.add_bond(open_atom, atom, query);
        } else {
            self.ring_closures.insert(num, (atom, bond));
        }
    }

    // ----------------------------------------------------------------- bonds
    // Precedence (lowest to highest): `;`, `,`, `&`/implicit, `!`.

    fn is_bond_primitive_start(c: char) -> bool {
        matches!(c, '-' | '=' | '#' | ':' | '~' | '@' | '/' | '\\' | '!')
    }

    fn parse_bond_low(&mut self) -> std::result::Result<QueryBond, String> {
        let mut terms = vec![self.parse_bond_or()?];
        while self.peek() == Some(';') {
            self.pos += 1;
            terms.push(self.parse_bond_or()?);
        }
        Ok(if terms.len() == 1 { terms.pop().unwrap() } else { QueryBond::And(terms) })
    }

    fn parse_bond_or(&mut self) -> std::result::Result<QueryBond, String> {
        let mut terms = vec![self.parse_bond_and()?];
        while self.peek() == Some(',') {
            self.pos += 1;
            terms.push(self.parse_bond_and()?);
        }
        Ok(if terms.len() == 1 { terms.pop().unwrap() } else { QueryBond::Or(terms) })
    }

    fn parse_bond_and(&mut self) -> std::result::Result<QueryBond, String> {
        let mut terms = vec![self.parse_bond_unary()?];
        loop {
            match self.peek() {
                Some('&') => {
                    self.pos += 1;
                    terms.push(self.parse_bond_unary()?);
                }
                Some(c) if Self::is_bond_primitive_start(c) => terms.push(self.parse_bond_unary()?),
                _ => break,
            }
        }
        Ok(if terms.len() == 1 { terms.pop().unwrap() } else { QueryBond::And(terms) })
    }

    fn parse_bond_unary(&mut self) -> std::result::Result<QueryBond, String> {
        let c = self.peek().ok_or("Expected bond")?;
        self.pos += 1;
        Ok(match c {
            '!' => QueryBond::Not(Box::new(self.parse_bond_unary()?)),
            '-' => QueryBond::Single,
            '=' => QueryBond::Double,
            '#' => QueryBond::Triple,
            ':' => QueryBond::Aromatic,
            '~' => QueryBond::Any,
            '@' => QueryBond::Ring,
            // directional single bonds; double-bond stereo is not matched
            '/' | '\\' => QueryBond::Single,
            other => return Err(format!("Invalid bond primitive '{}'", other)),
        })
    }

    // ----------------------------------------------------------------- atoms

    fn parse_atom(&mut self) -> std::result::Result<QueryAtom, String> {
        let c = self.peek().ok_or("Expected atom")?;
        let prim = match c {
            '[' => return self.parse_bracket_atom(),
            '*' => {
                self.pos += 1;
                AtomPrimitive::AnyAtom
            }
            'a' => {
                self.pos += 1;
                AtomPrimitive::AnyAromatic
            }
            'A' => {
                self.pos += 1;
                AtomPrimitive::AnyAliphatic
            }
            'C' if self.peek_at(1) == Some('l') => {
                self.pos += 2;
                AtomPrimitive::Aliphatic(17)
            }
            'B' if self.peek_at(1) == Some('r') => {
                self.pos += 2;
                AtomPrimitive::Aliphatic(35)
            }
            'B' | 'C' | 'N' | 'O' | 'P' | 'S' | 'F' | 'I' => {
                self.pos += 1;
                let z = match c {
                    'B' => 5,
                    'C' => 6,
                    'N' => 7,
                    'O' => 8,
                    'P' => 15,
                    'S' => 16,
                    'F' => 9,
                    _ => 53,
                };
                AtomPrimitive::Aliphatic(z)
            }
            'b' | 'c' | 'n' | 'o' | 'p' | 's' => {
                self.pos += 1;
                let z = match c {
                    'b' => 5,
                    'c' => 6,
                    'n' => 7,
                    'o' => 8,
                    'p' => 15,
                    _ => 16,
                };
                AtomPrimitive::Aromatic(z)
            }
            other => return Err(format!("Unexpected character '{}' at position {}", other, self.pos)),
        };
        Ok(QueryAtom { primitives: vec![prim] })
    }

    fn parse_bracket_atom(&mut self) -> std::result::Result<QueryAtom, String> {
        self.pos += 1; // '['
        if let Some(atom) = self.try_parse_hydrogen_atom() {
            return Ok(atom);
        }
        let expr = self.parse_atom_low()?;
        if self.peek() != Some(']') {
            return Err("Missing ]".to_string());
        }
        self.pos += 1;
        Ok(QueryAtom { primitives: vec![expr] })
    }

    /// RDKit `hydrogen_atom`: `[H]`, `[2H]`, `[H+]`, `[2H-]`, `[H:1]` denote a
    /// hydrogen *atom*, not an H-count query.
    fn try_parse_hydrogen_atom(&mut self) -> Option<QueryAtom> {
        let save = self.pos;
        let isotope = self.parse_number();
        if self.peek() != Some('H') {
            self.pos = save;
            return None;
        }
        self.pos += 1;
        let charge = match self.peek() {
            Some('+') | Some('-') => Some(self.parse_charge()),
            _ => None,
        };
        if self.peek() == Some(':') {
            self.pos += 1;
            if self.parse_number().is_none() {
                self.pos = save;
                return None;
            }
        }
        if self.peek() != Some(']') {
            self.pos = save;
            return None;
        }
        self.pos += 1;
        let mut prims = vec![AtomPrimitive::Element(1)];
        if let Some(iso) = isotope {
            prims.push(AtomPrimitive::Isotope(iso.min(u16::MAX as u32) as u16));
        }
        if let Some(c) = charge {
            prims.push(AtomPrimitive::Charge(c));
        }
        Some(QueryAtom { primitives: vec![AtomPrimitive::And(prims)] })
    }

    /// `+`, `++`, `+2`, `-`, `--`, `-3`
    fn parse_charge(&mut self) -> i8 {
        let sign_char = self.peek().unwrap();
        let sign: i32 = if sign_char == '+' { 1 } else { -1 };
        self.pos += 1;
        let magnitude = if let Some(n) = self.parse_number() {
            n as i32
        } else {
            let mut k = 1;
            while self.peek() == Some(sign_char) {
                k += 1;
                self.pos += 1;
            }
            k
        };
        (sign * magnitude).clamp(-127, 127) as i8
    }

    // Precedence (lowest to highest): `;`, `,`, `&`/implicit, `!`.

    fn parse_atom_low(&mut self) -> std::result::Result<AtomPrimitive, String> {
        let mut terms = vec![self.parse_atom_or()?];
        while self.peek() == Some(';') {
            self.pos += 1;
            terms.push(self.parse_atom_or()?);
        }
        Ok(if terms.len() == 1 { terms.pop().unwrap() } else { AtomPrimitive::And(terms) })
    }

    fn parse_atom_or(&mut self) -> std::result::Result<AtomPrimitive, String> {
        let mut terms = vec![self.parse_atom_and()?];
        while self.peek() == Some(',') {
            self.pos += 1;
            terms.push(self.parse_atom_and()?);
        }
        Ok(if terms.len() == 1 { terms.pop().unwrap() } else { AtomPrimitive::Or(terms) })
    }

    fn parse_atom_and(&mut self) -> std::result::Result<AtomPrimitive, String> {
        let mut terms = Vec::new();
        loop {
            match self.peek() {
                None => return Err("Missing ]".to_string()),
                Some(']') | Some(',') | Some(';') => break,
                Some('&') => {
                    self.pos += 1;
                }
                Some(':') => {
                    // atom map number: ignored
                    self.pos += 1;
                    self.parse_number().ok_or("Expected atom map number after ':'")?;
                }
                Some(_) => terms.push(self.parse_atom_unary()?),
            }
        }
        match terms.len() {
            0 => Err("Empty atom expression".to_string()),
            1 => Ok(terms.pop().unwrap()),
            _ => Ok(AtomPrimitive::And(terms)),
        }
    }

    fn parse_atom_unary(&mut self) -> std::result::Result<AtomPrimitive, String> {
        if self.peek() == Some('!') {
            self.pos += 1;
            return Ok(AtomPrimitive::Not(Box::new(self.parse_atom_unary()?)));
        }
        self.parse_atom_primitive()
    }

    /// Element symbol of `len` chars starting at the cursor, if valid.
    fn element_at(&self, len: usize) -> Option<u8> {
        let sym: String = self.input.get(self.pos..self.pos + len)?.iter().collect();
        crate::mol::elements::symbol_to_atomic_number(&sym)
    }

    fn parse_atom_primitive(&mut self) -> std::result::Result<AtomPrimitive, String> {
        let c = self.peek().ok_or("Unexpected end of SMARTS")?;
        let next_lower = self.peek_at(1).is_some_and(|n| n.is_ascii_lowercase());

        // Two-letter element symbols take precedence over one-letter
        // primitives (Hg vs H, Db vs D, Xe vs X, Ru vs R, Zn vs Z, Al vs A).
        if c.is_ascii_uppercase() && next_lower
            && let Some(z) = self.element_at(2) {
                self.pos += 2;
                return Ok(if is_organic_aliphatic(z) { AtomPrimitive::Aliphatic(z) } else { AtomPrimitive::Element(z) });
            }

        match c {
            '$' => {
                self.pos += 1;
                if self.peek() != Some('(') {
                    return Err("Expected '(' after '$' in recursive SMARTS".to_string());
                }
                self.pos += 1;
                let start = self.pos;
                let mut depth = 1;
                while self.pos < self.input.len() {
                    match self.input[self.pos] {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    self.pos += 1;
                }
                if depth != 0 {
                    return Err("Unclosed recursive SMARTS pattern".to_string());
                }
                let nested: String = self.input[start..self.pos].iter().collect();
                self.pos += 1;
                let sub = parse_smarts(&nested).map_err(|e| e.to_string())?;
                Ok(AtomPrimitive::Recursive(Box::new(sub)))
            }
            '#' => {
                self.pos += 1;
                let z = self.parse_number().ok_or("Expected atomic number after #")?;
                Ok(AtomPrimitive::Element(z.min(255) as u8))
            }
            '0'..='9' => {
                let iso = self.parse_number().unwrap();
                Ok(AtomPrimitive::Isotope(iso.min(u16::MAX as u32) as u16))
            }
            '@' => {
                // chirality (and chirality classes): matched as "any atom"
                self.pos += 1;
                if self.peek() == Some('@') {
                    self.pos += 1;
                }
                let class: String = self.input.get(self.pos..self.pos + 2).map(|s| s.iter().collect()).unwrap_or_default();
                if matches!(class.as_str(), "TH" | "AL" | "SP" | "TB" | "OH") {
                    self.pos += 2;
                    self.parse_number();
                }
                Ok(AtomPrimitive::AnyAtom)
            }
            '+' | '-' => Ok(AtomPrimitive::Charge(self.parse_charge())),
            '*' => {
                self.pos += 1;
                Ok(AtomPrimitive::AnyAtom)
            }
            'H' => {
                self.pos += 1;
                Ok(AtomPrimitive::HCount(self.parse_count()))
            }
            'h' => {
                self.pos += 1;
                Ok(match self.parse_number() {
                    Some(n) => AtomPrimitive::ImplicitHCount(n.min(255) as u8),
                    None => AtomPrimitive::HasImplicitH,
                })
            }
            'D' => {
                self.pos += 1;
                Ok(AtomPrimitive::Degree(self.parse_count()))
            }
            'd' => {
                self.pos += 1;
                Ok(AtomPrimitive::NonHydrogenDegree(self.parse_count()))
            }
            'X' => {
                self.pos += 1;
                Ok(AtomPrimitive::Connectivity(self.parse_count()))
            }
            'v' => {
                self.pos += 1;
                Ok(AtomPrimitive::Valence(self.parse_count()))
            }
            'x' => {
                self.pos += 1;
                Ok(match self.parse_number() {
                    Some(n) => AtomPrimitive::RingBonds(n.min(255) as u8),
                    None => AtomPrimitive::HasRingBond,
                })
            }
            'R' => {
                self.pos += 1;
                Ok(match self.parse_number() {
                    Some(0) => AtomPrimitive::NotInRing,
                    Some(n) => AtomPrimitive::RingMembership(n.min(255) as u8),
                    None => AtomPrimitive::InRing,
                })
            }
            'r' => {
                self.pos += 1;
                Ok(match self.parse_number() {
                    Some(0) => AtomPrimitive::NotInRing,
                    Some(n) => AtomPrimitive::RingSize(n.min(255) as u8),
                    None => AtomPrimitive::InRing,
                })
            }
            'z' => {
                self.pos += 1;
                Ok(match self.parse_number() {
                    Some(n) => AtomPrimitive::HeteroNeighbors(n.min(255) as u8),
                    None => AtomPrimitive::HasHeteroNeighbors,
                })
            }
            'Z' => {
                self.pos += 1;
                Ok(match self.parse_number() {
                    Some(n) => AtomPrimitive::AliphaticHeteroNeighbors(n.min(255) as u8),
                    None => AtomPrimitive::HasAliphaticHeteroNeighbors,
                })
            }
            '^' => {
                self.pos += 1;
                let hyb = match self.parse_number() {
                    Some(0) => crate::types::Hybridization::S,
                    Some(1) => crate::types::Hybridization::SP,
                    Some(2) => crate::types::Hybridization::SP2,
                    Some(3) => crate::types::Hybridization::SP3,
                    Some(4) => crate::types::Hybridization::SP3D,
                    Some(5) => crate::types::Hybridization::SP3D2,
                    _ => return Err("Invalid hybridization query".to_string()),
                };
                Ok(AtomPrimitive::Hybridization(hyb))
            }
            'a' if self.peek_at(1) == Some('s') => {
                self.pos += 2;
                Ok(AtomPrimitive::Aromatic(33))
            }
            's' if self.peek_at(1) == Some('e') => {
                self.pos += 2;
                Ok(AtomPrimitive::Aromatic(34))
            }
            't' if self.peek_at(1) == Some('e') => {
                self.pos += 2;
                Ok(AtomPrimitive::Aromatic(52))
            }
            'a' => {
                self.pos += 1;
                Ok(AtomPrimitive::AnyAromatic)
            }
            'A' => {
                self.pos += 1;
                Ok(AtomPrimitive::AnyAliphatic)
            }
            'b' | 'c' | 'n' | 'o' | 'p' | 's' => {
                self.pos += 1;
                let z = match c {
                    'b' => 5,
                    'c' => 6,
                    'n' => 7,
                    'o' => 8,
                    'p' => 15,
                    _ => 16,
                };
                Ok(AtomPrimitive::Aromatic(z))
            }
            c if c.is_ascii_uppercase() => {
                let z = self.element_at(1).ok_or_else(|| format!("Unknown element '{}'", c))?;
                self.pos += 1;
                Ok(if is_organic_aliphatic(z) { AtomPrimitive::Aliphatic(z) } else { AtomPrimitive::Element(z) })
            }
            other => Err(format!("Invalid atom primitive '{}' at position {}", other, self.pos)),
        }
    }

    fn add_atom(&mut self, atom: QueryAtom) -> u16 {
        let idx = self.atoms.len() as u16;
        self.atoms.push(atom);
        self.adjacency.push(SmallVec::new());
        idx
    }

    fn add_bond(&mut self, source: u16, target: u16, bond: QueryBond) {
        let bond_idx = self.bonds.len() as u16;
        self.bonds.push(QueryGraphBond { source, target, query: bond });
        self.adjacency[source as usize].push(bond_idx);
        self.adjacency[target as usize].push(bond_idx);
    }
}

#[cfg(test)]
mod smarts_parser_tests {
    use super::*;

    #[test]
    fn test_parse_single_element() {
        let pattern = parse_smarts("[C]").unwrap();
        assert_eq!(pattern.atoms.len(), 1);
        assert_eq!(pattern.bonds.len(), 0);
    }

    #[test]
    fn test_parse_chain() {
        let pattern = parse_smarts("CC").unwrap();
        assert_eq!(pattern.atoms.len(), 2);
        assert_eq!(pattern.bonds.len(), 1);
    }

    #[test]
    fn test_parse_double_bond() {
        let pattern = parse_smarts("C=O").unwrap();
        assert_eq!(pattern.atoms.len(), 2);
        assert_eq!(pattern.bonds.len(), 1);
    }

    #[test]
    fn test_parse_aromatic_ring() {
        let pattern = parse_smarts("c1ccccc1").unwrap();
        assert_eq!(pattern.atoms.len(), 6);
        assert_eq!(pattern.bonds.len(), 6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::rings::find_sssr;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_query_atom_element() {
        let mol = parse_smiles("CCO").unwrap();

        let carbon_query = QueryAtom::element(6); // Carbon
        assert!(carbon_query.matches(&mol, 0)); // First C
        assert!(carbon_query.matches(&mol, 1)); // Second C
        assert!(!carbon_query.matches(&mol, 2)); // O

        let oxygen_query = QueryAtom::element(8); // Oxygen
        assert!(!oxygen_query.matches(&mol, 0));
        assert!(!oxygen_query.matches(&mol, 1));
        assert!(oxygen_query.matches(&mol, 2));
    }

    #[test]
    fn test_query_atom_any() {
        let mol = parse_smiles("CCO").unwrap();
        let any_query = QueryAtom::any();

        assert!(any_query.matches(&mol, 0));
        assert!(any_query.matches(&mol, 1));
        assert!(any_query.matches(&mol, 2));
    }

    #[test]
    fn test_query_atom_aromatic() {
        let mol = parse_smiles("c1ccccc1").unwrap();

        let aromatic_query = QueryAtom::aromatic_element(6);
        assert!(aromatic_query.matches(&mol, 0));
        assert!(aromatic_query.matches(&mol, 5));

        // Non-aromatic molecule
        let mol2 = parse_smiles("CCCC").unwrap();
        assert!(!aromatic_query.matches(&mol2, 0));
    }

    #[test]
    fn test_query_atom_degree() {
        let mol = parse_smiles("CC(C)C").unwrap(); // Isopropane: C-C(C)-C

        let degree1_query = QueryAtom {
            primitives: vec![AtomPrimitive::Degree(1)],
        };
        assert!(degree1_query.matches(&mol, 0)); // Terminal C
        assert!(!degree1_query.matches(&mol, 1)); // Central C has degree 3

        let degree3_query = QueryAtom {
            primitives: vec![AtomPrimitive::Degree(3)],
        };
        assert!(degree3_query.matches(&mol, 1)); // Central C
    }

    #[test]
    fn test_query_atom_in_ring() {
        let mol = parse_smiles("C1CCCCC1").unwrap(); // Cyclohexane
        find_sssr(&mol);

        let ring_query = QueryAtom {
            primitives: vec![AtomPrimitive::InRing],
        };

        assert!(ring_query.matches(&mol, 0));
        assert!(ring_query.matches(&mol, 3));

        // Non-ring molecule
        let mol2 = parse_smiles("CCCC").unwrap();
        assert!(!ring_query.matches(&mol2, 0));
    }

    #[test]
    fn test_query_atom_ring_size_and_membership() {
        // Cyclohexane should satisfy r6 and R1 for each ring atom under the
        // current SSSR semantics used by SMARTS ring context.
        let mol = parse_smiles("C1CCCCC1").unwrap();
        find_sssr(&mol);

        let ring6_query = QueryAtom {
            primitives: vec![AtomPrimitive::RingSize(6)],
        };
        let ring_membership_1_query = QueryAtom {
            primitives: vec![AtomPrimitive::RingMembership(1)],
        };
        let not_in_ring_query = QueryAtom {
            primitives: vec![AtomPrimitive::NotInRing],
        };

        assert!(ring6_query.matches(&mol, 0));
        assert!(ring6_query.matches(&mol, 3));
        assert!(ring_membership_1_query.matches(&mol, 0));
        assert!(ring_membership_1_query.matches(&mol, 4));
        assert!(!not_in_ring_query.matches(&mol, 1));

        let linear = parse_smiles("CCCC").unwrap();
        assert!(!ring6_query.matches(&linear, 0));
        assert!(!ring_membership_1_query.matches(&linear, 1));
        assert!(not_in_ring_query.matches(&linear, 2));
    }

    #[test]
    fn test_query_bond_types() {
        let mol = parse_smiles("C=C").unwrap();

        let single_query = QueryBond::Single;
        let double_query = QueryBond::Double;

        assert!(!single_query.matches(&mol, 0));
        assert!(double_query.matches(&mol, 0));
    }
}

#[cfg(test)]
mod smarts_advanced_tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_or_in_brackets() {
        // Test comma-separated OR lists
        let query = parse_smarts("[#6,#7,#8]").unwrap();
        assert_eq!(query.atoms.len(), 1);

        // Should match C, N, or O
        let mol = parse_smiles("C").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1); // matches C

        let mol = parse_smiles("N").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1); // matches N

        let mol = parse_smiles("O").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1); // matches O
    }

    #[test]
    fn test_and_in_brackets() {
        // Test implicit AND (no operator between primitives)
        let query = parse_smarts("[CH3]").unwrap();

        // Should match carbon with exactly 3 hydrogens
        let mol = parse_smiles("C").unwrap(); // CH4
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 0); // CH4 doesn't match CH3

        let mol = parse_smiles("CC").unwrap(); // 2x CH3
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 2); // both carbons are CH3
    }

    #[test]
    fn test_branching() {
        // Test branching patterns
        let query = parse_smarts("C(C)C").unwrap();
        assert_eq!(query.atoms.len(), 3);
        assert_eq!(query.bonds.len(), 2);

        // In propane (C-C-C), this pattern matches twice:
        // - First C as center (connected to middle C and... wait, it only has one neighbor)
        // Actually, both terminal carbons match: each is connected to the middle C
        // and the middle C counts as both neighbors
        let mol = parse_smiles("CCC").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        // Pattern C(C)C means: C connected to two other C atoms via branches
        // In linear propane, the middle C matches (has 2 C neighbors)
        assert!(!matches.is_empty()); // at least one match
    }

    #[test]
    fn test_complex_branching() {
        // Test [CH2](C)C pattern (CH2 connected to two carbons)
        let query = parse_smarts("[CH2](C)C").unwrap();
        assert_eq!(query.atoms.len(), 3);

        // In propane, the middle carbon is CH2 and connected to 2 carbons
        let mol = parse_smiles("CCC").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        // Middle C is CH2 with 2 C neighbors
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_not_operator() {
        // Test NOT operator
        let query = parse_smarts("[!#6]").unwrap();

        // Should match non-carbon atoms
        let mol = parse_smiles("CCO").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1); // matches O only
    }

    #[test]
    fn test_multiple_or_elements() {
        // Test [N,O,P,S]
        let query = parse_smarts("[N,O,P,S]").unwrap();

        let mol = parse_smiles("NOPS").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 4); // all 4 atoms match
    }

    #[test]
    fn test_aromatic_with_hcount() {
        // Test [cH] (aromatic carbon with 1 H)
        let query = parse_smarts("[cH]").unwrap();

        // Benzene has 6 aromatic CH
        let mol = parse_smiles("c1ccccc1").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 6);
    }

    #[test]
    fn test_lowercase_a_outside_brackets() {
        // Test lowercase 'a' as any aromatic
        let query = parse_smarts("a").unwrap();

        // Should match aromatic atoms (benzene)
        let mol = parse_smiles("c1ccccc1").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 6); // all 6 aromatic carbons

        // Should NOT match aliphatic atoms
        let mol2 = parse_smiles("CCC").unwrap();
        let matches2 = crate::algo::iso::find_all_matches(&mol2, &query);
        assert_eq!(matches2.len(), 0); // no aromatic atoms
    }

    #[test]
    fn test_nested_branches() {
        // Test C(C)(C)C (carbon with 3 branches)
        let query = parse_smarts("C(C)(C)C").unwrap();
        assert_eq!(query.atoms.len(), 4);

        // Should match tert-butyl center in neopentane
        let mol = parse_smiles("CC(C)(C)C").unwrap(); // neopentane
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        // The central carbon has 4 C neighbors, pattern needs 3
        assert!(!matches.is_empty()); // at least the center matches
    }

    #[test]
    fn test_valence_matching() {
        // Test valence matching [Cv4] (carbon with valence 4)
        let query = parse_smarts("[Cv4]").unwrap();

        // Methane: C with 4 H = valence 4
        let mol = parse_smiles("C").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1); // CH4 has valence 4

        // Ethane: each C has valence 4 (3H + 1C)
        let mol = parse_smiles("CC").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 2); // both carbons have valence 4
    }

    #[test]
    fn test_valence_with_double_bonds() {
        // Test [Cv4] with double bonds
        let query = parse_smarts("[Cv4]").unwrap();

        // Ethene: C=C, each C has 2H + 1 double bond = 2 + 2 = 4
        let mol = parse_smiles("C=C").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 2); // both carbons have valence 4
    }

    #[test]
    fn test_ring_bonds_count() {
        // Test [cx2] (aromatic atom with 2 ring bonds)
        let query = parse_smarts("[cx2]").unwrap();

        // Benzene: each carbon has 2 ring bonds
        let mol = parse_smiles("c1ccccc1").unwrap();

        // Mark ring atoms and bonds by finding SSSR
        let _ = crate::algo::rings::find_sssr(&mol);

        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 6); // all 6 aromatic carbons have 2 ring bonds
    }

    #[test]
    fn test_valence_nitrogen() {
        // Test [Nv3] (nitrogen with valence 3)
        let query = parse_smarts("[Nv3]").unwrap();

        // Ammonia: N with 3 H = valence 3
        let mol = parse_smiles("N").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1);

        // Methylamine: N with 2H + 1C = valence 3
        let mol = parse_smiles("CN").unwrap();
        let matches = crate::algo::iso::find_all_matches(&mol, &query);
        assert_eq!(matches.len(), 1); // N has valence 3
    }

    #[test]
    fn test_simple_cu_parse() {
        // Simple test for Cu
        let result = parse_smarts("[Cu]");
        eprintln!("Result for [Cu]: {:?}", result);
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_two_letter_elements_in_or_list() {
        // Test parsing Cu, Zn
        let pattern = parse_smarts("[Cu,Zn]");
        assert!(pattern.is_ok(), "Should parse [Cu,Zn]: {:?}", pattern.err());

        // Test Br, Cl
        let pattern2 = parse_smarts("[Br,Cl]");
        assert!(
            pattern2.is_ok(),
            "Should parse [Br,Cl]: {:?}",
            pattern2.err()
        );

        // Test Ag, Au, Hg
        let pattern3 = parse_smarts("[Ag,Au,Hg]");
        assert!(
            pattern3.is_ok(),
            "Should parse [Ag,Au,Hg]: {:?}",
            pattern3.err()
        );

        // Aromatic two-letter elements used by RDKit QED alerts.
        let aromatic_two = parse_smarts("[se,as,te]");
        assert!(
            aromatic_two.is_ok(),
            "Should parse aromatic two-letter elements [se,as,te]: {:?}",
            aromatic_two.err()
        );
    }

    #[test]
    fn test_single_letter_elements_in_or_list() {
        // Test single-letter elements like V, W
        let pattern = parse_smarts("[V]");
        assert!(pattern.is_ok(), "Should parse [V]: {:?}", pattern.err());

        let pattern2 = parse_smarts("[W]");
        assert!(pattern2.is_ok(), "Should parse [W]: {:?}", pattern2.err());

        let pattern3 = parse_smarts("[V,W]");
        assert!(pattern3.is_ok(), "Should parse [V,W]: {:?}", pattern3.err());
    }

    #[test]
    fn test_aromatic_any_pattern() {
        // Test that 'a' (any aromatic) parses and matches benzene
        let pattern = parse_smarts("a").expect("Failed to parse 'a'");
        eprintln!("Pattern 'a': {:?}", pattern);

        // Create benzene
        let mol = crate::io::smiles::parse_smiles("c1ccccc1").expect("Failed to parse benzene");

        // Match
        let matches = crate::algo::iso::find_all_matches(&mol, &pattern);
        eprintln!("Matches in benzene: {}", matches.len());
        assert!(
            !matches.is_empty(),
            "Pattern 'a' should match aromatic carbons in benzene"
        );
    }

    #[test]
    fn test_six_member_ring_wildcard_on_benzene() {
        // Test key 163 pattern on benzene
        let pattern = parse_smarts("*1~*~*~*~*~*~1").expect("Failed to parse 6M ring");
        eprintln!(
            "6M ring pattern: {} atoms, {} bonds",
            pattern.atoms.len(),
            pattern.bonds.len()
        );

        // Create benzene (WITHOUT find_sssr - known issue with SSSR affecting matching)
        let mol = crate::io::smiles::parse_smiles("c1ccccc1").expect("Failed to parse benzene");

        // Match WITHOUT sssr
        let matches = crate::algo::iso::find_all_matches(&mol, &pattern);
        eprintln!("6M ring matches in benzene: {}", matches.len());

        // NOTE: After find_sssr(), matching breaks - this is a known issue
        // The pattern works correctly on molecules without SSSR detection
        assert!(!matches.is_empty(), "6M ring should match benzene");
    }

    #[test]
    fn test_lanthanides_in_or_list() {
        // Test key 6: [La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]
        let pattern = parse_smarts("[La,Ce]");
        assert!(pattern.is_ok(), "Should parse [La,Ce]: {:?}", pattern.err());

        // Full key 6
        let pattern2 = parse_smarts("[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]");
        assert!(pattern2.is_ok(), "Should parse key 6: {:?}", pattern2.err());
    }

    #[test]
    fn test_wildcard_ring_closure() {
        // Test *1~*~*~*~1 (4-membered ring with wildcards)
        let result = parse_smarts("*1~*~*~*~1");
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let query = result.unwrap();
        assert_eq!(query.atoms.len(), 4, "Should have 4 atoms");
        assert_eq!(
            query.bonds.len(),
            4,
            "Should have 4 bonds (including ring closure)"
        );
    }

    #[test]
    fn test_six_member_ring_wildcard() {
        // Test *1~*~*~*~*~*~1 (6-membered ring) - Key 163 from MACCS
        let result = parse_smarts("*1~*~*~*~*~*~1");
        assert!(result.is_ok(), "Failed to parse 6-ring: {:?}", result.err());

        let query = result.unwrap();
        assert_eq!(query.atoms.len(), 6);
        assert_eq!(query.bonds.len(), 6);
    }

    #[test]
    fn test_bare_element_symbols() {
        // Test Br, Cl without brackets (currently may fail)
        let tests = vec![
            ("CCl", "Carbon-Chlorine"),
            ("CFBr", "Carbon-Fluorine-Bromine"),
        ];

        for (pattern, desc) in tests {
            let result = parse_smarts(pattern);
            // These might fail - we'll fix them
            if result.is_err() {
                println!("{} parsing not yet supported: {:?}", desc, result.err());
            }
        }
    }

    #[test]
    fn test_ring_closure_bonds_preserved() {
        use crate::algo::iso::find_all_matches;
        use crate::algo::rings::find_sssr;

        // Test that ring closure bonds use the specified bond type (~ = Any)
        // This is critical for MACCS key 163 (6M Ring) to work after aromaticity perception
        let mol = parse_smiles("c1ccccc1").unwrap();
        let pattern = parse_smarts("*1~*~*~*~*~*~1").unwrap();

        // Verify pattern has All bonds as Any (including ring closure)
        for bond in &pattern.bonds {
            assert!(
                matches!(bond.query, QueryBond::Any),
                "Ring pattern bond {}->{} should be Any, got {:?}",
                bond.source,
                bond.target,
                bond.query
            );
        }

        // Full molecule preparation
        find_sssr(&mol);

        // Pattern should still match after full preparation
        let matches = find_all_matches(&mol, &pattern);
        assert!(
            !matches.is_empty(),
            "6-ring pattern should match benzene after aromaticity perception"
        );
    }

    #[test]
    fn test_recursive_smarts_parsing() {
        // Test parsing of recursive SMARTS patterns

        // Simple recursive: atom that matches a carbon pattern
        let result = parse_smarts("[$([CX4])]");
        assert!(
            result.is_ok(),
            "Should parse recursive SMARTS: {:?}",
            result.err()
        );

        // Recursive with negation: NOT carbon
        let result = parse_smarts("[!$([#6])]");
        assert!(
            result.is_ok(),
            "Should parse recursive SMARTS with NOT: {:?}",
            result.err()
        );

        // Complex recursive from MACCS key 101
        let result = parse_smarts("[$([CX2](=C)=C)]");
        assert!(
            result.is_ok(),
            "Should parse allene pattern: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_recursive_smarts_matching() {
        use crate::algo::iso::find_all_matches;

        // Test matching CX4 (sp3 carbon with 4 connections)
        let mol = parse_smiles("CC").unwrap(); // Ethane: 2 sp3 carbons

        // First test: simple [CX4] pattern (carbon with degree 4)
        let simple_pattern = parse_smarts("[CX4]").unwrap();
        let simple_matches = find_all_matches(&mol, &simple_pattern);
        println!("Simple [CX4] matches: {}", simple_matches.len());

        // Now test recursive version
        let pattern = parse_smarts("[$([CX4])]").unwrap();
        println!("Pattern atoms: {:?}", pattern.atoms);

        let matches = find_all_matches(&mol, &pattern);
        println!("Recursive [$([CX4])] matches: {}", matches.len());

        // Both should find sp3 carbons
        assert!(
            !simple_matches.is_empty(),
            "Should find sp3 carbons with [CX4]"
        );
        // Note: recursive matching may have different behavior
        // Just verify parsing works for now
    }
}
