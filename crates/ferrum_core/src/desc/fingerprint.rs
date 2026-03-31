//! Morgan/ECFP circular fingerprints.
//!
//! Implements Extended-Connectivity Fingerprints (ECFP) using the Morgan algorithm.
//! Circular fingerprints encode the local chemical environment around each atom
//! up to a given radius, making them the most widely used fingerprints in ML/QSAR.
//!
//! Variants:
//! - ECFP2 (radius=1): immediate neighbors
//! - ECFP4 (radius=2): 2-bond neighborhood (default, most common)
//! - ECFP6 (radius=3): 3-bond neighborhood
//!
//! Reference: Rogers, D. & Hahn, M. (2010) "Extended-Connectivity Fingerprints"
//! J. Chem. Inf. Model. 50(5), 742-754.


//! ## File guide: Fingerprint
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn rdkit_hash_combine`
//! - `fn rdkit_hash_u32_slice`
//! - `fn rdkit_hash_pair_i32_u32`
//! - `struct BondNeighborhood`
//! - `fn new`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - ECFP/FCFP hashing follows RDKit-compatible combine/folding behavior so bit collisions mirror parity expectations.
//! - Feature invariants depend on SMARTS-derived donor/acceptor and atom typing semantics from `io::smarts`.
//! - Radius expansion assumes stable bond neighborhoods and deterministic atom ordering across runs.
//! ### Chemistry/ML references
//! - Rogers, D. & Hahn, M. (2010). Extended-Connectivity Fingerprints (ECFP).
//! - RDKit Morgan fingerprint implementation notes for invariant generation and hash folding.
//! - Daylight SMARTS syntax used for pharmacophore-like feature detection primitives.

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::iso::visit_matches;
use crate::io::smarts::parse_smarts;
use crate::mol::elements::ATOMIC_MASSES;
use crate::mol::molecule::Molecule;
use crate::types::BondType;
use bitvec::prelude::*;
use once_cell::sync::Lazy;
use std::collections::HashSet;

/// Implements hash combine.
///
/// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn rdkit_hash_combine(seed: &mut u32, value: u32) {
    *seed ^= value
        .wrapping_add(0x9e37_79b9)
        .wrapping_add(seed.wrapping_shl(6))
        .wrapping_add(seed.wrapping_shr(2));
}

/// Implements hash u32 slice.
///
/// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn rdkit_hash_u32_slice(values: &[u32]) -> u32 {
    let mut seed = 0u32;
    for &value in values {
        rdkit_hash_combine(&mut seed, value);
    }
    seed
}

/// Implements hash pair i32 u32.
///
/// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn rdkit_hash_pair_i32_u32(first: i32, second: u32) -> u32 {
    let mut seed = 0u32;
    rdkit_hash_combine(&mut seed, first as u32);
    rdkit_hash_combine(&mut seed, second);
    seed
}

#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct BondNeighborhood(Vec<u64>);

impl BondNeighborhood {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn new(word_count: usize) -> Self {
        Self(vec![0u64; word_count])
    }

    #[inline]
    /// Sets bit.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn set_bit(&mut self, bit_idx: usize) {
        let word = bit_idx / 64;
        let bit = bit_idx % 64;
        self.0[word] |= 1u64 << bit;
    }

    #[inline]
    /// Implements with.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn union_with(&mut self, other: &Self) {
        for (slot, rhs) in self.0.iter_mut().zip(other.0.iter()) {
            *slot |= *rhs;
        }
    }
}

/// Implements mass.
///
/// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn atom_mass(atom: &crate::mol::atom::Atom) -> f32 {
    if atom.isotope > 0 {
        atom.isotope as f32
    } else {
        ATOMIC_MASSES[atom.atomic_number as usize]
    }
}

/// Default RDKit-like connectivity invariants for Morgan fingerprints.
///
/// Components:
/// - atomic number
/// - total degree (explicit degree + total hydrogens)
/// - total hydrogen count
/// - formal charge
/// - delta mass (isotope-aware)
/// - optional ring membership flag
fn morgan_connectivity_invariants(mol: &Molecule, include_ring_membership: bool) -> Vec<u32> {
    let mut invars = vec![0u32; mol.num_atoms()];
    for atom_idx in 0..mol.num_atoms() {
        let atom = &mol.atoms[atom_idx];
        let total_h = atom.total_h() as u32;
        let total_degree = mol.get_degree(atom_idx as u16) as u32 + total_h;
        let base_mass = ATOMIC_MASSES[atom.atomic_number as usize];
        let delta_mass = (atom_mass(atom) - base_mass) as i32;

        let mut components = [0u32; 6];
        let mut len = 0usize;
        components[len] = atom.atomic_number as u32;
        len += 1;
        components[len] = total_degree;
        len += 1;
        components[len] = total_h;
        len += 1;
        components[len] = (atom.formal_charge as i32) as u32;
        len += 1;
        components[len] = delta_mass as u32;
        len += 1;
        if include_ring_membership && atom.is_in_ring() {
            components[len] = 1;
            len += 1;
        }
        invars[atom_idx] = rdkit_hash_u32_slice(&components[..len]);
    }
    invars
}

const FCFP_FEATURE_SMARTS: [&str; 6] = [
    "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]",
    "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]",
    "[a]",
    "[F,Cl,Br,I]",
    "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]",
    "[$([C,S](=[O,S,P])-[O;H1,-1])]",
];

static FCFP_FEATURE_PATTERNS: Lazy<Vec<crate::io::smarts::QueryGraph>> = Lazy::new(|| {
    FCFP_FEATURE_SMARTS
        .iter()
        .map(|smarts| parse_smarts(smarts).expect("valid RDKit FCFP SMARTS pattern"))
        .collect()
});

/// RDKit-like Gobbi/Poppinger feature invariants used by FCFP.
fn morgan_feature_invariants(mol: &Molecule) -> Vec<u32> {
    let mut invars = vec![0u32; mol.num_atoms()];
    for (feature_idx, pattern) in FCFP_FEATURE_PATTERNS.iter().enumerate() {
        let mask = 1u32 << feature_idx;
        visit_matches(mol, pattern, |mapping| {
            if let Some(&target_atom) = mapping.first() {
                invars[target_atom as usize] |= mask;
            }
            true
        });
    }
    invars
}

/// Implements bond invariant.
///
/// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn morgan_bond_invariant(bond: &crate::mol::bond::Bond) -> i32 {
    match bond.bond_type {
        BondType::Single => 1,
        BondType::Double => 2,
        BondType::Triple => 3,
        BondType::Aromatic => 12,
        BondType::Dative => 17,
        BondType::Unknown => 0,
    }
}

/// Implements features from invariants.
///
/// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn morgan_features_from_invariants(mol: &Molecule, radius: u32, atom_invariants: Vec<u32>) -> Vec<u64> {
    let n_atoms = mol.num_atoms();
    if n_atoms == 0 {
        return vec![];
    }

    let include_redundant_environments = false;
    let mut features = Vec::with_capacity((radius as usize + 1) * n_atoms);
    let mut current_invariants = atom_invariants;
    let mut next_layer_invariants = vec![0u32; n_atoms];
    let bond_invariants: Vec<i32> = mol.bonds.iter().map(morgan_bond_invariant).collect();
    let mut dead_atoms = vec![false; n_atoms];

    // Layer-0 environments.
    features.extend(current_invariants.iter().map(|&v| v as u64));

    let neighborhood_word_count = (mol.num_bonds() + 63) / 64;
    let mut atom_neighborhoods = vec![BondNeighborhood::new(neighborhood_word_count); n_atoms];
    let mut round_atom_neighborhoods = atom_neighborhoods.clone();
    let mut seen_neighborhoods = HashSet::with_capacity((radius as usize + 1) * n_atoms);

    for layer in 0..radius {
        let mut all_neighborhoods_this_round: Vec<(BondNeighborhood, u32, usize)> =
            Vec::with_capacity(n_atoms);

        for atom_idx in 0..n_atoms {
            if dead_atoms[atom_idx] {
                continue;
            }
            if mol.get_degree(atom_idx as u16) == 0 {
                dead_atoms[atom_idx] = true;
                continue;
            }

            let mut neighborhood_invariants: Vec<(i32, u32)> =
                Vec::with_capacity(mol.adjacency[atom_idx].len());

            for &bond_idx_u16 in &mol.adjacency[atom_idx] {
                let bond_idx = bond_idx_u16 as usize;
                let bond = &mol.bonds[bond_idx];
                let neighbor_idx = if bond.source as usize == atom_idx {
                    bond.target as usize
                } else {
                    bond.source as usize
                };

                round_atom_neighborhoods[atom_idx].set_bit(bond_idx);
                round_atom_neighborhoods[atom_idx].union_with(&atom_neighborhoods[neighbor_idx]);
                neighborhood_invariants.push((bond_invariants[bond_idx], current_invariants[neighbor_idx]));
            }

            neighborhood_invariants.sort_unstable();

            let mut invariant = layer;
            rdkit_hash_combine(&mut invariant, current_invariants[atom_idx]);
            for (bond_inv, neighbor_inv) in neighborhood_invariants {
                let pair_hash = rdkit_hash_pair_i32_u32(bond_inv, neighbor_inv);
                rdkit_hash_combine(&mut invariant, pair_hash);
            }

            next_layer_invariants[atom_idx] = invariant;
            all_neighborhoods_this_round.push((
                round_atom_neighborhoods[atom_idx].clone(),
                invariant,
                atom_idx,
            ));
        }

        all_neighborhoods_this_round.sort_unstable();
        for (neighborhood, invariant, atom_idx) in all_neighborhoods_this_round {
            if include_redundant_environments || !seen_neighborhoods.contains(&neighborhood) {
                features.push(invariant as u64);
                seen_neighborhoods.insert(neighborhood);
            } else {
                dead_atoms[atom_idx] = true;
            }
        }

        current_invariants.copy_from_slice(&next_layer_invariants);
        next_layer_invariants.fill(0);
        atom_neighborhoods = round_atom_neighborhoods.clone();
    }

    features
}

/// ECFP features: Morgan with atomic invariants.
fn morgan_features(mol: &Molecule, radius: u32) -> Vec<u64> {
    let atom_invariants = morgan_connectivity_invariants(mol, true);
    morgan_features_from_invariants(mol, radius, atom_invariants)
}

/// FCFP features: Morgan with pharmacophoric invariants.
fn fcfp_features(mol: &Molecule, radius: u32) -> Vec<u64> {
    let atom_invariants = morgan_feature_invariants(mol);
    morgan_features_from_invariants(mol, radius, atom_invariants)
}

/// Compute Tanimoto similarity between two BitVec fingerprints.
///
/// Tanimoto = |A ∩ B| / |A ∪ B|
pub fn tanimoto_similarity(fp1: &BitVec<u8, Lsb0>, fp2: &BitVec<u8, Lsb0>) -> f32 {
    debug_assert_eq!(fp1.len(), fp2.len());
    let mut and_count = 0u32;
    let mut or_count = 0u32;

    for (a, b) in fp1.iter().by_vals().zip(fp2.iter().by_vals()) {
        if a && b {
            and_count += 1;
        }
        if a || b {
            or_count += 1;
        }
    }

    if or_count == 0 {
        1.0 // Both empty = identical
    } else {
        and_count as f32 / or_count as f32
    }
}

// ===========================================================================
// ECFP/FCFP Descriptors (Binary BitVec)
// ===========================================================================

/// Fold feature hashes into a fixed-size bit vector.
fn fold_fingerprint(features: &[u64], num_bits: usize) -> BitVec<u8, Lsb0> {
    let mut bits = bitvec![u8, Lsb0; 0; num_bits];
    for &hash in features {
        let bit = (hash as usize) % num_bits;
        bits.set(bit, true);
    }
    bits
}

/// ECFP (Extended-Connectivity Fingerprints) descriptor.
///
/// Circular fingerprints using atomic invariants (element, degree, H-count, charge,
/// aromaticity, ring membership). Returns bit-packed fingerprint.
///
/// Variants:
/// - `radius=1, bits=2048` → ECFP2
/// - `radius=2, bits=2048` → ECFP4 (most common)
/// - `radius=3, bits=2048` → ECFP6
///
/// # Examples
///
/// ```
/// use ferrum_core::desc::fingerprint::ECFP;
/// use ferrum_core::desc::typed::Descriptor;
/// use ferrum_core::io::smiles::parse_smiles;
/// use ferrum_core::algo::aromaticity::perceive_aromaticity;
///
/// let mut mol = parse_smiles("c1ccccc1").unwrap();
/// perceive_aromaticity(&mut mol);
///
/// let descriptor = ECFP::ecfp4();
/// let fp = descriptor.calculate(&mut mol, &mut ferrum_core::desc::context::CalculationContext::new());
///
/// let bits = fp.as_binary();
/// assert_eq!(bits.len(), 2048);
/// assert!(bits.count_ones() > 0);
/// ```
pub struct ECFP {
    pub radius: u8,
    pub num_bits: usize,
}

/// Backward-compatible Morgan alias for ECFP.
pub type Morgan = ECFP;

impl ECFP {
    /// Create ECFP2 (radius=1, 2048 bits).
    pub fn ecfp2() -> Self {
        Self {
            radius: 1,
            num_bits: 2048,
        }
    }

    /// Create ECFP4 (radius=2, 2048 bits) - most common variant.
    pub fn ecfp4() -> Self {
        Self {
            radius: 2,
            num_bits: 2048,
        }
    }

    /// Create ECFP6 (radius=3, 2048 bits).
    pub fn ecfp6() -> Self {
        Self {
            radius: 3,
            num_bits: 2048,
        }
    }

    /// Create with custom parameters.
    pub fn new(radius: u8, num_bits: usize) -> Self {
        Self { radius, num_bits }
    }

    /// Get the raw (unfolded) feature hashes for a molecule.
    pub fn raw_features(&self, mol: &Molecule) -> Vec<u64> {
        morgan_features(mol, self.radius as u32)
    }
}

impl Descriptor for ECFP {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        match self.radius {
            1 => "ECFP2",
            2 => "ECFP4",
            3 => "ECFP6",
            _ => "ECFP",
        }
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let features = morgan_features(mol, self.radius as u32);
        let bits = fold_fingerprint(&features, self.num_bits);
        DescriptorValue::Binary(bits)
    }
}

/// FCFP (Functional-Class Fingerprints) descriptor.
///
/// Like ECFP but uses pharmacophoric atom features instead of atomic identity.
/// Returns bit-packed fingerprint.
///
/// # Examples
///
/// ```
/// use ferrum_core::desc::fingerprint::FCFP;
/// use ferrum_core::desc::typed::Descriptor;
/// use ferrum_core::io::smiles::parse_smiles;
/// use ferrum_core::algo::aromaticity::perceive_aromaticity;
///
/// let mut mol = parse_smiles("c1ccccc1").unwrap();
/// perceive_aromaticity(&mut mol);
///
/// let descriptor = FCFP::fcfp4();
/// let fp = descriptor.calculate(&mut mol, &mut ferrum_core::desc::context::CalculationContext::new());
///
/// let bits = fp.as_binary();
/// assert_eq!(bits.len(), 2048);
/// ```
pub struct FCFP {
    pub radius: u8,
    pub num_bits: usize,
}

impl FCFP {
    /// Create FCFP2 (radius=1, 2048 bits).
    pub fn fcfp2() -> Self {
        Self {
            radius: 1,
            num_bits: 2048,
        }
    }

    /// Create FCFP4 (radius=2, 2048 bits) - most common variant.
    pub fn fcfp4() -> Self {
        Self {
            radius: 2,
            num_bits: 2048,
        }
    }

    /// Create FCFP6 (radius=3, 2048 bits).
    pub fn fcfp6() -> Self {
        Self {
            radius: 3,
            num_bits: 2048,
        }
    }

    /// Create with custom parameters.
    pub fn new(radius: u8, num_bits: usize) -> Self {
        Self { radius, num_bits }
    }

    /// Get the raw (unfolded) feature hashes for a molecule.
    pub fn raw_features(&self, mol: &Molecule) -> Vec<u64> {
        fcfp_features(mol, self.radius as u32)
    }
}

impl Descriptor for FCFP {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        match self.radius {
            1 => "FCFP2",
            2 => "FCFP4",
            3 => "FCFP6",
            _ => "FCFP",
        }
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let features = fcfp_features(mol, self.radius as u32);
        let bits = fold_fingerprint(&features, self.num_bits);
        DescriptorValue::Binary(bits)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ECFP>;
    let _ = assert_send_sync::<FCFP>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements fp.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_fp(smiles: &str, radius: u8, bits: usize) -> BitVec<u8, Lsb0> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        let desc = ECFP::new(radius, bits);
        desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_binary().clone()
    }

    /// Counts bits.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn count_bits(fp: &BitVec<u8, Lsb0>) -> usize {
        fp.count_ones()
    }

    #[test]
    /// Runs regression test for methane has bits.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_methane_has_bits() {
        let fp = calc_fp("C", 2, 2048);
        let bits = count_bits(&fp);
        assert!(bits > 0, "Methane should set some bits");
        assert!(bits < 50, "Methane should set few bits, got {}", bits);
    }

    #[test]
    /// Runs regression test for benzene has bits.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_benzene_has_bits() {
        let fp = calc_fp("c1ccccc1", 2, 2048);
        let bits = count_bits(&fp);
        assert!(bits > 0, "Benzene should set bits");
    }

    #[test]
    /// Runs regression test for empty molecule.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_empty_molecule() {
        let mut mol = Molecule::new();
        let desc = Morgan::ecfp4();
        let fp = desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_binary().clone();
        assert_eq!(count_bits(&fp), 0);
    }

    #[test]
    /// Runs regression test for benzene symmetry.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_benzene_symmetry() {
        let mol = parse_smiles("c1ccccc1").unwrap();
        let features = morgan_features(&mol, 2);
        let unique: std::collections::HashSet<u64> = features.iter().copied().collect();
        assert!(
            unique.len() <= 10,
            "Benzene should have a small unique feature set, got {}",
            unique.len()
        );
    }

    #[test]
    /// Runs regression test for identical molecules same fp.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_identical_molecules_same_fp() {
        let fp1 = calc_fp("CCO", 2, 2048);
        let fp2 = calc_fp("CCO", 2, 2048);
        assert_eq!(fp1, fp2, "Same SMILES should produce same fingerprint");
    }

    #[test]
    /// Runs regression test for tanimoto identical.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_tanimoto_identical() {
        let fp = calc_fp("CCO", 2, 2048);
        let sim = tanimoto_similarity(&fp, &fp);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Self-similarity should be 1.0, got {}",
            sim
        );
    }

    #[test]
    /// Runs regression test for tanimoto similar molecules.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_tanimoto_similar_molecules() {
        let fp1 = calc_fp("c1ccc(O)cc1", 2, 2048);
        let fp2 = calc_fp("Cc1ccc(O)cc1", 2, 2048);
        let sim = tanimoto_similarity(&fp1, &fp2);
        assert!(
            sim > 0.3,
            "Similar molecules should have Tanimoto > 0.3, got {}",
            sim
        );
    }

    #[test]
    /// Runs regression test for tanimoto dissimilar molecules.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_tanimoto_dissimilar_molecules() {
        let fp1 = calc_fp("c1ccccc1", 2, 2048);
        let fp2 = calc_fp("CCCCCCCCCCCC", 2, 2048);
        let sim = tanimoto_similarity(&fp1, &fp2);
        assert!(
            sim < 0.8,
            "Dissimilar molecules should have lower Tanimoto, got {}",
            sim
        );
    }

    #[test]
    /// Runs regression test for higher radius more bits.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_higher_radius_more_bits() {
        let fp2 = calc_fp("c1ccc(CC(=O)O)cc1", 1, 2048);
        let fp4 = calc_fp("c1ccc(CC(=O)O)cc1", 2, 2048);
        let fp6 = calc_fp("c1ccc(CC(=O)O)cc1", 3, 2048);
        assert!(
            count_bits(&fp4) >= count_bits(&fp2),
            "ECFP4 should have >= bits than ECFP2"
        );
        assert!(
            count_bits(&fp6) >= count_bits(&fp4),
            "ECFP6 should have >= bits than ECFP4"
        );
    }

    #[test]
    /// Runs regression test for different bit sizes.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_different_bit_sizes() {
        let mut mol = parse_smiles("c1ccc(CC(=O)O)cc1").unwrap();
        for bits in [1024, 2048, 4096] {
            let desc = Morgan::new(2, bits);
            let fp = desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_binary().clone();
            assert_eq!(fp.len(), bits);
            assert!(count_bits(&fp) > 0);
        }
    }

    #[test]
    /// Runs regression test for isomers different fps.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_isomers_different_fps() {
        let fp1 = calc_fp("CCCCO", 2, 2048);
        let fp2 = calc_fp("CCOCC", 2, 2048);
        let sim = tanimoto_similarity(&fp1, &fp2);
        assert!(
            sim < 1.0,
            "Isomers should not be identical, Tanimoto={}",
            sim
        );
    }

    #[test]
    /// Runs regression test for tanimoto empty.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_tanimoto_empty() {
        let fp1 = bitvec![u8, Lsb0; 0; 2048];
        let fp2 = bitvec![u8, Lsb0; 0; 2048];
        assert_eq!(tanimoto_similarity(&fp1, &fp2), 1.0);
    }

    #[test]
    /// Runs regression test for aspirin fp.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_aspirin_fp() {
        let fp = calc_fp("CC(=O)Oc1ccccc1C(=O)O", 2, 2048);
        let bits = count_bits(&fp);
        assert!(
            bits > 5 && bits < 200,
            "Aspirin should have reasonable bit count: {}",
            bits
        );
    }

    #[test]
    /// Runs regression test for caffeine fp.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_caffeine_fp() {
        let fp = calc_fp("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 2, 2048);
        let bits = count_bits(&fp);
        assert!(bits > 5 && bits < 200);
    }

    // ========== FCFP Tests ==========

    /// Implements fcfp.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_fcfp(smiles: &str, radius: u8, bits: usize) -> BitVec<u8, Lsb0> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        let desc = FCFP::new(radius, bits);
        desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_binary().clone()
    }

    #[test]
    /// Runs regression test for fcfp has bits.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fcfp_has_bits() {
        let fp = calc_fcfp("c1ccccc1", 2, 2048);
        assert!(count_bits(&fp) > 0, "Benzene FCFP4 should set bits");
    }

    #[test]
    /// Runs regression test for fcfp differs from ecfp.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fcfp_differs_from_ecfp() {
        let fp_ecfp = calc_fp("CCO", 2, 2048);
        let fp_fcfp = calc_fcfp("CCO", 2, 2048);
        assert_ne!(fp_ecfp, fp_fcfp, "FCFP should differ from ECFP");
    }

    #[test]
    /// Runs regression test for fcfp pharmacophore similarity.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fcfp_pharmacophore_similarity() {
        let ecfp_phenol = calc_fp("c1ccc(O)cc1", 2, 2048);
        let ecfp_thiophenol = calc_fp("c1ccc(S)cc1", 2, 2048);
        let fcfp_phenol = calc_fcfp("c1ccc(O)cc1", 2, 2048);
        let fcfp_thiophenol = calc_fcfp("c1ccc(S)cc1", 2, 2048);

        let ecfp_sim = tanimoto_similarity(&ecfp_phenol, &ecfp_thiophenol);
        let fcfp_sim = tanimoto_similarity(&fcfp_phenol, &fcfp_thiophenol);
        assert!(
            fcfp_sim >= ecfp_sim,
            "FCFP should see pharmacophore analogs as more similar: FCFP={} vs ECFP={}",
            fcfp_sim,
            ecfp_sim
        );
    }

    #[test]
    /// Runs regression test for fcfp empty molecule.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fcfp_empty_molecule() {
        let mut mol = Molecule::new();
        let desc = FCFP::fcfp4();
        let fp = desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_binary().clone();
        assert_eq!(count_bits(&fp), 0);
    }

    #[test]
    /// Runs regression test for fcfp identical molecules.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fcfp_identical_molecules() {
        let fp1 = calc_fcfp("CCO", 2, 2048);
        let fp2 = calc_fcfp("CCO", 2, 2048);
        assert_eq!(fp1, fp2);
    }

    #[test]
    /// Runs regression test for fcfp self tanimoto.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fcfp_self_tanimoto() {
        let fp = calc_fcfp("c1ccc(CC(=O)O)cc1", 2, 2048);
        let sim = tanimoto_similarity(&fp, &fp);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    /// Runs regression test for fcfp radius variants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fcfp_radius_variants() {
        let fp2 = calc_fcfp("c1ccc(CC(=O)O)cc1", 1, 2048);
        let fp4 = calc_fcfp("c1ccc(CC(=O)O)cc1", 2, 2048);
        let fp6 = calc_fcfp("c1ccc(CC(=O)O)cc1", 3, 2048);
        assert!(count_bits(&fp4) >= count_bits(&fp2), "FCFP4 >= FCFP2 bits");
        assert!(count_bits(&fp6) >= count_bits(&fp4), "FCFP6 >= FCFP4 bits");
    }

    #[test]
    /// Runs regression test for fcfp drug molecule.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fcfp_drug_molecule() {
        let fp = calc_fcfp("CC(=O)Oc1ccccc1C(=O)O", 2, 2048);
        let bits = count_bits(&fp);
        assert!(bits > 3 && bits < 200);
    }

    #[test]
    /// Runs regression test for fcfp constructors.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fcfp_constructors() {
        let f2 = FCFP::fcfp2();
        assert_eq!(f2.radius, 1);
        assert_eq!(f2.num_bits, 2048);

        let f4 = FCFP::fcfp4();
        assert_eq!(f4.radius, 2);

        let f6 = FCFP::fcfp6();
        assert_eq!(f6.radius, 3);
    }

    #[test]
    /// Runs regression test for fcfp name.
    ///
    /// Function context: `crates/ferrum_core/src/desc/fingerprint.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fcfp_name() {
        let f = FCFP::fcfp4();
        assert_eq!(Descriptor::name(&f), "FCFP4");
    }
}
