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

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::BondType;
use bitvec::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Compute a hash from a sequence of u64 values.
fn combine_hash(values: &[u64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for v in values {
        v.hash(&mut hasher);
    }
    hasher.finish()
}

/// Compute initial atom invariant hash (ECFP-style).
///
/// Uses: atomic_number, degree, total_h, formal_charge, is_aromatic, is_in_ring
fn atom_invariant_ecfp(mol: &Molecule, atom_idx: usize) -> u64 {
    let atom = &mol.atoms[atom_idx];
    let degree = mol.get_degree(atom_idx as u16) as u64;
    combine_hash(&[
        atom.atomic_number as u64,
        degree,
        atom.total_h() as u64,
        (atom.formal_charge as i64 + 128) as u64, // shift to avoid negative hash issues
        atom.is_aromatic() as u64,
        atom.is_in_ring() as u64,
    ])
}

/// Pharmacophoric feature flags for FCFP.
///
/// Each atom is classified by functional role rather than atomic identity:
/// - HBond Donor: N-H, O-H
/// - HBond Acceptor: N, O (with lone pairs)
/// - Positive Ionizable: basic amines
/// - Negative Ionizable: carboxylic acids, etc.
/// - Aromatic: aromatic atoms
/// - Hydrophobic: non-polar carbons, halogens (Cl, Br, I, but not F)
fn atom_invariant_fcfp(mol: &Molecule, atom_idx: usize) -> u64 {
    let atom = &mol.atoms[atom_idx];
    let an = atom.atomic_number;
    let total_h = atom.total_h();
    let charge = atom.formal_charge;

    // HBond Donor: N or O with at least one H
    let is_donor = (an == 7 || an == 8) && total_h > 0;

    // HBond Acceptor: N or O (standard definition)
    let is_acceptor = an == 7 || an == 8;

    // Positive Ionizable: N with positive charge, or basic amine (N with H, not amide-like)
    let is_pos_ionizable =
        (an == 7 && charge > 0) || (an == 7 && total_h > 0 && !atom.is_aromatic());

    // Negative Ionizable: O with negative charge, or acidic O-H
    let is_neg_ionizable = (an == 8 && charge < 0) || (an == 8 && total_h > 0);

    // Aromatic
    let is_aromatic = atom.is_aromatic();

    // Hydrophobic: C with no heteroatom neighbors, or Cl/Br/I
    let is_hydrophobic = if an == 6 {
        // Carbon: hydrophobic if all heavy neighbors are C
        mol.adjacency[atom_idx].iter().all(|&bond_idx| {
            let bond = &mol.bonds[bond_idx as usize];
            let neighbor = if bond.source as usize == atom_idx {
                bond.target as usize
            } else {
                bond.source as usize
            };
            mol.atoms[neighbor].atomic_number == 6
        })
    } else {
        // Cl, Br, I (but not F - too electronegative)
        an == 17 || an == 35 || an == 53
    };

    // Hash pharmacophoric features (not atomic identity)
    combine_hash(&[
        is_donor as u64,
        is_acceptor as u64,
        is_pos_ionizable as u64,
        is_neg_ionizable as u64,
        is_aromatic as u64,
        is_hydrophobic as u64,
    ])
}

/// Encode bond type as a u64 for hashing.
fn bond_type_hash(bt: BondType, is_aromatic: bool) -> u64 {
    if is_aromatic {
        return 4; // aromatic bond
    }
    match bt {
        BondType::Single => 1,
        BondType::Double => 2,
        BondType::Triple => 3,
        BondType::Aromatic => 4,
        BondType::Dative => 5,
        BondType::Unknown => 0,
    }
}

/// Core Morgan fingerprint computation with pluggable atom invariants.
///
/// Returns a set of unique feature hashes collected across all radii.
/// The `init_fn` determines ECFP vs FCFP behavior.
fn morgan_features_with<F>(mol: &Molecule, radius: u32, init_fn: F) -> Vec<u64>
where
    F: Fn(&Molecule, usize) -> u64,
{
    let n = mol.num_atoms();
    if n == 0 {
        return vec![];
    }

    // Initialize with atom invariants (radius 0)
    let mut current_hashes: Vec<u64> = (0..n).map(|i| init_fn(mol, i)).collect();

    // Collect features: radius 0 features
    let mut all_features: Vec<u64> = Vec::with_capacity(n * (radius as usize + 1));
    // Tag radius-0 hashes to distinguish from higher radii
    for &h in &current_hashes {
        all_features.push(combine_hash(&[0, h]));
    }

    // Iterative expansion
    for r in 1..=radius {
        let mut new_hashes = vec![0u64; n];

        for atom_idx in 0..n {
            // Collect (bond_hash, neighbor_hash) pairs and sort for order independence
            let mut neighbor_info: Vec<(u64, u64)> = Vec::new();

            for &bond_idx in &mol.adjacency[atom_idx] {
                let bond = &mol.bonds[bond_idx as usize];
                let neighbor = if bond.source as usize == atom_idx {
                    bond.target as usize
                } else {
                    bond.source as usize
                };

                let bh = bond_type_hash(bond.bond_type, bond.is_aromatic());
                neighbor_info.push((bh, current_hashes[neighbor]));
            }

            // Sort for canonical ordering (bond type first, then neighbor hash)
            neighbor_info.sort();

            // Build hash: [radius, current_hash, bond1, neigh1, bond2, neigh2, ...]
            let mut hash_input: Vec<u64> = Vec::with_capacity(2 + neighbor_info.len() * 2);
            hash_input.push(r as u64);
            hash_input.push(current_hashes[atom_idx]);
            for (bh, nh) in &neighbor_info {
                hash_input.push(*bh);
                hash_input.push(*nh);
            }

            new_hashes[atom_idx] = combine_hash(&hash_input);
        }

        // Collect features at this radius
        for &h in &new_hashes {
            all_features.push(h);
        }

        current_hashes = new_hashes;
    }

    // Deduplicate features (same environment at different atoms = one bit)
    all_features.sort();
    all_features.dedup();
    all_features
}

/// ECFP features: Morgan with atomic invariants.
fn morgan_features(mol: &Molecule, radius: u32) -> Vec<u64> {
    morgan_features_with(mol, radius, atom_invariant_ecfp)
}

/// FCFP features: Morgan with pharmacophoric invariants.
fn fcfp_features(mol: &Molecule, radius: u32) -> Vec<u64> {
    morgan_features_with(mol, radius, atom_invariant_fcfp)
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
    fn name(&self) -> &str {
        match self.radius {
            1 => "ECFP2",
            2 => "ECFP4",
            3 => "ECFP6",
            _ => "ECFP",
        }
    }

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
    fn name(&self) -> &str {
        match self.radius {
            1 => "FCFP2",
            2 => "FCFP4",
            3 => "FCFP6",
            _ => "FCFP",
        }
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let features = fcfp_features(mol, self.radius as u32);
        let bits = fold_fingerprint(&features, self.num_bits);
        DescriptorValue::Binary(bits)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ECFP>;
    let _ = assert_send_sync::<FCFP>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_fp(smiles: &str, radius: u8, bits: usize) -> BitVec<u8, Lsb0> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        let desc = ECFP::new(radius, bits);
        desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_binary().clone()
    }

    fn count_bits(fp: &BitVec<u8, Lsb0>) -> usize {
        fp.count_ones()
    }

    #[test]
    fn test_methane_has_bits() {
        let fp = calc_fp("C", 2, 2048);
        let bits = count_bits(&fp);
        assert!(bits > 0, "Methane should set some bits");
        assert!(bits < 50, "Methane should set few bits, got {}", bits);
    }

    #[test]
    fn test_benzene_has_bits() {
        let fp = calc_fp("c1ccccc1", 2, 2048);
        let bits = count_bits(&fp);
        assert!(bits > 0, "Benzene should set bits");
    }

    #[test]
    fn test_empty_molecule() {
        let mut mol = Molecule::new();
        let desc = Morgan::ecfp4();
        let fp = desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_binary().clone();
        assert_eq!(count_bits(&fp), 0);
    }

    #[test]
    fn test_benzene_symmetry() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        let features = morgan_features(&mol, 2);
        assert!(
            features.len() <= 10,
            "Benzene has high symmetry, got {} features",
            features.len()
        );
    }

    #[test]
    fn test_identical_molecules_same_fp() {
        let fp1 = calc_fp("CCO", 2, 2048);
        let fp2 = calc_fp("CCO", 2, 2048);
        assert_eq!(fp1, fp2, "Same SMILES should produce same fingerprint");
    }

    #[test]
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
    fn test_tanimoto_empty() {
        let fp1 = bitvec![u8, Lsb0; 0; 2048];
        let fp2 = bitvec![u8, Lsb0; 0; 2048];
        assert_eq!(tanimoto_similarity(&fp1, &fp2), 1.0);
    }

    #[test]
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
    fn test_caffeine_fp() {
        let fp = calc_fp("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 2, 2048);
        let bits = count_bits(&fp);
        assert!(bits > 5 && bits < 200);
    }

    // ========== FCFP Tests ==========

    fn calc_fcfp(smiles: &str, radius: u8, bits: usize) -> BitVec<u8, Lsb0> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        let desc = FCFP::new(radius, bits);
        desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_binary().clone()
    }

    #[test]
    fn test_fcfp_has_bits() {
        let fp = calc_fcfp("c1ccccc1", 2, 2048);
        assert!(count_bits(&fp) > 0, "Benzene FCFP4 should set bits");
    }

    #[test]
    fn test_fcfp_differs_from_ecfp() {
        let fp_ecfp = calc_fp("CCO", 2, 2048);
        let fp_fcfp = calc_fcfp("CCO", 2, 2048);
        assert_ne!(fp_ecfp, fp_fcfp, "FCFP should differ from ECFP");
    }

    #[test]
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
    fn test_fcfp_empty_molecule() {
        let mut mol = Molecule::new();
        let desc = FCFP::fcfp4();
        let fp = desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_binary().clone();
        assert_eq!(count_bits(&fp), 0);
    }

    #[test]
    fn test_fcfp_identical_molecules() {
        let fp1 = calc_fcfp("CCO", 2, 2048);
        let fp2 = calc_fcfp("CCO", 2, 2048);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fcfp_self_tanimoto() {
        let fp = calc_fcfp("c1ccc(CC(=O)O)cc1", 2, 2048);
        let sim = tanimoto_similarity(&fp, &fp);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_fcfp_radius_variants() {
        let fp2 = calc_fcfp("c1ccc(CC(=O)O)cc1", 1, 2048);
        let fp4 = calc_fcfp("c1ccc(CC(=O)O)cc1", 2, 2048);
        let fp6 = calc_fcfp("c1ccc(CC(=O)O)cc1", 3, 2048);
        assert!(count_bits(&fp4) >= count_bits(&fp2), "FCFP4 >= FCFP2 bits");
        assert!(count_bits(&fp6) >= count_bits(&fp4), "FCFP6 >= FCFP4 bits");
    }

    #[test]
    fn test_fcfp_drug_molecule() {
        let fp = calc_fcfp("CC(=O)Oc1ccccc1C(=O)O", 2, 2048);
        let bits = count_bits(&fp);
        assert!(bits > 3 && bits < 200);
    }

    #[test]
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
    fn test_fcfp_name() {
        let f = FCFP::fcfp4();
        assert_eq!(Descriptor::name(&f), "FCFP4");
    }
}
