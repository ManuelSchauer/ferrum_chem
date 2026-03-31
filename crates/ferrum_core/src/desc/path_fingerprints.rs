//! Atom Pair and Topological Torsion fingerprints.
//!
//! RDKit-aligned hashed path fingerprints with default count simulation
//! (`nBitsPerEntry=4`, thresholds `[1,2,4,8]`) for parity with:
//! - `GetHashedAtomPairFingerprintAsBitVect()`
//! - `GetHashedTopologicalTorsionFingerprintAsBitVect()`


//! ## File guide: Path Fingerprints
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
//! - `fn bond_neighbor`
//! - `fn bond_between_atoms`
//! - `fn is_effectively_sp3`
//! - `fn num_pi_electrons`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - Atom-pair and torsion encodings assume shortest-path and ordered-atom canonicalization are deterministic.
//! - Bit mapping expects fixed-size folded spaces where collisions are acceptable but reproducible.
//! - Topological paths are computed on bond graphs only; stereochemical 3D torsion geometry is out-of-scope.
//! ### Chemistry/ML references
//! - Carhart et al. (1985). Atom pair molecular representations.
//! - Nilakantan et al. (1987). Topological torsion descriptors.
//! - RDKit AtomPair/TopologicalTorsion behavior as external parity reference.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType, Hybridization};
use bitvec::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

const NUM_BITS_PER_ENTRY: usize = 4;
const COUNT_BOUNDS: [u32; NUM_BITS_PER_ENTRY] = [1, 2, 4, 8];
const MAX_PAIR_DISTANCE: u32 = 30; // RDKit default: maxPathLen-1 with numPathBits=5.
const TT_ATOM_COUNT: usize = 4;

// RDKit FingerprintUtil.h constants for AtomPair/Torsion atom coding.
const ATOM_NUMBER_TYPES: [u8; 15] = [5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 51, 52, 53];
const NUM_TYPE_BITS: u32 = 4;
const NUM_PI_BITS: u32 = 2;
const NUM_BRANCH_BITS: u32 = 3;
const MAX_NUM_PI: u32 = (1u32 << NUM_PI_BITS) - 1; // 3
const MAX_NUM_BRANCHES: u32 = (1u32 << NUM_BRANCH_BITS) - 1; // 7
const CODE_SIZE: u32 = NUM_TYPE_BITS + NUM_PI_BITS + NUM_BRANCH_BITS; // 9
const CODE_SIZE_LIMIT: u32 = (1u32 << CODE_SIZE) - 1; // 511

#[inline]
/// Implements hash combine.
///
/// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn rdkit_hash_combine(seed: &mut u32, value: u32) {
    *seed ^= value
        .wrapping_add(0x9e37_79b9)
        .wrapping_add(seed.wrapping_shl(6))
        .wrapping_add(seed.wrapping_shr(2));
}

#[inline]
/// Implements neighbor.
///
/// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn bond_neighbor(mol: &Molecule, atom_idx: usize, bond_idx: usize) -> usize {
    let bond = &mol.bonds[bond_idx];
    if bond.source as usize == atom_idx {
        bond.target as usize
    } else {
        bond.source as usize
    }
}

/// Implements between atoms.
///
/// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn bond_between_atoms(mol: &Molecule, a: usize, b: usize) -> Option<usize> {
    for &bond_idx_u16 in &mol.adjacency[a] {
        let bond_idx = bond_idx_u16 as usize;
        if bond_neighbor(mol, a, bond_idx) == b {
            return Some(bond_idx);
        }
    }
    None
}

/// Checks whether effectively sp3.
///
/// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn is_effectively_sp3(mol: &Molecule, idx: usize) -> bool {
    let atom = &mol.atoms[idx];
    if atom.is_aromatic() {
        return false;
    }

    if atom.hybridization == Hybridization::SP3 {
        return true;
    }
    if atom.hybridization != Hybridization::Unspecified {
        return false;
    }

    !mol.adjacency[idx].iter().any(|&bond_idx_u16| {
        matches!(
            mol.bonds[bond_idx_u16 as usize].bond_type,
            BondType::Double | BondType::Triple | BondType::Aromatic
        )
    })
}

/// Implements pi electrons.
///
/// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn num_pi_electrons(mol: &Molecule, idx: usize) -> u32 {
    let atom = &mol.atoms[idx];
    if atom.is_aromatic() {
        return 1;
    }
    if is_effectively_sp3(mol, idx) {
        return 0;
    }

    let mut pi = 0u32;
    for &bond_idx_u16 in &mol.adjacency[idx] {
        let bond = &mol.bonds[bond_idx_u16 as usize];
        pi += match bond.bond_type {
            BondType::Double => 1,
            BondType::Triple => 2,
            BondType::Aromatic => 1,
            _ => 0,
        };
    }
    pi
}

/// RDKit AtomPairs::getAtomCode equivalent (without chirality bits).
fn atom_code(mol: &Molecule, idx: usize, branch_subtract: u32) -> u32 {
    let degree = mol.get_degree(idx as AtomIdx) as u32;
    let num_branches = degree.saturating_sub(branch_subtract);
    let mut code = num_branches % MAX_NUM_BRANCHES;

    let n_pi = num_pi_electrons(mol, idx) % MAX_NUM_PI;
    code |= n_pi << NUM_BRANCH_BITS;

    let atomic_num = mol.atoms[idx].atomic_number;
    let mut type_idx = 0usize;
    let n_types = 1usize << NUM_TYPE_BITS; // 16
    while type_idx < n_types {
        let probe = ATOM_NUMBER_TYPES.get(type_idx).copied().unwrap_or(0);
        if probe == atomic_num {
            break;
        } else if probe > atomic_num {
            type_idx = n_types;
            break;
        }
        type_idx += 1;
    }
    if type_idx == n_types {
        type_idx -= 1;
    }
    code |= (type_idx as u32) << (NUM_BRANCH_BITS + NUM_PI_BITS);

    code
}

/// Builds unweighted dist matrix.
///
/// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn build_unweighted_dist_matrix(mol: &Molecule) -> Vec<u8> {
    let n = mol.num_atoms();
    let mut distances = vec![u8::MAX; n * n];
    if n == 0 {
        return distances;
    }

    for start in 0..n {
        distances[start * n + start] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(start as AtomIdx);

        while let Some(current) = queue.pop_front() {
            let current_usize = current as usize;
            let current_dist = distances[start * n + current_usize];
            for &bond_idx in &mol.adjacency[current_usize] {
                let bond = &mol.bonds[bond_idx as usize];
                let neighbor = if bond.source == current {
                    bond.target
                } else {
                    bond.source
                } as usize;
                let slot = &mut distances[start * n + neighbor];
                if *slot == u8::MAX {
                    *slot = current_dist.saturating_add(1);
                    queue.push_back(neighbor as AtomIdx);
                }
            }
        }
    }

    distances
}

/// Implements dist matrix.
///
/// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn unweighted_dist_matrix(mol: &mut Molecule) -> Vec<u8> {
    if let Some(cached) = mol.cached_unweighted_dist_matrix() {
        return cached.to_vec();
    }
    let matrix = build_unweighted_dist_matrix(mol);
    mol.store_unweighted_dist_matrix(matrix.clone());
    matrix
}

/// Implements count simulated hashes.
///
/// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn fold_count_simulated_hashes(hashes: &[u32], num_bits: usize) -> BitVec<u8, Lsb0> {
    let mut bits = bitvec![u8, Lsb0; 0; num_bits];
    if num_bits == 0 {
        return bits;
    }

    let block_len = num_bits / NUM_BITS_PER_ENTRY;
    if block_len == 0 {
        return bits;
    }

    let mut counts: HashMap<usize, u32> = HashMap::new();
    for &hash in hashes {
        let bit_id = (hash as usize) % block_len;
        *counts.entry(bit_id).or_insert(0) += 1;
    }

    for (bit_id, count) in counts {
        let offset = bit_id * NUM_BITS_PER_ENTRY;
        for (i, bound) in COUNT_BOUNDS.iter().enumerate() {
            if count >= *bound {
                let idx = offset + i;
                if idx < num_bits {
                    bits.set(idx, true);
                }
            }
        }
    }

    bits
}

/// Implements pair hashes.
///
/// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn atom_pair_hashes(mol: &mut Molecule) -> Vec<u32> {
    let n = mol.num_atoms();
    if n < 2 {
        return vec![];
    }

    let distances = unweighted_dist_matrix(mol);
    let atom_invariants: Vec<u32> = (0..n).map(|i| atom_code(mol, i, 0)).collect();
    let mut hashes = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = distances[i * n + j];
            if dist == u8::MAX {
                continue;
            }
            let d = dist as u32;
            if d == 0 || d > MAX_PAIR_DISTANCE {
                continue;
            }

            let code_i = atom_invariants[i] % CODE_SIZE_LIMIT;
            let code_j = atom_invariants[j] % CODE_SIZE_LIMIT;
            let (lo, hi) = if code_i <= code_j {
                (code_i, code_j)
            } else {
                (code_j, code_i)
            };

            let mut hash = 0u32;
            rdkit_hash_combine(&mut hash, lo);
            rdkit_hash_combine(&mut hash, d);
            rdkit_hash_combine(&mut hash, hi);
            hashes.push(hash);
        }
    }

    hashes
}

/// Implements paths len4.
///
/// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn dfs_paths_len4(mol: &Molecule, path: &mut Vec<usize>, out: &mut Vec<[usize; 4]>) {
    if path.len() == TT_ATOM_COUNT {
        out.push([path[0], path[1], path[2], path[3]]);
        return;
    }

    let current = *path.last().unwrap();
    for &bond_idx_u16 in &mol.adjacency[current] {
        let bond_idx = bond_idx_u16 as usize;
        let next = bond_neighbor(mol, current, bond_idx);

        if let Some(existing_pos) = path.iter().position(|&a| a == next) {
            // Match RDKit path-extension behavior for ring closure candidates:
            // only allow a repeated atom at final extension and not immediate backtrack.
            if path.len() == TT_ATOM_COUNT - 1
                && existing_pos == 0
                && path[path.len() - 2] != next
            {
                path.push(next);
                out.push([path[0], path[1], path[2], path[3]]);
                path.pop();
            }
            continue;
        }

        path.push(next);
        dfs_paths_len4(mol, path, out);
        path.pop();
    }
}

/// Implements torsion paths.
///
/// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn unique_torsion_paths(mol: &Molecule) -> Vec<[usize; 4]> {
    let n = mol.num_atoms();
    if n < TT_ATOM_COUNT {
        return vec![];
    }

    let mut all_paths = Vec::new();
    let mut path = Vec::with_capacity(TT_ATOM_COUNT);
    for start in 0..n {
        path.clear();
        path.push(start);
        dfs_paths_len4(mol, &mut path, &mut all_paths);
    }

    // Deduplicate by bond-set signature (RDKit path uniqueness criterion).
    let mut seen_bond_sets: HashSet<[usize; TT_ATOM_COUNT - 1]> = HashSet::new();
    let mut unique = Vec::new();
    for p in all_paths {
        let mut sig = [0usize; TT_ATOM_COUNT - 1];
        let mut ok = true;
        for k in 0..(TT_ATOM_COUNT - 1) {
            match bond_between_atoms(mol, p[k], p[k + 1]) {
                Some(bond_idx) => sig[k] = bond_idx,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        sig.sort_unstable();
        if seen_bond_sets.insert(sig) {
            unique.push(p);
        }
    }

    unique
}

/// Implements hashes.
///
/// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn torsion_hashes(mol: &mut Molecule) -> Vec<u32> {
    let n = mol.num_atoms();
    if n < TT_ATOM_COUNT {
        return vec![];
    }

    // TopologicalTorsion uses AtomPair invariants with correction (subtract 2).
    let atom_invariants: Vec<u32> = (0..n).map(|i| atom_code(mol, i, 0).wrapping_sub(2)).collect();
    let paths = unique_torsion_paths(mol);
    let mut hashes = Vec::with_capacity(paths.len());

    for path in paths {
        let mut path_codes = [0u32; TT_ATOM_COUNT];
        for (pos, &atom_idx) in path.iter().enumerate() {
            let mut code = atom_invariants[atom_idx] % CODE_SIZE_LIMIT + 1;
            // Interior atom correction in TT path encoding.
            if pos != 0 && pos + 1 != TT_ATOM_COUNT {
                code = code.saturating_sub(1);
            }
            path_codes[pos] = code;
        }

        // Canonicalize path direction before hashing.
        let mut reverse = false;
        let mut i = 0usize;
        let mut j = TT_ATOM_COUNT - 1;
        while i < j {
            if path_codes[i] > path_codes[j] {
                reverse = true;
                break;
            } else if path_codes[i] < path_codes[j] {
                break;
            }
            i += 1;
            j -= 1;
        }

        let mut hash = 0u32;
        if reverse {
            for idx in (0..TT_ATOM_COUNT).rev() {
                rdkit_hash_combine(&mut hash, path_codes[idx]);
            }
        } else {
            for &code in &path_codes {
                rdkit_hash_combine(&mut hash, code);
            }
        }
        hashes.push(hash);
    }

    hashes
}

/// Atom Pair Fingerprint descriptor.
///
/// Uses RDKit-style atom coding and count simulation defaults.
pub struct AtomPairFingerprint {
    pub num_bits: usize,
}

impl AtomPairFingerprint {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn new(num_bits: usize) -> Self {
        Self { num_bits }
    }

    /// Default: 2048 bits.
    pub fn default_size() -> Self {
        Self { num_bits: 2048 }
    }

    /// Raw hashed atom-pair environment identifiers (pre-folding).
    pub fn raw_features(&self, mol: &Molecule) -> Vec<u64> {
        let mut clone = mol.clone();
        atom_pair_hashes(&mut clone)
            .into_iter()
            .map(|v| v as u64)
            .collect()
    }
}

impl Descriptor for AtomPairFingerprint {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "AtomPair"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let hashes = atom_pair_hashes(mol);
        let bits = fold_count_simulated_hashes(&hashes, self.num_bits);
        DescriptorValue::Binary(bits)
    }
}

/// Topological Torsion Fingerprint descriptor.
///
/// Uses RDKit-style atom coding/path canonicalization and count simulation defaults.
pub struct TopologicalTorsionFingerprint {
    pub num_bits: usize,
}

impl TopologicalTorsionFingerprint {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn new(num_bits: usize) -> Self {
        Self { num_bits }
    }

    /// Implements size.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn default_size() -> Self {
        Self { num_bits: 2048 }
    }

    /// Raw hashed torsion environment identifiers (pre-folding).
    pub fn raw_features(&self, mol: &Molecule) -> Vec<u64> {
        let mut clone = mol.clone();
        torsion_hashes(&mut clone)
            .into_iter()
            .map(|v| v as u64)
            .collect()
    }
}

impl Descriptor for TopologicalTorsionFingerprint {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "TopologicalTorsion"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let hashes = torsion_hashes(mol);
        let bits = fold_count_simulated_hashes(&hashes, self.num_bits);
        DescriptorValue::Binary(bits)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<AtomPairFingerprint>;
    let _ = assert_send_sync::<TopologicalTorsionFingerprint>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::fingerprint::tanimoto_similarity;
    use crate::io::smiles::parse_smiles;

    /// Counts bits.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn count_bits(fp: &BitVec<u8, Lsb0>) -> usize {
        fp.count_ones()
    }

    // ========== Atom Pair Tests ==========

    /// Implements ap.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_ap(smiles: &str, bits: usize) -> BitVec<u8, Lsb0> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        let ap = AtomPairFingerprint::new(bits);
        ap.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        )
        .as_binary()
        .clone()
    }

    #[test]
    /// Runs regression test for ap methane no pairs.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ap_methane_no_pairs() {
        let fp = calc_ap("C", 2048);
        assert_eq!(count_bits(&fp), 0);
    }

    #[test]
    /// Runs regression test for ap ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ap_ethane() {
        let fp = calc_ap("CC", 2048);
        assert!(
            count_bits(&fp) > 0,
            "Ethane should have at least 1 atom pair"
        );
    }

    #[test]
    /// Runs regression test for ap benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ap_benzene() {
        let fp = calc_ap("c1ccccc1", 2048);
        let bits = count_bits(&fp);
        assert!(bits > 0 && bits < 200);
    }

    #[test]
    /// Runs regression test for ap identical.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ap_identical() {
        let fp1 = calc_ap("CCO", 2048);
        let fp2 = calc_ap("CCO", 2048);
        assert_eq!(fp1, fp2);
    }

    #[test]
    /// Runs regression test for ap self tanimoto.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ap_self_tanimoto() {
        let fp = calc_ap("c1ccc(CC(=O)O)cc1", 2048);
        let sim = tanimoto_similarity(&fp, &fp);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    /// Runs regression test for ap similar molecules.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ap_similar_molecules() {
        let fp1 = calc_ap("c1ccc(O)cc1", 2048);
        let fp2 = calc_ap("Cc1ccc(O)cc1", 2048);
        let sim = tanimoto_similarity(&fp1, &fp2);
        assert!(
            sim > 0.2,
            "Similar molecules should have some AP overlap: {}",
            sim
        );
    }

    #[test]
    /// Runs regression test for ap larger molecules more bits.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ap_larger_molecules_more_bits() {
        let fp_small = calc_ap("CC", 2048);
        let fp_large = calc_ap("CCCCCCCCCC", 2048);
        assert!(count_bits(&fp_large) >= count_bits(&fp_small));
    }

    #[test]
    /// Runs regression test for ap drug molecule.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_ap_drug_molecule() {
        let fp = calc_ap("CC(=O)Oc1ccccc1C(=O)O", 2048);
        let bits = count_bits(&fp);
        assert!(bits > 5 && bits < 800);
    }

    // ========== Topological Torsion Tests ==========

    /// Implements tt.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_tt(smiles: &str, bits: usize) -> BitVec<u8, Lsb0> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        let tt = TopologicalTorsionFingerprint::new(bits);
        tt.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        )
        .as_binary()
        .clone()
    }

    #[test]
    /// Runs regression test for tt small molecules no paths.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_tt_small_molecules_no_paths() {
        assert_eq!(count_bits(&calc_tt("C", 2048)), 0);
        assert_eq!(count_bits(&calc_tt("CC", 2048)), 0);
        assert_eq!(count_bits(&calc_tt("CCC", 2048)), 0);
    }

    #[test]
    /// Runs regression test for tt butane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_tt_butane() {
        let fp = calc_tt("CCCC", 2048);
        assert!(count_bits(&fp) > 0, "Butane has one 4-atom path");
    }

    #[test]
    /// Runs regression test for tt benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_tt_benzene() {
        let fp = calc_tt("c1ccccc1", 2048);
        assert!(count_bits(&fp) > 0);
    }

    #[test]
    /// Runs regression test for tt identical.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_tt_identical() {
        let fp1 = calc_tt("CCCCC", 2048);
        let fp2 = calc_tt("CCCCC", 2048);
        assert_eq!(fp1, fp2);
    }

    #[test]
    /// Runs regression test for tt self tanimoto.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_tt_self_tanimoto() {
        let fp = calc_tt("c1ccc(CC(=O)O)cc1", 2048);
        let sim = tanimoto_similarity(&fp, &fp);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    /// Runs regression test for tt different molecules.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_tt_different_molecules() {
        let fp1 = calc_tt("CCCCCC", 2048);
        let fp2 = calc_tt("c1ccccc1", 2048);
        assert_ne!(fp1, fp2, "Hexane and benzene should differ");
    }

    #[test]
    /// Runs regression test for tt drug molecule.
    ///
    /// Function context: `crates/ferrum_core/src/desc/path_fingerprints.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_tt_drug_molecule() {
        let fp = calc_tt("CC(=O)Oc1ccccc1C(=O)O", 2048);
        let bits = count_bits(&fp);
        assert!(bits > 3 && bits < 800);
    }
}
