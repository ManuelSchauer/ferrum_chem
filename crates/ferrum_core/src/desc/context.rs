//! Descriptor module: Context.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

use crate::mol::molecule::Molecule;
use smallvec::SmallVec;

#[derive(Default)]
/// Implements CalculationContext.
pub struct CalculationContext {
    pub active_molecule_key: Option<u64>,

    /// Topological distance matrix (row-major, `DIST_INF` = disconnected).
    pub dist: Vec<u16>,
    pub has_dist: bool,
    pub dist_bo: Vec<f64>,
    pub has_dist_bo: bool,

    // Distance matrix workspace
    pub dist_f64: Vec<f64>,
    pub row_f64: Vec<f64>,
    pub visited_bool: Vec<bool>,
    pub weighted_neighbors: Vec<SmallVec<[(usize, f64); 4]>>,

    pub bfs_queue: Vec<usize>,
    pub bfs_dist: Vec<usize>,

    // Bitset BFS workspace
    pub bfs_visited_u64: Vec<u64>,
    pub bfs_frontier_u64: Vec<u64>,
    pub bfs_next_frontier_u64: Vec<u64>,

    // Reusable float workspaces for matrix operations (e.g. Ipc)
    pub mat_a: Vec<f64>,
    pub mat_b: Vec<f64>,
    pub mat_c: Vec<f64>,
    pub mat_d: Vec<f64>,
    pub vec_a: Vec<f64>,
    pub vec_b: Vec<f64>,

    // BertzCT symmetry class keys
    pub bertz_keys: Vec<(SmallVec<[i32; 64]>, usize)>,

    // Morgan/ECFP/FCFP reusable scratch + cache
    pub morgan_next_layer_u32: Vec<u32>,
    pub morgan_dead_atoms: Vec<bool>,
    pub morgan_neighbor_invariants: Vec<(i32, u32)>,
    pub morgan_features: Vec<u64>,

    /// CDK atom-type view with `Aromaticity.cdkLegacy()` perceived, keyed by
    /// `molecule_cache_key` (shared by Autocorrelation, BCUT, APol, BPol).
    pub(crate) cdk_legacy_view: Option<(u64, crate::desc::cdk_perception::CdkView)>,
    // --- topological caches (keyed by `molecule_cache_key`) ---
    /// `|characteristic polynomial|` coefficients shared by Ipc and AvgIpc.
    pub char_poly_abs: Vec<f64>,
    pub char_poly_key: Option<u64>,
    /// Bond-order weighted distances in sixths (BalabanJ, BertzCT).
    pub bo_dist6: Vec<u32>,
    pub bo_dist6_key: Option<u64>,
    /// Legacy stereo perception (StereoCenters, SPS).
    pub legacy_stereo: Option<crate::algo::stereo::LegacyStereo>,
    pub legacy_stereo_key: Option<u64>,
    /// Atom paths of 1..=4 bonds (`algo::paths::cached_paths_of_length`;
    /// ChiIndices, ChiNIndices, KappaIndices).
    pub paths_by_len: Vec<Vec<crate::types::AtomIdx>>,
    pub paths_key: Option<u64>,
    /// E-State indices (EStateIndices, EStateAbs, EState_VSA, VSA_EState).
    pub estates: Vec<f64>,
    pub estates_key: Option<u64>,
    pub(crate) cdk_path_fps: Option<(u64, Box<crate::desc::cdk_path_fingerprints::CdkPathFps>)>,
    /// Prepared CDK SMARTS target and key counts of the CDK SMARTS-key
    /// fingerprints (CDKMACCS, CDKEState, CDKSubstructure, KlekotaRoth and
    /// their counts).
    pub(crate) cdk_keys: crate::desc::cdk_smarts::CdkKeyCache,
}

#[inline]
/// Creates a deterministic cache key for molecule-topology-dependent descriptor caches.
pub(crate) fn molecule_cache_key(mol: &Molecule) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0001_0000_01b3;

    #[inline]
    fn mix(mut hash: u64, value: u64) -> u64 {
        hash ^= value;
        hash.wrapping_mul(FNV_PRIME)
    }

    let mut hash = FNV_OFFSET;
    hash = mix(hash, mol.num_atoms() as u64);
    hash = mix(hash, mol.num_bonds() as u64);

    for atom in &mol.atoms {
        hash = mix(hash, atom.atomic_number as u64);
        hash = mix(hash, atom.formal_charge as i16 as u16 as u64);
        hash = mix(hash, atom.isotope as u64);
        hash = mix(hash, atom.num_explicit_h as u64);
        hash = mix(hash, atom.num_implicit_h as u64);
        hash = mix(hash, atom.flags as u64);
    }

    for bond in &mol.bonds {
        let a = bond.source.min(bond.target) as u64;
        let b = bond.source.max(bond.target) as u64;
        let bond_type = match bond.bond_type {
            crate::types::BondType::Unknown => 0u64,
            crate::types::BondType::Single => 1u64,
            crate::types::BondType::Double => 2u64,
            crate::types::BondType::Triple => 3u64,
            crate::types::BondType::Aromatic => 4u64,
            crate::types::BondType::Dative => 5u64,
        };
        hash = mix(hash, a);
        hash = mix(hash, b);
        hash = mix(hash, bond_type);
        hash = mix(hash, bond.flags as u64);
    }

    // `[H]n1cccc1` and `c1cc[nH]c1` differ only here (CDK keeps the H atom).
    for h in mol.removed_hydrogens() {
        let up = match h.up {
            None => 0u64,
            Some(true) => 1,
            Some(false) => 2,
        };
        hash = mix(hash, h.atom as u64 | (h.charge as u8 as u64) << 16 | up << 24 | 1 << 32);
    }

    hash
}

impl CalculationContext {
    pub fn new() -> Self {
        Self {
            active_molecule_key: None,
            dist: Vec::new(),
            has_dist: false,
            dist_bo: Vec::new(),
            has_dist_bo: false,
            dist_f64: Vec::new(),
            row_f64: Vec::new(),
            visited_bool: Vec::new(),
            weighted_neighbors: Vec::new(),
            bfs_queue: Vec::new(),
            bfs_dist: Vec::new(),
            mat_a: Vec::new(),
            mat_b: Vec::new(),
            mat_c: Vec::new(),
            mat_d: Vec::new(),
            vec_a: Vec::new(),
            vec_b: Vec::new(),
            bertz_keys: Vec::new(),
            morgan_next_layer_u32: Vec::new(),
            morgan_dead_atoms: Vec::new(),
            morgan_neighbor_invariants: Vec::new(),
            morgan_features: Vec::new(),
            bfs_visited_u64: Vec::new(),
            bfs_frontier_u64: Vec::new(),
            bfs_next_frontier_u64: Vec::new(),
            cdk_legacy_view: None,
            char_poly_abs: Vec::new(),
            char_poly_key: None,
            bo_dist6: Vec::new(),
            bo_dist6_key: None,
            legacy_stereo: None,
            legacy_stereo_key: None,
            paths_by_len: Vec::new(),
            paths_key: None,
            estates: Vec::new(),
            estates_key: None,
            cdk_path_fps: None,
            cdk_keys: Default::default(),
        }
    }

    pub fn reset_for_new_molecule(&mut self) {
        self.active_molecule_key = None;
        self.has_dist = false;
        self.has_dist_bo = false;
        self.bfs_queue.clear();
        self.bfs_dist.clear();
        self.bfs_visited_u64.clear();
        self.bfs_frontier_u64.clear();
        self.bfs_next_frontier_u64.clear();

        self.dist_f64.clear();
        self.row_f64.clear();
        self.visited_bool.clear();
        self.weighted_neighbors.clear();
        self.mat_a.clear();
        self.mat_b.clear();
        self.mat_c.clear();
        self.mat_d.clear();
        self.vec_a.clear();
        self.vec_b.clear();
        self.bertz_keys.clear();
        self.morgan_next_layer_u32.clear();
        self.morgan_dead_atoms.clear();
        self.morgan_neighbor_invariants.clear();
        self.morgan_features.clear();
        self.cdk_legacy_view = None;
        self.char_poly_key = None;
        self.bo_dist6_key = None;
        self.legacy_stereo_key = None;
        self.paths_key = None;
        self.estates_key = None;
        self.cdk_path_fps = None;
        self.cdk_keys.reset();
    }
}
