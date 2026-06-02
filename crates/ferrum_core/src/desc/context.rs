//! Descriptor module: Context.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

//! ## File guide: Context
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct CalculationContext`
//! - `fn new`
//! - `fn reset_for_new_molecule`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Context` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Context` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/context.rs` defines context behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Context` workflows.
//! - Repository unit and parity tests covering `Context` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/context.rs`.

use crate::mol::molecule::Molecule;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[derive(Default)]
/// Implements CalculationContext.
///
/// Function context: `crates/ferrum_core/src/desc/context.rs`.
/// Descriptor-layer note: computes molecular properties for ML pipelines.
pub struct CalculationContext {
    pub active_molecule_key: Option<u64>,

    pub dist_u8: Vec<u8>,
    pub has_dist_u8: bool,
    pub dist_u8_global_cache: FxHashMap<u64, Vec<u8>>,
    pub dist_bo: Vec<f64>,
    pub has_dist_bo: bool,

    // Distance matrix workspace
    pub dist_f64: Vec<f64>,
    pub row_f64: Vec<f64>,
    pub visited_bool: Vec<bool>,
    pub weighted_neighbors: Vec<SmallVec<[(usize, f64); 4]>>,

    pub bfs_queue: Vec<usize>,
    pub bfs_dist: Vec<usize>,

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
    pub morgan_features_global_cache: FxHashMap<(u64, u32, u8), Vec<u64>>,
}

#[inline]
/// Creates a deterministic cache key for molecule-topology-dependent descriptor caches.
pub(crate) fn molecule_cache_key(mol: &Molecule) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0001_0000_01b3;

    #[inline]
    /// Implements mix.
    ///
    /// Function context: `crates/ferrum_core/src/desc/context.rs`.
    /// Descriptor-layer note: computes molecular properties for ML pipelines.
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

    hash
}

impl CalculationContext {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_core/src/desc/context.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn new() -> Self {
        Self {
            active_molecule_key: None,
            dist_u8: Vec::new(),
            has_dist_u8: false,
            dist_u8_global_cache: FxHashMap::default(),
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
            morgan_features_global_cache: FxHashMap::default(),
        }
    }

    /// Implements for new molecule.
    ///
    /// Function context: `crates/ferrum_core/src/desc/context.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn reset_for_new_molecule(&mut self) {
        self.active_molecule_key = None;
        self.has_dist_u8 = false;
        self.has_dist_bo = false;
        self.bfs_queue.clear();
        self.bfs_dist.clear();

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
    }
}
