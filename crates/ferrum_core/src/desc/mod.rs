use crate::mol::molecule::Molecule;
use crate::types::Real;

// Public modules
pub mod molecular_weight;
pub mod fingerprint;
pub mod topo;
pub mod atom_count;
pub mod bond_count;
pub mod aromatic_ring_count;
pub mod formal_charge;
pub mod hbond;
pub mod logp;
pub mod lipinski;
pub mod chains;
pub mod pi_system;
pub mod maccs;
pub mod small_ring;
pub mod wiener_numbers;
pub mod simple_counts;
pub mod zagreb;
pub mod balaban;
pub mod eccentric;
pub mod ring_types;
pub mod carbon_types;
pub mod spiro;
pub mod gravitational;
pub mod petitjean;
pub mod tpsa;
pub mod batch;
pub mod kappa;
pub mod functional_groups;
pub mod qed;
pub mod path_fingerprints;
pub mod pubchem;
pub mod chi;
pub mod estate;
pub mod autocorrelation;
pub mod bcut;
pub mod weighted_path;
pub mod mde;
pub mod extra_counts;
pub mod vsa;

/// Generic interface for all descriptors.
///
/// # Thread Safety
/// All descriptor implementations MUST be `Send + Sync` for parallel batch processing.
/// The trait is object-safe to allow `Box<dyn Descriptor + Sync>` in batch calculations.
///
/// # Implementation Requirements
/// - Implementations must be stateless or use thread-safe internal state
/// - The `calculate` method must not mutate `self` (only takes `&self`)
/// - All calculations must be deterministic and reentrant
pub trait Descriptor: Send + Sync {
    /// Returns the name of the descriptor (for CSV headers).
    fn name(&self) -> String;

    /// The number of values this descriptor produces.
    fn size(&self) -> usize;

    /// Calculates the descriptor and writes results to `output`.
    /// `output` must have the correct length (equal to `self.size()`).
    ///
    /// # Thread Safety
    /// This method may be called concurrently from multiple threads.
    /// Implementations must not use mutable shared state.
    fn calculate(&self, mol: &Molecule, output: &mut [Real]);
}
