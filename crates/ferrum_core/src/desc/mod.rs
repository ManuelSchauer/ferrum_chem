use crate::mol::molecule::Molecule;
use crate::types::Real;

pub mod molecular_weight;
pub mod fingerprint;
pub mod topo;
mod batch;
mod wiener_numbers;
mod atom_count;
mod bond_count;

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
