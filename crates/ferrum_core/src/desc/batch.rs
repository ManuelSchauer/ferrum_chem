use crate::mol::molecule::Molecule;
use crate::desc::Descriptor;
use crate::types::Real;
use rayon::prelude::*;

/// Calculates descriptors for many molecules in parallel.
///
/// # Thread Safety
/// This function is designed for maximum parallelism:
/// - Uses Rayon for automatic work-stealing parallelization
/// - Molecules are shared immutably across threads (`&[Molecule]`)
/// - Descriptors must implement `Sync` for safe concurrent access
/// - Each thread has its own output buffer (no shared mutable state)
///
/// # Performance
/// Scales linearly with CPU cores. Typical speedup: 4-8x on modern hardware.
/// Best performance with batch sizes > 1000 molecules.
///
/// # Arguments
/// * `molecules` - Slice of molecules already in RAM (shared immutably)
/// * `descriptors` - Descriptors to calculate (must be `Sync` for thread safety)
///
/// # Returns
/// Vector of feature vectors (one per molecule), preserved in input order.
///
/// # Example
/// ```ignore
/// let molecules = vec![parse_smiles("C"), parse_smiles("CC")];
/// let descriptors: Vec<Box<dyn Descriptor + Sync>> = vec![
///     Box::new(MolecularWeight),
/// ];
/// let results = calculate_batch(&molecules, &descriptors);
/// ```
pub fn calculate_batch(
    molecules: &[Molecule],
    descriptors: &[Box<dyn Descriptor + Sync>] // Sync is critical for thread safety!
) -> Vec<Vec<Real>> {

    // Calculate total feature vector size once upfront
    let total_size: usize = descriptors.iter().map(|d| d.size()).sum();

    // par_iter() converts normal iterator to parallel iterator.
    // Rayon handles load-balancing and thread management automatically.
    molecules.par_iter().map(|mol| {

        // This block runs in parallel across all CPU cores
        // Each thread has its own feature buffer (no contention)
        let mut features = vec![0.0; total_size];
        let mut offset = 0;

        for desc in descriptors {
            let size = desc.size();
            // Slice the appropriate part of the buffer
            let dest_slice = &mut features[offset..offset + size];

            // Descriptor writes directly to memory
            // Thread-safe because each molecule is processed by one thread
            desc.calculate(mol, dest_slice);

            offset += size;
        }

        features
    }).collect() // Collects results in original order (Rayon guarantees this)
}
