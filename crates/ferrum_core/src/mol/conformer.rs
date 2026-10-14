//! 3D Conformer representations.
//!
//! This module defines the 3D conformer data structures used for
//! 3D descriptors and shape analysis.

#[derive(Debug, Clone)]
/// Implements Conformer.
pub struct Conformer {
    /// 3D coordinates for each atom in the molecule (x, y, z) in Angstroms.
    /// The length of this vector must exactly match the number of atoms.
    pub positions: Vec<[f64; 3]>,
}

impl Conformer {
    pub fn new(positions: Vec<[f64; 3]>) -> Self {
        Self { positions }
    }
}
