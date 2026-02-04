use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::Real;

/// Berechnet das Molekulargewicht (MW).
pub struct BondCountDescriptor;

impl Descriptor for BondCountDescriptor {
    fn name(&self) -> String { "Bond Count".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        todo!()
    }
}
