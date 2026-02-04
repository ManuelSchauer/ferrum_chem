use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::Real;

/// Berechnet das Molekulargewicht (MW).
pub struct AtomCount;

impl Descriptor for AtomCount {
    fn name(&self) -> String { "Atom Count".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        todo!()
    }
}
