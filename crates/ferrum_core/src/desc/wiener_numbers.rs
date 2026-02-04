use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::Real;

/// Berechnet die Wiener Numbers für ein Molekül.
pub struct WienerNumbers;

impl Descriptor for WienerNumbers {
    fn name(&self) -> String { "MW".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        todo!()
    }
}
