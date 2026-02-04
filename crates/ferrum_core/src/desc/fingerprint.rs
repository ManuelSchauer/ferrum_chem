use super::Descriptor;
use crate::mol::molecule::Molecule;
use crate::types::Real;

pub struct ECFP4 { pub n_bits: usize }
impl Descriptor for ECFP4 {
    fn name(&self) -> String { "ECFP4".into() }
    fn size(&self) -> usize { self.n_bits }
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        // Hashing Logik
        todo!()
    }
}