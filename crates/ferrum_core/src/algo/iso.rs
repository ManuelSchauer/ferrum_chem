use crate::mol::molecule::Molecule;
use crate::io::smarts::QueryGraph;

/// Prüft, ob das Pattern im Target enthalten ist (VF2 Algorithmus).
pub fn matches(target: &Molecule, pattern: &QueryGraph) -> bool {
    todo!()
}