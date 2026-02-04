use crate::error::Result;
use crate::mol::molecule::Molecule;
use std::path::Path;

/// Parst eine SDF Datei und gibt einen Vektor von Molekülen zurück.
pub fn parse_sdf(path: &Path) -> Result<Vec<Molecule>> {
    // Hier V2000 Block Parsing implementieren
    todo!("Implement SDF parser")
}
