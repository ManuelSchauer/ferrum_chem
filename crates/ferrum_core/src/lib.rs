//! FerrumChem Core Library
//!
//! Eine High-Performance QSAR-Engine geschrieben in Rust.

pub mod error;
pub mod types;
pub mod mol;
pub mod io;
pub mod algo;
pub mod desc;
pub mod ml;

// Re-Export wichtiger Structs für bequemeren Zugriff
pub use mol::molecule::Molecule;
pub use mol::atom::Atom;
