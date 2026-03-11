//! FerrumChem Core Library
//!
//! Eine High-Performance QSAR-Engine geschrieben in Rust.

pub mod algo;
pub mod desc;
pub mod error;
pub mod io;
pub mod ml;
pub mod mol;
pub mod types;

// Re-Export wichtiger Structs für bequemeren Zugriff
pub use mol::atom::Atom;
pub use mol::molecule::Molecule;
