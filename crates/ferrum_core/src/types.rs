
/// Index type for atoms (max 65,535 atoms per molecule).
pub type AtomIdx = u16;

/// Index type for bonds.
pub type BondIdx = u16;

/// Floating point type for descriptors (switchable f32/f64).
pub type Real = f64;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Implements BondType.
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
    Dative,
    Unknown,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Implements Hybridization.
pub enum Hybridization {
    /// s orbital only (e.g. H, isolated ions); RDKit `S`.
    S,
    SP,
    SP2,
    SP3,
    SP3D,
    SP3D2,
    Unspecified,
}
