/// Index type for atoms (max 65,535 atoms per molecule).
pub type AtomIdx = u16;

/// Index type for bonds.
pub type BondIdx = u16;

/// Floating point type for descriptors (switchable f32/f64).
pub type Real = f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
    Dative,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hybridization {
    SP,
    SP2,
    SP3,
    SP3D,
    SP3D2,
    Unspecified,
}
