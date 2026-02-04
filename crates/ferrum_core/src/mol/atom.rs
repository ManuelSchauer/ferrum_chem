use crate::types::Hybridization;

/// Atom representation in a molecular graph.
/// 
/// # Thread Safety
/// `Atom` is `Copy + Send + Sync`, making it trivially thread-safe.
/// The struct is designed to be compact (12 bytes) and cache-friendly
/// for high-performance parallel processing.
#[derive(Debug, Clone, Copy)]
pub struct Atom {
    pub atomic_number: u8,
    pub formal_charge: i8,
    pub num_explicit_h: u8,
    pub num_implicit_h: u8,  // Calculated implicit hydrogens
    pub isotope: u16,
    pub hybridization: Hybridization,
    pub flags: u8, // Bit 0: Aromatic, Bit 1: Ring
}

// Static assertion: Atom must be Send + Sync
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Atom>;
};

impl Atom {
    pub fn new(element: u8) -> Self {
        Self {
            atomic_number: element,
            formal_charge: 0,
            num_explicit_h: 0,
            num_implicit_h: 0,
            isotope: 0,
            hybridization: Hybridization::Unspecified,
            flags: 0,
        }
    }

    pub fn is_aromatic(&self) -> bool { self.flags & 1 != 0 }
    pub fn set_aromatic(&mut self, val: bool) {
        if val { self.flags |= 1; } else { self.flags &= !1; }
    }
    
    pub fn is_in_ring(&self) -> bool { self.flags & 0b10 != 0 }
    pub fn set_in_ring(&mut self, val: bool) {
        if val { self.flags |= 0b10; } else { self.flags &= !0b10; }
    }
    
    /// Total hydrogen count (explicit + implicit)
    pub fn total_h(&self) -> u8 {
        self.num_explicit_h + self.num_implicit_h
    }
}
