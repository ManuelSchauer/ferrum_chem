use crate::types::{AtomIdx, BondType};

/// Bond representation in a molecular graph.
///
/// # Thread Safety
/// `Bond` is `Copy + Send + Sync`, making it trivially thread-safe.
/// The struct is compact (6 bytes) for efficient parallel processing.
#[derive(Debug, Clone, Copy)]
pub struct Bond {
    pub source: AtomIdx,
    pub target: AtomIdx,
    pub bond_type: BondType,
    pub flags: u8, // Bit 0: Aromatic, Bit 1: Ring
}

// Static assertion: Bond must be Send + Sync
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<Bond>;
};

impl Bond {
    pub fn new(u: AtomIdx, v: AtomIdx, t: BondType) -> Self {
        Self {
            source: u,
            target: v,
            bond_type: t,
            flags: 0,
        }
    }

    /// Check if this bond is aromatic (bit 0 of flags).
    /// Note: Aromatic bonds may have bond_type Single or Double after kekulization.
    pub fn is_aromatic(&self) -> bool {
        self.flags & 0b01 != 0
    }

    pub fn set_aromatic(&mut self, val: bool) {
        if val {
            self.flags |= 0b01;
        } else {
            self.flags &= !0b01;
        }
    }

    pub fn is_in_ring(&self) -> bool {
        self.flags & 0b10 != 0
    }
    pub fn set_in_ring(&mut self, val: bool) {
        if val {
            self.flags |= 0b10;
        } else {
            self.flags &= !0b10;
        }
    }
}
