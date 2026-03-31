//! Descriptor module: Atom Count.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.


//! ## File guide: Atom Count
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn compute_atom_count`
//! - `fn compute_heavy_atom_count`
//! - `fn compute_aromatic_atom_count`
//! - `struct AtomCount`
//! - `fn name`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Atom Count` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Atom Count` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/atom_count.rs` defines atom count behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Atom Count` workflows.
//! - Repository unit and parity tests covering `Atom Count` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/atom_count.rs`.

use super::typed::Descriptor;
use crate::mol::molecule::Molecule;

// ---------------------------------------------------------------------------
// Public computation functions
// ---------------------------------------------------------------------------

/// Compute total atom count (heavy atoms + hydrogens).
pub fn compute_atom_count(mol: &Molecule) -> u32 {
    let heavy_count = mol.num_atoms() as u32;
    let hydrogen_count: u32 = mol.atoms.iter().map(|atom| atom.total_h() as u32).sum();
    heavy_count + hydrogen_count
}

/// Compute heavy atom count (non-hydrogen atoms).
pub fn compute_heavy_atom_count(mol: &Molecule) -> u32 {
    mol.atoms
        .iter()
        .filter(|atom| atom.atomic_number > 1)
        .count() as u32
}

/// Compute aromatic atom count.
pub fn compute_aromatic_atom_count(mol: &Molecule) -> u32 {
    mol.atoms.iter().filter(|atom| atom.is_aromatic()).count() as u32
}

// ---------------------------------------------------------------------------
// Typed Descriptors
// ---------------------------------------------------------------------------

/// Counts the total number of atoms in the molecule.
/// Includes all heavy atoms plus all hydrogens (explicit and implicit).
pub struct AtomCount;

impl Descriptor for AtomCount {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "AtomCount"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> super::typed::DescriptorValue {
        super::typed::DescriptorValue::Int(compute_atom_count(mol))
    }
}

/// Counts only heavy atoms (non-hydrogen atoms).
pub struct HeavyAtomCount;

impl Descriptor for HeavyAtomCount {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "HeavyAtomCount"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> super::typed::DescriptorValue {
        super::typed::DescriptorValue::Int(compute_heavy_atom_count(mol))
    }
}

/// Element-wise atom counts.
///
/// Returns IntU16Vec with counts for C, N, O, S, P, F, Cl, Br, I, Others, H, Heavy.
/// Total: 12 values. Uses u16 for memory efficiency.
pub struct ElementCounts;

impl Descriptor for ElementCounts {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "ElementCounts"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> super::typed::DescriptorValue {
        let mut counts = vec![0u16; 12]; // C, N, O, S, P, F, Cl, Br, I, Others, H, Heavy

        let mut hydrogen_count = 0u16;
        let mut heavy_count = 0u16;

        for atom in &mol.atoms {
            let an = atom.atomic_number;
            if an <= 1 {
                continue;
            }

            heavy_count += 1;
            hydrogen_count += atom.total_h() as u16;

            let idx = match an {
                6 => 0,  // C
                7 => 1,  // N
                8 => 2,  // O
                16 => 3, // S
                15 => 4, // P
                9 => 5,  // F
                17 => 6, // Cl
                35 => 7, // Br
                53 => 8, // I
                _ => 9,  // Others
            };
            counts[idx] += 1;
        }

        counts[10] = hydrogen_count; // H
        counts[11] = heavy_count; // Heavy

        super::typed::DescriptorValue::IntU16Vec(counts)
    }
}

/// Counts atoms of a specific element in the molecule.
///
/// ```
/// use ferrum_core::desc::atom_count::SingleAtomCount;
/// use ferrum_core::desc::Descriptor;
/// use ferrum_core::io::smiles::parse_smiles;
///
/// let mut mol = parse_smiles("CCO").unwrap();
/// let carbon_count = SingleAtomCount::new("C");
/// assert_eq!(carbon_count.calculate(&mut mol, &mut ferrum_core::desc::context::CalculationContext::new()).as_int(), 2);
/// ```
pub struct SingleAtomCount {
    element_symbol: String,
    atomic_number: u8,
}

impl SingleAtomCount {
    /// Creates a new value with the defaults and invariants required by this module.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn new(element: &str) -> Self {
        let atomic_number = crate::mol::elements::symbol_to_atomic_number(element).unwrap_or(0);
        Self {
            element_symbol: element.to_string(),
            atomic_number,
        }
    }
}

impl Descriptor for SingleAtomCount {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        &self.element_symbol
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> super::typed::DescriptorValue {
        if self.atomic_number == 1 {
            // Hydrogen: count explicit + implicit
            let count: u32 = mol.atoms.iter().map(|a| a.total_h() as u32).sum();
            super::typed::DescriptorValue::Int(count)
        } else {
            let count = mol
                .atoms
                .iter()
                .filter(|a| a.atomic_number == self.atomic_number)
                .count() as u32;
            super::typed::DescriptorValue::Int(count)
        }
    }
}

/// Counts aromatic atoms in the molecule.
pub struct AromaticAtomCount;

impl Descriptor for AromaticAtomCount {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "AromaticAtomCount"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> super::typed::DescriptorValue {
        super::typed::DescriptorValue::Int(compute_aromatic_atom_count(mol))
    }
}

// Thread safety
const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<AtomCount>;
    let _ = assert_send_sync::<HeavyAtomCount>;
    let _ = assert_send_sync::<ElementCounts>;
    let _ = assert_send_sync::<SingleAtomCount>;
    let _ = assert_send_sync::<AromaticAtomCount>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::aromaticity::perceive_aromaticity;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for atom count.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_atom_count() {
        let mut mol = parse_smiles("CCO").unwrap(); // Ethanol: C2H6O
        assert_eq!(AtomCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 9); // 2C + 1O + 6H
    }

    #[test]
    /// Runs regression test for atom count methane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_atom_count_methane() {
        let mut mol = parse_smiles("C").unwrap(); // Methane: CH4
        assert_eq!(AtomCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 5); // 1C + 4H
    }

    #[test]
    /// Runs regression test for atom count benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_atom_count_benzene() {
        let mut mol = parse_smiles("c1ccccc1").unwrap(); // Benzene: C6H6
        assert_eq!(AtomCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 12); // 6C + 6H
    }

    #[test]
    /// Runs regression test for heavy atom count.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_heavy_atom_count() {
        let mut mol = parse_smiles("CCO").unwrap(); // Ethanol
        assert_eq!(HeavyAtomCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 3);
    }

    #[test]
    /// Runs regression test for aromatic atom count.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_aromatic_atom_count() {
        let mut mol = parse_smiles("c1ccccc1").unwrap(); // Benzene
        perceive_aromaticity(&mut mol);
        assert_eq!(AromaticAtomCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 6);
    }

    #[test]
    /// Runs regression test for aromatic atom count non aromatic.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_aromatic_atom_count_non_aromatic() {
        let mut mol = parse_smiles("CCCC").unwrap(); // Butane
        perceive_aromaticity(&mut mol);
        assert_eq!(AromaticAtomCount.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);
    }

    #[test]
    /// Runs regression test for compute atom count.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_compute_atom_count() {
        let mol = parse_smiles("CCO").unwrap();
        assert_eq!(compute_atom_count(&mol), 9);
    }

    #[test]
    /// Runs regression test for compute heavy atom count.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_compute_heavy_atom_count() {
        let mol = parse_smiles("CCO").unwrap();
        assert_eq!(compute_heavy_atom_count(&mol), 3);
    }

    #[test]
    /// Runs regression test for compute aromatic atom count.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_compute_aromatic_atom_count() {
        let mut mol = parse_smiles("c1ccccc1").unwrap();
        perceive_aromaticity(&mut mol);
        assert_eq!(compute_aromatic_atom_count(&mol), 6);
    }

    #[test]
    /// Runs regression test for single atom count carbon.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_single_atom_count_carbon() {
        let mut mol = parse_smiles("CCO").unwrap();
        let desc = SingleAtomCount::new("C");
        assert_eq!(desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 2);
    }

    #[test]
    /// Runs regression test for single atom count oxygen.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_single_atom_count_oxygen() {
        let mut mol = parse_smiles("CCO").unwrap();
        let desc = SingleAtomCount::new("O");
        assert_eq!(desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    /// Runs regression test for single atom count nitrogen.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_single_atom_count_nitrogen() {
        let mut mol = parse_smiles("CCN").unwrap();
        let desc = SingleAtomCount::new("N");
        assert_eq!(desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 1);
    }

    #[test]
    /// Runs regression test for single atom count absent element.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_single_atom_count_absent_element() {
        let mut mol = parse_smiles("CCCC").unwrap();
        let desc = SingleAtomCount::new("O");
        assert_eq!(desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 0);
    }

    #[test]
    /// Runs regression test for single atom count hydrogen.
    ///
    /// Function context: `crates/ferrum_core/src/desc/atom_count.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_single_atom_count_hydrogen() {
        let mut mol = parse_smiles("C").unwrap(); // CH4
        let desc = SingleAtomCount::new("H");
        assert_eq!(desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_int(), 4);
    }
}
