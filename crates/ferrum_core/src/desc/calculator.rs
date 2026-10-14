//! The single entry point for descriptor calculation.
//!
//! A [`DescriptorCalculator`] is created once and then shared by every thread
//! for every molecule: it is immutable after construction and `Send + Sync`.
//! Molecules are only read (`&Molecule`); derived per-molecule data is cached
//! thread-safely inside the molecule (`OnceLock`), and scratch space lives in
//! one [`CalculationContext`] per thread, which the calculator resets before
//! every molecule.
//!
//! ```
//! use ferrum_core::desc::calculator::DescriptorCalculator;
//!
//! let calc = DescriptorCalculator::all(); // build once, share everywhere
//! let table = calc.calculate_smiles(&["c1ccccc1O", "CCN", "not a smiles"]);
//! assert_eq!(table.rows, 3);
//! assert_eq!(table.valid, vec![true, true, false]);
//! let logp = table.scalar_row(0)[table.scalar_names.iter().position(|n| n == "MolLogP").unwrap()];
//! assert!((logp - 1.3922).abs() < 1e-9);
//! ```

use std::cell::RefCell;

use rayon::prelude::*;

use super::batch::BatchDescriptor;
use super::context::CalculationContext;
use super::table::{compute_table, Cancelled, CountFingerprintFormat, DescriptorTable, MoleculeSlot, TableLayout};
use crate::io::smiles::parse_smiles;
use crate::mol::molecule::Molecule;

thread_local! {
    /// Scratch space of the current thread (reused across molecules and calls).
    static CONTEXT: RefCell<CalculationContext> = RefCell::new(CalculationContext::new());
}

/// Runs `f` with this thread's calculation context, reset for a new molecule.
pub(crate) fn with_thread_context<R>(f: impl FnOnce(&mut CalculationContext) -> R) -> R {
    CONTEXT.with(|cell| match cell.try_borrow_mut() {
        Ok(mut ctx) => {
            ctx.reset_for_new_molecule();
            f(&mut ctx)
        }
        // re-entrant use on the same thread: use a fresh context
        Err(_) => f(&mut CalculationContext::new()),
    })
}

/// Molecules used to initialise every lazily compiled pattern and table at
/// construction, so the first real molecule does not pay for it.
const WARM_UP: &[&str] = &["CC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O", "c1ccc2[nH]ccc2c1", "[Na+].[Cl-]", "OP(=O)(O)OCC1OC(n2cnc3c2ncnc3N)C(O)C1O"];

/// Computes a fixed selection of descriptors for any number of molecules.
///
/// Immutable after construction and `Send + Sync`: create one and use it from
/// all threads (e.g. behind an `Arc` or a `&'static`).
#[derive(Debug, Clone)]
pub struct DescriptorCalculator {
    layout: TableLayout,
}

impl DescriptorCalculator {
    /// Calculator for the given descriptor families (output order: scalars
    /// in the given order, then fingerprints in the given order).
    pub fn new(descriptors: &[BatchDescriptor]) -> Self {
        let calc = Self { layout: TableLayout::new(descriptors) };
        let warm: Vec<Option<Molecule>> = WARM_UP.iter().map(|s| parse_smiles(s).ok()).collect();
        let _ = calc.calculate(&warm);
        calc
    }

    /// Calculator for the complete default bundle ([`BatchDescriptor::all`]).
    pub fn all() -> Self {
        Self::new(BatchDescriptor::all())
    }

    /// The default bundle without the six bit fingerprints.
    pub fn scalars_only() -> Self {
        let scalars: Vec<BatchDescriptor> =
            BatchDescriptor::all().iter().copied().filter(|d| !d.is_fingerprint()).collect();
        Self::new(&scalars)
    }

    /// The same calculator with count fingerprints (`KRFPC1..`, `SubFPC1..`)
    /// output as `format`: scalar columns (default) or sparse
    /// [`super::table::CountBlock`]s.
    ///
    /// ```
    /// use ferrum_core::desc::batch::BatchDescriptor;
    /// use ferrum_core::desc::calculator::DescriptorCalculator;
    /// use ferrum_core::desc::table::CountFingerprintFormat;
    ///
    /// let calc = DescriptorCalculator::new(&[BatchDescriptor::KlekotaRothCount])
    ///     .with_count_format(CountFingerprintFormat::Sparse);
    /// let table = calc.calculate_smiles(&["CC(=O)Oc1ccccc1C(=O)O"]);
    /// let kr = table.count_fingerprint("KlekotaRothCount").unwrap();
    /// assert_eq!(table.n_scalars(), 0);
    /// let (keys, counts) = kr.row(0); // non-zero counts only, keys ascending
    /// assert!(!keys.is_empty() && counts.iter().all(|&c| c > 0));
    /// ```
    pub fn with_count_format(self, format: CountFingerprintFormat) -> Self {
        Self { layout: TableLayout::with_count_format(self.layout.descriptors(), format) }
    }

    /// How count fingerprints are output.
    pub fn count_format(&self) -> CountFingerprintFormat {
        self.layout.count_format()
    }

    /// Column layout (scalar names, fingerprint shapes).
    pub fn layout(&self) -> &TableLayout {
        &self.layout
    }

    /// Names of the scalar columns, in output order.
    pub fn scalar_names(&self) -> &[String] {
        self.layout.scalar_names()
    }

    /// Computes all descriptors for a batch of molecules in parallel. Items
    /// may be `Molecule`, `&Molecule`, `Option<Molecule>` or `Option<&Molecule>`
    /// (`None` gives an invalid row).
    pub fn calculate<M: MoleculeSlot>(&self, molecules: &[M]) -> DescriptorTable {
        match compute_table(molecules, &self.layout, &|| false) {
            Ok(table) => table,
            Err(Cancelled) => unreachable!("never cancelled"),
        }
    }

    /// [`Self::calculate`] with cooperative cancellation between molecules.
    pub fn calculate_cancellable<M, F>(&self, molecules: &[M], should_cancel: &F) -> Result<DescriptorTable, Cancelled>
    where
        M: MoleculeSlot,
        F: Fn() -> bool + Sync,
    {
        compute_table(molecules, &self.layout, should_cancel)
    }

    /// Descriptors of a single molecule (a one-row table), computed on the
    /// calling thread.
    pub fn calculate_one(&self, molecule: &Molecule) -> DescriptorTable {
        self.calculate(&[molecule])
    }

    /// Parses SMILES in parallel and computes their descriptors; unparseable
    /// SMILES give invalid rows.
    pub fn calculate_smiles<S: AsRef<str> + Sync>(&self, smiles: &[S]) -> DescriptorTable {
        let molecules: Vec<Option<Molecule>> = smiles.par_iter().map(|s| parse_smiles(s.as_ref()).ok()).collect();
        self.calculate(&molecules)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DescriptorCalculator>();
    assert_send_sync::<Molecule>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn one_calculator_shared_by_threads_gives_identical_results() {
        let calc = Arc::new(DescriptorCalculator::all());
        let smiles = ["CC(=O)Oc1ccccc1C(=O)O", "C1CCC2(CC1)CCCC2", "c1ccc2ccccc2c1", "[NH4+].[Cl-]"];
        let reference = calc.calculate_smiles(&smiles);
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let calc = Arc::clone(&calc);
                std::thread::spawn(move || calc.calculate_smiles(&smiles))
            })
            .collect();
        for h in handles {
            let t = h.join().unwrap();
            assert_eq!(t.valid, reference.valid);
            for (a, b) in t.scalars.iter().zip(&reference.scalars) {
                assert!(a == b || (a.is_nan() && b.is_nan()));
            }
            assert_eq!(t.fingerprints, reference.fingerprints);
        }
    }

    #[test]
    fn molecules_are_read_only_and_can_be_shared() {
        let calc = DescriptorCalculator::scalars_only();
        let mol = Arc::new(parse_smiles("CCOC(=O)c1ccccc1").unwrap());
        let a = calc.calculate_one(&mol);
        // the same molecule from several threads at once
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let (calc, mol) = (calc.clone(), Arc::clone(&mol));
                std::thread::spawn(move || calc.calculate_one(&mol))
            })
            .collect();
        for h in handles {
            // the first call filled the molecule's caches; later calls read them
            for (x, y) in h.join().unwrap().scalars.iter().zip(&a.scalars) {
                assert!(x == y || (x.is_nan() && y.is_nan()));
            }
        }
        assert!(calc.layout().fingerprint_descriptors().is_empty());
    }

    #[test]
    fn order_of_molecules_does_not_change_values() {
        // stale per-thread state would make results depend on the previous molecule
        let calc = DescriptorCalculator::all();
        let x = ["CCCCCCCCCC", "c1ccccc1", "C1CC1"];
        let y = ["c1ccccc1", "C1CC1", "CCCCCCCCCC"];
        let (tx, ty) = (calc.calculate_smiles(&x), calc.calculate_smiles(&y));
        for (i, j) in [(0, 2), (1, 0), (2, 1)] {
            assert_eq!(tx.scalar_row(i).len(), ty.scalar_row(j).len());
            for (a, b) in tx.scalar_row(i).iter().zip(ty.scalar_row(j)) {
                assert!(a == b || (a.is_nan() && b.is_nan()));
            }
        }
    }

    #[test]
    fn cancellation() {
        let calc = DescriptorCalculator::scalars_only();
        let mols: Vec<Option<Molecule>> = vec![parse_smiles("C").ok(); 64];
        assert_eq!(calc.calculate_cancellable(&mols, &|| true), Err(Cancelled));
    }
}
