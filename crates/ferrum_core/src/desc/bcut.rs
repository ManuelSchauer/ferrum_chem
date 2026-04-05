//! BCUT (Burden-CAS-UT) descriptors.
//!
//! Modified Burden adjacency matrix with atomic property weights on the
//! diagonal. The highest and lowest eigenvalues capture molecular shape
//! and property extremes.
//!
//! Weights: atomic mass, electronegativity, polarizability, van der Waals volume.
//! Each produces 2 eigenvalues (highest, lowest) → 8 total.
//!
//! Reference: Pearlman, R.S. & Smith, K.M. (1999) "Metric Validation and
//!            the Receptor-Relevant Subspace Concept", JCICS 39, 28-35.

//! ## File guide: Bcut
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn fill_burden_matrix`
//! - `fn burden_bond_value`
//! - `struct BcutWorkspace`
//! - `fn with_size`
//! - `fn reset_uniform_start`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - BCUT uses exact eigendecomposition on small matrices and power-iteration approximations on larger matrices.
//! - Burden matrix operator form assumes sparse bond deltas over constant off-diagonal baseline for performance.
//! - Relative-error checks (not strict absolute) are used to keep large-magnitude eigenvalue comparisons stable.
//! ### Chemistry/ML references
//! - Burden, F. (1989). Molecular identification number for substructure searches (Burden eigenvalues).
//! - Pearlman et al. BCUT descriptor literature for weighted adjacency eigenvalue descriptors.
//! - RDKit BCUT2D behavior used for parity and tolerance calibration.

use super::atomic_props::{
    atomic_mass, burden_polarizability, burden_vdw_volume, pauling_electronegativity,
};
use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use smallvec::SmallVec;

const JACOBI_EXACT_MAX_N: usize = 24;
const POWER_MAX_ITERS: usize = 96;
const POWER_TOL: f64 = 1e-6;
const BURDEN_OFF_DIAG_BASE: f64 = 0.001;

// ---------------------------------------------------------------------------
// Burden matrix + eigenvalues
// ---------------------------------------------------------------------------

/// Build the modified Burden matrix:
/// - Diagonal: atomic property weight
/// - Off-diagonal connected: bond order / √(bond_order)
/// - Off-diagonal disconnected: 0.001 (small perturbation)
fn fill_burden_matrix(mol: &Molecule, weight_fn: fn(u8) -> f64, mat: &mut [f64]) {
    let n = mol.num_atoms();
    debug_assert_eq!(mat.len(), n * n);
    mat.fill(BURDEN_OFF_DIAG_BASE);

    // Diagonal: property weights
    let mut diag_idx = 0usize;
    for atom in &mol.atoms {
        mat[diag_idx] = weight_fn(atom.atomic_number);
        diag_idx += n + 1;
    }

    // Connected atoms: bond contribution
    for bond in &mol.bonds {
        let i = bond.source as usize;
        let j = bond.target as usize;
        let val = burden_bond_value(bond.bond_type);
        mat[i * n + j] = val;
        mat[j * n + i] = val;
    }
}

#[inline]
/// Implements bond value.
///
/// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn burden_bond_value(bond_type: crate::types::BondType) -> f64 {
    let bo = match bond_type {
        crate::types::BondType::Single => 1.0,
        crate::types::BondType::Double => 2.0,
        crate::types::BondType::Triple => 3.0,
        crate::types::BondType::Aromatic => 1.5,
        _ => 1.0,
    };
    (bo as f64).sqrt() * 0.1
}

struct BcutWorkspace {
    x: Vec<f64>,
    y: Vec<f64>,
    ax: Vec<f64>,
}

impl BcutWorkspace {
    /// Implements size.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn with_size(n: usize) -> Self {
        Self {
            x: vec![0.0; n],
            y: vec![0.0; n],
            ax: vec![0.0; n],
        }
    }

    /// Implements uniform start.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn reset_uniform_start(&mut self) {
        let n = self.x.len();
        if n == 0 {
            return;
        }
        let v = 1.0 / (n as f64).sqrt();
        self.x.fill(v);
        self.y.fill(0.0);
        self.ax.fill(0.0);
    }
}

struct BurdenLinearOperator {
    n: usize,
    diag: Vec<f64>,
    edge_deltas: Vec<SmallVec<[(usize, f64); 4]>>,
}

impl BurdenLinearOperator {
    /// Constructs molecule between representations.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn from_molecule(mol: &Molecule, weight_fn: fn(u8) -> f64) -> Self {
        let n = mol.num_atoms();
        let mut diag = Vec::with_capacity(n);
        for atom in &mol.atoms {
            diag.push(weight_fn(atom.atomic_number));
        }

        let mut edge_deltas = vec![SmallVec::<[(usize, f64); 4]>::new(); n];
        for bond in &mol.bonds {
            let i = bond.source as usize;
            let j = bond.target as usize;
            let delta = burden_bond_value(bond.bond_type) - BURDEN_OFF_DIAG_BASE;
            edge_deltas[i].push((j, delta));
            edge_deltas[j].push((i, delta));
        }

        Self {
            n,
            diag,
            edge_deltas,
        }
    }

    #[inline]
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n);
        debug_assert_eq!(y.len(), self.n);
        let sum_x: f64 = x.iter().sum();
        for i in 0..self.n {
            let mut acc = BURDEN_OFF_DIAG_BASE.mul_add(sum_x - x[i], self.diag[i] * x[i]);
            for &(j, delta) in &self.edge_deltas[i] {
                acc = delta.mul_add(x[j], acc);
            }
            y[i] = acc;
        }
    }

    /// Implements row abs sum.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn max_row_abs_sum(&self) -> f64 {
        let c_abs = BURDEN_OFF_DIAG_BASE.abs();
        let mut max_row = 0.0f64;
        for i in 0..self.n {
            let degree = self.edge_deltas[i].len();
            let mut row_sum = self.diag[i].abs();
            row_sum += (self.n.saturating_sub(1 + degree) as f64) * c_abs;
            for &(_, delta) in &self.edge_deltas[i] {
                row_sum += (BURDEN_OFF_DIAG_BASE + delta).abs();
            }
            if row_sum > max_row {
                max_row = row_sum;
            }
        }
        max_row
    }
}

/// Implements diagonalize in place.
///
/// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn jacobi_diagonalize_in_place(a: &mut [f64], n: usize) {
    let max_iter = 100 * n * n;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = a[i * n + j].abs();
                if v > max_val {
                    max_val = v;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-10 {
            break;
        }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();
        let c2 = c * c;
        let s2 = s * s;
        let cs2 = 2.0 * c * s;

        // Apply Jacobi rotation in-place.
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                let new_ip = c.mul_add(aip, s * aiq);
                let new_iq = (-s).mul_add(aip, c * aiq);
                a[i * n + p] = new_ip;
                a[p * n + i] = new_ip;
                a[i * n + q] = new_iq;
                a[q * n + i] = new_iq;
            }
        }
        a[p * n + p] = cs2.mul_add(apq, c2.mul_add(app, s2 * aqq));
        a[q * n + q] = (-cs2).mul_add(apq, s2.mul_add(app, c2 * aqq));
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;
    }
}

/// Implements in place.
///
/// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn normalize_in_place(v: &mut [f64]) -> f64 {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
    for value in v.iter_mut() {
        *value /= norm;
    }
    norm
}

/// Implements eigenvalue power.
///
/// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn largest_eigenvalue_power(a: &[f64], n: usize, max_iters: usize, tol: f64) -> f64 {
    debug_assert_eq!(a.len(), n * n);
    if n == 0 {
        return 0.0;
    }

    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    let mut y = vec![0.0; n];
    let mut lambda = 0.0;

    for _ in 0..max_iters {
        for i in 0..n {
            let row = &a[i * n..(i + 1) * n];
            y[i] = row.iter().zip(x.iter()).map(|(&aij, &xj)| aij * xj).sum();
        }

        let new_lambda: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let norm = normalize_in_place(&mut y);
        if !new_lambda.is_finite() || !norm.is_finite() {
            break;
        }

        let converged = (new_lambda - lambda).abs() <= tol * (1.0 + new_lambda.abs());
        x.copy_from_slice(&y);
        lambda = new_lambda;
        if converged {
            break;
        }
    }

    lambda
}

/// Implements eigenvalue shifted power.
///
/// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn largest_eigenvalue_shifted_power(
    a: &[f64],
    n: usize,
    shift: f64,
    max_iters: usize,
    tol: f64,
) -> f64 {
    debug_assert_eq!(a.len(), n * n);
    if n == 0 {
        return 0.0;
    }

    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    let mut y = vec![0.0; n];
    let mut lambda = 0.0;

    for _ in 0..max_iters {
        for i in 0..n {
            let row = &a[i * n..(i + 1) * n];
            let ax_i: f64 = row.iter().zip(x.iter()).map(|(&aij, &xj)| aij * xj).sum();
            y[i] = shift * x[i] - ax_i;
        }

        let new_lambda: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let norm = normalize_in_place(&mut y);
        if !new_lambda.is_finite() || !norm.is_finite() {
            break;
        }

        let converged = (new_lambda - lambda).abs() <= tol * (1.0 + new_lambda.abs());
        x.copy_from_slice(&y);
        lambda = new_lambda;
        if converged {
            break;
        }
    }

    lambda
}

/// Implements eigenvalue power operator.
///
/// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn largest_eigenvalue_power_operator(
    op: &BurdenLinearOperator,
    ws: &mut BcutWorkspace,
    max_iters: usize,
    tol: f64,
) -> f64 {
    if op.n == 0 {
        return 0.0;
    }

    ws.reset_uniform_start();
    let mut lambda = 0.0;

    for _ in 0..max_iters {
        op.matvec(&ws.x, &mut ws.y);
        let new_lambda: f64 = ws.x.iter().zip(ws.y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let norm = normalize_in_place(&mut ws.y);
        if !new_lambda.is_finite() || !norm.is_finite() {
            break;
        }

        let converged = (new_lambda - lambda).abs() <= tol * (1.0 + new_lambda.abs());
        ws.x.copy_from_slice(&ws.y);
        lambda = new_lambda;
        if converged {
            break;
        }
    }

    lambda
}

/// Implements eigenvalue shifted power operator.
///
/// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn largest_eigenvalue_shifted_power_operator(
    op: &BurdenLinearOperator,
    ws: &mut BcutWorkspace,
    shift: f64,
    max_iters: usize,
    tol: f64,
) -> f64 {
    if op.n == 0 {
        return 0.0;
    }

    ws.reset_uniform_start();
    let mut lambda = 0.0;

    for _ in 0..max_iters {
        op.matvec(&ws.x, &mut ws.ax);
        for i in 0..op.n {
            ws.y[i] = shift * ws.x[i] - ws.ax[i];
        }

        let new_lambda: f64 = ws.x.iter().zip(ws.y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let norm = normalize_in_place(&mut ws.y);
        if !new_lambda.is_finite() || !norm.is_finite() {
            break;
        }

        let converged = (new_lambda - lambda).abs() <= tol * (1.0 + new_lambda.abs());
        ws.x.copy_from_slice(&ws.y);
        lambda = new_lambda;
        if converged {
            break;
        }
    }

    lambda
}

/// Implements bounds burden operator.
///
/// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn eigenvalue_bounds_burden_operator(
    op: &BurdenLinearOperator,
    ws: &mut BcutWorkspace,
) -> (f64, f64) {
    debug_assert!(op.n > 0);
    if op.n == 1 {
        let v = op.diag[0];
        return (v, v);
    }

    let max_eig = largest_eigenvalue_power_operator(op, ws, POWER_MAX_ITERS, POWER_TOL);
    let shift = op.max_row_abs_sum().max(max_eig.abs()) + 1e-6;
    let shifted_max =
        largest_eigenvalue_shifted_power_operator(op, ws, shift, POWER_MAX_ITERS, POWER_TOL);
    let min_eig = shift - shifted_max;

    if max_eig.is_finite() && min_eig.is_finite() {
        if min_eig <= max_eig {
            (min_eig, max_eig)
        } else {
            (max_eig, min_eig)
        }
    } else {
        let bound = op.max_row_abs_sum();
        (-bound, bound)
    }
}

/// Implements bounds if exact.
///
/// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn diagonal_bounds_if_exact(matrix: &[f64], n: usize) -> Option<(f64, f64)> {
    let mut min_diag = matrix[0];
    let mut max_diag = matrix[0];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if matrix[i * n + j].abs() > 1e-15 {
                return None;
            }
        }
        let d = matrix[i * n + i];
        if d < min_diag {
            min_diag = d;
        }
        if d > max_diag {
            max_diag = d;
        }
    }
    Some((min_diag, max_diag))
}

/// Implements bounds symmetric inplace.
///
/// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn eigenvalue_bounds_symmetric_inplace(matrix: &mut [f64], n: usize) -> (f64, f64) {
    debug_assert!(n > 0);
    if n == 1 {
        let v = matrix[0];
        return (v, v);
    }
    if let Some(bounds) = diagonal_bounds_if_exact(matrix, n) {
        return bounds;
    }
    if n <= JACOBI_EXACT_MAX_N {
        jacobi_diagonalize_in_place(matrix, n);

        let mut min_eig = matrix[0];
        let mut max_eig = matrix[0];
        for i in 1..n {
            let v = matrix[i * n + i];
            if v < min_eig {
                min_eig = v;
            }
            if v > max_eig {
                max_eig = v;
            }
        }
        return (min_eig, max_eig);
    }

    // Fast path for larger matrices: estimate extremal eigenvalues using power iteration.
    let max_eig = largest_eigenvalue_power(matrix, n, POWER_MAX_ITERS, POWER_TOL);
    let mut row_sum_bound = 0.0f64;
    for i in 0..n {
        let row = &matrix[i * n..(i + 1) * n];
        let abs_sum = row.iter().map(|v| v.abs()).sum::<f64>();
        if abs_sum > row_sum_bound {
            row_sum_bound = abs_sum;
        }
    }
    let shift = row_sum_bound.max(max_eig.abs()) + 1e-6;
    let shifted_max =
        largest_eigenvalue_shifted_power(matrix, n, shift, POWER_MAX_ITERS, POWER_TOL);
    let min_eig = shift - shifted_max;

    if max_eig.is_finite() && min_eig.is_finite() {
        if min_eig <= max_eig {
            (min_eig, max_eig)
        } else {
            (max_eig, min_eig)
        }
    } else {
        // Conservative fallback for pathological numerical cases.
        jacobi_diagonalize_in_place(matrix, n);
        let mut min_eig = matrix[0];
        let mut max_eig = matrix[0];
        for i in 1..n {
            let v = matrix[i * n + i];
            if v < min_eig {
                min_eig = v;
            }
            if v > max_eig {
                max_eig = v;
            }
        }
        (min_eig, max_eig)
    }
}

/// Compute eigenvalues of a symmetric matrix using Jacobi iteration.
/// Returns sorted eigenvalues (ascending).
#[cfg(test)]
fn eigenvalues_symmetric(matrix: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![matrix[0]];
    }

    let mut a = matrix.to_vec();
    jacobi_diagonalize_in_place(&mut a, n);

    // Extract diagonal (eigenvalues)
    let mut eigs: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigs
}

// ---------------------------------------------------------------------------
// Descriptor
// ---------------------------------------------------------------------------

/// BCUT descriptors.
///
/// For each of 4 atomic properties (mass, electronegativity, polarizability,
/// vdW volume), computes the highest and lowest eigenvalue of the modified
/// Burden matrix.
///
/// Returns FloatVec with 8 values:
/// [mass_low, mass_high, en_low, en_high, pol_low, pol_high, vdw_low, vdw_high]
pub struct BCUT;

impl Descriptor for BCUT {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "BCUT"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut values = vec![0.0f32; 8];

        let n = mol.num_atoms();
        if n == 0 {
            return DescriptorValue::FloatVec(values);
        }

        let weight_fns: [fn(u8) -> f64; 4] = [
            atomic_mass,
            pauling_electronegativity,
            burden_polarizability,
            burden_vdw_volume,
        ];
        let mut matrix = if n <= JACOBI_EXACT_MAX_N {
            vec![0.0; n * n]
        } else {
            Vec::new()
        };
        let mut workspace = if n > JACOBI_EXACT_MAX_N {
            Some(BcutWorkspace::with_size(n))
        } else {
            None
        };

        for (wi, wfn) in weight_fns.iter().enumerate() {
            let (low, high) = if n <= JACOBI_EXACT_MAX_N {
                fill_burden_matrix(mol, *wfn, &mut matrix);
                eigenvalue_bounds_symmetric_inplace(&mut matrix, n)
            } else {
                let op = BurdenLinearOperator::from_molecule(mol, *wfn);
                eigenvalue_bounds_burden_operator(
                    &op,
                    workspace
                        .as_mut()
                        .expect("workspace initialized for large BCUT"),
                )
            };
            values[wi * 2] = low as f32;
            values[wi * 2 + 1] = high as f32;
        }

        DescriptorValue::FloatVec(values)
    }
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<BCUT>;
};

#[cfg(test)]
mod tests {
    use super::super::typed::Descriptor;
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements bcut.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_bcut(smiles: &str) -> Vec<f32> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match BCUT.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    /// Implements bounds.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn exact_bounds(mol: &Molecule, weight_fn: fn(u8) -> f64) -> (f64, f64) {
        let n = mol.num_atoms();
        let mut mat = vec![0.0; n * n];
        fill_burden_matrix(mol, weight_fn, &mut mat);
        let eigs = eigenvalues_symmetric(&mat, n);
        (eigs[0], eigs[n - 1])
    }

    #[test]
    /// Runs regression test for bcut len.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bcut_len() {
        let b = calc_bcut("CC");
        assert_eq!(b.len(), 8);
    }

    #[test]
    /// Runs regression test for bcut ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bcut_ethane() {
        let b = calc_bcut("CC");
        assert!(b[1] > 0.0, "mass high > 0");
    }

    #[test]
    /// Runs regression test for bcut ethanol different from ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bcut_ethanol_different_from_ethane() {
        let ethane = calc_bcut("CC");
        let ethanol = calc_bcut("CCO");
        assert_ne!(ethane, ethanol);
    }

    #[test]
    /// Runs regression test for bcut high ge low.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bcut_high_ge_low() {
        let b = calc_bcut("c1ccccc1");
        for i in 0..4 {
            assert!(
                b[i * 2 + 1] >= b[i * 2],
                "prop {}: high {} >= low {}",
                i,
                b[i * 2 + 1],
                b[i * 2]
            );
        }
    }

    #[test]
    /// Runs regression test for bcut deterministic.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bcut_deterministic() {
        let b1 = calc_bcut("CCCC");
        let b2 = calc_bcut("CCCC");
        assert_eq!(b1, b2);
    }

    #[test]
    /// Runs regression test for bcut single atom.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bcut_single_atom() {
        let b = calc_bcut("C");
        assert!(b[0] > 0.0);
        assert!((b[0] - b[1]).abs() < 0.01, "single atom: low ≈ high");
    }

    #[test]
    /// Runs regression test for bcut aspirin.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bcut_aspirin() {
        let b = calc_bcut("CC(=O)Oc1ccccc1C(=O)O");
        assert!(b[1] > b[0], "mass: high > low for mixed atoms");
    }

    #[test]
    /// Runs regression test for eigenvalues 2x2.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_eigenvalues_2x2() {
        let mat = vec![4.0, 1.0, 1.0, 4.0];
        let eigs = eigenvalues_symmetric(&mat, 2);
        assert!((eigs[0] - 3.0).abs() < 0.01, "λ₁ = {}", eigs[0]);
        assert!((eigs[1] - 5.0).abs() < 0.01, "λ₂ = {}", eigs[1]);
    }

    #[test]
    /// Runs regression test for eigenvalues identity.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_eigenvalues_identity() {
        let mat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let eigs = eigenvalues_symmetric(&mat, 3);
        for e in &eigs {
            assert!((e - 1.0).abs() < 0.01, "identity eigenvalue = {}", e);
        }
    }

    #[test]
    /// Runs regression test for eigenvalue bounds large diagonal uses fast path.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_eigenvalue_bounds_large_diagonal_uses_fast_path() {
        let n = JACOBI_EXACT_MAX_N + 16;
        let mut mat = vec![0.0; n * n];
        for i in 0..n {
            mat[i * n + i] = (i + 1) as f64;
        }

        let (min_eig, max_eig) = eigenvalue_bounds_symmetric_inplace(&mut mat, n);
        assert!((min_eig - 1.0).abs() < 1e-3, "min={min_eig}");
        assert!((max_eig - n as f64).abs() < 1e-2, "max={max_eig}");
    }

    #[test]
    /// Runs regression test for burden operator matches matrix bounds.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_burden_operator_matches_matrix_bounds() {
        let mut mol = parse_smiles(&"C".repeat(80)).expect("valid long alkane");
        let n = mol.num_atoms();
        let mut mat = vec![0.0; n * n];
        fill_burden_matrix(&mol, atomic_mass, &mut mat);
        let mut matrix_copy = mat.clone();
        let matrix_bounds = eigenvalue_bounds_symmetric_inplace(&mut matrix_copy, n);

        let op = BurdenLinearOperator::from_molecule(&mol, atomic_mass);
        let mut ws = BcutWorkspace::with_size(n);
        let op_bounds = eigenvalue_bounds_burden_operator(&op, &mut ws);

        assert!((op_bounds.0 - matrix_bounds.0).abs() < 1e-4);
        assert!((op_bounds.1 - matrix_bounds.1).abs() < 1e-4);
    }

    #[test]
    /// Runs regression test for bcut large molecule guardrail against exact jacobi.
    ///
    /// Function context: `crates/ferrum_core/src/desc/bcut.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_bcut_large_molecule_guardrail_against_exact_jacobi() {
        let smiles = "CCO".repeat(14);
        let mut mol = parse_smiles(&smiles).expect("valid large mixed-element chain");
        assert!(mol.num_atoms() > JACOBI_EXACT_MAX_N);

        let got = match BCUT.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        };
        let refs = [
            exact_bounds(&mol, atomic_mass),
            exact_bounds(&mol, pauling_electronegativity),
            exact_bounds(&mol, burden_polarizability),
            exact_bounds(&mol, burden_vdw_volume),
        ];

        for (i, (exp_low, exp_high)) in refs.iter().enumerate() {
            let low = got[i * 2] as f64;
            let high = got[i * 2 + 1] as f64;
            let low_rel = (low - exp_low).abs() / exp_low.abs().max(1e-9);
            let high_rel = (high - exp_high).abs() / exp_high.abs().max(1e-9);
            assert!(
                low_rel < 3e-2,
                "weight {i} low mismatch: got={low} exp={exp_low} rel={low_rel}"
            );
            assert!(
                high_rel < 3e-2,
                "weight {i} high mismatch: got={high} exp={exp_high} rel={high_rel}"
            );
        }
    }
}
