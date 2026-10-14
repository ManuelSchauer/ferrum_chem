//! CDK `GasteigerMarsiliPartialCharges` (PEOE sigma charges), ported 1:1.
//!
//! Unlike the RDKit Gasteiger implementation in `algo::electronic`, CDK starts
//! every atom at charge 0 (formal charges are ignored), selects parameters
//! from the element, the maximum (Kekulé) bond order and the formal charge,
//! uses 20.02 as denominator for hydrogen, damps with 0.5^k and stops after
//! 20 iterations or when no |charge| changes by more than 0.001 (with CDK's
//! quirk that only the first atom's previous charge is initialised to 20).

use super::cdk_perception::CdkView;
use crate::mol::molecule::Molecule;

const DEOC_HYDROGEN: f64 = 20.02;
const MX_DAMP: f64 = 0.5;
const MX_ITERATIONS: usize = 20;

/// Gasteiger-Marsili parameters `[a, b, c]` (`assignGasteigerSigmaMarsiliFactors`).
/// `max_order` is `IAtomContainer.getMaximumBondOrder` (0 = UNSET).
/// Returns `None` for elements CDK does not parameterise (CDK throws).
pub(crate) fn gm_factors(z: u8, max_order: u8, charge: i8) -> Option<[f64; 3]> {
    Some(match z {
        1 => [7.17, 6.24, -0.56],
        6 => {
            if max_order == 2 || (max_order == 1 && (charge == -1 || charge == 1)) {
                [8.79, 9.32, 1.51]
            } else if max_order == 1 && charge == 0 {
                [7.98, 9.18, 1.88]
            } else if max_order >= 3 {
                [10.39, 9.45, 0.73]
            } else {
                [0.0; 3]
            }
        }
        7 => {
            if max_order == 1 && charge != -1 {
                [11.54, 10.82, 1.36]
            } else if max_order == 2 || max_order == 1 {
                [12.87, 11.15, 0.85]
            } else if max_order >= 3 {
                [17.68, 12.70, -0.27]
            } else {
                [0.0; 3]
            }
        }
        8 => {
            if max_order == 1 && charge != -1 {
                [14.18, 12.92, 1.39]
            } else if max_order == 2 || max_order == 1 {
                [17.07, 13.79, 0.47]
            } else {
                [0.0; 3]
            }
        }
        14 => [8.10, 7.92, 1.78],
        15 => [8.90, 8.32, 1.58],
        16 => [10.14, 9.13, 1.38],
        9 => [14.66, 13.85, 2.31],
        17 => [12.31, 10.84, 1.512],
        35 => [11.44, 9.63, 1.31],
        53 => [9.88, 7.95, 0.945],
        _ => return None,
    })
}

/// Runs the Gasteiger-Marsili iteration on a graph given as parameters per atom
/// and a bond list (CDK bond iteration order).
pub(crate) fn gasteiger_marsili(
    factors: &[[f64; 3]],
    is_h: &[bool],
    bonds: &[(u32, u32)],
) -> Vec<f64> {
    let n = factors.len();
    let denom: Vec<f64> = factors
        .iter()
        .map(|f| {
            if f[0] == 0.0 && f[1] == 0.0 && f[2] == 0.0 {
                1.0
            } else {
                f[0] + f[1] + f[2]
            }
        })
        .collect();
    let mut q = vec![0.0f64; n];
    let mut chi = vec![0.0f64; n];
    let mut q_old = vec![0.0f64; n];
    if n > 0 {
        q_old[0] = 20.0;
    }
    let mut alpha = 1.0f64;
    for _ in 0..MX_ITERATIONS {
        alpha *= MX_DAMP;
        let mut different = false;
        for j in 0..n {
            let qj = q[j];
            if (q_old[j].abs() - qj.abs()).abs() > 0.001 {
                different = true;
            }
            q_old[j] = qj;
            let f = &factors[j];
            chi[j] = f[2] * qj * qj + f[1] * qj + f[0];
        }
        if !different {
            break;
        }
        for &(u, v) in bonds {
            let (u, v) = (u as usize, v as usize);
            let deoc = if chi[u] >= chi[v] {
                if is_h[v] { DEOC_HYDROGEN } else { denom[v] }
            } else if is_h[u] {
                DEOC_HYDROGEN
            } else {
                denom[u]
            };
            let t = (chi[u] - chi[v]) / deoc;
            q[u] -= t * alpha;
            q[v] += t * alpha;
        }
    }
    q
}

/// CDK maximum bond order of an atom (`getMaximumBondOrder`): over the graph
/// bonds, SINGLE for a bond-less atom with implicit hydrogens, else UNSET (0).
#[inline]
fn max_order_with_h(view: &CdkView, a: usize, n_h: u8) -> u8 {
    let m = view.max_order[a];
    if m == 0 && n_h > 0 { 1 } else { m }
}

/// Charges on the hydrogen-suppressed graph (`AutocorrelationDescriptorCharge`).
/// `None` if an element is not parameterised.
pub(crate) fn charges_implicit_h(mol: &Molecule, view: &CdkView) -> Option<Vec<f64>> {
    let n = mol.num_atoms();
    let mut factors = Vec::with_capacity(n);
    let mut is_h = Vec::with_capacity(n);
    for a in 0..n {
        let atom = &mol.atoms[a];
        let mo = max_order_with_h(view, a, view.h[a]);
        factors.push(gm_factors(atom.atomic_number, mo, view.charge[a])?);
        is_h.push(atom.atomic_number == 1);
    }
    let bonds: Vec<(u32, u32)> = mol
        .bonds
        .iter()
        .map(|b| (b.source as u32, b.target as u32))
        .collect();
    Some(gasteiger_marsili(&factors, &is_h, &bonds))
}

/// Charges with all hydrogens explicit (`BCUTDescriptor`). Returns the charges
/// of the molecule's atoms and, per atom, the charge of its (identical) added
/// hydrogens. `None` if an element is not parameterised.
pub(crate) fn charges_explicit_h(mol: &Molecule, view: &CdkView) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = mol.num_atoms();
    let n_h: usize = view.h.iter().map(|&h| h as usize).sum();
    let mut factors = Vec::with_capacity(n + n_h);
    let mut is_h = Vec::with_capacity(n + n_h);
    for a in 0..n {
        let atom = &mol.atoms[a];
        let mo = max_order_with_h(view, a, view.h[a]).max(if view.h[a] > 0 { 1 } else { 0 });
        factors.push(gm_factors(atom.atomic_number, mo, view.charge[a])?);
        is_h.push(atom.atomic_number == 1);
    }
    let mut bonds: Vec<(u32, u32)> = Vec::with_capacity(mol.num_bonds() + n_h);
    bonds.extend(mol.bonds.iter().map(|b| (b.source as u32, b.target as u32)));
    let mut first_h = vec![u32::MAX; n];
    for a in 0..n {
        for k in 0..view.h[a] {
            let idx = factors.len() as u32;
            if k == 0 {
                first_h[a] = idx;
            }
            factors.push([7.17, 6.24, -0.56]);
            is_h.push(true);
            bonds.push((a as u32, idx));
        }
    }
    let q = gasteiger_marsili(&factors, &is_h, &bonds);
    let qh = (0..n)
        .map(|a| {
            if first_h[a] == u32::MAX {
                0.0
            } else {
                q[first_h[a] as usize]
            }
        })
        .collect();
    Some((q[..n].to_vec(), qh))
}
