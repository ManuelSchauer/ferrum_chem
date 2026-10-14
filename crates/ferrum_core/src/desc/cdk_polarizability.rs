//! CDK `Polarizability`: Kang-Jhon atomic polarizability factors and the
//! Gasteiger-Hückel effective atom polarizability
//! (`calculateGHEffectiveAtomPolarizability` with a topological distance matrix).
//!
//! CDK evaluates it on a molecule with explicit hydrogens and `cdkLegacy`
//! aromaticity; hydrogens are folded in analytically here (an H on atom `j`
//! is at distance `d(i, j) + 1` from atom `i`).

use super::cdk_perception::CdkView;
use crate::algo::paths::DIST_INF;
use crate::mol::molecule::Molecule;

const H_FACTOR: f64 = 0.387;

/// `getKJPolarizabilityFactor` for a non-hydrogen atom of the explicit-H graph.
/// `charge` is the atom's partial charge (`None` when unset).
fn kj_factor(z: u8, aromatic: bool, max_order: u8, n_h: u8, charge: Option<f64>) -> f64 {
    match z {
        1 => H_FACTOR,
        6 => {
            if aromatic {
                1.230
            } else {
                match max_order {
                    1 => 1.064,
                    2 => {
                        if n_h == 0 {
                            1.382
                        } else {
                            1.37
                        }
                    }
                    3 | 4 => 1.279,
                    _ => 0.0,
                }
            }
        }
        7 => {
            if charge.is_some_and(|q| q < 0.0) {
                1.090
            } else {
                match max_order {
                    1 => 1.094,
                    2 => 1.030,
                    _ => 0.852,
                }
            }
        }
        8 => {
            if charge == Some(-1.0) {
                1.791
            } else if charge == Some(1.0) {
                0.422
            } else {
                match max_order {
                    1 => 0.664,
                    2 => 0.460,
                    _ => 0.0,
                }
            }
        }
        // P: 0 in every branch of CDK's switch (4-connected P=X and otherwise).
        16 => {
            if aromatic {
                3.38
            } else {
                match max_order {
                    1 => 3.20,
                    2 => {
                        if n_h == 0 {
                            3.51
                        } else {
                            3.50
                        }
                    }
                    _ => 3.42,
                }
            }
        }
        9 => 0.296,
        17 => 2.343,
        35 => 3.5,
        53 => 5.79,
        _ => 0.0,
    }
}

/// Effective polarizabilities of all atoms of `mol` (the molecule graph; its
/// own H atoms count as hydrogens). `dist` is the unweighted distance matrix
/// of `mol`; `charges` are partial charges (BCUT sets Gasteiger-Marsili charges
/// before computing polarizabilities, the autocorrelation leaves them unset).
pub(crate) fn effective_polarizabilities(
    mol: &Molecule,
    view: &CdkView,
    dist: &[u16],
    charges: Option<&[f64]>,
) -> Vec<f64> {
    let n = mol.num_atoms();
    let mut f = Vec::with_capacity(n);
    for a in 0..n {
        let atom = &mol.atoms[a];
        let h = view.h[a];
        // explicit-H graph: hydrogens add single bonds
        let max_order = if h > 0 {
            view.max_order[a].max(1)
        } else {
            view.max_order[a]
        };
        let n_h = h + mol.adjacency[a]
            .iter()
            .filter(|&&b| {
                let bd = &mol.bonds[b as usize];
                let o = if bd.source as usize == a {
                    bd.target
                } else {
                    bd.source
                };
                mol.atoms[o as usize].atomic_number == 1
            })
            .count() as u8;
        f.push(kj_factor(
            atom.atomic_number,
            view.arom_atom[a],
            max_order,
            n_h,
            charges.map(|q| q[a]),
        ));
    }
    if !f.iter().all(|v| v.is_finite()) {
        return effective_polarizabilities_skip(mol, view, dist, &f);
    }
    // Branch-free weights indexed by min(distance, n + 1): `w_heavy[d]` =
    // 0.5^(d-1) for another atom at distance d (0 for the atom itself),
    // `w_h[d]` = 0.5^d for its hydrogens; index n + 1 (disconnected,
    // `DIST_INF`) weighs 0.
    let cap = n + 1;
    let mut w_heavy = vec![0.0f64; cap + 1];
    let mut w_h = vec![0.0f64; cap + 1];
    let mut half = 1.0f64;
    for d in 0..cap {
        w_h[d] = half;
        if d + 1 < cap {
            w_heavy[d + 1] = half;
        }
        half *= 0.5;
    }
    let hw: Vec<f64> = view.h.iter().map(|&h| h as f64).collect();
    debug_assert!(DIST_INF as usize >= cap);
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let row = &dist[i * n..(i + 1) * n];
        let mut p = f[i];
        let mut hsum = 0.0;
        for ((&d, &fj), &hj) in row.iter().zip(&f).zip(&hw) {
            let d = (d as usize).min(cap);
            p += fj * w_heavy[d];
            hsum += hj * w_h[d];
        }
        out[i] = p + H_FACTOR * hsum;
    }
    out
}

/// Reference form of the distance-weighted sum that skips disconnected atoms
/// (used when some factor is not finite, so that `NaN * 0` cannot leak in).
fn effective_polarizabilities_skip(
    mol: &Molecule,
    view: &CdkView,
    dist: &[u16],
    f: &[f64],
) -> Vec<f64> {
    let n = mol.num_atoms();
    let mut half = vec![1.0f64; n + 2];
    for k in 1..half.len() {
        half[k] = half[k - 1] * 0.5;
    }
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let row = &dist[i * n..(i + 1) * n];
        let mut p = f[i];
        let mut hsum = 0.0;
        for j in 0..n {
            let d = row[j];
            if d == DIST_INF {
                continue;
            }
            let d = d as usize;
            if j != i {
                p += f[j] * half[d - 1];
            }
            if view.h[j] > 0 {
                hsum += view.h[j] as f64 * half[d];
            }
        }
        out[i] = p + H_FACTOR * hsum;
    }
    out
}
