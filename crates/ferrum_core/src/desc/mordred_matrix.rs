//! Mordred (mordred-community 2.0.7) matrix descriptors on the
//! hydrogen-suppressed graph (`Chem.RemoveHs`, FerrumChem's molecule; aromatic
//! bonds not kekulized):
//!
//! - `AdjacencyMatrix` (`*_A`, 12 values): adjacency matrix.
//! - `BaryszMatrix` (`*_Dz<p>`, 8 x 13 values): Barysz distance matrix
//!   weighted by the atomic property `p` in `Z, m, v, se, pe, are, p, i`
//!   (`mordred_data.rs`): the shortest-path matrix of the graph whose bond
//!   `(i, j)` weighs `C² / (P_i P_j π_ij)`, diagonal `1 - C / P_i` (`C` the
//!   value of carbon, `π_ij` RDKit's `GetBondTypeAsDouble`, 1.5 aromatic).
//!   Missing when an atom has no value for `p` (or a weight divides by zero).
//! - `DetourMatrix` (`*_Dt`, 13 values, plus `DetourIndex = ½ Σ Δ`): matrix of
//!   longest simple path lengths (`algo::detour`).
//! - `DistanceMatrix` (`*_D`, 12 values): topological distance matrix.
//!
//! Every matrix `M` is reduced by Mordred's `_matrix_attributes` with the
//! eigenvalues `λ` (`numpy.linalg.eigh`) and the unit eigenvector `v` of the
//! largest one (`algo::eigen_full`), `A` atoms:
//! `SpAbs = Σ|λ|`, `SpMax = λmax`, `SpDiam = λmax - λmin`,
//! `SpAD = Σ|λ - mean(λ)|`, `SpMAD = SpAD / A`,
//! `LogEE = ln(Σ exp λ + 1)` (evaluated as `a + ln(Σ exp(λ - a) + exp(-a))`,
//! `a = max(λmax, 0)`), `SM1 = tr M` (not for `A` and `D`),
//! `VE1 = Σ|v_i|`, `VE2 = VE1 / A`, `VE3 = ln(0.1 A VE1)`,
//! `VR1 = Σ_bonds (v_i v_j)^-½`, `VR2 = VR1 / A`, `VR3 = ln(0.1 A VR1)`
//! (`VE3` / `VR3` missing when the logarithm's argument is not positive).
//!
//! All four matrices are non-negative off the diagonal and irreducible for a
//! connected molecule, so (Perron-Frobenius) the largest eigenvalue is simple
//! and its eigenvector has entries of one sign: `VE*` and `VR*` do not depend
//! on numpy's arbitrary eigenvector sign. Mordred requires a connected
//! molecule; disconnected ones give missing values everywhere.
//!
//! Known deviation: `VR1` sums `(v_i v_j)^-½`, so it is dominated by the
//! smallest eigenvector entries. In large molecules with long chains far from
//! the most connected part, entries of the adjacency matrix's leading
//! eigenvector go down to ~1e-11; numpy's are only accurate to ~1e-17
//! absolute there, which moves Mordred's `VR1_A` / `VR2_A` (and `VR3_A`) by up
//! to ~1e-7 relative. FerrumChem's values agree with a 40-digit reference to
//! ~1e-11 or better in these cases (and to ~1e-15 typically).

use super::mordred_data::{
    MORDRED_ALLRED_ROCOW_EN, MORDRED_IONIZATION_POTENTIAL, MORDRED_MASS, MORDRED_PAULING_EN,
    MORDRED_POLARIZABILITY94, MORDRED_SANDERSON_EN, MORDRED_VDW_VOLUME,
};
use super::typed::{Descriptor, DescriptorValue};
use crate::algo::detour::BlockTree;
use crate::algo::eigen_full::{LaneEigen, SymEigen, solve_batch};
use crate::algo::paths::{DIST_INF, compute_unweighted_distance_matrix_cached};
use crate::desc::context::CalculationContext;
use crate::mol::molecule::Molecule;
use crate::types::BondType;
use std::cell::RefCell;

/// Mordred's matrix aggregating methods (`_matrix_attributes.methods`), in order.
const METHODS: [&str; N_METHODS] =
    ["SpAbs", "SpMax", "SpDiam", "SpAD", "SpMAD", "LogEE", "SM1", "VE1", "VE2", "VE3", "VR1", "VR2", "VR3"];
const N_METHODS: usize = 13;
/// Index of `SM1`, which Mordred's `AdjacencyMatrix` / `DistanceMatrix` presets drop.
const SM1: usize = 6;

/// Barysz matrix weights (`get_properties()`), in Mordred's order.
const BARYSZ_PROPS: [&str; 8] = ["Z", "m", "v", "se", "pe", "are", "p", "i"];
/// Element tables of `m, v, se, pe, are, p, i` (`Z` is the atomic number).
const BARYSZ_TABLES: [&[f64; 119]; 7] = [
    &MORDRED_MASS,
    &MORDRED_VDW_VOLUME,
    &MORDRED_SANDERSON_EN,
    &MORDRED_PAULING_EN,
    &MORDRED_ALLRED_ROCOW_EN,
    &MORDRED_POLARIZABILITY94,
    &MORDRED_IONIZATION_POTENTIAL,
];

const LEN_A: usize = N_METHODS - 1;
const LEN_DZ: usize = BARYSZ_PROPS.len() * N_METHODS;
const LEN_DT: usize = N_METHODS + 1;
/// Number of matrix descriptor columns.
pub const MORDRED_MATRIX_LEN: usize = LEN_A + LEN_DZ + LEN_DT + (N_METHODS - 1);

/// Column names in Mordred's order: `AdjacencyMatrix`, `BaryszMatrix`,
/// `DetourMatrix` (with `DetourIndex`), `DistanceMatrix`.
pub fn mordred_matrix_names() -> Vec<String> {
    let mut v = Vec::with_capacity(MORDRED_MATRIX_LEN);
    let skip_sm1 = |suffix: &str, v: &mut Vec<String>| {
        v.extend(METHODS.iter().enumerate().filter(|&(k, _)| k != SM1).map(|(_, m)| format!("{m}_{suffix}")));
    };
    skip_sm1("A", &mut v);
    for p in BARYSZ_PROPS {
        v.extend(METHODS.iter().map(|m| format!("{m}_Dz{p}")));
    }
    v.extend(METHODS.iter().map(|m| format!("{m}_Dt")));
    v.push("DetourIndex".to_string());
    skip_sm1("D", &mut v);
    v
}

/// The dense matrices of one molecule (decomposed in SIMD lanes; the
/// adjacency matrix is handled on its own).
#[derive(Clone, Copy)]
enum Job {
    Distance,
    /// Barysz matrix weighted by `BARYSZ_PROPS[p]`
    Barysz(usize),
    Detour,
}

/// Number of Barysz weightings.
const N_PROPS: usize = BARYSZ_PROPS.len();

/// Number of dense matrices: distance, 8 Barysz, detour.
const N_JOBS: usize = 2 + N_PROPS;

/// Upper bound on the matrix values held at once (decompositions are batched).
const MAX_BATCH_VALUES: usize = 1 << 20;

/// Largest `n` for which four matrices share a reduction pass (beyond it
/// two per pass are as fast: the working set outgrows the L1 cache).
const QUAD_MAX_N: usize = 56;

/// Largest molecule (atoms) whose buffers the per-thread [`Workspace`] keeps
/// for the next molecule (about 1 MB).
const KEEP_WORKSPACE_ATOMS: usize = 96;

/// Reusable buffers (one per thread).
#[derive(Default)]
struct Workspace {
    adj: SymEigen,
    groups: Vec<LaneEigen>,
    tree: BlockTree,
    dt: Vec<u32>,
    bonds: Vec<(u32, u32)>,
    /// per atom / bond / column: the eight Barysz weightings side by side
    prop: Vec<[f64; N_PROPS]>,
    weight: Vec<[f64; N_PROPS]>,
    row: Vec<[f64; N_PROPS]>,
}

thread_local! {
    static WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::default());
}

/// RDKit `Bond::getBondTypeAsDouble`.
#[inline]
fn bond_order(t: BondType) -> f64 {
    match t {
        BondType::Single | BondType::Dative => 1.0,
        BondType::Double => 2.0,
        BondType::Triple => 3.0,
        BondType::Aromatic => 1.5,
        BondType::Unknown => 0.0,
    }
}

/// `ln x` as Mordred's `rethrow_zerodiv` sees it: missing unless `x > 0`.
#[inline]
fn checked_ln(x: f64) -> f64 {
    if x > 0.0 { x.ln() } else { f64::NAN }
}

/// Mordred's 13 matrix attributes from the eigenvalues `w`, the smallest and
/// largest one and the leading eigenvector (`v(i)`: entry of atom `i`).
fn attributes(
    w: &[f64],
    (sp_min, sp_max): (f64, f64),
    v: impl Fn(usize) -> f64,
    trace: f64,
    bonds: &[(u32, u32)],
) -> [f64; N_METHODS] {
    let n_atoms = w.len();
    let n = n_atoms as f64;
    let sp_abs: f64 = w.iter().map(|x| x.abs()).sum();
    let mean = w.iter().sum::<f64>() / n;
    let sp_ad: f64 = w.iter().map(|x| (x - mean).abs()).sum();
    let a = sp_max.max(0.0);
    let sx = w.iter().map(|x| (x - a).exp()).sum::<f64>() + (-a).exp();
    let ve1: f64 = (0..n_atoms).map(|i| v(i).abs()).sum();
    let vr1: f64 = bonds.iter().map(|&(i, j)| (v(i as usize) * v(j as usize)).powf(-0.5)).sum();
    [
        sp_abs,
        sp_max,
        sp_max - sp_min,
        sp_ad,
        sp_ad / n,
        a + sx.ln(),
        trace,
        ve1,
        ve1 / n,
        checked_ln(0.1 * n * ve1),
        vr1,
        vr1 / n,
        checked_ln(0.1 * n * vr1),
    ]
}

/// Mordred matrix descriptor values in [`mordred_matrix_names`] order.
pub fn mordred_matrix(mol: &Molecule, ctx: &mut CalculationContext) -> Vec<f64> {
    let mut out = vec![f64::NAN; MORDRED_MATRIX_LEN];
    let n = mol.num_atoms();
    if n == 0 {
        return out;
    }
    let dist = compute_unweighted_distance_matrix_cached(mol, ctx);
    if dist[..n].contains(&DIST_INF) {
        return out; // disconnected (Mordred `require_connected`)
    }
    WORKSPACE.with(|ws| {
        let mut ws = ws.borrow_mut();
        fill(mol, dist, &mut ws, MAX_BATCH_VALUES, &mut out);
        if n > KEEP_WORKSPACE_ATOMS {
            // the buffers grow with n^2 (up to ~20 MB per thread): keep them
            // only at the size of typical molecules
            *ws = Workspace::default();
        }
    });
    out
}

/// Position of the element `(i, j)` (atoms) in the packed storage of a
/// [`LaneEigen`] with `w` lanes, lane `l`. The matrices are packed in reverse
/// atom order (atom `i` at index `n - 1 - i`), so that the Householder
/// reduction, which starts from the last index, starts from atom 0 like
/// numpy's (`dsytrd` on the lower triangle).
#[inline(always)]
fn pack_index(n: usize, w: usize, l: usize, i: usize, j: usize) -> usize {
    let (r, c) = (n - 1 - i.min(j), n - 1 - i.max(j));
    w * (r * (r + 1) / 2 + c) + l
}

/// Writes the 13 values of a matrix to its columns.
fn store(out: &mut [f64], job: Option<Job>, vals: &[f64; N_METHODS]) {
    let no_sm1 = |dst: &mut [f64]| {
        dst[..SM1].copy_from_slice(&vals[..SM1]);
        dst[SM1..].copy_from_slice(&vals[SM1 + 1..]);
    };
    let off_dt = LEN_A + LEN_DZ;
    match job {
        None => no_sm1(&mut out[..LEN_A]),
        Some(Job::Distance) => no_sm1(&mut out[off_dt + LEN_DT..]),
        Some(Job::Barysz(p)) => out[LEN_A + p * N_METHODS..][..N_METHODS].copy_from_slice(vals),
        Some(Job::Detour) => out[off_dt..off_dt + N_METHODS].copy_from_slice(vals),
    }
}

/// Fills `out`; at most `max_values` packed matrix values (or one group)
/// are decomposed at a time.
fn fill(mol: &Molecule, dist: &[u16], ws: &mut Workspace, max_values: usize, out: &mut [f64]) {
    let n = mol.num_atoms();
    let Workspace { adj, groups, tree, dt, bonds, prop, weight, row } = ws;
    bonds.clear();
    bonds.extend(mol.bonds.iter().map(|b| (b.source as u32, b.target as u32)));

    let neighbors = |a: usize| {
        mol.adjacency[a].iter().map(move |&bi| {
            let b = &mol.bonds[bi as usize];
            (if b.source as usize == a { b.target as usize } else { b.source as usize }, bi as usize)
        })
    };
    let tree_ok = tree.build(n, neighbors);
    dt.clear();
    dt.resize(n * n, 0);
    let detour_ok = tree_ok && tree.detour_matrix(dt);
    if detour_ok {
        out[LEN_A + LEN_DZ + N_METHODS] = (dt.iter().map(|&d| d as u64).sum::<u64>() / 2) as f64;
    }
    let barysz_ok = if tree_ok { barysz_weights(mol, prop, weight) } else { [false; N_PROPS] };
    if barysz_ok.contains(&true) {
        tree.weighted_local_distances(weight);
    }

    // The adjacency matrix: scalar reduction from column 0 (the Perron
    // vector of long chains has entries down to ~1e-11, which `VR1_A` sums
    // as `(v_i v_j)^-1/2`; this reduction keeps them most accurate).
    let m = adj.matrix_mut(n);
    m.fill(0.0);
    for &(i, j) in bonds.iter() {
        m[i as usize * n + j as usize] = 1.0;
        m[j as usize * n + i as usize] = 1.0;
    }

    // the dense matrices, two or four per SIMD pass (the Barysz matrices
    // first: all eight valid, weighting `p` is lane `p % w` of group `p / w`)
    let mut jobs = [Job::Distance; N_JOBS];
    let mut n_jobs = 0;
    for job in (0..N_PROPS).map(Job::Barysz).chain([Job::Distance, Job::Detour]) {
        let ok = match job {
            Job::Barysz(p) => barysz_ok[p],
            Job::Detour => detour_ok,
            Job::Distance => true,
        };
        if ok {
            jobs[n_jobs] = job;
            n_jobs += 1;
        }
    }
    let jobs = &jobs[..n_jobs];
    let quad = n <= QUAD_MAX_N;
    let mut layout = [(0usize, 0usize); N_JOBS]; // job -> (group, lane)
    let mut sizes = [(0usize, 0usize); N_JOBS]; // group -> (lanes, used)
    let mut n_groups = 0;
    let mut k = 0;
    while k < jobs.len() {
        let rest = jobs.len() - k;
        let lanes = if quad && rest >= 3 { 4 } else { 2 };
        let used = rest.min(lanes);
        for (l, slot) in layout[k..k + used].iter_mut().enumerate() {
            *slot = (n_groups, l);
        }
        sizes[n_groups] = (lanes, used);
        n_groups += 1;
        k += used;
    }
    // groups decomposed per call (bounded memory); group `g` of a call
    // starting at group `g0` uses workspace `g - g0`
    let per_call = (max_values / (4 * n * (n + 1) / 2)).max(1);
    if groups.len() < per_call.min(n_groups) {
        groups.resize_with(per_call.min(n_groups), LaneEigen::new);
    }
    let c = barysz_carbon();
    for g0 in (0..n_groups).step_by(per_call) {
        let g1 = (g0 + per_call).min(n_groups);
        let in_call = |k: usize| (g0..g1).contains(&layout[k].0);
        // packed storage of the groups of this call
        let mut packed: [&mut [f64]; N_JOBS] = Default::default();
        for (slot, (eig, &(lanes, used))) in packed.iter_mut().zip(groups.iter_mut().zip(&sizes[g0..g1])) {
            *slot = eig.packed_mut(n, lanes, used);
        }
        let mut barysz = false;
        for (k, &job) in jobs.iter().enumerate() {
            if !in_call(k) {
                continue;
            }
            let (g, l) = layout[k];
            let (w, a) = (sizes[g].0, &mut *packed[g - g0]);
            match job {
                Job::Distance => pack_rows(n, w, l, a, |i, j| dist[i * n + j] as f64),
                Job::Detour => pack_rows(n, w, l, a, |i, j| dt[i * n + j] as f64),
                Job::Barysz(_) => barysz = true,
            }
        }
        if barysz {
            // row by row, all weightings at once; the lower triangle
            // (numpy's `eigh` reads that one; the assembled matrix is
            // symmetric up to rounding)
            row.clear();
            row.resize(n, [0.0; N_PROPS]);
            let w = sizes[0].0;
            let aligned = g0 == 0 && N_PROPS / w <= g1 && barysz_ok.iter().all(|&ok| ok);
            for s in 0..n {
                let diag = std::array::from_fn(|p| 1.0 - c[p] / prop[s][p]);
                tree.assemble_row(s, diag, row);
                if aligned {
                    for (t, x) in row[..=s].iter().enumerate() {
                        let at = pack_index(n, w, 0, s, t);
                        for (a, x) in packed.iter_mut().zip(x.chunks_exact(w)) {
                            a[at..at + w].copy_from_slice(x);
                        }
                    }
                    continue;
                }
                for (k, &job) in jobs.iter().enumerate() {
                    let Job::Barysz(p) = job else { continue };
                    if !in_call(k) {
                        continue;
                    }
                    let (g, l) = layout[k];
                    let (w, a) = (sizes[g].0, &mut *packed[g - g0]);
                    for (t, x) in row[..=s].iter().enumerate() {
                        a[pack_index(n, w, l, s, t)] = x[p];
                    }
                }
            }
        }
        let singles = if g0 == 0 { std::slice::from_mut(adj) } else { &mut [] };
        solve_batch(singles, &mut groups[..g1 - g0]);
        if g0 == 0 {
            let vals = if adj.is_ok() {
                let v = adj.max_vector();
                let w = adj.values();
                attributes(w, (w[0], w[n - 1]), |i| v[i], 0.0, bonds)
            } else {
                [f64::NAN; N_METHODS]
            };
            store(out, None, &vals);
        }
        for (k, &job) in jobs.iter().enumerate() {
            if !in_call(k) {
                continue;
            }
            let (g, l) = layout[k];
            let eig = &groups[g - g0];
            let vals = if eig.is_ok(l) {
                let trace = match job {
                    Job::Barysz(p) => prop.iter().map(|x| 1.0 - c[p] / x[p]).sum(),
                    _ => 0.0,
                };
                let v = eig.max_vector(l);
                attributes(eig.values(l), eig.extremes(l), |i| v[n - 1 - i], trace, bonds)
            } else {
                [f64::NAN; N_METHODS]
            };
            store(out, Some(job), &vals);
        }
    }
}

/// Packs the symmetric matrix `m(i, j)` (atoms) into lane `l` of the `w`-lane
/// storage `a` (reverse atom order, see [`pack_index`]).
#[inline(always)]
fn pack_rows(n: usize, w: usize, l: usize, a: &mut [f64], m: impl Fn(usize, usize) -> f64) {
    for r in 0..n {
        // packed row r = atom n - 1 - r, columns 0..=r = atoms n - 1 down to n - 1 - r
        let i = n - 1 - r;
        let base = w * (r * (r + 1) / 2) + l;
        for c in 0..=r {
            a[base + w * c] = m(i, n - 1 - c);
        }
    }
}

/// Value of carbon for each Barysz weighting.
fn barysz_carbon() -> [f64; N_PROPS] {
    std::array::from_fn(|p| barysz_value(p, 6))
}

/// Value of the element `z` for Barysz weighting `p` (NaN when missing).
#[inline]
fn barysz_value(p: usize, z: usize) -> f64 {
    if p == 0 { z as f64 } else { BARYSZ_TABLES[p - 1].get(z).copied().unwrap_or(f64::NAN) }
}

/// Atom values (`prop`) and bond weights `C² / (P_i P_j π_ij)` (`weight`)
/// of the eight Barysz weightings; which of them Mordred does not report as
/// missing (a missing atom value or a zero denominator).
fn barysz_weights(mol: &Molecule, prop: &mut Vec<[f64; N_PROPS]>, weight: &mut Vec<[f64; N_PROPS]>) -> [bool; N_PROPS] {
    let c = barysz_carbon();
    prop.clear();
    prop.extend(mol.atoms.iter().map(|a| std::array::from_fn(|p| barysz_value(p, a.atomic_number as usize))));
    let mut ok: [bool; N_PROPS] = std::array::from_fn(|p| prop.iter().all(|x| !x[p].is_nan()));
    weight.clear();
    for b in &mol.bonds {
        let (pi, pj, bo) = (prop[b.source as usize], prop[b.target as usize], bond_order(b.bond_type));
        weight.push(std::array::from_fn(|p| {
            // Mordred: (C * C) / (P[i] * P[j] * pi), i / j = begin / end atom
            let den = pi[p] * pj[p] * bo;
            if den == 0.0 {
                ok[p] = false; // ZeroDivisionError
            }
            (c[p] * c[p]) / den
        }));
    }
    ok
}

/// Mordred AdjacencyMatrix, BaryszMatrix, DetourMatrix and DistanceMatrix
/// descriptor families (142 values, see the module docs).
pub struct MordredMatrix;

impl Descriptor for MordredMatrix {
    fn name(&self) -> &str {
        "MordredMatrix"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::FloatVec(mordred_matrix(mol, ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc(smiles: &str) -> Vec<f64> {
        let mol = parse_smiles(smiles).unwrap();
        mordred_matrix(&mol, &mut CalculationContext::new())
    }

    fn col(name: &str) -> usize {
        mordred_matrix_names().iter().position(|c| c == name).unwrap()
    }

    #[test]
    fn names_follow_mordred_order() {
        let names = mordred_matrix_names();
        assert_eq!(names.len(), MORDRED_MATRIX_LEN);
        assert_eq!(MORDRED_MATRIX_LEN, 142);
        assert_eq!(names[0], "SpAbs_A");
        assert_eq!(names[12], "SpAbs_DzZ");
        assert_eq!(names[LEN_A + LEN_DZ + N_METHODS], "DetourIndex");
        assert_eq!(names[MORDRED_MATRIX_LEN - 1], "VR3_D");
    }

    #[test]
    fn propane_values() {
        let v = calc("CCC");
        // P3 adjacency: eigenvalues -sqrt2, 0, sqrt2
        let s2 = 2f64.sqrt();
        assert!((v[col("SpMax_A")] - s2).abs() < 1e-14);
        assert!((v[col("SpAbs_A")] - 2.0 * s2).abs() < 1e-14);
        // detour == distance for a tree
        assert!((v[col("SpMax_Dt")] - v[col("SpMax_D")]).abs() < 1e-13);
        assert_eq!(v[col("DetourIndex")], 4.0);
        assert_eq!(v[col("SM1_Dt")], 0.0);
        // all-carbon Barysz matrix: diagonal 0
        assert_eq!(v[col("SM1_DzZ")], 0.0);
    }

    #[test]
    fn batched_calls_match_single_call() {
        // one group per call (small limit) gives the same values
        for smiles in ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "OCC1OC(O)C(O)C(O)C1O", "C1=CC=C2C=CC=CC2=C1", "C1CC1C2CCC2"] {
            let mol = parse_smiles(smiles).unwrap();
            let mut ctx = CalculationContext::new();
            let n = mol.num_atoms();
            let dist = compute_unweighted_distance_matrix_cached(&mol, &mut ctx).to_vec();
            let mut ws = Workspace::default();
            let (mut a, mut b) = (vec![f64::NAN; MORDRED_MATRIX_LEN], vec![f64::NAN; MORDRED_MATRIX_LEN]);
            fill(&mol, &dist, &mut ws, MAX_BATCH_VALUES, &mut a);
            fill(&mol, &dist, &mut ws, 1, &mut b);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a), bits(&b), "{smiles}");
            assert!(n > 0 && a.iter().filter(|x| x.is_nan()).count() < MORDRED_MATRIX_LEN);
        }
    }

    #[test]
    fn disconnected_and_missing() {
        assert!(calc("C.C").iter().all(|x| x.is_nan()));
        // methane: VR1 = 0, so VR3 = ln 0 is missing
        let v = calc("C");
        assert_eq!(v[col("VR1_A")], 0.0);
        assert!(v[col("VR3_A")].is_nan());
        assert!((v[col("LogEE_A")] - 2f64.ln()).abs() < 1e-15);
        // helium has no Sanderson electronegativity
        let v = calc("[He]");
        assert!(v[col("SpMax_Dzse")].is_nan());
    }
}
