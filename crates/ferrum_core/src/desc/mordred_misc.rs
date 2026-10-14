//! Descriptor module: Mordred small descriptor modules.
//!
//! Ports of the small mordred-community 2.0.7 modules that only produce a
//! handful of columns each (ABCIndex, AcidBase, AtomCount, BalabanJ,
//! BondCount, CPSA 2D, CarbonTypes, EccentricConnectivityIndex,
//! FragmentComplexity, Framework, KappaShapeIndex, Lipinski, LogS,
//! McGowanVolume, Polarizability, RotatableBond, TopoPSA, TopologicalIndex,
//! VdwVolumeABC, VertexAdjacencyInformation, Weight, WienerIndex,
//! ZagrebIndex). Columns that an existing FerrumChem column already covers
//! are not repeated (see `mordred_names.rs`).
//!
//! Mordred evaluates each descriptor either on the hydrogen-suppressed
//! molecule (`Chem.RemoveHs`, which is FerrumChem's molecule: isotopic H and
//! the like stay atoms) or on the molecule after `Chem.AddHs`. The latter is
//! never built here: every implicit hydrogen is accounted for analytically
//! (one atom and one single bond each, appended after the heavy atoms like
//! `AddHs` does).
//!
//! Distances use RDKit's `GetDistanceMatrix`, in which unreachable pairs
//! (disconnected molecules) are `1e8`; Mordred inherits that value.

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::paths::{DIST_INF, compute_unweighted_distance_matrix_cached};
use crate::mol::molecule::Molecule;
use crate::types::{BondType, Hybridization};

/// RDKit `GetDistanceMatrix` distance between disconnected atoms.
const RDKIT_UNREACHABLE: f64 = 1e8;

/// Number of columns of [`MordredMisc`].
pub const MORDRED_MISC_LEN: usize = 50;

/// Output columns of [`MordredMisc`], in Mordred's descriptor order.
pub const MORDRED_MISC_NAMES: [&str; MORDRED_MISC_LEN] = [
    "ABC",
    "ABCGG",
    "nAcid",
    "nBase",
    "Mordred_nHeavyAtom",
    "Mordred_nH",
    "nB",
    "nX",
    "Mordred_BalabanJ",
    "Mordred_nBonds",
    "Mordred_nBondsS",
    "nBondsM",
    "nBondsKS",
    "nBondsKD",
    "RNCG",
    "RPCG",
    "Mordred_C1SP1",
    "Mordred_C2SP1",
    "Mordred_C1SP2",
    "Mordred_C2SP2",
    "Mordred_C3SP2",
    "Mordred_C1SP3",
    "Mordred_C2SP3",
    "Mordred_C3SP3",
    "Mordred_C4SP3",
    "Mordred_HybRatio",
    "ECIndex",
    "fragCpx",
    "fMF",
    "Mordred_Kier1",
    "Mordred_Kier2",
    "Mordred_Kier3",
    "Lipinski",
    "GhoseFilter",
    "FilterItLogS",
    "VMcGowan",
    "Mordred_apol",
    "Mordred_bpol",
    "RotRatio",
    "TopoPSA",
    "TopoShapeIndex",
    "PetitjeanIndex",
    "Vabc",
    "Mordred_VAdjMat",
    "AMW",
    "WPath",
    "WPol",
    "Mordred_Zagreb1",
    "mZagreb1",
    "mZagreb2",
];

/// Mordred `VdwVolumeABC.bondi_radii` as atomic volumes 4/3·π·r³.
fn bondi_volume(z: u8) -> f64 {
    let r: f64 = match z {
        1 => 1.20,
        6 => 1.70,
        7 => 1.55,
        8 => 1.52,
        9 => 1.47,
        17 => 1.75,
        35 => 1.85,
        15 => 1.80,
        16 => 1.80,
        33 => 1.85,
        5 => 2.13,
        14 => 2.10,
        34 => 1.90,
        _ => return f64::NAN,
    };
    4.0 / 3.0 * std::f64::consts::PI * r.powf(3.0)
}

#[inline]
fn other(mol: &Molecule, bi: u16, a: usize) -> usize {
    let b = &mol.bonds[bi as usize];
    if b.source as usize == a { b.target as usize } else { b.source as usize }
}

#[inline]
fn b2f(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

/// Mordred small-module descriptors; columns in [`MORDRED_MISC_NAMES`].
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct MordredMisc;

impl Descriptor for MordredMisc {
    fn name(&self) -> &str {
        "MordredMisc"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let mut out = vec![f64::NAN; MORDRED_MISC_LEN];
        if mol.num_atoms() > 0 {
            fill(mol, ctx, &mut out);
        }
        DescriptorValue::FloatVec(out)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MordredMisc>;
};

/// Graph quantities from the topological distance matrix (hydrogen-suppressed
/// molecule, RDKit `GetDistanceMatrix` semantics).
struct DistanceStats {
    /// `int(0.5 * D.sum())` (Mordred `WPath`).
    wpath: f64,
    /// Number of atom pairs at distance 3 (Mordred `WPol`).
    wpol: f64,
    /// Smallest / largest eccentricity (Mordred `Radius` / `Diameter`).
    radius: f64,
    diameter: f64,
    /// Σ eccentricity · degree (Mordred `ECIndex`).
    ec_index: f64,
    /// RDKit `BalabanJ(mol, dMat=D)` on the topological distance matrix.
    balaban_j: f64,
    /// Mordred `ABCGG`.
    abcgg: f64,
}

fn distance_stats(mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DistanceStats {
    let n = mol.num_atoms();
    let dist = compute_unweighted_distance_matrix_cached(mol, ctx);
    // Row sums (= column sums, D is symmetric) and eccentricities, in integer
    // arithmetic; an unreachable pair (DIST_INF) counts as RDKit's 1e8.
    let unreachable = RDKIT_UNREACHABLE as u64;
    let mut sums = vec![0.0f64; n];
    let mut total = 0u64;
    let mut n_three = 0u64;
    let mut radius = u64::MAX;
    let mut diameter = 0u64;
    let mut ec_index = 0u64;
    for (i, sum) in sums.iter_mut().enumerate() {
        let row = &dist[i * n..(i + 1) * n];
        let (mut s, mut n_inf, mut three, mut max) = (0u64, 0u64, 0u64, 0u16);
        for &v in row {
            let inf = (v == DIST_INF) as u64;
            s += v as u64 * (1 - inf);
            n_inf += inf;
            three += (v == 3) as u64;
            max = max.max(v);
        }
        let s = s + n_inf * unreachable;
        let ecc = if max == DIST_INF { unreachable } else { max as u64 };
        *sum = s as f64;
        total += s;
        n_three += three;
        radius = radius.min(ecc);
        diameter = diameter.max(ecc);
        ec_index += ecc * mol.adjacency[i].len() as u64;
    }

    // RDKit GraphDescriptors.BalabanJ with Mordred's (non bond-order) dMat.
    let q = mol.num_bonds() as i64;
    let mu1 = q - n as i64 + 2;
    let balaban_j = if mu1 != 0 {
        let mut acc = 0.0f64;
        let mut nbrs: smallvec::SmallVec<[usize; 8]> = smallvec::SmallVec::new();
        for i in 0..n {
            nbrs.clear();
            nbrs.extend(mol.adjacency[i].iter().map(|&bi| other(mol, bi, i)).filter(|&j| j > i));
            nbrs.sort_unstable();
            for &j in &nbrs {
                acc += 1.0 / (sums[i] * sums[j]).sqrt();
            }
        }
        q as f64 / mu1 as f64 * acc
    } else {
        0.0
    };

    // Graovac-Ghorbani ABC: atoms closer to u than to v and vice versa.
    let mut abcgg = 0.0f64;
    for b in &mol.bonds {
        let (u, v) = (b.source as usize, b.target as usize);
        // u and v share their component, so unreachable (DIST_INF) never decides
        let (mut nu, mut nv) = (0u32, 0u32);
        for (&du, &dv) in dist[u * n..(u + 1) * n].iter().zip(&dist[v * n..(v + 1) * n]) {
            nu += (du < dv) as u32;
            nv += (dv < du) as u32;
        }
        let (nu, nv) = (nu as f64, nv as f64);
        abcgg += ((nu + nv - 2.0) / (nu * nv)).sqrt();
    }

    DistanceStats {
        wpath: (total / 2) as f64,
        wpol: (n_three / 2) as f64,
        radius: radius as f64,
        diameter: diameter as f64,
        ec_index: ec_index as f64,
        balaban_j,
        abcgg,
    }
}

fn fill(mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext, out: &mut [f64]) {
    use crate::desc::mordred_data::{MORDRED_MCGOWAN_VOLUME, MORDRED_POLARIZABILITY94};

    let n = mol.num_atoms();
    let n_bonds = mol.num_bonds();
    // Hydrogens Chem.AddHs would add, and the explicit-H atom / bond counts.
    let n_implicit_h: usize = mol.atoms.iter().map(|a| a.total_h() as usize).sum();
    let n_atoms_h = n + n_implicit_h;
    let n_bonds_h = n_bonds + n_implicit_h;
    let degree = |a: usize| mol.adjacency[a].len();

    let ds = distance_stats(mol, ctx);

    // --- ABCIndex -------------------------------------------------------
    let mut abc = 0.0f64;
    for b in &mol.bonds {
        let du = degree(b.source as usize) as f64;
        let dv = degree(b.target as usize) as f64;
        abc += ((du + dv - 2.0) / (du * dv)).sqrt();
    }
    out[0] = abc;
    out[1] = ds.abcgg;

    // --- AcidBase -------------------------------------------------------
    let (n_acid, n_base) = acid_base_counts(mol);
    out[2] = n_acid as f64;
    out[3] = n_base as f64;

    // --- AtomCount ------------------------------------------------------
    out[4] = n as f64;
    let n_h_atoms = mol.atoms.iter().filter(|a| a.atomic_number == 1).count();
    out[5] = (n_implicit_h + n_h_atoms) as f64;
    out[6] = mol.atoms.iter().filter(|a| a.atomic_number == 5).count() as f64;
    out[7] = mol
        .atoms
        .iter()
        .filter(|a| matches!(a.atomic_number, 9 | 17 | 35 | 53 | 85 | 117))
        .count() as f64;

    // --- BalabanJ -------------------------------------------------------
    out[8] = ds.balaban_j;

    // --- BondCount ------------------------------------------------------
    let (mut n_single, mut n_double, mut n_multiple, mut n_arom) = (0usize, 0usize, 0usize, 0usize);
    for b in &mol.bonds {
        match b.bond_type {
            BondType::Single => n_single += 1,
            BondType::Double => n_double += 1,
            BondType::Aromatic => n_arom += 1,
            _ => {}
        }
        if b.is_aromatic() || b.bond_type != BondType::Single {
            n_multiple += 1;
        }
    }
    out[9] = n_bonds_h as f64;
    out[10] = (n_single + n_implicit_h) as f64;
    out[11] = n_multiple as f64;
    // Chem.Kekulize: aromatic bonds become single / double bonds.
    let kekule_double = if n_arom > 0 { kekule_double_bonds(mol) } else { 0 };
    out[12] = (n_single + n_implicit_h + n_arom - kekule_double) as f64;
    out[13] = (n_double + kekule_double) as f64;

    // --- CPSA (2D): RNCG / RPCG on Gasteiger charges with explicit H ----
    let (rncg, rpcg) = relative_charges(mol);
    out[14] = rncg;
    out[15] = rpcg;

    // --- CarbonTypes (C1SP1 .. C4SP3, HybRatio) ---------------------------
    // RDKit hybridization (CDK's CarbonTypes columns of the same names use
    // CDK atom typing, e.g. for carbanions, hence the Mordred_ prefix).
    let mut ct = [[0u32; 5]; 4]; // [SP][n carbon neighbours, 0..=4]
    for (a, atom) in mol.atoms.iter().enumerate() {
        if atom.atomic_number != 6 {
            continue;
        }
        let sp = match atom.hybridization {
            Hybridization::SP => 1,
            Hybridization::SP2 => 2,
            Hybridization::SP3 | Hybridization::SP3D | Hybridization::SP3D2 => 3,
            _ => continue,
        };
        let nc = mol.adjacency[a].iter().filter(|&&bi| mol.atoms[other(mol, bi, a)].atomic_number == 6).count();
        if nc <= 4 {
            ct[sp][nc] += 1;
        }
    }
    out[16] = ct[1][1] as f64;
    out[17] = ct[1][2] as f64;
    out[18] = ct[2][1] as f64;
    out[19] = ct[2][2] as f64;
    out[20] = ct[2][3] as f64;
    out[21] = ct[3][1] as f64;
    out[22] = ct[3][2] as f64;
    out[23] = ct[3][3] as f64;
    out[24] = ct[3][4] as f64;
    let (n_sp2, n_sp3) = (ct[2].iter().sum::<u32>(), ct[3].iter().sum::<u32>());
    out[25] = if n_sp2 + n_sp3 == 0 { f64::NAN } else { n_sp3 as f64 / (n_sp2 + n_sp3) as f64 };

    // --- EccentricConnectivityIndex / FragmentComplexity / Framework ----
    out[26] = ds.ec_index;
    let a = n as i64;
    let b = n_bonds as i64;
    let hetero = mol.atoms.iter().filter(|x| x.atomic_number != 6).count();
    out[27] = (b * b - a * a + a).abs() as f64 + hetero as f64 / 100.0;
    out[28] = framework_atoms(mol) as f64 / n_atoms_h as f64;

    // --- KappaShapeIndex (Kier1..3) -------------------------------------
    let (k1, k2, k3) = kier(mol);
    out[29] = k1;
    out[30] = k2;
    out[31] = k3;

    // --- Lipinski / GhoseFilter -----------------------------------------
    let mw = crate::desc::rdkit_extra::exact_mol_wt(mol);
    let (logp, mr) = crate::desc::logp::compute_logp_cached(mol);
    let hbd = crate::desc::hbond::compute_hbond_donors_cached(mol);
    let hba = crate::desc::hbond::compute_hbond_acceptors_cached(mol);
    out[32] = b2f(hbd <= 5 && hba <= 10 && mw <= 500.0 && logp <= 5.0);
    out[33] = b2f(
        (160.0..=480.0).contains(&mw)
            && (20..=70).contains(&n_atoms_h)
            && (-0.4..=5.6).contains(&logp)
            && (40.0..=130.0).contains(&mr),
    );

    // --- LogS (Filter-it) -----------------------------------------------
    let avg_mw = crate::desc::molecular_weight::compute_molecular_weight(mol);
    let mut logs = 0.89823 - 0.10369 * avg_mw.sqrt();
    for (count, w) in logs_counts(mol).into_iter().zip(LOGS_WEIGHTS) {
        logs += count as f64 * w;
    }
    out[34] = logs;

    // --- McGowanVolume / Polarizability / VdwVolumeABC ------------------
    // Sums over the explicit-H molecule: heavy atoms first, then the added
    // hydrogens in parent-atom order (Chem.AddHs).
    let atom_sum = |table: &dyn Fn(u8) -> f64| -> f64 {
        let mut s = 0.0f64;
        for x in &mol.atoms {
            s += table(x.atomic_number);
        }
        let h = table(1);
        for _ in 0..n_implicit_h {
            s += h;
        }
        s
    };
    let mcgowan = atom_sum(&|z| MORDRED_MCGOWAN_VOLUME[z as usize]);
    out[35] = mcgowan - n_bonds_h as f64 * 6.56;
    let pol = |z: u8| MORDRED_POLARIZABILITY94[z as usize];
    out[36] = atom_sum(&pol);
    let mut bpol = 0.0f64;
    for bd in &mol.bonds {
        bpol += (pol(mol.atoms[bd.source as usize].atomic_number) - pol(mol.atoms[bd.target as usize].atomic_number)).abs();
    }
    for x in &mol.atoms {
        let d = (pol(x.atomic_number) - pol(1)).abs();
        for _ in 0..x.total_h() {
            bpol += d;
        }
    }
    out[37] = bpol;

    // --- RotatableBond ---------------------------------------------------
    let n_rot = crate::desc::bond_count::compute_rotatable_bond_count_cached(mol);
    out[38] = if n_bonds == 0 { f64::NAN } else { n_rot as f64 / n_bonds as f64 };

    // --- TopoPSA (N, O, S, P) --------------------------------------------
    out[39] = topo_psa(mol);

    // --- TopologicalIndex -----------------------------------------------
    out[40] = if ds.radius == 0.0 { f64::NAN } else { (ds.diameter - ds.radius) / ds.radius };
    out[41] = if ds.diameter == 0.0 { f64::NAN } else { (ds.diameter - ds.radius) / ds.diameter };

    // --- VdwVolumeABC ---------------------------------------------------
    let (mut n_arom_rings, mut n_aliph_rings) = (0usize, 0usize);
    if let Some(info) = mol.ring_info() {
        for r in &info.atom_rings {
            if r.iter().all(|&x| mol.atoms[x as usize].is_aromatic()) {
                n_arom_rings += 1;
            } else {
                n_aliph_rings += 1;
            }
        }
    }
    let ac = atom_sum(&bondi_volume);
    out[42] = ac - 5.92 * n_bonds_h as f64 - 14.7 * n_arom_rings as f64 - 3.8 * n_aliph_rings as f64;

    // --- VertexAdjacencyInformation -------------------------------------
    let m = mol
        .bonds
        .iter()
        .filter(|bd| mol.atoms[bd.source as usize].atomic_number != 1 && mol.atoms[bd.target as usize].atomic_number != 1)
        .count();
    out[43] = if m == 0 { f64::NAN } else { 1.0 + (m as f64).log2() };

    // --- Weight (AMW) -----------------------------------------------------
    out[44] = mw / n_atoms_h as f64;

    // --- WienerIndex ------------------------------------------------------
    out[45] = ds.wpath;
    out[46] = ds.wpol;

    // --- ZagrebIndex (lambda = 1, -1) -------------------------------------
    let mut z1 = 0.0f64;
    let mut mz1 = 0.0f64;
    for x in 0..n {
        let d = degree(x) as f64;
        z1 += d * d;
        mz1 += 1.0 / (d * d);
    }
    let mut mz2 = 0.0f64;
    for bd in &mol.bonds {
        mz2 += 1.0 / (degree(bd.source as usize) as f64 * degree(bd.target as usize) as f64);
    }
    out[47] = z1;
    // numpy raises on 0 ** -2 (an isolated atom): Mordred reports it missing
    out[48] = if mz1.is_finite() { mz1 } else { f64::NAN };
    out[49] = mz2;
}

/// Hydrogen count of the SMARTS `H` primitive (RDKit `getTotalNumHs(true)`:
/// implicit/explicit H plus hydrogen-atom neighbours).
fn smarts_h(mol: &Molecule, a: usize) -> usize {
    mol.atoms[a].total_h() as usize
        + mol.adjacency[a].iter().filter(|&&bi| mol.atoms[other(mol, bi, a)].atomic_number == 1).count()
}

/// Neighbours of `a` reached through a bond of type `bt`.
fn nbrs_by(mol: &Molecule, a: usize, bt: BondType) -> impl Iterator<Item = usize> + '_ {
    mol.adjacency[a]
        .iter()
        .filter(move |&&bi| mol.bonds[bi as usize].bond_type == bt)
        .map(move |&bi| other(mol, bi, a))
}

/// Neighbours of `a` reached through a single or aromatic bond (the SMARTS
/// default bond).
fn nbrs_single_or_arom(mol: &Molecule, a: usize) -> impl Iterator<Item = usize> + '_ {
    mol.adjacency[a]
        .iter()
        .filter(move |&&bi| matches!(mol.bonds[bi as usize].bond_type, BondType::Single | BondType::Aromatic))
        .map(move |&bi| other(mol, bi, a))
}

/// Mordred `AcidBase` (`nAcid`, `nBase`): number of atoms matching the
/// recursive SMARTS
///
/// ```text
/// nAcid: [$([O;H1]-[C,S,P]=O),$([*;-;!$(*~[*;+])]),$([NH](S(=O)=O)C(F)(F)F),$(n1nnnc1)]
/// nBase: [$([NH2]-[CX4]),$([NH](-[CX4])-[CX4]),$(N(-[CX4])(-[CX4])-[CX4]),
///         $([*;+;!$(*~[*;-])]),$(N=C-N),$(N-C=N)]
/// ```
///
/// evaluated directly on the graph (Mordred matches them on the explicit-H
/// molecule; none of the primitives depends on that). The unit tests check
/// this against FerrumChem's SMARTS matcher.
fn acid_base_counts(mol: &Molecule) -> (usize, usize) {
    let atoms = &mol.atoms;
    let aliph = |a: usize, z: u8| atoms[a].atomic_number == z && !atoms[a].is_aromatic();
    let cx4 = |a: usize| aliph(a, 6) && mol.adjacency[a].len() + atoms[a].total_h() as usize == 4;
    let charged_nbr = |a: usize, q: i8| mol.adjacency[a].iter().any(|&bi| atoms[other(mol, bi, a)].formal_charge == q);
    let (mut n_acid, mut n_base) = (0, 0);
    for (a, atom) in atoms.iter().enumerate() {
        let acid = (atom.formal_charge == -1 && !charged_nbr(a, 1))
            || match (atom.atomic_number, atom.is_aromatic()) {
                // [O;H1]-[C,S,P]=O
                (8, false) => {
                    smarts_h(mol, a) == 1
                        && nbrs_by(mol, a, BondType::Single).any(|x| {
                            (aliph(x, 6) || aliph(x, 16) || aliph(x, 15))
                                && nbrs_by(mol, x, BondType::Double).any(|o| aliph(o, 8))
                        })
                }
                // [NH](S(=O)=O)C(F)(F)F
                (7, false) => {
                    smarts_h(mol, a) == 1
                        && nbrs_single_or_arom(mol, a).any(|x| {
                            aliph(x, 16) && nbrs_by(mol, x, BondType::Double).filter(|&o| aliph(o, 8)).count() >= 2
                        })
                        && nbrs_single_or_arom(mol, a)
                            .any(|x| aliph(x, 6) && nbrs_single_or_arom(mol, x).filter(|&f| aliph(f, 9)).count() >= 3)
                }
                // n1nnnc1
                (7, true) => tetrazole_start(mol, a),
                _ => false,
            };
        let base = (atom.formal_charge == 1 && !charged_nbr(a, -1))
            || (aliph(a, 7) && {
                let h = smarts_h(mol, a);
                let n_cx4 = nbrs_by(mol, a, BondType::Single).filter(|&x| cx4(x)).count();
                (h == 2 && n_cx4 >= 1)
                    || (h == 1 && n_cx4 >= 2)
                    || n_cx4 >= 3
                    // N=C-N, N-C=N
                    || nbrs_by(mol, a, BondType::Double)
                        .any(|c| aliph(c, 6) && nbrs_by(mol, c, BondType::Single).any(|x| aliph(x, 7)))
                    || nbrs_by(mol, a, BondType::Single)
                        .any(|c| aliph(c, 6) && nbrs_by(mol, c, BondType::Double).any(|x| aliph(x, 7)))
            });
        n_acid += acid as usize;
        n_base += base as usize;
    }
    (n_acid, n_base)
}

/// Whether the SMARTS `n1nnnc1` matches rooted at the aromatic nitrogen `s`:
/// a 5-cycle s-n-n-n-c of aromatic atoms over single or aromatic bonds.
fn tetrazole_start(mol: &Molecule, s: usize) -> bool {
    let arom = |a: usize, z: u8| mol.atoms[a].atomic_number == z && mol.atoms[a].is_aromatic();
    nbrs_single_or_arom(mol, s).any(|a| {
        arom(a, 7)
            && nbrs_single_or_arom(mol, a).any(|b| {
                b != s
                    && arom(b, 7)
                    && nbrs_single_or_arom(mol, b).any(|c| {
                        c != a
                            && c != s
                            && arom(c, 7)
                            && nbrs_single_or_arom(mol, c)
                                .any(|d| arom(d, 6) && nbrs_single_or_arom(mol, d).any(|e| e == s))
                    })
            })
    })
}

/// Mordred `LogS._smarts_logs` weights, in Mordred's (dict) order; the atom
/// patterns are listed in [`logs_counts`].
const LOGS_WEIGHTS: [f64; 16] = [
    0.71535, 0.41056, 0.82535, 0.31464, 0.14787, 0.62998, -0.35634, -0.33888, -0.21912, -0.23057, -0.37570,
    -0.22435, -0.21728, -0.49721, -0.57982, -0.51547,
];

/// Number of atoms matching each Filter-it LogS atom SMARTS
///
/// ```text
/// [NH0;X3;v3] [NH2;X3;v3] [nH0;X3] [OH0;X2;v2] [OH0;X1;v2] [OH1;X2;v2]
/// [CH2;!R] [CH3;!R] [CH0;R] [CH2;R] [ch0] [ch1] F Cl Br I
/// ```
///
/// with RDKit's primitives: `H` = total H incl. H neighbours, `h` = implicit
/// (+ explicit count) H, `X` = degree + H, `v` = total valence, `R` = ring
/// atom. The unit tests check this against FerrumChem's SMARTS matcher.
fn logs_counts(mol: &Molecule) -> [usize; 16] {
    let mut c = [0usize; 16];
    for (a, atom) in mol.atoms.iter().enumerate() {
        let x = mol.adjacency[a].len() + atom.total_h() as usize;
        let k = match (atom.atomic_number, atom.is_aromatic()) {
            (7, false) => match (smarts_h(mol, a), x, crate::algo::sanitize::total_valence(mol, a)) {
                (0, 3, 3) => 0,
                (2, 3, 3) => 1,
                _ => continue,
            },
            (7, true) if smarts_h(mol, a) == 0 && x == 3 => 2,
            (8, false) => match (smarts_h(mol, a), x, crate::algo::sanitize::total_valence(mol, a)) {
                (0, 2, 2) => 3,
                (0, 1, 2) => 4,
                (1, 2, 2) => 5,
                _ => continue,
            },
            (6, false) => match (smarts_h(mol, a), atom.is_in_ring()) {
                (2, false) => 6,
                (3, false) => 7,
                (0, true) => 8,
                (2, true) => 9,
                _ => continue,
            },
            (6, true) => match atom.total_h() {
                0 => 10,
                1 => 11,
                _ => continue,
            },
            (9, false) => 12,
            (17, false) => 13,
            (35, false) => 14,
            (53, false) => 15,
            _ => continue,
        };
        c[k] += 1;
    }
    c
}

/// Double bonds `Chem.Kekulize` puts on the aromatic bonds: every aromatic
/// atom that still needs one gets exactly one (the candidate rule of
/// `cdk_perception::kekule_orders`), so their number is half the candidates.
fn kekule_double_bonds(mol: &Molecule) -> usize {
    let mut need = 0usize;
    for (a, atom) in mol.atoms.iter().enumerate() {
        let mut base = atom.total_h() as i32;
        let mut n_arom = 0;
        for &bi in &mol.adjacency[a] {
            let b = &mol.bonds[bi as usize];
            base += match b.bond_type {
                BondType::Aromatic => {
                    n_arom += 1;
                    1
                }
                BondType::Single => 1,
                BondType::Double => 2,
                BondType::Triple => 3,
                BondType::Dative => (b.target as usize == a) as i32,
                BondType::Unknown => 0,
            };
        }
        if n_arom > 0 && crate::algo::sanitize::total_valence(mol, a) - base >= 1 {
            need += 1;
        }
    }
    if need.is_multiple_of(2) {
        need / 2
    } else {
        // no perfect matching (RDKit would not sanitize this): count what
        // the Kekulé assignment actually places
        let orders = crate::desc::cdk_perception::kekule_orders(mol);
        mol.bonds
            .iter()
            .zip(&orders)
            .filter(|(b, o)| b.bond_type == BondType::Aromatic && **o == 2)
            .count()
    }
}

/// Mordred `CPSA.RNCG` / `RPCG`: the most negative (positive) Gasteiger charge
/// divided by the sum of all negative (positive) charges. Charges come from
/// `ComputeGasteigerCharges` on the explicit-H molecule; Mordred reports the
/// descriptor as missing when any charge is NaN.
fn relative_charges(mol: &Molecule) -> (f64, f64) {
    let (charges, h_charges) = crate::algo::electronic::gasteiger_charges_with_hydrogens(mol, true);
    let mut neg_sum = 0.0f64;
    let mut neg_max = 0.0f64;
    let mut pos_sum = 0.0f64;
    let mut pos_max = 0.0f64;
    let mut any_nan = false;
    let mut visit = |q: f64| {
        if q.is_nan() {
            any_nan = true;
        } else if q < 0.0 {
            neg_sum += q;
            if q.abs() > neg_max.abs() {
                neg_max = q;
            }
        } else if q > 0.0 {
            pos_sum += q;
            if q.abs() > pos_max.abs() {
                pos_max = q;
            }
        }
    };
    for &q in &charges {
        visit(q);
    }
    for (a, atom) in mol.atoms.iter().enumerate() {
        let nh = atom.total_h();
        if nh > 0 {
            let q = h_charges[a]; // charge of each added hydrogen
            for _ in 0..nh {
                visit(q);
            }
        }
    }
    if any_nan {
        return (f64::NAN, f64::NAN);
    }
    let rel = |max: f64, sum: f64| if max == 0.0 { 0.0 } else { max / sum };
    (rel(neg_max, neg_sum), rel(pos_max, pos_sum))
}

/// Mordred `Framework`: number of ring atoms plus linker atoms (non-ring
/// atoms on a path between two rings).
///
/// Mordred collects the linkers from shortest paths between every pair of
/// (contracted) rings. A non-ring atom only has bridge bonds, so it lies on
/// such a path exactly when rings hang off at least two of its branches, i.e.
/// when it survives repeatedly pruning ring-free branches (non-ring atoms of
/// degree <= 1).
fn framework_atoms(mol: &Molecule) -> usize {
    let n = mol.num_atoms();
    let mut deg: Vec<u32> = (0..n).map(|a| mol.adjacency[a].len() as u32).collect();
    let mut removed = vec![false; n];
    let mut stack: Vec<usize> = (0..n).filter(|&a| !mol.atoms[a].is_in_ring() && deg[a] <= 1).collect();
    while let Some(x) = stack.pop() {
        if removed[x] {
            continue;
        }
        removed[x] = true;
        for &bi in &mol.adjacency[x] {
            let y = other(mol, bi, x);
            if removed[y] {
                continue;
            }
            deg[y] -= 1;
            if deg[y] <= 1 && !mol.atoms[y].is_in_ring() {
                stack.push(y);
            }
        }
    }
    removed.iter().filter(|&&r| !r).count()
}

/// Mordred `KappaShapeIndex1..3` (unweighted Kier shape indices).
///
/// `P_k` is the number of `k`-bond path subgraphs of RDKit
/// `FindAllSubgraphsOfLengthN` (bonds to hydrogen atoms excluded) that
/// Mordred's DFS classifies as paths: all bonds for `k = 1`, adjacent bond
/// pairs for `k = 2` and linear 3-bond chains that are not a 3-membered ring
/// for `k = 3`. `A` is the number of atoms of the hydrogen-suppressed
/// molecule; `P = 0` makes the index missing.
fn kier(mol: &Molecule) -> (f64, f64, f64) {
    let n = mol.num_atoms();
    let is_h = |a: usize| mol.atoms[a].atomic_number == 1;
    let heavy_nbrs = |a: usize| mol.adjacency[a].iter().map(move |&bi| other(mol, bi, a)).filter(move |&b| !is_h(b));
    let deg: Vec<i64> = (0..n).map(|a| if is_h(a) { 0 } else { heavy_nbrs(a).count() as i64 }).collect();
    let mut p1 = 0i64;
    let mut p3 = 0i64;
    for bd in &mol.bonds {
        let (u, v) = (bd.source as usize, bd.target as usize);
        if is_h(u) || is_h(v) {
            continue;
        }
        p1 += 1;
        // 3-membered rings through this bond (only ring bonds can have any)
        let common = if bd.is_in_ring() {
            heavy_nbrs(u).filter(|&w| w != v && heavy_nbrs(v).any(|x| x == w)).count() as i64
        } else {
            0
        };
        p3 += (deg[u] - 1) * (deg[v] - 1) - common;
    }
    let p2: i64 = deg.iter().map(|&d| d * (d - 1) / 2).sum();

    let a = n as i64;
    let af = a as f64;
    let kappa = |num: f64, p: i64| if p == 0 { f64::NAN } else { num / ((p * p) as f64) };
    let pmax1 = 0.5 * af * (a - 1) as f64;
    let k1 = kappa(2.0 * pmax1 * (a - 1) as f64, p1);
    let pmax2 = 0.5 * (a - 1) as f64 * (a - 2) as f64;
    let k2 = kappa(2.0 * pmax2 * (a - 2) as f64, p2);
    let pmax3 = if a % 2 == 0 {
        0.25 * ((a - 2) * (a - 2)) as f64
    } else {
        0.25 * (a - 1) as f64 * (a - 3) as f64
    };
    let k3 = kappa(4.0 * pmax3 * (a - 3) as f64, p3);
    (k1, k2, k3)
}

/// Mordred `TopoPSA(no_only=False)`: RDKit TPSA (N, O) plus Ertl's
/// phosphorus and sulfur contributions, evaluated on the explicit-H molecule
/// (implicit hydrogens count as single bonds).
fn topo_psa(mol: &Molecule) -> f64 {
    let mut tpsa = crate::desc::tpsa::compute_tpsa_cached(mol, false);
    for (a, atom) in mol.atoms.iter().enumerate() {
        let z = atom.atomic_number;
        if z != 15 && z != 16 {
            continue;
        }
        // bond type histogram (aromatic flag wins) and hydrogen count
        let (mut single, mut double, mut arom, mut others) = (atom.total_h() as u32, 0u32, 0u32, 0u32);
        let mut n_h = atom.total_h() as u32;
        for &bi in &mol.adjacency[a] {
            let b = &mol.bonds[bi as usize];
            if mol.atoms[other(mol, bi, a)].atomic_number == 1 {
                n_h += 1;
            }
            if b.is_aromatic() || b.bond_type == BondType::Aromatic {
                arom += 1;
            } else {
                match b.bond_type {
                    BondType::Single => single += 1,
                    BondType::Double => double += 1,
                    _ => others += 1,
                }
            }
        }
        let is = |s: u32, d: u32, ar: u32| others == 0 && single == s && double == d && arom == ar;
        let contrib = if atom.formal_charge != 0 {
            0.0
        } else if z == 15 {
            if atom.is_aromatic() {
                0.0
            } else if n_h == 1 && is(3, 1, 0) {
                23.47
            } else if n_h == 0 && is(3, 0, 0) {
                13.59
            } else if n_h == 0 && is(1, 1, 0) {
                34.14
            } else if n_h == 0 && is(3, 1, 0) {
                9.81
            } else {
                0.0
            }
        } else if atom.is_aromatic() {
            if n_h == 0 && is(0, 0, 2) {
                28.24
            } else if n_h == 0 && is(0, 1, 2) {
                21.70
            } else {
                0.0
            }
        } else if n_h == 1 && is(2, 0, 0) {
            38.80
        } else if n_h == 0 && is(2, 0, 0) {
            25.30
        } else if n_h == 0 && is(0, 1, 0) {
            32.09
        } else if n_h == 0 && is(2, 1, 0) {
            19.21
        } else if n_h == 0 && is(2, 2, 0) {
            8.38
        } else {
            0.0
        };
        tpsa += contrib;
    }
    tpsa
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc(smiles: &str) -> Vec<f64> {
        let mol = parse_smiles(smiles).unwrap();
        MordredMisc
            .calculate(&mol, &mut crate::desc::context::CalculationContext::new())
            .as_float_vec()
            .to_vec()
    }

    fn col(name: &str) -> usize {
        MORDRED_MISC_NAMES.iter().position(|n| *n == name).unwrap()
    }

    #[test]
    fn names_match_length() {
        assert_eq!(calc("CCO").len(), MORDRED_MISC_LEN);
    }

    #[test]
    fn counts_on_acetic_acid() {
        let v = calc("CC(=O)O");
        assert_eq!(v[col("nAcid")], 1.0);
        assert_eq!(v[col("Mordred_nH")], 4.0);
        assert_eq!(v[col("Mordred_nBonds")], 7.0);
        assert_eq!(v[col("Mordred_nBondsS")], 6.0);
        assert_eq!(v[col("nBondsKD")], 1.0);
    }

    #[test]
    fn benzene_kekule_and_kappa() {
        let v = calc("c1ccccc1");
        assert_eq!(v[col("nBondsKD")], 3.0);
        assert_eq!(v[col("nBondsKS")], 9.0);
        assert_eq!(v[col("nBondsM")], 6.0);
        // Kier1 = 2 * 15 * 5 / 36
        assert!((v[col("Mordred_Kier1")] - 150.0 / 36.0).abs() < 1e-12);
        assert_eq!(v[col("fMF")], 0.5);
    }

    #[test]
    fn isolated_atom_gives_missing_values() {
        let v = calc("[Na+].[Cl-]");
        assert!(v[col("mZagreb1")].is_nan());
        assert!(v[col("Mordred_VAdjMat")].is_nan());
        assert!(v[col("Mordred_Kier1")].is_nan());
        assert!(v[col("RotRatio")].is_nan());
    }

    /// The graph ports of the AcidBase and LogS SMARTS agree with
    /// FerrumChem's SMARTS matcher (Mordred: `len(GetSubstructMatches)`) on
    /// the parity molecules (or the SMILES file in `FERRUM_SMARTS_CHECK`).
    #[test]
    fn native_patterns_match_smarts() {
        use crate::algo::iso::{IsoTarget, count_matches_target};
        use crate::io::smarts::parse_smarts;
        let acid = parse_smarts("[$([O;H1]-[C,S,P]=O),$([*;-;!$(*~[*;+])]),$([NH](S(=O)=O)C(F)(F)F),$(n1nnnc1)]").unwrap();
        let base = parse_smarts(
            "[$([NH2]-[CX4]),$([NH](-[CX4])-[CX4]),$(N(-[CX4])(-[CX4])-[CX4]),$([*;+;!$(*~[*;-])]),$(N=C-N),$(N-C=N)]",
        )
        .unwrap();
        let logs: Vec<_> = [
            "[NH0;X3;v3]", "[NH2;X3;v3]", "[nH0;X3]", "[OH0;X2;v2]", "[OH0;X1;v2]", "[OH1;X2;v2]", "[CH2;!R]",
            "[CH3;!R]", "[CH0;R]", "[CH2;R]", "[ch0]", "[ch1]", "F", "Cl", "Br", "I",
        ]
        .iter()
        .map(|s| parse_smarts(s).unwrap())
        .collect();
        let path = std::env::var("FERRUM_SMARTS_CHECK")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/parity_molecules.smi").to_string());
        let extra = "c1nnn[nH]1\nCc1nn[n-]n1\nFC(F)(F)S(=O)(=O)NC\nNC=N\nCN=C(N)N\nC[N+](C)(C)C\n[O-]C(=O)C.[NH4+]\n[NH3+]CC([O-])=O\nOP(=O)(O)O\nOS(=O)(=O)c1ccccc1";
        let text = std::fs::read_to_string(&path).unwrap() + "\n" + extra;
        let mut n = 0;
        for line in text.lines() {
            let Some(smi) = line.split_whitespace().next() else { continue };
            let Ok(mol) = parse_smiles(smi) else { continue };
            let t = IsoTarget::new(&mol);
            let (na, nb) = acid_base_counts(&mol);
            assert_eq!(na, count_matches_target(&t, &acid), "nAcid {smi}");
            assert_eq!(nb, count_matches_target(&t, &base), "nBase {smi}");
            let c = logs_counts(&mol);
            for (k, q) in logs.iter().enumerate() {
                assert_eq!(c[k], count_matches_target(&t, q), "LogS pattern {k} {smi}");
            }
            n += 1;
        }
        assert!(n > 1000);
    }
}
