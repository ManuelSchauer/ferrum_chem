//! Descriptor module: Mordred ExtendedTopochemicalAtom.
//!
//! Mordred (mordred-community 2.0.7) `ExtendedTopochemicalAtom`: the
//! extended topochemical atom (ETA) indices of Roy et al. (45 columns). Every
//! descriptor requires a single fragment (Mordred `require_connected`; NaN
//! otherwise) and works on the Kekulé form of the molecule.
//!
//! Atomic quantities (`_atomic_property.py`), with `Zv` RDKit's number of
//! outer electrons and `PN` the period:
//!
//! ```text
//! α_i  = (Z - Zv) / (Zv · (PN - 1))          (core count, 0 for hydrogen)
//! ε_i  = 0.3 · Zv - α_i
//! βσ_i = Σ_{heavy nbrs j} (0.5 if |ε_i - ε_j| ≤ 0.3 else 0.75)
//! βπ_i = Σ_{non-single bonds to heavy j} y · f
//!        f = 2 for triple bonds, else 1
//!        y = 2 (aromatic bond), 1.5 (|ε_i - ε_j| > 0.3), 1 (otherwise)
//! βδ_i = 0.5 if i is not aromatic, not in a ring, Zv > total valence and
//!        has an aromatic neighbour, else 0
//! γ_i  = α_i / (βσ_i + βπ_i + βδ_i)          (NaN if the sum is 0)
//! ```
//!
//! Aromatic bonds are Kekulé single (contributing nothing) or double bonds
//! that keep their aromatic flag (`y = 2`), so `βπ` only depends on which
//! atoms the Kekulé structure gives an aromatic double bond, not on the
//! particular structure.
//!
//! Mordred evaluates the descriptors on several molecules:
//!
//! - the hydrogen-suppressed molecule (`Chem.RemoveHs`, FerrumChem's graph:
//!   isotopic H and the like stay atoms) for α, shape, β, η, ε2, ψ;
//! - the hydrogen-explicit molecule (`Chem.AddHs`) for ε1 and ε5 (ε5 skips
//!   hydrogens bonded to carbon);
//! - the *reference alkane* (`AlterMolecule`): every non-hydrogen atom
//!   becomes a carbon, every bond between them a single bond; Mordred fails
//!   (NaN) when any atom has more than four neighbours (counted in the
//!   molecule it starts from, with or without explicit hydrogens). Used for
//!   η_R / η_RL (hydrogen-suppressed, `γ_i = 1 / degree`), α_R (0.5 per
//!   atom) and ε3 (hydrogen-explicit, `4 - degree` hydrogens per carbon);
//! - the *saturated* molecule for ε4: the heavy atoms with element and
//!   formal charge only, C-C bonds single, other bonds with their Kekulé
//!   order (`Chem.Kekulize(Chem.AddHs(mol))`, RDKit 2026.03 canonical), then
//!   `SanitizeMol` (implicit hydrogens from RDKit's valence model, NaN on a
//!   valence error) and `AddHs`.
//!
//! The molecular indices, with `A` the number of atoms of the molecule a
//! descriptor works on and `r_ij` the topological distance:
//!
//! ```text
//! α = Σ α_i                      shape_t = Σ_{degree(i) = 1, 3, 4} α_i / α
//! β_s = Σ βσ_i / 2               β_ns = Σ (βπ_i / 2 + βδ_i)
//! β = β_s + β_ns                 β_ns_d = Σ βδ_i
//! η = Σ_{i<j} sqrt(γ_i γ_j / r_ij²)       η_L: pairs with r_ij = 1
//! η_F = η_R - η                  η_FL = η_RL - η_L
//! η_B = η_NL - η_RL (+ 0.086 · nRing for η_BR),
//!       η_NL = 1 (A = 2), sqrt(2) + 0.5 (A - 3) (A > 2), NaN for A < 2
//! Δα_A = max((α - α_R) / A, 0)   Δα_B = max((α_R - α) / A, 0)
//! ε_k = mean ε_i over the atoms of the k-th molecule
//! Δε_A = ε1 - ε3, Δε_B = ε1 - ε4, Δε_C = ε3 - ε4, Δε_D = ε2 - ε5
//! Δβ = β_ns - β_s                ψ1 = α / (A · ε2)
//! Δψ_A = max(0.714 - ψ1, 0)      Δψ_B = max(ψ1 - 0.714, 0)
//! ```
//!
//! The `A`-prefixed columns divide by the number of atoms (of the reference
//! alkane for `AETA_eta_R` / `AETA_eta_RL`). Helium makes Mordred's core
//! count divide by zero: every value that uses its ε or α is NaN then.

use super::mordred_data::{MORDRED_PERIOD, RDKIT_N_OUTER_ELECS};
use super::typed::{Descriptor, DescriptorValue};
use crate::algo::paths::{DIST_INF, compute_unweighted_distance_matrix_cached};
use crate::desc::context::CalculationContext;
use crate::mol::molecule::Molecule;
use crate::mol::rdkit_data::ELEMENTS;
use crate::types::BondType;

/// Number of columns of [`MordredEta`].
pub const MORDRED_ETA_LEN: usize = 45;

/// Output columns of [`MordredEta`], in Mordred's descriptor order.
pub const MORDRED_ETA_NAMES: [&str; MORDRED_ETA_LEN] = [
    "ETA_alpha",
    "AETA_alpha",
    "ETA_shape_p",
    "ETA_shape_y",
    "ETA_shape_x",
    "ETA_beta",
    "AETA_beta",
    "ETA_beta_s",
    "AETA_beta_s",
    "ETA_beta_ns",
    "AETA_beta_ns",
    "ETA_beta_ns_d",
    "AETA_beta_ns_d",
    "ETA_eta",
    "AETA_eta",
    "ETA_eta_L",
    "AETA_eta_L",
    "ETA_eta_R",
    "AETA_eta_R",
    "ETA_eta_RL",
    "AETA_eta_RL",
    "ETA_eta_F",
    "AETA_eta_F",
    "ETA_eta_FL",
    "AETA_eta_FL",
    "ETA_eta_B",
    "AETA_eta_B",
    "ETA_eta_BR",
    "AETA_eta_BR",
    "ETA_dAlpha_A",
    "ETA_dAlpha_B",
    "ETA_epsilon_1",
    "ETA_epsilon_2",
    "ETA_epsilon_3",
    "ETA_epsilon_4",
    "ETA_epsilon_5",
    "ETA_dEpsilon_A",
    "ETA_dEpsilon_B",
    "ETA_dEpsilon_C",
    "ETA_dEpsilon_D",
    "ETA_dBeta",
    "AETA_dBeta",
    "ETA_psi_1",
    "ETA_dPsi_A",
    "ETA_dPsi_B",
];

/// RDKit `GetDistanceMatrix` distance between disconnected atoms.
const RDKIT_UNREACHABLE: f64 = 1e8;

/// Mordred `get_core_count` (NaN for helium, where Mordred divides by zero).
#[inline]
fn core_count(z: u8) -> f64 {
    if z == 1 {
        return 0.0;
    }
    let zv = RDKIT_N_OUTER_ELECS[z as usize] as f64;
    (z as f64 - zv) / (zv * (MORDRED_PERIOD[z as usize] - 1.0))
}

/// Mordred `get_eta_epsilon`.
#[inline]
fn epsilon(z: u8) -> f64 {
    0.3 * RDKIT_N_OUTER_ELECS[z as usize] as f64 - core_count(z)
}

/// Python `max(v, 0.0)`: NaN stays NaN.
#[inline]
fn py_max0(v: f64) -> f64 {
    if v.is_nan() || v >= 0.0 { v } else { 0.0 }
}

/// Python `a / b` for floats: division by zero is an error (NaN here).
#[inline]
fn py_div(a: f64, b: f64) -> f64 {
    if b == 0.0 { f64::NAN } else { a / b }
}

/// `Σ_{i<j} sqrt(γ_i γ_j) / r_ij` over all pairs and over bonded pairs, with
/// `dist(i, j)` the topological distance (`None` if unreachable).
fn eta_sums(sqrt_gamma: &[f64], dist: impl Fn(usize, usize) -> Option<u32>) -> (f64, f64) {
    let n = sqrt_gamma.len();
    let (mut all, mut local) = (0.0f64, 0.0f64);
    for i in 0..n {
        let gi = sqrt_gamma[i];
        let (mut row, mut row_local) = (0.0f64, 0.0f64);
        for (j, &gj) in sqrt_gamma.iter().enumerate().skip(i + 1) {
            match dist(i, j) {
                Some(1) => {
                    let v = gi * gj;
                    row += v;
                    row_local += v;
                }
                Some(r) => row += gi * gj / r as f64,
                None => row += gi * gj / RDKIT_UNREACHABLE,
            }
        }
        all += row;
        local += row_local;
    }
    (all, local)
}

/// [`eta_sums`] for two γ vectors over the same connected molecule, as
/// `[η(γ), η_L(γ), η(γ'), η_L(γ')]`; `dist` is the distance matrix.
fn eta_sums_pair(sqrt_gamma: &[f64], sqrt_gamma2: &[f64], dist: &[u16]) -> [f64; 4] {
    let n = sqrt_gamma.len();
    let mut acc = [0.0f64; 4];
    // the last row has no pairs (and a lone atom none at all)
    for i in 0..n.saturating_sub(1) {
        let row = &dist[i * n..(i + 1) * n];
        let (mut far, mut near, mut far2, mut near2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for j in i + 1..n {
            let r = row[j];
            if r == 1 {
                near += sqrt_gamma[j];
                near2 += sqrt_gamma2[j];
            } else {
                let inv = 1.0 / r as f64;
                far += sqrt_gamma[j] * inv;
                far2 += sqrt_gamma2[j] * inv;
            }
        }
        acc[0] += sqrt_gamma[i] * (near + far);
        acc[1] += sqrt_gamma[i] * near;
        acc[2] += sqrt_gamma2[i] * (near2 + far2);
        acc[3] += sqrt_gamma2[i] * near2;
    }
    acc
}

/// RDKit `canBeHypervalent` (`Atom.cpp`).
#[inline]
fn can_be_hypervalent(z: usize, eff: usize) -> bool {
    (eff > 16 && (z == 15 || z == 16)) || (eff > 34 && (z == 33 || z == 34))
}

/// Implicit hydrogen count RDKit's `SanitizeMol` gives a new, non-aromatic
/// atom (no explicit H, no radicals) of element `z` and formal charge
/// `charge` with explicit valence `ev`; `None` on a valence error
/// (`calculateExplicitValence` / `calculateImplicitValence`, strict).
fn sanitized_implicit_hs(z: u8, charge: i8, ev: i32) -> Option<i32> {
    let z = z as usize;
    if z == 0 {
        return Some(0);
    }
    let ovalens = ELEMENTS[z.min(118)].valences;
    let ovalens_last = *ovalens.last().unwrap() as i32;
    let mut eff = if ovalens.len() > 1 || ovalens[0] != -1 {
        (z as i32 - charge as i32).clamp(0, 118) as usize
    } else {
        z
    };
    // calculateExplicitValence (strict)
    let valens = ELEMENTS[eff].valences;
    let (mut max_valence, mut offset) = (*valens.last().unwrap() as i32, 0);
    if can_be_hypervalent(z, eff) {
        max_valence = ovalens_last;
        offset -= charge as i32;
    }
    if z == 1 && charge == -1 {
        max_valence = 2;
    }
    if max_valence >= 0 && ovalens_last >= 0 && ev + offset > max_valence {
        return None;
    }
    // calculateImplicitValence (strict)
    if z == 1 && ev == 0 {
        return match charge {
            1 | -1 => Some(0),
            0 => Some(1),
            _ => None,
        };
    }
    if eff == 0 {
        return Some(0);
    }
    let dv = ELEMENTS[eff].valences[0] as i32;
    if dv == -1 {
        return Some(0);
    }
    let mut epr = ev;
    if can_be_hypervalent(z, eff) {
        eff = z;
        epr -= charge as i32;
    }
    let valens = ELEMENTS[eff].valences;
    for &v in valens.iter().take_while(|&&v| v >= 0) {
        if epr <= v as i32 {
            return Some(v as i32 - epr);
        }
    }
    if *valens.last().unwrap() != -1 && ovalens_last > 0 { None } else { Some(0) }
}

/// RDKit `Bond::getValenceContrib` of a non-aromatic bond type for `atom`.
#[inline]
fn valence_contrib(t: BondType, is_target: bool) -> i32 {
    match t {
        BondType::Single => 1,
        BondType::Double => 2,
        BondType::Triple => 3,
        BondType::Dative => is_target as i32,
        BondType::Aromatic | BondType::Unknown => 0,
    }
}

/// Mordred ExtendedTopochemicalAtom values in [`MORDRED_ETA_NAMES`] order.
pub fn mordred_eta(mol: &Molecule, ctx: &mut CalculationContext) -> Vec<f64> {
    let mut out = vec![f64::NAN; MORDRED_ETA_LEN];
    let n = mol.num_atoms();
    if n == 0 {
        return out;
    }
    let other = |b: usize, a: usize| {
        let bond = &mol.bonds[b];
        if bond.source as usize == a { bond.target as usize } else { bond.source as usize }
    };
    let z_of = |a: usize| mol.atoms[a].atomic_number;

    // Kekulé structure. Every aromatic atom that still needs a double bond
    // gets exactly one aromatic double bond (the candidate rule of
    // `mordred_misc::kekule_double_bonds`); the hydrogen-suppressed quantities
    // only depend on these atoms. So does ε4 unless an aromatic bond joins two
    // heteroatoms (its Kekulé order is kept in the saturated molecule): that
    // case uses RDKit's canonical structure.
    let is_aromatic_bond = |b: usize| mol.bonds[b].is_aromatic() || mol.bonds[b].bond_type == BondType::Aromatic;
    let mut arom_double = vec![false; n];
    let mut kekule: Option<Vec<BondType>> = None;
    if (0..mol.num_bonds()).any(is_aromatic_bond) {
        let mut need = 0usize;
        for (a, atom) in mol.atoms.iter().enumerate() {
            let (mut base, mut n_arom) = (atom.total_h() as i32, 0usize);
            for &bi in &mol.adjacency[a] {
                let b = bi as usize;
                if is_aromatic_bond(b) {
                    n_arom += 1;
                    base += 1;
                } else {
                    base += valence_contrib(mol.bonds[b].bond_type, mol.bonds[b].target as usize == a);
                }
            }
            if n_arom > 0 && crate::algo::sanitize::total_valence(mol, a) - base >= 1 {
                arom_double[a] = true;
                need += 1;
            }
        }
        let hetero_pair = (0..mol.num_bonds()).any(|b| {
            is_aromatic_bond(b) && z_of(mol.bonds[b].source as usize) != 6 && z_of(mol.bonds[b].target as usize) != 6
        });
        if hetero_pair || !need.is_multiple_of(2) {
            let k = if hetero_pair {
                super::mordred_kekulize::mordred_kekule_bond_types(mol, ctx)
            } else {
                crate::algo::sanitize::kekule_bond_types(mol, None)
            };
            let Some(k) = k else {
                return out;
            };
            arom_double.fill(false);
            for (b, t) in k.iter().enumerate() {
                if *t == BondType::Double && is_aromatic_bond(b) {
                    arom_double[mol.bonds[b].source as usize] = true;
                    arom_double[mol.bonds[b].target as usize] = true;
                }
            }
            kekule = Some(k);
        }
    }

    let dist = compute_unweighted_distance_matrix_cached(mol, ctx);
    if dist[..n].contains(&DIST_INF) {
        // more than one fragment (`require_connected`)
        return out;
    }
    let nf = n as f64;

    // --- atomic quantities on the hydrogen-suppressed molecule ---
    let mut core = Vec::with_capacity(n);
    let mut eps = Vec::with_capacity(n);
    let mut helium = false;
    // helium in a non-single bond: β_ns needs its ε too
    let mut helium_pi = false;
    for atom in &mol.atoms {
        let z = atom.atomic_number;
        helium |= z == 2;
        core.push(core_count(z));
        eps.push(epsilon(z));
    }
    let mut sqrt_gamma = Vec::with_capacity(n);
    // γ of the reference alkane (heavy atoms only; unused for hydrogens)
    let mut sqrt_gamma_ref = Vec::with_capacity(n);
    let (mut n_h_atoms, mut ref_degree_sum) = (0usize, 0usize);
    let (mut alpha, mut beta, mut beta_s, mut beta_ns, mut beta_ns_d) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut shape = [0.0f64; 3];
    let mut max_degree = 0usize;
    let mut max_degree_h = 0usize;
    let mut bridging_h = false;
    for a in 0..n {
        let atom = &mol.atoms[a];
        let degree = mol.adjacency[a].len();
        max_degree = max_degree.max(degree);
        max_degree_h = max_degree_h.max(degree + atom.total_h() as usize);
        if atom.atomic_number == 1 && degree > 1 {
            bridging_h = true;
        }
        let (mut sigma, mut nonsigma, mut aromatic_nbr, mut heavy_degree) = (0.0f64, 0.0f64, false, 0usize);
        for &bi in &mol.adjacency[a] {
            let b = bi as usize;
            let o = other(b, a);
            aromatic_nbr |= mol.atoms[o].is_aromatic();
            if z_of(o) == 1 {
                continue;
            }
            heavy_degree += 1;
            let d_eps = (eps[o] - eps[a]).abs();
            sigma += if d_eps <= 0.3 { 0.5 } else { 0.75 };
            let t = mol.bonds[b].bond_type;
            if t != BondType::Single && !is_aromatic_bond(b) {
                helium_pi |= z_of(a) == 2 || z_of(o) == 2;
                let f = if t == BondType::Triple { 2.0 } else { 1.0 };
                nonsigma += if d_eps > 0.3 { 1.5 * f } else { f };
            }
        }
        if arom_double[a] {
            // the aromatic Kekulé double bond (y = 2, f = 1)
            nonsigma += 2.0;
        }
        let delta = if aromatic_nbr
            && !atom.is_aromatic()
            && !atom.is_in_ring()
            && (RDKIT_N_OUTER_ELECS[atom.atomic_number as usize] as i32)
                > crate::algo::sanitize::total_valence(mol, a)
        {
            0.5
        } else {
            0.0
        };
        alpha += core[a];
        match degree {
            1 => shape[0] += core[a],
            3 => shape[1] += core[a],
            4 => shape[2] += core[a],
            _ => {}
        }
        let (s, ns) = (sigma / 2.0, nonsigma / 2.0 + delta);
        beta += s + ns;
        beta_s += s;
        beta_ns += ns;
        beta_ns_d += delta;
        let total = sigma + nonsigma + delta;
        sqrt_gamma.push(if total == 0.0 { f64::NAN } else { (core[a] / total).sqrt() });
        if atom.atomic_number == 1 {
            n_h_atoms += 1;
            sqrt_gamma_ref.push(0.0);
        } else {
            ref_degree_sum += heavy_degree;
            sqrt_gamma_ref.push(if heavy_degree == 0 { f64::NAN } else { (1.0 / heavy_degree as f64).sqrt() });
        }
    }
    // --- reference alkane (hydrogen-suppressed): γ_i = 1 / degree ---
    let n_ref = n - n_h_atoms;
    let ref_ok = max_degree <= 4;
    let ((eta, eta_l), (eta_r, eta_rl)) = if n_h_atoms == 0 {
        let [eta, eta_l, eta_r, eta_rl] = eta_sums_pair(&sqrt_gamma, &sqrt_gamma_ref, dist);
        ((eta, eta_l), (eta_r, eta_rl))
    } else {
        let real = eta_sums(&sqrt_gamma, |i, j| Some(dist[i * n + j] as u32));
        let heavy: Vec<usize> = (0..n).filter(|&a| z_of(a) != 1).collect();
        let g_ref: Vec<f64> = heavy.iter().map(|&a| sqrt_gamma_ref[a]).collect();
        let reference = if bridging_h {
            let d = reference_distances(mol, &heavy);
            eta_sums(&g_ref, |i, j| d[i * n_ref + j])
        } else {
            // hydrogens are leaves: the heavy-atom distances do not change
            eta_sums(&g_ref, |i, j| Some(dist[heavy[i] * n + heavy[j]] as u32))
        };
        (real, reference)
    };
    let (eta_r, eta_rl) = if ref_ok { (eta_r, eta_rl) } else { (f64::NAN, f64::NAN) };
    let alpha_r = if ref_ok { 0.5 * n_ref as f64 } else { f64::NAN };

    // --- ε values ---
    let n_h: usize = mol.atoms.iter().map(|a| a.total_h() as usize).sum();
    let eps_h = epsilon(1);
    let eps_sum: f64 = eps.iter().sum();
    let eps1 = (eps_sum + n_h as f64 * eps_h) / (n + n_h) as f64;
    let eps2 = eps_sum / nf;
    let eps3 = if max_degree_h > 4 {
        f64::NAN
    } else {
        let n_ref_h = 4 * n_ref - ref_degree_sum;
        py_div(n_ref as f64 * epsilon(6) + n_ref_h as f64 * eps_h, (n_ref + n_ref_h) as f64)
    };
    let eps4 = saturated_epsilon(mol, kekule.as_deref(), &arom_double);
    let eps5 = {
        let (mut sum, mut count, mut ok) = (0.0f64, 0usize, true);
        for a in 0..n {
            let atom = &mol.atoms[a];
            let include = if atom.atomic_number != 1 {
                true
            } else if let Some(&b) = mol.adjacency[a].first() {
                z_of(other(b as usize, a)) != 6
            } else if atom.total_h() > 0 {
                true
            } else {
                ok = false;
                break;
            };
            if include {
                sum += eps[a];
                count += 1;
            }
            if atom.atomic_number != 6 {
                // hydrogens AddHs attaches to a non-carbon atom
                sum += atom.total_h() as f64 * eps_h;
                count += atom.total_h() as usize;
            }
        }
        if ok { sum / count as f64 } else { f64::NAN }
    };

    let n_rings = mol.ring_info().map_or(0, |r| r.num_rings()) as f64;
    let eta_nl = match n {
        1 => f64::NAN,
        2 => 1.0,
        _ => std::f64::consts::SQRT_2 + 0.5 * (nf - 3.0),
    };
    let psi = py_div(alpha, nf * eps2);

    out[0] = alpha;
    out[1] = alpha / nf;
    for (k, s) in shape.iter().enumerate() {
        out[2 + k] = py_div(*s, alpha);
    }
    for (k, v) in [beta, beta_s, beta_ns, beta_ns_d, eta, eta_l].into_iter().enumerate() {
        out[5 + 2 * k] = v;
        out[6 + 2 * k] = v / nf;
    }
    out[17] = eta_r;
    out[18] = py_div(eta_r, n_ref as f64);
    out[19] = eta_rl;
    out[20] = py_div(eta_rl, n_ref as f64);
    out[21] = eta_r - eta;
    out[22] = (eta_r - eta) / nf;
    out[23] = eta_rl - eta_l;
    out[24] = (eta_rl - eta_l) / nf;
    let eta_b = eta_nl - eta_rl;
    let eta_br = eta_nl - eta_rl + 0.086 * n_rings;
    out[25] = eta_b;
    out[26] = eta_b / nf;
    out[27] = eta_br;
    out[28] = eta_br / nf;
    out[29] = py_max0((alpha - alpha_r) / nf);
    out[30] = py_max0((alpha_r - alpha) / nf);
    out[31] = eps1;
    out[32] = eps2;
    out[33] = eps3;
    out[34] = eps4;
    out[35] = eps5;
    out[36] = eps1 - eps3;
    out[37] = eps1 - eps4;
    out[38] = eps3 - eps4;
    out[39] = eps2 - eps5;
    out[40] = beta_ns - beta_s;
    out[41] = (beta_ns - beta_s) / nf;
    out[42] = psi;
    out[43] = py_max0(0.714 - psi);
    out[44] = py_max0(psi - 0.714);

    if helium {
        // Mordred's core count of helium divides by zero; only β_ns_d, β_ns
        // (unless helium has a multiple bond) and the reference-alkane values
        // (η_R, η_RL, η_B, ε3) do not use it.
        for (k, v) in out.iter_mut().enumerate() {
            let keep = match k {
                9 | 10 => !helium_pi,
                11 | 12 | 17..=20 | 25..=28 | 33 => true,
                _ => false,
            };
            if !keep {
                *v = f64::NAN;
            }
        }
    }
    out
}

/// Topological distances between the heavy atoms of the reference alkane
/// (the graph without hydrogen atoms), `None` if unreachable. Only needed
/// when a hydrogen atom bridges two atoms.
fn reference_distances(mol: &Molecule, heavy: &[usize]) -> Vec<Option<u32>> {
    let n_ref = heavy.len();
    let mut index = vec![usize::MAX; mol.num_atoms()];
    for (i, &a) in heavy.iter().enumerate() {
        index[a] = i;
    }
    let mut d = vec![None; n_ref * n_ref];
    let mut queue = Vec::with_capacity(n_ref);
    for s in 0..n_ref {
        let row = &mut d[s * n_ref..(s + 1) * n_ref];
        row[s] = Some(0);
        queue.clear();
        queue.push(s);
        let mut head = 0;
        while head < queue.len() {
            let i = queue[head];
            head += 1;
            let di = row[i].unwrap();
            let a = heavy[i];
            for &bi in &mol.adjacency[a] {
                let bond = &mol.bonds[bi as usize];
                let o = if bond.source as usize == a { bond.target } else { bond.source } as usize;
                let j = index[o];
                if j != usize::MAX && row[j].is_none() {
                    row[j] = Some(di + 1);
                    queue.push(j);
                }
            }
        }
    }
    d
}

/// Mordred `ETA_epsilon_4`: mean ε over the saturated molecule with explicit
/// hydrogens; NaN if RDKit cannot sanitize it.
///
/// `kekule` holds the Kekulé bond orders when they matter. Otherwise every
/// aromatic bond joins a carbon and each aromatic heteroatom of
/// `arom_double` keeps a double bond to some carbon neighbour: that carbon
/// gets one hydrogen less than with a single bond, whichever it is.
fn saturated_epsilon(mol: &Molecule, kekule: Option<&[BondType]>, arom_double: &[bool]) -> f64 {
    let (mut sum, mut n_heavy, mut n_h, mut hetero_double) = (0.0f64, 0usize, 0usize, 0usize);
    for (a, atom) in mol.atoms.iter().enumerate() {
        let z = atom.atomic_number;
        if z == 1 {
            continue;
        }
        n_heavy += 1;
        let mut ev = 0i32;
        for &bi in &mol.adjacency[a] {
            let b = bi as usize;
            let bond = &mol.bonds[b];
            let o = if bond.source as usize == a { bond.target } else { bond.source } as usize;
            let zo = mol.atoms[o].atomic_number;
            if zo == 1 {
                continue;
            }
            ev += if z == 6 && zo == 6 {
                1
            } else if let Some(k) = kekule {
                valence_contrib(k[b], bond.target as usize == a)
            } else if bond.is_aromatic() || bond.bond_type == BondType::Aromatic {
                1
            } else {
                valence_contrib(bond.bond_type, bond.target as usize == a)
            };
        }
        if kekule.is_none() && z != 6 && arom_double[a] {
            ev += 1;
            hetero_double += 1;
        }
        let Some(h) = sanitized_implicit_hs(z, atom.formal_charge, ev) else {
            return f64::NAN;
        };
        sum += epsilon(z);
        n_h += h as usize;
    }
    let n_h = n_h.saturating_sub(hetero_double);
    py_div(sum + n_h as f64 * epsilon(1), (n_heavy + n_h) as f64)
}

/// Mordred ExtendedTopochemicalAtom descriptor family (45 values, see the
/// module docs).
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct MordredEta;

impl Descriptor for MordredEta {
    fn name(&self) -> &str {
        "MordredEta"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::FloatVec(mordred_eta(mol, ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    const NAN: f64 = f64::NAN;

    fn check(smi: &str, want: [f64; MORDRED_ETA_LEN]) {
        let mol = parse_smiles(smi).unwrap();
        let got = mordred_eta(&mol, &mut CalculationContext::new());
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            let ok = if w.is_nan() { g.is_nan() } else { (g - w).abs() <= 1e-12 + 1e-12 * w.abs() };
            assert!(ok, "{smi} {}: got {g}, want {w}", MORDRED_ETA_NAMES[k]);
        }
    }

    // Expected: mordred-community 2.0.7 on RDKit 2026.03.2.
    #[test]
    fn aspirin() {
        check(
            "CC(=O)Oc1ccccc1C(=O)O",
            [
                5.833333333333333, 0.4487179487179487, 0.2571428571428572, 0.34285714285714286, 0.0, 17.25,
                1.3269230769230769, 7.75, 0.5961538461538461, 9.5, 0.7307692307692307, 0.5, 0.038461538461538464,
                6.568476902903792, 0.5052674540695224, 2.3085173269798047, 0.17757825592152343, 17.88224391821074,
                1.3755572244777492, 6.1090609052806215, 0.4699277619446632, 11.313767015306947, 0.8702897704082267,
                3.800543578300817, 0.29234950602313975, 0.3051526570924734, 0.023473281314805645, 0.3911526570924734,
                0.03008866593019026, 0.0, 0.05128205128205131, 0.6936507936507936, 0.9358974358974359,
                0.4333333333333333, 0.6061728395061728, 0.8904761904761903, 0.26031746031746034, 0.08747795414462078,
                -0.17283950617283955, 0.04542124542124559, 1.75, 0.1346153846153846, 0.4794520547945205,
                0.23454794520547945, 0.0,
            ],
        );
    }

    /// Aromatic N=N: ε4 needs RDKit's canonical Kekulé structure.
    #[test]
    fn pyridazine() {
        check(
            "N1=NC=CC=C1",
            [
                2.8, 0.4666666666666666, 0.0, 0.0, 0.0, 9.5, 1.5833333333333333, 3.5, 0.5833333333333334, 6.0, 1.0,
                0.0, 0.0, 1.4729276101463156, 0.24548793502438593, 0.8852082638152766, 0.14753471063587945, 5.0,
                0.8333333333333334, 3.0, 0.5, 3.5270723898536844, 0.5878453983089474, 2.1147917361847233,
                0.3524652893641205, -0.08578643762690508, -0.01429773960448418, 0.00021356237309491655,
                3.559372884915276e-05, 0.0, 0.03333333333333336, 0.62, 0.8333333333333334, 0.43333333333333335,
                0.5285714285714286, 0.8333333333333334, 0.18666666666666665, 0.09142857142857141,
                -0.09523809523809523, 0.0, 2.5, 0.4166666666666667, 0.5599999999999999, 0.15400000000000003, 0.0,
            ],
        );
    }

    /// Isotopic hydrogens stay atoms of the hydrogen-suppressed molecule.
    #[test]
    fn deuterated_toluene() {
        check(
            "[2H]C([2H])c1ccccc1",
            [
                3.5, 0.3888888888888889, 0.0, 0.2857142857142857, 0.0, 10.25, 1.1388888888888888, 4.25,
                0.4722222222222222, 6.0, 0.6666666666666666, 0.0, 0.0, 2.785895974049715, 0.309543997116635,
                1.3532378396000777, 0.1503597599555642, 6.626799234676718, 0.9466856049538169, 3.3938468501173515,
                0.4848352643024788, 3.840903260627003, 0.4267670289585559, 2.040609010517274, 0.22673433450191935,
                1.0203667122557434, 0.11337407913952705, 1.1063667122557435, 0.12292963469508261, 0.0, 0.0,
                0.48666666666666664, 0.6111111111111112, 0.4333333333333333, 0.4333333333333333, 0.7,
                0.053333333333333344, 0.053333333333333344, 0.0, -0.0888888888888888, 1.75, 0.19444444444444445,
                0.6363636363636364, 0.07763636363636361, 0.0,
            ],
        );
    }

    /// More than four neighbours: no reference alkane.
    #[test]
    fn sulfur_hexafluoride() {
        check(
            "FS(F)(F)(F)(F)F",
            [
                2.5476190476190474, 0.3639455782312925, 0.6728971962616822, 0.0, 0.0, 4.5, 0.6428571428571429, 4.5,
                0.6428571428571429, 0.0, 0.0, 0.0, 0.0, 4.4507810029220485, 0.6358258575602926, 1.5936381457791915,
                0.2276625922541702, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
                1.6931972789115648, 1.6931972789115648, NAN, 1.6931972789115648, 1.6931972789115648, NAN, 0.0, NAN,
                0.0, -4.5, -0.6428571428571429, 0.2149457613499397, 0.49905423865006027, 0.0,
            ],
        );
    }

    #[test]
    fn methane_and_fragments() {
        check(
            "C",
            [
                0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, NAN, NAN, NAN, NAN, 0.0, 0.0, 0.38, 0.7, 0.38, 0.38, 0.7, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.7142857142857143, 0.0, 0.00028571428571433355,
            ],
        );
        check("CC.O", [NAN; MORDRED_ETA_LEN]);
    }
}
