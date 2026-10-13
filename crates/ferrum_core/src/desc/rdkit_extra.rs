//! RDKit descriptors that complete the `rdkit.Chem.Descriptors` catalogue.
//!
//! Each descriptor is a direct port of the RDKit definition named in its doc
//! comment and is checked column by column in `tests/rdkit_parity.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::mol::rdkit_data::{ELEMENTS, atom_mass};

/// RDKit `ROMol::getNumHeavyAtoms`.
fn num_heavy_atoms(mol: &Molecule) -> usize {
    mol.atoms.iter().filter(|a| a.atomic_number > 1).count()
}

/// Sum over RDKit `findAllPathsOfLengthN(mol, n + 1, useBonds = false)` paths
/// (hydrogen atoms excluded) of the product of atom weights; ring-closing
/// paths count their repeated atom once (RDKit `calcChiNv`/`calcChiNn`).
fn chi_path_sum(
    mol: &Molecule,
    ctx: &mut crate::desc::context::CalculationContext,
    weights: &[f64],
    n_bonds: usize,
) -> f64 {
    let mut res = 0.0;
    for path in crate::algo::paths::cached_paths_of_length(mol, ctx, n_bonds).chunks_exact(n_bonds + 1) {
        if path.iter().any(|&a| mol.atoms[a as usize].atomic_number == 1) {
            continue;
        }
        let mut accum = 1.0;
        for &a in &path[..n_bonds] {
            accum *= weights[a as usize];
        }
        if path[n_bonds] != path[0] {
            accum *= weights[path[n_bonds] as usize];
        }
        res += accum;
    }
    res
}

// ---------------------------------------------------------------------------
// HeavyAtomMolWt, ExactMolWt
// ---------------------------------------------------------------------------

/// `[HeavyAtomMolWt, ExactMolWt]` (RDKit `MolOps::getAvgMolWt(onlyHeavy=true)`,
/// `MolOps::getExactMolWt`).
pub struct MolWeights;

const ELECTRON_MASS: f64 = 0.00054857991;

pub fn heavy_atom_mol_wt(mol: &Molecule) -> f64 {
    mol.atoms
        .iter()
        .filter(|a| a.atomic_number != 1)
        .map(|a| atom_mass(a.atomic_number, a.isotope))
        .sum()
}

pub fn exact_mol_wt(mol: &Molecule) -> f64 {
    let mut res = 0.0;
    let mut n_hs = 0u32;
    for a in &mol.atoms {
        res += if a.isotope == 0 {
            ELEMENTS[a.atomic_number as usize].common_isotope_mass
        } else {
            atom_mass(a.atomic_number, a.isotope)
        };
        res -= ELECTRON_MASS * a.formal_charge as f64;
        n_hs += a.total_h() as u32;
    }
    res + n_hs as f64 * ELEMENTS[1].common_isotope_mass
}

impl Descriptor for MolWeights {
    fn name(&self) -> &str {
        "MolWeights"
    }
    fn calculate(&self, mol: &mut Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::FloatVec(vec![heavy_atom_mol_wt(mol), exact_mol_wt(mol)])
    }
}

// ---------------------------------------------------------------------------
// Chi0n .. Chi4n
// ---------------------------------------------------------------------------

/// `[Chi0n, Chi1n, Chi2n, Chi3n, Chi4n]` (RDKit `calcChi0n` .. `calcChi4n`).
pub struct ChiNIndices;

/// RDKit `nVals`: 1 / sqrt(outer electrons - total H), 0 when that is 0.
fn n_vals(mol: &Molecule) -> Vec<f64> {
    mol.atoms
        .iter()
        .map(|a| {
            let v = ELEMENTS[a.atomic_number as usize].n_outer as f64 - a.total_h() as f64;
            if v != 0.0 { 1.0 / v.sqrt() } else { 0.0 }
        })
        .collect()
}

impl Descriptor for ChiNIndices {
    fn name(&self) -> &str {
        "ChiNIndices"
    }
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let nv = n_vals(mol);
        let chi0: f64 = nv.iter().sum();
        let chi1: f64 = mol.bonds.iter().map(|b| nv[b.source as usize] * nv[b.target as usize]).sum();
        let mut out = vec![chi0 as f64, chi1];
        for n in 2..=4 {
            out.push(chi_path_sum(mol, ctx, &nv, n) as f64);
        }
        DescriptorValue::FloatVec(out)
    }
}

// ---------------------------------------------------------------------------
// MaxAbsEStateIndex, MinAbsEStateIndex, EState_VSA, VSA_EState
// ---------------------------------------------------------------------------

/// `[MaxAbsEStateIndex, MinAbsEStateIndex]` (RDKit `EState.EState`).
pub struct EStateAbs;

impl Descriptor for EStateAbs {
    fn name(&self) -> &str {
        "EStateAbs"
    }
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let es = crate::desc::estate::compute_estates(mol, ctx);
        if es.is_empty() {
            return DescriptorValue::FloatVec(vec![0.0, 0.0]);
        }
        let max = es.iter().map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max);
        let min = es.iter().map(|x| x.abs()).fold(f64::INFINITY, f64::min);
        DescriptorValue::FloatVec(vec![max as f64, min as f64])
    }
}

const ESTATE_BINS: [f64; 10] = [-0.390, 0.290, 0.717, 1.165, 1.540, 1.807, 2.05, 4.69, 9.17, 15.0];
const VSA_BINS: [f64; 9] = [4.78, 5.00, 5.410, 5.740, 6.00, 6.07, 6.45, 7.00, 11.0];

/// Python `bisect.bisect_right`.
fn bisect_right(bins: &[f64], x: f64) -> usize {
    bins.iter().take_while(|&&b| b <= x).count()
}

/// `EState_VSA1` .. `EState_VSA11` (RDKit `EState.EState_VSA`): Labute VSA binned by E-state.
pub struct EStateVsa;

/// `VSA_EState1` .. `VSA_EState10` (RDKit `EState.EState_VSA`): E-state binned by Labute VSA.
pub struct VsaEState;

fn estate_and_vsa(mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> (Vec<f64>, Vec<f64>) {
    let es = crate::desc::estate::compute_estates(mol, ctx);
    let (vsa, _) = crate::desc::vsa::compute_labute_contribs(mol, ctx);
    (es, vsa)
}

impl Descriptor for EStateVsa {
    fn name(&self) -> &str {
        "EStateVsa"
    }
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let (es, vsa) = estate_and_vsa(mol, ctx);
        let mut out = [0.0f64; 11];
        for (e, v) in es.iter().zip(&vsa) {
            out[bisect_right(&ESTATE_BINS, *e)] += v;
        }
        DescriptorValue::FloatVec(out.to_vec())
    }
}

impl Descriptor for VsaEState {
    fn name(&self) -> &str {
        "VsaEState"
    }
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let (es, vsa) = estate_and_vsa(mol, ctx);
        let mut out = [0.0f64; 10];
        for (e, v) in es.iter().zip(&vsa) {
            out[bisect_right(&VSA_BINS, *v)] += e;
        }
        DescriptorValue::FloatVec(out.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Ring counts
// ---------------------------------------------------------------------------

/// `[RingCount, NumAromaticCarbocycles, NumAromaticHeterocycles,
/// NumAliphaticCarbocycles, NumAliphaticHeterocycles, NumSaturatedCarbocycles,
/// NumSaturatedHeterocycles, NumHeterocycles, NumSpiroAtoms, NumBridgeheadAtoms]`
/// (RDKit `Lipinski.cpp`, on the symmetrized-SSSR ring info).
pub struct RingCounts;

pub fn ring_counts(mol: &Molecule) -> [u32; 10] {
    let Some(info) = mol.ring_info() else { return [0; 10] };
    let is_arom = |b: u16| mol.bonds[b as usize].is_aromatic();
    let is_hetero_bond = |b: u16| {
        let bond = &mol.bonds[b as usize];
        mol.atoms[bond.source as usize].atomic_number != 6 || mol.atoms[bond.target as usize].atomic_number != 6
    };
    let is_saturated = |b: u16| mol.bonds[b as usize].bond_type == crate::types::BondType::Single && !is_arom(b);

    let mut out = [0u32; 10];
    out[0] = info.num_rings() as u32;
    for ring in &info.bond_rings {
        let all_arom = ring.iter().all(|&b| is_arom(b));
        let any_aliph = ring.iter().any(|&b| !is_arom(b));
        let hetero = ring.iter().any(|&b| is_hetero_bond(b));
        let all_saturated = ring.iter().all(|&b| is_saturated(b));
        if all_arom && !hetero {
            out[1] += 1;
        }
        if all_arom && hetero {
            out[2] += 1;
        }
        if any_aliph && !hetero {
            out[3] += 1;
        }
        if any_aliph && hetero {
            out[4] += 1;
        }
        if all_saturated && !hetero {
            out[5] += 1;
        }
        if all_saturated && hetero {
            out[6] += 1;
        }
    }
    out[7] = info
        .atom_rings
        .iter()
        .filter(|r| r.iter().any(|&a| mol.atoms[a as usize].atomic_number != 6))
        .count() as u32;

    // spiro atoms: two atom rings sharing exactly one atom
    let mut spiro: Vec<u16> = Vec::new();
    for i in 0..info.atom_rings.len() {
        for j in i + 1..info.atom_rings.len() {
            let shared: Vec<u16> =
                info.atom_rings[i].iter().copied().filter(|a| info.atom_rings[j].contains(a)).collect();
            if shared.len() == 1 && !spiro.contains(&shared[0]) {
                spiro.push(shared[0]);
            }
        }
    }
    out[8] = spiro.len() as u32;

    // bridgehead atoms: bond rings sharing more than one bond; atoms that occur
    // once among the endpoints of the shared bonds
    let mut bridgeheads: Vec<usize> = Vec::new();
    let mut counts = vec![0u32; mol.num_atoms()];
    for i in 0..info.bond_rings.len() {
        for j in i + 1..info.bond_rings.len() {
            let shared: Vec<u16> =
                info.bond_rings[i].iter().copied().filter(|b| info.bond_rings[j].contains(b)).collect();
            if shared.len() > 1 {
                counts.fill(0);
                for &b in &shared {
                    counts[mol.bonds[b as usize].source as usize] += 1;
                    counts[mol.bonds[b as usize].target as usize] += 1;
                }
                for (a, &c) in counts.iter().enumerate() {
                    if c == 1 && !bridgeheads.contains(&a) {
                        bridgeheads.push(a);
                    }
                }
            }
        }
    }
    out[9] = bridgeheads.len() as u32;
    out
}

impl Descriptor for RingCounts {
    fn name(&self) -> &str {
        "RingCounts"
    }
    fn calculate(&self, mol: &mut Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::IntVec(ring_counts(mol).to_vec())
    }
}

// ---------------------------------------------------------------------------
// NHOHCount, NOCount
// ---------------------------------------------------------------------------

/// `[NHOHCount, NOCount]` (RDKit `calcLipinskiHBD`, `calcLipinskiHBA`).
pub struct LipinskiNO;

impl Descriptor for LipinskiNO {
    fn name(&self) -> &str {
        "LipinskiNO"
    }
    fn calculate(&self, mol: &mut Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let mut nhoh = 0u32;
        let mut no = 0u32;
        for (i, a) in mol.atoms.iter().enumerate() {
            if a.atomic_number == 7 || a.atomic_number == 8 {
                no += 1;
                nhoh += mol.total_hs_with_neighbors(i);
            }
        }
        DescriptorValue::IntVec(vec![nhoh, no])
    }
}

// ---------------------------------------------------------------------------
// AvgIpc, Phi
// ---------------------------------------------------------------------------

/// `[AvgIpc, Phi]` (RDKit `GraphDescriptors.AvgIpc`, `calcPhi`).
pub struct AvgIpcPhi;

pub fn avg_ipc(mol: &Molecule) -> f64 {
    avg_ipc_with_context(mol, &mut crate::desc::context::CalculationContext::new())
}

/// [`avg_ipc`] sharing the characteristic polynomial with `Ipc` via `ctx`.
pub fn avg_ipc_with_context(mol: &Molecule, ctx: &mut crate::desc::context::CalculationContext) -> f64 {
    if mol.num_atoms() == 0 {
        return 0.0;
    }
    let cpoly = crate::desc::extra_counts::abs_characteristic_polynomial_cached(mol, ctx);
    crate::desc::extra_counts::info_entropy(cpoly)
}

pub fn phi(mol: &Molecule) -> f64 {
    let a = num_heavy_atoms(mol) as f64;
    if a == 0.0 {
        return 0.0;
    }
    let alpha = crate::desc::extra_counts::compute_hall_kier_alpha_f64(mol);
    let p1 = mol.num_bonds() as f64;
    // Two-bond paths without hydrogens: one per pair of heavy-atom bonds at a
    // heavy centre (what `find_all_paths_of_length(mol, 2)` enumerates).
    let mut p2_count = 0usize;
    for (c, bonds) in mol.adjacency.iter().enumerate() {
        if mol.atoms[c].atomic_number == 1 {
            continue;
        }
        let k = bonds
            .iter()
            .filter(|&&bi| {
                let b = &mol.bonds[bi as usize];
                let o = if b.source as usize == c { b.target } else { b.source };
                mol.atoms[o as usize].atomic_number != 1
            })
            .count();
        p2_count += k * k.saturating_sub(1) / 2;
    }
    let p2 = p2_count as f64;
    let denom1 = p1 + alpha;
    let kappa1 = if denom1 != 0.0 { (a + alpha) * (a + alpha - 1.0).powi(2) / (denom1 * denom1) } else { 0.0 };
    let denom2 = (p2 + alpha) * (p2 + alpha);
    let kappa2 = if denom2 != 0.0 { (a + alpha - 1.0) * (a + alpha - 2.0).powi(2) / denom2 } else { 0.0 };
    kappa1 * kappa2 / a
}

impl Descriptor for AvgIpcPhi {
    fn name(&self) -> &str {
        "AvgIpcPhi"
    }
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::FloatVec(vec![avg_ipc_with_context(mol, ctx), phi(mol)])
    }
}

// ---------------------------------------------------------------------------
// FpDensityMorgan1..3
// ---------------------------------------------------------------------------

/// `[FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3]`: distinct Morgan
/// identifiers per heavy atom (RDKit `Descriptors.FpDensityMorgan*`).
pub struct FpDensityMorgan;

impl Descriptor for FpDensityMorgan {
    fn name(&self) -> &str {
        "FpDensityMorgan"
    }
    fn calculate(&self, mol: &mut Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let heavy = num_heavy_atoms(mol);
        let mut out = Vec::with_capacity(3);
        for radius in 1..=3u8 {
            if heavy == 0 {
                out.push(0.0);
                continue;
            }
            let mut ids = crate::desc::fingerprint::ECFP::new(radius, 2048).raw_features(mol);
            ids.sort_unstable();
            ids.dedup();
            out.push((ids.len() as f64 / heavy as f64) as f64);
        }
        DescriptorValue::FloatVec(out)
    }
}

#[allow(dead_code)]
fn _assert_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<MolWeights>();
    check::<ChiNIndices>();
    check::<EStateAbs>();
    check::<EStateVsa>();
    check::<VsaEState>();
    check::<RingCounts>();
    check::<LipinskiNO>();
    check::<AvgIpcPhi>();
    check::<FpDensityMorgan>();
}

// ---------------------------------------------------------------------------
// fr_* fragment counts
// ---------------------------------------------------------------------------

/// The 85 `fr_*` descriptors of `rdkit.Chem.Fragments`: unique SMARTS match
/// counts (`len(mol.GetSubstructMatches(patt))`).
pub struct Fragments;

static FRAGMENT_QUERIES: once_cell::sync::Lazy<Vec<crate::io::smarts::QueryGraph>> =
    once_cell::sync::Lazy::new(|| {
        crate::desc::rdkit_fragments_data::FRAGMENTS
            .iter()
            .map(|(name, smarts)| {
                crate::io::smarts::parse_smarts(smarts)
                    .unwrap_or_else(|e| panic!("RDKit fragment {name} SMARTS {smarts}: {e}"))
            })
            .collect()
    });

impl Descriptor for Fragments {
    fn name(&self) -> &str {
        "Fragments"
    }
    fn calculate(&self, mol: &mut Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let target = crate::algo::iso::IsoTarget::new(mol);
        DescriptorValue::IntVec(
            FRAGMENT_QUERIES
                .iter()
                .map(|q| crate::algo::iso::count_matches_target(&target, q) as u32)
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn all_fragment_smarts_parse() {
        let failed: Vec<_> = crate::desc::rdkit_fragments_data::FRAGMENTS
            .iter()
            .filter(|(_, s)| crate::io::smarts::parse_smarts(s).is_err())
            .collect();
        assert!(failed.is_empty(), "unparseable fragment SMARTS: {failed:?}");
    }
}

// ---------------------------------------------------------------------------
// BCUT2D
// ---------------------------------------------------------------------------

/// `[BCUT2D_MWHI, BCUT2D_MWLOW, BCUT2D_CHGHI, BCUT2D_CHGLO, BCUT2D_LOGPHI,
/// BCUT2D_LOGPLOW, BCUT2D_MRHI, BCUT2D_MRLOW]` (RDKit `Descriptors/BCUT.cpp`):
/// highest/lowest eigenvalues of the Burden matrix (0.001 off-diagonal,
/// 1/sqrt(bond order) for bonds) with atomic mass, Gasteiger charge, Crippen
/// logP and Crippen MR on the diagonal, computed on the H-free molecule.
pub struct Bcut2D;

pub fn bcut2d(mol: &Molecule) -> [f64; 8] {
    if mol.num_atoms() == 0 {
        return [0.0; 8];
    }
    let stripped = crate::algo::sanitize::without_all_hydrogens(mol);
    let mol = stripped.as_ref().unwrap_or(mol);
    match bcut2d_burden(mol) {
        Err(v) => v,
        Ok(burden) => {
            let charges = crate::algo::electronic::compute_gasteiger_charges(mol);
            let (logp, mr) = crate::desc::logp::crippen_atom_contributions(mol);
            bcut2d_eigen(mol, burden, &charges, &logp, &mr)
        }
    }
}

/// Bond entries of the Burden matrix of the H-free molecule (0.001
/// elsewhere), or the final result where RDKit returns early (no atoms) or
/// throws (NaN).
fn bcut2d_burden(mol: &Molecule) -> Result<Vec<(u32, u32, f64)>, [f64; 8]> {
    let n = mol.num_atoms();
    if n == 0 {
        return Err([0.0; 8]);
    }
    // RDKit throws for missing Gasteiger parameters and non-standard bond types
    if !crate::algo::electronic::has_all_gasteiger_params(mol) {
        return Err([f64::NAN; 8]);
    }
    let mut burden = Vec::with_capacity(mol.bonds.len());
    for bond in &mol.bonds {
        let score = match bond.bond_type {
            crate::types::BondType::Aromatic => 0.816_496_580_927_726_1,
            crate::types::BondType::Single => 1.0,
            crate::types::BondType::Double => 0.707_106_781_186_547_5,
            crate::types::BondType::Triple => 0.577_350_269_189_625_8,
            _ => return Err([f64::NAN; 8]),
        };
        burden.push((bond.source as u32, bond.target as u32, score));
    }
    Ok(burden)
}

/// Extreme eigenvalues of the Burden matrix with each atomic property on the
/// diagonal.
fn bcut2d_eigen(mol: &Molecule, burden: Vec<(u32, u32, f64)>, charges: &[f64], logp: &[f64], mr: &[f64]) -> [f64; 8] {
    let n = mol.num_atoms();
    let masses: Vec<f64> = mol.atoms.iter().map(|a| atom_mass(a.atomic_number, a.isotope)).collect();
    let jobs = [&masses[..], charges, logp, mr]
        .map(|diag| crate::algo::linalg::BurdenMatrix { fill: 0.001, entries: &burden, diag });
    let mut out = [0.0f64; 8];
    for (k, (lo, hi)) in crate::algo::linalg::burden_extremes(n, &jobs).into_iter().enumerate() {
        out[2 * k] = hi;
        out[2 * k + 1] = lo;
    }
    out
}

impl Descriptor for Bcut2D {
    fn name(&self) -> &str {
        "BCUT2D"
    }
    fn calculate(&self, mol: &mut Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        if mol.num_atoms() == 0 || mol.atoms.iter().any(|a| a.atomic_number == 1) {
            return DescriptorValue::FloatVec(bcut2d(mol).to_vec());
        }
        // H-free molecule: share the per-molecule Gasteiger / Crippen caches
        // with the other descriptors (values identical to `bcut2d`).
        let v = match bcut2d_burden(mol) {
            Err(v) => v,
            Ok(burden) => {
                let charges = crate::algo::electronic::compute_gasteiger_charges_cached(mol);
                let (logp, mr) = crate::desc::logp::cached_crippen_atom_contributions(mol);
                bcut2d_eigen(mol, burden, &charges, &logp, &mr)
            }
        };
        DescriptorValue::FloatVec(v.to_vec())
    }
}

/// RDKit `NumAtomStereoCenters` / `NumUnspecifiedAtomStereoCenters`.
pub struct StereoCenters;

impl Descriptor for StereoCenters {
    fn name(&self) -> &str {
        "StereoCenters"
    }
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let (all, unspecified) = crate::algo::stereo::atom_stereo_center_counts_with_context(mol, ctx);
        DescriptorValue::IntVec(vec![all, unspecified])
    }
}

/// RDKit `SPS`: normalized spacial score (NaN without heavy atoms, where RDKit raises).
pub struct Sps;

impl Descriptor for Sps {
    fn name(&self) -> &str {
        "SPS"
    }
    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::Float(crate::algo::stereo::spacial_score_with_context(mol, ctx))
    }
}
