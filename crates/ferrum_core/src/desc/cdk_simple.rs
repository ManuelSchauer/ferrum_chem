//! Simple CDK 2.13 count and property descriptors.
//!
//! One pass over the CDK view of the molecule computes the CDK descriptors
//! `AtomCountDescriptor` (`CDK_nAtom`), `BondCountDescriptor` (`CDK_nB`),
//! `AromaticAtomsCountDescriptor` (`naAromAtom`), `AromaticBondsCountDescriptor`
//! (`CDK_nAromBond`), `HBondAcceptorCountDescriptor` (`CDK_nHBAcc`),
//! `HBondDonorCountDescriptor` (`CDK_nHBDon`), `RotatableBondsCountDescriptor`
//! (`nRotB`), `TPSADescriptor` (`CDK_TopoPSA`), `FractionalPSADescriptor`
//! (`tpsaEfficiency`), `WeightDescriptor` (`CDK_MW`), `FMFDescriptor` (`FMF`),
//! `FractionalCSP3Descriptor` (`Fsp3`), `SpiroAtomCountDescriptor`
//! (`nSpiroAtoms`) and `VABCDescriptor` (`VABC`), all with CDK's default
//! parameters, plus `RuleOfFiveDescriptor` (`LipinskiFailures`, see
//! [`cdk_rule_of_five`], which needs CDK's XLogP).
//!
//! Column names: CDK's descriptor names, prefixed with `CDK_` where the name is
//! already a FerrumChem column (RDKit) or a Mordred descriptor name with another
//! definition (the same rule as Mordred's `Mordred_` prefix, see
//! `desc::mordred_names`). `AminoAcidCountDescriptor`'s columns all carry the
//! prefix (`CDK_nA`, ...) since `nC`, `nN`, ... would read as element counts.
//!
//! The molecule is the one the CDK reference prepares: Kekulé bond orders as
//! CDK's SMILES parser assigns them, CDK atom types and the hydrogens
//! `CDKHydrogenAdder` adds, and aromaticity from `Aromaticity(daylight(),
//! Cycles.or(all(), all(6)))` (see `desc::cdk_cycles`). The descriptors that do
//! not re-perceive aromaticity (their `checkAromaticity` parameter defaults to
//! false) read those flags.

use crate::desc::cdk_atom_types::formal_neighbour_count;
use crate::desc::cdk_cycles::{all_rings_finder_completes, daylight_prepared_aromaticity};
use crate::desc::cdk_daylight::CdkMol;
use crate::desc::cdk_perception::{CdkView, Hyb, MAJOR_ISOTOPE_MASS, NATURAL_MASS};
use crate::desc::context::CalculationContext;
use crate::desc::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Output columns of [`CdkSimpleCounts`], in output order.
pub const CDK_SIMPLE_NAMES: [&str; 14] = [
    "CDK_nAtom",
    "CDK_nB",
    "naAromAtom",
    "CDK_nAromBond",
    "CDK_nHBAcc",
    "CDK_nHBDon",
    "nRotB",
    "CDK_TopoPSA",
    "tpsaEfficiency",
    "CDK_MW",
    "FMF",
    "Fsp3",
    "nSpiroAtoms",
    "VABC",
];

/// The simple CDK 2.13 count/property descriptors (see the module docs);
/// returns FloatVec in the order of [`CDK_SIMPLE_NAMES`].
pub struct CdkSimpleCounts;

impl Descriptor for CdkSimpleCounts {
    fn name(&self) -> &str {
        "CdkSimpleCounts"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::FloatVec(CdkView::with_legacy_cached(mol, ctx, |view, _| {
            cdk_simple_counts(mol, view).to_vec()
        }))
    }
}

/// CDK `RuleOfFiveDescriptor` (`LipinskiFailures`, default parameters).
///
/// Counts the failed rules XLogP > 5, H-bond acceptors > 10, donors > 5,
/// molecular weight > 500 and rotatable bonds > 10. CDK evaluates them with
/// `XLogPDescriptor(checkAromaticity = false, salicylFlag = true)`,
/// `HBondAcceptorCountDescriptor`, `HBondDonorCountDescriptor`,
/// `WeightDescriptor("*")` and `RotatableBondsCountDescriptor(includeTerminals
/// = false, excludeAmides = true)`.
///
/// `xlogp` is the default `XLogPDescriptor` value (`XLogP` column, both
/// parameters false); the salicylic acid correction (+0.554 when
/// `O=C(O)c1ccccc1O` matches) that `salicylFlag = true` adds is applied here.
/// The `LipinskiFailures` column is emitted by `cdk_logp::CdkLogP`, which has
/// the XLogP at hand.
pub fn cdk_rule_of_five(mol: &Molecule, ctx: &mut CalculationContext, xlogp: f64) -> u32 {
    CdkView::with_legacy_cached(mol, ctx, |view, _| Prepared::with(mol, view, |p| rule_of_five(p, xlogp)))
}

fn rule_of_five(p: &Prepared, xlogp: f64) -> u32 {
    let mut failures = 0;
    let xlogp = if salicylic_acid(p.m) { xlogp + 0.554 } else { xlogp };
    if xlogp > 5.0 {
        failures += 1;
    }
    if p.hbond_acceptors() > 10 {
        failures += 1;
    }
    if p.hbond_donors() > 5 {
        failures += 1;
    }
    if p.weight() > 500.0 {
        failures += 1;
    }
    if p.rotatable_bonds(true) > 10 {
        failures += 1;
    }
    failures
}

/// CDK view of the prepared molecule shared by the descriptors of this module.
pub(crate) struct Prepared<'a> {
    /// Kekulé orders / formal charges as CDK's SMILES parser keeps them;
    /// `m.hcount` holds the implicit hydrogens `CDKHydrogenAdder` assigns.
    pub m: &'a CdkMol,
    /// Implicit hydrogens after `CDKHydrogenAdder` (`m.hcount`).
    pub h: &'a [u8],
    /// Daylight aromaticity of the reference preparation (atoms, bonds).
    pub arom_atom: &'a [bool],
    pub arom_bond: &'a [bool],
}

/// The molecule-dependent part of [`Prepared`], kept per thread for the
/// molecule last seen (keyed by `molecule_cache_key`) so that the descriptors
/// of this module, `KierHallSmarts` and `AminoAcidCount` build it once per
/// molecule.
type PreparedCore = (CdkMol, Vec<bool>, Vec<bool>);

thread_local! {
    static PREPARED_CACHE: std::cell::RefCell<Option<(u64, PreparedCore)>> =
        const { std::cell::RefCell::new(None) };
}

/// Runs `f` on the cached [`PreparedCore`] of `mol` (built when missing).
fn with_core<R>(mol: &Molecule, view: &CdkView, f: impl FnOnce(&PreparedCore) -> R) -> R {
    let key = crate::desc::context::molecule_cache_key(mol);
    let core = match PREPARED_CACHE.with(|c| c.borrow_mut().take()) {
        Some((k, core)) if k == key => core,
        _ => Prepared::core(mol, view),
    };
    let r = f(&core);
    PREPARED_CACHE.with(|c| *c.borrow_mut() = Some((key, core)));
    r
}

/// CDK's view of `mol` as its SMILES parser and `CDKHydrogenAdder` leave it:
/// Kekulé orders and charges as written, `hcount` = CDK's implicit hydrogens.
pub(crate) fn cdk_parser_mol(mol: &Molecule, view: &CdkView) -> CdkMol {
    let mut m = CdkMol::new(mol);
    restore_parser_state(mol, &mut m);
    let h = cdk_hydrogens(mol, &m, view);
    // CDK keeps no radicals: an aromatic radical atom that the hydrogen adder
    // fills up (`[C]1=CC=CC=C1` is benzene) needs its pi bond in the Kekulé form
    let filled = |a: usize| {
        let atom = &mol.atoms[a];
        atom.is_aromatic() && atom.num_radicals() > 0 && h[a] > atom.total_h()
    };
    if (0..m.n()).any(filled) {
        let mut copy = mol.clone();
        for a in 0..m.n() {
            if filled(a) {
                let atom = &mut copy.atoms[a];
                atom.num_explicit_h = h[a] - atom.num_implicit_h;
                atom.set_num_radicals(0);
            }
        }
        m = CdkMol::new(&copy);
        restore_parser_state(&copy, &mut m);
    }
    m.hcount = h;
    m
}

/// Runs `f` on [`cdk_parser_mol`] of `mol` (shared per-thread cache).
pub(crate) fn with_parser_mol<R>(mol: &Molecule, view: &CdkView, f: impl FnOnce(&CdkMol) -> R) -> R {
    with_core(mol, view, |core| f(&core.0))
}

/// Like [`with_parser_mol`], also passing the bond aromaticity of the
/// reference preparation.
pub(crate) fn with_prepared_bonds<R>(mol: &Molecule, view: &CdkView, f: impl FnOnce(&CdkMol, &[bool]) -> R) -> R {
    with_core(mol, view, |core| f(&core.0, &core.2))
}

impl<'a> Prepared<'a> {
    fn core(mol: &Molecule, view: &CdkView) -> PreparedCore {
        let m = cdk_parser_mol(mol, view);
        let (arom_atom, arom_bond) = daylight_prepared_aromaticity(&m)
            .unwrap_or_else(|| (vec![false; m.n()], vec![false; m.ends.len()]));
        (m, arom_atom, arom_bond)
    }

    /// Runs `f` on the prepared molecule, reusing the per-thread cache.
    pub(crate) fn with<R>(mol: &Molecule, view: &CdkView, f: impl FnOnce(&Prepared) -> R) -> R {
        with_core(mol, view, |(m, arom_atom, arom_bond)| {
            f(&Prepared { m, h: &m.hcount, arom_atom, arom_bond })
        })
    }

    #[inline]
    fn z(&self, a: usize) -> u8 {
        self.m.z[a]
    }

    /// Explicit hydrogen neighbours.
    fn h_neighbors(&self, a: usize) -> usize {
        self.m.adj[a].iter().filter(|&&(w, _)| self.m.z[w as usize] == 1).count()
    }

    /// `HBondAcceptorCountDescriptor` (checkAromaticity = false).
    fn hbond_acceptors(&self) -> u32 {
        let mut count = 0;
        'atoms: for a in 0..self.m.n() {
            let z = self.z(a);
            if self.m.charge[a] > 0 {
                continue;
            }
            if z == 7 {
                let mut n_pi = 0;
                for &(w, bi) in &self.m.adj[a] {
                    if self.m.z[w as usize] == 8 {
                        continue 'atoms;
                    }
                    if self.m.order[bi as usize] == 2 {
                        n_pi += 1;
                    }
                }
                if self.arom_atom[a] && n_pi == 0 {
                    continue;
                }
                count += 1;
            } else if z == 8 {
                let degree = self.m.adj[a].len() - self.h_neighbors(a);
                for &(w, _) in &self.m.adj[a] {
                    let w = w as usize;
                    if self.m.z[w] == 7 || (self.m.z[w] == 6 && self.arom_atom[w] && degree != 1) {
                        continue 'atoms;
                    }
                }
                count += 1;
            }
        }
        count
    }

    /// `HBondDonorCountDescriptor`.
    fn hbond_donors(&self) -> u32 {
        (0..self.m.n())
            .filter(|&a| {
                matches!(self.z(a), 7 | 8)
                    && self.m.charge[a] >= 0
                    && (self.h[a] > 0 || self.h_neighbors(a) > 0)
            })
            .count() as u32
    }

    /// `WeightDescriptor` ("*"): natural masses plus implicit hydrogens.
    fn weight(&self) -> f64 {
        let h_mass = NATURAL_MASS[1];
        let mut w = 0.0;
        for a in 0..self.m.n() {
            w += NATURAL_MASS.get(self.z(a) as usize).copied().unwrap_or(f64::NAN);
            w += self.h[a] as f64 * h_mass;
        }
        w
    }

    /// `RotatableBondsCountDescriptor` (includeTerminals = false).
    fn rotatable_bonds(&self, exclude_amides: bool) -> u32 {
        let m = &self.m;
        let max_order = |a: usize| m.adj[a].iter().map(|&(_, bi)| m.order[bi as usize]).max().unwrap_or(0);
        let heavy_degree = |a: usize| m.adj[a].len() - self.h_neighbors(a);
        let is_amide = |c: usize, n: usize| {
            m.z[c] == 6
                && m.z[n] == 7
                && m.adj[c].iter().any(|&(w, bi)| m.z[w as usize] == 8 && m.order[bi as usize] == 2)
        };
        let mut count = 0;
        for (bi, &(s, t)) in m.ends.iter().enumerate() {
            let (s, t) = (s as usize, t as usize);
            if m.z[s] == 1 || m.z[t] == 1 || m.order[bi] != 1 || m.ring_bond[bi] {
                continue;
            }
            if max_order(s) >= 3 || max_order(t) >= 3 {
                continue;
            }
            if exclude_amides && (is_amide(s, t) || is_amide(t, s)) {
                continue;
            }
            if heavy_degree(s) == 1 || heavy_degree(t) == 1 {
                continue;
            }
            count += 1;
        }
        count
    }

    /// `TPSADescriptor` (checkAromaticity = false); NaN when CDK's
    /// `AllRingsFinder` gives up.
    fn tpsa(&self) -> f64 {
        let m = &self.m;
        let n = m.n();
        // AllRingsFinder can only give up on large ring systems (see
        // `cdk_cycles`): skip the check when the whole cyclic part is small
        let rank = cycle_rank(m);
        let ring_atoms = m.ring_atom.iter().filter(|&&r| r).count();
        if rank > 0 && (rank >= 16 || (ring_atoms.saturating_sub(1) << rank) > 684) {
            let graph: Vec<Vec<u32>> = (0..n).map(|a| m.adj[a].iter().map(|&(w, _)| w).collect()).collect();
            if !all_rings_finder_completes(&graph) {
                return f64::NAN;
            }
        }
        let mut tpsa = 0.0;
        for a in 0..n {
            let z = m.z[a];
            if !matches!(z, 7 | 8 | 15 | 16) {
                continue;
            }
            let (mut single, mut double, mut triple, mut arom) = (0i32, 0i32, 0i32, 0i32);
            let mut h_count = 0i32;
            for &(w, bi) in &m.adj[a] {
                let bi = bi as usize;
                if self.arom_bond[bi] {
                    arom += 1;
                } else {
                    match m.order[bi] {
                        1 => single += 1,
                        2 => double += 1,
                        3 => triple += 1,
                        _ => {}
                    }
                }
                if m.z[w as usize] == 1 {
                    h_count += 1;
                }
            }
            let mut neighbours = m.adj[a].len() as i32;
            let implicit = self.h[a] as i32;
            h_count += implicit;
            neighbours += implicit;
            single += implicit;
            // bond orders in halves (1.5 -> 3)
            let sum2 = 2 * single + 4 * double + 6 * triple + 3 * arom;
            let mut max2 = 0;
            if single > 0 {
                max2 = 2;
            }
            if arom > 0 {
                max2 = 3;
            }
            if double > 0 {
                max2 = 4;
            }
            if triple > 0 {
                max2 = 6;
            }
            let in3 = in_three_ring(m, a) as i32;
            let key = [
                z as i32,
                max2,
                sum2,
                neighbours,
                h_count,
                m.charge[a] as i32,
                arom,
                in3,
                single,
                double,
                triple,
            ];
            let key = tpsa_key(&key);
            if let Some(i) = TPSA_KEYS.iter().position(|&k| k == key) {
                tpsa += TPSA_PROFILES[i].1;
            }
        }
        tpsa
    }

    /// `FractionalPSADescriptor`: TPSA over the mass from major isotopes.
    fn tpsa_efficiency(&self, tpsa: f64) -> f64 {
        let mut weight = 0.0;
        for a in 0..self.m.n() {
            weight += MAJOR_ISOTOPE_MASS.get(self.z(a) as usize).copied().unwrap_or(f64::NAN);
            weight += self.h[a] as f64 * 1.00782504;
        }
        if weight == 0.0 { 0.0 } else { tpsa / weight }
    }
}

/// Restores the charges and bond orders CDK's SMILES parser keeps for oxides
/// of N, P and the halogens. RDKit's clean-up turns `N(=O)=O` into
/// `[N+](=O)[O-]` (reverted here, same rule as `cdk_perception`: only atoms
/// written without brackets can be clean-up results); `CdkMol` reverts the
/// halogen form `[Cl+3]([O-])...` even when it was written with brackets,
/// which is undone here.
pub(crate) fn restore_parser_state(mol: &Molecule, m: &mut CdkMol) {
    for a in 0..m.n() {
        if !matches!(m.z[a], 17 | 35 | 53) || m.charge[a] == mol.atoms[a].formal_charge {
            continue;
        }
        let bracket = mol.atoms[a].no_implicit_h();
        for k in 0..m.adj[a].len() {
            let (o, bi) = m.adj[a][k];
            let (o, bi) = (o as usize, bi as usize);
            if m.charge[o] != mol.atoms[o].formal_charge && (bracket || mol.atoms[o].no_implicit_h()) {
                m.charge[o] = mol.atoms[o].formal_charge;
                m.order[bi] = 1;
                m.charge[a] += 1;
            }
        }
    }
    for a in 0..m.n() {
        let atom = &mol.atoms[a];
        if !matches!(m.z[a], 7 | 15) || m.charge[a] <= 0 || atom.no_implicit_h() {
            continue;
        }
        for k in 0..m.adj[a].len() {
            let (o, bi) = m.adj[a][k];
            let (o, bi) = (o as usize, bi as usize);
            if m.charge[o] != -1 || mol.atoms[o].no_implicit_h() {
                continue;
            }
            if m.z[o] == 8 && m.order[bi] == 1 {
                m.order[bi] = 2;
            } else if m.z[a] == 7 && m.z[o] == 7 && m.order[bi] == 2 {
                m.order[bi] = 3;
            } else {
                continue;
            }
            m.charge[o] = 0;
            m.charge[a] -= 1;
            break;
        }
    }
}

/// Whether the SMARTS `O=C(O)c1ccccc1O` (XLogP's `SALICYLIC_ACID_PTRN`)
/// matches; aromaticity as `SmartsPattern` prepares it (Daylight model).
fn salicylic_acid(m: &CdkMol) -> bool {
    let n = m.n();
    let order = |bi: u32| m.order[bi as usize];
    // carboxyl-like carbons: C(=O)(-O) with a carbon neighbour (cheap pre-check)
    let candidates: Vec<usize> = (0..n)
        .filter(|&c| {
            m.z[c] == 6
                && m.adj[c].iter().any(|&(w, bi)| m.z[w as usize] == 8 && order(bi) == 2)
                && m.adj[c].iter().any(|&(w, bi)| m.z[w as usize] == 8 && order(bi) == 1)
                && m.adj[c].iter().any(|&(w, _)| m.z[w as usize] == 6 && m.ring_atom[w as usize])
        })
        .collect();
    if candidates.is_empty() {
        return false;
    }
    let (arom, arom_bond) = crate::desc::cdk_daylight::smarts_daylight_aromaticity(m);
    let single_or_arom = |bi: u32| arom_bond[bi as usize] || m.order[bi as usize] == 1;
    let arom_c = |a: usize| m.z[a] == 6 && arom[a];
    let aliph_o = |a: usize| m.z[a] == 8 && !arom[a];
    for c2 in candidates {
        if arom[c2] {
            continue;
        }
        for &(o1, b1) in &m.adj[c2] {
            if !aliph_o(o1 as usize) || arom_bond[b1 as usize] || order(b1) != 2 {
                continue;
            }
            for &(o3, b3) in &m.adj[c2] {
                if o3 == o1 || !aliph_o(o3 as usize) || !single_or_arom(b3) {
                    continue;
                }
                for &(c4, b4) in &m.adj[c2] {
                    if !arom_c(c4 as usize) || !single_or_arom(b4) {
                        continue;
                    }
                    for &(c9, b49) in &m.adj[c4 as usize] {
                        if !arom_c(c9 as usize) || !single_or_arom(b49) {
                            continue;
                        }
                        let has_o = m.adj[c9 as usize].iter().any(|&(o10, b10)| {
                            o10 != o1 && o10 != o3 && aliph_o(o10 as usize) && single_or_arom(b10)
                        });
                        if has_o && ring_path(m, &arom_c, &single_or_arom, c4 as usize, c9 as usize) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Whether a path of five bonds through four other aromatic carbons leads from
/// `start` to `end` (the `c1ccccc1` ring closing over the bond `start`-`end`).
fn ring_path(
    m: &CdkMol,
    arom_c: &dyn Fn(usize) -> bool,
    bond_ok: &dyn Fn(u32) -> bool,
    start: usize,
    end: usize,
) -> bool {
    fn dfs(
        m: &CdkMol,
        arom_c: &dyn Fn(usize) -> bool,
        bond_ok: &dyn Fn(u32) -> bool,
        path: &mut Vec<usize>,
        end: usize,
    ) -> bool {
        let a = *path.last().unwrap();
        if path.len() == 5 {
            return m.adj[a].iter().any(|&(w, bi)| w as usize == end && bond_ok(bi));
        }
        for &(w, bi) in &m.adj[a] {
            let w = w as usize;
            if w == end || path.contains(&w) || !arom_c(w) || !bond_ok(bi) {
                continue;
            }
            path.push(w);
            let found = dfs(m, arom_c, bond_ok, path, end);
            path.pop();
            if found {
                return true;
            }
        }
        false
    }
    let mut path = vec![start];
    dfs(m, arom_c, bond_ok, &mut path, end)
}

/// Whether the atom lies on a 3-membered ring.
fn in_three_ring(m: &CdkMol, a: usize) -> bool {
    if !m.ring_atom[a] {
        return false;
    }
    let nb = &m.adj[a];
    for i in 0..nb.len() {
        for j in i + 1..nb.len() {
            if m.bond(nb[i].0 as usize, nb[j].0 as usize).is_some() {
                return true;
            }
        }
    }
    false
}

/// Packs a TPSA profile into one integer (5 bits per field, charge offset by
/// 16); profiles with a field outside the range get a key matching nothing.
const fn tpsa_key(p: &[i32; 11]) -> u64 {
    let mut key = 0u64;
    let mut i = 0;
    while i < 11 {
        let v = if i == 5 { p[i] + 16 } else { p[i] };
        if v < 0 || v >= 32 {
            return u64::MAX;
        }
        key = (key << 5) | v as u64;
        i += 1;
    }
    key
}

/// [`tpsa_key`] of every entry of [`TPSA_PROFILES`].
const TPSA_KEYS: [u64; 43] = {
    let mut keys = [0u64; 43];
    let mut i = 0;
    while i < 43 {
        keys[i] = tpsa_key(&TPSA_PROFILES[i].0);
        i += 1;
    }
    keys
};

/// `TPSADescriptor` atom profiles: (atomic number, max bond order x2, bond order
/// sum x2, neighbours, hydrogens, charge, aromatic bonds, in 3-ring, single,
/// double, triple bonds) -> contribution (Ertl et al. 2000, CDK's table).
#[rustfmt::skip]
const TPSA_PROFILES: [([i32; 11], f64); 43] = [
    ([7, 2, 6, 3, 0, 0, 0, 0, 3, 0, 0], 3.24),
    ([7, 4, 6, 2, 0, 0, 0, 0, 1, 1, 0], 12.36),
    ([7, 6, 6, 1, 0, 0, 0, 0, 0, 0, 1], 23.79),
    ([7, 4, 10, 3, 0, 0, 0, 0, 1, 2, 0], 11.68),
    ([7, 6, 10, 2, 0, 0, 0, 0, 0, 1, 1], 13.6),
    ([7, 2, 6, 3, 0, 0, 0, 1, 3, 0, 0], 3.01),
    ([7, 2, 6, 3, 1, 0, 0, 0, 3, 0, 0], 12.03),
    ([7, 2, 6, 3, 1, 0, 0, 1, 3, 0, 0], 21.94),
    ([7, 4, 6, 2, 1, 0, 0, 0, 1, 1, 0], 23.85),
    ([7, 2, 6, 3, 2, 0, 0, 0, 3, 0, 0], 26.02),
    ([7, 2, 8, 4, 0, 1, 0, 0, 4, 0, 0], 0.0),
    ([7, 4, 8, 3, 0, 1, 0, 0, 2, 1, 0], 3.01),
    ([7, 6, 8, 2, 0, 1, 0, 0, 1, 0, 1], 4.36),
    ([7, 2, 8, 4, 1, 1, 0, 0, 4, 0, 0], 4.44),
    ([7, 4, 8, 3, 1, 1, 0, 0, 2, 1, 0], 13.97),
    ([7, 2, 8, 4, 2, 1, 0, 0, 4, 0, 0], 16.61),
    ([7, 4, 8, 3, 2, 1, 0, 0, 2, 1, 0], 25.59),
    ([7, 2, 8, 4, 3, 1, 0, 0, 4, 0, 0], 27.64),
    ([7, 3, 6, 2, 0, 0, 2, 0, 0, 0, 0], 12.89),
    ([7, 3, 9, 3, 0, 0, 3, 0, 0, 0, 0], 4.41),
    ([7, 3, 8, 3, 0, 0, 2, 0, 1, 0, 0], 4.93),
    ([7, 4, 10, 3, 0, 0, 2, 0, 0, 1, 0], 8.39),
    ([7, 3, 8, 3, 1, 0, 2, 0, 1, 0, 0], 15.79),
    ([7, 3, 9, 3, 0, 1, 3, 0, 0, 0, 0], 4.1),
    ([7, 3, 8, 3, 0, 1, 2, 0, 1, 0, 0], 3.88),
    ([7, 3, 8, 3, 1, 1, 2, 0, 1, 0, 0], 14.14),
    ([8, 2, 4, 2, 0, 0, 0, 0, 2, 0, 0], 9.23),
    ([8, 2, 4, 2, 0, 0, 0, 1, 2, 0, 0], 12.53),
    ([8, 4, 4, 1, 0, 0, 0, 0, 0, 1, 0], 17.07),
    ([8, 2, 2, 1, 0, -1, 0, 0, 1, 0, 0], 23.06),
    ([8, 2, 4, 2, 1, 0, 0, 0, 2, 0, 0], 20.23),
    ([8, 3, 6, 2, 0, 0, 2, 0, 0, 0, 0], 13.14),
    ([16, 2, 4, 2, 0, 0, 0, 0, 2, 0, 0], 25.3),
    ([16, 4, 4, 1, 0, 0, 0, 0, 0, 1, 0], 32.09),
    ([16, 4, 8, 3, 0, 0, 0, 0, 2, 1, 0], 19.21),
    ([16, 4, 12, 4, 0, 0, 0, 0, 2, 2, 0], 8.38),
    ([16, 2, 4, 2, 1, 0, 0, 0, 2, 0, 0], 38.8),
    ([16, 3, 6, 2, 0, 0, 2, 0, 0, 0, 0], 28.24),
    ([16, 4, 10, 3, 0, 0, 2, 0, 0, 1, 0], 21.7),
    ([15, 2, 6, 3, 0, 0, 0, 0, 3, 0, 0], 13.59),
    ([15, 4, 6, 3, 0, 0, 0, 0, 1, 1, 0], 34.14),
    ([15, 4, 10, 4, 0, 0, 0, 0, 3, 1, 0], 9.81),
    ([15, 4, 10, 4, 1, 0, 0, 0, 3, 1, 0], 23.47),
];

/// Computes the values of [`CDK_SIMPLE_NAMES`].
pub(crate) fn cdk_simple_counts(mol: &Molecule, view: &CdkView) -> [f64; 14] {
    Prepared::with(mol, view, |p| simple_counts(mol, view, p))
}

fn simple_counts(mol: &Molecule, view: &CdkView, p: &Prepared) -> [f64; 14] {
    let m = &p.m;
    let n = m.n();
    let mut out = [0.0; 14];
    if n == 0 {
        out.fill(f64::NAN);
        return out;
    }
    // AtomCountDescriptor: atoms plus implicit hydrogens
    out[0] = (n + p.h.iter().map(|&h| h as usize).sum::<usize>()) as f64;
    // BondCountDescriptor: bonds without a hydrogen end
    out[1] = m.ends.iter().filter(|&&(s, t)| m.z[s as usize] != 1 && m.z[t as usize] != 1).count() as f64;
    out[2] = p.arom_atom.iter().filter(|&&x| x).count() as f64;
    out[3] = p.arom_bond.iter().filter(|&&x| x).count() as f64;
    out[4] = p.hbond_acceptors() as f64;
    out[5] = p.hbond_donors() as f64;
    out[6] = p.rotatable_bonds(false) as f64;
    let tpsa = p.tpsa();
    out[7] = tpsa;
    out[8] = p.tpsa_efficiency(tpsa);
    out[9] = p.weight();
    out[10] = fmf(m);
    out[11] = fsp3(m, view);
    out[12] = spiro_atoms(m) as f64;
    out[13] = vabc(mol, m, view);
    out
}

/// `FMFDescriptor`: Murcko framework atoms (`MurckoFragmenter(true, 3)`, i.e.
/// the scaffold left after repeatedly removing terminal atoms) over all atoms.
fn fmf(m: &CdkMol) -> f64 {
    let n = m.n();
    let mut bcount: Vec<i32> = (0..n).map(|a| m.adj[a].len() as i32).collect();
    let mut queue: std::collections::VecDeque<usize> = (0..n).filter(|&a| bcount[a] == 1).collect();
    while let Some(a) = queue.pop_front() {
        bcount[a] = 0;
        for &(w, _) in &m.adj[a] {
            let w = w as usize;
            bcount[w] -= 1;
            if bcount[w] == 1 {
                queue.push_back(w);
            }
        }
    }
    let scaffold = bcount.iter().filter(|&&c| c > 0).count();
    if scaffold >= 3 { scaffold as f64 / n as f64 } else { 0.0 }
}

/// `FractionalCSP3Descriptor`: carbons whose CDK atom type is sp3.
fn fsp3(m: &CdkMol, view: &CdkView) -> f64 {
    let (mut n_c, mut n_sp3) = (0, 0);
    for a in 0..m.n() {
        if m.z[a] == 6 {
            n_c += 1;
            if view.types[a].is_some_and(|t| t.hyb == Hyb::Sp3) {
                n_sp3 += 1;
            }
        }
    }
    if n_c == 0 { 0.0 } else { n_sp3 as f64 / n_c as f64 }
}

/// `SpiroAtomCountDescriptor`: ring atoms with at least four ring bonds whose
/// ring bonds lead into two or more components once the atom is removed.
fn spiro_atoms(m: &CdkMol) -> u32 {
    let n = m.n();
    let mut count = 0;
    let mut seen = Vec::new();
    let mut stack = Vec::new();
    for a in 0..n {
        if !m.ring_atom[a] || m.adj[a].iter().filter(|&&(_, bi)| m.ring_bond[bi as usize]).count() < 4 {
            continue;
        }
        seen.clear();
        seen.resize(n, false);
        seen[a] = true;
        let mut degree = 0;
        for &(w, bi) in &m.adj[a] {
            if !m.ring_bond[bi as usize] || seen[w as usize] {
                continue;
            }
            degree += 1;
            seen[w as usize] = true;
            stack.push(w as usize);
            while let Some(x) = stack.pop() {
                for &(y, _) in &m.adj[x] {
                    if !seen[y as usize] {
                        seen[y as usize] = true;
                        stack.push(y as usize);
                    }
                }
            }
        }
        if degree >= 2 {
            count += 1;
        }
    }
    count
}

/// Implicit hydrogens after `CDKHydrogenAdder`: the formal neighbour count of
/// the atom type minus the degree, the parsed count for untyped (`X`) atoms
/// (e.g. a radical `[B]` gets two hydrogens, `[Zn+2]` two, like in CDK).
fn cdk_hydrogens(mol: &Molecule, m: &CdkMol, view: &CdkView) -> Vec<u8> {
    (0..m.n())
        .map(|a| match m.z[a] {
            6 | 7 | 8 | 15 | 16 | 33 => view.h[a],
            _ => formal_neighbour_count(mol, m, view, a)
                .map_or(view.h[a], |fnc| fnc.saturating_sub(m.adj[a].len()) as u8),
        })
        .collect()
}

/// Number of independent cycles (bonds - atoms + connected components).
fn cycle_rank(m: &CdkMol) -> usize {
    let n = m.n();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let mut components = n;
    for &(s, t) in &m.ends {
        let (rs, rt) = (find(&mut parent, s), find(&mut parent, t));
        if rs != rt {
            parent[rs as usize] = rt;
            components -= 1;
        }
    }
    m.ends.len() + components - n
}

/// Bondi volume per element (`VABCVolume`); `None`: unsupported element.
fn bondi_volume(z: u8) -> Option<f64> {
    Some(match z {
        1 => 7.2382293504,
        6 => 20.5795259250667,
        7 => 15.5985308577667,
        8 => 14.7102267005611,
        17 => 22.4492971208333,
        35 => 26.5218483279667,
        9 => 13.3057882007064,
        53 => 32.5150310206656,
        16 | 15 => 24.4290240576,
        33 => 26.5218483279667,
        5 => 40.48,
        34 => 28.7309115245333,
        14 => 38.7923854248,
        _ => return None,
    })
}

/// `VABCDescriptor` (`VABCVolume.calculate`): NaN for unsupported elements.
/// Hydrogens are the formal neighbour count of the atom type minus the
/// explicit degree; CDK's `X` type (no match) has a formal neighbour count of
/// 0, so untyped atoms subtract their degree.
fn vabc(mol: &Molecule, m: &CdkMol, view: &CdkView) -> f64 {
    let n = m.n();
    let h_vol = 7.2382293504;
    let mut sum = 0.0;
    let mut total_h = 0i64;
    for a in 0..n {
        let Some(v) = bondi_volume(m.z[a]) else {
            return f64::NAN;
        };
        sum += v;
        let fnc = formal_neighbour_count(mol, m, view, a).unwrap_or(0) as i64;
        let h = fnc - m.adj[a].len() as i64;
        sum += h as f64 * h_vol;
        total_h += h;
    }
    sum -= 5.92 * (m.ends.len() as i64 + total_h) as f64;
    // cdkLegacy aromaticity; rings of the SSSR (as many as the cycle rank).
    // The symmetrized SSSR is the SSSR unless it has extra rings (cages).
    let rank = cycle_rank(m);
    let mut arom_rings = 0;
    let symmetrized = &mol.ring_info().expect("ring info").atom_rings;
    if symmetrized.len() == rank {
        for ring in symmetrized {
            if ring.iter().all(|&a| view.arom_atom[a as usize]) {
                arom_rings += 1;
            }
        }
    } else {
        for ring in crate::algo::rings::minimum_cycle_basis_bonds_cached(mol) {
            let aromatic = ring.iter().all(|&b| {
                let bond = &mol.bonds[b as usize];
                view.arom_atom[bond.source as usize] && view.arom_atom[bond.target as usize]
            });
            if aromatic {
                arom_rings += 1;
            }
        }
    }
    let other_rings = rank.saturating_sub(arom_rings);
    if arom_rings + other_rings > 0 {
        sum -= 14.7 * arom_rings as f64;
        sum -= 3.8 * other_rings as f64;
    }
    sum
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<CdkSimpleCounts>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// `LipinskiFailures` of the CDK reference, given the reference `XLogP`.
    #[test]
    fn rule_of_five_matches_cdk_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/cdk_parity_reference.csv");
        let mut rdr = csv::ReaderBuilder::new().comment(Some(b'#')).from_path(path).unwrap();
        let header = rdr.headers().unwrap().clone();
        let col = |name: &str| header.iter().position(|h| h == name).unwrap();
        let (c_smi, c_xlogp, c_ro5) = (col("smiles"), col("XLogP"), col("LipinskiFailures"));
        let mut checked = 0;
        for rec in rdr.records() {
            let rec = rec.unwrap();
            let (Ok(xlogp), Ok(expected)) = (rec[c_xlogp].parse::<f64>(), rec[c_ro5].parse::<u32>()) else {
                continue;
            };
            let mol = parse_smiles(&rec[c_smi]).unwrap();
            let mut ctx = CalculationContext::new();
            assert_eq!(cdk_rule_of_five(&mol, &mut ctx, xlogp), expected, "{}", &rec[c_smi]);
            checked += 1;
        }
        assert!(checked > 1200);
    }

    #[test]
    fn salicylic_acid_pattern() {
        let check = |smi: &str| {
            let mol = parse_smiles(smi).unwrap();
            let mut m = CdkMol::new(&mol);
            restore_parser_state(&mol, &mut m);
            salicylic_acid(&m)
        };
        assert!(check("OC(=O)c1ccccc1O"));
        assert!(check("CC(=O)Oc1ccccc1C(O)=O"));
        assert!(!check("OC(=O)c1ccc(O)cc1"));
        assert!(!check("OC(=O)C1=CCCC=C1O"));
    }
}
