//! CDK logP / molar refractivity models (CDK 2.13).
//!
//! One batch descriptor computes the four CDK models on a shared CDK view of
//! the molecule:
//!
//! - `ALOGPDescriptor` (`ALogP`, `ALogp2`, `AMR`, see `cdk_alogp`),
//! - `XLogPDescriptor` with default parameters (`XLogP`, see `cdk_xlogp`),
//! - `MannholdLogPDescriptor` (`MLogP`),
//! - `JPlogPDescriptor` with default parameters (`JPLogP`, see `cdk_jplogp`).
//!
//! The view reproduces the molecule as the CDK reference prepares it
//! (`SmilesParser`, `percieveAtomTypesAndConfigureAtoms`, `CDKHydrogenAdder`,
//! `Aromaticity(ElectronDonation.daylight(), Cycles.or(Cycles.all(), Cycles.all(6)))`):
//!
//! - CDK's bond order and Kekulé structure (`cdk_daylight::CdkMol`; a Kekulé
//!   structure written in the input is kept, see `Bond::is_input_double`),
//! - RDKit's clean-up of nitro-type groups undone, bracket halogen oxides kept
//!   charged,
//! - the hydrogen counts `CDKHydrogenAdder` assigns (radicals filled),
//! - the Daylight aromaticity flags of atoms and bonds.
//!
//! Key symbols: `struct CdkLogP`, `fn cdk_logp_values`, `struct LogpView`.

use crate::desc::cdk_daylight::{CdkMol, daylight_model_contrib};
use crate::desc::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::BondType;
use smallvec::SmallVec;

/// CDK 2.13 logP / molar refractivity models.
///
/// Returns FloatVec with 6 values:
/// - [0]: `ALogP` - Ghose-Crippen logKow (`ALOGPDescriptor`)
/// - [1]: `ALogp2` - its square
/// - [2]: `AMR` - Ghose-Crippen molar refractivity
/// - [3]: `XLogP` - `XLogPDescriptor` (default parameters)
/// - [4]: `MLogP` - `MannholdLogPDescriptor`
/// - [5]: `JPLogP` - `JPlogPDescriptor` (default parameters; NaN when an atom
///   type has no coefficient, as in CDK)
///
/// # Thread Safety
/// Thread-safe: stateless and uses only immutable references.
pub struct CdkLogP;

/// Output column names of [`CdkLogP`].
pub const CDK_LOGP_NAMES: [&str; 7] = ["ALogP", "ALogp2", "AMR", "XLogP", "MLogP", "JPLogP", "LipinskiFailures"];

impl Descriptor for CdkLogP {
    fn name(&self) -> &str {
        "CdkLogP"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::FloatVec(cdk_logp_values(mol, ctx).to_vec())
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<CdkLogP>;
};

/// The values of [`CdkLogP`]. `LipinskiFailures` (CDK `RuleOfFiveDescriptor`)
/// is computed here because it needs this XLogP (CDK's rule of five calls
/// `XLogPDescriptor` with the salicylic-acid correction, which
/// `cdk_rule_of_five` adds).
pub(crate) fn cdk_logp_values(
    mol: &Molecule,
    ctx: &mut crate::desc::context::CalculationContext,
) -> [f64; 7] {
    let mlogp = mannhold_logp(mol);
    if mol.num_atoms() == 0 {
        return [0.0, 0.0, 0.0, 0.0, mlogp, 0.0, 0.0];
    }
    let v = LogpView::new(mol);
    let [alogp, alogp2, amr] = super::cdk_alogp::alogp(mol, &v);
    let xlogp = super::cdk_xlogp::xlogp(mol, &v);
    let jplogp = super::cdk_jplogp::jplogp(mol, &v, ctx);
    let lipinski = super::cdk_simple::cdk_rule_of_five(mol, ctx, xlogp) as f64;
    [alogp, alogp2, amr, xlogp, mlogp, jplogp, lipinski]
}

/// CDK 2.13 `MannholdLogPDescriptor`: `1.46 + 0.11 * nC - 0.11 * nHetero`,
/// counting every non-hydrogen atom that is not carbon as hetero atom.
fn mannhold_logp(mol: &Molecule) -> f64 {
    let mut carbon = 0i32;
    let mut hetero = 0i32;
    for a in &mol.atoms {
        match a.atomic_number {
            1 => {}
            6 => carbon += 1,
            _ => hetero += 1,
        }
    }
    1.46 + 0.11 * carbon as f64 - 0.11 * hetero as f64
}

/// The molecule as the CDK reference preparation leaves it (see module docs).
pub(crate) struct LogpView {
    /// Elements, charges, implicit H counts, Kekulé orders, CDK bond order.
    pub m: CdkMol,
    /// Daylight aromaticity flags of atoms.
    pub arom: Vec<bool>,
    /// Daylight aromaticity flags of bonds.
    pub arom_bond: Vec<bool>,
}

impl LogpView {
    pub(crate) fn new(mol: &Molecule) -> Self {
        let mut m = CdkMol::new(mol);
        restore_input_kekule(mol, &mut m);
        restore_bracket_halogens(mol, &mut m);
        revert_oxide_cleanup(mol, &mut m);
        for a in 0..m.n() {
            m.hcount[a] = cdk_hydrogens(mol, a);
        }
        let (arom, arom_bond) = daylight_flags(mol, &m);
        Self { m, arom, arom_bond }
    }

    /// Number of atoms (explicit hydrogens of the input included).
    #[inline]
    pub fn n(&self) -> usize {
        self.m.n()
    }

    /// Hydrogens attached to `a` once all are explicit: implicit ones plus
    /// hydrogen atoms of the input.
    #[inline]
    pub fn total_h(&self, a: usize) -> usize {
        self.m.hcount[a] as usize
            + self.m.adj[a]
                .iter()
                .filter(|&&(w, _)| self.m.z[w as usize] == 1)
                .count()
    }

    /// Bond count once all hydrogens are explicit.
    #[inline]
    pub fn degree_all(&self, a: usize) -> usize {
        self.m.adj[a].len() + self.m.hcount[a] as usize
    }
}

/// CDK's SMILES parser keeps a Kekulé structure as written; only aromatic
/// (lower-case) input is kekulized, which `CdkMol` reproduces for every
/// RDKit-aromatic bond. For each connected system of RDKit-aromatic bonds
/// whose input double bonds ([`Bond::is_input_double`]) form a matching that
/// covers every atom `CdkMol` gave a double bond (more atoms when the input
/// has radicals, which CDK fills with hydrogens), the written orders are
/// restored.
///
/// [`Bond::is_input_double`]: crate::mol::bond::Bond::is_input_double
fn restore_input_kekule(mol: &Molecule, m: &mut CdkMol) {
    let nb = mol.num_bonds();
    let arom = |b: usize| mol.bonds[b].bond_type == BondType::Aromatic;
    if !(0..nb).any(|b| arom(b) && mol.bonds[b].is_input_double()) {
        return;
    }
    let n = m.n();
    let mut seen = vec![false; nb];
    let mut stack: Vec<usize> = Vec::new();
    let mut system: Vec<usize> = Vec::new();
    let mut mark = vec![0u8; n]; // bit 0: Beam double, bit 1: written double
    for start in 0..nb {
        if seen[start] || !arom(start) {
            continue;
        }
        // collect the connected system of aromatic bonds
        system.clear();
        seen[start] = true;
        stack.push(start);
        while let Some(b) = stack.pop() {
            system.push(b);
            let (s, t) = m.ends[b];
            for x in [s as usize, t as usize] {
                for &(_, bi) in &m.adj[x] {
                    let bi = bi as usize;
                    if !seen[bi] && arom(bi) {
                        seen[bi] = true;
                        stack.push(bi);
                    }
                }
            }
        }
        let mut ok = system.iter().any(|&b| mol.bonds[b].is_input_double());
        for &b in &system {
            let (s, t) = (m.ends[b].0 as usize, m.ends[b].1 as usize);
            if m.order[b] == 2 {
                mark[s] |= 1;
                mark[t] |= 1;
            }
            if mol.bonds[b].is_input_double() {
                if mark[s] & 2 != 0 || mark[t] & 2 != 0 {
                    ok = false; // not a matching
                }
                mark[s] |= 2;
                mark[t] |= 2;
            }
        }
        for &b in &system {
            let (s, t) = (m.ends[b].0 as usize, m.ends[b].1 as usize);
            for x in [s, t] {
                if mark[x] == 1 {
                    ok = false; // an atom `CdkMol` matched has no written double bond
                }
            }
        }
        if ok {
            for &b in &system {
                m.order[b] = if mol.bonds[b].is_input_double() { 2 } else { 1 };
            }
        }
        for &b in &system {
            mark[m.ends[b].0 as usize] = 0;
            mark[m.ends[b].1 as usize] = 0;
        }
    }
}

/// `CdkMol` turns every charged halogen oxide back into the `Cl(=O)` form
/// RDKit's `cleanUp` started from; a halogen written as a bracket atom
/// (`[O-][Cl+3]([O-])([O-])O`) was not cleaned up, and CDK keeps it charged.
fn restore_bracket_halogens(mol: &Molecule, m: &mut CdkMol) {
    for a in 0..m.n() {
        let atom = &mol.atoms[a];
        if !matches!(m.z[a], 17 | 35 | 53)
            || !atom.no_implicit_h()
            || m.charge[a] == atom.formal_charge
        {
            continue;
        }
        m.charge[a] = atom.formal_charge;
        for k in 0..m.adj[a].len() {
            let (w, bi) = m.adj[a][k];
            m.charge[w as usize] = mol.atoms[w as usize].formal_charge;
            m.order[bi as usize] = match mol.bonds[bi as usize].bond_type {
                BondType::Double => 2,
                BondType::Triple => 3,
                _ => 1,
            };
        }
    }
}

/// Hydrogen count CDK assigns to an atom (`CDKHydrogenAdder` after
/// `CDKAtomTypeMatcher`), as verified for the PubChem fingerprint
/// (`pubchem_cdk::cdk_hydrogens`): the RDKit count, except for radicals that
/// CDK fills with hydrogens and a few atom types with unusual formal
/// neighbour counts.
fn cdk_hydrogens(mol: &Molecule, a: usize) -> u8 {
    let atom = &mol.atoms[a];
    let h = atom.total_h();
    let z = atom.atomic_number;
    let q = atom.formal_charge;
    let degree = mol.adjacency[a].len();
    let mut valence = h as u32;
    let mut doubles = 0;
    for &b in &mol.adjacency[a] {
        valence += match mol.bonds[b as usize].bond_type {
            BondType::Double => {
                doubles += 1;
                2
            }
            BondType::Triple => 3,
            _ => 1,
        };
    }
    // atom types with fixed formal neighbour counts
    let fnc: Option<u8> = match (z, q, degree) {
        (12, 0, 0) => Some(4),                    // Mg.neutral
        (14, -2, 0) => Some(6),                   // Si.2minus.6
        (26, -3 | -2, 0) => Some(6),              // Fe.3minus / Fe.2minus
        (30, 2, 0) => Some(2),                    // Zn.2plus
        (33, -1, 0) => Some(6),                   // As.minus
        (37, 0, 0) => Some(1),                    // Rb.neutral
        (44, 0, _) | (44, -3 | -2, 0) => Some(6), // Ru.6, Ru.3minus.6, Ru.2minus.6
        (80, -1, 0) => Some(2),                   // Hg.minus
        (80, 1, 0) => Some(1),                    // Hg.plus
        (16, 0, 3) if doubles == 3 => Some(6),    // S.trioxide
        _ => None,
    };
    if let Some(fnc) = fnc {
        return fnc.saturating_sub(degree as u8);
    }
    let rad = atom.num_radicals();
    if rad == 0 {
        return h;
    }
    let filled = valence + rad as u32;
    let fill = match z {
        5 | 6 | 7 | 8 | 9 | 17 | 32 | 33 | 35 | 50 | 53 => true,
        // S.3 (divalent) and S.sp3d1 (four neighbours, one double bond)
        16 => filled == 2 || (degree == 4 && doubles == 1),
        // Se.3: divalent with neighbours (a lone [Se] is typed Se.2 without H)
        34 => filled == 2 && degree > 0,
        // P.ine (0 or 2 single-bonded neighbours) and P.ate (3 neighbours, one =X)
        15 => match degree {
            0 => filled == 3,
            2 => doubles == 0 && filled == 3,
            3 => doubles == 1 && filled == 5,
            _ => false,
        },
        _ => false,
    };
    if fill { h + rad } else { h }
}

/// Undoes RDKit's `cleanUp` of hypervalent nitrogen and phosphorus oxides
/// (`N(=O)=O` -> `[N+](=O)[O-]`, `C=N#N`-type groups), which CDK keeps as
/// written. Atoms from the SMILES organic subset cannot carry a charge, so
/// charged non-bracket atoms identify clean-up results.
fn revert_oxide_cleanup(mol: &Molecule, m: &mut CdkMol) {
    for a in 0..m.n() {
        let atom = &mol.atoms[a];
        let z = m.z[a];
        if m.charge[a] <= 0 || atom.no_implicit_h() || !matches!(z, 7 | 15) {
            continue;
        }
        let mut restored = 0i8;
        for k in 0..m.adj[a].len() {
            let (o, bi) = m.adj[a][k];
            let (o, bi) = (o as usize, bi as usize);
            if m.charge[o] != -1 || mol.atoms[o].no_implicit_h() {
                continue;
            }
            if m.z[o] == 8 && m.order[bi] == 1 {
                m.order[bi] = 2;
            } else if z == 7 && m.z[o] == 7 && m.order[bi] == 2 {
                m.order[bi] = 3;
            } else {
                continue;
            }
            m.charge[o] = 0;
            restored += 1;
            break;
        }
        m.charge[a] -= restored;
    }
}

/// `(sum - 2) % 4 == 0` with Java's remainder semantics.
#[inline]
fn huckel(sum: i32) -> bool {
    (sum - 2) % 4 == 0
}

/// Daylight aromaticity as set by `Aromaticity(ElectronDonation.daylight(),
/// Cycles.or(Cycles.all(), Cycles.all(6)))`: a bond is aromatic when it lies
/// on a simple cycle of contributing atoms whose electron sum satisfies
/// Hückel's rule; an atom when one of its bonds is.
fn daylight_flags(mol: &Molecule, m: &CdkMol) -> (Vec<bool>, Vec<bool>) {
    let n = m.n();
    let nb = m.ends.len();
    let mut arom = vec![false; n];
    let mut arom_bond = vec![false; nb];
    // candidate bonds: ring bonds between contributing atoms
    if !(0..nb).any(|b| m.ring_bond[b]) {
        return (arom, arom_bond);
    }
    let contrib = daylight_model_contrib(m);
    let cand = |b: usize| {
        let (s, t) = m.ends[b];
        m.ring_bond[b] && contrib[s as usize] >= 0 && contrib[t as usize] >= 0
    };
    if !(0..nb).any(cand) {
        return (arom, arom_bond);
    }
    // shortcut: rings of the ring perception that are Hückel cycles
    if let Some(info) = mol.ring_info() {
        for ring in &info.atom_rings {
            if ring.iter().all(|&a| contrib[a as usize] >= 0)
                && huckel(ring.iter().map(|&a| contrib[a as usize]).sum())
            {
                let k = ring.len();
                for i in 0..k {
                    let (u, v) = (ring[i] as usize, ring[(i + 1) % k] as usize);
                    if let Some(bi) = m.bond(u, v) {
                        arom_bond[bi] = true;
                    }
                }
            }
        }
    }
    let mut on_path = vec![false; n];
    for b in 0..nb {
        if arom_bond[b] || !cand(b) {
            continue;
        }
        arom_bond[b] = huckel_cycle_through(m, &contrib, b, &mut on_path);
    }
    for b in 0..nb {
        if arom_bond[b] {
            let (s, t) = m.ends[b];
            arom[s as usize] = true;
            arom[t as usize] = true;
        }
    }
    (arom, arom_bond)
}

/// Whether bond `b` lies on a Hückel cycle (see [`daylight_flags`]). The
/// search is bounded; when the budget is exhausted only cycles up to size 6
/// are considered (`Cycles.or(Cycles.all(), Cycles.all(6))`).
fn huckel_cycle_through(m: &CdkMol, contrib: &[i32], b: usize, on_path: &mut [bool]) -> bool {
    let (u, v) = (m.ends[b].0 as usize, m.ends[b].1 as usize);
    let mut budget: usize = 2_000_000;
    on_path[u] = true;
    on_path[v] = true;
    let start = contrib[u] + contrib[v];
    let mut found = path_dfs(
        m,
        contrib,
        u,
        v,
        b,
        start,
        2,
        usize::MAX,
        on_path,
        &mut budget,
    );
    if !found && budget == 0 {
        let mut budget = usize::MAX;
        found = path_dfs(m, contrib, u, v, b, start, 2, 6, on_path, &mut budget);
    }
    on_path[u] = false;
    on_path[v] = false;
    found
}

/// Depth-first search for a path from `a` back to `target` closing a cycle of
/// at most `max_len` atoms with a Hückel electron sum.
#[allow(clippy::too_many_arguments)]
fn path_dfs(
    m: &CdkMol,
    contrib: &[i32],
    target: usize,
    a: usize,
    prev: usize,
    sum: i32,
    len: usize,
    max_len: usize,
    on_path: &mut [bool],
    budget: &mut usize,
) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    for &(w, bi) in &m.adj[a] {
        let (w, bi) = (w as usize, bi as usize);
        if bi == prev || !m.ring_bond[bi] || contrib[w] < 0 {
            continue;
        }
        if w == target {
            if len >= 3 && huckel(sum) {
                return true;
            }
            continue;
        }
        if on_path[w] || len >= max_len {
            continue;
        }
        on_path[w] = true;
        let r = path_dfs(
            m,
            contrib,
            target,
            w,
            bi,
            sum + contrib[w],
            len + 1,
            max_len,
            on_path,
            budget,
        );
        on_path[w] = false;
        if r {
            return true;
        }
        if *budget == 0 {
            return false;
        }
    }
    false
}

/// CDK's minimum cycle basis (`Cycles.sssr` -> `MinimumCycleBasis` of
/// `InitialCycles.ofBiconnectedComponent`) of one fused ring system, given by
/// its atoms in ascending order. Returns the basis cycles as atom lists.
///
/// The basis is not unique; CDK's choice follows from its vertex ordering
/// (degree, then index), the first shortest path of its ordered BFS and the
/// lexicographic order of equally long candidate cycles, all reproduced here.
pub(crate) fn cdk_mcb(v: &LogpView, verts: &[usize]) -> Vec<Vec<usize>> {
    let m = &v.m;
    let n = verts.len();
    let pos = |x: usize| verts.binary_search(&x).ok();
    // GraphUtil.subgraph: adjacency in CDK bond order
    let graph: Vec<SmallVec<[usize; 4]>> = verts
        .iter()
        .map(|&x| {
            m.adj[x]
                .iter()
                .filter_map(|&(w, _)| pos(w as usize))
                .collect()
        })
        .collect();
    let n_edges = graph.iter().map(|g| g.len()).sum::<usize>() / 2;
    // InitialCycles.ordering: counting sort by degree, stable in vertex order
    let max_deg = graph.iter().map(|g| g.len()).max().unwrap_or(0);
    let mut count = vec![0usize; max_deg + 2];
    for g in &graph {
        count[g.len() + 1] += 1;
    }
    for i in 1..count.len() {
        count[i] += count[i - 1];
    }
    let mut ord = vec![0usize; n];
    for x in 0..n {
        ord[x] = count[graph[x].len()];
        count[graph[x].len()] += 1;
    }
    let n_deg2 = count.get(2).copied().unwrap_or(0);
    let mut vertices = vec![0usize; n];
    for x in 0..n {
        vertices[ord[x]] = x;
    }
    // edge indices for edge vectors
    let mut edge_index: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); n];
    {
        let mut k = 0;
        for x in 0..n {
            for &w in &graph[x] {
                if w > x {
                    edge_index[x].push(k);
                    k += 1;
                } else {
                    edge_index[x].push(usize::MAX);
                }
            }
        }
        for x in 0..n {
            for p in 0..graph[x].len() {
                let w = graph[x][p];
                if w < x {
                    let q = graph[w].iter().position(|&y| y == x).unwrap();
                    edge_index[x][p] = edge_index[w][q];
                }
            }
        }
    }
    let edge_of =
        |a: usize, b: usize| edge_index[a][graph[a].iter().position(|&y| y == b).unwrap()];
    let words = n_edges.div_ceil(64).max(1);
    let first = if n_deg2 < n { n_deg2 } else { 2 };
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    let mut dist = vec![usize::MAX; n];
    let mut parent = vec![usize::MAX; n];
    let mut precedes = vec![false; n];
    let mut queue: Vec<usize> = Vec::with_capacity(n);
    let path_to = |x: usize, dist: &[usize], parent: &[usize]| -> Vec<usize> {
        let mut p = vec![0usize; dist[x] + 1];
        let mut y = x;
        loop {
            p[dist[y]] = y;
            if dist[y] == 0 {
                break;
            }
            y = parent[y];
        }
        p
    };
    let singleton = |p: &[usize], q: &[usize]| (1..p.len()).all(|i| p[i] != q[i]);
    for i in first..n {
        let r = vertices[i];
        // ShortestPaths with ordering
        dist.fill(usize::MAX);
        precedes.fill(false);
        dist[r] = 0;
        precedes[r] = true;
        queue.clear();
        queue.push(r);
        let mut head = 0;
        while head < queue.len() {
            let x = queue[head];
            head += 1;
            let d = dist[x] + 1;
            for &w in &graph[x] {
                if d < dist[w] {
                    dist[w] = d;
                    parent[w] = x;
                    precedes[w] = precedes[x] && ord[w] < ord[r];
                    queue.push(w);
                } else if dist[w] == d && precedes[x] && ord[w] < ord[r] && !precedes[w] {
                    precedes[w] = true;
                    parent[w] = x;
                }
            }
        }
        for &y in &vertices[..i] {
            if !precedes[y] {
                continue;
            }
            let mut s: SmallVec<[usize; 4]> = SmallVec::new();
            for &z in &graph[y] {
                if !precedes[z] {
                    continue;
                }
                if dist[z] + 1 == dist[y] {
                    s.push(z);
                } else if dist[z] == dist[y] && ord[z] < ord[y] {
                    let py = path_to(y, &dist, &parent);
                    let pz = path_to(z, &dist, &parent);
                    if singleton(&pz, &py) {
                        let mut c = py;
                        c.extend(pz.iter().rev());
                        cycles.push(c);
                    }
                }
            }
            for k in 0..s.len() {
                for l in k + 1..s.len() {
                    let pp = path_to(s[k], &dist, &parent);
                    let pq = path_to(s[l], &dist, &parent);
                    if singleton(&pp, &pq) {
                        let mut c = pp;
                        c.push(y);
                        c.extend(pq.iter().rev());
                        cycles.push(c);
                    }
                }
            }
        }
    }
    // Cycle.compareTo: length, then the closed walk lexicographically
    cycles.sort_unstable_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    cycles.dedup();
    // GreedyBasis
    let lim = n_edges + 1 - n;
    let mut rows: Vec<(Vec<u64>, usize)> = Vec::new(); // reduced rows with pivot bit
    let mut out: Vec<Vec<usize>> = Vec::new();
    for c in cycles {
        if out.len() >= lim {
            break;
        }
        let mut row = vec![0u64; words];
        for w in c.windows(2) {
            let e = edge_of(w[0], w[1]);
            row[e / 64] |= 1 << (e % 64);
        }
        for (brow, piv) in &rows {
            if row[piv / 64] >> (piv % 64) & 1 == 1 {
                for k in 0..words {
                    row[k] ^= brow[k];
                }
            }
        }
        let Some(piv) = (0..n_edges).find(|&e| row[e / 64] >> (e % 64) & 1 == 1) else {
            continue; // dependent
        };
        for (brow, _) in rows.iter_mut() {
            if brow[piv / 64] >> (piv % 64) & 1 == 1 {
                for k in 0..words {
                    brow[k] ^= row[k];
                }
            }
        }
        rows.push((row, piv));
        out.push(c[..c.len() - 1].iter().map(|&x| verts[x]).collect());
    }
    out
}

/// Biconnected components of the subgraph of bonds selected by `keep`:
/// the block of every bond (`u32::MAX` outside the subgraph) and the number
/// of bonds of every block.
pub(crate) fn edge_blocks(v: &LogpView, keep: impl Fn(usize) -> bool) -> (Vec<u32>, Vec<u32>) {
    let m = &v.m;
    let n = m.n();
    let nb = m.ends.len();
    let mut block = vec![u32::MAX; nb];
    let mut size: Vec<u32> = Vec::new();
    let mut disc = vec![u32::MAX; n];
    let mut low = vec![0u32; n];
    let mut time = 0u32;
    let mut edge_stack: Vec<usize> = Vec::new();
    // (vertex, parent bond, next adjacency position)
    let mut stack: Vec<(usize, usize, usize)> = Vec::new();
    for root in 0..n {
        if disc[root] != u32::MAX {
            continue;
        }
        disc[root] = time;
        low[root] = time;
        time += 1;
        stack.push((root, usize::MAX, 0));
        while let Some(&mut (u, pb, ref mut k)) = stack.last_mut() {
            if *k < m.adj[u].len() {
                let (w, b) = m.adj[u][*k];
                *k += 1;
                let (w, b) = (w as usize, b as usize);
                if b == pb || !keep(b) {
                    continue;
                }
                if disc[w] == u32::MAX {
                    edge_stack.push(b);
                    disc[w] = time;
                    low[w] = time;
                    time += 1;
                    stack.push((w, b, 0));
                } else if disc[w] < disc[u] {
                    edge_stack.push(b);
                    low[u] = low[u].min(disc[w]);
                }
            } else {
                stack.pop();
                if let Some(&(p, _, _)) = stack.last() {
                    low[p] = low[p].min(low[u]);
                    if low[u] >= disc[p] {
                        let id = size.len() as u32;
                        let mut count = 0;
                        while let Some(e) = edge_stack.pop() {
                            block[e] = id;
                            count += 1;
                            if e == pb {
                                break;
                            }
                        }
                        size.push(count);
                    }
                }
            }
        }
    }
    (block, size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn values(smi: &str) -> [f64; 7] {
        let mol = parse_smiles(smi).unwrap();
        let mut ctx = crate::desc::context::CalculationContext::new();
        cdk_logp_values(&mol, &mut ctx)
    }

    fn close(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 + 1e-9 * b.abs()
    }

    #[test]
    fn mannhold() {
        let mol = parse_smiles("CCO").unwrap();
        assert!((mannhold_logp(&mol) - (1.46 + 0.22 - 0.11)).abs() < 1e-12);
    }

    #[test]
    fn daylight_flags_basic() {
        let mol = parse_smiles("c1ccccc1C1CC1").unwrap();
        let v = LogpView::new(&mol);
        assert_eq!(v.arom.iter().filter(|&&x| x).count(), 6);
        assert_eq!(v.arom_bond.iter().filter(|&&x| x).count(), 6);
    }

    /// CDK 2.13 values `[ALogP, XLogP, JPLogP]` for cases that need the
    /// special handling of this module.
    #[test]
    fn cdk_reference_values() {
        let cases: [(&str, [f64; 3]); 6] = [
            // written Kekulé structure kept (hydrogen on C=N)
            (
                "[N+]1(C(N)=CC=CC=1)[O-]",
                [0.24239999999999973, 1.6750000000000003, 0.2259773812690915],
            ),
            // radical carbon filled with hydrogen, ring as written
            (
                "CC(=O)[O-].[C]1=CC=CC=C1.[Hg+]",
                [0.922299999999999, 1.461, f64::NAN],
            ),
            // untyped bracket atom keeps its hydrogen
            (
                "C[S@@H+](C)[O-]",
                [-0.5895000000000004, 0.589, -0.45479542505941994],
            ),
            // bracket perchlorate stays charged
            (
                "C1CCN2CC3CC(CN4CCCC[C@@H]34)[C@H]2C1.[O-][Cl+3]([O-])([O-])O",
                [-0.47900000000000076, 0.7639999999999998, f64::NAN],
            ),
            // CDK's minimum cycle basis puts the pyrrole N into the 13-ring
            (
                "COC1=CC(=C2N=CC=C2Cl)OC1=C1C2=C3C=C(CCCCCCCCC1[C@@H]3C(C)C)N2",
                [6.577199999999994, 7.799000000000001, 7.5433423995291005],
            ),
            // AllRingsFinder exceeds its threshold: ALogP is NaN
            (
                "C1C2[C@H]3C[C@H]4[C@@H]5C[C@@H]6C7CC8[C@@H]9C[C@@H]%10[C@H]%11C[C@H]%12C1[C@@H]%10C([C@H]69)([C@H]25)C([C@H]3%12)([C@H]74)[C@@H]8%11",
                [f64::NAN, 9.908000000000005, 4.0603124587253925],
            ),
        ];
        for (smi, [alogp, xlogp, jplogp]) in cases {
            let v = values(smi);
            assert!(
                close(v[0], alogp) && close(v[3], xlogp) && close(v[5], jplogp),
                "{smi}: {v:?}"
            );
        }
    }
}
