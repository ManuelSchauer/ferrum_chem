//! Kier-Hall Chi (χ) molecular connectivity indices.
//!
//! Two descriptor families live here:
//!
//! - [`ChiIndices`]: RDKit's `Chi0`, `Chi1`, `Chi0v`..`Chi4v` (7 values),
//!   computed on the RDKit-perceived molecule.
//! - [`CdkChiIndices`]: CDK 2.13's `ChiPathDescriptor` (`SP-0..7`, `VP-0..7`),
//!   `ChiClusterDescriptor` (`SC-3..6`, `VC-3..6`), `ChiChainDescriptor`
//!   (`SCH-3..7`, `VCH-3..7`) and `ChiPathClusterDescriptor` (`SPC-4..6`,
//!   `VPC-4..6`) — 40 values in that order.
//!
//! ## CDK semantics reproduced by [`CdkChiIndices`]
//!
//! - Hydrogen-suppressed graph as produced by CDK's `removeHydrogens`
//!   (`Hydrogens.isSafeToSuppress`): neutral, non-isotopic H with exactly one
//!   non-H neighbour are folded into the neighbour's H count; all other H atoms
//!   stay in the graph.
//! - Simple delta = number of connected bonds in that graph.
//! - Valence delta (`ChiIndexUtils.evalValenceIndex`):
//!   `(Zv - h) / (Z - Zv - 1)` with `Zv = AtomValenceTool.getValence - charge`,
//!   plus Kier & Hall's empirical values for S (S-S 0.89, S=O 1.33, O=S=O 2.67)
//!   and P (2.22 for four-connected P with `=O`/single bonds).
//! - CDK perception differences handled locally (shared perception untouched):
//!   groups that RDKit's cleanUp charge-separated (`N(=O)=O`, `N=N#N`,
//!   `Cl(=O)(=O)(=O)O`, ...) are restored to CDK's neutral form when the
//!   charged atoms are organic-subset atoms (no-implicit-H flag unset; SMILES
//!   can only charge bracket atoms), and neutral radical centres (`[N]`,
//!   `[CH2]`, ...) are filled with H as CDK's hydrogen adder does.
//! - Known gap: CDK reads the S/P special cases from Kekulé bond orders;
//!   aromatic bonds are counted as neither single nor double here (no such
//!   case occurs in the parity set).
//! - Fragments (`ChiIndexUtils.getFragments`): CDK matches each query SMILES
//!   with the bond-based `UniversalIsomorphismTester` and keeps the unique sets
//!   of target bonds per query (fragments matched by two different queries are
//!   counted twice, as in CDK). By Whitney's theorem a bond set matches a
//!   connected query iff it is isomorphic to the query graph (the K3/K1,3
//!   ambiguity only yields 3-atom fragments for the 3-star, which CDK's
//!   atom-count filter discards). We therefore enumerate atom-injective
//!   embeddings of each query and divide by the query's automorphism count.
//! - Chain order 3 uses the 3-membered rings of the SSSR.
//! - Each fragment contributes `1/sqrt(prod)` unless `prod == 0` (NaN and
//!   negative products propagate as NaN, exactly like Java's `Math.sqrt`).
//!
//! Reference: Kier, L.B. & Hall, L.H. (1986) "Molecular Connectivity in
//!            Structure-Activity Analysis", Research Studies Press.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::elements::outer_electrons;
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType};
use smallvec::SmallVec;
use std::sync::OnceLock;

/// Compute simple delta (graph degree) for each heavy atom.
/// δ_i = number of heavy-atom neighbors (i.e., graph degree).
fn simple_deltas(mol: &Molecule) -> Vec<f64> {
    (0..mol.num_atoms())
        .map(|i| mol.get_degree(i as AtomIdx) as f64)
        .collect()
}

/// Compute valence delta for each heavy atom.
/// δᵥ = (Zᵥ - Hᵢ) / (Z - Zᵥ - 1)
/// where Zᵥ = valence electrons, Hᵢ = total H count, Z = atomic number.
/// For hydrogen or denominator=0, returns 0.
fn valence_deltas(mol: &Molecule) -> Vec<f64> {
    mol.atoms
        .iter()
        .map(|atom| {
            let z = atom.atomic_number as f64;
            let zv = outer_electrons(atom.atomic_number) as f64;
            let h = atom.total_h() as f64;
            let denom = z - zv - 1.0;
            if denom <= 0.0 || atom.atomic_number <= 1 {
                // For H and 1st-row elements where Z-Zv-1=0, use simple delta
                // δᵥ(C) = (4-H)/(6-4-1) = (4-H)/1 = 4-H
                // For H itself: return 0
                if atom.atomic_number == 1 {
                    return 0.0;
                }
                // 1st row: C(6),N(7),O(8),F(9) → denom=1,1,1,1
                // But O has Z-Zv-1 = 8-6-1=1, N has 7-5-1=1, C has 6-4-1=1
                // Only H (1-1-1=-1) truly fails. For other elements where
                // denom is exactly 0 (shouldn't happen for common elements), fallback:
                zv - h
            } else {
                (zv - h) / denom
            }
        })
        .collect()
}

/// Chi-0 (zero-order): χ⁰ = Σ 1/√δᵢ for each atom with δ > 0.
fn chi0(deltas: &[f64]) -> f64 {
    deltas
        .iter()
        .filter(|&&d| d > 0.0)
        .map(|&d| 1.0 / d.sqrt())
        .sum()
}

/// Chi-1 (first-order): χ¹ = Σ 1/√(δᵢ·δⱼ) over all bonds.
fn chi1(mol: &Molecule, deltas: &[f64]) -> f64 {
    mol.bonds
        .iter()
        .map(|bond| {
            let di = deltas[bond.source as usize];
            let dj = deltas[bond.target as usize];
            let prod = di * dj;
            if prod > 0.0 { 1.0 / prod.sqrt() } else { 0.0 }
        })
        .sum()
}

/// Find all paths of exactly `length` bonds and compute chi contribution.
/// χⁿ = Σ 1/√(δ₁·δ₂·...·δₙ₊₁) over all n-bond paths.
fn chi_path(mol: &Molecule, deltas: &[f64], length: usize) -> f64 {
    if length == 0 {
        return chi0(deltas);
    }
    if length == 1 {
        return chi1(mol, deltas);
    }

    let n = mol.num_atoms();
    if n < length + 1 {
        return 0.0;
    }

    // DFS to enumerate all paths of exact length
    let mut sum = 0.0f64;
    let mut visited = vec![false; n];

    for start in 0..n {
        let d0 = deltas[start];
        if d0 <= 0.0 {
            continue;
        }
        visited[start] = true;
        dfs_paths_acc(
            mol,
            deltas,
            start as AtomIdx,
            1,
            length + 1,
            d0,
            &mut visited,
            &mut sum,
        );
        visited[start] = false;
    }

    // Each path is found twice (forward + reverse), so divide by 2
    sum / 2.0
}

/// Implements path rdkit like.
///
/// Function context: `crates/ferrum_core/src/desc/chi.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn chi_path_rdkit_like(
    mol: &Molecule,
    ctx: &mut crate::desc::context::CalculationContext,
    deltas: &[f64],
    length: usize,
) -> f64 {
    let mut sum = 0.0;
    for path in crate::algo::paths::cached_paths_of_length(mol, ctx, length).chunks_exact(length + 1) {
        let atoms = if path.first() == path.last() {
            &path[..path.len() - 1]
        } else {
            path
        };

        let mut prod = 1.0f64;
        for &idx in atoms {
            prod *= deltas[idx as usize];
        }
        if prod > 0.0 {
            sum += 1.0 / prod.sqrt();
        }
    }
    sum
}

#[allow(clippy::too_many_arguments)]
fn dfs_paths_acc(
    mol: &Molecule,
    deltas: &[f64],
    current: AtomIdx,
    depth: usize,
    target_depth: usize,
    prod: f64,
    visited: &mut [bool],
    sum: &mut f64,
) {
    if depth == target_depth {
        *sum += 1.0 / prod.sqrt();
        return;
    }

    for &bond_idx in &mol.adjacency[current as usize] {
        let bond = &mol.bonds[bond_idx as usize];
        let neighbor = if bond.source == current {
            bond.target
        } else {
            bond.source
        };

        let ni = neighbor as usize;
        if visited[ni] {
            continue;
        }
        let d = deltas[ni];
        if d <= 0.0 {
            continue;
        }
        visited[ni] = true;
        dfs_paths_acc(
            mol,
            deltas,
            neighbor,
            depth + 1,
            target_depth,
            prod * d,
            visited,
            sum,
        );
        visited[ni] = false;
    }
}

// ===========================================================================
// RDKit connectivity indices
// ===========================================================================

/// RDKit connectivity indices: `Chi0`, `Chi1`, `Chi0v`, `Chi1v`, `Chi2v`,
/// `Chi3v`, `Chi4v` (7 values).
pub struct ChiIndices;

impl Descriptor for ChiIndices {
    fn name(&self) -> &str {
        "ChiIndices"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let simple = simple_deltas(mol);
        let valence = valence_deltas(mol);
        let values = [
            chi_path(mol, &simple, 0),
            chi_path(mol, &simple, 1),
            chi_path(mol, &valence, 0),
            chi_path(mol, &valence, 1),
            chi_path(mol, &valence, 2),
            chi_path_rdkit_like(mol, ctx, &valence, 3),
            chi_path_rdkit_like(mol, ctx, &valence, 4),
        ];
        DescriptorValue::FloatVec(values.to_vec())
    }
}

// ===========================================================================
// CDK ChiPath / ChiCluster / ChiChain / ChiPathCluster
// ===========================================================================

/// Maximum number of atoms in a CDK chi query (8-atom path for `SP-7`).
const MAX_QUERY_ATOMS: usize = 8;

/// `AtomValenceTool.getValence` (CDK): valence electrons by element. `None`
/// for elements missing from CDK's table (CDK throws there).
fn cdk_valence(z: u8) -> Option<i32> {
    Some(match z {
        1 | 3 | 11 | 19 | 37 | 55 | 87 => 1,
        4 | 12 | 20 | 38 | 56 | 88 | 80 | 29 | 25 | 27 => 2,
        5 | 13 | 31 | 49 | 81 => 3,
        6 | 14 | 32 | 50 | 82 => 4,
        7 | 15 | 33 | 51 | 83 => 5,
        8 | 16 | 34 | 52 | 84 => 6,
        9 | 17 | 35 | 53 | 85 => 7,
        2 | 10 | 18 | 36 | 54 | 86 => 8,
        _ => return None,
    })
}

/// CDK's hydrogen-suppressed graph with per-atom simple and valence deltas.
struct CdkGraph {
    /// CSR offsets into `nbrs` / `nbr_bonds`.
    offs: Vec<u32>,
    nbrs: Vec<u32>,
    nbr_bonds: Vec<u32>,
    bonds: Vec<(u32, u32)>,
    /// Simple delta (connected bond count).
    simple: Vec<f64>,
    /// Valence delta (`ChiIndexUtils.evalValenceIndex`).
    valence: Vec<f64>,
    /// Bit `k` set when the atom lies on a simple cycle of exactly `k` atoms
    /// (`k <= MAX_RING_BIT`).
    ring_bits: Vec<u8>,
    /// `by_degree[k]`: atoms with degree >= k (k <= 4), candidate query roots.
    by_degree: [Vec<u32>; 5],
}

/// Largest cycle size tracked in [`CdkGraph::ring_bits`] (the chain queries
/// contain rings of 3 to 6 atoms; larger query rings are not filtered).
const MAX_RING_BIT: usize = 6;

impl CdkGraph {
    #[inline]
    fn n(&self) -> usize {
        self.offs.len() - 1
    }

    #[inline]
    fn nbrs(&self, a: usize) -> &[u32] {
        &self.nbrs[self.offs[a] as usize..self.offs[a + 1] as usize]
    }

    #[inline]
    fn degree(&self, a: usize) -> usize {
        (self.offs[a + 1] - self.offs[a]) as usize
    }

    #[inline]
    fn adjacent(&self, a: usize, b: usize) -> bool {
        self.nbrs(a).contains(&(b as u32))
    }

    /// Builds a graph from an edge list (all deltas set to `delta`).
    fn from_edges(n: usize, edges: &[(u32, u32)], delta: f64) -> Self {
        let mut adj: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        for (bi, &(u, v)) in edges.iter().enumerate() {
            adj[u as usize].push((v, bi as u32));
            adj[v as usize].push((u, bi as u32));
        }
        Self::from_adjacency(adj, edges.to_vec(), vec![delta; n], vec![delta; n])
    }

    fn from_adjacency(
        adj: Vec<Vec<(u32, u32)>>,
        bonds: Vec<(u32, u32)>,
        simple: Vec<f64>,
        valence: Vec<f64>,
    ) -> Self {
        let mut offs = Vec::with_capacity(adj.len() + 1);
        let mut nbrs = Vec::with_capacity(bonds.len() * 2);
        let mut nbr_bonds = Vec::with_capacity(bonds.len() * 2);
        offs.push(0u32);
        for list in &adj {
            for &(nb, bi) in list {
                nbrs.push(nb);
                nbr_bonds.push(bi);
            }
            offs.push(nbrs.len() as u32);
        }
        let mut g =
            Self { offs, nbrs, nbr_bonds, bonds, simple, valence, ring_bits: Vec::new(), by_degree: Default::default() };
        g.ring_bits = g.small_cycle_bits();
        for k in 0..5 {
            g.by_degree[k] = (0..g.n() as u32).filter(|&a| g.degree(a as usize) >= k).collect();
        }
        g
    }

    /// Per atom, the sizes (bits) of the simple cycles of at most
    /// `MAX_RING_BIT` atoms through it. Each cycle is found from its lowest
    /// atom by a depth-limited search over higher atoms.
    fn small_cycle_bits(&self) -> Vec<u8> {
        fn dfs(g: &CdkGraph, s: usize, path: &mut Vec<u32>, on_path: &mut [bool], bits: &mut [u8]) {
            let u = *path.last().unwrap() as usize;
            if path.len() >= 3 && g.adjacent(u, s) {
                let bit = 1u8 << path.len();
                for &a in path.iter() {
                    bits[a as usize] |= bit;
                }
            }
            if path.len() == MAX_RING_BIT {
                return;
            }
            for &v in g.nbrs(u) {
                let v = v as usize;
                if v > s && !on_path[v] {
                    on_path[v] = true;
                    path.push(v as u32);
                    dfs(g, s, path, on_path, bits);
                    path.pop();
                    on_path[v] = false;
                }
            }
        }
        let n = self.n();
        let mut bits = vec![0u8; n];
        // only atoms of the 2-core can lie on a cycle
        let mut deg: Vec<u32> = (0..n).map(|a| self.degree(a) as u32).collect();
        let mut in_core = vec![true; n];
        let mut stack: Vec<usize> = (0..n).filter(|&a| deg[a] <= 1).collect();
        while let Some(a) = stack.pop() {
            if !in_core[a] {
                continue;
            }
            in_core[a] = false;
            for &b in self.nbrs(a) {
                let b = b as usize;
                if in_core[b] {
                    deg[b] -= 1;
                    if deg[b] == 1 {
                        stack.push(b);
                    }
                }
            }
        }
        // atoms outside the core are marked visited so the search skips them
        let mut on_path: Vec<bool> = in_core.iter().map(|&c| !c).collect();
        let mut path = Vec::with_capacity(MAX_RING_BIT);
        for s in (0..n).filter(|&s| in_core[s]) {
            on_path[s] = true;
            path.push(s as u32);
            dfs(self, s, &mut path, &mut on_path, &mut bits);
            path.pop();
            on_path[s] = false;
        }
        bits
    }

    /// CDK `removeHydrogens` + atom typing + `CDKHydrogenAdder` +
    /// `ChiIndexUtils` deltas on top of the RDKit-perceived molecule.
    fn from_molecule(mol: &Molecule) -> Self {
        let n = mol.num_atoms();
        let other = |bi: usize, a: usize| {
            let b = &mol.bonds[bi];
            if b.source as usize == a { b.target as usize } else { b.source as usize }
        };

        // Hydrogens.isSafeToSuppress
        let suppress: Vec<bool> = (0..n)
            .map(|a| {
                let at = &mol.atoms[a];
                if at.atomic_number != 1 || at.formal_charge != 0 || at.isotope != 0 || at.total_h() != 0 {
                    return false;
                }
                if mol.adjacency[a].len() != 1 {
                    return false;
                }
                let nb = other(mol.adjacency[a][0] as usize, a);
                !matches!(mol.atoms[nb].atomic_number, 0 | 1)
            })
            .collect();

        let mut new_idx = vec![u32::MAX; n];
        let mut kept = Vec::with_capacity(n);
        for a in 0..n {
            if !suppress[a] {
                new_idx[a] = kept.len() as u32;
                kept.push(a);
            }
        }

        let mut adj: Vec<Vec<(u32, u32)>> = vec![Vec::new(); kept.len()];
        let mut bonds = Vec::with_capacity(mol.num_bonds());
        let mut orders = Vec::with_capacity(mol.num_bonds());
        for b in &mol.bonds {
            let (u, v) = (new_idx[b.source as usize], new_idx[b.target as usize]);
            if u == u32::MAX || v == u32::MAX {
                continue;
            }
            let bi = bonds.len() as u32;
            adj[u as usize].push((v, bi));
            adj[v as usize].push((u, bi));
            bonds.push((u, v));
            orders.push(b.bond_type);
        }

        // Formal charges and bond orders as CDK sees them. RDKit's cleanUp
        // (sanitize) rewrites neutral hypervalent groups into charge-separated
        // form: `N(=O)=O` -> `[N+](=O)[O-]`, `N=N#N` -> `N=[N+]=[N-]`,
        // `Cl(=O)(=O)(=O)O` -> `[Cl+3]([O-])([O-])([O-])O`, `P(=O)(=C)` ->
        // `[P+]([O-])(=C)`; CDK keeps the neutral form. SMILES can only charge
        // bracket atoms, so a charged atom without the no-implicit-H flag (an
        // organic-subset atom) was charged by cleanUp and is restored here.
        let mut charge: Vec<i32> = kept.iter().map(|&a| mol.atoms[a].formal_charge as i32).collect();
        let cleaned = |k: usize| !mol.atoms[kept[k]].no_implicit_h();
        for k in 0..kept.len() {
            let z = mol.atoms[kept[k]].atomic_number;
            if charge[k] <= 0 || !cleaned(k) || !matches!(z, 7 | 15 | 17 | 35 | 53) {
                continue;
            }
            for &(nb, bi) in &adj[k] {
                if charge[k] == 0 {
                    break;
                }
                let nb = nb as usize;
                let nz = mol.atoms[kept[nb]].atomic_number;
                let order = orders[bi as usize];
                if charge[nb] != -1 || !cleaned(nb) {
                    continue;
                }
                let restored = match (nz, order) {
                    (8, BondType::Single) => BondType::Double,
                    (7, BondType::Double) if z == 7 => BondType::Triple,
                    _ => continue,
                };
                orders[bi as usize] = restored;
                charge[nb] = 0;
                charge[k] -= 1;
            }
        }

        let simple: Vec<f64> = adj.iter().map(|l| l.len() as f64).collect();
        let valence: Vec<f64> = kept
            .iter()
            .enumerate()
            .map(|(k, &a)| {
                let at = &mol.atoms[a];
                let suppressed_h = mol.adjacency[a]
                    .iter()
                    .filter(|&&bi| suppress[other(bi as usize, a)])
                    .count() as i32;
                let h = cdk_hydrogen_count(
                    at.atomic_number,
                    charge[k],
                    adj[k].len(),
                    at.total_h() as i32 + suppressed_h,
                    at.num_radicals() as i32,
                );
                let nbr = |&(nb, bi): &(u32, u32)| (mol.atoms[kept[nb as usize]].atomic_number, orders[bi as usize]);
                cdk_valence_delta(at.atomic_number, charge[k], h, adj[k].iter().map(nbr))
            })
            .collect();

        Self::from_adjacency(adj, bonds, simple, valence)
    }
}

/// Hydrogen count after CDK's atom typing + `CDKHydrogenAdder` (implicit H =
/// the type's formal neighbour count minus connected bonds) where that
/// differs from RDKit's count: RDKit keeps neutral bracket atoms with missing
/// valence (`[N]`, `[O]`, `[C]`, `[CH2]`) as radicals, CDK types them as
/// ordinary closed-shell atoms and fills them up with hydrogens. Isolated Se
/// is the exception (typed `Se.2`, no hydrogens).
fn cdk_hydrogen_count(z: u8, charge: i32, degree: usize, h: i32, radicals: i32) -> i32 {
    let fills = matches!(z, 5..=8 | 14..=16 | 32..=34 | 50..=52) && !(z == 34 && degree == 0);
    if radicals > 0 && charge == 0 && fills {
        return h + radicals;
    }
    h
}

/// Valence delta of one atom (`ChiIndexUtils.evalValenceIndex`).
/// `nbrs` yields (atomic number, bond order) of each neighbour.
fn cdk_valence_delta(
    z: u8,
    charge: i32,
    h: i32,
    nbrs: impl Iterator<Item = (u8, BondType)> + Clone,
) -> f64 {
    if z == 16 {
        // deltavSulphur
        if nbrs.clone().any(|(nz, o)| nz == 16 && o == BondType::Single) {
            return 0.89;
        }
        match nbrs.clone().filter(|&(nz, o)| nz == 8 && o == BondType::Double).count() {
            1 => return 1.33,
            2 => return 2.67,
            _ => {}
        }
    }
    if z == 15 {
        // deltavPhosphorous
        let mut conditions = 0;
        if nbrs.clone().count() == 4 {
            conditions += 1;
        }
        for (nz, o) in nbrs.clone() {
            if nz == 8 && o == BondType::Double {
                conditions += 1;
            }
            if o == BondType::Single {
                conditions += 1;
            }
        }
        if conditions == 5 {
            return 2.22;
        }
    }
    let Some(valence) = cdk_valence(z) else { return f64::NAN };
    let zv = valence - charge;
    (zv - h) as f64 / (z as i32 - zv - 1) as f64
}

/// A CDK chi query graph compiled for embedding enumeration.
struct ChiQuery {
    n: usize,
    /// Minimum target degree for the atom at each matching position.
    deg: [u8; MAX_QUERY_ATOMS],
    /// Earlier position the atom at each position is grown from.
    anchor: [u8; MAX_QUERY_ATOMS],
    /// Other earlier positions that must also be bonded to it (ring closures).
    extra: [SmallVec<[u8; 2]>; MAX_QUERY_ATOMS],
    /// Symmetry breaking: a leaf placed after a twin leaf (same anchor) must
    /// map to a higher atom index than it (`u8::MAX` = no constraint).
    twin_prev: [u8; MAX_QUERY_ATOMS],
    /// Required `CdkGraph::ring_bits` (the query ring's size for ring atoms).
    ring_req: [u8; MAX_QUERY_ATOMS],
    /// The query's ring size bit (0 for acyclic queries).
    ring_bit: u8,
    /// Embeddings per unique bond set: the automorphisms that survive the
    /// twin-leaf symmetry breaking (counted by embedding the query into itself).
    aut: f64,
}

impl ChiQuery {
    /// Parses a carbon-only SMILES (`C`, branches, ring-closure digits) as used
    /// by the CDK chi descriptors.
    fn new(smiles: &str) -> Self {
        let mut n = 0usize;
        let mut edges: Vec<(u32, u32)> = Vec::new();
        let mut prev: Option<usize> = None;
        let mut stack = Vec::new();
        let mut rings: [Option<usize>; 10] = [None; 10];
        for c in smiles.chars() {
            match c {
                'C' => {
                    if let Some(p) = prev {
                        edges.push((p as u32, n as u32));
                    }
                    prev = Some(n);
                    n += 1;
                }
                '(' => stack.push(prev.expect("branch after atom")),
                ')' => prev = Some(stack.pop().expect("balanced branches")),
                d if d.is_ascii_digit() => {
                    let k = d as usize - '0' as usize;
                    let cur = prev.expect("ring closure after atom");
                    match rings[k].take() {
                        Some(o) => edges.push((o as u32, cur as u32)),
                        None => rings[k] = Some(cur),
                    }
                }
                _ => panic!("unsupported query character {c:?} in {smiles}"),
            }
        }
        assert!((2..=MAX_QUERY_ATOMS).contains(&n), "query {smiles}");
        let graph = CdkGraph::from_edges(n, &edges, 1.0);

        // Matching order: start at the highest-degree atom, then repeatedly
        // take the atom with most bonds to already placed atoms.
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut placed = vec![false; n];
        let first = (0..n).max_by_key(|&a| (graph.degree(a), std::cmp::Reverse(a))).unwrap();
        order.push(first);
        placed[first] = true;
        while order.len() < n {
            let next = (0..n)
                .filter(|&a| !placed[a])
                .filter(|&a| graph.nbrs(a).iter().any(|&b| placed[b as usize]))
                .max_by_key(|&a| {
                    let links = graph.nbrs(a).iter().filter(|&&b| placed[b as usize]).count();
                    // leaves sharing their atom with another leaf: placing them
                    // early lets the twin-leaf symmetry breaking prune sooner
                    let twin = graph.degree(a) == 1 && {
                        let hub = graph.nbrs(a)[0] as usize;
                        graph.nbrs(hub).iter().any(|&b| b as usize != a && graph.degree(b as usize) == 1)
                    };
                    (links, twin, graph.degree(a), std::cmp::Reverse(a))
                })
                .expect("connected query");
            order.push(next);
            placed[next] = true;
        }
        let mut pos_of = vec![0usize; n];
        for (p, &a) in order.iter().enumerate() {
            pos_of[a] = p;
        }

        let mut q = ChiQuery {
            n,
            deg: [0; MAX_QUERY_ATOMS],
            anchor: [0; MAX_QUERY_ATOMS],
            extra: Default::default(),
            twin_prev: [u8::MAX; MAX_QUERY_ATOMS],
            ring_req: [0; MAX_QUERY_ATOMS],
            ring_bit: 0,
            aut: 1.0,
        };
        // unicyclic queries: ring atoms must lie on a cycle of the ring's size
        if edges.len() == n {
            let core: Vec<usize> = (0..n).filter(|&a| graph.ring_bits[a] != 0).collect();
            let bit = if (3..=MAX_RING_BIT).contains(&core.len()) { 1u8 << core.len() } else { 0 };
            if bit != 0 && core.iter().all(|&a| graph.ring_bits[a] == bit) {
                q.ring_bit = bit;
                for &a in &core {
                    q.ring_req[pos_of[a]] = bit;
                }
            }
        }
        for (p, &a) in order.iter().enumerate() {
            q.deg[p] = graph.degree(a) as u8;
            let mut earlier: Vec<u8> = graph
                .nbrs(a)
                .iter()
                .map(|&b| pos_of[b as usize])
                .filter(|&bp| bp < p)
                .map(|bp| bp as u8)
                .collect();
            earlier.sort_unstable();
            if p > 0 {
                q.anchor[p] = earlier[0];
                q.extra[p] = earlier[1..].iter().copied().collect();
            }
        }
        for p in 1..n {
            if q.deg[p] != 1 {
                continue;
            }
            if let Some(prev) = (1..p).rev().find(|&r| q.deg[r] == 1 && q.anchor[r] == q.anchor[p]) {
                q.twin_prev[p] = prev as u8;
            }
        }
        q.aut = q.embed_sums(&graph, &mut vec![false; n])[0];
        q
    }

    /// Sums `1/sqrt(prod δ)` (simple, valence) over all embeddings of the
    /// query into `g`; every unique bond set is counted `aut` times.
    /// `used`: all-false scratch of the graph's size (left all-false).
    fn embed_sums(&self, g: &CdkGraph, used: &mut [bool]) -> [f64; 2] {
        let mut acc = [0.0f64; 2];
        let n = g.n();
        if n < self.n {
            return acc;
        }
        if self.ring_bit != 0 && !g.ring_bits.iter().any(|&b| b & self.ring_bit != 0) {
            return acc;
        }
        let mut map = [0u32; MAX_QUERY_ATOMS];
        for &t in &g.by_degree[(self.deg[0] as usize).min(4)] {
            let t = t as usize;
            if g.degree(t) < self.deg[0] as usize || g.ring_bits[t] & self.ring_req[0] != self.ring_req[0] {
                continue;
            }
            used[t] = true;
            map[0] = t as u32;
            self.extend(g, 1, &mut map, used, g.simple[t], g.valence[t], &mut acc);
            used[t] = false;
        }
        acc
    }

    #[allow(clippy::too_many_arguments)]
    fn extend(
        &self,
        g: &CdkGraph,
        pos: usize,
        map: &mut [u32; MAX_QUERY_ATOMS],
        used: &mut [bool],
        ps: f64,
        pv: f64,
        acc: &mut [f64; 2],
    ) {
        if pos == self.n {
            if ps != 0.0 {
                acc[0] += 1.0 / ps.sqrt();
            }
            if pv != 0.0 {
                acc[1] += 1.0 / pv.sqrt();
            }
            return;
        }
        let anchor = map[self.anchor[pos] as usize] as usize;
        let min_deg = self.deg[pos] as usize;
        let req = self.ring_req[pos];
        let above = match self.twin_prev[pos] {
            u8::MAX => 0,
            prev => map[prev as usize] as usize + 1,
        };
        'next: for &t in g.nbrs(anchor) {
            let t = t as usize;
            if used[t] || t < above || g.degree(t) < min_deg || g.ring_bits[t] & req != req {
                continue;
            }
            for &e in &self.extra[pos] {
                if !g.adjacent(map[e as usize] as usize, t) {
                    continue 'next;
                }
            }
            used[t] = true;
            map[pos] = t as u32;
            self.extend(g, pos + 1, map, used, ps * g.simple[t], pv * g.valence[t], acc);
            used[t] = false;
        }
    }
}

/// `ChiIndexUtils.getFragments` + `evalSimpleIndex` / `evalValenceIndex` for
/// a set of queries: (simple, valence).
fn cdk_fragment_sums(g: &CdkGraph, queries: &[ChiQuery], used: &mut [bool]) -> [f64; 2] {
    let mut out = [0.0f64; 2];
    for q in queries {
        let s = q.embed_sums(g, used);
        out[0] += s[0] / q.aut;
        out[1] += s[1] / q.aut;
    }
    out
}

/// Chain order 3 (`ChiChainDescriptor.order3`): the 3-membered rings of the
/// SSSR. Triangles are the smallest cycles, so the SSSR holds a maximal
/// GF(2)-independent subset of all triangles (all of them unless they are
/// linearly dependent, e.g. in K4).
fn cdk_chain3_sums(g: &CdkGraph) -> [f64; 2] {
    let mut out = [0.0f64; 2];
    let words = g.bonds.len().div_ceil(64).max(1);
    let mut basis: Vec<(usize, Vec<u64>)> = Vec::new(); // (pivot bit, vector)
    let n = g.n();
    let bond_between = |a: usize, b: usize| -> usize {
        let s = g.offs[a] as usize;
        let k = g.nbrs(a).iter().position(|&x| x as usize == b).unwrap();
        g.nbr_bonds[s + k] as usize
    };
    for i in 0..n {
        for &j in g.nbrs(i) {
            let j = j as usize;
            if j <= i {
                continue;
            }
            for &k in g.nbrs(j) {
                let k = k as usize;
                if k <= j || !g.adjacent(i, k) {
                    continue;
                }
                let mut v = vec![0u64; words];
                for b in [bond_between(i, j), bond_between(j, k), bond_between(i, k)] {
                    v[b / 64] ^= 1 << (b % 64);
                }
                for (pivot, bv) in &basis {
                    if v[pivot / 64] >> (pivot % 64) & 1 == 1 {
                        for (x, y) in v.iter_mut().zip(bv) {
                            *x ^= y;
                        }
                    }
                }
                let Some(w) = v.iter().position(|&x| x != 0) else { continue };
                let pivot = w * 64 + v[w].trailing_zeros() as usize;
                basis.push((pivot, v));
                let ps = g.simple[i] * g.simple[j] * g.simple[k];
                let pv = g.valence[i] * g.valence[j] * g.valence[k];
                if ps != 0.0 {
                    out[0] += 1.0 / ps.sqrt();
                }
                if pv != 0.0 {
                    out[1] += 1.0 / pv.sqrt();
                }
            }
        }
    }
    out
}

/// The query sets of the four CDK chi descriptors.
/// Sums over all simple paths of 2..=7 bonds (`[simple, valence]` per
/// length), each path counted once (from its lower end atom); CDK's ChiPath
/// queries `CCC`..`CCCCCCCC` enumerate exactly these subgraphs.
fn cdk_path_sums(g: &CdkGraph) -> [[f64; 2]; 6] {
    #[allow(clippy::too_many_arguments)]
    fn dfs(g: &CdkGraph, s: usize, u: usize, bonds: usize, ps: f64, pv: f64, used: &mut [bool], acc: &mut [[f64; 2]; 6]) {
        for &v in g.nbrs(u) {
            let v = v as usize;
            if used[v] {
                continue;
            }
            let (ps, pv) = (ps * g.simple[v], pv * g.valence[v]);
            let len = bonds + 1;
            if len >= 2 && v > s {
                let a = &mut acc[len - 2];
                if ps != 0.0 {
                    a[0] += 1.0 / ps.sqrt();
                }
                if pv != 0.0 {
                    a[1] += 1.0 / pv.sqrt();
                }
            }
            if len < 7 {
                used[v] = true;
                dfs(g, s, v, len, ps, pv, used, acc);
                used[v] = false;
            }
        }
    }
    let n = g.n();
    let mut acc = [[0.0f64; 2]; 6];
    let mut used = vec![false; n];
    for s in 0..n {
        used[s] = true;
        dfs(g, s, s, 0, g.simple[s], g.valence[s], &mut used, &mut acc);
        used[s] = false;
    }
    acc
}

struct CdkChiQueries {
    /// Cluster orders 3..=6.
    cluster: [Vec<ChiQuery>; 4],
    /// Chain orders 4..=7 (order 3 uses SSSR rings).
    chain: [Vec<ChiQuery>; 4],
    /// Path-cluster orders 4..=6.
    path_cluster: [Vec<ChiQuery>; 3],
}

fn cdk_chi_queries() -> &'static CdkChiQueries {
    static QUERIES: OnceLock<CdkChiQueries> = OnceLock::new();
    QUERIES.get_or_init(|| {
        let set = |smiles: &[&str]| smiles.iter().map(|s| ChiQuery::new(s)).collect::<Vec<_>>();
        CdkChiQueries {
            cluster: [
                set(&["C(C)(C)(C)"]),
                set(&["C(C)(C)(C)(C)"]),
                set(&["CC(C)C(C)(C)"]),
                set(&["CC(C)C(C)(C)C"]),
            ],
            chain: [
                set(&["C1CCC1", "CC1CC1"]),
                set(&["C1CCCC1", "CC1CCC1", "CC1CC1(C)"]),
                set(&[
                    "CC1CCCC1", "CC1CC(C)C1", "CC1(C)(CCC1)", "CCC1CCC1", "C1CCCCC1", "CC1CCC1(C)",
                    "CC1C(C)C1(C)", "CCCC1CC1", "CCC1CC1(C)",
                ]),
                set(&[
                    "C1CCCCC1C",
                    // 5-ring cases
                    "C1CCCC1(C)(C)", "C1(C)C(C)CCC1", "C1(C)CC(C)CC1", "C1CCCC1(CC)",
                    // 4-ring cases
                    "C1(C)C(C)C(C)C1", "C1CC(C)C1(CC)", "C1C(C)CC1(CC)", "C1CCC1(CCC)", "C1CCC1C(C)(C)",
                    "C1CCC1(C)(CC)", "C1CC(C)C1(C)(C)", "C1C(C)CC1(C)(C)",
                    // 3-ring cases
                    "C1(C)C(C)C1(CC)", "C1C(C)(C)C1(C)(C)", "C1CC1CCCC", "C1C(C)C1(CCC)", "C1C(CC)C1(CC)",
                    "C1C(C)C1C(C)(C)", "C1C(C)C1(C)(CC)", "C1CC1CC(C)(C)", "C1CC1C(C)CC", "C1CC1C(C)(C)(C)",
                ]),
            ],
            path_cluster: [
                set(&["CC(C)CC"]),
                set(&["CC(C)CCC", "CC(C)(C)CC", "CCC(C)CC"]),
                set(&["CC(C)(C)CCC", "CC(C)C(C)CC", "CC(C)CC(C)C", "CC(C)CCCC", "CCC(C)CCC", "CC(C)(CC)CC"]),
            ],
        }
    })
}

/// CDK ChiPath / ChiCluster / ChiChain / ChiPathCluster descriptors
/// (`SP-0..7`, `VP-0..7`, `SC-3..6`, `VC-3..6`, `SCH-3..7`, `VCH-3..7`,
/// `SPC-4..6`, `VPC-4..6`).
pub struct CdkChiIndices;

impl CdkChiIndices {
    /// The 40 values in feature order, as f64.
    fn values(mol: &Molecule) -> [f64; 40] {
        let g = CdkGraph::from_molecule(mol);
        let qs = cdk_chi_queries();
        let mut sp = [0.0f64; 8];
        let mut vp = [0.0f64; 8];

        // order 0: every atom; order 1: every bond
        for a in 0..g.n() {
            if g.simple[a] != 0.0 {
                sp[0] += 1.0 / g.simple[a].sqrt();
            }
            if g.valence[a] != 0.0 {
                vp[0] += 1.0 / g.valence[a].sqrt();
            }
        }
        for &(u, v) in &g.bonds {
            let (u, v) = (u as usize, v as usize);
            let ps = g.simple[u] * g.simple[v];
            let pv = g.valence[u] * g.valence[v];
            if ps != 0.0 {
                sp[1] += 1.0 / ps.sqrt();
            }
            if pv != 0.0 {
                vp[1] += 1.0 / pv.sqrt();
            }
        }
        for (i, s) in cdk_path_sums(&g).iter().enumerate() {
            sp[i + 2] = s[0];
            vp[i + 2] = s[1];
        }

        let mut used = vec![false; g.n()];
        let cluster: Vec<[f64; 2]> = qs.cluster.iter().map(|set| cdk_fragment_sums(&g, set, &mut used)).collect();
        let mut chain = vec![cdk_chain3_sums(&g)];
        chain.extend(qs.chain.iter().map(|set| cdk_fragment_sums(&g, set, &mut used)));
        let path_cluster: Vec<[f64; 2]> =
            qs.path_cluster.iter().map(|set| cdk_fragment_sums(&g, set, &mut used)).collect();

        let mut out = [0.0f64; 40];
        let mut k = 0;
        let mut push = |v: f64| {
            out[k] = v;
            k += 1;
        };
        sp.iter().for_each(|&v| push(v));
        vp.iter().for_each(|&v| push(v));
        for which in 0..2 {
            cluster.iter().for_each(|s| push(s[which]));
        }
        for which in 0..2 {
            chain.iter().for_each(|s| push(s[which]));
        }
        for which in 0..2 {
            path_cluster.iter().for_each(|s| push(s[which]));
        }
        debug_assert_eq!(k, 40);
        out
    }
}

impl Descriptor for CdkChiIndices {
    fn name(&self) -> &str {
        "CdkChiIndices"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::FloatVec(Self::values(mol).to_vec())
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ChiIndices>;
    let _ = assert_send_sync::<CdkChiIndices>;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc(desc: &dyn Descriptor, smiles: &str) -> Vec<f64> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    fn assert_close(got: &[f64], expected: &[f64], what: &str) {
        assert_eq!(got.len(), expected.len(), "{what}: length");
        for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
            let ok = if e.is_nan() { g.is_nan() } else { (g as f64 - e).abs() <= 1e-5 + 1e-6 * e.abs() };
            assert!(ok, "{what}[{i}]: got {g}, expected {e}");
        }
    }

    // RDKit layout: Chi0, Chi1, Chi0v, Chi1v, Chi2v, Chi3v, Chi4v
    // (expected values from RDKit 2025.03).
    #[test]
    /// Chi0/Chi1 (simple), Chi0v..Chi2v (paths) and Chi3v/Chi4v (RDKit counts
    /// 3-/4-membered ring closures as paths) against RDKit 2025.03 (7 decimals).
    #[allow(clippy::approx_constant)] // RDKit reference values, 7 decimals
    fn rdkit_chi_indices() {
        let cases: [(&str, [f64; 7]); 10] = [
            ("CC", [2.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0]),
            ("CCC", [2.7071068, 1.4142136, 2.7071068, 1.4142136, 0.7071068, 0.0, 0.0]),
            ("CCCC", [3.4142136, 1.9142136, 3.4142136, 1.9142136, 1.0, 0.5, 0.0]),
            ("CC(C)C", [3.5773503, 1.7320508, 3.5773503, 1.7320508, 1.7320508, 0.0, 0.0]),
            ("c1ccccc1", [4.2426407, 3.0, 3.4641016, 2.0, 1.1547005, 0.6666667, 0.3849002]),
            ("CCO", [2.7071068, 1.4142136, 2.1543204, 1.0233345, 0.3162278, 0.0, 0.0]),
            ("C1CC1", [2.1213203, 1.5, 2.1213203, 1.5, 1.0606602, 0.3535534, 0.0]),
            ("C1CCC1", [2.8284271, 2.0, 2.8284271, 2.0, 1.4142136, 1.0, 0.25]),
            ("CC1(C)C2CC=CC1C2", [6.4831277, 4.2103434, 6.2236146, 3.8770101, 4.0243406, 3.3352995, 2.1899135]),
            ("CS(=O)(=O)c1ccc(Cl)cc1", [8.4831277, 4.9990186, 7.4845359, 5.2254312, 4.678257, 2.6222367, 1.4382492]),
        ];
        for (smiles, expected) in cases {
            let got = calc(&ChiIndices, smiles);
            assert_eq!(got.len(), 7);
            for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
                assert!((g as f64 - e).abs() < 1e-6, "{smiles}[{i}]: got {g}, expected {e}");
            }
        }
    }

    #[test]
    fn rdkit_chi_branching_and_heteroatoms() {
        let lin = calc(&ChiIndices, "CCCC");
        let bra = calc(&ChiIndices, "CC(C)C");
        assert!(lin[1] > bra[1], "linear Chi1 must exceed branched Chi1");
        let c = calc(&ChiIndices, "CCO");
        assert!((c[1] - c[3]).abs() > 1e-3, "Chi1v must differ from Chi1 for ethanol");
    }


    // CDK layout (expected values from CDK 2.13, see
    // tests/data/cdk_parity_reference.csv).
    #[test]
    fn cdk_layout_and_simple_alkanes() {
        let v = calc(&CdkChiIndices, "CCCCC");
        assert_eq!(v.len(), 40);
        // pentane: one 4-bond path (1*2*2*2*1), no clusters/chains/path clusters
        assert!((v[4] as f64 - 1.0 / 8f64.sqrt()).abs() < 1e-6, "SP-4 = {}", v[4]);
        assert!(v[16..40].iter().all(|&x| x == 0.0), "{v:?}");
        // neopentane: one 4-star (SC-4) and four 3-stars (SC-3)
        let v = calc(&CdkChiIndices, "CC(C)(C)C");
        assert!((v[17] - 0.5).abs() < 1e-6, "SC-4 = {}", v[17]);
        assert!((v[16] - 2.0).abs() < 1e-6, "SC-3 = {}", v[16]);
        // cyclopropane: one 3-chain; the 3-star query must not match the ring
        let v = calc(&CdkChiIndices, "C1CC1");
        assert!((v[24] as f64 - 1.0 / 8f64.sqrt()).abs() < 1e-6, "SCH-3 = {}", v[24]);
        assert_eq!(v[16], 0.0, "SC-3 must ignore triangles");
    }

    #[test]
    fn cdk_query_automorphisms() {
        // automorphisms left after twin-leaf symmetry breaking (k twin leaves
        // on one atom are enumerated in a single order instead of k!)
        let aut = |s: &str| ChiQuery::new(s).aut;
        assert_eq!(aut("CCC"), 1.0); // 2 / 2!
        assert_eq!(aut("CCCC"), 2.0);
        assert_eq!(aut("C(C)(C)(C)"), 1.0); // 6 / 3!
        assert_eq!(aut("C(C)(C)(C)(C)"), 1.0); // 24 / 4!
        assert_eq!(aut("CC(C)CC"), 1.0); // 2 / 2!
        assert_eq!(aut("C1CCC1"), 8.0);
        assert_eq!(aut("C1CCCCC1"), 12.0);
        assert_eq!(aut("CC1CC1"), 2.0);
        assert_eq!(aut("C1C(C)(C)C1(C)(C)"), 2.0); // 8 / (2! 2!)
    }

    #[test]
    /// Cubane: 4- to 7-chains and path clusters.
    fn cdk_cubane_chains() {
        let expected = [
            4.618802153517007, 4.0, 4.618802153517005, 5.3333333333333295,
            4.618802153517002, 4.444444444444453, 2.052800957118669, 0.8888888888888906,
            4.618802153517007, 4.0, 4.618802153517005, 5.3333333333333295,
            4.618802153517002, 4.444444444444453, 2.052800957118669, 0.8888888888888906,
            0.8888888888888891, 0.0, 0.4444444444444444, 0.0,
            0.8888888888888891, 0.0, 0.4444444444444444, 0.0,
            0.0, 0.6666666666666667, 1.5396007178390023, 3.703703703703709,
            5.6452026320763045, 0.0, 0.6666666666666667, 1.5396007178390023,
            3.703703703703709, 5.6452026320763045, 3.079201435678001, 4.444444444444453,
            6.158402871355967, 3.079201435678001, 4.444444444444453, 6.158402871355967,
        ];
        assert_close(&calc(&CdkChiIndices, "C12C3C4C1C5C2C3C45"), &expected, "C12C3C4C1C5C2C3C45");
    }

    #[test]
    /// Fused 4-ring with gem-dimethyl: clusters, chains, path clusters.
    fn cdk_bicyclic_terpene() {
        let expected = [
            11.922285251880869, 7.415520785076279, 7.561510596411609, 5.929354719193864,
            4.1871392151882425, 3.4050185530236416, 2.119577582209704, 1.3980450979086563,
            10.62901187137467, 6.450069451222892, 6.327241493285348, 4.704088115587196,
            3.3405865110599695, 2.4156936919584258, 1.4642290710679133, 0.9339213364087006,
            2.0115174905242097, 0.20412414523193154, 0.5955220308284978, 0.08333333333333333,
            1.666141303559806, 0.20412414523193154, 0.430753575088734, 0.08333333333333333,
            0.0, 0.11785113019775793, 0.3870769754728264, 0.6329795093937626,
            0.8440671177570169, 0.0, 0.11785113019775793, 0.3779611588277282,
            0.5754814494946513, 0.6759641700729735, 3.9149653008725687, 4.814438719876381,
            6.96830738968948, 2.970117035396212, 3.6779986171969683, 4.618040829535253,
        ];
        assert_close(&calc(&CdkChiIndices, "C=C1CCCC(=C)C2CC(C)(C)C2CC1O"), &expected, "C=C1CCCC(=C)C2CC(C)(C)C2CC1O");
    }

    #[test]
    /// CDK keeps N(=O)=O neutral (RDKit cleanUp charge-separates it).
    fn cdk_neutral_nitro() {
        let expected = [
            6.690234444311989, 4.304530452640312, 3.6421070322115843, 2.5926389558431224,
            1.8008334588458519, 1.250443623153493, 0.4511844635310912, 0.23570226039551587,
            4.650461522375813, 2.4994387719430486, 1.5926589440218353, 0.967012110859851,
            0.5497102281601602, 0.3124133261225012, 0.06926080682414132, 0.023424278964210215,
            0.5, 0.0, 0.16666666666666666, 0.0,
            0.1118033988749895, 0.0, 0.012422599874998832, 0.0,
            0.0, 0.0, 0.0, 0.10206207261596577,
            0.05892556509887897, 0.0, 0.0, 0.0,
            0.032075014954979206, 0.014344382763731175, 1.040440114519881, 1.054737854124365,
            1.040440114519881, 0.18995750681859905, 0.1696536358926213, 0.12663833787906603,
        ];
        assert_close(&calc(&CdkChiIndices, "c1ccccc1N(=O)=O"), &expected, "c1ccccc1N(=O)=O");
    }

    #[test]
    /// CDK keeps Cl(=O)(=O)=O neutral.
    fn cdk_perchloric_acid() {
        let expected = [
            4.5, 2.0, 3.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            2.8058518859192287, 1.895822702495937, 1.1880057129219597, 0.0,
            0.0, 0.0, 0.0, 0.0,
            2.0, 0.5, 0.0, 0.0,
            0.33069795139960095, 0.03450327796711771, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_close(&calc(&CdkChiIndices, "OCl(=O)(=O)=O"), &expected, "OCl(=O)(=O)=O");
    }

    #[test]
    /// CDK fills the [N] radical with one H.
    fn cdk_radical_nitrogen() {
        let expected = [
            3.5355339059327373, 2.5, 1.7677669529663687, 1.25,
            0.8838834764831843, 0.0, 0.0, 0.0,
            2.809401076758503, 1.5773502691896257, 0.8849001794597504, 0.4960112905708617,
            0.2777777777777778, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.17677669529663687, 0.0,
            0.0, 0.0, 0.0, 0.05555555555555555,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_close(&calc(&CdkChiIndices, "[N]1C=CC=C1"), &expected, "[N]1C=CC=C1");
    }

    #[test]
    /// Empirical S delta for O=S=O (2.67).
    fn cdk_sulfone_deltas() {
        let expected = [
            10.983127663125444, 6.210343431045079, 7.292503577864646, 4.123697052034566,
            2.7629282173543084, 2.2016280866845843, 1.433566316292564, 0.6767766952966366,
            8.300267780365846, 4.7386544161867805, 4.133045331375583, 2.2631937765683543,
            1.2943532747198359, 0.9020138006648145, 0.4993703132996029, 0.19595012197521397,
            3.0207259421636903, 0.5773502691896258, 0.8660254037844387, 0.2886751345948129,
            1.0567888709231998, 0.1088267970375785, 0.21171286324244715, 0.036275599012526166,
            0.0, 0.0, 0.0, 0.08333333333333333,
            0.08333333333333333, 0.0, 0.0, 0.0,
            0.027777777777777776, 0.03399944785345025, 3.686305958466216, 3.9653743626207643,
            4.810963430442338, 1.337386948916672, 1.1828577701912601, 1.3002556235922536,
        ];
        assert_close(&calc(&CdkChiIndices, "CS(=O)(=O)c1cccc(S(=O)(=O)Cl)c1"), &expected, "CS(=O)(=O)c1cccc(S(=O)(=O)Cl)c1");
    }

    #[test]
    /// Empirical P delta (2.22).
    fn cdk_phosphonate_delta() {
        let expected = [
            4.5, 2.0, 3.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            2.973831536677803, 1.545454592577284, 1.2536012891040005, 0.0,
            0.0, 0.0, 0.0, 0.0,
            2.0, 0.5, 0.0, 0.0,
            0.4341024141758598, 0.0547996624351191, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_close(&calc(&CdkChiIndices, "CP(=O)(O)O"), &expected, "CP(=O)(O)O");
    }

    #[test]
    /// Isotopic H stays explicit: delta_v(D) = -1, so CDK reports NaN.
    fn cdk_deuterium_nan() {
        let expected = [
            4.5, 2.0, 3.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            f64::NAN, f64::NAN, f64::NAN, 0.0,
            0.0, 0.0, 0.0, 0.0,
            2.0, 0.5, 0.0, 0.0,
            f64::NAN, f64::NAN, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_close(&calc(&CdkChiIndices, "[2H]C([2H])([2H])Cl"), &expected, "[2H]C([2H])([2H])Cl");
    }
}
