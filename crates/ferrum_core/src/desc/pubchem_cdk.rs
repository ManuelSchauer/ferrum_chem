//! CDK perception used by the PubChem fingerprint (private to `pubchem`).
//!
//! CDK's `PubchemFingerprinter` works on a molecule with explicit hydrogens and
//! (re-)perceives aromaticity itself (`SmartsPattern.prepare` → CDK's Daylight
//! model applied to all cycles). Its ring bits use the ESSSR ring set
//! (`Cycles.tripletShort`). FerrumChem's shared perception is RDKit's, so this
//! module rebuilds the CDK view of a sanitized [`Molecule`] locally:
//!
//! 1. a Kekulé structure for RDKit-aromatic bonds (CDK keeps Kekulé bond orders
//!    and only flags aromaticity),
//! 2. CDK's Daylight aromaticity (`AromaticTypeMatcher` + `Aromaticity.apply`),
//! 3. the ESSSR ring set (`RingSearch`, `InitialCycles`, `MinimumCycleBasis`,
//!    `TripletShortCycles`).
//!
//! Sources (CDK 2.13): `aromaticity/{Aromaticity,AromaticTypeMatcher,AromaticTypeModel}.java`,
//! `graph/{Cycles,InitialCycles,MinimumCycleBasis,GreedyBasis,BitMatrix,ShortestPaths,TripletShortCycles}.java`,
//! `ringsearch/RegularCyclicVertexSearch.java`.

use crate::mol::molecule::Molecule;
use crate::mol::rdkit_data::ELEMENTS;
use crate::types::BondType;
use smallvec::SmallVec;

/// Bond order codes (CDK `IBond.Order`): 0 = unset/other.
pub(super) const ORD_SINGLE: u8 = 1;
pub(super) const ORD_DOUBLE: u8 = 2;
pub(super) const ORD_TRIPLE: u8 = 3;

/// The CDK view of a molecule (heavy atoms + any hydrogens kept as graph atoms;
/// the remaining hydrogens are implicit here but treated as explicit H atoms
/// where CDK's result depends on it).
pub(super) struct CdkMol {
    pub z: Vec<u8>,
    pub charge: Vec<i8>,
    /// Hydrogens CDK holds as explicit H atoms not present in the graph.
    pub h_fields: Vec<u8>,
    /// Total hydrogen count (CDK `TOTAL_H_COUNT`: H atoms + graph H neighbours).
    pub total_h: Vec<u8>,
    pub atom_ring: Vec<bool>,
    pub atom_arom: Vec<bool>,
    /// Adjacency (CSR, CDK bond order): neighbour atom and bond index.
    pub adj_start: Vec<u32>,
    pub adj_nbr: Vec<u32>,
    pub adj_bond: Vec<u32>,
    pub bond_a: Vec<u32>,
    pub bond_b: Vec<u32>,
    /// Kekulé bond order (`ORD_*`, 0 = other).
    pub bond_order: Vec<u8>,
    pub bond_ring: Vec<bool>,
    pub bond_arom: Vec<bool>,
    /// Ring system (biconnected component) of each ring bond, `u32::MAX` for chain bonds.
    bond_comp: Vec<u32>,
    n_comp: usize,
}

impl CdkMol {
    #[inline]
    pub fn n(&self) -> usize {
        self.z.len()
    }

    #[inline]
    pub fn nbrs(&self, a: usize) -> std::ops::Range<usize> {
        self.adj_start[a] as usize..self.adj_start[a + 1] as usize
    }

    pub fn build(mol: &Molecule) -> Self {
        let n = mol.atoms.len();
        let m = mol.bonds.len();
        let mut c = CdkMol {
            z: Vec::with_capacity(n),
            charge: Vec::with_capacity(n),
            h_fields: Vec::with_capacity(n),
            total_h: Vec::with_capacity(n),
            atom_ring: vec![false; n],
            atom_arom: vec![false; n],
            adj_start: Vec::with_capacity(n + 1),
            adj_nbr: Vec::with_capacity(2 * m),
            adj_bond: Vec::with_capacity(2 * m),
            bond_a: Vec::with_capacity(m),
            bond_b: Vec::with_capacity(m),
            bond_order: Vec::with_capacity(m),
            bond_ring: vec![false; m],
            bond_arom: vec![false; m],
            bond_comp: vec![u32::MAX; m],
            n_comp: 0,
        };
        for (i, a) in mol.atoms.iter().enumerate() {
            let h = cdk_hydrogens(mol, i);
            c.z.push(a.atomic_number);
            c.charge.push(a.formal_charge);
            c.h_fields.push(h);
            c.total_h.push(h);
        }
        for b in &mol.bonds {
            c.bond_a.push(b.source as u32);
            c.bond_b.push(b.target as u32);
            c.bond_order.push(match b.bond_type {
                BondType::Single | BondType::Dative => ORD_SINGLE,
                BondType::Double => ORD_DOUBLE,
                BondType::Triple => ORD_TRIPLE,
                BondType::Aromatic => ORD_SINGLE, // fixed by kekulize()
                BondType::Unknown => 0,
            });
        }
        // RDKit's cleanUp rewrites neutral hypervalent groups into charge-separated
        // form (`N(=O)=O` -> `[N+](=O)[O-]`, `N=N#N` -> `N=[N+]=[N-]`, ...); CDK
        // keeps the input form. SMILES can only charge bracket atoms, so a charged
        // atom without the no-implicit-H flag was charged by cleanUp: restore it
        // (same rule as the CDK chi descriptors).
        let cleaned = |a: usize| !mol.atoms[a].no_implicit_h();
        for a in 0..n {
            if c.charge[a] <= 0 || !cleaned(a) || !matches!(c.z[a], 7 | 15 | 17 | 35 | 53) {
                continue;
            }
            for &bi in &mol.adjacency[a] {
                if c.charge[a] == 0 {
                    break;
                }
                let bi = bi as usize;
                let nb = if c.bond_a[bi] as usize == a { c.bond_b[bi] } else { c.bond_a[bi] } as usize;
                if c.charge[nb] != -1 || !cleaned(nb) {
                    continue;
                }
                let restored = match (c.z[nb], c.bond_order[bi]) {
                    (8, ORD_SINGLE) if mol.bonds[bi].bond_type == BondType::Single => ORD_DOUBLE,
                    (7, ORD_DOUBLE) if c.z[a] == 7 => ORD_TRIPLE,
                    _ => continue,
                };
                c.bond_order[bi] = restored;
                c.charge[nb] = 0;
                c.charge[a] -= 1;
            }
        }
        // Bond order as CDK sees it. CDK's SMILES parser (Beam) adds a chain bond
        // when the child atom is read and a ring-closure bond when the ring
        // closes; FerrumChem keeps chain bonds first (in parse order) followed by
        // ring closures (in closing order), so Beam's edge chronology is
        // recoverable. BeamToCDK then creates bonds by higher endpoint, in that
        // atom's edge order (`Graph.edges()`), and `GraphUtil.toAdjList` builds
        // the adjacency from that bond list.
        let mut beam_time = vec![0u32; m];
        let mut last_child = None;
        let mut closures = false;
        for (b, time) in beam_time.iter_mut().enumerate() {
            let hi = c.bond_a[b].max(c.bond_b[b]);
            closures |= last_child.is_some_and(|l| hi <= l);
            if closures {
                *time = 2 * hi + 1;
            } else {
                *time = 2 * hi;
                last_child = Some(hi);
            }
        }
        let mut cdk_bonds: Vec<u32> = (0..m as u32).collect();
        cdk_bonds.sort_by_key(|&b| {
            let b = b as usize;
            (c.bond_a[b].max(c.bond_b[b]), beam_time[b], b)
        });
        let mut deg = vec![0u32; n + 1];
        for b in 0..m {
            deg[c.bond_a[b] as usize] += 1;
            deg[c.bond_b[b] as usize] += 1;
        }
        let mut acc = 0u32;
        for d in deg.iter().take(n) {
            c.adj_start.push(acc);
            acc += d;
        }
        c.adj_start.push(acc);
        c.adj_nbr.resize(acc as usize, 0);
        c.adj_bond.resize(acc as usize, 0);
        let mut fill: Vec<u32> = c.adj_start[..n].to_vec();
        for &b in &cdk_bonds {
            let b = b as usize;
            let (u, v) = (c.bond_a[b] as usize, c.bond_b[b] as usize);
            c.adj_nbr[fill[u] as usize] = v as u32;
            c.adj_bond[fill[u] as usize] = b as u32;
            fill[u] += 1;
            c.adj_nbr[fill[v] as usize] = u as u32;
            c.adj_bond[fill[v] as usize] = b as u32;
            fill[v] += 1;
        }
        // CDK marks ring atoms/bonds on its own graph (Cycles.markRingAtomsAndBonds),
        // where bonds RDKit makes dative (e.g. to metals) are ordinary bonds.
        let mut comp = std::mem::take(&mut c.bond_comp);
        c.n_comp = ring_components(&c, &mut comp);
        c.bond_comp = comp;
        for b in 0..m {
            if c.bond_comp[b] != u32::MAX {
                c.bond_ring[b] = true;
                c.atom_ring[c.bond_a[b] as usize] = true;
                c.atom_ring[c.bond_b[b] as usize] = true;
            }
        }
        for a in 0..n {
            let hn = c
                .nbrs(a)
                .filter(|&k| c.z[c.adj_nbr[k] as usize] == 1)
                .count() as u8;
            c.total_h[a] = c.total_h[a].saturating_add(hn);
        }
        let arom_bonds: Vec<bool> = mol
            .bonds
            .iter()
            .map(|b| b.bond_type == BondType::Aromatic || b.is_aromatic())
            .collect();
        c.kekulize(mol, &arom_bonds, &beam_time);
        c.apply_daylight_aromaticity();
        c
    }

    // -----------------------------------------------------------------------
    // Kekulé structure for RDKit-aromatic bonds
    // -----------------------------------------------------------------------

    /// Assigns single/double orders to RDKit-aromatic bonds the way CDK's SMILES
    /// parser (Beam `Localise`) kekulises aromatic SMILES: every aromatic atom
    /// whose valence is one short needs a double bond; the matching is Beam's
    /// (greedy initial matching, single augmentation, then Edmonds' maximum
    /// matching, all following Beam's edge order). The choice matters only for
    /// bonds that RDKit but not CDK considers aromatic.
    fn kekulize(&mut self, mol: &Molecule, arom_bonds: &[bool], beam_time: &[u32]) {
        if !arom_bonds.iter().any(|&x| x) {
            return;
        }
        let n = self.n();
        let mut subset = vec![false; n];
        for (a, in_subset) in subset.iter_mut().enumerate() {
            let mut n_arom = 0;
            let mut sum = self.h_fields[a] as i32;
            for k in self.nbrs(a) {
                let b = self.adj_bond[k] as usize;
                if arom_bonds[b] {
                    n_arom += 1;
                    sum += 1;
                } else {
                    sum += match mol.bonds[b].bond_type {
                        BondType::Dative => 0,
                        _ => self.bond_order[b] as i32,
                    };
                }
            }
            if n_arom == 0 {
                continue;
            }
            let zz = (self.z[a] as i32 - self.charge[a] as i32).clamp(0, 118) as usize;
            let vals = ELEMENTS[zz].valences;
            if vals.contains(&-1) {
                continue;
            }
            // one short of the smallest allowed valence: needs a double bond
            *in_subset = vals
                .iter()
                .find(|&&v| v as i32 >= sum)
                .is_some_and(|&t| t as i32 - sum >= 1);
        }
        // Beam adjacency (edge chronology), restricted to aromatic bonds
        let mut adj: Vec<SmallVec<[(u32, u32); 4]>> = vec![SmallVec::new(); n];
        for a in 0..n {
            if !subset[a] {
                continue;
            }
            for k in self.nbrs(a) {
                let b = self.adj_bond[k];
                if arom_bonds[b as usize] {
                    adj[a].push((self.adj_nbr[k], b));
                }
            }
            adj[a].sort_by_key(|&(_, b)| (beam_time[b as usize], b));
        }
        let m = beam::kekule_matching(&adj, &subset);
        for (b, &arom) in arom_bonds.iter().enumerate() {
            if arom {
                let (u, v) = (self.bond_a[b] as usize, self.bond_b[b] as usize);
                self.bond_order[b] = if m.unmatched(u) || m.other(u) != v {
                    ORD_SINGLE
                } else {
                    ORD_DOUBLE
                };
            }
        }
    }

    // -----------------------------------------------------------------------
    // CDK Daylight aromaticity (AromaticTypeModel.DAYLIGHT + Aromaticity.apply)
    // -----------------------------------------------------------------------

    /// Number of bonds CDK sees on the explicit-H molecule.
    #[inline]
    fn bond_count(&self, a: usize) -> usize {
        (self.adj_start[a + 1] - self.adj_start[a]) as usize + self.h_fields[a] as usize
    }

    /// CDK `binfo` (implicit H = 0 on the explicit-H molecule).
    fn binfo(&self, a: usize) -> u32 {
        if self.bond_count(a) > 15 {
            return 0x1000;
        }
        let mut info = self.h_fields[a] as u32; // explicit H atoms: single bonds
        for k in self.nbrs(a) {
            let b = self.adj_bond[k] as usize;
            info += match self.bond_order[b] {
                ORD_SINGLE => 0x0001,
                ORD_DOUBLE => {
                    if self.bond_ring[b] {
                        0x0010
                    } else {
                        0x0100
                    }
                }
                _ => 0x1000,
            };
        }
        info
    }

    fn has_exo_het(&self, a: usize) -> bool {
        for k in self.nbrs(a) {
            let b = self.adj_bond[k] as usize;
            if self.bond_ring[b] || self.bond_order[b] != ORD_DOUBLE {
                continue;
            }
            return self.z[self.adj_nbr[k] as usize] != 6;
        }
        false
    }

    fn has_oxide(&self, a: usize) -> bool {
        for k in self.nbrs(a) {
            let b = self.adj_bond[k] as usize;
            if self.bond_ring[b] {
                continue;
            }
            let w = self.adj_nbr[k] as usize;
            if self.z[w] != 8 {
                return false;
            }
            if self.bond_order[b] == ORD_DOUBLE {
                return true;
            }
            return self.bond_order[b] == ORD_SINGLE && self.charge[a] == 1 && self.charge[w] == -1;
        }
        // only explicit H atoms remain: not oxygen
        false
    }

    /// π-electron contribution of the Daylight model (-1 = cannot be aromatic).
    fn daylight_contrib(&self, a: usize) -> i32 {
        if !self.atom_ring[a] {
            return -1;
        }
        let z = self.z[a];
        if self.bond_count(a) > 3 && z != 15 {
            return -1;
        }
        let q = self.charge[a];
        let bi = self.binfo(a);
        match z {
            6 => match q {
                -1 => match bi {
                    0x0011 => 1, // C2_MINUS
                    0x0003 => 2, // C3_MINUS
                    _ => -1,
                },
                0 => match bi {
                    0x0012 => 1, // C3
                    0x0102 => {
                        if self.has_exo_het(a) {
                            0 // C3_ENEG_EXO
                        } else {
                            1 // C3_EXO
                        }
                    }
                    _ => -1,
                },
                1 => match bi {
                    0x0011 => 1, // C2_PLUS
                    0x0003 => 0, // C3_PLUS
                    _ => -1,
                },
                _ => -1,
            },
            7 => {
                if q == -1 && bi == 0x0002 {
                    2 // N2_MINUS
                } else if q == 0 {
                    match bi {
                        0x0003 => 2,                      // N3
                        0x0011 => 1,                      // N2
                        0x0111 if self.has_oxide(a) => 1, // N3_OXIDE
                        _ => -1,
                    }
                } else if q == 1 && bi == 0x0012 {
                    1 // N3_OXIDE_PLUS / N3_PLUS
                } else {
                    -1
                }
            }
            15 => {
                if q == -1 && bi == 0x0002 {
                    2 // P2_MINUS
                } else if q == 0 {
                    match bi {
                        0x0013 => -1,                     // P4 (not in Daylight)
                        0x0003 => 2,                      // P3
                        0x0011 => 1,                      // P2
                        0x0111 if self.has_oxide(a) => 1, // P3_OXIDE
                        _ => -1,
                    }
                } else if q == 1 && bi == 0x0012 {
                    1 // P3_OXIDE_PLUS / P3_PLUS
                } else {
                    -1
                }
            }
            33 => {
                // As3 only (As2/As3_PLUS are not part of the Daylight model)
                if q == 0 && bi == 0x0003 { 2 } else { -1 }
            }
            8 => {
                if q == 0 && bi == 0x0002 {
                    2 // O2
                } else if q == 1 && bi == 0x0011 {
                    1 // O2_PLUS
                } else {
                    -1
                }
            }
            16 => {
                if q == 0 {
                    match bi {
                        0x0002 => 2,                      // S2
                        0x0102 if self.has_oxide(a) => 2, // S3_OXIDE
                        _ => -1,                          // S3, S2_CUML: not Daylight
                    }
                } else if q == 1 && bi == 0x0011 {
                    1 // S2_PLUS
                } else {
                    -1
                }
            }
            34 => {
                if q == 0 {
                    match bi {
                        0x0002 => 2,                      // Se2
                        0x0102 if self.has_oxide(a) => 2, // Se3_OXIDE
                        _ => -1,
                    }
                } else if q == 1 && bi == 0x0011 {
                    1 // Se2_PLUS
                } else {
                    -1
                }
            }
            _ => -1,
        }
    }

    #[inline]
    fn make_aromatic(&mut self, b: usize) {
        self.bond_arom[b] = true;
        self.atom_arom[self.bond_a[b] as usize] = true;
        self.atom_arom[self.bond_b[b] as usize] = true;
    }

    /// `Aromaticity.apply(Model.Daylight, mol)` (backtracking variant over all cycles).
    fn apply_daylight_aromaticity(&mut self) {
        let n = self.n();
        let mut contrib: Vec<i32> = (0..n).map(|a| self.daylight_contrib(a)).collect();
        if contrib.iter().all(|&c| c < 0) {
            return;
        }
        // prune terminal atoms
        for a in 0..n {
            if contrib[a] == -1 {
                continue;
            }
            let mut atom = Some(a);
            while let Some(x) = atom {
                let mut next = None;
                let mut d2 = false;
                for k in self.nbrs(x) {
                    if !self.bond_ring[self.adj_bond[k] as usize] {
                        continue;
                    }
                    let w = self.adj_nbr[k] as usize;
                    if contrib[w] == -1 {
                        continue;
                    }
                    if next.is_some() {
                        d2 = true;
                        break;
                    }
                    next = Some(w);
                }
                if d2 {
                    break;
                }
                contrib[x] = -1;
                atom = next;
            }
        }
        let mut visit = vec![0u8; n];
        for a in 0..n {
            if contrib[a] < 0 {
                visit[a] = 1;
            }
        }
        // fusion / bridge atoms: flood fill their ring systems with 2
        let mut d3 = Vec::new();
        for a in 0..n {
            if self.bond_count(a) < 3 || contrib[a] < 0 {
                continue;
            }
            let cnt = self
                .nbrs(a)
                .filter(|&k| {
                    self.bond_ring[self.adj_bond[k] as usize]
                        && contrib[self.adj_nbr[k] as usize] != -1
                })
                .count();
            if cnt >= 3 {
                d3.push(a);
                // flood fill (iterative)
                let mut stack = vec![a];
                visit[a] = 2;
                while let Some(x) = stack.pop() {
                    for k in self.nbrs(x) {
                        if !self.bond_ring[self.adj_bond[k] as usize] {
                            continue;
                        }
                        let w = self.adj_nbr[k] as usize;
                        if visit[w] == 0 {
                            visit[w] = 2;
                            stack.push(w);
                        }
                    }
                }
            }
        }
        // simple rings
        let mut ring: Vec<usize> = Vec::new();
        for start in 0..n {
            if visit[start] != 0 {
                continue;
            }
            ring.clear();
            let mut sum = 0;
            let mut prev = usize::MAX;
            let mut atom = Some(start);
            while let Some(x) = atom {
                let mut next = None;
                visit[x] = 1;
                sum += contrib[x];
                for k in self.nbrs(x) {
                    let b = self.adj_bond[k] as usize;
                    if b == prev || !self.bond_ring[b] {
                        continue;
                    }
                    let w = self.adj_nbr[k] as usize;
                    if visit[w] != 0 {
                        continue;
                    }
                    ring.push(b);
                    prev = b;
                    next = Some(w);
                }
                atom = next;
            }
            if (sum - 2) % 4 != 0 {
                continue;
            }
            for &b in &ring {
                self.make_aromatic(b);
            }
        }
        if d3.is_empty() {
            return;
        }
        for v in visit.iter_mut() {
            if *v == 2 {
                *v = 0;
            }
        }
        // complex ring systems: iterative deepening DFS from the d3 atoms
        let max_ring = n.max(3) + self.h_fields.iter().map(|&h| h as usize).sum::<usize>();
        let mut st = AllCyclesState {
            visit,
            contrib,
            num_states: 0,
            max_states: 0,
            error: false,
        };
        if max_ring > 6 {
            st.process(self, &d3, 6);
        }
        if max_ring > 10 {
            st.process(self, &d3, 10);
        }
        st.max_states = 1_048_576;
        st.process(self, &d3, max_ring);
    }
}

struct AllCyclesState {
    visit: Vec<u8>,
    contrib: Vec<i32>,
    num_states: u32,
    max_states: u32,
    error: bool,
}

impl AllCyclesState {
    fn mark_path(
        &mut self,
        m: &mut CdkMol,
        atom: usize,
        prev: usize,
        limit: usize,
        mut pi: i32,
        depth: usize,
    ) -> bool {
        if limit != 0 && depth > limit {
            return false;
        }
        if self.max_states != 0 {
            self.num_states += 1;
            if self.num_states >= self.max_states {
                return false;
            }
        }
        pi += self.contrib[atom];
        self.visit[atom] = 1;
        for k in m.nbrs(atom) {
            let b = m.adj_bond[k] as usize;
            if b == prev || !m.bond_ring[b] {
                continue;
            }
            let w = m.adj_nbr[k] as usize;
            if self.visit[w] == 2 {
                if (pi - 2) % 4 == 0 {
                    m.make_aromatic(b);
                    self.visit[atom] = 0;
                    return true;
                }
            } else if self.visit[w] == 0 && self.mark_path(m, w, b, limit, pi, depth + 1) {
                m.make_aromatic(b);
                self.visit[atom] = 0;
                return true;
            }
        }
        self.visit[atom] = 0;
        false
    }

    fn process(&mut self, m: &mut CdkMol, d3: &[usize], limit: usize) {
        for &atom in d3 {
            let pi = self.contrib[atom];
            self.visit[atom] = 2;
            for k in m.nbrs(atom) {
                let b = m.adj_bond[k] as usize;
                if m.bond_arom[b] || !m.bond_ring[b] {
                    continue;
                }
                let w = m.adj_nbr[k] as usize;
                if self.visit[w] != 0 {
                    continue;
                }
                self.num_states = 0;
                if self.mark_path(m, w, b, limit, pi, 2) {
                    m.make_aromatic(b);
                }
                if self.max_states != 0 && self.num_states >= self.max_states {
                    self.error = true;
                    return;
                }
            }
            self.visit[atom] = 0;
        }
    }
}

/// Hydrogen count CDK assigns to an atom (`CDKHydrogenAdder` after
/// `CDKAtomTypeMatcher`), where it differs from RDKit's.
///
/// CDK sets the implicit H count to the formal neighbour count of the perceived
/// atom type minus the number of bonds. For ordinary atoms this equals RDKit's
/// count; it differs for
///
/// * radicals: CDK's SMILES parser records no unpaired electrons, so for most
///   main-group elements the radical electrons become hydrogens (`[CH3]` →
///   methane, `[N]1C=CC=C1` → pyrrole), while metals, silicon and atoms without
///   a matching CDK type (e.g. `C[S](=O)=O`) keep their bracket H count;
/// * a few CDK atom types with unusual formal neighbour counts (`S.trioxide`,
///   bare metal ions such as `Zn.2plus`, `Ru.6`, `Mg.neutral`).
///
/// The cases follow `CDKAtomTypeMatcher` and were verified against CDK 2.13.
pub(super) fn cdk_hydrogens(mol: &Molecule, a: usize) -> u8 {
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

// ---------------------------------------------------------------------------
// ESSSR ring set (Cycles.tripletShort)
// ---------------------------------------------------------------------------

/// Shortest paths from one vertex (CDK `ShortestPaths`), keeping every
/// predecessor in discovery order so `path_to`/`paths_to` reproduce CDK's
/// route order.
struct ShortestPaths {
    dist: Vec<u32>,
    preds: Vec<SmallVec<[u32; 2]>>,
    precedes: Vec<bool>,
}

impl ShortestPaths {
    /// Plain BFS (`compute(adjacent)`), never entering `blocked`.
    fn new(g: &[Vec<u32>], start: usize, blocked: usize) -> Self {
        let n = g.len();
        let mut sp = ShortestPaths {
            dist: vec![u32::MAX; n],
            preds: vec![SmallVec::new(); n],
            precedes: Vec::new(),
        };
        sp.dist[start] = 0;
        let mut queue = Vec::with_capacity(n);
        queue.push(start);
        let mut i = 0;
        while i < queue.len() {
            let v = queue[i];
            i += 1;
            let d = sp.dist[v] + 1;
            for &w in &g[v] {
                let w = w as usize;
                if w == blocked {
                    continue;
                }
                if d < sp.dist[w] {
                    sp.dist[w] = d;
                    sp.preds[w].clear();
                    sp.preds[w].push(v as u32);
                    queue.push(w);
                } else if sp.dist[w] == d {
                    sp.preds[w].push(v as u32);
                }
            }
        }
        sp
    }

    /// BFS with vertex ordering (`compute(adjacent, ordering)`).
    fn with_ordering(g: &[Vec<u32>], start: usize, ordering: &[u32]) -> Self {
        let n = g.len();
        let mut sp = ShortestPaths {
            dist: vec![u32::MAX; n],
            preds: vec![SmallVec::new(); n],
            precedes: vec![false; n],
        };
        sp.dist[start] = 0;
        sp.precedes[start] = true;
        let ord_start = ordering[start];
        let mut queue = Vec::with_capacity(n);
        queue.push(start);
        let mut i = 0;
        while i < queue.len() {
            let v = queue[i];
            i += 1;
            let d = sp.dist[v] + 1;
            for &w in &g[v] {
                let w = w as usize;
                if d < sp.dist[w] {
                    sp.dist[w] = d;
                    sp.preds[w].clear();
                    sp.preds[w].push(v as u32);
                    sp.precedes[w] = sp.precedes[v] && ordering[w] < ord_start;
                    queue.push(w);
                } else if sp.dist[w] == d && sp.precedes[v] && ordering[w] < ord_start {
                    if sp.precedes[w] {
                        sp.preds[w].push(v as u32);
                    } else {
                        sp.precedes[w] = true;
                        sp.preds[w].clear();
                        sp.preds[w].push(v as u32);
                    }
                }
            }
        }
        sp
    }

    fn path_to(&self, end: usize) -> Vec<u32> {
        let len = self.dist[end] as usize + 1;
        let mut p = vec![0u32; len];
        let mut v = end;
        for i in (0..len).rev() {
            p[i] = v as u32;
            if i > 0 {
                v = self.preds[v][0] as usize;
            }
        }
        p
    }

    fn paths_to(&self, end: usize, out: &mut Vec<Vec<u32>>) {
        if self.dist[end] == u32::MAX {
            return;
        }
        if self.dist[end] == 0 {
            out.push(vec![end as u32]);
            return;
        }
        for &p in &self.preds[end] {
            let start = out.len();
            self.paths_to(p as usize, out);
            for path in &mut out[start..] {
                path.push(end as u32);
            }
        }
    }
}

/// GF(2) span test over edge vectors (CDK `GreedyBasis` / `BitMatrix`).
struct Gf2Basis {
    /// Reduced rows with their pivot bit.
    rows: Vec<(usize, Vec<u64>)>,
    size: usize,
}

impl Gf2Basis {
    fn reduce(&self, v: &mut [u64]) {
        for (piv, row) in &self.rows {
            if v[piv / 64] >> (piv % 64) & 1 == 1 {
                for (x, y) in v.iter_mut().zip(row) {
                    *x ^= y;
                }
            }
        }
    }

    /// Adds `v` if it is linearly independent of the basis; returns whether it was added.
    fn add_if_independent(&mut self, v: &[u64]) -> bool {
        let mut r = v.to_vec();
        self.reduce(&mut r);
        let Some(w) = r.iter().position(|&x| x != 0) else {
            return false;
        };
        let piv = w * 64 + r[w].trailing_zeros() as usize;
        // keep rows reduced w.r.t. the new pivot
        for (_, row) in self.rows.iter_mut() {
            if row[piv / 64] >> (piv % 64) & 1 == 1 {
                for (x, y) in row.iter_mut().zip(&r) {
                    *x ^= y;
                }
            }
        }
        self.rows.push((piv, r));
        self.size += 1;
        true
    }
}

/// Canonical (lexicographic) form of a cycle given without the closing vertex
/// (CDK `TripletShortCycles.lexicographic`).
fn lexicographic(p: &[u32]) -> Vec<u32> {
    let len = p.len();
    let off = (0..len).min_by_key(|&i| (p[i], i)).unwrap();
    let rev = p[(off + 1) % len] > p[(len + off - 1) % len];
    let mut q = vec![0u32; len];
    if rev {
        for i in 0..len {
            q[(len - i) % len] = p[(off + i) % len];
        }
    } else {
        for i in 0..len {
            q[i] = p[(off + i) % len];
        }
    }
    q
}

/// `InitialCycles.ofBiconnectedComponent` + `MinimumCycleBasis(ic, true)`; returns
/// the basis cycles as closed walks.
pub(super) fn minimum_cycle_basis(g: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let n = g.len();
    // vertex ordering by degree (counting sort, stable)
    let max_deg = g.iter().map(|x| x.len()).max().unwrap_or(0);
    let mut count = vec![0u32; max_deg.max(4) + 2];
    for adj in g {
        count[adj.len() + 1] += 1;
    }
    for i in 1..count.len() {
        count[i] += count[i - 1];
    }
    let mut ordering = vec![0u32; n];
    for v in 0..n {
        ordering[v] = count[g[v].len()];
        count[g[v].len()] += 1;
    }
    let n_deg2 = count[2] as usize;
    let mut vertices = vec![0usize; n];
    for v in 0..n {
        vertices[ordering[v] as usize] = v;
    }
    // edge indices
    let mut edge_index: Vec<SmallVec<[(u32, u32); 4]>> = vec![SmallVec::new(); n];
    let mut n_edges = 0u32;
    for v in 0..n {
        for &w in &g[v] {
            if w as usize > v {
                edge_index[v].push((w, n_edges));
                n_edges += 1;
            }
        }
    }
    let edge_of = |u: u32, v: u32| -> u32 {
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        edge_index[a as usize].iter().find(|e| e.0 == b).unwrap().1
    };
    let words = (n_edges as usize).div_ceil(64).max(1);

    let singleton_intersect = |p: &[u32], q: &[u32]| -> bool { (1..p.len()).all(|i| p[i] != q[i]) };

    let mut cycles: Vec<Vec<u32>> = Vec::new();
    let first = if n_deg2 < n { n_deg2 } else { 2 };
    let mut s: Vec<usize> = Vec::new();
    for i in first..n {
        let r = vertices[i];
        let sp = ShortestPaths::with_ordering(g, r, &ordering);
        for &y in vertices.iter().take(i) {
            if !sp.precedes[y] {
                continue;
            }
            s.clear();
            for &z in &g[y] {
                let z = z as usize;
                if !sp.precedes[z] {
                    continue;
                }
                let dz = sp.dist[z];
                let dy = sp.dist[y];
                if dz + 1 == dy {
                    s.push(z);
                } else if dz == dy && ordering[z] < ordering[y] {
                    let py = sp.path_to(y);
                    let pz = sp.path_to(z);
                    if singleton_intersect(&pz, &py) {
                        let mut path = py;
                        path.extend(pz.iter().rev());
                        cycles.push(path);
                    }
                }
            }
            for k in 0..s.len() {
                for l in k + 1..s.len() {
                    let pp = sp.path_to(s[k]);
                    let pq = sp.path_to(s[l]);
                    if singleton_intersect(&pp, &pq) {
                        let mut path = pp;
                        path.push(y as u32);
                        path.extend(pq.iter().rev());
                        cycles.push(path);
                    }
                }
            }
        }
    }
    cycles.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    cycles.dedup();

    let lim = n_edges as usize + 1 - n; // cyclomatic number
    let mut basis = Gf2Basis {
        rows: Vec::new(),
        size: 0,
    };
    let mut members = Vec::new();
    let mut vec_buf = vec![0u64; words];
    for c in cycles {
        if basis.size >= lim {
            break;
        }
        vec_buf.iter_mut().for_each(|x| *x = 0);
        for w in c.windows(2) {
            let e = edge_of(w[0], w[1]) as usize;
            vec_buf[e / 64] |= 1 << (e % 64);
        }
        if basis.add_if_independent(&vec_buf) {
            members.push(c);
        }
    }
    members
}

/// `TripletShortCycles(mcb, false)` on one fused ring system; cycles are pushed
/// (canonical, without closing vertex) into `out`.
fn triplet_short(g: &[Vec<u32>], out: &mut Vec<Vec<u32>>) {
    let n = g.len();
    let mcb = minimum_cycle_basis(g);
    let mut n_cycles = vec![0u32; n];
    for c in &mcb {
        for &v in &c[..c.len() - 1] {
            n_cycles[v as usize] += 1;
        }
        out.push(lexicographic(&c[..c.len() - 1]));
    }
    let mut paths = Vec::new();
    for v in 0..n {
        if n_cycles[v] <= 1 {
            continue;
        }
        let ws = &g[v];
        for i in 0..ws.len() {
            let sp = ShortestPaths::new(g, ws[i] as usize, v);
            for &wj in &ws[i + 1..] {
                paths.clear();
                sp.paths_to(wj as usize, &mut paths);
                for p in paths.drain(..) {
                    let mut p = p;
                    p.push(v as u32);
                    out.push(lexicographic(&p));
                }
            }
        }
    }
}

/// ESSSR ring set of the molecule graph: every ring as its atom cycle (atom
/// indices in ring order, without repeating the first atom).
pub(super) fn esssr(m: &CdkMol) -> Vec<Vec<u32>> {
    let n = m.n();
    let mut rings: Vec<Vec<u32>> = Vec::new();
    // ring systems = biconnected components (CDK RingSearch isolated/fused)
    let comp = &m.bond_comp;
    for c in 0..m.n_comp as u32 {
        let bonds: Vec<usize> = (0..comp.len()).filter(|&b| comp[b] == c).collect();
        let mut verts: Vec<u32> = bonds
            .iter()
            .flat_map(|&b| [m.bond_a[b], m.bond_b[b]])
            .collect();
        verts.sort_unstable();
        verts.dedup();
        let mut map = vec![u32::MAX; n];
        for (i, &v) in verts.iter().enumerate() {
            map[v as usize] = i as u32;
        }
        let sub: Vec<Vec<u32>> = verts
            .iter()
            .map(|&v| {
                m.nbrs(v as usize)
                    .filter(|&k| comp[m.adj_bond[k] as usize] == c)
                    .map(|k| map[m.adj_nbr[k] as usize])
                    .collect()
            })
            .collect();
        if bonds.len() == verts.len() {
            // isolated cycle: walk it
            let mut cyc = Vec::with_capacity(verts.len());
            let mut prev = u32::MAX;
            let mut cur = 0u32;
            loop {
                cyc.push(verts[cur as usize]);
                let nx = if sub[cur as usize][0] != prev {
                    sub[cur as usize][0]
                } else {
                    sub[cur as usize][1]
                };
                prev = cur;
                cur = nx;
                if cur == 0 {
                    break;
                }
            }
            rings.push(cyc);
        } else {
            let mut local = Vec::new();
            triplet_short(&sub, &mut local);
            local.sort();
            local.dedup();
            for c in local {
                rings.push(c.into_iter().map(|v| verts[v as usize]).collect());
            }
        }
    }
    rings
}

/// Labels the bonds of each ring system (biconnected component with more than
/// one bond, iterative Tarjan); returns the number of ring systems. Chain bonds
/// (bridges) keep `u32::MAX`.
fn ring_components(m: &CdkMol, comp: &mut [u32]) -> usize {
    let n = m.n();
    let mut disc = vec![u32::MAX; n];
    let mut low = vec![0u32; n];
    let mut time = 0u32;
    let mut n_comp = 0usize;
    let mut edge_stack: Vec<usize> = Vec::new();
    // stack frames: (vertex, parent bond, next adjacency position)
    let mut stack: Vec<(usize, usize, usize)> = Vec::new();
    for root in 0..n {
        if disc[root] != u32::MAX {
            continue;
        }
        disc[root] = time;
        low[root] = time;
        time += 1;
        stack.push((root, usize::MAX, m.adj_start[root] as usize));
        while let Some(&mut (v, pb, ref mut pos)) = stack.last_mut() {
            if *pos < m.adj_start[v + 1] as usize {
                let k = *pos;
                *pos += 1;
                let b = m.adj_bond[k] as usize;
                if b == pb {
                    continue;
                }
                let w = m.adj_nbr[k] as usize;
                if disc[w] == u32::MAX {
                    edge_stack.push(b);
                    disc[w] = time;
                    low[w] = time;
                    time += 1;
                    stack.push((w, b, m.adj_start[w] as usize));
                } else if disc[w] < disc[v] {
                    edge_stack.push(b);
                    low[v] = low[v].min(disc[w]);
                }
            } else {
                stack.pop();
                if let Some(&(u, _, _)) = stack.last() {
                    low[u] = low[u].min(low[v]);
                    if low[v] >= disc[u] {
                        // pop the component ending with the tree bond u-v
                        let start = edge_stack
                            .iter()
                            .rposition(|&e| e == pb)
                            .expect("tree bond on stack");
                        if edge_stack.len() - start > 1 {
                            for &e in &edge_stack[start..] {
                                comp[e] = n_comp as u32;
                            }
                            n_comp += 1;
                        }
                        edge_stack.truncate(start);
                    }
                }
            }
        }
    }
    n_comp
}

/// Port of Beam's (CDK's SMILES parser) Kekulé matching: `ArbitraryMatching`,
/// `MaximumMatching` and `UnionFind` from uk.ac.ebi.beam 1.3.10.
mod beam {
    use smallvec::SmallVec;
    use rustc_hash::FxHashMap as HashMap;

    const NIL: i32 = -1;

    /// Beam `Matching`: a vertex is matched only if its partner points back.
    pub(super) struct Matching {
        mate: Vec<i32>,
    }

    impl Matching {
        #[inline]
        pub(super) fn unmatched(&self, v: usize) -> bool {
            let w = self.mate[v];
            w < 0 || self.mate[w as usize] != v as i32
        }

        #[inline]
        pub(super) fn other(&self, v: usize) -> usize {
            self.mate[v] as usize
        }

        #[inline]
        fn set(&mut self, u: usize, v: usize) {
            self.mate[u] = v as i32;
            self.mate[v] = u as i32;
        }
    }

    type Adj = [SmallVec<[(u32, u32); 4]>];

    /// Matching of the `subset` atoms over `adj` (Localise.generateKekuleForm).
    pub(super) fn kekule_matching(adj: &Adj, subset: &[bool]) -> Matching {
        let n = subset.len();
        let mut m = Matching { mate: vec![NIL; n] };
        let n_subset = subset.iter().filter(|&&s| s).count();
        // ArbitraryMatching.initial
        let mut n_matched = 0;
        for v in 0..n {
            if !subset[v] || !m.unmatched(v) {
                continue;
            }
            for &(w, _) in &adj[v] {
                let w = w as usize;
                if m.unmatched(w) && subset[w] {
                    m.set(v, w);
                    n_matched += 2;
                    break;
                }
            }
        }
        if n_matched < n_subset {
            if n_subset - n_matched == 2 {
                n_matched = augment_once(adj, &mut m, n_matched, subset);
            }
            if n_matched < n_subset {
                MaximumMatching::run(adj, &mut m, subset);
            }
        }
        m
    }

    /// ArbitraryMatching.augmentOnce
    fn augment_once(adj: &Adj, m: &mut Matching, n_matched: usize, subset: &[bool]) -> usize {
        let n = subset.len();
        let Some(start) = (0..n).find(|&v| subset[v] && m.unmatched(v)) else {
            return n_matched;
        };
        let Some(end) = (start + 1..n).find(|&v| subset[v] && m.unmatched(v)) else {
            return n_matched;
        };
        let mut unvisited = subset.to_vec();
        let mut path = Vec::with_capacity(n);
        if find_path(adj, start, end, &mut unvisited, &mut path, m, false) {
            for pair in path.chunks(2) {
                m.set(pair[0], pair[1]);
            }
            return n_matched + 2;
        }
        n_matched
    }

    /// ArbitraryMatching.findPath (alternating path from `v` to `end`).
    fn find_path(
        adj: &Adj,
        v: usize,
        end: usize,
        unvisited: &mut [bool],
        path: &mut Vec<usize>,
        m: &Matching,
        match_needed: bool,
    ) -> bool {
        unvisited[v] = false;
        path.push(v);
        for &(w, _) in &adj[v] {
            let w = w as usize;
            if !unvisited[w] {
                continue;
            }
            if w == end {
                unvisited[v] = true;
                if path.len() % 2 == 1 {
                    path.push(w);
                    return true;
                }
                // odd length: give up on this branch (Beam returns 0 here)
                path.pop();
                return false;
            }
            if (m.other(w) == v) == match_needed
                && find_path(adj, w, end, unvisited, path, m, !match_needed)
            {
                unvisited[v] = true;
                return true;
            }
        }
        unvisited[v] = true;
        path.pop();
        false
    }

    /// Beam `UnionFind` (union by size, path compression).
    struct UnionFind {
        forest: Vec<i32>,
    }

    impl UnionFind {
        fn find(&mut self, u: usize) -> usize {
            if self.forest[u] < 0 {
                return u;
            }
            let r = self.find(self.forest[u] as usize);
            self.forest[u] = r as i32;
            r
        }

        fn union(&mut self, u: usize, v: usize) {
            let (ur, vr) = (self.find(u), self.find(v));
            if ur == vr {
                return;
            }
            let (s, l) = if self.forest[ur] < self.forest[vr] {
                (vr, ur)
            } else {
                (ur, vr)
            };
            self.forest[s] += self.forest[l];
            self.forest[l] = s as i32;
        }
    }

    /// Edmonds' maximum matching (Beam `MaximumMatching`).
    struct MaximumMatching<'a> {
        adj: &'a Adj,
        subset: &'a [bool],
        even: Vec<i32>,
        odd: Vec<i32>,
        queue: Vec<usize>,
        qi: usize,
        uf: UnionFind,
        bridges: HashMap<usize, (usize, usize)>,
        v_anc: Vec<bool>,
        w_anc: Vec<bool>,
    }

    impl<'a> MaximumMatching<'a> {
        fn run(adj: &'a Adj, m: &mut Matching, subset: &'a [bool]) {
            let n = subset.len();
            let mut mm = MaximumMatching {
                adj,
                subset,
                even: vec![NIL; n],
                odd: vec![NIL; n],
                queue: Vec::with_capacity(n),
                qi: 0,
                uf: UnionFind {
                    forest: vec![-1; n],
                },
                bridges: HashMap::default(),
                v_anc: vec![false; n],
                w_anc: vec![false; n],
            };
            while mm.augment(m) {}
        }

        fn augment(&mut self, m: &mut Matching) -> bool {
            let n = self.subset.len();
            self.even.iter_mut().for_each(|x| *x = NIL);
            self.odd.iter_mut().for_each(|x| *x = NIL);
            self.uf.forest.iter_mut().for_each(|x| *x = -1);
            self.bridges.clear();
            self.queue.clear();
            self.qi = 0;
            for v in 0..n {
                if self.subset[v] && m.unmatched(v) {
                    self.even[v] = v as i32;
                    self.queue.push(v);
                }
            }
            while self.qi < self.queue.len() {
                let v = self.queue[self.qi];
                self.qi += 1;
                for &(w, _) in &self.adj[v] {
                    let w = w as usize;
                    if !self.subset[w] {
                        continue;
                    }
                    let fw = self.uf.find(w);
                    if self.even[fw] != NIL {
                        if self.check(m, v, w) {
                            return true;
                        }
                    } else if self.odd[w] == NIL {
                        self.odd[w] = v as i32;
                        let u = m.other(w);
                        let fu = self.uf.find(u);
                        if self.even[fu] == NIL {
                            self.even[u] = w as i32;
                            self.queue.push(u);
                        }
                    }
                }
            }
            false
        }

        fn check(&mut self, m: &mut Matching, v: usize, w: usize) -> bool {
            if self.uf.find(v) == self.uf.find(w) {
                return false;
            }
            self.v_anc.iter_mut().for_each(|x| *x = false);
            self.w_anc.iter_mut().for_each(|x| *x = false);
            let (mut v_cur, mut w_cur) = (v, w);
            loop {
                v_cur = self.parent(true, v_cur);
                w_cur = self.parent(false, w_cur);
                if v_cur == w_cur {
                    self.blossom(v, w, v_cur);
                    return false;
                }
                let ev = self.even[v_cur] as usize;
                let ew = self.even[w_cur] as usize;
                if self.uf.find(ev) == v_cur && self.uf.find(ew) == w_cur {
                    self.augment_path(m, v);
                    self.augment_path(m, w);
                    m.set(v, w);
                    return true;
                }
                if self.w_anc[v_cur] {
                    self.blossom(v, w, v_cur);
                    return false;
                }
                if self.v_anc[w_cur] {
                    self.blossom(v, w, w_cur);
                    return false;
                }
            }
        }

        fn parent(&mut self, v_side: bool, curr: usize) -> usize {
            let curr = self.uf.find(curr);
            if v_side {
                self.v_anc[curr] = true;
            } else {
                self.w_anc[curr] = true;
            }
            let parent = self.uf.find(self.even[curr] as usize);
            if parent == curr {
                return curr;
            }
            if v_side {
                self.v_anc[parent] = true;
            } else {
                self.w_anc[parent] = true;
            }
            self.uf.find(self.odd[parent] as usize)
        }

        fn blossom(&mut self, v: usize, w: usize, base: usize) {
            let base = self.uf.find(base);
            let s1 = self.blossom_supports(v, w, base);
            let s2 = self.blossom_supports(w, v, base);
            for &x in &s1 {
                self.uf.union(x, s1[0]);
            }
            for &x in &s2 {
                self.uf.union(x, s2[0]);
            }
            let fb = self.uf.find(base);
            self.even[fb] = self.even[base];
        }

        fn blossom_supports(&mut self, v: usize, w: usize, base: usize) -> Vec<usize> {
            let mut path = vec![self.uf.find(v)];
            while *path.last().unwrap() != base {
                let u = self.even[*path.last().unwrap()] as usize;
                path.push(u);
                self.bridges.insert(u, (v, w));
                self.queue.push(u);
                let fo = self.uf.find(self.odd[u] as usize);
                path.push(fo);
            }
            path
        }

        fn augment_path(&mut self, m: &mut Matching, v: usize) {
            let mut path = Vec::new();
            self.build_path(m, &mut path, v, usize::MAX);
            let mut i = 2;
            while i < path.len() {
                m.set(path[i], path[i - 1]);
                i += 2;
            }
        }

        fn build_path(&self, m: &Matching, path: &mut Vec<usize>, mut start: usize, goal: usize) {
            loop {
                while self.odd[start] != NIL {
                    let (first, second) = self.bridges[&start];
                    let i = path.len();
                    self.build_path(m, path, first, start);
                    path[i..].reverse();
                    start = second;
                }
                path.push(start);
                if m.unmatched(start) {
                    return;
                }
                let o = m.other(start);
                path.push(o);
                if o == goal {
                    return;
                }
                start = self.odd[o] as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Sorted (ring size, all bonds aromatic) of the ESSSR ring set.
    fn rings(smiles: &str) -> Vec<(usize, bool)> {
        let mol = parse_smiles(smiles).unwrap();
        let m = CdkMol::build(&mol);
        let mut out: Vec<(usize, bool)> = esssr(&m)
            .iter()
            .map(|r| {
                let arom = (0..r.len()).all(|i| {
                    let (u, v) = (r[i] as usize, r[(i + 1) % r.len()]);
                    m.nbrs(u)
                        .any(|k| m.adj_nbr[k] == v && m.bond_arom[m.adj_bond[k] as usize])
                });
                (r.len(), arom)
            })
            .collect();
        out.sort_unstable();
        out
    }

    // Expected values: CDK 2.13 Cycles.tripletShort after SmartsPattern.prepare.
    #[test]
    fn esssr_matches_cdk() {
        assert_eq!(
            rings("c1ccc2ccccc2c1"),
            vec![(6, true), (6, true), (10, true)]
        );
        assert_eq!(rings("c1ccc(cc1)-c1ccccc1"), vec![(6, true), (6, true)]);
        assert_eq!(rings("C12C3C4C1C5C2C3C45"), vec![(4, false); 6]); // cubane
        assert_eq!(rings("C1C2CC3CC1CC(C2)C3"), vec![(6, false); 4]); // adamantane
        assert_eq!(
            rings("C1CC2CC1C2"),
            vec![(4, false), (5, false), (5, false)]
        ); // norbornane
        assert_eq!(rings("C1CC2(C1)CCC2"), vec![(4, false), (4, false)]); // spiro
    }

    #[test]
    fn daylight_aromaticity_matches_cdk() {
        // azulene: only the 10-membered envelope is aromatic in CDK's Daylight model
        assert_eq!(
            rings("C1=CC2=CC=CC=CC2=C1"),
            vec![(5, false), (7, false), (10, true)]
        );
        assert_eq!(rings("O=c1cccc[nH]1"), vec![(6, true)]);
    }

    #[test]
    fn hydrogens_follow_cdk_atom_types() {
        let h = |smiles: &str| {
            let mol = parse_smiles(smiles).unwrap();
            CdkMol::build(&mol).h_fields
        };
        assert_eq!(h("[CH3]"), vec![4]);
        assert_eq!(h("C[N]C"), vec![3, 1, 3]);
        assert_eq!(h("C[Si](C)C"), vec![3, 0, 3, 3]);
        assert_eq!(h("NC[S](=O)=O"), vec![2, 2, 0, 0, 0]);
        assert_eq!(h("[Zn+2]"), vec![2]);
        assert_eq!(h("C=S(=O)=O"), vec![2, 3, 0, 0]);
    }
}
