//! CDK-compatible molecule view for descriptors ported from CDK 2.13.
//!
//! FerrumChem perceives molecules the RDKit way (aromatic bond type, RDKit
//! `cleanUp`). Several CDK descriptors instead read the Kekulé bond orders that
//! CDK's SMILES parser assigns, and/or CDK's Daylight-style aromaticity flags.
//! This module rebuilds that view locally without touching shared perception:
//!
//! - **Bond/atom order**: CDK (via Beam) stores bonds ordered by the larger atom
//!   index, then by creation time; Beam lists the edges of an atom in creation
//!   order. Both are reproduced by sorting on `(max(begin, end), bond index)`.
//! - **Kekulé orders**: aromatic bonds are localised the way Beam's
//!   `Localise.generateKekuleForm` does it (greedy matching in atom/edge order,
//!   `augmentOnce`, then a maximum matching).
//! - **RDKit cleanUp undo**: halogen oxides (`[Cl+3]([O-])...`) are turned back
//!   into the `Cl(=O)` form CDK keeps.
//! - **Aromaticity**: `Aromaticity.Model.Daylight` (aromatic types + the
//!   backtracking cycle search, as used by `SmartsPattern`) and
//!   `ElectronDonation.daylight()` (`DaylightModel`, as used by the CDK
//!   reference preparation).

use crate::mol::molecule::Molecule;
use crate::types::BondType;
use smallvec::SmallVec;

/// CDK-style view of a molecule (Kekulé bond orders, CDK bond order).
pub(crate) struct CdkMol {
    /// Atomic numbers.
    pub z: Vec<u8>,
    /// Formal charges (after undoing RDKit halogen cleanUp).
    pub charge: Vec<i8>,
    /// Implicit (and RDKit-folded explicit) hydrogen count.
    pub hcount: Vec<u8>,
    /// Ring membership of atoms.
    pub ring_atom: Vec<bool>,
    /// Neighbours as `(atom, bond)` in CDK/Beam bond order.
    pub adj: Vec<SmallVec<[(u32, u32); 4]>>,
    /// Bond end points.
    pub ends: Vec<(u32, u32)>,
    /// Kekulé bond order (1, 2, 3).
    pub order: Vec<u8>,
    /// Ring membership of bonds.
    pub ring_bond: Vec<bool>,
}

impl CdkMol {
    /// Builds the CDK view (including Kekulé bond orders) of a sanitized molecule.
    pub fn new(mol: &Molecule) -> Self {
        let n = mol.num_atoms();
        let nb = mol.num_bonds();
        let mut z = Vec::with_capacity(n);
        let mut charge = Vec::with_capacity(n);
        let mut hcount = Vec::with_capacity(n);
        let mut ring_atom = Vec::with_capacity(n);
        for a in &mol.atoms {
            z.push(a.atomic_number);
            charge.push(a.formal_charge);
            // CDK's SMILES parser keeps no radicals: CDKHydrogenAdder fills the
            // typed valence with hydrogens (e.g. `[As]1C=CC=C1` becomes arsole).
            let rad = if matches!(a.atomic_number, 5 | 6 | 7 | 8 | 14 | 15 | 16 | 33 | 34) {
                a.num_radicals()
            } else {
                0
            };
            hcount.push(a.total_h() + rad);
            ring_atom.push(a.is_in_ring());
        }
        let mut ends = Vec::with_capacity(nb);
        let mut order = Vec::with_capacity(nb);
        let mut ring_bond = Vec::with_capacity(nb);
        let mut arom_bond = Vec::with_capacity(nb);
        for b in &mol.bonds {
            ends.push((b.source as u32, b.target as u32));
            let aromatic = b.bond_type == BondType::Aromatic;
            arom_bond.push(aromatic);
            order.push(match b.bond_type {
                BondType::Double => 2,
                BondType::Triple => 3,
                _ => 1,
            });
            ring_bond.push(b.is_in_ring());
        }
        let mut adj: Vec<SmallVec<[(u32, u32); 4]>> = Vec::with_capacity(n);
        for a in 0..n {
            let mut v: SmallVec<[(u32, u32); 4]> = mol.adjacency[a]
                .iter()
                .map(|&bi| {
                    let (s, t) = ends[bi as usize];
                    (if s as usize == a { t } else { s }, bi as u32)
                })
                .collect();
            v.sort_unstable_by_key(|&(_, bi)| {
                let (s, t) = ends[bi as usize];
                (s.max(t), bi)
            });
            adj.push(v);
        }
        let mut m = CdkMol { z, charge, hcount, ring_atom, adj, ends, order, ring_bond };
        m.undo_halogen_cleanup(mol);
        if arom_bond.iter().any(|&x| x) {
            m.kekulize(mol, &arom_bond);
        }
        m
    }

    /// Number of atoms.
    #[inline]
    pub fn n(&self) -> usize {
        self.z.len()
    }

    /// Bond between two atoms (if any).
    #[inline]
    pub fn bond(&self, a: usize, b: usize) -> Option<usize> {
        self.adj[a].iter().find(|&&(w, _)| w as usize == b).map(|&(_, bi)| bi as usize)
    }

    /// Heavy (explicit) degree.
    #[inline]
    pub fn degree(&self, a: usize) -> usize {
        self.adj[a].len()
    }

    /// Undoes RDKit's halogen cleanUp (`Cl(=O)(=O)(=O)O` -> `[Cl+3]([O-])...`).
    /// A halogen written in brackets (`[Cl+3]`) was charged in the input and
    /// keeps its charge, as in CDK.
    fn undo_halogen_cleanup(&mut self, mol: &Molecule) {
        for a in 0..self.n() {
            if !matches!(self.z[a], 17 | 35 | 53) || mol.atoms[a].no_implicit_h() {
                continue;
            }
            let q = self.charge[a];
            if !(1..=3).contains(&q) || self.adj[a].is_empty() {
                continue;
            }
            if !self.adj[a].iter().all(|&(w, _)| self.z[w as usize] == 8) {
                continue;
            }
            let cands: SmallVec<[(u32, u32); 4]> = self.adj[a]
                .iter()
                .copied()
                .filter(|&(w, bi)| self.charge[w as usize] == -1 && self.order[bi as usize] == 1)
                .collect();
            if cands.len() < q as usize {
                continue;
            }
            for &(w, bi) in cands.iter().take(q as usize) {
                self.order[bi as usize] = 2;
                self.charge[w as usize] = 0;
            }
            self.charge[a] = 0;
        }
    }

    /// Assigns Kekulé orders to the RDKit-aromatic bonds (Beam's localisation).
    fn kekulize(&mut self, mol: &Molecule, arom_bond: &[bool]) {
        let n = self.n();
        let mut in_subset = vec![false; n];
        let mut subset: Vec<usize> = Vec::new();
        for a in 0..n {
            if !mol.atoms[a].is_aromatic() {
                continue;
            }
            let mut fixed = self.hcount[a] as i32;
            let mut n_arom = 0;
            for &(_, bi) in &self.adj[a] {
                if arom_bond[bi as usize] {
                    n_arom += 1;
                } else if mol.bonds[bi as usize].bond_type != BondType::Dative {
                    fixed += self.order[bi as usize] as i32;
                }
            }
            if n_arom == 0 {
                continue;
            }
            let need = crate::algo::sanitize::total_valence(mol, a) - fixed - n_arom;
            if need == 1 {
                in_subset[a] = true;
                subset.push(a);
            }
        }
        if subset.is_empty() {
            return;
        }
        let mut mate = vec![u32::MAX; n];
        let total = subset.len();
        // ArbitraryMatching.initial
        let mut matched = 0;
        for &v in &subset {
            if mate[v] != u32::MAX {
                continue;
            }
            for &(w, bi) in &self.adj[v] {
                let w = w as usize;
                if arom_bond[bi as usize] && mate[w] == u32::MAX && in_subset[w] {
                    mate[v] = w as u32;
                    mate[w] = v as u32;
                    matched += 2;
                    break;
                }
            }
        }
        if matched < total && total - matched == 2 {
            matched = self.augment_once(&mut mate, matched, &subset, &in_subset, arom_bond);
        }
        if matched < total {
            self.maximum_matching(&mut mate, &subset, &in_subset, arom_bond);
        }
        for &v in &subset {
            let w = mate[v];
            if w != u32::MAX && (w as usize) > v
                && let Some(bi) = self.bond(v, w as usize) {
                    self.order[bi] = 2;
                }
        }
    }

    /// Beam `ArbitraryMatching.augmentOnce`.
    fn augment_once(
        &self,
        mate: &mut [u32],
        matched: usize,
        subset: &[usize],
        in_subset: &[bool],
        arom_bond: &[bool],
    ) -> usize {
        let mut unmatched = subset.iter().copied().filter(|&v| mate[v] == u32::MAX);
        let (Some(start), Some(end)) = (unmatched.next(), unmatched.next()) else {
            return matched;
        };
        let mut unvisited = in_subset.to_vec();
        let mut path = Vec::with_capacity(subset.len());
        if self.find_path(start, end, &mut unvisited, &mut path, mate, false, arom_bond) {
            for pair in path.chunks(2) {
                if let [a, b] = *pair {
                    mate[a] = b as u32;
                    mate[b] = a as u32;
                }
            }
            return matched + 2;
        }
        matched
    }

    #[allow(clippy::too_many_arguments)]
    fn find_path(
        &self,
        v: usize,
        end: usize,
        unvisited: &mut [bool],
        path: &mut Vec<usize>,
        mate: &[u32],
        match_needed: bool,
        arom_bond: &[bool],
    ) -> bool {
        unvisited[v] = false;
        path.push(v);
        for &(w, bi) in &self.adj[v] {
            if !arom_bond[bi as usize] {
                continue;
            }
            let w = w as usize;
            if !unvisited[w] {
                continue;
            }
            if w == end {
                path.push(w);
                unvisited[v] = true;
                if path.len() % 2 == 1 {
                    path.pop();
                    path.pop();
                    return false;
                }
                return true;
            } else if (mate[w] == v as u32) == match_needed
                && self.find_path(w, end, unvisited, path, mate, !match_needed, arom_bond)
            {
                unvisited[v] = true;
                return true;
            }
        }
        unvisited[v] = true;
        path.pop();
        false
    }

    /// Edmonds' blossom algorithm on the aromatic subgraph, extending `mate`.
    fn maximum_matching(&self, mate: &mut [u32], subset: &[usize], in_subset: &[bool], arom_bond: &[bool]) {
        let n = self.n();
        let none = usize::MAX;
        let mut m: Vec<usize> = mate.iter().map(|&x| if x == u32::MAX { none } else { x as usize }).collect();
        let nbrs = |v: usize| {
            self.adj[v]
                .iter()
                .filter(move |&&(w, bi)| arom_bond[bi as usize] && in_subset[w as usize])
                .map(|&(w, _)| w as usize)
        };
        let mut parent = vec![none; n];
        let mut base: Vec<usize> = (0..n).collect();
        let mut used = vec![false; n];
        let mut blossom = vec![false; n];
        let mut q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

        let lca = |a0: usize, b0: usize, m: &[usize], base: &[usize], parent: &[usize]| -> usize {
            let mut seen = vec![false; n];
            let mut a = a0;
            loop {
                a = base[a];
                seen[a] = true;
                if m[a] == none {
                    break;
                }
                a = parent[m[a]];
            }
            let mut b = b0;
            loop {
                b = base[b];
                if seen[b] {
                    return b;
                }
                b = parent[m[b]];
            }
        };

        for &root in subset {
            if m[root] != none {
                continue;
            }
            for i in 0..n {
                parent[i] = none;
                base[i] = i;
                used[i] = false;
            }
            used[root] = true;
            q.clear();
            q.push_back(root);
            let mut found = none;
            'bfs: while let Some(v) = q.pop_front() {
                for to in nbrs(v) {
                    if base[v] == base[to] || m[v] == to {
                        continue;
                    }
                    if to == root || (m[to] != none && parent[m[to]] != none) {
                        let cur = lca(v, to, &m, &base, &parent);
                        for x in blossom.iter_mut() {
                            *x = false;
                        }
                        // mark path v -> cur and to -> cur
                        for (start, child0) in [(v, to), (to, v)] {
                            let mut x = start;
                            let mut child = child0;
                            while base[x] != cur {
                                blossom[base[x]] = true;
                                blossom[base[m[x]]] = true;
                                parent[x] = child;
                                child = m[x];
                                x = parent[m[x]];
                            }
                        }
                        for i in 0..n {
                            if blossom[base[i]] {
                                base[i] = cur;
                                if !used[i] {
                                    used[i] = true;
                                    q.push_back(i);
                                }
                            }
                        }
                    } else if parent[to] == none {
                        parent[to] = v;
                        if m[to] == none {
                            found = to;
                            break 'bfs;
                        }
                        let mt = m[to];
                        used[mt] = true;
                        q.push_back(mt);
                    }
                }
            }
            if found != none {
                let mut v = found;
                while v != none {
                    let pv = parent[v];
                    let ppv = m[pv];
                    m[v] = pv;
                    m[pv] = v;
                    v = ppv;
                }
            }
        }
        for v in 0..n {
            mate[v] = if m[v] == none { u32::MAX } else { m[v] as u32 };
        }
    }
}

// ---------------------------------------------------------------------------
// Aromaticity: shared helpers
// ---------------------------------------------------------------------------

#[inline]
fn huckel(sum: i32) -> bool {
    (sum - 2) % 4 == 0
}

// ---------------------------------------------------------------------------
// Aromaticity.Model.Daylight (AromaticTypeModel DAYLIGHT) + backtracking search
// ---------------------------------------------------------------------------

/// CDK `AromaticTypeMatcher` packed bond info.
fn binfo(m: &CdkMol, a: usize) -> u32 {
    let mut info = m.hcount[a] as u32;
    if m.degree(a) + info as usize > 15 {
        return 0x1000;
    }
    for &(_, bi) in &m.adj[a] {
        let bi = bi as usize;
        match m.order[bi] {
            1 => info += 0x0001,
            2 => info += if m.ring_bond[bi] { 0x0010 } else { 0x0100 },
            _ => info += 0x1000,
        }
    }
    info
}

fn has_exo_het(m: &CdkMol, a: usize) -> bool {
    for &(w, bi) in &m.adj[a] {
        let bi = bi as usize;
        if m.ring_bond[bi] || m.order[bi] != 2 {
            continue;
        }
        return m.z[w as usize] != 6;
    }
    false
}

fn has_oxide(m: &CdkMol, a: usize) -> bool {
    for &(w, bi) in &m.adj[a] {
        let bi = bi as usize;
        if m.ring_bond[bi] {
            continue;
        }
        let w = w as usize;
        if m.z[w] != 8 {
            return false;
        }
        if m.order[bi] == 2 {
            return true;
        }
        return m.order[bi] == 1 && m.charge[a] == 1 && m.charge[w] == -1;
    }
    false
}

/// Electron contribution under `Aromaticity.Model.Daylight` (-1 = not aromatic).
fn aromatic_type_daylight_contrib(m: &CdkMol, a: usize) -> i32 {
    if !m.ring_atom[a] {
        return -1;
    }
    let z = m.z[a];
    if m.degree(a) > 3 && z != 15 {
        return -1;
    }
    if m.hcount[a] > 1 && z != 15 {
        return -1;
    }
    let q = m.charge[a];
    match z {
        6 => match (q, binfo(m, a)) {
            (-1, 0x0011) => 1, // C2_MINUS
            (-1, 0x0003) => 2, // C3_MINUS
            (0, 0x0012) => 1,  // C3
            (0, 0x0102) => {
                if has_exo_het(m, a) {
                    0 // C3_ENEG_EXO
                } else {
                    1 // C3_EXO
                }
            }
            (1, 0x0011) => 1, // C2_PLUS
            (1, 0x0003) => 0, // C3_PLUS
            _ => -1,
        },
        7 => {
            let bi = binfo(m, a);
            match q {
                -1 if bi == 0x0002 => 2, // N2_MINUS
                0 => match bi {
                    0x0003 => 2, // N3
                    0x0011 => 1, // N2
                    0x0111 if has_oxide(m, a) => 1,
                    _ => -1,
                },
                1 if bi == 0x0012 => 1, // N3_OXIDE_PLUS / N3_PLUS
                _ => -1,
            }
        }
        15 => {
            let bi = binfo(m, a);
            match q {
                -1 if bi == 0x0002 => 2, // P2_MINUS
                0 => match bi {
                    0x0013 => -1, // P4 (not in Daylight)
                    0x0003 => 2,  // P3
                    0x0011 => 1,  // P2
                    0x0111 if has_oxide(m, a) => 1,
                    _ => -1,
                },
                1 if bi == 0x0012 => 1, // P3_OXIDE_PLUS / P3_PLUS
                _ => -1,
            }
        }
        33 => {
            let bi = binfo(m, a);
            match q {
                0 if bi == 0x0003 => 2, // As3
                _ => -1,                // As2_MINUS, As2, As3_PLUS not in Daylight
            }
        }
        8 => match (q, binfo(m, a)) {
            (0, 0x0002) => 2, // O2
            (1, 0x0011) => 1, // O2_PLUS
            _ => -1,
        },
        16 => {
            let bi = binfo(m, a);
            if q == 0 {
                match bi {
                    0x0012 => return -1, // S3 (not in Daylight)
                    0x0002 => return 2,  // S2
                    0x0020 => return -1, // S2_CUML
                    0x0102 if has_oxide(m, a) => return 2, // S3_OXIDE
                    _ => {}
                }
            }
            if q == 1 && bi == 0x0011 {
                return 1; // S2_PLUS
            }
            -1
        }
        34 => {
            let bi = binfo(m, a);
            if q == 0 {
                match bi {
                    0x0002 => return 2,                    // Se2
                    0x0012 => return -1,                   // Se3
                    0x0102 if has_oxide(m, a) => return 2, // Se3_OXIDE
                    _ => {}
                }
            }
            if q == 1 && bi == 0x0011 {
                return 1; // Se2_PLUS
            }
            -1
        }
        _ => -1,
    }
}

/// Aromatic atom/bond flags as set by CDK `SmartsPattern.prepare`
/// (`Aromaticity.apply(Aromaticity.Model.Daylight, mol)`).
pub(crate) fn smarts_daylight_aromaticity(m: &CdkMol) -> (Vec<bool>, Vec<bool>) {
    let n = m.n();
    let contrib: Vec<i32> = (0..n).map(|a| aromatic_type_daylight_contrib(m, a)).collect();
    let mut arom_atom = vec![false; n];
    let mut arom_bond = vec![false; m.ends.len()];
    if contrib.iter().all(|&c| c < 0) {
        return (arom_atom, arom_bond);
    }
    let mut st = Backtrack {
        m,
        contrib,
        visit: vec![0u8; n],
        arom_atom: &mut arom_atom,
        arom_bond: &mut arom_bond,
        num_states: 0,
        max_states: 0,
        error: false,
    };
    st.apply(n.max(3));
    (arom_atom, arom_bond)
}

const ALLOWED_STATE_COUNT: usize = 1_048_576;

struct Backtrack<'a> {
    m: &'a CdkMol,
    contrib: Vec<i32>,
    visit: Vec<u8>,
    arom_atom: &'a mut Vec<bool>,
    arom_bond: &'a mut Vec<bool>,
    num_states: usize,
    max_states: usize,
    error: bool,
}

impl Backtrack<'_> {
    fn make_aromatic(&mut self, bi: usize) {
        self.arom_bond[bi] = true;
        let (s, t) = self.m.ends[bi];
        self.arom_atom[s as usize] = true;
        self.arom_atom[t as usize] = true;
    }

    fn prune_from(&mut self, start: usize) {
        let mut atom = Some(start);
        while let Some(a) = atom {
            let mut next = None;
            for &(w, bi) in &self.m.adj[a] {
                if !self.m.ring_bond[bi as usize] || self.contrib[w as usize] == -1 {
                    continue;
                }
                if next.is_some() {
                    return;
                }
                next = Some(w as usize);
            }
            self.contrib[a] = -1;
            atom = next;
        }
    }

    fn is_fusion_or_bridge(&self, a: usize) -> bool {
        if self.m.degree(a) < 3 || self.contrib[a] < 0 {
            return false;
        }
        let count = self.m.adj[a]
            .iter()
            .filter(|&&(w, bi)| self.m.ring_bond[bi as usize] && self.contrib[w as usize] != -1)
            .count();
        count >= 3
    }

    fn flood_fill(&mut self, a: usize, prev: usize, label: u8) {
        self.visit[a] = label;
        for k in 0..self.m.adj[a].len() {
            let (w, bi) = self.m.adj[a][k];
            if bi as usize == prev || !self.m.ring_bond[bi as usize] {
                continue;
            }
            if self.visit[w as usize] == 0 {
                self.flood_fill(w as usize, bi as usize, label);
            }
        }
    }

    fn apply(&mut self, max_ring_size: usize) {
        let n = self.m.n();
        for a in 0..n {
            if self.contrib[a] != -1 {
                self.prune_from(a);
            }
        }
        for a in 0..n {
            if self.contrib[a] < 0 {
                self.visit[a] = 1;
            }
        }
        let mut d3 = Vec::new();
        for a in 0..n {
            if self.is_fusion_or_bridge(a) {
                d3.push(a);
                self.flood_fill(a, usize::MAX, 2);
            }
        }
        self.process_simple_rings();
        if d3.is_empty() {
            return;
        }
        for v in self.visit.iter_mut() {
            if *v == 2 {
                *v = 0;
            }
        }
        self.max_states = 0;
        if max_ring_size > 6 {
            self.process_complex_rings(&d3, 6);
        }
        if max_ring_size > 10 {
            self.process_complex_rings(&d3, 10);
        }
        self.max_states = ALLOWED_STATE_COUNT;
        self.process_complex_rings(&d3, max_ring_size);
    }

    fn process_simple_rings(&mut self) {
        let n = self.m.n();
        let mut ring: Vec<usize> = Vec::new();
        for start in 0..n {
            if self.visit[start] != 0 {
                continue;
            }
            ring.clear();
            let mut sum = 0;
            let mut prev = usize::MAX;
            let mut atom = Some(start);
            while let Some(a) = atom {
                let mut next = None;
                self.visit[a] = 1;
                sum += self.contrib[a];
                for &(w, bi) in &self.m.adj[a] {
                    let bi = bi as usize;
                    if bi == prev || !self.m.ring_bond[bi] {
                        continue;
                    }
                    if self.visit[w as usize] != 0 {
                        continue;
                    }
                    ring.push(bi);
                    prev = bi;
                    next = Some(w as usize);
                }
                atom = next;
            }
            if !huckel(sum) {
                continue;
            }
            for k in 0..ring.len() {
                self.make_aromatic(ring[k]);
            }
        }
    }

    fn mark_path(&mut self, a: usize, prev: usize, limit: usize, mut pi: i32, depth: usize) -> bool {
        if limit != 0 && depth > limit {
            return false;
        }
        if self.max_states != 0 {
            self.num_states += 1;
            if self.num_states >= self.max_states {
                return false;
            }
        }
        pi += self.contrib[a];
        self.visit[a] = 1;
        for k in 0..self.m.adj[a].len() {
            let (w, bi) = self.m.adj[a][k];
            let (w, bi) = (w as usize, bi as usize);
            if bi == prev || !self.m.ring_bond[bi] {
                continue;
            }
            if self.visit[w] == 2 {
                if huckel(pi) {
                    self.make_aromatic(bi);
                    self.visit[a] = 0;
                    return true;
                }
            } else if self.visit[w] == 0 && self.mark_path(w, bi, limit, pi, depth + 1) {
                self.make_aromatic(bi);
                self.visit[a] = 0;
                return true;
            }
        }
        self.visit[a] = 0;
        false
    }

    fn process_complex_rings(&mut self, d3: &[usize], limit: usize) {
        if self.error {
            return;
        }
        for &a in d3 {
            let pi = self.contrib[a];
            self.visit[a] = 2;
            for k in 0..self.m.adj[a].len() {
                let (w, bi) = self.m.adj[a][k];
                let (w, bi) = (w as usize, bi as usize);
                if self.arom_bond[bi] || !self.m.ring_bond[bi] {
                    continue;
                }
                if self.visit[w] != 0 {
                    continue;
                }
                self.num_states = 0;
                if self.mark_path(w, bi, limit, pi, 2) {
                    self.make_aromatic(bi);
                }
                if self.max_states != 0 && self.num_states >= self.max_states {
                    self.error = true;
                    return;
                }
            }
            self.visit[a] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// ElectronDonation.daylight() (DaylightModel) + all-cycles search
// ---------------------------------------------------------------------------

fn dm_normal(z: u8, q: i8, valence: i32) -> bool {
    match z {
        0 => true,
        6 => {
            if q == -1 || q == 1 {
                valence == 3
            } else {
                q == 0 && valence == 4
            }
        }
        7 | 15 | 33 => {
            if q == -1 {
                valence == 2
            } else if q == 1 {
                valence == 4
            } else {
                q == 0 && (valence == 3 || (valence == 5 && z == 7))
            }
        }
        8 => {
            if q == 1 {
                valence == 3
            } else {
                q == 0 && valence == 2
            }
        }
        16 | 34 => {
            if q == 1 {
                valence == 3
            } else {
                q == 0 && (valence == 2 || valence == 4 || valence == 6)
            }
        }
        _ => false,
    }
}

fn dm_valence(element: i32) -> i32 {
    match element {
        5 | 13 | 31 => 3,
        6 | 14 | 32 => 4,
        7 | 15 | 33 => 5,
        8 | 16 | 34 => 6,
        9 | 17 | 35 => 7,
        _ => -100, // CDK throws; never reached for valid input
    }
}

/// Electron contributions of CDK's `DaylightModel` (-1 = cannot be aromatic).
pub(crate) fn daylight_model_contrib(m: &CdkMol) -> Vec<i32> {
    let n = m.n();
    let mut out = vec![-1; n];
    for i in 0..n {
        let z = m.z[i];
        let q = m.charge[i];
        let h = m.hcount[i] as i32;
        let mut degree = h;
        let mut valence = h;
        let mut n_cyclic_pi = 0;
        let mut exo: Option<usize> = None;
        for &(w, bi) in &m.adj[i] {
            let bi = bi as usize;
            degree += 1;
            let o = m.order[bi] as i32;
            if o == 2 {
                if m.ring_bond[bi] {
                    n_cyclic_pi += 1;
                } else {
                    exo = Some(w as usize);
                }
            }
            valence += o;
        }
        if !dm_normal(z, q, valence) {
            continue;
        }
        if !matches!(z, 0 | 6 | 7 | 8 | 15 | 16 | 33 | 34) || !m.ring_atom[i] || degree > 3 || n_cyclic_pi > 1 {
            continue;
        }
        out[i] = if let Some(o) = exo {
            let other = m.z[o];
            match z {
                0 | 6 => {
                    if other != 6 {
                        0
                    } else {
                        1
                    }
                }
                7 | 15 => {
                    if q == 1 || (q == 0 && other == 8 && n_cyclic_pi == 1) {
                        1
                    } else {
                        -1
                    }
                }
                16
                    if q == 0 && other == 8 => {
                        2
                    }
                _ => -1,
            }
        } else if n_cyclic_pi == 1 {
            if z == 33 {
                -1
            } else {
                1
            }
        } else if q <= 0 && q > -3 {
            if z == 0 || dm_valence(z as i32 - q as i32) - valence >= 2 {
                2
            } else {
                -1
            }
        } else if z == 6 && q > 0 {
            0
        } else {
            -1
        };
    }
    out
}

/// Whether atom `t` lies on a simple cycle (through ring bonds of atoms with
/// `contrib >= 0`) whose electron sum satisfies Hückel's rule — i.e. whether
/// CDK's `Aromaticity(daylight(), Cycles.all())` marks it aromatic.
/// The search is bounded; when the budget is exhausted only cycles up to size 6
/// are considered (mirrors `Cycles.or(Cycles.all(), Cycles.all(6))`).
pub(crate) fn on_huckel_cycle(m: &CdkMol, contrib: &[i32], t: usize) -> bool {
    if contrib[t] < 0 || !m.ring_atom[t] {
        return false;
    }
    let n = m.n();
    let mut on_path = vec![false; n];
    let mut budget: usize = 2_000_000;
    on_path[t] = true;
    if cycle_dfs(m, contrib, t, t, usize::MAX, contrib[t], 1, usize::MAX, &mut on_path, &mut budget) {
        return true;
    }
    if budget == 0 {
        // intractable: fall back to cycles of length <= 6
        let mut on_path = vec![false; n];
        on_path[t] = true;
        let mut budget = usize::MAX;
        return cycle_dfs(m, contrib, t, t, usize::MAX, contrib[t], 1, 6, &mut on_path, &mut budget);
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn cycle_dfs(
    m: &CdkMol,
    contrib: &[i32],
    start: usize,
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
        if w == start {
            if len >= 3 && huckel(sum) {
                return true;
            }
            continue;
        }
        if on_path[w] || len >= max_len {
            continue;
        }
        on_path[w] = true;
        let r = cycle_dfs(m, contrib, start, w, bi, sum + contrib[w], len + 1, max_len, on_path, budget);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn kek(smi: &str) -> Vec<u8> {
        let mol = parse_smiles(smi).unwrap();
        CdkMol::new(&mol).order
    }

    #[test]
    fn kekule_benzene_alternates() {
        let o = kek("c1ccccc1");
        assert_eq!(o.iter().filter(|&&x| x == 2).count(), 3);
        // Beam: greedy matching in atom order -> 0=1, 2=3, 4=5
        assert_eq!(o, vec![2, 1, 2, 1, 2, 1]);
    }

    #[test]
    fn kekule_pyrrole_and_fused() {
        assert_eq!(kek("c1cc[nH]c1").iter().filter(|&&x| x == 2).count(), 2);
        assert_eq!(kek("c1ccc2ccccc2c1").iter().filter(|&&x| x == 2).count(), 5);
        assert_eq!(kek("c1ccc2c(c1)c1cccc3cccc2c31").iter().filter(|&&x| x == 2).count(), 8);
    }

    /// Every aromatic atom that needs a pi bond gets exactly one (perfect
    /// matching) on the whole parity set.
    #[test]
    fn kekule_is_perfect_on_parity_set() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/parity_molecules.smi");
        let text = std::fs::read_to_string(path).unwrap();
        for smi in text.lines() {
            let Ok(mol) = parse_smiles(smi) else { continue };
            let m = CdkMol::new(&mol);
            for a in 0..mol.num_atoms() {
                if !mol.atoms[a].is_aromatic() {
                    continue;
                }
                let mut fixed = m.hcount[a] as i32 - mol.atoms[a].num_radicals() as i32;
                let (mut n_arom, mut n_dbl) = (0, 0);
                for &(_, bi) in &m.adj[a] {
                    let b = &mol.bonds[bi as usize];
                    if b.bond_type == BondType::Aromatic {
                        n_arom += 1;
                        n_dbl += (m.order[bi as usize] == 2) as i32;
                    } else if b.bond_type != BondType::Dative {
                        fixed += m.order[bi as usize] as i32;
                    }
                }
                let need = crate::algo::sanitize::total_valence(&mol, a) - fixed - n_arom;
                assert_eq!(n_dbl, need.clamp(0, 1), "{smi} atom {a}");
            }
        }
    }

    #[test]
    fn halogen_oxide_restored() {
        let mol = parse_smiles("OCl(=O)(=O)=O").unwrap();
        let m = CdkMol::new(&mol);
        assert_eq!(m.charge.iter().filter(|&&q| q != 0).count(), 0);
        assert_eq!(m.order.iter().filter(|&&x| x == 2).count(), 3);
    }

    #[test]
    fn smarts_daylight_flags() {
        let mol = parse_smiles("NN1C(=N)NNC1=N").unwrap();
        let m = CdkMol::new(&mol);
        let (aa, _) = smarts_daylight_aromaticity(&m);
        // Daylight treats the ring (exocyclic C=N) as aromatic
        assert!(aa[1] && aa[2]);
        let mol = parse_smiles("c1ccsc1").unwrap();
        let m = CdkMol::new(&mol);
        let (aa, ab) = smarts_daylight_aromaticity(&m);
        assert!(aa.iter().all(|&x| x) && ab.iter().all(|&x| x));
    }
}
