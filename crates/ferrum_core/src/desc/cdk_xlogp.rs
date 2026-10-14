//! XLogP (CDK 2.13 `XLogPDescriptor`, default parameters).
//!
//! CDK makes every hydrogen explicit, assigns one of 90 atom types per heavy
//! atom from local counts (neighbours, pi systems, aromatic rings of at least
//! six atoms from `Cycles.sssr`) and adds correction factors: hydrophobic carbons,
//! 1-3 halogen pairs, 1-5 sp2 oxygen pairs, internal hydrogen bonds (1-4 / 1-5
//! acceptor-donor pairs near rings) and three SMARTS patterns (C-terminal
//! amino acid, p-amino sulphonic acid, ortho oxygen pair). With the default
//! `checkAromaticity = false` the aromatic flags of the input (the reference
//! preparation's Daylight model) are used; the SMARTS patterns re-perceive
//! aromaticity (`SmartsPattern.prepare`).
//!
//! Hydrogens stay implicit here: every helper counts the implicit hydrogens
//! of an atom as single-bonded, non-aromatic, acyclic neighbours that CDK
//! appends after the heavy atoms.
//!
//! Key symbols: `fn xlogp`, `struct XView`.

use crate::desc::cdk_daylight::smarts_daylight_aromaticity;
use crate::desc::cdk_logp::{LogpView, cdk_mcb, edge_blocks};
use crate::mol::molecule::Molecule;
use smallvec::SmallVec;

/// Atom data of `XLogPDescriptor` with hydrogens made explicit.
struct XView<'a> {
    v: &'a LogpView,
    /// `IS_IN_AROMATIC_RING`: aromatic and in an SSSR ring of size >= 6.
    in_arom_ring: Vec<bool>,
    /// `PART_OF_RING_OF_SIZE`: smallest SSSR ring containing the atom (0: none).
    ring_size: Vec<usize>,
    /// Number of oxygens double-bonded to the atom.
    double_o: Vec<u8>,
    /// All atoms within two bonds are C or H.
    ok2: Vec<bool>,
}

/// CDK `XLogPDescriptor` value (default parameters).
pub(crate) fn xlogp(mol: &Molecule, v: &LogpView) -> f64 {
    let m = &v.m;
    let n = v.n();
    // Smallest ring per atom: the same for every minimum cycle basis, so the
    // ring perception's rings give it.
    let mut ring_size = vec![0usize; n];
    if let Some(info) = mol.ring_info() {
        for ring in &info.atom_rings {
            for &a in ring {
                let a = a as usize;
                if ring_size[a] == 0 || ring.len() < ring_size[a] {
                    ring_size[a] = ring.len();
                }
            }
        }
    }
    // Membership in a basis ring of size >= 6 (only aromatic atoms matter):
    // certain when the smallest ring has >= 6 atoms; otherwise it depends on
    // CDK's choice of basis in the atom's fused ring system.
    let mut big_ring: Vec<bool> = ring_size.iter().map(|&r| r >= 6).collect();
    if (0..n).any(|a| v.arom[a] && ring_size[a] > 0 && ring_size[a] < 6) {
        let (block, size) = edge_blocks(v, |b| m.ring_bond[b]);
        let mut verts: Vec<Vec<usize>> = vec![Vec::new(); size.len()];
        for b in 0..m.ends.len() {
            let k = block[b];
            if k != u32::MAX {
                let (x, y) = m.ends[b];
                verts[k as usize].extend([x as usize, y as usize]);
            }
        }
        for (k, vs) in verts.iter_mut().enumerate() {
            vs.sort_unstable();
            vs.dedup();
            // isolated cycles (in every basis) are covered by `ring_size`
            if (size[k] as usize) <= vs.len() || !vs.iter().any(|&a| v.arom[a] && !big_ring[a]) {
                continue;
            }
            for ring in cdk_mcb(v, vs) {
                if ring.len() >= 6 {
                    for a in ring {
                        big_ring[a] = true;
                    }
                }
            }
        }
    }
    let in_arom_ring = (0..n).map(|a| v.arom[a] && big_ring[a]).collect();
    let ch = |x: usize| matches!(m.z[x], 1 | 6);
    let ok1: Vec<bool> = (0..n)
        .map(|a| m.adj[a].iter().all(|&(w, _)| ch(w as usize)))
        .collect();
    let ok2: Vec<bool> = (0..n)
        .map(|a| ok1[a] && m.adj[a].iter().all(|&(w, _)| ok1[w as usize]))
        .collect();
    let double_o: Vec<u8> = (0..n)
        .map(|a| {
            m.adj[a]
                .iter()
                .filter(|&&(o, b)| m.z[o as usize] == 8 && m.order[b as usize] == 2)
                .count() as u8
        })
        .collect();
    let x = XView {
        v,
        in_arom_ring,
        ring_size,
        double_o,
        ok2,
    };
    x.calculate()
}

impl XView<'_> {
    #[inline]
    fn z(&self, a: usize) -> u8 {
        self.v.m.z[a]
    }

    #[inline]
    fn q(&self, a: usize) -> i8 {
        self.v.m.charge[a]
    }

    #[inline]
    fn arom(&self, a: usize) -> bool {
        self.v.arom[a]
    }

    #[inline]
    fn in_ring(&self, a: usize) -> bool {
        self.v.m.ring_atom[a]
    }

    /// Heavy (input) neighbours with bond order.
    #[inline]
    fn nbrs(&self, a: usize) -> impl Iterator<Item = (usize, u8)> + '_ {
        self.v.m.adj[a]
            .iter()
            .map(|&(w, b)| (w as usize, self.v.m.order[b as usize]))
    }

    /// `getConnectedBondsCount` with explicit hydrogens.
    #[inline]
    fn bond_count(&self, a: usize) -> usize {
        self.v.degree_all(a)
    }

    /// `getHydrogenCount`.
    #[inline]
    fn hs(&self, a: usize) -> usize {
        self.v.total_h(a)
    }

    /// `getMaximumBondOrder` (0 without bonds).
    fn max_order(&self, a: usize) -> u8 {
        let heavy = self.nbrs(a).map(|(_, o)| o).max().unwrap_or(0);
        if self.v.m.hcount[a] > 0 {
            heavy.max(1)
        } else {
            heavy
        }
    }

    fn bond_order_sum(&self, a: usize) -> usize {
        self.nbrs(a).map(|(_, o)| o as usize).sum::<usize>() + self.v.m.hcount[a] as usize
    }

    fn atom_type_x_count(&self, a: usize) -> usize {
        self.nbrs(a)
            .filter(|&(w, o)| matches!(self.z(w), 7 | 8) && !self.in_arom_ring[w] && o != 2)
            .count()
    }

    fn aromatic_carbons(&self, a: usize) -> usize {
        self.nbrs(a)
            .filter(|&(w, _)| self.z(w) == 6 && self.arom(w))
            .count()
    }

    fn carbons(&self, a: usize) -> usize {
        self.nbrs(a)
            .filter(|&(w, _)| self.z(w) == 6 && !self.arom(w))
            .count()
    }

    fn oxygens(&self, a: usize) -> usize {
        self.nbrs(a)
            .filter(|&(w, _)| self.z(w) == 8 && !self.arom(w))
            .count()
    }

    fn double_bonded_carbons(&self, a: usize) -> usize {
        self.nbrs(a)
            .filter(|&(w, o)| self.z(w) == 6 && o == 2)
            .count()
    }

    fn double_bonded_oxygens(&self, a: usize) -> usize {
        let charged = self.q(a) >= 1;
        let mut c = 0;
        for (w, o) in self.nbrs(a) {
            if self.z(w) == 8 {
                if charged && self.q(w) == -1 && o == 1 {
                    c += 1;
                }
                if !self.arom(w) && o == 2 {
                    c += 1;
                }
            }
        }
        c
    }

    fn double_bonded_sulfurs(&self, a: usize) -> usize {
        let mut c = 0;
        for (w, o) in self.nbrs(a) {
            if self.z(w) == 16 {
                if self.q(a) == 1 && self.q(w) == -1 {
                    c += 1;
                }
                if !self.arom(w) && o == 2 {
                    c += 1;
                }
            }
        }
        c
    }

    fn double_bonded_nitrogens(&self, a: usize) -> usize {
        self.nbrs(a)
            .filter(|&(w, o)| self.z(w) == 7 && !self.arom(w) && o == 2)
            .count()
    }

    fn aromatic_nitrogens(&self, a: usize) -> usize {
        self.nbrs(a)
            .filter(|&(w, _)| self.z(w) == 7 && self.in_arom_ring[w])
            .count()
    }

    /// `getPiSystemsCount`: non-single bonds of the neighbours (other than P
    /// and S) that do not lead back to the atom.
    fn pi_systems(&self, a: usize) -> usize {
        let mut c = 0;
        for &(w, _) in &self.v.m.adj[a] {
            let w = w as usize;
            if matches!(self.z(w), 15 | 16) {
                continue;
            }
            c += self.v.m.adj[w]
                .iter()
                .filter(|&&(x, b)| x as usize != a && self.v.m.order[b as usize] != 1)
                .count();
        }
        c
    }

    /// `getPresenceOfHydroxy` (first neighbour in CDK order).
    fn presence_of_hydroxy(&self, a: usize) -> bool {
        let Some(&(c, _)) = self.v.m.adj[a].first() else {
            // only hydrogens: a hydrogen is not a carbon
            return false;
        };
        let c = c as usize;
        if self.z(c) != 6 {
            return false;
        }
        for (o, ord) in self.nbrs(c) {
            if self.z(o) == 8 && ord == 1 {
                return !(self.bond_count(o) > 1 && self.hs(o) == 0);
            }
        }
        false
    }

    fn presence_of_nitro(&self, a: usize) -> bool {
        self.nbrs(a)
            .any(|(w, _)| self.z(w) == 7 && self.nbrs(w).any(|(o, ord)| self.z(o) == 8 && ord == 2))
    }

    fn presence_of_sulfat(&self, a: usize) -> bool {
        self.nbrs(a)
            .any(|(w, _)| self.z(w) == 16 && self.oxygens(w) >= 2 && self.bond_count(w) == 4)
    }

    /// `getPresenceOfCarbonil`: C=O double bonds on carbon neighbours.
    fn carbonyls(&self, a: usize) -> usize {
        self.nbrs(a)
            .filter(|&(w, _)| self.z(w) == 6)
            .map(|(w, _)| self.double_o[w] as usize)
            .sum()
    }

    /// `getIfCarbonIsHydrophobic`: every atom reached by a walk of one to
    /// three bonds is C or H (implicit hydrogens only lead back to checked atoms).
    fn hydrophobic(&self, a: usize) -> bool {
        self.bond_count(a) > 0
            && self
                .nbrs(a)
                .all(|(w, _)| matches!(self.z(w), 1 | 6) && self.ok2[w])
    }

    fn halogens(&self, a: usize) -> usize {
        self.nbrs(a)
            .filter(|&(w, _)| matches!(self.z(w), 9 | 17 | 35 | 53))
            .count()
    }

    /// `checkRingLink`: the atom or a neighbour is in a ring.
    fn ring_link(&self, a: usize) -> bool {
        self.in_ring(a) || self.nbrs(a).any(|(w, _)| self.in_ring(w))
    }

    fn calculate(&self) -> f64 {
        let n = self.v.n();
        let mut xlogp = 0.0f64;
        let mut acceptors: SmallVec<[usize; 16]> = SmallVec::new();
        let mut donors: SmallVec<[usize; 16]> = SmallVec::new();
        for i in 0..n {
            let hs = self.hs(i);
            let bc = self.bond_count(i);
            let max_order = self.max_order(i);
            match self.z(i) {
                6 => {
                    if bc == 2 {
                        if hs >= 1 {
                            xlogp += 0.209;
                        } else if max_order == 2 {
                            xlogp += 2.073;
                        } else if max_order == 3 {
                            xlogp += 0.33;
                        }
                    }
                    if bc == 3 {
                        if self.in_arom_ring[i] {
                            if self.aromatic_carbons(i) >= 2 && self.aromatic_nitrogens(i) == 0 {
                                if hs == 0 {
                                    if self.atom_type_x_count(i) == 0 {
                                        xlogp += 0.296;
                                    } else {
                                        xlogp -= 0.151;
                                    }
                                } else {
                                    xlogp += 0.337;
                                }
                            } else if self.aromatic_nitrogens(i) >= 1 {
                                if hs == 0 {
                                    if self.atom_type_x_count(i) == 0 {
                                        xlogp += 0.174;
                                    } else {
                                        xlogp += 0.366;
                                    }
                                } else if hs == 1 {
                                    xlogp += 0.126;
                                }
                            }
                        } else {
                            let x = self.atom_type_x_count(i);
                            let pi = self.pi_systems(i);
                            if hs == 0 {
                                if x == 0 {
                                    xlogp += if pi <= 1 { 0.05 } else { 0.013 };
                                } else if x == 1 {
                                    xlogp -= if pi == 0 { 0.03 } else { 0.027 };
                                } else if x == 2 {
                                    if pi == 0 {
                                        xlogp += 0.005;
                                    } else {
                                        xlogp -= 0.315;
                                    }
                                }
                            }
                            if hs == 1 {
                                if x == 0 {
                                    if pi == 0 {
                                        xlogp += 0.466;
                                    }
                                    if pi == 1 {
                                        xlogp += 0.136;
                                    }
                                } else {
                                    if pi == 0 {
                                        xlogp += 0.001;
                                    }
                                    if pi == 1 {
                                        xlogp -= 0.31;
                                    }
                                }
                            }
                            if hs == 2 {
                                xlogp += 0.42;
                            }
                            if self.hydrophobic(i) {
                                xlogp += 0.211;
                            }
                        }
                    }
                    if bc == 4 {
                        let x = self.atom_type_x_count(i);
                        let pi = self.pi_systems(i);
                        if hs == 0 {
                            if x == 0 {
                                match pi {
                                    0 => xlogp -= 0.006,
                                    1 => xlogp -= 0.57,
                                    _ => xlogp -= 0.317,
                                }
                            } else if pi == 0 {
                                xlogp -= 0.316;
                            } else {
                                xlogp -= 0.723;
                            }
                        }
                        if hs == 1 {
                            if x == 0 {
                                match pi {
                                    0 => xlogp += 0.127,
                                    1 => xlogp -= 0.243,
                                    _ => xlogp -= 0.499,
                                }
                            } else {
                                match pi {
                                    0 => xlogp -= 0.205,
                                    1 => xlogp -= 0.305,
                                    _ => xlogp -= 0.709,
                                }
                            }
                        }
                        if hs == 2 {
                            if x == 0 {
                                match pi {
                                    0 => xlogp += 0.358,
                                    1 => xlogp -= 0.008,
                                    2 => xlogp -= 0.185,
                                    _ => {}
                                }
                            } else {
                                match pi {
                                    0 => xlogp -= 0.137,
                                    1 => xlogp -= 0.303,
                                    2 => xlogp -= 0.815,
                                    _ => {}
                                }
                            }
                        }
                        if hs > 2 {
                            if x == 0 {
                                match pi {
                                    0 => xlogp += 0.528,
                                    1 => xlogp += 0.267,
                                    _ => {}
                                }
                            } else {
                                xlogp -= 0.032;
                            }
                        }
                        if self.hydrophobic(i) {
                            xlogp += 0.211;
                        }
                    }
                }
                7 => self.nitrogen(i, hs, bc, max_order, &mut xlogp, &mut donors),
                8 => {
                    if bc == 1 && max_order == 2 {
                        xlogp -= 0.399;
                        if !self.presence_of_hydroxy(i) {
                            acceptors.push(i);
                        }
                    } else if (bc == 1
                        && hs == 0
                        && (self.presence_of_nitro(i) || self.carbonyls(i) == 1))
                        || self.presence_of_sulfat(i)
                    {
                        xlogp -= 0.399;
                        if !self.presence_of_hydroxy(i) {
                            acceptors.push(i);
                        }
                    } else if bc >= 1 {
                        let x = self.atom_type_x_count(i);
                        if hs == 0 && bc == 2 {
                            if x == 0 {
                                let pi = self.pi_systems(i);
                                if pi == 0 {
                                    xlogp += 0.084;
                                }
                                if pi > 0 {
                                    xlogp += 0.435;
                                }
                            } else if x == 1 {
                                xlogp += 0.105;
                            }
                        } else if x == 0 {
                            let pi = self.pi_systems(i);
                            if pi == 0 {
                                xlogp -= 0.467;
                                donors.push(i);
                                acceptors.push(i);
                            }
                            if pi == 1 {
                                xlogp += 0.082;
                                donors.push(i);
                                acceptors.push(i);
                            }
                        } else if x == 1 {
                            xlogp -= 0.522;
                            donors.push(i);
                            acceptors.push(i);
                        }
                    }
                }
                16 => {
                    if (bc == 1 && max_order == 2) || (bc == 1 && self.q(i) == -1) {
                        xlogp -= 0.148;
                    } else if bc == 2 {
                        xlogp += if hs == 0 { 0.255 } else { 0.419 };
                    } else if bc == 3 {
                        if self.oxygens(i) >= 1 {
                            xlogp -= 1.375;
                        }
                    } else if bc == 4 && self.double_bonded_oxygens(i) >= 2 {
                        xlogp -= 0.168;
                    }
                }
                15 => {
                    if self.double_bonded_sulfurs(i) >= 1 && bc >= 4 {
                        xlogp += 1.253;
                    } else if self.oxygens(i) >= 1
                        || (self.double_bonded_oxygens(i) == 1 && bc >= 4)
                    {
                        xlogp -= 0.447;
                    }
                }
                9 | 17 | 35 | 53 => {
                    let pi = self.pi_systems(i);
                    match (self.z(i), pi) {
                        (9, 0) => xlogp += 0.375,
                        (9, 1) => xlogp += 0.202,
                        (17, 0) => xlogp += 0.512,
                        (17, _) => xlogp += 0.663,
                        (35, 0) => xlogp += 0.85,
                        (35, 1) => xlogp += 0.839,
                        (53, 0) => xlogp += 1.05,
                        (53, 1) => xlogp += 1.109,
                        _ => {}
                    }
                }
                _ => {}
            }
            // halogen 1-3 pairs
            match self.halogens(i) {
                2 => xlogp += 0.137,
                3 => xlogp += 3.0 * 0.137,
                4 => xlogp += 6.0 * 0.137,
                _ => {}
            }
            // sp2 oxygen 1-5 pair
            if self.carbonyls(i) == 2 && !self.in_ring(i) {
                xlogp += 0.580;
            }
            // the implicit hydrogens on this atom (appended by CDK): a hydrogen
            // on a carbon carrying two C=O sees a 1-5 pair
            let h = self.v.m.hcount[i];
            if h > 0 && self.z(i) == 6 && self.double_o[i] == 2 {
                xlogp += 0.580 * h as f64;
            }
        }
        xlogp += self.hbond_pairs(&acceptors, &donors);
        xlogp += self.smarts_corrections();
        xlogp
    }

    fn nitrogen(
        &self,
        i: usize,
        hs: usize,
        bc: usize,
        max_order: u8,
        xlogp: &mut f64,
        donors: &mut SmallVec<[usize; 16]>,
    ) {
        if self.bond_order_sum(i) >= 3 && self.oxygens(i) >= 2 && max_order == 2 {
            *xlogp += 1.178;
            return;
        }
        if self.carbonyls(i) >= 1 {
            // amidic nitrogen
            let x = self.atom_type_x_count(i);
            if hs == 0 {
                if x == 0 {
                    *xlogp += 0.078;
                }
                if x == 1 {
                    *xlogp -= 0.118;
                }
            }
            if hs == 1 {
                *xlogp -= if x == 0 { 0.096 } else { 0.044 };
                donors.push(i);
            }
            if hs == 2 {
                *xlogp -= 0.646;
                donors.push(i);
            }
            return;
        }
        match bc {
            1 => {
                if self.carbons(i) == 1 {
                    *xlogp -= 0.566;
                }
            }
            2 => {
                if self.in_arom_ring[i] {
                    *xlogp -= 0.493;
                } else {
                    let dc = self.double_bonded_carbons(i);
                    if dc == 0 {
                        let dn = self.double_bonded_nitrogens(i);
                        if dn == 0 && self.double_bonded_oxygens(i) == 1 {
                            *xlogp += 0.427;
                        }
                        if dn == 1 {
                            match self.atom_type_x_count(i) {
                                0 => *xlogp += 0.536,
                                1 => *xlogp -= 0.597,
                                _ => {}
                            }
                        }
                    } else if dc == 1 {
                        let x = self.atom_type_x_count(i);
                        let pi = self.pi_systems(i);
                        if x == 0 {
                            match pi {
                                0 => *xlogp += 0.007,
                                1 => *xlogp -= 0.275,
                                _ => {}
                            }
                        } else if x == 1 {
                            match pi {
                                0 => *xlogp += 0.366,
                                1 => *xlogp += 0.251,
                                _ => {}
                            }
                        }
                    }
                }
            }
            3 => {
                let x = self.atom_type_x_count(i);
                let pi = self.pi_systems(i);
                let ringed = self.in_ring(i) && self.ring_size[i] > 3;
                if hs == 0 {
                    if self.arom(i) || (ringed && pi >= 1) {
                        if x == 0 {
                            *xlogp += 0.881;
                        } else {
                            *xlogp -= 0.01;
                        }
                    } else if x == 0 {
                        if pi == 0 {
                            *xlogp += 0.159;
                        }
                        if pi > 0 {
                            *xlogp += 0.761;
                        }
                    } else {
                        *xlogp -= 0.239;
                    }
                } else if hs == 1 {
                    if x == 0 {
                        if self.arom(i) || (ringed && pi >= 2) {
                            *xlogp += 0.545;
                            donors.push(i);
                        } else {
                            if pi == 0 {
                                *xlogp -= 0.112;
                                donors.push(i);
                            }
                            if pi > 0 {
                                *xlogp += 0.166;
                                donors.push(i);
                            }
                        }
                    } else if self.in_ring(i) {
                        *xlogp += 0.153;
                        donors.push(i);
                    } else {
                        *xlogp += 0.324;
                        donors.push(i);
                    }
                } else if hs == 2 {
                    if x == 0 {
                        if pi == 0 {
                            *xlogp -= 0.534;
                            donors.push(i);
                        }
                        if pi == 1 {
                            *xlogp -= 0.329;
                            donors.push(i);
                        }
                    } else {
                        *xlogp -= 1.082;
                        donors.push(i);
                    }
                }
            }
            _ => {}
        }
    }

    /// Internal hydrogen bonds: acceptor-donor pairs 1-4 apart when both
    /// touch a ring, 1-5 apart when one does (each pair counted once).
    fn hbond_pairs(&self, acceptors: &[usize], donors: &[usize]) -> f64 {
        if acceptors.is_empty() || donors.is_empty() {
            return 0.0;
        }
        let donor_ring: SmallVec<[bool; 16]> = donors.iter().map(|&d| self.ring_link(d)).collect();
        let any_donor_ring = donor_ring.iter().any(|&r| r);
        let n = self.v.n();
        let mut dist = vec![u8::MAX; n];
        let mut queue: Vec<usize> = Vec::new();
        let mut seen: SmallVec<[(usize, usize); 8]> = SmallVec::new();
        let mut sum = 0.0;
        for &acc in acceptors {
            let ra = self.ring_link(acc);
            if !ra && !any_donor_ring {
                continue;
            }
            // breadth-first distances up to 4 bonds
            for &x in &queue {
                dist[x] = u8::MAX;
            }
            queue.clear();
            dist[acc] = 0;
            queue.push(acc);
            let mut head = 0;
            while head < queue.len() {
                let u = queue[head];
                head += 1;
                if dist[u] == 4 {
                    continue;
                }
                for &(w, _) in &self.v.m.adj[u] {
                    let w = w as usize;
                    if dist[w] == u8::MAX {
                        dist[w] = dist[u] + 1;
                        queue.push(w);
                    }
                }
            }
            for (k, &don) in donors.iter().enumerate() {
                let rd = donor_ring[k];
                if !(ra || rd) {
                    continue;
                }
                let want = if ra && rd { 3 } else { 4 };
                let key = (acc.min(don), acc.max(don));
                if dist[don] == want && !seen.contains(&key) {
                    sum += 0.429;
                    seen.push(key);
                }
            }
        }
        sum
    }

    /// The SMARTS corrections (`C_TERMINAL_AMINO_ACID`, `P_AMINO_SULPHONIC_ACID`,
    /// `ORTHOPAIR`) on `SmartsPattern`'s Daylight aromaticity.
    fn smarts_corrections(&self) -> f64 {
        let m = &self.v.m;
        let mut flags: Option<(Vec<bool>, Vec<bool>)> = None;
        let mut sum = 0.0;
        // quick pre-checks on elements before perceiving aromaticity
        let has = |z: u8| m.z.contains(&z);
        // the pattern without its aromaticity constraints first
        if has(7)
            && has(8)
            && self.c_terminal_amino_acid(&vec![false; m.n()], &vec![false; m.ends.len()])
        {
            let (sa, sb) = flags.get_or_insert_with(|| smarts_daylight_aromaticity(m));
            if self.c_terminal_amino_acid(sa, sb) {
                sum -= 2.166;
            }
        }
        if has(16) && has(7) && self.sulphonyl_candidate() {
            let (sa, sb) = flags.get_or_insert_with(|| smarts_daylight_aromaticity(m));
            if self.p_amino_sulphonic_acid(sa, sb) {
                sum -= 0.501;
            }
        }
        if self.ortho_pair_candidate() {
            let (sa, sb) = flags.get_or_insert_with(|| smarts_daylight_aromaticity(m));
            if self.ortho_pair(sa, sb) {
                sum -= 0.268;
            }
        }
        sum
    }

    #[inline]
    fn single_or_arom(&self, b: usize, sb: &[bool]) -> bool {
        sb[b] || self.v.m.order[b] == 1
    }

    #[inline]
    fn aliphatic_order(&self, b: usize, sb: &[bool], ord: u8) -> bool {
        !sb[b] && self.v.m.order[b] == ord
    }

    /// Cheap test for an S(=O)(=O) bonded to a ring carbon.
    fn sulphonyl_candidate(&self) -> bool {
        let m = &self.v.m;
        (0..self.v.n()).any(|s| {
            m.z[s] == 16
                && m.adj[s]
                    .iter()
                    .filter(|&&(o, b)| m.z[o as usize] == 8 && m.order[b as usize] == 2)
                    .count()
                    >= 2
                && m.adj[s]
                    .iter()
                    .any(|&(c, _)| m.z[c as usize] == 6 && m.ring_atom[c as usize])
        })
    }

    /// Cheap test for two bonded ring carbons that each carry a singly bonded
    /// oxygen (aromatic atoms are ring atoms).
    fn ortho_pair_candidate(&self) -> bool {
        let m = &self.v.m;
        let o_on = |c: usize| {
            m.adj[c]
                .iter()
                .filter(|&&(o, b)| m.z[o as usize] == 8 && m.order[b as usize] == 1)
                .count()
        };
        (0..m.ends.len()).any(|b| {
            let (x, y) = (m.ends[b].0 as usize, m.ends[b].1 as usize);
            m.ring_atom[x]
                && m.ring_atom[y]
                && m.z[x] == 6
                && m.z[y] == 6
                && o_on(x) >= 1
                && o_on(y) >= 1
        })
    }

    /// `N!@C-C(=O)-[O;X2H1+0,X1H0-]`.
    fn c_terminal_amino_acid(&self, sa: &[bool], sb: &[bool]) -> bool {
        let m = &self.v.m;
        let aliphatic = |a: usize, z: u8| m.z[a] == z && !sa[a];
        for c2 in 0..self.v.n() {
            if !aliphatic(c2, 6) {
                continue;
            }
            let mut has_o1 = SmallVec::<[usize; 2]>::new();
            let mut has_o2 = SmallVec::<[usize; 2]>::new();
            for &(o, b) in &m.adj[c2] {
                let (o, b) = (o as usize, b as usize);
                if !aliphatic(o, 8) {
                    continue;
                }
                if self.aliphatic_order(b, sb, 2) {
                    has_o1.push(o);
                }
                if self.aliphatic_order(b, sb, 1) {
                    let x = self.v.degree_all(o);
                    let h = self.v.total_h(o);
                    let q = m.charge[o];
                    if (x == 2 && h == 1 && q == 0) || (x == 1 && h == 0 && q == -1) {
                        has_o2.push(o);
                    }
                }
            }
            if !has_o1.iter().any(|o1| has_o2.iter().any(|o2| o1 != o2)) {
                continue;
            }
            for &(c1, b12) in &m.adj[c2] {
                let (c1, b12) = (c1 as usize, b12 as usize);
                if !aliphatic(c1, 6) || !self.aliphatic_order(b12, sb, 1) {
                    continue;
                }
                if m.adj[c1]
                    .iter()
                    .any(|&(nn, b)| aliphatic(nn as usize, 7) && !m.ring_bond[b as usize])
                {
                    return true;
                }
            }
        }
        false
    }

    /// `CS(=O)(=O)c1ccc(N)cc1`.
    fn p_amino_sulphonic_acid(&self, sa: &[bool], sb: &[bool]) -> bool {
        let m = &self.v.m;
        let aliphatic = |a: usize, z: u8| m.z[a] == z && !sa[a];
        let arom_c = |a: usize| m.z[a] == 6 && sa[a];
        for s in 0..self.v.n() {
            if !aliphatic(s, 16) {
                continue;
            }
            let n_o = m.adj[s]
                .iter()
                .filter(|&&(o, b)| {
                    aliphatic(o as usize, 8) && self.aliphatic_order(b as usize, sb, 2)
                })
                .count();
            if n_o < 2 {
                continue;
            }
            if !m.adj[s]
                .iter()
                .any(|&(c, b)| aliphatic(c as usize, 6) && self.single_or_arom(b as usize, sb))
            {
                continue;
            }
            for &(a0, b0) in &m.adj[s] {
                let a0 = a0 as usize;
                if !arom_c(a0) || !self.single_or_arom(b0 as usize, sb) {
                    continue;
                }
                // 6-cycle a0..a5 of aromatic carbons, nitrogen on a3
                let mut path = [a0, 0, 0, 0, 0, 0];
                if self.ring_path(&mut path, 1, sa, sb) {
                    return true;
                }
            }
        }
        false
    }

    fn ring_path(&self, path: &mut [usize; 6], k: usize, sa: &[bool], sb: &[bool]) -> bool {
        let m = &self.v.m;
        let prev = path[k - 1];
        for &(w, b) in &m.adj[prev] {
            let w = w as usize;
            if m.z[w] != 6
                || !sa[w]
                || !self.single_or_arom(b as usize, sb)
                || path[..k].contains(&w)
            {
                continue;
            }
            path[k] = w;
            if k == 3
                && !m.adj[w].iter().any(|&(nn, bn)| {
                    m.z[nn as usize] == 7
                        && !sa[nn as usize]
                        && self.single_or_arom(bn as usize, sb)
                })
            {
                continue;
            }
            if k == 5 {
                if m.bond(w, path[0])
                    .is_some_and(|bc| self.single_or_arom(bc, sb))
                {
                    return true;
                }
                continue;
            }
            if self.ring_path(path, k + 1, sa, sb) {
                return true;
            }
        }
        false
    }

    /// `OccO`.
    fn ortho_pair(&self, sa: &[bool], sb: &[bool]) -> bool {
        let m = &self.v.m;
        let o_on = |c: usize, skip: usize| {
            m.adj[c].iter().any(|&(o, b)| {
                let o = o as usize;
                o != skip && m.z[o] == 8 && !sa[o] && self.single_or_arom(b as usize, sb)
            })
        };
        for c1 in 0..self.v.n() {
            if m.z[c1] != 6 || !sa[c1] {
                continue;
            }
            for &(o1, b1) in &m.adj[c1] {
                let o1 = o1 as usize;
                if m.z[o1] != 8 || sa[o1] || !self.single_or_arom(b1 as usize, sb) {
                    continue;
                }
                for &(c2, b2) in &m.adj[c1] {
                    let c2 = c2 as usize;
                    if m.z[c2] == 6
                        && sa[c2]
                        && self.single_or_arom(b2 as usize, sb)
                        && o_on(c2, o1)
                    {
                        return true;
                    }
                }
            }
        }
        false
    }
}
