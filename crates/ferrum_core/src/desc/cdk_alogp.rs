//! Ghose-Crippen ALogP and molar refractivity (CDK 2.13 `ALOGPDescriptor`).
//!
//! CDK types every atom with `EStateAtomTypeMatcher` (fragment strings such as
//! `SsCH3`, `SaaCH`, `SdssC`), refines them into the 120 Ghose-Crippen groups
//! with graph tests and sums the group coefficients. The reference molecule
//! keeps its hydrogens implicit; hydrogen groups are counted from the heavy
//! atom they sit on (`getHAtomType`). "Same aromatic ring" tests use the ring
//! set of `AllRingsFinder` (all simple cycles); they are answered exactly with
//! biconnected components:
//!
//! - EState matcher: some cycle whose atoms are all aromatic contains both
//!   atoms - for bonded atoms, the bond is not a bridge of the subgraph of
//!   aromatic atoms;
//! - ALOGP groups 25-35/43/44: some cycle whose bonds are all aromatic
//!   contains both atoms - they share a block of the aromatic-bond subgraph.
//!
//! `AllRingsFinder` gives up (CDK then reports NaN) when its path graph
//! exceeds the `PubChem_99` threshold; that test is reproduced.
//!
//! Key symbols: `fn alogp`, `fn h_atom_type`, `fn all_rings_tractable`.

use crate::desc::cdk_logp::{LogpView, edge_blocks};
use crate::mol::elements::ELEMENT_SYMBOLS;
use crate::mol::molecule::Molecule;
use smallvec::SmallVec;

/// `ALOGPDescriptor.FRAGVAL` (Ghose et al., 1998), index 1..=120.
const FRAGVAL: [f64; 121] = [
    0.0, -1.5603, -1.012, -0.6681, -0.3698, -1.788, -1.2486, -1.0305, -0.6805, -0.3858, 0.7555,
    -0.2849, 0.02, 0.7894, 1.6422, -0.7866, -0.3962, 0.0383, -0.8051, -0.2129, 0.2432, 0.4697,
    0.2952, 0.0, -0.3251, 0.1492, 0.1539, 0.0005, 0.2361, 0.3514, 0.1814, 0.0901, 0.5142, -0.3723,
    0.2813, 0.1191, -0.132, -0.0244, -0.2405, -0.0909, -0.1002, 0.4182, -0.2147, -0.0009, 0.1388,
    0.0, 0.7341, 0.6301, 0.518, -0.0371, -0.1036, 0.5234, 0.6666, 0.5372, 0.6338, 0.362, -0.3567,
    -0.0127, -0.0233, -0.1541, 0.0324, 1.052, -0.7941, 0.4165, 0.6601, 0.0, -0.5427, -0.3168,
    0.0132, -0.3883, -0.0389, 0.1087, -0.5113, 0.1259, 0.1349, -0.1624, -2.0585, -1.915, 0.4208,
    -1.4439, 0.0, 0.4797, 0.2358, 0.1029, 0.3566, 0.1988, 0.7443, 0.5337, 0.2996, 0.8155, 0.4856,
    0.8888, 0.7452, 0.5034, 0.8995, 0.5946, 1.4201, 1.1472, 0.0, 0.7293, 0.7173, 0.0, -2.6737,
    -2.4178, -3.1121, 0.0, 0.6146, 0.5906, 0.8758, -0.4979, -0.3786, 1.5188, 1.0255, 0.0, 0.0, 0.0,
    -0.9359, -0.1726, -0.7966, 0.6705, -0.4801,
];

/// `ALOGPDescriptor.REFRACVAL` (Viswanadhan et al., 1989), index 1..=120.
const REFRACVAL: [f64; 121] = [
    0.0, 2.968, 2.9116, 2.8028, 2.6205, 3.015, 2.9244, 2.6329, 2.504, 2.377, 2.5559, 2.303, 2.3006,
    2.9627, 2.3038, 3.2001, 4.2654, 3.9392, 3.6005, 4.487, 3.2001, 3.4825, 4.2817, 3.9556, 3.4491,
    3.8821, 3.7593, 2.5009, 2.5, 3.0627, 2.5009, 0.0, 2.6632, 3.4671, 3.6842, 2.9372, 4.019, 4.777,
    3.9031, 3.9964, 3.4986, 3.4997, 2.7784, 2.6267, 2.5, 0.0, 0.8447, 0.8939, 0.8005, 0.832, 0.8,
    0.8188, 0.9215, 0.9769, 0.7701, 0.0, 1.7646, 1.4778, 1.4429, 1.6191, 1.3502, 1.945, 0.0, 0.0,
    11.1366, 13.1149, 2.6221, 2.5, 2.898, 3.6841, 4.2808, 3.6189, 2.5, 2.7956, 2.7, 4.2063, 4.0184,
    3.0009, 4.7142, 0.0, 0.0, 0.8725, 1.1837, 1.1573, 0.8001, 1.5013, 5.6156, 6.1022, 5.9921,
    5.3885, 6.1363, 8.5991, 8.9188, 8.8006, 8.2065, 8.7352, 13.9462, 14.0792, 14.073, 12.9918,
    13.3408, 0.0, 0.0, 0.0, 0.0, 0.0, 7.8916, 7.7935, 9.4338, 7.7223, 5.7558, 0.0, 0.0, 0.0, 0.0,
    0.0, 5.5306, 5.5152, 6.836, 10.0101, 5.2806,
];

/// `AllRingsFinder.Threshold.PubChem_99`.
const ALL_RINGS_THRESHOLD: usize = 684;

/// EState fragment string of an atom (`EStateAtomTypeMatcher`).
type Frag = SmallVec<[u8; 24]>;

/// `ALOGPDescriptor.isHetero`: B, N, O, F, Si, P, S, Cl, Se, Br, I.
#[inline]
fn is_hetero(z: u8) -> bool {
    matches!(z, 5 | 7 | 8 | 9 | 14 | 15 | 16 | 17 | 34 | 35 | 53)
}

/// `AtomicProperties.getNormalizedElectronegativity(symbol) > 1`; `None` for
/// elements missing from CDK's `whim_weights.txt` (CDK then throws).
fn electronegative(z: u8) -> Option<bool> {
    match z {
        7 | 8 | 9 | 16 | 17 | 33 | 34 | 35 | 53 => Some(true),
        1 | 5 | 6 | 13 | 14 | 15 | 24..=32 | 42 | 47..=52 | 64 | 78..=83 => Some(false),
        _ => None,
    }
}

/// Error raised by a group test (CDK reports NaN for the whole descriptor).
struct Failed;

struct Alogp<'a> {
    v: &'a LogpView,
    frag: Vec<Frag>,
    /// Bond lies on a cycle of aromatic atoms (EState "same aromatic ring").
    estate_ring_bond: Vec<bool>,
    /// Blocks (with a cycle) of the aromatic-bond subgraph each atom is in.
    arom_blocks: Vec<SmallVec<[u32; 2]>>,
    frags: [i32; 121],
}

/// CDK `ALOGPDescriptor` values `[ALogP, ALogp2, AMR]` (NaN when CDK fails).
pub(crate) fn alogp(mol: &Molecule, v: &LogpView) -> [f64; 3] {
    let m = &v.m;
    let n = v.n();
    if !all_rings_tractable(mol, v) {
        return [f64::NAN; 3];
    }
    let nb = m.ends.len();
    // Aromatic bonds lie on an all-aromatic (Hückel) cycle and acyclic bonds
    // on none; only ring bonds between aromatic atoms that are not aromatic
    // themselves need the block decomposition.
    let ambiguous = (0..nb).any(|b| {
        let (s, t) = m.ends[b];
        m.ring_bond[b] && !v.arom_bond[b] && v.arom[s as usize] && v.arom[t as usize]
    });
    let mut estate_ring_bond: Vec<bool> = v.arom_bond.clone();
    let mut arom_blocks: Vec<SmallVec<[u32; 2]>> = Vec::new();
    if ambiguous {
        let (blk, size) = edge_blocks(v, |b| {
            let (s, t) = m.ends[b];
            v.arom[s as usize] && v.arom[t as usize]
        });
        for b in 0..nb {
            estate_ring_bond[b] = blk[b] != u32::MAX && size[blk[b] as usize] >= 2;
        }
        let (blk, size) = edge_blocks(v, |b| v.arom_bond[b]);
        arom_blocks = vec![SmallVec::new(); n];
        for b in 0..nb {
            let k = blk[b];
            if k == u32::MAX || size[k as usize] < 2 {
                continue;
            }
            let (s, t) = m.ends[b];
            for x in [s as usize, t as usize] {
                if !arom_blocks[x].contains(&k) {
                    arom_blocks[x].push(k);
                }
            }
        }
    }
    let mut st = Alogp {
        v,
        frag: Vec::with_capacity(n),
        estate_ring_bond,
        arom_blocks,
        frags: [0; 121],
    };
    for a in 0..n {
        let f = st.estate_fragment(a);
        st.frag.push(f);
    }
    for a in 0..n {
        if st.atom_groups(a).is_err() {
            return [f64::NAN; 3];
        }
    }
    let mut alogp = 0.0;
    let mut amr = 0.0;
    for i in 1..=120 {
        alogp += FRAGVAL[i] * st.frags[i] as f64;
        amr += REFRACVAL[i] * st.frags[i] as f64;
    }
    [alogp, alogp * alogp, amr]
}

impl<'a> Alogp<'a> {
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
    fn nbrs(&self, a: usize) -> impl Iterator<Item = (usize, usize)> + use<'a> {
        let v: &'a LogpView = self.v;
        v.m.adj[a].iter().map(|&(w, b)| (w as usize, b as usize))
    }

    #[inline]
    fn nbr_at(&self, a: usize, k: usize) -> usize {
        self.v.m.adj[a][k].0 as usize
    }

    #[inline]
    fn order(&self, b: usize) -> u8 {
        self.v.m.order[b]
    }

    #[inline]
    fn has_double(&self, a: usize) -> bool {
        self.nbrs(a).any(|(_, b)| self.order(b) == 2)
    }

    #[inline]
    fn implicit_h(&self, a: usize) -> i32 {
        self.v.m.hcount[a] as i32
    }

    #[inline]
    fn is(&self, a: usize, f: &str) -> bool {
        self.frag[a].as_slice() == f.as_bytes()
    }

    /// `EStateAtomTypeMatcher.findMatchingAtomType`.
    fn estate_fragment(&self, a: usize) -> Frag {
        let mut nh = self.implicit_h(a);
        let mut single = nh;
        let (mut double, mut triple, mut aromatic) = (0, 0, 0);
        for (w, b) in self.nbrs(a) {
            if self.z(w) == 1 {
                nh += 1;
            }
            if self.arom(a) && self.arom(w) && self.estate_ring_bond[b] {
                aromatic += 1;
            } else {
                match self.order(b) {
                    1 => single += 1,
                    2 => double += 1,
                    3 => triple += 1,
                    _ => {}
                }
            }
        }
        single -= nh;
        let mut f = Frag::new();
        f.push(b'S');
        for _ in 0..triple {
            f.push(b't');
        }
        for _ in 0..double {
            f.push(b'd');
        }
        for _ in 0..single {
            f.push(b's');
        }
        for _ in 0..aromatic {
            f.push(b'a');
        }
        f.extend_from_slice(
            ELEMENT_SYMBOLS
                .get(self.z(a) as usize)
                .copied()
                .unwrap_or("*")
                .as_bytes(),
        );
        match self.q(a) {
            1 => f.push(b'p'),
            -1 => f.push(b'm'),
            _ => {}
        }
        if nh == 1 {
            f.push(b'H');
        } else if nh > 1 {
            f.push(b'H');
            if nh >= 10 {
                f.push(b'0' + (nh / 10 % 10) as u8);
            }
            f.push(b'0' + (nh % 10) as u8);
        }
        f
    }

    /// `ALOGPDescriptor.inSameAromaticRing` (ring whose bonds are all aromatic).
    fn same_aromatic_ring(&self, a: usize, w: usize, b: usize) -> bool {
        if self.v.arom_bond[b] {
            true
        } else if !self.v.m.ring_bond[b] || self.arom_blocks.is_empty() {
            false
        } else {
            self.arom_blocks[a]
                .iter()
                .any(|k| self.arom_blocks[w].contains(k))
        }
    }

    /// `isPyrroleLikeHetero`.
    fn pyrrole_like(&self, a: usize) -> bool {
        if !self.arom(a) {
            return false;
        }
        let deg = self.v.m.adj[a].len();
        match self.z(a) {
            7 | 15 => {
                (deg == 3 && self.q(a) == 0)
                    || (deg == 2 && self.implicit_h(a) == 1)
                    || (deg == 2 && self.q(a) == -1)
            }
            8 | 16 => true,
            _ => false,
        }
    }

    /// `ALOGPDescriptor.getHAtomType`: group of the hydrogens on `a` (0: none).
    fn h_atom_type(&self, a: usize) -> usize {
        let (mut ndoub, mut ntrip, mut ox, mut xcount) = (0, 0, 0, 0);
        let mut conj = false;
        for (w, b) in self.nbrs(a) {
            match self.order(b) {
                2 => ndoub += 1,
                3 => ntrip += 1,
                _ => {}
            }
            if is_hetero(self.z(w)) {
                if self.v.arom_bond[b] {
                    if self.order(b) == 1 {
                        if !self.pyrrole_like(w) && !conj {
                            ox += 2;
                            conj = true;
                        } else {
                            ox += 1;
                        }
                    } else if !conj {
                        conj = true;
                        ox += 2;
                    } else {
                        ox += 1;
                    }
                } else {
                    ox += self.order(b) as i32;
                }
            } else if self.z(w) == 6 {
                xcount += self.nbrs(w).filter(|&(x, _)| is_hetero(self.z(x))).count();
            }
        }
        #[derive(PartialEq)]
        enum Hyb {
            Sp1,
            Sp2,
            Sp3,
        }
        let hyb = if ndoub == 0 && ntrip == 0 {
            Hyb::Sp3
        } else if ndoub == 1 && ntrip == 0 {
            Hyb::Sp2
        } else if ndoub == 2 || ntrip == 1 {
            Hyb::Sp1
        } else {
            return 0;
        };
        let mut alpha = false;
        if self.z(a) == 6 && hyb == Hyb::Sp3 {
            for (w, _) in self.nbrs(a) {
                if is_hetero(self.z(w)) {
                    alpha = false;
                    break;
                } else if self.z(w) == 6 {
                    let mut x = 0;
                    for (w2, b2) in self.nbrs(w) {
                        if is_hetero(self.z(w2)) {
                            match self.order(b2) {
                                1 if self.v.arom_bond[b2] => x += 1,
                                2 | 3 => x += 1,
                                _ => {}
                            }
                        }
                    }
                    if x == 1 {
                        alpha = true;
                    }
                }
            }
        }
        if alpha {
            return 51;
        }
        match hyb {
            Hyb::Sp1 => match ox {
                0 => 48,
                1 => 49,
                _ => 0,
            },
            Hyb::Sp2 => match ox {
                0 => 47,
                1 => 48,
                2 | 3 => 49,
                _ => 0,
            },
            Hyb::Sp3 => match ox {
                0 => match xcount {
                    0 => 46,
                    1 => 52,
                    2 => 53,
                    3 => 54,
                    4 => 55,
                    _ => 0,
                },
                1 => 47,
                2 => 48,
                3 => 49,
                _ => 0,
            },
        }
    }

    #[inline]
    fn add(&mut self, group: usize) {
        self.frags[group] += 1;
    }

    /// All `calcGroup*` tests of `ALOGPDescriptor.calculate` for atom `a`.
    fn atom_groups(&mut self, a: usize) -> Result<(), Failed> {
        self.group_001_005(a);
        self.group_002_006_007(a);
        self.group_003_008_009_010(a);
        self.group_004_011_to_014(a);
        self.group_015(a);
        self.group_016_018_036_037(a);
        self.group_017_019_020_038_to_041(a);
        self.group_021_to_023_040(a);
        self.group_024_027_030_033_042(a)?;
        self.group_025_to_035_043_044(a);
        self.group_056_057(a);
        self.group_058_061(a);
        self.group_059_060_063(a);
        self.group_066_to_079(a);
        self.group_081_to_100(a)?;
        self.group_101_to_104(a);
        let f = self.frag[a].clone();
        match f.as_slice() {
            b"SsSH" => {
                self.add(106);
                self.add(50);
            }
            b"SssS" | b"SaaS" => self.add(107),
            b"SdS" => self.add(108),
            b"SdssS" | b"SddssS" => {
                let n_do = self
                    .nbrs(a)
                    .filter(|&(w, b)| self.order(b) == 2 && self.z(w) == 8)
                    .count();
                if f.as_slice() == b"SdssS" && n_do == 1 {
                    self.add(109);
                } else if f.as_slice() == b"SddssS" && n_do == 2 {
                    self.add(110);
                }
            }
            b"SssssSi" => self.add(111),
            b"SsssB" | b"SssBm" => self.add(112),
            b"SssssPp" => self.add(115),
            b"SdsssP" => {
                let (mut x, mut r, mut pdx) = (0, 0, false);
                for (w, b) in self.nbrs(a) {
                    match self.order(b) {
                        1 if self.z(w) == 6 => r += 1,
                        1 => x += 1,
                        2 if self.z(w) != 6 => pdx = true,
                        _ => {}
                    }
                }
                if pdx {
                    if r == 3 {
                        self.add(116);
                    } else if x == 3 {
                        self.add(117);
                    } else if x == 2 && r == 1 {
                        self.add(120);
                    }
                }
            }
            b"SsssP" => {
                let (mut x, mut r) = (0, 0);
                for (w, b) in self.nbrs(a) {
                    if self.order(b) == 1 {
                        if self.z(w) == 6 {
                            r += 1;
                        } else {
                            x += 1;
                        }
                    }
                }
                if x == 3 {
                    self.add(118);
                } else if r == 3 {
                    self.add(119);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// C in CH3R / CH3X.
    fn group_001_005(&mut self, a: usize) {
        if !self.is(a, "SsCH3") {
            return;
        }
        let ht = self.h_atom_type(a);
        self.frags[ht] += self.implicit_h(a);
        for k in 0..self.v.m.adj[a].len() {
            let w = self.nbr_at(a, k);
            match self.z(w) {
                6 => self.add(1),
                1 => self.add(ht),
                _ => self.add(5),
            }
        }
    }

    /// C in CH2R2 / CH2RX / CH2X2.
    fn group_002_006_007(&mut self, a: usize) {
        if !self.is(a, "SssCH2") {
            return;
        }
        let ht = self.h_atom_type(a);
        self.frags[ht] += self.implicit_h(a);
        let (c, x) = self.count_c_x_h(a, ht);
        match (c, x) {
            (2, 0) => self.add(2),
            (1, 1) => self.add(6),
            (0, 2) => self.add(7),
            _ => {}
        }
    }

    /// C in CHR3 / CHR2X / CHRX2 / CHX3.
    fn group_003_008_009_010(&mut self, a: usize) {
        if !self.is(a, "SsssCH") {
            return;
        }
        let ht = self.h_atom_type(a);
        self.frags[ht] += self.implicit_h(a);
        let (c, x) = self.count_c_x_h(a, ht);
        match (c, x) {
            (3, 0) => self.add(3),
            (2, 1) => self.add(8),
            (1, 2) => self.add(9),
            (0, 3) => self.add(10),
            _ => {}
        }
    }

    /// Carbon / hetero neighbour counts; explicit hydrogens go to group `ht`.
    fn count_c_x_h(&mut self, a: usize, ht: usize) -> (i32, i32) {
        let (mut c, mut x) = (0, 0);
        for k in 0..self.v.m.adj[a].len() {
            let w = self.nbr_at(a, k);
            match self.z(w) {
                6 => c += 1,
                1 => self.add(ht),
                _ => x += 1,
            }
        }
        (c, x)
    }

    /// C in CR4 / CR3X / CR2X2 / CRX3 / CX4.
    fn group_004_011_to_014(&mut self, a: usize) {
        if !self.is(a, "SssssC") {
            return;
        }
        let c = self.nbrs(a).filter(|&(w, _)| self.z(w) == 6).count();
        let x = self.v.m.adj[a].len() - c;
        match (c, x) {
            (4, 0) => self.add(4),
            (3, 1) => self.add(11),
            (2, 2) => self.add(12),
            (1, 3) => self.add(13),
            (0, 4) => self.add(14),
            _ => {}
        }
    }

    /// =CH2.
    fn group_015(&mut self, a: usize) {
        if !self.is(a, "SdCH2") {
            return;
        }
        self.add(15);
        let ht = self.h_atom_type(a);
        self.frags[ht] += 2;
    }

    /// =CH- groups.
    fn group_016_018_036_037(&mut self, a: usize) {
        if !self.is(a, "SdsCH") {
            return;
        }
        let ht = self.h_atom_type(a);
        self.add(ht);
        let (mut cdx, mut csx, mut csar) = (false, false, false);
        for (w, b) in self.nbrs(a) {
            if self.z(w) == 1 {
                continue;
            }
            match self.order(b) {
                1 => {
                    if self.z(w) != 6 {
                        csx = true;
                    }
                    if self.arom(w) {
                        csar = true;
                    }
                }
                2 if self.z(w) != 6 => cdx = true,
                _ => {}
            }
        }
        let g = if cdx {
            if csar { 37 } else { 36 }
        } else if csx {
            18
        } else {
            16
        };
        self.add(g);
    }

    /// =C< groups (including aromatic carbons with an exocyclic double bond).
    fn group_017_019_020_038_to_041(&mut self, a: usize) {
        if !self.is(a, "SdssC") && !self.is(a, "SdaaC") {
            return;
        }
        let (mut r, mut x, mut ar, mut cdx) = (0, 0, 0, false);
        for (w, b) in self.nbrs(a) {
            match self.order(b) {
                1 => {
                    if self.z(w) == 6 {
                        r += 1;
                        if self.arom(w) {
                            ar += 1;
                        }
                    } else {
                        x += 1;
                    }
                }
                2 if is_hetero(self.z(w)) => cdx = true,
                _ => {}
            }
        }
        let g = if cdx {
            if ar >= 1 {
                if x == 1 { 40 } else { 39 }
            } else if r == 1 && x == 1 {
                40
            } else if r == 0 && x == 2 {
                41
            } else {
                38
            }
        } else {
            match (r, x) {
                (2, 0) => 17,
                (1, 1) => 19,
                (0, 2) => 20,
                _ => return,
            }
        };
        self.add(g);
    }

    /// Triple-bonded and allene carbons.
    fn group_021_to_023_040(&mut self, a: usize) {
        if self.is(a, "StCH") {
            self.add(21);
            let ht = self.h_atom_type(a);
            self.add(ht);
        } else if self.is(a, "SddC") {
            let (n0, n1) = (self.nbr_at(a, 0), self.nbr_at(a, 1));
            if self.z(n0) == 6 && self.z(n1) == 6 {
                self.add(22);
            } else if self.z(n0) != 6 && self.z(n1) != 6 {
                self.add(40);
            }
        } else if self.is(a, "StsC") {
            let (mut ctx, mut csx) = (false, false);
            for (w, b) in self.nbrs(a) {
                match self.order(b) {
                    1 if self.z(w) != 6 => csx = true,
                    3 if self.z(w) != 6 => ctx = true,
                    _ => {}
                }
            }
            let g = if ctx && !csx {
                40
            } else if csx {
                23
            } else {
                22
            };
            self.add(g);
        }
    }

    /// Aromatic CH.
    fn group_024_027_030_033_042(&mut self, a: usize) -> Result<(), Failed> {
        if !self.is(a, "SaaCH") {
            return Ok(());
        }
        let ht = self.h_atom_type(a);
        self.add(ht);
        let adj = &self.v.m.adj[a];
        let nb = |k: usize| adj.get(k).map(|&(w, _)| w as usize).ok_or(Failed);
        let (ca0, ca1) = if self.implicit_h(a) == 1 {
            (nb(0)?, nb(1)?)
        } else if self.z(nb(0)?) == 1 {
            (nb(1)?, nb(2)?)
        } else if self.z(nb(1)?) == 1 {
            (nb(0)?, nb(2)?)
        } else if self.z(nb(2)?) == 1 {
            (nb(0)?, nb(1)?)
        } else {
            return Err(Failed);
        };
        if self.z(ca0) == 6 && self.z(ca1) == 6 {
            self.add(24);
            return Ok(());
        }
        let d1 = self.has_double(ca0);
        let d2 = self.has_double(ca1);
        // CDK tests `nbors.get(1)` here, not `ca1`
        let n1 = nb(1)?;
        if self.z(ca0) != 6 && self.z(n1) != 6 {
            self.add(if d1 && d2 { 30 } else { 42 });
        } else if (self.z(ca0) == 6) != (self.z(ca1) == 6) {
            self.add(if d1 && d2 { 27 } else { 33 });
        }
        Ok(())
    }

    /// Substituted aromatic carbons.
    fn group_025_to_035_043_044(&mut self, a: usize) {
        if !self.is(a, "SsaaC") && !self.is(a, "SaaaC") {
            return;
        }
        let nbors: SmallVec<[usize; 4]> =
            self.v.m.adj[a].iter().map(|&(w, _)| w as usize).collect();
        let same_ring: SmallVec<[bool; 4]> = self
            .nbrs(a)
            .map(|(w, b)| self.same_aromatic_ring(a, w, b))
            .collect();
        let same = same_ring.iter().filter(|&&x| x).count();
        let (r0, r1, nonring) = if same == 2 {
            let mut ring: SmallVec<[usize; 2]> = SmallVec::new();
            let mut nonring = None;
            for (k, &w) in nbors.iter().enumerate() {
                if same_ring[k] {
                    ring.push(w);
                } else {
                    nonring = Some(w);
                }
            }
            (ring[0], ring[1], nonring)
        } else {
            // CDK: stable sort putting hetero atoms first
            let mut sorted = nbors.clone();
            sorted.sort_by_key(|&w| !is_hetero(self.z(w)));
            (sorted[0], sorted[1], sorted.get(2).copied())
        };
        // an unassigned non-ring atom is a fresh `IAtom` (no element)
        let nonring_c = nonring.is_some_and(|w| self.z(w) == 6);
        let d1 = self.has_double(r0);
        let d2 = self.has_double(r1);
        let (c0, c1) = (self.z(r0) == 6, self.z(r1) == 6);
        let g = if !c0 && !c1 {
            if d1 && d2 {
                if nonring_c { 31 } else { 32 }
            } else if nonring_c {
                43
            } else {
                44
            }
        } else if c0 && c1 {
            if nonring_c { 25 } else { 26 }
        } else if d1 && d2 {
            if nonring_c { 28 } else { 29 }
        } else if nonring_c {
            34
        } else {
            35
        };
        self.add(g);
    }

    /// O in alcohols (56) and phenol / enol / carboxyl (57).
    fn group_056_057(&mut self, a: usize) {
        if !self.is(a, "SsOH") {
            return;
        }
        self.add(50);
        let ca0 = self
            .nbrs(a)
            .filter(|&(w, _)| self.z(w) == 6)
            .last()
            .map(|(w, _)| w);
        if let Some(c) = ca0 {
            if self.arom(c) {
                self.add(57);
                return;
            }
            let enol = self
                .nbrs(c)
                .any(|(w, b)| w != a && self.order(b) == 2 && matches!(self.z(w), 6 | 8));
            if enol {
                self.add(57);
                return;
            }
        }
        self.add(56);
    }

    /// =O (58), O in nitro / N-oxides (61), O- (62).
    fn group_058_061(&mut self, a: usize) {
        let om = self.is(a, "SsOm");
        if !om && !self.is(a, "SdO") {
            return;
        }
        let ca0 = self.nbr_at(a, 0);
        if self.z(ca0) == 7 && self.q(ca0) == 1 {
            self.add(61);
        } else {
            self.add(if om { 62 } else { 58 });
        }
    }

    /// Ether-type oxygens.
    fn group_059_060_063(&mut self, a: usize) {
        if self.is(a, "SaaO") {
            self.add(60);
            return;
        }
        if !self.is(a, "SssO") {
            return;
        }
        let (ca0, ca1) = (self.nbr_at(a, 0), self.nbr_at(a, 1));
        if self.arom(ca0) || self.arom(ca1) {
            self.add(60);
            return;
        }
        for (w, _) in self.nbrs(a) {
            if self.z(w) == 6
                && self
                    .nbrs(w)
                    .any(|(x, b)| self.order(b) == 2 && self.z(x) != 6)
            {
                self.add(60);
                return;
            }
        }
        if self.z(ca0) == 8 || self.z(ca1) == 8 {
            self.add(63);
        } else {
            self.add(59);
        }
    }

    /// Nitrogen groups (and the hydrogens on nitrogen).
    fn group_066_to_079(&mut self, a: usize) {
        if self.z(a) != 7 {
            return;
        }
        let explicit_h = self.nbrs(a).filter(|&(w, _)| self.z(w) == 1).count() as i32;
        self.frags[50] += explicit_h + self.implicit_h(a);
        let (mut n_ar, mut n_al) = (0, 0);
        for (w, _) in self.nbrs(a) {
            if self.z(w) == 1 {
                continue;
            }
            if self.arom(w) {
                n_ar += 1;
            } else {
                n_al += 1;
            }
        }
        let f = self.frag[a].clone();
        let f = f.as_slice();
        if matches!(f, b"SsssN" | b"SssNH" | b"SsNH2") {
            // RC(=O)N or NX=X
            for (j, _) in self.nbrs(a) {
                if self.z(j) == 1 {
                    continue;
                }
                for (k, bjk) in self.nbrs(j) {
                    if k != a
                        && self.z(k) != 6
                        && !self.arom(k)
                        && !self.arom(j)
                        && !self.arom(a)
                        && self.order(bjk) == 2
                    {
                        self.add(72);
                        return;
                    }
                }
            }
        }
        match f {
            b"SsNH2" => {
                let Some((ca0, _)) = self.nbrs(a).find(|&(w, _)| self.z(w) != 1) else {
                    return;
                };
                self.add(if self.arom(ca0) || self.z(ca0) != 6 {
                    69
                } else {
                    66
                });
            }
            b"SaaNH" | b"SsaaN" | b"SaaaN" | b"SaaNm" => self.add(73),
            b"SssNH" => match (n_ar, n_al) {
                (2, 0) => self.add(73),
                (1, 1) => self.add(70),
                (0, 2) => self.add(67),
                _ => {}
            },
            b"SsssN" => match (n_ar, n_al) {
                (3, 0) | (2, 1) => self.add(73),
                (1, 2) => self.add(71),
                (0, 3) => self.add(68),
                _ => {}
            },
            b"SaaN" => self.add(75),
            b"SdssNp" | b"SddsN" => {
                let (mut ss_om, mut sd_o, mut ar) = (0, 0, false);
                for (w, _) in self.nbrs(a) {
                    if self.is(w, "SsOm") {
                        ss_om += 1;
                    } else if self.is(w, "SdO") {
                        sd_o += 1;
                    } else if self.arom(w) {
                        ar = true;
                    }
                }
                let nitro = sd_o == 2 || (ss_om >= 1 && sd_o >= 1);
                self.add(match (nitro, ar) {
                    (true, true) => 76,
                    (true, false) => 77,
                    _ => 79,
                });
            }
            b"StN" => {
                if self.z(self.nbr_at(a, 0)) == 6 {
                    self.add(74);
                }
            }
            b"SdNH" | b"SdsN" => {
                if f == b"SdsN" && self.z(self.nbr_at(a, 0)) == 8 && self.z(self.nbr_at(a, 1)) == 8
                {
                    self.add(76);
                    return;
                }
                let (mut flag1, mut flag2) = (false, false);
                for (w, b) in self.nbrs(a) {
                    if self.z(w) == 1 {
                        continue;
                    }
                    if self.order(b) == 2 {
                        if self.z(w) == 6 {
                            self.add(74);
                            return;
                        }
                        flag1 = true;
                    } else if self.z(w) != 6 || self.arom(w) {
                        flag2 = true;
                    }
                    if flag1 && flag2 {
                        self.add(78);
                        return;
                    }
                }
            }
            _ => {
                if f.contains(&b'p') {
                    self.add(79);
                }
            }
        }
    }

    /// Halogens on carbon (81-100).
    fn group_081_to_100(&mut self, a: usize) -> Result<(), Failed> {
        let base = match self.frag[a].as_slice() {
            b"SsF" => 81,
            b"SsCl" => 86,
            b"SsBr" => 91,
            b"SsI" => 96,
            _ => return Ok(()),
        };
        let ca0 = self.nbr_at(a, 0);
        let (mut d, mut t) = (0, 0);
        for (_, b) in self.nbrs(ca0) {
            match self.order(b) {
                2 => d += 1,
                3 => t += 1,
                _ => {}
            }
        }
        #[derive(PartialEq)]
        enum Hyb {
            Sp3,
            Sp2,
            Sp,
            None,
        }
        let hyb = if d == 0 && t == 0 {
            Hyb::Sp3
        } else if d == 1 {
            Hyb::Sp2
        } else if d == 2 || t == 1 {
            Hyb::Sp
        } else {
            Hyb::None
        };
        let mut ox = 0;
        for (w, b) in self.nbrs(ca0) {
            let polar = if base == 81 {
                is_hetero(self.z(w))
            } else {
                electronegative(self.z(w)).ok_or(Failed)?
            };
            if polar {
                ox += self.order(b) as i32;
            }
        }
        if self.z(ca0) != 6 {
            self.add(base + 4);
            return Ok(());
        }
        let g = match (&hyb, ox) {
            (Hyb::Sp3, 1) => base,
            (Hyb::Sp3, 2) => base + 1,
            (Hyb::Sp3, 3) => base + 2,
            (Hyb::Sp2, 1) => base + 3,
            (Hyb::Sp2, o) if o > 1 => base + 4,
            (Hyb::Sp, o) if o >= 1 => base + 4,
            (Hyb::Sp3, 4) => base + 4,
            _ => return Ok(()),
        };
        self.add(g);
        Ok(())
    }

    /// Halide ions and hydrogen halides (101-104).
    fn group_101_to_104(&mut self, a: usize) {
        let deg = self.v.m.adj[a].len();
        let h_only =
            (deg == 0 && self.implicit_h(a) == 1) || (deg == 1 && self.z(self.nbr_at(a, 0)) == 1);
        if self.q(a) == -1 || (self.q(a) == 0 && h_only) {
            match self.z(a) {
                9 => self.add(101),
                17 => self.add(102),
                35 => self.add(103),
                53 => self.add(104),
                _ => {}
            }
        }
    }
}

/// Whether `AllRingsFinder.findAllRings` (threshold `PubChem_99`) completes:
/// CDK's `AllCycles` path-graph reduction is replayed on every fused ring
/// system, stopping when a vertex to remove has more than 684 path edges.
fn all_rings_tractable(mol: &Molecule, v: &LogpView) -> bool {
    let m = &v.m;
    if !m.ring_bond.iter().any(|&r| r) {
        return true;
    }
    // Shortcut with the ring perception: no atom in two rings means only
    // isolated cycles; otherwise the path bound below (see the per-block test)
    // with all ring atoms and an upper bound of the cyclomatic number.
    if let Some(info) = mol.ring_info() {
        let ring_atoms = info.atom_ring_counts.iter().filter(|&&c| c > 0).count();
        let r = info.num_rings();
        if info.atom_ring_counts.iter().all(|&c| c <= 1)
            || (r < 16
                && ring_atoms.saturating_sub(1) * ((1usize << (r + 1)) - 1) <= ALL_RINGS_THRESHOLD)
        {
            return true;
        }
    }
    let (block, size) = edge_blocks(v, |b| m.ring_bond[b]);
    // vertices of every block with more bonds than an isolated cycle could have
    let mut verts: Vec<Vec<usize>> = vec![Vec::new(); size.len()];
    for b in 0..m.ends.len() {
        let k = block[b];
        if k != u32::MAX && size[k as usize] >= 4 {
            let (x, y) = m.ends[b];
            verts[k as usize].extend([x as usize, y as usize]);
        }
    }
    for (k, vs) in verts.iter_mut().enumerate() {
        if vs.is_empty() {
            continue;
        }
        vs.sort_unstable();
        vs.dedup();
        // isolated cycles need no search
        let (e, nv) = (size[k] as usize, vs.len());
        if e <= nv {
            continue;
        }
        // A path edge is a simple path; with cyclomatic number r there are at
        // most 2^(r+1) - 1 simple paths between two vertices, so no vertex can
        // collect more than (|V| - 1)(2^(r+1) - 1) path edges.
        let r = e - nv + 1;
        if r < 16 && (nv - 1) * ((1usize << (r + 1)) - 1) <= ALL_RINGS_THRESHOLD {
            continue;
        }
        if !path_graph_completes(v, vs) {
            return false;
        }
    }
    true
}

/// Replays `AllCycles` (`RegularPathGraph` / `JumboPathGraph`) on the subgraph
/// induced by `verts` and reports whether every vertex could be removed.
fn path_graph_completes(v: &LogpView, verts: &[usize]) -> bool {
    let m = &v.m;
    let ord = verts.len();
    let pos = |x: usize| verts.binary_search(&x).ok();
    let graph: Vec<Vec<usize>> = verts
        .iter()
        .map(|&x| {
            m.adj[x]
                .iter()
                .filter_map(|&(w, _)| pos(w as usize))
                .collect()
        })
        .collect();
    // AllCycles.rank: counting sort by degree (stable in vertex order)
    let mut count = vec![0usize; ord + 2];
    for g in &graph {
        count[g.len() + 1] += 1;
    }
    for i in 0..=ord {
        count[i + 1] += count[i];
    }
    let mut rank = vec![0usize; ord];
    for x in 0..ord {
        rank[x] = count[graph[x].len()];
        count[graph[x].len()] += 1;
    }
    let mut order = vec![0usize; ord];
    for x in 0..ord {
        order[rank[x]] = x;
    }
    let words = ord.div_ceil(64);
    // path edges: endpoints + interior vertex set
    let mut ends: Vec<(u32, u32)> = Vec::new();
    let mut sets: Vec<u64> = Vec::new();
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); ord];
    let add = |e: u32, ends: &[(u32, u32)], lists: &mut Vec<Vec<u32>>| {
        let (x, y) = ends[e as usize];
        let (x, y) = (x as usize, y as usize);
        if rank[x] < rank[y] {
            lists[x].push(e);
        } else {
            lists[y].push(e);
        }
    };
    for x in 0..ord {
        for &w in &graph[x] {
            if w > x {
                let e = ends.len() as u32;
                ends.push((x as u32, w as u32));
                sets.extend(std::iter::repeat_n(0, words));
                add(e, &ends, &mut lists);
            }
        }
    }
    for &x in &order {
        if lists[x].len() > ALL_RINGS_THRESHOLD {
            return false;
        }
        let edges = std::mem::take(&mut lists[x]);
        for i in 0..edges.len() {
            let e = edges[i] as usize;
            for &f in &edges[i + 1..] {
                let f = f as usize;
                let (se, sf) = (
                    &sets[e * words..(e + 1) * words],
                    &sets[f * words..(f + 1) * words],
                );
                if se.iter().zip(sf).any(|(p, q)| p & q != 0) {
                    continue;
                }
                let other = |g: usize| {
                    let (p, q) = ends[g];
                    if p as usize == x { q } else { p }
                };
                let (u, w) = (other(e), other(f));
                if u == w {
                    continue; // a cycle
                }
                let id = ends.len() as u32;
                let mut s: SmallVec<[u64; 4]> = se.iter().zip(sf).map(|(p, q)| p | q).collect();
                s[x / 64] |= 1 << (x % 64);
                ends.push((u, w));
                sets.extend_from_slice(&s);
                add(id, &ends, &mut lists);
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use crate::desc::cdk_logp::LogpView;
    use crate::io::smiles::parse_smiles;

    fn alogp(smi: &str) -> [f64; 3] {
        let mol = parse_smiles(smi).unwrap();
        super::alogp(&mol, &LogpView::new(&mol))
    }

    #[test]
    fn toluene_and_ethanol() {
        // toluene: 5 x C24 + C25 + C1, 5 x H47 (aromatic CH) + 3 x H46
        let v = alogp("Cc1ccccc1");
        let exp = 5.0 * -0.3251 + 0.1492 - 1.5603 + 5.0 * 0.6301 + 3.0 * 0.7341;
        assert!((v[0] - exp).abs() < 1e-12, "{v:?} {exp}");
        // ethanol: C1 + C6 + O56 + 3 x H52 (CH3 beta to O) + 2 x H47 + H50
        let v = alogp("CCO");
        let exp = -1.5603 - 1.2486 - 0.3567 + 3.0 * 0.6666 + 2.0 * 0.6301 - 0.1036;
        assert!((v[0] - exp).abs() < 1e-12, "{v:?} {exp}");
    }
}
