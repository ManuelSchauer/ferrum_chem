//! Minimal SMARTS compiler/matcher for the PubChem keys (private to `pubchem`).
//!
//! Implements exactly the SMARTS subset used by CDK's `PubchemFingerprinter`
//! with CDK semantics (`org.openscience.cdk.isomorphism.matchers.Expr`):
//!
//! * atoms: `#n`, element symbols (lower case = aromatic, upper case organic
//!   subset outside brackets = aliphatic), `H<n>` (total H count), `!H0`,
//!   `&` and `,` (`,` binds weaker than `&`);
//! * bonds: `-` `=` `#` (aliphatic orders), `:` (aromatic), `~` (any), `,`
//!   lists; an implicit bond is single-or-aromatic;
//! * branches and single-digit ring closures.
//!
//! Only "does the pattern match at least once" is needed (all PubChem SMARTS
//! bits test `count > 0`), so matching stops at the first embedding.

use smallvec::SmallVec;

/// Target bond classes (bit index into a bond mask).
pub(super) const BC_SINGLE: u8 = 0;
pub(super) const BC_DOUBLE: u8 = 1;
pub(super) const BC_TRIPLE: u8 = 2;
pub(super) const BC_AROM: u8 = 3;
pub(super) const BC_OTHER: u8 = 4;
const BM_ANY: u8 = 0x1F;

/// Aromaticity requirement of an atom alternative.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arom {
    Any,
    Aliphatic,
    Aromatic,
}

#[derive(Clone, Copy, Debug)]
struct AtomAlt {
    /// Atomic number (0 = any).
    z: u8,
    arom: Arom,
    /// Allowed total H counts (bit h; bit 15 = 15 or more).
    hmask: u16,
}

#[derive(Clone, Debug)]
struct PAtom {
    alts: SmallVec<[AtomAlt; 1]>,
    /// Number of pattern bonds (a target atom needs at least as many neighbours).
    degree: u32,
}

/// Compiled pattern in matching order.
#[derive(Debug)]
pub(super) struct Pattern {
    atoms: Vec<PAtom>,
    /// For atom i > 0 (in matching order): index of the already placed anchor
    /// atom and the bond mask to it.
    anchor: Vec<(u8, u8)>,
    /// Extra bonds to already placed atoms (closure checks): (atom i, j < i, mask).
    checks: Vec<SmallVec<[(u8, u8); 2]>>,
    /// Required number of atoms per atomic number (screen).
    pub(super) need: SmallVec<[(u8, u8); 4]>,
    /// Required molecule features (`FEAT_*`, screen).
    pub(super) feats: u8,
}

/// Molecule features used for screening.
pub(super) const FEAT_AROM_ATOM: u8 = 1;
pub(super) const FEAT_DOUBLE: u8 = 2;
pub(super) const FEAT_TRIPLE: u8 = 4;
pub(super) const FEAT_AROM_BOND: u8 = 8;

/// Per-molecule target for matching.
pub(super) struct Target<'a> {
    pub z: &'a [u8],
    pub arom: &'a [bool],
    pub hcount: &'a [u8],
    pub adj_start: &'a [u32],
    pub adj_nbr: &'a [u32],
    /// Bond class (`BC_*`) per adjacency entry.
    pub adj_class: &'a [u8],
    /// Atoms grouped by atomic number: `by_z[by_z_start[z]..by_z_start[z + 1]]`.
    pub by_z_start: &'a [u32],
    pub by_z: &'a [u32],
}

impl AtomAlt {
    #[inline]
    fn matches(&self, t: &Target, a: usize) -> bool {
        (self.z == 0 || self.z == t.z[a])
            && match self.arom {
                Arom::Any => true,
                Arom::Aliphatic => !t.arom[a],
                Arom::Aromatic => t.arom[a],
            }
            && self.hmask >> (t.hcount[a].min(15)) & 1 == 1
    }
}

impl PAtom {
    #[inline]
    fn matches(&self, t: &Target, a: usize) -> bool {
        t.adj_start[a + 1] - t.adj_start[a] >= self.degree
            && self.alts.iter().any(|x| x.matches(t, a))
    }
}

fn element(sym: &str) -> Option<u8> {
    Some(match sym {
        "H" => 1,
        "Li" => 3,
        "B" => 5,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "Na" => 11,
        "Mg" => 12,
        "Al" => 13,
        "Si" => 14,
        "P" => 15,
        "S" => 16,
        "Cl" => 17,
        "K" => 19,
        "As" => 33,
        "Se" => 34,
        "Br" => 35,
        "I" => 53,
        _ => return None,
    })
}

struct Parser<'s> {
    s: &'s [u8],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }

    fn number(&mut self) -> Option<u32> {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        (self.pos > start).then(|| {
            std::str::from_utf8(&self.s[start..self.pos])
                .unwrap()
                .parse()
                .unwrap()
        })
    }

    /// Element symbol at the cursor (two-letter symbols first).
    fn symbol(&mut self) -> Option<(u8, bool)> {
        let c = self.peek()?;
        if c.is_ascii_lowercase() {
            let z = match c {
                b'c' => 6,
                b'n' => 7,
                b'o' => 8,
                b's' => 16,
                b'p' => 15,
                _ => return None,
            };
            self.pos += 1;
            return Some((z, true));
        }
        if !c.is_ascii_uppercase() {
            return None;
        }
        if let Some(z) = self
            .s
            .get(self.pos + 1)
            .filter(|c2| c2.is_ascii_lowercase())
            .and_then(|_| element(std::str::from_utf8(&self.s[self.pos..self.pos + 2]).unwrap()))
        {
            self.pos += 2;
            return Some((z, false));
        }
        let one = std::str::from_utf8(&self.s[self.pos..self.pos + 1]).unwrap();
        let z = element(one)?;
        self.pos += 1;
        Some((z, false))
    }

    /// One `&`-conjunction inside brackets.
    fn conjunction(&mut self) -> AtomAlt {
        let mut alt = AtomAlt {
            z: 0,
            arom: Arom::Any,
            hmask: u16::MAX,
        };
        loop {
            match self.peek() {
                Some(b'#') => {
                    self.pos += 1;
                    alt.z = self.number().expect("atomic number") as u8;
                }
                Some(b'H')
                    if alt.z != 0
                        || self
                            .s
                            .get(self.pos + 1)
                            .is_none_or(|c| !c.is_ascii_lowercase()) =>
                {
                    // H count (an element would have been read first)
                    self.pos += 1;
                    let h = self.number().unwrap_or(1);
                    alt.hmask &= 1 << h;
                }
                Some(b'!') => {
                    self.pos += 1;
                    assert_eq!(self.peek(), Some(b'H'), "only !H<n> is supported");
                    self.pos += 1;
                    let h = self.number().unwrap_or(1);
                    alt.hmask &= !(1 << h);
                }
                Some(b'&') => self.pos += 1,
                Some(b',') | Some(b']') => return alt,
                _ => {
                    let (z, arom) = self.symbol().expect("element symbol");
                    alt.z = z;
                    alt.arom = if arom {
                        Arom::Aromatic
                    } else if matches!(z, 5 | 6 | 7 | 8 | 15 | 16) {
                        Arom::Aliphatic
                    } else {
                        Arom::Any
                    };
                }
            }
        }
    }

    fn bond(&mut self) -> Option<u8> {
        let mut mask = 0u8;
        let mut any = false;
        loop {
            let m = match self.peek() {
                Some(b'-') => 1 << BC_SINGLE,
                Some(b'=') => 1 << BC_DOUBLE,
                Some(b'#') => 1 << BC_TRIPLE,
                Some(b':') => 1 << BC_AROM,
                Some(b'~') => BM_ANY,
                Some(b',') if any => {
                    self.pos += 1;
                    continue;
                }
                _ => break,
            };
            self.pos += 1;
            mask |= m;
            any = true;
        }
        any.then_some(mask)
    }
}

const IMPLICIT_BOND: u8 = (1 << BC_SINGLE) | (1 << BC_AROM);

/// Parses a SMARTS of the supported subset into atoms (in SMARTS order) and bonds.
fn parse(smarts: &str) -> (Vec<PAtom>, Vec<(usize, usize, u8)>) {
    let mut p = Parser {
        s: smarts.as_bytes(),
        pos: 0,
    };
    let mut atoms: Vec<PAtom> = Vec::new();
    let mut bonds: Vec<(usize, usize, u8)> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut prev: Option<usize> = None;
    let mut pending: Option<u8> = None;
    let mut rings: [Option<(usize, Option<u8>)>; 10] = [None; 10];
    while let Some(c) = p.peek() {
        match c {
            b'(' => {
                p.pos += 1;
                stack.push(prev.expect("branch without atom"));
            }
            b')' => {
                p.pos += 1;
                prev = stack.pop();
            }
            b'0'..=b'9' => {
                p.pos += 1;
                let r = (c - b'0') as usize;
                let cur = prev.expect("ring closure without atom");
                if let Some((other, b0)) = rings[r].take() {
                    let mask = pending.take().or(b0).unwrap_or(IMPLICIT_BOND);
                    bonds.push((other, cur, mask));
                } else {
                    rings[r] = Some((cur, pending.take()));
                }
            }
            b'-' | b'=' | b'#' | b':' | b'~' => {
                pending = p.bond();
            }
            b'[' => {
                p.pos += 1;
                let mut alts = SmallVec::new();
                loop {
                    alts.push(p.conjunction());
                    match p.peek() {
                        Some(b',') => p.pos += 1,
                        Some(b']') => {
                            p.pos += 1;
                            break;
                        }
                        other => panic!("unexpected {other:?} in {smarts}"),
                    }
                }
                let idx = atoms.len();
                atoms.push(PAtom { alts, degree: 0 });
                if let Some(pr) = prev {
                    bonds.push((pr, idx, pending.take().unwrap_or(IMPLICIT_BOND)));
                }
                prev = Some(idx);
            }
            _ => {
                let (z, arom) = p.symbol().unwrap_or_else(|| panic!("bad atom in {smarts}"));
                let a = if arom {
                    Arom::Aromatic
                } else if matches!(z, 5 | 6 | 7 | 8 | 15 | 16) {
                    Arom::Aliphatic
                } else {
                    Arom::Any
                };
                let idx = atoms.len();
                atoms.push(PAtom {
                    alts: smallvec::smallvec![AtomAlt {
                        z,
                        arom: a,
                        hmask: u16::MAX
                    }],
                    degree: 0,
                });
                if let Some(pr) = prev {
                    bonds.push((pr, idx, pending.take().unwrap_or(IMPLICIT_BOND)));
                }
                prev = Some(idx);
            }
        }
    }
    (atoms, bonds)
}

/// Rough rarity of an atom query (higher = fewer expected matches).
fn rarity(a: &PAtom) -> u32 {
    let alt = &a.alts[0];
    let mut r = match alt.z {
        0 => 0,
        6 => 1,
        8 => 3,
        7 => 4,
        _ => 6,
    };
    if a.alts.len() == 1 {
        if alt.arom != Arom::Any {
            r += 1;
        }
        if alt.hmask != u16::MAX {
            r += 1;
        }
    }
    r
}

impl Pattern {
    pub(super) fn compile(smarts: &str) -> Pattern {
        let (mut atoms, bonds) = parse(smarts);
        for &(a, b, _) in &bonds {
            atoms[a].degree += 1;
            atoms[b].degree += 1;
        }
        let n = atoms.len();
        assert!(n > 0 && n < 255);
        // matching order: start at the rarest atom, then grow connected,
        // preferring atoms with more bonds into the placed set, then rarity
        let mut placed = vec![usize::MAX; n]; // position in order
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let start = (0..n)
            .max_by_key(|&i| (rarity(&atoms[i]), atoms[i].degree, std::cmp::Reverse(i)))
            .unwrap();
        placed[start] = 0;
        order.push(start);
        while order.len() < n {
            let mut best: Option<(usize, (u32, u32, u32))> = None;
            for i in 0..n {
                if placed[i] != usize::MAX {
                    continue;
                }
                let links = bonds
                    .iter()
                    .filter(|&&(a, b, _)| {
                        (a == i && placed[b] != usize::MAX) || (b == i && placed[a] != usize::MAX)
                    })
                    .count() as u32;
                if links == 0 {
                    continue;
                }
                let key = (links, atoms[i].degree, rarity(&atoms[i]));
                if best.is_none_or(|(_, k)| key > k) {
                    best = Some((i, key));
                }
            }
            let (i, _) = best.expect("disconnected SMARTS");
            placed[i] = order.len();
            order.push(i);
        }
        let mut anchor = vec![(0u8, 0u8); n];
        let mut checks: Vec<SmallVec<[(u8, u8); 2]>> = vec![SmallVec::new(); n];
        for (pos, &i) in order.iter().enumerate().skip(1) {
            let mut first = true;
            for &(a, b, mask) in &bonds {
                let other = if a == i {
                    b
                } else if b == i {
                    a
                } else {
                    continue;
                };
                if placed[other] >= pos {
                    continue;
                }
                if first {
                    anchor[pos] = (placed[other] as u8, mask);
                    first = false;
                } else {
                    checks[pos].push((placed[other] as u8, mask));
                }
            }
        }
        let ordered: Vec<PAtom> = order.iter().map(|&i| atoms[i].clone()).collect();
        let mut need: SmallVec<[(u8, u8); 4]> = SmallVec::new();
        for a in &ordered {
            let z = a.alts[0].z;
            if z == 0 || a.alts.iter().any(|x| x.z != z) {
                continue;
            }
            match need.iter_mut().find(|e| e.0 == z) {
                Some(e) => e.1 += 1,
                None => need.push((z, 1)),
            }
        }
        let mut feats = 0u8;
        if ordered
            .iter()
            .any(|a| a.alts.iter().all(|x| x.arom == Arom::Aromatic))
        {
            feats |= FEAT_AROM_ATOM;
        }
        for &(_, _, mask) in &bonds {
            feats |= match mask {
                m if m == 1 << BC_DOUBLE => FEAT_DOUBLE,
                m if m == 1 << BC_TRIPLE => FEAT_TRIPLE,
                m if m == 1 << BC_AROM => FEAT_AROM_BOND,
                _ => 0,
            };
        }
        Pattern {
            atoms: ordered,
            anchor,
            checks,
            need,
            feats,
        }
    }

    /// True if the pattern has at least one embedding in the target.
    pub(super) fn matches(&self, t: &Target) -> bool {
        let mut map: SmallVec<[u32; 16]> = smallvec::smallvec![u32::MAX; self.atoms.len()];
        let first = &self.atoms[0];
        let mut try_start = |a: usize| -> bool {
            if first.matches(t, a) {
                map[0] = a as u32;
                if self.atoms.len() == 1 || self.extend(t, &mut map, 1) {
                    return true;
                }
            }
            false
        };
        let z = first.alts[0].z as usize;
        if z != 0 && first.alts.iter().all(|x| x.z as usize == z) {
            // only atoms of that element can start an embedding
            t.by_z[t.by_z_start[z] as usize..t.by_z_start[z + 1] as usize]
                .iter()
                .any(|&a| try_start(a as usize))
        } else {
            (0..t.z.len()).any(try_start)
        }
    }

    fn extend(&self, t: &Target, map: &mut [u32], pos: usize) -> bool {
        let (anc, mask) = self.anchor[pos];
        let from = map[anc as usize] as usize;
        let pa = &self.atoms[pos];
        'cand: for k in t.adj_start[from] as usize..t.adj_start[from + 1] as usize {
            if mask >> t.adj_class[k] & 1 == 0 {
                continue;
            }
            let w = t.adj_nbr[k];
            if map[..pos].contains(&w) || !pa.matches(t, w as usize) {
                continue;
            }
            for &(j, m2) in &self.checks[pos] {
                let other = map[j as usize];
                let ws = w as usize;
                let ok = (t.adj_start[ws] as usize..t.adj_start[ws + 1] as usize)
                    .any(|q| t.adj_nbr[q] == other && m2 >> t.adj_class[q] & 1 == 1);
                if !ok {
                    continue 'cand;
                }
            }
            map[pos] = w;
            if pos + 1 == self.atoms.len() || self.extend(t, map, pos + 1) {
                return true;
            }
        }
        map[pos] = u32::MAX;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_all_pubchem_smarts() {
        for s in super::super::pubchem_keys::SMARTS_BITS {
            let p = Pattern::compile(s);
            assert!(!p.atoms.is_empty(), "{s}");
        }
    }

    #[test]
    fn parses_h_alternatives() {
        let (atoms, bonds) = parse("[#6H,#6H2,#6H3]=,:[#7&!H0]");
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms[0].alts.len(), 3);
        assert_eq!(atoms[0].alts[1].hmask, 1 << 2);
        assert_eq!(atoms[1].alts[0].hmask, !1u16);
        assert_eq!(bonds, vec![(0, 1, (1 << BC_DOUBLE) | (1 << BC_AROM))]);
    }

    #[test]
    fn parses_ring_closures() {
        let (atoms, bonds) = parse("Brc1c(Br)cccc1");
        assert_eq!(atoms.len(), 8);
        assert_eq!(bonds.len(), 8);
        assert!(bonds.contains(&(1, 7, IMPLICIT_BOND)));
        assert_eq!(atoms[0].alts[0].z, 35);
        assert_eq!(atoms[1].alts[0].arom, Arom::Aromatic);
    }
}
