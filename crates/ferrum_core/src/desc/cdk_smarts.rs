//! CDK 2.13 SMARTS semantics for the CDK SMARTS-key fingerprints
//! (`MACCSFingerprinter`, `EStateFingerprinter`, `SubstructureFingerprinter`,
//! `KlekotaRothFingerprinter`).
//!
//! FerrumChem's general SMARTS engine (`io::smarts`, `algo::iso`) follows
//! RDKit. CDK differs in perception (Kekulé bond orders, Daylight aromaticity
//! applied by `SmartsPattern.prepare`, CDK implicit hydrogens, ring counts from
//! the minimum cycle basis) and in a few primitives, so the CDK keys get their
//! own small engine that ports exactly what CDK does:
//!
//! - the SMARTS grammar of CDK `Smarts` (default flavour `FLAVOR_LOOSE`):
//!   primitive meanings, operator precedence, single-atom `$(...)` inlined,
//!   ring-closure bond rules, component groups `(...).(...)`, query stereo
//!   (`@`/`@@` on atoms with three or four query neighbours, `/` `\` around a
//!   double bond);
//! - atom/bond predicates of CDK `Expr.matches` (`TOTAL_H_COUNT` incl. hydrogen
//!   neighbours, `D` = graph degree, `X` = degree + implicit H, `R<n>` = number
//!   of minimum-cycle-basis rings, `r<n>` = `isInSmallRingSize`, ...);
//! - `DfPattern` matching with the post-match filters `QueryStereoFilter` and
//!   `ComponentFilter`, `Pattern.matches` (any match) and
//!   `Mappings.countUnique()` (number of distinct matched atom sets).
//!
//! The molecule is the CDK reference preparation (`SmilesParser`, atom types,
//! implicit hydrogens) after `SmartsPattern.prepare` (ring flags +
//! `Aromaticity.Model.Daylight`), rebuilt locally from the RDKit-perceived
//! [`Molecule`] in [`CdkTarget::new`]. It is built once per molecule and
//! shared by all four fingerprints ([`CdkKeyCache`]).
//!
//! Speed (the 4,860 Klekota-Roth keys are matched in about 0.15 ms per
//! molecule, CDK needs about 35 ms for bits and counts):
//!
//! - all keys of a fingerprint share one table of distinct atom expressions,
//!   evaluated per molecule once as atom bit masks: the parts that only read
//!   the atom itself once per class of equal atoms, recursive SMARTS lazily
//!   for all candidate roots in one search, memoised;
//! - implied constraints are added to every query (atoms and bonds on a query
//!   cycle need ring atoms/bonds, an atom with `d` query neighbours needs at
//!   least `d` neighbours), which sharpens all screens;
//! - a key is only searched when each of its atom expressions and each of its
//!   (atom, bond, atom) expression triples matches at least as many target
//!   atoms/bonds as the key has;
//! - the search starts at the query atom with the fewest candidates, extends
//!   along the rarest neighbours (iteratively, with distance bounds for ring
//!   closures), and when counting it only follows the lexicographic leader
//!   of each class of mappings that differ by a query automorphism
//!   (stabilizer-chain symmetry breaking), hashing the remaining atom sets.
//!
//! Sources (CDK 2.13): `smarts/{Smarts,SmartsPattern}.java`,
//! `isomorphism/{DfPattern,DfState,Mappings,QueryStereoFilter,ComponentFilter}.java`,
//! `isomorphism/matchers/Expr.java`.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::cdk_daylight::{CdkMol, smarts_daylight_aromaticity};
use super::context::molecule_cache_key;
use crate::mol::atom::ChiralTag;
use crate::mol::molecule::Molecule;
use crate::types::BondType;

// ===========================================================================
// Expressions
// ===========================================================================

/// `IStereoElement.LEFT` (`@`; for bonds `OPPOSITE`).
const LEFT: u8 = 1;
/// `IStereoElement.RIGHT` (`@@`; for bonds `TOGETHER`).
const RIGHT: u8 = 2;
/// `Expr.UNKNOWN_STEREO`: stereo is not checked during the search.
const UNKNOWN_STEREO: i32 = -1;

/// Atom expression (subset of CDK `Expr.Type` used by the key sets).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum AExpr {
    True,
    False,
    /// `ELEMENT`
    Elem(u8),
    /// `ALIPHATIC_ELEMENT`
    Aliph(u8),
    /// `AROMATIC_ELEMENT`
    Arom(u8),
    IsArom,
    IsAliph,
    InRing,
    InChain,
    /// `HAS_IMPLICIT_HYDROGEN`
    HasImplH,
    /// `IMPL_H_COUNT`
    ImplH(u8),
    /// `TOTAL_H_COUNT`
    TotalH(u8),
    /// `DEGREE`
    Degree(u8),
    /// `TOTAL_DEGREE`
    TotalDegree(u8),
    Valence(u8),
    Isotope(u16),
    /// `HAS_UNSPEC_ISOTOPE`
    UnspecIsotope,
    Charge(i8),
    RingCount(u8),
    RingSmallest(u8),
    RingBondCount(u8),
    /// `STEREOCHEMISTRY` (`LEFT`/`RIGHT`, 0 = unspecified)
    Stereo(u8),
    And(Box<AExpr>, Box<AExpr>),
    Or(Box<AExpr>, Box<AExpr>),
    Not(Box<AExpr>),
    /// `RECURSIVE`: index of the sub-query.
    Rec(u32),
    /// Implied by the query graph (not SMARTS): at least this many neighbours.
    MinDegree(u8),
    /// Implied by the query graph (not SMARTS): at least this many ring bonds.
    MinRingBonds(u8),
}

/// Bond expression (subset of CDK `Expr.Type`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum BExpr {
    True,
    /// `ALIPHATIC_ORDER`
    AliphOrder(u8),
    /// `ORDER`
    Order(u8),
    IsArom,
    InRing,
    InChain,
    SingleOrArom,
    /// `STEREOCHEMISTRY`: `b'/'` or `b'\\'` as parsed, then the double-bond
    /// configuration (`TOGETHER` = `RIGHT`, `OPPOSITE` = `LEFT`).
    Stereo(u8),
    And(Box<BExpr>, Box<BExpr>),
    Or(Box<BExpr>, Box<BExpr>),
    Not(Box<BExpr>),
}

fn and(l: AExpr, r: AExpr) -> AExpr {
    match (l, r) {
        (AExpr::True, x) | (x, AExpr::True) => x,
        (l, r) => AExpr::And(Box::new(l), Box::new(r)),
    }
}

fn negate(e: AExpr) -> AExpr {
    match e {
        AExpr::True => AExpr::False,
        AExpr::False => AExpr::True,
        AExpr::IsArom => AExpr::IsAliph,
        AExpr::IsAliph => AExpr::IsArom,
        AExpr::InRing => AExpr::InChain,
        AExpr::InChain => AExpr::InRing,
        AExpr::Not(x) => *x,
        e => AExpr::Not(Box::new(e)),
    }
}

/// `Smarts.Parser.hasAliphaticDoubleBond`.
fn has_aliphatic_double(e: &BExpr) -> bool {
    match e {
        BExpr::Not(x) => has_aliphatic_double(x),
        BExpr::And(l, r) | BExpr::Or(l, r) => has_aliphatic_double(l) || has_aliphatic_double(r),
        BExpr::AliphOrder(2) => true,
        _ => false,
    }
}

/// `Smarts.Parser.strip(expr, STEREOCHEMISTRY)`.
fn strip_stereo(e: BExpr) -> Option<BExpr> {
    match e {
        BExpr::Stereo(_) => None,
        BExpr::And(l, r) => match (strip_stereo(*l), strip_stereo(*r)) {
            (Some(l), Some(r)) => Some(BExpr::And(Box::new(l), Box::new(r))),
            (x, None) | (None, x) => x,
        },
        BExpr::Or(l, r) => match (strip_stereo(*l), strip_stereo(*r)) {
            (Some(l), Some(r)) => Some(BExpr::Or(Box::new(l), Box::new(r))),
            (x, None) | (None, x) => x,
        },
        BExpr::Not(x) => strip_stereo(*x).map(|x| BExpr::Not(Box::new(x))),
        e => Some(e),
    }
}

// ===========================================================================
// Parser (CDK `Smarts.Parser`)
// ===========================================================================

/// A parsed query graph (before compilation).
#[derive(Debug, Default)]
struct Parsed {
    atoms: Vec<AExpr>,
    /// Component group of each atom (0 = none).
    comp: Vec<u8>,
    /// `(begin, end, expression)`
    bonds: Vec<(u8, u8, BExpr)>,
    /// Tetrahedral query stereo: focus and its four ligands (the focus itself
    /// stands for an implicit neighbour).
    tetra: Vec<(u8, [u8; 4])>,
    /// Double-bond query stereo: (stereo bond, left carrier bond, right carrier bond).
    geo: Vec<(u8, u8, u8)>,
}

/// Recursive sub-queries collected while parsing (shared by text).
#[derive(Default)]
struct SubQueries {
    parsed: Vec<Parsed>,
    by_text: FxHashMap<String, u32>,
}

struct Parser<'s, 'q> {
    s: &'s [u8],
    pos: usize,
    subs: &'q mut SubQueries,
    out: Parsed,
    /// Ordered local bonds of each atom (CDK `LocalNbrs`) and whether the atom
    /// started a fragment.
    local: Vec<(SmallVec<[usize; 4]>, bool)>,
    /// Bonds whose expression is still open (ring openings without expression).
    open_expr: Vec<bool>,
    astereo: Vec<usize>,
    bstereo: Vec<usize>,
    prev: Option<usize>,
    bond: Option<BExpr>,
    has_bond: bool,
    stack: Vec<usize>,
    rings: [Option<usize>; 100],
    n_comp: u8,
    cur_comp: u8,
}

type PResult<T> = Result<T, String>;

impl<'s, 'q> Parser<'s, 'q> {
    fn new(s: &'s str, subs: &'q mut SubQueries) -> Self {
        Self {
            s: s.as_bytes(),
            pos: 0,
            subs,
            out: Parsed::default(),
            local: Vec::new(),
            open_expr: Vec::new(),
            astereo: Vec::new(),
            bstereo: Vec::new(),
            prev: None,
            bond: None,
            has_bond: false,
            stack: Vec::new(),
            rings: [None; 100],
            n_comp: 0,
            cur_comp: 0,
        }
    }

    fn peek(&self) -> u8 {
        self.s.get(self.pos).copied().unwrap_or(0)
    }

    fn next(&mut self) -> u8 {
        let c = self.peek();
        self.pos += 1;
        c
    }

    fn uint(&mut self) -> Option<u32> {
        if !self.peek().is_ascii_digit() {
            return None;
        }
        let mut v = 0u32;
        while self.peek().is_ascii_digit() {
            v = v * 10 + (self.next() - b'0') as u32;
        }
        Some(v)
    }

    fn err<T>(&self, msg: &str) -> PResult<T> {
        Err(format!("{msg} at {} in {}", self.pos, String::from_utf8_lossy(self.s)))
    }

    fn parse(mut self) -> PResult<Parsed> {
        while self.pos < self.s.len() {
            let c = self.next();
            match c {
                b'*' => self.append(AExpr::True)?,
                b'A' => self.append(AExpr::IsAliph)?,
                b'B' => {
                    if self.peek() == b'r' {
                        self.pos += 1;
                        self.append(AExpr::Elem(35))?;
                    } else {
                        self.append(AExpr::Aliph(5))?;
                    }
                }
                b'C' => {
                    if self.peek() == b'l' {
                        self.pos += 1;
                        self.append(AExpr::Elem(17))?;
                    } else {
                        self.append(AExpr::Aliph(6))?;
                    }
                }
                b'N' => self.append(AExpr::Aliph(7))?,
                b'O' => self.append(AExpr::Aliph(8))?,
                b'P' => self.append(AExpr::Aliph(15))?,
                b'S' => self.append(AExpr::Aliph(16))?,
                b'F' => self.append(AExpr::Elem(9))?,
                b'I' => self.append(AExpr::Elem(53))?,
                b'a' => self.append(AExpr::IsArom)?,
                b'b' => self.append(AExpr::Arom(5))?,
                b'c' => self.append(AExpr::Arom(6))?,
                b'n' => self.append(AExpr::Arom(7))?,
                b'o' => self.append(AExpr::Arom(8))?,
                b'p' => self.append(AExpr::Arom(15))?,
                b's' => self.append(AExpr::Arom(16))?,
                b'[' => {
                    let atom_idx = self.out.atoms.len();
                    let e = self.bracket_atom(atom_idx)?;
                    self.append(e)?;
                }
                b'.' => self.prev = None,
                b'-' | b'=' | b'#' | b'$' | b':' | b'@' | b'~' | b'!' | b'/' | b'\\' => {
                    if self.prev.is_none() {
                        return self.err("bond without previous atom");
                    }
                    self.pos -= 1;
                    let e = self.bond_expr(0)?;
                    self.bond = Some(e);
                    self.has_bond = true;
                }
                b'(' => {
                    if let Some(prev) = self.prev {
                        if self.has_bond {
                            return self.err("bond before branch");
                        }
                        self.stack.push(prev);
                    } else {
                        self.n_comp += 1;
                        self.cur_comp = self.n_comp;
                    }
                }
                b')' => {
                    if let Some(p) = self.stack.pop() {
                        if self.has_bond {
                            return self.err("bond before branch close");
                        }
                        self.prev = Some(p);
                    } else {
                        if self.cur_comp == 0 {
                            return self.err("closing unopened component group");
                        }
                        self.cur_comp = 0;
                    }
                }
                b'0'..=b'9' => self.ring_closure((c - b'0') as usize)?,
                b'%' => {
                    let (d1, d2) = (self.next(), self.next());
                    if !d1.is_ascii_digit() || !d2.is_ascii_digit() {
                        return self.err("bad ring number");
                    }
                    self.ring_closure(((d1 - b'0') * 10 + (d2 - b'0')) as usize)?;
                }
                _ => return self.err("unexpected character"),
            }
        }
        self.finish()
    }

    /// `Parser.append`
    fn append(&mut self, e: AExpr) -> PResult<()> {
        let idx = self.out.atoms.len();
        if idx >= 250 {
            return self.err("query too large");
        }
        self.out.atoms.push(e);
        self.out.comp.push(self.cur_comp);
        if let Some(prev) = self.prev {
            let expr = self.bond.take().unwrap_or(BExpr::SingleOrArom);
            self.has_bond = false;
            let b = self.out.bonds.len();
            self.out.bonds.push((prev as u8, idx as u8, expr));
            self.open_expr.push(false);
            if self.bstereo.last() == Some(&usize::MAX) {
                *self.bstereo.last_mut().unwrap() = b;
            }
            self.local[prev].0.push(b);
            self.local.push((SmallVec::from_slice(&[b]), false));
        } else {
            self.local.push((SmallVec::new(), true));
        }
        self.prev = Some(idx);
        Ok(())
    }

    fn ring_closure(&mut self, rnum: usize) -> PResult<()> {
        let Some(prev) = self.prev else { return self.err("ring closure without atom") };
        match self.rings[rnum].take() {
            None => {
                // open: the bond expression (if any) is stored on the bond
                let b = self.out.bonds.len();
                let expr = self.bond.take();
                self.has_bond = false;
                self.open_expr.push(expr.is_none());
                if self.bstereo.last() == Some(&usize::MAX) {
                    *self.bstereo.last_mut().unwrap() = b;
                }
                self.out.bonds.push((prev as u8, u8::MAX, expr.unwrap_or(BExpr::True)));
                self.local[prev].0.push(b);
                self.rings[rnum] = Some(b);
            }
            Some(b) => {
                let close = self.bond.take();
                self.has_bond = false;
                if let Some(close) = close {
                    if self.open_expr[b] {
                        self.out.bonds[b].2 = close;
                        if self.bstereo.last() == Some(&usize::MAX) {
                            *self.bstereo.last_mut().unwrap() = b;
                        }
                    } else if self.out.bonds[b].2 != close {
                        return self.err("open/close ring bond expressions differ");
                    }
                } else if self.open_expr[b] {
                    self.out.bonds[b].2 = BExpr::SingleOrArom;
                }
                self.open_expr[b] = false;
                self.out.bonds[b].1 = prev as u8;
                self.local[prev].0.push(b);
            }
        }
        Ok(())
    }

    /// Bracket atom (after `[`), CDK `parseExplicitHydrogen` / `parseAtomExpr`.
    fn bracket_atom(&mut self, atom_idx: usize) -> PResult<AExpr> {
        // [H], [2H], [H+], [H-2], ... are hydrogen atoms
        let mark = self.pos;
        let iso = self.uint();
        if self.peek() == b'H' {
            self.pos += 1;
            let mut e = match iso {
                None => AExpr::Elem(1),
                Some(i) => and(AExpr::Isotope(i as u16), AExpr::Elem(1)),
            };
            if matches!(self.peek(), b'+' | b'-') {
                let q = self.charge()?;
                e = and(e, AExpr::Charge(q));
            }
            if self.peek() == b']' {
                self.pos += 1;
                return Ok(e);
            }
        }
        self.pos = mark;
        let e = self.atom_low(atom_idx)?;
        if self.next() != b']' {
            return self.err("unclosed bracket atom");
        }
        Ok(e)
    }

    fn charge(&mut self) -> PResult<i8> {
        let sign = if self.next() == b'+' { 1i32 } else { -1 };
        let num = match self.uint() {
            Some(v) => v as i32,
            None => {
                let mut n = 1;
                let c = if sign > 0 { b'+' } else { b'-' };
                while self.peek() == c {
                    self.pos += 1;
                    n += 1;
                }
                n
            }
        };
        Ok((sign * num) as i8)
    }

    // precedence: `;` < `,` < `&`/implicit < `!`
    fn atom_low(&mut self, ai: usize) -> PResult<AExpr> {
        let mut e = self.atom_or(ai)?;
        while self.peek() == b';' {
            self.pos += 1;
            let r = self.atom_or(ai)?;
            e = and(e, r);
        }
        Ok(e)
    }

    fn atom_or(&mut self, ai: usize) -> PResult<AExpr> {
        let mut e = self.atom_and(ai)?;
        while self.peek() == b',' {
            self.pos += 1;
            let r = self.atom_and(ai)?;
            e = AExpr::Or(Box::new(e), Box::new(r));
        }
        Ok(e)
    }

    fn atom_and(&mut self, ai: usize) -> PResult<AExpr> {
        let mut e = self.atom_unary(ai)?;
        loop {
            match self.peek() {
                b'&' => {
                    self.pos += 1;
                    let r = self.atom_unary(ai)?;
                    e = and(e, r);
                }
                b',' | b';' | b']' | 0 => return Ok(e),
                _ => {
                    let r = self.atom_unary(ai)?;
                    e = and(e, r);
                }
            }
        }
    }

    fn atom_unary(&mut self, ai: usize) -> PResult<AExpr> {
        if self.peek() == b'!' {
            self.pos += 1;
            let e = self.atom_unary(ai)?;
            return Ok(negate(e));
        }
        self.atom_primitive(ai)
    }

    fn atom_primitive(&mut self, ai: usize) -> PResult<AExpr> {
        let c = self.next();
        let e = match c {
            b'*' => AExpr::True,
            b'A'..=b'Z' => {
                let l = self.peek();
                let two = l.is_ascii_lowercase()
                    .then(|| std::str::from_utf8(&[c, l]).ok().and_then(crate::mol::elements::symbol_to_atomic_number))
                    .flatten();
                if let Some(z) = two {
                    self.pos += 1;
                    match z {
                        13 | 14 | 32 | 33 | 34 | 51 | 52 => AExpr::Aliph(z),
                        _ => AExpr::Elem(z),
                    }
                } else {
                    match c {
                        b'A' => AExpr::IsAliph,
                        b'D' => AExpr::Degree(self.uint().unwrap_or(1) as u8),
                        b'H' => AExpr::TotalH(self.uint().unwrap_or(1) as u8),
                        b'X' => AExpr::TotalDegree(self.uint().unwrap_or(1) as u8),
                        b'R' => match self.uint() {
                            None => AExpr::InRing,
                            Some(0) => AExpr::InChain,
                            Some(n) => AExpr::RingCount(n as u8),
                        },
                        b'B' | b'C' | b'N' | b'O' | b'P' | b'S' => {
                            AExpr::Aliph(crate::mol::elements::symbol_to_atomic_number(
                                std::str::from_utf8(&[c]).unwrap(),
                            )
                            .unwrap())
                        }
                        _ => match crate::mol::elements::symbol_to_atomic_number(std::str::from_utf8(&[c]).unwrap()) {
                            Some(z) => AExpr::Elem(z),
                            None => return self.err("unknown element"),
                        },
                    }
                }
            }
            b'a' => match self.peek() {
                b'l' => {
                    self.pos += 1;
                    AExpr::Arom(13)
                }
                b's' => {
                    self.pos += 1;
                    AExpr::Arom(33)
                }
                _ => AExpr::IsArom,
            },
            b'b' => AExpr::Arom(5),
            b'c' => AExpr::Arom(6),
            b'n' => AExpr::Arom(7),
            b'o' => AExpr::Arom(8),
            b'p' => AExpr::Arom(15),
            b's' => match self.peek() {
                b'b' => {
                    self.pos += 1;
                    AExpr::Arom(51)
                }
                b'e' => {
                    self.pos += 1;
                    AExpr::Arom(34)
                }
                b'i' => {
                    self.pos += 1;
                    AExpr::Arom(14)
                }
                _ => AExpr::Arom(16),
            },
            b't' if self.peek() == b'e' => {
                self.pos += 1;
                AExpr::Arom(52)
            }
            b'r' => match self.uint() {
                None => AExpr::InRing,
                Some(0) => AExpr::InChain,
                Some(n) if n > 2 => AExpr::RingSmallest(n as u8),
                Some(_) => return self.err("r1/r2"),
            },
            b'v' => AExpr::Valence(self.uint().unwrap_or(1) as u8),
            b'h' => match self.uint() {
                None => AExpr::HasImplH,
                Some(n) => AExpr::ImplH(n as u8),
            },
            b'x' => match self.uint() {
                None => AExpr::InRing,
                Some(0) => AExpr::InChain,
                Some(n) if n > 1 => AExpr::RingBondCount(n as u8),
                Some(_) => return self.err("x1"),
            },
            b'#' => match self.uint() {
                Some(z) => AExpr::Elem(z as u8),
                None => return self.err("#X/#G not supported"),
            },
            b'0'..=b'9' => {
                self.pos -= 1;
                match self.uint().unwrap() {
                    0 => AExpr::UnspecIsotope,
                    i => AExpr::Isotope(i as u16),
                }
            }
            b'+' | b'-' => {
                self.pos -= 1;
                AExpr::Charge(self.charge()?)
            }
            b'@' => {
                let mut v = LEFT;
                if self.peek() == b'@' {
                    self.pos += 1;
                    v = RIGHT;
                } else if [b"TH", b"AL", b"SP", b"TB", b"OH"].iter().any(|p| self.s[self.pos..].starts_with(*p)) {
                    return self.err("@TH/@AL/... not supported");
                }
                let mut e = AExpr::Stereo(v);
                if self.peek() == b'?' {
                    self.pos += 1;
                    e = AExpr::Or(Box::new(e), Box::new(AExpr::Stereo(0)));
                }
                if !self.astereo.contains(&ai) {
                    self.astereo.push(ai);
                }
                e
            }
            b'$' => {
                if self.next() != b'(' {
                    return self.err("$ without (");
                }
                let beg = self.pos;
                let mut depth = 1;
                let mut end = beg;
                while end < self.s.len() {
                    match self.s[end] {
                        b'(' => depth += 1,
                        b')' => depth -= 1,
                        _ => {}
                    }
                    end += 1;
                    if depth == 0 {
                        break;
                    }
                }
                if depth != 0 {
                    return self.err("unclosed recursive SMARTS");
                }
                let text = std::str::from_utf8(&self.s[beg..end - 1]).unwrap().to_string();
                self.pos = end;
                if let Some(&id) = self.subs.by_text.get(&text) {
                    let p = &self.subs.parsed[id as usize];
                    if p.atoms.len() == 1 { p.atoms[0].clone() } else { AExpr::Rec(id) }
                } else {
                    let parsed = Parser::new(&text, self.subs).parse()?;
                    let single = (parsed.atoms.len() == 1).then(|| parsed.atoms[0].clone());
                    let id = self.subs.parsed.len() as u32;
                    self.subs.parsed.push(parsed);
                    self.subs.by_text.insert(text, id);
                    single.unwrap_or(AExpr::Rec(id))
                }
            }
            b':' => {
                // atom map: ignored
                if self.uint().is_none() {
                    return self.err("bad atom map");
                }
                AExpr::True
            }
            _ => return self.err("bad atom primitive"),
        };
        Ok(e)
    }

    fn bond_expr(&mut self, depth: u8) -> PResult<BExpr> {
        // `;` < `,` < `&`/implicit < `!`
        let _ = depth;
        let mut e = self.bond_or()?;
        while self.peek() == b';' {
            self.pos += 1;
            let r = self.bond_or()?;
            e = BExpr::And(Box::new(e), Box::new(r));
        }
        Ok(e)
    }

    fn bond_or(&mut self) -> PResult<BExpr> {
        let mut e = self.bond_and()?;
        while self.peek() == b',' {
            self.pos += 1;
            let r = self.bond_and()?;
            e = BExpr::Or(Box::new(e), Box::new(r));
        }
        Ok(e)
    }

    fn is_bond_char(c: u8) -> bool {
        matches!(c, b'-' | b'=' | b'#' | b'$' | b':' | b'~' | b'@' | b'!' | b'/' | b'\\')
    }

    fn bond_and(&mut self) -> PResult<BExpr> {
        let mut e = self.bond_unary()?;
        loop {
            match self.peek() {
                b'&' => {
                    self.pos += 1;
                    let r = self.bond_unary()?;
                    e = BExpr::And(Box::new(e), Box::new(r));
                }
                c if Self::is_bond_char(c) => {
                    let r = self.bond_unary()?;
                    e = BExpr::And(Box::new(e), Box::new(r));
                }
                _ => return Ok(e),
            }
        }
    }

    fn bond_unary(&mut self) -> PResult<BExpr> {
        let c = self.next();
        Ok(match c {
            b'!' => {
                let x = self.bond_unary()?;
                match x {
                    BExpr::InRing => BExpr::InChain,
                    BExpr::InChain => BExpr::InRing,
                    BExpr::Not(x) => *x,
                    x => BExpr::Not(Box::new(x)),
                }
            }
            b'-' => BExpr::AliphOrder(1),
            b'=' => BExpr::AliphOrder(2),
            b'#' => BExpr::AliphOrder(3),
            b'$' => BExpr::AliphOrder(4),
            b':' => BExpr::IsArom,
            b'~' => BExpr::True,
            b'@' => BExpr::InRing,
            b'/' | b'\\' => {
                if self.peek() == b'?' {
                    return self.err("unspecified bond direction not supported");
                }
                // the bond this expression belongs to is created next
                if self.bstereo.last() != Some(&usize::MAX) {
                    self.bstereo.push(usize::MAX);
                }
                BExpr::Stereo(c)
            }
            _ => return self.err("bad bond primitive"),
        })
    }

    /// `Parser.finish`: query stereo elements.
    fn finish(mut self) -> PResult<Parsed> {
        if self.rings.iter().any(|r| r.is_some()) || self.cur_comp != 0 || !self.stack.is_empty() || self.has_bond {
            return self.err("unclosed ring, component group or branch");
        }
        let other = |bonds: &[(u8, u8, BExpr)], b: usize, a: usize| {
            let (x, y, _) = &bonds[b];
            if *x as usize == a { *y } else { *x }
        };
        for &a in &self.astereo {
            let (ref lb, is_first) = self.local[a];
            let degree = lb.len();
            if degree != 3 && degree != 4 {
                continue; // no stereo element: `@` matches any atom
            }
            let input: SmallVec<[u8; 4]> = lb.iter().map(|&b| other(&self.out.bonds, b, a)).collect();
            let mut lig = [a as u8; 4];
            if degree == 4 {
                lig.copy_from_slice(&input);
            } else {
                // insertImplicitRefs: the focus stands for the implicit neighbour
                let mut src = 0;
                let mut dst = 0;
                if !is_first {
                    lig[dst] = input[src];
                    dst += 1;
                    src += 1;
                }
                lig[dst] = a as u8;
                dst += 1;
                while src < degree {
                    lig[dst] = input[src];
                    dst += 1;
                    src += 1;
                }
            }
            self.out.tetra.push((a as u8, lig));
        }
        let bstereo: Vec<usize> = self.bstereo.iter().copied().filter(|&b| b != usize::MAX).collect();
        if !bstereo.is_empty() {
            for b in 0..self.out.bonds.len() {
                if !has_aliphatic_double(&self.out.bonds[b].2) {
                    continue;
                }
                let (beg, end) = (self.out.bonds[b].0 as usize, self.out.bonds[b].1 as usize);
                let left = self.local[beg].0.iter().rev().copied().find(|x| bstereo.contains(x));
                let right = self.local[end].0.iter().rev().copied().find(|x| bstereo.contains(x));
                let (Some(left), Some(right)) = (left, right) else { continue };
                let dir = |e: &BExpr| match e {
                    BExpr::Stereo(d) => Some(*d),
                    _ => None,
                };
                let (Some(dl), Some(dr)) = (dir(&self.out.bonds[left].2), dir(&self.out.bonds[right].2)) else {
                    return self.err("complex bond stereo not supported");
                };
                let mut cfg = if dl == dr { RIGHT } else { LEFT }; // TOGETHER : OPPOSITE
                if (self.out.bonds[left].0 as usize == beg) != (self.out.bonds[right].0 as usize == end) {
                    cfg ^= 3;
                }
                let e = std::mem::replace(&mut self.out.bonds[b].2, BExpr::True);
                self.out.bonds[b].2 = BExpr::And(Box::new(e), Box::new(BExpr::Stereo(cfg)));
                self.out.geo.push((b as u8, left as u8, right as u8));
            }
            for &b in &bstereo {
                let e = std::mem::replace(&mut self.out.bonds[b].2, BExpr::True);
                self.out.bonds[b].2 = match strip_stereo(e) {
                    None => BExpr::Order(1),
                    Some(e) => BExpr::And(Box::new(e), Box::new(BExpr::Order(1))),
                };
            }
        }
        Ok(self.out)
    }
}

// ===========================================================================
// Compiled queries
// ===========================================================================

/// A query compiled for matching.
#[derive(Debug)]
struct Query {
    n: usize,
    /// Interned atom expression of each query atom (with the implied ring constraint).
    atom_expr: Vec<u32>,
    /// Bond expression of each query bond (with the implied ring constraint).
    bond_expr: Vec<u16>,
    /// Neighbours (atom, bond expression).
    nbrs: Vec<SmallVec<[(u8, u16); 4]>>,
    /// Static selectivity of each atom (lower = rarer).
    weight: Vec<u32>,
    /// Query distances (`n x n`).
    qd: Vec<u8>,
    /// Whether automorphic mappings may be skipped when counting (no stereo or
    /// component filters, which could tell them apart).
    symmetry_ok: bool,
    /// Distinct atom expressions with the number of query atoms using them and
    /// the first atom using them: cheap (local) expressions first, each group
    /// most selective first (screen, start atoms).
    need: Vec<(u32, u8, u8)>,
    /// Search plans by start atom (one per entry of `need`), built on first use;
    /// a recursive query always starts at atom 0 (`root_plan`).
    plans: Vec<std::sync::OnceLock<Plan>>,
    root_plan: Option<Box<Plan>>,
    /// Bond screen: distinct (atom, bond, atom) expression triples (see
    /// [`BondScreen`]) with the number of query bonds of that kind.
    bond_need: Vec<(u32, u8)>,
    tetra: Vec<(u8, [u8; 4])>,
    /// (begin, end, bond expr, left carrier (begin side atom), right carrier (end side atom))
    geo: Vec<(u8, u8, u16, u8, u8)>,
    /// Component group per atom, `None` without grouping.
    comp: Option<Vec<u8>>,
}

/// The placement order of a search and what to check at each position.
#[derive(Debug)]
struct Plan {
    steps: Box<[Step]>,
    /// Variable-length data of the steps (see [`Step`]).
    extra: Box<[u8]>,
}

/// What the search does at one position of a [`Plan`] (compact: the plans
/// of all keys are visited once per molecule, so they should be small).
#[derive(Debug, Clone, Copy)]
struct Step {
    /// Atom expression of the query atom placed here.
    expr: u32,
    /// The query atom.
    atom: u8,
    /// The already placed neighbour this atom is reached from (position;
    /// `NONE` for a component start) and the bond expression to it.
    from: u8,
    be: u8,
    /// Numbers of entries in `Plan::extra` from `extra` on: earlier positions
    /// whose target atoms must have smaller indices (count mode only; symmetry
    /// breaking, see [`Query::build_plan`]), then further bonds to placed
    /// atoms (position, bond expression), then distance checks (position,
    /// largest target distance = their query distance).
    n_lower: u8,
    n_clos: u8,
    n_dist: u8,
    extra: u16,
}

impl Step {
    #[inline(always)]
    fn lower<'p>(&self, extra: &'p [u8]) -> &'p [u8] {
        let o = self.extra as usize;
        &extra[o..o + self.n_lower as usize]
    }

    #[inline(always)]
    fn closures<'p>(&self, extra: &'p [u8]) -> &'p [u8] {
        let o = self.extra as usize + self.n_lower as usize;
        &extra[o..o + 2 * self.n_clos as usize]
    }

    #[inline(always)]
    fn dist<'p>(&self, extra: &'p [u8]) -> &'p [u8] {
        let o = self.extra as usize + self.n_lower as usize + 2 * self.n_clos as usize;
        &extra[o..o + 2 * self.n_dist as usize]
    }
}

const NONE: u8 = u8::MAX;

impl Query {
    fn has_filters(&self) -> bool {
        !self.tetra.is_empty() || !self.geo.is_empty() || self.comp.is_some()
    }

    /// Plan starting at the first atom of `need[i]`.
    fn plan(&self, i: usize) -> &Plan {
        self.plans[i].get_or_init(|| self.build_plan(self.need[i].2 as usize))
    }

    /// Placement: start at `start`, then always extend with the rarest
    /// frontier atom; further components start at their rarest atom.
    fn build_plan(&self, start: usize) -> Plan {
        let n = self.n;
        let w = &self.weight;
        let nbrs = &self.nbrs;
        let mut pos_of = vec![u8::MAX; n];
        let mut order: Vec<u8> = Vec::with_capacity(n);
        let mut anchor = Vec::with_capacity(n);
        while order.len() < n {
            let s = if order.is_empty() {
                start
            } else {
                (0..n).filter(|&a| pos_of[a] == u8::MAX).min_by_key(|&a| (w[a], a)).unwrap()
            };
            pos_of[s] = order.len() as u8;
            order.push(s as u8);
            anchor.push(None);
            loop {
                let mut best: Option<(usize, u8, u16)> = None;
                for &placed in &order {
                    for &(nb, be) in &nbrs[placed as usize] {
                        let nb = nb as usize;
                        if pos_of[nb] != u8::MAX {
                            continue;
                        }
                        let better = match best {
                            None => true,
                            Some((b, _, _)) => (w[nb], nb) < (w[b], b),
                        };
                        if better {
                            best = Some((nb, pos_of[placed as usize], be));
                        }
                    }
                }
                let Some((a, from, be)) = best else { break };
                pos_of[a] = order.len() as u8;
                order.push(a as u8);
                anchor.push(Some((from, be)));
            }
        }
        let mut closures: Vec<SmallVec<[(u8, u16); 2]>> = vec![SmallVec::new(); n];
        for (k, &a) in order.iter().enumerate() {
            let used_anchor = anchor[k];
            let mut anchor_seen = false;
            for &(nb, be) in &nbrs[a as usize] {
                let pn = pos_of[nb as usize];
                if (pn as usize) >= k {
                    continue;
                }
                if !anchor_seen && used_anchor == Some((pn, be)) {
                    anchor_seen = true;
                    continue;
                }
                closures[k].push((pn, be));
            }
        }
        // distance checks: a placed pair can be at most as far apart in the
        // target as in the query; only pairs whose query distance is shorter
        // than their path in the placement tree add information
        let tree_edges: Vec<(u8, u8)> =
            (0..n).filter_map(|k| anchor[k].map(|(from, _)| (order[k], order[from as usize]))).collect();
        let td = query_distances(n, &tree_edges);
        let mut dist_checks: Vec<SmallVec<[(u8, u8); 2]>> = vec![SmallVec::new(); n];
        for k in 0..n {
            let a = order[k] as usize;
            let mut cands: SmallVec<[(u8, u8); 8]> = SmallVec::new();
            for j in 0..k {
                let b = order[j] as usize;
                let (q, t) = (self.qd[a * n + b], td[a * n + b]);
                if q >= 2 && q < t {
                    cands.push((q, j as u8));
                }
            }
            cands.sort_unstable();
            dist_checks[k] = cands.iter().take(3).map(|&(q, j)| (j, q)).collect();
        }
        // Symmetry breaking for counting: mappings that differ by a query
        // automorphism have the same atom set, so only the lexicographic leader
        // (in placement order) of each class is needed. That is the mapping
        // with m(order[j]) < m(x) for every j and every x != order[j] in the
        // orbit of order[j] under the automorphisms fixing order[..j]
        // (stabilizer chain); x is placed after j, where the bound is checked.
        let mut lower: Vec<SmallVec<[u8; 2]>> = vec![SmallVec::new(); n];
        let connected = anchor.iter().skip(1).all(|a| a.is_some());
        if self.symmetry_ok && connected {
            for j in 0..n {
                let a = order[j] as usize;
                for k in j + 1..n {
                    let x = order[k] as usize;
                    if self.atom_expr[x] == self.atom_expr[a]
                        && self.nbrs[x].len() == self.nbrs[a].len()
                        && self.automorphism_exists(&order, &anchor, &closures, j, x)
                    {
                        lower[k].push(j as u8);
                    }
                }
            }
        }
        let mut extra: Vec<u8> = Vec::new();
        let steps: Vec<Step> = (0..n)
            .map(|k| {
                let off = extra.len() as u16;
                extra.extend_from_slice(&lower[k]);
                for &(p, e) in &closures[k] {
                    extra.extend_from_slice(&[p, e as u8]);
                }
                for &(p, d) in &dist_checks[k] {
                    extra.extend_from_slice(&[p, d]);
                }
                Step {
                    expr: self.atom_expr[order[k] as usize],
                    atom: order[k],
                    from: anchor[k].map_or(NONE, |a| a.0),
                    be: anchor[k].map_or(0, |a| a.1 as u8),
                    n_lower: lower[k].len() as u8,
                    n_clos: closures[k].len() as u8,
                    n_dist: dist_checks[k].len() as u8,
                    extra: off,
                }
            })
            .collect();
        Plan { steps: steps.into_boxed_slice(), extra: extra.into_boxed_slice() }
    }

    /// Whether a query automorphism fixes `order[..j]` and maps `order[j]` to `x`.
    fn automorphism_exists(
        &self,
        order: &[u8],
        anchor: &[Option<(u8, u16)>],
        closures: &[SmallVec<[(u8, u16); 2]>],
        j: usize,
        x: usize,
    ) -> bool {
        let n = self.n;
        let bond = |a: usize, b: usize| self.nbrs[a].iter().find(|&&(w, _)| w as usize == b).map(|&(_, e)| e);
        // image of each position
        let mut img = vec![u8::MAX; n];
        let mut used = vec![false; n];
        for i in 0..j {
            img[i] = order[i];
            used[order[i] as usize] = true;
        }
        if used[x] {
            return false;
        }
        let fits = |k: usize, c: usize, img: &[u8]| {
            closures[k].iter().all(|&(p, pe)| bond(c, img[p as usize] as usize) == Some(pe))
                && anchor[k].is_none_or(|(from, be)| bond(c, img[from as usize] as usize) == Some(be))
        };
        if !fits(j, x, &img) {
            return false;
        }
        img[j] = x as u8;
        used[x] = true;
        fn rec(
            q: &Query,
            k: usize,
            order: &[u8],
            anchor: &[Option<(u8, u16)>],
            img: &mut [u8],
            used: &mut [bool],
            fits: &dyn Fn(usize, usize, &[u8]) -> bool,
        ) -> bool {
            if k == q.n {
                return true;
            }
            let a = order[k] as usize;
            let (from, _) = anchor[k].expect("connected");
            let src = img[from as usize] as usize;
            for &(c, _) in &q.nbrs[src] {
                let c = c as usize;
                if used[c] || q.atom_expr[c] != q.atom_expr[a] || q.nbrs[c].len() != q.nbrs[a].len() || !fits(k, c, img) {
                    continue;
                }
                img[k] = c as u8;
                used[c] = true;
                if rec(q, k + 1, order, anchor, img, used, fits) {
                    return true;
                }
                used[c] = false;
            }
            img[k] = u8::MAX;
            false
        }
        rec(self, j + 1, order, anchor, &mut img, &mut used, &fits)
    }
}

/// Rough static selectivity of an atom expression (lower = rarer); only used
/// to choose the placement order.
fn weight(e: &AExpr) -> u32 {
    match e {
        AExpr::True => 1000,
        AExpr::False => 0,
        AExpr::Elem(z) | AExpr::Aliph(z) | AExpr::Arom(z) => {
            match (z, e) {
                (6, AExpr::Elem(_)) => 600,
                (6, AExpr::Aliph(_)) => 400,
                (6, AExpr::Arom(_)) => 350,
                (7, AExpr::Arom(_)) => 40,
                (7, _) => 100,
                (8, AExpr::Arom(_)) => 10,
                (8, _) => 120,
                (16, AExpr::Arom(_)) => 8,
                (16, _) => 30,
                (9, _) | (17, _) => 20,
                (35, _) | (15, _) => 8,
                (1, _) => 30,
                _ => 3,
            }
        }
        AExpr::IsArom => 400,
        AExpr::IsAliph => 700,
        AExpr::InRing => 500,
        AExpr::InChain => 600,
        AExpr::HasImplH => 600,
        AExpr::ImplH(h) | AExpr::TotalH(h) => match h {
            0 => 500,
            1 => 450,
            2 => 300,
            3 => 150,
            _ => 20,
        },
        AExpr::Degree(d) | AExpr::TotalDegree(d) => match d {
            1 => 300,
            2 | 3 => 500,
            4 => 300,
            _ => 20,
        },
        AExpr::Valence(_) => 500,
        AExpr::Isotope(_) => 1,
        AExpr::UnspecIsotope => 1000,
        AExpr::Charge(0) => 950,
        AExpr::Charge(_) => 10,
        AExpr::RingCount(_) | AExpr::RingSmallest(_) | AExpr::RingBondCount(_) => 200,
        AExpr::Stereo(_) => 1000,
        AExpr::And(l, r) => weight(l).min(weight(r)) * 7 / 10,
        AExpr::Or(l, r) => (weight(l) + weight(r)).min(1000),
        AExpr::Not(x) => 1000u32.saturating_sub(weight(x)).max(50),
        AExpr::Rec(_) => 200,
        AExpr::MinDegree(d) => match d {
            0..=2 => 800,
            3 => 400,
            _ => 60,
        },
        AExpr::MinRingBonds(_) => 150,
    }
}

/// The compiled key set of one fingerprint.
pub(crate) struct KeySet {
    exprs: Vec<AExpr>,
    bexprs: Vec<BExpr>,
    /// Interned expression id -> needs per-atom stereo re-evaluation.
    keys: Vec<Query>,
    subs: Vec<Query>,
    /// Per atom expression: its node in `nodes` when it depends only on the
    /// atom's own properties (see [`is_local`]), `u32::MAX` otherwise.
    local: Vec<u32>,
    /// Hash-consed sub-expressions of the local atom expressions: primitives
    /// are evaluated once per class of equal atoms, the rest by bit operations
    /// on the masks of their children.
    nodes: Vec<Node>,
    /// Every atom expression with its local parts as nodes.
    xexprs: Vec<XExpr>,
    bond_screen: Option<BondScreen>,
    /// The screens of all keys in one array (cache friendly): key `i` uses
    /// `screen[screen_start[i]..screen_start[i + 1]]`, entries
    /// `id << 10 | kind << 8 | count` (`SCREEN_*` kinds).
    screen: Vec<u32>,
    screen_start: Vec<u32>,
    /// Number of atoms of each key.
    key_n: Vec<u8>,
}

/// Screen entry kinds: an entry of `Query::need` (atom expression), a bond
/// triple, a local bound of an expensive atom expression.
const SCREEN_NEED: u32 = 0;
const SCREEN_TRIPLE: u32 = 1;
const SCREEN_BOUND: u32 = 2;

/// A node of the local expression DAG (see [`KeySet::nodes`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Node {
    Prim(AExpr),
    And(u32, u32),
    Or(u32, u32),
    Not(u32),
    /// Always true (`!@` while stereo is unknown).
    All,
}

fn node_of(e: &AExpr, nodes: &mut Vec<Node>, ids: &mut FxHashMap<Node, u32>) -> u32 {
    let node = match e {
        AExpr::And(l, r) => Node::And(node_of(l, nodes, ids), node_of(r, nodes, ids)),
        AExpr::Or(l, r) => Node::Or(node_of(l, nodes, ids), node_of(r, nodes, ids)),
        AExpr::Not(x) if matches!(**x, AExpr::Stereo(_)) => Node::All,
        AExpr::Not(x) => Node::Not(node_of(x, nodes, ids)),
        prim => Node::Prim(prim.clone()),
    };
    if let Some(&id) = ids.get(&node) {
        return id;
    }
    let id = nodes.len() as u32;
    nodes.push(node.clone());
    ids.insert(node, id);
    id
}

/// An atom expression with its local parts replaced by DAG nodes.
#[derive(Debug)]
enum XExpr {
    Local(u32),
    And(Box<XExpr>, Box<XExpr>),
    Or(Box<XExpr>, Box<XExpr>),
    Not(Box<XExpr>),
    Rec(u32),
    /// A primitive that needs more than the atom's own properties.
    Prim(AExpr),
}

fn annotate(e: &AExpr, nodes: &mut Vec<Node>, ids: &mut FxHashMap<Node, u32>) -> XExpr {
    if is_local(e) {
        return XExpr::Local(node_of(e, nodes, ids));
    }
    match e {
        AExpr::And(l, r) => XExpr::And(Box::new(annotate(l, nodes, ids)), Box::new(annotate(r, nodes, ids))),
        AExpr::Or(l, r) => XExpr::Or(Box::new(annotate(l, nodes, ids)), Box::new(annotate(r, nodes, ids))),
        AExpr::Not(x) => XExpr::Not(Box::new(annotate(x, nodes, ids))),
        AExpr::Rec(id) => XExpr::Rec(*id),
        prim => XExpr::Prim(prim.clone()),
    }
}

/// A local expression matching every atom `e` matches (a cheap screen for
/// expressions with recursive SMARTS), if there is a useful one.
fn local_bound(e: &AExpr, int: &Interner, subs: &[Query]) -> Option<AExpr> {
    if is_local(e) {
        return Some(e.clone());
    }
    match e {
        AExpr::And(l, r) => match (local_bound(l, int, subs), local_bound(r, int, subs)) {
            (Some(a), Some(b)) => Some(and(a, b)),
            (x, None) | (None, x) => x,
        },
        AExpr::Or(l, r) => match (local_bound(l, int, subs), local_bound(r, int, subs)) {
            (Some(a), Some(b)) => Some(AExpr::Or(Box::new(a), Box::new(b))),
            _ => None,
        },
        // the root of a recursive SMARTS matches its first atom
        AExpr::Rec(id) => local_bound(&int.exprs[subs[*id as usize].atom_expr[0] as usize], int, subs),
        _ => None,
    }
}

/// Whether an expression only reads the atom's own properties (no recursive
/// SMARTS, ring counts or smallest-ring sizes).
fn is_local(e: &AExpr) -> bool {
    match e {
        AExpr::Rec(_) | AExpr::RingCount(_) | AExpr::RingSmallest(_) => false,
        AExpr::And(l, r) | AExpr::Or(l, r) => is_local(l) && is_local(r),
        AExpr::Not(x) => is_local(x),
        _ => true,
    }
}

struct Interner {
    exprs: Vec<AExpr>,
    by_expr: FxHashMap<AExpr, u32>,
    bexprs: Vec<BExpr>,
    by_bexpr: FxHashMap<BExpr, u16>,
}

impl Interner {
    fn atom(&mut self, e: &AExpr) -> u32 {
        if let Some(&id) = self.by_expr.get(e) {
            return id;
        }
        let id = self.exprs.len() as u32;
        self.exprs.push(e.clone());
        self.by_expr.insert(e.clone(), id);
        id
    }

    fn bond(&mut self, e: &BExpr) -> u16 {
        if let Some(&id) = self.by_bexpr.get(e) {
            return id;
        }
        let id = self.bexprs.len() as u16;
        self.bexprs.push(e.clone());
        self.by_bexpr.insert(e.clone(), id);
        id
    }
}

/// Shortest path lengths between the atoms of a query (`u8::MAX`: disconnected).
fn query_distances(n: usize, edges: &[(u8, u8)]) -> Vec<u8> {
    let mut adj: Vec<SmallVec<[u8; 4]>> = vec![SmallVec::new(); n];
    for &(a, b) in edges {
        adj[a as usize].push(b);
        adj[b as usize].push(a);
    }
    let mut d = vec![u8::MAX; n * n];
    let mut queue = Vec::with_capacity(n);
    for s in 0..n {
        let row = &mut d[s * n..(s + 1) * n];
        row[s] = 0;
        queue.clear();
        queue.push(s as u8);
        let mut head = 0;
        while head < queue.len() {
            let v = queue[head] as usize;
            head += 1;
            for &w in &adj[v] {
                if row[w as usize] == u8::MAX {
                    row[w as usize] = row[v] + 1;
                    queue.push(w);
                }
            }
        }
    }
    d
}

/// Whether each query bond lies on a cycle (is not a bridge).
fn query_ring_bonds(n: usize, bonds: &[(u8, u8, BExpr)]) -> Vec<bool> {
    // a bond is a ring bond iff its ends stay connected without it
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    (0..bonds.len())
        .map(|skip| {
            for (i, v) in parent.iter_mut().enumerate() {
                *v = i;
            }
            for (i, &(a, b, _)) in bonds.iter().enumerate() {
                if i != skip {
                    let (ra, rb) = (find(&mut parent, a as usize), find(&mut parent, b as usize));
                    parent[ra] = rb;
                }
            }
            let (a, b, _) = bonds[skip];
            find(&mut parent, a as usize) == find(&mut parent, b as usize)
        })
        .collect()
}

fn compile(p: &Parsed, int: &mut Interner, rooted: bool) -> Query {
    let n = p.atoms.len();
    // Implied ring constraints: a query bond on a query cycle can only map to
    // a ring bond (an injective mapping turns a cycle into a cycle), and its
    // atoms to ring atoms. They change no result but prune the search and
    // sharpen the screens.
    let ring_bond = query_ring_bonds(n, &p.bonds);
    let mut ring_atom = vec![false; n];
    for (b, &(x, y, _)) in p.bonds.iter().enumerate() {
        if ring_bond[b] {
            ring_atom[x as usize] = true;
            ring_atom[y as usize] = true;
        }
    }
    // likewise a query atom with d neighbours (k ring bonds) needs a target
    // atom with at least d neighbours (k ring bonds)
    let mut degree = vec![0u8; n];
    let mut ring_bonds = vec![0u8; n];
    for (b, &(x, y, _)) in p.bonds.iter().enumerate() {
        for a in [x, y] {
            degree[a as usize] += 1;
            ring_bonds[a as usize] += u8::from(ring_bond[b]);
        }
    }
    let atom_expr: Vec<u32> = p
        .atoms
        .iter()
        .enumerate()
        .map(|(a, e)| {
            let mut e = e.clone();
            if ring_atom[a] {
                e = and(e, AExpr::InRing);
            }
            if ring_bonds[a] >= 3 {
                e = and(e, AExpr::MinRingBonds(ring_bonds[a]));
            }
            if degree[a] >= 2 {
                e = and(e, AExpr::MinDegree(degree[a]));
            }
            int.atom(&e)
        })
        .collect();
    let bond_expr: Vec<u16> = p
        .bonds
        .iter()
        .enumerate()
        .map(|(b, (_, _, e))| {
            if ring_bond[b] {
                int.bond(&BExpr::And(Box::new(e.clone()), Box::new(BExpr::InRing)))
            } else {
                int.bond(e)
            }
        })
        .collect();
    let mut nbrs: Vec<SmallVec<[(u8, u16); 4]>> = vec![SmallVec::new(); n];
    for (b, (a, c, _)) in p.bonds.iter().enumerate() {
        nbrs[*a as usize].push((*c, bond_expr[b]));
        nbrs[*c as usize].push((*a, bond_expr[b]));
    }
    let w: Vec<u32> = atom_expr.iter().map(|&e| weight(&int.exprs[e as usize])).collect();
    let edges: Vec<(u8, u8)> = p.bonds.iter().map(|&(x, y, _)| (x, y)).collect();
    let qd = query_distances(n, &edges);
    let mut need: Vec<(u32, u8, u8)> = Vec::new();
    for (a, &e) in atom_expr.iter().enumerate() {
        match need.iter_mut().find(|x| x.0 == e) {
            Some(x) => x.1 += 1,
            None => need.push((e, 1, a as u8)),
        }
    }
    need.sort_by_key(|&(e, c, _)| (!is_local(&int.exprs[e as usize]), weight(&int.exprs[e as usize]) / c as u32, e));
    let other_end = |b: usize, a: u8| {
        let (x, y, _) = &p.bonds[b];
        if *x == a { *y } else { *x }
    };
    let geo = p
        .geo
        .iter()
        .map(|&(b, l, r)| {
            let (beg, end, _) = p.bonds[b as usize];
            (beg, end, bond_expr[b as usize], other_end(l as usize, beg), other_end(r as usize, end))
        })
        .collect();
    let comp = p.comp.iter().any(|&c| c != 0).then(|| p.comp.clone());
    let symmetry_ok = p.tetra.is_empty() && p.geo.is_empty() && comp.is_none();
    let mut q = Query {
        n,
        atom_expr,
        bond_expr,
        nbrs,
        weight: w,
        qd,
        symmetry_ok,
        plans: (0..need.len()).map(|_| std::sync::OnceLock::new()).collect(),
        need,
        root_plan: None,
        bond_need: Vec::new(),
        tetra: p.tetra.clone(),
        geo,
        comp,
    };
    if rooted {
        q.root_plan = Some(Box::new(q.build_plan(0)));
    }
    q
}

/// Screen on query bonds: every key bond is an (atom expression, bond
/// expression, atom expression) triple, and a molecule can only match a key
/// with `k` bonds of one triple if at least `k` of its bonds match that triple
/// (in either direction). Only triples of expressions without recursive SMARTS
/// are used, so their per-molecule counts are cheap.
struct BondScreen {
    /// Atom expression of each compact index.
    exprs: Vec<u32>,
    n_bexprs: usize,
    /// `(a * exprs.len() + b) * n_bexprs + bond expr` -> triple id (`u32::MAX`: none).
    table: Vec<u32>,
    n_triples: usize,
}

fn has_rec(e: &AExpr) -> bool {
    match e {
        AExpr::Rec(_) => true,
        AExpr::And(l, r) | AExpr::Or(l, r) => has_rec(l) || has_rec(r),
        AExpr::Not(x) => has_rec(x),
        _ => false,
    }
}

impl BondScreen {
    fn build(parsed: &[Parsed], keys: &mut [Query], int: &mut Interner) -> Self {
        let mut idx = vec![u16::MAX; int.exprs.len()];
        let mut exprs = Vec::new();
        let mut triples: FxHashMap<(u32, u16, u32), u32> = FxHashMap::default();
        let mut by_id: Vec<(u32, u16, u32)> = Vec::new();
        for (p, q) in parsed.iter().zip(keys.iter_mut()) {
            let mut need: Vec<(u32, u8)> = Vec::new();
            for (bi, (a, b, _)) in p.bonds.iter().enumerate() {
                let (ea, eb) = (q.atom_expr[*a as usize], q.atom_expr[*b as usize]);
                if has_rec(&int.exprs[ea as usize]) || has_rec(&int.exprs[eb as usize]) {
                    continue;
                }
                for x in [ea, eb] {
                    if idx[x as usize] == u16::MAX {
                        idx[x as usize] = exprs.len() as u16;
                        exprs.push(x);
                    }
                }
                let key = (ea.min(eb), q.bond_expr[bi], ea.max(eb));
                let n = triples.len() as u32;
                let tid = *triples.entry(key).or_insert_with(|| {
                    by_id.push(key);
                    n
                });
                match need.iter_mut().find(|x| x.0 == tid) {
                    Some(x) => x.1 += 1,
                    None => need.push((tid, 1)),
                }
            }
            // rarest first: by the static weight of the rarer end
            need.sort_by_key(|&(tid, c)| {
                let (ea, _, eb) = by_id[tid as usize];
                let w = weight(&int.exprs[ea as usize]).min(weight(&int.exprs[eb as usize]));
                (w / c as u32, tid)
            });
            q.bond_need = need;
        }
        let nc = exprs.len();
        let n_bexprs = int.bexprs.len();
        let mut table = vec![u32::MAX; nc * nc * n_bexprs];
        for (&(ea, be, eb), &tid) in &triples {
            let (a, b) = (idx[ea as usize] as usize, idx[eb as usize] as usize);
            table[(a * nc + b) * n_bexprs + be as usize] = tid;
            table[(b * nc + a) * n_bexprs + be as usize] = tid;
        }
        BondScreen { exprs, n_bexprs, table, n_triples: triples.len() }
    }
}

impl KeySet {
    /// Parses and compiles the keys (panics on a SMARTS the port does not
    /// support; the key tables are fixed).
    /// `bond_screen`: also screen on bond triples (worth it for many small keys).
    pub(crate) fn new(smarts: &[&str], bond_screen: bool) -> Self {
        let mut subs = SubQueries::default();
        let parsed: Vec<Parsed> = smarts
            .iter()
            .map(|s| Parser::new(s, &mut subs).parse().unwrap_or_else(|e| panic!("CDK SMARTS {s}: {e}")))
            .collect();
        let mut int = Interner {
            exprs: Vec::new(),
            by_expr: FxHashMap::default(),
            bexprs: Vec::new(),
            by_bexpr: FxHashMap::default(),
        };
        let mut keys: Vec<Query> = parsed.iter().map(|p| compile(p, &mut int, false)).collect();
        let subs: Vec<Query> = subs.parsed.iter().map(|p| compile(p, &mut int, true)).collect();
        let bond_screen = bond_screen.then(|| BondScreen::build(&parsed, &mut keys, &mut int));
        assert!(int.bexprs.len() <= 64, "too many distinct bond expressions");
        // screen entries: bond triples, then the cheap (local) atom
        // expressions, then local upper bounds of the others (e.g. the `C` of
        // `[C;!$(...)]`), and only then the expensive ones
        let mut screen = Vec::new();
        let mut screen_start = Vec::with_capacity(keys.len() + 1);
        for q in &keys {
            screen_start.push(screen.len() as u32);
            if bond_screen.is_some() {
                screen.extend(q.bond_need.iter().map(|&(tid, c)| tid << 10 | SCREEN_TRIPLE << 8 | c as u32));
            }
            // `need` is ordered cheap first (see `compile`)
            let n_cheap = q.need.iter().take_while(|&&(e, _, _)| is_local(&int.exprs[e as usize])).count();
            let (cheap, costly) = q.need.split_at(n_cheap);
            screen.extend(cheap.iter().map(|&(e, c, _)| e << 10 | SCREEN_NEED << 8 | c as u32));
            for &(e, c, _) in costly {
                let bound = local_bound(&int.exprs[e as usize].clone(), &int, &subs);
                if let Some(b) = bound.filter(|b| *b != AExpr::True) {
                    screen.push(int.atom(&b) << 10 | SCREEN_BOUND << 8 | c as u32);
                }
            }
            screen.extend(costly.iter().map(|&(e, c, _)| e << 10 | SCREEN_NEED << 8 | c as u32));
        }
        screen_start.push(screen.len() as u32);
        let mut nodes = Vec::new();
        let mut node_ids = FxHashMap::default();
        let xexprs: Vec<XExpr> = int.exprs.iter().map(|e| annotate(e, &mut nodes, &mut node_ids)).collect();
        let local = xexprs.iter().map(|x| if let XExpr::Local(n) = x { *n } else { u32::MAX }).collect();
        let key_n = keys.iter().map(|q| q.n.min(255) as u8).collect();
        KeySet {
            exprs: int.exprs,
            bexprs: int.bexprs,
            keys,
            subs,
            local,
            nodes,
            xexprs,
            bond_screen,
            key_n,
            screen,
            screen_start,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    /// `Pattern.matches` of every key (bit `i` set when key `i` matches).
    pub(crate) fn match_bits(&self, t: &CdkTarget, out: &mut [bool]) {
        let mut m = Matcher::new(self, t);
        for (i, q) in self.keys.iter().enumerate() {
            out[i] = m.screen_key(i) && m.search(q, Mode::Exists) > 0;
        }
    }

    /// `Mappings.countUnique()` of every key.
    pub(crate) fn match_counts(&self, t: &CdkTarget, out: &mut [u32]) {
        let mut m = Matcher::new(self, t);
        for (i, q) in self.keys.iter().enumerate() {
            out[i] = if m.screen_key(i) { m.search(q, Mode::CountUnique) } else { 0 };
        }
    }

    /// `Mappings.uniqueAtoms().atLeast(min[i])` of every key (`Pattern.matches`
    /// for `min[i] <= 1`).
    pub(crate) fn match_at_least(&self, t: &CdkTarget, min: &[u32], out: &mut [bool]) {
        let mut m = Matcher::new(self, t);
        for (i, q) in self.keys.iter().enumerate() {
            out[i] = m.screen_key(i)
                && match min[i] {
                    0 | 1 => m.search(q, Mode::Exists) > 0,
                    k => m.search(q, Mode::CountUnique) >= k,
                };
        }
    }
}

// ===========================================================================
// Target molecule (CDK view)
// ===========================================================================

/// Tetrahedral stereo of a target atom: ligands (the atom itself for an
/// implicit hydrogen / lone pair) and `CLOCKWISE`.
#[derive(Clone, Copy, Debug)]
struct TetraT {
    ligands: [u32; 4],
    clockwise: bool,
}

/// Double-bond stereo of the target (`DoubleBondStereochemistry`).
#[derive(Clone, Copy, Debug)]
struct GeoT {
    begin: u32,
    end: u32,
    /// Carrier atoms on the begin and end side.
    carriers: [u32; 2],
    /// `TOGETHER` (2) or `OPPOSITE` (1).
    config: u8,
}

/// Key-set independent matching data of a target, shared by the key sets.
struct TargetIndex {
    /// Classes of atoms with equal local properties: a representative and the
    /// atom mask of each class, and the class of each atom.
    class_rep: Vec<u32>,
    class_mask: Vec<u64>,
    class_of: Vec<u32>,
    /// Target adjacency (CSR): neighbour and bond of each entry.
    adj_start: Vec<u32>,
    adj_nbr: Vec<u32>,
    adj_bond: Vec<u32>,
    /// Random atom keys for hashing matched atom sets.
    zob: Vec<(u64, u64)>,
}

/// The CDK view of a molecule after `SmartsPattern.prepare`.
pub(crate) struct CdkTarget {
    n: usize,
    z: Vec<u8>,
    charge: Vec<i8>,
    isotope: Vec<u16>,
    arom: Vec<bool>,
    pub(super) ring: Vec<bool>,
    /// Implicit hydrogens (`CDKHydrogenAdder`).
    hcount: Vec<u8>,
    /// Implicit + hydrogen neighbours.
    total_h: Vec<u8>,
    /// Valence (implicit H + bond orders).
    valence: Vec<u16>,
    /// Neighbours (atom, bond) in CDK order.
    pub(super) adj: Vec<SmallVec<[(u32, u32); 4]>>,
    ends: Vec<(u32, u32)>,
    order: Vec<u8>,
    pub(super) bond_arom: Vec<bool>,
    pub(super) bond_ring: Vec<bool>,
    /// Ring count per atom (minimum cycle basis), when a key set needs it.
    ring_count: std::cell::OnceCell<Vec<u8>>,
    index: std::cell::OnceCell<TargetIndex>,
    tetra: Vec<Option<TetraT>>,
    /// Index into `geo` per atom (`u32::MAX`: none).
    geo_of: Vec<u32>,
    geo: Vec<GeoT>,
    /// Connected component per atom.
    comp: Vec<u32>,
}

impl CdkTarget {
    pub(crate) fn new(mol: &Molecule) -> Self {
        let mut m = CdkMol::new(mol);
        restore_cleaned_charges(mol, &mut m);
        // implicit hydrogens as `CDKHydrogenAdder` assigns them (`[Zn+2]` gets
        // two, the radical S of `N#C[S](O)O` is `S.anyl` with one)
        m.hcount = super::cdk_perception::cdk_hydrogen_counts(mol);
        // CDK keeps the hydrogen atoms written in the SMILES (`[H]OC`,
        // `[H][C@](C)(N)O`, `[H]n1cccc1`), RDKit folds them into their
        // neighbour; restore them from the molecule's record (appended in
        // input order, before the aromaticity perception, which reads degrees).
        let n0 = m.n();
        let removed = mol.removed_hydrogens();
        for h in removed {
            let a = h.atom as usize;
            m.hcount[a] = m.hcount[a].saturating_sub(1);
            let hi = m.z.len() as u32;
            let b = m.ends.len() as u32;
            m.z.push(1);
            m.charge.push(h.charge);
            m.hcount.push(0);
            m.ring_atom.push(false);
            m.adj.push(SmallVec::from_slice(&[(a as u32, b)]));
            m.adj[a].push((hi, b));
            m.ends.push((a as u32, hi));
            m.order.push(1);
            m.ring_bond.push(false);
        }
        let (arom, bond_arom) = smarts_daylight_aromaticity(&m);
        let n = m.n();
        let adj: Vec<SmallVec<[(u32, u32); 4]>> = m.adj.clone();
        let mut total_h = m.hcount.clone();
        let mut valence: Vec<u16> = m.hcount.iter().map(|&h| h as u16).collect();
        for a in 0..n {
            for &(w, b) in &adj[a] {
                if m.z[w as usize] == 1 {
                    total_h[a] = total_h[a].saturating_add(1);
                }
                valence[a] += m.order[b as usize] as u16;
            }
        }
        // stereo as written in the SMILES (CDK keeps every `@`/`@@`, `/`, `\`)
        let mut tetra = vec![None; n];
        for a in 0..n0 {
            let tag = mol.atoms[a].chiral_tag();
            if !tag.is_tetrahedral() {
                continue;
            }
            // RDKit moved the bonds of removed hydrogens to the end of the
            // bond list (in input order) and adjusted the tag accordingly
            let mut nbrs: SmallVec<[u32; 4]> = mol.adjacency[a]
                .iter()
                .map(|&b| {
                    let b = &mol.bonds[b as usize];
                    if b.source as usize == a { b.target as u32 } else { b.source as u32 }
                })
                .collect();
            for (k, h) in removed.iter().enumerate() {
                if h.atom as usize == a {
                    nbrs.push((n0 + k) as u32);
                }
            }
            let ligands = match nbrs.len() {
                4 => [nbrs[0], nbrs[1], nbrs[2], nbrs[3]],
                3 => [nbrs[0], nbrs[1], nbrs[2], a as u32],
                _ => continue,
            };
            tetra[a] = Some(TetraT { ligands, clockwise: tag == ChiralTag::TetrahedralCW });
        }
        let mut geo = Vec::new();
        let mut geo_of = vec![u32::MAX; n];
        for b in &mol.bonds {
            if b.bond_type != BondType::Double {
                continue;
            }
            let (u, v) = (b.source as usize, b.target as usize);
            // directional bond on each side, relative to the double-bond atom
            let side = |x: usize, other: usize| -> Option<(u32, bool)> {
                mol.adjacency[x].iter().find_map(|&bi| {
                    let bd = &mol.bonds[bi as usize];
                    let w = if bd.source as usize == x { bd.target as usize } else { bd.source as usize };
                    if w == other {
                        return None;
                    }
                    let up = match crate::algo::stereo::bond_direction(bd) {
                        crate::algo::stereo::BondDir::None => return None,
                        crate::algo::stereo::BondDir::EndUpRight => true,
                        crate::algo::stereo::BondDir::EndDownRight => false,
                    };
                    Some((w as u32, if bd.source as usize == x { up } else { !up }))
                })
            };
            // or the bond to a restored hydrogen (`[H]/C(C)=C/C`)
            let carrier = |x: usize, other: usize| {
                side(x, other).or_else(|| {
                    removed.iter().enumerate().find_map(|(k, h)| match h.up {
                        Some(up) if h.atom as usize == x => Some(((n0 + k) as u32, up)),
                        _ => None,
                    })
                })
            };
            let (Some((cu, du)), Some((cv, dv))) = (carrier(u, v), carrier(v, u)) else { continue };
            let idx = geo.len() as u32;
            geo.push(GeoT { begin: u as u32, end: v as u32, carriers: [cu, cv], config: if du == dv { RIGHT } else { LEFT } });
            geo_of[u] = idx;
            geo_of[v] = idx;
        }
        let mut comp = vec![u32::MAX; n];
        let mut stack = Vec::new();
        let mut nc = 0;
        for s in 0..n {
            if comp[s] != u32::MAX {
                continue;
            }
            comp[s] = nc;
            stack.push(s);
            while let Some(a) = stack.pop() {
                for &(w, _) in &adj[a] {
                    if comp[w as usize] == u32::MAX {
                        comp[w as usize] = nc;
                        stack.push(w as usize);
                    }
                }
            }
            nc += 1;
        }
        CdkTarget {
            n,
            isotope: (0..n).map(|a| mol.atoms.get(a).map_or(0, |x| x.isotope)).collect(),
            z: m.z,
            charge: m.charge,
            arom,
            ring: m.ring_atom,
            hcount: m.hcount,
            total_h,
            valence,
            adj,
            ends: m.ends,
            order: m.order,
            bond_arom,
            bond_ring: m.ring_bond,
            ring_count: std::cell::OnceCell::new(),
            index: std::cell::OnceCell::new(),
            tetra,
            geo_of,
            geo,
            comp,
        }
    }

    /// Number of atoms (with the hydrogen atoms of the SMILES).
    pub(crate) fn num_atoms(&self) -> usize {
        self.n
    }

    /// Some atom has a mass number.
    pub(crate) fn has_isotope(&self) -> bool {
        self.isotope.iter().any(|&i| i != 0)
    }

    /// Number of connected components.
    pub(crate) fn num_components(&self) -> usize {
        self.comp.iter().max().map_or(0, |&c| c as usize + 1)
    }

    /// The matching data shared by all key sets.
    fn index(&self) -> &TargetIndex {
        self.index.get_or_init(|| {
            let (n, words) = (self.n, self.n.div_ceil(64).max(1));
            let mut ix = TargetIndex {
                class_rep: Vec::new(),
                class_mask: Vec::new(),
                class_of: Vec::with_capacity(n),
                adj_start: Vec::with_capacity(n + 1),
                adj_nbr: Vec::with_capacity(2 * self.ends.len()),
                adj_bond: Vec::with_capacity(2 * self.ends.len()),
                zob: (0..n as u32).map(zobrist).collect(),
            };
            let mut classes: FxHashMap<[u16; 8], u32> = FxHashMap::default();
            for a in 0..n {
                let ring_bonds = self.adj[a].iter().filter(|&&(_, b)| self.bond_ring[b as usize]).count();
                let sig = [
                    self.z[a] as u16 | (self.arom[a] as u16) << 8 | (self.ring[a] as u16) << 9,
                    self.charge[a] as u8 as u16,
                    self.hcount[a] as u16,
                    self.total_h[a] as u16,
                    self.adj[a].len() as u16,
                    self.valence[a],
                    self.isotope[a],
                    ring_bonds as u16,
                ];
                let c = *classes.entry(sig).or_insert_with(|| {
                    ix.class_rep.push(a as u32);
                    ix.class_mask.resize(ix.class_mask.len() + words, 0);
                    ix.class_rep.len() as u32 - 1
                }) as usize;
                ix.class_mask[c * words + a / 64] |= 1 << (a % 64);
                ix.class_of.push(c as u32);
            }
            let mut acc = 0u32;
            for nb in &self.adj {
                ix.adj_start.push(acc);
                acc += nb.len() as u32;
                for &(w, b) in nb {
                    ix.adj_nbr.push(w);
                    ix.adj_bond.push(b);
                }
            }
            ix.adj_start.push(acc);
            ix
        })
    }

    /// Number of minimum-cycle-basis rings through each atom (`Expr.getRingCounts`).
    fn ring_counts(&self) -> &[u8] {
        self.ring_count.get_or_init(|| mcb_ring_counts(&self.adj, &self.bond_ring))
    }

    /// `Expr.isInSmallRingSize` (only called for ring atoms).
    fn in_small_ring_size(&self, atom: usize, size: usize) -> bool {
        let n = self.n;
        let mut dist = vec![1 + n; n];
        dist[atom] = 0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(atom);
        let mut smallest = 1 + n;
        while let Some(a) = queue.pop_front() {
            let d = 1 + dist[a];
            for &(w, b) in &self.adj[a] {
                if !self.bond_ring[b as usize] {
                    continue;
                }
                let w = w as usize;
                if d < dist[w] {
                    dist[w] = d;
                    queue.push_back(w);
                } else if d != 2 + dist[w] {
                    let tmp = d + dist[w];
                    if tmp < smallest {
                        smallest = tmp;
                    }
                }
            }
            if 2 * d > 1 + size {
                break;
            }
        }
        smallest == size
    }

    fn bond_between(&self, a: u32, b: u32) -> Option<u32> {
        self.adj[a as usize].iter().find(|&&(w, _)| w == b).map(|&(_, bi)| bi)
    }
}

/// Per-molecule cache of the CDK SMARTS-key fingerprints (in
/// [`CalculationContext`](super::context::CalculationContext)): the prepared
/// target shared by CDKMACCS, CDKEState, CDKSubstructure and KlekotaRoth, and
/// the key counts of the substructure key sets (see `cdk_substructure`).
#[derive(Default)]
pub(crate) struct CdkKeyCache {
    key: Option<u64>,
    target: Option<CdkTarget>,
    pub(super) counts: [Option<Vec<u32>>; 2],
}

impl CdkKeyCache {
    /// Forgets the cached molecule.
    pub(crate) fn reset(&mut self) {
        self.key = None;
    }

    /// Makes `mol` the cached molecule (dropping the data of another one).
    pub(super) fn select(&mut self, mol: &Molecule) {
        let key = molecule_cache_key(mol);
        if self.key != Some(key) {
            self.key = Some(key);
            self.target = None;
            self.counts = [None, None];
        }
    }

    /// The prepared target of `mol`, built once per molecule.
    pub(super) fn target(&mut self, mol: &Molecule) -> &CdkTarget {
        self.select(mol);
        self.target.get_or_insert_with(|| CdkTarget::new(mol))
    }
}

/// Number of minimum-cycle-basis rings through each atom (CDK
/// `Expr.getRingCounts`: `Cycles.mcb`, i.e. every isolated ring plus the
/// `MinimumCycleBasis` of each fused ring system).
fn mcb_ring_counts(adj: &[SmallVec<[(u32, u32); 4]>], bond_ring: &[bool]) -> Vec<u8> {
    let n = adj.len();
    let mut counts = vec![0u8; n];
    // ring systems: biconnected components of the ring bonds (iterative Tarjan)
    let mut comp = vec![u32::MAX; bond_ring.len()];
    let mut n_comp = 0u32;
    let mut disc = vec![u32::MAX; n];
    let mut low = vec![0u32; n];
    let mut time = 0u32;
    let mut edge_stack: Vec<u32> = Vec::new();
    let mut stack: Vec<(usize, u32, usize)> = Vec::new();
    for root in 0..n {
        if disc[root] != u32::MAX || !adj[root].iter().any(|&(_, b)| bond_ring[b as usize]) {
            continue;
        }
        disc[root] = time;
        low[root] = time;
        time += 1;
        stack.push((root, u32::MAX, 0));
        while let Some(&mut (v, pb, ref mut pos)) = stack.last_mut() {
            if *pos < adj[v].len() {
                let (w, b) = adj[v][*pos];
                *pos += 1;
                if b == pb || !bond_ring[b as usize] {
                    continue;
                }
                let w = w as usize;
                if disc[w] == u32::MAX {
                    edge_stack.push(b);
                    disc[w] = time;
                    low[w] = time;
                    time += 1;
                    stack.push((w, b, 0));
                } else if disc[w] < disc[v] {
                    edge_stack.push(b);
                    low[v] = low[v].min(disc[w]);
                }
            } else {
                stack.pop();
                if let Some(&(u, _, _)) = stack.last() {
                    low[u] = low[u].min(low[v]);
                    if low[v] >= disc[u] {
                        let start = edge_stack.iter().rposition(|&e| e == pb).expect("tree bond on stack");
                        if edge_stack.len() - start > 1 {
                            for &e in &edge_stack[start..] {
                                comp[e as usize] = n_comp;
                            }
                            n_comp += 1;
                        }
                        edge_stack.truncate(start);
                    }
                }
            }
        }
    }
    for c in 0..n_comp {
        let mut verts: Vec<u32> = Vec::new();
        let mut n_bonds = 0;
        for a in 0..n {
            if adj[a].iter().any(|&(_, b)| comp[b as usize] == c) {
                verts.push(a as u32);
            }
        }
        for &cb in &comp {
            n_bonds += usize::from(cb == c);
        }
        if n_bonds == verts.len() {
            for &v in &verts {
                counts[v as usize] = counts[v as usize].saturating_add(1);
            }
            continue;
        }
        let mut map = vec![u32::MAX; n];
        for (i, &v) in verts.iter().enumerate() {
            map[v as usize] = i as u32;
        }
        let sub: Vec<Vec<u32>> = verts
            .iter()
            .map(|&v| {
                adj[v as usize].iter().filter(|&&(_, b)| comp[b as usize] == c).map(|&(w, _)| map[w as usize]).collect()
            })
            .collect();
        for cycle in super::pubchem_cdk::minimum_cycle_basis(&sub) {
            for &v in &cycle[1..] {
                let a = verts[v as usize] as usize;
                counts[a] = counts[a].saturating_add(1);
            }
        }
    }
    counts
}

/// Undoes RDKit's `cleanUp` charge separation of neutral hypervalent N/P
/// groups (`N(=O)=O` -> `[N+](=O)[O-]`, `N=N#N` -> `N=[N+]=[N-]`); CDK keeps
/// the input form. SMILES can only charge bracket atoms, so a charged atom
/// without the no-implicit-H flag was charged by cleanUp (same rule as the
/// PubChem fingerprint's CDK view). Halogen oxides are restored by [`CdkMol`]
/// (a bracket halogen keeps its charge; of the terminal oxygens of
/// `[O-]Cl(=O)(=O)=O` another one than the bracket `[O-]` may stay charged,
/// which gives the same graph).
fn restore_cleaned_charges(mol: &Molecule, m: &mut CdkMol) {
    let cleaned = |a: usize| !mol.atoms[a].no_implicit_h();
    for a in 0..m.n() {
        if m.charge[a] <= 0 || !cleaned(a) || !matches!(m.z[a], 7 | 15) {
            continue;
        }
        for k in 0..m.adj[a].len() {
            if m.charge[a] == 0 {
                break;
            }
            let (nb, bi) = m.adj[a][k];
            let (nb, bi) = (nb as usize, bi as usize);
            if m.charge[nb] != -1 || !cleaned(nb) {
                continue;
            }
            let restored = match (m.z[nb], m.order[bi]) {
                (8, 1) if mol.bonds[bi].bond_type == BondType::Single => 2,
                (7, 2) if m.z[a] == 7 => 3,
                _ => continue,
            };
            m.order[bi] = restored;
            m.charge[nb] = 0;
            m.charge[a] -= 1;
        }
    }
}

// ===========================================================================
// Matching
// ===========================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Exists,
    CountUnique,
    /// Which atoms of a set are roots of a match (recursive SMARTS).
    Roots,
}

/// Candidate iterator of one search position: an adjacency range `lo..hi`,
/// or (component start) the mask words `lo..hi` with the pending `bits` of
/// word `lo`; candidates below `lower` are skipped (symmetry breaking).
#[derive(Clone, Copy, Default)]
struct Iter {
    lo: u32,
    hi: u32,
    bits: u64,
    lower: u32,
}

/// The fixed parts of one search.
struct SearchCtx<'q> {
    q: &'q Query,
    steps: &'q [Step],
    extra: &'q [u8],
    /// Mask offset of the atom expression at each position.
    offs: &'q [u32],
    count_mode: bool,
    filters: bool,
    /// Root candidates of the first position (`Mode::Roots`), else empty.
    roots: &'q [u64],
    mode: Mode,
}

/// Per-molecule matching state of one key set.
struct Matcher<'a> {
    ks: &'a KeySet,
    t: &'a CdkTarget,
    words: usize,
    /// Offset into `masks` of each computed atom expression (`u32::MAX`: not yet).
    mask_off: Vec<u32>,
    popcount: Vec<u32>,
    masks: Vec<u64>,
    /// Recursive queries: atoms evaluated / matched (bit sets).
    rec_known: Vec<u64>,
    rec_val: Vec<u64>,
    /// Bond expressions matched by each bond (bit per bond expression).
    bond_ok: Vec<u64>,
    /// Number of bonds matching each bond-screen triple.
    triple_count: Vec<u16>,
    /// Topological distances from a source atom (rows computed on demand):
    /// offset of each atom's row in `dists` (`u32::MAX`: not yet).
    drow: Vec<u32>,
    dists: Vec<u8>,
    bfs: Vec<u32>,
    /// Random atom keys for hashing matched atom sets.
    zob: &'a [(u64, u64)],
    /// A search is running (its state must survive nested searches).
    searching: bool,
    /// Screen result of each recursive query (0: not yet, 1: pass, 2: fail).
    sub_screen: Vec<u8>,
    /// Roots found by a `Mode::Roots` search.
    root_hits: SmallVec<[u64; 4]>,
    /// Start (`need` entry) of the next key search, set by `screen_key`.
    best: usize,
    /// Classes of atoms with equal local properties (see [`TargetIndex`]).
    class_rep: &'a [u32],
    class_mask: &'a [u64],
    class_of: &'a [u32],
    /// Masks of the local expression nodes (offsets, `u32::MAX`: not yet).
    node_off: Vec<u32>,
    node_masks: Vec<u64>,
    /// All atoms.
    all: SmallVec<[u64; 4]>,
    /// Target adjacency (CSR) with the bond expressions each entry's bond matches.
    adj_start: &'a [u32],
    adj_nbr: &'a [u32],
    adj_ok: Vec<u64>,
    /// Search state.
    map: Vec<u32>,
    used: Vec<u64>,
    seen: FxHashSet<(u64, u64)>,
}

#[inline]
fn bit(words: &[u64], i: usize) -> bool {
    words[i >> 6] >> (i & 63) & 1 == 1
}

#[inline]
fn zobrist(i: u32) -> (u64, u64) {
    fn mix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }
    (mix(i as u64), mix(i as u64 ^ 0x5851_f42d_4c95_7f2d))
}

impl<'a> Matcher<'a> {
    fn new(ks: &'a KeySet, t: &'a CdkTarget) -> Self {
        let words = t.n.div_ceil(64).max(1);
        let ix = t.index();
        let mut bond_ok = vec![0u64; t.ends.len()];
        for (b, ok) in bond_ok.iter_mut().enumerate() {
            for (i, e) in ks.bexprs.iter().enumerate() {
                if eval_bond(e, t, b, UNKNOWN_STEREO) {
                    *ok |= 1 << i;
                }
            }
        }
        let adj_ok = ix.adj_bond.iter().map(|&b| bond_ok[b as usize]).collect();
        let mut m = Matcher {
            ks,
            t,
            words,
            mask_off: vec![u32::MAX; ks.exprs.len()],
            popcount: vec![0; ks.exprs.len()],
            masks: Vec::new(),
            rec_known: vec![0; ks.subs.len() * words],
            rec_val: vec![0; ks.subs.len() * words],
            bond_ok,
            triple_count: Vec::new(),
            drow: vec![u32::MAX; t.n],
            dists: Vec::new(),
            bfs: Vec::new(),
            zob: &ix.zob,
            searching: false,
            sub_screen: vec![0; ks.subs.len()],
            root_hits: SmallVec::new(),
            best: 0,
            class_rep: &ix.class_rep,
            class_mask: &ix.class_mask,
            class_of: &ix.class_of,
            node_off: vec![u32::MAX; ks.nodes.len()],
            node_masks: Vec::new(),
            all: {
                let mut all: SmallVec<[u64; 4]> = SmallVec::from_elem(!0u64, words);
                if !t.n.is_multiple_of(64) {
                    all[words - 1] = (1u64 << (t.n % 64)) - 1;
                }
                if t.n == 0 {
                    all[0] = 0;
                }
                all
            },
            adj_start: &ix.adj_start,
            adj_nbr: &ix.adj_nbr,
            adj_ok,
            map: Vec::new(),
            used: vec![0; words],
            seen: FxHashSet::default(),
        };
        if let Some(bs) = &ks.bond_screen {
            m.count_triples(bs);
        }
        m
    }

    /// Topological distance between `a` and `b` (255 for 255 or more / disconnected).
    #[inline]
    fn dist(&mut self, src: u32, a: u32) -> u8 {
        let mut off = self.drow[src as usize];
        if off == u32::MAX {
            let n = self.t.n;
            off = self.dists.len() as u32;
            self.dists.resize(self.dists.len() + n, u8::MAX);
            let row = &mut self.dists[off as usize..off as usize + n];
            row[src as usize] = 0;
            self.bfs.clear();
            self.bfs.push(src);
            let mut head = 0;
            while head < self.bfs.len() {
                let v = self.bfs[head] as usize;
                head += 1;
                let d = row[v].saturating_add(1);
                for &(w, _) in &self.t.adj[v] {
                    if row[w as usize] == u8::MAX && w != src {
                        row[w as usize] = d;
                        self.bfs.push(w);
                    }
                }
            }
            self.drow[src as usize] = off;
        }
        self.dists[off as usize + a as usize]
    }

    /// Per-molecule counts of the bond-screen triples.
    fn count_triples(&mut self, bs: &BondScreen) {
        let t = self.t;
        let nc = bs.exprs.len();
        // compact indices of the screen expressions each atom class matches
        let n_class = self.class_rep.len();
        let mut class_exprs: Vec<SmallVec<[u16; 16]>> = vec![SmallVec::new(); n_class];
        for (x, &e) in bs.exprs.iter().enumerate() {
            let off = self.mask(e);
            for (c, &rep) in self.class_rep.iter().enumerate() {
                let r = rep as usize;
                if self.masks[off + r / 64] >> (r % 64) & 1 == 1 {
                    class_exprs[c].push(x as u16);
                }
            }
        }
        // bonds with equal end classes and bond expressions match the same triples
        let mut groups: FxHashMap<(u32, u32, u64), u16> = FxHashMap::default();
        for (b, &(u, v)) in t.ends.iter().enumerate() {
            *groups.entry((self.class_of[u as usize], self.class_of[v as usize], self.bond_ok[b])).or_insert(0) += 1;
        }
        let mut count = vec![0u16; bs.n_triples];
        let mut stamp = vec![u32::MAX; bs.n_triples];
        for (g, (&(cu, cv, ok), &mult)) in groups.iter().enumerate() {
            let (lu, lv) = (&class_exprs[cu as usize], &class_exprs[cv as usize]);
            let mut bes = ok;
            while bes != 0 {
                let be = bes.trailing_zeros() as usize;
                bes &= bes - 1;
                for &x in lu {
                    let row = x as usize * nc;
                    for &y in lv {
                        let tid = bs.table[(row + y as usize) * bs.n_bexprs + be];
                        if tid != u32::MAX && stamp[tid as usize] != g as u32 {
                            stamp[tid as usize] = g as u32;
                            count[tid as usize] = count[tid as usize].saturating_add(mult);
                        }
                    }
                }
            }
        }
        self.triple_count = count;
    }

    /// Offset of the (lazily computed) atom mask of expression `id`.
    fn mask(&mut self, id: u32) -> usize {
        let off = self.mask_off[id as usize];
        if off != u32::MAX {
            return off as usize;
        }
        let ks = self.ks;
        let mut all: SmallVec<[u64; 4]> = SmallVec::from_elem(!0u64, self.words);
        let rem = self.t.n % 64;
        if rem != 0 {
            all[self.words - 1] = (1u64 << rem) - 1;
        }
        if self.t.n == 0 {
            all[0] = 0;
        }
        let out: SmallVec<[u64; 4]> = if ks.local[id as usize] != u32::MAX {
            let off = self.node_mask(ks.local[id as usize]);
            SmallVec::from_slice(&self.node_masks[off..off + self.words])
        } else {
            self.eval(&ks.xexprs[id as usize], &all)
        };
        let off = self.masks.len();
        self.popcount[id as usize] = out.iter().map(|w| w.count_ones()).sum();
        self.masks.extend_from_slice(&out);
        self.mask_off[id as usize] = off as u32;
        off
    }

    /// Offset in `node_masks` of the (lazily computed) mask of a local node.
    fn node_mask(&mut self, id: u32) -> usize {
        let off = self.node_off[id as usize];
        if off != u32::MAX {
            return off as usize;
        }
        let w = self.words;
        let mut out: SmallVec<[u64; 4]> = SmallVec::from_elem(0, w);
        match &self.ks.nodes[id as usize] {
            Node::Prim(p) => {
                // once per class of atoms with equal properties
                for c in 0..self.class_rep.len() {
                    if eval_prim(p, self.t, self.class_rep[c] as usize, UNKNOWN_STEREO) {
                        for (o, m) in out.iter_mut().zip(&self.class_mask[c * w..(c + 1) * w]) {
                            *o |= m;
                        }
                    }
                }
            }
            &Node::And(l, r) => {
                let (l, r) = (self.node_mask(l), self.node_mask(r));
                for i in 0..w {
                    out[i] = self.node_masks[l + i] & self.node_masks[r + i];
                }
            }
            &Node::Or(l, r) => {
                let (l, r) = (self.node_mask(l), self.node_mask(r));
                for i in 0..w {
                    out[i] = self.node_masks[l + i] | self.node_masks[r + i];
                }
            }
            &Node::Not(x) => {
                let x = self.node_mask(x);
                for i in 0..w {
                    out[i] = self.all[i] & !self.node_masks[x + i];
                }
            }
            Node::All => out.copy_from_slice(&self.all),
        }
        let off = self.node_masks.len();
        self.node_masks.extend_from_slice(&out);
        self.node_off[id as usize] = off as u32;
        off
    }

    /// Atoms of `cand` matching `e` (stereo unknown).
    fn eval(&mut self, e: &XExpr, cand: &[u64]) -> SmallVec<[u64; 4]> {
        let mut out: SmallVec<[u64; 4]> = SmallVec::from_elem(0, cand.len());
        match e {
            &XExpr::Local(node) => {
                let off = self.node_mask(node);
                for i in 0..out.len() {
                    out[i] = cand[i] & self.node_masks[off + i];
                }
            }
            XExpr::And(l, r) => {
                let a = self.eval(l, cand);
                if a.iter().all(|&w| w == 0) {
                    return a;
                }
                return self.eval(r, &a);
            }
            XExpr::Or(l, r) => {
                let a = self.eval(l, cand);
                let rest: SmallVec<[u64; 4]> = cand.iter().zip(&a).map(|(c, x)| c & !x).collect();
                let b = if rest.iter().any(|&w| w != 0) { self.eval(r, &rest) } else { rest };
                for i in 0..out.len() {
                    out[i] = a[i] | b[i];
                }
            }
            XExpr::Not(x) => {
                let a = self.eval(x, cand);
                for i in 0..out.len() {
                    out[i] = cand[i] & !a[i];
                }
            }
            &XExpr::Rec(id) => {
                let id = id as usize;
                let base = id * self.words;
                let unknown: SmallVec<[u64; 4]> =
                    cand.iter().enumerate().map(|(wi, &c)| c & !self.rec_known[base + wi]).collect();
                if unknown.iter().any(|&w| w != 0) {
                    let hits = self.match_roots(id, &unknown);
                    for wi in 0..self.words {
                        self.rec_known[base + wi] |= unknown[wi];
                        self.rec_val[base + wi] |= hits[wi];
                    }
                }
                for wi in 0..out.len() {
                    out[wi] = cand[wi] & self.rec_val[base + wi];
                }
            }
            XExpr::Prim(prim) => {
                for (wi, &w) in cand.iter().enumerate() {
                    let mut rest = w;
                    while rest != 0 {
                        let b = rest.trailing_zeros() as usize;
                        rest &= rest - 1;
                        if eval_prim(prim, self.t, wi * 64 + b, UNKNOWN_STEREO) {
                            out[wi] |= 1 << b;
                        }
                    }
                }
            }
        }
        out
    }

    /// Scalar evaluation with a known stereo value (stereo filter).
    fn eval_atom(&mut self, e: &AExpr, a: usize, stereo: i32) -> bool {
        match e {
            AExpr::And(l, r) => self.eval_atom(l, a, stereo) && self.eval_atom(r, a, stereo),
            AExpr::Or(l, r) => self.eval_atom(l, a, stereo) || self.eval_atom(r, a, stereo),
            AExpr::Not(x) => {
                !self.eval_atom(x, a, stereo) || (stereo == UNKNOWN_STEREO && matches!(**x, AExpr::Stereo(_)))
            }
            AExpr::Rec(id) => {
                let mut cand: SmallVec<[u64; 4]> = SmallVec::from_elem(0, self.words);
                cand[a >> 6] |= 1 << (a & 63);
                let r = self.eval(&XExpr::Rec(*id), &cand);
                bit(&r, a)
            }
            prim => eval_prim(prim, self.t, a, stereo),
        }
    }

    /// [`Self::screen`] of key `i` from the flat screen table.
    /// Also sets `self.best`: the entry of `need` (in order) with the fewest
    /// candidate atoms, where the search starts.
    fn screen_key(&mut self, i: usize) -> bool {
        let ks = self.ks;
        if ks.key_n[i] as usize > self.t.n {
            return false;
        }
        let (mut need_i, mut best, mut best_pop) = (0u32, 0u32, u32::MAX);
        for &x in &ks.screen[ks.screen_start[i] as usize..ks.screen_start[i + 1] as usize] {
            let (id, kind, c) = (x >> 10, x >> 8 & 3, x & 255);
            if kind == SCREEN_TRIPLE {
                if (self.triple_count[id as usize] as u32) < c {
                    return false;
                }
            } else {
                if self.mask_off[id as usize] == u32::MAX {
                    self.mask(id);
                }
                let pop = self.popcount[id as usize];
                if pop < c {
                    return false;
                }
                if kind == SCREEN_NEED {
                    if pop < best_pop {
                        best_pop = pop;
                        best = need_i;
                    }
                    need_i += 1;
                }
            }
        }
        self.best = best as usize;
        true
    }

    /// Screen: every atom expression of `q` matches enough atoms.
    fn screen(&mut self, q: &Query) -> bool {
        if q.n > self.t.n {
            return false;
        }
        if !self.triple_count.is_empty() {
            for &(tid, c) in &q.bond_need {
                if self.triple_count[tid as usize] < c as u16 {
                    return false;
                }
            }
        }
        for &(e, c, _) in &q.need {
            self.mask(e);
            if self.popcount[e as usize] < c as u32 {
                return false;
            }
        }
        true
    }

    /// `DfPattern.matchRoot(atom).atLeast(1)` of recursive query `id` for every
    /// atom of `roots` (one search over all of them); returns the matching ones.
    fn match_roots(&mut self, id: usize, roots: &[u64]) -> SmallVec<[u64; 4]> {
        let ks = self.ks;
        let q = &ks.subs[id];
        // the screen does not depend on the root: once per molecule
        if self.sub_screen[id] == 0 {
            self.sub_screen[id] = if self.screen(q) { 1 } else { 2 };
        }
        if self.sub_screen[id] == 2 {
            return SmallVec::from_elem(0, self.words);
        }
        let pl = q.root_plan.as_deref().expect("rooted plan");
        if !self.searching {
            self.root_hits = SmallVec::from_elem(0, self.words);
            self.search_with(q, pl, roots, Mode::Roots);
            return std::mem::take(&mut self.root_hits);
        }
        // called from a running search's stereo filter: search with own state
        let saved_map = std::mem::take(&mut self.map);
        let saved_used = std::mem::replace(&mut self.used, vec![0; self.words]);
        let saved_seen = std::mem::take(&mut self.seen);
        let saved_hits = std::mem::replace(&mut self.root_hits, SmallVec::from_elem(0, self.words));
        self.search_with(q, pl, roots, Mode::Roots);
        let hits = std::mem::replace(&mut self.root_hits, saved_hits);
        self.map = saved_map;
        self.used = saved_used;
        self.seen = saved_seen;
        hits
    }

    /// Searches `q`; returns 1/0 (`Exists`) or the number of unique atom sets.
    /// The search starts at the query atom with the fewest candidates in this
    /// molecule, as found by the preceding [`Self::screen_key`].
    fn search(&mut self, q: &Query, mode: Mode) -> u32 {
        self.search_with(q, q.plan(self.best), &[], mode)
    }

    /// Searches `q` along plan `pl`; with `Mode::Roots` the first position
    /// only takes the atoms of `roots` and each root is recorded in
    /// `root_hits` at its first match.
    fn search_with(&mut self, q: &Query, pl: &Plan, roots: &[u64], mode: Mode) -> u32 {
        // atom masks of all query atoms must exist before the search
        let offs: SmallVec<[u32; 16]> = pl
            .steps
            .iter()
            .map(|st| match self.mask_off[st.expr as usize] {
                u32::MAX => self.mask(st.expr) as u32,
                off => off,
            })
            .collect();
        let filters = q.has_filters();
        if filters {
            self.map.clear();
            self.map.resize(q.n, u32::MAX);
        }
        if !self.seen.is_empty() {
            self.seen.clear();
        }
        let sc = SearchCtx { q, steps: &pl.steps, extra: &pl.extra, offs: &offs, count_mode: mode == Mode::CountUnique, filters, roots, mode };
        let mut count = 0u32;
        let mut pos_map: SmallVec<[u32; 16]> = SmallVec::from_elem(u32::MAX, q.n);
        let outer = std::mem::replace(&mut self.searching, true);
        self.run(&sc, &mut pos_map, &mut count);
        self.searching = outer;
        for w in self.used.iter_mut() {
            *w = 0;
        }
        count
    }

    /// Sets up the candidate iterator of position `k`.
    #[inline(always)]
    fn init_pos(&self, sc: &SearchCtx, it: &mut [Iter], k: usize, pos_map: &[u32]) {
        let st = &sc.steps[k];
        let mut low = 0u32;
        if sc.count_mode {
            for &j in st.lower(sc.extra) {
                low = low.max(pos_map[j as usize] + 1);
            }
        }
        let i = &mut it[k];
        i.lower = low;
        if st.from == NONE {
            i.lo = 0;
            i.hi = self.words as u32;
            i.bits = self.masks[sc.offs[k] as usize] & !self.used[0];
            if k == 0 && !sc.roots.is_empty() {
                i.bits &= sc.roots[0];
            }
        } else {
            let src = pos_map[st.from as usize] as usize;
            i.lo = self.adj_start[src];
            i.hi = self.adj_start[src + 1];
        }
    }

    /// The search proper (iterative backtracking over the plan's positions);
    /// returns `true` when stopped at the first match (Exists mode).
    fn run(&mut self, sc: &SearchCtx, pos_map: &mut [u32], count: &mut u32) -> bool {
        let n = sc.steps.len();
        if n == 0 {
            return false;
        }
        let mut it: SmallVec<[Iter; 16]> = SmallVec::from_elem(Iter::default(), n);
        let mut k = 0usize;
        self.init_pos(sc, &mut it, 0, pos_map);
        loop {
            // next candidate of position k
            let st = &sc.steps[k];
            let cand = if st.from == NONE {
                let i = &mut it[k];
                loop {
                    if i.bits != 0 {
                        let b = i.bits.trailing_zeros();
                        i.bits &= i.bits - 1;
                        let a = i.lo * 64 + b;
                        if a < i.lower {
                            continue;
                        }
                        break Some(a);
                    }
                    i.lo += 1;
                    if i.lo >= i.hi {
                        break None;
                    }
                    let wi = i.lo as usize;
                    i.bits = self.masks[sc.offs[k] as usize + wi] & !self.used[wi];
                    if k == 0 && !sc.roots.is_empty() {
                        i.bits &= sc.roots[wi];
                    }
                }
            } else {
                let off = sc.offs[k] as usize;
                let i = &mut it[k];
                let mut found = None;
                while i.lo < i.hi {
                    let x = i.lo as usize;
                    i.lo += 1;
                    if self.adj_ok[x] >> st.be & 1 == 0 {
                        continue;
                    }
                    let w = self.adj_nbr[x];
                    let (wi, bit) = ((w >> 6) as usize, 1u64 << (w & 63));
                    if self.used[wi] & bit != 0 || self.masks[off + wi] & bit == 0 || w < i.lower {
                        continue;
                    }
                    found = Some(w);
                    break;
                }
                found
            };
            let Some(a) = cand else {
                // backtrack
                if k == 0 {
                    return false;
                }
                k -= 1;
                let a = pos_map[k];
                self.used[(a >> 6) as usize] &= !(1u64 << (a & 63));
                pos_map[k] = u32::MAX;
                continue;
            };
            if !self.place_ok(st, sc.extra, a, pos_map) {
                continue;
            }
            pos_map[k] = a;
            if k + 1 < n {
                self.used[(a >> 6) as usize] |= 1u64 << (a & 63);
                k += 1;
                self.init_pos(sc, &mut it, k, pos_map);
                continue;
            }
            // complete mapping
            if self.leaf(sc, pos_map, count) {
                if sc.mode != Mode::Roots {
                    return true;
                }
                // root found: continue with the next root
                for p in pos_map[..k].iter_mut() {
                    self.used[(*p >> 6) as usize] &= !(1u64 << (*p & 63));
                    *p = u32::MAX;
                }
                pos_map[k] = u32::MAX;
                k = 0;
                continue;
            }
            pos_map[k] = u32::MAX;
        }
    }

    /// Closures and distance checks of a candidate atom `a` for step `st`.
    #[inline(always)]
    fn place_ok(&mut self, st: &Step, extra: &[u8], a: u32, pos_map: &[u32]) -> bool {
        for &[p, be] in st.closures(extra).as_chunks::<2>().0 {
            let other = pos_map[p as usize];
            let (l, h) = (self.adj_start[a as usize] as usize, self.adj_start[a as usize + 1] as usize);
            match (l..h).find(|&i| self.adj_nbr[i] == other) {
                Some(i) if self.adj_ok[i] >> be & 1 == 1 => {}
                _ => return false,
            }
        }
        for &[p, maxd] in st.dist(extra).as_chunks::<2>().0 {
            if self.dist(pos_map[p as usize], a) > maxd {
                return false;
            }
        }
        true
    }

    /// A complete mapping: filters, then record it; `true` stops the search.
    fn leaf(&mut self, sc: &SearchCtx, pos_map: &[u32], count: &mut u32) -> bool {
        if sc.filters {
            for (st, &a) in sc.steps.iter().zip(pos_map.iter()) {
                self.map[st.atom as usize] = a;
            }
            if !self.filters_pass(sc.q) {
                return false;
            }
        }
        if sc.mode == Mode::Roots {
            let r = pos_map[0];
            self.root_hits[(r >> 6) as usize] |= 1u64 << (r & 63);
            return true;
        }
        if !sc.count_mode {
            *count = 1;
            return true;
        }
        let mut h = (0u64, 0u64);
        for &t in pos_map.iter() {
            let z = self.zob[t as usize];
            h.0 = h.0.wrapping_add(z.0);
            h.1 = h.1.wrapping_add(z.1);
        }
        if self.seen.insert(h) {
            *count += 1;
        }
        false
    }

    /// `QueryStereoFilter` and `ComponentFilter` on the complete mapping `self.map`.
    fn filters_pass(&mut self, q: &Query) -> bool {
        let t = self.t;
        if let Some(comp) = &q.comp {
            // atoms of one query group in one target component, different groups apart
            let mut pairs: SmallVec<[(u8, u32); 4]> = SmallVec::new();
            for (qa, &g) in comp.iter().enumerate() {
                if g == 0 {
                    continue;
                }
                let tc = t.comp[self.map[qa] as usize];
                for &(g2, tc2) in &pairs {
                    if (g2 == g) != (tc2 == tc) {
                        return false;
                    }
                }
                pairs.push((g, tc));
            }
        }
        let map = self.map.clone();
        for &(u, lig) in &q.tetra {
            let v = map[u as usize] as usize;
            let expr = &self.ks.exprs[q.atom_expr[u as usize] as usize];
            let Some(te) = t.tetra[v] else {
                // no target stereo (or a double-bond stereo atom)
                if t.geo_of[v] != u32::MAX {
                    return false;
                }
                if !self.eval_atom(expr, v, 0) {
                    return false;
                }
                continue;
            };
            let mut us: [u32; 4] = lig.map(|x| map[x as usize]);
            let vs = te.ligands;
            for &x in &vs {
                if !us.contains(&x)
                    && let Some(i) = us.iter().position(|&y| y as usize == v)
                {
                    us[i] = x;
                }
            }
            let parity = perm_parity(&us) * perm_parity(&vs) * if te.clockwise { 1 } else { -1 };
            let stereo = if parity < 0 { LEFT } else { RIGHT } as i32;
            if !self.eval_atom(expr, v, stereo) {
                return false;
            }
        }
        for &(u1, u2, be, lc, rc) in &q.geo {
            let (v1, v2) = (map[u1 as usize], map[u2 as usize]);
            let (g1, g2) = (t.geo_of[v1 as usize], t.geo_of[v2 as usize]);
            if t.tetra[v1 as usize].is_some() || t.tetra[v2 as usize].is_some() {
                return false;
            }
            let Some(tb) = t.bond_between(v1, v2) else { return false };
            let config = if g1 != u32::MAX && g2 != u32::MAX {
                let g = t.geo[g1 as usize];
                if !((g.begin == v1 || g.end == v1) && (g.begin == v2 || g.end == v2)) {
                    return false;
                }
                let carriers = if g.begin == v1 { g.carriers } else { [g.carriers[1], g.carriers[0]] };
                let same_l = map[lc as usize] == carriers[0];
                let same_r = map[rc as usize] == carriers[1];
                if same_l != same_r { g.config ^ 3 } else { g.config }
            } else if g1 != u32::MAX || g2 != u32::MAX {
                return false;
            } else {
                0
            };
            if !eval_bond(&self.ks.bexprs[be as usize], t, tb as usize, config as i32) {
                return false;
            }
        }
        true
    }
}

/// `QueryStereoFilter.permutationParity`.
fn perm_parity(v: &[u32; 4]) -> i32 {
    let mut n = 0;
    for i in 0..4 {
        for j in i + 1..4 {
            if v[i] > v[j] {
                n += 1;
            }
        }
    }
    if n & 1 == 1 { -1 } else { 1 }
}

/// CDK `Expr.matches(IAtom, stereo)` for primitives.
fn eval_prim(e: &AExpr, t: &CdkTarget, a: usize, stereo: i32) -> bool {
    match *e {
        AExpr::True => true,
        AExpr::False => false,
        AExpr::Elem(z) => t.z[a] == z,
        AExpr::Aliph(z) => !t.arom[a] && t.z[a] == z,
        AExpr::Arom(z) => t.arom[a] && t.z[a] == z,
        AExpr::IsArom => t.arom[a],
        AExpr::IsAliph => !t.arom[a],
        AExpr::InRing => t.ring[a],
        AExpr::InChain => !t.ring[a],
        AExpr::HasImplH => t.hcount[a] > 0,
        AExpr::ImplH(h) => t.hcount[a] == h,
        AExpr::TotalH(h) => t.hcount[a] <= h && t.total_h[a] == h,
        AExpr::Degree(d) => t.adj[a].len() == d as usize,
        AExpr::TotalDegree(d) => t.adj[a].len() + t.hcount[a] as usize == d as usize,
        AExpr::Valence(v) => t.hcount[a] as u16 <= v as u16 && t.valence[a] == v as u16,
        AExpr::Isotope(i) => t.isotope[a] == i,
        AExpr::UnspecIsotope => t.isotope[a] == 0,
        AExpr::Charge(q) => t.charge[a] == q,
        AExpr::RingCount(c) => t.ring[a] && t.ring_counts()[a] == c,
        AExpr::RingSmallest(s) => t.ring[a] && t.in_small_ring_size(a, s as usize),
        AExpr::RingBondCount(c) => {
            if !t.ring[a] || t.adj[a].len() < c as usize {
                return false;
            }
            t.adj[a].iter().filter(|&&(_, b)| t.bond_ring[b as usize]).count() == c as usize
        }
        AExpr::Stereo(v) => stereo == UNKNOWN_STEREO || stereo == v as i32,
        AExpr::MinDegree(d) => t.adj[a].len() >= d as usize,
        AExpr::MinRingBonds(c) => t.adj[a].iter().filter(|&&(_, b)| t.bond_ring[b as usize]).count() >= c as usize,
        AExpr::And(..) | AExpr::Or(..) | AExpr::Not(..) | AExpr::Rec(_) => unreachable!("not a primitive"),
    }
}

/// CDK `Expr.matches(IBond, stereo)`.
fn eval_bond(e: &BExpr, t: &CdkTarget, b: usize, stereo: i32) -> bool {
    match e {
        BExpr::True => true,
        BExpr::AliphOrder(o) => !t.bond_arom[b] && t.order[b] == *o,
        BExpr::Order(o) => t.order[b] == *o,
        BExpr::IsArom => t.bond_arom[b],
        BExpr::InRing => t.bond_ring[b],
        BExpr::InChain => !t.bond_ring[b],
        BExpr::SingleOrArom => t.bond_arom[b] || t.order[b] == 1,
        BExpr::Stereo(v) => stereo == UNKNOWN_STEREO || stereo == *v as i32,
        BExpr::And(l, r) => eval_bond(l, t, b, stereo) && eval_bond(r, t, b, stereo),
        BExpr::Or(l, r) => eval_bond(l, t, b, stereo) || eval_bond(r, t, b, stereo),
        BExpr::Not(x) => {
            !eval_bond(x, t, b, stereo) || (stereo == UNKNOWN_STEREO && matches!(**x, BExpr::Stereo(_)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn count(smarts: &str, smiles: &str) -> u32 {
        let ks = KeySet::new(&[smarts], true);
        let t = CdkTarget::new(&parse_smiles(smiles).unwrap());
        let mut out = [0u32];
        ks.match_counts(&t, &mut out);
        let mut bits = [false];
        ks.match_bits(&t, &mut bits);
        assert_eq!(bits[0], out[0] > 0, "{smarts} on {smiles}");
        out[0]
    }

    #[test]
    fn unique_counts() {
        assert_eq!(count("[!#1][CH]([!#1])[!#1]", "CC(C)C"), 1);
        assert_eq!(count("[!#1][CH2][!#1]", "CCCCC"), 3);
        assert_eq!(count("c1ccccc1", "c1ccccc1"), 1);
        assert_eq!(count("CCC", "C1CC1"), 1);
        assert_eq!(count("[OX2H][CX4]", "OCCO"), 2);
        assert_eq!(count("[$(*=O)]", "CC(=O)O"), 1);
    }

    #[test]
    fn cdk_primitives() {
        // R2 counts minimum-cycle-basis rings; ring closure inside $()
        assert_eq!(count("[R2]", "c1ccc2ccccc2c1"), 2);
        assert_eq!(count("[$([CX4;!$([H0])]1[CX3]=[CX3][CX3]=[CX3]1)]", "C1C=CC=C1"), 1);
        // component groups
        assert_eq!(count("([-1,-2]).([+1,+2])", "[Na+].[Cl-]"), 1);
        assert_eq!(count("([-1,-2]).([+1,+2])", "C[N+](C)(C)CC(=O)[O-]"), 0);
        // query stereo with fewer than three neighbours matches anything
        assert_eq!(count("[$([*@H](~*)~*)]", "c1ccccc1"), 6);
        assert_eq!(count("[$([*@H](*)(*)*)]", "CC(C)O"), 0);
        assert_eq!(count("[$([*@H](*)(*)*)]", "C[C@H](N)O"), 1);
        // cis/trans
        assert_eq!(count(r"*/[D2]=[D2]\*", r"C/C=C\C"), 1);
        assert_eq!(count(r"*/[D2]=[D2]/*", r"C/C=C\C"), 0);
        assert_eq!(count(r"*/[D2]=[D2]/*", r"C/C=C/C"), 1);
        assert_eq!(count(r"*/[D2]=[D2]/*", "CC=CC"), 0);
        // nitro groups keep CDK's neutral form
        assert_eq!(count("[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]", "CN(=O)=O"), 1);
        assert_eq!(count("[N+](=O)[O-]", "CN(=O)=O"), 0);
        assert_eq!(count("[N+](=O)[O-]", "C[N+](=O)[O-]"), 1);
    }

    /// Hydrogen atoms of the SMILES stay atoms (and stereo neighbours), as in
    /// CDK 2.13 (`SmartsPattern.matchAll(mol).countUnique()` on the reference
    /// preparation).
    #[test]
    fn restored_hydrogen_atoms() {
        let tetra = ["[C@](C)(N)O", "[C@@](C)(N)O", "[C@]([#1])(C)(N)O", "[C@@H](C)(N)O", "[#1][C@](C)(N)O",
                     "[C@](*)(*)(*)*", "[C@@](*)(*)(*)*", "[C@](F)(Cl)Br"];
        let cases: [(&str, [u32; 8]); 8] = [
            ("[H][C@](C)(N)O", [1, 0, 1, 0, 1, 1, 1, 0]),
            ("N[C@@]([H])(C)C(=O)O", [0, 0, 0, 0, 0, 1, 1, 0]),
            ("[H][C@]([H])(O)C", [0, 0, 0, 0, 0, 1, 1, 0]),
            ("[H][C@@H](N)O", [0, 0, 0, 0, 0, 0, 0, 0]),
            ("F[C@@](Cl)([H])Br", [0, 0, 0, 0, 0, 1, 1, 0]),
            ("[H][C@@](F)(Cl)Br", [0, 0, 0, 0, 0, 1, 1, 0]),
            ("F[C@@](Cl)(Br)[H]", [0, 0, 0, 0, 0, 1, 1, 1]),
            ("N[C@@H](C)O", [0, 1, 0, 1, 0, 0, 0, 0]),
        ];
        for (smiles, expected) in cases {
            let got: Vec<u32> = tetra.iter().map(|q| count(q, smiles)).collect();
            assert_eq!(got, expected, "{smiles}");
        }
        let geo = [r"C/C=C/C", r"C/C=C\C", r"[#1]/C=C/C", r"[#1]/C=C\C", r"*/C=C/*"];
        let cases: [(&str, [u32; 5]); 6] = [
            (r"[H]/C(C)=C/C", [0, 1, 1, 0, 1]),
            (r"C\C([H])=C\C", [1, 0, 0, 1, 1]),
            (r"[H]\C(C)=C(/C)[H]", [1, 0, 0, 2, 2]),
            (r"[H]/C(Cl)=C(\[H])Br", [0, 0, 0, 0, 2]),
            (r"C/C(C)=C/C", [1, 1, 0, 0, 1]),
            (r"C/C=C/C", [1, 0, 0, 0, 1]),
        ];
        for (smiles, expected) in cases {
            let got: Vec<u32> = geo.iter().map(|q| count(q, smiles)).collect();
            assert_eq!(got, expected, "{smiles}");
        }
        // aromatic N with a hydrogen atom: degree 3 (`H` counts hydrogen neighbours)
        for smiles in ["[H]n1cccc1", "[H]N1C=CC=C1"] {
            assert_eq!(count("[nD3]", smiles), 1, "{smiles}");
            assert_eq!(count("[nH]", smiles), 1, "{smiles}");
            assert_eq!(count("[nD3H0]", smiles), 0, "{smiles}");
        }
        assert_eq!(count("[nD3]", "c1cc[nH]c1"), 0);
    }
}
