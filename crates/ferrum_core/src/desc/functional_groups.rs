//! Functional group counting descriptors using SMARTS pattern matching.
//!
//! Provides counts for common functional groups, acidic groups, and basic groups.

//! ## File guide: Functional Groups
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct FuncGroupPattern`
//! - `struct FunctionalGroupCounts`
//! - `fn name`
//! - `fn calculate`
//! - `struct AcidicGroupCount`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Functional Groups` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Functional Groups` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/functional_groups.rs` defines functional groups behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Functional Groups` workflows.
//! - Repository unit and parity tests covering `Functional Groups` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/functional_groups.rs`.

use super::typed::{Descriptor, DescriptorValue};
use crate::io::smarts::{QueryGraph, parse_smarts};
use crate::mol::molecule::Molecule;
use once_cell::sync::Lazy;

/// A named SMARTS pattern for functional group detection.
struct FuncGroupPattern {
    #[allow(dead_code)] // documentation only
    name: &'static str,
    query: QueryGraph,
    screen: crate::algo::iso::QueryScreen,
}

macro_rules! fg_patterns {
    ($name:ident, $( ($label:expr, $smarts:expr) ),+ $(,)?) => {
        static $name: Lazy<Vec<FuncGroupPattern>> = Lazy::new(|| {
            vec![
                $(
                    {
                        let q = parse_smarts($smarts).expect(concat!("Bad SMARTS: ", $smarts));
                        FuncGroupPattern { name: $label, screen: crate::algo::iso::QueryScreen::build(&q), query: q }
                    },
                )+
            ]
        });
    };
}

// Core functional group SMARTS patterns
fg_patterns!(
    FUNC_GROUP_PATTERNS,
    ("Hydroxyl", "[OX2H]"),
    ("Aldehyde", "[CX3H1](=O)[#6]"),
    ("Ketone", "[#6][CX3](=O)[#6]"),
    ("CarboxylicAcid", "[CX3](=O)[OX2H1]"),
    ("Ester", "[#6][CX3](=O)[OX2H0][#6]"),
    ("Amide", "[NX3][CX3](=O)[#6]"),
    ("PrimaryAmine", "[NX3;H2;!$(NC=O)]"),
    ("SecondaryAmine", "[NX3;H1;!$(NC=O)]"),
    ("TertiaryAmine", "[NX3;H0;!$(NC=O);!$([NX3](=O)=O)]"),
    ("Nitro", "[$([NX3](=O)=O)]"),
    ("Sulfonamide", "[#16X4]([NX3])(=O)=O"),
    ("Sulfone", "[#16X4](=O)(=O)([#6])[#6]"),
    ("Thiol", "[#16X2H]"),
    ("Ether", "[OD2]([#6])[#6]"),
    ("Nitrile", "[NX1]#[CX2]"),
    ("Halide", "[F,Cl,Br,I]"),
);

/// Functional Group Counts descriptor.
///
/// Counts 16 common functional groups using SMARTS patterns.
/// Output order: Hydroxyl, Aldehyde, Ketone, CarboxylicAcid, Ester, Amide,
/// PrimaryAmine, SecondaryAmine, TertiaryAmine, Nitro, Sulfonamide, Sulfone,
/// Thiol, Ether, Nitrile, Halide.
pub struct FunctionalGroupCounts;

impl Descriptor for FunctionalGroupCounts {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn name(&self) -> &str {
        "FunctionalGroupCounts"
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let iso_target = crate::algo::iso::IsoTarget::new(mol);
        let counts: Vec<u16> = FUNC_GROUP_PATTERNS
            .iter()
            .map(|p| {
                if !p.screen.passes_target(&iso_target) {
                    0
                } else {
                    crate::algo::iso::count_matches_target(&iso_target, &p.query) as u16
                }
            })
            .collect();
        DescriptorValue::IntU16Vec(counts)
    }
}

/// Acidic Group Count (CDK 2.13 `AcidicGroupCountDescriptor`, `nAcid`).
///
/// Counts atoms matching CDK's SMARTS `[$([O;H1]-[C,S,P]=O)]`,
/// `[$([*;-;!$(*~[*;+])])]`, `[$([NH](S(=O)=O)C(F)(F)F)]` and `[$(n1nnnc1)]`
/// with CDK SMARTS semantics: Kekulé bond orders and CDK's Daylight aromaticity
/// (`SmartsPattern` re-perceives it), `-`/`=` only match non-aromatic bonds.
pub struct AcidicGroupCount;

impl Descriptor for AcidicGroupCount {
    fn name(&self) -> &str {
        "AcidicGroupCount"
    }

    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(cdk_group_counts(mol).0)
    }
}

/// Basic Group Count (CDK 2.13 `BasicGroupCountDescriptor`, `nBase`).
///
/// Counts atoms matching CDK's SMARTS `[$([NH2]-[CX4])]`,
/// `[$([NH](-[CX4])-[CX4])]`, `[$(N(-[CX4])(-[CX4])-[CX4])]`,
/// `[$([*;+;!$(*~[*;-])])]`, `[$(N=C-N)]` and `[$(N-C=N)]` (CDK SMARTS
/// semantics, see [`AcidicGroupCount`]).
pub struct BasicGroupCount;

impl Descriptor for BasicGroupCount {
    fn name(&self) -> &str {
        "BasicGroupCount"
    }

    fn calculate(
        &self,
        mol: &mut Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Int(cdk_group_counts(mol).1)
    }
}

/// CDK acidic and basic group counts `(nAcid, nBase)`.
pub(crate) fn cdk_group_counts(mol: &Molecule) -> (u32, u32) {
    use crate::desc::cdk_daylight::{CdkMol, smarts_daylight_aromaticity};
    let n = mol.num_atoms();
    if n == 0 {
        return (0, 0);
    }
    let m = CdkMol::new(mol);
    let (arom, arom_bond) = smarts_daylight_aromaticity(&m);
    let h = |a: usize| m.hcount[a] as u32;
    let aliph = |a: usize, z: u8| m.z[a] == z && !arom[a];
    // `-`, `=`: aliphatic order; implicit bond: single or aromatic
    let ord = |bi: u32, o: u8| !arom_bond[bi as usize] && m.order[bi as usize] == o;
    let single_or_arom = |bi: u32| arom_bond[bi as usize] || m.order[bi as usize] == 1;
    let cx4 = |a: usize| aliph(a, 6) && m.degree(a) + m.hcount[a] as usize == 4;
    // '=O' neighbours via aliphatic double bonds
    let count_dbl_o = |a: usize| m.adj[a].iter().filter(|&&(w, bi)| ord(bi, 2) && aliph(w as usize, 8)).count();

    let mut acid = 0u32;
    let mut base = 0u32;
    for a in 0..n {
        let q = m.charge[a];
        // [$([*;-;!$(*~[*;+])])] / [$([*;+;!$(*~[*;-])])]
        if q == -1 && !m.adj[a].iter().any(|&(w, _)| m.charge[w as usize] == 1) {
            acid += 1;
        }
        if q == 1 && !m.adj[a].iter().any(|&(w, _)| m.charge[w as usize] == -1) {
            base += 1;
        }
        match m.z[a] {
            8 if !arom[a] => {
                // [$([O;H1]-[C,S,P]=O)]
                if h(a) == 1
                    && m.adj[a].iter().any(|&(x, bi)| {
                        let x = x as usize;
                        ord(bi, 1) && (aliph(x, 6) || aliph(x, 16) || aliph(x, 15)) && count_dbl_o(x) >= 1
                    })
                {
                    acid += 1;
                }
            }
            7 if !arom[a] => {
                // [$([NH](S(=O)=O)C(F)(F)F)]
                if h(a) == 1 {
                    let s_ok = m.adj[a].iter().any(|&(x, bi)| {
                        single_or_arom(bi) && aliph(x as usize, 16) && count_dbl_o(x as usize) >= 2
                    });
                    let cf3 = m.adj[a].iter().any(|&(x, bi)| {
                        let x = x as usize;
                        single_or_arom(bi)
                            && aliph(x, 6)
                            && m.adj[x]
                                .iter()
                                .filter(|&&(f, fb)| single_or_arom(fb) && m.z[f as usize] == 9)
                                .count()
                                >= 3
                    });
                    if s_ok && cf3 {
                        acid += 1;
                    }
                }
                // amines: single (aliphatic) bonds to sp3 carbons
                let n_cx4 = m.adj[a].iter().filter(|&&(x, bi)| ord(bi, 1) && cx4(x as usize)).count();
                if h(a) == 2 && n_cx4 >= 1 {
                    base += 1; // [$([NH2]-[CX4])]
                }
                if h(a) == 1 && n_cx4 >= 2 {
                    base += 1; // [$([NH](-[CX4])-[CX4])]
                }
                if n_cx4 >= 3 {
                    base += 1; // [$(N(-[CX4])(-[CX4])-[CX4])]
                }
                // [$(N=C-N)]
                if m.adj[a].iter().any(|&(c, bi)| {
                    let c = c as usize;
                    ord(bi, 2)
                        && aliph(c, 6)
                        && m.adj[c].iter().any(|&(x, cb)| x as usize != a && ord(cb, 1) && aliph(x as usize, 7))
                }) {
                    base += 1;
                }
                // [$(N-C=N)]
                if m.adj[a].iter().any(|&(c, bi)| {
                    let c = c as usize;
                    ord(bi, 1)
                        && aliph(c, 6)
                        && m.adj[c].iter().any(|&(x, cb)| x as usize != a && ord(cb, 2) && aliph(x as usize, 7))
                }) {
                    base += 1;
                }
            }
            7 => {
                // [$(n1nnnc1)]: aromatic n -> n -> n -> n -> c -> back
                if tetrazole_start(&m, &arom, &single_or_arom, a) {
                    acid += 1;
                }
            }
            _ => {}
        }
    }
    (acid, base)
}

/// Whether aromatic N `a` starts a path `n n n n c` closing back to `a`
/// (CDK SMARTS `n1nnnc1`, implicit bonds = single or aromatic).
fn tetrazole_start(
    m: &crate::desc::cdk_daylight::CdkMol,
    arom: &[bool],
    bond_ok: &dyn Fn(u32) -> bool,
    a: usize,
) -> bool {
    let is = |x: usize, z: u8| m.z[x] == z && arom[x];
    for &(b, b1) in &m.adj[a] {
        let b = b as usize;
        if !bond_ok(b1) || !is(b, 7) {
            continue;
        }
        for &(c, b2) in &m.adj[b] {
            let c = c as usize;
            if c == a || !bond_ok(b2) || !is(c, 7) {
                continue;
            }
            for &(d, b3) in &m.adj[c] {
                let d = d as usize;
                if d == a || d == b || !bond_ok(b3) || !is(d, 7) {
                    continue;
                }
                for &(e, b4) in &m.adj[d] {
                    let e = e as usize;
                    if e == a || e == b || e == c || !bond_ok(b4) || !is(e, 6) {
                        continue;
                    }
                    if m.adj[e].iter().any(|&(x, b5)| x as usize == a && bond_ok(b5)) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

const _: () = {
    /// Asserts send sync to guard invariants.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<FunctionalGroupCounts>;
    let _ = assert_send_sync::<AcidicGroupCount>;
    let _ = assert_send_sync::<BasicGroupCount>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    /// Implements fg.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_fg(smiles: &str) -> Vec<u16> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match FunctionalGroupCounts.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::IntU16Vec(v) => v,
            other => panic!("expected IntU16Vec, got {:?}", other),
        }
    }

    /// Implements acidic.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_acidic(smiles: &str) -> u32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match AcidicGroupCount.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::Int(v) => v,
            other => panic!("expected Int, got {:?}", other),
        }
    }

    /// Implements basic.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn calc_basic(smiles: &str) -> u32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        match BasicGroupCount.calculate(
            &mut mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::Int(v) => v,
            other => panic!("expected Int, got {:?}", other),
        }
    }

    #[test]
    /// Runs regression test for fg ethanol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fg_ethanol() {
        let fg = calc_fg("CCO");
        assert!(fg[0] >= 1, "Ethanol should have hydroxyl: {:?}", fg);
    }

    #[test]
    /// Runs regression test for fg acetic acid.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fg_acetic_acid() {
        let fg = calc_fg("CC(=O)O");
        assert!(fg[3] >= 1, "Acetic acid should have COOH: {:?}", fg);
    }

    #[test]
    /// Runs regression test for fg acetone.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fg_acetone() {
        let fg = calc_fg("CC(=O)C");
        assert!(fg[2] >= 1, "Acetone should have ketone: {:?}", fg);
    }

    #[test]
    /// Runs regression test for fg methylamine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fg_methylamine() {
        let fg = calc_fg("CN");
        assert!(
            fg[6] >= 1,
            "Methylamine should have primary amine: {:?}",
            fg
        );
    }

    #[test]
    /// Runs regression test for fg chlorobenzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fg_chlorobenzene() {
        let fg = calc_fg("c1ccc(Cl)cc1");
        assert!(fg[15] >= 1, "Chlorobenzene should have halide: {:?}", fg);
    }

    #[test]
    /// Runs regression test for fg benzene no groups.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_fg_benzene_no_groups() {
        let fg = calc_fg("c1ccccc1");
        let total: u16 = fg.iter().sum();
        assert_eq!(
            total, 0,
            "Benzene should have no functional groups: {:?}",
            fg
        );
    }

    #[test]
    /// Runs regression test for acidic acetic acid.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_acidic_acetic_acid() {
        assert!(calc_acidic("CC(=O)O") >= 1);
    }

    #[test]
    /// Runs regression test for acidic phenol.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_acidic_phenol() {
        // CDK has no phenol pattern
        assert_eq!(calc_acidic("c1ccc(O)cc1"), 0);
    }

    #[test]
    /// Runs regression test for acidic ethane.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_acidic_ethane() {
        assert_eq!(calc_acidic("CC"), 0);
    }

    #[test]
    /// Runs regression test for basic methylamine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_basic_methylamine() {
        assert!(calc_basic("CN") >= 1);
    }

    #[test]
    /// Runs regression test for basic dimethylamine.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_basic_dimethylamine() {
        assert!(calc_basic("CNC") >= 1);
    }

    #[test]
    /// Runs regression test for basic benzene.
    ///
    /// Function context: `crates/ferrum_core/src/desc/functional_groups.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_basic_benzene() {
        assert_eq!(calc_basic("c1ccccc1"), 0);
    }
}
