//! CDK 2.13 `AminoAcidCountDescriptor` (`CDK_nA` ... `CDK_nW`).
//!
//! CDK counts, for each of the 20 amino-acid residue templates of
//! `AminoAcids.createAAs()`, the number of subgraph mappings that
//! `UniversalIsomorphismTester.getSubgraphMaps` finds. That tester maps
//! *bonds*, not atoms: a mapping is an injective map from template bonds to
//! molecule bonds such that
//!
//! - mapped bonds are compatible (same order and aromatic flag, or both
//!   aromatic; end symbols equal in either orientation), and
//! - every pair of template bonds shares an atom of a given symbol exactly when
//!   the pair of mapped bonds shares an atom of that symbol.
//!
//! Every such mapping is a solution of CDK's resolution-graph clique search
//! (findAllStructure, findAllMap), so symmetric residues are counted once per
//! automorphism (valine's two methyls give 2 per residue). The molecule's bond
//! orders and aromatic flags are those of the CDK reference preparation
//! (`desc::cdk_cycles::daylight_prepared_aromaticity`); the templates carry the
//! orders and `cdk()`-model aromatic flags CDK assigns them. Searching the edge
//! mappings directly gives exactly CDK's counts without its slow clique search.

use crate::desc::cdk_daylight::CdkMol;
use crate::desc::cdk_perception::CdkView;
use crate::desc::cdk_simple::with_prepared_bonds;
use crate::desc::context::CalculationContext;
use crate::desc::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// One-letter codes of `AminoAcids.createAAs()`, in CDK's output order
/// (columns `CDK_n<code>`).
pub const AMINO_ACID_CODES: [&str; 20] = [
    "A", "R", "N", "D", "C", "F", "Q", "E", "G", "H", "I", "P", "L", "K", "M", "S", "T", "Y", "V", "W",
];

/// Column names (`CDK_nA`, ...).
pub fn amino_acid_names() -> Vec<String> {
    AMINO_ACID_CODES.iter().map(|c| format!("CDK_n{c}")).collect()
}

/// A residue template: atom elements and bonds `(begin, end, order, aromatic)`.
struct Template {
    atoms: &'static [u8],
    bonds: &'static [(u8, u8, u8, bool)],
}

/// Index of glycine in [`TEMPLATES`].
const GLYCINE: usize = 8;
/// Index of alanine in [`TEMPLATES`].
const ALANINE: usize = 0;

/// Bit of a bond kind used by the templates (0 for other kinds): aromatic
/// bonds match any aromatic bond with the same end elements, other bonds need
/// the same order.
fn bond_kind(za: u8, zb: u8, order: u8, arom: bool) -> u32 {
    let (lo, hi) = if za <= zb { (za, zb) } else { (zb, za) };
    let bit = match (lo, hi, arom, order) {
        (C, C, true, _) => 0,
        (C, N, true, _) => 1,
        (C, C, false, 1) => 2,
        (C, C, false, 2) => 3,
        (C, N, false, 1) => 4,
        (C, N, false, 2) => 5,
        (C, O, false, 1) => 6,
        (C, O, false, 2) => 7,
        (C, S, false, 1) => 8,
        (N, N, true, _) => 9,
        _ => return 0,
    };
    1 << bit
}

const C: u8 = 6;
const N: u8 = 7;
const O: u8 = 8;
const S: u8 = 16;

/// Templates as `AminoAcids.createAAs()` builds them (residue atoms after
/// `removeAcidicOxygen`, CDK `list_aminoacids.cml` atom and bond order).
#[rustfmt::skip]
const TEMPLATES: [Template; 20] = [
    // A
    Template {
        atoms: &[C, N, C, C, O],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (3, 4, 2, false),
        ],
    },
    // R
    Template {
        atoms: &[C, C, C, N, C, O, C, N, C, N, N],
        bonds: &[
            (0, 1, 1, false), (0, 2, 1, false), (0, 3, 1, false), (1, 4, 1, false), (2, 5, 2, false),
            (4, 6, 1, false), (6, 7, 1, false), (7, 8, 1, false), (8, 9, 2, false), (8, 10, 1, false),
        ],
    },
    // N
    Template {
        atoms: &[C, N, C, C, C, O, N, O],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 1, false), (3, 5, 2, false),
            (4, 6, 1, false), (4, 7, 2, false),
        ],
    },
    // D
    Template {
        atoms: &[C, N, C, C, C, O, O, O],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 1, false), (3, 5, 2, false),
            (4, 6, 1, false), (4, 7, 2, false),
        ],
    },
    // C
    Template {
        atoms: &[C, N, C, C, S, O],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 1, false), (3, 5, 2, false),
        ],
    },
    // F
    Template {
        atoms: &[C, N, C, C, O, C, C, C, C, C, C],
        bonds: &[
            (0, 1, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 2, false), (3, 5, 1, false),
            (5, 6, 1, true), (6, 7, 2, true), (7, 8, 1, true), (8, 9, 2, true), (9, 10, 1, true),
            (10, 5, 2, true),
        ],
    },
    // Q
    Template {
        atoms: &[C, N, C, C, O, C, C, O, N],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 2, false), (3, 5, 1, false),
            (5, 6, 1, false), (6, 7, 2, false), (6, 8, 1, false),
        ],
    },
    // E
    Template {
        atoms: &[C, N, C, C, O, C, C, O, O],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 2, false), (3, 5, 1, false),
            (5, 6, 1, false), (6, 7, 2, false), (6, 8, 1, false),
        ],
    },
    // G
    Template {
        atoms: &[C, N, C, O],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (2, 3, 2, false),
        ],
    },
    // H
    Template {
        atoms: &[C, N, C, C, O, C, N, C, N, C],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 2, false), (3, 5, 1, false),
            (5, 6, 1, true), (6, 7, 2, true), (7, 8, 1, true), (8, 9, 1, true), (9, 5, 2, true),
        ],
    },
    // I
    Template {
        atoms: &[C, N, C, C, O, C, C, C],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 2, false), (3, 5, 1, false),
            (3, 6, 1, false), (6, 7, 1, false),
        ],
    },
    // P
    Template {
        atoms: &[C, C, C, N, C, C, O],
        bonds: &[
            (1, 0, 1, false), (2, 0, 1, false), (1, 3, 1, false), (1, 4, 1, false), (5, 2, 1, false),
            (3, 5, 1, false), (4, 6, 2, false),
        ],
    },
    // L
    Template {
        atoms: &[C, N, C, C, O, C, C, C],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 2, false), (3, 5, 1, false),
            (5, 6, 1, false), (5, 7, 1, false),
        ],
    },
    // K
    Template {
        atoms: &[N, C, C, C, O, C, C, C, N],
        bonds: &[
            (1, 0, 1, false), (1, 2, 1, false), (1, 3, 1, false), (2, 4, 2, false), (3, 5, 1, false),
            (5, 6, 1, false), (6, 7, 1, false), (7, 8, 1, false),
        ],
    },
    // M
    Template {
        atoms: &[C, N, C, C, C, O, S, C],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 1, false), (3, 5, 2, false),
            (4, 6, 1, false), (6, 7, 1, false),
        ],
    },
    // S
    Template {
        atoms: &[C, N, C, C, O, O],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 1, false), (3, 5, 2, false),
        ],
    },
    // T
    Template {
        atoms: &[C, N, C, C, O, O, C],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 2, false), (3, 5, 1, false),
            (3, 6, 1, false),
        ],
    },
    // Y
    Template {
        atoms: &[C, N, C, C, O, C, C, C, C, C, C, O],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 2, false), (3, 5, 1, false),
            (5, 6, 1, true), (7, 5, 2, true), (6, 8, 2, true), (9, 7, 1, true), (8, 10, 1, true),
            (10, 9, 2, true), (10, 11, 1, false),
        ],
    },
    // V
    Template {
        atoms: &[C, N, C, C, O, C, C],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (2, 4, 2, false), (3, 5, 1, false),
            (3, 6, 1, false),
        ],
    },
    // W
    Template {
        atoms: &[C, C, N, C, C, O, C, N, C, C, C, C, C, C],
        bonds: &[
            (1, 0, 1, false), (0, 2, 1, false), (0, 3, 1, false), (1, 4, 1, false), (3, 5, 2, false),
            (4, 6, 2, true), (6, 7, 1, true), (7, 8, 1, true), (8, 9, 1, true), (9, 4, 1, true),
            (9, 10, 2, true), (10, 11, 1, true), (11, 12, 2, true), (12, 13, 1, true), (13, 8, 2, true),
        ],
    },
];

/// CDK 2.13 `AminoAcidCountDescriptor`: IntU16Vec of the 20 counts in the order
/// of [`AMINO_ACID_CODES`].
pub struct CdkAminoAcidCount;

impl Descriptor for CdkAminoAcidCount {
    fn name(&self) -> &str {
        "AminoAcidCount"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::IntU16Vec(
            CdkView::with_legacy_cached(mol, ctx, |view, _| with_prepared_bonds(mol, view, amino_acid_counts)).to_vec(),
        )
    }
}

/// A bond of the molecule graph or a template: end elements, order, aromatic flag.
#[derive(Clone, Copy)]
struct BondInfo {
    a: u32,
    b: u32,
    order: u8,
    arom: bool,
}

/// `getCommonSymbol`: element of the shared atom of two bonds (0: none).
#[inline]
fn common(x: &BondInfo, y: &BondInfo, z: &[u8]) -> u8 {
    if x.a == y.a || x.b == y.a {
        z[y.a as usize]
    } else if x.a == y.b || x.b == y.b {
        z[y.b as usize]
    } else {
        0
    }
}

/// Counts of [`AMINO_ACID_CODES`] exactly as CDK 2.13 computes them.
///
/// `m` is CDK's parser view of the molecule, `arom_bond` the bond aromaticity
/// of the reference preparation.
pub(crate) fn amino_acid_counts(m: &CdkMol, arom_bond: &[bool]) -> [u16; 20] {
    let mol_z: Vec<u8> = m.z.iter().map(|&z| if z == 0 { u8::MAX } else { z }).collect();
    let mol_bonds: Vec<BondInfo> = m
        .ends
        .iter()
        .enumerate()
        .map(|(bi, &(s, t))| BondInfo { a: s, b: t, order: m.order[bi], arom: arom_bond[bi] })
        .collect();
    let mut counts = [0u16; 20];
    // every template starts with its carbonyl C=O (see `Search::prepare`)
    let carbonyls: Vec<u32> = (0..mol_bonds.len() as u32)
        .filter(|&g| {
            let b = &mol_bonds[g as usize];
            let (za, zb) = (mol_z[b.a as usize], mol_z[b.b as usize]);
            b.order == 2 && !b.arom && ((za == C && zb == O) || (za == O && zb == C))
        })
        .collect();
    if carbonyls.is_empty() {
        return counts;
    }
    // bond kinds present (a template bond needs a molecule bond of its kind)
    let mut kinds = 0u32;
    for b in &mol_bonds {
        kinds |= bond_kind(mol_z[b.a as usize], mol_z[b.b as usize], b.order, b.arom);
    }
    let templates = ordered_templates();
    let mut search = Search {
        mol_z: &mol_z,
        mol_bonds: &mol_bonds,
        adj: &m.adj,
        t: &templates[GLYCINE],
        image: [u32::MAX; MAX_TEMPLATE_BONDS],
        used: vec![false; mol_bonds.len()],
        first: &carbonyls,
        count: 0,
    };
    // Glycine's bonds (N-C-C=O) are part of every template and alanine's
    // (N-C(-C)-C=O) of every template but glycine; restricting a mapping to
    // them gives a glycine (alanine) mapping, so without one there is none.
    for k in [GLYCINE, ALANINE].into_iter().chain((0..TEMPLATES.len()).filter(|&k| k != GLYCINE && k != ALANINE)) {
        let t = &templates[k];
        if t.needed & !kinds != 0 {
            continue;
        }
        search.t = t;
        search.count = 0;
        search.extend(0);
        counts[k] = search.count.min(u16::MAX as u64) as u16;
        if (k == GLYCINE || k == ALANINE) && counts[k] == 0 {
            break;
        }
    }
    counts
}

/// Most bonds of a template (tryptophan).
const MAX_TEMPLATE_BONDS: usize = 15;

/// A template prepared for the search: bonds in search order.
struct OrderedTemplate {
    z: &'static [u8],
    /// Template bonds in search order (the carbonyl first, then breadth first).
    bonds: Vec<BondInfo>,
    /// For bond k > 0, an earlier template bond sharing an atom with it.
    anchor: Vec<usize>,
    /// [`bond_kind`] bits of its bonds.
    needed: u32,
}

fn ordered_templates() -> &'static [OrderedTemplate] {
    static ORDERED: std::sync::OnceLock<Vec<OrderedTemplate>> = std::sync::OnceLock::new();
    ORDERED.get_or_init(|| TEMPLATES.iter().map(OrderedTemplate::new).collect())
}

impl OrderedTemplate {
    fn new(t: &'static Template) -> Self {
        let all: Vec<BondInfo> = t
            .bonds
            .iter()
            .map(|&(a, b, order, arom)| BondInfo { a: a as u32, b: b as u32, order, arom })
            .collect();
        let start = all
            .iter()
            .position(|b| b.order == 2 && !b.arom && (t.atoms[b.a as usize] == O || t.atoms[b.b as usize] == O))
            .expect("every template has a carbonyl");
        let mut order = vec![start];
        let mut anchor = vec![usize::MAX];
        let mut placed = vec![false; all.len()];
        placed[start] = true;
        let mut i = 0;
        while i < order.len() {
            let cur = all[order[i]];
            for (j, b) in all.iter().enumerate() {
                if !placed[j] && (b.a == cur.a || b.a == cur.b || b.b == cur.a || b.b == cur.b) {
                    placed[j] = true;
                    anchor.push(i);
                    order.push(j);
                }
            }
            i += 1;
        }
        let needed = all
            .iter()
            .fold(0, |acc, b| acc | bond_kind(t.atoms[b.a as usize], t.atoms[b.b as usize], b.order, b.arom));
        OrderedTemplate { z: t.atoms, bonds: order.iter().map(|&j| all[j]).collect(), anchor, needed }
    }
}

struct Search<'a> {
    mol_z: &'a [u8],
    mol_bonds: &'a [BondInfo],
    /// Neighbours `(atom, bond)` per atom.
    adj: &'a [smallvec::SmallVec<[(u32, u32); 4]>],
    t: &'a OrderedTemplate,
    image: [u32; MAX_TEMPLATE_BONDS],
    used: Vec<bool>,
    /// Candidate images of the first template bond (the carbonyl).
    first: &'a [u32],
    count: u64,
}

impl Search<'_> {
    /// Node compatibility of a template bond and a molecule bond.
    #[inline]
    fn compatible(&self, t: &BondInfo, g: &BondInfo) -> bool {
        if !((g.order == t.order && g.arom == t.arom) || (g.arom && t.arom)) {
            return false;
        }
        let (ga, gb) = (self.mol_z[g.a as usize], self.mol_z[g.b as usize]);
        let (ta, tb) = (self.t.z[t.a as usize], self.t.z[t.b as usize]);
        (ga == ta && gb == tb) || (ga == tb && gb == ta)
    }

    /// Whether molecule bond `g` can be the image of template bond `k`, given
    /// the images of bonds `0..k`.
    fn consistent(&self, k: usize, g: u32) -> bool {
        let gb = &self.mol_bonds[g as usize];
        let tb = &self.t.bonds[k];
        (0..k).all(|j| {
            let gj = &self.mol_bonds[self.image[j] as usize];
            common(gb, gj, self.mol_z) == common(tb, &self.t.bonds[j], self.t.z)
        })
    }

    fn extend(&mut self, k: usize) {
        if k == self.t.bonds.len() {
            self.count += 1;
            return;
        }
        let tb = self.t.bonds[k];
        if k == 0 {
            for i in 0..self.first.len() {
                let g = self.first[i];
                if self.compatible(&tb, &self.mol_bonds[g as usize]) {
                    self.try_bond(k, g);
                }
            }
            return;
        }
        let anchor_image = self.mol_bonds[self.image[self.t.anchor[k]] as usize];
        for end in [anchor_image.a, anchor_image.b] {
            for idx in 0..self.adj[end as usize].len() {
                let g = self.adj[end as usize][idx].1;
                if !self.used[g as usize]
                    && self.compatible(&tb, &self.mol_bonds[g as usize])
                    && self.consistent(k, g)
                {
                    self.try_bond(k, g);
                }
            }
        }
    }

    fn try_bond(&mut self, k: usize, g: u32) {
        self.used[g as usize] = true;
        self.image[k] = g;
        self.extend(k + 1);
        self.used[g as usize] = false;
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<CdkAminoAcidCount>;
};
