//! MACCS (Molecular ACCess System) Keys fingerprints.
//!
//! MACCS keys are a set of 166 structural keys used for molecular similarity searching.
//! Each key represents the presence or absence of a specific structural feature.
//!
//! Reference: Durant, J. L., et al. (2002). Reoptimization of MDL Keys for Use in
//! Drug Discovery. Journal of Chemical Information and Computer Sciences, 42(6), 1273-1280.
//!
//! Note: This is the public MACCS keys set with 166 keys (numbered 1-166).
//! Key 0 is reserved and always returns 0.

use super::aromatic_ring_count::AromaticRingCount;
use super::typed::{Descriptor, DescriptorValue};
use crate::algo::iso::{has_at_least_matches, has_substruct_match};
use crate::io::smarts::{AtomPrimitive, QueryAtom, QueryGraph, parse_smarts};
use crate::mol::molecule::Molecule;
use bitvec::prelude::*;
use once_cell::sync::Lazy;
use smallvec::SmallVec;

#[derive(Clone)]
struct QueryScreen {
    min_atoms: usize,
    min_bonds: usize,
    required_elements: SmallVec<[(u8, u16); 8]>,
    requires_aromatic_atom: bool,
}

#[derive(Clone)]
struct ScreenedQuery {
    query: QueryGraph,
    screen: QueryScreen,
}

#[derive(Clone, Copy)]
struct MoleculeScreenStats {
    num_atoms: usize,
    num_bonds: usize,
    element_counts: [u16; 119],
    aromatic_atom_count: u16,
}

impl MoleculeScreenStats {
    fn from_molecule(mol: &Molecule) -> Self {
        let mut element_counts = [0u16; 119];
        let mut aromatic_atom_count = 0u16;
        for atom in &mol.atoms {
            let idx = atom.atomic_number as usize;
            if idx < element_counts.len() {
                element_counts[idx] = element_counts[idx].saturating_add(1);
            }
            if atom.is_aromatic() {
                aromatic_atom_count = aromatic_atom_count.saturating_add(1);
            }
        }
        Self {
            num_atoms: mol.num_atoms(),
            num_bonds: mol.num_bonds(),
            element_counts,
            aromatic_atom_count,
        }
    }

    fn passes(&self, screen: &QueryScreen) -> bool {
        if screen.min_atoms > self.num_atoms || screen.min_bonds > self.num_bonds {
            return false;
        }
        if screen.requires_aromatic_atom && self.aromatic_atom_count == 0 {
            return false;
        }
        screen
            .required_elements
            .iter()
            .all(|&(element, count)| self.element_counts[element as usize] >= count)
    }
}

fn build_query_screen(query: &QueryGraph) -> QueryScreen {
    let mut required = [0u16; 119];
    let mut requires_aromatic_atom = false;
    for atom in &query.atoms {
        if let Some(element) = fixed_atomic_number(atom) {
            let idx = element as usize;
            if idx < required.len() {
                required[idx] = required[idx].saturating_add(1);
            }
        }
        requires_aromatic_atom |= query_atom_requires_aromatic(atom);
    }
    let required_elements = required
        .iter()
        .enumerate()
        .filter_map(|(idx, &count)| (count > 0).then_some((idx as u8, count)))
        .collect();
    QueryScreen {
        min_atoms: query.atoms.len(),
        min_bonds: query.bonds.len(),
        required_elements,
        requires_aromatic_atom,
    }
}

fn fixed_atomic_number(query_atom: &QueryAtom) -> Option<u8> {
    let mut fixed = None;
    for primitive in &query_atom.primitives {
        let Some(element) = fixed_atomic_number_primitive(primitive) else {
            continue;
        };
        if let Some(prev) = fixed {
            if prev != element {
                return None;
            }
        } else {
            fixed = Some(element);
        }
    }
    fixed
}

fn fixed_atomic_number_primitive(primitive: &AtomPrimitive) -> Option<u8> {
    match primitive {
        AtomPrimitive::Element(z) | AtomPrimitive::Aromatic(z) | AtomPrimitive::Aliphatic(z) => {
            Some(*z)
        }
        AtomPrimitive::And(items) => {
            let mut fixed = None;
            for item in items {
                let Some(element) = fixed_atomic_number_primitive(item) else {
                    continue;
                };
                if let Some(prev) = fixed {
                    if prev != element {
                        return None;
                    }
                } else {
                    fixed = Some(element);
                }
            }
            fixed
        }
        AtomPrimitive::Or(options) => {
            let mut elements = options.iter().filter_map(fixed_atomic_number_primitive);
            let first = elements.next()?;
            if elements.all(|element| element == first) {
                Some(first)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn query_atom_requires_aromatic(query_atom: &QueryAtom) -> bool {
    query_atom.primitives.iter().any(primitive_requires_aromatic_atom)
}

fn primitive_requires_aromatic_atom(primitive: &AtomPrimitive) -> bool {
    match primitive {
        AtomPrimitive::AnyAromatic | AtomPrimitive::Aromatic(_) => true,
        AtomPrimitive::Not(_) => false,
        AtomPrimitive::And(options) => options.iter().any(primitive_requires_aromatic_atom),
        AtomPrimitive::Or(options) => {
            !options.is_empty() && options.iter().all(primitive_requires_aromatic_atom)
        }
        AtomPrimitive::Recursive(sub_pattern) => sub_pattern
            .atoms
            .first()
            .is_some_and(query_atom_requires_aromatic),
        _ => false,
    }
}

/// Pre-parsed + screened MACCS keys patterns for fast fingerprint calculation.
static SCREENED_KEYS: Lazy<Vec<Option<ScreenedQuery>>> = Lazy::new(|| {
    MACCS_KEYS
        .iter()
        .map(|key| {
            if key.smarts.is_empty() {
                None // Special keys that need custom logic
            } else {
                parse_smarts(key.smarts).ok().map(|query| {
                    let screen = build_query_screen(&query);
                    ScreenedQuery { query, screen }
                })
            }
        })
        .collect()
});

fn required_match_count(key_number: usize) -> Option<usize> {
    match key_number {
        118 | 120 | 127 | 130 | 131 | 136 | 138 | 142 | 145 | 149 | 159 => Some(1),
        141 | 146 => Some(2),
        140 => Some(3),
        _ => None,
    }
}

/// MACCS key definition.
struct MaccsKey {
    number: usize,
    smarts: &'static str,
    description: &'static str,
}

/// MACCS Keys fingerprint descriptor.
///
/// Returns 166 binary values representing the presence/absence
/// of 166 predefined structural features. Returns bit-packed fingerprint.
pub struct MaccsKeys;

impl Descriptor for MaccsKeys {
    fn name(&self) -> &str {
        "MaccsKeys"
    }

    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        let mut bits = bitvec![u8, Lsb0; 0; 166];
        let stats = MoleculeScreenStats::from_molecule(mol);

        for (i, key_def) in MACCS_KEYS.iter().enumerate() {
            let key_number = key_def.number;
            let key_present = if let Some(Some(screened)) = SCREENED_KEYS.get(i) {
                if !stats.passes(&screened.screen) {
                    false
                } else if let Some(required_count) = required_match_count(key_number) {
                    has_at_least_matches(mol, &screened.query, required_count + 1)
                } else {
                    has_substruct_match(mol, &screened.query)
                }
            } else {
                calculate_special_key(mol, key_number, ctx)
            };

            if key_present {
                bits.set(i, true);
            }
        }

        DescriptorValue::Binary(bits)
    }
}

/// Calculate special keys that don't use SMARTS patterns or need custom logic.
fn calculate_special_key(
    mol: &mut Molecule,
    key_number: usize,
    ctx: &mut crate::desc::context::CalculationContext,
) -> bool {
    match key_number {
        // Key 1: ISOTOPE - Not implemented (requires isotope information in Atom struct)
        1 => false,

        // Key 125: Aromatic Ring > 1
        125 => AromaticRingCount.calculate(mol, ctx).as_int() > 1,

        // Key 166: Fragments > 1
        166 => crate::algo::components::num_fragments(mol) > 1,

        _ => false,
    }
}

/// MACCS Keys definitions (all 166 keys from RDKit).
///
/// These are the publicly available MACCS keys with SMARTS patterns
/// as defined in RDKit. Some keys require special logic beyond SMARTS matching.
///
/// Reference: https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py
static MACCS_KEYS: &[MaccsKey] = &[
    // Key 1: ISOTOPE (special)
    MaccsKey {
        number: 1,
        smarts: "",
        description: "ISOTOPE",
    },
    // Key 2-10
    MaccsKey {
        number: 2,
        smarts: "[#104]",
        description: "atomic num >103",
    },
    MaccsKey {
        number: 3,
        smarts: "[#32,#33,#34,#50,#51,#52,#82,#83,#84]",
        description: "Group IVa,Va,VIa Rows 4-6",
    },
    MaccsKey {
        number: 4,
        smarts: "[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]",
        description: "actinide",
    },
    MaccsKey {
        number: 5,
        smarts: "[Sc,Ti,Y,Zr,Hf]",
        description: "Group IIIB,IVB (Sc...)",
    },
    MaccsKey {
        number: 6,
        smarts: "[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]",
        description: "Lanthanide",
    },
    MaccsKey {
        number: 7,
        smarts: "[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]",
        description: "Group VB,VIB,VIIB",
    },
    MaccsKey {
        number: 8,
        smarts: "[!#6;!#1]1~*~*~*~1",
        description: "QAAA@1",
    },
    MaccsKey {
        number: 9,
        smarts: "[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]",
        description: "Group VIII (Fe...)",
    },
    MaccsKey {
        number: 10,
        smarts: "[Be,Mg,Ca,Sr,Ba,Ra]",
        description: "Group IIa (Alkaline earth)",
    },
    // Key 11-20
    MaccsKey {
        number: 11,
        smarts: "*1~*~*~*~1",
        description: "4M Ring",
    },
    MaccsKey {
        number: 12,
        smarts: "[Cu,Zn,Ag,Cd,Au,Hg]",
        description: "Group IB,IIB (Cu..)",
    },
    MaccsKey {
        number: 13,
        smarts: "[#8]~[#7](~[#6])~[#6]",
        description: "ON(C)C",
    },
    MaccsKey {
        number: 14,
        smarts: "[#16]-[#16]",
        description: "S-S",
    },
    MaccsKey {
        number: 15,
        smarts: "[#8]~[#6](~[#8])~[#8]",
        description: "OC(O)O",
    },
    MaccsKey {
        number: 16,
        smarts: "[!#6;!#1]1~*~*~1",
        description: "QAA@1",
    },
    MaccsKey {
        number: 17,
        smarts: "[#6]#[#6]",
        description: "CTC",
    },
    MaccsKey {
        number: 18,
        smarts: "[#5,#13,#31,#49,#81]",
        description: "Group IIIA (B...)",
    },
    MaccsKey {
        number: 19,
        smarts: "*1~*~*~*~*~*~*~1",
        description: "7M Ring",
    },
    MaccsKey {
        number: 20,
        smarts: "[#14]",
        description: "Si",
    },
    // Key 21-30
    MaccsKey {
        number: 21,
        smarts: "[#6]=[#6](~[!#6;!#1])~[!#6;!#1]",
        description: "C=C(Q)Q",
    },
    MaccsKey {
        number: 22,
        smarts: "*1~*~*~1",
        description: "3M Ring",
    },
    MaccsKey {
        number: 23,
        smarts: "[#7]~[#6](~[#8])~[#8]",
        description: "NC(O)O",
    },
    MaccsKey {
        number: 24,
        smarts: "[#7]-[#8]",
        description: "N-O",
    },
    MaccsKey {
        number: 25,
        smarts: "[#7]~[#6](~[#7])~[#7]",
        description: "NC(N)N",
    },
    MaccsKey {
        number: 26,
        smarts: "[#6]=;@[#6](@*)@*",
        description: "C$=C($A)$A",
    },
    MaccsKey {
        number: 27,
        smarts: "[I]",
        description: "I",
    },
    MaccsKey {
        number: 28,
        smarts: "[!#6;!#1]~[CH2]~[!#6;!#1]",
        description: "QCH2Q",
    },
    MaccsKey {
        number: 29,
        smarts: "[#15]",
        description: "P",
    },
    MaccsKey {
        number: 30,
        smarts: "[#6]~[!#6;!#1](~[#6])(~[#6])~*",
        description: "CQ(C)(C)A",
    },
    // Key 31-40
    MaccsKey {
        number: 31,
        smarts: "[!#6;!#1]~[F,Cl,Br,I]",
        description: "QX",
    },
    MaccsKey {
        number: 32,
        smarts: "[#6]~[#16]~[#7]",
        description: "CSN",
    },
    MaccsKey {
        number: 33,
        smarts: "[#7]~[#16]",
        description: "NS",
    },
    MaccsKey {
        number: 34,
        smarts: "[CH2]=*",
        description: "CH2=A",
    },
    MaccsKey {
        number: 35,
        smarts: "[Li,Na,K,Rb,Cs,Fr]",
        description: "Group IA (Alkali Metal)",
    },
    MaccsKey {
        number: 36,
        smarts: "[#16R]",
        description: "S Heterocycle",
    },
    MaccsKey {
        number: 37,
        smarts: "[#7]~[#6](~[#8])~[#7]",
        description: "NC(O)N",
    },
    MaccsKey {
        number: 38,
        smarts: "[#7]~[#6](~[#6])~[#7]",
        description: "NC(C)N",
    },
    MaccsKey {
        number: 39,
        smarts: "[#8]~[#16](~[#8])~[#8]",
        description: "OS(O)O",
    },
    MaccsKey {
        number: 40,
        smarts: "[#16]-[#8]",
        description: "S-O",
    },
    // Key 41-50
    MaccsKey {
        number: 41,
        smarts: "[#6]#[#7]",
        description: "CTN",
    },
    MaccsKey {
        number: 42,
        smarts: "F",
        description: "F",
    },
    MaccsKey {
        number: 43,
        smarts: "[!#6;!#1;!H0]~*~[!#6;!#1;!H0]",
        description: "QHAQH",
    },
    MaccsKey {
        number: 44,
        smarts: "[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]",
        description: "OTHER",
    },
    MaccsKey {
        number: 45,
        smarts: "[#6]=[#6]~[#7]",
        description: "C=CN",
    },
    MaccsKey {
        number: 46,
        smarts: "Br",
        description: "BR",
    },
    MaccsKey {
        number: 47,
        smarts: "[#16]~*~[#7]",
        description: "SAN",
    },
    MaccsKey {
        number: 48,
        smarts: "[#8]~[!#6;!#1](~[#8])(~[#8])",
        description: "OQ(O)O",
    },
    MaccsKey {
        number: 49,
        smarts: "[!+0]",
        description: "CHARGE",
    },
    MaccsKey {
        number: 50,
        smarts: "[#6]=[#6](~[#6])~[#6]",
        description: "C=C(C)C",
    },
    // Key 51-60
    MaccsKey {
        number: 51,
        smarts: "[#6]~[#16]~[#8]",
        description: "CSO",
    },
    MaccsKey {
        number: 52,
        smarts: "[#7]~[#7]",
        description: "NN",
    },
    MaccsKey {
        number: 53,
        smarts: "[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]",
        description: "QHAAAQH",
    },
    MaccsKey {
        number: 54,
        smarts: "[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]",
        description: "QHAAQH",
    },
    MaccsKey {
        number: 55,
        smarts: "[#8]~[#16]~[#8]",
        description: "OSO",
    },
    MaccsKey {
        number: 56,
        smarts: "[#8]~[#7](~[#8])~[#6]",
        description: "ON(O)C",
    },
    MaccsKey {
        number: 57,
        smarts: "[#8R]",
        description: "O Heterocycle",
    },
    MaccsKey {
        number: 58,
        smarts: "[!#6;!#1]~[#16]~[!#6;!#1]",
        description: "QSQ",
    },
    MaccsKey {
        number: 59,
        smarts: "[#16]!:*:*",
        description: "Snot%A%A",
    },
    MaccsKey {
        number: 60,
        smarts: "[#16]=[#8]",
        description: "S=O",
    },
    // Key 61-70
    MaccsKey {
        number: 61,
        smarts: "*~[#16](~*)~*",
        description: "AS(A)A",
    },
    MaccsKey {
        number: 62,
        smarts: "*@*!@*@*",
        description: "A$!A$A",
    },
    MaccsKey {
        number: 63,
        smarts: "[#7]=[#8]",
        description: "N=O",
    },
    MaccsKey {
        number: 64,
        smarts: "*@*!@[#16]",
        description: "A$A!S",
    },
    MaccsKey {
        number: 65,
        smarts: "c:n",
        description: "C%N",
    },
    MaccsKey {
        number: 66,
        smarts: "[#6]~[#6](~[#6])(~[#6])~*",
        description: "CC(C)(C)A",
    },
    MaccsKey {
        number: 67,
        smarts: "[!#6;!#1]~[#16]",
        description: "QS",
    },
    MaccsKey {
        number: 68,
        smarts: "[!#6;!#1;!H0]~[!#6;!#1;!H0]",
        description: "QHQH",
    },
    MaccsKey {
        number: 69,
        smarts: "[!#6;!#1]~[!#6;!#1;!H0]",
        description: "QQH",
    },
    MaccsKey {
        number: 70,
        smarts: "[!#6;!#1]~[#7]~[!#6;!#1]",
        description: "QNQ",
    },
    // Key 71-80
    MaccsKey {
        number: 71,
        smarts: "[#7]~[#8]",
        description: "NO",
    },
    MaccsKey {
        number: 72,
        smarts: "[#8]~*~*~[#8]",
        description: "OAAO",
    },
    MaccsKey {
        number: 73,
        smarts: "[#16]=*",
        description: "S=A",
    },
    MaccsKey {
        number: 74,
        smarts: "[CH3]~*~[CH3]",
        description: "CH3ACH3",
    },
    MaccsKey {
        number: 75,
        smarts: "*!@[#7]@*",
        description: "A!N$A",
    },
    MaccsKey {
        number: 76,
        smarts: "[#6]=[#6](~*)~*",
        description: "C=C(A)A",
    },
    MaccsKey {
        number: 77,
        smarts: "[#7]~*~[#7]",
        description: "NAN",
    },
    MaccsKey {
        number: 78,
        smarts: "[#6]=[#7]",
        description: "C=N",
    },
    MaccsKey {
        number: 79,
        smarts: "[#7]~*~*~[#7]",
        description: "NAAN",
    },
    MaccsKey {
        number: 80,
        smarts: "[#7]~*~*~*~[#7]",
        description: "NAAAN",
    },
    // Key 81-90
    MaccsKey {
        number: 81,
        smarts: "[#16]~*(~*)~*",
        description: "SA(A)A",
    },
    MaccsKey {
        number: 82,
        smarts: "*~[CH2]~[!#6;!#1;!H0]",
        description: "ACH2QH",
    },
    MaccsKey {
        number: 83,
        smarts: "[!#6;!#1]1~*~*~*~*~1",
        description: "QAAAA@1",
    },
    MaccsKey {
        number: 84,
        smarts: "[NH2]",
        description: "NH2",
    },
    MaccsKey {
        number: 85,
        smarts: "[#6]~[#7](~[#6])~[#6]",
        description: "CN(C)C",
    },
    MaccsKey {
        number: 86,
        smarts: "[C;H2,H3][!#6;!#1][C;H2,H3]",
        description: "CH2QCH2",
    },
    MaccsKey {
        number: 87,
        smarts: "[F,Cl,Br,I]!@*@*",
        description: "X!A$A",
    },
    MaccsKey {
        number: 88,
        smarts: "[#16]",
        description: "S",
    },
    MaccsKey {
        number: 89,
        smarts: "[#8]~*~*~*~[#8]",
        description: "OAAAO",
    },
    MaccsKey {
        number: 90,
        smarts: "[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]",
        description: "QHAACH2A",
    },
    // Key 91-100
    MaccsKey {
        number: 91,
        smarts: "[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]",
        description: "QHAAACH2A",
    },
    MaccsKey {
        number: 92,
        smarts: "[#8]~[#6](~[#7])~[#6]",
        description: "OC(N)C",
    },
    MaccsKey {
        number: 93,
        smarts: "[!#6;!#1]~[CH3]",
        description: "QCH3",
    },
    MaccsKey {
        number: 94,
        smarts: "[!#6;!#1]~[#7]",
        description: "QN",
    },
    MaccsKey {
        number: 95,
        smarts: "[#7]~*~*~[#8]",
        description: "NAAO",
    },
    MaccsKey {
        number: 96,
        smarts: "*1~*~*~*~*~1",
        description: "5 M ring",
    },
    MaccsKey {
        number: 97,
        smarts: "[#7]~*~*~*~[#8]",
        description: "NAAAO",
    },
    MaccsKey {
        number: 98,
        smarts: "[!#6;!#1]1~*~*~*~*~*~1",
        description: "QAAAAA@1",
    },
    MaccsKey {
        number: 99,
        smarts: "[#6]=[#6]",
        description: "C=C",
    },
    MaccsKey {
        number: 100,
        smarts: "*~[CH2]~[#7]",
        description: "ACH2N",
    },
    // Key 101: 8M Ring or larger (up to ring size 14)
    MaccsKey {
        number: 101,
        smarts: "[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]",
        description: "8M Ring or larger",
    },
    // Key 102-110
    MaccsKey {
        number: 102,
        smarts: "[!#6;!#1]~[#8]",
        description: "QO",
    },
    MaccsKey {
        number: 103,
        smarts: "Cl",
        description: "CL",
    },
    MaccsKey {
        number: 104,
        smarts: "[!#6;!#1;!H0]~*~[CH2]~*",
        description: "QHACH2A",
    },
    MaccsKey {
        number: 105,
        smarts: "*@*(@*)@*",
        description: "A$A($A)$A",
    },
    MaccsKey {
        number: 106,
        smarts: "[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]",
        description: "QA(Q)Q",
    },
    MaccsKey {
        number: 107,
        smarts: "[F,Cl,Br,I]~*(~*)~*",
        description: "XA(A)A",
    },
    MaccsKey {
        number: 108,
        smarts: "[CH3]~*~*~*~[CH2]~*",
        description: "CH3AAACH2A",
    },
    MaccsKey {
        number: 109,
        smarts: "*~[CH2]~[#8]",
        description: "ACH2O",
    },
    MaccsKey {
        number: 110,
        smarts: "[#7]~[#6]~[#8]",
        description: "NCO",
    },
    // Key 111-120
    MaccsKey {
        number: 111,
        smarts: "[#7]~*~[CH2]~*",
        description: "NACH2A",
    },
    MaccsKey {
        number: 112,
        smarts: "*~*(~*)(~*)~*",
        description: "AA(A)(A)A",
    },
    MaccsKey {
        number: 113,
        smarts: "[#8]!:*:*",
        description: "Onot%A%A",
    },
    MaccsKey {
        number: 114,
        smarts: "[CH3]~[CH2]~*",
        description: "CH3CH2A",
    },
    MaccsKey {
        number: 115,
        smarts: "[CH3]~*~[CH2]~*",
        description: "CH3ACH2A",
    },
    MaccsKey {
        number: 116,
        smarts: "[$([CH3]~*~*~[CH2]~*),$([CH3]~*1~*~[CH2]1)]",
        description: "CH3AACH2A",
    },
    MaccsKey {
        number: 117,
        smarts: "[#7]~*~[#8]",
        description: "NAO",
    },
    MaccsKey {
        number: 118,
        smarts: "[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]",
        description: "ACH2CH2A",
    }, // count > 1
    MaccsKey {
        number: 119,
        smarts: "[#7]=*",
        description: "N=A",
    },
    MaccsKey {
        number: 120,
        smarts: "[!#6;R]",
        description: "Heterocyclic atom",
    }, // count > 1
    // Key 121-130
    MaccsKey {
        number: 121,
        smarts: "[#7;R]",
        description: "N Heterocycle",
    },
    MaccsKey {
        number: 122,
        smarts: "*~[#7](~*)~*",
        description: "AN(A)A",
    },
    MaccsKey {
        number: 123,
        smarts: "[#8]~[#6]~[#8]",
        description: "OCO",
    },
    MaccsKey {
        number: 124,
        smarts: "[!#6;!#1]~[!#6;!#1]",
        description: "QQ",
    },
    MaccsKey {
        number: 125,
        smarts: "",
        description: "Aromatic Ring > 1",
    }, // special
    MaccsKey {
        number: 126,
        smarts: "*!@[#8]!@*",
        description: "A!O!A",
    },
    MaccsKey {
        number: 127,
        smarts: "*@*!@[#8]",
        description: "A$A!O",
    }, // count > 1
    MaccsKey {
        number: 128,
        smarts: "[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]",
        description: "ACH2AAACH2A",
    },
    MaccsKey {
        number: 129,
        smarts: "[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]",
        description: "ACH2AACH2A",
    },
    MaccsKey {
        number: 130,
        smarts: "[!#6;!#1]~[!#6;!#1]",
        description: "QQ",
    }, // count > 1
    // Key 131-140
    MaccsKey {
        number: 131,
        smarts: "[!#6;!#1;!H0]",
        description: "QH",
    }, // count > 1
    MaccsKey {
        number: 132,
        smarts: "[#8]~*~[CH2]~*",
        description: "OACH2A",
    },
    MaccsKey {
        number: 133,
        smarts: "*@*!@[#7]",
        description: "A$A!N",
    },
    MaccsKey {
        number: 134,
        smarts: "[F,Cl,Br,I]",
        description: "X (HALOGEN)",
    },
    MaccsKey {
        number: 135,
        smarts: "[#7]!:*:*",
        description: "Nnot%A%A",
    },
    MaccsKey {
        number: 136,
        smarts: "[#8]=*",
        description: "O=A",
    }, // count > 1
    MaccsKey {
        number: 137,
        smarts: "[!C;!c;R]",
        description: "Heterocycle",
    },
    MaccsKey {
        number: 138,
        smarts: "[!#6;!#1]~[CH2]~*",
        description: "QCH2A",
    }, // count > 1
    MaccsKey {
        number: 139,
        smarts: "[O;!H0]",
        description: "OH",
    },
    MaccsKey {
        number: 140,
        smarts: "[#8]",
        description: "O",
    }, // count > 3
    // Key 141-150
    MaccsKey {
        number: 141,
        smarts: "[CH3]",
        description: "CH3",
    }, // count > 2
    MaccsKey {
        number: 142,
        smarts: "[#7]",
        description: "N",
    }, // count > 1
    MaccsKey {
        number: 143,
        smarts: "*@*!@[#8]",
        description: "A$A!O",
    },
    MaccsKey {
        number: 144,
        smarts: "*!:*:*!:*",
        description: "Anot%A%Anot%A",
    },
    MaccsKey {
        number: 145,
        smarts: "*1~*~*~*~*~*~1",
        description: "6M ring",
    }, // count > 1
    MaccsKey {
        number: 146,
        smarts: "[#8]",
        description: "O",
    }, // count > 2
    MaccsKey {
        number: 147,
        smarts: "[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]",
        description: "ACH2CH2A",
    },
    MaccsKey {
        number: 148,
        smarts: "*~[!#6;!#1](~*)~*",
        description: "AQ(A)A",
    },
    MaccsKey {
        number: 149,
        smarts: "[C;H3,H4]",
        description: "CH3",
    }, // count > 1
    MaccsKey {
        number: 150,
        smarts: "*!@*@*!@*",
        description: "A!A$A!A",
    },
    // Key 151-160
    MaccsKey {
        number: 151,
        smarts: "[#7;!H0]",
        description: "NH",
    },
    MaccsKey {
        number: 152,
        smarts: "[#8]~[#6](~[#6])~[#6]",
        description: "OC(C)C",
    },
    MaccsKey {
        number: 153,
        smarts: "[!#6;!#1]~[CH2]~*",
        description: "QCH2A",
    },
    MaccsKey {
        number: 154,
        smarts: "[#6]=[#8]",
        description: "C=O",
    },
    MaccsKey {
        number: 155,
        smarts: "*!@[CH2]!@*",
        description: "A!CH2!A",
    },
    MaccsKey {
        number: 156,
        smarts: "[#7]~*(~*)~*",
        description: "NA(A)A",
    },
    MaccsKey {
        number: 157,
        smarts: "[#6]-[#8]",
        description: "C-O",
    },
    MaccsKey {
        number: 158,
        smarts: "[#6]-[#7]",
        description: "C-N",
    },
    MaccsKey {
        number: 159,
        smarts: "[#8]",
        description: "O",
    }, // count > 1
    MaccsKey {
        number: 160,
        smarts: "[C;H3,H4]",
        description: "CH3",
    },
    // Key 161-166
    MaccsKey {
        number: 161,
        smarts: "[#7]",
        description: "N",
    },
    MaccsKey {
        number: 162,
        smarts: "a",
        description: "Aromatic",
    },
    MaccsKey {
        number: 163,
        smarts: "*1~*~*~*~*~*~1",
        description: "6M Ring",
    },
    MaccsKey {
        number: 164,
        smarts: "[#8]",
        description: "O",
    },
    MaccsKey {
        number: 165,
        smarts: "[R]",
        description: "Ring",
    },
    MaccsKey {
        number: 166,
        smarts: "",
        description: "Fragments",
    }, // special: num frags > 1
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::aromaticity::perceive_aromaticity;
    use crate::algo::rings::find_sssr;
    use crate::algo::valency::assign_implicit_hydrogens;
    use crate::io::smiles::parse_smiles;

    fn parse_and_prepare(smiles: &str) -> Molecule {
        let mut mol = parse_smiles(smiles).unwrap();
        assign_implicit_hydrogens(&mut mol).unwrap();
        find_sssr(&mut mol);
        perceive_aromaticity(&mut mol);
        mol
    }

    fn get_set_bits(bits: &BitVec<u8, Lsb0>) -> Vec<usize> {
        bits.iter()
            .by_vals()
            .enumerate()
            .filter_map(|(i, val)| if val { Some(i + 1) } else { None })
            .collect()
    }

    fn calc_maccs(smiles: &str) -> BitVec<u8, Lsb0> {
        let mut mol = parse_and_prepare(smiles);
        MaccsKeys.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()).as_binary().clone()
    }

    #[test]
    fn test_pattern_parsing() {
        let parsed_count = SCREENED_KEYS.iter().filter(|p| p.is_some()).count();
        assert!(
            parsed_count >= 150,
            "Too many patterns failed to parse: only {} succeeded",
            parsed_count
        );
    }

    #[test]
    fn test_maccs_descriptor_name() {
        let desc = MaccsKeys;
        assert_eq!(Descriptor::name(&desc), "MaccsKeys");
    }

    #[test]
    fn test_maccs_benzene() {
        let fp = calc_maccs("c1ccccc1");
        assert!(fp[161], "Key 162: Aromatic");
        assert!(fp[162], "Key 163: 6M Ring");
        assert!(fp[164], "Key 165: Ring");
    }

    #[test]
    fn test_maccs_ethanol() {
        let fp = calc_maccs("CCO");
        assert!(fp[163], "Key 164: O");
        assert!(fp[138], "Key 139: OH");
        assert!(fp[156], "Key 157: C-O");
        assert!(fp[159], "Key 160: CH3");
    }

    #[test]
    fn test_maccs_aspirin() {
        let fp = calc_maccs("CC(=O)Oc1ccccc1C(=O)O");
        let set_bits = get_set_bits(&fp);
        assert!(fp[153], "Key 154: C=O");
        assert!(fp[161], "Key 162: Aromatic");
        assert!(fp[163], "Key 164: O");
        assert!(fp[164], "Key 165: Ring");
        let count = set_bits.len();
        assert!(
            count >= 15 && count <= 25,
            "Expected ~21 bits, got {}",
            count
        );
    }

    #[test]
    fn test_maccs_caffeine() {
        let fp = calc_maccs("CN1C=NC2=C1C(=O)N(C(=O)N2C)C");
        let set_bits = get_set_bits(&fp);
        assert!(fp[120], "Key 121: N Heterocycle");
        assert!(fp[160], "Key 161: N");
        assert!(fp[163], "Key 164: O");
        assert!(fp[164], "Key 165: Ring");
        assert!(
            set_bits.len() >= 20,
            "Expected at least 20 bits, got {}",
            set_bits.len()
        );
    }

    #[test]
    fn test_maccs_hexane() {
        let fp = calc_maccs("CCCCCC");
        assert!(fp[159], "Key 160: CH3");
        assert!(!fp[163], "Key 164: O (should not be set)");
        assert!(!fp[164], "Key 165: Ring (should not be set)");
        let set_bits = get_set_bits(&fp);
        assert!(
            set_bits.len() >= 5,
            "Expected at least 5 bits, got {}",
            set_bits.len()
        );
    }

    #[test]
    fn test_maccs_pyridine() {
        let fp = calc_maccs("c1ccncc1");
        assert!(fp[64], "Key 65: C%N");
        assert!(fp[120], "Key 121: N Heterocycle");
        assert!(fp[136], "Key 137: Heterocycle");
        assert!(fp[160], "Key 161: N");
        assert!(fp[161], "Key 162: Aromatic");
        assert!(fp[164], "Key 165: Ring");
    }

    #[test]
    fn test_maccs_threshold_keys() {
        let fp = calc_maccs("CCO");
        assert!(
            !fp[145],
            "Key 146 (1-based): O count > 2 should be false for ethanol"
        );
        assert!(
            !fp[139],
            "Key 140 (1-based): O count > 3 should be false for ethanol"
        );

        let fp = calc_maccs("OCC(O)CO");
        assert!(
            fp[145],
            "Key 146 (1-based): O count > 2 should be true for glycerol"
        );
        assert!(
            !fp[139],
            "Key 140 (1-based): O count > 3 should be false for glycerol"
        );
    }
}

// Thread safety
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MaccsKeys>;
};
