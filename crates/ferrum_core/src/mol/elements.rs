//! ## File guide: Elements
//!
//! This file belongs to FerrumChem's **molecular graph model layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn symbol_to_atomic_number`
//! - `fn is_aromatic_symbol`
//! - `mod tests`
//! - `fn test_atomic_masses`
//! - `fn test_symbol_lookup`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Elements` encodes graph primitives with compact integer indices to keep memory predictable on large CSV batches.
//! - `Elements` expects topology invariants (bond endpoints, adjacency symmetry) to hold after all mutating operations.
//! - Bitflag/state fields in `Elements` are treated as cached derivations and must be invalidated after structural edits.
//! - Specific scope anchor: `crates/ferrum_core/src/mol/elements.rs` defines elements behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Graph-based cheminformatics data-model conventions for atom/bond/molecule containers.
//! - IUPAC element/valence background relevant to `Elements` typed fields.
//! - Rust ownership/borrowing patterns for safe high-performance molecular graph manipulation.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/mol/elements.rs`.

/// Element data lookup tables for high-performance atomic operations.
/// Uses const arrays indexed by atomic number for zero-cost abstractions.

/// Standard atomic masses (in Daltons) indexed by atomic number.
/// Index 0 is unused, elements 1-118 are defined.
pub const ATOMIC_MASSES: [f32; 119] = [
    0.0,    // 0: placeholder
    1.008,  // 1: H
    4.003,  // 2: He
    6.941,  // 3: Li
    9.012,  // 4: Be
    10.81,  // 5: B
    12.011, // 6: C
    14.007, // 7: N
    15.999, // 8: O
    18.998, // 9: F
    20.180, // 10: Ne
    22.990, // 11: Na
    24.305, // 12: Mg
    26.982, // 13: Al
    28.085, // 14: Si
    30.974, // 15: P
    32.067, // 16: S
    35.453, // 17: Cl
    39.948, // 18: Ar
    39.098, // 19: K
    40.078, // 20: Ca
    44.956, // 21: Sc
    47.867, // 22: Ti
    50.942, // 23: V
    51.996, // 24: Cr
    54.938, // 25: Mn
    55.845, // 26: Fe
    58.933, // 27: Co
    58.693, // 28: Ni
    63.546, // 29: Cu
    65.38,  // 30: Zn
    69.723, // 31: Ga
    72.630, // 32: Ge
    74.922, // 33: As
    78.971, // 34: Se
    79.904, // 35: Br
    83.798, // 36: Kr
    85.468, // 37: Rb
    87.62,  // 38: Sr
    88.906, // 39: Y
    91.224, // 40: Zr
    92.906, // 41: Nb
    95.95,  // 42: Mo
    98.0,   // 43: Tc
    101.07, // 44: Ru
    102.91, // 45: Rh
    106.42, // 46: Pd
    107.87, // 47: Ag
    112.41, // 48: Cd
    114.82, // 49: In
    118.71, // 50: Sn
    121.76, // 51: Sb
    127.60, // 52: Te
    126.90, // 53: I
    131.29, // 54: Xe
    132.91, // 55: Cs
    137.33, // 56: Ba
    138.91, // 57: La
    140.12, // 58: Ce
    140.91, // 59: Pr
    144.24, // 60: Nd
    145.0,  // 61: Pm
    150.36, // 62: Sm
    151.96, // 63: Eu
    157.25, // 64: Gd
    158.93, // 65: Tb
    162.50, // 66: Dy
    164.93, // 67: Ho
    167.26, // 68: Er
    168.93, // 69: Tm
    173.05, // 70: Yb
    174.97, // 71: Lu
    178.49, // 72: Hf
    180.95, // 73: Ta
    183.84, // 74: W
    186.21, // 75: Re
    190.23, // 76: Os
    192.22, // 77: Ir
    195.08, // 78: Pt
    196.97, // 79: Au
    200.59, // 80: Hg
    204.38, // 81: Tl
    207.2,  // 82: Pb
    208.98, // 83: Bi
    209.0,  // 84: Po
    210.0,  // 85: At
    222.0,  // 86: Rn
    223.0,  // 87: Fr
    226.0,  // 88: Ra
    227.0,  // 89: Ac
    232.04, // 90: Th
    231.04, // 91: Pa
    238.03, // 92: U
    237.0,  // 93: Np
    244.0,  // 94: Pu
    243.0,  // 95: Am
    247.0,  // 96: Cm
    247.0,  // 97: Bk
    251.0,  // 98: Cf
    252.0,  // 99: Es
    257.0,  // 100: Fm
    258.0,  // 101: Md
    259.0,  // 102: No
    262.0,  // 103: Lr
    267.0,  // 104: Rf
    268.0,  // 105: Db
    271.0,  // 106: Sg
    272.0,  // 107: Bh
    270.0,  // 108: Hs
    276.0,  // 109: Mt
    281.0,  // 110: Ds
    280.0,  // 111: Rg
    285.0,  // 112: Cn
    284.0,  // 113: Nh
    289.0,  // 114: Fl
    288.0,  // 115: Mc
    293.0,  // 116: Lv
    294.0,  // 117: Ts
    294.0,  // 118: Og
];

/// Element symbols indexed by atomic number.
pub const ELEMENT_SYMBOLS: [&str; 119] = [
    "*", // 0: wildcard/unknown
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", // 1-10
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", // 11-20
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", // 21-30
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", // 31-40
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", // 41-50
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", // 51-60
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", // 61-70
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", // 71-80
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", // 81-90
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", // 91-100
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", // 101-110
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", // 111-118
];

/// Typical valences for organic elements (for implicit hydrogen calculation).
/// Returns common valence states as a slice.
#[inline]
pub const fn get_valences(atomic_number: u8) -> &'static [u8] {
    match atomic_number {
        1 => &[1],        // H
        5 => &[3],        // B
        6 => &[4],        // C
        7 => &[3, 5],     // N
        8 => &[2],        // O
        9 => &[1],        // F
        14 => &[4],       // Si
        15 => &[3, 5],    // P
        16 => &[2, 4, 6], // S
        17 => &[1],       // Cl
        35 => &[1],       // Br
        53 => &[1],       // I
        _ => &[],         // Unknown/metal
    }
}

/// Aromatic valences per the OpenSMILES specification.
///
/// For aromatic atoms, implicit H is calculated using degree (number of neighbors)
/// and this aromatic valence table, NOT standard bond-order-based valences.
///
/// The aromatic valence represents the maximum number of σ-bonds (including H)
/// the atom can form while participating in the aromatic π system.
///
/// Examples:
/// - Aromatic C (benzene): aromatic valence 3 → degree 2 + 1 H
/// - Aromatic N (pyridine): aromatic valence 2 → degree 2 + 0 H
/// - Aromatic N (pyrrole): aromatic valence 3 → degree 2 + 1 H (but [nH] is explicit)
/// - Aromatic S (thiophene): aromatic valence 2 → degree 2 + 0 H
#[inline]
pub const fn get_aromatic_valences(atomic_number: u8) -> &'static [u8] {
    match atomic_number {
        5 => &[3],        // B
        6 => &[3],        // C: 1 π electron, 3 σ bonds
        7 => &[2, 3],     // N: pyridine-type (2) or pyrrole-type (3)
        8 => &[2],        // O: lone pair contributes 2 π electrons
        15 => &[2, 3, 5], // P
        16 => &[2],       // S: lone pair contributes 2 π electrons
        34 => &[2],       // Se
        52 => &[2],       // Te
        _ => &[],         // Non-aromatic or unknown
    }
}

/// Number of valence (outer shell) electrons for an element.
/// Returns `true` if this element is a transition metal with no default valence.
/// These elements can form dative/coordinate bonds with donor atoms (N, O, S, P).
#[inline]
pub const fn is_transition_metal(atomic_number: u8) -> bool {
    matches!(
        atomic_number,
        21..=30 | 39..=48 | 57..=80 | 89..=99
    )
}

/// Used for calculating total valence electron count in molecules.
///
/// # Examples
/// - H (1): 1 valence electron
/// - C (6): 4 valence electrons
/// - N (7): 5 valence electrons
/// - O (8): 6 valence electrons
#[inline]
pub const fn outer_electrons(atomic_number: u8) -> u8 {
    match atomic_number {
        1 => 1,  // H
        2 => 2,  // He
        3 => 1,  // Li
        4 => 2,  // Be
        5 => 3,  // B
        6 => 4,  // C
        7 => 5,  // N
        8 => 6,  // O
        9 => 7,  // F
        10 => 8, // Ne
        11 => 1, // Na
        12 => 2, // Mg
        13 => 3, // Al
        14 => 4, // Si
        15 => 5, // P
        16 => 6, // S
        17 => 7, // Cl
        18 => 8, // Ar
        19 => 1, // K
        20 => 2, // Ca
        // Transition metals (Sc-Zn)
        21 => 3,
        22 => 4,
        23 => 5,
        24 => 6,
        25 => 7,
        26 => 8,
        27 => 9,
        28 => 10,
        29 => 2,
        30 => 2,
        31 => 3, // Ga
        32 => 4, // Ge
        33 => 5, // As
        34 => 6, // Se
        35 => 7, // Br
        36 => 8, // Kr
        37 => 1, // Rb
        38 => 2, // Sr
        // Transition metals (Y-Cd)
        39 => 3,
        40 => 4,
        41 => 5,
        42 => 6,
        43 => 7,
        44 => 8,
        45 => 9,
        46 => 10,
        47 => 2,
        48 => 2,
        49 => 3, // In
        50 => 4, // Sn
        51 => 5, // Sb
        52 => 6, // Te
        53 => 7, // I
        54 => 8, // Xe
        // 3rd-row transition metals (Hf-Hg): 5d + 6s electrons
        55 => 1,  // Cs
        56 => 2,  // Ba
        72 => 4,  // Hf
        73 => 5,  // Ta
        74 => 6,  // W
        75 => 7,  // Re
        76 => 8,  // Os
        77 => 9,  // Ir
        78 => 10, // Pt (5d9 6s1 → 10 outer)
        79 => 11, // Au (5d10 6s1 → 11 outer)
        80 => 2,  // Hg
        81 => 3,  // Tl
        82 => 4,  // Pb
        83 => 5,  // Bi
        // For other elements, use simplified model
        _ => {
            if atomic_number % 10 < 8 {
                atomic_number % 10
            } else {
                8
            }
        }
    }
}

/// Parse element symbol (1 or 2 characters) to atomic number.
/// Returns None if not recognized.
pub fn symbol_to_atomic_number(symbol: &str) -> Option<u8> {
    match symbol.as_bytes() {
        b"H" => Some(1),
        b"He" => Some(2),
        b"Li" => Some(3),
        b"Be" => Some(4),
        b"B" => Some(5),
        b"C" => Some(6),
        b"N" => Some(7),
        b"O" => Some(8),
        b"F" => Some(9),
        b"Ne" => Some(10),
        b"Na" => Some(11),
        b"Mg" => Some(12),
        b"Al" => Some(13),
        b"Si" => Some(14),
        b"P" => Some(15),
        b"S" => Some(16),
        b"Cl" => Some(17),
        b"Ar" => Some(18),
        b"K" => Some(19),
        b"Ca" => Some(20),
        b"Sc" => Some(21),
        b"Ti" => Some(22),
        b"V" => Some(23),
        b"Cr" => Some(24),
        b"Mn" => Some(25),
        b"Fe" => Some(26),
        b"Co" => Some(27),
        b"Ni" => Some(28),
        b"Cu" => Some(29),
        b"Zn" => Some(30),
        b"Ga" => Some(31),
        b"Ge" => Some(32),
        b"As" => Some(33),
        b"Se" => Some(34),
        b"Br" => Some(35),
        b"Kr" => Some(36),
        b"Rb" => Some(37),
        b"Sr" => Some(38),
        b"Y" => Some(39),
        b"Zr" => Some(40),
        b"Nb" => Some(41),
        b"Mo" => Some(42),
        b"Tc" => Some(43),
        b"Ru" => Some(44),
        b"Rh" => Some(45),
        b"Pd" => Some(46),
        b"Ag" => Some(47),
        b"Cd" => Some(48),
        b"In" => Some(49),
        b"Sn" => Some(50),
        b"Sb" => Some(51),
        b"Te" => Some(52),
        b"I" => Some(53),
        b"Xe" => Some(54),
        b"Cs" => Some(55),
        b"Ba" => Some(56),
        b"La" => Some(57),
        b"Ce" => Some(58),
        b"Pr" => Some(59),
        b"Nd" => Some(60),
        b"Pm" => Some(61),
        b"Sm" => Some(62),
        b"Eu" => Some(63),
        b"Gd" => Some(64),
        b"Tb" => Some(65),
        b"Dy" => Some(66),
        b"Ho" => Some(67),
        b"Er" => Some(68),
        b"Tm" => Some(69),
        b"Yb" => Some(70),
        b"Lu" => Some(71),
        b"Hf" => Some(72),
        b"Ta" => Some(73),
        b"W" => Some(74),
        b"Re" => Some(75),
        b"Os" => Some(76),
        b"Ir" => Some(77),
        b"Pt" => Some(78),
        b"Au" => Some(79),
        b"Hg" => Some(80),
        b"Tl" => Some(81),
        b"Pb" => Some(82),
        b"Bi" => Some(83),
        b"Po" => Some(84),
        b"At" => Some(85),
        b"Rn" => Some(86),
        b"Fr" => Some(87),
        b"Ra" => Some(88),
        b"Ac" => Some(89),
        b"Th" => Some(90),
        b"Pa" => Some(91),
        b"U" => Some(92),
        b"Np" => Some(93),
        b"Pu" => Some(94),
        b"Am" => Some(95),
        b"Cm" => Some(96),
        b"Bk" => Some(97),
        b"Cf" => Some(98),
        b"Es" => Some(99),
        b"Fm" => Some(100),
        b"Md" => Some(101),
        b"No" => Some(102),
        b"Lr" => Some(103),
        b"Rf" => Some(104),
        b"Db" => Some(105),
        b"Sg" => Some(106),
        b"Bh" => Some(107),
        b"Hs" => Some(108),
        b"Mt" => Some(109),
        b"Ds" => Some(110),
        b"Rg" => Some(111),
        b"Cn" => Some(112),
        b"Nh" => Some(113),
        b"Fl" => Some(114),
        b"Mc" => Some(115),
        b"Lv" => Some(116),
        b"Ts" => Some(117),
        b"Og" => Some(118),
        // Lowercase organic subset (SMILES convention)
        b"c" => Some(6),   // aromatic carbon
        b"n" => Some(7),   // aromatic nitrogen
        b"o" => Some(8),   // aromatic oxygen
        b"s" => Some(16),  // aromatic sulfur
        b"p" => Some(15),  // aromatic phosphorus
        b"b" => Some(5),   // aromatic boron
        b"as" => Some(33), // aromatic arsenic (bracket form)
        b"se" => Some(34), // aromatic selenium (bracket form)
        b"te" => Some(52), // aromatic tellurium (bracket form)
        _ => None,
    }
}

/// Check if character is an aromatic SMILES atom symbol.
#[inline]
pub fn is_aromatic_symbol(c: char) -> bool {
    matches!(c, 'a' | 'b' | 'c' | 'n' | 'o' | 'p' | 's' | 't')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Runs regression test for atomic masses.
    ///
    /// Function context: `crates/ferrum_core/src/mol/elements.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    fn test_atomic_masses() {
        assert_eq!(ATOMIC_MASSES[1], 1.008); // H
        assert_eq!(ATOMIC_MASSES[6], 12.011); // C
        assert_eq!(ATOMIC_MASSES[8], 15.999); // O
    }

    #[test]
    /// Runs regression test for symbol lookup.
    ///
    /// Function context: `crates/ferrum_core/src/mol/elements.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    fn test_symbol_lookup() {
        assert_eq!(symbol_to_atomic_number("C"), Some(6));
        assert_eq!(symbol_to_atomic_number("N"), Some(7));
        assert_eq!(symbol_to_atomic_number("Cl"), Some(17));
        assert_eq!(symbol_to_atomic_number("c"), Some(6)); // aromatic
        assert_eq!(symbol_to_atomic_number("te"), Some(52)); // aromatic bracket tellurium
        assert_eq!(symbol_to_atomic_number("Xx"), None);
    }

    #[test]
    /// Runs regression test for valences.
    ///
    /// Function context: `crates/ferrum_core/src/mol/elements.rs`.
    /// Molecule-model note: preserve atom/bond index invariants and cached-property invalidation rules.
    fn test_valences() {
        assert_eq!(get_valences(6), &[4]); // C
        assert_eq!(get_valences(7), &[3, 5]); // N
        assert_eq!(get_valences(16), &[2, 4, 6]); // S
    }
}
