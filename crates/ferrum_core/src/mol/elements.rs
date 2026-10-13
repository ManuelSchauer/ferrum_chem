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
pub const ATOMIC_MASSES: [f64; 119] = build_atomic_masses();

/// Average atomic weights from RDKit's periodic table (see `rdkit_data`).
const fn build_atomic_masses() -> [f64; 119] {
    let mut out = [0.0f64; 119];
    let mut i = 0;
    while i < 119 {
        out[i] = crate::mol::rdkit_data::ELEMENTS[i].mass;
        i += 1;
    }
    out
}

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
const VALENCE_EMPTY: &[u8] = &[];
const VALENCE_1: &[u8] = &[1];
const VALENCE_2: &[u8] = &[2];
const VALENCE_3: &[u8] = &[3];
const VALENCE_4: &[u8] = &[4];
const VALENCE_3_5: &[u8] = &[3, 5];
const VALENCE_2_4_6: &[u8] = &[2, 4, 6];

const fn build_valence_table() -> [&'static [u8]; 119] {
    let mut table = [VALENCE_EMPTY; 119];
    table[1] = VALENCE_1; // H
    table[5] = VALENCE_3; // B
    table[6] = VALENCE_4; // C
    table[7] = VALENCE_3_5; // N
    table[8] = VALENCE_2; // O
    table[9] = VALENCE_1; // F
    table[14] = VALENCE_4; // Si
    table[15] = VALENCE_3_5; // P
    table[16] = VALENCE_2_4_6; // S
    table[17] = VALENCE_1; // Cl
    table[35] = VALENCE_1; // Br
    table[53] = VALENCE_1; // I
    table
}

const VALENCE_TABLE: [&[u8]; 119] = build_valence_table();

#[inline]
pub const fn get_valences(atomic_number: u8) -> &'static [u8] {
    if (atomic_number as usize) < VALENCE_TABLE.len() {
        VALENCE_TABLE[atomic_number as usize]
    } else {
        VALENCE_EMPTY
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
const fn build_transition_metal_mask() -> [bool; 119] {
    let mut mask = [false; 119];
    let mut z = 21usize;
    while z <= 30 {
        mask[z] = true;
        z += 1;
    }
    z = 39usize;
    while z <= 48 {
        mask[z] = true;
        z += 1;
    }
    z = 57usize;
    while z <= 80 {
        mask[z] = true;
        z += 1;
    }
    z = 89usize;
    while z <= 99 {
        mask[z] = true;
        z += 1;
    }
    mask
}

const TRANSITION_METAL_MASK: [bool; 119] = build_transition_metal_mask();

#[inline]
pub const fn is_transition_metal(atomic_number: u8) -> bool {
    if (atomic_number as usize) < TRANSITION_METAL_MASK.len() {
        TRANSITION_METAL_MASK[atomic_number as usize]
    } else {
        false
    }
}

#[inline]
pub const fn outer_electrons(atomic_number: u8) -> u8 {
    // RDKit periodic table (e.g. Cu = 11, Fe = 8), see `rdkit_data`.
    if (atomic_number as usize) < crate::mol::rdkit_data::ELEMENTS.len() {
        crate::mol::rdkit_data::ELEMENTS[atomic_number as usize].n_outer
    } else {
        0
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
