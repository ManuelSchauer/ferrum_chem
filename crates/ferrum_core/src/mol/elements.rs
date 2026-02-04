/// Element data lookup tables for high-performance atomic operations.
/// Uses const arrays indexed by atomic number for zero-cost abstractions.

/// Standard atomic masses (in Daltons) indexed by atomic number.
/// Index 0 is unused, elements 1-118 are defined.
pub const ATOMIC_MASSES: [f32; 119] = [
    0.0,     // 0: placeholder
    1.008,   // 1: H
    4.003,   // 2: He
    6.941,   // 3: Li
    9.012,   // 4: Be
    10.81,   // 5: B
    12.011,  // 6: C
    14.007,  // 7: N
    15.999,  // 8: O
    18.998,  // 9: F
    20.180,  // 10: Ne
    22.990,  // 11: Na
    24.305,  // 12: Mg
    26.982,  // 13: Al
    28.085,  // 14: Si
    30.974,  // 15: P
    32.06,   // 16: S
    35.45,   // 17: Cl
    39.948,  // 18: Ar
    39.098,  // 19: K
    40.078,  // 20: Ca
    44.956,  // 21: Sc
    47.867,  // 22: Ti
    50.942,  // 23: V
    51.996,  // 24: Cr
    54.938,  // 25: Mn
    55.845,  // 26: Fe
    58.933,  // 27: Co
    58.693,  // 28: Ni
    63.546,  // 29: Cu
    65.38,   // 30: Zn
    69.723,  // 31: Ga
    72.630,  // 32: Ge
    74.922,  // 33: As
    78.971,  // 34: Se
    79.904,  // 35: Br
    83.798,  // 36: Kr
    85.468,  // 37: Rb
    87.62,   // 38: Sr
    88.906,  // 39: Y
    91.224,  // 40: Zr
    92.906,  // 41: Nb
    95.95,   // 42: Mo
    98.0,    // 43: Tc
    101.07,  // 44: Ru
    102.91,  // 45: Rh
    106.42,  // 46: Pd
    107.87,  // 47: Ag
    112.41,  // 48: Cd
    114.82,  // 49: In
    118.71,  // 50: Sn
    121.76,  // 51: Sb
    127.60,  // 52: Te
    126.90,  // 53: I
    131.29,  // 54: Xe
    132.91,  // 55: Cs
    137.33,  // 56: Ba
    138.91,  // 57: La
    140.12,  // 58: Ce
    140.91,  // 59: Pr
    144.24,  // 60: Nd
    145.0,   // 61: Pm
    150.36,  // 62: Sm
    151.96,  // 63: Eu
    157.25,  // 64: Gd
    158.93,  // 65: Tb
    162.50,  // 66: Dy
    164.93,  // 67: Ho
    167.26,  // 68: Er
    168.93,  // 69: Tm
    173.05,  // 70: Yb
    174.97,  // 71: Lu
    178.49,  // 72: Hf
    180.95,  // 73: Ta
    183.84,  // 74: W
    186.21,  // 75: Re
    190.23,  // 76: Os
    192.22,  // 77: Ir
    195.08,  // 78: Pt
    196.97,  // 79: Au
    200.59,  // 80: Hg
    204.38,  // 81: Tl
    207.2,   // 82: Pb
    208.98,  // 83: Bi
    209.0,   // 84: Po
    210.0,   // 85: At
    222.0,   // 86: Rn
    223.0,   // 87: Fr
    226.0,   // 88: Ra
    227.0,   // 89: Ac
    232.04,  // 90: Th
    231.04,  // 91: Pa
    238.03,  // 92: U
    237.0,   // 93: Np
    244.0,   // 94: Pu
    243.0,   // 95: Am
    247.0,   // 96: Cm
    247.0,   // 97: Bk
    251.0,   // 98: Cf
    252.0,   // 99: Es
    257.0,   // 100: Fm
    258.0,   // 101: Md
    259.0,   // 102: No
    262.0,   // 103: Lr
    267.0,   // 104: Rf
    268.0,   // 105: Db
    271.0,   // 106: Sg
    272.0,   // 107: Bh
    270.0,   // 108: Hs
    276.0,   // 109: Mt
    281.0,   // 110: Ds
    280.0,   // 111: Rg
    285.0,   // 112: Cn
    284.0,   // 113: Nh
    289.0,   // 114: Fl
    288.0,   // 115: Mc
    293.0,   // 116: Lv
    294.0,   // 117: Ts
    294.0,   // 118: Og
];

/// Element symbols indexed by atomic number.
pub const ELEMENT_SYMBOLS: [&str; 119] = [
    "*",  // 0: wildcard/unknown
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",  // 1-10
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",  // 11-20
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",  // 21-30
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",  // 31-40
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",  // 41-50
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",  // 51-60
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",  // 61-70
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",  // 71-80
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",  // 81-90
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",  // 91-100
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",  // 101-110
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",  // 111-118
];

/// Typical valences for organic elements (for implicit hydrogen calculation).
/// Returns common valence states as a slice.
#[inline]
pub const fn get_valences(atomic_number: u8) -> &'static [u8] {
    match atomic_number {
        1 => &[1],           // H
        5 => &[3],           // B
        6 => &[4],           // C
        7 => &[3, 5],        // N
        8 => &[2],           // O
        9 => &[1],           // F
        14 => &[4],          // Si
        15 => &[3, 5],       // P
        16 => &[2, 4, 6],    // S
        17 => &[1],          // Cl
        35 => &[1],          // Br
        53 => &[1],          // I
        _ => &[],            // Unknown/metal
    }
}

/// Parse element symbol (1 or 2 characters) to atomic number.
/// Returns None if not recognized.
pub fn symbol_to_atomic_number(symbol: &str) -> Option<u8> {
    match symbol {
        "H" => Some(1),
        "He" => Some(2),
        "Li" => Some(3),
        "Be" => Some(4),
        "B" => Some(5),
        "C" => Some(6),
        "N" => Some(7),
        "O" => Some(8),
        "F" => Some(9),
        "Ne" => Some(10),
        "Na" => Some(11),
        "Mg" => Some(12),
        "Al" => Some(13),
        "Si" => Some(14),
        "P" => Some(15),
        "S" => Some(16),
        "Cl" => Some(17),
        "Ar" => Some(18),
        "K" => Some(19),
        "Ca" => Some(20),
        "Sc" => Some(21),
        "Ti" => Some(22),
        "V" => Some(23),
        "Cr" => Some(24),
        "Mn" => Some(25),
        "Fe" => Some(26),
        "Co" => Some(27),
        "Ni" => Some(28),
        "Cu" => Some(29),
        "Zn" => Some(30),
        "Ga" => Some(31),
        "Ge" => Some(32),
        "As" => Some(33),
        "Se" => Some(34),
        "Br" => Some(35),
        "Kr" => Some(36),
        "Rb" => Some(37),
        "Sr" => Some(38),
        "Y" => Some(39),
        "Zr" => Some(40),
        "Nb" => Some(41),
        "Mo" => Some(42),
        "Tc" => Some(43),
        "Ru" => Some(44),
        "Rh" => Some(45),
        "Pd" => Some(46),
        "Ag" => Some(47),
        "Cd" => Some(48),
        "In" => Some(49),
        "Sn" => Some(50),
        "Sb" => Some(51),
        "Te" => Some(52),
        "I" => Some(53),
        "Xe" => Some(54),
        "Cs" => Some(55),
        "Ba" => Some(56),
        "La" => Some(57),
        "Ce" => Some(58),
        "Pr" => Some(59),
        "Nd" => Some(60),
        "Pm" => Some(61),
        "Sm" => Some(62),
        "Eu" => Some(63),
        "Gd" => Some(64),
        "Tb" => Some(65),
        "Dy" => Some(66),
        "Ho" => Some(67),
        "Er" => Some(68),
        "Tm" => Some(69),
        "Yb" => Some(70),
        "Lu" => Some(71),
        "Hf" => Some(72),
        "Ta" => Some(73),
        "W" => Some(74),
        "Re" => Some(75),
        "Os" => Some(76),
        "Ir" => Some(77),
        "Pt" => Some(78),
        "Au" => Some(79),
        "Hg" => Some(80),
        "Tl" => Some(81),
        "Pb" => Some(82),
        "Bi" => Some(83),
        "Po" => Some(84),
        "At" => Some(85),
        "Rn" => Some(86),
        "Fr" => Some(87),
        "Ra" => Some(88),
        "Ac" => Some(89),
        "Th" => Some(90),
        "Pa" => Some(91),
        "U" => Some(92),
        "Np" => Some(93),
        "Pu" => Some(94),
        "Am" => Some(95),
        "Cm" => Some(96),
        "Bk" => Some(97),
        "Cf" => Some(98),
        "Es" => Some(99),
        "Fm" => Some(100),
        "Md" => Some(101),
        "No" => Some(102),
        "Lr" => Some(103),
        "Rf" => Some(104),
        "Db" => Some(105),
        "Sg" => Some(106),
        "Bh" => Some(107),
        "Hs" => Some(108),
        "Mt" => Some(109),
        "Ds" => Some(110),
        "Rg" => Some(111),
        "Cn" => Some(112),
        "Nh" => Some(113),
        "Fl" => Some(114),
        "Mc" => Some(115),
        "Lv" => Some(116),
        "Ts" => Some(117),
        "Og" => Some(118),
        // Lowercase organic subset (SMILES convention)
        "c" => Some(6),   // aromatic carbon
        "n" => Some(7),   // aromatic nitrogen
        "o" => Some(8),   // aromatic oxygen
        "s" => Some(16),  // aromatic sulfur
        "p" => Some(15),  // aromatic phosphorus
        "b" => Some(5),   // aromatic boron
        _ => None,
    }
}

/// Check if character is an aromatic SMILES atom symbol.
#[inline]
pub fn is_aromatic_symbol(c: char) -> bool {
    matches!(c, 'b' | 'c' | 'n' | 'o' | 'p' | 's')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_masses() {
        assert_eq!(ATOMIC_MASSES[1], 1.008);  // H
        assert_eq!(ATOMIC_MASSES[6], 12.011); // C
        assert_eq!(ATOMIC_MASSES[8], 15.999); // O
    }

    #[test]
    fn test_symbol_lookup() {
        assert_eq!(symbol_to_atomic_number("C"), Some(6));
        assert_eq!(symbol_to_atomic_number("N"), Some(7));
        assert_eq!(symbol_to_atomic_number("Cl"), Some(17));
        assert_eq!(symbol_to_atomic_number("c"), Some(6));  // aromatic
        assert_eq!(symbol_to_atomic_number("Xx"), None);
    }

    #[test]
    fn test_valences() {
        assert_eq!(get_valences(6), &[4]);      // C
        assert_eq!(get_valences(7), &[3, 5]);   // N
        assert_eq!(get_valences(16), &[2, 4, 6]); // S
    }
}

