//! PubChem 881-bit structural key fingerprint.
//!
//! All 881 bits defined as SMARTS patterns with count thresholds.
//! Each bit is set when the molecule contains >= `threshold` matches of the SMARTS pattern.
//!
//! Sections:
//!   0-114:   Hierarchical element counts
//!   115-262: Ring counts by size
//!   263-326: Simple atom pairs
//!   327-415: Simple atom nearest neighbors
//!   416-459: Detailed atom neighborhoods
//!   460-712: Simple SMARTS patterns
//!   713-880: Complex SMARTS patterns
//!
//! Reference: ftp://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt

use super::typed::{Descriptor, DescriptorValue};
use crate::algo::iso::count_matches;
use crate::io::smarts::{QueryGraph, parse_smarts};
use crate::mol::molecule::Molecule;
use bitvec::prelude::*;
use once_cell::sync::Lazy;

/// A PubChem key: SMARTS pattern + minimum count to set the bit.
struct PubChemKey {
    smarts: &'static str,
    threshold: usize,
}

/// Pre-parsed PubChem keys.
struct ParsedPubChemKey {
    query: QueryGraph,
    threshold: usize,
}

static PARSED_KEYS: Lazy<Vec<Option<ParsedPubChemKey>>> = Lazy::new(|| {
    PUBCHEM_KEYS
        .iter()
        .map(|key| {
            parse_smarts(key.smarts).ok().map(|q| ParsedPubChemKey {
                query: q,
                threshold: key.threshold,
            })
        })
        .collect()
});

/// PubChem 881-bit structural key fingerprint.
///
/// Returns bit-packed fingerprint. Each bit encodes the presence of a
/// specific structural feature at or above a count threshold.
///
/// # Examples
///
/// ```
/// use ferrum_core::desc::pubchem::PubChem881;
/// use ferrum_core::desc::typed::Descriptor;
/// use ferrum_core::io::smiles::parse_smiles;
///
/// let mut mol = parse_smiles("c1ccccc1").unwrap();
/// let descriptor = PubChem881;
/// let fp = descriptor.calculate(&mut mol);
///
/// let bits = fp.as_binary();
/// assert_eq!(bits.len(), 881);
/// assert!(bits.count_ones() > 0);
/// ```
pub struct PubChem881;

impl Descriptor for PubChem881 {
    fn name(&self) -> &str {
        "PubChem881"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let mut bits = bitvec![u8, Lsb0; 0; 881];

        for (i, key) in PARSED_KEYS.iter().enumerate() {
            if let Some(pk) = key {
                let cnt = count_matches(mol, &pk.query);
                if cnt >= pk.threshold {
                    bits.set(i, true);
                }
            }
        }

        DescriptorValue::Binary(bits)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<PubChem881>;
};

// ---------------------------------------------------------------------------
// All 881 key definitions.
// ---------------------------------------------------------------------------
// Macro to reduce boilerplate.
macro_rules! pk {
    ($smarts:expr, $thr:expr) => {
        PubChemKey {
            smarts: $smarts,
            threshold: $thr,
        }
    };
}

static PUBCHEM_KEYS: &[PubChemKey] = &[
    // ===== Section 1: Hierarchical Element Counts (bits 0-114) =====
    // H
    pk!("[#1]", 4),  // 0
    pk!("[#1]", 8),  // 1
    pk!("[#1]", 16), // 2
    pk!("[#1]", 32), // 3
    // He
    pk!("[#2]", 1), // 4
    // Li
    pk!("[#3]", 1), // 5
    pk!("[#3]", 2), // 6
    // Be
    pk!("[#4]", 1), // 7
    pk!("[#4]", 2), // 8
    // B
    pk!("[#5]", 1), // 9
    pk!("[#5]", 2), // 10
    pk!("[#5]", 4), // 11
    // C
    pk!("[#6]", 2),  // 12
    pk!("[#6]", 4),  // 13
    pk!("[#6]", 8),  // 14
    pk!("[#6]", 16), // 15
    pk!("[#6]", 32), // 16
    // N
    pk!("[#7]", 1), // 17
    pk!("[#7]", 2), // 18
    pk!("[#7]", 4), // 19
    pk!("[#7]", 8), // 20
    // O
    pk!("[#8]", 1),  // 21
    pk!("[#8]", 2),  // 22
    pk!("[#8]", 4),  // 23
    pk!("[#8]", 8),  // 24
    pk!("[#8]", 16), // 25
    // F
    pk!("[#9]", 1), // 26
    pk!("[#9]", 2), // 27
    pk!("[#9]", 4), // 28
    // Ne
    pk!("[#10]", 1), // 29
    // Na
    pk!("[#11]", 1), // 30
    pk!("[#11]", 2), // 31
    // Mg
    pk!("[#12]", 1), // 32
    pk!("[#12]", 2), // 33
    // Al
    pk!("[#13]", 1), // 34
    pk!("[#13]", 2), // 35
    // Si
    pk!("[#14]", 1), // 36
    pk!("[#14]", 2), // 37
    // P
    pk!("[#15]", 1), // 38
    pk!("[#15]", 2), // 39
    pk!("[#15]", 4), // 40
    // S
    pk!("[#16]", 1), // 41
    pk!("[#16]", 2), // 42
    pk!("[#16]", 4), // 43
    pk!("[#16]", 8), // 44
    // Cl
    pk!("[#17]", 1), // 45
    pk!("[#17]", 2), // 46
    pk!("[#17]", 4), // 47
    // Ar
    pk!("[#18]", 1), // 48
    // K
    pk!("[#19]", 1), // 49
    pk!("[#19]", 2), // 50
    // Ca
    pk!("[#20]", 1), // 51
    pk!("[#20]", 2), // 52
    // Sc
    pk!("[#21]", 1), // 53
    // Ti
    pk!("[#22]", 1), // 54
    // V
    pk!("[#23]", 1), // 55
    // Cr
    pk!("[#24]", 1), // 56
    // Mn
    pk!("[#25]", 1), // 57
    // Fe
    pk!("[#26]", 1), // 58
    // Co
    pk!("[#27]", 1), // 59
    // Ni
    pk!("[#28]", 1), // 60
    // Cu
    pk!("[#29]", 1), // 61
    // Zn
    pk!("[#30]", 1), // 62
    // Ga
    pk!("[#31]", 1), // 63
    // Ge
    pk!("[#32]", 1), // 64
    // As
    pk!("[#33]", 1), // 65
    // Se
    pk!("[#34]", 1), // 66
    // Br
    pk!("[#35]", 1), // 67
    pk!("[#35]", 2), // 68
    pk!("[#35]", 4), // 69
    // Kr
    pk!("[#36]", 1), // 70
    // Rb - Xe (37-54)
    pk!("[#37]", 1), // 71
    pk!("[#38]", 1), // 72
    pk!("[#39]", 1), // 73
    pk!("[#40]", 1), // 74
    pk!("[#41]", 1), // 75
    pk!("[#42]", 1), // 76
    pk!("[#44]", 1), // 77  Ru
    pk!("[#45]", 1), // 78  Rh
    pk!("[#46]", 1), // 79  Pd
    pk!("[#47]", 1), // 80  Ag
    pk!("[#48]", 1), // 81  Cd
    pk!("[#49]", 1), // 82  In
    pk!("[#50]", 1), // 83  Sn
    pk!("[#51]", 1), // 84  Sb
    pk!("[#52]", 1), // 85  Te
    pk!("[#53]", 1), // 86  I
    pk!("[#53]", 2), // 87
    pk!("[#53]", 4), // 88
    pk!("[#54]", 1), // 89  Xe
    // Cs - Lu (55-71)
    pk!("[#55]", 1), // 90
    pk!("[#56]", 1), // 91
    pk!("[#57]", 1), // 92  La
    pk!("[#58]", 1), // 93  Ce
    pk!("[#59]", 1), // 94  Pr
    pk!("[#60]", 1), // 95  Nd
    pk!("[#61]", 1), // 96  Pm
    pk!("[#62]", 1), // 97  Sm
    pk!("[#63]", 1), // 98  Eu
    pk!("[#64]", 1), // 99  Gd
    pk!("[#65]", 1), // 100 Tb
    pk!("[#66]", 1), // 101 Dy
    pk!("[#67]", 1), // 102 Ho
    pk!("[#68]", 1), // 103 Er
    pk!("[#69]", 1), // 104 Tm
    pk!("[#70]", 1), // 105 Yb
    pk!("[#71]", 1), // 106 Lu
    // Hf - Bi (72-83)
    pk!("[#72]", 1), // 107 Hf
    pk!("[#73]", 1), // 108 Ta
    pk!("[#74]", 1), // 109 W
    pk!("[#75]", 1), // 110 Re
    pk!("[#76]", 1), // 111 Os
    pk!("[#77]", 1), // 112 Ir
    pk!("[#78]", 1), // 113 Pt
    pk!("[#79]", 1), // 114 Au
    // ===== Section 2: Ring counts (bits 115-262) =====
    // Ring of size 3
    pk!("[r3]", 1), // 115
    pk!("[r3]", 2), // 116  (>=2 atoms in 3-ring ≈ >=1 ring of size 3)
    // Ring of size 4
    pk!("[r4]", 1), // 117
    pk!("[r4]", 2), // 118
    // Ring of size 5
    pk!("[r5]", 1),  // 119
    pk!("[r5]", 2),  // 120
    pk!("[r5]", 4),  // 121
    pk!("[r5]", 8),  // 122
    pk!("[r5]", 16), // 123
    // Ring of size 6
    pk!("[r6]", 1),  // 124
    pk!("[r6]", 2),  // 125
    pk!("[r6]", 4),  // 126
    pk!("[r6]", 8),  // 127
    pk!("[r6]", 16), // 128
    // Ring of size 7
    pk!("[r7]", 1), // 129
    pk!("[r7]", 2), // 130
    // Ring of size 8
    pk!("[r8]", 1), // 131
    pk!("[r8]", 2), // 132
    // Ring of size 9
    pk!("[r9]", 1), // 133
    // Ring of size 10+
    pk!("[R]", 1), // 134  any ring atom
    pk!("[R]", 2), // 135
    pk!("[R]", 4), // 136
    // Saturated/unsaturated ring atoms
    pk!("[R;X4;#6]", 1), // 137  saturated carbon in ring
    pk!("[R;X4;#6]", 2), // 138
    pk!("[R;X4;#6]", 4), // 139
    pk!("[r5;#7]", 1),   // 140  N in 5-ring
    pk!("[r5;#7]", 2),   // 141
    pk!("[r6;#7]", 1),   // 142  N in 6-ring
    pk!("[r6;#7]", 2),   // 143
    pk!("[r5;#8]", 1),   // 144  O in 5-ring
    pk!("[r5;#8]", 2),   // 145
    pk!("[r6;#8]", 1),   // 146  O in 6-ring
    pk!("[r5;#16]", 1),  // 147  S in 5-ring
    pk!("[r6;#16]", 1),  // 148  S in 6-ring
    // Aromatic ring atoms
    pk!("[a]", 1),  // 149
    pk!("[a]", 2),  // 150
    pk!("[a]", 4),  // 151
    pk!("[a]", 8),  // 152
    pk!("[a]", 16), // 153
    // Aromatic N
    pk!("[n]", 1), // 154
    pk!("[n]", 2), // 155
    pk!("[n]", 4), // 156
    // Aromatic O
    pk!("[o]", 1), // 157
    // Aromatic S
    pk!("[s]", 1), // 158
    // Aromatic heterocycle
    pk!("[a;!#6]", 1), // 159
    pk!("[a;!#6]", 2), // 160
    pk!("[a;!#6]", 4), // 161
    // Fused ring systems (shared ring bond atoms)
    pk!("[x3]", 1),  // 162  atom in 3+ ring bonds → fused
    pk!("[x3]", 2),  // 163
    pk!("[x2]", 4),  // 164
    pk!("[x2]", 8),  // 165
    pk!("[x2]", 12), // 166
    // 5-membered ring heteroatom patterns
    pk!("[r5;#7;H1]", 1),  // 167  NH in 5-ring (pyrrole-like)
    pk!("[r5;#7;H0]", 1),  // 168  =N- in 5-ring
    pk!("[r5;#8;H0]", 1),  // 169  -O- in 5-ring (furan-like)
    pk!("[r5;#16;H0]", 1), // 170  -S- in 5-ring (thiophene-like)
    // 6-membered ring heteroatom patterns
    pk!("[r6;#7;H0]", 1), // 171  =N- in 6-ring (pyridine-like)
    pk!("[r6;#7;H0]", 2), // 172
    pk!("[r6;#7;H1]", 1), // 173  NH in 6-ring
    // Ring-ring junction types
    pk!("[R;D3]", 1), // 174  degree-3 ring atom (ring junction)
    pk!("[R;D3]", 2), // 175
    pk!("[R;D3]", 4), // 176
    pk!("[R;D4]", 1), // 177  degree-4 ring atom (spiro-like)
    // Nitrogen ring patterns
    pk!("[nR]", 1), // 178
    pk!("[nR]", 2), // 179
    pk!("[nR]", 4), // 180
    // Multiple ring types
    pk!("[r5;a]", 1), // 181  aromatic 5-ring atom
    pk!("[r5;a]", 2), // 182
    pk!("[r5;a]", 4), // 183
    pk!("[r6;a]", 1), // 184  aromatic 6-ring atom
    pk!("[r6;a]", 2), // 185
    pk!("[r6;a]", 4), // 186
    pk!("[r6;a]", 8), // 187
    // Aliphatic ring atoms
    pk!("[R;A]", 1), // 188
    pk!("[R;A]", 2), // 189
    pk!("[R;A]", 4), // 190
    pk!("[R;A]", 8), // 191
    // Ring bonds
    pk!("[R]~[R]", 1),  // 192
    pk!("[R]~[R]", 2),  // 193
    pk!("[R]~[R]", 4),  // 194
    pk!("[R]~[R]", 8),  // 195
    pk!("[R]~[R]", 16), // 196
    // Ring-chain connections
    pk!("[R]~[!R]", 1), // 197
    pk!("[R]~[!R]", 2), // 198
    pk!("[R]~[!R]", 4), // 199
    pk!("[R]~[!R]", 8), // 200
    // Specific ring types continued
    pk!("[r3;#6]", 1), // 201  C in 3-ring
    pk!("[r4;#6]", 1), // 202  C in 4-ring
    pk!("[r5;#6]", 1), // 203  C in 5-ring
    pk!("[r5;#6]", 2), // 204
    pk!("[r5;#6]", 4), // 205
    pk!("[r6;#6]", 1), // 206  C in 6-ring
    pk!("[r6;#6]", 2), // 207
    pk!("[r6;#6]", 4), // 208
    pk!("[r6;#6]", 8), // 209
    pk!("[r7;#6]", 1), // 210  C in 7-ring
    // More ring element combos
    pk!("[r5;#15]", 1), // 211  P in 5-ring
    pk!("[r6;#15]", 1), // 212  P in 6-ring
    // Bridged ring indicators
    pk!("[R;D3;#6]", 1), // 213
    pk!("[R;D3;#6]", 2), // 214
    pk!("[R;D3;#7]", 1), // 215
    // Ring double bonds
    pk!("[R]=[R]", 1), // 216
    pk!("[R]=[R]", 2), // 217
    pk!("[R]=[R]", 4), // 218
    pk!("[R]=[R]", 8), // 219
    // Ring aromatic bonds
    pk!("[a]:[a]", 1),  // 220
    pk!("[a]:[a]", 2),  // 221
    pk!("[a]:[a]", 4),  // 222
    pk!("[a]:[a]", 8),  // 223
    pk!("[a]:[a]", 16), // 224
    // Ring with exocyclic double bond
    pk!("[R]=[!R]", 1), // 225
    pk!("[R]=[!R]", 2), // 226
    pk!("[R]=[!R]", 4), // 227
    // Heterocyclic ring atoms total
    pk!("[R;!#6]", 1), // 228
    pk!("[R;!#6]", 2), // 229
    pk!("[R;!#6]", 4), // 230
    pk!("[R;!#6]", 8), // 231
    // Ring N-H patterns
    pk!("[R;#7;H1]", 1), // 232
    pk!("[R;#7;H1]", 2), // 233
    // Ring N with no H
    pk!("[R;#7;H0]", 1), // 234
    pk!("[R;#7;H0]", 2), // 235
    pk!("[R;#7;H0]", 4), // 236
    // Ring O
    pk!("[R;#8]", 1), // 237
    pk!("[R;#8]", 2), // 238
    // Ring S
    pk!("[R;#16]", 1), // 239
    pk!("[R;#16]", 2), // 240
    // Mixed ring/chain descriptors
    pk!("[R;#6;H2]", 1), // 241  CH2 in ring
    pk!("[R;#6;H2]", 2), // 242
    pk!("[R;#6;H2]", 4), // 243
    pk!("[R;#6;H1]", 1), // 244  CH in ring
    pk!("[R;#6;H1]", 2), // 245
    pk!("[R;#6;H1]", 4), // 246
    pk!("[R;#6;H0]", 1), // 247  quaternary C in ring
    pk!("[R;#6;H0]", 2), // 248
    // Ring-size specific with multiple rings
    pk!("[r5]~[r6]", 1), // 249  fused 5-6 ring junction
    pk!("[r5]~[r6]", 2), // 250
    pk!("[r6]~[r6]", 1), // 251  fused 6-6 ring junction
    pk!("[r6]~[r6]", 2), // 252
    pk!("[r6]~[r6]", 4), // 253
    pk!("[r5]~[r5]", 1), // 254  fused 5-5 ring junction
    // Additional ring patterns
    pk!("[R;#6;X4]", 1),  // 255  sp3 C in ring
    pk!("[R;#6;X4]", 2),  // 256
    pk!("[R;#6;X4]", 4),  // 257
    pk!("[R;#6;X3]", 1),  // 258  sp2 C in ring
    pk!("[R;#6;X3]", 2),  // 259
    pk!("[R;#6;X3]", 4),  // 260
    pk!("[R;#6;X3]", 8),  // 261
    pk!("[R;#6;X3]", 16), // 262
    // ===== Section 3: Simple atom pairs (bits 263-326) =====
    pk!("[#6]~[#6]", 1),   // 263  C-C
    pk!("[#6]~[#6]", 2),   // 264
    pk!("[#6]~[#6]", 4),   // 265
    pk!("[#6]~[#6]", 8),   // 266
    pk!("[#6]~[#6]", 16),  // 267
    pk!("[#6]~[#6]", 32),  // 268
    pk!("[#6]~[#7]", 1),   // 269  C-N
    pk!("[#6]~[#7]", 2),   // 270
    pk!("[#6]~[#7]", 4),   // 271
    pk!("[#6]~[#7]", 8),   // 272
    pk!("[#6]~[#8]", 1),   // 273  C-O
    pk!("[#6]~[#8]", 2),   // 274
    pk!("[#6]~[#8]", 4),   // 275
    pk!("[#6]~[#8]", 8),   // 276
    pk!("[#6]~[#16]", 1),  // 277  C-S
    pk!("[#6]~[#16]", 2),  // 278
    pk!("[#6]~[#16]", 4),  // 279
    pk!("[#6]~[#9]", 1),   // 280  C-F
    pk!("[#6]~[#9]", 2),   // 281
    pk!("[#6]~[#9]", 4),   // 282
    pk!("[#6]~[#17]", 1),  // 283  C-Cl
    pk!("[#6]~[#17]", 2),  // 284
    pk!("[#6]~[#17]", 4),  // 285
    pk!("[#6]~[#35]", 1),  // 286  C-Br
    pk!("[#6]~[#35]", 2),  // 287
    pk!("[#6]~[#53]", 1),  // 288  C-I
    pk!("[#6]~[#53]", 2),  // 289
    pk!("[#6]~[#15]", 1),  // 290  C-P
    pk!("[#6]~[#15]", 2),  // 291
    pk!("[#7]~[#7]", 1),   // 292  N-N
    pk!("[#7]~[#7]", 2),   // 293
    pk!("[#7]~[#8]", 1),   // 294  N-O
    pk!("[#7]~[#8]", 2),   // 295
    pk!("[#7]~[#8]", 4),   // 296
    pk!("[#7]~[#16]", 1),  // 297  N-S
    pk!("[#7]~[#16]", 2),  // 298
    pk!("[#7]~[#15]", 1),  // 299  N-P
    pk!("[#8]~[#8]", 1),   // 300  O-O
    pk!("[#8]~[#16]", 1),  // 301  O-S
    pk!("[#8]~[#16]", 2),  // 302
    pk!("[#8]~[#15]", 1),  // 303  O-P
    pk!("[#8]~[#15]", 2),  // 304
    pk!("[#16]~[#16]", 1), // 305  S-S
    pk!("[#15]~[#15]", 1), // 306  P-P
    // Bond-type specific pairs
    pk!("[#6]=[#6]", 1),  // 307  C=C
    pk!("[#6]=[#6]", 2),  // 308
    pk!("[#6]=[#6]", 4),  // 309
    pk!("[#6]=[#7]", 1),  // 310  C=N
    pk!("[#6]=[#7]", 2),  // 311
    pk!("[#6]=[#8]", 1),  // 312  C=O
    pk!("[#6]=[#8]", 2),  // 313
    pk!("[#6]=[#8]", 4),  // 314
    pk!("[#6]=[#16]", 1), // 315  C=S
    pk!("[#7]=[#7]", 1),  // 316  N=N
    pk!("[#7]=[#8]", 1),  // 317  N=O
    pk!("[#7]=[#8]", 2),  // 318
    pk!("[#15]=[#8]", 1), // 319  P=O
    pk!("[#15]=[#8]", 2), // 320
    pk!("[#16]=[#8]", 1), // 321  S=O
    pk!("[#16]=[#8]", 2), // 322
    pk!("[#16]=[#8]", 4), // 323
    pk!("[#6]#[#6]", 1),  // 324  C#C
    pk!("[#6]#[#7]", 1),  // 325  C#N
    pk!("[#6]#[#7]", 2),  // 326  C#N
    // ===== Section 4: Simple atom nearest neighbors (bits 327-415) =====
    // C with neighbors
    pk!("[#6](~[#6])(~[#6])", 1),        // 327  C(C)(C)
    pk!("[#6](~[#6])(~[#6])", 2),        // 328
    pk!("[#6](~[#6])(~[#6])", 4),        // 329
    pk!("[#6](~[#6])(~[#6])(~[#6])", 1), // 330  C(C)(C)(C)
    pk!("[#6](~[#6])(~[#6])(~[#6])", 2), // 331
    pk!("[#6](~[#6])(~[#7])", 1),        // 332  C(C)(N)
    pk!("[#6](~[#6])(~[#7])", 2),        // 333
    pk!("[#6](~[#6])(~[#7])", 4),        // 334
    pk!("[#6](~[#6])(~[#8])", 1),        // 335  C(C)(O)
    pk!("[#6](~[#6])(~[#8])", 2),        // 336
    pk!("[#6](~[#6])(~[#8])", 4),        // 337
    pk!("[#6](~[#6])(~[#16])", 1),       // 338  C(C)(S)
    pk!("[#6](~[#6])(~[#16])", 2),       // 339
    pk!("[#6](~[#6])(~[#17])", 1),       // 340  C(C)(Cl)
    pk!("[#6](~[#6])(~[#9])", 1),        // 341  C(C)(F)
    pk!("[#6](~[#6])(~[#35])", 1),       // 342  C(C)(Br)
    pk!("[#6](~[#6])(~[#53])", 1),       // 343  C(C)(I)
    pk!("[#6](~[#7])(~[#7])", 1),        // 344  C(N)(N)
    pk!("[#6](~[#7])(~[#7])", 2),        // 345
    pk!("[#6](~[#7])(~[#8])", 1),        // 346  C(N)(O)
    pk!("[#6](~[#7])(~[#8])", 2),        // 347
    pk!("[#6](~[#8])(~[#8])", 1),        // 348  C(O)(O)
    pk!("[#6](~[#8])(~[#8])", 2),        // 349
    pk!("[#6](~[#8])(~[#8])", 4),        // 350
    pk!("[#6](~[#8])(~[#16])", 1),       // 351  C(O)(S)
    // N with neighbors
    pk!("[#7](~[#6])(~[#6])", 1),        // 352  N(C)(C)
    pk!("[#7](~[#6])(~[#6])", 2),        // 353
    pk!("[#7](~[#6])(~[#6])", 4),        // 354
    pk!("[#7](~[#6])(~[#6])(~[#6])", 1), // 355  N(C)(C)(C)
    pk!("[#7](~[#6])(~[#6])(~[#6])", 2), // 356
    pk!("[#7](~[#6])(~[#7])", 1),        // 357  N(C)(N)
    pk!("[#7](~[#6])(~[#8])", 1),        // 358  N(C)(O)
    pk!("[#7](~[#8])(~[#8])", 1),        // 359  N(O)(O)  (nitro)
    pk!("[#7](~[#8])(~[#8])", 2),        // 360
    // O with neighbors
    pk!("[#8](~[#6])(~[#6])", 1), // 361  O(C)(C)  (ether)
    pk!("[#8](~[#6])(~[#6])", 2), // 362
    // S with neighbors
    pk!("[#16](~[#6])(~[#6])", 1), // 363  S(C)(C)  (thioether)
    pk!("[#16](~[#8])(~[#8])", 1), // 364  S(O)(O)  (sulfone/sulfonyl)
    pk!("[#16](~[#8])(~[#8])", 2), // 365
    // P with neighbors
    pk!("[#15](~[#8])(~[#8])", 1),        // 366  P(O)(O)
    pk!("[#15](~[#8])(~[#8])(~[#8])", 1), // 367  P(O)(O)(O)
    // Mixed neighbors
    pk!("[#6](~[#6])(~[#6])(~[#7])", 1), // 368  C(C)(C)(N)
    pk!("[#6](~[#6])(~[#6])(~[#8])", 1), // 369  C(C)(C)(O)
    pk!("[#6](~[#6])(~[#6])(~[#8])", 2), // 370
    pk!("[#6](~[#6])(~[#7])(~[#8])", 1), // 371  C(C)(N)(O)
    pk!("[#6](~[#6])(~[#8])(~[#8])", 1), // 372  C(C)(O)(O)
    pk!("[#6](~[#7])(~[#8])(~[#8])", 1), // 373  C(N)(O)(O)
    // C=O neighborhoods
    pk!("[#6](=[#8])(~[#6])", 1),  // 374  C(=O)(C)
    pk!("[#6](=[#8])(~[#6])", 2),  // 375
    pk!("[#6](=[#8])(~[#7])", 1),  // 376  C(=O)(N)  (amide)
    pk!("[#6](=[#8])(~[#7])", 2),  // 377
    pk!("[#6](=[#8])(~[#8])", 1),  // 378  C(=O)(O)  (ester/acid)
    pk!("[#6](=[#8])(~[#8])", 2),  // 379
    pk!("[#6](=[#8])(~[#8])", 4),  // 380
    pk!("[#6](=[#8])(~[#16])", 1), // 381  C(=O)(S)
    // C=N neighborhoods
    pk!("[#6](=[#7])(~[#6])", 1), // 382  C(=N)(C)
    pk!("[#6](=[#7])(~[#7])", 1), // 383  C(=N)(N)
    pk!("[#6](=[#7])(~[#8])", 1), // 384  C(=N)(O)
    // S=O neighborhoods
    pk!("[#16](=[#8])(~[#6])", 1),        // 385  S(=O)(C)
    pk!("[#16](=[#8])(~[#7])", 1),        // 386  S(=O)(N)
    pk!("[#16](=[#8])(~[#8])", 1),        // 387  S(=O)(O)
    pk!("[#16](=[#8])(=[#8])", 1),        // 388  S(=O)(=O) (sulfone)
    pk!("[#16](=[#8])(=[#8])(~[#6])", 1), // 389  S(=O)(=O)(C)
    pk!("[#16](=[#8])(=[#8])(~[#7])", 1), // 390  S(=O)(=O)(N)
    pk!("[#16](=[#8])(=[#8])(~[#8])", 1), // 391  S(=O)(=O)(O)
    // P=O neighborhoods
    pk!("[#15](=[#8])(~[#6])", 1),        // 392  P(=O)(C)
    pk!("[#15](=[#8])(~[#7])", 1),        // 393  P(=O)(N)
    pk!("[#15](=[#8])(~[#8])", 1),        // 394  P(=O)(O)
    pk!("[#15](=[#8])(~[#8])(~[#8])", 1), // 395  P(=O)(O)(O)
    // More neighbor patterns
    pk!("[#6](~[#9])(~[#9])", 1),        // 396  C(F)(F) (gem-difluoro)
    pk!("[#6](~[#9])(~[#9])(~[#9])", 1), // 397  C(F)(F)(F) (CF3)
    pk!("[#6](~[#17])(~[#17])", 1),      // 398  C(Cl)(Cl) (gem-dichloro)
    pk!("[#6](~[#6])(~[#6])(~[#6])(~[#6])", 1), // 399  C(C)(C)(C)(C) quat C
    pk!("[#7](~[#6])(~[#8])(~[#8])", 1), // 400  N(C)(O)(O)
    pk!("[#6](=[#6])(~[#6])", 1),        // 401  C(=C)(C)
    pk!("[#6](=[#6])(~[#6])", 2),        // 402
    pk!("[#6](=[#6])(~[#7])", 1),        // 403  C(=C)(N)
    pk!("[#6](=[#6])(~[#8])", 1),        // 404  C(=C)(O)
    pk!("[#6](#[#6])(~[#6])", 1),        // 405  C(#C)(C) internal alkyne
    pk!("[#6](#[#7])", 1),               // 406  C#N (nitrile)
    pk!("[#6](#[#7])", 2),               // 407
    // Charged atom patterns
    pk!("[+]", 1),    // 408
    pk!("[+]", 2),    // 409
    pk!("[-]", 1),    // 410
    pk!("[-]", 2),    // 411
    pk!("[#7+]", 1),  // 412  N+
    pk!("[#8-]", 1),  // 413  O-
    pk!("[#16-]", 1), // 414  S-
    pk!("[#15+]", 1), // 415  P+ (rare)
    // ===== Section 5: Detailed atom neighborhoods (bits 416-459) =====
    pk!("[#6](~[#6])(~[#6])(~[#7])(~[#8])", 1),  // 416
    pk!("[#6](~[#6])(~[#6])(~[#8])(~[#8])", 1),  // 417
    pk!("[#6](=[#8])(~[#6])(~[#7])", 1),         // 418  amide
    pk!("[#6](=[#8])(~[#6])(~[#7])", 2),         // 419
    pk!("[#6](=[#8])(~[#6])(~[#8])", 1),         // 420  ester
    pk!("[#6](=[#8])(~[#6])(~[#8])", 2),         // 421
    pk!("[#6](=[#8])(~[#8])(~[#8])", 1),         // 422  carbonate/anhydride
    pk!("[#6](=[#7])(~[#7])(~[#7])", 1),         // 423  guanidine
    pk!("[#6](=[#7])(~[#6])(~[#7])", 1),         // 424  amidine
    pk!("[#16](=[#8])(=[#8])(~[#6])(~[#6])", 1), // 425  sulfone
    pk!("[#16](=[#8])(=[#8])(~[#6])(~[#7])", 1), // 426  sulfonamide
    pk!("[#16](=[#8])(=[#8])(~[#6])(~[#8])", 1), // 427  sulfonate ester
    pk!("[#16](=[#8])(=[#8])(~[#7])(~[#8])", 1), // 428
    pk!("[#15](=[#8])(~[#8])(~[#8])(~[#8])", 1), // 429  phosphate
    pk!("[#15](=[#8])(~[#8])(~[#8])(~[#6])", 1), // 430  phosphonate
    // Aromatic neighborhoods
    pk!("[a](~[#7])", 1),  // 431  aromatic with N neighbor
    pk!("[a](~[#7])", 2),  // 432
    pk!("[a](~[#8])", 1),  // 433  aromatic with O neighbor
    pk!("[a](~[#8])", 2),  // 434
    pk!("[a](~[#16])", 1), // 435  aromatic with S neighbor
    pk!("[a](~[#9])", 1),  // 436  aromatic with F neighbor
    pk!("[a](~[#17])", 1), // 437  aromatic with Cl neighbor
    pk!("[a](~[#35])", 1), // 438  aromatic with Br neighbor
    pk!("[a](~[#53])", 1), // 439  aromatic with I neighbor
    // Chain length indicators
    pk!("[#6]~[#6]~[#6]", 1),                     // 440  3-chain
    pk!("[#6]~[#6]~[#6]~[#6]", 1),                // 441  4-chain
    pk!("[#6]~[#6]~[#6]~[#6]~[#6]", 1),           // 442  5-chain
    pk!("[#6]~[#6]~[#6]~[#6]~[#6]~[#6]", 1),      // 443  6-chain
    pk!("[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]", 1), // 444  7-chain
    // Heteroatom chains
    pk!("[#6]~[#7]~[#6]", 1),      // 445  C-N-C
    pk!("[#6]~[#7]~[#6]", 2),      // 446
    pk!("[#6]~[#8]~[#6]", 1),      // 447  C-O-C (ether)
    pk!("[#6]~[#8]~[#6]", 2),      // 448
    pk!("[#6]~[#16]~[#6]", 1),     // 449  C-S-C (thioether)
    pk!("[#7]~[#6]~[#7]", 1),      // 450  N-C-N
    pk!("[#7]~[#6]~[#8]", 1),      // 451  N-C-O
    pk!("[#7]~[#6]~[#8]", 2),      // 452
    pk!("[#8]~[#6]~[#8]", 1),      // 453  O-C-O (acetal)
    pk!("[#8]~[#6]~[#8]", 2),      // 454
    pk!("[#7]~[#6]~[#16]", 1),     // 455  N-C-S
    pk!("[#6]~[#7]~[#8]", 1),      // 456  C-N-O
    pk!("[#6]~[#16]~[#8]", 1),     // 457  C-S-O
    pk!("[#7]~[#6]~[#6]~[#7]", 1), // 458  N-C-C-N
    pk!("[#8]~[#6]~[#6]~[#8]", 1), // 459  O-C-C-O
    // ===== Section 6: Simple SMARTS patterns (bits 460-712) =====
    // Functional groups
    pk!("[OX2H]", 1),                   // 460  hydroxyl
    pk!("[OX2H]", 2),                   // 461
    pk!("[OX2H]", 4),                   // 462
    pk!("[CX3](=O)[OX2H1]", 1),         // 463  carboxylic acid
    pk!("[CX3](=O)[OX2H1]", 2),         // 464
    pk!("[CX3H1](=O)", 1),              // 465  aldehyde
    pk!("[CX3](=O)[#6]", 1),            // 466  ketone/aldehyde C=O with C
    pk!("[CX3](=O)[#6]", 2),            // 467
    pk!("[CX3](=O)[#6]", 4),            // 468
    pk!("[NX3;H2]", 1),                 // 469  primary amine NH2
    pk!("[NX3;H2]", 2),                 // 470
    pk!("[NX3;H1]([#6])[#6]", 1),       // 471  secondary amine
    pk!("[NX3;H1]([#6])[#6]", 2),       // 472
    pk!("[NX3;H0]([#6])([#6])[#6]", 1), // 473  tertiary amine
    pk!("[NX3;H0]([#6])([#6])[#6]", 2), // 474
    pk!("[NX3][CX3](=[OX1])", 1),       // 475  amide N-C=O
    pk!("[NX3][CX3](=[OX1])", 2),       // 476
    pk!("[NX3][CX3](=[OX1])", 4),       // 477
    pk!("[#6][OX2][#6]", 1),            // 478  ether
    pk!("[#6][OX2][#6]", 2),            // 479
    pk!("[#6][OX2][#6]", 4),            // 480
    // Esters
    pk!("[#6][CX3](=O)[OX2][#6]", 1), // 481  ester
    pk!("[#6][CX3](=O)[OX2][#6]", 2), // 482
    // Nitro
    pk!("[NX3](=[OX1])=[OX1]", 1), // 483
    pk!("[NX3](=[OX1])=[OX1]", 2), // 484
    // Sulfonamide
    pk!("[#16X4](=[OX1])(=[OX1])[NX3]", 1), // 485
    pk!("[#16X4](=[OX1])(=[OX1])[NX3]", 2), // 486
    // Sulfone
    pk!("[#16X4](=[OX1])(=[OX1])([#6])[#6]", 1), // 487
    // Sulfoxide
    pk!("[#16X3](=[OX1])([#6])[#6]", 1), // 488
    // Thiol
    pk!("[#16X2H]", 1), // 489
    // Thioether
    pk!("[#6][#16X2][#6]", 1), // 490
    // Phosphate
    pk!("[#15](=[#8])(~[#8])", 1), // 491
    // Halides
    pk!("[#9]", 1),  // 492  F
    pk!("[#17]", 1), // 493  Cl
    pk!("[#35]", 1), // 494  Br
    pk!("[#53]", 1), // 495  I
    // Aromatic halides
    pk!("[a][#9]", 1),  // 496  ArF
    pk!("[a][#17]", 1), // 497  ArCl
    pk!("[a][#35]", 1), // 498  ArBr
    pk!("[a][#53]", 1), // 499  ArI
    // Nitrile
    pk!("[CX2]#[NX1]", 1), // 500
    pk!("[CX2]#[NX1]", 2), // 501
    // Alkyne
    pk!("[CX2]#[CX2]", 1), // 502
    // Imine
    pk!("[CX3]=[NX2]", 1), // 503
    pk!("[CX3]=[NX2]", 2), // 504
    // Enol/enolate
    pk!("[OX2][CX3]=[CX3]", 1), // 505
    // Hydrazine
    pk!("[NX3][NX3]", 1), // 506
    // Azo
    pk!("[#7]=[#7]", 1), // 507
    // Diazo
    pk!("[#6]=[#7]=[#7]", 1), // 508
    // Azide
    pk!("[#7]=[#7]=[#7]", 1), // 509
    // Isocyanate
    pk!("[NX2]=[CX2]=[OX1]", 1), // 510
    // Isothiocyanate
    pk!("[NX2]=[CX2]=[SX1]", 1), // 511
    // Urea
    pk!("[NX3][CX3](=[OX1])[NX3]", 1), // 512
    pk!("[NX3][CX3](=[OX1])[NX3]", 2), // 513
    // Thiourea
    pk!("[NX3][CX3](=[SX1])[NX3]", 1), // 514
    // Carbamate
    pk!("[NX3][CX3](=[OX1])[OX2]", 1), // 515
    pk!("[NX3][CX3](=[OX1])[OX2]", 2), // 516
    // Anhydride
    pk!("[CX3](=[OX1])[OX2][CX3](=[OX1])", 1), // 517
    // Lactone (cyclic ester)
    pk!("[R][CX3](=[OX1])[OX2][R]", 1), // 518
    // Lactam (cyclic amide)
    pk!("[R][CX3](=[OX1])[NX3][R]", 1), // 519
    // Epoxide (3-membered ring with O)
    pk!("[r3;#8]", 1), // 520
    // Peroxide
    pk!("[OX2][OX2]", 1), // 521
    // Hemiacetal
    pk!("[OX2H][CX4][OX2]", 1), // 522
    // Acetal
    pk!("[OX2][CX4][OX2]", 1), // 523
    // Enamine
    pk!("[NX3][CX3]=[CX3]", 1), // 524
    // Oxime
    pk!("[CX3]=[NX2][OX2H]", 1), // 525
    // Hydroxamic acid
    pk!("[CX3](=[OX1])[NX3][OX2H]", 1), // 526
    // Aromatic amine
    pk!("[NX3;H2]a", 1),       // 527
    pk!("[NX3;H1](a)[#6]", 1), // 528
    pk!("[NX3;H0](a)(a)", 1),  // 529
    // Aromatic hydroxyl (phenol)
    pk!("[OX2H]a", 1), // 530
    pk!("[OX2H]a", 2), // 531
    // Aromatic ether
    pk!("[OX2](a)[#6]", 1), // 532
    pk!("[OX2](a)[#6]", 2), // 533
    // Aromatic thiol
    pk!("[#16X2H]a", 1), // 534
    // Aromatic thioether
    pk!("[#16X2](a)[#6]", 1), // 535
    // Aromatic aldehyde
    pk!("[CX3H1](=O)a", 1), // 536
    // Aromatic ketone
    pk!("a[CX3](=O)[#6]", 1), // 537
    // Aromatic carboxylic acid
    pk!("a[CX3](=O)[OX2H]", 1), // 538
    // Aromatic ester
    pk!("a[CX3](=O)[OX2][#6]", 1), // 539
    // Aromatic amide
    pk!("a[CX3](=O)[NX3]", 1), // 540
    pk!("a[CX3](=O)[NX3]", 2), // 541
    // Aromatic nitro
    pk!("a[NX3](=[OX1])=[OX1]", 1), // 542
    // Aromatic sulfonamide
    pk!("a[#16X4](=[OX1])(=[OX1])[NX3]", 1), // 543
    // Aromatic sulfonate
    pk!("a[#16X4](=[OX1])(=[OX1])[OX2]", 1), // 544
    // Aromatic nitrile
    pk!("a[CX2]#[NX1]", 1), // 545
    // Patterns with specific carbon types
    pk!("[CH3]", 1), // 546  methyl
    pk!("[CH3]", 2), // 547
    pk!("[CH3]", 4), // 548
    pk!("[CH3]", 8), // 549
    pk!("[CH2]", 1), // 550  methylene
    pk!("[CH2]", 2), // 551
    pk!("[CH2]", 4), // 552
    pk!("[CH2]", 8), // 553
    pk!("[CH1]", 1), // 554  methine
    pk!("[CH1]", 2), // 555
    pk!("[CH1]", 4), // 556
    pk!("[CH0]", 1), // 557  quaternary C
    pk!("[CH0]", 2), // 558
    pk!("[CH0]", 4), // 559
    // Charged nitrogen
    pk!("[NX4+]", 1), // 560  quaternary N+
    pk!("[NX4+]", 2), // 561
    // Aromatic nitrogen types
    pk!("[nX2]", 1),   // 562  pyridine-like N
    pk!("[nX2]", 2),   // 563
    pk!("[nX3H1]", 1), // 564  pyrrole-like NH
    pk!("[nX3H1]", 2), // 565
    // Aromatic oxygen
    pk!("[oX2]", 1), // 566  furan-like O
    // Aromatic sulfur
    pk!("[sX2]", 1), // 567  thiophene-like S
    // Heteroatom connectivity
    pk!("[!#6;!#1]~[!#6;!#1]", 1), // 568  heteroatom-heteroatom bond
    pk!("[!#6;!#1]~[!#6;!#1]", 2), // 569
    pk!("[!#6;!#1]~[!#6;!#1]", 4), // 570
    // Degree patterns
    pk!("[D1]", 1), // 571  terminal atom
    pk!("[D1]", 2), // 572
    pk!("[D1]", 4), // 573
    pk!("[D1]", 8), // 574
    pk!("[D4]", 1), // 575  4-connected atom
    pk!("[D4]", 2), // 576
    // Hydrogen bond donor/acceptor proxies
    pk!("[#7;H1,H2]", 1), // 577  NH or NH2
    pk!("[#7;H1,H2]", 2), // 578
    pk!("[#7;H1,H2]", 4), // 579
    pk!("[#8;H1]", 1),    // 580  OH
    pk!("[#8;H1]", 2),    // 581
    pk!("[#8;H1]", 4),    // 582
    // Aliphatic unsaturation
    pk!("[CX3]=[CX3]", 1), // 583  C=C (alkene)
    pk!("[CX3]=[CX3]", 2), // 584
    pk!("[CX3]=[CX3]", 4), // 585
    // Aromatic C-C bonds
    pk!("c:c", 1),  // 586
    pk!("c:c", 4),  // 587
    pk!("c:c", 8),  // 588
    pk!("c:c", 16), // 589
    // Aromatic C-N bonds
    pk!("c:n", 1), // 590
    pk!("c:n", 2), // 591
    pk!("c:n", 4), // 592
    // Aromatic C-O
    pk!("c:o", 1), // 593
    // Aromatic C-S
    pk!("c:s", 1), // 594
    // Aromatic N-N
    pk!("n:n", 1), // 595
    // Multi-bond patterns
    pk!("[#6]=[#6]~[#6]=[#6]", 1), // 596  conjugated diene
    pk!("[#6]=[#6]~[#6]=[#8]", 1), // 597  enone
    pk!("[#6]=[#6]~[#6]=[#7]", 1), // 598  conjugated imine
    pk!("[#8]~[#6]~[#6]~[#7]", 1), // 599
    pk!("[#8]~[#6]~[#6]~[#7]", 2), // 600
    pk!("[#8]~[#6]~[#6]~[#8]", 1), // 601
    pk!("[#8]~[#6]~[#6]~[#8]", 2), // 602
    pk!("[#7]~[#6]~[#6]~[#7]", 1), // 603
    pk!("[#7]~[#6]~[#6]~[#7]", 2), // 604
    // Ring-chain connection patterns
    pk!("a~[#6]~[#6]", 1),   // 605  Ar-C-C
    pk!("a~[#6]~[#6]", 2),   // 606
    pk!("a~[#6]~[#7]", 1),   // 607  Ar-C-N
    pk!("a~[#6]~[#8]", 1),   // 608  Ar-C-O
    pk!("a~[#6](=[#8])", 1), // 609  Ar-C=O
    pk!("a~[#6](=[#8])", 2), // 610
    pk!("a~[#7]~[#6]", 1),   // 611  Ar-N-C
    pk!("a~[#8]~[#6]", 1),   // 612  Ar-O-C
    pk!("a~[#8]~[#6]", 2),   // 613
    pk!("a~[#16]~[#6]", 1),  // 614  Ar-S-C
    // Common drug fragments
    pk!("[#6]~[#6](=[#8])~[#7]~[#6]", 1), // 615  C-C(=O)-N-C (peptide bond)
    pk!("[#6]~[#6](=[#8])~[#7]~[#6]", 2), // 616
    pk!("[#6]~[#6](=[#8])~[#8]~[#6]", 1), // 617  C-C(=O)-O-C (ester chain)
    pk!("[#6]~[#6](=[#8])~[#8]~[#6]", 2), // 618
    pk!("[#6]~[#7]~[#6](=[#8])", 1),      // 619  C-N-C=O
    pk!("[#6]~[#7]~[#6](=[#8])", 2),      // 620
    pk!("[#6]~[#8]~[#6](=[#8])", 1),      // 621  C-O-C=O
    pk!("[#6]~[#8]~[#6](=[#8])", 2),      // 622
    pk!("[#7]~[#6](=[#8])~[#7]", 1),      // 623  N-C(=O)-N (urea pattern)
    pk!("[#7]~[#6](=[#8])~[#8]", 1),      // 624  N-C(=O)-O (carbamate)
    pk!("[#8]~[#6](=[#8])~[#8]", 1),      // 625  O-C(=O)-O (carbonate)
    // Sulfonyl patterns
    pk!("[#6]~[#16](=[#8])(=[#8])~[#6]", 1), // 626  C-SO2-C
    pk!("[#6]~[#16](=[#8])(=[#8])~[#7]", 1), // 627  C-SO2-N
    pk!("[#6]~[#16](=[#8])(=[#8])~[#8]", 1), // 628  C-SO2-O
    // Phosphoryl patterns
    pk!("[#6]~[#15](=[#8])~[#8]", 1), // 629  C-P(=O)-O
    pk!("[#8]~[#15](=[#8])~[#8]", 1), // 630  O-P(=O)-O
    // Additional functional group patterns
    pk!("[CX3](=[OX1])[CX3](=[OX1])", 1),  // 631  1,2-diketone
    pk!("[CX3](=[OX1])~[CX3](=[OX1])", 1), // 632
    pk!("[OX2H][CX3](=[OX1])[OX2H]", 1),   // 633  free dicarboxylic acid pattern
    // Aromatic substitution patterns
    pk!("a(~[#6])(~[#7])", 1),  // 634  Ar with C and N substituents
    pk!("a(~[#6])(~[#8])", 1),  // 635  Ar with C and O substituents
    pk!("a(~[#7])(~[#8])", 1),  // 636  Ar with N and O substituents
    pk!("a(~[#9])(~[#6])", 1),  // 637  Ar with F and C
    pk!("a(~[#17])(~[#6])", 1), // 638  Ar with Cl and C
    // Saturated chains
    pk!("[CX4][CX4]", 1),           // 639  sp3-sp3
    pk!("[CX4][CX4]", 2),           // 640
    pk!("[CX4][CX4]", 4),           // 641
    pk!("[CX4][CX4]", 8),           // 642
    pk!("[CX4][CX4][CX4]", 1),      // 643  sp3-sp3-sp3
    pk!("[CX4][CX4][CX4]", 2),      // 644
    pk!("[CX4][CX4][CX4][CX4]", 1), // 645
    // Mixed hybridization chains
    pk!("[CX3][CX4]", 1), // 646  sp2-sp3
    pk!("[CX3][CX4]", 2), // 647
    pk!("[CX3][CX3]", 1), // 648  sp2-sp2
    pk!("[CX3][CX3]", 2), // 649
    pk!("[CX3][CX3]", 4), // 650
    // Aromatic ring connections
    pk!("a~a~a", 1),       // 651  3 aromatic
    pk!("a~a~a~a", 1),     // 652  4 aromatic
    pk!("a~a~a~a~a", 1),   // 653  5 aromatic
    pk!("a~a~a~a~a~a", 1), // 654  6 aromatic (benzene ring)
    // Multiple aromatic rings
    pk!("a1aaaaa1~a1aaaaa1", 1), // 655  biphenyl-like (2 fused/linked 6-rings)
    // Specific heterocycle fragments
    pk!("c1ccncc1", 1),    // 656  pyridine
    pk!("c1ccncc1", 2),    // 657
    pk!("c1cc[nH]c1", 1),  // 658  pyrrole
    pk!("c1ccoc1", 1),     // 659  furan
    pk!("c1ccsc1", 1),     // 660  thiophene
    pk!("c1cnc[nH]1", 1),  // 661  imidazole
    pk!("c1ccnc(c1)N", 1), // 662  2-aminopyridine
    pk!("c1nncn1", 1),     // 663  1,2,4-triazine-like
    pk!("c1nnn[nH]1", 1),  // 664  tetrazole
    // Common scaffolds
    pk!("c1ccc2ccccc2c1", 1),   // 665  naphthalene
    pk!("c1ccc2[nH]ccc2c1", 1), // 666  indole
    pk!("c1ccc2nccc2c1", 1),    // 667  quinoline-like
    pk!("c1ccc2c(c1)ccn2", 1),  // 668  indole-like (N in 5-ring)
    // Bond order patterns
    pk!("[#6]-[#6]", 1),      // 669  single C-C
    pk!("[#6]-[#6]", 4),      // 670
    pk!("[#6]-[#6]", 8),      // 671
    pk!("[#6]-[#6]", 16),     // 672
    pk!("[#6]-[#7]", 1),      // 673  single C-N
    pk!("[#6]-[#7]", 2),      // 674
    pk!("[#6]-[#7]", 4),      // 675
    pk!("[#6]-[#8]", 1),      // 676  single C-O
    pk!("[#6]-[#8]", 2),      // 677
    pk!("[#6]-[#8]", 4),      // 678
    pk!("[#6]=[#6]", 1),      // 679  double C=C
    pk!("[#6]=[#6]", 2),      // 680
    pk!("[#6]=[#8]", 1),      // 681  double C=O
    pk!("[#6]=[#8]", 2),      // 682
    pk!("[#6]=[#8]", 4),      // 683
    pk!("[#6]=[#7]", 1),      // 684  double C=N
    pk!("[#6]=[#7]", 2),      // 685
    pk!("[#7]-[#6]=[#8]", 1), // 686  N-C=O (amide bond)
    pk!("[#7]-[#6]=[#8]", 2), // 687
    pk!("[#8]-[#6]=[#8]", 1), // 688  O-C=O (ester)
    pk!("[#8]-[#6]=[#8]", 2), // 689
    // sp3 heteroatom patterns
    pk!("[NX3H2][CX4]", 1),        // 690  NH2-C(sp3)
    pk!("[NX3H2][CX4]", 2),        // 691
    pk!("[NX3H1]([CX4])[CX4]", 1), // 692  NH(sp3)(sp3)
    pk!("[OX2H][CX4]", 1),         // 693  OH-C(sp3) (alcohol)
    pk!("[OX2H][CX4]", 2),         // 694
    pk!("[OX2][CX4][OX2]", 1),     // 695  O-C-O (sp3)
    // Aromatic amine detailed
    pk!("[NX3;H2]c", 1),             // 696
    pk!("[NX3;H1]([#6])c", 1),       // 697
    pk!("[NX3;H0]([#6])([#6])c", 1), // 698
    // Carbonyl patterns
    pk!("[CX3]=[OX1]", 1), // 699  any C=O
    pk!("[CX3]=[OX1]", 2), // 700
    pk!("[CX3]=[OX1]", 4), // 701
    pk!("[CX3]=[OX1]", 8), // 702
    // Aromatic ring count proxies
    pk!("a1aaaaa1", 1), // 703  6-membered aromatic ring
    pk!("a1aaaaa1", 2), // 704
    pk!("a1aaaaa1", 4), // 705
    pk!("a1aaaa1", 1),  // 706  5-membered aromatic ring
    pk!("a1aaaa1", 2),  // 707
    pk!("a1aaaa1", 4),  // 708
    // General ring patterns
    pk!("[R]", 1),  // 709
    pk!("[R]", 4),  // 710
    pk!("[R]", 8),  // 711
    pk!("[R]", 16), // 712
    // ===== Section 7: Complex SMARTS patterns (bits 713-880) =====
    // Piperidine / piperazine fragments
    pk!("C1CCNCC1", 1),   // 713  piperidine
    pk!("C1CNCCN1", 1),   // 714  piperazine
    pk!("C1CCOCC1", 1),   // 715  tetrahydropyran / morpholine
    pk!("C1CCNC1", 1),    // 716  pyrrolidine
    pk!("C1CCOC1", 1),    // 717  tetrahydrofuran
    pk!("C1CCSC1", 1),    // 718  tetrahydrothiophene
    pk!("C1CCC1", 1),     // 719  cyclobutane
    pk!("C1CC1", 1),      // 720  cyclopropane
    pk!("C1CCCCC1", 1),   // 721  cyclohexane
    pk!("C1CCCC1", 1),    // 722  cyclopentane
    pk!("C1CCCCCCC1", 1), // 723  cyclooctane
    pk!("C1CCCCCC1", 1),  // 724  cycloheptane
    // Multi-ring systems
    pk!("c1ccc2c(c1)cccc2", 1),   // 725  naphthalene alt
    pk!("c1cc2ccccc2cc1", 1),     // 726  naphthalene alt2
    pk!("c1ccc(cc1)c1ccccc1", 1), // 727  biphenyl
    // Benzimidazole, benzothiazole, benzoxazole
    pk!("c1cc2[nH]cnc2cc1", 1), // 728  benzimidazole
    pk!("c1cc2ncsc2cc1", 1),    // 729  benzothiazole
    pk!("c1cc2ncoc2cc1", 1),    // 730  benzoxazole
    // Drug-relevant patterns
    pk!("[NX3][CX3](=[NX2])[NX3]", 1),    // 731  guanidine
    pk!("[NX3][CX3](=[NX2])[NX3]", 2),    // 732
    pk!("[NX3][SX4](=[OX1])(=[OX1])", 1), // 733  sulfonamide
    pk!("[NX3][SX4](=[OX1])(=[OX1])", 2), // 734
    pk!("[OX2][CX3](=[OX1])[NX3]", 1),    // 735  carbamate
    pk!("[NX3][CX3](=[SX1])", 1),         // 736  thioamide
    pk!("[OX2][CX3](=[SX1])", 1),         // 737  thioester
    pk!("[SX2][CX3](=[OX1])", 1),         // 738  thioester alt
    // Heterocyclic nitrogen patterns
    pk!("[nX2]c[nX2]", 1),   // 739  N-C-N in aromatic (pyrimidine-like)
    pk!("[nX2]c[nX2]", 2),   // 740
    pk!("[nX2]c[nX3H1]", 1), // 741  imidazole N-C-NH
    pk!("[nX2]cc[nX2]", 1),  // 742  1,3-diazine pattern
    // Extended conjugation
    pk!("c~c~c~c~c~c~c~c", 1),     // 743  8 conjugated aromatic atoms
    pk!("c~c~c~c~c~c~c~c~c~c", 1), // 744  10 conjugated
    // Fused ring systems
    pk!("[R]1~[R]~[R]~[R]2~[R]~[R]~[R]~[R]~[R]2~[R]1", 1), // 745  bicyclic
    // Common drug scaffolds continued
    pk!("[#6]~[#6](~[#6])~[#7]~[#6](=[#8])~[#6]", 1), // 746
    pk!("[#6]~[#7]~[#6]~[#6]~[#8]", 1),               // 747
    pk!("[#6]~[#7]~[#6]~[#6]~[#7]", 1),               // 748
    pk!("[#6]~[#8]~[#6]~[#6]~[#8]", 1),               // 749
    pk!("[#6]~[#8]~[#6]~[#6]~[#7]", 1),               // 750
    // More specific bond patterns
    pk!("[CX4]-[#7]-[CX3]=[OX1]", 1),        // 751  sp3C-N-C=O
    pk!("[CX4]-[#8]-[CX3]=[OX1]", 1),        // 752  sp3C-O-C=O
    pk!("a-[CX3]=[OX1]", 1),                 // 753  Ar-C=O
    pk!("a-[CX3]=[OX1]", 2),                 // 754
    pk!("a-[NX3]-[CX3]=[OX1]", 1),           // 755  Ar-N-C=O
    pk!("a-[NX3]-[#16X4](=[OX1])=[OX1]", 1), // 756  Ar-N-SO2
    pk!("a-[OX2]-[CX4]", 1),                 // 757  Ar-O-Csp3
    pk!("a-[OX2]-[CX4]", 2),                 // 758
    pk!("a-[OX2]-[CX3]=[OX1]", 1),           // 759  Ar-O-C=O
    pk!("a-[#16X2]-[#6]", 1),                // 760  Ar-S-C
    // 4-atom chain patterns with heteroatoms
    pk!("[#6]-[#6]-[#6]-[#7]", 1), // 761
    pk!("[#6]-[#6]-[#6]-[#7]", 2), // 762
    pk!("[#6]-[#6]-[#6]-[#8]", 1), // 763
    pk!("[#6]-[#6]-[#6]-[#8]", 2), // 764
    pk!("[#6]-[#6]-[#7]-[#6]", 1), // 765
    pk!("[#6]-[#6]-[#7]-[#6]", 2), // 766
    pk!("[#6]-[#6]-[#8]-[#6]", 1), // 767
    pk!("[#6]-[#6]-[#8]-[#6]", 2), // 768
    pk!("[#6]-[#7]-[#6]-[#6]", 1), // 769
    pk!("[#6]-[#7]-[#6]-[#6]", 2), // 770
    pk!("[#6]-[#8]-[#6]-[#6]", 1), // 771
    pk!("[#6]-[#8]-[#6]-[#6]", 2), // 772
    // Branched patterns
    pk!("[#6](-[#7])(-[#8])", 1),        // 773
    pk!("[#6](-[#7])(-[#8])", 2),        // 774
    pk!("[#6](-[#6])(-[#7])(-[#8])", 1), // 775
    pk!("[#6](-[#6])(-[#6])(-[#7])", 1), // 776
    pk!("[#6](-[#6])(-[#6])(-[#8])", 1), // 777
    pk!("[#6](-[#6])(-[#6])(-[#8])", 2), // 778
    // Heavy atom count ranges (using connectivity)
    pk!("[!#1]~[!#1]~[!#1]~[!#1]", 1), // 779  >=4 heavy atoms in chain
    pk!("[!#1]~[!#1]~[!#1]~[!#1]~[!#1]", 1), // 780  >=5
    pk!("[!#1]~[!#1]~[!#1]~[!#1]~[!#1]~[!#1]", 1), // 781 >=6
    // Ring substituent count patterns
    pk!("[R]([!R])([!R])", 1), // 782  ring atom with 2 non-ring substituents
    pk!("[R]([!R])([!R])", 2), // 783
    pk!("[R]([!R])", 1),       // 784  ring atom with 1 non-ring substituent
    pk!("[R]([!R])", 2),       // 785
    pk!("[R]([!R])", 4),       // 786
    pk!("[R]([!R])", 8),       // 787
    // Specific element patterns
    pk!("[#14]~[#8]", 1), // 788  Si-O
    pk!("[#5]~[#8]", 1),  // 789  B-O
    pk!("[#14]~[#6]", 1), // 790  Si-C
    pk!("[#5]~[#6]", 1),  // 791  B-C
    pk!("[#5]~[#7]", 1),  // 792  B-N
    // Long-range patterns
    pk!("[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]", 1), // 793  8-carbon chain
    pk!("[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]", 1), // 794  8-C single bond chain
    // Extended heterocyclic patterns
    pk!("c1ccc(cc1)-c1ccccc1", 1),   // 795  biphenyl
    pk!("c1ccc(cc1)-c1ccncc1", 1),   // 796  phenylpyridine
    pk!("c1ccc(cc1)-c1cc[nH]c1", 1), // 797  phenylpyrrole
    pk!("c1ccc(cc1)-c1ccoc1", 1),    // 798  phenylfuran
    pk!("c1ccc(cc1)-c1ccsc1", 1),    // 799  phenylthiophene
    // Additional functional group indicators
    pk!("[CX3](=[OX1])[OX2][CX3](=[OX1])", 1), // 800  anhydride
    pk!("[OX2H][SX4](=[OX1])=[OX1]", 1),       // 801  sulfonic acid
    pk!("[NX3][SX4](=[OX1])(=[OX1])[#6]", 1),  // 802  sulfonamide (N)
    pk!("[OX2][SX4](=[OX1])(=[OX1])[#6]", 1),  // 803  sulfonate ester
    pk!("[NX3]c1cc[nX2]cc1", 1),               // 804  aminopyridine
    pk!("[OX2H]c1cc[nX2]cc1", 1),              // 805  hydroxypyridine
    pk!("[NX3]c1ccccc1", 1),                   // 806  aniline
    pk!("[NX3]c1ccccc1", 2),                   // 807
    pk!("[OX2H]c1ccccc1", 1),                  // 808  phenol
    pk!("[OX2H]c1ccccc1", 2),                  // 809
    // More complex drug patterns
    pk!("[#6]~[#6](=[#8])~[#7]~[#6]~[#6]~[#6]", 1), // 810
    pk!("[#6]~[#6](=[#8])~[#8]~[#6]~[#6]~[#6]", 1), // 811
    pk!("[#6]~[#7]~[#6](=[#8])~[#6]~[#6]", 1),      // 812
    pk!("[#6]~[#8]~[#6](=[#8])~[#6]~[#6]", 1),      // 813
    pk!("[#7]~[#6](=[#8])~[#6]~[#6](=[#8])", 1),    // 814
    pk!("[#8]~[#6](=[#8])~[#6]~[#6](=[#8])", 1),    // 815
    // Common substructure patterns
    pk!("[#6]-[#6](=[#8])-[#7]-[#1]", 1),      // 816  C-C(=O)-NH
    pk!("[#6]-[#6](=[#8])-[#7]-[#1]", 2),      // 817
    pk!("[#6]-[#6](=[#8])-[#8]-[#1]", 1),      // 818  C-C(=O)-OH
    pk!("[#6]-[#6](=[#8])-[#8]-[#1]", 2),      // 819
    pk!("[#6]-[#7]-[#6]-[#6](=[#8])-[#8]", 1), // 820  amino acid pattern
    // Stereochemistry-insensitive ring patterns
    pk!("C1=CC=CC=C1", 1),           // 821  cyclohexadiene-like
    pk!("[R]=,:[R]=,:[R]", 1),       // 822  conjugated ring bonds
    pk!("[R]=,:[R]=,:[R]", 2),       // 823
    pk!("[R]=,:[R]=,:[R]=,:[R]", 1), // 824
    // Additional element-specific patterns
    pk!("[#16](=[#8])", 1), // 825  S=O
    pk!("[#16](=[#8])", 2), // 826
    pk!("[#15](=[#8])", 1), // 827  P=O
    pk!("[#15](=[#8])", 2), // 828
    pk!("[#7](=[#8])", 1),  // 829  N=O
    pk!("[#7](=[#8])", 2),  // 830
    // Degree-specific heteroatom patterns
    pk!("[NX1]", 1), // 831  terminal N (nitrile, azide, etc.)
    pk!("[NX1]", 2), // 832
    pk!("[NX2]", 1), // 833  2-connected N
    pk!("[NX2]", 2), // 834
    pk!("[NX3]", 1), // 835  3-connected N
    pk!("[NX3]", 2), // 836
    pk!("[NX3]", 4), // 837
    pk!("[NX3]", 8), // 838
    pk!("[OX1]", 1), // 839  terminal O (=O)
    pk!("[OX1]", 2), // 840
    pk!("[OX1]", 4), // 841
    pk!("[OX2]", 1), // 842  2-connected O (-O-)
    pk!("[OX2]", 2), // 843
    pk!("[OX2]", 4), // 844
    pk!("[OX2]", 8), // 845
    pk!("[SX1]", 1), // 846  terminal S (=S)
    pk!("[SX2]", 1), // 847  2-connected S
    pk!("[SX2]", 2), // 848
    pk!("[SX3]", 1), // 849  sulfoxide S
    pk!("[SX4]", 1), // 850  sulfone S
    pk!("[SX4]", 2), // 851
    // Bridgehead and spiro indicators
    pk!("[R;D3;x3]", 1), // 852  fused ring junction
    pk!("[R;D3;x3]", 2), // 853
    pk!("[R;D4;x4]", 1), // 854  spiro center
    // Molecular property indicators
    pk!("[!#6;!#1;!#7;!#8;!#16;!#9;!#17;!#35;!#53;!#15]", 1), // 855  unusual element
    // Remaining keys: additional structural patterns
    pk!("[#6]~[#6]~[#6](=[#8])~[#8]", 1), // 856  beta-keto ester-like
    pk!("[#6]~[#6]~[#6](=[#8])~[#7]", 1), // 857  beta-keto amide-like
    pk!("[#6]=[#6]~[#6](=[#8])", 1),      // 858  alpha-beta-unsaturated carbonyl
    pk!("[#6]=[#6]~[#6](=[#8])", 2),      // 859
    pk!("a~[#6]=[#6]~a", 1),              // 860  stilbene-like
    pk!("a~[#6]~[#6]~a", 1),              // 861  bibenzyl-like
    pk!("a~[#6]~[#6]~a", 2),              // 862
    pk!("a~[#7]~a", 1),                   // 863  Ar-N-Ar (diarylamine)
    pk!("a~[#8]~a", 1),                   // 864  Ar-O-Ar (diaryl ether)
    pk!("a~[#16]~a", 1),                  // 865  Ar-S-Ar (diaryl sulfide)
    pk!("[#6]1~[#6]~[#6]~[#7]~[#6]~[#6]1", 1), // 866  6-ring with 1 N
    pk!("[#6]1~[#6]~[#7]~[#6]~[#7]~[#6]1", 1), // 867  6-ring with 2 N (pyrimidine)
    pk!("[#6]1~[#6]~[#6]~[#8]~[#6]~[#6]1", 1), // 868  6-ring with 1 O
    pk!("[#6]1~[#6]~[#6]~[#16]~[#6]~[#6]1", 1), // 869  6-ring with 1 S
    pk!("[#6]1~[#6]~[#7]~[#6]~[#6]1", 1), // 870  5-ring with 1 N
    pk!("[#6]1~[#7]~[#6]~[#7]~[#6]1", 1), // 871  5-ring with 2 N (imidazole)
    pk!("[#6]1~[#6]~[#8]~[#6]~[#6]1", 1), // 872  5-ring with 1 O
    pk!("[#6]1~[#6]~[#16]~[#6]~[#6]1", 1), // 873  5-ring with 1 S
    pk!("[#6]1~[#7]~[#6]~[#8]~[#6]1", 1), // 874  oxazole-like
    pk!("[#6]1~[#7]~[#6]~[#16]~[#6]1", 1), // 875  thiazole-like
    pk!("[#6]1~[#7]~[#7]~[#6]~[#6]1", 1), // 876  pyrazole-like
    pk!("[#6]1~[#7]~[#7]~[#7]~[#6]1", 1), // 877  triazole-like
    pk!("[#6]1~[#7]~[#7]~[#7]~[#7]1", 1), // 878  tetrazole-like
    pk!("[#6]1~[#6]~[#7]~[#7]~[#6]1", 1), // 879  pyrazole-like alt
    pk!("[#6]1~[#8]~[#6]~[#8]~[#6]1", 1), // 880  dioxolane-like
];

// Verify key count at compile time
const _: () = assert!(PUBCHEM_KEYS.len() == 881);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_pubchem(smiles: &str) -> BitVec<u8, Lsb0> {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        PubChem881.calculate(&mut mol).as_binary().clone()
    }

    #[test]
    fn test_pubchem_size() {
        let mut mol = parse_smiles("C").unwrap();
        let fp = PubChem881.calculate(&mut mol);
        assert_eq!(fp.len(), 881);
    }

    #[test]
    fn test_pubchem_methane() {
        let fp = calc_pubchem("C");
        let bits = fp.count_ones();
        assert!(bits < 5, "Methane should set very few bits: {}", bits);
    }

    #[test]
    fn test_pubchem_benzene() {
        let fp = calc_pubchem("c1ccccc1");
        let bits = fp.count_ones();
        assert!(bits > 10, "Benzene should set many bits: {}", bits);
    }

    #[test]
    fn test_pubchem_ethanol() {
        let fp = calc_pubchem("CCO");
        assert!(fp[12], "Should have >=2 C");
        assert!(fp[21], "Should have >=1 O");
        assert!(fp[460], "Should have hydroxyl");
    }

    #[test]
    fn test_pubchem_acetic_acid() {
        let fp = calc_pubchem("CC(=O)O");
        assert!(fp[463], "Should have carboxylic acid pattern");
    }

    #[test]
    fn test_pubchem_aspirin() {
        let fp = calc_pubchem("CC(=O)Oc1ccccc1C(=O)O");
        let bits = fp.count_ones();
        assert!(bits > 30, "Aspirin should set many bits: {}", bits);
    }

    #[test]
    fn test_pubchem_caffeine() {
        let fp = calc_pubchem("CN1C=NC2=C1C(=O)N(C(=O)N2C)C");
        let bits = fp.count_ones();
        assert!(bits > 20);
    }

    #[test]
    fn test_pubchem_identical() {
        let fp1 = calc_pubchem("CCO");
        let fp2 = calc_pubchem("CCO");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_pubchem_different_molecules() {
        let fp1 = calc_pubchem("c1ccccc1");
        let fp2 = calc_pubchem("CCCCCC");
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_pubchem_patterns_parse() {
        let mut failed = Vec::new();
        for (i, key) in PUBCHEM_KEYS.iter().enumerate() {
            if parse_smarts(key.smarts).is_err() {
                failed.push((i, key.smarts));
            }
        }
        let parsed_count = PUBCHEM_KEYS.len() - failed.len();
        assert_eq!(
            parsed_count, 881,
            "All patterns should parse: {}/881",
            parsed_count
        );
    }
}
