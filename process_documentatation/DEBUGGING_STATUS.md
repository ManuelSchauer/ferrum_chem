# Ferrum Chem — Debugging Status & Test Guide

## How to Run Tests

### Core Library Tests (626 unit tests)
```bash
cargo test -p ferrum_core --lib
```

### RDKit Comparison (27 torture molecules × 72 descriptors)
```bash
# All 4 test groups
cargo test -p ferrum_core --test rdkit_comparison -- --nocapture

# Individual groups
cargo test -p ferrum_core --test rdkit_comparison rdkit_scalar -- --nocapture
cargo test -p ferrum_core --test rdkit_comparison rdkit_vsa -- --nocapture
cargo test -p ferrum_core --test rdkit_comparison rdkit_chi_kappa -- --nocapture
cargo test -p ferrum_core --test rdkit_comparison rdkit_ring -- --nocapture
```

### SMARTS Edge Cases (55 tests)
```bash
cargo test -p ferrum_core --test smarts_edge -- --nocapture
```

### Core Debug Diagnostics
```bash
cargo test -p ferrum_core --test debug_core -- --nocapture
```

### Regenerate RDKit Reference Data
```bash
python3 scripts/generate_rdkit_reference.py
```

---

## Test File Locations

| File | Purpose |
|---|---|
| `crates/ferrum_core/tests/rdkit_comparison.rs` | Main RDKit comparison: 4 tests covering scalar, VSA, Chi/Kappa, ring descriptors |
| `crates/ferrum_core/tests/rdkit_reference.csv` | RDKit reference values for 27 molecules × 60+ descriptors |
| `crates/ferrum_core/tests/smarts_edge.rs` | 55 SMARTS edge case tests across 4 test functions |
| `crates/ferrum_core/tests/debug_core.rs` | Diagnostic test dumping atom-level molecule details |
| `scripts/generate_rdkit_reference.py` | Python script to regenerate reference CSV via RDKit |
| `torture_set.csv` | Input: 30 challenging molecules (27 parseable by RDKit) |

---

## Current Accuracy (27 Torture Molecules vs RDKit)

### Overall Pass Rates

| Test Group | Total | Passed | Rate |
|---|---|---|---|
| Ring Descriptors | 108 | 108 | **100.0%** |
| Chi/Kappa Indices | 270 | 270 | **100.0%** |
| VSA Descriptors | 972 | 972 | **100.0%** |
| Scalar Descriptors | 594 | 594 | **100.0%** |
| **TOTAL** | **1944** | **1944** | **100.0%** |

### ✅ All 72 Descriptors at 100%

All 72 descriptors across 27 torture molecules match RDKit exactly.

---

## Bugs Fixed (Chronological)

### Core SMILES/Aromaticity (Session 1)

| Bug | File | Fix |
|---|---|---|
| `Cn` parsed as Copernicium(Z=112) | `io/smiles.rs` | Only `Br`/`Cl` as 2-char organic subset |
| Aromatic S got 1 implicit H | `algo/valency.rs` | Rewrote with OpenSMILES aromatic valences |
| Aromatic N with 3 neighbors got 1 implicit H | `algo/valency.rs` | Added `get_aromatic_valences()` |

### Descriptor Algorithm Fixes (Session 2)

| Descriptor | Before → After | File | Fix |
|---|---|---|---|
| LabuteASA | 0% → 96.3% | `desc/vsa.rs` | Complete rewrite: Rb0 covalent radii, bond-based formula |
| HallKierAlpha | 11% → 96.3% | `desc/extra_counts.rs` | Hybridization-dependent alpha lookup table |
| BalabanJ | 11% → 100% | `desc/balaban.rs` | Dijkstra bond-order-weighted distances; return 0 for 0-bond molecules |
| TPSA | 89% → 100% | `desc/tpsa.rs` | `include_sp` false instead of true |
| HBondAcceptor/Donor | 63% → 96.3% | `desc/hbond.rs` | Lipinski SMARTS patterns instead of simple count |
| Kappa1/2/3 | 3–7% → 96.3% | `desc/kappa.rs` | Alpha correction in both numerator AND denominator; removed small-molecule guards |
| BertzCT | 15% → 96.3% | `desc/extra_counts.rs` | Complete rewrite: symmetry classes, connection counting, Shannon entropy |
| Ipc | 33% → 100% | `desc/extra_counts.rs` | Characteristic polynomial via Le Verrier-Faddeev-Frame |
| Qed | 56% → 100% | `desc/qed.rs` | Fixed weights (HBD 0.26→0.61), added 35 SMARTS structural alert groups |

### SMARTS Parser Fixes (Session 3)

| Bug | File | Fix |
|---|---|---|
| Uppercase `C`/`N`/`O`/`S` matched any aromaticity | `io/smarts.rs` | `Element(z)` → `Aliphatic(z)` for bracket + organic atoms |
| Comma/semicolon precedence wrong | `io/smarts.rs` | 3-level parser: implicit-AND > comma-OR > semicolon-AND |
| Default bond was `Single` instead of `SingleOrAromatic` | `io/smarts.rs` | All `unwrap_or(QueryBond::Single)` → `SingleOrAromatic` |
| Aromatic bond valence counted as 1 | `io/smarts.rs` | Half-integer math: aromatic = 1.5 (×2 internally) |
| `[R1]` parsed as ring-size-1 instead of ring-membership-1 | `io/smarts.rs` | Added `RingMembership(u8)` primitive; `R`→membership, `r`→size |

### LogP Fixes (Session 3)

| Bug | File | Fix |
|---|---|---|
| H classification wrong | `desc/logp.rs` | Added `classify_implicit_h()` matching RDKit C++ (H1/H2/H3/H4/HS) |
| VSA bins used total LogP instead of heavy-atom-only | `desc/logp.rs` + `desc/vsa.rs` | `crippen_atom_contributions()` returns pure heavy-atom values |

### Aromaticity & Remaining Fixes (Session 4)

| Bug | File | Fix |
|---|---|---|
| No aromaticity re-perception | `algo/aromaticity.rs` | Strip non-aromatic atoms, Kekulize boundary bonds after SSSR + Hückel check |
| N π-electron counting wrong | `algo/aromaticity.rs` | Combinatorial approach: try all 1π/2π combos for N atoms |
| Isatin 5-ring incorrectly aromatic | `algo/aromaticity.rs` | C with only exocyclic C=O contributes 0π to ring |
| Cyclopentadienyl C⁻ not aromatic | `algo/aromaticity.rs` | C⁻ always contributes 2π (check charge before ring bonds) |
| TotalRings overcounted | `desc/small_ring.rs` | Use cyclomatic number (M-N+C) instead of exhaustive ring enumeration |
| HallKierAlpha for sulfone S | `algo/electronic.rs` | Hypervalent S/P (degree≥3) → SP3 not SP2 |
| Gasteiger used C_SP3 fallback for metals | `algo/electronic.rs` | Return `None` for unknown elements (keep formal charge) |

### Cisplatin & Metal Complex Fixes (Session 5)

| Bug | File | Fix |
|---|---|---|
| Missing Pt covalent radius | `desc/vsa.rs` | Extended `rb0_radius()` from Z=53 to Z=83 (Pt=1.30Å) |
| Missing Pt outer electrons | `mol/elements.rs` | Extended `outer_electrons()` with 3rd-row transition metals (Pt=10) |
| No dative bond detection | `algo/valency.rs` | Added `assign_dative_bonds()`: Single bonds between donor (N,O,S,P) and transition metal → Dative, directional |
| Dative bonds in SMILES pipeline | `io/smiles.rs` | Added `assign_dative_bonds()` call after implicit H, before aromaticity |
| BertzCT wrong for dative bonds | `desc/extra_counts.rs` | Dative bond order = 17.0 (matching RDKit `float(BondType.DATIVE)`) |
| SMARTS valence for dative bonds | `io/smarts.rs` | Directional: 0 to source (donor), 1 to target (metal) |
| Gasteiger NaN not propagated | `algo/electronic.rs` | Per-fragment flood-fill: if any atom in component has no params + has bonds → all NaN |

### Gasteiger Charge Accuracy Fixes (Session 6)

| Bug | File | Fix |
|---|---|---|
| Missing Mg/Al/Si/B Gasteiger params | `algo/electronic.rs` | Added 11 new param sets from RDKit's `additionalParamData` (PyBabel-derived) |
| Missing P_SP2 params | `algo/electronic.rs` | Added P_SP2 (a=9.665, b=8.530, c=0.735) |
| S so/so2 params unused | `algo/electronic.rs` | Added S_SO, S_SO2; S param selection checks bonded-O count when hyb unset |
| Formal charges not split across conjugation | `algo/electronic.rs` | Implemented `split_charge_conjugated()`: same-element atoms in conjugated systems share formal charge (e.g., NO2: both O start at -0.5) |
| Charges initialized wrong | `algo/electronic.rs` | Changed from formal-charge init to 0 + `split_charge_conjugated()` + fallback |
| Wrong Mg/Al/Si/B outer shell electrons | `algo/electronic.rs` | Added to `outer_shell_electrons()`: Mg=2, Al=3, Si=4, B=3 |

---

## Key Source Files

### Core Parsing
| File | Purpose |
|---|---|
| `crates/ferrum_core/src/io/smiles.rs` | SMILES parser |
| `crates/ferrum_core/src/io/smarts.rs` | SMARTS parser + matcher (3-level bracket precedence) |
| `crates/ferrum_core/src/algo/valency.rs` | Implicit hydrogen assignment |
| `crates/ferrum_core/src/algo/aromaticity.rs` | Aromaticity perception (Hückel 4n+2 with re-perception) |
| `crates/ferrum_core/src/algo/electronic.rs` | Gasteiger charges, hybridization |
| `crates/ferrum_core/src/algo/rings.rs` | SSSR ring finding |
| `crates/ferrum_core/src/mol/elements.rs` | Element data, `get_aromatic_valences()` |

### Descriptors
| File | Descriptors |
|---|---|
| `crates/ferrum_core/src/desc/logp.rs` | LogP, MolarRefractivity, Crippen atom contributions |
| `crates/ferrum_core/src/desc/vsa.rs` | LabuteASA, SlogpVSA[0–11], SmrVSA[0–9], PeoeVSA[0–13] |
| `crates/ferrum_core/src/desc/extra_counts.rs` | FractionCSP3, NumAmideBonds, HallKierAlpha, BertzCT, Ipc |
| `crates/ferrum_core/src/desc/chi.rs` | Chi0, Chi1, Chi0v–Chi4v |
| `crates/ferrum_core/src/desc/kappa.rs` | Kappa1, Kappa2, Kappa3 |
| `crates/ferrum_core/src/desc/hbond.rs` | HBondAcceptor, HBondDonor (Lipinski SMARTS) |
| `crates/ferrum_core/src/desc/tpsa.rs` | Topological Polar Surface Area |
| `crates/ferrum_core/src/desc/qed.rs` | QED (8 properties + 35 structural alerts) |
| `crates/ferrum_core/src/desc/balaban.rs` | BalabanJ, `compute_distance_matrix()` |
| `crates/ferrum_core/src/desc/small_ring.rs` | SmallRingDescriptor (total, aromatic, size counts) |

---

## SMARTS Edge Cases Tested (55 total)

### `test_smarts_edge_cases` — 19 basic tests
Covers: `#6`, `[CH3]`, `[C;H3]`, `[OH,NH2]`, `[N;!H0]`, `[!#1]`, `$(...)` recursive, `[X3]`, `[D3]`, `[R]`, `[!R]`, `[r6]`, `[H1]`, `[+1]`, `[-1]`, `[cx2]`, `[c;r6]`, `[O,S;H1;+0]`, `[OH,OH2]`

### `test_smarts_hba_edge_cases` — 10 HBA tests
Covers: `[oH0;X2]`, `[s;a]`, `[nH0;X2]`, `[nH0,o,s;+0]` (comma-OR), `$(...)` recursive with comma-OR, `[N,O;H1;+0]` (semicolon-AND), `[R]`

### `test_smarts_bond_and_valence` — 11 bond/valence tests
Covers: `CC` vs `cc` (aliphatic vs aromatic default bond), `C-C`, `C-c`, `[cv4]`, `[cv3]`, `[cX3]`, `[!c]`, `[$([NH2+0][A;!#1])]`, `[N;!H0;v3]`

### `test_smarts_ring_bond_features` — 15 ring/bond tests
Covers: `C!@C` (not-ring bond), `C@C` (ring bond), `C~C` (any bond), `[R1]` (ring membership), `[R0]`, `[D1]`, `[D2]`, `[r5]`, `[r6]`
