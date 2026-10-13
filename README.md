# FerrumChem

**A molecular descriptor engine in Rust that reproduces RDKit, CDK, Mordred and
Biosynfoni exactly, and computes them 10–900× faster.**

FerrumChem calculates 531 scalar descriptors and 6 fingerprints (9,239 bits)
per molecule: the complete RDKit descriptor catalogue, the CDK and Mordred
descriptors that are commonly used in QSAR, the Biosynfoni natural-product
fingerprint and a few extras. It is available as a Rust crate
(`ferrum_core`), a command-line tool with an interactive terminal UI
(`ferrum_cli`) and Python bindings (`ferrum_py`).

There are no options for aromaticity models, hydrogen handling or charge
schemes: the chemistry is preconfigured to match the reference toolkits, so
the same molecule always gives the same numbers as RDKit/CDK/Mordred would.

---

## Why FerrumChem

- **Exact, not approximate.** 512 of the 531 scalar columns and all 6
  fingerprints are compared molecule by molecule against the original
  implementation of RDKit 2026.03.2, CDK 2.13, Mordred 1.2.0 and Biosynfoni
  (`|a − b| ≤ 1e-9 + 1e-9·|b|`; fingerprints and atom perception bit-exact).
  The tests fail when any column gets worse.
- **Fast.** On one core: SMILES parsing 4× faster than RDKit, the complete
  RDKit descriptor catalogue 25–28× faster, the five RDKit fingerprints
  7–10× faster, all tested CDK descriptors ~900× faster than CDK. On all cores
  the CLI processes about 14,000 molecules per second with the full bundle.
- **Constant memory.** Input is streamed in chunks; 100,000 molecules need
  about 400 MB peak, the same order as 10,000.
- **One chemistry model.** A port of RDKit's `SanitizeMol` (valence model,
  Kekulé assignment, aromaticity, conjugation, hybridization, ring
  perception) is used everywhere. Where CDK descriptors depend on CDK's own
  perception (atom types, cdkLegacy/Daylight aromaticity, explicit
  hydrogens), that view is rebuilt locally inside those descriptors only.
- **ML-ready output.** All scalars are `f64` end to end; fingerprints are
  bit-packed (`u64` words, NumPy `packbits(bitorder="little")` layout); invalid
  input gives NaN rows plus a validity mask, so rows always line up with the
  input.
- **Thread-safe by construction.** Descriptors are stateless `Send + Sync`
  types; parallelism is built in (Rayon) and every core is used without
  locks.

## At a glance

| | |
|---|---|
| scalar descriptors | 531 (88 descriptor families) |
| fingerprints | MACCS (166), PubChem (881), ECFP4, FCFP4, AtomPair, TopologicalTorsion (2,048 each) |
| validated against a reference | 512 scalar columns + all 6 fingerprints + per-atom perception |
| references | RDKit 2026.03.2 (2D + 3D), CDK 2.13, Mordred 1.2.0, Biosynfoni `full_1103` |
| input | SMILES (`.smi`), CSV with a SMILES column, SDF (V2000/V3000, incl. 3D conformers) |
| output | CSV (exact round-trip `f64`, fingerprints as hex / on-bits / 0-1 columns), NumPy arrays, Rust `DescriptorTable` |
| throughput (Apple M4 Pro, 14 cores) | ~14,000 molecules/s, full bundle, end to end |

---

## Quick tutorial

### Build

```bash
cargo build --release -p ferrum_cli
```

### Command line

```bash
# all descriptors for a SMILES, CSV or SDF file
target/release/ferrum_cli calc -i molecules.smi -o descriptors.csv

# fingerprints as on-bit lists instead of hex, smaller chunks for less memory
target/release/ferrum_cli calc -i molecules.sdf -o descriptors.csv --fingerprints onbits --chunk-size 1024

# interactive terminal UI: load a dataset, pick descriptors, preview, export
target/release/ferrum_cli tui
```

`--fingerprints` takes `hex` (default; one column per fingerprint), `onbits`,
`bits` (one 0/1 column per bit) or `none`. Every input record gives one output
row; records that cannot be parsed have empty cells.

### Python

```bash
cd crates/ferrum_py && maturin develop --release
```

```python
import numpy as np
import ferrum_py

smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "not a smiles"]

calc = ferrum_py.Calculator()                  # build once, reuse for every call
names, X, fps, valid = calc.compute(smiles)
# names: 531 column names; X: float64 (3, 531), NaN rows where valid is False
# fps:   {"MACCS": uint8 (3, 21), "PubChem": (3, 111), "ECFP4": (3, 256), ...}

bits = calc.fingerprint_bits                   # {"MACCS": 166, "ECFP4": 2048, ...}
ecfp4 = np.unpackbits(fps["ECFP4"], axis=1, bitorder="little", count=bits["ECFP4"])

scalars = ferrum_py.Calculator(fingerprints=False)                 # scalars only
names, X, fps, valid = ferrum_py.compute_descriptors(smiles)       # same, shared module-level calculator
bio_names, bio = ferrum_py.compute_biosynfoni(smiles)              # 39 counts
```

The calculation runs in parallel and releases the GIL; one `Calculator` can be
used from several Python threads at the same time.

### Rust

```rust
use ferrum_core::desc::calculator::DescriptorCalculator;
use ferrum_core::io::smiles::parse_smiles;

// one calculator for all molecules and all threads (Send + Sync)
let calc = DescriptorCalculator::all(); // or ::new(&[BatchDescriptor::MolecularWeight, ...]), ::scalars_only()

let table = calc.calculate_smiles(&["c1ccccc1O", "CCN"]); // parallel
let logp = table.scalar_row(0)[table.scalar_names.iter().position(|n| n == "MolLogP").unwrap()];
let ecfp4_bits: Vec<usize> = table.fingerprint("ECFP4").unwrap().on_bits(0).collect();

// molecules are only read, so they can be reused or shared between threads
let mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O").unwrap();
let one = calc.calculate_one(&mol);
```

`calculate` accepts slices of `Molecule`, `&Molecule`, `Option<Molecule>` or
`Option<&Molecule>` (`None` gives an invalid row). For large files,
`ferrum_core::io::stream::MoleculeReader` reads SMILES, CSV and SDF in chunks,
and `desc::table::{write_csv_header, write_csv_rows}` write the CSV format used
by the CLI.

---

## Descriptor catalogue

531 scalar columns in 88 families, plus 6 fingerprints. "Reference" is the
implementation each column is tested against; column names follow that
reference.

### RDKit (321 scalar columns + 5 fingerprints)

| family | columns |
|---|---|
| Molecular weight | `MolWt`, `HeavyAtomMolWt`, `ExactMolWt` |
| Crippen | `MolLogP`, `MolMR` |
| Polar surface / H-bonding | `TPSA`, `NumHDonors`, `NumHAcceptors`, `NHOHCount`, `NOCount`, `LipinskiViolations` (MolWt > 500, MolLogP > 5, HBD > 5, HBA > 10) |
| Counts | `HeavyAtomCount`, `NumHeteroatoms`, `NumRotatableBonds`, `NumValenceElectrons`, `NumRadicalElectrons`, `FormalCharge`, `NumAmideBonds`, `FractionCSP3`, `AromaticProportion`, element counts `nC`, `nN`, `nO`, `nS`, `nP`, `nF`, `nCl`, `nBr`, `nI`, `nH`, … (12), atom and bond counts (`nAtom`, `nAromAtom`, `nBonds`, `nBondsS/D/T/A`, …) |
| Rings | `RingCount`, `NumAromaticRings`, `NumSaturatedRings`, `NumAliphaticRings`, `NumAromatic/Aliphatic/Saturated Carbo-/Heterocycles`, `NumHeterocycles`, `NumSpiroAtoms`, `NumBridgeheadAtoms` |
| Topology | `BalabanJ`, `BertzCT`, `Ipc`, `AvgIpc`, `Phi`, `HallKierAlpha`, `Kappa1–3`, `Chi0`, `Chi1`, `Chi0v–Chi4v`, `Chi0n–Chi4n` |
| E-state | `SumEState`, `MinEStateIndex`, `MaxEStateIndex`, `MaxAbsEStateIndex`, `MinAbsEStateIndex`, 79 E-state atom-type sums `SEstate_*` (RDKit `EState.Fingerprinter`) |
| Surface area / VSA | `LabuteASA`, `SlogP_VSA1–12`, `SMR_VSA1–10`, `PEOE_VSA1–14`, `EState_VSA1–11`, `VSA_EState1–10` |
| Partial charges (Gasteiger) | `MaxPartialCharge`, `MinPartialCharge`, `MaxAbsPartialCharge`, `MinAbsPartialCharge` |
| Drug-likeness | `qed` |
| Fragments | all 85 `fr_*` counts of `rdkit.Chem.Fragments` |
| BCUT2D | `BCUT2D_MWHI/LOW`, `BCUT2D_CHGHI/LO`, `BCUT2D_LOGPHI/LOW`, `BCUT2D_MRHI/LOW` |
| Morgan density | `FpDensityMorgan1–3` |
| Stereo | `NumAtomStereoCenters`, `NumUnspecifiedAtomStereoCenters`, `SPS` |
| 3D (needs a conformer, e.g. from SDF; NaN for SMILES) | `PMI1`, `PMI2`, `PMI3`, `Asphericity` |
| Fingerprints | MACCS keys (166 bits), ECFP4 and FCFP4 (Morgan radius 2, 2,048 bits), AtomPair (2,048), TopologicalTorsion (2,048) |

This covers every entry of `rdkit.Chem.Descriptors._descList` (217 of 217).

### CDK (123 scalar columns + PubChem fingerprint)

| family | columns |
|---|---|
| Chi indices | `SP-0..7`, `VP-0..7`, `SC-3..6`, `VC-3..6`, `SPC-4..6`, `VPC-4..6`, `SCH-3..7`, `VCH-3..7` (40) |
| Autocorrelation | `ATSm1–5` (mass), `ATSp1–5` (polarizability), `ATSc1–5` (charge) |
| BCUT | `BCUTw-1l/h`, `BCUTc-1l/h`, `BCUTp-1l/h` |
| Molecular distance edge | `MDEC-11` … `MDEN-33` (19) |
| Weighted path | `WTPT-1..5` |
| Carbon types / hybridization | `C1SP1` … `C4SP3` (9), `HybRatio` |
| Topology | `WienerPath`, `WienerPolarity`, `Zagreb1`, `PetitjeanNumber`, `ECCEN`, `VAdjMat`, `fragC`, `Kier1–3` |
| Rings and chains | `nSmallRings`, `nAromRings`, `nRingBlocks`, `nAromBlocks`, `nRings3–9`, `nAtomLC`, `nAtomLAC`, `nAtomP` |
| Polarizability | `apol`, `bpol` |
| Acid/base groups | `nAcidicGroups`, `nBasicGroups` |
| Fingerprint | PubChem (881 bits, CDK `PubchemFingerprinter`) |

### Mordred (25 columns)

`Radius`, `Diameter`, `Zagreb2`, `ATS0pe–ATS8pe` (Pauling electronegativity),
`ATS0v–ATS8v` (van der Waals volume), `BCUTpe-1l/h`, `BCUTv-1l/h`.

### Biosynfoni (39 columns)

The `full_1103` substructure counts of the Biosynfoni natural-product
fingerprint (`Biosynfoni_*`: coenzymes, amino acids, sugars, biosynthetic
building blocks, halogens, ring sizes), with the default overlap settings.

### FerrumChem only (19 columns, unit-tested, no external reference)

`ElementEntropy`, `LargestPiSystemSize`, `nPiSystems` and 16 functional-group
counts `fg_Hydroxyl`, `fg_Aldehyde`, `fg_Ketone`, `fg_CarboxylicAcid`,
`fg_Ester`, `fg_Amide`, `fg_PrimaryAmine`, `fg_SecondaryAmine`,
`fg_TertiaryAmine`, `fg_Nitro`, `fg_Sulfonamide`, `fg_Sulfone`, `fg_Thiol`,
`fg_Ether`, `fg_Nitrile`, `fg_Halide`.

---

## Parity with the reference toolkits

Each reference has a CSV generated by the original software from the same
1,277 test molecules (drugs, natural products, metals, isotopes, radicals,
charged and unusual valence states), and an integration test that recomputes
everything with FerrumChem:

| test | reference | molecules | columns that match on every molecule |
|---|---|---|---|
| `rdkit_parity` | RDKit 2026.03.2 | 1,277 | 325 of 327 (321 scalars, 5 fingerprints, atom perception) |
| `rdkit_3d_parity` | RDKit 2026.03.2, ETKDG conformers | 296 | 4 of 5 (PMI1–3, Asphericity; perception see below) |
| `cdk_parity` | CDK 2.13 | 1,277 | 124 of 124 |
| `mordred_parity` | Mordred 1.2.0 | 1,277 | 25 of 25 |
| `biosynfoni_parity` | Biosynfoni (original Python package) | 1,277 | 39 of 39 |

Known deviations, tracked per column in `tests/data/*_baseline.tsv` (a column
may never get worse):

- **`Ipc`, `AvgIpc` (31 of 1,277 molecules):** RDKit evaluates the characteristic
  polynomial with a float64 recurrence that is numerically unstable for large,
  flexible molecules; its result depends on BLAS rounding (e.g. 2.2e24 for a
  lipid whose exact value is 3.7e21). FerrumChem uses the same recurrence and
  matches wherever RDKit's value is stable.
- **3D perception of `N[Pt](N)(Cl)Cl`:** RDKit's mol-block reader marks Pt as
  chiral from the 3D coordinates and then reports SP3 instead of unspecified
  hybridization. The 3D descriptors themselves match.

Beyond the test set:

- **Atom perception:** 715,790 of 715,822 COCONUT natural products identical to
  RDKit (aromaticity, H counts, hybridization, radicals, rings).
- **Stereo descriptors:** identical on all of COCONUT except 3 SPS values.
- **Biosynfoni:** identical on 20,000 COCONUT molecules.
- **RDKit, 5,000 mixed molecules:** identical except Ipc/AvgIpc (as above).

All descriptors that the CDK computes on its own perception (atom typing,
Kekulé structure from its SMILES parser, cdkLegacy or Daylight aromaticity,
explicit hydrogens, neutral nitro groups) reconstruct that view locally, so
the shared RDKit perception is never changed for them.

---

## Performance

Apple M4 Pro, 14 cores; all numbers for identical results (see
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) for methodology, datasets and
more tables).

| single thread, µs per molecule | RDKit | FerrumChem | speed-up |
|---|---|---|---|
| parse SMILES + sanitize | 52–79 | 13–18 | 4× |
| full RDKit catalogue (217 descriptors) | 3,874–6,605 | 152–240 | 25–28× |
| 5 fingerprints | 658–1,119 | 89–114 | 7–10× |
| FerrumChem full bundle (531 scalars + 6 fingerprints) | – | 399–627 | – |

| | reference | FerrumChem |
|---|---|---|
| all CDK descriptors, 1,273 molecules, 1 thread | CDK 147.5 s | 0.165 s |
| all cores, catalogue + fingerprints, 5,000 molecules | RDKit (multiprocessing) 744 mol/s | 7,629 mol/s with the full bundle |
| CLI, 100,000 molecules, all cores | – | 7.0 s, 14,203 mol/s, 406 MB peak RSS |
| CLI, 10,000 molecules, 1 thread | – | 6.8 s, 148 MB peak RSS |

Peak memory depends on the largest molecules and the number of threads, not
on the input size. The most expensive families are now PubChem (69 µs/mol),
MACCS (67) and the CDK Chi indices (53); the CDK weighted-path descriptor,
whose cost grows exponentially with fused rings in CDK, is computed exactly
with a memoised search (a 30-ring ladder polyether: 40 s → < 10 ms).

Reproduce with `python scripts/benchmark.py MOLECULES.smi` and
`cargo run --release -p ferrum_core --example descriptor_timing -- MOLECULES.smi`.

---

## Output format

**Scalars** are `f64`. Counts are exact; values that cannot be computed (e.g.
3D descriptors without a conformer, or where the reference reports an error)
are NaN. In CSV, values are written in their shortest exact round-trip form,
integers without a fraction, and rows of unparseable input are empty.

**Fingerprints** are bit-packed per row: bit `i` is bit `i % 64` of `u64` word
`i / 64`; as bytes, bit `i % 8` of byte `i / 8`. This is NumPy's
`packbits(bitorder="little")` layout, so
`np.unpackbits(a, axis=1, bitorder="little", count=n_bits)` restores the bit
matrix. In CSV, `hex` writes these bytes as one hex string per fingerprint.

**`DescriptorTable`** (Rust) holds `scalar_names`, the row-major `scalars`, one
`FingerprintBlock` per fingerprint (`row`, `get`, `on_bits`, `count_ones`,
`to_bytes`) and the `valid` mask.

---

## Thread safety and architecture

- **One calculator for everything.** `DescriptorCalculator` (Rust) /
  `ferrum_py.Calculator` (Python) is built once, is immutable afterwards and
  `Send + Sync`; the same object computes every molecule on every thread. Its
  constructor also initialises all compiled SMARTS patterns and data tables,
  so the first molecule does not pay for them.
- **Descriptors are stateless.** Every descriptor is a type without mutable
  state (at most immutable settings such as a fingerprint's radius and bit
  count) or a value of the `BatchDescriptor` enum, and implements
  `Descriptor: Send + Sync`, checked at compile time. Precompiled patterns
  and tables live in `OnceLock` statics. The library uses no locks and no
  `static mut`; the only shared mutable value is an atomic id counter for
  compiled SMARTS patterns.
- **Molecules are read-only during calculation** (`&Molecule`). Results that
  several descriptors need (ring information, Gasteiger and Crippen
  contributions, TPSA, H-bond counts, …) are cached inside the molecule in
  `OnceLock` fields: computed once by the first descriptor that needs them,
  thread-safely, and read-only afterwards. A molecule can therefore be reused
  or shared between threads (e.g. in an `Arc`).
- **Scratch space is per thread.** Each worker thread owns one
  `CalculationContext` (reusable buffers and per-molecule intermediates such as
  distance matrices, the characteristic polynomial, path tables, stereo
  perception and the CDK atom-type view). The calculator resets it before
  every molecule, so results never depend on the previous molecule.
- **Parallelism** comes from Rayon. The CLI streams its input in chunks and
  keeps several chunks in flight, so a single slow molecule does not stall the
  other cores; the TUI streams the same way and keeps only a preview in memory.

Repository layout:

```
crates/ferrum_core/   library
  src/io/             SMILES, SMARTS, SDF (V2000/V3000), CSV, chunked readers
  src/mol/            molecule graph, element and isotope data (generated from RDKit)
  src/algo/           sanitize (RDKit port), SSSR, stereo/CIP, SMARTS matcher,
                      distance matrices, paths, eigenvalues
  src/desc/           descriptor families, batch registry, table output
  tests/              parity tests and reference data
  examples/           descriptor_timing (per-family benchmark)
crates/ferrum_cli/    `calc` command and terminal UI
crates/ferrum_py/     Python bindings (PyO3 + NumPy)
scripts/              reference generators, data generators, benchmarks
docs/PERFORMANCE.md   benchmark methodology and results
```

---

## Development

```bash
# all tests, including every parity test
cargo test -p ferrum_core --release

# details of deviating columns / tighten a baseline after a fix
FERRUM_PARITY_VERBOSE=1 cargo test -p ferrum_core --release --test rdkit_parity -- --nocapture
FERRUM_UPDATE_PARITY_BASELINE=1 cargo test -p ferrum_core --release --test cdk_parity
```

Regenerating the references (only needed after upgrading a reference toolkit):

| reference | command |
|---|---|
| RDKit (2D + 3D) | `python scripts/generate_parity_reference.py` (`--check` verifies, `--embed-3d` re-embeds the 3D set) |
| CDK | `bash scripts/cdk_reference/run.sh` (`CDK_JAR` points to a CDK bundle jar; also prints CDK timings) |
| Mordred | `MORDRED_SRC=~/mordred python scripts/generate_mordred_reference.py` |
| Biosynfoni | `BIOSYNFONI_SRC=~/biosynfoni/src python scripts/generate_biosynfoni_reference.py` |
| data tables | `scripts/gen_element_data.py`, `gen_fragment_data.py`, `gen_mordred_data.py` |

CI (`.github/workflows/ci.yml`) runs the tests in release mode, verifies the
committed RDKit reference against `rdkit==2026.3.2`, and smoke-tests the Python
bindings.

## License

No license has been chosen yet.
