# Performance

Measured on an Apple M4 Pro (14 cores: 10 performance + 4 efficiency), macOS
(Darwin 27), Rust 1.93.1 (`--release`), RDKit 2026.03.2 (Python), CDK 2.13 on
Java 21. Every FerrumChem value in these runs is checked against the reference
implementations by the parity tests (`|a - b| <= 1e-9 + 1e-9·|b|`, fingerprints
bit-exact), so the speed-ups compare identical results.

## Summary

| | Reference | FerrumChem | speed-up |
|---|---|---|---|
| parse SMILES + sanitize, 1 thread | RDKit 52–79 µs/mol | 13–18 µs/mol | **4.0–4.4×** |
| full RDKit descriptor catalogue (217 descriptors), 1 thread | RDKit 3.9–6.6 ms/mol | 152–240 µs/mol | **25–28×** |
| 5 fingerprints (MACCS, ECFP4, FCFP4, AtomPair, TopoTorsion), 1 thread | RDKit 0.66–1.12 ms/mol | 89–114 µs/mol | **7–10×** |
| all CDK descriptors and fingerprints (243 columns, 9 fingerprints, 2 count fingerprints), 1,277 molecules, 1 thread | CDK 232 s | 0.58 s | **~400×** |
| catalogue + fingerprints, all 14 cores | RDKit 744 mol/s | ~6,400 mol/s with the *full* bundle | **8.6×** |
| CLI, 100,000 molecules, full bundle, all cores | – | 17.0 s (5,870 mol/s), 880 MB peak RSS | – |

The FerrumChem full bundle computes 2,157 scalar columns, 2 count fingerprints
(5,167 counts) and 14 bit fingerprints (18,503 bits): the complete RDKit
catalogue, all CDK 2.13 2D descriptors and fingerprints PaDEL uses, all 1,613
Mordred 2D descriptors, Biosynfoni and a few FerrumChem-only descriptors. It
takes 1,437 µs/mol on one core for the mixed set (882 µs/mol for the parity
set). History: 531 columns 634 / 399 µs/mol; with all Mordred descriptors
(2,037 columns) 1,058 / 616 µs/mol; the CDK additions (Klekota-Roth counts
alone 174 µs/mol) bring it to the current value.

## Datasets

- **parity set**: `crates/ferrum_core/tests/data/parity_molecules.smi`
  (1,273 molecules; drugs, natural products and edge cases).
- **mixed set**: a 100,000-molecule set (NCI, RDKit `canonSmiles.long`, a
  seeded COCONUT sample; large natural products with up to 1,500 SMILES
  characters). The 5,000-molecule tables use every 20th molecule of it. It is
  not committed; any large SMILES file can be used with the commands below.

## Single thread vs RDKit

µs per molecule. RDKit timings use fresh molecule copies (no cached Crippen /
Gasteiger contributions). The FerrumChem catalogue row runs every descriptor
family that produces one of RDKit's 217 columns (some of these families compute
extra columns as well).

| group | RDKit (mixed) | FerrumChem (mixed) | speed-up | RDKit (parity) | FerrumChem (parity) | speed-up |
|---|---|---|---|---|---|---|
| parse SMILES (+ sanitize) | 79 | 18 | 4.4× | 52 | 13 | 4.0× |
| RDKit descriptor catalogue (217/217) | 6,605 | 240 | 27.5× | 3,874 | 152 | 25.5× |
| 5 fingerprints | 1,119 | 114 | 9.8× | 658 | 89 | 7.4× |
| FerrumChem full bundle (7,324 numeric columns + 14 fingerprints) | – | 1,437 | – | – | 882 | – |

Selected descriptors (first 1,500 molecules of the mixed 5k sample, µs/mol,
each group run on its own):

| descriptor | RDKit / original | FerrumChem | speed-up |
|---|---|---|---|
| MACCS keys | 694 | 68 | 10× |
| QED | 713 | 32 | 22× |
| 85 `fr_*` fragment counts | 265 | 24 | 11× |
| Crippen LogP + MR | 198 | 13 | 15× |
| Biosynfoni (original Python package) | 418 | 37 | 11× |

## All cores

RDKit has no parallel descriptor API; it runs in a `multiprocessing` pool with
one process per core (catalogue + 5 fingerprints + parsing). FerrumChem runs
`ferrum_cli calc` end to end with the full bundle, including reading and CSV
writing.

| | RDKit (mixed, 5k) | FerrumChem (mixed, 5k) |
|---|---|---|
| wall time | 6.72 s | 0.66 s |
| molecules/s | 744 | 7,629 |

| `ferrum_cli calc`, mixed set, full bundle | wall time | molecules/s | peak RSS |
|---|---|---|---|
| 10,000 molecules, 1 thread | 15.9 s | 628 | 247 MB (156 MB with `--count-fingerprints sparse`) |
| 10,000 molecules, 14 cores | 1.60 s | 6,247 | 660 MB |
| 100,000 molecules, 14 cores | 17.0 s | 5,872 | 882 MB (730 MB sparse) |

(The parity-set run on all cores, 0.40 s for 1,273 molecules, is dominated by
process start-up.)

## CDK

`scripts/cdk_reference/run.sh` times every CDK descriptor while generating the
CDK reference (one pass over the parity set, JIT warm-up included). FerrumChem
runs the same CDK descriptor families and fingerprints single-threaded on the
same 1,277 molecules (its BCUT and autocorrelation families additionally
compute the Mordred columns).

| | CDK 2.13 | FerrumChem |
|---|---|---|
| Chi indices (path, cluster, path-cluster, chain; 40 columns) | 195.6 s | 0.044 s |
| Klekota-Roth fingerprint + counts (4,860 SMARTS keys) | 25.3 s | 0.187 s |
| PubChem fingerprint | 2.21 s | 0.064 s |
| CDK substructure fingerprint + counts (307 keys) | 1.45 s | 0.063 s |
| BCUT | 0.67 s | 0.058 s |
| WeightedPath (WTPT) | 0.81 s | 0.017 s |
| all other CDK descriptors and fingerprints | 5.6 s | 0.15 s |
| **total** | **231.6 s** | **0.58 s** |

## Streaming and memory

The CLI reads its input in chunks (`--chunk-size`, default 1,024 records),
computes several chunks concurrently, formats the CSV rows in the worker pool
and writes them in input order, so memory does not grow with the input length.
The number of records in flight is bounded by a budget of 4 million output
values, so wider rows mean fewer records in flight. The CLI uses mimalloc as
its allocator: with the macOS system allocator the peak RSS of a
single-threaded 10,000-molecule run was 442 MB instead of 135 MB, because freed
chunk buffers were not returned.

| input | peak RSS, count columns (default) | peak RSS, `--count-fingerprints sparse` |
|---|---|---|
| 10,000 molecules, 1 thread | 247 MB | 156 MB |
| 10,000 molecules, 14 cores | 660 MB | – |
| 100,000 molecules, 14 cores | 882 MB | 730 MB |

Peak memory depends on the largest molecules and the number of threads
working at once, not on the file size. CSV size for 100,000 molecules with the
full bundle: 4.0 GB with hex-packed fingerprints and count columns (default),
3.0 GB with `--count-fingerprints sparse`.

## Cost per descriptor family

Single thread, mixed set (5,000 molecules), µs per molecule, the 15 most
expensive families of the full bundle:

| family | µs/mol |
|---|---|
| MordredMatrix (11 full eigen-decompositions, detour matrix) | 212 |
| KlekotaRothCount (4,860 CDK SMARTS keys, unique matches) | 174 |
| BCUT (CDK + 24 Mordred: 30 eigenvalue problems) | 78 |
| PubChem881 | 73 |
| MaccsKeys | 70 |
| CDKSubstructureCount (307 CDK SMARTS keys) | 66 |
| CdkChiIndices | 57 |
| MordredMolecularId | 53 |
| Biosynfoni | 42 |
| CdkMaccs | 40 |
| MordredInformationContent | 38 |
| WeightedPath | 30 |
| QED | 30 |
| Fragments | 28 |
| SPS (pays the shared stereo perception) | 27 |

The CDK bit fingerprints `CDKSubstructure` and `KlekotaRoth` are read from the
cached counts (1 and 6 µs/mol).

When several descriptors share an intermediate result (distance matrices,
characteristic polynomial, path tables, stereo perception, Crippen and
Gasteiger contributions, the CDK atom-type view), the first descriptor that
needs it pays for it and the others read the per-molecule cache.

## What made it fast

| change | effect (single thread, mixed set) |
|---|---|
| SMARTS matcher rewrite: per-pattern compiled atom/bond checks and screens, per-molecule shared neighbour/element/ring tables, rarest-atom-first ordering where the result is order-independent, lazily memoised recursive SMARTS, allocation-free unique-match counting | MACCS 709 → 68, Biosynfoni 282 → 41, QED 147 → 29, Fragments 133 → 27, LogP 50 → 14 µs/mol |
| extreme eigenvalues only (tridiagonalisation + Laguerre with Sturm-count brackets), two matrices per SIMD pass, Burden matrices built into reused buffers, CDK atom-type view shared by BCUT/ATS/APol/BPol | BCUT2D 362 → 29, BCUT 359 → 45 µs/mol |
| shared characteristic polynomial (Ipc/AvgIpc), exact integer bond-order distances (BalabanJ/BertzCT), shared path table (Chi/ChiN/Kappa), shared stereo perception (SPS/StereoCenters), single-pass CDK Chi subgraph enumeration | 605 → 167 µs/mol for these descriptors |
| WTPT: pendant trees folded, degree-2 chains contracted, memoised search over reachable core regions for > 8 rings | ladder polyether 40.4 s → < 10 ms; average 804 → 29 µs/mol |
| CLI: chunked streaming with several chunks in flight, CSV formatting in the worker pool (`ryu`/`itoa`) | 100,000 molecules: 81 s → 7 s |
| BCUT solver: 4 matrices per SIMD pass with fused passes; band reduction (reverse Cuthill–McKee + Schwarz) for large sparse Burden matrices | BCUT 104 → 78, BCUT2D 28 → 20 µs/mol; a 344-atom peptide 32.7 → 5.7 ms |
| Mordred MolecularId / InformationContent / Chi / PathCount: per-block branch lists, single-neighbour atoms folded into their neighbour, bitset subgraph search, paths of pendant atoms derived from their parent | 192 → 129 µs/mol for the four |
| Mordred matrices: dense spectra in SIMD lanes, 8 Barysz weightings in one Floyd–Warshall, detour search with dominance memo and tighter bounds | 351 → 207 µs/mol; worst molecule on 100k 1.33 s → 0.19 s |

Build options were measured too: `lto = "fat"` + `codegen-units = 1` and
`-C target-cpu=native` gave no measurable gain, so the default release profile
is used. The workload is dominated by graph traversal and pattern matching,
where explicit SIMD does not help; the dense eigenvalue kernel uses NEON with
a portable fallback.

## Reproducing

```bash
# FerrumChem vs RDKit, single thread per group and all cores
python scripts/benchmark.py MOLECULES.smi --n 5000 --markdown result.md

# FerrumChem cost per descriptor family, slowest single calculations
cargo run --release -p ferrum_core --example descriptor_timing -- MOLECULES.smi
FERRUM_ONLY=WeightedPath,MaccsKeys cargo run --release -p ferrum_core --example descriptor_timing -- MOLECULES.smi

# CDK timings (printed while regenerating the CDK reference)
bash scripts/cdk_reference/run.sh

# end to end with peak memory (macOS: /usr/bin/time -l, Linux: /usr/bin/time -v)
/usr/bin/time -l target/release/ferrum_cli calc -i MOLECULES.smi -o out.csv
```
