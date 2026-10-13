#!/usr/bin/env python3
"""Speed benchmark: FerrumChem vs RDKit on the same molecules and descriptors.

Single-thread comparison per descriptor group (FerrumChem timings come from
`cargo run --release -p ferrum_core --example descriptor_timing`), plus an
all-cores comparison: RDKit with a multiprocessing pool vs the FerrumChem CLI
(`ferrum_cli calc`, end to end incl. CSV writing).

    python scripts/benchmark.py MOLECULES.smi [--n 5000] [--markdown out.md]

Groups:
- parse:       SMILES -> sanitized molecule
- catalogue:   RDKit `Descriptors._descList` (all 217 descriptors) vs every
               FerrumChem descriptor family that produces one of those columns
- fingerprints: MACCS, ECFP4, FCFP4, AtomPair, TopologicalTorsion (2048 bit)
- full bundle: FerrumChem's complete default bundle (RDKit + CDK + extras)
"""
import argparse
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors as D
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as FG

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
FP_NAMES = ["MaccsKeys", "ECFP4", "FCFP4", "AtomPair", "TopologicalTorsion"]


def rdkit_fingerprints(m, gens):
    MACCSkeys.GenMACCSKeys(m)
    for g in gens:
        g.GetFingerprint(m)


def make_gens():
    return [
        FG.GetMorganGenerator(radius=2, fpSize=2048),
        FG.GetMorganGenerator(radius=2, fpSize=2048,
                              atomInvariantsGenerator=FG.GetMorganFeatureAtomInvGen()),
        FG.GetAtomPairGenerator(fpSize=2048),
        FG.GetTopologicalTorsionGenerator(fpSize=2048),
    ]


def rdkit_catalogue(m):
    for _, fn in D._descList:
        try:
            fn(m)
        except Exception:
            pass


def rdkit_single(smiles):
    out = {}
    t = time.perf_counter()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    out["parse"] = time.perf_counter() - t
    mols = [m for m in mols if m is not None]
    for name, fn in [("catalogue", rdkit_catalogue),
                     ("fingerprints", lambda m, g=make_gens(): rdkit_fingerprints(m, g))]:
        fresh = [Chem.Mol(m) for m in mols]  # no cached charges/Crippen contributions
        t = time.perf_counter()
        for m in fresh:
            fn(m)
        out[name] = time.perf_counter() - t
    return {k: v / len(smiles) * 1e6 for k, v in out.items()}  # us/mol


def _rdkit_worker(chunk):
    gens = make_gens()
    for s in chunk:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            rdkit_catalogue(m)
            rdkit_fingerprints(m, gens)
    return len(chunk)


def rdkit_all_cores(smiles, procs):
    chunks = [smiles[i:i + 64] for i in range(0, len(smiles), 64)]
    t = time.perf_counter()
    with mp.Pool(procs) as pool:
        sum(pool.imap_unordered(_rdkit_worker, chunks))
    return time.perf_counter() - t


def ferrum_timing(smi_file, only=None):
    env = dict(os.environ, FERRUM_TSV="1")
    if only:
        env["FERRUM_ONLY"] = ",".join(only)
    out = subprocess.run(
        ["cargo", "run", "-q", "--release", "-p", "ferrum_core", "--example", "descriptor_timing", "--", smi_file],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
    rows = dict(line.split("\t") for line in out.strip().splitlines())
    parse = float(rows.pop("parse"))
    rows.pop("molecules")
    return parse, {k: float(v) for k, v in rows.items()}


def ferrum_families():
    out = subprocess.run(
        ["cargo", "run", "-q", "--release", "-p", "ferrum_core", "--example", "descriptor_timing", "--", "-"],
        cwd=ROOT, env=dict(os.environ, FERRUM_LIST="1"), capture_output=True, text=True, check=True).stdout
    return {name: feats.split(",") for name, feats in (l.split("\t") for l in out.strip().splitlines())}


def ferrum_cli_all_cores(smi_file):
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "ferrum_cli"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory() as d:
        t = time.perf_counter()
        subprocess.run([str(ROOT / "target/release/ferrum_cli"), "calc", "-i", smi_file, "-o", f"{d}/out.csv"],
                       check=True, capture_output=True)
        return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("smiles")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--markdown")
    args = ap.parse_args()

    smiles = []
    for line in open(args.smiles):
        if line.strip() and not line.startswith("#"):
            s = line.split()[0]
            if Chem.MolFromSmiles(s) is not None:
                smiles.append(s)
        if len(smiles) == args.n:
            break
    tmp = tempfile.NamedTemporaryFile("w", suffix=".smi", delete=False)
    tmp.write("\n".join(smiles) + "\n")
    tmp.close()
    n = len(smiles)
    procs = os.cpu_count()

    families = ferrum_families()
    catalogue_names = {name for name, _ in D._descList}
    catalogue = [f for f, feats in families.items() if catalogue_names & set(feats)]
    covered = catalogue_names & {x for f in catalogue for x in families[f]}

    print(f"{n} molecules, {procs} cores; RDKit {rdkit.__version__}", file=sys.stderr)
    rd = rdkit_single(smiles)
    f_parse, f_cat = ferrum_timing(tmp.name, catalogue)
    _, f_fp = ferrum_timing(tmp.name, FP_NAMES)
    _, f_all = ferrum_timing(tmp.name)
    rd_mp = rdkit_all_cores(smiles, procs)
    fe_mp = ferrum_cli_all_cores(tmp.name)
    os.unlink(tmp.name)

    def row(name, r, f):
        return f"| {name} | {r:,.0f} | {f:,.0f} | {r / f:.1f}x |" if r else f"| {name} | – | {f:,.0f} | – |"

    lines = [
        f"Dataset: {n} molecules from `{Path(args.smiles).name}`; {platform.machine()}, {procs} cores, "
        f"{platform.system()} {platform.release()}; RDKit {rdkit.__version__}.",
        "",
        "Single thread, µs per molecule:",
        "",
        "| group | RDKit | FerrumChem | speed-up |",
        "|---|---|---|---|",
        row("parse SMILES (+ sanitize)", rd["parse"], f_parse),
        row(f"RDKit descriptor catalogue ({len(covered)}/{len(catalogue_names)} columns)",
            rd["catalogue"], sum(f_cat.values())),
        row("5 fingerprints (MACCS, ECFP4, FCFP4, AtomPair, TopoTorsion)", rd["fingerprints"], sum(f_fp.values())),
        row("FerrumChem full bundle (RDKit + CDK + extras)", None, sum(f_all.values())),
        "",
        f"All {procs} cores, catalogue + 5 fingerprints incl. parsing (RDKit: multiprocessing pool; "
        f"FerrumChem: `ferrum_cli calc` with the full bundle and CSV output):",
        "",
        "| | RDKit | FerrumChem (full bundle) |",
        "|---|---|---|",
        f"| wall time | {rd_mp:.2f} s | {fe_mp:.2f} s |",
        f"| molecules/s | {n / rd_mp:,.0f} | {n / fe_mp:,.0f} |",
        "",
        "Most expensive FerrumChem descriptor families (single thread, µs per molecule):",
        "",
        "| family | µs/mol |",
        "|---|---|",
    ]
    for name, us in sorted(f_all.items(), key=lambda kv: -kv[1])[:15]:
        lines.append(f"| {name} | {us:,.1f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.markdown:
        Path(args.markdown).write_text(text)


if __name__ == "__main__":
    main()
