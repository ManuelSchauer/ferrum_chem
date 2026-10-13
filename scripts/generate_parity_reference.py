#!/usr/bin/env python3
"""Generate the RDKit references used by `crates/ferrum_core/tests/rdkit_parity.rs`
and `crates/ferrum_core/tests/rdkit_3d_parity.rs`.

Reads `tests/data/parity_molecules.smi` and writes
`tests/data/rdkit_parity_reference.csv` with one row per molecule. Columns use
FerrumChem's feature names (RDKit names wherever an RDKit equivalent exists),
fingerprints are stored as on-bit lists, and `perception` holds the per-atom
RDKit sanitization result (element, charge, aromaticity, H count,
hybridization, radicals, ring membership).

3D descriptors (PMI1-3, Asphericity) are checked on `tests/data/parity_3d.sdf`:
~300 molecules of `parity_molecules.smi` with explicit hydrogens, embedded once
with ETKDGv3 (`--embed-3d`; coordinates depend on the RDKit build, so the SDF is
committed input data). `tests/data/rdkit_3d_parity_reference.csv` is computed
from the molecules read back from that SDF with RDKit defaults
(`SDMolSupplier`: sanitize, removeHs), which is what FerrumChem's SDF reader
does too.

Molecules are processed and written one at a time (constant memory).

    python scripts/generate_parity_reference.py             # (re)write both references
    python scripts/generate_parity_reference.py --check     # verify the committed files
    python scripts/generate_parity_reference.py --embed-3d  # re-embed parity_3d.sdf (then rewrite)
"""
import csv
import math
import sys
from pathlib import Path

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors as D
from rdkit.Chem import GraphDescriptors as G
from rdkit.Chem import Lipinski, MACCSkeys, QED
from rdkit.Chem import rdFingerprintGenerator as FG
from rdkit.Chem import rdMolDescriptors as R
from rdkit.Chem.EState import AtomTypes, EState, Fingerprinter

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "crates/ferrum_core/tests/data"
SMILES_FILE = DATA / "parity_molecules.smi"
OUT_FILE = DATA / "rdkit_parity_reference.csv"
SDF_3D_FILE = DATA / "parity_3d.sdf"
OUT_3D_FILE = DATA / "rdkit_3d_parity_reference.csv"
EMBED_SEED = 0xF3D
EMBED_STRIDE = 4  # every 4th molecule of parity_molecules.smi ...
EMBED_MAX_HEAVY = 60  # ... with at most 60 heavy atoms (keeps the SDF small)
EMBED_COUNT = 300

HYB = {
    Chem.HybridizationType.UNSPECIFIED: "U", Chem.HybridizationType.S: "S",
    Chem.HybridizationType.SP: "SP", Chem.HybridizationType.SP2: "SP2",
    Chem.HybridizationType.SP3: "SP3", Chem.HybridizationType.SP2D: "SP2D",
    Chem.HybridizationType.SP3D: "SP3D", Chem.HybridizationType.SP3D2: "SP3D2",
    Chem.HybridizationType.OTHER: "O",
}
ELEMENT_COLS = [("nC", 6), ("nN", 7), ("nO", 8), ("nS", 16), ("nP", 15), ("nF", 9),
                ("nCl", 17), ("nBr", 35), ("nI", 53)]

ECFP4 = FG.GetMorganGenerator(radius=2, fpSize=2048)
FCFP4 = FG.GetMorganGenerator(radius=2, fpSize=2048,
                              atomInvariantsGenerator=FG.GetMorganFeatureAtomInvGen())
ATOM_PAIR = FG.GetAtomPairGenerator(fpSize=2048)
TORSION = FG.GetTopologicalTorsionGenerator(fpSize=2048)

AtomTypes.BuildPatts()
ESTATE_TYPES = [name for name, _ in AtomTypes.esPatterns]


def scalar_columns(m):
    """RDKit values keyed by FerrumChem feature name."""
    total_h = sum(a.GetTotalNumHs() for a in m.GetAtoms())
    heavy = m.GetNumHeavyAtoms()
    n_arom = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
    bond_types = [b.GetBondType() for b in m.GetBonds()]
    estate = EState.EStateIndices(m)

    v = {
        "MolWt": D.MolWt(m), "MolLogP": D.MolLogP(m), "MolMR": D.MolMR(m), "TPSA": D.TPSA(m),
        "NumHDonors": D.NumHDonors(m), "NumHAcceptors": D.NumHAcceptors(m),
        "NumRotatableBonds": D.NumRotatableBonds(m), "HeavyAtomCount": D.HeavyAtomCount(m),
        "nAtom": m.GetNumAtoms() + total_h, "nH": total_h, "nHeavyAtom": heavy,
        "nAromAtom": n_arom, "AromaticProportion": n_arom / heavy if heavy else 0.0,
        "NumHeteroatoms": D.NumHeteroatoms(m),
        "NumAromaticRings": D.NumAromaticRings(m), "NumAromaticRingsSSSR": D.NumAromaticRings(m),
        "NumSaturatedRings": D.NumSaturatedRings(m), "NumAliphaticRings": D.NumAliphaticRings(m),
        "nBondsHeavy": m.GetNumBonds(), "nBonds": m.GetNumBonds(),
        "nBondsS": bond_types.count(Chem.BondType.SINGLE),
        "nBondsD": bond_types.count(Chem.BondType.DOUBLE),
        "nBondsT": bond_types.count(Chem.BondType.TRIPLE),
        "nBondsA": bond_types.count(Chem.BondType.AROMATIC),
        "NumValenceElectrons": D.NumValenceElectrons(m),
        "NumRadicalElectrons": D.NumRadicalElectrons(m),
        "FormalCharge": Chem.GetFormalCharge(m), "FractionCSP3": D.FractionCSP3(m),
        "BertzCT": D.BertzCT(m), "Ipc": D.Ipc(m), "LabuteASA": D.LabuteASA(m), "qed": QED.qed(m),
        "NumAmideBonds": R.CalcNumAmideBonds(m), "HallKierAlpha": D.HallKierAlpha(m),
        "BalabanJ": G.BalabanJ(m),
        "Chi0": D.Chi0(m), "Chi1": D.Chi1(m),
        "Chi0v": D.Chi0v(m), "Chi1v": D.Chi1v(m), "Chi2v": D.Chi2v(m), "Chi3v": D.Chi3v(m),
        "Chi4v": D.Chi4v(m),
        "Kappa1": D.Kappa1(m), "Kappa2": D.Kappa2(m), "Kappa3": D.Kappa3(m),
        "SumEState": sum(estate), "MinEStateIndex": min(estate) if len(estate) else 0.0,
        "MaxEStateIndex": max(estate) if len(estate) else 0.0,
        "MaxPartialCharge": D.MaxPartialCharge(m), "MinPartialCharge": D.MinPartialCharge(m),
        "MaxAbsPartialCharge": D.MaxAbsPartialCharge(m), "MinAbsPartialCharge": D.MinAbsPartialCharge(m),
        # FerrumChem composite of RDKit values (rule of five, 4 rules)
        "LipinskiViolations": int(D.MolWt(m) > 500) + int(D.MolLogP(m) > 5)
        + int(Lipinski.NumHDonors(m) > 5) + int(Lipinski.NumHAcceptors(m) > 10),
    }
    # E-State atom-type sums (Hall & Kier types, RDKit order)
    for name, x in zip(ESTATE_TYPES, Fingerprinter.FingerprintMol(m)[1]):
        v[f"SEstate_{name}"] = float(x)
    for name, z in ELEMENT_COLS:
        v[name] = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == z)
    known = {z for _, z in ELEMENT_COLS}
    v["nOtherElements"] = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1 and a.GetAtomicNum() not in known)
    for i, x in enumerate(R.SlogP_VSA_(m), 1):
        v[f"SlogP_VSA{i}"] = x
    for i, x in enumerate(R.SMR_VSA_(m), 1):
        v[f"SMR_VSA{i}"] = x
    for i, x in enumerate(R.PEOE_VSA_(m), 1):
        v[f"PEOE_VSA{i}"] = x
    # the complete rdkit.Chem.Descriptors catalogue
    for name, fn in D._descList:
        if name in v:
            continue
        try:
            v[name] = fn(m)
        except Exception:
            v[name] = float("nan")
    return v


def fingerprint_columns(m):
    on = lambda fp: " ".join(map(str, fp.GetOnBits()))
    # RDKit MACCS has 167 bits with bit 0 unused; FerrumChem column MACCS{i} is key i.
    return {
        "fp:MACCS": " ".join(str(b) for b in MACCSkeys.GenMACCSKeys(m).GetOnBits() if b > 0),
        "fp:ECFP4": on(ECFP4.GetFingerprint(m)),
        "fp:FCFP4": on(FCFP4.GetFingerprint(m)),
        "fp:AtomPair": on(ATOM_PAIR.GetFingerprint(m)),
        "fp:TopoTorsion": on(TORSION.GetFingerprint(m)),
    }


def perception(m):
    return " ".join(
        f"{a.GetAtomicNum()},{a.GetFormalCharge()},{int(a.GetIsAromatic())},{a.GetTotalNumHs()},"
        f"{HYB[a.GetHybridization()]},{a.GetNumRadicalElectrons()},{int(a.IsInRing())}"
        for a in m.GetAtoms()
    )


def fmt(x):
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return repr(float(x))
    return str(x)


def rows():
    """Yields (columns, row) one molecule at a time."""
    probe = Chem.MolFromSmiles("CCO")
    columns = ["smiles", "perception"] + list(scalar_columns(probe)) + list(fingerprint_columns(probe))
    with open(SMILES_FILE) as src:
        for line in src:
            smi = line.strip()
            if not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                yield columns, [smi, "ERR"] + [""] * (len(columns) - 2)
            else:
                vals = {**scalar_columns(m), **fingerprint_columns(m)}
                yield columns, [smi, perception(m)] + [fmt(vals[c]) for c in columns[2:]]


def embed_3d():
    """Writes parity_3d.sdf: ETKDGv3 conformers of a stride sample of the parity set."""
    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    n = 0
    w = Chem.SDWriter(str(SDF_3D_FILE))
    with open(SMILES_FILE) as src:
        lines = [l.strip() for l in src if l.strip()]
    for i in range(0, len(lines), EMBED_STRIDE):
        m = Chem.MolFromSmiles(lines[i])
        if m is None or m.GetNumHeavyAtoms() > EMBED_MAX_HEAVY:
            continue
        mh = Chem.AddHs(m)
        if AllChem.EmbedMolecule(mh, params) != 0:
            continue
        mh.SetProp("_Name", f"parity_molecules.smi:{i + 1}")
        mh.SetProp("smiles", lines[i])
        w.write(mh)
        n += 1
        if n == EMBED_COUNT:
            break
    w.close()
    print(f"wrote {SDF_3D_FILE} ({n} molecules)")


COLUMNS_3D = ["name", "perception", "PMI1", "PMI2", "PMI3", "Asphericity"]


def rows_3d():
    """Yields (columns, row) for the molecules of parity_3d.sdf as RDKit reads them."""
    for m in Chem.SDMolSupplier(str(SDF_3D_FILE)):  # sanitize=True, removeHs=True
        if m is None:
            sys.exit("RDKit cannot read a record of parity_3d.sdf")
        yield COLUMNS_3D, [
            m.GetProp("_Name"), perception(m), fmt(R.CalcPMI1(m)), fmt(R.CalcPMI2(m)),
            fmt(R.CalcPMI3(m)), fmt(R.CalcAsphericity(m)),
        ]


def same(a, b):
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return abs(x - y) <= 1e-9 + 1e-9 * abs(y)


def check_file(out_file, generator):
    """Compares a fresh RDKit run with a committed reference (float tolerance 1e-9)."""
    with open(out_file, newline="") as f:
        ref = list(csv.reader(line for line in f if not line.startswith("#")))
    header, expected_rows = ref[0], ref[1:]
    fresh = list(generator())
    if len(fresh) != len(expected_rows):
        sys.exit(f"{out_file.name}: {len(expected_rows)} committed rows, RDKit gives {len(fresh)}")
    bad = 0
    for (columns, row), expected in zip(fresh, expected_rows):
        if columns != header:
            sys.exit(f"{out_file.name}: column layout changed; regenerate the reference")
        for col, a, b in zip(columns, row, expected):
            if not same(a, b):
                bad += 1
                if bad <= 10:
                    print(f"{row[0]}: {col}: committed={b} rdkit={a}")
    if bad:
        sys.exit(f"{out_file.name}: {bad} values differ from the committed reference")
    print(f"committed {out_file.name} ({len(fresh)} molecules) matches RDKit", rdkit.__version__)


def check():
    check_file(OUT_FILE, rows)
    check_file(OUT_3D_FILE, rows_3d)


def write_reference(out_file, generator):
    n = 0
    with open(out_file, "w", newline="") as dst:
        w = csv.writer(dst, lineterminator="\n")
        dst.write(f"# RDKit {rdkit.__version__} reference, generated by scripts/generate_parity_reference.py\n")
        for columns, row in generator():
            if n == 0:
                w.writerow(columns)
            w.writerow(row)
            n += 1
    print(f"wrote {out_file} ({n} molecules)")


def main():
    if "--check" in sys.argv:
        check()
        return
    if "--embed-3d" in sys.argv:
        embed_3d()
    write_reference(OUT_FILE, rows)
    write_reference(OUT_3D_FILE, rows_3d)


if __name__ == "__main__":
    main()
