#!/usr/bin/env python3
"""Generate RDKit reference descriptor values for the torture_set molecules."""

import csv
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

INPUT_CSV = "torture_set.csv"
OUTPUT_CSV = "crates/ferrum_core/tests/rdkit_reference.csv"

# Scalar descriptors: (column_name, rdkit_function)
SCALAR_DESCRIPTORS = [
    ("MolecularWeight", lambda m: Descriptors.MolWt(m)),
    ("LogP", lambda m: Descriptors.MolLogP(m)),
    ("MolarRefractivity", lambda m: Descriptors.MolMR(m)),
    ("TPSA", lambda m: Descriptors.TPSA(m)),
    ("HBondDonor", lambda m: Descriptors.NumHDonors(m)),
    ("HBondAcceptor", lambda m: Descriptors.NumHAcceptors(m)),
    ("RotatableBondCount", lambda m: Descriptors.NumRotatableBonds(m)),
    ("HeavyAtomCount", lambda m: Descriptors.HeavyAtomCount(m)),
    ("NumAtoms", lambda m: m.GetNumAtoms(onlyExplicit=False)),
    ("AromaticAtomCount", lambda m: sum(1 for a in m.GetAtoms() if a.GetIsAromatic())),
    ("HeteroatomCount", lambda m: Descriptors.NumHeteroatoms(m)),
    ("AromaticRingCount", lambda m: Descriptors.NumAromaticRings(m)),
    ("FormalCharge", lambda m: Chem.GetFormalCharge(m)),
    ("FractionCSP3", lambda m: Descriptors.FractionCSP3(m)),
    ("BertzCT", lambda m: Descriptors.BertzCT(m)),
    ("LabuteASA", lambda m: Descriptors.LabuteASA(m)),
    ("Ipc", lambda m: Descriptors.Ipc(m)),
    ("Qed", lambda m: QED.qed(m)),
    ("NumAmideBonds", lambda m: rdMolDescriptors.CalcNumAmideBonds(m)),
    ("HallKierAlpha", lambda m: Descriptors.HallKierAlpha(m)),
    ("BalabanJ", lambda m: Descriptors.BalabanJ(m)),
    ("BondCount", lambda m: m.GetNumBonds()),
    ("RingCount", lambda m: Descriptors.RingCount(m)),
    ("NumAliphaticRings", lambda m: Descriptors.NumAliphaticRings(m)),
    ("NumSaturatedRings", lambda m: Descriptors.NumSaturatedRings(m)),
]

# VSA descriptors (vectors) - use rdMolDescriptors functions
SLOGP_VSA_NAMES = [f"SlogP_VSA{i}" for i in range(1, 13)]
SMR_VSA_NAMES = [f"SMR_VSA{i}" for i in range(1, 11)]
PEOE_VSA_NAMES = [f"PEOE_VSA{i}" for i in range(1, 15)]

# Chi indices
CHI_NAMES = [
    "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v",
    "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",
]

# Kappa indices
KAPPA_NAMES = ["Kappa1", "Kappa2", "Kappa3"]


def safe_calc(func, mol, default="NaN"):
    """Calculate a descriptor, returning default on error."""
    try:
        val = func(mol)
        if val is None:
            return default
        return val
    except Exception as e:
        print(f"  Warning: {e}", file=sys.stderr)
        return default


def main():
    """
    Implements main.
    """
    molecules = []
    with open(INPUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row["SMILES"].strip()
            name = row["Name"].strip()
            if smiles:
                molecules.append((smiles, name))

    # Build header
    header = ["SMILES", "Name"]
    for col_name, _ in SCALAR_DESCRIPTORS:
        header.append(col_name)
    header.extend(SLOGP_VSA_NAMES)
    header.extend(SMR_VSA_NAMES)
    header.extend(PEOE_VSA_NAMES)
    header.extend(CHI_NAMES)
    header.extend(KAPPA_NAMES)

    rows = []
    for smiles, name in molecules:
        print(f"Processing: {name} ({smiles})")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  FAILED to parse: {smiles}", file=sys.stderr)
            continue

        # Add explicit Hs for NumAtoms comparison
        mol_h = Chem.AddHs(mol)

        row = [smiles, name]

        # Scalar descriptors
        for col_name, func in SCALAR_DESCRIPTORS:
            if col_name == "NumAtoms":
                val = safe_calc(func, mol_h)
            else:
                val = safe_calc(func, mol)
            row.append(val)

        # SlogP_VSA
        slogp_vsa = safe_calc(lambda m: list(rdMolDescriptors.SlogP_VSA_(m)), mol, default=None)
        if slogp_vsa is None:
            row.extend(["NaN"] * 12)
        else:
            row.extend(slogp_vsa[:12])

        # SMR_VSA
        smr_vsa = safe_calc(lambda m: list(rdMolDescriptors.SMR_VSA_(m)), mol, default=None)
        if smr_vsa is None:
            row.extend(["NaN"] * 10)
        else:
            row.extend(smr_vsa[:10])

        # PEOE_VSA
        peoe_vsa = safe_calc(lambda m: list(rdMolDescriptors.PEOE_VSA_(m)), mol, default=None)
        if peoe_vsa is None:
            row.extend(["NaN"] * 14)
        else:
            row.extend(peoe_vsa[:14])

        # Chi indices
        chi_funcs = [
            Descriptors.Chi0, Descriptors.Chi0n, Descriptors.Chi0v,
            Descriptors.Chi1, Descriptors.Chi1n, Descriptors.Chi1v,
            Descriptors.Chi2n, Descriptors.Chi2v, Descriptors.Chi3n,
            Descriptors.Chi3v, Descriptors.Chi4n, Descriptors.Chi4v,
        ]
        for func in chi_funcs:
            row.append(safe_calc(func, mol))

        # Kappa indices
        kappa_funcs = [
            Descriptors.Kappa1, Descriptors.Kappa2, Descriptors.Kappa3,
        ]
        for func in kappa_funcs:
            row.append(safe_calc(func, mol))

        rows.append(row)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} molecules to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
