#!/usr/bin/env python3
import os
import statistics
import time

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

ITERATIONS = int(os.getenv("FERRUM_BENCH_ITERATIONS", "10000"))
REPEATS = int(os.getenv("FERRUM_BENCH_REPEATS", "3"))
WARMUP = int(os.getenv("FERRUM_BENCH_WARMUP", "200"))

CASES = [
    ("Ethane", "CC"),
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("Caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
    (
        "LargeDruglike",
        "O=C(N[C@@H](OC[C@@H]1CCCN2CCCC[C@H]12)C(Cl)(Cl)Cl)C1=CC=CC2=CC=CC=C12",
    ),
]

SCALAR_DESCRIPTORS = [
    lambda m: Descriptors.MolWt(m),
    lambda m: Descriptors.MolLogP(m),
    lambda m: Descriptors.MolMR(m),
    lambda m: Descriptors.TPSA(m),
    lambda m: Descriptors.NumHDonors(m),
    lambda m: Descriptors.NumHAcceptors(m),
    lambda m: Descriptors.NumRotatableBonds(m),
    lambda m: Descriptors.HeavyAtomCount(m),
    lambda m: m.GetNumAtoms(onlyExplicit=False),
    lambda m: sum(1 for a in m.GetAtoms() if a.GetIsAromatic()),
    lambda m: Descriptors.NumHeteroatoms(m),
    lambda m: Descriptors.NumAromaticRings(m),
    lambda m: Chem.GetFormalCharge(m),
    lambda m: Descriptors.FractionCSP3(m),
    lambda m: Descriptors.BertzCT(m),
    lambda m: Descriptors.LabuteASA(m),
    lambda m: QED.qed(m),
    lambda m: rdMolDescriptors.CalcNumAmideBonds(m),
    lambda m: Descriptors.HallKierAlpha(m),
    lambda m: Descriptors.BalabanJ(m),
    lambda m: m.GetNumBonds(),
]


def bundle_output_len(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    for func in SCALAR_DESCRIPTORS:
        func(mol)
    rdMolDescriptors.SlogP_VSA_(mol)
    rdMolDescriptors.SMR_VSA_(mol)
    rdMolDescriptors.PEOE_VSA_(mol)
    return 57


def run_bundle(smiles: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    for func in SCALAR_DESCRIPTORS:
        func(mol)
    rdMolDescriptors.SlogP_VSA_(mol)
    rdMolDescriptors.SMR_VSA_(mol)
    rdMolDescriptors.PEOE_VSA_(mol)


def main() -> None:
    print(
        "tool\tcase\tsmiles\theavy_atoms\titerations\toutputs\tmedian_seconds\tms_per_molecule\tmolecules_per_second"
    )
    for name, smiles in CASES:
        mol = Chem.MolFromSmiles(smiles)
        heavy_atoms = mol.GetNumHeavyAtoms()
        outputs = bundle_output_len(smiles)

        for _ in range(WARMUP):
            run_bundle(smiles)

        timings = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            for _ in range(ITERATIONS):
                run_bundle(smiles)
            timings.append(time.perf_counter() - start)

        median_seconds = statistics.median(timings)
        ms_per_molecule = median_seconds * 1000.0 / ITERATIONS
        molecules_per_second = ITERATIONS / median_seconds
        print(
            f"rdkit\t{name}\t{smiles}\t{heavy_atoms}\t{ITERATIONS}\t{outputs}\t{median_seconds:.6f}\t{ms_per_molecule:.6f}\t{molecules_per_second:.2f}"
        )


if __name__ == "__main__":
    main()
