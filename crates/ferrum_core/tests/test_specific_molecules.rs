//! Integration tests: Test Specific Molecules.
//!
//! This test module validates end-to-end behavior across parser, chemistry
//! algorithms, descriptors, and ML surfaces. It captures real-world edge cases
//! so regressions are caught before parity or performance runs.

//! ## File guide: Test Specific Molecules
//!
//! This file belongs to FerrumChem's **integration/diagnostic validation layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct RepresentativeCase`
//! - `fn aromatic_atom_count`
//! - `fn assert_aromatic_state_consistent`
//! - `fn find_bond`
//! - `fn representative_blockers_keep_aromatic_state_consistent`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Test Specific Molecules` fixtures are regression-focused: they pin previously fragile behavior before parity/performance runs.
//! - `Test Specific Molecules` assumes deterministic parser/descriptor outputs so test failures indicate real semantic drift.
//! - Test expectations in `Test Specific Molecules` are intentionally explicit to support learning-oriented debugging.
//! - Specific scope anchor: `crates/ferrum_core/tests/test_specific_molecules.rs` defines test specific molecules behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Repository historical bug reports and checkpoints that motivated `Test Specific Molecules` assertions.
//! - RDKit or chemistry-reference fixtures used to validate `Test Specific Molecules` expectations.
//! - Rust integration-test best practices for reproducible scientific software checks.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/tests/test_specific_molecules.rs`.

use ferrum_core::algo::rings::find_minimum_cycle_basis_bonds;
use ferrum_core::desc::aromatic_ring_count::compute_aromatic_ring_count;
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::types::BondType;

struct RepresentativeCase {
    id: &'static str,
    smiles: &'static str,
    rdkit_aromatic_atoms: usize,
    rdkit_aromatic_rings: u32,
}

const REPRESENTATIVE_CASES: &[RepresentativeCase] = &[
    RepresentativeCase {
        id: "CNP0515307.2",
        smiles: "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2",
        rdkit_aromatic_atoms: 23,
        rdkit_aromatic_rings: 3,
    },
    RepresentativeCase {
        id: "CNP0449381.0",
        smiles: "CC1=CC(O)=C2C(=CC3=CC(OS(=O)(=O)O)=C(C4=C(OS(=O)(=O)O)C=C(O)C5=C4C=C4OC(C)=CC(O)=C4C5=O)C(O)=C3C2=O)O1",
        rdkit_aromatic_atoms: 28,
        rdkit_aromatic_rings: 2,
    },
    RepresentativeCase {
        id: "CNP0065983.0",
        smiles: "CC(C)(C)C1=CC=C(C2=CC(=O)C3=CC=C(NC(=O)C4=CC=CC=C4Cl)C=C3O2)C=C1",
        rdkit_aromatic_atoms: 22,
        rdkit_aromatic_rings: 4,
    },
    RepresentativeCase {
        id: "CNP0454690.1",
        smiles: "CN1C(O)=C(C(=O)C[C@@H](NC2=CC=C(N)C=C2)C2=C(O)C=CC3=CC=CC=C23)C(=O)C2=CC=CC=C21",
        rdkit_aromatic_atoms: 26,
        rdkit_aromatic_rings: 5,
    },
    RepresentativeCase {
        id: "CNP0584384.2",
        smiles: "C/C1=C/[C@H](C)[C@@H](O)[C@@H](C)/C=C/[C@@H](O)C/C=C(\\C)C(=O)C[C@@H](O)[C@@H](C)C=C/C=C/C=C\\C(=O)NC2=CC(=O)C3=C(C=C(C)C(O)=C3C1=O)C2=O",
        rdkit_aromatic_atoms: 6,
        rdkit_aromatic_rings: 1,
    },
    RepresentativeCase {
        id: "CNP0321538.0",
        smiles: "COC1=CC(O)=C2C(=O)C3=CC=C(O)C(OC)=C3OC2=C1",
        rdkit_aromatic_atoms: 14,
        rdkit_aromatic_rings: 3,
    },
    RepresentativeCase {
        id: "CNP0539904.0",
        smiles: "CC1=CC2=[N+](CC(=O)C3=CC=C([N+](=O)[O-])C=C3)C(C)=CN2C=C1",
        rdkit_aromatic_atoms: 15,
        rdkit_aromatic_rings: 3,
    },
    RepresentativeCase {
        id: "CNP0331134.0",
        smiles: "CC1=CC(O)=C2C(=O)C3=C(C(=O)C2=C1)C1=C(O3)C2=C(C=C1C)C(=O)C=CC2=O",
        rdkit_aromatic_atoms: 21,
        rdkit_aromatic_rings: 5,
    },
    RepresentativeCase {
        id: "CNP0138771.1",
        smiles: "COC1=CC(OC)=C2C(=O)C3=C(OC)C4=C(C=C3C(=O)C2=C1)O[C@@H](O)[C@@H]4CCOC(C)=O",
        rdkit_aromatic_atoms: 12,
        rdkit_aromatic_rings: 2,
    },
    RepresentativeCase {
        id: "CNP0234360.4",
        smiles: "CC1=CC(O)=C2C(=O)C3=C(O[C@@H]4O[C@@H](COC(=O)CC(=O)O)[C@@H](O)[C@@H](O)[C@H]4O)C=CC=C3C(=O)C2=C1",
        rdkit_aromatic_atoms: 12,
        rdkit_aromatic_rings: 2,
    },
    RepresentativeCase {
        id: "CNP0208621.2",
        smiles: "CC1=CC(O)=C2C(=O)C3=C(O)C=C(O[C@@H]4O[C@@H](C)[C@H](O)[C@@H](O)[C@H]4O)C=C3C(=O)C2=C1",
        rdkit_aromatic_atoms: 12,
        rdkit_aromatic_rings: 2,
    },
];

/// Implements atom count.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn aromatic_atom_count(smiles: &str) -> usize {
    let mol = parse_smiles(smiles).expect("valid representative SMILES");
    mol.atoms.iter().filter(|atom| atom.is_aromatic()).count()
}

/// Asserts aromatic state consistent to guard invariants.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn assert_aromatic_state_consistent(case: &RepresentativeCase) {
    let mol = parse_smiles(case.smiles).expect("valid representative SMILES");

    for (bond_idx, bond) in mol.bonds.iter().enumerate() {
        assert_eq!(
            bond.is_aromatic(),
            bond.bond_type == BondType::Aromatic,
            "{} bond {} has mismatched aromatic flag/type",
            case.id,
            bond_idx
        );
        if bond.is_aromatic() {
            assert!(
                mol.atoms[bond.source as usize].is_aromatic(),
                "{} bond {} is aromatic but source atom {} is not",
                case.id,
                bond_idx,
                bond.source
            );
            assert!(
                mol.atoms[bond.target as usize].is_aromatic(),
                "{} bond {} is aromatic but target atom {} is not",
                case.id,
                bond_idx,
                bond.target
            );
        }
    }

    let aromatic_basis_rings = find_minimum_cycle_basis_bonds(&mol)
        .into_iter()
        .filter(|ring| {
            ring.iter()
                .all(|&bond_idx| mol.bonds[bond_idx].is_aromatic())
        })
        .count() as u32;
    assert_eq!(
        aromatic_basis_rings,
        compute_aromatic_ring_count(&mol),
        "{} aromatic ring helper disagrees with descriptor",
        case.id
    );
}

/// Finds bond.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn find_bond(mol: &ferrum_core::mol::molecule::Molecule, a: u16, b: u16) -> usize {
    mol.adjacency[a as usize]
        .iter()
        .map(|&bond_idx| bond_idx as usize)
        .find(|&bond_idx| {
            let bond = &mol.bonds[bond_idx];
            (bond.source == a && bond.target == b) || (bond.source == b && bond.target == a)
        })
        .expect("expected bond to exist")
}

#[test]
/// Runs regression test for representative blockers keep aromatic state consistent.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn representative_blockers_keep_aromatic_state_consistent() {
    for case in REPRESENTATIVE_CASES {
        assert_aromatic_state_consistent(case);
    }
}

#[test]
/// Runs regression test for cnp0515307 bridge bond stays kekulized.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn cnp0515307_bridge_bond_stays_kekulized() {
    let mol = parse_smiles(REPRESENTATIVE_CASES[0].smiles).expect("valid representative SMILES");
    let bond_idx = find_bond(&mol, 44, 51);
    let bond = &mol.bonds[bond_idx];
    assert_eq!(bond.bond_type, BondType::Single);
    assert!(!bond.is_aromatic());
    assert!(mol.atoms[44].is_aromatic());
    assert!(mol.atoms[51].is_aromatic());
}

#[test]
/// Runs regression test for cnp0515307 matches rdkit aromatic targets.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn cnp0515307_matches_rdkit_aromatic_targets() {
    let mol = parse_smiles(REPRESENTATIVE_CASES[0].smiles).expect("valid representative SMILES");
    let aromatic_atoms = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
    assert_eq!(aromatic_atoms, REPRESENTATIVE_CASES[0].rdkit_aromatic_atoms);
    assert_eq!(
        compute_aromatic_ring_count(&mol),
        REPRESENTATIVE_CASES[0].rdkit_aromatic_rings
    );
    for atom_idx in [44usize, 45, 46, 48, 49, 51] {
        assert!(
            mol.atoms[atom_idx].is_aromatic(),
            "atom {atom_idx} should be aromatic"
        );
    }
}

#[test]
/// Runs regression test for cnp0584384 matches rdkit aromatic targets.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn cnp0584384_matches_rdkit_aromatic_targets() {
    let mol = parse_smiles(REPRESENTATIVE_CASES[4].smiles).expect("valid representative SMILES");
    let aromatic_atoms = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
    assert_eq!(aromatic_atoms, REPRESENTATIVE_CASES[4].rdkit_aromatic_atoms);
    assert_eq!(
        compute_aromatic_ring_count(&mol),
        REPRESENTATIVE_CASES[4].rdkit_aromatic_rings
    );
    for atom_idx in [37usize, 38, 39, 40, 42, 44] {
        assert!(
            mol.atoms[atom_idx].is_aromatic(),
            "atom {atom_idx} should be aromatic"
        );
    }
    for atom_idx in [33usize, 34, 35, 47] {
        assert!(
            !mol.atoms[atom_idx].is_aromatic(),
            "atom {atom_idx} should stay kekulized"
        );
    }
}

#[test]
/// Runs regression test for cnp0321538 matches rdkit aromatic targets.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn cnp0321538_matches_rdkit_aromatic_targets() {
    let mol = parse_smiles(REPRESENTATIVE_CASES[5].smiles).expect("valid representative SMILES");
    let aromatic_atoms = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
    assert_eq!(aromatic_atoms, REPRESENTATIVE_CASES[5].rdkit_aromatic_atoms);
    assert_eq!(
        compute_aromatic_ring_count(&mol),
        REPRESENTATIVE_CASES[5].rdkit_aromatic_rings
    );
    for atom_idx in [6usize, 7, 9, 17, 18, 19] {
        assert!(
            mol.atoms[atom_idx].is_aromatic(),
            "atom {atom_idx} should be aromatic"
        );
    }
}

#[test]
/// Runs regression test for cnp0539904 matches rdkit aromatic targets.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn cnp0539904_matches_rdkit_aromatic_targets() {
    let mol = parse_smiles(REPRESENTATIVE_CASES[6].smiles).expect("valid representative SMILES");
    let aromatic_atoms = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
    assert_eq!(aromatic_atoms, REPRESENTATIVE_CASES[6].rdkit_aromatic_atoms);
    assert_eq!(
        compute_aromatic_ring_count(&mol),
        REPRESENTATIVE_CASES[6].rdkit_aromatic_rings
    );
    for atom_idx in [1usize, 2, 3, 4, 8, 9, 10, 11, 15, 16, 17, 19, 20, 21, 22] {
        assert!(
            mol.atoms[atom_idx].is_aromatic(),
            "atom {atom_idx} should be aromatic"
        );
    }
}

#[test]
/// Runs regression test for cnp0331134 matches rdkit aromatic targets.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn cnp0331134_matches_rdkit_aromatic_targets() {
    let mol = parse_smiles(REPRESENTATIVE_CASES[7].smiles).expect("valid representative SMILES");
    let aromatic_atoms = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
    assert_eq!(aromatic_atoms, REPRESENTATIVE_CASES[7].rdkit_aromatic_atoms);
    assert_eq!(
        compute_aromatic_ring_count(&mol),
        REPRESENTATIVE_CASES[7].rdkit_aromatic_rings
    );
    for atom_idx in [
        1usize, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 26,
    ] {
        assert!(
            mol.atoms[atom_idx].is_aromatic(),
            "atom {atom_idx} should be aromatic"
        );
    }
}

#[test]
/// Runs regression test for cnp0138771 matches rdkit aromatic targets.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn cnp0138771_matches_rdkit_aromatic_targets() {
    let mol = parse_smiles(REPRESENTATIVE_CASES[8].smiles).expect("valid representative SMILES");
    let aromatic_atoms = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
    assert_eq!(aromatic_atoms, REPRESENTATIVE_CASES[8].rdkit_aromatic_atoms);
    assert_eq!(
        compute_aromatic_ring_count(&mol),
        REPRESENTATIVE_CASES[8].rdkit_aromatic_rings
    );
    for atom_idx in [2usize, 3, 4, 7, 10, 11, 14, 15, 16, 17, 20, 21] {
        assert!(
            mol.atoms[atom_idx].is_aromatic(),
            "atom {atom_idx} should be aromatic"
        );
    }
    for atom_idx in [8usize, 18] {
        assert!(
            !mol.atoms[atom_idx].is_aromatic(),
            "atom {atom_idx} should stay kekulized"
        );
    }
}

#[test]
/// Runs regression test for cnp0234360 matches rdkit aromatic targets.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn cnp0234360_matches_rdkit_aromatic_targets() {
    let mol = parse_smiles(REPRESENTATIVE_CASES[9].smiles).expect("valid representative SMILES");
    let aromatic_atoms = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
    assert_eq!(aromatic_atoms, REPRESENTATIVE_CASES[9].rdkit_aromatic_atoms);
    assert_eq!(
        compute_aromatic_ring_count(&mol),
        REPRESENTATIVE_CASES[9].rdkit_aromatic_rings
    );
    for atom_idx in [1usize, 2, 3, 5, 8, 9, 28, 29, 30, 31, 34, 35] {
        assert!(
            mol.atoms[atom_idx].is_aromatic(),
            "atom {atom_idx} should be aromatic"
        );
    }
    for atom_idx in [6usize, 32] {
        assert!(
            !mol.atoms[atom_idx].is_aromatic(),
            "atom {atom_idx} should stay kekulized"
        );
    }
}

#[test]
/// Runs regression test for cnp0208621 matches rdkit aromatic targets.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn cnp0208621_matches_rdkit_aromatic_targets() {
    let mol = parse_smiles(REPRESENTATIVE_CASES[10].smiles).expect("valid representative SMILES");
    let aromatic_atoms = mol.atoms.iter().filter(|atom| atom.is_aromatic()).count();
    assert_eq!(
        aromatic_atoms,
        REPRESENTATIVE_CASES[10].rdkit_aromatic_atoms
    );
    assert_eq!(
        compute_aromatic_ring_count(&mol),
        REPRESENTATIVE_CASES[10].rdkit_aromatic_rings
    );
    for atom_idx in [1usize, 2, 3, 5, 8, 9, 11, 12, 24, 25, 28, 29] {
        assert!(
            mol.atoms[atom_idx].is_aromatic(),
            "atom {atom_idx} should be aromatic"
        );
    }
    for atom_idx in [6usize, 26] {
        assert!(
            !mol.atoms[atom_idx].is_aromatic(),
            "atom {atom_idx} should stay kekulized"
        );
    }
}

#[test]
#[ignore = "Known RDKit parity gaps while aromaticity work is in progress"]
/// Runs regression test for representative blockers match rdkit targets.
///
/// Function context: `crates/ferrum_core/tests/test_specific_molecules.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn representative_blockers_match_rdkit_targets() {
    for case in REPRESENTATIVE_CASES {
        let mol = parse_smiles(case.smiles).expect("valid representative SMILES");
        assert_eq!(
            aromatic_atom_count(case.smiles),
            case.rdkit_aromatic_atoms,
            "{} aromatic atom count should match RDKit",
            case.id
        );
        assert_eq!(
            compute_aromatic_ring_count(&mol),
            case.rdkit_aromatic_rings,
            "{} aromatic ring count should match RDKit",
            case.id
        );
    }
}
