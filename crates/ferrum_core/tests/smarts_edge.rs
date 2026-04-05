//! Integration tests: Smarts Edge.
//!
//! This test module validates end-to-end behavior across parser, chemistry
//! algorithms, descriptors, and ML surfaces. It captures real-world edge cases
//! so regressions are caught before parity or performance runs.

//! ## File guide: Smarts Edge
//!
//! This file belongs to FerrumChem's **integration/diagnostic validation layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `fn test_smarts_edge_cases`
//! - `fn test_smarts_hba_edge_cases`
//! - `fn test_smarts_bond_and_valence`
//! - `fn test_smarts_ring_bond_features`
//! - `fn test_smarts_disconnected_components_and_default_bonds`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - Edge-case SMARTS fixtures intentionally target precedence (`&,;,,!`), aromatic/aliphatic classing, and ring-bond corner cases.
//! - Expected match counts are pinned to current RDKit-aligned semantics and should change only with deliberate parser updates.
//! - Disconnected-component and default-bond tests validate language semantics, not synthetic chemistry likelihood.
//! ### Chemistry/ML references
//! - Daylight SMARTS specification examples for logical operators and atom/bond primitives.
//! - RDKit SMARTS matching behavior used when curating expected counts.
//! - Internal regression history around SMARTS precedence fixes in FerrumChem.

#[test]
/// Runs regression test for smarts edge cases.
///
/// Function context: `crates/ferrum_core/tests/smarts_edge.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn test_smarts_edge_cases() {
    use ferrum_core::algo::iso::{count_matches, find_all_matches};
    use ferrum_core::io::smarts::parse_smarts;
    use ferrum_core::io::smiles::parse_smiles;

    let cases = vec![
        // (SMARTS, SMILES, expected_count, description)
        (
            "[c](:a)(:a):a",
            "c1ccccc1",
            0,
            "aromatic bond in benzene: no 3-aro-nbr atoms",
        ),
        (
            "[c](:a)(:a)=[C,N,O]",
            "O=c1[nH]cccc1",
            2,
            "aromatic+double bond in 2-pyridone (2 symmetry matches)",
        ),
        ("[C]=[!C;A;!#1]", "C=O", 1, "aliphatic C=O"),
        (
            "[C]=[!C;A;!#1]",
            "O=c1[nH]cccc1",
            0,
            "[C] should NOT match aromatic c",
        ),
        (
            "[N]",
            "c1ccncc1",
            0,
            "aliphatic [N] should NOT match aromatic n",
        ),
        ("[n]", "c1ccncc1", 1, "aromatic [n] matches pyridine N"),
        ("[N]", "N", 1, "aliphatic [N] matches ammonia"),
        (
            "[!#6;!#7;!#8]",
            "c1cc[nH]c1",
            0,
            "atomic number negation in pyrrole",
        ),
        ("[O,S;H1;+0]", "CO", 1, "comma-semicolon precedence"),
        ("[O,S;H1;+0]", "CS", 1, "comma-semicolon with S"),
        ("[NH2+0][A;!#1]", "NC", 1, "NH2 aliphatic"),
        ("[NH2+0]a", "Nc1ccccc1", 1, "NH2 on aromatic"),
        ("[Nv3]", "N", 1, "N valence 3"),
        (
            "[Nv3]",
            "c1cc[nH]c1",
            0,
            "aromatic N should NOT match [Nv3]",
        ),
        ("[OH,OH2]", "O", 1, "comma OR in brackets: water"),
        ("[OH,OH2]", "CO", 1, "comma OR in brackets: methanol"),
        (
            "[CH3]a",
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            3,
            "CH3 on aromatic: caffeine",
        ),
        ("[cH]", "c1ccccc1", 6, "aromatic CH: benzene"),
        ("[n+0]", "c1ccncc1", 1, "aromatic n charge 0: pyridine"),
    ];

    let mut passed = 0;
    let mut failed = 0;
    for (smarts, smiles, expected, desc) in &cases {
        let q = match parse_smarts(smarts) {
            Ok(q) => q,
            Err(e) => {
                println!("  FAIL  {:40} parse error: {}", smarts, e);
                failed += 1;
                continue;
            }
        };
        let mol = parse_smiles(smiles).unwrap();
        let actual = count_matches(&mol, &q);
        if actual == *expected {
            passed += 1;
        } else {
            println!(
                "  FAIL  {:40} + {:30} expected={} got={} ({})",
                smarts, smiles, expected, actual, desc
            );
            failed += 1;
        }
    }
    println!("\nSMARTS edge cases: {}/{} passed", passed, passed + failed);
    assert_eq!(failed, 0, "{} SMARTS edge cases failed", failed);
}

#[test]
/// Runs regression test for smarts hba edge cases.
///
/// Function context: `crates/ferrum_core/tests/smarts_edge.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn test_smarts_hba_edge_cases() {
    use ferrum_core::algo::iso::count_matches;
    use ferrum_core::io::smarts::parse_smarts;
    use ferrum_core::io::smiles::parse_smiles;

    let cases = vec![
        ("[oH0;X2]", "c1ccoc1", 1, "furan o acceptor"),
        ("[s;a]", "c1ccsc1", 1, "aromatic s"),
        ("[nH0;X2]", "c1ccncc1", 1, "pyridine N acceptor"),
        ("[nH0,o,s;+0]", "c1ccoc1", 1, "furan o in comma-OR"),
        ("[nH0,o,s;+0]", "c1ccsc1", 1, "thiophene s in comma-OR"),
        ("[$([nH0,o,s;+0])]", "c1ccoc1", 1, "recursive furan o"),
        ("[$([nH0,o,s;+0])]", "c1ccsc1", 1, "recursive thiophene s"),
        ("[N,O;H1;+0]", "CO", 1, "comma-semi: methanol OH"),
        (
            "[N,O;H1;+0]",
            "CN",
            0,
            "comma-semi: methylamine NH2 (not H1)",
        ),
        ("[R]", "c1ccccc1", 6, "ring atoms in benzene"),
    ];

    let mut passed = 0;
    let mut failed = 0;
    for (smarts, smiles, expected, desc) in &cases {
        let q = match parse_smarts(smarts) {
            Ok(q) => q,
            Err(e) => {
                println!("  FAIL  {:40} parse error: {}", smarts, e);
                failed += 1;
                continue;
            }
        };
        let mol = parse_smiles(smiles).unwrap();
        let actual = count_matches(&mol, &q);
        if actual == *expected {
            passed += 1;
        } else {
            println!(
                "  FAIL  {:40} + {:30} expected={} got={} ({})",
                smarts, smiles, expected, actual, desc
            );
            failed += 1;
        }
    }
    println!(
        "\nHBA SMARTS edge cases: {}/{} passed",
        passed,
        passed + failed
    );
    assert_eq!(failed, 0, "{} HBA SMARTS edge cases failed", failed);
}

#[test]
/// Runs regression test for smarts bond and valence.
///
/// Function context: `crates/ferrum_core/tests/smarts_edge.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn test_smarts_bond_and_valence() {
    use ferrum_core::algo::iso::count_matches;
    use ferrum_core::io::smarts::parse_smarts;
    use ferrum_core::io::smiles::parse_smiles;

    let cases = vec![
        // Default bond matching
        (
            "CC",
            "c1ccccc1",
            0,
            "aliphatic CC should NOT match aromatic benzene",
        ),
        ("cc", "c1ccccc1", 12, "aromatic cc in benzene"),
        ("C-C", "CC", 2, "explicit single bond"),
        (
            "C-c",
            "Cc1ccccc1",
            1,
            "aliphatic C - aromatic c single bond",
        ),
        // Valence
        ("[cv4]", "c1ccccc1", 6, "aromatic C valence 4"),
        ("[cv3]", "c1ccccc1", 0, "aromatic C NOT valence 3"),
        ("[cX3]", "c1ccccc1", 6, "aromatic C connectivity 3"),
        // Negation
        ("[!c]", "c1ccccc1", 0, "NOT aromatic C in benzene"),
        ("[!c]", "Cc1ccccc1", 1, "NOT aromatic C in toluene"),
        // Recursive SMARTS
        ("[$([NH2+0][A;!#1])]", "NC", 1, "recursive NH2"),
        // Semicolons
        ("[N;!H0;v3]", "N", 1, "N with H, valence 3"),
    ];

    let mut passed = 0;
    let mut failed = 0;
    for (smarts, smiles, expected, desc) in &cases {
        let q = match parse_smarts(smarts) {
            Ok(q) => q,
            Err(e) => {
                println!("  FAIL  {:40} parse error: {}", smarts, e);
                failed += 1;
                continue;
            }
        };
        let mol = parse_smiles(smiles).unwrap();
        let actual = count_matches(&mol, &q);
        if actual == *expected {
            passed += 1;
        } else {
            println!(
                "  FAIL  {:40} + {:30} expected={} got={} ({})",
                smarts, smiles, expected, actual, desc
            );
            failed += 1;
        }
    }
    println!(
        "\nBond/valence SMARTS: {}/{} passed",
        passed,
        passed + failed
    );
    assert_eq!(failed, 0, "{} bond/valence SMARTS cases failed", failed);
}

#[test]
/// Runs regression test for smarts ring bond features.
///
/// Function context: `crates/ferrum_core/tests/smarts_edge.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn test_smarts_ring_bond_features() {
    use ferrum_core::algo::iso::count_matches;
    use ferrum_core::io::smarts::parse_smarts;
    use ferrum_core::io::smiles::parse_smiles;

    let cases = vec![
        // Ring bond (@) and not-ring bond (!@)
        ("C!@C", "C1CCCC1", 0, "!@ not-ring bond in cyclopentane"),
        ("C!@C", "CCCCC", 8, "!@ not-ring bond in pentane"),
        ("C@C", "C1CCCC1", 10, "@ ring bond in cyclopentane"),
        ("C@C", "CCCCC", 0, "@ ring bond in pentane"),
        (
            "C=!@C",
            "C1=CCCCC1",
            0,
            "=!@ should reject ring alkene bonds",
        ),
        ("C=!@C", "CC=CC", 2, "=!@ should match acyclic alkene bonds"),
        (
            "C=!@CC=[O,S]",
            "O=C1C=CCC1",
            0,
            "QED alert enone pattern should reject ring enones",
        ),
        (
            "C=!@CC=[O,S]",
            "CC=CC=O",
            1,
            "QED alert enone pattern should match acyclic enones",
        ),
        (
            "[D2][D2][D2][D2]",
            "CC=CC=CC",
            0,
            "omitted bonds between bracket atoms should not match across double bonds",
        ),
        // Any bond (~)
        ("C~C", "CC", 2, "~ any bond: ethane"),
        ("C~C", "C=C", 2, "~ any bond: ethylene"),
        ("c~c", "c1ccccc1", 12, "~ any bond: benzene"),
        // Ring membership
        ("[R1]", "C1CCCC1", 5, "R1 in cyclopentane"),
        ("[R0]", "C1CCCC1", 0, "R0 in cyclopentane"),
        ("[R0]", "CCCCC", 5, "R0 in pentane"),
        // Degree
        ("[D2]", "CCC", 1, "D2 middle of propane"),
        ("[D1]", "CCC", 2, "D1 ends of propane"),
        // Ring size
        ("[r5]", "C1CCCC1", 5, "r5 in cyclopentane"),
        ("[r6]", "c1ccccc1", 6, "r6 in benzene"),
        ("[r5]", "c1ccccc1", 0, "r5 in benzene (should not match)"),
    ];

    let mut passed = 0;
    let mut failed = 0;
    for (smarts, smiles, expected, desc) in &cases {
        let q = match parse_smarts(smarts) {
            Ok(q) => q,
            Err(e) => {
                println!("  FAIL  {:40} parse error: {}", smarts, e);
                failed += 1;
                continue;
            }
        };
        let mol = parse_smiles(smiles).unwrap();
        let actual = count_matches(&mol, &q);
        if actual == *expected {
            passed += 1;
        } else {
            println!(
                "  FAIL  {:40} + {:30} expected={} got={} ({})",
                smarts, smiles, expected, actual, desc
            );
            failed += 1;
        }
    }
    println!(
        "\nRing/bond features SMARTS: {}/{} passed",
        passed,
        passed + failed
    );
    assert_eq!(failed, 0, "{} ring/bond SMARTS cases failed", failed);
}

#[test]
/// Runs regression test for smarts disconnected components and default bonds.
///
/// Function context: `crates/ferrum_core/tests/smarts_edge.rs`.
/// Test-layer note: this assertion path pins expected behavior to prevent regressions.
fn test_smarts_disconnected_components_and_default_bonds() {
    use ferrum_core::algo::iso::has_substruct_match;
    use ferrum_core::io::smarts::parse_smarts;
    use ferrum_core::io::smiles::parse_smiles;

    let cases = vec![
        (
            "[D2]~[D2]~[D2]~[D2]",
            "CC=CC=CC",
            true,
            "explicit any bonds should match across alternating double bonds",
        ),
        (
            "C.C",
            "CC",
            true,
            "dot-separated SMARTS components should match within one connected molecule",
        ),
        (
            "C.C.C",
            "CCC",
            true,
            "multiple SMARTS components should map to distinct atoms",
        ),
        (
            "C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]",
            "CO[C@@H]1C=C(COC(=O)C2=CC=CC=C2)[C@H](OC(C)=O)[C@H](OC(=O)C2=CC=CC=C2)[C@@H]1O",
            true,
            "three-component ester alert should match tri-ester motif",
        ),
    ];

    let mut passed = 0;
    let mut failed = 0;
    for (smarts, smiles, expected, desc) in &cases {
        let q = match parse_smarts(smarts) {
            Ok(q) => q,
            Err(e) => {
                println!("  FAIL  {:40} parse error: {}", smarts, e);
                failed += 1;
                continue;
            }
        };
        let mol = parse_smiles(smiles).unwrap();
        let actual = has_substruct_match(&mol, &q);
        if actual == *expected {
            passed += 1;
        } else {
            println!(
                "  FAIL  {:40} + {:30} expected={} got={} ({})",
                smarts, smiles, expected, actual, desc
            );
            failed += 1;
        }
    }

    println!(
        "\nDisconnected/default-bond SMARTS: {}/{} passed",
        passed,
        passed + failed
    );
    assert_eq!(
        failed, 0,
        "{} disconnected/default-bond SMARTS cases failed",
        failed
    );
}
