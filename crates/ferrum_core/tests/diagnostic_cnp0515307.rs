use ferrum_core::algo::aromaticity::perceive_aromaticity;
use ferrum_core::algo::rings::find_minimum_cycle_basis_bonds;
use ferrum_core::io::smiles::parse_smiles;
use std::collections::HashSet;

#[test]
fn diagnostic_cnp0515307_2() {
    let smiles = "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2";
    let mut mol = parse_smiles(smiles).unwrap();

    eprintln!("\n=== CNP0515307.2 Diagnostic ===");
    eprintln!("Atoms: {}, Bonds: {}", mol.num_atoms(), mol.num_bonds());

    perceive_aromaticity(&mut mol);

    // Collect aromatic atoms
    let aromatic_atoms: Vec<usize> = mol
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.is_aromatic())
        .map(|(i, _)| i)
        .collect();

    eprintln!(
        "\nAromatic atoms ({}): {:?}",
        aromatic_atoms.len(),
        aromatic_atoms
    );

    // Find aromatic bonds
    let aromatic_bonds: Vec<(usize, usize, usize)> = mol
        .bonds
        .iter()
        .enumerate()
        .filter(|(_, b)| b.is_aromatic())
        .map(|(idx, b)| (idx, b.source as usize, b.target as usize))
        .collect();

    eprintln!("Aromatic bonds ({}): ", aromatic_bonds.len());

    let mut bond_44_51_found = false;
    for (bond_idx, src, tgt) in &aromatic_bonds {
        eprintln!("  Bond {}: {} -- {}", bond_idx, src, tgt);
        if (*src == 44 && *tgt == 51) || (*src == 51 && *tgt == 44) {
            bond_44_51_found = true;
            eprintln!("    ^^^ PROBLEM: Bond 44-51 is aromatic!");
        }
    }

    // Get ring basis and identify aromatic rings
    let ring_basis = find_minimum_cycle_basis_bonds(&mol);
    eprintln!("\nAromatic ring basis:");

    let mut aromatic_ring_count = 0;
    let mut rings_with_44_51 = Vec::new();

    for (ri, ring_bonds) in ring_basis.iter().enumerate() {
        let all_aromatic = ring_bonds.iter().all(|&bi| mol.bonds[bi].is_aromatic());

        if all_aromatic {
            aromatic_ring_count += 1;
            let ring_atoms: Vec<usize> = ring_bonds
                .iter()
                .flat_map(|&bi| {
                    let b = &mol.bonds[bi];
                    vec![b.source as usize, b.target as usize]
                })
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();

            let has_44_51 = ring_bonds.iter().any(|&bi| {
                let b = &mol.bonds[bi];
                (b.source == 44 && b.target == 51) || (b.source == 51 && b.target == 44)
            });

            if has_44_51 {
                rings_with_44_51.push(ri);
                eprintln!(
                    "  Ring {} [AROMATIC] (contains 44-51): {} bonds, atoms: {:?}",
                    ri,
                    ring_bonds.len(),
                    ring_atoms
                );
            } else {
                eprintln!(
                    "  Ring {} [AROMATIC]: {} bonds, atoms: {:?}",
                    ri,
                    ring_bonds.len(),
                    ring_atoms
                );
            }
        }
    }

    eprintln!("\nSummary:");
    eprintln!("  Aromatic atoms: {}", aromatic_atoms.len());
    eprintln!("  Aromatic rings: {}", aromatic_ring_count);
    eprintln!("  Bond 44-51 aromatic: {}", bond_44_51_found);
    eprintln!("  Rings containing 44-51: {:?}", rings_with_44_51);
    eprintln!("\nExpected (RDKit): atoms=23, rings=3");
    eprintln!("Current (Safe baseline): atoms=19, rings=4");
}
