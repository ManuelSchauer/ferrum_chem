use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::algo::rings::find_minimum_cycle_basis_bonds;

#[test]
fn debug_ring_structure() {
    let smiles = "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2";
    let mol = parse_smiles(smiles).unwrap();
    
    eprintln!("\n=== Ring Basis (as bonds) ===");
    let rings = find_minimum_cycle_basis_bonds(&mol);
    
    for (ri, ring_bonds) in rings.iter().enumerate() {
        eprintln!("\nRing {}:", ri);
        eprintln!("  Bond indices: {:?}", ring_bonds);
        
        // Collect atoms in this ring
        let mut atoms = std::collections::HashSet::new();
        for &bi in ring_bonds {
            let b = &mol.bonds[bi];
            atoms.insert(b.source as usize);
            atoms.insert(b.target as usize);
        }
        let mut atom_vec: Vec<_> = atoms.into_iter().collect();
        atom_vec.sort();
        eprintln!("  Atoms (sorted): {:?}", atom_vec);
        
        // Show bonds details
        eprintln!("  Bond details:");
        for &bi in ring_bonds {
            let b = &mol.bonds[bi];
            eprintln!("    Bond {}: {} -- {}, type={:?}", bi, b.source, b.target, b.bond_type);
        }
    }
}
