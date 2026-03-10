use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::types::BondType;

#[test]
fn trace_ring5_aromaticity() {
    let smiles = "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2";
    let mol = parse_smiles(smiles).unwrap();
    
    eprintln!("\n=== Analyzing Ring 5: atoms [3, 4, 8, 44, 51, 52] ===");
    let ring = vec![3, 4, 8, 44, 51, 52];
    
    eprintln!("\nAtom properties:");
    for &ai in &ring {
        let a = &mol.atoms[ai];
        let pi_bonds = mol.adjacency[ai].iter().filter(|&&bi| {
            let b = &mol.bonds[bi as usize];
            matches!(b.bond_type, BondType::Double | BondType::Aromatic)
        }).count();
        eprintln!("  Atom {}: Z={}, h={}, chg={}, pi_bonds={}", ai, a.atomic_number, a.total_h(), a.formal_charge, pi_bonds);
    }
    
    eprintln!("\nRing edges (consecutive pairs):");
    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];
        eprintln!("  {} -- {}: ", from, to);
        
        // Find bond between them
        let mut found_bond = false;
        for &bi in &mol.adjacency[from] {
            let b = &mol.bonds[bi as usize];
            if (b.source as usize == from && b.target as usize == to) ||
               (b.source as usize == to && b.target as usize == from) {
                eprintln!("    Bond {}: type={:?}", bi, b.bond_type);
                found_bond = true;
                break;
            }
        }
        if !found_bond {
            eprintln!("    NO DIRECT BOND FOUND!");
        }
    }
    
    eprintln!("\nIs this a valid ring?");
    eprintln!("  All consecutive atoms connected: checking...");
    let mut all_connected = true;
    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];
        let mut found = false;
        for &bi in &mol.adjacency[from] {
            let b = &mol.bonds[bi as usize];
            if (b.source as usize == from && b.target as usize == to) ||
               (b.source as usize == to && b.target as usize == from) {
                found = true;
                break;
            }
        }
        if !found {
            all_connected = false;
            eprintln!("    MISSING: {} -- {}", from, to);
        }
    }
    eprintln!("  Result: {}", if all_connected { "valid ring" } else { "INVALID RING!" });
}
