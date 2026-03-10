use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::algo::aromaticity::perceive_aromaticity;

#[test]
fn debug_ring_0_aromatic() {
    let smiles = "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2";
    let mut mol = parse_smiles(smiles).unwrap();
    
    eprintln!("\n=== Before aromaticity ===");
    eprintln!("Ring 0 atoms [8, 3, 51, 52, 4, 44] properties:");
    for &idx in &[8, 3, 51, 52, 4, 44] {
        let a = &mol.atoms[idx];
        eprintln!("  Atom {}: Z={}, arom={}, h={}, chg={}, bonds=[", idx, 
            a.atomic_number, a.is_aromatic(), a.total_h(), a.formal_charge);
        
        for &bi in &mol.adjacency[idx] {
            let b = &mol.bonds[bi as usize];
            let other = if b.source as usize == idx { b.target } else { b.source };
            eprintln!("    bond {} to atom {}: type={:?}, arom={}", bi, other, b.bond_type, b.is_aromatic());
        }
        eprintln!("  ]");
    }
    
    // Check specific bonds in ring 0
    eprintln!("\nRing 0 bonds (consecutive pairs):");
    let ring = vec![8, 3, 51, 52, 4, 44];
    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];
        eprintln!("  Edge {}: {} -- {}", i, from, to);
        
        // Find bond
        for &bi in &mol.adjacency[from] {
            let b = &mol.bonds[bi as usize];
            if (b.source == from as u16 && b.target == to as u16) || 
               (b.source == to as u16 && b.target == from as u16) {
                eprintln!("    Bond {}: type={:?}, arom={}", bi, b.bond_type, b.is_aromatic());
            }
        }
    }
    
    perceive_aromaticity(&mut mol);
    
    eprintln!("\n=== After aromaticity ===");
    eprintln!("Ring 0 atoms [8, 3, 51, 52, 4, 44] are aromatic:");
    for &idx in &[8, 3, 51, 52, 4, 44] {
        let a = &mol.atoms[idx];
        eprintln!("  Atom {}: arom={}", idx, a.is_aromatic());
    }
    
    eprintln!("\nRing 0 bonds (consecutive pairs) are aromatic:");
    let ring = vec![8, 3, 51, 52, 4, 44];
    for i in 0..ring.len() {
        let from = ring[i];
        let to = ring[(i + 1) % ring.len()];
        
        // Find bond
        for &bi in &mol.adjacency[from] {
            let b = &mol.bonds[bi as usize];
            if (b.source == from as u16 && b.target == to as u16) || 
               (b.source == to as u16 && b.target == from as u16) {
                eprintln!("  Edge {}: {} -- {} [bond {}] arom={}", i, from, to, bi, b.is_aromatic());
            }
        }
    }
}
