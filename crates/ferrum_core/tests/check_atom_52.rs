use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::types::BondType;

#[test]
fn check_atom_52() {
    let smiles = "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2";
    let mol = parse_smiles(smiles).unwrap();
    
    eprintln!("\n=== Analyzing Atom 52 (N in Ring 5) ===");
    let ring: Vec<u16> = vec![3, 4, 8, 44, 51, 52];
    
    let a52 = &mol.atoms[52];
    eprintln!("Atom 52: Z={}, total_h={}, charge={}, degree={}", 
        a52.atomic_number, a52.total_h(), a52.formal_charge, mol.get_degree(52));
    
    eprintln!("\nBonds from atom 52:");
    for &bi in &mol.adjacency[52] {
        let b = &mol.bonds[bi as usize];
        let other = if b.source == 52 { b.target } else { b.source } as u16;
        let in_ring = ring.contains(&other);
        eprintln!("  Bond {}: 52 -- {}, type={:?}, in_ring={}", bi, other, b.bond_type, in_ring);
    }
    
    eprintln!("\nhas_ring_pi_bond check (bonds with pi type WITHIN ring):");
    let has_ring_pi = mol.adjacency[52].iter().any(|&bi| {
        let b = &mol.bonds[bi as usize];
        if !matches!(b.bond_type, BondType::Double | BondType::Aromatic) {
            return false;
        }
        let other = if b.source == 52 { b.target } else { b.source };
        ring.contains(&(other as u16))
    });
    
    eprintln!("  has_ring_pi_bond = {}", has_ring_pi);
    
    eprintln!("\nConclusion: Atom 52 (N) with ring_pi=true should count as:");
    if has_ring_pi {
        let deg = mol.get_degree(52);
        if deg >= 3 {
            eprintln!("  2π (degree >= 3)");
        } else {
            eprintln!("  AMBIGUOUS [1 or 2] (degree < 3)");
        }
    } else {
        eprintln!("  AMBIGUOUS [1 or 2] (no ring_pi)");
    }
}
