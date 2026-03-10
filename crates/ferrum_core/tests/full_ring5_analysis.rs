use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::types::BondType;
use std::collections::HashSet;

#[test]
fn full_ring5_analysis() {
    let smiles = "CC1=CC2=C(C=C1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N3C=NC4=C(N)N=CN=C43)[C@H](O)[C@@H]1O)C1=NC(=O)N=C(O)C1=N2";
    let mol = parse_smiles(smiles).unwrap();
    
    eprintln!("\n=== Complete Ring 5 Analysis [3-4-8-44-51-52] ===");
    let ring: Vec<u16> = vec![3, 4, 8, 44, 51, 52];
    
    eprintln!("\n1. Conjugation Check:");
    let mut conjugated = true;
    for i in 0..ring.len() {
        let from = ring[i] as usize;
        let to = ring[(i + 1) % ring.len()] as usize;
        
        let mut found = false;
        let mut bond_type = "NOT FOUND";
        
        for &bi in &mol.adjacency[from] {
            let b = &mol.bonds[bi as usize];
            if (b.source as usize == from && b.target as usize == to) ||
               (b.source as usize == to && b.target as usize == from) {
                found = true;
                bond_type = match b.bond_type {
                    BondType::Single => "Single",
                    BondType::Double => "Double",
                    BondType::Aromatic => "Aromatic",
                    BondType::Triple => "Triple",
                    _ => "Other",
                };
                if b.bond_type == BondType::Triple {
                    conjugated = false;
                }
                break;
            }
        }
        
        eprintln!("  {} -- {}: {} {}", from, to, bond_type, if !found { "[NOT FOUND!]" } else { "" });
    }
    eprintln!("  Result: {}", if conjugated { "CONJUGATED ✓" } else { "NOT CONJUGATED ✗" });
    
    eprintln!("\n2. π-Electron Count:");
    let mut fixed_pi: u8 = 0;
    let mut n_ambiguous = 0;
    
    for &ai in &ring {
        let ai = ai as usize;
        let a = &mol.atoms[ai];
        
        match a.atomic_number {
            6 => {
                let has_any_pi = mol.adjacency[ai].iter().any(|&bi| {
                    let b = &mol.bonds[bi as usize];
                    matches!(b.bond_type, BondType::Double | BondType::Aromatic)
                });
                if has_any_pi {
                    fixed_pi += 1;
                    eprintln!("  Atom {}: C (neutral, sp2) → 1π", ai);
                } else {
                    eprintln!("  Atom {}: C (sp3?) → 0π", ai);
                }
            }
            7 => {
                let has_h = a.total_h() > 0;
                let has_ring_pi = ring.iter().any(|&o| {
                    mol.adjacency[ai].iter().any(|&bi| {
                        let b = &mol.bonds[bi as usize];
                        matches!(b.bond_type, BondType::Double | BondType::Aromatic) &&
                        ((b.source as usize == ai && b.target as usize == o as usize) ||
                         (b.target as usize == ai && b.source as usize == o as usize))
                    })
                });
                
                if has_h {
                    eprintln!("  Atom {}: N (with H) → 2π", ai);
                    fixed_pi += 2;
                } else if has_ring_pi {
                    let deg = mol.get_degree(ai as u16);
                    if deg >= 3 {
                        eprintln!("  Atom {}: N (degree {}, ring_pi) → 2π", ai, deg);
                        fixed_pi += 2;
                    } else {
                        eprintln!("  Atom {}: N (degree {}, ring_pi) → ambiguous [1 or 2]", ai, deg);
                        n_ambiguous += 1;
                    }
                } else {
                    eprintln!("  Atom {}: N (neutral, no ring_pi) → ambiguous [1 or 2]", ai);
                    n_ambiguous += 1;
                }
            }
            _ => {
                eprintln!("  Atom {}: Z={} → INVALID", ai, a.atomic_number);
            }
        }
    }
    
    eprintln!("\n  Fixed π: {}", fixed_pi);
    eprintln!("  Ambiguous N count: {}", n_ambiguous);
    eprintln!("  Possible totals:");
    let mut totals = vec![fixed_pi];
    for _ in 0..n_ambiguous {
        let mut next = Vec::new();
        for &t in &totals {
            next.push(t + 1);
            next.push(t + 2);
        }
        next.sort_unstable();
        next.dedup();
        totals = next;
    }
    
    for &t in &totals {
        let huckel = (t as i32 - 2) % 4 == 0 && t >= 2;
        eprintln!("    {} electrons: {} {}", t, if huckel { "✓ Hückel (4n+2)" } else { "✗ Not Hückel" }, if huckel { "AROMATIC" } else { "" });
    }
    
    eprintln!("\n3. Chord Check:");
    eprintln!("  Checking for internal chords in ring...");
    // For simplicity, just report atoms
    let ring_set: HashSet<u16> = ring.iter().copied().collect();
    let mut has_chord = false;
    for &ai in &ring {
        for &bi in &mol.adjacency[ai as usize] {
            let b = &mol.bonds[bi as usize];
            let other = if b.source == ai { b.target } else { b.source };
            
            if !ring_set.contains(&other) {
                continue;
            }
            
            // Check if it's a ring edge
            let is_consecutive = (0..ring.len()).any(|i| {
                let from = ring[i];
                let to = ring[(i + 1) % ring.len()];
                (from == ai && to == other) || (from == other && to == ai)
            });
            
            if !is_consecutive {
                eprintln!("  Found chord: {} -- {}", ai, other);
                has_chord = true;
            }
        }
    }
    if !has_chord {
        eprintln!("  No chords found ✓");
    }
}
