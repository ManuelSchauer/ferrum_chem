use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::io::smarts::parse_smarts;
use ferrum_core::algo::iso::{has_substruct_match, count_matches};
use ferrum_core::algo::aromaticity;

fn main() {
    println!("=== SMARTS Pattern Matching Demo ===\n");
    
    // Test molecules
    let test_mols = vec![
        ("CCO", "Ethanol"),
        ("CC(C)O", "Isopropanol"),
        ("c1ccccc1", "Benzene"),
        ("c1ccccc1O", "Phenol"),
        ("C=O", "Formaldehyde"),
        ("CC(=O)C", "Acetone"),
        ("CC(=O)O", "Acetic acid"),
        ("CCN", "Ethylamine"),
    ];
    
    // Test patterns
    let patterns = vec![
        ("[C]", "Any carbon"),
        ("[O]", "Any oxygen"),
        ("CO", "Carbon-oxygen"),
        ("C=O", "Carbonyl"),
        ("[c]", "Aromatic carbon"),
        ("[OH]", "Hydroxyl (explicit H)"),
        ("c1ccccc1", "Benzene ring"),
    ];
    
    println!("Parsing {} molecules and {} patterns...\n", test_mols.len(), patterns.len());
    
    // Parse all molecules
    let mut molecules: Vec<_> = test_mols.iter()
        .filter_map(|(smiles, name)| {
            match parse_smiles(smiles) {
                Ok(mut mol) => {
                    // Perceive aromaticity for aromatic patterns
                    aromaticity::perceive_aromaticity(&mut mol);
                    Some((mol, *name))
                }
                Err(e) => {
                    eprintln!("Failed to parse {}: {}", smiles, e);
                    None
                }
            }
        })
        .collect();
    
    // Parse all patterns
    let query_patterns: Vec<_> = patterns.iter()
        .filter_map(|(smarts, desc)| {
            match parse_smarts(smarts) {
                Ok(pattern) => Some((pattern, *desc)),
                Err(e) => {
                    eprintln!("Failed to parse SMARTS {}: {}", smarts, e);
                    None
                }
            }
        })
        .collect();
    
    println!("Successfully parsed {} molecules and {} patterns\n", 
             molecules.len(), query_patterns.len());
    
    // Create match matrix
    println!("{:<20} {}", "Pattern", 
             test_mols.iter().map(|(_, name)| format!("{:>12}", name)).collect::<Vec<_>>().join(" "));
    println!("{}", "-".repeat(120));
    
    for (pattern, desc) in &query_patterns {
        print!("{:<20}", desc);
        
        for (mol, _) in &molecules {
            let count = count_matches(mol, pattern);
            if count > 0 {
                print!(" {:>12}", count);
            } else {
                print!(" {:>12}", "-");
            }
        }
        println!();
    }
    
    println!("\n=== Detailed Analysis ===\n");
    
    // Show detailed matches for specific patterns
    let hydroxyl = parse_smarts("CO").unwrap();
    println!("Molecules containing C-O:");
    for (mol, name) in &molecules {
        if has_substruct_match(mol, &hydroxyl) {
            let count = count_matches(mol, &hydroxyl);
            println!("  {} - {} matches", name, count);
        }
    }
    
    println!("\nMolecules containing carbonyl (C=O):");
    let carbonyl = parse_smarts("C=O").unwrap();
    for (mol, name) in &molecules {
        if has_substruct_match(mol, &carbonyl) {
            println!("  {}", name);
        }
    }
    
    println!("\nAromatic molecules:");
    let aromatic_carbon = parse_smarts("[c]").unwrap();
    for (mol, name) in &molecules {
        if has_substruct_match(mol, &aromatic_carbon) {
            let count = count_matches(mol, &aromatic_carbon);
            println!("  {} - {} aromatic carbons", name, count);
        }
    }
    
    println!("\n=== Performance Test ===\n");
    
    // Simple performance benchmark
    let test_mol = parse_smiles("c1ccccc1").unwrap();
    let simple_pattern = parse_smarts("[C]").unwrap();
    
    let iterations = 100_000;
    let start = std::time::Instant::now();
    
    for _ in 0..iterations {
        let _ = has_substruct_match(&test_mol, &simple_pattern);
    }
    
    let elapsed = start.elapsed();
    let matches_per_sec = iterations as f64 / elapsed.as_secs_f64();
    
    println!("Simple pattern matching: {:.0} matches/sec", matches_per_sec);
    println!("Time per match: {:.2} µs", elapsed.as_micros() as f64 / iterations as f64);
    
    println!("\nDemo complete!");
}
