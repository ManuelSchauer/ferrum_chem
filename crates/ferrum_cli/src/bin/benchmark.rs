use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::desc::{Descriptor, molecular_weight};
use std::time::Instant;

fn main() {
    println!("=== FerrumChem Performance Benchmarks ===\n");

    // Test molecules of varying complexity
    let test_smiles = vec![
        ("Methane", "C"),
        ("Ethanol", "CCO"),
        ("Benzene", "c1ccccc1"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("Complex", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O.CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ];

    println!("--- SMILES Parsing Performance (with automatic implicit H) ---");
    for (name, smiles) in &test_smiles {
        let iterations = 10000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = parse_smiles(smiles).unwrap();
        }

        let duration = start.elapsed();
        let per_parse = duration.as_nanos() / iterations;
        let per_second = 1_000_000_000.0 / per_parse as f64;

        println!(
            "{:12} ({:2} chars): {:6} ns/parse  ({:8.0} molecules/sec)",
            name, smiles.len(), per_parse, per_second
        );
    }

    println!("\n--- Molecular Weight Calculation ---");
    let descriptor = molecular_weight::MolecularWeight;

    for (name, smiles) in &test_smiles {
        let mol = parse_smiles(smiles).unwrap();

        let iterations = 100000;
        let mut result = vec![0.0; 1];

        let start = Instant::now();
        for _ in 0..iterations {
            descriptor.calculate(&mol, &mut result);
        }

        let duration = start.elapsed();
        let per_calc = duration.as_nanos() / iterations;
        let per_second = 1_000_000_000.0 / per_calc as f64;

        println!(
            "{:12}: MW = {:7.2}, {:5} ns/calc  ({:9.0} calcs/sec)",
            name, result[0], per_calc, per_second
        );
    }

    println!("\n--- Combined Pipeline (Parse + Calculate MW) ---");
    for (name, smiles) in &test_smiles {
        let iterations = 10000;
        let mut result = vec![0.0; 1];

        let start = Instant::now();
        for _ in 0..iterations {
            let mol = parse_smiles(smiles).unwrap();
            descriptor.calculate(&mol, &mut result);
        }

        let duration = start.elapsed();
        let per_molecule = duration.as_nanos() / iterations;
        let per_second = 1_000_000_000.0 / per_molecule as f64;

        println!(
            "{:12}: {:6} ns/molecule  ({:8.0} molecules/sec)",
            name, per_molecule, per_second
        );
    }

    println!("\n--- Batch Processing Simulation ---");
    // Simulate processing a large dataset
    let dataset_sizes = vec![1000, 10000, 100000];
    let sample_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"; // Ibuprofen

    for size in dataset_sizes {
        let start = Instant::now();

        for _ in 0..size {
            let mol = parse_smiles(sample_smiles).unwrap();
            let mut result = vec![0.0; 1];
            descriptor.calculate(&mol, &mut result);
        }

        let duration = start.elapsed();
        let per_molecule = duration.as_micros() as f64 / size as f64;
        let throughput = size as f64 / duration.as_secs_f64();

        println!(
            "{:6} molecules: {:7.2} ms total, {:6.2} μs/mol, {:9.0} mol/sec",
            size,
            duration.as_secs_f64() * 1000.0,
            per_molecule,
            throughput
        );
    }

    println!("\n--- Memory Usage Estimate ---");
    for (name, smiles) in &test_smiles {
        let mol = parse_smiles(smiles).unwrap();
        let atoms = mol.num_atoms();
        let bonds = mol.num_bonds();

        // Rough estimate:
        // - Atom struct: 8 bytes
        // - Bond struct: 6 bytes
        // - Adjacency list: ~8 bytes/atom (SmallVec overhead)
        let estimated_bytes = atoms * 16 + bonds * 6;

        println!(
            "{:12}: {} atoms, {} bonds → ~{} bytes",
            name, atoms, bonds, estimated_bytes
        );
    }
}

