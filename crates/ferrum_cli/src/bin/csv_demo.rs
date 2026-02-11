use ferrum_core::io::csv_reader::CsvReader;
use ferrum_core::algo::valency::assign_implicit_hydrogens;
use ferrum_core::algo::aromaticity::perceive_aromaticity;
use ferrum_core::desc::Descriptor;
use ferrum_core::desc::batch::{calculate_batch, write_csv, write_json};
use ferrum_core::desc::molecular_weight::MolecularWeight;
use ferrum_core::desc::atom_count::{AtomCount, HeavyAtomCount, AromaticAtomCount};
use ferrum_core::desc::bond_count::BondCount;
use ferrum_core::desc::aromatic_ring_count::AromaticRingCount;
use ferrum_core::desc::formal_charge::FormalCharge;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("🧪 FerrumChem - Phase 5: CSV Batch Processing Demo\n");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create a sample CSV file for demonstration
    let csv_content = "\
SMILES
CCO
c1ccccc1
c1ccncc1
CCCCCC
C=CC=C
[NH4+]
c1ccc2ccccc2c1
Cc1ccccc1
INVALID_SMILES
CC(C)CC
";

    // Write to temporary file
    let temp_file = "/tmp/ferrum_test_molecules.csv";
    let mut file = File::create(temp_file).expect("Failed to create temp file");
    write!(file, "{}", csv_content).expect("Failed to write CSV");
    drop(file);
    
    println!("📄 Step 1: Reading CSV file...");
    println!("   File: {}", temp_file);
    println!("   Format: Just SMILES (no IDs needed)");
    
    // Read and parse CSV
    let reader = CsvReader::from_path(temp_file)
        .expect("Failed to create reader")
        .with_smiles_column("SMILES");
    
    let stats = reader.parse_with_stats()
        .expect("Failed to parse CSV");
    
    println!();
    stats.print_summary();
    println!();
    
    // Extract valid molecules with their SMILES
    let mut molecules = Vec::new();
    let mut smiles_list = Vec::new();
    
    for record in stats.records {
        if let Some(mut mol) = record.molecule {
            // Preprocess molecule
            assign_implicit_hydrogens(&mut mol).ok();
            perceive_aromaticity(&mut mol);
            
            molecules.push(mol);
            smiles_list.push(record.smiles);
        }
    }
    
    println!("✅ Successfully parsed {} molecules\n", molecules.len());
    
    // Define descriptors to calculate
    println!("📊 Step 2: Calculating descriptors...");
    
    let descriptors: Vec<Box<dyn Descriptor>> = vec![
        Box::new(MolecularWeight),
        Box::new(AtomCount),
        Box::new(HeavyAtomCount),
        Box::new(AromaticAtomCount),
        Box::new(BondCount),
        Box::new(AromaticRingCount),
        Box::new(FormalCharge),
    ];
    
    println!("   Descriptors: {}", descriptors.len());
    for desc in &descriptors {
        println!("   - {}", desc.name());
    }
    println!();
    
    // Calculate descriptors in parallel
    let matrix = calculate_batch(&mut molecules, &descriptors);
    
    println!("✅ Calculated {} x {} descriptor matrix\n", 
             matrix.len(), matrix[0].len());
    
    // Display sample results
    println!("📈 Step 3: Sample Results (first 3 molecules):");
    println!("───────────────────────────────────────────────────────────────");
    
    for i in 0..3.min(matrix.len()) {
        println!("\n{} (row {}):", smiles_list[i], i + 1);
        for (desc, &value) in descriptors.iter().zip(&matrix[i]) {
            println!("  {:25} {:.2}", desc.name(), value);
        }
    }
    println!();
    
    // Write CSV output
    println!("💾 Step 4: Writing output files...");
    
    let csv_output = "/tmp/ferrum_descriptors.csv";
    let mut csv_file = File::create(csv_output).expect("Failed to create CSV output");
    write_csv(&mut csv_file, &smiles_list, &descriptors, &matrix)
        .expect("Failed to write CSV");
    println!("   ✓ CSV: {}", csv_output);
    
    // Write JSON output
    let json_output = "/tmp/ferrum_descriptors.json";
    let mut json_file = File::create(json_output).expect("Failed to create JSON output");
    write_json(&mut json_file, &smiles_list, &descriptors, &matrix)
        .expect("Failed to write JSON");
    println!("   ✓ JSON: {}", json_output);
    
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("\n✨ Phase 5 Complete!");
    println!("\nFeatures demonstrated:");
    println!("  ✅ CSV file reading (SMILES only, no IDs required)");
    println!("  ✅ Parallel SMILES parsing");
    println!("  ✅ Error handling (invalid SMILES)");
    println!("  ✅ Batch descriptor calculation");
    println!("  ✅ CSV output with SMILES as identifier");
    println!("  ✅ JSON output with SMILES");
    println!("\nOutput format:");
    println!("  SMILES,MolecularWeight,AtomCount,...");
    println!("  CCO,46.07,3.00,...");
    println!("  c1ccccc1,78.11,6.00,...");
    println!("\nOutput files:");
    println!("  - {}", csv_output);
    println!("  - {}", json_output);
    println!("\n🚀 Ready for production ML workflows!");
}
