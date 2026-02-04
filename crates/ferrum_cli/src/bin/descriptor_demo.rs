use ferrum_core::desc::Descriptor;
use ferrum_core::desc::molecular_weight::MolecularWeight;
use ferrum_core::desc::atom_count::{AtomCount, HeavyAtomCount, SingleAtomCount, AromaticAtomCount};
use ferrum_core::desc::bond_count::{BondCountDescriptor, RotatableBondCount};
use ferrum_core::desc::aromatic_ring_count::AromaticRingCount;
use ferrum_core::desc::formal_charge::FormalCharge;
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::algo::valency::assign_implicit_hydrogens;
use ferrum_core::algo::aromaticity::perceive_aromaticity;

fn main() {
    println!("🧪 FerrumChem - Basic Descriptors Demo\n");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Test molecules
    let test_cases = vec![
        ("CCO", "Ethanol"),
        ("c1ccccc1", "Benzene"),
        ("c1ccncc1", "Pyridine"),
        ("c1ccc2ccccc2c1", "Naphthalene"),
        ("CCCCCC", "Hexane"),
        ("C=CC=C", "1,3-Butadiene"),
        ("C#N", "Hydrogen Cyanide"),
        ("[NH4+]", "Ammonium Ion"),
        ("[OH-]", "Hydroxide Ion"),
        ("Cc1ccccc1", "Toluene"),
    ];

    for (smiles, name) in test_cases {
        println!("📊 {} ({})", name, smiles);
        println!("───────────────────────────────────────────────────────────────");
        
        match parse_smiles(smiles) {
            Ok(mut mol) => {
                // Process molecule
                assign_implicit_hydrogens(&mut mol).unwrap();
                perceive_aromaticity(&mut mol);
                
                // Calculate descriptors
                let descriptors: Vec<(Box<dyn Descriptor>, usize)> = vec![
                    (Box::new(MolecularWeight), 1),
                    (Box::new(AtomCount), 1),
                    (Box::new(HeavyAtomCount), 1),
                    (Box::new(SingleAtomCount::new("C")), 1),
                    (Box::new(SingleAtomCount::new("N")), 1),
                    (Box::new(SingleAtomCount::new("O")), 1),
                    (Box::new(AromaticAtomCount), 1),
                    (Box::new(BondCountDescriptor::new(false)), 1),
                    (Box::new(BondCountDescriptor::new(true)), 4),
                    (Box::new(RotatableBondCount), 1),
                    (Box::new(AromaticRingCount), 1),
                    (Box::new(FormalCharge), 1),
                ];
                
                for (desc, size) in descriptors {
                    let mut result = vec![0.0; size];
                    desc.calculate(&mol, &mut result);
                    
                    print!("  {:20} ", desc.name());
                    if size == 1 {
                        println!("{:.2}", result[0]);
                    } else if size == 4 {
                        // Detailed bond count: [total, single, double, triple]
                        println!("[Total: {:.0}, Single: {:.0}, Double: {:.0}, Triple: {:.0}]",
                                 result[0], result[1], result[2], result[3]);
                    } else {
                        println!("{:?}", result);
                    }
                }
            }
            Err(e) => {
                println!("  ❌ Error parsing SMILES: {}", e);
            }
        }
        
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("\n✅ Demo complete! All {} descriptors tested.", 12);
    println!("\nDescriptor Summary:");
    println!("  • Molecular Weight - Accurate mass calculation");
    println!("  • Atom Counts - Total, heavy, specific elements");
    println!("  • Aromatic Atoms - Counts atoms in aromatic systems");
    println!("  • Bond Counts - Total and detailed (single/double/triple)");
    println!("  • Rotatable Bonds - Conformational flexibility metric");
    println!("  • Aromatic Rings - Aromatic ring detection");
    println!("  • Formal Charge - Net molecular charge");
    println!("\n🚀 Ready for Phase 5: CSV reading and batch processing!");
}
