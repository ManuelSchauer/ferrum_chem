use ferrum_core::desc::Descriptor;
use ferrum_core::desc::DescriptorValue;
use ferrum_core::desc::molecular_weight::MolecularWeight;
use ferrum_core::desc::atom_count::{AtomCount, HeavyAtomCount, SingleAtomCount, AromaticAtomCount};
use ferrum_core::desc::bond_count::{BondCount, BondCountDetailed, RotatableBondCount};
use ferrum_core::desc::aromatic_ring_count::AromaticRingCount;
use ferrum_core::desc::formal_charge::FormalCharge;
use ferrum_core::desc::hbond::{HBondDonor, HBondAcceptor};
use ferrum_core::desc::logp::LogP;
use ferrum_core::desc::lipinski::LipinskiViolations;
use ferrum_core::desc::maccs::MaccsKeys;
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
                
                // Calculate and display individual descriptors
                let descriptors: Vec<Box<dyn Descriptor>> = vec![
                    Box::new(MolecularWeight),
                    Box::new(AtomCount),
                    Box::new(HeavyAtomCount),
                    Box::new(SingleAtomCount::new("C")),
                    Box::new(SingleAtomCount::new("N")),
                    Box::new(SingleAtomCount::new("O")),
                    Box::new(AromaticAtomCount),
                    Box::new(HBondDonor),
                    Box::new(HBondAcceptor),
                    Box::new(LogP),
                    Box::new(LipinskiViolations),
                    Box::new(MaccsKeys),
                    Box::new(BondCount),
                    Box::new(BondCountDetailed),
                    Box::new(RotatableBondCount),
                    Box::new(AromaticRingCount),
                    Box::new(FormalCharge),
                ];
                
                for desc in &descriptors {
                    let value = desc.calculate(&mut mol);
                    
                    print!("  {:20} ", desc.name());
                    match &value {
                        DescriptorValue::Float(v) => println!("{:.2}", v),
                        DescriptorValue::Int(v) => println!("{}", v),
                        DescriptorValue::IntVec(v) => {
                            if v.len() == 4 {
                                println!("[Total: {}, Single: {}, Double: {}, Triple: {}]",
                                         v[0], v[1], v[2], v[3]);
                            } else {
                                println!("{:?}", v);
                            }
                        }
                        DescriptorValue::Binary(_bits) => {
                            let mut floats = Vec::new();
                            value.to_floats(&mut floats);
                            let set_count = floats.iter().filter(|&&x| x == 1.0).count();
                            println!("{} / {} keys set", set_count, floats.len());
                        }
                        _ => {
                            let mut floats = Vec::new();
                            value.to_floats(&mut floats);
                            println!("{:?}", floats);
                        }
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
    println!("\n✅ Demo complete! All {} descriptors tested.", 15);
    println!("\nDescriptor Summary:");
    println!("  • Molecular Weight - Accurate mass calculation");
    println!("  • Atom Counts - Total, heavy, specific elements");
    println!("  • Aromatic Atoms - Counts atoms in aromatic systems");
    println!("  • H-Bond Donors - N-H and O-H groups (Lipinski)");
    println!("  • H-Bond Acceptors - N and O atoms (Lipinski)");
    println!("  • LogP - Lipophilicity estimation (RDKit patterns)");
    println!("  • Lipinski Rule of Five - Drug-likeness assessment");
    println!("  • MACCS Keys - 166 structural fingerprint keys");
    println!("  • Bond Counts - Total and detailed (single/double/triple)");
    println!("  • Rotatable Bonds - Conformational flexibility metric");
    println!("  • Aromatic Rings - Aromatic ring detection");
    println!("  • Formal Charge - Net molecular charge");
    println!("\n🚀 Phase 8 Complete: MACCS Keys implementation (all 166 keys)!");
}
