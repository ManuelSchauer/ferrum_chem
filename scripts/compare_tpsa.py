#!/usr/bin/env python3
"""
Compare our TPSA implementation with RDKit for the test molecules.
"""

import subprocess
import json

# Rust test program
rust_code = """
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::desc::tpsa::Tpsa;
use ferrum_core::desc::Descriptor;

fn main() {
    let test_cases = vec![
        ("Butane", "CCCC"),
        ("Ethanol", "CCO"),
        ("Dimethyl ether", "COC"),
        ("Acetone", "CC(=O)C"),
        ("Acetic acid", "CC(=O)O"),
        ("Methyl acetate", "CC(=O)OC"),
        ("Methylamine", "CN"),
        ("Dimethylamine", "CNC"),
        ("Trimethylamine", "CN(C)C"),
        ("Pyridine", "c1cccnc1"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("Acetamide", "CC(=O)N"),
        ("Benzene", "c1ccccc1"),
        ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"),
        ("Metformin", "CN(C)C(=N)NC(=N)N"),
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ];
    
    let desc = Tpsa;
    for (name, smiles) in test_cases {
        match parse_smiles(smiles) {
            Ok(mol) => {
                let mut output = vec![0.0];
                desc.calculate(&mol, &mut output);
                println!("{},{:.2}", name, output[0]);
            }
            Err(e) => {
                eprintln!("{},ERROR: {:?}", name, e);
            }
        }
    }
}
"""

# Write and compile temporary Rust program
with open("/tmp/test_tpsa.rs", "w") as f:
    f.write(rust_code)

print("Compiling Rust test program...")
result = subprocess.run(
    ["rustc", "--edition", "2021", "/tmp/test_tpsa.rs", 
     "-L", "target/debug/deps",
     "--extern", "ferrum_core=target/debug/libferrum_core.rlib",
     "-o", "/tmp/test_tpsa"],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("Compilation failed:")
    print(result.stderr)
    exit(1)

print("\nOur TPSA values:")
print("-" * 40)
result = subprocess.run(["/tmp/test_tpsa"], capture_output=True, text=True)
print(result.stdout)

if result.stderr:
    print("Errors:")
    print(result.stderr)
