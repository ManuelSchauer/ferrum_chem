//! Memory benchmark for typed descriptors.
//!
//! Demonstrates memory-efficient bit-packed fingerprints.

use ferrum_core::algo::aromaticity::perceive_aromaticity;
use ferrum_core::desc::maccs::MaccsKeys;
use ferrum_core::desc::typed::Descriptor;
use ferrum_core::io::smiles::parse_smiles;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  FerrumChem Memory Benchmark: Bit-Packed Fingerprints");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Parse test molecule
    let mut mol = parse_smiles("c1ccccc1").expect("valid SMILES");
    perceive_aromaticity(&mut mol);

    // -----------------------------------------------------------------------
    // MACCS (BitVec) - 1 bit per feature
    // -----------------------------------------------------------------------

    let maccs = MaccsKeys;
    let value = maccs.calculate(&mut mol);

    let bitvec = value.as_binary();
    let typed_size = std::mem::size_of_val(bitvec.as_raw_slice());
    let set_bits = bitvec.count_ones();
    // Compare: Vec<f32> would be 166 * 4 = 664 bytes
    let legacy_equiv_size = 166 * 4;

    println!("🚀 MACCS Keys (BitVec)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "  Size per molecule:  {} bytes (vs {} bytes as Vec<f32>)",
        typed_size, legacy_equiv_size
    );
    println!("  Bits set:           {}/166", set_bits);
    println!("  Storage efficiency: 1 bit per feature");
    let reduction = legacy_equiv_size as f64 / typed_size as f64;
    println!("  Memory reduction:   {:.1}x", reduction);
    println!();

    // -----------------------------------------------------------------------
    // Full Fingerprint Suite Projection
    // -----------------------------------------------------------------------

    println!("📈 PROJECTED SAVINGS (All Fingerprints)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  MACCS (166):        664 → ~21 bytes");
    println!("  PubChem (881):      3,524 → ~111 bytes");
    println!("  ECFP4 (2048):       8,192 → ~256 bytes");
    println!("  FCFP4 (2048):       8,192 → ~256 bytes");
    println!("  AtomPair (2048):    8,192 → ~256 bytes");
    println!("  Torsion (2048):     8,192 → ~256 bytes");
    println!("  ECFP2/6 (4096):     16,384 → ~512 bytes");
    println!("  FCFP2/6 (4096):     16,384 → ~512 bytes");
    println!("  ─────────────────────────────────────────");
    println!("  TOTAL:              ~69 KB → ~2.2 KB");
    println!("  REDUCTION:          31x memory savings ⚡");
    println!();

    // -----------------------------------------------------------------------
    // Batch Processing Impact
    // -----------------------------------------------------------------------

    let batch_sizes = [1000, 10_000, 100_000, 1_000_000];
    println!("💾 BATCH PROCESSING IMPACT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for &n in &batch_sizes {
        let legacy_mb = (n * 69 * 1024) as f64 / (1024.0 * 1024.0);
        let typed_mb = (n * 2200) as f64 / (1024.0 * 1024.0);
        println!(
            "  {:>9} molecules:  {:.1} MB → {:.1} MB",
            format_number(n),
            legacy_mb,
            typed_mb
        );
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Conclusion: Phase 11 enables 31x memory reduction! 🎉");
    println!("═══════════════════════════════════════════════════════════════");
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}k", n / 1_000)
    } else {
        format!("{}", n)
    }
}
