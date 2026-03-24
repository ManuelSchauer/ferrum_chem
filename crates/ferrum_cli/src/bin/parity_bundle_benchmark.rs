#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use ferrum_core::desc::batch::BatchDescriptor;
use ferrum_core::io::smiles::parse_smiles;
use std::env;
use std::hint::black_box;
use std::time::Instant;

const DEFAULT_ITERATIONS: usize = 10_000;
const DEFAULT_REPEATS: usize = 3;
const DEFAULT_WARMUP: usize = 200;

struct Case {
    name: &'static str,
    smiles: &'static str,
}

fn bundle_descriptors() -> Vec<BatchDescriptor> {
    vec![
        BatchDescriptor::MolecularWeight,
        BatchDescriptor::LogP,
        BatchDescriptor::MolarRefractivity,
        BatchDescriptor::Tpsa,
        BatchDescriptor::HBondDonor,
        BatchDescriptor::HBondAcceptor,
        BatchDescriptor::RotatableBondCount,
        BatchDescriptor::HeavyAtomCount,
        BatchDescriptor::AtomCount,
        BatchDescriptor::AromaticAtomCount,
        BatchDescriptor::HeteroatomCount,
        BatchDescriptor::AromaticRingCount,
        BatchDescriptor::FormalCharge,
        BatchDescriptor::FractionCSP3,
        BatchDescriptor::BertzCT,
        BatchDescriptor::LabuteAsa,
        BatchDescriptor::Qed,
        BatchDescriptor::NumAmideBonds,
        BatchDescriptor::HallKierAlpha,
        BatchDescriptor::BalabanIndex,
        BatchDescriptor::BondCount,
        BatchDescriptor::SlogpVsa,
        BatchDescriptor::SmrVsa,
        BatchDescriptor::PeoeVsa,
    ]
}

fn bundle_output_len(smiles: &str, descriptors: &[BatchDescriptor]) -> usize {
    let mut mol = parse_smiles(smiles).expect("valid benchmark SMILES");
    let mut total = 0usize;
    let mut ctx = ferrum_core::desc::context::CalculationContext::new();
    for &desc in descriptors {
        total += desc.calculate(&mut mol, &mut ctx).len();
    }
    total
}

fn run_bundle(smiles: &str, descriptors: &[BatchDescriptor]) {
    let mut mol = parse_smiles(smiles).expect("valid benchmark SMILES");
    let mut ctx = ferrum_core::desc::context::CalculationContext::new();
    for &desc in descriptors {
        black_box(desc.calculate(&mut mol, &mut ctx));
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    values[values.len() / 2]
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

fn main() {
    let iterations = env_usize("FERRUM_BENCH_ITERATIONS", DEFAULT_ITERATIONS);
    let repeats = env_usize("FERRUM_BENCH_REPEATS", DEFAULT_REPEATS);
    let warmup = env_usize("FERRUM_BENCH_WARMUP", DEFAULT_WARMUP);

    let cases = [
        Case {
            name: "Ethane",
            smiles: "CC",
        },
        Case {
            name: "Aspirin",
            smiles: "CC(=O)OC1=CC=CC=C1C(=O)O",
        },
        Case {
            name: "Caffeine",
            smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        },
        Case {
            name: "LargeDruglike",
            smiles: "O=C(N[C@@H](OC[C@@H]1CCCN2CCCC[C@H]12)C(Cl)(Cl)Cl)C1=CC=CC2=CC=CC=C12",
        },
    ];

    let descriptors = bundle_descriptors();
    println!(
        "tool\tcase\tsmiles\theavy_atoms\titerations\toutputs\tmedian_seconds\tms_per_molecule\tmolecules_per_second"
    );
    for case in cases {
        let mol = parse_smiles(case.smiles).expect("valid benchmark SMILES");
        let heavy_atoms = mol.num_atoms();
        let outputs = bundle_output_len(case.smiles, &descriptors);
        for _ in 0..warmup {
            run_bundle(case.smiles, &descriptors);
        }

        let mut timings = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            for _ in 0..iterations {
                run_bundle(case.smiles, &descriptors);
            }
            timings.push(start.elapsed().as_secs_f64());
        }

        let median_seconds = median(&mut timings);
        let ms_per_molecule = median_seconds * 1000.0 / iterations as f64;
        let molecules_per_second = iterations as f64 / median_seconds;
        println!(
            "ferrum\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.2}",
            case.name,
            case.smiles,
            heavy_atoms,
            iterations,
            outputs,
            median_seconds,
            ms_per_molecule,
            molecules_per_second
        );
    }
}
