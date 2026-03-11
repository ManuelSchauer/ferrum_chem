use ferrum_core::desc::aromatic_ring_count::AromaticRingCount;
use ferrum_core::desc::atom_count::{AromaticAtomCount, AtomCount, HeavyAtomCount};
use ferrum_core::desc::balaban::BalabanIndex;
use ferrum_core::desc::bond_count::{BondCount, RotatableBondCount};
use ferrum_core::desc::extra_counts::{BertzCT, FractionCSP3, HallKierAlpha, NumAmideBonds};
use ferrum_core::desc::formal_charge::FormalCharge;
use ferrum_core::desc::hbond::{HBondAcceptor, HBondDonor};
use ferrum_core::desc::logp::{LogP, MolarRefractivity};
use ferrum_core::desc::molecular_weight::MolecularWeight;
use ferrum_core::desc::qed::Qed;
use ferrum_core::desc::simple_counts::HeteroatomCount;
use ferrum_core::desc::tpsa::Tpsa;
use ferrum_core::desc::typed::Descriptor;
use ferrum_core::desc::vsa::{LabuteASA, PeoeVsa, SlogpVSA, SmrVSA};
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

fn scalar_descriptors() -> Vec<Box<dyn Descriptor>> {
    vec![
        Box::new(MolecularWeight),
        Box::new(LogP),
        Box::new(MolarRefractivity),
        Box::new(Tpsa),
        Box::new(HBondDonor),
        Box::new(HBondAcceptor),
        Box::new(RotatableBondCount),
        Box::new(HeavyAtomCount),
        Box::new(AtomCount),
        Box::new(AromaticAtomCount),
        Box::new(HeteroatomCount),
        Box::new(AromaticRingCount),
        Box::new(FormalCharge),
        Box::new(FractionCSP3),
        Box::new(BertzCT),
        Box::new(LabuteASA),
        Box::new(Qed),
        Box::new(NumAmideBonds),
        Box::new(HallKierAlpha),
        Box::new(BalabanIndex),
        Box::new(BondCount),
    ]
}

fn bundle_output_len(smiles: &str, scalar: &[Box<dyn Descriptor>]) -> usize {
    let mut mol = parse_smiles(smiles).expect("valid benchmark SMILES");
    let mut total = 0usize;
    for desc in scalar {
        total += desc.calculate(&mut mol).len();
    }
    total += SlogpVSA.calculate(&mut mol).len();
    total += SmrVSA.calculate(&mut mol).len();
    total += PeoeVsa.calculate(&mut mol).len();
    total
}

fn run_bundle(smiles: &str, scalar: &[Box<dyn Descriptor>]) {
    let mut mol = parse_smiles(smiles).expect("valid benchmark SMILES");
    for desc in scalar {
        black_box(desc.calculate(&mut mol));
    }
    black_box(SlogpVSA.calculate(&mut mol));
    black_box(SmrVSA.calculate(&mut mol));
    black_box(PeoeVsa.calculate(&mut mol));
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

    let scalar = scalar_descriptors();
    println!(
        "tool\tcase\tsmiles\theavy_atoms\titerations\toutputs\tmedian_seconds\tms_per_molecule\tmolecules_per_second"
    );
    for case in cases {
        let mol = parse_smiles(case.smiles).expect("valid benchmark SMILES");
        let heavy_atoms = mol.num_atoms();
        let outputs = bundle_output_len(case.smiles, &scalar);

        for _ in 0..warmup {
            run_bundle(case.smiles, &scalar);
        }

        let mut timings = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            for _ in 0..iterations {
                run_bundle(case.smiles, &scalar);
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
