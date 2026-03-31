use clap::{Parser, Subcommand};
use ferrum_core::desc::Descriptor;
use ferrum_core::{algo, desc, io};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "FerrumChem")]
#[command(about = "High-Performance QSAR Engine in Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Berechnet Deskriptoren für eine Datei
    Calc {
        /// Input Datei (.smi oder .sdf)
        #[arg(short, long)]
        input: PathBuf,

        /// Output Datei (.csv)
        #[arg(short, long)]
        output: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Calc { input, output } => {
            println!("Calculating descriptors for {:?}", input);

            // 1. Datei einlesen (hier nur Dummy Logik)
            let raw_mol = io::smiles::parse_smiles("c1ccccc1").unwrap();

            // 2. Standardisieren
            let mut mol = algo::standardize::process(raw_mol).unwrap();

            // 3. Rechnen
            let mw = desc::molecular_weight::MolecularWeight;
            let result = mw.calculate(
                &mut mol,
                &mut ferrum_core::desc::context::CalculationContext::new(),
            );

            println!("Ergebnis: {:?}", result);
        }
    }
}
