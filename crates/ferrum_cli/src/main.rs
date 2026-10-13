use clap::{Parser, Subcommand, ValueEnum};
use ferrum_core::desc::table::FingerprintFormat;
use ferrum_cli::calc::run_calc;
use ferrum_cli::tui::run_tui;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ferrum_cli")]
#[command(author = "Manuel Schauer <info@ferrumchem.com>")]
#[command(version = "0.1.0")]
#[command(about = "FerrumChem - High-Performance Molecular Descriptor Engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Calculate descriptors for input molecules (.smi, .csv, or .sdf) and export to CSV
    Calc {
        /// Input dataset (.smi, .csv, or .sdf)
        #[arg(short, long)]
        input: PathBuf,

        /// Output CSV destination
        #[arg(short, long)]
        output: PathBuf,

        /// How fingerprints are written: one hex column per fingerprint
        /// (little-endian bit order), space-separated on-bit indices, one 0/1
        /// column per bit, or no fingerprints
        #[arg(long, value_enum, default_value_t = FpFormat::Hex)]
        fingerprints: FpFormat,

        /// Records processed per chunk (memory scales with this)
        #[arg(long, default_value_t = 4096)]
        chunk_size: usize,
    },
    /// Launch the interactive Terminal User Interface (TUI)
    Tui,
}

#[derive(Clone, Copy, ValueEnum)]
enum FpFormat {
    Hex,
    Onbits,
    Bits,
    None,
}

impl From<FpFormat> for FingerprintFormat {
    fn from(f: FpFormat) -> Self {
        match f {
            FpFormat::Hex => FingerprintFormat::Hex,
            FpFormat::Onbits => FingerprintFormat::OnBits,
            FpFormat::Bits => FingerprintFormat::Bits,
            FpFormat::None => FingerprintFormat::None,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Calc { input, output, fingerprints, chunk_size }) => {
            if let Err(e) = run_calc(&input, &output, fingerprints.into(), chunk_size) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        Some(Commands::Tui) | None => {
            run_tui()?;
        }
    }

    Ok(())
}
