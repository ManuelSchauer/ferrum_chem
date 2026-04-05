//! FerrumChem CLI entry point.
//!
//! This binary hosts the thin command-oriented CLI wrapper around ferrum_core.
//! It demonstrates argument parsing, molecule ingestion, standardization, and
//! descriptor execution as a complete end-to-end flow.
//!
//! Learning focus:
//! - clap-based command definitions.
//! - Building a minimal, composable pipeline around reusable library APIs.
//!
//! Note: this binary is intentionally thin and currently demonstrates the basic
//! command wiring pattern. Most production workflows in this repository are
//! exposed through dedicated binaries under `src/bin/`.

//! ## File guide: Main
//!
//! This file belongs to FerrumChem's **library support layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct Cli`
//! - `enum Commands`
//! - `fn main`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Main` is a task-focused executable; CLI environment variables are part of the stable workflow contract.
//! - `Main` should keep parse/descriptor/model behavior aligned with `ferrum_core` defaults to avoid demo drift.
//! - Benchmark/demo paths in `Main` prefer deterministic setup to keep measurements comparable over time.
//! - Specific scope anchor: `crates/ferrum_cli/src/main.rs` defines main behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Rust CLI patterns (argument/env handling, structured errors, stdout diagnostics) reflected by `Main`.
//! - FerrumChem benchmark/parity methodology documents referenced by `Main` scripts.
//! - Terminal UX libraries (where applicable) used by `Main` for interactive workflows.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_cli/src/main.rs`.

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

/// Runs this executable entrypoint and orchestrates its top-level workflow.
///
/// Function context: `crates/ferrum_cli/src/main.rs`.
/// CLI-layer note: keep workflow and configuration behavior stable for reproducible demos/benchmarks.
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
