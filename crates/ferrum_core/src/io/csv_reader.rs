//! I/O module: Csv Reader.
//!
//! This module handles external data interchange (SMILES/SMARTS/CSV/SDF) and
//! conversion into FerrumChem graph types. It is the boundary between textual
//! chemical formats and internal, typed molecular structures.
//!
//! Learning focus:
//! - Parser/state-machine techniques in Rust.
//! - Error-aware ingestion pipelines for large datasets.

//! ## File guide: Csv Reader
//!
//! This file belongs to FerrumChem's **input/output parsing layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct CsvReader`
//! - `struct MoleculeRecord`
//! - `fn from_path`
//! - `fn with_smiles_column`
//! - `fn with_delimiter`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Csv Reader` treats textual chemistry formats as untrusted input and prefers explicit parse errors over silent repair.
//! - `Csv Reader` must preserve deterministic tokenization so repeated reads produce identical molecule/query graphs.
//! - Format-specific edge cases in `Csv Reader` are validated against toolkit-compatible semantics where possible.
//! - Specific scope anchor: `crates/ferrum_core/src/io/csv_reader.rs` defines csv reader behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - SMILES/SMARTS/CTfile family specifications relevant to `Csv Reader` parser or reader behavior.
//! - RDKit parsing behavior used as external compatibility reference for `Csv Reader`.
//! - Rust parser/state-machine patterns for reliable streaming and error reporting.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/io/csv_reader.rs`.

use crate::error::{FerrumError, Result};
use crate::io::smiles::parse_smiles;
use crate::mol::molecule::Molecule;
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

/// CSV reader for batch processing of molecular datasets.
///
/// Supports:
/// - Auto-detection of SMILES column
/// - Custom column specification
/// - Parallel parsing with Rayon
/// - Error handling (skip invalid SMILES)
/// - Memory-efficient streaming
///
/// # Example
/// ```ignore
/// use ferrum_core::io::csv_reader::CsvReader;
///
/// let reader = CsvReader::from_path("molecules.csv")?;
/// let results = reader
///     .with_smiles_column("SMILES")
///     .parse()?;
///
/// for (id, molecule, properties) in results {
///     println!("Molecule {}: {} atoms", id, molecule.num_atoms());
/// }
/// ```
pub struct CsvReader {
    path: String,
    smiles_column: Option<String>,
    id_column: Option<String>,
    delimiter: u8,
    include_properties: bool,
    parse_chunk_size: usize,
    row_limit: Option<usize>,
}

/// Result of parsing a single CSV row.
#[derive(Debug, Clone)]
pub struct MoleculeRecord {
    /// Row index (0-based, excluding header)
    pub row_index: usize,
    /// Parsed molecule (if successful)
    pub molecule: Option<Molecule>,
    /// Original SMILES string (used as identifier)
    pub smiles: String,
    /// Error message (if parsing failed)
    pub error: Option<String>,
    /// Additional properties from other columns
    pub properties: Vec<(String, String)>,
}

impl CsvReader {
    /// Creates a new CSV reader for the given file path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Check if file exists
        if !path.as_ref().exists() {
            return Err(FerrumError::ParseError(format!(
                "File not found: {}",
                path_str
            )));
        }

        Ok(Self {
            path: path_str,
            smiles_column: None,
            id_column: None,
            delimiter: b',',
            include_properties: true,
            parse_chunk_size: 10_000,
            row_limit: None,
        })
    }

    /// Specifies the name of the SMILES column.
    /// If not set, will auto-detect common column names.
    pub fn with_smiles_column(mut self, column_name: &str) -> Self {
        self.smiles_column = Some(column_name.to_string());
        self
    }

    /// Sets the delimiter (default: comma).
    /// Use b'\t' for TSV files.
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Enables or disables capturing non-SMILES columns as String properties.
    /// Disable this for lower memory usage in descriptor-only pipelines.
    pub fn with_properties(mut self, include_properties: bool) -> Self {
        self.include_properties = include_properties;
        self
    }

    /// Sets how many CSV rows are parsed in each parallel chunk.
    /// Values < 1 are clamped to 1.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.parse_chunk_size = chunk_size.max(1);
        self
    }

    /// Limits the number of CSV rows consumed (excluding header).
    /// Use this to bound memory/time for benchmarking and sampling.
    pub fn with_row_limit(mut self, row_limit: usize) -> Self {
        self.row_limit = Some(row_limit);
        self
    }

    /// Auto-detects the SMILES column from common names.
    fn detect_smiles_column(headers: &csv::StringRecord) -> Result<usize> {
        let common_names = [
            "SMILES",
            "smiles",
            "Smiles",
            "SMILES_STRING",
            "smiles_string",
            "CANONICAL_SMILES",
            "canonical_smiles",
            "Canonical_Smiles",
        ];

        for (idx, header) in headers.iter().enumerate() {
            if common_names.contains(&header) {
                return Ok(idx);
            }
        }

        Err(FerrumError::ParseError(
            "Could not auto-detect SMILES column. Please specify with with_smiles_column()"
                .to_string(),
        ))
    }

    /// Parses chunk records.
    ///
    /// Function context: `crates/ferrum_core/src/io/csv_reader.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn parse_chunk_records(
        headers: &csv::StringRecord,
        smiles_idx: usize,
        include_properties: bool,
        records: Vec<(usize, csv::StringRecord)>,
    ) -> Vec<MoleculeRecord> {
        records
            .into_par_iter()
            .map(|(row_idx, record)| {
                let smiles = record.get(smiles_idx).unwrap_or("").trim().to_string();

                let properties = if include_properties {
                    let mut properties: Vec<(String, String)> =
                        Vec::with_capacity(headers.len().saturating_sub(1));
                    for (idx, (header, value)) in headers.iter().zip(record.iter()).enumerate() {
                        if idx != smiles_idx {
                            properties.push((header.to_string(), value.to_string()));
                        }
                    }
                    properties
                } else {
                    Vec::new()
                };

                let (molecule, error) = match parse_smiles(&smiles) {
                    Ok(mol) => (Some(mol), None),
                    Err(e) => (None, Some(e.to_string())),
                };

                MoleculeRecord {
                    row_index: row_idx,
                    molecule,
                    smiles,
                    error,
                    properties,
                }
            })
            .collect()
    }

    /// Parses the CSV file in bounded chunks and streams records to a callback.
    ///
    /// This avoids materializing the full `Vec<MoleculeRecord>` in memory.
    pub fn parse_stream<F>(&self, mut on_record: F) -> Result<ParseSummary>
    where
        F: FnMut(MoleculeRecord),
    {
        let file = File::open(&self.path)?;

        let mut reader = csv::ReaderBuilder::new()
            .delimiter(self.delimiter)
            .from_reader(file);

        let headers = reader
            .headers()
            .map_err(|e| FerrumError::ParseError(format!("Failed to read CSV headers: {}", e)))?
            .clone();

        let smiles_idx = if let Some(ref col_name) = self.smiles_column {
            headers.iter().position(|h| h == col_name).ok_or_else(|| {
                FerrumError::ParseError(format!("SMILES column '{}' not found in CSV", col_name))
            })?
        } else {
            Self::detect_smiles_column(&headers)?
        };

        let chunk_size = self.parse_chunk_size.max(1);
        let mut chunk_records: Vec<(usize, csv::StringRecord)> = Vec::with_capacity(chunk_size);
        let mut summary = ParseSummary::default();

        let row_limit = self.row_limit.unwrap_or(usize::MAX);
        for (row_idx, result) in reader.records().take(row_limit).enumerate() {
            if let Ok(record) = result {
                chunk_records.push((row_idx, record));
            }

            if chunk_records.len() == chunk_size {
                let current_chunk =
                    std::mem::replace(&mut chunk_records, Vec::with_capacity(chunk_size));
                let chunk_results = Self::parse_chunk_records(
                    &headers,
                    smiles_idx,
                    self.include_properties,
                    current_chunk,
                );
                for record in chunk_results {
                    summary.total_rows += 1;
                    if record.molecule.is_some() {
                        summary.valid_molecules += 1;
                    } else {
                        summary.failed_parses += 1;
                    }
                    on_record(record);
                }
            }
        }

        if !chunk_records.is_empty() {
            let chunk_results = Self::parse_chunk_records(
                &headers,
                smiles_idx,
                self.include_properties,
                chunk_records,
            );
            for record in chunk_results {
                summary.total_rows += 1;
                if record.molecule.is_some() {
                    summary.valid_molecules += 1;
                } else {
                    summary.failed_parses += 1;
                }
                on_record(record);
            }
        }

        Ok(summary)
    }

    /// Parses the CSV file and returns all records.
    ///
    /// # Returns
    /// Vector of `MoleculeRecord` containing molecules and metadata.
    /// Invalid SMILES are included with error messages (not skipped).
    pub fn parse(&self) -> Result<Vec<MoleculeRecord>> {
        let mut all_results = Vec::new();
        self.parse_stream(|record| all_results.push(record))?;
        Ok(all_results)
    }

    /// Parses the CSV file and returns only successfully parsed molecules with their SMILES.
    ///
    /// # Returns
    /// Vector of (SMILES, Molecule) tuples. Failed parses are skipped.
    pub fn parse_valid_only(&self) -> Result<Vec<(String, Molecule)>> {
        let mut valid = Vec::new();
        self.parse_stream(|record| {
            if let Some(mol) = record.molecule {
                valid.push((record.smiles, mol));
            }
        })?;
        Ok(valid)
    }

    /// Parses the CSV file and returns statistics.
    pub fn parse_with_stats(&self) -> Result<ParseStats> {
        let mut records = Vec::new();
        let mut error_messages = Vec::new();
        let summary = self.parse_stream(|record| {
            if let Some(error) = record.error.as_ref() {
                error_messages.push((record.row_index, record.smiles.clone(), error.clone()));
            }
            records.push(record);
        })?;

        Ok(ParseStats {
            total_rows: summary.total_rows,
            valid_molecules: summary.valid_molecules,
            failed_parses: summary.failed_parses,
            error_details: error_messages,
            records,
        })
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ParseSummary {
    pub total_rows: usize,
    pub valid_molecules: usize,
    pub failed_parses: usize,
}

/// Statistics from CSV parsing.
#[derive(Debug)]
pub struct ParseStats {
    pub total_rows: usize,
    pub valid_molecules: usize,
    pub failed_parses: usize,
    pub error_details: Vec<(usize, String, String)>, // (row, smiles, error)
    pub records: Vec<MoleculeRecord>,
}

impl ParseStats {
    /// Prints a summary to stdout.
    pub fn print_summary(&self) {
        println!("CSV Parsing Summary:");
        println!("  Total rows:        {}", self.total_rows);
        println!(
            "  Valid molecules:   {} ({:.1}%)",
            self.valid_molecules,
            (self.valid_molecules as f64 / self.total_rows as f64) * 100.0
        );
        println!(
            "  Failed parses:     {} ({:.1}%)",
            self.failed_parses,
            (self.failed_parses as f64 / self.total_rows as f64) * 100.0
        );

        if !self.error_details.is_empty() {
            println!("\nFirst {} errors:", self.error_details.len().min(5));
            for (row, smiles, error) in self.error_details.iter().take(5) {
                println!("  Row {}: '{}' - {}", row, smiles, error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Creates test csv.
    ///
    /// Function context: `crates/ferrum_core/src/io/csv_reader.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn create_test_csv(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", content).unwrap();
        file
    }

    #[test]
    /// Runs regression test for csv reader basic.
    ///
    /// Function context: `crates/ferrum_core/src/io/csv_reader.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_csv_reader_basic() {
        let csv_content = "\
SMILES,Name
CCO,Ethanol
c1ccccc1,Benzene
CC,Ethane
";
        let file = create_test_csv(csv_content);

        let reader = CsvReader::from_path(file.path()).unwrap();
        let records = reader.parse().unwrap();

        assert_eq!(records.len(), 3);
        assert!(records[0].molecule.is_some());
        assert_eq!(records[0].smiles, "CCO");
        assert!(records[1].molecule.is_some());
        assert_eq!(records[1].smiles, "c1ccccc1");
    }

    #[test]
    /// Runs regression test for csv reader with errors.
    ///
    /// Function context: `crates/ferrum_core/src/io/csv_reader.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_csv_reader_with_errors() {
        let csv_content = "\
SMILES,Name
CCO,Ethanol
INVALID,Bad
CC,Ethane
";
        let file = create_test_csv(csv_content);

        let reader = CsvReader::from_path(file.path()).unwrap();
        let stats = reader.parse_with_stats().unwrap();

        assert_eq!(stats.total_rows, 3);
        assert_eq!(stats.valid_molecules, 2);
        assert_eq!(stats.failed_parses, 1);
    }

    #[test]
    /// Runs regression test for csv reader custom column.
    ///
    /// Function context: `crates/ferrum_core/src/io/csv_reader.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_csv_reader_custom_column() {
        let csv_content = "\
Compound,Structure
Ethanol,CCO
Benzene,c1ccccc1
";
        let file = create_test_csv(csv_content);

        let reader = CsvReader::from_path(file.path())
            .unwrap()
            .with_smiles_column("Structure");
        let records = reader.parse().unwrap();

        assert_eq!(records.len(), 2);
        assert!(records[0].molecule.is_some());
    }

    #[test]
    /// Runs regression test for csv reader with id.
    ///
    /// Function context: `crates/ferrum_core/src/io/csv_reader.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_csv_reader_with_id() {
        let csv_content = "\
SMILES,Name
CCO,Ethanol
c1ccccc1,Benzene
";
        let file = create_test_csv(csv_content);

        let reader = CsvReader::from_path(file.path()).unwrap();
        let records = reader.parse().unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].smiles, "CCO");
        assert_eq!(records[1].smiles, "c1ccccc1");
        // Properties should include Name column
        assert_eq!(records[0].properties[0].0, "Name");
        assert_eq!(records[0].properties[0].1, "Ethanol");
    }

    #[test]
    /// Runs regression test for csv reader valid only.
    ///
    /// Function context: `crates/ferrum_core/src/io/csv_reader.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_csv_reader_valid_only() {
        let csv_content = "\
SMILES
CCO
INVALID
CC
";
        let file = create_test_csv(csv_content);

        let reader = CsvReader::from_path(file.path()).unwrap();
        let valid = reader.parse_valid_only().unwrap();

        assert_eq!(valid.len(), 2);
    }

    #[test]
    /// Runs regression test for csv reader without properties.
    ///
    /// Function context: `crates/ferrum_core/src/io/csv_reader.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_csv_reader_without_properties() {
        let csv_content = "\
SMILES,Name
CCO,Ethanol
CC,Ethane
";
        let file = create_test_csv(csv_content);

        let reader = CsvReader::from_path(file.path())
            .unwrap()
            .with_properties(false);
        let records = reader.parse().unwrap();

        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|record| record.properties.is_empty()));
    }

    #[test]
    /// Runs regression test for csv reader chunked parsing keeps row order.
    ///
    /// Function context: `crates/ferrum_core/src/io/csv_reader.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_csv_reader_chunked_parsing_keeps_row_order() {
        let csv_content = "\
SMILES
CCO
c1ccccc1
CC
";
        let file = create_test_csv(csv_content);

        let reader = CsvReader::from_path(file.path())
            .unwrap()
            .with_chunk_size(1);
        let records = reader.parse().unwrap();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].row_index, 0);
        assert_eq!(records[1].row_index, 1);
        assert_eq!(records[2].row_index, 2);
        assert_eq!(records[0].smiles, "CCO");
        assert_eq!(records[1].smiles, "c1ccccc1");
        assert_eq!(records[2].smiles, "CC");
    }

    #[test]
    /// Runs regression test for csv reader parse stream summary and order.
    ///
    /// Function context: `crates/ferrum_core/src/io/csv_reader.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_csv_reader_parse_stream_summary_and_order() {
        let csv_content = "\
SMILES
CCO
INVALID
CC
";
        let file = create_test_csv(csv_content);

        let reader = CsvReader::from_path(file.path())
            .unwrap()
            .with_chunk_size(1)
            .with_properties(false);
        let mut seen = Vec::new();
        let summary = reader
            .parse_stream(|record| {
                seen.push((record.row_index, record.smiles, record.molecule.is_some()));
            })
            .unwrap();

        assert_eq!(summary.total_rows, 3);
        assert_eq!(summary.valid_molecules, 2);
        assert_eq!(summary.failed_parses, 1);
        assert_eq!(seen[0], (0, "CCO".to_string(), true));
        assert_eq!(seen[1], (1, "INVALID".to_string(), false));
        assert_eq!(seen[2], (2, "CC".to_string(), true));
    }

    #[test]
    /// Runs regression test for csv reader row limit.
    ///
    /// Function context: `crates/ferrum_core/src/io/csv_reader.rs`.
    /// I/O-layer note: parsing/ingestion should surface explicit errors instead of silently repairing malformed input.
    fn test_csv_reader_row_limit() {
        let csv_content = "\
SMILES
CCO
CC
CCC
";
        let file = create_test_csv(csv_content);

        let reader = CsvReader::from_path(file.path()).unwrap().with_row_limit(2);
        let records = reader.parse().unwrap();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].smiles, "CCO");
        assert_eq!(records[1].smiles, "CC");
    }
}
