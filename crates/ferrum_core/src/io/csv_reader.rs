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
    
    /// Auto-detects the SMILES column from common names.
    fn detect_smiles_column(headers: &csv::StringRecord) -> Result<usize> {
        let common_names = [
            "SMILES", "smiles", "Smiles",
            "SMILES_STRING", "smiles_string",
            "CANONICAL_SMILES", "canonical_smiles",
            "Canonical_Smiles",
        ];
        
        for (idx, header) in headers.iter().enumerate() {
            if common_names.contains(&header) {
                return Ok(idx);
            }
        }
        
        Err(FerrumError::ParseError(
            "Could not auto-detect SMILES column. Please specify with with_smiles_column()".to_string()
        ))
    }
    
    /// Parses the CSV file and returns all records.
    ///
    /// # Returns
    /// Vector of `MoleculeRecord` containing molecules and metadata.
    /// Invalid SMILES are included with error messages (not skipped).
    pub fn parse(&self) -> Result<Vec<MoleculeRecord>> {
        let file = File::open(&self.path)?;
        
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(self.delimiter)
            .from_reader(file);
        
        // Get headers
        let headers = reader.headers()
            .map_err(|e| FerrumError::ParseError(format!("Failed to read CSV headers: {}", e)))?
            .clone();
        
        // Find SMILES column
        let smiles_idx = if let Some(ref col_name) = self.smiles_column {
            headers.iter().position(|h| h == col_name)
                .ok_or_else(|| FerrumError::ParseError(
                    format!("SMILES column '{}' not found in CSV", col_name)
                ))?
        } else {
            Self::detect_smiles_column(&headers)?
        };
        
        // Parse all rows (sequential, to preserve order)
        let records: Vec<(usize, csv::StringRecord)> = reader.records()
            .enumerate()
            .filter_map(|(row_idx, result)| {
                result.ok().map(|record| (row_idx, record))
            })
            .collect();
        
        // Parallel processing of SMILES strings
        let results: Vec<MoleculeRecord> = records.par_iter()
            .map(|(row_idx, record)| {
                let smiles = record.get(smiles_idx)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                
                // Collect other properties
                let mut properties: Vec<(String, String)> = Vec::new();
                for (idx, (header, value)) in headers.iter().zip(record.iter()).enumerate() {
                    if idx != smiles_idx {
                        properties.push((header.to_string(), value.to_string()));
                    }
                }
                
                // Parse SMILES
                let (molecule, error) = match parse_smiles(&smiles) {
                    Ok(mol) => (Some(mol), None),
                    Err(e) => (None, Some(e.to_string())),
                };
                
                MoleculeRecord {
                    row_index: *row_idx,
                    molecule,
                    smiles,
                    error,
                    properties,
                }
            })
            .collect();
        
        Ok(results)
    }
    
    /// Parses the CSV file and returns only successfully parsed molecules with their SMILES.
    ///
    /// # Returns
    /// Vector of (SMILES, Molecule) tuples. Failed parses are skipped.
    pub fn parse_valid_only(&self) -> Result<Vec<(String, Molecule)>> {
        let records = self.parse()?;
        
        Ok(records.into_iter()
            .filter_map(|record| {
                record.molecule.map(|mol| (record.smiles, mol))
            })
            .collect())
    }
    
    /// Parses the CSV file and returns statistics.
    pub fn parse_with_stats(&self) -> Result<ParseStats> {
        let records = self.parse()?;
        
        let total = records.len();
        let valid = records.iter().filter(|r| r.molecule.is_some()).count();
        let errors = total - valid;
        
        let error_messages: Vec<_> = records.iter()
            .filter_map(|r| {
                r.error.as_ref().map(|e| (r.row_index, r.smiles.clone(), e.clone()))
            })
            .collect();
        
        Ok(ParseStats {
            total_rows: total,
            valid_molecules: valid,
            failed_parses: errors,
            error_details: error_messages,
            records,
        })
    }
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
        println!("  Valid molecules:   {} ({:.1}%)", 
                 self.valid_molecules,
                 (self.valid_molecules as f64 / self.total_rows as f64) * 100.0);
        println!("  Failed parses:     {} ({:.1}%)", 
                 self.failed_parses,
                 (self.failed_parses as f64 / self.total_rows as f64) * 100.0);
        
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
    
    fn create_test_csv(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", content).unwrap();
        file
    }
    
    #[test]
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
    fn test_csv_reader_with_id() {
        let csv_content = "\
SMILES,Name
CCO,Ethanol
c1ccccc1,Benzene
";
        let file = create_test_csv(csv_content);
        
        let reader = CsvReader::from_path(file.path())
            .unwrap();
        let records = reader.parse().unwrap();
        
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].smiles, "CCO");
        assert_eq!(records[1].smiles, "c1ccccc1");
        // Properties should include Name column
        assert_eq!(records[0].properties[0].0, "Name");
        assert_eq!(records[0].properties[0].1, "Ethanol");
    }
    
    #[test]
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
}
