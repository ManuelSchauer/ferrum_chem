//! Chunked molecule input for constant-memory batch runs.
//!
//! [`MoleculeReader`] reads `.smi`/`.smiles`, `.csv` and `.sdf` files lazily
//! and returns records in chunks; each chunk is parsed in parallel. Every
//! input record yields exactly one [`InputRecord`] (parse failures included),
//! so output rows stay aligned with input records.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use rayon::prelude::*;

use crate::error::{FerrumError, Result};
use crate::mol::molecule::Molecule;

/// One input record: its identifier (SMILES text or SDF title) and the parsed
/// molecule or the parse error.
#[derive(Debug)]
pub struct InputRecord {
    pub id: String,
    pub molecule: std::result::Result<Molecule, String>,
}

enum Raw {
    Smiles(String),
    MolBlock(String),
    Error(String),
}

/// One input record as read, not yet parsed ([`MoleculeReader::next_record`]);
/// [`RawRecord::parse`] turns it into an [`InputRecord`], e.g. on a worker
/// thread.
pub struct RawRecord {
    id: String,
    raw: Raw,
}

impl RawRecord {
    /// Identifier of the record (SMILES text or SDF title).
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Parses the molecule (SMILES or molfile block).
    pub fn parse(self) -> InputRecord {
        let molecule = match self.raw {
            Raw::Smiles(s) => super::smiles::parse_smiles(&s).map_err(|e| e.to_string()),
            Raw::MolBlock(b) => super::sdf::parse_mol_block(&b).map_err(|e| e.to_string()),
            Raw::Error(e) => Err(e),
        };
        InputRecord { id: self.id, molecule }
    }
}

enum Source {
    Smiles(BufReader<File>),
    Csv { reader: csv::Reader<File>, smiles_idx: usize },
    Sdf { reader: BufReader<File>, count: usize },
}

/// Lazily reads molecule records from a file.
pub struct MoleculeReader {
    source: Source,
    chunk_size: usize,
    line: String,
}

impl MoleculeReader {
    /// Default number of records per chunk.
    pub const DEFAULT_CHUNK_SIZE: usize = 4096;

    /// Opens `path`, choosing the format by extension (`.smi`, `.smiles`, `.csv`, `.sdf`).
    pub fn open(path: &Path) -> Result<Self> {
        let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
        let file = File::open(path)?;
        let source = match ext.as_str() {
            "smi" | "smiles" => Source::Smiles(BufReader::new(file)),
            "sdf" => Source::Sdf { reader: BufReader::new(file), count: 0 },
            "csv" => {
                let mut reader = csv::ReaderBuilder::new().from_reader(file);
                let headers = reader
                    .headers()
                    .map_err(|e| FerrumError::ParseError(format!("Failed to read CSV headers: {e}")))?
                    .clone();
                let smiles_idx = super::csv_reader::CsvReader::detect_smiles_column(&headers)?;
                Source::Csv { reader, smiles_idx }
            }
            _ => {
                return Err(FerrumError::ParseError(format!(
                    "Unsupported file format '{}': expected .smi, .smiles, .csv or .sdf",
                    path.display()
                )))
            }
        };
        Ok(Self { source, chunk_size: Self::DEFAULT_CHUNK_SIZE, line: String::new() })
    }

    /// Sets the number of records per chunk (memory scales with it).
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }

    /// Name of the identifier column: `"SMILES"`, or `"Name"` for SDF input.
    pub fn id_column(&self) -> &'static str {
        match self.source {
            Source::Sdf { .. } => "Name",
            _ => "SMILES",
        }
    }

    /// Reads and parses the next chunk; `None` at end of input.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<InputRecord>>> {
        let mut raw = Vec::with_capacity(self.chunk_size);
        while raw.len() < self.chunk_size {
            match self.next_record()? {
                Some(r) => raw.push(r),
                None => break,
            }
        }
        if raw.is_empty() {
            return Ok(None);
        }
        Ok(Some(raw.into_par_iter().map(RawRecord::parse).collect()))
    }

    /// Reads the next record without parsing it; `None` at end of input.
    pub fn next_record(&mut self) -> Result<Option<RawRecord>> {
        Ok(self.next_raw()?.map(|(id, raw)| RawRecord { id, raw }))
    }

    fn next_raw(&mut self) -> Result<Option<(String, Raw)>> {
        match &mut self.source {
            Source::Smiles(reader) => loop {
                self.line.clear();
                if reader.read_line(&mut self.line)? == 0 {
                    return Ok(None);
                }
                let t = self.line.trim();
                if t.is_empty() || t.starts_with('#') {
                    continue;
                }
                let smi = t.split_whitespace().next().unwrap_or_default().to_string();
                return Ok(Some((smi.clone(), Raw::Smiles(smi))));
            },
            Source::Csv { reader, smiles_idx } => {
                let mut record = csv::StringRecord::new();
                match reader.read_record(&mut record) {
                    Ok(false) => Ok(None),
                    Ok(true) => {
                        let smi = record.get(*smiles_idx).unwrap_or("").trim().to_string();
                        Ok(Some((smi.clone(), Raw::Smiles(smi))))
                    }
                    Err(e) => Ok(Some((String::new(), Raw::Error(format!("Malformed CSV row: {e}"))))),
                }
            }
            Source::Sdf { reader, count } => {
                let mut block = String::new();
                loop {
                    self.line.clear();
                    if reader.read_line(&mut self.line)? == 0 {
                        // trailing record without `$$$$`
                        if block.trim().is_empty() {
                            return Ok(None);
                        }
                        break;
                    }
                    if self.line.trim_end() == "$$$$" {
                        break;
                    }
                    block.push_str(&self.line);
                }
                *count += 1;
                let title = block.lines().next().unwrap_or("").trim();
                let id = if title.is_empty() { format!("SDF_Mol_{count}") } else { title.to_string() };
                Ok(Some((id, Raw::MolBlock(block))))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(ext: &str, content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(ext).tempfile().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    fn read_all(path: &Path, chunk: usize) -> Vec<InputRecord> {
        let mut r = MoleculeReader::open(path).unwrap().with_chunk_size(chunk);
        let mut out = Vec::new();
        while let Some(c) = r.next_chunk().unwrap() {
            assert!(c.len() <= chunk);
            out.extend(c);
        }
        out
    }

    #[test]
    fn smiles_file_keeps_failed_records() {
        let f = write_tmp(".smi", "CCO ethanol\n\n# comment\nC1CC\nc1ccccc1\n");
        let recs = read_all(f.path(), 2);
        assert_eq!(recs.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["CCO", "C1CC", "c1ccccc1"]);
        assert_eq!(recs.iter().map(|r| r.molecule.is_ok()).collect::<Vec<_>>(), [true, false, true]);
    }

    #[test]
    fn csv_file_detects_smiles_column() {
        let f = write_tmp(".csv", "id,smiles\n1,CCO\n2,xx\n");
        let recs = read_all(f.path(), 10);
        assert_eq!(recs.len(), 2);
        assert!(recs[0].molecule.is_ok() && recs[1].molecule.is_err());
        assert_eq!(MoleculeReader::open(f.path()).unwrap().id_column(), "SMILES");
    }

    #[test]
    fn sdf_file_streams_records() {
        let block = "water\n  test\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n> <prop>\nx\n\n$$$$\n";
        let f = write_tmp(".sdf", &format!("{block}{}", block.replacen("water", "", 1)));
        let recs = read_all(f.path(), 1);
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].id, "water");
        assert_eq!(recs[1].id, "SDF_Mol_2");
        assert_eq!(recs[0].molecule.as_ref().unwrap().num_atoms(), 1);
    }
}
