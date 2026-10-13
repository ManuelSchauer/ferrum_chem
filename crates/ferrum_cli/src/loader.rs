use ferrum_core::io::csv_reader::CsvReader;
use ferrum_core::io::smiles::parse_smiles;
use ferrum_core::mol::molecule::Molecule;
use std::fs;
use std::path::Path;

/// Loads molecules and SMILES strings from CSV, SMILES, or SDF files.
/// Returns (molecules, smiles_strings, failed_rows_count).
pub fn load_dataset(path: &Path) -> Result<(Vec<Molecule>, Vec<String>, usize), String> {
    let extension = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    match extension.as_str() {
        "csv" => load_from_csv(path),
        "smi" | "smiles" => load_from_smi(path),
        "sdf" => load_from_sdf(path),
        _ => Err(format!(
            "Unsupported file format for '{}'. Expected .csv, .smi, or .sdf",
            path.display()
        )),
    }
}

fn load_from_csv(path: &Path) -> Result<(Vec<Molecule>, Vec<String>, usize), String> {
    let records = CsvReader::from_path(path)
        .map_err(|e| format!("Failed to open CSV '{}': {}", path.display(), e))?
        .parse()
        .map_err(|e| format!("Failed to parse CSV '{}': {}", path.display(), e))?;

    let mut molecules = Vec::new();
    let mut smiles_vec = Vec::new();
    let mut failed_rows = 0;

    for record in records {
        if let Some(mol) = record.molecule {
            molecules.push(mol);
            smiles_vec.push(record.smiles);
        } else {
            failed_rows += 1;
        }
    }

    Ok((molecules, smiles_vec, failed_rows))
}

fn load_from_smi(path: &Path) -> Result<(Vec<Molecule>, Vec<String>, usize), String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read SMILES file '{}': {}", path.display(), e))?;

    let mut molecules = Vec::new();
    let mut smiles_vec = Vec::new();
    let mut failed_rows = 0;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let smi = trimmed
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_string();
        if smi.is_empty() {
            continue;
        }

        match parse_smiles(&smi) {
            Ok(mol) => {
                molecules.push(mol);
                smiles_vec.push(smi);
            }
            Err(_) => {
                failed_rows += 1;
            }
        }
    }

    Ok((molecules, smiles_vec, failed_rows))
}

fn load_from_sdf(path: &Path) -> Result<(Vec<Molecule>, Vec<String>, usize), String> {
    let records = ferrum_core::io::sdf::read_sdf(path)
        .map_err(|e| format!("Failed to read SDF '{}': {}", path.display(), e))?;

    let mut molecules = Vec::with_capacity(records.len());
    let mut names = Vec::with_capacity(records.len());
    let mut failed_rows = 0;
    for (i, record) in records.into_iter().enumerate() {
        match record.molecule {
            Ok(mol) => {
                molecules.push(mol);
                names.push(if record.name.is_empty() {
                    format!("SDF_Mol_{}", i + 1)
                } else {
                    record.name
                });
            }
            Err(_) => failed_rows += 1,
        }
    }
    Ok((molecules, names, failed_rows))
}
