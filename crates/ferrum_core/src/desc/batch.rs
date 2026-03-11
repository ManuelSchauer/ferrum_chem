use crate::desc::typed::{Descriptor, build_ml_vector};
use crate::error::Result;
use crate::mol::molecule::Molecule;
use rayon::prelude::*;
use std::io::Write;

/// Calculates descriptors for many molecules in parallel.
///
/// Uses Rayon for automatic work-stealing parallelization.
/// Scales linearly with CPU cores.
pub fn calculate_batch(
    molecules: &mut [Molecule],
    descriptors: &[Box<dyn Descriptor>],
) -> Vec<Vec<f32>> {
    molecules
        .par_iter_mut()
        .map(|mol| build_ml_vector(mol, descriptors))
        .collect()
}

/// Writes descriptor matrix to CSV format.
pub fn write_csv<W: Write>(
    writer: &mut W,
    smiles: &[String],
    descriptors: &[Box<dyn Descriptor>],
    matrix: &[Vec<f32>],
) -> Result<()> {
    write!(writer, "SMILES")?;
    for desc in descriptors {
        write!(writer, ",{}", desc.name())?;
    }
    writeln!(writer)?;

    for (smiles_str, features) in smiles.iter().zip(matrix.iter()) {
        write!(writer, "{}", smiles_str)?;
        for &value in features {
            write!(writer, ",{:.6}", value)?;
        }
        writeln!(writer)?;
    }

    Ok(())
}

/// Writes descriptor matrix to JSON format.
pub fn write_json<W: Write>(
    writer: &mut W,
    smiles: &[String],
    descriptors: &[Box<dyn Descriptor>],
    matrix: &[Vec<f32>],
) -> Result<()> {
    use serde_json::json;

    let descriptor_names: Vec<String> = descriptors.iter().map(|d| d.name().to_string()).collect();

    let mut records = Vec::new();
    for (smiles_str, features) in smiles.iter().zip(matrix.iter()) {
        let mut record = serde_json::Map::new();
        record.insert("smiles".to_string(), json!(smiles_str));

        for (name, &value) in descriptor_names.iter().zip(features.iter()) {
            record.insert(name.clone(), json!(value));
        }

        records.push(json!(record));
    }

    let output = json!({
        "molecules": records,
        "descriptor_count": descriptor_names.len(),
        "molecule_count": smiles.len(),
    });

    serde_json::to_writer_pretty(&mut *writer, &output)?;
    writeln!(writer)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::molecular_weight::MolecularWeight;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn test_calculate_batch() {
        let mut molecules = vec![
            parse_smiles("C").unwrap(),
            parse_smiles("CC").unwrap(),
            parse_smiles("CCC").unwrap(),
        ];

        let descriptors: Vec<Box<dyn Descriptor>> = vec![Box::new(MolecularWeight)];

        let results = calculate_batch(&mut molecules, &descriptors);

        assert_eq!(results.len(), 3);
        assert!(!results[0].is_empty());
    }

    #[test]
    fn test_write_csv() {
        let smiles = vec!["C".to_string(), "CC".to_string()];
        let descriptors: Vec<Box<dyn Descriptor>> = vec![Box::new(MolecularWeight)];
        let matrix = vec![vec![16.0], vec![30.0]];

        let mut output = Vec::new();
        write_csv(&mut output, &smiles, &descriptors, &matrix).unwrap();

        let csv_string = String::from_utf8(output).unwrap();
        assert!(csv_string.contains("SMILES,MolecularWeight"));
    }

    #[test]
    fn test_write_json() {
        let smiles = vec!["C".to_string()];
        let descriptors: Vec<Box<dyn Descriptor>> = vec![Box::new(MolecularWeight)];
        let matrix = vec![vec![16.0]];

        let mut output = Vec::new();
        write_json(&mut output, &smiles, &descriptors, &matrix).unwrap();

        let json_string = String::from_utf8(output).unwrap();
        assert!(json_string.contains("\"molecules\""));
        assert!(json_string.contains("\"smiles\": \"C\""));
    }
}
