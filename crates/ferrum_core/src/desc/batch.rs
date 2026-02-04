use crate::mol::molecule::Molecule;
use crate::desc::Descriptor;
use crate::types::Real;
use crate::error::Result;
use rayon::prelude::*;
use std::io::Write;

/// Calculates descriptors for many molecules in parallel.
///
/// # Thread Safety
/// This function is designed for maximum parallelism:
/// - Uses Rayon for automatic work-stealing parallelization
/// - Molecules are shared immutably across threads (`&[Molecule]`)
/// - Descriptors must implement `Sync` for safe concurrent access
/// - Each thread has its own output buffer (no shared mutable state)
///
/// # Performance
/// Scales linearly with CPU cores. Typical speedup: 4-8x on modern hardware.
/// Best performance with batch sizes > 1000 molecules.
///
/// # Arguments
/// * `molecules` - Slice of molecules already in RAM (shared immutably)
/// * `descriptors` - Descriptors to calculate (must be `Sync` for thread safety)
///
/// # Returns
/// Vector of feature vectors (one per molecule), preserved in input order.
///
/// # Example
/// ```ignore
/// let molecules = vec![parse_smiles("C"), parse_smiles("CC")];
/// let descriptors: Vec<Box<dyn Descriptor + Sync>> = vec![
///     Box::new(MolecularWeight),
/// ];
/// let results = calculate_batch(&molecules, &descriptors);
/// ```
pub fn calculate_batch(
    molecules: &[Molecule],
    descriptors: &[Box<dyn Descriptor + Sync>] // Sync is critical for thread safety!
) -> Vec<Vec<Real>> {

    // Calculate total feature vector size once upfront
    let total_size: usize = descriptors.iter().map(|d| d.size()).sum();

    // par_iter() converts normal iterator to parallel iterator.
    // Rayon handles load-balancing and thread management automatically.
    molecules.par_iter().map(|mol| {

        // This block runs in parallel across all CPU cores
        // Each thread has its own feature buffer (no contention)
        let mut features = vec![0.0; total_size];
        let mut offset = 0;

        for desc in descriptors {
            let size = desc.size();
            // Slice the appropriate part of the buffer
            let dest_slice = &mut features[offset..offset + size];

            // Descriptor writes directly to memory
            // Thread-safe because each molecule is processed by one thread
            desc.calculate(mol, dest_slice);

            offset += size;
        }

        features
    }).collect() // Collects results in original order (Rayon guarantees this)
}

/// Writes descriptor matrix to CSV format.
///
/// # Arguments
/// * `writer` - Output writer (file, stdout, etc.)
/// * `smiles` - SMILES strings (one per row)
/// * `descriptors` - Descriptor instances (for header names)
/// * `matrix` - Feature matrix (rows = molecules, cols = descriptors)
pub fn write_csv<W: Write>(
    writer: &mut W,
    smiles: &[String],
    descriptors: &[Box<dyn Descriptor + Sync>],
    matrix: &[Vec<Real>],
) -> Result<()> {
    // Write header
    write!(writer, "SMILES")?;
    for desc in descriptors {
        if desc.size() == 1 {
            write!(writer, ",{}", desc.name())?;
        } else {
            // Multi-value descriptors get numbered columns
            for i in 0..desc.size() {
                write!(writer, ",{}_{}", desc.name(), i)?;
            }
        }
    }
    writeln!(writer)?;
    
    // Write data rows
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
///
/// # Arguments
/// * `writer` - Output writer (file, stdout, etc.)
/// * `smiles` - SMILES strings
/// * `descriptors` - Descriptor instances (for names)
/// * `matrix` - Feature matrix
pub fn write_json<W: Write>(
    writer: &mut W,
    smiles: &[String],
    descriptors: &[Box<dyn Descriptor + Sync>],
    matrix: &[Vec<Real>],
) -> Result<()> {
    use serde_json::json;
    
    // Build descriptor names list
    let mut descriptor_names = Vec::new();
    for desc in descriptors {
        if desc.size() == 1 {
            descriptor_names.push(desc.name());
        } else {
            for i in 0..desc.size() {
                descriptor_names.push(format!("{}_{}", desc.name(), i));
            }
        }
    }
    
    // Build JSON structure
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
    use crate::io::smiles::parse_smiles;
    use crate::desc::molecular_weight::MolecularWeight;
    use crate::desc::atom_count::AtomCount;
    
    #[test]
    fn test_calculate_batch() {
        let molecules = vec![
            parse_smiles("C").unwrap(),
            parse_smiles("CC").unwrap(),
            parse_smiles("CCC").unwrap(),
        ];
        
        let descriptors: Vec<Box<dyn Descriptor + Sync>> = vec![
            Box::new(MolecularWeight),
            Box::new(AtomCount),
        ];
        
        let results = calculate_batch(&molecules, &descriptors);
        
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].len(), 2); // MW + AtomCount
    }
    
    #[test]
    fn test_write_csv() {
        let smiles = vec!["C".to_string(), "CC".to_string()];
        let descriptors: Vec<Box<dyn Descriptor + Sync>> = vec![
            Box::new(MolecularWeight),
            Box::new(AtomCount),
        ];
        let matrix = vec![
            vec![16.04, 1.0],
            vec![30.07, 2.0],
        ];
        
        let mut output = Vec::new();
        write_csv(&mut output, &smiles, &descriptors, &matrix).unwrap();
        
        let csv_string = String::from_utf8(output).unwrap();
        assert!(csv_string.contains("SMILES,MolecularWeight,AtomCount"));
        assert!(csv_string.contains("C,16.04"));
        assert!(csv_string.contains("CC,30.07"));
    }
    
    #[test]
    fn test_write_json() {
        let smiles = vec!["C".to_string()];
        let descriptors: Vec<Box<dyn Descriptor + Sync>> = vec![
            Box::new(MolecularWeight),
        ];
        let matrix = vec![vec![16.04]];
        
        let mut output = Vec::new();
        write_json(&mut output, &smiles, &descriptors, &matrix).unwrap();
        
        let json_string = String::from_utf8(output).unwrap();
        assert!(json_string.contains("\"molecules\""));
        assert!(json_string.contains("\"smiles\": \"C\""));
    }
}

