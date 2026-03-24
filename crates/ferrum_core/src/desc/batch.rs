use super::aromatic_ring_count::AromaticRingCount;
use super::atom_count::{AromaticAtomCount, AtomCount, HeavyAtomCount};
use super::balaban::BalabanIndex;
use super::bond_count::{BondCount, RotatableBondCount};
use super::extra_counts::{BertzCT, FractionCSP3, HallKierAlpha, NumAmideBonds};
use super::formal_charge::FormalCharge;
use super::hbond::{HBondAcceptor, HBondDonor};
use super::logp::{LogP, MolarRefractivity};
use super::molecular_weight::MolecularWeight;
use super::qed::Qed;
use super::simple_counts::HeteroatomCount;
use super::tpsa::Tpsa;
use super::typed::{Descriptor, DescriptorValue};
use super::vsa::{
    fill_peoe_vsa_into, fill_slogp_vsa_into, fill_smr_vsa_into, LabuteASA, PeoeVsa, SlogpVSA,
    SmrVSA,
};
use crate::error::Result;
use crate::mol::molecule::Molecule;
use rayon::prelude::*;
use std::io::Write;

/// Flat descriptor matrix optimized for ML consumers.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatDescriptorMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl FlatDescriptorMatrix {
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn row(&self, idx: usize) -> &[f32] {
        assert!(idx < self.rows, "row index out of bounds");
        if self.cols == 0 {
            return &[];
        }
        let start = idx * self.cols;
        &self.data[start..start + self.cols]
    }

    pub fn into_nested(self) -> Vec<Vec<f32>> {
        if self.cols == 0 {
            return vec![Vec::new(); self.rows];
        }
        self.data
            .chunks(self.cols)
            .map(|row| row.to_vec())
            .collect()
    }

    pub fn into_shape_vec(self) -> ((usize, usize), Vec<f32>) {
        ((self.rows, self.cols), self.data)
    }
}

/// Static-dispatch batch descriptor set for high-throughput ML/batch paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchDescriptor {
    MolecularWeight,
    LogP,
    MolarRefractivity,
    Tpsa,
    HBondDonor,
    HBondAcceptor,
    RotatableBondCount,
    HeavyAtomCount,
    AtomCount,
    AromaticAtomCount,
    HeteroatomCount,
    AromaticRingCount,
    FormalCharge,
    FractionCSP3,
    BertzCT,
    LabuteAsa,
    Qed,
    NumAmideBonds,
    HallKierAlpha,
    BalabanIndex,
    BondCount,
    SlogpVsa,
    SmrVsa,
    PeoeVsa,
}

impl BatchDescriptor {
    pub fn name(self) -> &'static str {
        match self {
            Self::MolecularWeight => "MolecularWeight",
            Self::LogP => "LogP",
            Self::MolarRefractivity => "MolarRefractivity",
            Self::Tpsa => "TPSA",
            Self::HBondDonor => "HBondDonor",
            Self::HBondAcceptor => "HBondAcceptor",
            Self::RotatableBondCount => "RotatableBondCount",
            Self::HeavyAtomCount => "HeavyAtomCount",
            Self::AtomCount => "AtomCount",
            Self::AromaticAtomCount => "AromaticAtomCount",
            Self::HeteroatomCount => "HeteroatomCount",
            Self::AromaticRingCount => "AromaticRingCount",
            Self::FormalCharge => "FormalCharge",
            Self::FractionCSP3 => "FractionCSP3",
            Self::BertzCT => "BertzCT",
            Self::LabuteAsa => "LabuteASA",
            Self::Qed => "QED",
            Self::NumAmideBonds => "NumAmideBonds",
            Self::HallKierAlpha => "HallKierAlpha",
            Self::BalabanIndex => "BalabanIndex",
            Self::BondCount => "BondCount",
            Self::SlogpVsa => "SlogP_VSA",
            Self::SmrVsa => "SMR_VSA",
            Self::PeoeVsa => "PEOE_VSA",
        }
    }

    pub fn calculate(self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        match self {
            Self::MolecularWeight => MolecularWeight.calculate(mol, ctx),
            Self::LogP => LogP.calculate(mol, ctx),
            Self::MolarRefractivity => MolarRefractivity.calculate(mol, ctx),
            Self::Tpsa => Tpsa.calculate(mol, ctx),
            Self::HBondDonor => HBondDonor.calculate(mol, ctx),
            Self::HBondAcceptor => HBondAcceptor.calculate(mol, ctx),
            Self::RotatableBondCount => RotatableBondCount.calculate(mol, ctx),
            Self::HeavyAtomCount => HeavyAtomCount.calculate(mol, ctx),
            Self::AtomCount => AtomCount.calculate(mol, ctx),
            Self::AromaticAtomCount => AromaticAtomCount.calculate(mol, ctx),
            Self::HeteroatomCount => HeteroatomCount.calculate(mol, ctx),
            Self::AromaticRingCount => AromaticRingCount.calculate(mol, ctx),
            Self::FormalCharge => FormalCharge.calculate(mol, ctx),
            Self::FractionCSP3 => FractionCSP3.calculate(mol, ctx),
            Self::BertzCT => BertzCT.calculate(mol, ctx),
            Self::LabuteAsa => LabuteASA.calculate(mol, ctx),
            Self::Qed => Qed.calculate(mol, ctx),
            Self::NumAmideBonds => NumAmideBonds.calculate(mol, ctx),
            Self::HallKierAlpha => HallKierAlpha.calculate(mol, ctx),
            Self::BalabanIndex => BalabanIndex.calculate(mol, ctx),
            Self::BondCount => BondCount.calculate(mol, ctx),
            Self::SlogpVsa => SlogpVSA.calculate(mol, ctx),
            Self::SmrVsa => SmrVSA.calculate(mol, ctx),
            Self::PeoeVsa => PeoeVsa.calculate(mol, ctx),
        }
    }

    pub fn feature_len(self) -> usize {
        match self {
            Self::SlogpVsa => 12,
            Self::SmrVsa => 10,
            Self::PeoeVsa => 14,
            _ => 1,
        }
    }

    pub fn write_into(self, mol: &mut Molecule, row: &mut [f32], offset: &mut usize, ctx: &mut crate::desc::context::CalculationContext) {
        match self {
            Self::SlogpVsa => {
                let end = *offset + self.feature_len();
                fill_slogp_vsa_into(mol, &mut row[*offset..end], ctx);
                *offset = end;
            }
            Self::SmrVsa => {
                let end = *offset + self.feature_len();
                fill_smr_vsa_into(mol, &mut row[*offset..end], ctx);
                *offset = end;
            }
            Self::PeoeVsa => {
                let end = *offset + self.feature_len();
                fill_peoe_vsa_into(mol, &mut row[*offset..end], ctx);
                *offset = end;
            }
            _ => write_descriptor_value(self.calculate(mol, ctx), row, offset),
        }
    }
}

fn write_descriptor_value(value: DescriptorValue, row: &mut [f32], offset: &mut usize) {
    match value {
        DescriptorValue::Int(v) => {
            row[*offset] = v as f32;
            *offset += 1;
        }
        DescriptorValue::Float(v) => {
            row[*offset] = v;
            *offset += 1;
        }
        DescriptorValue::Float64(v) => {
            row[*offset] = v as f32;
            *offset += 1;
        }
        DescriptorValue::IntVec(v) => {
            for value in v {
                row[*offset] = value as f32;
                *offset += 1;
            }
        }
        DescriptorValue::IntU16Vec(v) => {
            for value in v {
                row[*offset] = value as f32;
                *offset += 1;
            }
        }
        DescriptorValue::FloatVec(v) => {
            let end = *offset + v.len();
            row[*offset..end].copy_from_slice(&v);
            *offset = end;
        }
        DescriptorValue::Binary(bits) => {
            for bit in bits.iter().by_vals() {
                row[*offset] = if bit { 1.0 } else { 0.0 };
                *offset += 1;
            }
        }
    }
}

fn infer_cols_dyn(probe: &mut Molecule, descriptors: &[Box<dyn Descriptor>]) -> usize {
    let mut ctx = crate::desc::context::CalculationContext::new();
    descriptors
        .iter()
        .map(|desc| desc.calculate(probe, &mut ctx).len())
        .sum()
}

fn infer_cols_static(descriptors: &[BatchDescriptor]) -> usize {
    descriptors.iter().map(|desc| desc.feature_len()).sum()
}

fn fill_row_dyn(row: &mut [f32], mol: &mut Molecule, descriptors: &[Box<dyn Descriptor>], ctx: &mut crate::desc::context::CalculationContext) {
    let mut offset = 0usize;
    for desc in descriptors {
        write_descriptor_value(desc.calculate(mol, ctx), row, &mut offset);
    }
    debug_assert_eq!(offset, row.len());
}

fn fill_row_static(row: &mut [f32], mol: &mut Molecule, descriptors: &[BatchDescriptor], ctx: &mut crate::desc::context::CalculationContext) {
    let mut offset = 0usize;
    for &desc in descriptors {
        desc.write_into(mol, row, &mut offset, ctx);
    }
    debug_assert_eq!(offset, row.len());
}

/// Calculates descriptors for many molecules in parallel into one contiguous buffer.
pub fn calculate_batch_flat(
    molecules: &mut [Molecule],
    descriptors: &[Box<dyn Descriptor>],
) -> FlatDescriptorMatrix {
    let rows = molecules.len();
    if rows == 0 || descriptors.is_empty() {
        return FlatDescriptorMatrix {
            rows,
            cols: 0,
            data: Vec::new(),
        };
    }

    let mut probe = molecules[0].clone();
    let cols = infer_cols_dyn(&mut probe, descriptors);
    if cols == 0 {
        return FlatDescriptorMatrix {
            rows,
            cols,
            data: Vec::new(),
        };
    }

    let mut data = vec![0.0f32; rows * cols];
    data.par_chunks_mut(cols)
        .zip(molecules.par_iter_mut())
        .for_each_init(crate::desc::context::CalculationContext::new, |ctx, (row, mol)| {
            ctx.reset_for_new_molecule();
            fill_row_dyn(row, mol, descriptors, ctx);
        });

    FlatDescriptorMatrix { rows, cols, data }
}

/// Calculates static-dispatch descriptors for many molecules in parallel into one contiguous buffer.
pub fn calculate_batch_flat_static(
    molecules: &mut [Molecule],
    descriptors: &[BatchDescriptor],
) -> FlatDescriptorMatrix {
    let rows = molecules.len();
    if rows == 0 || descriptors.is_empty() {
        return FlatDescriptorMatrix {
            rows,
            cols: 0,
            data: Vec::new(),
        };
    }

    let cols = infer_cols_static(descriptors);
    if cols == 0 {
        return FlatDescriptorMatrix {
            rows,
            cols,
            data: Vec::new(),
        };
    }

    let mut data = vec![0.0f32; rows * cols];
    data.par_chunks_mut(cols)
        .zip(molecules.par_iter_mut())
        .for_each_init(crate::desc::context::CalculationContext::new, |ctx, (row, mol)| {
            ctx.reset_for_new_molecule();
            fill_row_static(row, mol, descriptors, ctx);
        });

    FlatDescriptorMatrix { rows, cols, data }
}

/// Compatibility API: calculates descriptors in parallel and returns row-major vectors.
pub fn calculate_batch(
    molecules: &mut [Molecule],
    descriptors: &[Box<dyn Descriptor>],
) -> Vec<Vec<f32>> {
    calculate_batch_flat(molecules, descriptors).into_nested()
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

/// Writes flattened descriptor matrix to CSV format.
pub fn write_csv_flat<W: Write>(
    writer: &mut W,
    smiles: &[String],
    feature_names: &[String],
    matrix: &FlatDescriptorMatrix,
) -> Result<()> {
    write!(writer, "SMILES")?;
    for name in feature_names {
        write!(writer, ",{}", name)?;
    }
    writeln!(writer)?;

    for (row_idx, smiles_str) in smiles.iter().enumerate().take(matrix.rows) {
        write!(writer, "{}", smiles_str)?;
        for &value in matrix.row(row_idx) {
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

/// Writes flattened descriptor matrix to JSON format.
pub fn write_json_flat<W: Write>(
    writer: &mut W,
    smiles: &[String],
    feature_names: &[String],
    matrix: &FlatDescriptorMatrix,
) -> Result<()> {
    use serde_json::json;

    let mut records = Vec::new();
    for (row_idx, smiles_str) in smiles.iter().enumerate().take(matrix.rows) {
        let mut record = serde_json::Map::new();
        record.insert("smiles".to_string(), json!(smiles_str));
        for (name, &value) in feature_names.iter().zip(matrix.row(row_idx)) {
            record.insert(name.clone(), json!(value));
        }
        records.push(json!(record));
    }

    let output = json!({
        "molecules": records,
        "feature_count": feature_names.len(),
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
    fn test_calculate_batch_flat_shape() {
        let mut molecules = vec![parse_smiles("C").unwrap(), parse_smiles("CC").unwrap()];
        let descriptors: Vec<Box<dyn Descriptor>> = vec![Box::new(MolecularWeight)];
        let matrix = calculate_batch_flat(&mut molecules, &descriptors);

        assert_eq!(matrix.shape(), (2, 1));
        assert_eq!(matrix.data.len(), 2);
    }

    #[test]
    fn test_calculate_batch_flat_static_shape() {
        let mut molecules = vec![parse_smiles("C").unwrap(), parse_smiles("CC").unwrap()];
        let descriptors = [BatchDescriptor::MolecularWeight, BatchDescriptor::AtomCount];
        let matrix = calculate_batch_flat_static(&mut molecules, &descriptors);

        assert_eq!(matrix.shape(), (2, 2));
        assert_eq!(matrix.data.len(), 4);
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
    fn test_write_csv_flat() {
        let smiles = vec!["C".to_string(), "CC".to_string()];
        let feature_names = vec!["MolecularWeight".to_string()];
        let matrix = FlatDescriptorMatrix {
            rows: 2,
            cols: 1,
            data: vec![16.0, 30.0],
        };

        let mut output = Vec::new();
        write_csv_flat(&mut output, &smiles, &feature_names, &matrix).unwrap();
        let csv_string = String::from_utf8(output).unwrap();
        assert!(csv_string.contains("SMILES,MolecularWeight"));
        assert!(csv_string.contains("CC,30.000000"));
    }
}
