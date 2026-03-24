use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Compute Gasteiger partial charges for the molecule.
pub fn compute_charges(mol: &Molecule) -> Vec<f64> {
    crate::algo::electronic::compute_gasteiger_charges(mol)
}

fn compute_charges_cached(mol: &mut Molecule) -> Vec<f64> {
    crate::algo::electronic::compute_gasteiger_charges_cached(mol)
}

/// Maximum partial charge across all atoms.
pub struct MaxPartialCharge;

impl Descriptor for MaxPartialCharge {
    fn name(&self) -> &str {
        "MaxPartialCharge"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let charges = compute_charges_cached(mol);
        let max = charges.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        DescriptorValue::Float(if max.is_finite() { max as f32 } else { 0.0 })
    }
}

/// Minimum partial charge across all atoms.
pub struct MinPartialCharge;

impl Descriptor for MinPartialCharge {
    fn name(&self) -> &str {
        "MinPartialCharge"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let charges = compute_charges_cached(mol);
        let min = charges.iter().copied().fold(f64::INFINITY, f64::min);
        DescriptorValue::Float(if min.is_finite() { min as f32 } else { 0.0 })
    }
}

/// Maximum absolute partial charge across all atoms.
pub struct MaxAbsPartialCharge;

impl Descriptor for MaxAbsPartialCharge {
    fn name(&self) -> &str {
        "MaxAbsPartialCharge"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let charges = compute_charges_cached(mol);
        let max_abs = charges
            .iter()
            .map(|c| c.abs())
            .fold(f64::NEG_INFINITY, f64::max);
        DescriptorValue::Float(if max_abs.is_finite() {
            max_abs as f32
        } else {
            0.0
        })
    }
}

/// Minimum absolute partial charge across all atoms.
pub struct MinAbsPartialCharge;

impl Descriptor for MinAbsPartialCharge {
    fn name(&self) -> &str {
        "MinAbsPartialCharge"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let charges = compute_charges_cached(mol);
        let min_abs = charges
            .iter()
            .map(|c| c.abs())
            .fold(f64::INFINITY, f64::min);
        DescriptorValue::Float(if min_abs.is_finite() {
            min_abs as f32
        } else {
            0.0
        })
    }
}

/// Per-atom Gasteiger partial charges as a vector.
pub struct GasteigerCharges;

impl Descriptor for GasteigerCharges {
    fn name(&self) -> &str {
        "GasteigerCharges"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let charges = compute_charges_cached(mol);
        DescriptorValue::FloatVec(charges.iter().map(|&c| c as f32).collect())
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MaxPartialCharge>;
    let _ = assert_send_sync::<MinPartialCharge>;
    let _ = assert_send_sync::<MaxAbsPartialCharge>;
    let _ = assert_send_sync::<MinAbsPartialCharge>;
    let _ = assert_send_sync::<GasteigerCharges>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).expect("valid SMILES")
    }

    fn calc_float(desc: &dyn Descriptor, mol: &mut Molecule) -> f32 {
        match desc.calculate(mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::Float(v) => v,
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_max_partial_charge_formaldehyde() {
        let mut mol = parse("C=O");
        let max = calc_float(&MaxPartialCharge, &mut mol);
        assert!(
            max > 0.0,
            "max charge on C=O should be positive (the C), got {max}"
        );
    }

    #[test]
    fn test_min_partial_charge_formaldehyde() {
        let mut mol = parse("C=O");
        let min = calc_float(&MinPartialCharge, &mut mol);
        assert!(
            min < 0.0,
            "min charge on C=O should be negative (the O), got {min}"
        );
    }

    #[test]
    fn test_max_abs_partial_charge() {
        let mut mol = parse("C=O");
        let max_abs = calc_float(&MaxAbsPartialCharge, &mut mol);
        let min = calc_float(&MinPartialCharge, &mut mol);
        // O should have largest |charge|
        assert!(
            (max_abs - min.abs()).abs() < 1e-6 || max_abs >= min.abs(),
            "max abs should be >= |min|"
        );
        assert!(max_abs > 0.0);
    }

    #[test]
    fn test_min_abs_partial_charge() {
        let mut mol = parse("C=O");
        let min_abs = calc_float(&MinAbsPartialCharge, &mut mol);
        let max_abs = calc_float(&MaxAbsPartialCharge, &mut mol);
        assert!(
            min_abs <= max_abs,
            "min abs {min_abs} should be <= max abs {max_abs}"
        );
    }

    #[test]
    fn test_charges_vector_length() {
        let mut mol = parse("CCO");
        match GasteigerCharges.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::FloatVec(v) => {
                assert_eq!(
                    v.len(),
                    mol.atoms.len(),
                    "vector length should equal num_atoms"
                );
            }
            other => panic!("expected FloatVec, got {:?}", other),
        }
    }

    #[test]
    fn test_acetic_acid_charges() {
        // CC(=O)O — carbonyl C positive, oxygens negative
        let mut mol = parse("CC(=O)O");
        let charges = compute_charges(&mol);
        // Atom 1 is the carbonyl carbon (bonded to both O atoms)
        assert!(
            charges[1] > 0.0,
            "carbonyl C should be positive, got {}",
            charges[1]
        );
        // Atom 2 is =O, Atom 3 is -OH — both should be negative
        assert!(
            charges[2] < 0.0,
            "=O should be negative, got {}",
            charges[2]
        );
        assert!(
            charges[3] < 0.0,
            "-OH oxygen should be negative, got {}",
            charges[3]
        );
    }

    #[test]
    fn test_benzene_symmetry() {
        let mut mol = parse("c1ccccc1");
        let max = calc_float(&MaxPartialCharge, &mut mol);
        let min = calc_float(&MinPartialCharge, &mut mol);
        assert!(
            (max - min).abs() < 0.01,
            "benzene should be symmetric: max={max}, min={min}"
        );
    }

    #[test]
    fn test_empty_molecule() {
        let mut mol = Molecule::default();
        assert_eq!(
            MaxPartialCharge.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()),
            DescriptorValue::Float(0.0)
        );
        assert_eq!(
            MinPartialCharge.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()),
            DescriptorValue::Float(0.0)
        );
        assert_eq!(
            MaxAbsPartialCharge.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()),
            DescriptorValue::Float(0.0)
        );
        assert_eq!(
            MinAbsPartialCharge.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()),
            DescriptorValue::Float(0.0)
        );
        assert_eq!(
            GasteigerCharges.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()),
            DescriptorValue::FloatVec(vec![])
        );
    }

    #[test]
    fn test_ethanol_oxygen_negative() {
        let mut mol = parse("CCO");
        let min = calc_float(&MinPartialCharge, &mut mol);
        assert!(
            min < 0.0,
            "O in ethanol should give negative min charge, got {min}"
        );
    }
}
