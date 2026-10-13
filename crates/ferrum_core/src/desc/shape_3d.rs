//! 3D shape descriptors of the first conformer (RDKit `PMI.cpp`).
//!
//! - `PMI1..3`: principal moments of inertia, mass weighted, ascending
//!   (`rdMolDescriptors.CalcPMI1..3`, `useAtomicMasses=True`).
//! - `Asphericity`: from the eigenvalues of the mass-weighted gyration
//!   (covariance) matrix (`rdMolDescriptors.CalcAsphericity`).
//!
//! Atom weights are RDKit `Atom::getMass()` (isotope mass when an isotope is
//! set, otherwise the standard atomic weight); implicit hydrogens do not
//! contribute, explicit hydrogen atoms do. Molecules without a conformer
//! (e.g. from SMILES) give NaN.

use crate::desc::context::CalculationContext;
use crate::desc::Descriptor;
use crate::desc::DescriptorValue;
use crate::mol::molecule::Molecule;
use crate::mol::rdkit_data::atom_mass;

/// Positions of the first conformer and RDKit atom masses, if available.
fn weighted_positions(mol: &Molecule) -> Option<(&[[f64; 3]], Vec<f64>)> {
    let conf = mol.conformers.first()?;
    if conf.positions.len() != mol.atoms.len() {
        return None;
    }
    let weights = mol
        .atoms
        .iter()
        .map(|a| atom_mass(a.atomic_number, a.isotope))
        .collect();
    Some((&conf.positions, weights))
}

/// RDKit `MolTransforms::computeCentroid` with weights.
fn centroid(positions: &[[f64; 3]], weights: &[f64]) -> [f64; 3] {
    let mut c = [0.0f64; 3];
    let mut w_sum = 0.0;
    for (p, &w) in positions.iter().zip(weights) {
        w_sum += w;
        for k in 0..3 {
            c[k] += p[k] * w;
        }
    }
    c.map(|x| x / w_sum)
}

/// Eigenvalues of a symmetric 3x3 matrix in ascending order (cyclic Jacobi).
fn sym3_eigenvalues(mut a: [[f64; 3]; 3]) -> [f64; 3] {
    for _sweep in 0..64 {
        if a[0][1] == 0.0 && a[0][2] == 0.0 && a[1][2] == 0.0 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            // smaller root of t^2 + 2 theta t - 1 = 0 (|rotation angle| <= pi/4)
            let t = if theta.abs() > 1e150 {
                0.5 / theta
            } else {
                theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
            };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for row in a.iter_mut() {
                let (akp, akq) = (row[p], row[q]);
                row[p] = c * akp - s * akq;
                row[q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            a[p][q] = 0.0;
            a[q][p] = 0.0;
        }
    }
    let mut d = [a[0][0], a[1][1], a[2][2]];
    d.sort_by(f64::total_cmp);
    d
}

/// RDKit `computePrincipalAxesAndMoments` (mass weighted, H included):
/// eigenvalues of the inertia tensor about the centre of mass, ascending.
pub(crate) fn principal_moments(mol: &Molecule) -> Option<[f64; 3]> {
    let (positions, weights) = weighted_positions(mol)?;
    let c = centroid(positions, &weights);
    let (mut xx, mut xy, mut xz, mut yy, mut yz, mut zz) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (p, &w) in positions.iter().zip(&weights) {
        let (x, y, z) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
        xx += w * (y * y + z * z);
        yy += w * (x * x + z * z);
        zz += w * (y * y + x * x);
        xy -= w * x * y;
        xz -= w * x * z;
        yz -= w * z * y;
    }
    Some(sym3_eigenvalues([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]))
}

/// RDKit `computePrincipalAxesAndMomentsFromGyrationMatrix` (mass weighted,
/// normalised covariance about the centre of mass), ascending.
pub(crate) fn gyration_moments(mol: &Molecule) -> Option<[f64; 3]> {
    let (positions, weights) = weighted_positions(mol)?;
    let c = centroid(positions, &weights);
    let (mut xx, mut xy, mut xz, mut yy, mut yz, mut zz) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut w_sum = 0.0;
    for (p, &w) in positions.iter().zip(&weights) {
        let (x, y, z) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
        w_sum += w;
        xx += w * x * x;
        xy += w * x * y;
        xz += w * x * z;
        yy += w * y * y;
        yz += w * y * z;
        zz += w * z * z;
    }
    let [xx, xy, xz, yy, yz, zz] = [xx, xy, xz, yy, yz, zz].map(|v| v / w_sum);
    Some(sym3_eigenvalues([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]))
}

fn float_or_nan(v: Option<f64>) -> DescriptorValue {
    DescriptorValue::Float(v.unwrap_or(f64::NAN))
}

/// Smallest principal moment of inertia (RDKit `CalcPMI1`).
#[derive(Debug, Clone)]
pub struct Pmi1;
impl Descriptor for Pmi1 {
    fn name(&self) -> &str {
        "PMI1"
    }
    fn calculate(&self, mol: &Molecule, _: &mut CalculationContext) -> DescriptorValue {
        float_or_nan(principal_moments(mol).map(|m| m[0]))
    }
}

/// Middle principal moment of inertia (RDKit `CalcPMI2`).
#[derive(Debug, Clone)]
pub struct Pmi2;
impl Descriptor for Pmi2 {
    fn name(&self) -> &str {
        "PMI2"
    }
    fn calculate(&self, mol: &Molecule, _: &mut CalculationContext) -> DescriptorValue {
        float_or_nan(principal_moments(mol).map(|m| m[1]))
    }
}

/// Largest principal moment of inertia (RDKit `CalcPMI3`).
#[derive(Debug, Clone)]
pub struct Pmi3;
impl Descriptor for Pmi3 {
    fn name(&self) -> &str {
        "PMI3"
    }
    fn calculate(&self, mol: &Molecule, _: &mut CalculationContext) -> DescriptorValue {
        float_or_nan(principal_moments(mol).map(|m| m[2]))
    }
}

/// Asphericity (RDKit `CalcAsphericity`): with the gyration-matrix
/// eigenvalues `l1 <= l2 <= l3`,
/// `0.5 * ((l1-l2)^2 + (l1-l3)^2 + (l2-l3)^2) / (l1+l2+l3)^2`, and 0 when
/// `l3 < 1e-4` (no extent).
#[derive(Debug, Clone)]
pub struct Asphericity;
impl Descriptor for Asphericity {
    fn name(&self) -> &str {
        "Asphericity"
    }
    fn calculate(&self, mol: &Molecule, _: &mut CalculationContext) -> DescriptorValue {
        float_or_nan(gyration_moments(mol).map(|[p1, p2, p3]| {
            if p3 < 1e-4 {
                0.0
            } else {
                let denom = p1 + p2 + p3;
                0.5 * ((p1 - p2).powi(2) + (p1 - p3).powi(2) + (p2 - p3).powi(2)) / (denom * denom)
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::sdf::parse_mol_block;

    fn calc(desc: &dyn Descriptor, mol: &Molecule) -> f64 {
        match desc.calculate(mol, &mut CalculationContext::new()) {
            DescriptorValue::Float(x) => x,
            other => panic!("expected Float, got {other:?}"),
        }
    }

    #[test]
    fn eigenvalues_of_symmetric_matrix() {
        let d = sym3_eigenvalues([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]);
        let s = 2f64.sqrt();
        for (got, exp) in d.iter().zip([2.0 - s, 2.0, 2.0 + s]) {
            assert!((got - exp).abs() < 1e-14, "{d:?}");
        }
        assert_eq!(
            sym3_eigenvalues([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]),
            [1.0, 2.0, 3.0]
        );
    }

    /// Linear O=C=O: PMI1 = 0, PMI2 = PMI3 = 2 m_O d^2, asphericity 1
    /// (RDKit 2026.03.2 gives the same values).
    #[test]
    fn linear_molecule() {
        let block = "\n     RDKit          3D\n\n  3  2  0  0  0  0  0  0  0  0999 V2000\n   \
            -1.1600    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    \
            0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    \
            1.1600    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n  \
            1  2  2  0\n  2  3  2  0\nM  END\n";
        let mut mol = parse_mol_block(block).expect("valid mol block");
        let o = atom_mass(8, 0);
        let expected = 2.0 * o * 1.16 * 1.16;
        assert!(calc(&Pmi1, &mut mol).abs() < 1e-12);
        assert!((calc(&Pmi2, &mut mol) - expected).abs() < 1e-9);
        assert!((calc(&Pmi3, &mut mol) - expected).abs() < 1e-9);
        // linear: one non-zero gyration eigenvalue -> asphericity 1
        assert!((calc(&Asphericity, &mut mol) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_conformer_gives_nan() {
        let mut mol = crate::io::smiles::parse_smiles("CCO").unwrap();
        assert!(calc(&Pmi1, &mut mol).is_nan());
        assert!(calc(&Asphericity, &mut mol).is_nan());
    }
}
