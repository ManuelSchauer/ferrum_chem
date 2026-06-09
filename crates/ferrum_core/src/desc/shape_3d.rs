use crate::desc::context::CalculationContext;
use crate::desc::Descriptor;
use crate::mol::elements::ATOMIC_MASSES;
use crate::mol::molecule::Molecule;
use crate::desc::DescriptorValue;

/// Calculates the center of mass and the principal moments of inertia (PMI)
/// of the first conformer. Returns (PMI1, PMI2, PMI3) sorted such that PMI1 <= PMI2 <= PMI3.
fn compute_pmis(mol: &Molecule) -> Option<(f64, f64, f64)> {
    if mol.conformers.is_empty() {
        return None;
    }
    
    let conf = &mol.conformers[0];
    if conf.positions.len() != mol.atoms.len() {
        return None; // Mismatch
    }
    
    let mut total_mass = 0.0;
    let mut com = [0.0; 3];
    let mut masses = Vec::with_capacity(mol.atoms.len());
    
    for (i, atom) in mol.atoms.iter().enumerate() {
        let z = atom.atomic_number as usize;
        let mass = if z < ATOMIC_MASSES.len() { ATOMIC_MASSES[z] as f64 } else { 0.0 };
        masses.push(mass);
        total_mass += mass;
        
        com[0] += mass * conf.positions[i][0];
        com[1] += mass * conf.positions[i][1];
        com[2] += mass * conf.positions[i][2];
    }
    
    if total_mass == 0.0 {
        return None;
    }
    
    com[0] /= total_mass;
    com[1] /= total_mass;
    com[2] /= total_mass;
    
    // Compute inertia tensor
    let mut ixx = 0.0;
    let mut iyy = 0.0;
    let mut izz = 0.0;
    let mut ixy = 0.0;
    let mut ixz = 0.0;
    let mut iyz = 0.0;
    
    for (i, p) in conf.positions.iter().enumerate() {
        let dx = p[0] - com[0];
        let dy = p[1] - com[1];
        let dz = p[2] - com[2];
        let mass = masses[i];
        
        ixx += mass * (dy * dy + dz * dz);
        iyy += mass * (dx * dx + dz * dz);
        izz += mass * (dx * dx + dy * dy);
        ixy -= mass * dx * dy;
        ixz -= mass * dx * dz;
        iyz -= mass * dy * dz;
    }
    
    // Diagonalize 3x3 symmetric matrix (Inertia Tensor) using Jacobi or standard eigenvalue solver.
    // For simplicity, we use a basic analytical or iterative approach for 3x3 symmetric.
    // Here we use a robust Jacobi eigenvalue algorithm for a 3x3 matrix.
    let mut a = [
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ];
    let mut d = [a[0][0], a[1][1], a[2][2]];
    
    for _step in 0..50 {
        let mut sm = 0.0;
        sm += a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if sm == 0.0 {
            break;
        }
        
        for p in 0..2 {
            for q in (p+1)..3 {
                let h = d[q] - d[p];
                if a[p][q].abs() > 1e-12 {
                    let t = if h.abs() > 1e-12 {
                        let theta = 0.5 * h / a[p][q];
                        let t = 1.0 / (theta.abs() + (1.0 + theta * theta).sqrt());
                        if theta < 0.0 { -t } else { t }
                    } else {
                        1.0
                    };
                    
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    let tau = s / (1.0 + c);
                    let h2 = t * a[p][q];
                    
                    d[p] -= h2;
                    d[q] += h2;
                    a[p][q] = 0.0;
                    
                    for j in 0..p {
                        let g = a[j][p];
                        let h = a[j][q];
                        a[j][p] = g - s * (h + g * tau);
                        a[j][q] = h + s * (g - h * tau);
                    }
                    for j in (p+1)..q {
                        let g = a[p][j];
                        let h = a[j][q];
                        a[p][j] = g - s * (h + g * tau);
                        a[j][q] = h + s * (g - h * tau);
                    }
                    for j in (q+1)..3 {
                        let g = a[p][j];
                        let h = a[q][j];
                        a[p][j] = g - s * (h + g * tau);
                        a[q][j] = h + s * (g - h * tau);
                    }
                }
            }
        }
    }
    
    let mut pmis = [d[0], d[1], d[2]];
    pmis.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some((pmis[0], pmis[1], pmis[2]))
}

#[derive(Debug, Clone)]
/// Implements Pmi1.
///
/// Function context: `crates/ferrum_core/src/desc/shape_3d.rs`.
/// Descriptor-layer note: computes molecular properties for ML pipelines.
pub struct Pmi1;
impl Descriptor for Pmi1 {
    /// Implements name.
    ///
    /// Function context: `crates/ferrum_core/src/desc/shape_3d.rs`.
    /// Descriptor-layer note: computes molecular properties for ML pipelines.
    fn name(&self) -> &str { "PMI1" }
    /// Implements calculate.
    ///
    /// Function context: `crates/ferrum_core/src/desc/shape_3d.rs`.
    /// Descriptor-layer note: computes molecular properties for ML pipelines.
    fn calculate(&self, mol: &mut Molecule, _: &mut CalculationContext) -> DescriptorValue {
        compute_pmis(mol).map(|p| DescriptorValue::Float64(p.0)).unwrap_or(DescriptorValue::Float64(f64::NAN))
    }
}

#[derive(Debug, Clone)]
/// Implements Pmi2.
///
/// Function context: `crates/ferrum_core/src/desc/shape_3d.rs`.
/// Descriptor-layer note: computes molecular properties for ML pipelines.
pub struct Pmi2;
impl Descriptor for Pmi2 {
    /// Implements name.
    ///
    /// Function context: `crates/ferrum_core/src/desc/shape_3d.rs`.
    /// Descriptor-layer note: computes molecular properties for ML pipelines.
    fn name(&self) -> &str { "PMI2" }
    /// Implements calculate.
    ///
    /// Function context: `crates/ferrum_core/src/desc/shape_3d.rs`.
    /// Descriptor-layer note: computes molecular properties for ML pipelines.
    fn calculate(&self, mol: &mut Molecule, _: &mut CalculationContext) -> DescriptorValue {
        compute_pmis(mol).map(|p| DescriptorValue::Float64(p.1)).unwrap_or(DescriptorValue::Float64(f64::NAN))
    }
}

#[derive(Debug, Clone)]
/// Implements Pmi3.
///
/// Function context: `crates/ferrum_core/src/desc/shape_3d.rs`.
/// Descriptor-layer note: computes molecular properties for ML pipelines.
pub struct Pmi3;
impl Descriptor for Pmi3 {
    /// Implements name.
    ///
    /// Function context: `crates/ferrum_core/src/desc/shape_3d.rs`.
    /// Descriptor-layer note: computes molecular properties for ML pipelines.
    fn name(&self) -> &str { "PMI3" }
    /// Implements calculate.
    ///
    /// Function context: `crates/ferrum_core/src/desc/shape_3d.rs`.
    /// Descriptor-layer note: computes molecular properties for ML pipelines.
    fn calculate(&self, mol: &mut Molecule, _: &mut CalculationContext) -> DescriptorValue {
        compute_pmis(mol).map(|p| DescriptorValue::Float64(p.2)).unwrap_or(DescriptorValue::Float64(f64::NAN))
    }
}

#[derive(Debug, Clone)]
/// Implements Asphericity.
///
/// Function context: `crates/ferrum_core/src/desc/shape_3d.rs`.
/// Descriptor-layer note: computes molecular properties for ML pipelines.
pub struct Asphericity;
impl Descriptor for Asphericity {
    /// Implements name.
    ///
    /// Function context: `crates/ferrum_core/src/desc/shape_3d.rs`.
    /// Descriptor-layer note: computes molecular properties for ML pipelines.
    fn name(&self) -> &str { "Asphericity" }
    /// Implements calculate.
    ///
    /// Function context: `crates/ferrum_core/src/desc/shape_3d.rs`.
    /// Descriptor-layer note: computes molecular properties for ML pipelines.
    fn calculate(&self, mol: &mut Molecule, _: &mut CalculationContext) -> DescriptorValue {
        compute_pmis(mol)
            .map(|(pmi1, pmi2, pmi3)| {
                // Asphericity = 0.5 * ((I3-I2)^2 + (I3-I1)^2 + (I2-I1)^2) / (I1^2 + I2^2 + I3^2)
                let num = (pmi3 - pmi2).powi(2) + (pmi3 - pmi1).powi(2) + (pmi2 - pmi1).powi(2);
                let den = pmi1.powi(2) + pmi2.powi(2) + pmi3.powi(2);
                if den == 0.0 { DescriptorValue::Float64(0.0) } else { DescriptorValue::Float64(0.5 * num / den) }
            })
            .unwrap_or(DescriptorValue::Float64(f64::NAN))
    }
}
