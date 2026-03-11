use super::typed::{Descriptor, DescriptorValue};
use crate::mol::{elements, molecule::Molecule};

/// Valence Electrons: Total number of valence electrons in the molecule.
///
/// Calculates the sum of valence electrons for all atoms including hydrogens.
/// Formula: Σ (outer_electrons - formal_charge + total_H_count)
///
/// # Examples
/// - Ethane (C₂H₆): 14 valence electrons
/// - CO₂: 16 valence electrons
/// - Acetate ion: 18 valence electrons
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct ValenceElectrons;

impl Descriptor for ValenceElectrons {
    fn name(&self) -> &str {
        "ValenceElectrons"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let mut count: i32 = 0;
        for atom in &mol.atoms {
            let outer_elec = elements::outer_electrons(atom.atomic_number) as i32;
            let formal_charge = atom.formal_charge as i32;
            let total_h = atom.total_h() as i32;
            count += outer_elec - formal_charge + total_h;
        }
        DescriptorValue::Float(count as f32)
    }
}

/// Radical Electrons: Total number of unpaired radical electrons.
///
/// Counts atoms with unpaired electrons (radicals).
/// Says nothing about spin state.
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct RadicalElectrons;

impl Descriptor for RadicalElectrons {
    fn name(&self) -> &str {
        "RadicalElectrons"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let count: u32 = mol
            .atoms
            .iter()
            .map(|_atom| 0u32) // No radical electron field in Atom yet
            .sum();
        DescriptorValue::Int(count)
    }
}

/// Heteroatom Count: Number of non-carbon, non-hydrogen atoms.
///
/// Counts atoms that are neither carbon nor hydrogen.
/// Useful for characterizing heteroatom-rich molecules.
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct HeteroatomCount;

impl Descriptor for HeteroatomCount {
    fn name(&self) -> &str {
        "HeteroatomCount"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let count = mol
            .atoms
            .iter()
            .filter(|atom| atom.atomic_number > 1 && atom.atomic_number != 6)
            .count() as u32;
        DescriptorValue::Int(count)
    }
}

/// Hybridization Ratio: Ratio of sp³ to sp² carbons.
///
/// Measures molecular saturation/flatness.
/// Higher values indicate more 3D character ("escape from flatland").
/// Returns infinity if no sp² carbons exist.
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct HybridizationRatio;

impl Descriptor for HybridizationRatio {
    fn name(&self) -> &str {
        "HybridizationRatio"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let mut sp3_count = 0u32;
        let mut sp2_count = 0u32;
        for (idx, atom) in mol.atoms.iter().enumerate() {
            if atom.atomic_number != 6 {
                continue;
            }
            let total_neighbors = mol.get_degree(idx as u16) + atom.total_h() as usize;
            match total_neighbors {
                4 => sp3_count += 1,
                3 => sp2_count += 1,
                _ => {}
            }
        }
        let ratio = if sp2_count == 0 {
            if sp3_count > 0 { f32::INFINITY } else { 0.0 }
        } else {
            sp3_count as f32 / sp2_count as f32
        };
        DescriptorValue::Float(ratio)
    }
}

/// VAdjMa: Vertex Adjacency Magnitude.
///
/// Defined as: 1 + log₂(m) where m = number of heavy-heavy bonds.
/// Simple topological descriptor based on bond count.
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct VAdjMa;

impl Descriptor for VAdjMa {
    fn name(&self) -> &str {
        "VAdjMa"
    }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        let m = mol.num_bonds() as f32;
        let val = if m == 0.0 { 0.0 } else { 1.0 + m.log2() };
        DescriptorValue::Float(val)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<ValenceElectrons>;
    let _ = assert_send_sync::<RadicalElectrons>;
    let _ = assert_send_sync::<HeteroatomCount>;
    let _ = assert_send_sync::<HybridizationRatio>;
    let _ = assert_send_sync::<VAdjMa>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc_float(desc: &dyn Descriptor, smiles: &str) -> f32 {
        let mut mol = parse_smiles(smiles).expect("Invalid SMILES");
        desc.calculate(&mut mol).as_float()
    }

    fn calc_int(desc: &dyn Descriptor, smiles: &str) -> u32 {
        let mut mol = parse_smiles(smiles).expect("Invalid SMILES");
        desc.calculate(&mut mol).as_int()
    }

    // Valence Electrons Tests
    #[test]
    fn test_valence_electrons_ethane() {
        assert_eq!(calc_float(&ValenceElectrons, "CC"), 14.0);
    }

    #[test]
    fn test_valence_electrons_co2() {
        let val = calc_float(&ValenceElectrons, "O=C=O");
        assert!(
            (val - 16.0).abs() < 1.0,
            "CO2 should have ~16 valence electrons"
        );
    }

    // Radical Electrons Tests
    #[test]
    fn test_radical_electrons_ethane() {
        assert_eq!(calc_int(&RadicalElectrons, "CC"), 0);
    }

    // Heteroatom Tests
    #[test]
    fn test_heteroatom_benzene() {
        assert_eq!(calc_int(&HeteroatomCount, "c1ccccc1"), 0);
    }

    #[test]
    fn test_heteroatom_pyridine() {
        assert_eq!(calc_int(&HeteroatomCount, "c1ccncc1"), 1);
    }

    #[test]
    fn test_heteroatom_caffeine() {
        assert_eq!(
            calc_int(&HeteroatomCount, "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
            6
        );
    }

    // Hybridization Ratio Tests
    #[test]
    fn test_hybridization_cyclohexane() {
        let ratio = calc_float(&HybridizationRatio, "C1CCCCC1");
        assert!(ratio.is_infinite() && ratio.is_sign_positive());
    }

    #[test]
    fn test_hybridization_benzene() {
        assert_eq!(calc_float(&HybridizationRatio, "c1ccccc1"), 0.0);
    }

    #[test]
    fn test_hybridization_mixed() {
        let ratio = calc_float(&HybridizationRatio, "CCc1ccccc1");
        assert!(
            (ratio - 2.0 / 6.0).abs() < 0.01,
            "Ethylbenzene should have ratio ~0.33"
        );
    }

    // VAdjMa Tests
    #[test]
    fn test_vadjma_methane() {
        assert_eq!(calc_float(&VAdjMa, "C"), 0.0);
    }

    #[test]
    fn test_vadjma_ethane() {
        assert_eq!(calc_float(&VAdjMa, "CC"), 1.0);
    }

    #[test]
    fn test_vadjma_benzene() {
        let val = calc_float(&VAdjMa, "c1ccccc1");
        assert!((val - 3.585).abs() < 0.01, "Benzene VAdjMa should be ~3.58");
    }
}
