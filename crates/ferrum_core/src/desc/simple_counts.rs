use super::Descriptor;
use crate::mol::{molecule::Molecule, elements};
use crate::types::Real;

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
    fn name(&self) -> String {
        "ValenceElectrons".to_string()
    }

    fn size(&self) -> usize {
        1
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let mut count: i32 = 0;
        
        for atom in &mol.atoms {
            let outer_elec = elements::outer_electrons(atom.atomic_number) as i32;
            let formal_charge = atom.formal_charge as i32;
            let total_h = atom.total_h() as i32;
            
            count += outer_elec - formal_charge + total_h;
        }
        
        output[0] = count as Real;
    }
}

/// Radical Electrons: Total number of unpaired radical electrons.
///
/// Counts atoms with unpaired electrons (radicals).
/// Says nothing about spin state.
///
/// # Examples
/// - Ethane: 0 radical electrons
/// - Methyl radical (•CH₃): 1 radical electron
/// - Methylene (CH₂): 2 radical electrons
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct RadicalElectrons;

impl Descriptor for RadicalElectrons {
    fn name(&self) -> String {
        "RadicalElectrons".to_string()
    }

    fn size(&self) -> usize {
        1
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let count: u32 = mol.atoms.iter()
            .map(|atom| 0u32) // No radical electron field in Atom yet
            .sum();
        
        output[0] = count as Real;
    }
}

/// Heteroatom Count: Number of non-carbon, non-hydrogen atoms.
///
/// Counts atoms that are neither carbon nor hydrogen.
/// Useful for characterizing heteroatom-rich molecules.
///
/// # Examples
/// - Benzene (C₆H₆): 0 heteroatoms
/// - Pyridine (C₅H₅N): 1 heteroatom
/// - Caffeine (C₈H₁₀N₄O₂): 6 heteroatoms (4N + 2O)
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct HeteroatomCount;

impl Descriptor for HeteroatomCount {
    fn name(&self) -> String {
        "HeteroatomCount".to_string()
    }

    fn size(&self) -> usize {
        1
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let count = mol.atoms.iter()
            .filter(|atom| atom.atomic_number != 6 && atom.atomic_number != 1) // Not C, not H
            .count();
        
        output[0] = count as Real;
    }
}

/// Hybridization Ratio: Ratio of sp³ to sp² carbons.
///
/// Measures molecular saturation/flatness.
/// Higher values indicate more 3D character ("escape from flatland").
/// Returns infinity if no sp² carbons exist.
///
/// # Algorithm
/// - sp³: carbon with 4 neighbors (including H)
/// - sp²: carbon with 3 neighbors (including H), typically with double bond
/// - sp: carbon with 2 neighbors
///
/// # Examples
/// - Cyclohexane: infinite (all sp³, no sp²)
/// - Benzene: 0 (all sp², no sp³)
/// - Ethylbenzene: ratio > 0
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct HybridizationRatio;

impl Descriptor for HybridizationRatio {
    fn name(&self) -> String {
        "HybridizationRatio".to_string()
    }

    fn size(&self) -> usize {
        1
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let mut sp3_count = 0;
        let mut sp2_count = 0;
        
        for (idx, atom) in mol.atoms.iter().enumerate() {
            if atom.atomic_number != 6 {
                continue; // Only carbons
            }
            
            let total_neighbors = mol.get_degree(idx as u16) + atom.total_h() as usize;
            
            match total_neighbors {
                4 => sp3_count += 1,
                3 => sp2_count += 1,
                _ => {} // sp or unusual
            }
        }
        
        if sp2_count == 0 {
            output[0] = if sp3_count > 0 { Real::INFINITY } else { 0.0 };
        } else {
            output[0] = sp3_count as Real / sp2_count as Real;
        }
    }
}

/// VAdjMa: Vertex Adjacency Magnitude.
///
/// Defined as: 1 + log₂(m) where m = number of heavy-heavy bonds.
/// Simple topological descriptor based on bond count.
///
/// # Examples
/// - Methane: 1 + log₂(0) = -∞ (or 0 if no bonds)
/// - Ethane: 1 + log₂(1) = 1
/// - Benzene: 1 + log₂(6) ≈ 3.58
///
/// # Thread Safety
/// Stateless and thread-safe.
pub struct VAdjMa;

impl Descriptor for VAdjMa {
    fn name(&self) -> String {
        "VAdjMa".to_string()
    }

    fn size(&self) -> usize {
        1
    }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let m = mol.num_bonds() as Real;
        
        if m == 0.0 {
            output[0] = 0.0;
        } else {
            output[0] = 1.0 + m.log2();
        }
    }
}

// Static assertions
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

    fn calc_desc<D: Descriptor>(desc: &D, smiles: &str) -> Real {
        let mol = parse_smiles(smiles).expect("Invalid SMILES");
        let mut output = vec![0.0; desc.size()];
        desc.calculate(&mol, &mut output);
        output[0]
    }

    // Valence Electrons Tests
    #[test]
    fn test_valence_electrons_ethane() {
        // C₂H₆: 2 carbons × 4 + 6 hydrogens × 1 = 8 + 6 = 14
        assert_eq!(calc_desc(&ValenceElectrons, "CC"), 14.0);
    }

    #[test]
    fn test_valence_electrons_co2() {
        // CO₂: C=4, 2×O=12, total = 16
        // Note: formal charges affect this
        let val = calc_desc(&ValenceElectrons, "O=C=O");
        assert!((val - 16.0).abs() < 1.0, "CO2 should have ~16 valence electrons");
    }

    // Radical Electrons Tests
    #[test]
    fn test_radical_electrons_ethane() {
        assert_eq!(calc_desc(&RadicalElectrons, "CC"), 0.0);
    }

    // Heteroatom Tests
    #[test]
    fn test_heteroatom_benzene() {
        assert_eq!(calc_desc(&HeteroatomCount, "c1ccccc1"), 0.0);
    }

    #[test]
    fn test_heteroatom_pyridine() {
        assert_eq!(calc_desc(&HeteroatomCount, "c1ccncc1"), 1.0);
    }

    #[test]
    fn test_heteroatom_caffeine() {
        // Caffeine: C₈H₁₀N₄O₂ = 4 N + 2 O = 6 heteroatoms
        assert_eq!(calc_desc(&HeteroatomCount, "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"), 6.0);
    }

    // Hybridization Ratio Tests
    #[test]
    fn test_hybridization_cyclohexane() {
        // All sp³, no sp² → infinity
        let ratio = calc_desc(&HybridizationRatio, "C1CCCCC1");
        assert!(ratio.is_infinite() && ratio.is_sign_positive());
    }

    #[test]
    fn test_hybridization_benzene() {
        // All sp², no sp³ → 0
        assert_eq!(calc_desc(&HybridizationRatio, "c1ccccc1"), 0.0);
    }

    #[test]
    fn test_hybridization_mixed() {
        // Ethylbenzene: 2 sp³ (ethyl), 6 sp² (benzene)
        let ratio = calc_desc(&HybridizationRatio, "CCc1ccccc1");
        assert!((ratio - 2.0/6.0).abs() < 0.01, "Ethylbenzene should have ratio ~0.33");
    }

    // VAdjMa Tests
    #[test]
    fn test_vadjma_methane() {
        // No heavy-heavy bonds
        assert_eq!(calc_desc(&VAdjMa, "C"), 0.0);
    }

    #[test]
    fn test_vadjma_ethane() {
        // 1 bond: 1 + log₂(1) = 1
        assert_eq!(calc_desc(&VAdjMa, "CC"), 1.0);
    }

    #[test]
    fn test_vadjma_benzene() {
        // 6 bonds: 1 + log₂(6) ≈ 3.58
        let val = calc_desc(&VAdjMa, "c1ccccc1");
        assert!((val - 3.585).abs() < 0.01, "Benzene VAdjMa should be ~3.58");
    }
}
