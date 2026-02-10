//! Additional fraction and count descriptors commonly used in QSAR.
//!
//! Includes:
//! - FractionCSP3: fraction of sp³ carbons (drug-likeness)
//! - RingTypeCounts: aromatic, aliphatic, saturated ring counts as Descriptor
//! - NumAmideBonds: amide bond count via SMARTS
//! - NumHeavyAtoms: heavy atom count
//! - AromaticAtomFraction: fraction of atoms that are aromatic
//! - HallKierAlpha: molecular flexibility descriptor
//! - BertzCT: molecular complexity index
//! - Ipc: information content descriptor
//!
//! References:
//! - Lovering et al. (2009) J. Med. Chem. 52, 6752 (FractionCSP3)
//! - Bertz (1981) JACS 103, 3599 (BertzCT)
//! - Kier & Hall (1986) Molecular Connectivity (HallKierAlpha)

use super::Descriptor;
use crate::algo::rings::find_sssr;
use crate::mol::elements::outer_electrons;
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType, Real};

// ---------------------------------------------------------------------------
// FractionCSP3
// ---------------------------------------------------------------------------

/// Fraction of sp³ carbons (carbon sp3 fraction).
///
/// FractionCSP3 = (number of sp3 carbons) / (total number of carbons).
/// Returns 0 for molecules with no carbons.
/// Drug-likeness metric: higher values → more 3D character.
pub struct FractionCSP3;

impl Descriptor for FractionCSP3 {
    fn name(&self) -> String { "FractionCSP3".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let mut total_c = 0u32;
        let mut sp3_c = 0u32;

        for (i, atom) in mol.atoms.iter().enumerate() {
            if atom.atomic_number != 6 { continue; }
            total_c += 1;

            // Infer hybridization from bonds
            let has_triple = mol.adjacency[i].iter().any(|&bi| {
                mol.bonds[bi as usize].bond_type == BondType::Triple
            });
            let has_double_or_arom = mol.adjacency[i].iter().any(|&bi| {
                let bt = mol.bonds[bi as usize].bond_type;
                bt == BondType::Double || bt == BondType::Aromatic
            });

            if !has_triple && !has_double_or_arom && !atom.is_aromatic() {
                sp3_c += 1;
            }
        }

        output[0] = if total_c > 0 { sp3_c as Real / total_c as Real } else { 0.0 };
    }
}

// ---------------------------------------------------------------------------
// Ring Type Counts (as Descriptor struct)
// ---------------------------------------------------------------------------

/// Ring type counts: [aromatic, aliphatic, saturated, heteroaromatic, heterocyclic].
///
/// - Aromatic: all atoms aromatic
/// - Aliphatic: no atom aromatic
/// - Saturated: all bonds single
/// - Heteroaromatic: aromatic + contains non-C atom
/// - Heterocyclic: contains non-C atom (any ring type)
pub struct RingTypeCounts;

impl Descriptor for RingTypeCounts {
    fn name(&self) -> String { "RingTypeCounts".to_string() }
    fn size(&self) -> usize { 5 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        for v in output.iter_mut() { *v = 0.0; }

        let mut mol_clone = mol.clone();
        let rings = find_sssr(&mut mol_clone);

        for ring in &rings {
            let all_aromatic = ring.iter().all(|&ai| mol.atoms[ai as usize].is_aromatic());
            let any_aromatic = ring.iter().any(|&ai| mol.atoms[ai as usize].is_aromatic());
            let has_heteroatom = ring.iter().any(|&ai| {
                let an = mol.atoms[ai as usize].atomic_number;
                an != 6 && an != 1
            });

            let all_single = ring_all_single_bonds(mol, ring);

            if all_aromatic { output[0] += 1.0; }           // aromatic
            if !any_aromatic { output[1] += 1.0; }          // aliphatic
            if all_single { output[2] += 1.0; }             // saturated
            if all_aromatic && has_heteroatom { output[3] += 1.0; } // heteroaromatic
            if has_heteroatom { output[4] += 1.0; }         // heterocyclic
        }
    }
}

fn ring_all_single_bonds(mol: &Molecule, ring: &[AtomIdx]) -> bool {
    for i in 0..ring.len() {
        let a1 = ring[i];
        let a2 = ring[(i + 1) % ring.len()];
        let is_single = mol.adjacency[a1 as usize].iter().any(|&bi| {
            let bond = &mol.bonds[bi as usize];
            ((bond.source == a1 && bond.target == a2) || (bond.source == a2 && bond.target == a1))
                && bond.bond_type == BondType::Single
        });
        if !is_single { return false; }
    }
    true
}

// ---------------------------------------------------------------------------
// NumAmideBonds
// ---------------------------------------------------------------------------

/// Count of amide bonds: [NX3][CX3](=[OX1])[#6,#1,#7]
pub struct NumAmideBonds;

impl Descriptor for NumAmideBonds {
    fn name(&self) -> String { "NumAmideBonds".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        use crate::io::smarts::parse_smarts;
        use crate::algo::iso::count_matches;
        use once_cell::sync::Lazy;

        static AMIDE: Lazy<crate::io::smarts::QueryGraph> = Lazy::new(|| {
            parse_smarts("[NX3][CX3](=[OX1])").expect("valid SMARTS")
        });

        output[0] = count_matches(mol, &AMIDE) as Real;
    }
}

// ---------------------------------------------------------------------------
// NumHeavyAtoms
// ---------------------------------------------------------------------------

/// Count of heavy atoms (non-hydrogen).
pub struct NumHeavyAtoms;

impl Descriptor for NumHeavyAtoms {
    fn name(&self) -> String { "NumHeavyAtoms".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        output[0] = mol.atoms.iter()
            .filter(|a| a.atomic_number > 1)
            .count() as Real;
    }
}

// ---------------------------------------------------------------------------
// AromaticAtomFraction
// ---------------------------------------------------------------------------

/// Fraction of heavy atoms that are aromatic.
pub struct AromaticAtomFraction;

impl Descriptor for AromaticAtomFraction {
    fn name(&self) -> String { "AromaticAtomFraction".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let heavy: usize = mol.atoms.iter().filter(|a| a.atomic_number > 1).count();
        if heavy == 0 {
            output[0] = 0.0;
            return;
        }
        let aromatic: usize = mol.atoms.iter()
            .filter(|a| a.atomic_number > 1 && a.is_aromatic())
            .count();
        output[0] = aromatic as Real / heavy as Real;
    }
}

// ---------------------------------------------------------------------------
// HallKierAlpha
// ---------------------------------------------------------------------------

/// Hall-Kier alpha: molecular flexibility/shape modifier.
///
/// α = Σ (αᵢ) where αᵢ = (rᵢᶜ - rᶜ_C) / rᶜ_C
/// rᵢᶜ = covalent radius of atom i, rᶜ_C = covalent radius of carbon (0.77 Å).
///
/// Quantifies how much the molecule differs from an all-carbon skeleton.
pub struct HallKierAlpha;

fn covalent_radius(atomic_number: u8) -> f64 {
    match atomic_number {
        1 => 0.32, 5 => 0.82, 6 => 0.77, 7 => 0.75, 8 => 0.73,
        9 => 0.71, 14 => 1.11, 15 => 1.06, 16 => 1.02, 17 => 0.99,
        35 => 1.14, 53 => 1.33, _ => 0.77,
    }
}

impl Descriptor for HallKierAlpha {
    fn name(&self) -> String { "HallKierAlpha".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let rc_c = 0.77f64; // covalent radius of sp3 carbon
        let mut alpha = 0.0f64;

        for atom in &mol.atoms {
            if atom.atomic_number <= 1 { continue; }
            let rc = covalent_radius(atom.atomic_number);
            alpha += (rc - rc_c) / rc_c;
        }

        output[0] = alpha as Real;
    }
}

// ---------------------------------------------------------------------------
// BertzCT - Molecular Complexity
// ---------------------------------------------------------------------------

/// Bertz complexity index CT.
///
/// CT = Σ 2^bᵢ × log₂(2^bᵢ) + Σ elements × log₂(elements)
///
/// Simplified Bertz: counts bond types, element types, and symmetry.
/// Higher CT → more complex molecule.
pub struct BertzCT;

impl Descriptor for BertzCT {
    fn name(&self) -> String { "BertzCT".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        if mol.num_atoms() == 0 {
            output[0] = 0.0;
            return;
        }

        // Connection complexity: based on vertex degrees
        let n = mol.num_atoms();
        let mut degree_counts = [0u32; 10]; // degree 0-9
        for i in 0..n {
            let d = mol.get_degree(i as AtomIdx).min(9);
            degree_counts[d] += 1;
        }

        let conn_complexity = shannon_entropy(&degree_counts) * mol.num_bonds() as f64;

        // Element complexity: based on element diversity
        let mut element_counts = [0u32; 120];
        for atom in &mol.atoms {
            if atom.atomic_number > 0 {
                element_counts[atom.atomic_number as usize] += 1;
            }
        }
        let elem_complexity = shannon_entropy(&element_counts) * n as f64;

        // Bond type complexity
        let mut bond_type_counts = [0u32; 6]; // Single, Double, Triple, Aromatic, Other
        for bond in &mol.bonds {
            let idx = match bond.bond_type {
                BondType::Single => 0,
                BondType::Double => 1,
                BondType::Triple => 2,
                BondType::Aromatic => 3,
                _ => 4,
            };
            bond_type_counts[idx] += 1;
        }
        let bond_complexity = shannon_entropy(&bond_type_counts) * mol.num_bonds() as f64;

        output[0] = (conn_complexity + elem_complexity + bond_complexity) as Real;
    }
}

fn shannon_entropy(counts: &[u32]) -> f64 {
    let total: u32 = counts.iter().sum();
    if total == 0 { return 0.0; }
    let t = total as f64;
    counts.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / t;
            -p * p.log2()
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Ipc - Information Content
// ---------------------------------------------------------------------------

/// Information content (Bonchev-Trinajstić).
///
/// Based on the partition of edges by equivalence classes.
/// Ipc = N × log₂(N) - Σ nᵢ × log₂(nᵢ)
/// where N = total bonds, nᵢ = bonds in equivalence class i.
///
/// Simplified: uses bond-type × endpoint-degree as class key.
pub struct Ipc;

impl Descriptor for Ipc {
    fn name(&self) -> String { "Ipc".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let m = mol.num_bonds();
        if m == 0 {
            output[0] = 0.0;
            return;
        }

        // Classify each bond by (bond_type, min_degree, max_degree)
        let mut class_counts: std::collections::HashMap<(u8, usize, usize), u32> =
            std::collections::HashMap::new();

        for bond in &mol.bonds {
            let d1 = mol.get_degree(bond.source);
            let d2 = mol.get_degree(bond.target);
            let (lo, hi) = if d1 <= d2 { (d1, d2) } else { (d2, d1) };
            let bt = match bond.bond_type {
                BondType::Single => 1,
                BondType::Double => 2,
                BondType::Triple => 3,
                BondType::Aromatic => 4,
                _ => 0,
            };
            *class_counts.entry((bt, lo, hi)).or_insert(0) += 1;
        }

        let mf = m as f64;
        let total_info = mf * mf.log2();
        let class_info: f64 = class_counts.values()
            .map(|&c| {
                let cf = c as f64;
                cf * cf.log2()
            })
            .sum();

        output[0] = (total_info - class_info) as Real;
    }
}

// ---------------------------------------------------------------------------
// FragmentComplexity
// ---------------------------------------------------------------------------

/// Fragment complexity: Shannon entropy of atomic number distribution.
///
/// Measures elemental diversity in the molecule.
/// FC = -Σ pᵢ × log₂(pᵢ) where pᵢ = fraction of atoms with element i.
pub struct FragmentComplexity;

impl Descriptor for FragmentComplexity {
    fn name(&self) -> String { "FragmentComplexity".to_string() }
    fn size(&self) -> usize { 1 }

    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let mut element_counts = [0u32; 120];
        for atom in &mol.atoms {
            if atom.atomic_number > 0 {
                element_counts[atom.atomic_number as usize] += 1;
            }
        }
        output[0] = shannon_entropy(&element_counts) as Real;
    }
}

// ---------------------------------------------------------------------------
// Thread safety
// ---------------------------------------------------------------------------
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<FractionCSP3>;
    let _ = assert_send_sync::<RingTypeCounts>;
    let _ = assert_send_sync::<NumAmideBonds>;
    let _ = assert_send_sync::<NumHeavyAtoms>;
    let _ = assert_send_sync::<AromaticAtomFraction>;
    let _ = assert_send_sync::<HallKierAlpha>;
    let _ = assert_send_sync::<BertzCT>;
    let _ = assert_send_sync::<Ipc>;
    let _ = assert_send_sync::<FragmentComplexity>;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc<D: Descriptor>(d: &D, smiles: &str) -> Vec<Real> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        let mut output = vec![0.0; d.size()];
        d.calculate(&mol, &mut output);
        output
    }

    // -- FractionCSP3 --

    #[test]
    fn test_fsp3_ethane() {
        let v = calc(&FractionCSP3, "CC");
        assert!((v[0] - 1.0).abs() < 0.01, "all sp3: {}", v[0]);
    }

    #[test]
    fn test_fsp3_ethene() {
        let v = calc(&FractionCSP3, "C=C");
        assert!((v[0]).abs() < 0.01, "all sp2: {}", v[0]);
    }

    #[test]
    fn test_fsp3_propanol() {
        let v = calc(&FractionCSP3, "CCCO");
        // 3 carbons, all sp3
        assert!((v[0] - 1.0).abs() < 0.01, "all sp3: {}", v[0]);
    }

    #[test]
    fn test_fsp3_benzene() {
        let v = calc(&FractionCSP3, "c1ccccc1");
        assert!((v[0]).abs() < 0.01, "no sp3 carbons: {}", v[0]);
    }

    #[test]
    fn test_fsp3_toluene() {
        let v = calc(&FractionCSP3, "Cc1ccccc1");
        // 1 sp3 out of 7 carbons = 0.143
        assert!((v[0] - 1.0 / 7.0).abs() < 0.02, "fsp3 = {}", v[0]);
    }

    // -- RingTypeCounts --

    #[test]
    fn test_ring_types_benzene() {
        let v = calc(&RingTypeCounts, "c1ccccc1");
        assert!((v[0] - 1.0).abs() < 0.01, "1 aromatic: {}", v[0]);
        assert!((v[1]).abs() < 0.01, "0 aliphatic: {}", v[1]);
        assert!((v[2]).abs() < 0.01, "0 saturated: {}", v[2]);
    }

    #[test]
    fn test_ring_types_cyclohexane() {
        let v = calc(&RingTypeCounts, "C1CCCCC1");
        assert!((v[0]).abs() < 0.01, "0 aromatic");
        assert!((v[1] - 1.0).abs() < 0.01, "1 aliphatic: {}", v[1]);
        assert!((v[2] - 1.0).abs() < 0.01, "1 saturated: {}", v[2]);
    }

    #[test]
    fn test_ring_types_pyridine() {
        let v = calc(&RingTypeCounts, "c1ccncc1");
        assert!((v[0] - 1.0).abs() < 0.01, "1 aromatic");
        assert!((v[3] - 1.0).abs() < 0.01, "1 heteroaromatic: {}", v[3]);
        assert!((v[4] - 1.0).abs() < 0.01, "1 heterocyclic: {}", v[4]);
    }

    #[test]
    fn test_ring_types_morpholine() {
        // Saturated heterocyclic ring with O
        let v = calc(&RingTypeCounts, "C1COCCN1");
        assert!((v[2] - 1.0).abs() < 0.01, "1 saturated: {}", v[2]);
        assert!((v[4] - 1.0).abs() < 0.01, "1 heterocyclic: {}", v[4]);
    }

    // -- NumAmideBonds --

    #[test]
    fn test_amide_acetamide() {
        let v = calc(&NumAmideBonds, "CC(=O)N");
        assert!((v[0] - 1.0).abs() < 0.01, "1 amide: {}", v[0]);
    }

    #[test]
    fn test_amide_ethane() {
        let v = calc(&NumAmideBonds, "CC");
        assert!((v[0]).abs() < 0.01, "0 amides");
    }

    #[test]
    fn test_amide_peptide() {
        // Gly-Gly: two amide bonds
        let v = calc(&NumAmideBonds, "NCC(=O)NCC(=O)O");
        assert!(v[0] >= 1.0, "should have amide bonds: {}", v[0]);
    }

    // -- NumHeavyAtoms --

    #[test]
    fn test_heavy_atoms_ethanol() {
        let v = calc(&NumHeavyAtoms, "CCO");
        assert!((v[0] - 3.0).abs() < 0.01, "3 heavy atoms: {}", v[0]);
    }

    // -- AromaticAtomFraction --

    #[test]
    fn test_arom_frac_benzene() {
        let v = calc(&AromaticAtomFraction, "c1ccccc1");
        assert!((v[0] - 1.0).abs() < 0.01, "all aromatic: {}", v[0]);
    }

    #[test]
    fn test_arom_frac_hexane() {
        let v = calc(&AromaticAtomFraction, "CCCCCC");
        assert!((v[0]).abs() < 0.01, "no aromatic: {}", v[0]);
    }

    #[test]
    fn test_arom_frac_toluene() {
        let v = calc(&AromaticAtomFraction, "Cc1ccccc1");
        // 6 aromatic / 7 heavy = 0.857
        assert!((v[0] - 6.0 / 7.0).abs() < 0.02, "frac = {}", v[0]);
    }

    // -- HallKierAlpha --

    #[test]
    fn test_hka_all_carbon() {
        let v = calc(&HallKierAlpha, "CCCC");
        // All C: alpha = 0 (no deviation from carbon skeleton)
        assert!((v[0]).abs() < 0.01, "all-C alpha = {}", v[0]);
    }

    #[test]
    fn test_hka_oxygen() {
        let v = calc(&HallKierAlpha, "CCO");
        // O has smaller covalent radius than C: alpha < 0
        assert!(v[0] < 0.0, "O makes alpha negative: {}", v[0]);
    }

    #[test]
    fn test_hka_bromine() {
        let v = calc(&HallKierAlpha, "CCBr");
        // Br has larger covalent radius: alpha > 0
        assert!(v[0] > 0.0, "Br makes alpha positive: {}", v[0]);
    }

    // -- BertzCT --

    #[test]
    fn test_bertz_methane() {
        let v = calc(&BertzCT, "C");
        assert!((v[0]).abs() < 0.01, "methane CT = {}", v[0]);
    }

    #[test]
    fn test_bertz_complex_gt_simple() {
        let simple = calc(&BertzCT, "CCCC");
        let complex = calc(&BertzCT, "c1ccc(O)c(N)c1");
        assert!(complex[0] > simple[0],
            "complex {} > simple {}", complex[0], simple[0]);
    }

    #[test]
    fn test_bertz_benzene() {
        let v = calc(&BertzCT, "c1ccccc1");
        // Benzene: uniform degree, single element type, single bond type → CT ≈ 0
        assert!(v[0].abs() < 0.01, "benzene CT = {}", v[0]);
    }

    // -- Ipc --

    #[test]
    fn test_ipc_ethane() {
        let v = calc(&Ipc, "CC");
        // 1 bond, 1 class → Ipc = 0 (no diversity)
        assert!((v[0]).abs() < 0.01, "ethane Ipc = {}", v[0]);
    }

    #[test]
    fn test_ipc_diverse() {
        let v = calc(&Ipc, "CC=O");
        // Has single + double bonds with different degree endpoints
        assert!(v[0] > 0.0, "should have info content: {}", v[0]);
    }

    // -- FragmentComplexity --

    #[test]
    fn test_fc_all_carbon() {
        let v = calc(&FragmentComplexity, "CCCC");
        // All same element: entropy = 0
        assert!((v[0]).abs() < 0.01, "all-C FC = {}", v[0]);
    }

    #[test]
    fn test_fc_diverse() {
        let v = calc(&FragmentComplexity, "CCNO");
        // 3 elements: C, N, O → entropy > 0
        assert!(v[0] > 0.0, "diverse FC = {}", v[0]);
    }

    #[test]
    fn test_fc_more_diverse_higher() {
        let less = calc(&FragmentComplexity, "CCCC");
        let more = calc(&FragmentComplexity, "CCNO");
        assert!(more[0] > less[0]);
    }
}
