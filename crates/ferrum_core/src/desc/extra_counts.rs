//! Additional fraction and count descriptors commonly used in QSAR.
//!
//! Includes:
//! - FractionCSP3: fraction of sp³ carbons (drug-likeness)
//! - NumAmideBonds: amide bond count via SMARTS
//! - AromaticAtomFraction: fraction of atoms that are aromatic
//! - HallKierAlpha: molecular flexibility descriptor
//! - BertzCT: molecular complexity index
//! - Ipc: information content descriptor
//! - FragmentComplexity: Shannon entropy of element distribution
//!
//! References:
//! - Lovering et al. (2009) J. Med. Chem. 52, 6752 (FractionCSP3)
//! - Bertz (1981) JACS 103, 3599 (BertzCT)
//! - Kier & Hall (1986) Molecular Connectivity (HallKierAlpha)

use super::typed::{DescriptorValue, Descriptor};
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondType};

// ---------------------------------------------------------------------------
// FractionCSP3
// ---------------------------------------------------------------------------

/// Fraction of sp³ carbons (carbon sp3 fraction).
///
/// FractionCSP3 = (number of sp3 carbons) / (total number of carbons).
/// Returns 0 for molecules with no carbons.
/// Drug-likeness metric: higher values → more 3D character.
pub struct FractionCSP3;

fn compute_fraction_csp3(mol: &Molecule) -> f32 {
    let mut total_c = 0u32;
    let mut sp3_c = 0u32;

    for (i, atom) in mol.atoms.iter().enumerate() {
        if atom.atomic_number != 6 { continue; }
        total_c += 1;

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

    if total_c > 0 { sp3_c as f32 / total_c as f32 } else { 0.0 }
}

impl Descriptor for FractionCSP3 {
    fn name(&self) -> &str { "FractionCSP3" }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_fraction_csp3(mol))
    }
}

// ---------------------------------------------------------------------------
// Ring Type Counts (helper function only; typed descriptor in ring_types.rs)
// ---------------------------------------------------------------------------

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

/// Compute ring type counts: [aromatic, aliphatic, saturated, heteroaromatic, heterocyclic].
pub fn compute_ring_type_counts(mol: &Molecule) -> [u32; 5] {
    use crate::algo::rings::find_sssr;

    let mut counts = [0u32; 5];
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

        if all_aromatic { counts[0] += 1; }
        if !any_aromatic { counts[1] += 1; }
        if all_single { counts[2] += 1; }
        if all_aromatic && has_heteroatom { counts[3] += 1; }
        if has_heteroatom { counts[4] += 1; }
    }

    counts
}

// ---------------------------------------------------------------------------
// NumAmideBonds
// ---------------------------------------------------------------------------

/// Count of amide bonds: [NX3][CX3](=[OX1])
pub struct NumAmideBonds;

fn compute_num_amide_bonds(mol: &Molecule) -> u32 {
    use crate::io::smarts::parse_smarts;
    use crate::algo::iso::count_matches;
    use once_cell::sync::Lazy;

    static AMIDE: Lazy<crate::io::smarts::QueryGraph> = Lazy::new(|| {
        parse_smarts("[NX3][CX3](=[OX1])").expect("valid SMARTS")
    });

    count_matches(mol, &AMIDE) as u32
}

impl Descriptor for NumAmideBonds {
    fn name(&self) -> &str { "NumAmideBonds" }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Int(compute_num_amide_bonds(mol))
    }
}

// ---------------------------------------------------------------------------
// AromaticAtomFraction
// ---------------------------------------------------------------------------

/// Fraction of heavy atoms that are aromatic.
pub struct AromaticAtomFraction;

fn compute_aromatic_atom_fraction(mol: &Molecule) -> f32 {
    let heavy: usize = mol.atoms.iter().filter(|a| a.atomic_number > 1).count();
    if heavy == 0 { return 0.0; }
    let aromatic: usize = mol.atoms.iter()
        .filter(|a| a.atomic_number > 1 && a.is_aromatic())
        .count();
    aromatic as f32 / heavy as f32
}

impl Descriptor for AromaticAtomFraction {
    fn name(&self) -> &str { "AromaticAtomFraction" }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_aromatic_atom_fraction(mol))
    }
}

// ---------------------------------------------------------------------------
// HallKierAlpha
// ---------------------------------------------------------------------------

/// Hall-Kier alpha: molecular flexibility/shape modifier.
///
/// α = Σ (αᵢ) where αᵢ = (rᵢᶜ - rᶜ_C) / rᶜ_C
/// rᵢᶜ = covalent radius of atom i, rᶜ_C = covalent radius of carbon (0.77 Å).
pub struct HallKierAlpha;

fn covalent_radius(atomic_number: u8) -> f64 {
    match atomic_number {
        1 => 0.32, 5 => 0.82, 6 => 0.77, 7 => 0.75, 8 => 0.73,
        9 => 0.71, 14 => 1.11, 15 => 1.06, 16 => 1.02, 17 => 0.99,
        35 => 1.14, 53 => 1.33, _ => 0.77,
    }
}

fn compute_hall_kier_alpha(mol: &Molecule) -> f32 {
    let rc_c = 0.77f64;
    let mut alpha = 0.0f64;
    for atom in &mol.atoms {
        if atom.atomic_number <= 1 { continue; }
        let rc = covalent_radius(atom.atomic_number);
        alpha += (rc - rc_c) / rc_c;
    }
    alpha as f32
}

impl Descriptor for HallKierAlpha {
    fn name(&self) -> &str { "HallKierAlpha" }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_hall_kier_alpha(mol))
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

fn compute_bertz_ct(mol: &Molecule) -> f32 {
    if mol.num_atoms() == 0 { return 0.0; }

    let n = mol.num_atoms();
    let mut degree_counts = [0u32; 10];
    for i in 0..n {
        let d = mol.get_degree(i as AtomIdx).min(9);
        degree_counts[d] += 1;
    }
    let conn_complexity = shannon_entropy(&degree_counts) * mol.num_bonds() as f64;

    let mut element_counts = [0u32; 120];
    for atom in &mol.atoms {
        if atom.atomic_number > 0 {
            element_counts[atom.atomic_number as usize] += 1;
        }
    }
    let elem_complexity = shannon_entropy(&element_counts) * n as f64;

    let mut bond_type_counts = [0u32; 6];
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

    (conn_complexity + elem_complexity + bond_complexity) as f32
}

impl Descriptor for BertzCT {
    fn name(&self) -> &str { "BertzCT" }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_bertz_ct(mol))
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
///
/// Simplified: uses bond-type × endpoint-degree as class key.
pub struct Ipc;

fn compute_ipc(mol: &Molecule) -> f32 {
    let m = mol.num_bonds();
    if m == 0 { return 0.0; }

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

    (total_info - class_info) as f32
}

impl Descriptor for Ipc {
    fn name(&self) -> &str { "Ipc" }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_ipc(mol))
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

fn compute_fragment_complexity(mol: &Molecule) -> f32 {
    let mut element_counts = [0u32; 120];
    for atom in &mol.atoms {
        if atom.atomic_number > 0 {
            element_counts[atom.atomic_number as usize] += 1;
        }
    }
    shannon_entropy(&element_counts) as f32
}

impl Descriptor for FragmentComplexity {
    fn name(&self) -> &str { "FragmentComplexity" }

    fn calculate(&self, mol: &mut Molecule) -> DescriptorValue {
        DescriptorValue::Float(compute_fragment_complexity(mol))
    }
}

// ---------------------------------------------------------------------------
// Thread safety
// ---------------------------------------------------------------------------
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<FractionCSP3>;
    let _ = assert_send_sync::<NumAmideBonds>;
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

    fn calc_float(desc: &dyn Descriptor, smiles: &str) -> f32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        desc.calculate(&mut mol).as_float()
    }

    fn calc_int(desc: &dyn Descriptor, smiles: &str) -> u32 {
        let mut mol = parse_smiles(smiles).expect("valid SMILES");
        desc.calculate(&mut mol).as_int()
    }

    // -- FractionCSP3 --

    #[test]
    fn test_fsp3_ethane() {
        let v = calc_float(&FractionCSP3, "CC");
        assert!((v - 1.0).abs() < 0.01, "all sp3: {}", v);
    }

    #[test]
    fn test_fsp3_ethene() {
        let v = calc_float(&FractionCSP3, "C=C");
        assert!(v.abs() < 0.01, "all sp2: {}", v);
    }

    #[test]
    fn test_fsp3_propanol() {
        let v = calc_float(&FractionCSP3, "CCCO");
        assert!((v - 1.0).abs() < 0.01, "all sp3: {}", v);
    }

    #[test]
    fn test_fsp3_benzene() {
        let v = calc_float(&FractionCSP3, "c1ccccc1");
        assert!(v.abs() < 0.01, "no sp3 carbons: {}", v);
    }

    #[test]
    fn test_fsp3_toluene() {
        let v = calc_float(&FractionCSP3, "Cc1ccccc1");
        assert!((v - 1.0 / 7.0).abs() < 0.02, "fsp3 = {}", v);
    }

    // -- RingTypeCounts (helper function) --

    #[test]
    fn test_ring_types_benzene() {
        let mol = parse_smiles("c1ccccc1").expect("valid");
        let v = compute_ring_type_counts(&mol);
        assert_eq!(v[0], 1, "1 aromatic");
        assert_eq!(v[1], 0, "0 aliphatic");
        assert_eq!(v[2], 0, "0 saturated");
    }

    #[test]
    fn test_ring_types_cyclohexane() {
        let mol = parse_smiles("C1CCCCC1").expect("valid");
        let v = compute_ring_type_counts(&mol);
        assert_eq!(v[0], 0, "0 aromatic");
        assert_eq!(v[1], 1, "1 aliphatic");
        assert_eq!(v[2], 1, "1 saturated");
    }

    #[test]
    fn test_ring_types_pyridine() {
        let mol = parse_smiles("c1ccncc1").expect("valid");
        let v = compute_ring_type_counts(&mol);
        assert_eq!(v[0], 1, "1 aromatic");
        assert_eq!(v[3], 1, "1 heteroaromatic");
        assert_eq!(v[4], 1, "1 heterocyclic");
    }

    #[test]
    fn test_ring_types_morpholine() {
        let mol = parse_smiles("C1COCCN1").expect("valid");
        let v = compute_ring_type_counts(&mol);
        assert_eq!(v[2], 1, "1 saturated");
        assert_eq!(v[4], 1, "1 heterocyclic");
    }

    // -- NumAmideBonds --

    #[test]
    fn test_amide_acetamide() {
        assert_eq!(calc_int(&NumAmideBonds, "CC(=O)N"), 1);
    }

    #[test]
    fn test_amide_ethane() {
        assert_eq!(calc_int(&NumAmideBonds, "CC"), 0);
    }

    #[test]
    fn test_amide_peptide() {
        let v = calc_int(&NumAmideBonds, "NCC(=O)NCC(=O)O");
        assert!(v >= 1, "should have amide bonds: {}", v);
    }

    // -- AromaticAtomFraction --

    #[test]
    fn test_arom_frac_benzene() {
        let v = calc_float(&AromaticAtomFraction, "c1ccccc1");
        assert!((v - 1.0).abs() < 0.01, "all aromatic: {}", v);
    }

    #[test]
    fn test_arom_frac_hexane() {
        let v = calc_float(&AromaticAtomFraction, "CCCCCC");
        assert!(v.abs() < 0.01, "no aromatic: {}", v);
    }

    #[test]
    fn test_arom_frac_toluene() {
        let v = calc_float(&AromaticAtomFraction, "Cc1ccccc1");
        assert!((v - 6.0 / 7.0).abs() < 0.02, "frac = {}", v);
    }

    // -- HallKierAlpha --

    #[test]
    fn test_hka_all_carbon() {
        let v = calc_float(&HallKierAlpha, "CCCC");
        assert!(v.abs() < 0.01, "all-C alpha = {}", v);
    }

    #[test]
    fn test_hka_oxygen() {
        let v = calc_float(&HallKierAlpha, "CCO");
        assert!(v < 0.0, "O makes alpha negative: {}", v);
    }

    #[test]
    fn test_hka_bromine() {
        let v = calc_float(&HallKierAlpha, "CCBr");
        assert!(v > 0.0, "Br makes alpha positive: {}", v);
    }

    // -- BertzCT --

    #[test]
    fn test_bertz_methane() {
        let v = calc_float(&BertzCT, "C");
        assert!(v.abs() < 0.01, "methane CT = {}", v);
    }

    #[test]
    fn test_bertz_complex_gt_simple() {
        let simple = calc_float(&BertzCT, "CCCC");
        let complex = calc_float(&BertzCT, "c1ccc(O)c(N)c1");
        assert!(complex > simple, "complex {} > simple {}", complex, simple);
    }

    #[test]
    fn test_bertz_benzene() {
        let v = calc_float(&BertzCT, "c1ccccc1");
        assert!(v.abs() < 0.01, "benzene CT = {}", v);
    }

    // -- Ipc --

    #[test]
    fn test_ipc_ethane() {
        let v = calc_float(&Ipc, "CC");
        assert!(v.abs() < 0.01, "ethane Ipc = {}", v);
    }

    #[test]
    fn test_ipc_diverse() {
        let v = calc_float(&Ipc, "CC=O");
        assert!(v > 0.0, "should have info content: {}", v);
    }

    // -- FragmentComplexity --

    #[test]
    fn test_fc_all_carbon() {
        let v = calc_float(&FragmentComplexity, "CCCC");
        assert!(v.abs() < 0.01, "all-C FC = {}", v);
    }

    #[test]
    fn test_fc_diverse() {
        let v = calc_float(&FragmentComplexity, "CCNO");
        assert!(v > 0.0, "diverse FC = {}", v);
    }

    #[test]
    fn test_fc_more_diverse_higher() {
        let less = calc_float(&FragmentComplexity, "CCCC");
        let more = calc_float(&FragmentComplexity, "CCNO");
        assert!(more > less);
    }
}
