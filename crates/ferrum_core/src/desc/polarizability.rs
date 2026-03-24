use super::Descriptor;
use crate::desc::DescriptorValue;
use crate::mol::molecule::Molecule;

/// Atomic polarizabilities (ų) indexed by atomic number (0..54).
/// Data from Miller & Savchik / CRC Handbook.
const POLARIZABILITY: [f32; 54] = [
    0.0,   // 0  placeholder
    0.387, // 1  H
    0.205, // 2  He
    0.0,   // 3  Li
    0.0,   // 4  Be
    0.0,   // 5  B
    1.76,  // 6  C
    1.10,  // 7  N
    0.802, // 8  O
    0.557, // 9  F
    0.396, // 10 Ne
    0.0,   // 11 Na
    0.0,   // 12 Mg
    0.0,   // 13 Al
    5.38,  // 14 Si
    3.63,  // 15 P
    2.90,  // 16 S
    2.18,  // 17 Cl
    1.64,  // 18 Ar
    0.0,   // 19 K
    0.0,   // 20 Ca
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // 21-29
    0.0,  // 30 Zn
    0.0,  // 31 Ga
    0.0,  // 32 Ge
    0.0,  // 33 As
    0.0,  // 34 Se
    3.05, // 35 Br
    2.48, // 36 Kr
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // 37-45
    0.0,  // 46 Pd
    0.0,  // 47 Ag
    0.0,  // 48 Cd
    0.0,  // 49 In
    0.0,  // 50 Sn
    0.0,  // 51 Sb
    0.0,  // 52 Te
    5.35, // 53 I
];

fn atom_pol(atomic_number: u8) -> f32 {
    let idx = atomic_number as usize;
    if idx < POLARIZABILITY.len() {
        POLARIZABILITY[idx]
    } else {
        0.0
    }
}

/// APol — sum of atomic polarizabilities (heavy atoms + implicit H).
pub struct APol;

impl Descriptor for APol {
    fn name(&self) -> &str {
        "APol"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let h_pol = POLARIZABILITY[1];
        let mut sum = 0.0f32;
        for atom in &mol.atoms {
            sum += atom_pol(atom.atomic_number);
            sum += atom.total_h() as f32 * h_pol;
        }
        DescriptorValue::Float(sum)
    }
}

/// BPol — sum of absolute differences of polarizabilities over bonds.
pub struct BPol;

impl Descriptor for BPol {
    fn name(&self) -> &str {
        "BPol"
    }

    fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let mut sum = 0.0f32;
        for bond in &mol.bonds {
            let p_src = atom_pol(mol.atoms[bond.source as usize].atomic_number);
            let p_tgt = atom_pol(mol.atoms[bond.target as usize].atomic_number);
            sum += (p_src - p_tgt).abs();
        }
        DescriptorValue::Float(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn parse(smiles: &str) -> Molecule {
        parse_smiles(smiles).expect("valid SMILES")
    }

    fn calc(desc: &dyn Descriptor, smiles: &str) -> f32 {
        let mut mol = parse(smiles);
        match desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()) {
            DescriptorValue::Float(v) => v,
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_apol_methane() {
        // C: 1.76 + 4*0.387 = 3.308
        let v = calc(&APol, "[CH4]");
        assert!((v - 3.308).abs() < 0.01, "APol methane = {v}");
    }

    #[test]
    fn test_apol_ethanol() {
        // CCO: 2*1.76 + 0.802 + (3+2+1)*0.387 = 4.322 + 2.322 = 6.644
        let v = calc(&APol, "CCO");
        let expected = 2.0 * 1.76 + 0.802 + 6.0 * 0.387;
        assert!(
            (v - expected).abs() < 0.01,
            "APol ethanol = {v}, expected {expected}"
        );
    }

    #[test]
    fn test_bpol_ethanol() {
        // Bonds: C-C (|1.76-1.76|=0) + C-O (|1.76-0.802|=0.958)
        let v = calc(&BPol, "CCO");
        assert!((v - 0.958).abs() < 0.01, "BPol ethanol = {v}");
    }

    #[test]
    fn test_apol_benzene() {
        // c1ccccc1: 6*1.76 + 6*0.387 = 10.56 + 2.322 = 12.882
        let v = calc(&APol, "c1ccccc1");
        let expected = 6.0 * 1.76 + 6.0 * 0.387;
        assert!(
            (v - expected).abs() < 0.01,
            "APol benzene = {v}, expected {expected}"
        );
    }

    #[test]
    fn test_bpol_symmetric() {
        // Benzene: all C-C bonds → BPol = 0
        let v = calc(&BPol, "c1ccccc1");
        assert!(v.abs() < 1e-6, "BPol benzene = {v}");
    }

    #[test]
    fn test_apol_fluoromethane() {
        // CF: 1.76 + 0.557 + 3*0.387 = 3.478
        let v = calc(&APol, "CF");
        let expected = 1.76 + 0.557 + 3.0 * 0.387;
        assert!(
            (v - expected).abs() < 0.01,
            "APol CF = {v}, expected {expected}"
        );
    }

    #[test]
    fn test_empty_molecule() {
        let mut mol = Molecule::default();
        assert!(matches!(APol.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()), DescriptorValue::Float(v) if v == 0.0));
        assert!(matches!(BPol.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new()), DescriptorValue::Float(v) if v == 0.0));
    }
}
