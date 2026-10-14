//! Descriptor module: Mordred Constitutional.
//!
//! Mordred (mordred-community 2.0.7) `Constitutional.ConstitutionalSum` /
//! `ConstitutionalMean`: the sum (`S<p>`) and mean (`M<p>`) over all atoms of
//! the explicit-H molecule of an atomic property divided by the property of
//! carbon,
//!
//! ```text
//! S_p = Σ_i p_i / p_C        M_p = S_p / A
//! ```
//!
//! for the properties Z (atomic number), m (mass), v (vdW volume),
//! se (Sanderson EN), pe (Pauling EN), are (Allred-Rochow EN),
//! p (polarizability) and i (ionization potential). An atom without a value
//! makes Mordred report the descriptor as missing (NaN here).

use super::mordred_data::{
    MORDRED_ALLRED_ROCOW_EN, MORDRED_IONIZATION_POTENTIAL, MORDRED_MASS, MORDRED_PAULING_EN,
    MORDRED_POLARIZABILITY94, MORDRED_SANDERSON_EN, MORDRED_VDW_VOLUME,
};
use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Property suffixes in Mordred's order (`get_properties()`).
pub const MORDRED_CONSTITUTIONAL_PROPS: [&str; 8] = ["Z", "m", "v", "se", "pe", "are", "p", "i"];

/// Number of columns: `S<p>` for every property, then `M<p>`.
pub const MORDRED_CONSTITUTIONAL_LEN: usize = 2 * MORDRED_CONSTITUTIONAL_PROPS.len();

/// Column names (`SZ`, `Sm`, ..., `Si`, `MZ`, ..., `Mi`).
pub fn mordred_constitutional_names() -> Vec<String> {
    let props = MORDRED_CONSTITUTIONAL_PROPS.iter();
    props
        .clone()
        .map(|p| format!("S{p}"))
        .chain(props.map(|p| format!("M{p}")))
        .collect()
}

/// Atomic property `k` (in [`MORDRED_CONSTITUTIONAL_PROPS`] order) of element `z`.
#[inline]
fn prop(k: usize, z: usize) -> f64 {
    match k {
        0 => z as f64,
        1 => MORDRED_MASS[z],
        2 => MORDRED_VDW_VOLUME[z],
        3 => MORDRED_SANDERSON_EN[z],
        4 => MORDRED_PAULING_EN[z],
        5 => MORDRED_ALLRED_ROCOW_EN[z],
        6 => MORDRED_POLARIZABILITY94[z],
        _ => MORDRED_IONIZATION_POTENTIAL[z],
    }
}

/// Mordred constitutional sums and means; see the module docs.
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct MordredConstitutional;

impl Descriptor for MordredConstitutional {
    fn name(&self) -> &str {
        "MordredConstitutional"
    }

    fn calculate(&self, mol: &Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        const NP: usize = MORDRED_CONSTITUTIONAL_PROPS.len();
        let mut out = vec![f64::NAN; MORDRED_CONSTITUTIONAL_LEN];
        let n_h: usize = mol.atoms.iter().map(|a| a.total_h() as usize).sum();
        let n_atoms = mol.num_atoms() + n_h;
        if n_atoms == 0 {
            return DescriptorValue::FloatVec(out);
        }
        for k in 0..NP {
            let carbon = prop(k, 6);
            // heavy atoms first, then the hydrogens Chem.AddHs appends
            let mut s = 0.0f64;
            for a in &mol.atoms {
                s += prop(k, a.atomic_number as usize) / carbon;
            }
            let h = prop(k, 1) / carbon;
            for _ in 0..n_h {
                s += h;
            }
            out[k] = s;
            out[NP + k] = s / n_atoms as f64;
        }
        DescriptorValue::FloatVec(out)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MordredConstitutional>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn methane_atomic_number_sum() {
        let mol = parse_smiles("C").unwrap();
        let v = MordredConstitutional
            .calculate(&mol, &mut crate::desc::context::CalculationContext::new())
            .as_float_vec()
            .to_vec();
        assert_eq!(v.len(), MORDRED_CONSTITUTIONAL_LEN);
        // SZ = (6 + 4 * 1) / 6, MZ = SZ / 5
        assert!((v[0] - 10.0 / 6.0).abs() < 1e-12);
        assert!((v[8] - 10.0 / 30.0).abs() < 1e-12);
        assert_eq!(mordred_constitutional_names()[8], "MZ");
    }
}
