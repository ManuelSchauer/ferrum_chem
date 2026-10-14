//! Mordred (mordred-community 2.0.7) atomic properties
//! (`mordred/_atomic_property.py`), as used by the Mordred `Autocorrelation`
//! and `BCUT` modules.
//!
//! The twelve properties, in Mordred's order (`get_properties(charge=True,
//! valence=True)`):
//!
//! | short | property | value |
//! |-------|----------|-------|
//! | `c`   | Gasteiger charge | `_GasteigerCharge + _GasteigerHCharge` (RDKit `ComputeGasteigerCharges` on the molecule Mordred works on) |
//! | `dv`  | valence electrons | `(Zv - h) / (Z - Zv - 1)`, 0 for hydrogen (`Zv` outer electrons and `Z` atomic number, both minus the formal charge; `h` attached hydrogens) |
//! | `d`   | sigma electrons | number of non-hydrogen neighbours |
//! | `s`   | intrinsic state | `((2 / period)² · dv + 1) / d`, NaN when `d = 0` |
//! | `Z`   | atomic number | |
//! | `m`, `v`, `se`, `pe`, `are`, `p`, `i` | tables | mass, vdW volume, Sanderson / Pauling / Allred-Rochow electronegativity, polarizability, ionization potential (`mordred_data.rs`) |
//!
//! Mordred reports every descriptor of a property as missing when any atom
//! has no value (NaN) for it.
//!
//! Mordred evaluates `Autocorrelation` on the molecule with every hydrogen
//! explicit (`Chem.AddHs`) and `BCUT` on the hydrogen-suppressed one
//! (`Chem.RemoveHs`, i.e. the graph as parsed). The added hydrogens are not
//! materialised here: [`MordredAtomProps`] holds the property values of the
//! graph atoms and, per graph atom, the value shared by the hydrogens added to
//! it (they are equivalent, including their Gasteiger charge).

use super::mordred_data::{
    MORDRED_ALLRED_ROCOW_EN, MORDRED_IONIZATION_POTENTIAL, MORDRED_MASS, MORDRED_PAULING_EN, MORDRED_PERIOD,
    MORDRED_POLARIZABILITY94, MORDRED_SANDERSON_EN, MORDRED_VDW_VOLUME, RDKIT_N_OUTER_ELECS,
};
use crate::mol::molecule::Molecule;

/// Number of Mordred atomic properties.
pub(crate) const N_PROPS: usize = 12;

/// Mordred short names of the atomic properties, in Mordred's order.
pub(crate) const PROP_NAMES: [&str; N_PROPS] = ["c", "dv", "d", "s", "Z", "m", "v", "se", "pe", "are", "p", "i"];

/// Index of the Gasteiger charge `c` (the only property not in Mordred's `ATS` / `AATS`).
pub(crate) const PROP_C: usize = 0;

/// Element tables of `Z, m, v, se, pe, are, p, i` (property indices 4..12).
const TABLES: [&[f64; 119]; 7] = [
    &MORDRED_MASS,
    &MORDRED_VDW_VOLUME,
    &MORDRED_SANDERSON_EN,
    &MORDRED_PAULING_EN,
    &MORDRED_ALLRED_ROCOW_EN,
    &MORDRED_POLARIZABILITY94,
    &MORDRED_IONIZATION_POTENTIAL,
];

/// Mordred atomic property values of a molecule.
pub(crate) struct MordredAtomProps {
    /// `w[i][p]`: property `p` of graph atom `i`.
    pub w: Vec<[f64; N_PROPS]>,
    /// `u[i][p]`: property `p` of each hydrogen added to atom `i`
    /// (explicit-hydrogen mode only; unused otherwise).
    pub u: Vec<[f64; N_PROPS]>,
    /// Hydrogens added to atom `i` (`total_h`; 0 in hydrogen-suppressed mode).
    pub h: Vec<f64>,
    /// `false` where Mordred reports the property as missing (an atom has no value).
    pub valid: [bool; N_PROPS],
}

/// Property values that depend only on the element: `[Z, m, v, se, pe, are, p, i]`.
#[inline]
fn element_props(z: usize, out: &mut [f64]) {
    out[0] = z as f64;
    for (o, table) in out[1..].iter_mut().zip(TABLES) {
        *o = table.get(z).copied().unwrap_or(f64::NAN);
    }
}

/// Mordred `get_intrinsic_state` from `d` and `dv`.
#[inline]
fn intrinsic_state(z: usize, d: f64, dv: f64) -> f64 {
    if d == 0.0 {
        return f64::NAN;
    }
    let period = if z >= 1 { MORDRED_PERIOD.get(z).copied().unwrap_or(f64::NAN) } else { f64::NAN };
    let f = 2.0 / period;
    (f * f * dv + 1.0) / d
}

/// Mordred atomic properties of `mol` with every hydrogen explicit
/// (`explicit_h`, Mordred `explicit_hydrogens = True`) or of the molecule as
/// parsed (hydrogen-suppressed).
pub(crate) fn mordred_atom_props(mol: &Molecule, explicit_h: bool) -> MordredAtomProps {
    let n = mol.num_atoms();
    let (charges, h_charges) = crate::algo::electronic::gasteiger_charges_with_hydrogens(mol, explicit_h);
    let mut w = Vec::with_capacity(n);
    let mut u = Vec::with_capacity(if explicit_h { n } else { 0 });
    let mut h = Vec::with_capacity(n);
    let mut nan = [false; N_PROPS];
    let mut h_props = [0.0f64; N_PROPS];
    element_props(1, &mut h_props[4..]);
    for (i, atom) in mol.atoms.iter().enumerate() {
        let z = atom.atomic_number as usize;
        let mut p = [0.0f64; N_PROPS];
        // sigma electrons: non-hydrogen neighbours (the added hydrogens never count)
        let d = mol.adjacency[i]
            .iter()
            .filter(|&&b| {
                let bond = &mol.bonds[b as usize];
                let other = if bond.source as usize == i { bond.target } else { bond.source };
                mol.atoms[other as usize].atomic_number != 1
            })
            .count() as f64;
        p[2] = d;
        p[1] = if z == 1 {
            0.0
        } else {
            let fc = atom.formal_charge as i32;
            let zv = RDKIT_N_OUTER_ELECS.get(z).copied().unwrap_or(0) as i32 - fc;
            let zz = z as i32 - fc;
            let hs = mol.total_hs_with_neighbors(i) as i32;
            (zv - hs) as f64 / (zz - zv - 1) as f64
        };
        p[3] = intrinsic_state(z, d, p[1]);
        element_props(z, &mut p[4..]);
        p[PROP_C] = if explicit_h { charges[i] } else { charges[i] + h_charges[i] };
        for (k, v) in p.iter().enumerate() {
            nan[k] |= v.is_nan();
        }
        w.push(p);

        let n_h = if explicit_h { atom.total_h() } else { 0 };
        h.push(n_h as f64);
        if explicit_h && n_h == 0 {
            // finite placeholder: `u` is always used multiplied by `h = 0`
            u.push([0.0; N_PROPS]);
        } else if explicit_h {
            let mut q = h_props;
            q[PROP_C] = h_charges[i];
            // an added hydrogen's only neighbour is atom `i`; dv(H) = 0
            let dh = if z == 1 { 0.0 } else { 1.0 };
            q[2] = dh;
            q[3] = intrinsic_state(1, dh, 0.0);
            for (k, v) in q.iter().enumerate() {
                nan[k] |= v.is_nan();
            }
            u.push(q);
        }
    }
    MordredAtomProps { w, u, h, valid: nan.map(|x| !x) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn ethanol_properties() {
        let mol = parse_smiles("CCO").unwrap();
        let p = mordred_atom_props(&mol, true);
        assert_eq!(p.h, vec![3.0, 2.0, 1.0]);
        // C (CH3): dv = (4 - 3) / (6 - 4 - 1) = 1, d = 1, s = ((2/2)^2 * 1 + 1) / 1 = 2
        assert_eq!(&p.w[0][1..4], &[1.0, 1.0, 2.0]);
        // O (OH): dv = (6 - 1) / (8 - 6 - 1) = 5, d = 1, s = 6
        assert_eq!(&p.w[2][1..4], &[5.0, 1.0, 6.0]);
        // hydrogens: dv = 0, d = 1, s = 1, Z = 1
        assert_eq!(&p.u[0][1..5], &[0.0, 1.0, 1.0, 1.0]);
        assert!(p.valid.iter().all(|&v| v));
    }

    #[test]
    fn isolated_atom_has_no_intrinsic_state() {
        let mol = parse_smiles("C").unwrap();
        let p = mordred_atom_props(&mol, true);
        assert!(!p.valid[3]);
        assert!(p.valid[1] && p.valid[2]);
    }
}
