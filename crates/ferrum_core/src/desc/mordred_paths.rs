//! Mordred (mordred-community 2.0.7) `PathCount` descriptors on the
//! hydrogen-suppressed graph (`Chem.RemoveHs`; hydrogens RDKit keeps stay
//! graph atoms).
//!
//! Output (21 values, Mordred's order): `MPC2..MPC10`, `TMPC10`, `piPC1..piPC10`,
//! `TpiPC10`. `MPCk` is the number of simple paths of `k` bonds that RDKit's
//! `FindAllPathsOfLengthN` finds (bonds to hydrogen atoms are ignored,
//! `useHs = false`; Mordred drops the ring closures that revisit an atom),
//! `piPCk = ln(1 + Σ_paths Π bond orders)` (aromatic = 1.5). The totals
//! start from order 0 = number of atoms: `TMPC10 = A + Σ_{k=1..10} MPCk`,
//! `TpiPC10 = ln(1 + A + Σ_{k=1..10} Σ_paths Π bond orders)`.

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use crate::types::BondType;

/// Longest path (bonds) of Mordred's PathCount preset.
const MAX_PATH: usize = 10;

/// Number of PathCount columns.
pub const MORDRED_PATH_COUNT_LEN: usize = 21;

/// PathCount column names in Mordred's preset order.
pub fn mordred_path_count_names() -> Vec<String> {
    let mut v: Vec<String> = (2..=MAX_PATH).map(|k| format!("MPC{k}")).collect();
    v.push(format!("TMPC{MAX_PATH}"));
    v.extend((1..=MAX_PATH).map(|k| format!("piPC{k}")));
    v.push(format!("TpiPC{MAX_PATH}"));
    v
}

/// RDKit `Bond::getBondTypeAsDouble`.
#[inline]
fn bond_order(t: BondType) -> f64 {
    match t {
        BondType::Single | BondType::Dative => 1.0,
        BondType::Double => 2.0,
        BondType::Triple => 3.0,
        BondType::Aromatic => 1.5,
        BondType::Unknown => 0.0,
    }
}

/// Mordred PathCount values in [`mordred_path_count_names`] order.
pub fn mordred_path_count(mol: &Molecule) -> Vec<f64> {
    let n = mol.num_atoms();
    // adjacency over bonds between non-hydrogen atoms: (neighbour, bond order)
    let is_h: Vec<bool> = mol.atoms.iter().map(|a| a.atomic_number == 1).collect();
    let mut start = Vec::with_capacity(n + 1);
    let mut adj: Vec<(u32, f64)> = Vec::with_capacity(2 * mol.num_bonds());
    for a in 0..n {
        start.push(adj.len());
        if is_h[a] {
            continue;
        }
        for &bi in &mol.adjacency[a] {
            let b = &mol.bonds[bi as usize];
            let o = if b.source as usize == a { b.target } else { b.source } as usize;
            if !is_h[o] {
                adj.push((o as u32, bond_order(b.bond_type)));
            }
        }
    }
    start.push(adj.len());

    struct Walk<'a> {
        start: &'a [usize],
        adj: &'a [(u32, f64)],
        on_path: Vec<bool>,
        root: u32,
        count: [u64; MAX_PATH + 1],
        pi: [f64; MAX_PATH + 1],
    }
    fn dfs(w: &mut Walk, atom: usize, depth: usize, weight: f64) {
        for k in w.start[atom]..w.start[atom + 1] {
            let (nb, order) = w.adj[k];
            if w.on_path[nb as usize] {
                continue;
            }
            let wt = weight * order;
            // every simple path is walked from both ends; count it once
            if nb > w.root {
                w.count[depth + 1] += 1;
                w.pi[depth + 1] += wt;
            }
            if depth + 1 < MAX_PATH {
                w.on_path[nb as usize] = true;
                dfs(w, nb as usize, depth + 1, wt);
                w.on_path[nb as usize] = false;
            }
        }
    }
    let mut w = Walk {
        start: &start,
        adj: &adj,
        on_path: vec![false; n],
        root: 0,
        count: [0; MAX_PATH + 1],
        pi: [0.0; MAX_PATH + 1],
    };
    for s in 0..n {
        w.root = s as u32;
        w.on_path[s] = true;
        dfs(&mut w, s, 0, 1.0);
        w.on_path[s] = false;
    }

    let mut out = Vec::with_capacity(MORDRED_PATH_COUNT_LEN);
    out.extend((2..=MAX_PATH).map(|k| w.count[k] as f64));
    out.push((n as u64 + w.count[1..].iter().sum::<u64>()) as f64);
    out.extend((1..=MAX_PATH).map(|k| (w.pi[k] + 1.0).ln()));
    let mut total = n as f64;
    for k in 1..=MAX_PATH {
        total += w.pi[k];
    }
    out.push((total + 1.0).ln());
    out
}

/// Mordred PathCount descriptor family (21 values, see the module docs).
pub struct MordredPathCount;

impl Descriptor for MordredPathCount {
    fn name(&self) -> &str {
        "MordredPathCount"
    }

    fn calculate(&self, mol: &Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        DescriptorValue::FloatVec(mordred_path_count(mol))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn path_counts_of_cyclobutane_and_benzene() {
        let v = mordred_path_count(&parse_smiles("C1CCC1").unwrap());
        // 4 paths of 2 bonds, 4 of 3 bonds, none longer (ring closures dropped)
        assert_eq!(&v[..3], &[4.0, 4.0, 0.0]);
        assert_eq!(v[9], 4.0 + 4.0 + 4.0 + 4.0);
        let v = mordred_path_count(&parse_smiles("c1ccccc1").unwrap());
        assert!((v[10] - (6.0f64 * 1.5 + 1.0).ln()).abs() < 1e-12);
    }
}
