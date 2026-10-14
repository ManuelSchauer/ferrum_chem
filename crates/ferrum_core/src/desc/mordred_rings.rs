//! Descriptor module: Mordred RingCount.
//!
//! Mordred (mordred-community 2.0.7) `RingCount.RingCount`: counts of the
//! rings of `Chem.GetSymmSSSR` (FerrumChem's ring info, the RDKit
//! symmetrized SSSR), or of fused ring systems, filtered by
//!
//! - size: any, exactly 3..=12 (4..=12 for fused systems) or 12 and more (`G12`),
//! - aromaticity: any, aromatic (`a`, every atom aromatic) or not (`A`),
//! - heteroatoms: any or at least one non-carbon atom (`H`).
//!
//! A fused ring system is a connected component (of at least two rings) of
//! the graph whose edges join rings sharing two or more atoms; its size is
//! the number of atoms of the union (Mordred `FusedRings`, via networkx).
//!
//! Columns follow Mordred's preset order: for fused in (no, yes), aromatic in
//! (any, a, A), hetero in (any, H): total, then each size, then `G12`. The
//! very first one, `nRing`, is RDKit's `RingCount` (same ring set) and is not
//! repeated here (see `mordred_names::MORDRED_ALIASES`).

use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Size slots per (aromatic, hetero) block: total, 3..=12, G12 (plain rings).
const PLAIN_SLOTS: usize = 12;
/// Size slots per block for fused systems: total, 4..=12, G12.
const FUSED_SLOTS: usize = 11;
/// 6 (aromatic, hetero) blocks per family.
const BLOCKS: usize = 6;

/// Number of columns (Mordred's 138 minus `nRing`).
pub const MORDRED_RING_COUNT_LEN: usize = BLOCKS * (PLAIN_SLOTS + FUSED_SLOTS) - 1;

/// Column names in output order (`n3Ring`, ..., `nG12FAHRing`).
pub fn mordred_ring_count_names() -> Vec<String> {
    let mut names = Vec::with_capacity(MORDRED_RING_COUNT_LEN + 1);
    for fused in [false, true] {
        for arom in ["", "a", "A"] {
            for hetero in ["", "H"] {
                let f = if fused { "F" } else { "" };
                names.push(format!("n{f}{arom}{hetero}Ring"));
                for order in (if fused { 4 } else { 3 })..=12 {
                    names.push(format!("n{order}{f}{arom}{hetero}Ring"));
                }
                names.push(format!("nG12{f}{arom}{hetero}Ring"));
            }
        }
    }
    names.remove(0); // nRing (alias of RingCount)
    names
}

/// Adds one ring (or fused system) of `size` atoms to the counts of one family.
fn add(counts: &mut [f64], fused: bool, size: usize, aromatic: bool, hetero: bool) {
    let (slots, min_order) = if fused { (FUSED_SLOTS, 4) } else { (PLAIN_SLOTS, 3) };
    for (ai, arom_ok) in [true, aromatic, !aromatic].into_iter().enumerate() {
        if !arom_ok {
            continue;
        }
        for (hi, het_ok) in [true, hetero].into_iter().enumerate() {
            if !het_ok {
                continue;
            }
            let block = &mut counts[(ai * 2 + hi) * slots..(ai * 2 + hi + 1) * slots];
            block[0] += 1.0;
            if (min_order..=12).contains(&size) {
                block[1 + size - min_order] += 1.0;
            }
            if size >= 12 {
                block[slots - 1] += 1.0;
            }
        }
    }
}

/// Mordred ring counts; see the module docs.
///
/// # Thread Safety
/// Stateless and thread-safe (Send + Sync).
pub struct MordredRingCount;

impl Descriptor for MordredRingCount {
    fn name(&self) -> &str {
        "MordredRingCount"
    }

    fn calculate(&self, mol: &Molecule, _ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
        let mut counts = vec![0.0f64; BLOCKS * (PLAIN_SLOTS + FUSED_SLOTS)];
        let rings: &[_] = match mol.ring_info() {
            Some(info) => &info.atom_rings,
            None => &[],
        };
        let is_arom = |a: &u16| mol.atoms[*a as usize].is_aromatic();
        let is_het = |a: &u16| mol.atoms[*a as usize].atomic_number != 6;
        let (plain, fused) = counts.split_at_mut(BLOCKS * PLAIN_SLOTS);
        for r in rings {
            add(plain, false, r.len(), r.iter().all(is_arom), r.iter().any(is_het));
        }

        let nr = rings.len();
        if nr >= 2 {
            // rings sharing >= 2 atoms are joined (union-find)
            let n = mol.num_atoms();
            let mut parent: Vec<usize> = (0..nr).collect();
            fn find(p: &mut [usize], mut x: usize) -> usize {
                while p[x] != x {
                    p[x] = p[p[x]];
                    x = p[x];
                }
                x
            }
            let mut joined = vec![false; nr];
            let mut mark = vec![usize::MAX; n];
            for i in 0..nr {
                for &a in &rings[i] {
                    mark[a as usize] = i;
                }
                for j in i + 1..nr {
                    let shared = rings[j].iter().filter(|&&a| mark[a as usize] == i).take(2).count();
                    if shared >= 2 {
                        joined[i] = true;
                        joined[j] = true;
                        let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                        parent[ri] = rj;
                    }
                }
            }
            // union atom set of every component, in order of the first ring
            mark.fill(usize::MAX);
            for i in 0..nr {
                if !joined[i] || find(&mut parent, i) != i {
                    continue;
                }
                let (mut size, mut aromatic, mut hetero) = (0usize, true, false);
                for k in 0..nr {
                    if !joined[k] || find(&mut parent, k) != i {
                        continue;
                    }
                    for a in &rings[k] {
                        if mark[*a as usize] == i {
                            continue;
                        }
                        mark[*a as usize] = i;
                        size += 1;
                        aromatic &= is_arom(a);
                        hetero |= is_het(a);
                    }
                }
                add(fused, true, size, aromatic, hetero);
            }
        }
        counts.remove(0); // nRing
        DescriptorValue::FloatVec(counts)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<MordredRingCount>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn calc(smiles: &str) -> std::collections::HashMap<String, f64> {
        let mol = parse_smiles(smiles).unwrap();
        let v = MordredRingCount
            .calculate(&mol, &mut crate::desc::context::CalculationContext::new())
            .as_float_vec()
            .to_vec();
        mordred_ring_count_names().into_iter().zip(v).collect()
    }

    #[test]
    fn names_and_length() {
        let names = mordred_ring_count_names();
        assert_eq!(names.len(), MORDRED_RING_COUNT_LEN);
        assert_eq!(names[0], "n3Ring");
        assert_eq!(names.last().unwrap(), "nG12FAHRing");
    }

    #[test]
    fn indole_and_biphenyl() {
        let c = calc("c1ccc2[nH]ccc2c1");
        assert_eq!(c["n6aRing"], 1.0);
        assert_eq!(c["n5aHRing"], 1.0);
        assert_eq!(c["nFRing"], 1.0);
        assert_eq!(c["n9FaHRing"], 1.0);
        let c = calc("c1ccccc1-c1ccccc1");
        assert_eq!(c["n6aRing"], 2.0);
        assert_eq!(c["nFRing"], 0.0);
    }
}
