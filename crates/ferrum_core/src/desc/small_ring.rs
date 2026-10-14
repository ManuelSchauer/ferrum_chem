//! Small ring descriptor (CDK 2.13 `SmallRingDescriptor`).
//!
//! CDK enumerates chordless rings of size 3-7, groups ring atoms into ring
//! blocks and applies its own Kekulé-based aromaticity test (strict alternating
//! 6-rings, then a relaxed Hückel/fusion rule). The Kekulé structure and bond
//! order are rebuilt the way CDK's SMILES parser produces them
//! (`desc::cdk_daylight`).
//!
//! Key symbols: `struct SmallRingDescriptor`, `fn small_ring_counts`,
//! `fn chordless_rings`.

use crate::desc::cdk_daylight::CdkMol;
use crate::desc::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// CDK `SmallRingDescriptor` (CDK 2.13).
///
/// Returns IntU16Vec with 11 values:
/// - [0]: `nSmallRings` - chordless rings found by CDK's search (sizes 3-7 only)
/// - [1]: `nAromRings` - those rings whose bonds are all aromatic under the
///   descriptor's own Kekulé-based aromaticity test
/// - [2]: `nRingBlocks` - ring systems (spiro-joined rings form one block)
/// - [3]: `nAromBlocks` - connected components of aromatic bonds
/// - [4-10]: `nRings3`..`nRings9` - CDK ring counts by size (every chordless
///   ring of size 3-7; CDK never finds rings of size 8/9, so those are 0).
///
/// The aromaticity test is CDK's: a 6-ring alternating single/double in its
/// Kekulé form ("strict"), then any ring of size <= 7 whose atoms are all
/// pi atoms or lone-pair donors and that is either fused to an aromatic bond or
/// satisfies Hückel's rule ("relaxed"). Kekulé orders follow CDK's SMILES
/// parser (see `cdk_daylight`).
///
/// # Thread Safety
/// Thread-safe: stateless and uses only immutable references.
pub struct SmallRingDescriptor;

impl Descriptor for SmallRingDescriptor {
    fn name(&self) -> &str {
        "SmallRingDescriptor"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::IntU16Vec(small_ring_counts(mol).to_vec())
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<SmallRingDescriptor>;
};

/// Element block (s=1, p=2, d=3, f=4) by atomic number (CDK table).
const ELEMENT_BLOCKS: [u8; 113] = [
    0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2,
    1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3,
];

/// Valence electrons by atomic number (CDK table).
const ELEMENT_VALENCE: [i32; 113] = [
    0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6, 7,
    8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6, 7, 8, 1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 4, 5, 6, 7, 8, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12,
];

/// Largest ring size CDK's `SmallRingDescriptor` searches for.
const MAX_RING: usize = 7;

/// Computes the 11 CDK `SmallRingDescriptor` values exactly as CDK 2.13 does.
pub(crate) fn small_ring_counts(mol: &Molecule) -> [u16; 11] {
    let mut counts = [0u16; 11];
    let n = mol.num_atoms();
    if n == 0 || !mol.atoms.iter().any(|a| a.is_in_ring()) {
        return counts;
    }
    let m = CdkMol::new(mol);
    let nb = m.ends.len();

    // ring blocks: connected components over ring bonds (spiro atoms merge blocks)
    let mut block = vec![0u32; n];
    let mut n_blocks = 0u32;
    let mut stack = Vec::new();
    for s in 0..n {
        if block[s] != 0 || !m.adj[s].iter().any(|&(_, bi)| m.ring_bond[bi as usize]) {
            continue;
        }
        n_blocks += 1;
        block[s] = n_blocks;
        stack.push(s);
        while let Some(a) = stack.pop() {
            for &(w, bi) in &m.adj[a] {
                if m.ring_bond[bi as usize] && block[w as usize] == 0 {
                    block[w as usize] = n_blocks;
                    stack.push(w as usize);
                }
            }
        }
    }
    counts[2] = n_blocks as u16;

    // chordless cycles of size 3..=7 (CDK requires every ring atom to have exactly
    // two neighbours inside the ring)
    let rings = chordless_rings(&m, &block);
    for r in &rings {
        counts[4 + r.len() - 3] += 1;
    }
    counts[0] = rings.len() as u16;

    // conservative implicit hydrogen formula used by the descriptor
    let mut implicit_h = vec![0i32; n];
    for a in 0..n {
        let hy = match m.z[a] {
            6 => 4,
            7 => 3,
            8 => 2,
            16 => 2,
            15 => 3,
            _ => continue,
        };
        let mut ch = m.charge[a] as i32;
        if m.z[a] == 6 {
            ch = -ch.abs();
        }
        let mut hy = hy + ch;
        for &(_, bi) in &m.adj[a] {
            hy -= m.order[bi as usize] as i32;
        }
        implicit_h[a] = hy.max(0);
    }

    let mut bond_arom = vec![false; nb];
    if !rings.is_empty() {
        // strict aromaticity: 6-rings of pi atoms alternating 1/2
        let mut pi_atom = vec![false; n];
        for (bi, &(s, t)) in m.ends.iter().enumerate() {
            if m.order[bi] == 2 {
                pi_atom[s as usize] = true;
                pi_atom[t as usize] = true;
            }
        }
        let ring_bonds: Vec<Vec<usize>> = rings
            .iter()
            .map(|r| (0..r.len()).map(|i| m.bond(r[i], r[(i + 1) % r.len()]).unwrap()).collect())
            .collect();
        let mut maybe: Vec<usize> = (0..rings.len())
            .filter(|&k| {
                rings[k].len() == 6
                    && rings[k].iter().all(|&a| pi_atom[a])
                    && ring_bonds[k].iter().all(|&b| m.order[b] == 1 || m.order[b] == 2)
            })
            .collect();
        loop {
            let mut changed = false;
            maybe.retain(|&k| {
                let (mut phase1, mut phase2) = (true, true);
                for (i, &b) in ring_bonds[k].iter().enumerate() {
                    if bond_arom[b] {
                        continue;
                    }
                    phase1 = phase1 && m.order[b] as usize == 2 - (i & 1);
                    phase2 = phase2 && m.order[b] as usize == 1 + (i & 1);
                }
                if !phase1 && !phase2 {
                    return true;
                }
                for &b in &ring_bonds[k] {
                    bond_arom[b] = true;
                }
                changed = true;
                false
            });
            if !changed {
                break;
            }
        }

        // relaxed aromaticity
        let mut electrons = vec![0i32; n];
        for a in 0..n {
            let z = m.z[a] as usize;
            let (blk, val) = if z < ELEMENT_BLOCKS.len() { (ELEMENT_BLOCKS[z], ELEMENT_VALENCE[z]) } else { (0, 0) };
            electrons[a] = if blk == 2 { val } else { 0 } - m.charge[a] as i32 - implicit_h[a];
            for &(_, bi) in &m.adj[a] {
                electrons[a] -= m.order[bi as usize] as i32;
            }
        }
        let mut cand: Vec<usize> = (0..rings.len())
            .filter(|&k| {
                let mut already = true;
                for (i, &a) in rings[k].iter().enumerate() {
                    if !pi_atom[a] && electrons[a] < 2 {
                        return false;
                    }
                    let b = ring_bonds[k][i];
                    if m.order[b] != 1 && m.order[b] != 2 {
                        return false;
                    }
                    already = already && bond_arom[b];
                }
                !already
            })
            .collect();
        loop {
            let mut changed = false;
            let mut i = 0;
            while i < cand.len() {
                let k = cand[i];
                let r = &rings[k];
                let sz = r.len();
                let (mut pairs, mut maybe_e) = (0i32, 0i32);
                for j in 0..sz {
                    let b1 = ring_bonds[k][j];
                    let b2 = ring_bonds[k][(j + sz - 1) % sz];
                    if bond_arom[b1] {
                        maybe_e += 2;
                    } else if m.order[b1] == 2 {
                        pairs += 2;
                    } else if electrons[r[j]] >= 2 && m.order[b2] != 2 {
                        pairs += 2;
                    }
                }
                let mut arom = false;
                while maybe_e >= 0 {
                    if (pairs + maybe_e - 2) % 4 == 0 {
                        arom = true;
                        break;
                    }
                    maybe_e -= 2;
                }
                if arom {
                    for &b in &ring_bonds[k] {
                        bond_arom[b] = true;
                    }
                    cand.remove(i);
                    changed = true;
                } else {
                    i += 1;
                }
            }
            if !changed || cand.is_empty() {
                break;
            }
        }

        counts[1] = rings
            .iter()
            .enumerate()
            .filter(|&(k, _)| ring_bonds[k].iter().all(|&b| bond_arom[b]))
            .count() as u16;
    }

    // aromatic blocks: connected components of aromatic bonds
    let mut seen = vec![false; n];
    let mut n_arom_blocks = 0u16;
    for s in 0..n {
        if seen[s] || !m.adj[s].iter().any(|&(_, bi)| bond_arom[bi as usize]) {
            continue;
        }
        n_arom_blocks += 1;
        seen[s] = true;
        stack.push(s);
        while let Some(a) = stack.pop() {
            for &(w, bi) in &m.adj[a] {
                if bond_arom[bi as usize] && !seen[w as usize] {
                    seen[w as usize] = true;
                    stack.push(w as usize);
                }
            }
        }
    }
    counts[3] = n_arom_blocks;
    counts
}

/// All chordless cycles of size 3..=7 within ring blocks, each listed once
/// (starting at its lowest atom index).
fn chordless_rings(m: &CdkMol, block: &[u32]) -> Vec<Vec<usize>> {
    let n = m.n();
    let mut rings = Vec::new();
    let mut path: Vec<usize> = Vec::with_capacity(MAX_RING);
    let mut on_path = vec![false; n];
    for s in 0..n {
        if block[s] == 0 {
            continue;
        }
        path.clear();
        path.push(s);
        on_path[s] = true;
        extend_ring(m, block, s, &mut path, &mut on_path, &mut rings);
        on_path[s] = false;
    }
    rings
}

fn extend_ring(
    m: &CdkMol,
    block: &[u32],
    s: usize,
    path: &mut Vec<usize>,
    on_path: &mut [bool],
    rings: &mut Vec<Vec<usize>>,
) {
    let last = *path.last().unwrap();
    for &(w, bi) in &m.adj[last] {
        let w = w as usize;
        if w <= s || on_path[w] || !m.ring_bond[bi as usize] || block[w] != block[s] {
            continue;
        }
        // w may only touch `last` and (when closing the ring) `s` within the path
        let mut touches_start = false;
        let mut chord = false;
        for &(x, _) in &m.adj[w] {
            let x = x as usize;
            if x == last || !on_path[x] {
                continue;
            }
            if x == s {
                touches_start = true;
            } else {
                chord = true;
                break;
            }
        }
        if chord {
            continue;
        }
        path.push(w);
        if touches_start {
            // ring closes here (any longer ring would have the chord s-w)
            if path.len() >= 3 && path[1] < w {
                rings.push(path.clone());
            }
        } else if path.len() < MAX_RING {
            on_path[w] = true;
            extend_ring(m, block, s, path, on_path, rings);
            on_path[w] = false;
        }
        path.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn prepare_molecule(smiles: &str) -> Molecule {
        parse_smiles(smiles).unwrap()
    }

    fn calc_small_ring(smiles: &str) -> Vec<u16> {
        let mol = prepare_molecule(smiles);
        match SmallRingDescriptor.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        ) {
            DescriptorValue::IntU16Vec(v) => v,
            other => panic!("expected IntU16Vec, got {:?}", other),
        }
    }

    #[test]
    fn test_benzene() {
        let output = calc_small_ring("c1ccccc1");

        assert_eq!(output[0], 1, "Should have 1 small ring");
        assert_eq!(output[1], 1, "Should have 1 aromatic ring");
        assert_eq!(output[2], 1, "Should have 1 ring block");
        assert!(output[3] >= 1, "Should have at least 1 aromatic component");
        assert_eq!(
            output[4 + 3],
            1,
            "Should have 1 six-membered ring (index 7)"
        );
    }

    #[test]
    fn test_naphthalene() {
        let output = calc_small_ring("c1ccc2ccccc2c1");

        assert!(output[0] >= 2, "Should have at least 2 small rings");
        assert!(output[1] >= 2, "Should have at least 2 aromatic rings");
        assert_eq!(output[2], 1, "Should have 1 ring block (fused system)");
        assert_eq!(output[3], 1, "Should have 1 aromatic component");
    }

    #[test]
    fn test_cyclohexane() {
        let output = calc_small_ring("C1CCCCC1");

        assert_eq!(output[0], 1, "Should have 1 small ring");
        assert_eq!(output[1], 0, "Should have 0 aromatic rings");
        assert_eq!(output[2], 1, "Should have 1 ring block");
        assert_eq!(output[3], 0, "Should have 0 aromatic components");
        assert_eq!(output[4 + 3], 1, "Should have 1 six-membered ring");
    }

    #[test]
    fn test_cyclopropane() {
        let output = calc_small_ring("C1CC1");

        assert_eq!(output[0], 1, "Should have 1 small ring");
        assert_eq!(output[4], 1, "Should have 1 three-membered ring (index 4)");
    }

    #[test]
    /// CDK counts every chordless ring of size 3-7 (values verified against CDK 2.13).
    fn test_cubane_counts_all_chordless_rings() {
        let cdk = small_ring_counts(&prepare_molecule("C12C3C4C1C5C2C3C45"));
        assert_eq!(cdk[0], 10, "6 four-rings + 4 chordless six-rings");
        assert_eq!(cdk[5], 6);
        assert_eq!(cdk[7], 4);
        let output = calc_small_ring("C12C3C4C1C5C2C3C45");
        assert_eq!(output, cdk.to_vec());
    }

    #[test]
    /// CDK never finds rings larger than 7 atoms.
    fn test_no_rings_above_seven() {
        let cdk = small_ring_counts(&prepare_molecule("c1ccccccc1"));
        assert_eq!(cdk[0], 0);
        assert_eq!(cdk[2], 1, "the ring block is still counted");
        assert!(cdk[4..].iter().all(|&c| c == 0));
        assert_eq!(calc_small_ring("c1ccccccc1")[9], 0, "no nRings8 in CDK");
    }

    #[test]
    /// Kekulé-based relaxed aromaticity (azulene) and spiro ring blocks.
    fn test_cdk_aromaticity_and_blocks() {
        let output = calc_small_ring("C1=CC2=CC=CC=CC2=C1");
        assert_eq!(output[1], 2, "azulene: both rings aromatic");
        assert_eq!(output[3], 1);
        let output = calc_small_ring("C1CCC2(C1)CCCC2");
        assert_eq!(output[2], 1, "spiro rings form one ring block");
        let output = calc_small_ring("c1ccccc1-c1ccccc1");
        assert_eq!(output[2], 2);
        assert_eq!(output[3], 2);
    }

    #[test]
    /// Adamantane: 4 chordless six-rings (CDK 2.13).
    fn test_total_rings_uses_symmetrized_ring_count_for_adamantane() {
        let output = calc_small_ring("C1C2CC3CC1CC(C2)C3");
        assert_eq!(
            output[0], 4,
            "Adamantane has 4 chordless six-rings"
        );
    }

    #[test]
    fn test_no_rings() {
        let output = calc_small_ring("CCCC");

        for i in 0..11 {
            assert_eq!(
                output[i], 0,
                "All counts should be 0 for non-cyclic molecule"
            );
        }
    }
}
