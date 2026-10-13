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
//!
//! Every path is counted from both ends (and the sums halved), but walked
//! only from the atoms of the core left by repeatedly stripping degree-1
//! atoms: the paths of a stripped (pendant) atom `x` hanging on `u` are the
//! paths down its subtree plus the step to `u` followed by the paths of `u`
//! that do not start towards `x`. The bond-order products are multiples of
//! `2^-10` below `3^10`, so all sums (and differences) are exact and
//! independent of the order.

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

/// Directed paths per length (index = bonds): number and `Σ Π bond orders`.
#[derive(Clone, Copy)]
struct Paths {
    count: [u64; MAX_PATH + 1],
    pi: [f64; MAX_PATH + 1],
}

impl Paths {
    const ZERO: Paths = Paths { count: [0; MAX_PATH + 1], pi: [0.0; MAX_PATH + 1] };

    /// The paths starting with a bond of order `order` followed by `self`
    /// (lengths `1..MAX_PATH`).
    fn behind(&self, order: f64) -> Paths {
        let mut p = Paths::ZERO;
        p.count[1] = 1;
        p.pi[1] = order;
        for k in 1..MAX_PATH {
            p.count[k + 1] = self.count[k];
            p.pi[k + 1] = order * self.pi[k];
        }
        p
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
    let slots = |a: usize| start[a]..start[a + 1];
    let degree = |a: usize| start[a + 1] - start[a];

    // pendant trees: strip degree-1 atoms (children before parents), with
    // the paths from each pendant atom down its subtree
    let mut parent: Vec<(u32, f64)> = vec![(u32::MAX, 0.0); n];
    let mut stripped: Vec<u32> = Vec::new();
    let mut down: Vec<Paths> = vec![Paths::ZERO; n];
    {
        let mut deg: Vec<usize> = (0..n).map(degree).collect();
        let mut stack: Vec<u32> = (0..n as u32).filter(|&a| deg[a as usize] == 1).collect();
        let mut removed = vec![false; n];
        while let Some(x) = stack.pop() {
            let x = x as usize;
            if removed[x] || deg[x] != 1 {
                continue;
            }
            let Some(&(p, order)) = adj[slots(x)].iter().find(|e| !removed[e.0 as usize]) else {
                continue;
            };
            removed[x] = true;
            parent[x] = (p, order);
            stripped.push(x as u32);
            let mut d = Paths::ZERO;
            for &(y, o) in &adj[slots(x)] {
                if parent[y as usize].0 == x as u32 {
                    let b = down[y as usize].behind(o);
                    for k in 1..=MAX_PATH {
                        d.count[k] += b.count[k];
                        d.pi[k] += b.pi[k];
                    }
                }
            }
            down[x] = d;
            let p = p as usize;
            deg[p] -= 1;
            if deg[p] == 1 {
                stack.push(p as u32);
            }
        }
    }
    // neighbour entries flagged with "the neighbour has no other neighbour"
    let adj: Vec<(u32, f64, bool)> = adj.iter().map(|&(o, order)| (o, order, degree(o as usize) == 1)).collect();

    struct Walk<'a> {
        start: &'a [usize],
        adj: &'a [(u32, f64, bool)],
        on_path: Vec<bool>,
        paths: Paths,
    }
    fn dfs(w: &mut Walk, atom: usize, depth: usize, weight: f64) {
        for k in w.start[atom]..w.start[atom + 1] {
            let (nb, order, dead_end) = w.adj[k];
            if w.on_path[nb as usize] {
                continue;
            }
            let wt = weight * order;
            w.paths.count[depth + 1] += 1;
            w.paths.pi[depth + 1] += wt;
            if depth + 1 < MAX_PATH && !dead_end {
                w.on_path[nb as usize] = true;
                dfs(w, nb as usize, depth + 1, wt);
                w.on_path[nb as usize] = false;
            }
        }
    }
    let mut w = Walk { start: &start, adj: &adj, on_path: vec![false; n], paths: Paths::ZERO };
    // directed paths from every atom: walked from the core atoms
    let mut of_atom = vec![Paths::ZERO; n];
    for s in 0..n {
        if parent[s].0 == u32::MAX {
            w.paths = Paths::ZERO;
            w.on_path[s] = true;
            dfs(&mut w, s, 0, 1.0);
            w.on_path[s] = false;
            of_atom[s] = w.paths;
        }
    }
    // pendant atoms, parents first: down the subtree, or to the parent and
    // on along its paths that do not start towards x
    for &x in stripped.iter().rev() {
        let x = x as usize;
        let (u, order) = parent[x];
        let (pu, dx) = (&of_atom[u as usize], &down[x]);
        let towards_x = dx.behind(order);
        let mut rest = Paths::ZERO;
        for k in 1..=MAX_PATH {
            rest.count[k] = pu.count[k] - towards_x.count[k];
            rest.pi[k] = pu.pi[k] - towards_x.pi[k];
        }
        let up = rest.behind(order);
        let mut px = *dx;
        for k in 1..=MAX_PATH {
            px.count[k] += up.count[k];
            px.pi[k] += up.pi[k];
        }
        of_atom[x] = px;
    }
    let mut total = Paths::ZERO;
    for p in &of_atom {
        for k in 1..=MAX_PATH {
            total.count[k] += p.count[k];
            total.pi[k] += p.pi[k];
        }
    }
    // every path was counted from both ends
    let count = |k: usize| total.count[k] / 2;
    let pi = |k: usize| total.pi[k] / 2.0;

    let mut out = Vec::with_capacity(MORDRED_PATH_COUNT_LEN);
    out.extend((2..=MAX_PATH).map(|k| count(k) as f64));
    out.push((n as u64 + (1..=MAX_PATH).map(count).sum::<u64>()) as f64);
    out.extend((1..=MAX_PATH).map(|k| (pi(k) + 1.0).ln()));
    let mut t = n as f64;
    for k in 1..=MAX_PATH {
        t += pi(k);
    }
    out.push((t + 1.0).ln());
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

    /// Plain enumeration of every simple path (counted once, from the lower
    /// end).
    fn reference(mol: &Molecule) -> Vec<f64> {
        let n = mol.num_atoms();
        let is_h = |a: usize| mol.atoms[a].atomic_number == 1;
        let nbrs = |a: usize| -> Vec<(usize, f64)> {
            if is_h(a) {
                return Vec::new();
            }
            mol.adjacency[a]
                .iter()
                .map(|&b| {
                    let b = &mol.bonds[b as usize];
                    let o = if b.source as usize == a { b.target as usize } else { b.source as usize };
                    (o, bond_order(b.bond_type))
                })
                .filter(|&(o, _)| !is_h(o))
                .collect()
        };
        #[allow(clippy::too_many_arguments)]
        fn go(
            u: usize,
            d: usize,
            w: f64,
            root: usize,
            vis: &mut Vec<bool>,
            acc: &mut Paths,
            nb: &dyn Fn(usize) -> Vec<(usize, f64)>,
        ) {
            for (v, o) in nb(u) {
                if vis[v] {
                    continue;
                }
                if v > root {
                    acc.count[d + 1] += 1;
                    acc.pi[d + 1] += w * o;
                }
                if d + 1 < MAX_PATH {
                    vis[v] = true;
                    go(v, d + 1, w * o, root, vis, acc, nb);
                    vis[v] = false;
                }
            }
        }
        let mut acc = Paths::ZERO;
        let mut vis = vec![false; n];
        for s in 0..n {
            vis[s] = true;
            go(s, 0, 1.0, s, &mut vis, &mut acc, &nbrs);
            vis[s] = false;
        }
        let mut out: Vec<f64> = (2..=MAX_PATH).map(|k| acc.count[k] as f64).collect();
        out.push((n as u64 + acc.count[1..].iter().sum::<u64>()) as f64);
        out.extend((1..=MAX_PATH).map(|k| (acc.pi[k] + 1.0).ln()));
        let mut t = n as f64;
        for k in 1..=MAX_PATH {
            t += acc.pi[k];
        }
        out.push((t + 1.0).ln());
        out
    }

    #[test]
    fn pendant_paths_match_the_plain_enumeration() {
        for smi in [
            "C",
            "CC",
            "CCO",
            "CC(C)(C)C",
            "C1CC1",
            "c1ccccc1",
            "CCCCCCCCCCCCCCCCCCCC",
            "C1CCC2(CC1)CCCC2",
            "OC1CCC(CC1)C1CCCCC1",
            "C12C3C4C1C5C2C3C45",
            "CC(O)C1CC(N)C(C1)C(=O)O",
            "c1ccc2c(c1)ccc1ccccc12",
            "O=C(Nc1ccccc1)c1ccc(cc1)N1CCN(CC1)c1ncccn1",
            "[2H]C([2H])([2H])c1ccccc1",
            "[H]C([H])([H])C=O",
            "[Na+].[Cl-]",
            "CC.CCC1CC1",
            "C#CC(=O)OCC1=CC=CC=C1",
            "FC(F)(F)C(Cl)(Br)I",
            "O=C=O",
            "N#N",
            "CCCCCCCCCCCCCCCCCCCCC(CCCCCCCCCCCC)CCCCCCC(C)(C)C",
            "CC(C)(C)OC(=O)NC(CC(C)C)C(=O)OC1CCCCC1OCCOC",
        ] {
            let mol = parse_smiles(smi).unwrap();
            assert_eq!(mordred_path_count(&mol), reference(&mol), "{smi}");
        }
    }

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
