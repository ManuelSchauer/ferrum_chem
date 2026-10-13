//! The Kekulé form Mordred (mordred-community 2.0.7) works on: RDKit
//! 2026.03 `Chem.Kekulize(Chem.AddHs(mol))`.
//!
//! Since RDKit 2026.03, `Chem.Kekulize` defaults to `canonical = True`: the
//! atoms of a fused aromatic system are visited in the order of their
//! canonical rank (`Canon::rankFragmentAtoms` over all atoms and bonds of the
//! hydrogen-explicit molecule, ties broken), so the double bonds no longer
//! depend on the input atom order. This module ports that ranking
//! (`new_canon.h` / `new_canon.cpp`: `AtomCompareFunctor` with neighbours,
//! `SpecialChiralityAtomCompareFunctor`, `SpecialSymmetryAtomCompareFunctor`,
//! `RefinePartitions`, `BreakTies`, `hanoisort`) for molecules read from
//! SMILES (no atom maps, stereo groups or atropisomers); the double-bond
//! assignment itself is the rank-ordered `kekulizeWorker` in
//! `algo::sanitize`.

use crate::algo::stereo::{BondStereo, LegacyStereo, legacy_stereo_perception_cached};
use crate::desc::context::CalculationContext;
use crate::mol::atom::ChiralTag;
use crate::mol::molecule::Molecule;
use crate::types::BondType;

/// RDKit `Bond::BondType` value as used by `bondholder` (aromatic flag wins).
fn rdkit_bond_type(t: BondType, aromatic: bool) -> u8 {
    if aromatic {
        return 12;
    }
    match t {
        BondType::Single => 1,
        BondType::Double => 2,
        BondType::Triple => 3,
        BondType::Aromatic => 12,
        BondType::Dative => 17,
        BondType::Unknown => 0,
    }
}

/// RDKit `Bond::BondStereo` value.
fn rdkit_bond_stereo(s: BondStereo) -> u8 {
    match s {
        BondStereo::None => 0,
        BondStereo::Any => 1,
        BondStereo::Z => 2,
        BondStereo::E => 3,
    }
}

/// RDKit `Atom::ChiralType` value.
fn rdkit_chiral_tag(t: ChiralTag) -> u8 {
    match t {
        ChiralTag::Unspecified => 0,
        ChiralTag::TetrahedralCW => 1,
        ChiralTag::TetrahedralCCW => 2,
        ChiralTag::Other => 3,
    }
}

/// `Canon::bondholder` (no symbols, no CIS/TRANS controlling atoms).
#[derive(Clone, Copy)]
struct BondHolder {
    bond_type: u8,
    stereo: u8,
    nbr_idx: u32,
    nbr_class: i32,
}

impl BondHolder {
    /// `bondholder::compare`.
    fn compare(x: &BondHolder, y: &BondHolder) -> i32 {
        if x.bond_type != y.bond_type {
            return if x.bond_type < y.bond_type { -1 } else { 1 };
        }
        if x.stereo != y.stereo {
            return if x.stereo < y.stereo { -1 } else { 1 };
        }
        if x.nbr_class != y.nbr_class {
            return if x.nbr_class < y.nbr_class { -1 } else { 1 };
        }
        // both E/Z (or ANY) specified: equal stereo values compare equal
        0
    }
}

/// `countSwapsToInterconvert(ref, probe)`.
fn count_swaps(reference: &[u32], probe: &mut [u32]) -> u32 {
    let mut swaps = 0;
    for i in 0..reference.len() {
        if probe[i] != reference[i] {
            let j = (i..probe.len()).find(|&j| probe[j] == reference[i]).expect("probe element");
            probe.swap(i, j);
            swaps += 1;
        }
    }
    swaps
}

/// Hydrogen-explicit molecule with the per-atom canonicalisation state
/// (`Canon::canon_atom`).
struct Canon {
    z: Vec<u8>,
    isotope: Vec<u16>,
    charge: Vec<i8>,
    chiral: Vec<u8>,
    num_rings: Vec<u32>,
    /// Neighbours in RDKit order (`nbrIds`) and the bond holders (sorted,
    /// `canon_atom::bonds`), both at `nbr_start[i]..nbr_start[i + 1]`.
    nbr_start: Vec<u32>,
    nbrs: Vec<u32>,
    bonds: Vec<BondHolder>,
    /// Current class of each atom (`canon_atom::index`).
    index: Vec<i32>,
    neighbor_num: Vec<Vec<i32>>,
    revisited: Vec<Vec<i32>>,
    /// Bumped whenever `index` changes; an atom's bond list is re-sorted only
    /// when its `sorted_at` epoch is stale (the sort is deterministic, so this
    /// equals RDKit's re-sorting on every comparison).
    epoch: u64,
    sorted_at: Vec<u64>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Functor {
    /// `AtomCompareFunctor` with `df_useNbrs = true`.
    Atom,
    SpecialChirality,
    SpecialSymmetry,
}

impl Canon {
    fn new(mol: &Molecule, stereo: &LegacyStereo) -> Self {
        let n = mol.num_atoms();
        let n_h: usize = mol.atoms.iter().map(|a| a.total_h() as usize).sum();
        let total = n + n_h;
        let info = mol.ring_info().expect("ring info");
        let mut nbr_start = Vec::with_capacity(total + 1);
        let mut nbrs = Vec::with_capacity(2 * (mol.num_bonds() + n_h));
        let mut bonds = Vec::with_capacity(2 * (mol.num_bonds() + n_h));
        // neighbour order: existing bonds, then the added hydrogens
        let mut h = n as u32;
        for a in 0..n {
            nbr_start.push(nbrs.len() as u32);
            for &bi in &mol.adjacency[a] {
                let b = &mol.bonds[bi as usize];
                let o = if b.source as usize == a { b.target } else { b.source };
                nbrs.push(o as u32);
                bonds.push(BondHolder {
                    bond_type: rdkit_bond_type(b.bond_type, b.is_aromatic()),
                    stereo: rdkit_bond_stereo(stereo.bond_stereo[bi as usize]),
                    nbr_idx: o as u32,
                    nbr_class: 0,
                });
            }
            for _ in 0..mol.atoms[a].total_h() {
                nbrs.push(h);
                bonds.push(BondHolder { bond_type: 1, stereo: 0, nbr_idx: h, nbr_class: 0 });
                h += 1;
            }
        }
        for (a, atom) in mol.atoms.iter().enumerate() {
            for _ in 0..atom.total_h() {
                nbr_start.push(nbrs.len() as u32);
                nbrs.push(a as u32);
                bonds.push(BondHolder { bond_type: 1, stereo: 0, nbr_idx: a as u32, nbr_class: 0 });
            }
        }
        nbr_start.push(nbrs.len() as u32);
        let mut c = Canon {
            z: Vec::with_capacity(total),
            isotope: Vec::with_capacity(total),
            charge: Vec::with_capacity(total),
            chiral: Vec::with_capacity(total),
            num_rings: Vec::with_capacity(total),
            nbr_start,
            nbrs,
            bonds,
            index: vec![0; total],
            neighbor_num: vec![Vec::new(); total],
            revisited: vec![Vec::new(); total],
            epoch: 0,
            sorted_at: vec![u64::MAX; total],
        };
        for (a, atom) in mol.atoms.iter().enumerate() {
            c.z.push(atom.atomic_number);
            c.isotope.push(atom.isotope);
            c.charge.push(atom.formal_charge);
            c.chiral.push(rdkit_chiral_tag(stereo.chiral_tags[a]));
            c.num_rings.push(info.num_atom_rings(a) as u32);
        }
        c.z.resize(total, 1);
        c.isotope.resize(total, 0);
        c.charge.resize(total, 0);
        c.chiral.resize(total, 0);
        c.num_rings.resize(total, 0);
        c
    }

    #[inline]
    fn range(&self, i: usize) -> std::ops::Range<usize> {
        self.nbr_start[i] as usize..self.nbr_start[i + 1] as usize
    }

    #[inline]
    fn nb(&self, i: usize) -> &[u32] {
        &self.nbrs[self.range(i)]
    }

    #[inline]
    fn bh(&self, i: usize) -> &[BondHolder] {
        &self.bonds[self.range(i)]
    }

    fn len(&self) -> usize {
        self.z.len()
    }

    /// `updateAtomNeighborIndex`.
    fn update_nbr_index(&mut self, i: usize) {
        if self.sorted_at[i] == self.epoch {
            return;
        }
        self.sorted_at[i] = self.epoch;
        let r = self.range(i);
        let index = &self.index;
        let bonds = &mut self.bonds[r];
        for b in bonds.iter_mut() {
            b.nbr_class = index[b.nbr_idx as usize];
        }
        bonds.sort_by(|x, y| BondHolder::compare(y, x).cmp(&0));
    }

    fn compare_bond_lists(&self, i: usize, j: usize) -> i32 {
        for (x, y) in self.bh(i).iter().zip(self.bh(j)) {
            let c = BondHolder::compare(x, y);
            if c != 0 {
                return c;
            }
        }
        0
    }

    /// Anonymous-namespace `getChiralRank`.
    fn chiral_rank(&self, i: usize) -> u32 {
        let mut perm: Vec<u32> = Vec::with_capacity(self.nb(i).len());
        for &nb in self.nb(i) {
            let r = self.index[nb as usize] as u32;
            if perm.contains(&r) {
                break;
            }
            perm.push(r);
        }
        if perm.len() != self.nb(i).len() {
            return 0;
        }
        let tag = self.chiral[i];
        if tag != 1 && tag != 2 {
            return 0;
        }
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        let swaps = count_swaps(&perm, &mut sorted);
        let res = if tag == 1 { 2 } else { 1 };
        if swaps % 2 == 1 { 3 - res } else { res }
    }

    /// `AtomCompareFunctor::basecomp` (no atom maps, symbols or stereo groups;
    /// `totalNumHs` is 0 on the hydrogen-explicit graph).
    fn basecomp(&self, i: usize, j: usize) -> i32 {
        macro_rules! cmp {
            ($a:expr, $b:expr) => {
                if $a < $b {
                    return -1;
                } else if $a > $b {
                    return 1;
                }
            };
        }
        cmp!(self.index[i], self.index[j]);
        cmp!(self.nb(i).len(), self.nb(j).len());
        cmp!(self.z[i], self.z[j]);
        cmp!(self.isotope[i], self.isotope[j]);
        // RDKit compares the charges as `unsigned int`: negative charges rank last
        cmp!(self.charge[i] as i32 as u32, self.charge[j] as i32 as u32);
        // chirality (no stereo groups): presence of a tag, then its rank
        let (ci, cj) = (self.chiral[i] != 0, self.chiral[j] != 0);
        cmp!(ci, cj);
        if ci && cj {
            cmp!(self.chiral_rank(i), self.chiral_rank(j));
        }
        0
    }

    /// `updateAtomNeighborNumSwaps` (second elements, sorted by pair).
    fn neighbor_num_swaps(&self, atom: usize) -> Vec<(i32, u32)> {
        let ring_atom = self.num_rings[atom] > 0;
        let mut out = Vec::with_capacity(self.bh(atom).len());
        for b in self.bh(atom) {
            let nb = b.nbr_idx as usize;
            if ring_atom && self.chiral[nb] != 0 {
                let mut seen: Vec<i32> = Vec::new();
                let mut too_many = false;
                let reference: Vec<u32> = self.nb(nb).to_vec();
                for &x in self.nb(nb) {
                    if x as usize != atom {
                        let cls = self.index[x as usize];
                        if seen.contains(&cls) {
                            too_many = true;
                        } else {
                            seen.push(cls);
                        }
                    }
                }
                let mut probe: Vec<u32> = vec![atom as u32];
                probe.extend(self.bh(nb).iter().map(|x| x.nbr_idx).filter(|&x| x as usize != atom));
                if too_many {
                    out.push((b.nbr_class, 0));
                } else {
                    let swaps = count_swaps(&reference, &mut probe);
                    match self.chiral[nb] {
                        1 => out.push((b.nbr_class, if swaps % 2 == 1 { 2 } else { 1 })),
                        2 => out.push((b.nbr_class, if swaps % 2 == 1 { 1 } else { 2 })),
                        _ => {}
                    }
                }
            } else {
                out.push((b.nbr_class, 0));
            }
        }
        out.sort_unstable();
        out
    }

    fn compare(&mut self, f: Functor, i: usize, j: usize) -> i32 {
        match f {
            Functor::Atom => {
                let v = self.basecomp(i, j);
                if v != 0 {
                    return v;
                }
                self.update_nbr_index(i);
                self.update_nbr_index(j);
                let c = self.compare_bond_lists(i, j);
                if c != 0 {
                    return c;
                }
                (self.bh(i).len() as i32 - self.bh(j).len() as i32).signum()
            }
            Functor::SpecialChirality => {
                self.update_nbr_index(i);
                self.update_nbr_index(j);
                let c = self.compare_bond_lists(i, j);
                if c != 0 {
                    return c;
                }
                let (si, sj) = (self.neighbor_num_swaps(i), self.neighbor_num_swaps(j));
                for (x, y) in si.iter().zip(&sj) {
                    if x.1 != y.1 {
                        return if x.1 < y.1 { -1 } else { 1 };
                    }
                }
                0
            }
            Functor::SpecialSymmetry => {
                match self.neighbor_num[i].cmp(&self.neighbor_num[j]) {
                    std::cmp::Ordering::Less => return -1,
                    std::cmp::Ordering::Greater => return 1,
                    _ => {}
                }
                match self.revisited[i].cmp(&self.revisited[j]) {
                    std::cmp::Ordering::Less => return -1,
                    std::cmp::Ordering::Greater => return 1,
                    _ => {}
                }
                self.update_nbr_index(i);
                self.update_nbr_index(j);
                let c = self.compare_bond_lists(i, j);
                if c != 0 {
                    return c;
                }
                (self.bh(i).len() as i32 - self.bh(j).len() as i32).signum()
            }
        }
    }

    /// `compareRingAtomsConcerningNumNeighbors`.
    fn ring_neighbor_signatures(&mut self) {
        let n = self.len();
        let mut visited = vec![false; n];
        let mut last_level = vec![false; n];
        let mut current_level = vec![false; n];
        let mut revisited = vec![0i32; n];
        for idx in 0..n {
            if self.num_rings[idx] == 0 {
                continue;
            }
            visited.fill(false);
            last_level.fill(false);
            current_level.fill(false);
            revisited.fill(0);
            let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::from([idx]);
            let mut nnum = Vec::new();
            let mut rev = Vec::new();
            while !queue.is_empty() {
                let mut num_level = 0i32;
                let mut next_level = Vec::new();
                while let Some(x) = queue.pop_front() {
                    if self.num_rings[x] == 0 {
                        continue;
                    }
                    last_level[x] = true;
                    visited[x] = true;
                    for &y in self.nb(x) {
                        let y = y as usize;
                        if !visited[y] {
                            current_level[y] = true;
                            num_level += 1;
                            visited[y] = true;
                            next_level.push(y);
                        }
                    }
                }
                for i in 0..n {
                    if current_level[i] {
                        for &k in self.nb(i) {
                            let k = k as usize;
                            if current_level[k] || last_level[k] {
                                revisited[k] += 1;
                            }
                        }
                    }
                }
                last_level.copy_from_slice(&current_level);
                current_level.fill(false);
                let mut tmp: Vec<i32> = revisited.iter().copied().filter(|&r| r > 0).collect();
                tmp.sort_unstable();
                tmp.push(-1);
                rev.extend(tmp);
                revisited.fill(0);
                nnum.push(num_level);
                nnum.push(-1);
                queue.extend(next_level);
            }
            self.neighbor_num[idx] = nnum;
            self.revisited[idx] = rev;
        }
    }
}

/// RDKit `hanoi` merge sort over `buf[base..base + nel]` with scratch
/// `buf[temp..temp + nel]`; returns true when the result is in the scratch.
#[allow(clippy::too_many_arguments)]
fn hanoi(
    buf: &mut [usize],
    base: usize,
    nel: usize,
    temp: usize,
    count: &mut [i32],
    changed: &[i32],
    canon: &mut Canon,
    f: Functor,
) -> bool {
    if nel == 1 {
        count[buf[base]] = 1;
        return false;
    } else if nel == 2 {
        let (n1, n2) = (buf[base], buf[base + 1]);
        let stat = if changed[n1] != 0 || changed[n2] != 0 { canon.compare(f, n1, n2) } else { 0 };
        if stat == 0 {
            count[n1] = 2;
            count[n2] = 0;
        } else {
            count[n1] = 1;
            count[n2] = 1;
            if stat > 0 {
                buf[base] = n2;
                buf[base + 1] = n1;
            }
        }
        return false;
    }
    let mut n1 = nel / 2;
    let mut n2 = nel - n1;
    let (b1, t1, b2, t2) = (base, temp, base + n1, temp + n1);
    let result;
    let mut ptr;
    let mut s1;
    let s2_in_temp;
    if hanoi(buf, b1, n1, t1, count, changed, canon, f) {
        s2_in_temp = hanoi(buf, b2, n2, t2, count, changed, canon, f);
        result = false;
        ptr = base;
        s1 = t1;
    } else {
        s2_in_temp = hanoi(buf, b2, n2, t2, count, changed, canon, f);
        result = true;
        ptr = temp;
        s1 = b1;
    }
    let mut s2 = if s2_in_temp { t2 } else { b2 };
    loop {
        let (a1, a2) = (buf[s1], buf[s2]);
        let stat = if changed[a1] != 0 || changed[a2] != 0 { canon.compare(f, a1, a2) } else { 0 };
        let len1 = count[a1] as usize;
        let len2 = count[a2] as usize;
        if stat == 0 {
            count[a1] = (len1 + len2) as i32;
            count[a2] = 0;
            buf.copy_within(s1..s1 + len1, ptr);
            ptr += len1;
            n1 -= len1;
            if n1 == 0 {
                if ptr != s2 {
                    buf.copy_within(s2..s2 + n2, ptr);
                }
                return result;
            }
            s1 += len1;
            buf.copy_within(s2..s2 + len2, ptr);
            ptr += len2;
            n2 -= len2;
            if n2 == 0 {
                buf.copy_within(s1..s1 + n1, ptr);
                return result;
            }
            s2 += len2;
        } else if stat < 0 {
            buf.copy_within(s1..s1 + len1, ptr);
            ptr += len1;
            n1 -= len1;
            if n1 == 0 {
                if ptr != s2 {
                    buf.copy_within(s2..s2 + n2, ptr);
                }
                return result;
            }
            s1 += len1;
        } else {
            buf.copy_within(s2..s2 + len2, ptr);
            ptr += len2;
            n2 -= len2;
            if n2 == 0 {
                buf.copy_within(s1..s1 + n1, ptr);
                return result;
            }
            s2 += len2;
        }
    }
}

/// Partition-refinement state (`order`, `count`, `next`, `changed`, `touched`).
struct Refiner {
    order: Vec<usize>,
    count: Vec<i32>,
    next: Vec<i32>,
    changed: Vec<i32>,
    touched: Vec<bool>,
    /// The set entries of `touched` (scanned in ascending order like RDKit's
    /// loop over all partitions).
    touched_list: Vec<usize>,
    activeset: i32,
    buf: Vec<usize>,
}

impl Refiner {
    #[inline]
    fn touch(&mut self, partition: usize) {
        if !self.touched[partition] {
            self.touched[partition] = true;
            self.touched_list.push(partition);
        }
    }

    /// Activates the touched partitions that are still ties.
    fn flush_touched(&mut self) {
        self.touched_list.sort_unstable();
        for k in 0..self.touched_list.len() {
            let ii = self.touched_list[k];
            let p = self.order[ii];
            if self.count[p] > 1 && self.next[p] == -2 {
                self.next[p] = self.activeset;
                self.activeset = p as i32;
            }
            self.touched[ii] = false;
        }
        self.touched_list.clear();
    }

    fn hanoisort(&mut self, canon: &mut Canon, f: Functor, start: usize, len: usize) {
        self.buf.clear();
        self.buf.resize(2 * len, 0);
        self.buf[..len].copy_from_slice(&self.order[start..start + len]);
        let in_temp = hanoi(&mut self.buf, 0, len, len, &mut self.count, &self.changed, canon, f);
        let src = if in_temp { len } else { 0 };
        self.order[start..start + len].copy_from_slice(&self.buf[src..src + len]);
    }

    /// `ActivatePartitions`.
    fn activate(&mut self) {
        let n = self.order.len();
        self.activeset = -1;
        self.next.fill(-2);
        let mut i = 0;
        while i < n {
            let j = self.order[i];
            if self.count[j] > 1 {
                self.next[j] = self.activeset;
                self.activeset = j as i32;
                i += self.count[j] as usize;
            } else {
                i += 1;
            }
        }
        self.changed.fill(1);
    }

    /// `RefinePartitions` (mode = 1).
    fn refine(&mut self, canon: &mut Canon, f: Functor) {
        let mut symclass = 0i32;
        while self.activeset != -1 {
            let partition = self.activeset as usize;
            self.activeset = self.next[partition];
            self.next[partition] = -2;
            let len = self.count[partition] as usize;
            let offset = canon.index[partition] as usize;
            self.hanoisort(canon, f, offset, len);
            for k in 0..len {
                self.changed[self.order[offset + k]] = 0;
            }
            let first = self.count[self.order[offset]] as usize;
            for i in first..len {
                let index = self.order[offset + i];
                if self.count[index] != 0 {
                    symclass = (offset + i) as i32;
                }
                canon.index[index] = symclass;
                canon.epoch += 1;
                for &nb in canon.nb(index) {
                    self.changed[nb as usize] = 1;
                }
            }
            let first = self.count[self.order[offset]] as usize;
            for i in first..len {
                let index = self.order[offset + i];
                for &nb in canon.nb(index) {
                    self.touch(canon.index[nb as usize] as usize);
                }
            }
            self.flush_touched();
        }
    }

    /// `BreakTies` (mode = 1).
    fn break_ties(&mut self, canon: &mut Canon, f: Functor) {
        let n = self.order.len() as isize;
        let mut i: isize = 0;
        while i < n {
            let partition = self.order[i as usize];
            let old_part = canon.index[partition];
            while self.count[partition] > 1 {
                let len = self.count[partition];
                let offset = (canon.index[partition] + len - 1) as usize;
                let index = self.order[offset];
                canon.index[index] = offset as i32;
                canon.epoch += 1;
                self.count[partition] = len - 1;
                self.count[index] = 1;
                if canon.nb(index).is_empty() {
                    continue;
                }
                for &nb in canon.nb(index) {
                    self.touch(canon.index[nb as usize] as usize);
                    self.changed[nb as usize] = 1;
                }
                self.flush_touched();
                self.refine(canon, f);
            }
            if canon.index[partition] != old_part {
                i -= 1;
            }
            i += 1;
        }
    }
}

/// RDKit `Canon::rankFragmentAtoms(mol, res, all atoms, all bonds)` with the
/// default flags (break ties, chirality, isotopes) on the
/// hydrogen-explicit molecule (`Chem.AddHs`: hydrogens appended after the
/// existing atoms, grouped by parent atom). Returns one rank per atom of the
/// hydrogen-explicit molecule.
pub(crate) fn canonical_ranks_with_hs(mol: &Molecule, ctx: &mut CalculationContext) -> Vec<u32> {
    let stereo = legacy_stereo_perception_cached(mol, ctx);
    let mut canon = Canon::new(mol, stereo);
    let n = canon.len();
    if n == 0 {
        return Vec::new();
    }
    let mut r = Refiner {
        order: (0..n).collect(),
        count: vec![0; n],
        next: vec![-2; n],
        changed: vec![1; n],
        touched: vec![false; n],
        touched_list: Vec::new(),
        activeset: -1,
        buf: Vec::new(),
    };
    // CreateSinglePartition
    canon.index.fill(0);
    canon.epoch += 1;
    r.count[0] = n as i32;
    r.activate();
    r.refine(&mut canon, Functor::Atom);

    let ties = r.count.contains(&0);
    if ties {
        r.activate();
        r.refine(&mut canon, Functor::SpecialChirality);
    }
    let mut ties = false;
    let mut sym_ring_atoms = 0usize;
    let mut ring_atoms = 0usize;
    let mut branching_ring_atom = false;
    for i in 0..n {
        let o = r.order[i];
        if canon.num_rings[o] > 0 {
            if r.count[o] > 2 {
                sym_ring_atoms += r.count[o] as usize;
            }
            ring_atoms += 1;
            if canon.num_rings[o] > 1 && r.count[o] > 1 {
                branching_ring_atom = true;
            }
        }
        if r.count[i] == 0 {
            ties = true;
        }
    }
    if ties && ring_atoms > 0 && (sym_ring_atoms as f32) / (ring_atoms as f32) > 0.5 && branching_ring_atom {
        canon.ring_neighbor_signatures();
        r.activate();
        r.refine(&mut canon, Functor::SpecialSymmetry);
    }
    r.break_ties(&mut canon, Functor::Atom);
    canon.index.iter().map(|&i| i as u32).collect()
}

/// Kekulé bond types of `Chem.Kekulize(Chem.AddHs(mol))` (RDKit 2026.03,
/// `canonical = true`), one per bond of `mol`; `None` if RDKit could not
/// kekulize the molecule.
pub(crate) fn mordred_kekule_bond_types(mol: &Molecule, ctx: &mut CalculationContext) -> Option<Vec<BondType>> {
    if !mol.bonds.iter().any(|b| b.is_aromatic() || b.bond_type == BondType::Aromatic) {
        return Some(mol.bonds.iter().map(|b| b.bond_type).collect());
    }
    let ranks = canonical_ranks_with_hs(mol, ctx);
    crate::algo::sanitize::kekule_bond_types(mol, Some(&ranks[..mol.num_atoms()]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn ranks_match_rdkit() {
        // Expected: RDKit 2026.03.2
        // CanonicalRankAtomsInFragment(AddHs(mol), all atoms, all bonds).
        for (smi, want) in [
            ("CN(=O)=O", vec![6, 5, 4, 3, 0, 1, 2]),
            ("[O-][N+](C)=O", vec![4, 5, 6, 3, 0, 1, 2]),
            ("[cH-]1cccc1", vec![9, 7, 5, 6, 8, 4, 2, 0, 1, 3]),
            ("c1ccccc1N(=O)=O", vec![10, 8, 7, 9, 11, 12, 13, 6, 5, 3, 1, 0, 2, 4]),
        ] {
            let mol = parse_smiles(smi).unwrap();
            assert_eq!(canonical_ranks_with_hs(&mol, &mut CalculationContext::new()), want, "{smi}");
        }
    }

    #[test]
    fn kekule_structures_match_rdkit() {
        // Expected: double bonds of RDKit 2026.03.2 Chem.Kekulize(Chem.AddHs(mol)).
        for (smi, double) in [
            ("c1ccccc1", vec![(1, 2), (3, 4), (5, 0)]),
            ("NC1=CC=C(C=C1Cl)C2=CC(=C(N)C=C2)Cl", vec![(2, 3), (4, 5), (6, 1), (8, 9), (10, 11), (13, 14)]),
        ] {
            let mol = parse_smiles(smi).unwrap();
            let types = mordred_kekule_bond_types(&mol, &mut CalculationContext::new()).unwrap();
            let mut got: Vec<(u16, u16)> = mol
                .bonds
                .iter()
                .zip(&types)
                .filter(|(_, t)| **t == BondType::Double)
                .map(|(b, _)| (b.source, b.target))
                .collect();
            let mut want: Vec<(u16, u16)> = double.into_iter().collect();
            let norm = |v: &mut Vec<(u16, u16)>| {
                for p in v.iter_mut() {
                    *p = (p.0.min(p.1), p.0.max(p.1));
                }
                v.sort_unstable();
            };
            norm(&mut got);
            norm(&mut want);
            assert_eq!(got, want, "{smi}");
        }
    }
}
