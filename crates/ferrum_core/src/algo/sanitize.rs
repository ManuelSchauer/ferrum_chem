//! Molecule sanitization: a port of RDKit's `MolOps::sanitizeMol` pipeline.
//!
//! FerrumChem has exactly one chemistry model: RDKit's defaults. Every molecule
//! (SMILES or SDF) goes through this pipeline once at parse time, so all
//! descriptors see the same atom typing RDKit would produce:
//!
//! 1. `remove_hs`: fold explicit hydrogen atoms into their heavy neighbours
//!    (`MolFromSmiles` default `removeHs`), recording them for the CDK views
//! 2. `clean_up`: normalise nitro/N-oxide, P and halogen oxo forms
//! 3. `clean_up_organometallics`: hypervalent non-metal → metal single bonds
//!    become dative
//! 4. valence check + implicit H (`updatePropertyCache(strict)`)
//! 5. ring perception (symmetrized SSSR, see [`crate::algo::sssr`])
//! 6. `kekulize` aromatic input
//! 7. `assign_radicals`
//! 8. `set_aromaticity` (RDKit default model, incl. fused systems)
//! 9. `set_conjugation`, `set_hybridization`
//! 10. `adjust_hs` + final implicit H
//!
//! Sources: `Code/GraphMol/{MolOps,Atom,Kekulize,Aromaticity,ConjugHybrid,FindRings,AddHs}.cpp`.

use crate::algo::sssr;
use crate::error::{FerrumError, Result};
use crate::mol::molecule::{Molecule, RemovedHydrogen, RingInfo};
use crate::mol::rdkit_data::ELEMENTS;
use crate::types::{AtomIdx, BondIdx, BondType, Hybridization};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const MAX_BACKTRACKS: u32 = 100;
const MAX_FUSED_AROMATIC_RING_SIZE: usize = 24;
const MAX_NUM_FUSED_RINGS: usize = 6;

// ---------------------------------------------------------------------------
// Element helpers
// ---------------------------------------------------------------------------

#[inline]
fn valence_list(z: usize) -> &'static [i8] {
    ELEMENTS[z.min(118)].valences
}

#[inline]
fn default_valence(z: usize) -> i32 {
    valence_list(z)[0] as i32
}

#[inline]
fn last_valence(z: usize) -> i32 {
    *valence_list(z).last().unwrap() as i32
}

#[inline]
fn n_outer(z: usize) -> i32 {
    ELEMENTS[z.min(118)].n_outer as i32
}

/// RDKit `isEarlyAtom`: elements left of carbon in their period.
fn is_early_atom(z: usize) -> bool {
    matches!(
        z,
        3 | 4 | 5 | 11 | 12 | 13 | 19 | 20 | 21 | 22 | 30 | 31 | 32 | 37..=41 | 48..=51
            | 55..=61 | 72 | 73 | 80..=83 | 87..=93 | 104..=118
    )
}

/// RDKit `QueryOps::isMetal` (Marvin definition).
pub fn is_metal(z: u8) -> bool {
    !matches!(
        z,
        0 | 1 | 2 | 5 | 6 | 7 | 8 | 9 | 10 | 14 | 15 | 16 | 17 | 18 | 33 | 34 | 35 | 36 | 52 | 53 | 54 | 85 | 86
    )
}

/// RDKit `PeriodicTable::moreElectroNegative`.
fn more_electronegative(z1: usize, z2: usize) -> bool {
    let (ne1, ne2) = (n_outer(z1), n_outer(z2));
    ne1 > ne2 || (ne1 == ne2 && z1 < z2)
}

// ---------------------------------------------------------------------------
// Graph helpers
// ---------------------------------------------------------------------------

#[inline]
fn other_atom(mol: &Molecule, bond: usize, atom: usize) -> usize {
    let b = &mol.bonds[bond];
    if b.source as usize == atom { b.target as usize } else { b.source as usize }
}

fn bond_between(mol: &Molecule, a: usize, b: usize) -> Option<usize> {
    mol.adjacency[a]
        .iter()
        .map(|&bi| bi as usize)
        .find(|&bi| other_atom(mol, bi, a) == b)
}

/// RDKit `Bond::getValenceContrib`.
#[inline]
fn valence_contrib(mol: &Molecule, bond: usize, atom: usize) -> f64 {
    let b = &mol.bonds[bond];
    match b.bond_type {
        BondType::Single => 1.0,
        BondType::Double => 2.0,
        BondType::Triple => 3.0,
        BondType::Aromatic => 1.5,
        BondType::Dative => {
            if b.target as usize == atom {
                1.0
            } else {
                0.0
            }
        }
        BondType::Unknown => 0.0,
    }
}

/// RDKit `isAromaticAtom`: flagged aromatic or has an aromatic bond.
fn is_aromatic_atom(mol: &Molecule, a: usize) -> bool {
    mol.atoms[a].is_aromatic()
        || mol.adjacency[a].iter().any(|&bi| {
            let b = &mol.bonds[bi as usize];
            b.is_aromatic() || b.bond_type == BondType::Aromatic
        })
}

#[inline]
fn effective_atomic_num(mol: &Molecule, a: usize) -> usize {
    let atom = &mol.atoms[a];
    (atom.atomic_number as i32 - atom.formal_charge as i32).clamp(0, 118) as usize
}

fn can_be_hypervalent(z: usize, eff: usize) -> bool {
    (eff > 16 && (z == 15 || z == 16)) || (eff > 34 && (z == 33 || z == 34))
}

fn valence_error(mol: &Molecule, a: usize, msg: &str) -> FerrumError {
    let z = mol.atoms[a].atomic_number as usize;
    let sym = crate::mol::elements::ELEMENT_SYMBOLS.get(z).copied().unwrap_or("?");
    FerrumError::ParseError(format!("{} for atom #{} {}", msg, a, sym))
}

// ---------------------------------------------------------------------------
// Valence model (Atom.cpp)
// ---------------------------------------------------------------------------

/// Port of `calculateExplicitValence`.
fn calc_explicit_valence(mol: &Molecule, a: usize, strict: bool) -> Result<i32> {
    let atom = &mol.atoms[a];
    let mut accum: f64 = mol.adjacency[a]
        .iter()
        .map(|&bi| valence_contrib(mol, bi as usize, a))
        .sum::<f64>()
        + atom.num_explicit_h as f64;
    let z = atom.atomic_number as usize;
    let ovalens = valence_list(z);
    let eff = if ovalens.len() > 1 || ovalens[0] != -1 { effective_atomic_num(mol, a) } else { z };
    let dv = default_valence(eff);
    let valens = valence_list(eff);
    // `dv` is unsigned in RDKit: -1 never compares smaller than accum.
    if dv >= 0 && accum > dv as f64 && is_aromatic_atom(mol, a) {
        let mut pval = dv;
        for &v in valens {
            if v == -1 || v as f64 > accum {
                break;
            }
            pval = v as i32;
        }
        if accum - pval as f64 <= 1.5 {
            accum = pval as f64;
        }
    }
    accum += 0.1;
    let res = accum.round() as i32;
    if strict {
        let mut max_valence = *valens.last().unwrap() as i32;
        let mut offset = 0;
        if can_be_hypervalent(z, eff) {
            max_valence = *ovalens.last().unwrap() as i32;
            offset -= atom.formal_charge as i32;
        }
        if z == 1 && atom.formal_charge == -1 {
            max_valence = 2;
        }
        if max_valence >= 0 && *ovalens.last().unwrap() >= 0 && res + offset > max_valence {
            return Err(valence_error(mol, a, "Explicit valence is greater than permitted"));
        }
    }
    Ok(res)
}

/// Port of `calculateImplicitValence`.
fn calc_implicit_valence(mol: &Molecule, a: usize, ev: i32, strict: bool) -> Result<i32> {
    let atom = &mol.atoms[a];
    if atom.no_implicit_h() {
        return Ok(0);
    }
    let z = atom.atomic_number as usize;
    if z == 0 {
        return Ok(0);
    }
    let charge = atom.formal_charge as i32;
    let nrad = atom.num_radicals() as i32;
    if ev == 0 && nrad == 0 && z == 1 {
        return match charge {
            1 | -1 => Ok(0),
            0 => Ok(1),
            _ if strict => Err(valence_error(mol, a, "Unreasonable formal charge")),
            _ => Ok(0),
        };
    }
    let mut explicit_plus_rad = ev + nrad;
    let ovalens = valence_list(z);
    let mut eff = if ovalens.len() > 1 || ovalens[0] != -1 { effective_atomic_num(mol, a) } else { z };
    if eff == 0 {
        return Ok(0);
    }
    let dv = default_valence(eff);
    if dv == -1 {
        return Ok(0);
    }
    if can_be_hypervalent(z, eff) {
        eff = z;
        explicit_plus_rad -= charge;
    }
    let valens = valence_list(eff);
    let res = if is_aromatic_atom(mol, a) {
        if explicit_plus_rad <= dv {
            dv - explicit_plus_rad
        } else {
            let satis = valens.iter().take_while(|&&v| v > 0).any(|&v| v as i32 == explicit_plus_rad);
            if !satis && strict {
                return Err(valence_error(
                    mol,
                    a,
                    "Explicit valence for aromatic atom not equal to any accepted valence",
                ));
            }
            0
        }
    } else {
        let mut res = -1;
        for &v in valens.iter().take_while(|&&v| v >= 0) {
            if explicit_plus_rad <= v as i32 {
                res = v as i32 - explicit_plus_rad;
                break;
            }
        }
        if res < 0 {
            if strict && *valens.last().unwrap() != -1 && *ovalens.last().unwrap() > 0 {
                return Err(valence_error(mol, a, "Explicit valence is greater than permitted"));
            }
            res = 0;
        }
        res
    };
    Ok(res)
}

/// Cached valences, mirroring RDKit's `d_explicitValence`/`d_implicitValence`.
struct ValenceCache {
    explicit: Vec<i32>,
    implicit: Vec<i32>,
}

impl ValenceCache {
    #[inline]
    fn total(&self, a: usize) -> i32 {
        self.explicit[a] + self.implicit[a]
    }
}

fn update_atom_cache(mol: &mut Molecule, cache: &mut ValenceCache, a: usize, strict: bool) -> Result<()> {
    let ev = calc_explicit_valence(mol, a, strict)?;
    let iv = calc_implicit_valence(mol, a, ev, strict)?;
    cache.explicit[a] = ev;
    cache.implicit[a] = iv;
    mol.atoms[a].num_implicit_h = iv.clamp(0, 255) as u8;
    Ok(())
}

fn update_property_cache(mol: &mut Molecule, strict: bool) -> Result<ValenceCache> {
    let n = mol.num_atoms();
    let mut cache = ValenceCache { explicit: vec![0; n], implicit: vec![0; n] };
    for a in 0..n {
        update_atom_cache(mol, &mut cache, a, strict)?;
    }
    Ok(cache)
}

/// RDKit `Atom::getTotalValence` on a sanitized molecule (used by SMARTS `v`).
pub fn total_valence(mol: &Molecule, a: usize) -> i32 {
    let ev = calc_explicit_valence(mol, a, false).unwrap_or(0);
    let atom = &mol.atoms[a];
    ev + if atom.no_implicit_h() { 0 } else { atom.num_implicit_h as i32 }
}

#[inline]
fn total_num_hs(mol: &Molecule, cache: &ValenceCache, a: usize) -> i32 {
    let atom = &mol.atoms[a];
    atom.num_explicit_h as i32 + if atom.no_implicit_h() { 0 } else { cache.implicit[a] }
}

// ---------------------------------------------------------------------------
// Explicit hydrogen removal (AddHs.cpp `removeHs`, SMILES defaults)
// ---------------------------------------------------------------------------

/// Removes explicit hydrogen atoms the way `MolFromSmiles` does
/// (`RemoveHsParameters` defaults with `updateExplicitCount = true`).
///
/// Kept: isotopic H, H without neighbours, H with more than one neighbour,
/// H bonded only to H, hydrides `[H-]`... except that RDKit removes hydrides
/// by default, and H bonded to dummy atoms.
///
/// The removed hydrogens are recorded on the molecule
/// ([`Molecule::removed_hydrogens`]) for the CDK views, which keep them.
pub fn remove_hs(mol: &mut Molecule) {
    let n = mol.num_atoms();
    let mut remove = vec![false; n];
    let mut any = false;
    for a in 0..n {
        let atom = &mol.atoms[a];
        if atom.atomic_number != 1 || atom.isotope != 0 {
            continue;
        }
        let degree = mol.adjacency[a].len();
        if degree != 1 {
            continue;
        }
        let h_bond = mol.adjacency[a][0] as usize;
        let nbr = other_atom(mol, h_bond, a);
        let nbr_z = mol.atoms[nbr].atomic_number;
        if nbr_z == 0 || nbr_z == 1 {
            continue;
        }
        // keep H that defines double-bond stereo (`removeDefiningBondStereo = false`)
        if mol.bonds[h_bond].has_direction()
            && mol.adjacency[nbr].len() == 2
            && mol.adjacency[nbr].iter().any(|&bi| mol.bonds[bi as usize].bond_type == BondType::Double)
        {
            continue;
        }
        remove[a] = true;
        any = true;
    }
    if !any {
        return;
    }
    // Record the removed hydrogens (in input order) for the CDK views, which
    // keep them as atoms.
    let mut record = mol.take_removed_hydrogens();
    for a in 0..n {
        if remove[a] {
            let bi = mol.adjacency[a][0] as usize;
            let nbr = other_atom(mol, bi, a);
            mol.atoms[nbr].num_explicit_h = mol.atoms[nbr].num_explicit_h.saturating_add(1);
            let bond = &mol.bonds[bi];
            let up = match crate::algo::stereo::bond_direction(bond) {
                crate::algo::stereo::BondDir::None => None,
                dir => Some((dir == crate::algo::stereo::BondDir::EndUpRight) == (bond.source as usize == nbr)),
            };
            record.push(RemovedHydrogen { atom: nbr as AtomIdx, charge: mol.atoms[a].formal_charge, up });
        }
    }
    let new_idx = remove_marked_hydrogens(mol, &remove);
    for h in record.iter_mut() {
        h.atom = new_idx[h.atom as usize];
    }
    mol.set_removed_hydrogens(record);
}

/// RDKit `removeAllHs` followed by sanitization: every hydrogen atom is folded
/// into its neighbour's explicit H count. Returns `None` when the molecule has
/// no hydrogen atoms (the input can be used as is).
pub(crate) fn without_all_hydrogens(mol: &Molecule) -> Option<Molecule> {
    let remove: Vec<bool> = mol.atoms.iter().map(|a| a.atomic_number == 1).collect();
    if !remove.iter().any(|&r| r) {
        return None;
    }
    let mut out = mol.clone();
    for a in 0..out.num_atoms() {
        if remove[a] {
            for k in 0..out.adjacency[a].len() {
                let nbr = other_atom(&out, out.adjacency[a][k] as usize, a);
                if !remove[nbr] {
                    out.atoms[nbr].num_explicit_h = out.atoms[nbr].num_explicit_h.saturating_add(1);
                }
            }
        }
    }
    remove_marked_hydrogens(&mut out, &remove);
    sanitize(&mut out).ok()?;
    Some(out)
}

/// The molecule as CDK's `AtomContainerManipulator.removeHydrogens` leaves
/// it (`Hydrogens.isSafeToSuppress`): every neutral hydrogen atom without mass
/// number and implicit H, with exactly one neighbour that is neither H nor a
/// dummy atom, is folded into its neighbour. Unlike RDKit's `removeHs`, this
/// also removes hydrogens that define double-bond stereo (`[H]/N=C`). `None`
/// when nothing would be removed.
pub(crate) fn cdk_remove_hydrogens(mol: &Molecule) -> Option<Molecule> {
    let remove: Vec<bool> = (0..mol.num_atoms())
        .map(|a| {
            let atom = &mol.atoms[a];
            atom.atomic_number == 1
                && atom.formal_charge == 0
                && atom.isotope == 0
                && atom.total_h() == 0
                && mol.adjacency[a].len() == 1
                && !matches!(mol.atoms[other_atom(mol, mol.adjacency[a][0] as usize, a)].atomic_number, 0 | 1)
        })
        .collect();
    if !remove.iter().any(|&r| r) {
        return None;
    }
    let mut out = mol.clone();
    for a in 0..out.num_atoms() {
        if remove[a] {
            let nbr = other_atom(&out, out.adjacency[a][0] as usize, a);
            out.atoms[nbr].num_explicit_h = out.atoms[nbr].num_explicit_h.saturating_add(1);
        }
    }
    remove_marked_hydrogens(&mut out, &remove);
    sanitize(&mut out).ok()?;
    Some(out)
}

/// Deletes the marked atoms (keeping the order of the others) and returns the
/// new index of each old atom (`u16::MAX`: deleted). The removed-hydrogen
/// record is reset.
fn remove_marked_hydrogens(mol: &mut Molecule, remove: &[bool]) -> Vec<u16> {
    let n = mol.num_atoms();
    // Rebuild atoms/bonds/adjacency without the removed hydrogens, keeping order.
    let mut new_idx = vec![u16::MAX; n];
    let mut atoms = Vec::with_capacity(n);
    for a in 0..n {
        if !remove[a] {
            new_idx[a] = atoms.len() as u16;
            atoms.push(mol.atoms[a]);
        }
    }
    let mut bonds = Vec::with_capacity(mol.num_bonds());
    for b in &mol.bonds {
        if remove[b.source as usize] || remove[b.target as usize] {
            continue;
        }
        let mut nb = *b;
        nb.source = new_idx[b.source as usize];
        nb.target = new_idx[b.target as usize];
        bonds.push(nb);
    }
    let mut adjacency = vec![SmallVec::<[BondIdx; 4]>::new(); atoms.len()];
    for (bi, b) in bonds.iter().enumerate() {
        adjacency[b.source as usize].push(bi as BondIdx);
        adjacency[b.target as usize].push(bi as BondIdx);
    }
    let conformers = std::mem::take(&mut mol.conformers);
    *mol = Molecule::from_parts(atoms, bonds, adjacency);
    mol.conformers = conformers
        .into_iter()
        .map(|mut c| {
            let mut k = 0;
            c.positions.retain(|_| {
                let keep = k >= remove.len() || !remove[k];
                k += 1;
                keep
            });
            c
        })
        .collect();
    new_idx
}

// ---------------------------------------------------------------------------
// cleanUp / cleanUpOrganometallics (MolOps.cpp)
// ---------------------------------------------------------------------------

fn nitrogen_cleanup(mol: &mut Molecule, a: usize) {
    if mol.atoms[a].formal_charge != 0 {
        return;
    }
    let arom = mol.atoms[a].is_aromatic();
    mol.atoms[a].set_aromatic(false);
    let ev = calc_explicit_valence(mol, a, false).unwrap_or(0);
    mol.atoms[a].set_aromatic(arom);
    if ev != 5 {
        return;
    }
    for k in 0..mol.adjacency[a].len() {
        let bi = mol.adjacency[a][k] as usize;
        let nbr = other_atom(mol, bi, a);
        let nz = mol.atoms[nbr].atomic_number;
        let ncharge = mol.atoms[nbr].formal_charge;
        let bt = mol.bonds[bi].bond_type;
        if nz == 8 && ncharge == 0 && bt == BondType::Double {
            mol.bonds[bi].bond_type = BondType::Single;
            mol.atoms[a].formal_charge = 1;
            mol.atoms[nbr].formal_charge = -1;
            break;
        } else if nz == 7 && ncharge == 0 && bt == BondType::Triple {
            mol.bonds[bi].bond_type = BondType::Double;
            mol.atoms[a].formal_charge = 1;
            mol.atoms[nbr].formal_charge = -1;
            break;
        }
    }
}

fn phosphorus_cleanup(mol: &mut Molecule, a: usize) {
    if mol.atoms[a].formal_charge != 0 {
        return;
    }
    let ev = calc_explicit_valence(mol, a, false).unwrap_or(0);
    if ev != 5 || mol.adjacency[a].len() != 3 {
        return;
    }
    let mut dbl_to_o: Option<(usize, usize)> = None;
    let mut has_double_to_c_or_n = false;
    for &bi in &mol.adjacency[a] {
        let bi = bi as usize;
        let nbr = other_atom(mol, bi, a);
        let n = &mol.atoms[nbr];
        let bt = mol.bonds[bi].bond_type;
        if n.atomic_number == 8 && n.formal_charge == 0 && bt == BondType::Double {
            dbl_to_o = Some((bi, nbr));
        } else if (n.atomic_number == 6 || n.atomic_number == 7)
            && mol.adjacency[nbr].len() >= 2
            && bt == BondType::Double
        {
            has_double_to_c_or_n = true;
        }
    }
    if let (true, Some((bi, o))) = (has_double_to_c_or_n, dbl_to_o) {
        mol.atoms[o].formal_charge = -1;
        mol.bonds[bi].bond_type = BondType::Single;
        mol.atoms[a].formal_charge = 1;
    }
}

fn halogen_cleanup(mol: &mut Molecule, a: usize) {
    let ev = calc_explicit_valence(mol, a, false).unwrap_or(0);
    if mol.atoms[a].formal_charge != 0 || !(ev == 7 || ev == 5 || ev == 3) {
        return;
    }
    let all_o = mol.adjacency[a]
        .iter()
        .all(|&bi| mol.atoms[other_atom(mol, bi as usize, a)].atomic_number == 8);
    if !all_o {
        return;
    }
    let mut charge = 0;
    for k in 0..mol.adjacency[a].len() {
        let bi = mol.adjacency[a][k] as usize;
        if mol.bonds[bi].bond_type == BondType::Double {
            mol.bonds[bi].bond_type = BondType::Single;
            let o = other_atom(mol, bi, a);
            charge += 1;
            mol.atoms[o].formal_charge = -1;
        }
    }
    mol.atoms[a].formal_charge = charge;
}

fn clean_up(mol: &mut Molecule) {
    for a in 0..mol.num_atoms() {
        match mol.atoms[a].atomic_number {
            7 => nitrogen_cleanup(mol, a),
            15 => phosphorus_cleanup(mol, a),
            17 | 35 | 53 => halogen_cleanup(mol, a),
            _ => {}
        }
    }
}

fn is_hypervalent_non_metal(mol: &Molecule, a: usize) -> bool {
    let atom = &mol.atoms[a];
    if is_metal(atom.atomic_number) {
        return false;
    }
    let ev = calc_explicit_valence(mol, a, false).unwrap_or(0);
    let eff = atom.atomic_number as i32 - atom.formal_charge as i32;
    if eff <= 0 {
        return false;
    }
    let max_v = last_valence(eff.min(118) as usize);
    let total_degree = mol.adjacency[a].len() as i32 + atom.total_h() as i32;
    max_v > 0 && (ev > max_v || (ev == max_v && atom.is_aromatic() && total_degree == 4))
}

fn clean_up_organometallics(mol: &mut Molecule) {
    let no_dative = |z: u8| matches!(z, 1 | 2 | 9 | 10);
    let needs_fixing = (0..mol.num_atoms()).any(|a| {
        is_hypervalent_non_metal(mol, a)
            && !no_dative(mol.atoms[a].atomic_number)
            && mol.adjacency[a].iter().any(|&bi| {
                mol.bonds[bi as usize].bond_type == BondType::Single
                    && is_metal(mol.atoms[other_atom(mol, bi as usize, a)].atomic_number)
            })
    });
    if !needs_fixing {
        return;
    }
    // RDKit visits atoms in canonical-rank order; atom order is used here.
    for a in 0..mol.num_atoms() {
        if !is_hypervalent_non_metal(mol, a) || no_dative(mol.atoms[a].atomic_number) {
            continue;
        }
        let num_datives = |mol: &Molecule, m: usize| {
            mol.adjacency[m].iter().filter(|&&bi| mol.bonds[bi as usize].bond_type == BondType::Dative).count()
        };
        let mut metals: Vec<usize> = mol.adjacency[a]
            .iter()
            .filter(|&&bi| mol.bonds[bi as usize].bond_type == BondType::Single)
            .map(|&bi| other_atom(mol, bi as usize, a))
            .filter(|&m| is_metal(mol.atoms[m].atomic_number))
            .collect();
        if metals.is_empty() {
            continue;
        }
        metals.sort_by_key(|&m| num_datives(mol, m));
        let m = metals[0];
        if let Some(bi) = bond_between(mol, a, m) {
            let b = &mut mol.bonds[bi];
            b.bond_type = BondType::Dative;
            b.source = a as AtomIdx;
            b.target = m as AtomIdx;
        }
    }
}

// ---------------------------------------------------------------------------
// Rings
// ---------------------------------------------------------------------------

fn perceive_rings(mol: &mut Molecule) {
    let rings = sssr::symmetrized_sssr(mol);
    let mut info = RingInfo {
        atom_rings: Vec::with_capacity(rings.len()),
        bond_rings: Vec::with_capacity(rings.len()),
        atom_ring_counts: vec![0; mol.num_atoms()],
        bond_ring_counts: vec![0; mol.num_bonds()],
    };
    for r in &rings {
        let bonds = sssr::ring_atoms_to_bonds(mol, r);
        for &a in r {
            info.atom_ring_counts[a as usize] = info.atom_ring_counts[a as usize].saturating_add(1);
        }
        for &b in &bonds {
            info.bond_ring_counts[b as usize] = info.bond_ring_counts[b as usize].saturating_add(1);
        }
        info.atom_rings.push(r.iter().map(|&a| a as AtomIdx).collect());
        info.bond_rings.push(bonds.iter().map(|&b| b as BondIdx).collect());
    }
    for (a, atom) in mol.atoms.iter_mut().enumerate() {
        atom.set_in_ring(info.atom_ring_counts[a] > 0);
    }
    for (b, bond) in mol.bonds.iter_mut().enumerate() {
        bond.set_in_ring(info.bond_ring_counts[b] > 0);
    }
    mol.set_ring_info(info);
}

/// RDKit `RingUtils::makeRingNeighborMap`.
fn make_ring_neighbor_map(brings: &[Vec<usize>], max_size: usize, max_overlap: usize) -> BTreeMap<usize, Vec<usize>> {
    let mut map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let n = brings.len();
    for i in 0..n {
        map.entry(i).or_default();
        if max_size > 0 && brings[i].len() > max_size {
            continue;
        }
        for j in i + 1..n {
            if max_size > 0 && brings[j].len() > max_size {
                continue;
            }
            let inter = brings[i].iter().filter(|b| brings[j].contains(b)).count();
            if inter > 0 && (max_overlap == 0 || inter <= max_overlap) {
                map.entry(i).or_default().push(j);
                map.entry(j).or_default().push(i);
            }
        }
    }
    map
}

/// RDKit `RingUtils::pickFusedRings`.
fn pick_fused_rings(curr: usize, neigh: &BTreeMap<usize, Vec<usize>>, res: &mut Vec<usize>, done: &mut [bool]) {
    done[curr] = true;
    res.push(curr);
    if let Some(ns) = neigh.get(&curr) {
        for &n in ns {
            if !done[n] {
                pick_fused_rings(n, neigh, res, done);
            }
        }
    }
}

/// RDKit `RingUtils::checkFused`.
fn check_fused(rids: &[usize], neigh: &BTreeMap<usize, Vec<usize>>) -> bool {
    let n = neigh.len();
    let mut done = vec![false; n];
    for &rid in neigh.keys() {
        if !rids.contains(&rid) {
            done[rid] = true;
        }
    }
    let mut fused = Vec::new();
    pick_fused_rings(rids[0], neigh, &mut fused, &mut done);
    fused.len() == rids.len()
}

/// Union of ring atoms in order of first appearance (RDKit `Union`).
fn union_atoms(rings: &[&[AtomIdx]]) -> Vec<usize> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for r in rings {
        for &a in r.iter() {
            if seen.insert(a) {
                out.push(a as usize);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Kekulization (Kekulize.cpp)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn mark_dbond_cands(
    mol: &mut Molecule,
    cache: &ValenceCache,
    all_atms: &[usize],
    dbnd_cands: &mut [bool],
    done: &mut Vec<usize>,
) {
    let has_arom_or_dummy =
        all_atms.iter().any(|&a| mol.atoms[a].atomic_number == 0 || is_aromatic_atom(mol, a));
    if !has_arom_or_dummy {
        return;
    }
    let mut make_single: Vec<usize> = Vec::new();
    let mut in_all = vec![false; mol.num_atoms()];
    for &a in all_atms {
        in_all[a] = true;
        let z = mol.atoms[a].atomic_number as usize;
        if z != 0 && !is_aromatic_atom(mol, a) {
            done.push(a);
            continue;
        }
        let mut sbo = 0i32;
        let mut n_to_ignore = 0i32;
        for &bi in &mol.adjacency[a] {
            let bi = bi as usize;
            let b = &mol.bonds[bi];
            if b.is_aromatic() && matches!(b.bond_type, BondType::Single | BondType::Double | BondType::Aromatic) {
                sbo += 1;
                make_single.push(bi);
            } else {
                let c = valence_contrib(mol, bi, a).round() as i32;
                sbo += c;
                if c == 0 {
                    n_to_ignore += 1;
                }
            }
        }
        if z == 0 {
            // dummy atoms: always candidates (simplified; no permutation search)
            dbnd_cands[a] = true;
            continue;
        }
        let atom = &mol.atoms[a];
        sbo += total_num_hs(mol, cache, a);
        let mut dv = default_valence(z);
        let mut chrg = atom.formal_charge as i32;
        if is_early_atom(z) {
            chrg = -chrg;
        }
        if z == 6 && chrg > 0 {
            chrg = -chrg;
        }
        dv += chrg;
        let tbo = cache.total(a);
        let nrad = atom.num_radicals() as i32;
        let implicit_v = if atom.no_implicit_h() { 0 } else { cache.implicit[a] };
        let total_degree = mol.adjacency[a].len() as i32 + implicit_v - n_to_ignore;
        let val_list = valence_list(z);
        let mut vi = 1;
        while tbo > dv && vi < val_list.len() && val_list[vi] > 0 {
            dv = val_list[vi] as i32 + chrg;
            vi += 1;
        }
        if tbo == 5 && sbo == 4 && dv == 3 && total_degree == 3 && nrad == 0 && chrg == 0
            && total_num_hs(mol, cache, a) == 0
            && matches!(z, 7 | 15 | 33)
        {
            dv = 5;
        }
        if total_degree + nrad >= dv {
            continue;
        }
        if dv == sbo + 1 + nrad || (nrad == 0 && atom.no_implicit_h() && dv == sbo + 2) {
            dbnd_cands[a] = true;
        }
    }
    for bi in make_single {
        mol.bonds[bi].bond_type = BondType::Single;
    }
}

fn back_track(
    mol: &mut Molecule,
    last_opt: usize,
    done: &mut Vec<usize>,
    done_mask: &mut [bool],
    astack: &mut VecDeque<usize>,
    dbnd_cands: &mut [bool],
    dbnd_adds: &mut [bool],
) {
    let first = done.iter().position(|&x| x == last_opt).unwrap_or(done.len());
    let last = done.iter().rposition(|&x| x == last_opt).unwrap_or(0);
    for k in (last..done.len()).rev() {
        astack.push_front(done[k]);
    }
    let tdone: Vec<usize> = done[..first].to_vec();
    let mut in_tdone = vec![false; mol.num_atoms()];
    for &a in &tdone {
        in_tdone[a] = true;
    }
    for bi in 0..mol.num_bonds() {
        if dbnd_adds[bi] {
            let (a1, a2) = (mol.bonds[bi].source as usize, mol.bonds[bi].target as usize);
            if !in_tdone[a1] && !in_tdone[a2] {
                dbnd_adds[bi] = false;
                mol.bonds[bi].bond_type = BondType::Single;
                dbnd_cands[a1] = true;
                dbnd_cands[a2] = true;
            }
        }
    }
    for &a in done.iter() {
        done_mask[a] = false;
    }
    for &a in &tdone {
        done_mask[a] = true;
    }
    *done = tdone;
}

/// `kekulizeWorker`. With `ranks` (RDKit 2026.03 `Kekulize(canonical =
/// true)`), the start atoms are taken in rank order and the neighbours of
/// each atom are queued and tried in rank order; without, the atoms follow
/// the ring order and the neighbours are queued in reverse bond order.
fn kekulize_worker(
    mol: &mut Molecule,
    all_atms: &[usize],
    mut dbnd_cands: Vec<bool>,
    mut dbnd_adds: Vec<bool>,
    mut done: Vec<usize>,
    ranks: Option<&[u32]>,
) -> bool {
    let n = mol.num_atoms();
    let mut sorted_atms = all_atms.to_vec();
    if let Some(r) = ranks {
        sorted_atms.sort_by_key(|&a| (r[a], a));
    }
    let all_atms = &sorted_atms[..];
    let mut nbr_buf: Vec<(usize, usize)> = Vec::new();
    let mut in_all = vec![false; n];
    for &a in all_atms {
        in_all[a] = true;
    }
    let mut done_mask = vec![false; n];
    for &a in &done {
        done_mask[a] = true;
    }
    let mut astack: VecDeque<usize> = VecDeque::new();
    let mut options: BTreeMap<usize, VecDeque<usize>> = BTreeMap::new();
    let mut last_opt: Option<usize> = None;
    let mut local_added = vec![false; mol.num_bonds()];
    let mut btmoves: VecDeque<usize> = VecDeque::new();
    let mut num_bt = 0u32;

    while done.len() < all_atms.len() || !astack.is_empty() {
        let curr = if let Some(c) = astack.pop_front() {
            c
        } else {
            match all_atms.iter().copied().find(|&a| !done_mask[a]) {
                Some(c) => c,
                None => break,
            }
        };
        done.push(curr);
        done_mask[curr] = true;

        let c_cand = dbnd_cands[curr];
        let mut opts: VecDeque<usize>;
        if let Some(o) = options.get(&curr) {
            opts = o.clone();
        } else {
            opts = VecDeque::new();
            let mut lstack: VecDeque<usize> = VecDeque::new();
            nbr_buf.clear();
            for k in 0..mol.adjacency[curr].len() {
                let bi = mol.adjacency[curr][k] as usize;
                let nbr = other_atom(mol, bi, curr);
                if !done_mask[nbr] && in_all[nbr] {
                    nbr_buf.push((nbr, bi));
                }
            }
            if let Some(r) = ranks {
                nbr_buf.sort_by_key(|&(a, _)| (r[a], a));
            }
            for &(nbr, bi) in &nbr_buf {
                if !astack.contains(&nbr) {
                    if ranks.is_some() {
                        lstack.push_back(nbr);
                    } else {
                        lstack.push_front(nbr);
                    }
                }
                if c_cand
                    && dbnd_cands[nbr]
                    && (mol.bonds[bi].is_aromatic()
                        || mol.atoms[curr].atomic_number == 0
                        || mol.atoms[nbr].atomic_number == 0)
                {
                    opts.push_back(nbr);
                }
            }
            astack.extend(lstack);
        }
        if !c_cand {
            continue;
        }
        if let Some(ncnd) = opts.pop_front() {
            let bi = bond_between(mol, curr, ncnd).expect("neighbour bond");
            mol.bonds[bi].bond_type = BondType::Double;
            dbnd_cands[curr] = false;
            dbnd_cands[ncnd] = false;
            dbnd_adds[bi] = true;
            local_added[bi] = true;
            match options.entry(curr) {
                std::collections::btree_map::Entry::Occupied(mut e) => {
                    if opts.is_empty() {
                        options.remove(&curr);
                        btmoves.pop_back();
                        last_opt = btmoves.back().copied();
                    } else {
                        e.insert(opts);
                    }
                }
                std::collections::btree_map::Entry::Vacant(e) => if !opts.is_empty() {
                    last_opt = Some(curr);
                    btmoves.push_back(curr);
                    e.insert(opts);
                }
            }
        } else if mol.atoms[curr].atomic_number != 0 {
            match last_opt {
                Some(lo) if num_bt < MAX_BACKTRACKS => {
                    back_track(mol, lo, &mut done, &mut done_mask, &mut astack, &mut dbnd_cands, &mut dbnd_adds);
                    num_bt += 1;
                }
                _ => {
                    for (bi, &added) in local_added.iter().enumerate() {
                        if added {
                            mol.bonds[bi].bond_type = BondType::Single;
                        }
                    }
                    return false;
                }
            }
        }
    }
    true
}

fn kekulize_fused(
    mol: &mut Molecule,
    cache: &ValenceCache,
    frings: &[&[AtomIdx]],
    ranks: Option<&[u32]>,
) -> Result<()> {
    let all_atms = union_atoms(frings);
    let mut done = Vec::new();
    let mut dbnd_cands = vec![false; mol.num_atoms()];
    let dbnd_adds = vec![false; mol.num_bonds()];
    mark_dbond_cands(mol, cache, &all_atms, &mut dbnd_cands, &mut done);
    if !kekulize_worker(mol, &all_atms, dbnd_cands.clone(), dbnd_adds, done, ranks) {
        let bad: Vec<String> =
            (0..mol.num_atoms()).filter(|&a| dbnd_cands[a]).map(|a| a.to_string()).collect();
        return Err(FerrumError::ParseError(format!(
            "Can't kekulize mol. Unkekulized atoms: {}",
            bad.join(" ")
        )));
    }
    Ok(())
}

/// Port of `MolOps::Kekulize` (with `markAtomsBonds = true`).
fn kekulize(mol: &mut Molecule, cache: &mut ValenceCache) -> Result<()> {
    kekulize_ranked(mol, cache, None)
}

/// [`kekulize`] with optional canonical atom ranks (see [`kekulize_worker`]).
fn kekulize_ranked(mol: &mut Molecule, cache: &mut ValenceCache, ranks: Option<&[u32]>) -> Result<()> {
    let found_aromatic = mol.bonds.iter().any(|b| b.is_aromatic())
        || (0..mol.num_atoms()).any(|a| is_aromatic_atom(mol, a));
    if !found_aromatic {
        return Ok(());
    }
    let info = mol.ring_info().cloned().unwrap_or_default();
    let ring_ok: Vec<usize> = (0..info.atom_rings.len())
        .filter(|&r| info.atom_rings[r].iter().any(|&a| mol.atoms[a as usize].atomic_number != 0))
        .collect();
    let brings: Vec<Vec<usize>> =
        ring_ok.iter().map(|&r| info.bond_rings[r].iter().map(|&b| b as usize).collect()).collect();
    let neigh = make_ring_neighbor_map(&brings, 0, 0);
    let cnrs = ring_ok.len();
    let mut fus_done = vec![false; cnrs];
    let mut curr = 0;
    while curr < cnrs {
        let mut fused = Vec::new();
        pick_fused_rings(curr, &neigh, &mut fused, &mut fus_done);
        let frings: Vec<&[AtomIdx]> = fused.iter().map(|&k| info.atom_rings[ring_ok[k]].as_slice()).collect();
        kekulize_fused(mol, cache, &frings, ranks)?;
        match (0..cnrs).find(|&r| !fus_done[r]) {
            Some(r) => curr = r,
            None => break,
        }
    }
    for b in mol.bonds.iter_mut() {
        b.set_aromatic(false);
    }
    for a in 0..mol.num_atoms() {
        if !mol.atoms[a].is_aromatic() {
            continue;
        }
        if info.num_atom_rings(a) == 0 {
            return Err(FerrumError::ParseError(format!("non-ring atom {} marked aromatic", a)));
        }
        mol.atoms[a].set_aromatic(false);
        let atom = &mol.atoms[a];
        if (atom.atomic_number == 7 || atom.atomic_number == 15)
            && atom.formal_charge == 0
            && atom.num_explicit_h == 1
        {
            let before = cache.total(a);
            mol.atoms[a].set_no_implicit_h(false);
            mol.atoms[a].num_explicit_h = 0;
            update_atom_cache(mol, cache, a, false)?;
            if cache.total(a) != before {
                return Err(FerrumError::ParseError(format!(
                    "Kekulization somehow screwed up valence on {}",
                    a
                )));
            }
        }
    }
    // Any remaining aromatic bond outside a ring system cannot be kekulized.
    if mol.bonds.iter().any(|b| b.bond_type == BondType::Aromatic) {
        return Err(FerrumError::ParseError("Can't kekulize mol: aromatic bond outside ring".into()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Radicals (MolOps.cpp `assignRadicals`)
// ---------------------------------------------------------------------------

fn assign_radicals(mol: &mut Molecule) {
    for a in 0..mol.num_atoms() {
        let atom = &mol.atoms[a];
        if !atom.no_implicit_h() || atom.atomic_number == 0 {
            continue;
        }
        let z = atom.atomic_number as usize;
        let valens = valence_list(z);
        let chg = atom.formal_charge as i32;
        let nouter = n_outer(z);
        let nrad = if valens.len() != 1 || valens[0] != -1 {
            let accum: f64 = mol.adjacency[a]
                .iter()
                .map(|&bi| valence_contrib(mol, bi as usize, a))
                .sum::<f64>()
                + atom.num_explicit_h as f64;
            let total_valence = (accum + 0.1) as i32;
            let base_count = if z == 1 || z == 2 { 2 } else { 8 };
            let mut nrad = base_count - nouter - total_valence + chg;
            if nrad < 0 {
                nrad = 0;
                if valens.len() > 1 {
                    for &v in valens {
                        if v as i32 - total_valence + chg >= 0 {
                            nrad = v as i32 - total_valence + chg;
                            break;
                        }
                    }
                }
            }
            let nrad2 = nouter - total_valence - chg;
            if nrad2 >= 0 {
                nrad = nrad.min(nrad2);
            }
            nrad
        } else if !mol.adjacency[a].is_empty() {
            0
        } else {
            (nouter - chg).max(0) % 2
        };
        mol.atoms[a].set_num_radicals(nrad.clamp(0, 7) as u8);
    }
}

// ---------------------------------------------------------------------------
// Aromaticity (Aromaticity.cpp, AROMATICITY_RDKIT)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Edon {
    Vacant,
    One,
    Two,
    Any,
    No,
}

fn incident_non_cyclic_multiple_bond(mol: &Molecule, ri: &RingInfo, a: usize) -> Option<usize> {
    for &bi in &mol.adjacency[a] {
        let bi = bi as usize;
        if ri.num_bond_rings(bi) == 0 && valence_contrib(mol, bi, a) >= 2.0 {
            return Some(other_atom(mol, bi, a));
        }
    }
    None
}

fn incident_cyclic_multiple_bond(mol: &Molecule, ri: &RingInfo, a: usize) -> bool {
    mol.adjacency[a]
        .iter()
        .any(|&bi| ri.num_bond_rings(bi as usize) > 0 && valence_contrib(mol, bi as usize, a) >= 2.0)
}

fn incident_multiple_bond(mol: &Molecule, cache: &ValenceCache, a: usize) -> bool {
    let mut deg = mol.adjacency[a].len() as i32 + mol.atoms[a].num_explicit_h as i32;
    for &bi in &mol.adjacency[a] {
        if valence_contrib(mol, bi as usize, a).round() as i32 == 0 {
            deg -= 1;
        }
    }
    cache.explicit[a] != deg
}

/// Port of `MolOps::countAtomElec`.
fn count_atom_elec(mol: &Molecule, cache: &ValenceCache, a: usize) -> i32 {
    let atom = &mol.atoms[a];
    let z = atom.atomic_number as usize;
    let dv = default_valence(z);
    if dv <= 1 {
        return -1;
    }
    let mut degree = mol.adjacency[a].len() as i32 + total_num_hs(mol, cache, a);
    for &bi in &mol.adjacency[a] {
        if valence_contrib(mol, bi as usize, a) == 0.0 {
            degree -= 1;
        }
    }
    if degree > 3 {
        return -1;
    }
    let nlp = (n_outer(z) - dv - atom.formal_charge as i32).max(0);
    let nrad = atom.num_radicals() as i32;
    let mut res = (dv - degree) + nlp - nrad;
    if res > 1 {
        let n_unsat = cache.explicit[a] - mol.adjacency[a].len() as i32;
        if n_unsat > 1 {
            res = 1;
        }
    }
    res
}

fn donor_type(mol: &Molecule, cache: &ValenceCache, ri: &RingInfo, a: usize) -> Edon {
    let z = mol.atoms[a].atomic_number as usize;
    if z == 0 {
        return if incident_cyclic_multiple_bond(mol, ri, a) { Edon::One } else { Edon::Any };
    }
    let mut nelec = count_atom_elec(mol, cache, a);
    if nelec < 0 {
        Edon::No
    } else if nelec == 0 {
        if incident_non_cyclic_multiple_bond(mol, ri, a).is_some() {
            Edon::Vacant
        } else if incident_cyclic_multiple_bond(mol, ri, a) {
            Edon::One
        } else {
            Edon::No
        }
    } else if nelec == 1 {
        if let Some(who) = incident_non_cyclic_multiple_bond(mol, ri, a) {
            if more_electronegative(mol.atoms[who].atomic_number as usize, z) {
                Edon::Vacant
            } else {
                Edon::One
            }
        } else if incident_multiple_bond(mol, cache, a) {
            Edon::One
        } else if mol.atoms[a].formal_charge == 1 {
            Edon::Vacant
        } else {
            Edon::No
        }
    } else {
        if let Some(who) = incident_non_cyclic_multiple_bond(mol, ri, a)
            && more_electronegative(mol.atoms[who].atomic_number as usize, z) {
                nelec -= 1;
            }
        if nelec % 2 == 1 { Edon::One } else { Edon::Two }
    }
}

fn is_atom_cand_for_arom(mol: &Molecule, cache: &ValenceCache, a: usize, edon: Edon) -> bool {
    let atom = &mol.atoms[a];
    let z = atom.atomic_number as usize;
    if z > 18 && z != 34 && z != 52 {
        return false;
    }
    if edon == Edon::No {
        return false;
    }
    let def_val = default_valence(z);
    let eff = (z as i32 - atom.formal_charge as i32).clamp(0, 118) as usize;
    if def_val > 0 && cache.total(a) > default_valence(eff) {
        return false;
    }
    if atom.num_radicals() > 0 && (z != 6 || atom.formal_charge != 0) {
        return false;
    }
    let n_unsat = cache.explicit[a] - mol.adjacency[a].len() as i32;
    if n_unsat > 1 {
        let n_mult = mol.adjacency[a]
            .iter()
            .filter(|&&bi| matches!(mol.bonds[bi as usize].bond_type, BondType::Double | BondType::Triple))
            .count();
        if n_mult > 1 {
            return false;
        }
    }
    true
}

fn apply_huckel(ring: &[usize], edon: &[Edon]) -> bool {
    let (mut rlw, mut rup) = (0i32, 0i32);
    let mut n_any = 0;
    for &a in ring {
        let (lo, hi) = match edon[a] {
            Edon::Any => {
                n_any += 1;
                if n_any > 1 {
                    return false;
                }
                (1, 2)
            }
            Edon::One => (1, 1),
            Edon::Two => (2, 2),
            Edon::No | Edon::Vacant => (0, 0),
        };
        rlw += lo;
        rup += hi;
    }
    if rup >= 6 {
        (rlw..=rup).any(|rie| (rie - 2) % 4 == 0)
    } else {
        rup == 2
    }
}

fn mark_atoms_bonds_arom(
    mol: &mut Molecule,
    srings: &[&[AtomIdx]],
    brings: &[Vec<usize>],
    ring_ids: &[usize],
    done_bonds: &mut BTreeSet<usize>,
) {
    for &ri in ring_ids {
        for &a in srings[ri] {
            mol.atoms[a as usize].set_aromatic(true);
        }
    }
    let mut cnt: BTreeMap<usize, u32> = BTreeMap::new();
    for &ri in ring_ids {
        for &b in &brings[ri] {
            *cnt.entry(b).or_default() += 1;
        }
    }
    for (b, c) in cnt {
        if c == 1 {
            let bond = &mut mol.bonds[b];
            bond.set_aromatic(true);
            if matches!(bond.bond_type, BondType::Single | BondType::Double) {
                bond.bond_type = BondType::Aromatic;
            }
            done_bonds.insert(b);
        }
    }
}

/// Next combination of `comb.len()` indices out of `tot` (RDKit `nextCombination`).
fn next_combination(comb: &mut [usize], tot: usize) -> Option<usize> {
    let nelem = comb.len();
    let mut celem = nelem as isize - 1;
    while comb[celem as usize] == tot - nelem + celem as usize {
        celem -= 1;
        if celem < 0 {
            return None;
        }
    }
    let c = celem as usize;
    comb[c] += 1;
    for i in c + 1..nelem {
        comb[i] = comb[i - 1] + 1;
    }
    Some(c)
}

#[allow(clippy::too_many_arguments)]
fn apply_huckel_to_fused(
    mol: &mut Molecule,
    srings: &[&[AtomIdx]],
    brings: &[Vec<usize>],
    fused: &[usize],
    edon: &[Edon],
    ring_neighs: &BTreeMap<usize, Vec<usize>>,
    max_num_fused: usize,
) {
    let nrings = fused.len();
    let n_ring_bonds = {
        let mut set = BTreeSet::new();
        for &r in fused {
            set.extend(brings[r].iter().copied());
        }
        set.len()
    };
    let mut done_bonds: BTreeSet<usize> = BTreeSet::new();
    let mut cur_size = 0usize;
    let mut comb: Vec<usize> = Vec::new();
    let mut fresh = true;
    let mut ats_in_system = vec![0u32; mol.num_atoms()];
    loop {
        if fresh {
            if cur_size == 2 && nrings > 300 {
                break;
            }
            cur_size += 1;
            if cur_size > nrings.min(max_num_fused) || done_bonds.len() >= n_ring_bonds {
                break;
            }
            comb = (0..cur_size).collect();
            fresh = false;
        } else if next_combination(&mut comb, nrings).is_none() {
            fresh = true;
            continue;
        }
        let cur_rs: Vec<usize> = comb.iter().map(|&i| fused[i]).collect();
        if !ring_neighs.is_empty() && !check_fused(&cur_rs, ring_neighs) {
            continue;
        }
        for &r in &cur_rs {
            for &a in srings[r] {
                ats_in_system[a as usize] += 1;
            }
        }
        let mut unon = Vec::new();
        for (a, c) in ats_in_system.iter_mut().enumerate() {
            if *c == 1 || *c == 2 {
                unon.push(a);
            }
            *c = 0;
        }
        if apply_huckel(&unon, edon) {
            mark_atoms_bonds_arom(mol, srings, brings, &cur_rs, &mut done_bonds);
        }
    }
}

/// Port of `MolOps::setAromaticity` with the RDKit default model.
fn set_aromaticity(mol: &mut Molecule, cache: &ValenceCache) {
    let info = match mol.ring_info() {
        Some(i) if i.num_rings() > 0 => i.clone(),
        _ => return,
    };
    let n = mol.num_atoms();
    let mut edon = vec![Edon::No; n];
    let mut acands = vec![false; n];
    let mut aseen = vec![false; n];
    let mut c_rings: Vec<usize> = Vec::new();
    for (ri, ring) in info.atom_rings.iter().enumerate() {
        let mut all_arom = true;
        let mut all_dummy = true;
        for &a in ring.iter() {
            let a = a as usize;
            if all_dummy && mol.atoms[a].atomic_number != 0 {
                all_dummy = false;
            }
            if aseen[a] {
                if !acands[a] {
                    all_arom = false;
                }
                continue;
            }
            aseen[a] = true;
            edon[a] = donor_type(mol, cache, &info, a);
            acands[a] = is_atom_cand_for_arom(mol, cache, a, edon[a]);
            if !acands[a] {
                all_arom = false;
            }
        }
        if all_arom && !all_dummy {
            c_rings.push(ri);
        }
    }
    let srings: Vec<&[AtomIdx]> = c_rings.iter().map(|&r| info.atom_rings[r].as_slice()).collect();
    let brings: Vec<Vec<usize>> =
        c_rings.iter().map(|&r| info.bond_rings[r].iter().map(|&b| b as usize).collect()).collect();
    let neigh = make_ring_neighbor_map(&brings, MAX_FUSED_AROMATIC_RING_SIZE, 1);
    let cnrs = srings.len();
    let mut fus_done = vec![false; cnrs];
    let mut curr = 0;
    while curr < cnrs {
        let mut fused = Vec::new();
        pick_fused_rings(curr, &neigh, &mut fused, &mut fus_done);
        apply_huckel_to_fused(mol, &srings, &brings, &fused, &edon, &neigh, MAX_NUM_FUSED_RINGS);
        match (0..cnrs).find(|&r| !fus_done[r]) {
            Some(r) => curr = r,
            None => break,
        }
    }
}

// ---------------------------------------------------------------------------
// Conjugation & hybridization (ConjugHybrid.cpp)
// ---------------------------------------------------------------------------

fn is_atom_conjug_cand(mol: &Molecule, cache: &ValenceCache, a: usize) -> bool {
    let atom = &mol.atoms[a];
    let z = atom.atomic_number as usize;
    let vals = valence_list(z);
    if atom.formal_charge == 0 && vals[0] >= 0 && cache.total(a) > vals[0] as i32 {
        return false;
    }
    let nouter = n_outer(z);
    let total_degree = mol.adjacency[a].len() as i32 + total_num_hs(mol, cache, a);
    (z <= 10 || (nouter != 5 && nouter != 6) || (nouter == 6 && total_degree < 2))
        && count_atom_elec(mol, cache, a) > 0
}

fn set_conjugation(mol: &mut Molecule, cache: &ValenceCache) {
    for b in mol.bonds.iter_mut() {
        let arom = b.is_aromatic();
        b.set_conjugated(arom);
    }
    let cand: Vec<bool> = (0..mol.num_atoms()).map(|a| is_atom_conjug_cand(mol, cache, a)).collect();
    let sbo_of = |mol: &Molecule, a: usize| mol.adjacency[a].len() as i32 + total_num_hs(mol, cache, a);
    for a in 0..mol.num_atoms() {
        if !cand[a] {
            continue;
        }
        let sbo = sbo_of(mol, a);
        if !(2..=3).contains(&sbo) {
            continue;
        }
        let bonds: SmallVec<[usize; 6]> = mol.adjacency[a].iter().map(|&b| b as usize).collect();
        for &b1 in &bonds {
            if valence_contrib(mol, b1, a) < 1.5 || !cand[other_atom(mol, b1, a)] {
                continue;
            }
            for &b2 in &bonds {
                if b1 == b2 {
                    continue;
                }
                let at2 = other_atom(mol, b2, a);
                if sbo_of(mol, at2) > 3 {
                    continue;
                }
                if cand[at2] {
                    mol.bonds[b1].set_conjugated(true);
                    mol.bonds[b2].set_conjugated(true);
                }
            }
        }
    }
}

fn num_bonds_plus_lone_pairs(mol: &Molecule, cache: &ValenceCache, a: usize) -> i32 {
    let atom = &mol.atoms[a];
    let mut deg = mol.adjacency[a].len() as i32 + total_num_hs(mol, cache, a);
    for &bi in &mol.adjacency[a] {
        let b = &mol.bonds[bi as usize];
        if b.bond_type == BondType::Dative && b.target as usize != a {
            deg -= 1;
        }
    }
    let z = atom.atomic_number as usize;
    if z <= 1 {
        return deg;
    }
    let nouter = n_outer(z);
    let total_valence = cache.total(a);
    let chg = atom.formal_charge as i32;
    let n_free = nouter - (total_valence + chg);
    if total_valence + nouter - chg < 8 {
        let nrad = atom.num_radicals() as i32;
        deg + (n_free - nrad) / 2 + nrad
    } else {
        deg + n_free / 2
    }
}

fn set_hybridization(mol: &mut Molecule, cache: &ValenceCache) {
    for a in 0..mol.num_atoms() {
        let z = mol.atoms[a].atomic_number;
        let hyb = if z == 0 {
            Hybridization::Unspecified
        } else {
            let total_degree = mol.adjacency[a].len() as i32 + total_num_hs(mol, cache, a);
            let norbs = if z < 89 { num_bonds_plus_lone_pairs(mol, cache, a) } else { total_degree };
            match norbs {
                0 | 1 => Hybridization::S,
                2 => Hybridization::SP,
                3 => Hybridization::SP2,
                4 => {
                    let has_conj = mol.adjacency[a].iter().any(|&bi| mol.bonds[bi as usize].is_conjugated());
                    if total_degree > 3 || !has_conj { Hybridization::SP3 } else { Hybridization::SP2 }
                }
                5 => Hybridization::SP3D,
                6 => Hybridization::SP3D2,
                _ => Hybridization::Unspecified,
            }
        };
        mol.atoms[a].hybridization = hyb;
    }
}

/// Kekulé bond types RDKit's `Chem.Kekulize` assigns to a sanitized
/// (aromatic) molecule, one per bond; `None` if the molecule cannot be
/// kekulized. Bonds outside aromatic systems keep their type. `ranks` are the
/// canonical atom ranks of RDKit 2026.03 `Kekulize(canonical = true)` (one per
/// atom of `mol`), `None` for the atom-order kekulization of sanitization.
pub(crate) fn kekule_bond_types(mol: &Molecule, ranks: Option<&[u32]>) -> Option<Vec<BondType>> {
    if !mol.bonds.iter().any(|b| b.is_aromatic() || b.bond_type == BondType::Aromatic) {
        return Some(mol.bonds.iter().map(|b| b.bond_type).collect());
    }
    let mut work = mol.clone();
    let mut cache = update_property_cache(&mut work, false).ok()?;
    kekulize_ranked(&mut work, &mut cache, ranks).ok()?;
    Some(work.bonds.iter().map(|b| b.bond_type).collect())
}

// ---------------------------------------------------------------------------
// adjustHs + driver
// ---------------------------------------------------------------------------

fn adjust_hs(mol: &mut Molecule, cache: &ValenceCache) -> Result<()> {
    for a in 0..mol.num_atoms() {
        let orig_implicit = if mol.atoms[a].no_implicit_h() { 0 } else { cache.implicit[a] };
        let ev = calc_explicit_valence(mol, a, false)?;
        let new_implicit = calc_implicit_valence(mol, a, ev, false)?;
        if new_implicit < orig_implicit {
            let atom = &mut mol.atoms[a];
            atom.num_explicit_h = atom.num_explicit_h.saturating_add((orig_implicit - new_implicit) as u8);
        }
    }
    Ok(())
}

/// Runs the full RDKit-compatible sanitization on a freshly parsed molecule.
///
/// Returns an error for molecules RDKit would reject (valence violations,
/// unkekulizable aromatic systems).
pub fn sanitize(mol: &mut Molecule) -> Result<()> {
    mol.invalidate_cached_properties();
    remove_hs(mol);
    for b in mol.bonds.iter_mut() {
        if b.bond_type == BondType::Aromatic {
            b.set_aromatic(true);
        }
    }
    clean_up(mol);
    clean_up_organometallics(mol);
    let mut cache = update_property_cache(mol, true)?;
    perceive_rings(mol);
    kekulize(mol, &mut cache)?;
    assign_radicals(mol);
    // Radicals change the implicit valence of bracket atoms only (no implicit H),
    // so the cache stays valid.
    set_aromaticity(mol, &cache);
    set_conjugation(mol, &cache);
    set_hybridization(mol, &cache);
    adjust_hs(mol, &cache)?;
    update_property_cache(mol, true)?;
    Ok(())
}
