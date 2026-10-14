//! A molecule prepared the way CDK 2.13's `SmartsPattern.prepare` leaves it,
//! for matching CDK SMARTS keys with FerrumChem's substructure engine.
//!
//! CDK fingerprinters that match SMARTS keys (`MACCSFingerprinter`,
//! `EStateFingerprinter`, ...) run on the reference preparation (CDK SMILES
//! parser + atom typing + `CDKHydrogenAdder`) and call
//! `SmartsPattern.prepare`, which re-applies `Aromaticity.Model.Daylight`
//! (clearing all earlier flags). The target built here carries
//!
//! - CDK's Kekulé bond orders (Beam localisation, [`CdkMol`]) for every bond
//!   that is not aromatic in that model, and aromatic bond type for the others,
//! - CDK's formal charges: RDKit's `cleanUp` of hypervalent N, P and halogen
//!   oxides is reverted (CDK keeps `N(=O)=O` and `[O-][Cl+3]([O-])([O-])O`
//!   as written),
//! - CDK's implicit hydrogen counts (stored as explicit H counts),
//! - the aromatic atom flags of the Daylight model,
//! - explicit hydrogen atoms: CDK keeps `[H]` atoms of the SMILES as atoms
//!   (they match `*` and count for `D`), FerrumChem's parser folds them into
//!   the neighbour's explicit H count like RDKit. They are restored for atoms
//!   outside brackets (whose explicit H count comes from folded hydrogens,
//!   except the single H of ring N/P atoms, which kekulization stores there);
//!   for bracket atoms (`[H][C@]...`) the folded hydrogens cannot be told
//!   apart from `[C@H]` and stay implicit.
//!
//! With that, FerrumChem's SMARTS semantics coincide with CDK's `Expr`
//! semantics for the primitives used by CDK's key sets: `H<n>` is the total H
//! count including hydrogen neighbours, `D` the explicit degree, `-`/`=`/`#`
//! aliphatic orders, an implicit bond single-or-aromatic, `R`/`@`/`x` ring
//! membership from the ring flags, upper-case symbols aliphatic.

use super::cdk_daylight::{CdkMol, smarts_daylight_aromaticity};
use super::cdk_perception::CdkView;
use crate::io::smarts::{AtomPrimitive, QueryGraph, parse_smarts};
use crate::mol::atom::Atom;
use crate::mol::bond::Bond;
use crate::mol::molecule::Molecule;
use crate::types::{AtomIdx, BondIdx, BondType};

/// The CDK-prepared molecule (see the module documentation).
pub(crate) struct CdkSmartsTarget {
    pub mol: Molecule,
}

impl CdkSmartsTarget {
    pub(crate) fn new(mol: &Molecule) -> Self {
        let mut m = CdkMol::new(mol);
        revert_np_cleanup(mol, &mut m);
        radical_hydrogens(mol, &mut m);
        let (arom, arom_bond) = smarts_daylight_aromaticity(&m);
        let mut atoms = mol.atoms.clone();
        for (a, atom) in atoms.iter_mut().enumerate() {
            atom.formal_charge = m.charge[a];
            atom.num_explicit_h = m.hcount[a];
            atom.num_implicit_h = 0;
            atom.set_aromatic(arom[a]);
        }
        let mut bonds = mol.bonds.clone();
        let mut adjacency = mol.adjacency.clone();
        for (bi, b) in bonds.iter_mut().enumerate() {
            b.bond_type = if arom_bond[bi] {
                BondType::Aromatic
            } else {
                match m.order[bi] {
                    2 => BondType::Double,
                    3 => BondType::Triple,
                    _ => BondType::Single,
                }
            };
            b.set_aromatic(arom_bond[bi]);
        }
        // restore folded hydrogen atoms of atoms outside brackets
        for a in 0..mol.num_atoms() {
            let atom = &mol.atoms[a];
            // kekulization leaves the H of aromatic `[nH]`/`[pH]` as an explicit
            // count on a non-bracket atom: not a folded hydrogen atom
            let ring_nh = matches!(atom.atomic_number, 7 | 15)
                && atom.is_in_ring()
                && atom.formal_charge == 0
                && atom.num_explicit_h == 1;
            if atom.no_implicit_h() || atom.atomic_number <= 1 || ring_nh {
                continue;
            }
            for _ in 0..atom.num_explicit_h {
                let h = atoms.len() as AtomIdx;
                let bi = bonds.len() as BondIdx;
                atoms.push(Atom::new(1));
                atoms[h as usize].set_no_implicit_h(true);
                atoms[a].num_explicit_h -= 1;
                bonds.push(Bond::new(a as AtomIdx, h, BondType::Single));
                adjacency[a].push(bi);
                adjacency.push(smallvec::smallvec![bi]);
            }
        }
        Self {
            mol: Molecule::from_parts(atoms, bonds, adjacency),
        }
    }

    /// Runs `f` with the target of `mol`, cached in the calculation context so
    /// that every CDK SMARTS fingerprint of a molecule shares one preparation.
    pub(crate) fn with_cached<R>(
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
        f: impl FnOnce(&CdkSmartsTarget) -> R,
    ) -> R {
        let key = crate::desc::context::molecule_cache_key(mol);
        let target = match ctx.cdk_smarts_target.take() {
            Some((k, t)) if k == key => t,
            _ => Box::new(Self::new(mol)),
        };
        let r = f(&target);
        ctx.cdk_smarts_target = Some((key, target));
        r
    }
}

/// Hydrogen counts of radical atoms. CDK's SMILES parser keeps no radicals;
/// `CDKHydrogenAdder` fills the valence of the matched atom type with
/// hydrogens, and leaves the count alone when the type is unknown (`X`).
/// [`CdkMol`] fills every radical of B, C, N, O, Si, P, S, As and Se. Here N,
/// O, P, S and As follow [`CdkView`]'s atom typing (untyped atoms keep their
/// count: `C[S]=O` is `X` in CDK), silicon keeps its count (`C[Si](C)C` is
/// `Si.3`, `CC[Si]O...` is `Si.2`, both without H) and germanium is filled
/// (`C[Ge]C` gets two H).
fn radical_hydrogens(mol: &Molecule, m: &mut CdkMol) {
    if mol.atoms.iter().all(|a| a.num_radicals() == 0) {
        return;
    }
    let view = CdkView::new(mol);
    for (a, atom) in mol.atoms.iter().enumerate() {
        let rad = atom.num_radicals();
        if rad == 0 {
            continue;
        }
        m.hcount[a] = match atom.atomic_number {
            7 | 8 | 15 | 16 | 33 => view.types[a].map_or(atom.total_h(), |_| view.h[a]),
            5 | 6 | 32 | 34 => atom.total_h() + rad,
            _ => atom.total_h(),
        };
    }
}

/// Reverts RDKit's `cleanUp` of hypervalent N and P oxides (`N(=O)=O` ->
/// `[N+](=O)[O-]`, `N=N#N` -> `N=[N+]=[N-]`). Atoms from the SMILES organic
/// subset cannot carry a charge, so charged non-bracket atoms identify clean-up
/// results (as in `cdk_perception`). [`CdkMol`] restores halogen oxides
/// unconditionally; oxygens written as `[O-]` (`[O-][Cl+3]([O-])([O-])O`) are
/// kept charged here, as CDK reads them.
fn revert_np_cleanup(mol: &Molecule, m: &mut CdkMol) {
    for a in 0..mol.num_atoms() {
        let atom = &mol.atoms[a];
        let z = atom.atomic_number;
        if matches!(z, 17 | 35 | 53) && m.charge[a] != atom.formal_charge {
            for &bi in &mol.adjacency[a] {
                let bi = bi as usize;
                let b = &mol.bonds[bi];
                let o = if b.source as usize == a { b.target } else { b.source } as usize;
                if mol.atoms[o].no_implicit_h() && m.charge[o] != mol.atoms[o].formal_charge && m.order[bi] == 2 {
                    m.order[bi] = 1;
                    m.charge[o] = mol.atoms[o].formal_charge;
                    m.charge[a] += 1;
                }
            }
            continue;
        }
        if m.charge[a] <= 0 || atom.no_implicit_h() || !matches!(z, 7 | 15) {
            continue;
        }
        for &bi in &mol.adjacency[a] {
            let bi = bi as usize;
            let b = &mol.bonds[bi];
            let o = if b.source as usize == a { b.target } else { b.source } as usize;
            let nb = &mol.atoms[o];
            if m.charge[o] != -1 || nb.no_implicit_h() {
                continue;
            }
            if nb.atomic_number == 8 && m.order[bi] == 1 {
                m.order[bi] = 2;
            } else if z == 7 && nb.atomic_number == 7 && m.order[bi] == 2 {
                m.order[bi] = 3;
            } else {
                continue;
            }
            m.charge[o] = 0;
            m.charge[a] -= 1;
            break;
        }
    }
}

/// Parses a SMARTS pattern with CDK 2.13 `Smarts` semantics for the element
/// symbols FerrumChem's (RDKit's) parser reads differently: CDK treats the
/// bracket symbols `Si`, `Ge`, `As`, `Se`, `Sb` and `Te` as aliphatic
/// (`ALIPHATIC_ELEMENT`), RDKit as any aromaticity. An element is only
/// rewritten when the pattern does not also use its `#n` form.
pub(crate) fn parse_cdk_smarts(smarts: &str) -> QueryGraph {
    const CDK_ALIPHATIC: [u8; 6] = [14, 32, 33, 34, 51, 52];
    fn fix(p: &mut AtomPrimitive, zs: &[u8]) {
        match p {
            AtomPrimitive::Element(z) if zs.contains(z) => *p = AtomPrimitive::Aliphatic(*z),
            AtomPrimitive::Not(inner) => fix(inner, zs),
            AtomPrimitive::Or(v) | AtomPrimitive::And(v) => v.iter_mut().for_each(|x| fix(x, zs)),
            _ => {}
        }
    }
    let zs: Vec<u8> = CDK_ALIPHATIC
        .into_iter()
        .filter(|z| !smarts.contains(&format!("#{z}")))
        .collect();
    let mut q = parse_smarts(smarts).unwrap_or_else(|e| panic!("CDK SMARTS {smarts}: {e:?}"));
    for atom in &mut q.atoms {
        atom.primitives.iter_mut().for_each(|p| fix(p, &zs));
    }
    q
}
