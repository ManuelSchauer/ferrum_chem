//! PubChem 881-bit structural key fingerprint — a port of CDK 2.13's
//! `org.openscience.cdk.fingerprint.PubchemFingerprinter`.
//!
//! Bit `i` of the output is CDK's bit `i`:
//!
//! | bits     | section                                                         |
//! |----------|-----------------------------------------------------------------|
//! | 0-114    | element counts (all hydrogens counted)                          |
//! | 115-262  | ESSSR ring set: ring sizes, saturation, hetero atoms, aromaticity|
//! | 263-880  | SMARTS patterns (atom pairs, neighbours, sections 4-7)          |
//!
//! CDK computes the fingerprint on a molecule with explicit hydrogens and
//! perceives aromaticity itself (`SmartsPattern.prepare`: CDK's Daylight model
//! over all cycles), so the CDK view of the molecule is rebuilt locally in
//! [`super::pubchem_cdk`] (Kekulé bond orders, CDK aromaticity, ESSSR rings)
//! without touching FerrumChem's shared RDKit perception. The SMARTS keys are
//! matched with CDK semantics by [`super::pubchem_smarts`].
//!
//! Reference: ftp://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt

use super::pubchem_cdk::{CdkMol, ORD_DOUBLE, ORD_SINGLE, ORD_TRIPLE, esssr};
use super::pubchem_keys::{ELEMENT_BITS, RING_BITS, RingKind, SMARTS_BITS};
use super::pubchem_smarts::{
    BC_AROM, BC_DOUBLE, BC_OTHER, BC_SINGLE, BC_TRIPLE, FEAT_AROM_ATOM, FEAT_AROM_BOND,
    FEAT_DOUBLE, FEAT_TRIPLE, Pattern, Target,
};
use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use bitvec::prelude::*;
use std::sync::OnceLock;

/// Number of bits.
pub const PUBCHEM_FP_SIZE: usize = 881;
const RING_BIT_OFFSET: usize = 115;
const SMARTS_BIT_OFFSET: usize = 263;

fn patterns() -> &'static [Pattern] {
    static PATTERNS: OnceLock<Vec<Pattern>> = OnceLock::new();
    PATTERNS.get_or_init(|| SMARTS_BITS.iter().map(|s| Pattern::compile(s)).collect())
}

/// PubChem 881-bit structural key fingerprint (CDK `PubchemFingerprinter`).
///
/// # Examples
///
/// ```
/// use ferrum_core::desc::pubchem::PubChem881;
/// use ferrum_core::desc::typed::Descriptor;
/// use ferrum_core::io::smiles::parse_smiles;
///
/// let mut mol = parse_smiles("c1ccccc1").unwrap();
/// let descriptor = PubChem881;
/// let fp = descriptor.calculate(&mut mol, &mut ferrum_core::desc::context::CalculationContext::new());
///
/// let bits = fp.as_binary();
/// assert_eq!(bits.len(), 881);
/// assert!(bits.count_ones() > 0);
/// ```
pub struct PubChem881;

impl Descriptor for PubChem881 {
    fn name(&self) -> &str {
        "PubChem881"
    }

    fn calculate(
        &self,
        mol: &Molecule,
        _ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        DescriptorValue::Binary(pubchem_fingerprint(mol))
    }
}

/// Computes the 881 PubChem bits of `mol`.
pub fn pubchem_fingerprint(mol: &Molecule) -> BitVec<u8, Lsb0> {
    let mut bits = bitvec![u8, Lsb0; 0; PUBCHEM_FP_SIZE];
    let cm = CdkMol::build(mol);

    // Section 1: element counts (CountElements; implicit H are explicit atoms in CDK)
    let mut counts = [0u32; 120];
    for a in 0..cm.n() {
        counts[cm.z[a].min(119) as usize] += 1;
        counts[1] += cm.h_fields[a] as u32;
    }
    for (bit, &(z, min)) in ELEMENT_BITS.iter().enumerate() {
        if counts[z as usize] >= min as u32 {
            bits.set(bit, true);
        }
    }

    // Section 2: ESSSR ring set (CountRings)
    ring_bits(&cm, &mut bits);

    // Sections 3-7: SMARTS
    let adj_class: Vec<u8> = cm
        .adj_bond
        .iter()
        .map(|&b| {
            let b = b as usize;
            if cm.bond_arom[b] {
                BC_AROM
            } else {
                match cm.bond_order[b] {
                    ORD_SINGLE => BC_SINGLE,
                    ORD_DOUBLE => BC_DOUBLE,
                    ORD_TRIPLE => BC_TRIPLE,
                    _ => BC_OTHER,
                }
            }
        })
        .collect();
    let mut feats = 0u8;
    if cm.atom_arom.iter().any(|&x| x) {
        feats |= FEAT_AROM_ATOM;
    }
    for &c in &adj_class {
        feats |= match c {
            BC_DOUBLE => FEAT_DOUBLE,
            BC_TRIPLE => FEAT_TRIPLE,
            BC_AROM => FEAT_AROM_BOND,
            _ => 0,
        };
    }
    // atoms grouped by atomic number (counting sort)
    let mut heavy = [0u32; 121];
    for &z in &cm.z {
        heavy[z.min(119) as usize] += 1;
    }
    let mut by_z_start = [0u32; 121];
    for z in 0..120 {
        by_z_start[z + 1] = by_z_start[z] + heavy[z];
    }
    let mut fill = by_z_start;
    let mut by_z = vec![0u32; cm.n()];
    for (a, &z) in cm.z.iter().enumerate() {
        let z = z.min(119) as usize;
        by_z[fill[z] as usize] = a as u32;
        fill[z] += 1;
    }
    let target = Target {
        z: &cm.z,
        arom: &cm.atom_arom,
        hcount: &cm.total_h,
        adj_start: &cm.adj_start,
        adj_nbr: &cm.adj_nbr,
        adj_class: &adj_class,
        by_z_start: &by_z_start,
        by_z: &by_z,
    };
    for (i, p) in patterns().iter().enumerate() {
        if p.feats & !feats != 0 || p.need.iter().any(|&(z, k)| heavy[z as usize] < k as u32) {
            continue;
        }
        if p.matches(&target) {
            bits.set(SMARTS_BIT_OFFSET + i, true);
        }
    }
    bits
}

/// Ring bits 115-262 from the ESSSR ring set.
fn ring_bits(cm: &CdkMol, bits: &mut BitSlice<u8, Lsb0>) {
    // counts[kind][size] for sizes 3..=10; aromatic counts over all sizes
    let mut counts = [[0u32; 11]; 7];
    let mut n_arom = 0u32;
    let mut n_hetero_arom = 0u32;
    for ring in esssr(cm) {
        let size = ring.len();
        let mut carbon_only = true;
        let mut has_n = false;
        let mut has_hetero = false;
        for &a in &ring {
            let z = cm.z[a as usize];
            carbon_only &= z == 6;
            has_n |= z == 7;
            has_hetero |= z != 6 && z != 1;
        }
        let mut saturated = true;
        let mut aromatic = true;
        for i in 0..size {
            let (u, v) = (ring[i] as usize, ring[(i + 1) % size]);
            let b = cm
                .nbrs(u)
                .find(|&k| cm.adj_nbr[k] == v)
                .map(|k| cm.adj_bond[k] as usize)
                .expect("ring bond");
            let arom = cm.bond_arom[b];
            aromatic &= arom;
            if cm.bond_order[b] != ORD_SINGLE && !arom {
                saturated = false;
            }
        }
        if aromatic {
            n_arom += 1;
            if !carbon_only {
                n_hetero_arom += 1;
            }
        }
        if size > 10 {
            continue;
        }
        let sat_or_arom = saturated || aromatic;
        let unsat = !saturated && !aromatic;
        counts[0][size] += 1;
        counts[1][size] += (carbon_only && sat_or_arom) as u32;
        counts[2][size] += (has_n && sat_or_arom) as u32;
        counts[3][size] += (has_hetero && sat_or_arom) as u32;
        counts[4][size] += (carbon_only && unsat) as u32;
        counts[5][size] += (has_n && unsat) as u32;
        counts[6][size] += (has_hetero && unsat) as u32;
    }
    for (i, &(kind, size, min)) in RING_BITS.iter().enumerate() {
        let size = size as usize;
        let count = match kind {
            RingKind::Any => counts[0][size],
            RingKind::SatOrAromCarbon => counts[1][size],
            RingKind::SatOrAromNitrogen => counts[2][size],
            RingKind::SatOrAromHetero => counts[3][size],
            RingKind::UnsatCarbon => counts[4][size],
            RingKind::UnsatNitrogen => counts[5][size],
            RingKind::UnsatHetero => counts[6][size],
            RingKind::Aromatic => n_arom,
            RingKind::HeteroAromatic => n_hetero_arom,
        };
        if count >= min as u32 {
            bits.set(RING_BIT_OFFSET + i, true);
        }
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<PubChem881>;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn on_bits(smiles: &str) -> Vec<usize> {
        let mol = parse_smiles(smiles).expect("valid SMILES");
        pubchem_fingerprint(&mol).iter_ones().collect()
    }

    fn bits(list: &str) -> Vec<usize> {
        list.split_whitespace()
            .map(|b| b.parse().unwrap())
            .collect()
    }

    #[test]
    fn size_is_881() {
        let mol = parse_smiles("C").unwrap();
        let fp = PubChem881.calculate(
            &mol,
            &mut crate::desc::context::CalculationContext::new(),
        );
        assert_eq!(fp.as_binary().len(), 881);
    }

    // Expected values: CDK 2.13 PubchemFingerprinter (tests/data/cdk_parity_reference.csv).
    #[test]
    fn benzene_matches_cdk() {
        assert_eq!(
            on_bits("c1ccccc1"),
            bits(
                "0 9 10 178 179 255 283 284 332 344 355 370 371 384 416 434 441 446 470 490 516 520 524 552 \
                 556 564 570 578 582 584 595 599 603 608 618 634 640 660 664 668 677 678 679"
            )
        );
    }

    #[test]
    fn caffeine_matches_cdk() {
        assert_eq!(
            on_bits("CN1C=NC2=C1C(=O)N(C)C(=O)N2C"),
            bits(
                "0 1 9 10 11 14 15 16 18 19 143 145 146 178 180 181 241 243 244 255 256 257 258 259 260 283 \
                 284 285 286 332 340 351 352 355 357 358 359 365 373 374 375 376 377 378 379 381 384 386 387 \
                 388 389 390 391 396 397 403 416 418 420 431 437 438 439 441 442 443 447 449 450 451 453 464 \
                 472 482 484 485 487 491 493 494 495 499 504 506 519 521 523 530 535 536 538 540 545 547 549 \
                 553 555 560 569 572 580 585 593 596 601 602 611 613 621 624 628 636 645 646 647 650 654 657 \
                 673 674"
            )
        );
    }

    #[test]
    fn phenanthrene_matches_cdk() {
        assert_eq!(
            on_bits("c1ccc2c(c1)ccc1ccccc12"),
            bits(
                "0 1 9 10 11 178 179 185 186 192 193 248 249 255 257 259 261 283 284 332 333 344 355 356 370 \
                 371 384 385 416 430 434 441 446 470 490 516 520 524 552 556 564 570 578 582 584 595 599 603 \
                 608 618 634 640 660 664 668 677 678 679 688 696 697 708 709 710 712 713 734 755 776 797 818"
            )
        );
    }

    #[test]
    fn azulene_matches_cdk() {
        assert_eq!(
            on_bits("C1=CC2=CC=CC=CC2=C1"),
            bits(
                "0 1 9 10 11 143 144 213 214 248 249 255 283 284 332 333 344 355 356 370 371 384 416 430 434 \
                 441 446 470 490 516 520 524 552 556 564 570 578 582 584 595 599 603 608 618 634 640 660 664 \
                 668 677 678 679 688 696 697 708 709 710 712 860"
            )
        );
    }

    #[test]
    fn radical_input_matches_cdk() {
        // CDK reads [N]1C=CC=C1 as pyrrole (the radical becomes a hydrogen)
        assert_eq!(
            on_bits("[N]1C=CC=C1"),
            bits(
                "0 9 10 14 143 145 146 255 256 283 284 285 299 332 344 345 351 355 358 365 370 371 372 373 \
                 376 384 387 390 392 393 396 399 400 403 416 418 434 435 441 442 445 446 447 449 453 464 470 \
                 472 482 490 491 495 502 506 516 520 521 523 524 527 528 530 538 539 540 545 546 549 552 555 \
                 556 564 570 576 577 585 592 599 600 607 613 628 633 643 656 657 665"
            )
        );
    }

    #[test]
    fn deterministic_and_discriminating() {
        assert_eq!(on_bits("CCO"), on_bits("OCC"));
        assert_ne!(on_bits("c1ccccc1"), on_bits("CCCCCC"));
    }
}
