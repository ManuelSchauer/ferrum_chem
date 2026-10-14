//! CDK 2.13 substructure-key fingerprints and their count variants:
//!
//! - [`CdkSubstructure`] / [`CdkSubstructureCount`]: CDK `SubstructureFingerprinter()`
//!   with its default 307 functional-group keys (Laggner's `SMARTS_InteLigand.txt`);
//! - [`KlekotaRoth`] / [`KlekotaRothCount`]: CDK `KlekotaRothFingerprinter`
//!   (4,860 keys).
//!
//! The bit fingerprints are `getBitFingerprint` (bit `i` set when key `i`
//! matches), the counts `getCountFingerprint` (per key the number of unique
//! matches, CDK `Mappings.countUnique()`: distinct matched atom sets). CDK
//! prepares the molecule once with `SmartsPattern.prepare` (ring flags and
//! Daylight aromaticity); the SMARTS semantics are CDK's (see `cdk_smarts`,
//! whose prepared molecule is shared with CDKMACCS and CDKEState).
//!
//! When a key set's bits and counts are both requested for a molecule the
//! counts are computed once and the bits derived from them (`count > 0`).

use bitvec::prelude::*;
use once_cell::sync::Lazy;

use super::cdk_smarts::KeySet;
use super::cdk_substructure_keys::{CDK_SUBSTRUCTURE_SMARTS, KLEKOTA_ROTH_SMARTS};
use super::context::CalculationContext;
use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;

/// Number of CDK `SubstructureFingerprinter` keys.
pub const CDK_SUBSTRUCTURE_LEN: usize = 307;
/// Number of CDK `KlekotaRothFingerprinter` keys.
pub const KLEKOTA_ROTH_LEN: usize = 4860;

static SUBSTRUCTURE_KEYS: Lazy<KeySet> = Lazy::new(|| KeySet::new(&CDK_SUBSTRUCTURE_SMARTS, false));
static KLEKOTA_ROTH_KEYS: Lazy<KeySet> = Lazy::new(|| KeySet::new(&KLEKOTA_ROTH_SMARTS, true));

#[derive(Clone, Copy)]
enum Keys {
    Substructure = 0,
    KlekotaRoth = 1,
}

impl Keys {
    fn set(self) -> &'static KeySet {
        match self {
            Keys::Substructure => &SUBSTRUCTURE_KEYS,
            Keys::KlekotaRoth => &KLEKOTA_ROTH_KEYS,
        }
    }
}

fn counts(keys: Keys, mol: &Molecule, ctx: &mut CalculationContext) -> Vec<u32> {
    let cache = &mut ctx.cdk_keys;
    cache.select(mol);
    if let Some(c) = &cache.counts[keys as usize] {
        return c.clone();
    }
    let set = keys.set();
    let mut out = vec![0u32; set.len()];
    set.match_counts(cache.target(mol), &mut out);
    cache.counts[keys as usize] = Some(out.clone());
    out
}

fn bits(keys: Keys, mol: &Molecule, ctx: &mut CalculationContext) -> BitVec<u8, Lsb0> {
    let cache = &mut ctx.cdk_keys;
    cache.select(mol);
    let set = keys.set();
    let mut out = bitvec![u8, Lsb0; 0; set.len()];
    if let Some(c) = &cache.counts[keys as usize] {
        for (i, &n) in c.iter().enumerate() {
            out.set(i, n > 0);
        }
        return out;
    }
    let mut hit = vec![false; set.len()];
    set.match_bits(cache.target(mol), &mut hit);
    for (i, h) in hit.into_iter().enumerate() {
        out.set(i, h);
    }
    out
}

/// CDK `SubstructureFingerprinter()` (307 functional-group keys), bit fingerprint.
pub struct CdkSubstructure;

impl Descriptor for CdkSubstructure {
    fn name(&self) -> &str {
        "CDKSubstructure"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::Binary(bits(Keys::Substructure, mol, ctx))
    }
}

/// CDK `SubstructureFingerprinter().getCountFingerprint` (unique matches per key).
pub struct CdkSubstructureCount;

impl Descriptor for CdkSubstructureCount {
    fn name(&self) -> &str {
        "CDKSubstructureCount"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::IntVec(counts(Keys::Substructure, mol, ctx))
    }
}

/// CDK `KlekotaRothFingerprinter` (4,860 keys), bit fingerprint.
pub struct KlekotaRoth;

impl Descriptor for KlekotaRoth {
    fn name(&self) -> &str {
        "KlekotaRoth"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::Binary(bits(Keys::KlekotaRoth, mol, ctx))
    }
}

/// CDK `KlekotaRothFingerprinter.getCountFingerprint` (unique matches per key).
pub struct KlekotaRothCount;

impl Descriptor for KlekotaRothCount {
    fn name(&self) -> &str {
        "KlekotaRothCount"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        DescriptorValue::IntVec(counts(Keys::KlekotaRoth, mol, ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    fn on_bits(d: &dyn Descriptor, smiles: &str) -> Vec<usize> {
        let mol = parse_smiles(smiles).unwrap();
        d.calculate(&mol, &mut CalculationContext::new()).as_binary().iter_ones().collect()
    }

    fn nonzero(d: &dyn Descriptor, smiles: &str) -> Vec<(usize, u32)> {
        let mol = parse_smiles(smiles).unwrap();
        match d.calculate(&mol, &mut CalculationContext::new()) {
            DescriptorValue::IntVec(v) => v.into_iter().enumerate().filter(|&(_, c)| c > 0).collect(),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn key_tables_compile() {
        assert_eq!(SUBSTRUCTURE_KEYS.len(), CDK_SUBSTRUCTURE_LEN);
        assert_eq!(KLEKOTA_ROTH_KEYS.len(), KLEKOTA_ROTH_LEN);
    }

    #[test]
    fn substructure_matches_cdk() {
        // CDK 2.13 (reference preparation)
        assert_eq!(on_bits(&CdkSubstructure, "CCO"), vec![0, 11, 12, 294]);
        assert_eq!(on_bits(&CdkSubstructure, "CC(C)O"), vec![0, 11, 13, 294, 306]);
        assert_eq!(on_bits(&CdkSubstructure, "C[C@H](N)O"), vec![0, 62, 294, 306]);
        assert_eq!(on_bits(&CdkSubstructure, "c1ccccc1"), vec![273, 306]);
        assert_eq!(nonzero(&CdkSubstructureCount, "CC(C)O"), vec![(0, 2), (11, 1), (13, 1), (294, 1), (306, 1)]);
        assert_eq!(nonzero(&CdkSubstructureCount, "C[C@H](N)O"), vec![(0, 1), (62, 1), (294, 2), (306, 1)]);
        assert_eq!(nonzero(&CdkSubstructureCount, "c1ccccc1"), vec![(273, 6), (306, 6)]);
    }

    #[test]
    fn bits_agree_with_counts() {
        for smi in ["CC(=O)Oc1ccccc1C(=O)O", "c1ccc2[nH]ccc2c1", "C[N+](C)(C)C.[Cl-]", "OC1C(O)C(O)C(O)C(O)C1O"] {
            let mol = parse_smiles(smi).unwrap();
            for (b, c) in [
                (&CdkSubstructure as &dyn Descriptor, &CdkSubstructureCount as &dyn Descriptor),
                (&KlekotaRoth, &KlekotaRothCount),
            ] {
                let bits: Vec<usize> = b.calculate(&mol, &mut CalculationContext::new()).as_binary().iter_ones().collect();
                let counts: Vec<usize> = match c.calculate(&mol, &mut CalculationContext::new()) {
                    DescriptorValue::IntVec(v) => v.iter().enumerate().filter(|&(_, &n)| n > 0).map(|(i, _)| i).collect(),
                    other => panic!("{other:?}"),
                };
                assert_eq!(bits, counts, "{} on {smi}", b.name());
            }
        }
    }
}
