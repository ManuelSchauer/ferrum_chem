//! CDK E-State fingerprint: a port of CDK 2.13 `EStateFingerprinter` (79 bits).
//!
//! Bit `i` is set when the SMARTS of E-State fragment `i` (CDK
//! `EStateFragments`, the Hall/Kier atom types `sLi` ... `ssssPb` as converted
//! by RDKit) matches. CDK matches them with `SmartsPattern` on the molecule as
//! `SmartsPattern.prepare` leaves it (Daylight aromaticity), see `cdk_smarts`.

use super::cdk_smarts::KeySet;
use super::context::CalculationContext;
use super::typed::{Descriptor, DescriptorValue};
use crate::mol::molecule::Molecule;
use bitvec::prelude::*;
use once_cell::sync::Lazy;

/// CDK `EStateFragments` SMARTS, in bit order.
const SMARTS: [&str; 79] = [
    "[LiD1]-*",
    "[BeD2](-*)-*",
    "[BeD4](-*)(-*)(-*)-*",
    "[BD2H](-*)-*",
    "[BD3](-*)(-*)-*",
    "[BD4](-*)(-*)(-*)-*",
    "[CD1H3]-*",
    "[CD1H2]=*",
    "[CD2H2](-*)-*",
    "[CD1H]#*",
    "[CD2H](=*)-*",
    "[C,c;D2H](:*):*",
    "[CD3H](-*)(-*)-*",
    "[CD2H0](=*)=*",
    "[CD2H0](#*)-*",
    "[CD3H0](=*)(-*)-*",
    "[C,c;D3H0](:*)(:*)-*",
    "[C,c;D3H0](:*)(:*):*",
    "[CD4H0](-*)(-*)(-*)-*",
    "[ND1H3]-*",
    "[ND1H2]-*",
    "[ND2H2](-*)-*",
    "[ND1H]=*",
    "[ND2H](-*)-*",
    "[N,nD2H](:*):*",
    "[ND1H0]#*",
    "[ND3H](-*)(-*)-*",
    "[ND2H0](=*)-*",
    "[N,nD2H0](:*):*",
    "[ND3H0](-*)(-*)-*",
    "[ND3H0](~[OD1H0])(~[OD1H0])-,:*",
    "[N,nD3H0](:*)(:*)-,:*",
    "[ND4H0](-*)(-*)(-*)-*",
    "[OD1H]-*",
    "[OD1H0]=*",
    "[OD2H0](-*)-*",
    "[O,oD2H0](:*):*",
    "[FD1]-*",
    "[SiD1H3]-*",
    "[SiD2H2](-*)-*",
    "[SiD3H1](-*)(-*)-*",
    "[SiD4H0](-*)(-*)(-*)-*",
    "[PD1H2]-*",
    "[PD2H1](-*)-*",
    "[PD3H0](-*)(-*)-*",
    "[PD4H0](=*)(-*)(-*)-*",
    "[PD5H0](-*)(-*)(-*)(-*)-*",
    "[SD1H1]-*",
    "[SD1H0]=*",
    "[SD2H0](-*)-*",
    "[S,sD2H0](:*):*",
    "[SD3H0](=*)(-*)-*",
    "[SD4H0](~[OD1H0])(~[OD1H0])(-*)-*",
    "[ClD1]-*",
    "[GeD1H3](-*)",
    "[GeD2H2](-*)-*",
    "[GeD3H1](-*)(-*)-*",
    "[GeD4H0](-*)(-*)(-*)-*",
    "[AsD1H2]-*",
    "[AsD2H1](-*)-*",
    "[AsD3H0](-*)(-*)-*",
    "[AsD4H0](=*)(-*)(-*)-*",
    "[AsD5H0](-*)(-*)(-*)(-*)-*",
    "[SeD1H1]-*",
    "[SeD1H0]=*",
    "[SeD2H0](-*)-*",
    "[SeD2H0](:*):*",
    "[SeD3H0](=*)(-*)-*",
    "[SeD4H0](=*)(=*)(-*)-*",
    "[BrD1]-*",
    "[SnD1H3]-*",
    "[SnD2H2](-*)-*",
    "[SnD3H1](-*)(-*)-*",
    "[SnD4H0](-*)(-*)(-*)-*",
    "[ID1]-*",
    "[PbD1H3]-*",
    "[PbD2H2](-*)-*",
    "[PbD3H1](-*)(-*)-*",
    "[PbD4H0](-*)(-*)(-*)-*",
];

/// Number of CDK E-State fingerprint bits.
pub const CDK_ESTATE_FP_SIZE: usize = SMARTS.len();

static ESTATE_KEYS: Lazy<KeySet> = Lazy::new(|| KeySet::new(&SMARTS, false));

/// CDK E-State fingerprint (CDK 2.13 `EStateFingerprinter`, 79 bits).
///
/// # Examples
///
/// ```
/// use ferrum_core::desc::cdk_estate_fp::CdkEStateFingerprint;
/// use ferrum_core::desc::typed::Descriptor;
/// use ferrum_core::io::smiles::parse_smiles;
///
/// let mol = parse_smiles("CCO").unwrap();
/// let fp = CdkEStateFingerprint.calculate(&mol, &mut ferrum_core::desc::context::CalculationContext::new());
/// let bits = fp.as_binary();
/// assert_eq!(bits.len(), 79);
/// assert!(bits[6] && bits[8] && bits[33]); // sCH3, ssCH2, sOH
/// ```
pub struct CdkEStateFingerprint;

impl Descriptor for CdkEStateFingerprint {
    fn name(&self) -> &str {
        "CdkEStateFingerprint"
    }

    fn calculate(&self, mol: &Molecule, ctx: &mut CalculationContext) -> DescriptorValue {
        let mut hit = [false; CDK_ESTATE_FP_SIZE];
        ESTATE_KEYS.match_bits(ctx.cdk_keys.target(mol), &mut hit);
        let mut fp = bitvec![u8, Lsb0; 0; CDK_ESTATE_FP_SIZE];
        for (i, h) in hit.into_iter().enumerate() {
            fp.set(i, h);
        }
        DescriptorValue::Binary(fp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::parse_smiles;

    #[test]
    fn prepared_like_cdk() {
        // CDK 2.13 (reference preparation): hydrogen atoms of the SMILES on
        // ring N/P stay atoms, the radical Sn of `C[Sn](C)C` gets one H
        for (smiles, bits) in [("[H]N1CCCC1", vec![8, 26]), ("[H]P1CCCC1", vec![8]), ("C[Sn](C)C", vec![6, 72])] {
            let mol = parse_smiles(smiles).unwrap();
            let fp = CdkEStateFingerprint.calculate(&mol, &mut CalculationContext::new());
            assert_eq!(fp.as_binary().iter_ones().collect::<Vec<_>>(), bits, "{smiles}");
        }
    }
}
