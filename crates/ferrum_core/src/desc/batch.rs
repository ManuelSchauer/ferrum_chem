//! Descriptor module: Batch.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

use super::aromatic_ring_count::AromaticRingCount;
use super::atom_count::{AromaticAtomCount, AtomCount, ElementCounts, HeavyAtomCount};
use super::autocorrelation::Autocorrelation;
use super::balaban::BalabanIndex;
use super::bcut::BCUT;
use super::bond_count::{BondCount, BondCountDetailed, RotatableBondCount};
use super::carbon_types::CarbonTypes;
use super::chains::{LargestChain, LongestAliphaticChain};
use super::chi::ChiIndices;
use super::estate::EStateIndices;
use super::extra_counts::{
    AromaticAtomFraction, BertzCT, FractionCSP3, FragmentComplexity, HallKierAlpha, Ipc,
    NumAmideBonds,
};
use super::fingerprint::{ECFP, FCFP};
use super::formal_charge::FormalCharge;
use super::shape_3d::{Asphericity, Pmi1, Pmi2, Pmi3};
use super::biosynfoni::Biosynfoni;
use super::functional_groups::{AcidicGroupCount, BasicGroupCount, FunctionalGroupCounts};
use super::gasteiger_charge::{
    MaxAbsPartialCharge, MaxPartialCharge, MinAbsPartialCharge, MinPartialCharge,
};
use super::hbond::{HBondAcceptor, HBondDonor};
use super::kappa::KappaIndices;
use super::lipinski::LipinskiViolations;
use super::logp::{LogP, MolarRefractivity};
use super::maccs::MaccsKeys;
use super::mde::MDE;
use super::molecular_weight::MolecularWeight;
use super::path_fingerprints::{AtomPairFingerprint, TopologicalTorsionFingerprint};
use super::petitjean::PetitjeanIndex;
use super::pi_system::{LargestPiSystem, PiSystemCounts};
use super::polarizability::{APol, BPol};
use super::pubchem::PubChem881;
use super::qed::Qed;
use super::ring_types::RingTypeCounts;
use super::simple_counts::{
    HeteroatomCount, HybridizationRatio, RadicalElectrons, VAdjMa, ValenceElectrons,
};
use super::small_ring::SmallRingDescriptor;
use super::tpsa::Tpsa;
use super::typed::{Descriptor, DescriptorValue};
use super::vsa::{
    LabuteASA, PeoeVsa, SlogpVSA, SmrVSA, fill_peoe_vsa_into, fill_slogp_vsa_into,
    fill_smr_vsa_into,
};
use super::weighted_path::WeightedPath;
use super::wiener_numbers::WienerIndex;
use super::zagreb::ZagrebIndices;
use crate::mol::molecule::Molecule;

/// Static-dispatch batch descriptor set for high-throughput ML/batch paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchDescriptor {
    MolecularWeight,
    LogP,
    MolarRefractivity,
    Tpsa,
    HBondDonor,
    HBondAcceptor,
    RotatableBondCount,
    HeavyAtomCount,
    AtomCount,
    ElementCounts,
    AromaticAtomCount,
    HeteroatomCount,
    AromaticRingCount,
    BondCount,
    BondCountDetailed,
    ValenceElectrons,
    RadicalElectrons,
    HybridizationRatio,
    VAdjMa,
    FormalCharge,
    FractionCSP3,
    BertzCT,
    Ipc,
    FragmentComplexity,
    LabuteAsa,
    Qed,
    NumAmideBonds,
    HallKierAlpha,
    AromaticAtomFraction,
    LipinskiViolations,
    BalabanIndex,
    WienerIndex,
    ZagrebIndices,
    PetitjeanIndex,
    SmallRingDescriptor,
    RingTypeCounts,
    PiSystemCounts,
    LargestPiSystem,
    CarbonTypes,
    ChiIndices,
    KappaIndices,
    EStateIndices,
    Autocorrelation,
    Bcut,
    WeightedPath,
    Mde,
    FunctionalGroupCounts,
    AcidicGroupCount,
    BasicGroupCount,
    LongestAliphaticChain,
    LargestChain,
    MaxPartialCharge,
    MinPartialCharge,
    MaxAbsPartialCharge,
    MinAbsPartialCharge,
    APol,
    BPol,
    SlogpVsa,
    SmrVsa,
    PeoeVsa,
    MaccsKeys,
    PubChem881,
    Ecfp4,
    Fcfp4,
    AtomPairFingerprint,
    TopologicalTorsionFingerprint,
    Pmi1,
    Pmi2,
    Pmi3,
    Asphericity,
    Biosynfoni,
    MolWeights,
    ChiNIndices,
    EStateAbs,
    EStateVsa,
    VsaEState,
    RdkitRingCounts,
    LipinskiNO,
    AvgIpcPhi,
    FpDensityMorgan,
    Fragments,
    Bcut2D,
    CdkChiIndices,
    CdkFragmentComplexity,
    EccentricConnectivity,
    StereoCenters,
    Sps,
    CdkKappa,
    MordredEState,
    MordredMde,
    MordredMisc,
    MordredConstitutional,
    MordredRingCount,
    MordredChi,
    MordredPathCount,
    MordredWalkCount,
    MordredTopologicalCharge,
    MordredMolecularId,
    MordredInformationContent,
    MordredEta,
    MordredMatrix,
}

impl BatchDescriptor {
    pub fn all() -> &'static [Self] {
        const ALL: [BatchDescriptor; 101] = [
            BatchDescriptor::MolecularWeight,
            BatchDescriptor::LogP,
            BatchDescriptor::MolarRefractivity,
            BatchDescriptor::Tpsa,
            BatchDescriptor::HBondDonor,
            BatchDescriptor::HBondAcceptor,
            BatchDescriptor::RotatableBondCount,
            BatchDescriptor::HeavyAtomCount,
            BatchDescriptor::AtomCount,
            BatchDescriptor::ElementCounts,
            BatchDescriptor::AromaticAtomCount,
            BatchDescriptor::HeteroatomCount,
            BatchDescriptor::AromaticRingCount,
            BatchDescriptor::BondCount,
            BatchDescriptor::BondCountDetailed,
            BatchDescriptor::ValenceElectrons,
            BatchDescriptor::RadicalElectrons,
            BatchDescriptor::HybridizationRatio,
            BatchDescriptor::VAdjMa,
            BatchDescriptor::FormalCharge,
            BatchDescriptor::FractionCSP3,
            BatchDescriptor::BertzCT,
            BatchDescriptor::Ipc,
            BatchDescriptor::FragmentComplexity,
            BatchDescriptor::LabuteAsa,
            BatchDescriptor::Qed,
            BatchDescriptor::NumAmideBonds,
            BatchDescriptor::HallKierAlpha,
            BatchDescriptor::AromaticAtomFraction,
            BatchDescriptor::LipinskiViolations,
            BatchDescriptor::BalabanIndex,
            BatchDescriptor::WienerIndex,
            BatchDescriptor::ZagrebIndices,
            BatchDescriptor::PetitjeanIndex,
            BatchDescriptor::SmallRingDescriptor,
            BatchDescriptor::RingTypeCounts,
            BatchDescriptor::PiSystemCounts,
            BatchDescriptor::LargestPiSystem,
            BatchDescriptor::CarbonTypes,
            BatchDescriptor::ChiIndices,
            BatchDescriptor::KappaIndices,
            BatchDescriptor::EStateIndices,
            BatchDescriptor::Autocorrelation,
            BatchDescriptor::Bcut,
            BatchDescriptor::WeightedPath,
            BatchDescriptor::Mde,
            BatchDescriptor::FunctionalGroupCounts,
            BatchDescriptor::AcidicGroupCount,
            BatchDescriptor::BasicGroupCount,
            BatchDescriptor::LongestAliphaticChain,
            BatchDescriptor::LargestChain,
            BatchDescriptor::MaxPartialCharge,
            BatchDescriptor::MinPartialCharge,
            BatchDescriptor::MaxAbsPartialCharge,
            BatchDescriptor::MinAbsPartialCharge,
            BatchDescriptor::APol,
            BatchDescriptor::BPol,
            BatchDescriptor::SlogpVsa,
            BatchDescriptor::SmrVsa,
            BatchDescriptor::PeoeVsa,
            BatchDescriptor::MaccsKeys,
            BatchDescriptor::PubChem881,
            BatchDescriptor::Ecfp4,
            BatchDescriptor::Fcfp4,
            BatchDescriptor::AtomPairFingerprint,
            BatchDescriptor::TopologicalTorsionFingerprint,
            BatchDescriptor::Pmi1,
            BatchDescriptor::Pmi2,
            BatchDescriptor::Pmi3,
            BatchDescriptor::Asphericity,
            BatchDescriptor::Biosynfoni,
            BatchDescriptor::MolWeights,
            BatchDescriptor::ChiNIndices,
            BatchDescriptor::EStateAbs,
            BatchDescriptor::EStateVsa,
            BatchDescriptor::VsaEState,
            BatchDescriptor::RdkitRingCounts,
            BatchDescriptor::LipinskiNO,
            BatchDescriptor::AvgIpcPhi,
            BatchDescriptor::FpDensityMorgan,
            BatchDescriptor::Fragments,
            BatchDescriptor::Bcut2D,
            BatchDescriptor::CdkChiIndices,
            BatchDescriptor::CdkFragmentComplexity,
            BatchDescriptor::EccentricConnectivity,
            BatchDescriptor::StereoCenters,
            BatchDescriptor::Sps,
            BatchDescriptor::CdkKappa,
            BatchDescriptor::MordredEState,
            BatchDescriptor::MordredMde,
            BatchDescriptor::MordredMisc,
            BatchDescriptor::MordredConstitutional,
            BatchDescriptor::MordredRingCount,
            BatchDescriptor::MordredChi,
            BatchDescriptor::MordredPathCount,
            BatchDescriptor::MordredWalkCount,
            BatchDescriptor::MordredTopologicalCharge,
            BatchDescriptor::MordredMolecularId,
            BatchDescriptor::MordredInformationContent,
            BatchDescriptor::MordredEta,
            BatchDescriptor::MordredMatrix,
        ];
        &ALL
    }

    /// Returns a curated descriptor subset for fast, broad-coverage ML featurization.
    pub fn fast_ml_preset() -> &'static [Self] {
        const FAST_ML: [BatchDescriptor; 46] = [
            BatchDescriptor::MolecularWeight,
            BatchDescriptor::LogP,
            BatchDescriptor::MolarRefractivity,
            BatchDescriptor::Tpsa,
            BatchDescriptor::HBondDonor,
            BatchDescriptor::HBondAcceptor,
            BatchDescriptor::RotatableBondCount,
            BatchDescriptor::HeavyAtomCount,
            BatchDescriptor::AtomCount,
            BatchDescriptor::ElementCounts,
            BatchDescriptor::AromaticAtomCount,
            BatchDescriptor::HeteroatomCount,
            BatchDescriptor::AromaticRingCount,
            BatchDescriptor::BondCount,
            BatchDescriptor::BondCountDetailed,
            BatchDescriptor::ValenceElectrons,
            BatchDescriptor::HybridizationRatio,
            BatchDescriptor::VAdjMa,
            BatchDescriptor::FormalCharge,
            BatchDescriptor::FractionCSP3,
            BatchDescriptor::BertzCT,
            BatchDescriptor::FragmentComplexity,
            BatchDescriptor::LabuteAsa,
            BatchDescriptor::Qed,
            BatchDescriptor::NumAmideBonds,
            BatchDescriptor::HallKierAlpha,
            BatchDescriptor::AromaticAtomFraction,
            BatchDescriptor::LipinskiViolations,
            BatchDescriptor::BalabanIndex,
            BatchDescriptor::WienerIndex,
            BatchDescriptor::ZagrebIndices,
            BatchDescriptor::PetitjeanIndex,
            BatchDescriptor::RingTypeCounts,
            BatchDescriptor::PiSystemCounts,
            BatchDescriptor::LargestPiSystem,
            BatchDescriptor::CarbonTypes,
            BatchDescriptor::FunctionalGroupCounts,
            BatchDescriptor::AcidicGroupCount,
            BatchDescriptor::BasicGroupCount,
            BatchDescriptor::LongestAliphaticChain,
            BatchDescriptor::LargestChain,
            BatchDescriptor::APol,
            BatchDescriptor::BPol,
            BatchDescriptor::SlogpVsa,
            BatchDescriptor::SmrVsa,
            BatchDescriptor::PeoeVsa,
        ];
        &FAST_ML
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::MolecularWeight => "MolecularWeight",
            Self::LogP => "LogP",
            Self::MolarRefractivity => "MolarRefractivity",
            Self::Tpsa => "TPSA",
            Self::HBondDonor => "HBondDonor",
            Self::HBondAcceptor => "HBondAcceptor",
            Self::RotatableBondCount => "RotatableBondCount",
            Self::HeavyAtomCount => "HeavyAtomCount",
            Self::AtomCount => "AtomCount",
            Self::ElementCounts => "ElementCounts",
            Self::AromaticAtomCount => "AromaticAtomCount",
            Self::HeteroatomCount => "HeteroatomCount",
            Self::AromaticRingCount => "AromaticRingCount",
            Self::BondCount => "BondCount",
            Self::BondCountDetailed => "BondCountDetailed",
            Self::ValenceElectrons => "ValenceElectrons",
            Self::RadicalElectrons => "RadicalElectrons",
            Self::HybridizationRatio => "HybridizationRatio",
            Self::VAdjMa => "VAdjMa",
            Self::FormalCharge => "FormalCharge",
            Self::FractionCSP3 => "FractionCSP3",
            Self::BertzCT => "BertzCT",
            Self::Ipc => "Ipc",
            Self::FragmentComplexity => "FragmentComplexity",
            Self::LabuteAsa => "LabuteASA",
            Self::Qed => "QED",
            Self::NumAmideBonds => "NumAmideBonds",
            Self::HallKierAlpha => "HallKierAlpha",
            Self::AromaticAtomFraction => "AromaticAtomFraction",
            Self::LipinskiViolations => "LipinskiViolations",
            Self::BalabanIndex => "BalabanIndex",
            Self::WienerIndex => "WienerIndex",
            Self::ZagrebIndices => "ZagrebIndices",
            Self::PetitjeanIndex => "PetitjeanIndex",
            Self::SmallRingDescriptor => "SmallRingDescriptor",
            Self::RingTypeCounts => "RingTypeCounts",
            Self::PiSystemCounts => "PiSystemCounts",
            Self::LargestPiSystem => "LargestPiSystem",
            Self::CarbonTypes => "CarbonTypes",
            Self::ChiIndices => "ChiIndices",
            Self::KappaIndices => "KappaIndices",
            Self::EStateIndices => "EStateIndices",
            Self::Autocorrelation => "Autocorrelation",
            Self::Bcut => "BCUT",
            Self::WeightedPath => "WeightedPath",
            Self::Mde => "MDE",
            Self::FunctionalGroupCounts => "FunctionalGroupCounts",
            Self::AcidicGroupCount => "AcidicGroupCount",
            Self::BasicGroupCount => "BasicGroupCount",
            Self::LongestAliphaticChain => "LongestAliphaticChain",
            Self::LargestChain => "LargestChain",
            Self::MaxPartialCharge => "MaxPartialCharge",
            Self::MinPartialCharge => "MinPartialCharge",
            Self::MaxAbsPartialCharge => "MaxAbsPartialCharge",
            Self::MinAbsPartialCharge => "MinAbsPartialCharge",
            Self::APol => "APol",
            Self::BPol => "BPol",
            Self::SlogpVsa => "SlogP_VSA",
            Self::SmrVsa => "SMR_VSA",
            Self::PeoeVsa => "PEOE_VSA",
            Self::MaccsKeys => "MaccsKeys",
            Self::PubChem881 => "PubChem881",
            Self::Ecfp4 => "ECFP4",
            Self::Fcfp4 => "FCFP4",
            Self::AtomPairFingerprint => "AtomPair",
            Self::TopologicalTorsionFingerprint => "TopologicalTorsion",
            Self::Pmi1 => "PMI1",
            Self::Pmi2 => "PMI2",
            Self::Pmi3 => "PMI3",
            Self::Asphericity => "Asphericity",
            Self::Biosynfoni => "Biosynfoni",
            Self::MolWeights => "MolWeights",
            Self::ChiNIndices => "ChiNIndices",
            Self::EStateAbs => "EStateAbs",
            Self::EStateVsa => "EState_VSA",
            Self::VsaEState => "VSA_EState",
            Self::RdkitRingCounts => "RingCounts",
            Self::LipinskiNO => "LipinskiNO",
            Self::AvgIpcPhi => "AvgIpcPhi",
            Self::FpDensityMorgan => "FpDensityMorgan",
            Self::Fragments => "Fragments",
            Self::Bcut2D => "BCUT2D",
            Self::CdkChiIndices => "CdkChiIndices",
            Self::CdkFragmentComplexity => "FragmentComplexityCDK",
            Self::EccentricConnectivity => "EccentricConnectivityIndex",
            Self::StereoCenters => "StereoCenters",
            Self::Sps => "SPS",
            Self::CdkKappa => "CdkKappaShapeIndices",
            Self::MordredEState => "MordredEState",
            Self::MordredMde => "MordredMDE",
            Self::MordredMisc => "MordredMisc",
            Self::MordredConstitutional => "MordredConstitutional",
            Self::MordredRingCount => "MordredRingCount",
            Self::MordredChi => "MordredChi",
            Self::MordredPathCount => "MordredPathCount",
            Self::MordredWalkCount => "MordredWalkCount",
            Self::MordredTopologicalCharge => "MordredTopologicalCharge",
            Self::MordredMolecularId => "MordredMolecularId",
            Self::MordredInformationContent => "MordredInformationContent",
            Self::MordredEta => "MordredEta",
            Self::MordredMatrix => "MordredMatrix",
        }
    }

    pub fn calculate(
        self,
        mol: &Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue {
        match self {
            Self::MolecularWeight => MolecularWeight.calculate(mol, ctx),
            Self::LogP => LogP.calculate(mol, ctx),
            Self::MolarRefractivity => MolarRefractivity.calculate(mol, ctx),
            Self::Tpsa => Tpsa.calculate(mol, ctx),
            Self::HBondDonor => HBondDonor.calculate(mol, ctx),
            Self::HBondAcceptor => HBondAcceptor.calculate(mol, ctx),
            Self::RotatableBondCount => RotatableBondCount.calculate(mol, ctx),
            Self::HeavyAtomCount => HeavyAtomCount.calculate(mol, ctx),
            Self::AtomCount => AtomCount.calculate(mol, ctx),
            Self::ElementCounts => ElementCounts.calculate(mol, ctx),
            Self::AromaticAtomCount => AromaticAtomCount.calculate(mol, ctx),
            Self::HeteroatomCount => HeteroatomCount.calculate(mol, ctx),
            Self::AromaticRingCount => AromaticRingCount.calculate(mol, ctx),
            Self::BondCount => BondCount.calculate(mol, ctx),
            Self::BondCountDetailed => BondCountDetailed.calculate(mol, ctx),
            Self::ValenceElectrons => ValenceElectrons.calculate(mol, ctx),
            Self::RadicalElectrons => RadicalElectrons.calculate(mol, ctx),
            Self::HybridizationRatio => HybridizationRatio.calculate(mol, ctx),
            Self::VAdjMa => VAdjMa.calculate(mol, ctx),
            Self::FormalCharge => FormalCharge.calculate(mol, ctx),
            Self::FractionCSP3 => FractionCSP3.calculate(mol, ctx),
            Self::BertzCT => BertzCT.calculate(mol, ctx),
            Self::Ipc => Ipc.calculate(mol, ctx),
            Self::FragmentComplexity => FragmentComplexity.calculate(mol, ctx),
            Self::LabuteAsa => LabuteASA.calculate(mol, ctx),
            Self::Qed => Qed.calculate(mol, ctx),
            Self::NumAmideBonds => NumAmideBonds.calculate(mol, ctx),
            Self::HallKierAlpha => HallKierAlpha.calculate(mol, ctx),
            Self::AromaticAtomFraction => AromaticAtomFraction.calculate(mol, ctx),
            Self::LipinskiViolations => LipinskiViolations.calculate(mol, ctx),
            Self::BalabanIndex => BalabanIndex.calculate(mol, ctx),
            Self::WienerIndex => WienerIndex.calculate(mol, ctx),
            Self::ZagrebIndices => ZagrebIndices.calculate(mol, ctx),
            Self::PetitjeanIndex => PetitjeanIndex.calculate(mol, ctx),
            Self::SmallRingDescriptor => SmallRingDescriptor.calculate(mol, ctx),
            Self::RingTypeCounts => RingTypeCounts.calculate(mol, ctx),
            Self::PiSystemCounts => PiSystemCounts.calculate(mol, ctx),
            Self::LargestPiSystem => LargestPiSystem.calculate(mol, ctx),
            Self::CarbonTypes => CarbonTypes.calculate(mol, ctx),
            Self::ChiIndices => ChiIndices.calculate(mol, ctx),
            Self::KappaIndices => KappaIndices.calculate(mol, ctx),
            Self::EStateIndices => EStateIndices.calculate(mol, ctx),
            Self::Autocorrelation => Autocorrelation.calculate(mol, ctx),
            Self::Bcut => BCUT.calculate(mol, ctx),
            Self::WeightedPath => WeightedPath.calculate(mol, ctx),
            Self::Mde => MDE.calculate(mol, ctx),
            Self::FunctionalGroupCounts => FunctionalGroupCounts.calculate(mol, ctx),
            Self::AcidicGroupCount => AcidicGroupCount.calculate(mol, ctx),
            Self::BasicGroupCount => BasicGroupCount.calculate(mol, ctx),
            Self::LongestAliphaticChain => LongestAliphaticChain.calculate(mol, ctx),
            Self::LargestChain => LargestChain.calculate(mol, ctx),
            Self::MaxPartialCharge => MaxPartialCharge.calculate(mol, ctx),
            Self::MinPartialCharge => MinPartialCharge.calculate(mol, ctx),
            Self::MaxAbsPartialCharge => MaxAbsPartialCharge.calculate(mol, ctx),
            Self::MinAbsPartialCharge => MinAbsPartialCharge.calculate(mol, ctx),
            Self::APol => APol.calculate(mol, ctx),
            Self::BPol => BPol.calculate(mol, ctx),
            Self::SlogpVsa => SlogpVSA.calculate(mol, ctx),
            Self::SmrVsa => SmrVSA.calculate(mol, ctx),
            Self::PeoeVsa => PeoeVsa.calculate(mol, ctx),
            Self::MaccsKeys => MaccsKeys.calculate(mol, ctx),
            Self::PubChem881 => PubChem881.calculate(mol, ctx),
            Self::Ecfp4 => ECFP::ecfp4().calculate(mol, ctx),
            Self::Fcfp4 => FCFP::fcfp4().calculate(mol, ctx),
            Self::AtomPairFingerprint => AtomPairFingerprint::default_size().calculate(mol, ctx),
            Self::TopologicalTorsionFingerprint => {
                TopologicalTorsionFingerprint::default_size().calculate(mol, ctx)
            }
            Self::Pmi1 => Pmi1.calculate(mol, ctx),
            Self::Pmi2 => Pmi2.calculate(mol, ctx),
            Self::Pmi3 => Pmi3.calculate(mol, ctx),
            Self::Asphericity => Asphericity.calculate(mol, ctx),
            Self::Biosynfoni => Biosynfoni.calculate(mol, ctx),
            Self::MolWeights => super::rdkit_extra::MolWeights.calculate(mol, ctx),
            Self::ChiNIndices => super::rdkit_extra::ChiNIndices.calculate(mol, ctx),
            Self::EStateAbs => super::rdkit_extra::EStateAbs.calculate(mol, ctx),
            Self::EStateVsa => super::rdkit_extra::EStateVsa.calculate(mol, ctx),
            Self::VsaEState => super::rdkit_extra::VsaEState.calculate(mol, ctx),
            Self::RdkitRingCounts => super::rdkit_extra::RingCounts.calculate(mol, ctx),
            Self::LipinskiNO => super::rdkit_extra::LipinskiNO.calculate(mol, ctx),
            Self::AvgIpcPhi => super::rdkit_extra::AvgIpcPhi.calculate(mol, ctx),
            Self::FpDensityMorgan => super::rdkit_extra::FpDensityMorgan.calculate(mol, ctx),
            Self::Fragments => super::rdkit_extra::Fragments.calculate(mol, ctx),
            Self::Bcut2D => super::rdkit_extra::Bcut2D.calculate(mol, ctx),
            Self::CdkChiIndices => super::chi::CdkChiIndices.calculate(mol, ctx),
            Self::CdkFragmentComplexity => super::extra_counts::CdkFragmentComplexity.calculate(mol, ctx),
            Self::EccentricConnectivity => super::eccentric::EccentricConnectivityIndex.calculate(mol, ctx),
            Self::StereoCenters => super::rdkit_extra::StereoCenters.calculate(mol, ctx),
            Self::Sps => super::rdkit_extra::Sps.calculate(mol, ctx),
            Self::CdkKappa => super::kappa::CdkKappaShapeIndices.calculate(mol, ctx),
            Self::MordredEState => super::mordred_estate::MordredEState.calculate(mol, ctx),
            Self::MordredMde => super::mordred_mde::MordredMDE.calculate(mol, ctx),
            Self::MordredMisc => super::mordred_misc::MordredMisc.calculate(mol, ctx),
            Self::MordredConstitutional => super::mordred_constitutional::MordredConstitutional.calculate(mol, ctx),
            Self::MordredRingCount => super::mordred_rings::MordredRingCount.calculate(mol, ctx),
            Self::MordredChi => super::mordred_chi::MordredChi.calculate(mol, ctx),
            Self::MordredPathCount => super::mordred_paths::MordredPathCount.calculate(mol, ctx),
            Self::MordredWalkCount => super::mordred_topological::MordredWalkCount.calculate(mol, ctx),
            Self::MordredTopologicalCharge => super::mordred_topological::MordredTopologicalCharge.calculate(mol, ctx),
            Self::MordredMolecularId => super::mordred_molecular_id::MordredMolecularId.calculate(mol, ctx),
            Self::MordredInformationContent => {
                super::mordred_information::MordredInformationContent.calculate(mol, ctx)
            }
            Self::MordredEta => super::mordred_eta::MordredEta.calculate(mol, ctx),
            Self::MordredMatrix => super::mordred_matrix::MordredMatrix.calculate(mol, ctx),
        }
    }

    pub fn feature_len(self) -> usize {
        match self {
            Self::ElementCounts => 12,
            Self::BondCountDetailed => 5,
            Self::SmallRingDescriptor => 11,
            Self::RingTypeCounts => 3,
            Self::PiSystemCounts => 2,
            Self::CarbonTypes => 9,
            Self::ChiIndices => 7,
            Self::CdkChiIndices => 40,
            Self::KappaIndices => 3,
            Self::EStateIndices => 3 + super::estate::NUM_ESTATE_TYPES,
            Self::Autocorrelation => super::autocorrelation::N_OUT,
            Self::Bcut => super::bcut::N_OUT,
            Self::WeightedPath => 5,
            Self::Mde => 19,
            Self::WienerIndex => 2,
            Self::ZagrebIndices => 2,
            Self::PetitjeanIndex => 3,
            Self::FunctionalGroupCounts => 16,
            Self::SlogpVsa => 12,
            Self::SmrVsa => 10,
            Self::PeoeVsa => 14,
            Self::MaccsKeys => 166,
            Self::PubChem881 => 881,
            Self::Ecfp4
            | Self::Fcfp4
            | Self::AtomPairFingerprint
            | Self::TopologicalTorsionFingerprint => 2048,
            Self::Biosynfoni => super::biosynfoni::BIOSYNFONI_FEATURES.len(),
            Self::MolWeights => 2,
            Self::ChiNIndices => 5,
            Self::EStateAbs => 2,
            Self::EStateVsa => 11,
            Self::VsaEState => 10,
            Self::RdkitRingCounts => 10,
            Self::LipinskiNO => 2,
            Self::AvgIpcPhi => 2,
            Self::FpDensityMorgan => 3,
            Self::Fragments => super::rdkit_fragments_data::FRAGMENTS.len(),
            Self::Bcut2D => 8,
            Self::CdkFragmentComplexity => 1,
            Self::EccentricConnectivity => 1,
            Self::StereoCenters => 2,
            Self::Sps => 1,
            Self::CdkKappa => 3,
            Self::MordredEState => super::mordred_estate::MORDRED_ESTATE_LEN,
            Self::MordredMde => super::mordred_mde::MORDRED_MDE_NAMES.len(),
            Self::MordredMisc => super::mordred_misc::MORDRED_MISC_LEN,
            Self::MordredConstitutional => super::mordred_constitutional::MORDRED_CONSTITUTIONAL_LEN,
            Self::MordredRingCount => super::mordred_rings::MORDRED_RING_COUNT_LEN,
            Self::MordredChi => super::mordred_chi::MORDRED_CHI_LEN,
            Self::MordredPathCount => super::mordred_paths::MORDRED_PATH_COUNT_LEN,
            Self::MordredWalkCount => super::mordred_topological::MORDRED_WALK_COUNT_LEN,
            Self::MordredTopologicalCharge => super::mordred_topological::MORDRED_TOPOLOGICAL_CHARGE_LEN,
            Self::MordredMolecularId => super::mordred_molecular_id::MORDRED_MOLECULAR_ID_LEN,
            Self::MordredInformationContent => super::mordred_information::MORDRED_INFORMATION_CONTENT_LEN,
            Self::MordredEta => super::mordred_eta::MORDRED_ETA_LEN,
            Self::MordredMatrix => super::mordred_matrix::MORDRED_MATRIX_LEN,
            _ => 1,
        }
    }

    /// Output column names for this descriptor, in output order.
    ///
    /// RDKit names are used wherever the value is meant to reproduce an RDKit
    /// descriptor (e.g. `MolWt`, `Chi1v`, `PEOE_VSA3`, `MACCS42`), CDK names
    /// for CDK-derived descriptors (e.g. `C2SP3`, `nAtomP`), otherwise a
    /// descriptive name. Names are unique across [`BatchDescriptor::all`].
    pub fn feature_names(self) -> Vec<String> {
        fn seq(prefix: &str, range: std::ops::RangeInclusive<usize>) -> Vec<String> {
            range.map(|i| format!("{prefix}{i}")).collect()
        }
        fn list(names: &[&str]) -> Vec<String> {
            names.iter().map(|s| s.to_string()).collect()
        }
        let one = |name: &str| vec![name.to_string()];
        match self {
            Self::MolecularWeight => one("MolWt"),
            Self::LogP => one("MolLogP"),
            Self::MolarRefractivity => one("MolMR"),
            Self::Tpsa => one("TPSA"),
            Self::HBondDonor => one("NumHDonors"),
            Self::HBondAcceptor => one("NumHAcceptors"),
            Self::RotatableBondCount => one("NumRotatableBonds"),
            Self::HeavyAtomCount => one("HeavyAtomCount"),
            Self::AtomCount => one("nAtom"),
            Self::ElementCounts => list(&["nC", "nN", "nO", "nS", "nP", "nF", "nCl", "nBr", "nI", "nOtherElements", "nH", "nHeavyAtom"]),
            Self::AromaticAtomCount => one("nAromAtom"),
            Self::HeteroatomCount => one("NumHeteroatoms"),
            Self::AromaticRingCount => one("NumAromaticRings"),
            Self::BondCount => one("nBondsHeavy"),
            Self::BondCountDetailed => list(&["nBonds", "nBondsS", "nBondsD", "nBondsT", "nBondsA"]),
            Self::ValenceElectrons => one("NumValenceElectrons"),
            Self::RadicalElectrons => one("NumRadicalElectrons"),
            Self::HybridizationRatio => one("HybRatio"),
            Self::VAdjMa => one("VAdjMat"),
            Self::FormalCharge => one("FormalCharge"),
            Self::FractionCSP3 => one("FractionCSP3"),
            Self::BertzCT => one("BertzCT"),
            Self::Ipc => one("Ipc"),
            Self::FragmentComplexity => one("ElementEntropy"),
            Self::LabuteAsa => one("LabuteASA"),
            Self::Qed => one("qed"),
            Self::NumAmideBonds => one("NumAmideBonds"),
            Self::HallKierAlpha => one("HallKierAlpha"),
            Self::AromaticAtomFraction => one("AromaticProportion"),
            Self::LipinskiViolations => one("LipinskiViolations"),
            Self::BalabanIndex => one("BalabanJ"),
            Self::WienerIndex => list(&["WienerPath", "WienerPolarity"]),
            Self::ZagrebIndices => list(&["Zagreb1", "Zagreb2"]),
            Self::PetitjeanIndex => list(&["Radius", "Diameter", "PetitjeanNumber"]),
            Self::SmallRingDescriptor => {
                let mut v = list(&["nSmallRings", "nAromRings", "nRingBlocks", "nAromBlocks"]);
                v.extend(seq("nRings", 3..=9));
                v
            }
            Self::RingTypeCounts => list(&["NumSaturatedRings", "NumAliphaticRings", "NumAromaticRingsSSSR"]),
            Self::PiSystemCounts => list(&["LargestPiSystemSize", "nPiSystems"]),
            Self::LargestPiSystem => one("nAtomP"),
            Self::CarbonTypes => list(&["C1SP1", "C2SP1", "C1SP2", "C2SP2", "C3SP2", "C1SP3", "C2SP3", "C3SP3", "C4SP3"]),
            // RDKit connectivity indices
            Self::ChiIndices => list(&["Chi0", "Chi1", "Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v"]),
            // CDK ChiPath / ChiCluster / ChiChain / ChiPathCluster
            Self::CdkChiIndices => {
                let mut v = Vec::with_capacity(40);
                v.extend((0..=7).map(|i| format!("SP-{i}")));
                v.extend((0..=7).map(|i| format!("VP-{i}")));
                v.extend((3..=6).map(|i| format!("SC-{i}")));
                v.extend((3..=6).map(|i| format!("VC-{i}")));
                v.extend((3..=7).map(|i| format!("SCH-{i}")));
                v.extend((3..=7).map(|i| format!("VCH-{i}")));
                v.extend((4..=6).map(|i| format!("SPC-{i}")));
                v.extend((4..=6).map(|i| format!("VPC-{i}")));
                v
            }
            Self::KappaIndices => seq("Kappa", 1..=3),
            Self::EStateIndices => {
                let mut v = list(&["SumEState", "MinEStateIndex", "MaxEStateIndex"]);
                v.extend(super::estate::ESTATE_TYPE_NAMES.iter().map(|t| format!("SEstate_{t}")));
                v
            }
            // CDK ATSm/ATSp/ATSc (lags 0..4) + Mordred Autocorrelation module
            Self::Autocorrelation => {
                let mut v = Vec::with_capacity(super::autocorrelation::N_OUT);
                for w in ["m", "p", "c"] {
                    v.extend(seq(&format!("ATS{w}"), 1..=5));
                }
                v.extend(super::autocorrelation::mordred_feature_names());
                v
            }
            // CDK BCUT (w = mass, c = charge, p = polarizability) + Mordred BCUT module
            Self::Bcut => super::bcut::feature_names(),
            Self::WeightedPath => (1..=5).map(|i| format!("WTPT-{i}")).collect(),
            Self::Mde => list(&[
                "MDEC-11", "MDEC-12", "MDEC-13", "MDEC-14", "MDEC-22", "MDEC-23", "MDEC-24", "MDEC-33", "MDEC-34",
                "MDEC-44", "MDEO-11", "MDEO-12", "MDEO-22", "MDEN-11", "MDEN-12", "MDEN-13", "MDEN-22", "MDEN-23",
                "MDEN-33",
            ]),
            Self::FunctionalGroupCounts => [
                "Hydroxyl", "Aldehyde", "Ketone", "CarboxylicAcid", "Ester", "Amide", "PrimaryAmine",
                "SecondaryAmine", "TertiaryAmine", "Nitro", "Sulfonamide", "Sulfone", "Thiol", "Ether",
                "Nitrile", "Halide",
            ]
            .iter()
            .map(|g| format!("fg_{g}"))
            .collect(),
            Self::AcidicGroupCount => one("nAcidicGroups"),
            Self::BasicGroupCount => one("nBasicGroups"),
            Self::LongestAliphaticChain => one("nAtomLAC"),
            Self::LargestChain => one("nAtomLC"),
            Self::MaxPartialCharge => one("MaxPartialCharge"),
            Self::MinPartialCharge => one("MinPartialCharge"),
            Self::MaxAbsPartialCharge => one("MaxAbsPartialCharge"),
            Self::MinAbsPartialCharge => one("MinAbsPartialCharge"),
            Self::APol => one("apol"),
            Self::BPol => one("bpol"),
            Self::SlogpVsa => seq("SlogP_VSA", 1..=12),
            Self::SmrVsa => seq("SMR_VSA", 1..=10),
            Self::PeoeVsa => seq("PEOE_VSA", 1..=14),
            Self::MaccsKeys => seq("MACCS", 1..=166),
            Self::PubChem881 => seq("PubchemFP", 0..=880),
            Self::Ecfp4 => seq("ECFP4_", 0..=2047),
            Self::Fcfp4 => seq("FCFP4_", 0..=2047),
            Self::AtomPairFingerprint => seq("AtomPair_", 0..=2047),
            Self::TopologicalTorsionFingerprint => seq("TopoTorsion_", 0..=2047),
            Self::Pmi1 => one("PMI1"),
            Self::Pmi2 => one("PMI2"),
            Self::Pmi3 => one("PMI3"),
            Self::Asphericity => one("Asphericity"),
            Self::Biosynfoni => super::biosynfoni::BIOSYNFONI_FEATURES
                .iter()
                .map(|(key, _, _)| format!("Biosynfoni_{key}"))
                .collect(),
            Self::MolWeights => list(&["HeavyAtomMolWt", "ExactMolWt"]),
            Self::ChiNIndices => seq("Chi", 0..=4).into_iter().map(|c| format!("{c}n")).collect(),
            Self::EStateAbs => list(&["MaxAbsEStateIndex", "MinAbsEStateIndex"]),
            Self::EStateVsa => seq("EState_VSA", 1..=11),
            Self::VsaEState => seq("VSA_EState", 1..=10),
            Self::RdkitRingCounts => list(&["RingCount", "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumHeterocycles", "NumSpiroAtoms", "NumBridgeheadAtoms"]),
            Self::LipinskiNO => list(&["NHOHCount", "NOCount"]),
            Self::AvgIpcPhi => list(&["AvgIpc", "Phi"]),
            Self::FpDensityMorgan => seq("FpDensityMorgan", 1..=3),
            Self::Fragments => super::rdkit_fragments_data::FRAGMENTS.iter().map(|(n, _)| n.to_string()).collect(),
            Self::Bcut2D => list(&["BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW"]),
            Self::CdkFragmentComplexity => one("fragC"),
            Self::EccentricConnectivity => one("ECCEN"),
            Self::StereoCenters => list(&["NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters"]),
            Self::Sps => one("SPS"),
            Self::CdkKappa => seq("Kier", 1..=3),
            // Mordred AtomTypeEState: N / S / MAX / MIN per E-State type
            Self::MordredEState => super::mordred_estate::mordred_estate_names(),
            // Mordred MolecularDistanceEdge (plain MDE* names are CDK's)
            Self::MordredMde => super::mordred_mde::MORDRED_MDE_NAMES
                .iter()
                .map(|n| format!("{}{n}", super::mordred_names::MORDRED_PREFIX))
                .collect(),
            Self::MordredMisc => list(&super::mordred_misc::MORDRED_MISC_NAMES),
            Self::MordredConstitutional => super::mordred_constitutional::mordred_constitutional_names(),
            Self::MordredRingCount => super::mordred_rings::mordred_ring_count_names(),
            Self::MordredChi => super::mordred_chi::mordred_chi_names(),
            Self::MordredPathCount => super::mordred_paths::mordred_path_count_names(),
            Self::MordredWalkCount => super::mordred_topological::mordred_walk_count_names(),
            Self::MordredTopologicalCharge => super::mordred_topological::mordred_topological_charge_names(),
            Self::MordredMolecularId => super::mordred_molecular_id::mordred_molecular_id_names(),
            Self::MordredInformationContent => super::mordred_information::mordred_information_content_names(),
            Self::MordredEta => list(&super::mordred_eta::MORDRED_ETA_NAMES),
            // Mordred AdjacencyMatrix, BaryszMatrix, DetourMatrix, DistanceMatrix
            Self::MordredMatrix => super::mordred_matrix::mordred_matrix_names(),
        }
    }

    /// Short name of a fingerprint descriptor (bit-packed output block), or
    /// `None` for scalar descriptors.
    pub fn fingerprint_name(self) -> Option<&'static str> {
        match self {
            Self::MaccsKeys => Some("MACCS"),
            Self::PubChem881 => Some("PubChem"),
            Self::Ecfp4 => Some("ECFP4"),
            Self::Fcfp4 => Some("FCFP4"),
            Self::AtomPairFingerprint => Some("AtomPair"),
            Self::TopologicalTorsionFingerprint => Some("TopoTorsion"),
            _ => None,
        }
    }

    /// Whether this descriptor is a bit fingerprint (see [`Self::fingerprint_name`]).
    pub fn is_fingerprint(self) -> bool {
        self.fingerprint_name().is_some()
    }

    pub fn write_into(
        self,
        mol: &Molecule,
        row: &mut [f64],
        offset: &mut usize,
        ctx: &mut crate::desc::context::CalculationContext,
    ) {
        match self {
            Self::SlogpVsa => {
                let end = *offset + self.feature_len();
                fill_slogp_vsa_into(mol, &mut row[*offset..end], ctx);
                *offset = end;
            }
            Self::SmrVsa => {
                let end = *offset + self.feature_len();
                fill_smr_vsa_into(mol, &mut row[*offset..end], ctx);
                *offset = end;
            }
            Self::PeoeVsa => {
                let end = *offset + self.feature_len();
                fill_peoe_vsa_into(mol, &mut row[*offset..end], ctx);
                *offset = end;
            }
            _ => {
                let value = self.calculate(mol, ctx);
                let expected = self.feature_len();
                if value.len() == expected {
                    write_descriptor_value(value, row, offset);
                } else {
                    // Never shift columns: a descriptor with a wrong output
                    // length yields NaN in its own slot.
                    debug_assert!(false, "{} returned {} values, expected {}", self.name(), value.len(), expected);
                    row[*offset..*offset + expected].fill(f64::NAN);
                    *offset += expected;
                }
            }
        }
    }
}

/// Appends a descriptor value at `offset` (bits as 0.0 / 1.0).
fn write_descriptor_value(value: DescriptorValue, row: &mut [f64], offset: &mut usize) {
    let n = value.to_f64_slice(&mut row[*offset..]);
    *offset += n;
}

/// Column names for a descriptor set, in output order.
pub fn feature_names(descriptors: &[BatchDescriptor]) -> Vec<String> {
    descriptors.iter().flat_map(|d| d.feature_names()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_ml_preset_is_valid_subset_and_excludes_heavy_fingerprints() {
        let preset = BatchDescriptor::fast_ml_preset();
        assert!(!preset.is_empty(), "Fast ML preset must not be empty");

        let all = BatchDescriptor::all();
        for (idx, descriptor) in preset.iter().enumerate() {
            assert!(
                all.contains(descriptor),
                "Fast ML preset contains descriptor that is not in BatchDescriptor::all(): {:?}",
                descriptor
            );
            assert!(
                !preset[idx + 1..].contains(descriptor),
                "Fast ML preset must not contain duplicates: {:?}",
                descriptor
            );
        }

        assert!(
            !preset.iter().any(|descriptor| {
                matches!(
                    descriptor,
                    BatchDescriptor::MaccsKeys
                        | BatchDescriptor::PubChem881
                        | BatchDescriptor::Ecfp4
                        | BatchDescriptor::Fcfp4
                        | BatchDescriptor::AtomPairFingerprint
                        | BatchDescriptor::TopologicalTorsionFingerprint
                )
            }),
            "Fast ML preset should avoid high-dimensional fingerprint families"
        );
    }

    #[test]
    fn feature_names_match_lengths_and_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for &d in BatchDescriptor::all() {
            let names = d.feature_names();
            assert_eq!(names.len(), d.feature_len(), "{}", d.name());
            for n in names {
                assert!(seen.insert(n.clone()), "duplicate feature name {n}");
            }
        }
    }
}
