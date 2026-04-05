//! Descriptor module: Batch.
//!
//! This file implements one descriptor family (or supporting descriptor glue)
//! used in QSAR pipelines. Descriptors convert molecular structure into typed
//! numerical or binary features consumed by ML and analytics layers.
//!
//! Learning focus:
//! - Implementing stateless descriptor structs in Rust.
//! - Mapping chemistry concepts to reproducible numeric features.

//! ## File guide: Batch
//!
//! This file belongs to FerrumChem's **descriptor/fingerprint feature layer**.
//!
//! What to learn here:
//! - Which chemistry/ML concern this file owns in the pipeline.
//! - How Rust types and control-flow encode that concern.
//! - Which public symbols are the best entry points for reading/debugging.
//!
//! Key symbols in this file:
//! - `struct FlatDescriptorMatrix`
//! - `fn shape`
//! - `fn row`
//! - `fn into_nested`
//! - `fn into_shape_vec`
//!
//! Suggested reading workflow:
//! 1. Read top-level type definitions and public APIs.
//! 2. Trace helper functions used in hot/critical paths.
//! 3. Cross-reference tests/examples that exercise this file.

//!
//! ### Algorithmic assumptions for this file
//! - `Batch` assumes molecule preprocessing (implicit hydrogens, aromaticity flags, and ring annotations) is current before calculation.
//! - `Batch` remains deterministic for identical atom/bond indexing so batch and single-molecule runs agree bit-for-bit.
//! - Typed `DescriptorValue` output is part of the contract; downstream ML code should not rely on ad-hoc coercions.
//! - Specific scope anchor: `crates/ferrum_core/src/desc/batch.rs` defines batch behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Todeschini & Consonni. Handbook of Molecular Descriptors (general descriptor taxonomy).
//! - RDKit descriptor/fingerprint APIs used as practical parity baselines for `Batch` workflows.
//! - Repository unit and parity tests covering `Batch` behavior and edge molecules.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/desc/batch.rs`.

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
use crate::error::Result;
use crate::mol::molecule::Molecule;
use rayon::prelude::*;
use std::io::Write;

/// Flat descriptor matrix optimized for ML consumers.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatDescriptorMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl FlatDescriptorMatrix {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn row(&self, idx: usize) -> &[f32] {
        assert!(idx < self.rows, "row index out of bounds");
        if self.cols == 0 {
            return &[];
        }
        let start = idx * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Implements nested.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn into_nested(self) -> Vec<Vec<f32>> {
        if self.cols == 0 {
            return vec![Vec::new(); self.rows];
        }
        self.data
            .chunks(self.cols)
            .map(|row| row.to_vec())
            .collect()
    }

    /// Implements shape vec.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn into_shape_vec(self) -> ((usize, usize), Vec<f32>) {
        ((self.rows, self.cols), self.data)
    }
}

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
}

impl BatchDescriptor {
    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn all() -> &'static [Self] {
        const ALL: [BatchDescriptor; 66] = [
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
        ];
        &ALL
    }

    /// Returns a curated descriptor subset for fast, broad-coverage ML featurization.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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

    /// Implements the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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
        }
    }

    /// Calculates the behavior encoded by this function.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn calculate(
        self,
        mol: &mut Molecule,
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
        }
    }

    /// Implements len.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn feature_len(self) -> usize {
        match self {
            Self::ElementCounts => 12,
            Self::BondCountDetailed => 5,
            Self::SmallRingDescriptor => 11,
            Self::RingTypeCounts => 3,
            Self::PiSystemCounts => 2,
            Self::CarbonTypes => 9,
            Self::ChiIndices => 26,
            Self::KappaIndices => 3,
            Self::EStateIndices => 47,
            Self::Autocorrelation => 32,
            Self::Bcut => 8,
            Self::WeightedPath => 10,
            Self::Mde => 15,
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
            _ => 1,
        }
    }

    /// Writes into.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    pub fn write_into(
        self,
        mol: &mut Molecule,
        row: &mut [f32],
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
            _ => write_descriptor_value(self.calculate(mol, ctx), row, offset),
        }
    }
}

/// Writes descriptor value.
///
/// Function context: `crates/ferrum_core/src/desc/batch.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn write_descriptor_value(value: DescriptorValue, row: &mut [f32], offset: &mut usize) {
    match value {
        DescriptorValue::Int(v) => {
            row[*offset] = v as f32;
            *offset += 1;
        }
        DescriptorValue::Float(v) => {
            row[*offset] = v;
            *offset += 1;
        }
        DescriptorValue::Float64(v) => {
            row[*offset] = v as f32;
            *offset += 1;
        }
        DescriptorValue::IntVec(v) => {
            for value in v {
                row[*offset] = value as f32;
                *offset += 1;
            }
        }
        DescriptorValue::IntU16Vec(v) => {
            for value in v {
                row[*offset] = value as f32;
                *offset += 1;
            }
        }
        DescriptorValue::FloatVec(v) => {
            let end = *offset + v.len();
            row[*offset..end].copy_from_slice(&v);
            *offset = end;
        }
        DescriptorValue::Binary(bits) => {
            for bit in bits.iter().by_vals() {
                row[*offset] = if bit { 1.0 } else { 0.0 };
                *offset += 1;
            }
        }
    }
}

/// Implements cols dyn.
///
/// Function context: `crates/ferrum_core/src/desc/batch.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn infer_cols_dyn(probe: &mut Molecule, descriptors: &[Box<dyn Descriptor>]) -> usize {
    let mut ctx = crate::desc::context::CalculationContext::new();
    descriptors
        .iter()
        .map(|desc| desc.calculate(probe, &mut ctx).len())
        .sum()
}

/// Implements cols static.
///
/// Function context: `crates/ferrum_core/src/desc/batch.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn infer_cols_static(descriptors: &[BatchDescriptor]) -> usize {
    descriptors.iter().map(|desc| desc.feature_len()).sum()
}

/// Implements row dyn.
///
/// Function context: `crates/ferrum_core/src/desc/batch.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn fill_row_dyn(
    row: &mut [f32],
    mol: &mut Molecule,
    descriptors: &[Box<dyn Descriptor>],
    ctx: &mut crate::desc::context::CalculationContext,
) {
    let mut offset = 0usize;
    for desc in descriptors {
        write_descriptor_value(desc.calculate(mol, ctx), row, &mut offset);
    }
    debug_assert_eq!(offset, row.len());
}

/// Implements row static.
///
/// Function context: `crates/ferrum_core/src/desc/batch.rs`.
/// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
fn fill_row_static(
    row: &mut [f32],
    mol: &mut Molecule,
    descriptors: &[BatchDescriptor],
    ctx: &mut crate::desc::context::CalculationContext,
) {
    let mut offset = 0usize;
    for &desc in descriptors {
        desc.write_into(mol, row, &mut offset, ctx);
    }
    debug_assert_eq!(offset, row.len());
}

/// Calculates descriptors for many molecules in parallel into one contiguous buffer.
pub fn calculate_batch_flat(
    molecules: &mut [Molecule],
    descriptors: &[Box<dyn Descriptor>],
) -> FlatDescriptorMatrix {
    let rows = molecules.len();
    if rows == 0 || descriptors.is_empty() {
        return FlatDescriptorMatrix {
            rows,
            cols: 0,
            data: Vec::new(),
        };
    }

    let mut probe = molecules[0].clone();
    let cols = infer_cols_dyn(&mut probe, descriptors);
    if cols == 0 {
        return FlatDescriptorMatrix {
            rows,
            cols,
            data: Vec::new(),
        };
    }

    let mut data = vec![0.0f32; rows * cols];
    data.par_chunks_mut(cols)
        .zip(molecules.par_iter_mut())
        .for_each_init(
            crate::desc::context::CalculationContext::new,
            |ctx, (row, mol)| {
                ctx.reset_for_new_molecule();
                fill_row_dyn(row, mol, descriptors, ctx);
            },
        );

    FlatDescriptorMatrix { rows, cols, data }
}

/// Calculates static-dispatch descriptors for many molecules in parallel into one contiguous buffer.
pub fn calculate_batch_flat_static(
    molecules: &mut [Molecule],
    descriptors: &[BatchDescriptor],
) -> FlatDescriptorMatrix {
    let rows = molecules.len();
    if rows == 0 || descriptors.is_empty() {
        return FlatDescriptorMatrix {
            rows,
            cols: 0,
            data: Vec::new(),
        };
    }

    let cols = infer_cols_static(descriptors);
    if cols == 0 {
        return FlatDescriptorMatrix {
            rows,
            cols,
            data: Vec::new(),
        };
    }

    let mut data = vec![0.0f32; rows * cols];
    data.par_chunks_mut(cols)
        .zip(molecules.par_iter_mut())
        .for_each_init(
            crate::desc::context::CalculationContext::new,
            |ctx, (row, mol)| {
                ctx.reset_for_new_molecule();
                fill_row_static(row, mol, descriptors, ctx);
            },
        );

    FlatDescriptorMatrix { rows, cols, data }
}

/// Calculates static-dispatch descriptors and allows cooperative cancellation between rows.
pub fn calculate_batch_flat_static_cancellable<F>(
    molecules: &mut [Molecule],
    descriptors: &[BatchDescriptor],
    should_cancel: &F,
) -> std::result::Result<FlatDescriptorMatrix, ()>
where
    F: Fn() -> bool + Sync,
{
    if should_cancel() {
        return Err(());
    }

    let rows = molecules.len();
    if rows == 0 || descriptors.is_empty() {
        return Ok(FlatDescriptorMatrix {
            rows,
            cols: 0,
            data: Vec::new(),
        });
    }

    let cols = infer_cols_static(descriptors);
    if cols == 0 {
        return Ok(FlatDescriptorMatrix {
            rows,
            cols,
            data: Vec::new(),
        });
    }

    let mut data = vec![0.0f32; rows * cols];
    data.par_chunks_mut(cols)
        .zip(molecules.par_iter_mut())
        .try_for_each_init(
            crate::desc::context::CalculationContext::new,
            |ctx, (row, mol)| {
                if should_cancel() {
                    return Err(());
                }
                ctx.reset_for_new_molecule();
                fill_row_static(row, mol, descriptors, ctx);
                if should_cancel() {
                    return Err(());
                }
                Ok(())
            },
        )?;

    if should_cancel() {
        return Err(());
    }

    Ok(FlatDescriptorMatrix { rows, cols, data })
}

/// Compatibility API: calculates descriptors in parallel and returns row-major vectors.
pub fn calculate_batch(
    molecules: &mut [Molecule],
    descriptors: &[Box<dyn Descriptor>],
) -> Vec<Vec<f32>> {
    calculate_batch_flat(molecules, descriptors).into_nested()
}

/// Writes descriptor matrix to CSV format.
pub fn write_csv<W: Write>(
    writer: &mut W,
    smiles: &[String],
    descriptors: &[Box<dyn Descriptor>],
    matrix: &[Vec<f32>],
) -> Result<()> {
    write!(writer, "SMILES")?;
    for desc in descriptors {
        write!(writer, ",{}", desc.name())?;
    }
    writeln!(writer)?;

    for (smiles_str, features) in smiles.iter().zip(matrix.iter()) {
        write!(writer, "{}", smiles_str)?;
        for &value in features {
            write!(writer, ",{:.6}", value)?;
        }
        writeln!(writer)?;
    }

    Ok(())
}

/// Writes flattened descriptor matrix to CSV format.
pub fn write_csv_flat<W: Write>(
    writer: &mut W,
    smiles: &[String],
    feature_names: &[String],
    matrix: &FlatDescriptorMatrix,
) -> Result<()> {
    write!(writer, "SMILES")?;
    for name in feature_names {
        write!(writer, ",{}", name)?;
    }
    writeln!(writer)?;

    for (row_idx, smiles_str) in smiles.iter().enumerate().take(matrix.rows) {
        write!(writer, "{}", smiles_str)?;
        for &value in matrix.row(row_idx) {
            write!(writer, ",{:.6}", value)?;
        }
        writeln!(writer)?;
    }

    Ok(())
}

/// Writes descriptor matrix to JSON format.
pub fn write_json<W: Write>(
    writer: &mut W,
    smiles: &[String],
    descriptors: &[Box<dyn Descriptor>],
    matrix: &[Vec<f32>],
) -> Result<()> {
    use serde_json::json;

    let descriptor_names: Vec<String> = descriptors.iter().map(|d| d.name().to_string()).collect();

    let mut records = Vec::new();
    for (smiles_str, features) in smiles.iter().zip(matrix.iter()) {
        let mut record = serde_json::Map::new();
        record.insert("smiles".to_string(), json!(smiles_str));

        for (name, &value) in descriptor_names.iter().zip(features.iter()) {
            record.insert(name.clone(), json!(value));
        }

        records.push(json!(record));
    }

    let output = json!({
        "molecules": records,
        "descriptor_count": descriptor_names.len(),
        "molecule_count": smiles.len(),
    });

    serde_json::to_writer_pretty(&mut *writer, &output)?;
    writeln!(writer)?;

    Ok(())
}

/// Writes flattened descriptor matrix to JSON format.
pub fn write_json_flat<W: Write>(
    writer: &mut W,
    smiles: &[String],
    feature_names: &[String],
    matrix: &FlatDescriptorMatrix,
) -> Result<()> {
    use serde_json::json;

    let mut records = Vec::new();
    for (row_idx, smiles_str) in smiles.iter().enumerate().take(matrix.rows) {
        let mut record = serde_json::Map::new();
        record.insert("smiles".to_string(), json!(smiles_str));
        for (name, &value) in feature_names.iter().zip(matrix.row(row_idx)) {
            record.insert(name.clone(), json!(value));
        }
        records.push(json!(record));
    }

    let output = json!({
        "molecules": records,
        "feature_count": feature_names.len(),
        "molecule_count": smiles.len(),
    });

    serde_json::to_writer_pretty(&mut *writer, &output)?;
    writeln!(writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::molecular_weight::MolecularWeight;
    use crate::io::smiles::parse_smiles;

    #[test]
    /// Runs regression test for calculate batch.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_calculate_batch() {
        let mut molecules = vec![
            parse_smiles("C").unwrap(),
            parse_smiles("CC").unwrap(),
            parse_smiles("CCC").unwrap(),
        ];

        let descriptors: Vec<Box<dyn Descriptor>> = vec![Box::new(MolecularWeight)];
        let results = calculate_batch(&mut molecules, &descriptors);

        assert_eq!(results.len(), 3);
        assert!(!results[0].is_empty());
    }

    #[test]
    /// Runs regression test for calculate batch flat shape.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_calculate_batch_flat_shape() {
        let mut molecules = vec![parse_smiles("C").unwrap(), parse_smiles("CC").unwrap()];
        let descriptors: Vec<Box<dyn Descriptor>> = vec![Box::new(MolecularWeight)];
        let matrix = calculate_batch_flat(&mut molecules, &descriptors);

        assert_eq!(matrix.shape(), (2, 1));
        assert_eq!(matrix.data.len(), 2);
    }

    #[test]
    /// Runs regression test for calculate batch flat static shape.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_calculate_batch_flat_static_shape() {
        let mut molecules = vec![parse_smiles("C").unwrap(), parse_smiles("CC").unwrap()];
        let descriptors = [BatchDescriptor::MolecularWeight, BatchDescriptor::AtomCount];
        let matrix = calculate_batch_flat_static(&mut molecules, &descriptors);

        assert_eq!(matrix.shape(), (2, 2));
        assert_eq!(matrix.data.len(), 4);
    }

    #[test]
    /// Runs regression test for calculate batch flat static cancellable honors cancel flag.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_calculate_batch_flat_static_cancellable_honors_cancel_flag() {
        let mut molecules = vec![parse_smiles("C").unwrap(); 128];
        let descriptors = [BatchDescriptor::MolecularWeight];
        let canceled = std::sync::atomic::AtomicBool::new(true);

        let result = calculate_batch_flat_static_cancellable(&mut molecules, &descriptors, &|| {
            canceled.load(std::sync::atomic::Ordering::Relaxed)
        });

        assert!(
            result.is_err(),
            "Expected cancellation to stop batch calculation"
        );
    }

    #[test]
    /// Runs regression test for fast ml preset descriptor list consistency.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
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
    /// Runs regression test for static feature len matches runtime value len on torture set.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_static_feature_len_matches_runtime_value_len_on_torture_set() {
        let torture_smiles = [
            "c1ccc2ccccc2c1",
            "c1ccc2cccc2c1",
            "c1ccc2[nH]ccc2c1",
            "c1ccc2ncccc2c1",
            "c1ccc2cnccc2c1",
            "c1nc2c(n1)nc[nH]2",
            "C1=CC=CC=CC=C1",
            "[cH-]1cccc1",
            "c1cc[nH]c1",
            "c1ccoc1",
            "c1ccsc1",
            "CC[NH3+]",
            "C[N+](C)(C)C",
            "[O-]C(=O)C",
            "O=[N+](O)c1ccccc1",
            "CC(C)(C)C",
            "C12C3C4C1C5C4C3C25",
            "[13CH3]C([2H])([2H])[2H]",
            "C[C@H](O)C(=O)O",
            "C[C@@H](N)C(=O)O",
            "F/C=C/F",
            "F/C=C\\F",
            "C1CCCCC1",
            "CCCCC",
            "c1ccc2c(c1)ccc3ccccc23",
            "[NH4+].[Cl-]",
            "c1cccc2c1cccc2",
            "[O-][N+](=O)c1ccccc1",
            "c1[nH+]cccn1",
            "C[NH+](C)C",
        ];
        let mut molecules = torture_smiles
            .iter()
            .map(|s| parse_smiles(s).expect("valid torture SMILES"))
            .collect::<Vec<_>>();
        assert!(
            !molecules.is_empty(),
            "torture_set.csv produced no molecules"
        );

        let descriptors = BatchDescriptor::all().to_vec();
        let static_cols = descriptors.iter().map(|d| d.feature_len()).sum::<usize>();
        let mut probe = molecules[0].clone();
        let mut ctx = crate::desc::context::CalculationContext::new();
        let runtime_cols = descriptors
            .iter()
            .map(|d| d.calculate(&mut probe, &mut ctx).len())
            .sum::<usize>();
        assert_eq!(
            static_cols, runtime_cols,
            "BatchDescriptor::feature_len sum must match runtime descriptor value lengths"
        );

        let matrix = calculate_batch_flat_static(&mut molecules, &descriptors);
        assert_eq!(matrix.cols, static_cols);
        assert_eq!(matrix.data.len(), matrix.rows * matrix.cols);
    }

    #[test]
    /// Runs regression test for write csv.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_write_csv() {
        let smiles = vec!["C".to_string(), "CC".to_string()];
        let descriptors: Vec<Box<dyn Descriptor>> = vec![Box::new(MolecularWeight)];
        let matrix = vec![vec![16.0], vec![30.0]];

        let mut output = Vec::new();
        write_csv(&mut output, &smiles, &descriptors, &matrix).unwrap();

        let csv_string = String::from_utf8(output).unwrap();
        assert!(csv_string.contains("SMILES,MolecularWeight"));
    }

    #[test]
    /// Runs regression test for write csv flat.
    ///
    /// Function context: `crates/ferrum_core/src/desc/batch.rs`.
    /// Descriptor-path note: keep deterministic chemistry semantics so parity and batch runs stay reproducible.
    fn test_write_csv_flat() {
        let smiles = vec!["C".to_string(), "CC".to_string()];
        let feature_names = vec!["MolecularWeight".to_string()];
        let matrix = FlatDescriptorMatrix {
            rows: 2,
            cols: 1,
            data: vec![16.0, 30.0],
        };

        let mut output = Vec::new();
        write_csv_flat(&mut output, &smiles, &feature_names, &matrix).unwrap();
        let csv_string = String::from_utf8(output).unwrap();
        assert!(csv_string.contains("SMILES,MolecularWeight"));
        assert!(csv_string.contains("CC,30.000000"));
    }
}
