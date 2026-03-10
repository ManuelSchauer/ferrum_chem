// Public modules
pub mod aromatic_ring_count;
pub mod atom_count;
pub mod autocorrelation;
pub mod balaban;
pub mod batch;
pub mod bcut;
pub mod bond_count;
pub mod carbon_types;
pub mod chains;
pub mod chi;
pub mod eccentric;
pub mod estate;
pub mod extra_counts;
pub mod fingerprint;
pub mod formal_charge;
pub mod functional_groups;
pub mod gasteiger_charge;
pub mod gravitational;
pub mod hbond;
pub mod kappa;
pub mod lipinski;
pub mod logp;
pub mod maccs;
pub mod mde;
pub mod molecular_weight;
pub mod path_fingerprints;
pub mod petitjean;
pub mod pi_system;
pub mod polarizability;
pub mod pubchem;
pub mod qed;
pub mod ring_types;
pub mod simple_counts;
pub mod small_ring;
pub mod spiro;
pub mod topo;
pub mod tpsa;
pub mod typed;
pub mod vsa;
pub mod weighted_path;
pub mod wiener_numbers;
pub mod zagreb;

// Re-export the core descriptor API
pub use typed::{
    BitVector, Descriptor, DescriptorValue, build_ml_vector, build_ml_vectors_parallel,
};

#[cfg(test)]
mod descriptor_tests {
    use super::typed::{Descriptor, DescriptorValue};
    use crate::io::smiles::parse_smiles;

    fn parse(smiles: &str) -> crate::mol::molecule::Molecule {
        parse_smiles(smiles).expect("valid SMILES")
    }

    fn calc_float(desc: &dyn Descriptor, smiles: &str) -> f32 {
        let mut mol = parse(smiles);
        match desc.calculate(&mut mol) {
            DescriptorValue::Float(v) => v,
            DescriptorValue::Float64(v) => v as f32,
            other => panic!("{} expected Float, got {:?}", desc.name(), other),
        }
    }

    fn calc_int(desc: &dyn Descriptor, smiles: &str) -> u32 {
        let mut mol = parse(smiles);
        match desc.calculate(&mut mol) {
            DescriptorValue::Int(v) => v,
            other => panic!("{} expected Int, got {:?}", desc.name(), other),
        }
    }

    fn calc_float_vec(desc: &dyn Descriptor, smiles: &str) -> Vec<f32> {
        let mut mol = parse(smiles);
        match desc.calculate(&mut mol) {
            DescriptorValue::FloatVec(v) => v,
            other => panic!("{} expected FloatVec, got {:?}", desc.name(), other),
        }
    }

    fn calc_u16_vec(desc: &dyn Descriptor, smiles: &str) -> Vec<u16> {
        let mut mol = parse(smiles);
        match desc.calculate(&mut mol) {
            DescriptorValue::IntU16Vec(v) => v,
            other => panic!("{} expected IntU16Vec, got {:?}", desc.name(), other),
        }
    }

    fn calc_binary_len(desc: &dyn Descriptor, smiles: &str) -> usize {
        let mut mol = parse(smiles);
        match desc.calculate(&mut mol) {
            DescriptorValue::Binary(bv) => bv.len(),
            other => panic!("{} expected Binary, got {:?}", desc.name(), other),
        }
    }

    // ======================================================================
    // Basic Properties (4)
    // ======================================================================

    #[test]
    fn test_molecular_weight() {
        use super::molecular_weight::MolecularWeight;
        let mw = calc_float(&MolecularWeight, "C");
        assert!((mw - 16.04).abs() < 0.1, "methane MW = {}", mw);
        let mw = calc_float(&MolecularWeight, "c1ccccc1");
        assert!((mw - 78.11).abs() < 0.1, "benzene MW = {}", mw);
    }

    #[test]
    fn test_logp() {
        use super::logp::LogP;
        let v = calc_float(&LogP, "c1ccccc1");
        assert!(v > 1.0 && v < 3.0, "benzene logP = {}", v);
    }

    #[test]
    fn test_molar_refractivity() {
        use super::logp::MolarRefractivity;
        let v = calc_float(&MolarRefractivity, "c1ccccc1");
        assert!(v > 20.0, "benzene MR = {}", v);
    }

    #[test]
    fn test_tpsa() {
        use super::tpsa::Tpsa;
        let v = calc_float(&Tpsa, "c1ccccc1");
        assert!(v.abs() < 0.01, "benzene TPSA = {}", v);
        let v = calc_float(&Tpsa, "CC(=O)O");
        assert!(v > 30.0, "acetic acid TPSA = {}", v);
    }

    // ======================================================================
    // Atom/Bond Counts (9)
    // ======================================================================

    #[test]
    fn test_atom_count() {
        use super::atom_count::AtomCount;
        let c = calc_int(&AtomCount, "CCO");
        assert_eq!(c, 9, "ethanol: 2C + 1O + 6H = 9");
    }

    #[test]
    fn test_heavy_atom_count() {
        use super::atom_count::HeavyAtomCount;
        assert_eq!(calc_int(&HeavyAtomCount, "CCO"), 3);
    }

    #[test]
    fn test_bond_count() {
        use super::bond_count::BondCount;
        assert_eq!(calc_int(&BondCount, "CC"), 1);
    }

    #[test]
    fn test_bond_count_detailed() {
        use super::bond_count::BondCountDetailed;
        let mut mol = parse("c1ccccc1");
        match BondCountDetailed.calculate(&mut mol) {
            DescriptorValue::IntVec(v) => {
                assert_eq!(v.len(), 5, "should have 5 bond type counts");
                assert!(v[4] > 0, "benzene should have aromatic bonds");
            }
            other => panic!("expected IntVec, got {:?}", other),
        }
    }

    #[test]
    fn test_rotatable_bond_count() {
        use super::bond_count::RotatableBondCount;
        assert_eq!(calc_int(&RotatableBondCount, "C"), 0);
        assert!(calc_int(&RotatableBondCount, "CCCCC") > 0);
    }

    #[test]
    fn test_aromatic_ring_count() {
        use super::aromatic_ring_count::AromaticRingCount;
        assert_eq!(calc_int(&AromaticRingCount, "c1ccccc1"), 1);
        assert_eq!(calc_int(&AromaticRingCount, "CCCC"), 0);
    }

    #[test]
    fn test_element_counts() {
        use super::atom_count::ElementCounts;
        let v = calc_u16_vec(&ElementCounts, "CCO");
        assert_eq!(v[0], 2, "2 carbons");
        assert_eq!(v[2], 1, "1 oxygen");
    }

    #[test]
    fn test_heteroatom_count() {
        use super::simple_counts::HeteroatomCount;
        assert_eq!(calc_int(&HeteroatomCount, "c1ccccc1"), 0);
        assert_eq!(calc_int(&HeteroatomCount, "c1ccncc1"), 1);
    }

    #[test]
    fn test_valence_electrons() {
        use super::simple_counts::ValenceElectrons;
        let v = calc_float(&ValenceElectrons, "CC");
        assert!((v - 14.0).abs() < 0.1, "ethane VE = {}", v);
    }

    // ======================================================================
    // Hydrogen Bonding (2)
    // ======================================================================

    #[test]
    fn test_hbond_donor() {
        use super::hbond::HBondDonor;
        assert_eq!(calc_int(&HBondDonor, "CCO"), 1);
        assert_eq!(calc_int(&HBondDonor, "c1ccccc1"), 0);
    }

    #[test]
    fn test_hbond_acceptor() {
        use super::hbond::HBondAcceptor;
        assert!(calc_int(&HBondAcceptor, "CCO") >= 1);
    }

    // ======================================================================
    // Drug-Likeness (3)
    // ======================================================================

    #[test]
    fn test_lipinski_violations() {
        use super::lipinski::LipinskiViolations;
        let v = calc_int(&LipinskiViolations, "CCO");
        assert_eq!(v, 0, "ethanol should have 0 Lipinski violations");
    }

    #[test]
    fn test_qed() {
        use super::qed::Qed;
        let v = calc_float(&Qed, "c1ccccc1");
        assert!(v > 0.0 && v <= 1.0, "QED should be in (0,1]: {}", v);
    }

    #[test]
    fn test_fraction_csp3() {
        use super::extra_counts::FractionCSP3;
        let v = calc_float(&FractionCSP3, "CC");
        assert!((v - 1.0).abs() < 0.01, "ethane all sp3: {}", v);
        let v = calc_float(&FractionCSP3, "c1ccccc1");
        assert!(v.abs() < 0.01, "benzene no sp3: {}", v);
    }

    // ======================================================================
    // Ring Analysis (3)
    // ======================================================================

    #[test]
    fn test_small_ring_descriptor() {
        use super::small_ring::SmallRingDescriptor;
        let v = calc_u16_vec(&SmallRingDescriptor, "c1ccccc1");
        assert_eq!(v.len(), 11);
        assert!(v[3] >= 1, "benzene has a 6-membered ring"); // index 3 = size 6
    }

    #[test]
    fn test_ring_type_counts() {
        use super::ring_types::RingTypeCounts;
        let v = calc_u16_vec(&RingTypeCounts, "c1ccccc1");
        assert_eq!(v.len(), 3);
        assert!(v[2] >= 1, "benzene has aromatic ring"); // [saturated, aliphatic, aromatic]
    }

    #[test]
    fn test_pi_system_counts() {
        use super::pi_system::PiSystemCounts;
        let v = calc_u16_vec(&PiSystemCounts, "c1ccccc1");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], 6, "benzene pi system size = 6");
    }

    // ======================================================================
    // Carbon Classification (2)
    // ======================================================================

    #[test]
    fn test_carbon_types() {
        use super::carbon_types::CarbonTypes;
        let v = calc_u16_vec(&CarbonTypes, "CC");
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_hybridization_ratio() {
        use super::simple_counts::HybridizationRatio;
        let v = calc_float(&HybridizationRatio, "c1ccccc1");
        assert!((v - 0.0).abs() < 0.01, "benzene: 0 sp3/sp2 = 0");
    }

    // ======================================================================
    // Connectivity Indices (2)
    // ======================================================================

    #[test]
    fn test_chi_indices() {
        use super::chi::ChiIndices;
        let v = calc_float_vec(&ChiIndices, "c1ccccc1");
        assert_eq!(v.len(), 26, "chi indices should have 26 values");
        assert!(
            v.iter().any(|&x| x > 0.0),
            "benzene should have nonzero chi"
        );
    }

    #[test]
    fn test_kappa_indices() {
        use super::kappa::KappaIndices;
        let v = calc_float_vec(&KappaIndices, "c1ccccc1");
        assert_eq!(v.len(), 3, "kappa has 3 values");
        assert!(v[0] > 0.0, "kappa1 > 0 for benzene");
    }

    // ======================================================================
    // E-State (1)
    // ======================================================================

    #[test]
    fn test_estate_indices() {
        use super::estate::EStateIndices;
        let v = calc_float_vec(&EStateIndices, "CCO");
        assert_eq!(v.len(), 47, "E-State has 47 values");
    }

    // ======================================================================
    // Autocorrelation / BCUT / Weighted Path / MDE (4)
    // ======================================================================

    #[test]
    fn test_autocorrelation() {
        use super::autocorrelation::Autocorrelation;
        let v = calc_float_vec(&Autocorrelation, "c1ccccc1");
        assert_eq!(v.len(), 32, "ATS has 32 values");
    }

    #[test]
    fn test_bcut() {
        use super::bcut::BCUT;
        let v = calc_float_vec(&BCUT, "c1ccccc1");
        assert_eq!(v.len(), 8, "BCUT has 8 values");
    }

    #[test]
    fn test_weighted_path() {
        use super::weighted_path::WeightedPath;
        let v = calc_float_vec(&WeightedPath, "c1ccccc1");
        assert_eq!(v.len(), 10, "WeightedPath has 10 values");
    }

    #[test]
    fn test_mde() {
        use super::mde::MDE;
        let v = calc_float_vec(&MDE, "c1ccccc1");
        assert_eq!(v.len(), 15, "MDE has 15 values");
    }

    // ======================================================================
    // Topological Indices (3)
    // ======================================================================

    #[test]
    fn test_wiener_index() {
        use super::wiener_numbers::WienerIndex;
        let v = calc_float(&WienerIndex, "CCCC");
        assert!(v > 0.0, "wiener index > 0 for butane: {}", v);
    }

    #[test]
    fn test_zagreb_indices() {
        use super::zagreb::ZagrebIndices;
        let v = calc_float_vec(&ZagrebIndices, "c1ccccc1");
        assert_eq!(v.len(), 2, "zagreb has 2 values");
        assert!(v[0] > 0.0, "M1 > 0 for benzene");
    }

    #[test]
    fn test_balaban_index() {
        use super::balaban::BalabanIndex;
        let v = calc_float(&BalabanIndex, "c1ccccc1");
        assert!(v > 0.0, "Balaban J > 0: {}", v);
    }

    // ======================================================================
    // Shape Descriptors (1)
    // ======================================================================

    #[test]
    fn test_petitjean_index() {
        use super::petitjean::PetitjeanIndex;
        let v = calc_float_vec(&PetitjeanIndex, "CCCC");
        assert_eq!(v.len(), 3, "Petitjean has 3 values");
    }

    // ======================================================================
    // VSA (3)
    // ======================================================================

    #[test]
    fn test_slogp_vsa() {
        use super::vsa::SlogpVSA;
        let v = calc_float_vec(&SlogpVSA, "c1ccccc1");
        assert_eq!(v.len(), 12, "SlogpVSA has 12 bins");
    }

    #[test]
    fn test_smr_vsa() {
        use super::vsa::SmrVSA;
        let v = calc_float_vec(&SmrVSA, "c1ccccc1");
        assert_eq!(v.len(), 10, "SmrVSA has 10 bins");
    }

    #[test]
    fn test_labute_asa() {
        use super::vsa::LabuteASA;
        let v = calc_float(&LabuteASA, "c1ccccc1");
        assert!(v > 0.0, "LabuteASA > 0: {}", v);
    }

    // ======================================================================
    // Complexity (2)
    // ======================================================================

    #[test]
    fn test_bertz_ct() {
        use super::extra_counts::BertzCT;
        let simple = calc_float(&BertzCT, "CCCC");
        let complex = calc_float(&BertzCT, "c1ccc(O)c(N)c1");
        assert!(complex > simple, "complex {} > simple {}", complex, simple);
    }

    #[test]
    fn test_ipc() {
        use super::extra_counts::Ipc;
        let v = calc_float(&Ipc, "CC=O");
        assert!(v > 0.0, "acetaldehyde Ipc > 0: {}", v);
    }

    // ======================================================================
    // Fingerprints (6)
    // ======================================================================

    #[test]
    fn test_maccs_keys() {
        use super::maccs::MaccsKeys;
        let len = calc_binary_len(&MaccsKeys, "c1ccccc1");
        assert_eq!(len, 166, "MACCS has 166 bits");
    }

    #[test]
    fn test_pubchem_881() {
        use super::pubchem::PubChem881;
        let len = calc_binary_len(&PubChem881, "c1ccccc1");
        assert_eq!(len, 881, "PubChem has 881 bits");
    }

    #[test]
    fn test_ecfp() {
        use super::fingerprint::ECFP;
        let desc = ECFP {
            radius: 2,
            num_bits: 2048,
        };
        let len = calc_binary_len(&desc, "c1ccccc1");
        assert_eq!(len, 2048, "ECFP4 has 2048 bits");
    }

    #[test]
    fn test_fcfp() {
        use super::fingerprint::FCFP;
        let desc = FCFP {
            radius: 2,
            num_bits: 2048,
        };
        let len = calc_binary_len(&desc, "c1ccccc1");
        assert_eq!(len, 2048, "FCFP4 has 2048 bits");
    }

    #[test]
    fn test_atom_pair_fingerprint() {
        use super::path_fingerprints::AtomPairFingerprint;
        let desc = AtomPairFingerprint { num_bits: 2048 };
        let len = calc_binary_len(&desc, "c1ccccc1");
        assert_eq!(len, 2048);
    }

    #[test]
    fn test_topological_torsion() {
        use super::path_fingerprints::TopologicalTorsionFingerprint;
        let desc = TopologicalTorsionFingerprint { num_bits: 2048 };
        let len = calc_binary_len(&desc, "c1ccccc1");
        assert_eq!(len, 2048);
    }

    // ======================================================================
    // New descriptors from this session (16 new typed)
    // ======================================================================

    #[test]
    fn test_num_amide_bonds() {
        use super::extra_counts::NumAmideBonds;
        assert_eq!(calc_int(&NumAmideBonds, "CC(=O)N"), 1);
        assert_eq!(calc_int(&NumAmideBonds, "CC"), 0);
    }

    #[test]
    fn test_aromatic_atom_fraction() {
        use super::extra_counts::AromaticAtomFraction;
        let v = calc_float(&AromaticAtomFraction, "c1ccccc1");
        assert!((v - 1.0).abs() < 0.01, "benzene all aromatic: {}", v);
        let v = calc_float(&AromaticAtomFraction, "CCCCCC");
        assert!(v.abs() < 0.01, "hexane no aromatic: {}", v);
    }

    #[test]
    fn test_hall_kier_alpha() {
        use super::extra_counts::HallKierAlpha;
        let v = calc_float(&HallKierAlpha, "CCCC");
        assert!(v.abs() < 0.01, "all-carbon alpha ≈ 0: {}", v);
        let v = calc_float(&HallKierAlpha, "CCBr");
        assert!(v > 0.0, "Br makes alpha positive: {}", v);
    }

    #[test]
    fn test_fragment_complexity() {
        use super::extra_counts::FragmentComplexity;
        let v = calc_float(&FragmentComplexity, "CCCC");
        assert!(v.abs() < 0.01, "all-carbon FC ≈ 0: {}", v);
        let v = calc_float(&FragmentComplexity, "CCNO");
        assert!(v > 0.0, "diverse FC > 0: {}", v);
    }

    #[test]
    fn test_radical_electrons() {
        use super::simple_counts::RadicalElectrons;
        assert_eq!(calc_int(&RadicalElectrons, "CC"), 0);
    }

    #[test]
    fn test_vadjma() {
        use super::simple_counts::VAdjMa;
        let v = calc_float(&VAdjMa, "C");
        assert!(v.abs() < 0.01, "methane VAdjMa = 0: {}", v);
        let v = calc_float(&VAdjMa, "c1ccccc1");
        assert!((v - 3.585).abs() < 0.01, "benzene VAdjMa ≈ 3.58: {}", v);
    }

    #[test]
    fn test_functional_group_counts() {
        use super::functional_groups::FunctionalGroupCounts;
        let v = calc_u16_vec(&FunctionalGroupCounts, "CCO");
        assert_eq!(v.len(), 16, "16 functional groups");
        assert!(v[0] >= 1, "ethanol has hydroxyl");
    }

    #[test]
    fn test_acidic_group_count() {
        use super::functional_groups::AcidicGroupCount;
        assert!(calc_int(&AcidicGroupCount, "CC(=O)O") >= 1);
        assert_eq!(calc_int(&AcidicGroupCount, "CC"), 0);
    }

    #[test]
    fn test_basic_group_count() {
        use super::functional_groups::BasicGroupCount;
        assert!(calc_int(&BasicGroupCount, "CN") >= 1);
        assert_eq!(calc_int(&BasicGroupCount, "c1ccccc1"), 0);
    }

    #[test]
    fn test_formal_charge() {
        use super::formal_charge::FormalCharge;
        let v = calc_float(&FormalCharge, "CCO");
        assert!((v - 0.0).abs() < 0.01, "ethanol neutral: {}", v);
        let v = calc_float(&FormalCharge, "[NH4+]");
        assert!((v - 1.0).abs() < 0.01, "ammonium +1: {}", v);
        let v = calc_float(&FormalCharge, "[OH-]");
        assert!((v - (-1.0)).abs() < 0.01, "hydroxide -1: {}", v);
    }

    #[test]
    fn test_longest_aliphatic_chain() {
        use super::chains::LongestAliphaticChain;
        assert_eq!(calc_int(&LongestAliphaticChain, "CCCCCC"), 6);
        assert_eq!(calc_int(&LongestAliphaticChain, "c1ccccc1"), 0);
    }

    #[test]
    fn test_largest_chain() {
        use super::chains::LargestChain;
        assert_eq!(calc_int(&LargestChain, "CCCCCC"), 6);
        assert_eq!(calc_int(&LargestChain, "c1ccccc1"), 6);
    }

    #[test]
    fn test_aromatic_atom_count() {
        use super::atom_count::AromaticAtomCount;
        assert_eq!(calc_int(&AromaticAtomCount, "c1ccccc1"), 6);
        assert_eq!(calc_int(&AromaticAtomCount, "CCCC"), 0);
    }

    #[test]
    fn test_largest_pi_system() {
        use super::pi_system::LargestPiSystem;
        assert_eq!(calc_int(&LargestPiSystem, "c1ccccc1"), 6);
        assert_eq!(calc_int(&LargestPiSystem, "CC"), 0);
        assert_eq!(calc_int(&LargestPiSystem, "C=C"), 2);
    }

    // ======================================================================
    // Phase 12: Gasteiger charges + Polarizability + PEOE_VSA
    // ======================================================================

    #[test]
    fn test_max_partial_charge() {
        use super::gasteiger_charge::MaxPartialCharge;
        let v = calc_float(&MaxPartialCharge, "C=O");
        assert!(
            v > 0.0,
            "formaldehyde max charge should be positive (C): {v}"
        );
    }

    #[test]
    fn test_min_partial_charge() {
        use super::gasteiger_charge::MinPartialCharge;
        let v = calc_float(&MinPartialCharge, "C=O");
        assert!(
            v < 0.0,
            "formaldehyde min charge should be negative (O): {v}"
        );
    }

    #[test]
    fn test_gasteiger_charges_vector() {
        use super::gasteiger_charge::GasteigerCharges;
        let v = calc_float_vec(&GasteigerCharges, "CCO");
        assert_eq!(v.len(), 3, "ethanol has 3 heavy atoms");
    }

    #[test]
    fn test_apol() {
        use super::polarizability::APol;
        let v = calc_float(&APol, "C");
        // methane: C(1.76) + 4H(0.387) = 3.308
        assert!((v - 3.308).abs() < 0.01, "methane APol: {v}");
    }

    #[test]
    fn test_bpol() {
        use super::polarizability::BPol;
        let v = calc_float(&BPol, "c1ccccc1");
        assert!(v.abs() < 0.01, "benzene BPol should be ~0 (symmetric): {v}");
    }

    #[test]
    fn test_peoe_vsa() {
        use super::vsa::PeoeVsa;
        let v = calc_float_vec(&PeoeVsa, "CCO");
        assert_eq!(v.len(), 14, "PEOE_VSA should have 14 bins");
        let sum: f32 = v.iter().sum();
        assert!(sum > 0.0, "PEOE_VSA sum should be positive");
    }

    // ======================================================================
    // ML vector building with all typed descriptors
    // ======================================================================

    #[test]
    fn test_build_ml_vector_all_types() {
        use super::atom_count::HeavyAtomCount;
        use super::logp::LogP;
        use super::molecular_weight::MolecularWeight;
        use super::typed::build_ml_vector;

        let mut mol = parse("CCO");
        let descriptors: Vec<Box<dyn Descriptor>> = vec![
            Box::new(MolecularWeight),
            Box::new(HeavyAtomCount),
            Box::new(LogP),
        ];
        let vec = build_ml_vector(&mut mol, &descriptors);
        assert_eq!(vec.len(), 3);
        assert!((vec[0] - 46.07).abs() < 0.1, "ethanol MW");
        assert!((vec[1] - 3.0).abs() < 0.01, "ethanol heavy atoms");
    }

    #[test]
    fn test_all_descriptors_no_panic() {
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
            GasteigerCharges, MaxAbsPartialCharge, MaxPartialCharge, MinAbsPartialCharge,
            MinPartialCharge,
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
        use super::vsa::{LabuteASA, PeoeVsa, SlogpVSA, SmrVSA};
        use super::weighted_path::WeightedPath;
        use super::wiener_numbers::WienerIndex;
        use super::zagreb::ZagrebIndices;

        let descriptors: Vec<Box<dyn Descriptor>> = vec![
            Box::new(MolecularWeight),
            Box::new(LogP),
            Box::new(MolarRefractivity),
            Box::new(Tpsa),
            Box::new(AtomCount),
            Box::new(HeavyAtomCount),
            Box::new(ElementCounts),
            Box::new(AromaticAtomCount),
            Box::new(BondCount),
            Box::new(BondCountDetailed),
            Box::new(RotatableBondCount),
            Box::new(AromaticRingCount),
            Box::new(HBondDonor),
            Box::new(HBondAcceptor),
            Box::new(LipinskiViolations),
            Box::new(ValenceElectrons),
            Box::new(HeteroatomCount),
            Box::new(RadicalElectrons),
            Box::new(HybridizationRatio),
            Box::new(VAdjMa),
            Box::new(SmallRingDescriptor),
            Box::new(RingTypeCounts),
            Box::new(PiSystemCounts),
            Box::new(LargestPiSystem),
            Box::new(CarbonTypes),
            Box::new(ChiIndices),
            Box::new(KappaIndices),
            Box::new(EStateIndices),
            Box::new(Autocorrelation),
            Box::new(BCUT),
            Box::new(WeightedPath),
            Box::new(MDE),
            Box::new(WienerIndex),
            Box::new(ZagrebIndices),
            Box::new(BalabanIndex),
            Box::new(Qed),
            Box::new(PetitjeanIndex),
            Box::new(SlogpVSA),
            Box::new(SmrVSA),
            Box::new(LabuteASA),
            Box::new(FractionCSP3),
            Box::new(BertzCT),
            Box::new(NumAmideBonds),
            Box::new(AromaticAtomFraction),
            Box::new(HallKierAlpha),
            Box::new(Ipc),
            Box::new(FragmentComplexity),
            Box::new(FunctionalGroupCounts),
            Box::new(AcidicGroupCount),
            Box::new(BasicGroupCount),
            Box::new(FormalCharge),
            Box::new(LongestAliphaticChain),
            Box::new(LargestChain),
            Box::new(MaccsKeys),
            Box::new(PubChem881),
            Box::new(ECFP {
                radius: 2,
                num_bits: 2048,
            }),
            Box::new(FCFP {
                radius: 2,
                num_bits: 2048,
            }),
            Box::new(AtomPairFingerprint { num_bits: 2048 }),
            Box::new(TopologicalTorsionFingerprint { num_bits: 2048 }),
            // Phase 12: Gasteiger charges + polarizability
            Box::new(MaxPartialCharge),
            Box::new(MinPartialCharge),
            Box::new(MaxAbsPartialCharge),
            Box::new(MinAbsPartialCharge),
            Box::new(GasteigerCharges),
            Box::new(APol),
            Box::new(BPol),
            Box::new(PeoeVsa),
        ];

        assert_eq!(descriptors.len(), 67, "should have 67 typed descriptors");

        // Test on multiple molecules - none should panic
        let smiles_list = [
            "C", "CC", "CCO", "c1ccccc1", "CC(=O)O", "CN", "CC=O", "C=C", "CCBr",
        ];
        for smiles in &smiles_list {
            let mut mol = parse(smiles);
            for desc in &descriptors {
                let _value = desc.calculate(&mut mol);
            }
        }
    }
}
