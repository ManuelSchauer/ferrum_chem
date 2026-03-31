//! # Machine-learning toolkit for QSAR workflows
//!
//! The `ml` module provides in-crate ML components used by FerrumChem's QSAR
//! pipeline (no external high-level ML framework required).
//!
//! ## Main capabilities
//!
//! - dataset containers and feature/target handling (`dataset`)
//! - preprocessing (`scaling`, `feature_selection`)
//! - model implementations (`knn`, `random_forest`, `pls`, `elastic_net`)
//! - validation and splitting (`cv`, `splitting`, `metrics`)
//! - projection and clustering (`pca`, `art2a`)
//! - applicability-domain estimation (`ad`)
//! - orchestration (`pipeline`)
//!
//! ## Learning notes
//!
//! This module is a good study target for implementing end-to-end ML workflows
//! in Rust with explicit data flow and strong typing.
//!
//! ## File guide: ML Module Registry
//!
//! This file maps the ML subsystem into composable modules.
//!
//! Key symbols in this file:
//! - `mod dataset`
//! - `mod scaling`
//! - `mod cv`
//! - `mod pipeline`
//! - model modules: `knn`, `random_forest`, `pls`, `elastic_net`
//!
//! Suggested reading workflow:
//! 1. `dataset` -> `scaling` -> `splitting`/`cv`.
//! 2. One model implementation (e.g. `knn`) end-to-end.
//! 3. `pipeline` to see orchestration and multi-model workflows.
//! 4. `ad` + `art2a` for advanced analysis surfaces.

//!
//! ### Algorithmic assumptions for this file
//! - `Mod` assumes `ndarray` row-major sample layout (`n_samples x n_features`) throughout fit/predict paths.
//! - `Mod` prioritizes reproducibility via explicit seeds and deterministic iteration order where applicable.
//! - Model quality metrics for `Mod` depend on consistent preprocessing choices (scaling, splitting, and leakage control).
//! - Specific scope anchor: `crates/ferrum_core/src/ml/mod.rs` defines mod behavior and should be updated together with its direct tests.
//! ### Chemistry/ML references
//! - Canonical ML literature for `Mod` algorithm family (statistical learning and validation texts).
//! - scikit-learn/RDKit-style API ergonomics that inform `Mod` fit/transform/predict contracts.
//! - OECD QSAR validation principles where `Mod` is used in predictive chemistry contexts.
//! - Code-local source of truth: symbols and tests declared in `crates/ferrum_core/src/ml/mod.rs`.

pub mod ad;
pub mod art2a;
pub mod cv;
pub mod dataset;
pub mod elastic_net;
pub mod feature_selection;
pub mod knn;
pub mod metrics;
pub mod pca;
pub mod pipeline;
pub mod pls;
pub mod random_forest;
pub mod scaling;
pub mod splitting;
