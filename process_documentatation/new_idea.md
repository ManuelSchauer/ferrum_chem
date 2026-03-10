# Project Specification: QSAR-rs
**A Rust-based QSAR Analysis, Clustering, and Machine Learning Tool.**

## 1. Project Overview
We are building a high-performance CLI and TUI tool in Rust for Quantitative Structure-Activity Relationship (QSAR) analysis. The tool will handle molecular descriptor calculation, fingerprint generation, dimensionality reduction, and machine learning (training/inference) within a single binary.

**Core Philosophy:**
*   **Performance:** All calculations and ML training happen in pure Rust (no Python dependency).
*   **Modular Monolith:** Functionality is separated into internal modules but shipped as one tool.
*   **Hybrid Interface:** A "Headless Mode" (CLI) for batch processing and an "Interactive Mode" (TUI) for pipeline design.

---

## 2. Architecture & Workflow

### 2.1 The Two-Mode Design
The application entry point determines the mode:

1.  **Interactive Mode (TUI):** `qsar-rs ui`
    *   Built with `ratatui` + `crossterm`.
    *   Used for visual pipeline configuration, feature selection, and immediate feedback.
    *   **Feature:** "Export Configuration" (saves settings to a TOML file).
2.  **Headless Mode (CLI):** `qsar-rs run --config pipeline.toml` (or specific flags).
    *   Used for automation, server jobs, and batch processing.

### 2.2 The Data Pipeline
The data flows through distinct stages:
1.  **Ingest:** Read SMILES from CSV/SDF.
2.  **Featurization:** Generate the "Mega-Vector" (Descriptors + Fingerprints).
3.  **Preprocessing:** Z-Score Normalization -> Optional PCA.
4.  **Learning:** Train Model (KNN, Random Forest, ART) OR Cluster.
5.  **Output:** Save Model (`.bin`) / Save Predictions (`.csv`).

---

## 3. Feature Engineering: "The Mega-Vector"

We will generate a heterogeneous feature vector (approx. 5,000+ dimensions) for every molecule.

### 3.1 Vector Composition
The vector is a concatenation of the following components:

| Component | Type | Details | Status |
| :--- | :--- | :--- | :--- |
| **ECFP** | Binary | Extended Connectivity Fingerprint (Structural). Default Radius: 2 (ECFP4). | **Default** |
| **FCFP** | Binary | Functional Class Fingerprint (Pharmacophore). Default Radius: 2 (FCFP4). | **Optional** |
| **MACCS** | Binary | 166 Standard Structural Keys. | **Required** |
| **PubChem** | Binary | ~881 PubChem Substructure Keys. | **Required** |
| **Biosymphony** | Binary | Custom set of ~39 Natural Product/Biosynthetic SMARTS patterns. | **Custom Impl** |
| **Descriptors** | Float | 200+ Physicochemical properties (MW, LogP, TPSA, etc.). | **Required** |

### 3.2 Implementation Strategy
*   **Concatenation:** ECFP and FCFP should be concatenated end-to-end (e.g., 2048 bits + 2048 bits).
*   **Normalization:** Continuous descriptors **must** be Z-Score normalized (Mean=0, Std=1) so they do not overpower the binary fingerprints during distance calculations.
*   **Custom Fingerprints:** The "Biosymphony" set will be loaded from a definition file (SMARTS strings) and checked against every molecule.

---

## 4. Machine Learning & Algorithms

We utilize the `linfa` ecosystem. The training process produces a serialized `QsarPipeline` struct, not just a raw model.

### 4.1 The `QsarPipeline` Struct
To ensure reproducibility, the saved model file must contain:
1.  **Scaler Parameters:** (Mean/StdDev) to normalize future inputs.
2.  **PCA Projection Matrix:** (If PCA was used) to transform future inputs.
3.  **The Trained Model:** The actual Linfa model struct.

### 4.2 Algorithms & Configuration

*   **Dimensionality Reduction (PCA):**
    *   *Usage:* Critical for KNN/Clustering when using the Mega-Vector.
    *   *Config:* Number of Components (e.g., 50).
    *   *Logic:* If enabled, transforms 5000 dims -> 50 dims before training.

*   **K-Nearest Neighbors (KNN):**
    *   *Use Case:* Similarity search and Regression.
    *   *Params:* `K` (Neighbors), `Metric` (Euclidean vs Jaccard/Tanimoto).
    *   *Note:* Requires Tanimoto implementation for raw bit-vectors, or Euclidean if using PCA-reduced vectors.

*   **Adaptive Resonance Theory (ART) - *Custom Implementation*:**
    *   *Use Case:* Unsupervised Clustering / Applicability Domain.
    *   *Params:* `Vigilance` (rho), `Learning Rate` (beta).
    *   *Implementation:* Must be written as a custom module, potentially implementing `linfa::traits::Transformer`.

*   **Random Forest:**
    *   *Use Case:* Robust prediction on the raw Mega-Vector (handles mixed binary/float well).
    *   *Params:* `n_trees`, `max_depth`.

---

## 5. User Interface Design (TUI)

The TUI will be structured with a Tab-based layout.

### 5.1 Tabs
1.  **Data:** File picker for Input CSV, Output Path, and Target Column selection.
2.  **Features:** Checkbox list to toggle specific fingerprints (ECFP, FCFP, MACCS, BioSymph) and Descriptors.
    *   *Settings:* Slider for ECFP Radius (0, 2, 4, 6) and Bit Length (1024, 2048).
3.  **Model:** Dropdown to select Algorithm (KNN, ART, RF).
    *   *Dynamic Config:* Showing only relevant hyperparameters (e.g., hiding 'Vigilance' if KNN is selected).
    *   *Preprocessing:* Toggle for PCA and Normalization.
4.  **Run:** Execution view with Progress Bars.
    *   *Action:* "Export Config" button (generates `.toml` for CLI usage).

### 5.2 Technical TUI Requirements
*   **Async Pattern:** Use `tokio` channels (MPSC). The UI runs on the main thread; heavy QSAR math runs on a worker thread to prevent UI freezing.
*   **State Management:** A global `AppState` struct holding the current configuration before execution.

---

## 6. Implementation Stack
*   **Language:** Rust (Edition 2021+)
*   **CLI Parsing:** `clap` (derive feature).
*   **TUI:** `ratatui` + `crossterm`.
*   **ML Library:** `linfa` (clustering, reduction, trees, etc.).
*   **Chemistry:** `rdkit-sys` (bindings) or `chemfiles` / `purr` (if pure Rust is viable, otherwise RDKit bindings are preferred for ECFP reliability).
*   **Serialization:** `serde` + `bincode` (for saving models).
*   **Parallelism:** `rayon` (for calculating descriptors across 5000 molecules in parallel).
