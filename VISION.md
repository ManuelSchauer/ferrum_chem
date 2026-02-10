# FerrumChem - Long-Term Vision

**Based on architectural discussions and future roadmap planning**

## Overview

FerrumChem will evolve into a comprehensive QSAR analysis tool with both CLI and TUI interfaces, supporting the complete ML workflow from molecular input to model deployment.

## Core Principles (Maintained)

1. ✅ **Pure Rust** - No Python/C++ dependencies
2. ✅ **Thread-Safe** - Zero locks, compile-time guarantees
3. ✅ **Performance-First** - 1M+ molecules/sec throughput
4. ✅ **ML-Ready** - Direct integration with Rust ML ecosystem
5. ✅ **Incremental Development** - Build solid foundation first

## The Complete Data Pipeline (Future)

```
Input → Featurization → Preprocessing → ML → Output
  ↓          ↓              ↓           ↓      ↓
CSV      Mega-Vector    Normalize    Train   Model
SDF      (5000+ dims)   + PCA        Linfa   .bin
```

### Pipeline Stages

1. **Ingest**: Read molecules from CSV/SDF
2. **Featurization**: Generate comprehensive feature vector
3. **Preprocessing**: Z-score normalization, optional PCA
4. **Learning**: Train model (KNN, RF, ART) or cluster
5. **Output**: Serialized model + predictions

## The "Mega-Vector" Feature Set

**Total: ~5000+ dimensions**

### Binary Fingerprints (~4500 bits)
- **ECFP4** (2048 bits) - Extended Connectivity Fingerprint
- **FCFP4** (2048 bits) - Functional Class Fingerprint
- **MACCS** (166 bits) - Standard structural keys
- **PubChem** (881 bits) - Substructure keys
- **Biosymphony** (39 bits) - Natural product patterns (custom)

### Continuous Descriptors (~200-500 features)
- Physicochemical properties (MW, LogP, TPSA, etc.)
- Topological indices (Wiener, Zagreb, Balaban)
- Electronic descriptors (charges, polarizability)
- 3D descriptors (optional, if conformers available)

### Feature Engineering
```rust
pub struct MolecularFeatures {
    /// Unified feature vector (all as floats for ML)
    /// Fingerprint bits encoded as 0.0/1.0
    /// Continuous descriptors as native floats
    pub features: Vec<f32>,  // ~5000 dimensions
}

impl MolecularFeatures {
    pub fn as_ndarray(&self) -> ArrayView1<f32> {
        ArrayView1::from(&self.features)
    }
    
    pub fn normalize(&mut self, scaler: &Scaler) {
        // Only normalize continuous descriptors, keep fingerprints 0/1
        scaler.transform(&mut self.features[5182..]);
    }
    
    pub fn to_normalized(&self, scaler: &Scaler) -> Vec<f32> {
        let mut normalized = self.features.clone();
        normalized[5182..].iter_mut().enumerate().for_each(|(i, x)| {
            *x = (*x - scaler.means[i]) / scaler.std_devs[i];
        });
        normalized
    }
}
```

**Design Decision (Phase 6):**
- ✅ **Unified float representation** - No separate bit storage
- ✅ **ML-native** - Direct ndarray conversion
- ✅ **Zero conversions** - No temporary allocations
- ⚠️ **Memory trade-off** - 20 KB vs 2.5 KB per molecule (acceptable)

## The QsarPipeline Architecture

**Core concept for reproducible ML workflows**

```rust
#[derive(Serialize, Deserialize)]
pub struct QsarPipeline {
    /// Scaling parameters for normalization
    scaler: Scaler,
    
    /// Optional PCA transformation
    pca: Option<PcaTransform>,
    
    /// Trained model (KNN, RF, ART, etc.)
    model: Box<dyn Model>,
    
    /// Metadata for reproducibility
    metadata: PipelineMetadata,
}

pub struct Scaler {
    means: Vec<f32>,
    std_devs: Vec<f32>,
}

pub struct PipelineMetadata {
    created: DateTime,
    training_samples: usize,
    feature_names: Vec<String>,
    model_type: String,
}

impl QsarPipeline {
    /// Train a new pipeline from molecules
    pub fn train(molecules: &[Molecule], config: &Config) -> Result<Self>;
    
    /// Predict on new molecules
    pub fn predict(&self, molecules: &[Molecule]) -> Result<Vec<f32>>;
    
    /// Save to binary file
    pub fn save(&self, path: &Path) -> Result<()>;
    
    /// Load from binary file
    pub fn load(path: &Path) -> Result<Self>;
}
```

**Key features:**
- ✅ Self-contained (includes scaler + PCA + model)
- ✅ Reproducible predictions on new data
- ✅ Serializable with `bincode`
- ✅ Version-tracked metadata

## Dual-Mode Interface (Future)

### 1. Headless Mode (CLI) - Priority

**For batch processing and automation:**

```bash
# Calculate descriptors
ferrum calc --input compounds.csv --output features.csv --descriptors all

# Train model
ferrum train --features features.csv --target activity --model rf --output model.bin

# Predict
ferrum predict --model model.bin --input new_compounds.csv --output predictions.csv

# Pipeline mode (one command)
ferrum pipeline --config workflow.toml
```

**Configuration file (TOML):**
```toml
[input]
file = "compounds.csv"
smiles_column = "SMILES"
id_column = "ID"

[features]
fingerprints = ["ECFP4", "MACCS", "PubChem"]
descriptors = ["all"]

[preprocessing]
normalize = true
pca_components = 50

[model]
type = "RandomForest"
n_trees = 100
max_depth = 10

[output]
model_path = "qsar_model.bin"
predictions = "predictions.csv"
```

### 2. Interactive Mode (TUI) - Later Phase

**For pipeline design and experimentation:**

Built with `ratatui` + `crossterm`:

```
┌─ FerrumChem QSAR Pipeline ─────────────────────────────┐
│ [Data] [Features] [Model] [Run]                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ Features Selection:                                      │
│ ☑ ECFP4 (Radius: 2, Bits: 2048)                        │
│ ☐ FCFP4                                                 │
│ ☑ MACCS Keys                                            │
│ ☑ Physicochemical Descriptors (200)                    │
│                                                          │
│ Preprocessing:                                           │
│ ☑ Z-Score Normalization                                │
│ ☑ PCA (Components: 50)                                 │
│                                                          │
│ Model: [Random Forest ▼]                                │
│   Trees: [100    ]                                      │
│   Max Depth: [10     ]                                  │
│                                                          │
│ [Export Config] [Train Model]                           │
└─────────────────────────────────────────────────────────┘
```

**Features:**
- Tab-based navigation
- Live parameter adjustment
- Config export to TOML
- Progress visualization
- Async execution (tokio channels)

## Machine Learning Integration

### Algorithms Supported

#### 1. **K-Nearest Neighbors (KNN)**
```rust
use linfa::traits::Predict;

let knn = KNN::new(k, metric);
let model = knn.fit(&dataset)?;
```

**Metrics:**
- Euclidean (for normalized/PCA-reduced vectors)
- Tanimoto/Jaccard (for raw fingerprints)

**Use cases:**
- Similarity search
- Regression
- Applicability domain

#### 2. **Random Forest**
```rust
use linfa_trees::DecisionTree;

let rf = RandomForest::params()
    .n_trees(100)
    .max_depth(10)
    .fit(&dataset)?;
```

**Advantages:**
- Handles mixed binary/continuous features
- Robust to high dimensions
- Built-in feature importance

#### 3. **Adaptive Resonance Theory (ART) - Custom**
```rust
pub struct ArtClusterer {
    vigilance: f32,
    learning_rate: f32,
    prototypes: Vec<Vec<f32>>,
}

impl Transformer for ArtClusterer {
    fn transform(&self, input: &Array2<f32>) -> Array1<usize>;
}
```

**Use cases:**
- Unsupervised clustering
- Applicability domain definition
- Online learning

**Implementation notes:**
- Custom module (not in linfa)
- Incremental learning capability
- Critical for chemical space exploration

### Preprocessing Pipeline

```rust
pub struct Preprocessor {
    scaler: StandardScaler,
    pca: Option<Pca>,
}

impl Preprocessor {
    pub fn fit(&mut self, data: &Array2<f32>) {
        self.scaler.fit(data);
        if let Some(pca) = &mut self.pca {
            let scaled = self.scaler.transform(data);
            pca.fit(&scaled);
        }
    }
    
    pub fn transform(&self, data: &Array2<f32>) -> Array2<f32> {
        let scaled = self.scaler.transform(data);
        match &self.pca {
            Some(pca) => pca.transform(&scaled),
            None => scaled,
        }
    }
}
```

## Phased Implementation Timeline

### ✅ **Phase 1-2: Complete**
- SMILES parser
- Implicit hydrogens
- Molecular weight
- Thread safety

### 🔄 **Phase 3-5: Current Focus (Months 1-2)**
- Ring detection
- Aromaticity perception
- CSV reading
- Basic descriptors (20-30)

### 📋 **Phase 6-7: Core Features (Months 3-4)**
- SMARTS parser
- MACCS keys (166 patterns)
- ECFP/Morgan fingerprints
- 200+ physicochemical descriptors
- Topological indices

### 🎯 **Phase 8: ML Integration (Month 5)**
- QsarPipeline struct
- Standard scaler implementation
- PCA integration (linfa)
- Random Forest integration (linfa-trees)
- Model serialization (bincode)
- Basic CLI for training/prediction

### 🚀 **Phase 9: Advanced Features (Month 6)**
- FCFP fingerprints
- PubChem keys
- Custom pattern loading
- KNN with Tanimoto metric
- ART clustering (custom)

### 🎨 **Phase 10: TUI (Month 7-8)**
- ratatui interface
- Tab-based navigation
- Config management
- Async execution
- Progress visualization

### 🔧 **Phase 11: Polish (Month 9-10)**
- Performance optimization
- Comprehensive testing
- Documentation
- Examples and tutorials
- Benchmark suite

## Technical Stack (Confirmed)

**Core:**
- `ferrum_core` - Chemistry engine (current implementation)
- `ferrum_cli` - CLI interface
- `ferrum_tui` - Interactive interface (future)

**Dependencies:**
- `clap` - CLI parsing ✅ (already using)
- `rayon` - Parallelism ✅ (already using)
- `linfa` - ML algorithms
- `linfa-trees` - Random Forest
- `linfa-reduction` - PCA
- `ndarray` - Matrix operations
- `ratatui` - TUI framework (later)
- `crossterm` - Terminal control (later)
- `serde` + `bincode` - Serialization
- `tokio` - Async runtime (for TUI)
- `csv` - CSV parsing

**NO Python/C++ dependencies:**
- ❌ RDKit bindings (rejected)
- ✅ Pure Rust implementation

## Performance Targets

**Current (Achieved):**
- ✅ SMILES parsing: 1-9M mol/sec
- ✅ Implicit H: 50M calc/sec
- ✅ Thread-safe: Zero locks

**Future Goals:**
- Fingerprint generation: >500k mol/sec
- Descriptor calculation: >100k mol/sec (full mega-vector)
- Model training: <1 minute for 10k molecules
- Prediction: >10k mol/sec

## Use Cases

### 1. **Virtual Screening**
```bash
ferrum screen --library compounds.csv \
              --model trained_model.bin \
              --threshold 0.7 \
              --output hits.csv
```

### 2. **QSAR Model Development**
```bash
ferrum pipeline --config qsar_workflow.toml
# Handles: feature calculation → training → validation → export
```

### 3. **Chemical Space Analysis**
```bash
ferrum cluster --input library.csv \
               --algorithm art \
               --vigilance 0.8 \
               --output clusters.csv
```

### 4. **Similarity Search**
```bash
ferrum similarity --query "CC(C)Cc1ccc(cc1)C(C)C(=O)O" \
                  --database compounds.csv \
                  --k 10 \
                  --output similar.csv
```

## Comparison to Other Tools

| Feature | FerrumChem | RDKit | CDK |
|---------|-----------|-------|-----|
| Language | Rust | Python/C++ | Java |
| Performance | ⚡⚡⚡ | ⚡⚡ | ⚡ |
| Thread Safety | ✅ Guaranteed | ⚠️ Manual | ⚠️ Manual |
| ML Integration | Native (linfa) | scikit-learn | Weka |
| TUI | ✅ Planned | ❌ | ❌ |
| Binary Output | ✅ Single binary | ❌ Python + libs | ❌ JAR + JVM |
| Deployment | ✅ Easy | ⚠️ Complex | ⚠️ Medium |

**Key Advantages:**
1. Single binary deployment (no runtime dependencies)
2. Guaranteed thread safety (Rust type system)
3. Native performance (no Python overhead)
4. Interactive TUI (unique feature)
5. Reproducible pipelines (QsarPipeline serialization)

## Scientific Validity

All algorithms will be validated against reference implementations:

**Fingerprints:**
- ECFP4 vs RDKit Morgan fingerprint
- MACCS vs RDKit MACCS keys
- Tanimoto coefficient validation

**Descriptors:**
- Compare against RDKit descriptors (±0.01 tolerance)
- Validate on ChEMBL subset
- Document any differences

**ML Models:**
- Cross-validation on standard datasets
- Compare to sklearn benchmarks
- Publish validation reports

## Open Questions (To Resolve)

1. **SMARTS Implementation Strategy**
   - Pure Rust (complex but consistent)
   - Or pattern matching with custom DSL?

2. **3D Descriptor Support**
   - Requires conformer generation
   - Defer to later phase?

3. **GPU Acceleration**
   - Worth it for fingerprint generation?
   - Or is CPU with Rayon sufficient?

4. **Molecule Canonicalization**
   - Needed for duplicate detection
   - Morgan algorithm implementation timing?

## Success Metrics

**By Phase 10 (Complete):**
- ✅ Process 1M molecules in <10 minutes (full pipeline)
- ✅ Train RF model on 10k molecules in <1 minute
- ✅ Single binary <50MB
- ✅ Memory <2GB for 100k molecule dataset
- ✅ 100% thread-safe (compile-time guaranteed)
- ✅ Feature parity with RDKit descriptors
- ✅ Production deployments in real projects

---

**This vision will be achieved incrementally while maintaining code quality, test coverage, and performance at every step.**

**Next Immediate Action: Continue with Phase 3 (Ring Detection)**

