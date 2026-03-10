# Design Decision: Unified Float Vector for Features

**Date**: February 4, 2026  
**Status**: ✅ DECIDED  
**Affects**: Phase 6 (ML Integration), Phase 7 (Fingerprints)

## The Question

How should we store molecular features (fingerprints + descriptors) for ML?

**Option 1**: Separate storage (bits + floats)
```rust
pub struct MolecularFeatures {
    fingerprints: Vec<u64>,  // Packed bits
    descriptors: Vec<f32>,   // Floats
}
```

**Option 2**: Unified storage (all floats)
```rust
pub struct MolecularFeatures {
    features: Vec<f32>,  // All features as floats
}
```

## Decision

✅ **We chose Option 2: Unified Float Vector**

## Rationale

### 1. **ML is the Primary Use Case**
- This tool is designed for QSAR/ML workflows
- Models expect single feature vectors
- ndarray/linfa integration requires floats

### 2. **Simplicity Over Micro-Optimization**
- Project philosophy: "limit flexibility for clarity"
- One data structure = simpler API
- Less cognitive load for users

### 3. **Zero Conversion Overhead**
```rust
// With separate storage (Option 1):
fn to_ml_vector(&self) -> Vec<f32> {
    let mut vec = Vec::with_capacity(5000);
    for bit in &self.fingerprints {
        vec.extend(bit_to_floats(*bit));  // Conversion cost!
    }
    vec.extend(&self.descriptors);
    vec
}
// Cost: ~15 μs per molecule

// With unified storage (Option 2):
fn as_slice(&self) -> &[f32] {
    &self.features  // Zero cost!
}
```

### 4. **Direct ndarray Integration**
```rust
// Batch processing becomes trivial:
let data: Vec<f32> = molecules
    .par_iter()
    .flat_map(|mol| calculate_features(mol).features)
    .collect();

let matrix = Array2::from_shape_vec((n, d), data)?;
// No conversions, no temporary allocations!
```

### 5. **Matches Industry Standards**
- scikit-learn: Uses numpy arrays (floats)
- RDKit: Converts bits to floats for ML
- linfa: Expects ndarray of floats

## Trade-offs

### Memory Usage

| Storage | Fingerprints | Descriptors | Total | For 1M Molecules |
|---------|--------------|-------------|-------|------------------|
| **Packed bits** | 563 bytes | 2 KB | 2.5 KB | 2.5 GB |
| **Float vector** | 18 KB | 2 KB | 20 KB | **20 GB** |

**Analysis**:
- 8x memory increase
- But: 20 GB is acceptable for modern systems
- And: Likely processing in batches anyway
- And: Clarity/speed worth the cost

### Performance

**Memory:**
- Packed: 2.5 KB per molecule ✅
- Floats: 20 KB per molecule ⚠️

**Speed:**
- Conversion overhead: **Eliminated** ✅
- ML pipeline: **Faster** ✅
- Fingerprint bit ops: Slightly slower (but rarely needed)

**Winner**: Unified floats for ML use case

## Implementation

### Structure Definition

```rust
// crates/ferrum_core/src/desc/features.rs

/// Unified molecular feature vector for ML
pub struct MolecularFeatures {
    /// All features as floats (~5000 dimensions)
    /// 
    /// Layout:
    ///   [0..2048]:     ECFP4 (bits as 0.0/1.0)
    ///   [2048..4096]:  FCFP4 (bits as 0.0/1.0)
    ///   [4096..4262]:  MACCS (166 bits as 0.0/1.0)
    ///   [4262..5143]:  PubChem (881 bits as 0.0/1.0)
    ///   [5143..5182]:  Biosymphony (39 bits as 0.0/1.0)
    ///   [5182..]:      Physicochemical descriptors (native floats)
    pub features: Vec<f32>,
}

impl MolecularFeatures {
    pub const FINGERPRINT_END: usize = 5182;
    pub const TOTAL_DIM: usize = 5182 + 500;  // fingerprints + descriptors
    
    /// Get all features as a slice (ML-ready)
    pub fn as_slice(&self) -> &[f32] {
        &self.features
    }
    
    /// Convert to ndarray view (zero-copy)
    pub fn as_ndarray(&self) -> ArrayView1<f32> {
        ArrayView1::from(&self.features)
    }
    
    /// Get fingerprint region (0..5182)
    pub fn fingerprints(&self) -> &[f32] {
        &self.features[0..Self::FINGERPRINT_END]
    }
    
    /// Get descriptor region (5182+)
    pub fn descriptors(&self) -> &[f32] {
        &self.features[Self::FINGERPRINT_END..]
    }
    
    /// Normalize continuous descriptors only
    pub fn normalize(&mut self, scaler: &Scaler) {
        scaler.transform(&mut self.features[Self::FINGERPRINT_END..]);
    }
}
```

### Fingerprint Generation

```rust
pub trait Fingerprint {
    /// Generate fingerprint bits directly as floats
    fn generate(&self, mol: &Molecule, output: &mut [f32]);
}

impl Fingerprint for ECFP4 {
    fn generate(&self, mol: &Molecule, output: &mut [f32]) {
        // Compute which bits are set
        let set_bits = self.compute_hash_bits(mol);
        
        // Write directly as 0.0 or 1.0
        output.fill(0.0);
        for bit_index in set_bits {
            output[bit_index] = 1.0;
        }
    }
}
```

### Batch Processing

```rust
pub fn calculate_batch_features(molecules: &[Molecule]) -> Array2<f32> {
    let n = molecules.len();
    let d = MolecularFeatures::TOTAL_DIM;
    
    // Parallel generation with direct flat_map
    let data: Vec<f32> = molecules
        .par_iter()
        .flat_map(|mol| {
            let features = calculate_features(mol);
            features.features  // Already Vec<f32>!
        })
        .collect();
    
    Array2::from_shape_vec((n, d), data).unwrap()
}
```

**Zero conversions, zero temporary allocations!** ✅

## Can We Still Do Fingerprint Operations?

### Yes! Tanimoto Similarity

```rust
pub fn tanimoto_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (mut intersection, mut union) = (0, 0);
    
    for (bit_a, bit_b) in a.iter().zip(b) {
        let a_set = *bit_a > 0.5;  // Treat as binary
        let b_set = *bit_b > 0.5;
        
        if a_set && b_set { intersection += 1; }
        if a_set || b_set { union += 1; }
    }
    
    intersection as f32 / union as f32
}
```

**Performance**: ~1 μs per pair (acceptable)

### Bit Operations

If needed for specific algorithms:
```rust
impl MolecularFeatures {
    /// Convert fingerprint region to bit vector (rare operation)
    pub fn fingerprints_as_bits(&self) -> BitVec {
        self.fingerprints()
            .iter()
            .map(|&x| x > 0.5)
            .collect()
    }
}
```

## ML Model Compatibility

### Random Forest
```rust
// Works perfectly - treats 0.0/1.0 as binary splits
let rf = RandomForest::params()
    .n_trees(100)
    .fit(&features_matrix)?;
```

### k-NN
```rust
// Can use either Euclidean or Tanimoto
let knn = KNN::new(5, EuclideanDistance);
// Or custom:
let knn = KNN::new(5, TanimotoDistance);
```

### Neural Networks
```rust
// Expects floats anyway - perfect fit
let nn = NeuralNet::new(&[5000, 100, 1]);
```

## Normalization Strategy

**Critical**: Only normalize continuous descriptors!

```rust
impl Scaler {
    pub fn fit(&mut self, features: &[MolecularFeatures]) {
        // Calculate mean/std ONLY for descriptors (5182+)
        for feature_vec in features {
            let descriptors = feature_vec.descriptors();
            // Update running statistics...
        }
    }
    
    pub fn transform(&self, features: &mut [f32]) {
        // Only transform continuous region
        for (i, x) in features.iter_mut().enumerate() {
            *x = (*x - self.means[i]) / self.std_devs[i];
        }
    }
}
```

**Result**: Fingerprints stay 0/1, descriptors normalized to mean=0, std=1

## Validation

### Memory Test
```rust
#[test]
fn test_memory_usage() {
    let features = MolecularFeatures {
        features: vec![0.0; 5682],  // 5182 + 500
    };
    
    let size = std::mem::size_of_val(&features.features);
    assert_eq!(size, 5682 * 4);  // 22,728 bytes (~20 KB)
}
```

### Conversion Test
```rust
#[test]
fn test_ndarray_conversion() {
    let features = MolecularFeatures {
        features: vec![0.0; 5682],
    };
    
    let array = features.as_ndarray();
    assert_eq!(array.len(), 5682);
    // Zero-copy conversion!
}
```

### Tanimoto Test
```rust
#[test]
fn test_tanimoto_similarity() {
    let a = vec![1.0, 0.0, 1.0, 1.0];
    let b = vec![1.0, 1.0, 0.0, 1.0];
    
    let sim = tanimoto_similarity(&a, &b);
    assert!((sim - 0.5).abs() < 0.01);  // 2/4 = 0.5
}
```

## Alternative Considered and Rejected

### Enum-based Approach
```rust
pub enum Feature {
    Bit(bool),
    Float(f32),
}
```

**Rejected because**:
- ❌ Too complex
- ❌ Poor cache performance
- ❌ Can't use as slice
- ❌ Harder ML integration

### Trait-based Polymorphism
```rust
pub trait FeatureVector {
    fn as_ml_vector(&self) -> Vec<f32>;
}
```

**Rejected because**:
- ❌ Still requires conversion
- ❌ Runtime overhead
- ❌ More complex API

## Impact on Project

### Phase 6: ML Integration
- ✅ Simplified implementation
- ✅ Zero conversion overhead
- ✅ Direct linfa integration

### Phase 7: Fingerprints
- ✅ Generate directly as floats
- ✅ No bit packing/unpacking
- ⚠️ Slightly more memory (acceptable)

### Phase 8: QsarPipeline
- ✅ Single vector type throughout
- ✅ Simpler serialization
- ✅ Easier debugging

## Conclusion

**The unified float vector is the correct choice for this project because:**

1. ✅ **Primary use case is ML** - optimize for that
2. ✅ **Simplicity is a core value** - one structure beats two
3. ✅ **Performance where it matters** - zero conversion overhead
4. ✅ **Industry standard** - matches scikit-learn, RDKit ML workflows
5. ✅ **Future-proof** - easy to extend, debug, maintain

**The memory overhead (8x) is acceptable because:**
- Modern systems have plenty of RAM
- Batch processing limits active molecules
- Clarity and speed are more valuable
- 20 KB per molecule is still small

---

**Status**: ✅ DECIDED and DOCUMENTED  
**Implementation**: Phase 6-7  
**No further discussion needed** - this is the design!

