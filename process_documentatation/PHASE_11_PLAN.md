# Phase 11: Typed Descriptors Architecture — Implementation Plan

**Goal**: Replace `Vec<f32>` descriptor outputs with proper types for:
- 4x memory reduction on fingerprints (BitVec instead of Vec<f32>)
- Type safety for QSAR (u32 counts, not f32 with rounding errors)
- Better API ergonomics

**Status**: In Progress  
**Start Date**: February 11, 2026

---

## Design Overview

### DescriptorValue Enum

```rust
pub enum DescriptorValue {
    /// Single integer (e.g., atom count, ring count)
    Int(u32),
    
    /// Single floating point (e.g., molecular weight, LogP)
    Float(f32),
    
    /// Multiple integers (e.g., atom counts by element)
    IntVec(Vec<u32>),
    
    /// Multiple floats (e.g., Chi indices, E-State)
    FloatVec(Vec<f32>),
    
    /// Binary fingerprint (bit-packed for 4x memory savings)
    Binary(BitVec),
}
```

### Memory Savings Calculation

**Before (Vec<f32>)**:
- MACCS: 166 bits × 4 bytes = 664 bytes
- PubChem: 881 bits × 4 bytes = 3,524 bytes
- ECFP4: 2048 bits × 4 bytes = 8,192 bytes
- **Total fingerprints**: ~69 KB per molecule (17,431 bits × 4)

**After (BitVec)**:
- MACCS: 166 bits = 21 bytes (with overhead)
- PubChem: 881 bits = 111 bytes
- ECFP4: 2048 bits = 256 bytes
- **Total fingerprints**: ~2.2 KB per molecule

**Memory Reduction**: 69 KB → 2.2 KB = **31x reduction** for fingerprints!

---

## TypedDescriptor Trait

```rust
pub trait TypedDescriptor: Send + Sync {
    /// Descriptor name
    fn name(&self) -> &str;
    
    /// Calculate and return typed value
    fn calculate(&self, mol: &Molecule) -> DescriptorValue;
}
```

**Design Decision**: Owned return (`DescriptorValue`) not `&mut` slice:
- Cleaner API (no pre-allocation)
- Easier to use in iterators/functional style
- Still efficient (move semantics, no copying)

---

## Conversion Layer for ML

```rust
impl DescriptorValue {
    /// Convert to flat f32 vector for ML pipelines
    pub fn to_floats(&self, output: &mut Vec<f32>) {
        match self {
            DescriptorValue::Int(v) => output.push(*v as f32),
            DescriptorValue::Float(v) => output.push(*v),
            DescriptorValue::IntVec(v) => output.extend(v.iter().map(|&x| x as f32)),
            DescriptorValue::FloatVec(v) => output.extend_from_slice(v),
            DescriptorValue::Binary(bv) => {
                for i in 0..bv.len() {
                    output.push(if bv[i] { 1.0 } else { 0.0 });
                }
            }
        }
    }
    
    /// Number of float values this will produce
    pub fn len(&self) -> usize {
        match self {
            DescriptorValue::Int(_) | DescriptorValue::Float(_) => 1,
            DescriptorValue::IntVec(v) => v.len(),
            DescriptorValue::FloatVec(v) => v.len(),
            DescriptorValue::Binary(bv) => bv.len(),
        }
    }
}
```

---

## Migration Strategy

### Phase 1: Foundation (Week 1)
1. Add `bitvec` dependency to Cargo.toml
2. Create `desc/typed.rs` with DescriptorValue enum
3. Create TypedDescriptor trait
4. Implement conversion layer (to_floats)
5. Create example: MolecularWeightTyped
6. Add tests for conversion

### Phase 2: Fingerprints (Week 2)
**Biggest memory win, do first**

Migration order (by memory impact):
1. **ECFP/FCFP** (12,288 bits × 2 = 24,576 bits → 3 KB saved each)
2. **PubChem** (881 bits → 3.5 KB saved)
3. **AtomPair** (2,048 bits → 8 KB saved)
4. **TopologicalTorsion** (2,048 bits → 8 KB saved)
5. **MACCS** (166 bits → 664 bytes saved)

For each fingerprint:
- Create `*Typed` struct implementing TypedDescriptor
- Return `Binary(BitVec)`
- Update tests to use new API
- Keep old struct for backward compat (deprecated)

### Phase 3: Counting Descriptors (Week 3)
**Type safety win**

Migrate to `Int(u32)` or `IntVec(Vec<u32>)`:
- AtomCount → IntVec (12 values)
- BondCount → IntVec (6 values)
- AromaticRingCount → Int
- NumHeavyAtoms → Int
- SmallRing → IntVec (11 values)
- RingTypeCounts → IntVec (5 values)
- CarbonTypes → IntVec (9 values)
- FunctionalGroupCounts → IntVec (16 values)

### Phase 4: Continuous Descriptors (Week 3)
**Keep as Float/FloatVec**

Most continuous descriptors stay as-is, just wrapped:
- MolecularWeight → Float
- LogP → Float
- TPSA → Float
- Chi indices → FloatVec (26 values)
- E-State → FloatVec (47 values)
- etc.

---

## Backward Compatibility Strategy

Keep legacy `Descriptor` trait and old implementations:

```rust
// Old API (deprecated but still works)
pub trait Descriptor: Send + Sync {
    fn calculate(&self, mol: &Molecule, output: &mut [Real]);
}

// New API
pub trait TypedDescriptor: Send + Sync {
    fn calculate(&self, mol: &Molecule) -> DescriptorValue;
}

// Bridge: TypedDescriptor → Descriptor
impl<T: TypedDescriptor> Descriptor for T {
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        let value = TypedDescriptor::calculate(self, mol);
        let mut floats = Vec::new();
        value.to_floats(&mut floats);
        output.copy_from_slice(&floats);
    }
}
```

**Migration path**:
1. New code uses TypedDescriptor
2. Old code keeps working via bridge
3. Deprecate old API in v2.0
4. Remove old API in v3.0

---

## Testing Strategy

### Unit Tests
- Conversion layer (Int → f32, Binary → Vec<f32>)
- Each migrated descriptor (typed output correct)
- Backward compat bridge (old API still works)

### Integration Tests
- ML vector builder with mixed typed descriptors
- Batch processing with typed descriptors
- Memory benchmarks (before/after)

### Performance Tests
- BitVec operations vs Vec<f32>
- Conversion overhead (should be negligible)
- Full pipeline with typed descriptors

---

## Success Criteria

- [ ] All 503 tests still pass
- [ ] Memory usage for fingerprints: <3 KB per molecule (vs 69 KB)
- [ ] No performance regression (conversion < 1% overhead)
- [ ] Backward compatibility: old API still works
- [ ] New API: cleaner, type-safe
- [ ] Documentation updated

---

## Dependencies to Add

```toml
[dependencies]
bitvec = "1.0"  # Bit-packed vectors with efficient operations
```

**Why bitvec**:
- Industry standard for bit manipulation in Rust
- Zero-cost abstractions
- Efficient iteration, indexing, and set operations
- Perfect for Tanimoto similarity on fingerprints

---

## Example Usage (After Migration)

```rust
use ferrum_core::desc::typed::*;
use ferrum_core::io::smiles::parse_smiles;

let mol = parse_smiles("c1ccccc1").unwrap();

// New typed API
let mw_desc = MolecularWeightTyped;
let mw = mw_desc.calculate(&mol);
match mw {
    DescriptorValue::Float(f) => println!("MW: {:.2}", f),
    _ => unreachable!(),
}

// Fingerprint
let maccs_desc = MACCSTyped;
let maccs = maccs_desc.calculate(&mol);
match maccs {
    DescriptorValue::Binary(bits) => {
        println!("MACCS: {} bits set", bits.count_ones());
    }
    _ => unreachable!(),
}

// ML vector builder
let descriptors: Vec<Box<dyn TypedDescriptor>> = vec![
    Box::new(MolecularWeightTyped),
    Box::new(MACCSTyped),
    Box::new(LogPTyped),
];

let mut ml_vector = Vec::new();
for desc in &descriptors {
    let value = desc.calculate(&mol);
    value.to_floats(&mut ml_vector);
}
println!("ML vector size: {}", ml_vector.len());
```

---

## Timeline

**Week 1 (Feb 11-15)**:
- Day 1: Design + add bitvec dependency
- Day 2-3: DescriptorValue enum + TypedDescriptor trait + conversion
- Day 4: Example (MolecularWeightTyped) + tests
- Day 5: ML vector builder + integration tests

**Week 2 (Feb 18-22)**:
- Day 1-2: Migrate ECFP/FCFP (biggest win)
- Day 3: Migrate PubChem
- Day 4: Migrate AtomPair + TopologicalTorsion
- Day 5: Migrate MACCS + benchmark memory

**Week 3 (Feb 25-Mar 1)**:
- Day 1-2: Migrate counting descriptors (Int/IntVec)
- Day 3: Migrate continuous descriptors (Float/FloatVec)
- Day 4: Update all tests + documentation
- Day 5: Final integration tests + release prep

**Buffer**: 2-3 days for unexpected issues

---

## Risks & Mitigations

**Risk**: Breaking changes to public API  
**Mitigation**: Keep backward compat via bridge, deprecate slowly

**Risk**: Performance regression from conversions  
**Mitigation**: Benchmark early, optimize hot paths

**Risk**: BitVec operations slower than Vec<f32>  
**Mitigation**: Profile, bitvec is highly optimized

**Risk**: Complex migration, lots of files to update  
**Mitigation**: Incremental migration, test each step

---

## Next Immediate Steps

1. Add `bitvec = "1.0"` to Cargo.toml
2. Create `crates/ferrum_core/src/desc/typed.rs`
3. Implement DescriptorValue enum
4. Implement TypedDescriptor trait
5. Add conversion methods
6. Create MolecularWeightTyped as proof-of-concept
7. Write tests for foundation

Let's start! 🚀
