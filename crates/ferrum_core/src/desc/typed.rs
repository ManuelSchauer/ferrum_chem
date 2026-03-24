//! Descriptor API — type-safe descriptors with proper return types
//!
//! This module provides the descriptor API that returns proper types
//! instead of flat `Vec<f32>` arrays. Benefits:
//!
//! - **4x memory reduction** for fingerprints (BitVec instead of Vec<f32>)
//! - **Type safety** for QSAR (u32 counts, not f32 with rounding)
//! - **Better API** ergonomics (owned returns, no pre-allocation)
//!
//! # Design
//!
//! ```no_run
//! use ferrum_core::desc::typed::*;
//! use ferrum_core::io::smiles::parse_smiles;
//!
//! let mol = parse_smiles("c1ccccc1").unwrap();
//! // let mw_desc = MolecularWeight;
//! // let value = mw_desc.calculate(&mol, &mut ferrum_core::desc::context::CalculationContext::new());
//!
//! // match value {
//! //     DescriptorValue::Float(f) => println!("MW: {:.2}", f),
//! //     _ => unreachable!(),
//! // }
//! ```
//!
//! # Conversion to ML Vectors
//!
//! ```rust
//! # use ferrum_core::desc::typed::*;
//! # use ferrum_core::io::smiles::parse_smiles;
//! let mut mol = parse_smiles("CCO").unwrap();
//! let descriptors: Vec<Box<dyn Descriptor>> = vec![
//!     // ... descriptors
//! ];
//!
//! let mut ml_vector = Vec::new();
//! for desc in &descriptors {
//!     let value = desc.calculate(&mut mol, &mut ferrum_core::desc::context::CalculationContext::new());
//!     value.to_floats(&mut ml_vector);
//! }
//! ```

use crate::mol::molecule::Molecule;
use bitvec::prelude::*;

// Type alias for our standard bit vector representation
pub type BitVector = BitVec<u8, Lsb0>;

/// Typed descriptor value — replaces flat `Vec<f32>` with proper types.
///
/// # Memory Savings
///
/// Fingerprints see 4x reduction:
/// - MACCS: 664 bytes → 21 bytes
/// - PubChem: 3,524 bytes → 111 bytes
/// - ECFP4: 8,192 bytes → 256 bytes
///
/// # Type Safety
///
/// Integer counts avoid float rounding:
/// - Atom counts: u32 (exact)
/// - Ring counts: u32 (exact)
/// - Before: `[7.0000001_f32]` (rounding error)
/// - After: `7_u32` (exact)
#[derive(Debug, Clone, PartialEq)]
pub enum DescriptorValue {
    /// Single integer value (e.g., atom count, ring count)
    Int(u32),

    /// Single floating point value (e.g., molecular weight, LogP)
    Float(f32),

    /// High-precision scalar floating point value.
    ///
    /// Used for descriptors that can span very large dynamic ranges (e.g. Ipc),
    /// where f32 precision is insufficient for RDKit parity.
    Float64(f64),

    /// Multiple integer values (e.g., atom counts by element)
    IntVec(Vec<u32>),

    /// Multiple u16 integer values (memory-optimized for element counts)
    /// Use for counts that never exceed 65535 (e.g., individual element counts)
    IntU16Vec(Vec<u16>),

    /// Multiple floating point values (e.g., Chi indices, E-State)
    FloatVec(Vec<f32>),

    /// Binary fingerprint (bit-packed, 4x memory savings)
    ///
    /// Stored as `BitVec` (bit-packed) instead of `Vec<f32>`:
    /// - Memory: 1 bit per feature vs 32 bits
    /// - Operations: Direct bit manipulation
    /// - Tanimoto: Efficient popcount-based
    Binary(BitVector),
}

impl DescriptorValue {
    /// Convert to flat f32 vector for ML pipelines.
    ///
    /// This is the bridge to existing ML infrastructure (linfa, ndarray).
    /// Conversion is efficient:
    /// - Int/Float: single push
    /// - IntVec/FloatVec: extend (no allocation if pre-sized)
    /// - Binary: bit test (fast)
    ///
    /// # Example
    ///
    /// ```
    /// # use ferrum_core::desc::typed::*;
    /// # use bitvec::prelude::*;
    /// let value = DescriptorValue::Int(42);
    /// let mut vec = Vec::new();
    /// value.to_floats(&mut vec);
    /// assert_eq!(vec, vec![42.0]);
    ///
    /// let bits = bitvec![u8, Lsb0; 1, 0, 1];
    /// let value = DescriptorValue::Binary(bits);
    /// let mut vec = Vec::new();
    /// value.to_floats(&mut vec);
    /// assert_eq!(vec, vec![1.0, 0.0, 1.0]);
    /// ```
    pub fn to_floats(&self, output: &mut Vec<f32>) {
        match self {
            DescriptorValue::Int(v) => output.push(*v as f32),
            DescriptorValue::Float(v) => output.push(*v),
            DescriptorValue::Float64(v) => output.push(*v as f32),
            DescriptorValue::IntVec(v) => output.extend(v.iter().map(|&x| x as f32)),
            DescriptorValue::IntU16Vec(v) => output.extend(v.iter().map(|&x| x as f32)),
            DescriptorValue::FloatVec(v) => output.extend_from_slice(v),
            DescriptorValue::Binary(bv) => {
                for bit in bv.iter().by_vals() {
                    output.push(if bit { 1.0 } else { 0.0 });
                }
            }
        }
    }

    /// Number of float values this will produce when converted to ML vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use ferrum_core::desc::typed::*;
    /// # use bitvec::prelude::*;
    /// assert_eq!(DescriptorValue::Int(42).len(), 1);
    /// assert_eq!(DescriptorValue::FloatVec(vec![1.0, 2.0, 3.0]).len(), 3);
    /// assert_eq!(DescriptorValue::Binary(bitvec![u8, Lsb0; 1, 0, 1, 1]).len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        match self {
            DescriptorValue::Int(_) | DescriptorValue::Float(_) | DescriptorValue::Float64(_) => 1,
            DescriptorValue::IntVec(v) => v.len(),
            DescriptorValue::IntU16Vec(v) => v.len(),
            DescriptorValue::FloatVec(v) => v.len(),
            DescriptorValue::Binary(bv) => bv.len(),
        }
    }

    /// Check if this value is empty (returns 0 floats).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a Binary value from a boolean slice.
    ///
    /// # Example
    ///
    /// ```
    /// # use ferrum_core::desc::typed::*;
    /// let bits = vec![true, false, true];
    /// let value = DescriptorValue::from_bits(&bits);
    /// assert_eq!(value.len(), 3);
    /// ```
    pub fn from_bits(bits: &[bool]) -> Self {
        let mut bv = BitVec::with_capacity(bits.len());
        for &bit in bits {
            bv.push(bit);
        }
        DescriptorValue::Binary(bv)
    }

    /// Extract as Int (panics if not Int).
    pub fn as_int(&self) -> u32 {
        match self {
            DescriptorValue::Int(v) => *v,
            _ => panic!("Expected Int, got {:?}", self),
        }
    }

    /// Extract as Float (panics if not Float).
    pub fn as_float(&self) -> f32 {
        match self {
            DescriptorValue::Float(v) => *v,
            DescriptorValue::Float64(v) => *v as f32,
            _ => panic!("Expected Float, got {:?}", self),
        }
    }

    /// Extract as f64 scalar (panics if not scalar float).
    pub fn as_f64(&self) -> f64 {
        match self {
            DescriptorValue::Float(v) => *v as f64,
            DescriptorValue::Float64(v) => *v,
            _ => panic!("Expected Float/Float64, got {:?}", self),
        }
    }

    /// Extract as IntVec (panics if not IntVec).
    pub fn as_int_vec(&self) -> &[u32] {
        match self {
            DescriptorValue::IntVec(v) => v,
            _ => panic!("Expected IntVec, got {:?}", self),
        }
    }

    /// Extract as FloatVec (panics if not FloatVec).
    pub fn as_float_vec(&self) -> &[f32] {
        match self {
            DescriptorValue::FloatVec(v) => v,
            _ => panic!("Expected FloatVec, got {:?}", self),
        }
    }

    /// Extract as Binary (panics if not Binary).
    pub fn as_binary(&self) -> &BitVector {
        match self {
            DescriptorValue::Binary(bv) => bv,
            _ => panic!("Expected Binary, got {:?}", self),
        }
    }
}

/// Descriptor trait — returns proper types instead of flat f32 arrays.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync`.
///
/// # Example Implementation
///
/// ```
/// use ferrum_core::desc::typed::*;
/// use ferrum_core::mol::molecule::Molecule;
///
/// pub struct MyDescriptor;
///
/// impl Descriptor for MyDescriptor {
///     fn name(&self) -> &str {
///         "MyDescriptor"
///     }
///
///     fn calculate(&self, mol: &mut Molecule, ctx: &mut ferrum_core::desc::context::CalculationContext) -> DescriptorValue {
///         DescriptorValue::Int(mol.num_atoms() as u32)
///     }
/// }
/// ```
pub trait Descriptor: Send + Sync {
    /// Descriptor name (for CSV headers, logging).
    fn name(&self) -> &str;

    /// Calculate and return typed value.
    ///
    /// Returns an owned value. This is efficient due to move semantics
    /// and avoids pre-allocation complexity.
    ///
    /// Note: Takes `&mut Molecule` for algorithms that need to cache
    /// computed properties (e.g., ring perception sets ring flags).
    fn calculate(
        &self,
        mol: &mut Molecule,
        ctx: &mut crate::desc::context::CalculationContext,
    ) -> DescriptorValue;

    /// Number of float values when converted to ML vector.
    ///
    /// Default implementation calculates a value and returns its length.
    /// Override if this can be determined without calculation (e.g., constant size).
    fn size(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> usize {
        self.calculate(mol, ctx).len()
    }
}

/// ML vector builder — collects descriptors into flat f32 vector.
///
/// # Example
///
/// ```
/// use ferrum_core::desc::typed::*;
/// use ferrum_core::io::smiles::parse_smiles;
///
/// let mut mol = parse_smiles("CCO").unwrap();
/// let descriptors: Vec<Box<dyn Descriptor>> = vec![
///     // Box::new(MolecularWeight),
///     // Box::new(LogP),
///     // ... more descriptors
/// ];
///
/// let ml_vector = build_ml_vector(&mut mol, &descriptors);
/// // Use with linfa, ndarray, etc.
/// ```
pub fn build_ml_vector(mol: &mut Molecule, descriptors: &[Box<dyn Descriptor>]) -> Vec<f32> {
    let mut ctx = crate::desc::context::CalculationContext::new();
    let mut output = Vec::new();
    for desc in descriptors {
        let value = desc.calculate(mol, &mut ctx);
        value.to_floats(&mut output);
    }
    output
}

/// Batch ML vector builder — parallel processing with Rayon.
///
/// # Example
///
/// ```no_run
/// use ferrum_core::desc::typed::*;
/// use ferrum_core::io::smiles::parse_smiles;
/// use rayon::prelude::*;
///
/// let mut molecules = vec![
///     parse_smiles("CCO").unwrap(),
///     parse_smiles("c1ccccc1").unwrap(),
/// ];
///
/// let descriptors: Vec<Box<dyn Descriptor>> = vec![
///     // ... descriptors
/// ];
///
/// let vectors: Vec<Vec<f32>> = build_ml_vectors_parallel(&mut molecules, &descriptors);
/// ```
pub fn build_ml_vectors_parallel(
    molecules: &mut [Molecule],
    descriptors: &[Box<dyn Descriptor>],
) -> Vec<Vec<f32>> {
    use rayon::prelude::*;
    molecules
        .par_iter_mut()
        .map_init(crate::desc::context::CalculationContext::new, |ctx, mol| {
            ctx.reset_for_new_molecule();
            let mut output = Vec::new();
            for desc in descriptors {
                let value = desc.calculate(mol, ctx);
                value.to_floats(&mut output);
            }
            output
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Thread safety assertions
// ---------------------------------------------------------------------------

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<DescriptorValue>;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_to_floats() {
        let value = DescriptorValue::Int(42);
        let mut vec = Vec::new();
        value.to_floats(&mut vec);
        assert_eq!(vec, vec![42.0]);
        assert_eq!(value.len(), 1);
    }

    #[test]
    fn test_float_to_floats() {
        let value = DescriptorValue::Float(3.14);
        let mut vec = Vec::new();
        value.to_floats(&mut vec);
        assert_eq!(vec, vec![3.14]);
        assert_eq!(value.len(), 1);
    }

    #[test]
    fn test_int_vec_to_floats() {
        let value = DescriptorValue::IntVec(vec![1, 2, 3]);
        let mut vec = Vec::new();
        value.to_floats(&mut vec);
        assert_eq!(vec, vec![1.0, 2.0, 3.0]);
        assert_eq!(value.len(), 3);
    }

    #[test]
    fn test_float_vec_to_floats() {
        let value = DescriptorValue::FloatVec(vec![1.1, 2.2, 3.3]);
        let mut vec = Vec::new();
        value.to_floats(&mut vec);
        assert_eq!(vec, vec![1.1, 2.2, 3.3]);
        assert_eq!(value.len(), 3);
    }

    #[test]
    fn test_binary_to_floats() {
        let bv = bitvec![u8, Lsb0; 1, 0, 1, 1, 0];
        let value = DescriptorValue::Binary(bv);
        let mut vec = Vec::new();
        value.to_floats(&mut vec);
        assert_eq!(vec, vec![1.0, 0.0, 1.0, 1.0, 0.0]);
        assert_eq!(value.len(), 5);
    }

    #[test]
    fn test_from_bits() {
        let bits = vec![true, false, true];
        let value = DescriptorValue::from_bits(&bits);
        assert_eq!(value.len(), 3);
        match value {
            DescriptorValue::Binary(bv) => {
                assert_eq!(bv[0], true);
                assert_eq!(bv[1], false);
                assert_eq!(bv[2], true);
            }
            _ => panic!("Expected Binary"),
        }
    }

    #[test]
    fn test_extract_int() {
        let value = DescriptorValue::Int(99);
        assert_eq!(value.as_int(), 99);
    }

    #[test]
    fn test_extract_float() {
        let value = DescriptorValue::Float(1.23);
        assert_eq!(value.as_float(), 1.23);
    }

    #[test]
    fn test_extract_int_vec() {
        let value = DescriptorValue::IntVec(vec![1, 2, 3]);
        assert_eq!(value.as_int_vec(), &[1, 2, 3]);
    }

    #[test]
    fn test_extract_float_vec() {
        let value = DescriptorValue::FloatVec(vec![1.1, 2.2]);
        assert_eq!(value.as_float_vec(), &[1.1, 2.2]);
    }

    #[test]
    fn test_extract_binary() {
        let bv = bitvec![u8, Lsb0; 1, 0, 1];
        let value = DescriptorValue::Binary(bv.clone());
        assert_eq!(value.as_binary(), &bv);
    }

    #[test]
    fn test_is_empty() {
        assert!(!DescriptorValue::Int(1).is_empty());
        assert!(!DescriptorValue::Binary(bitvec![u8, Lsb0; 1]).is_empty());
        assert!(DescriptorValue::IntVec(vec![]).is_empty());
        assert!(DescriptorValue::FloatVec(vec![]).is_empty());
    }

    #[test]
    fn test_binary_memory_efficiency() {
        // 166 bits (MACCS size)
        let bv = bitvec![u8, Lsb0; 0; 166];
        let value = DescriptorValue::Binary(bv);

        // BitVec is much smaller than Vec<f32>
        // Vec<f32>: 166 * 4 = 664 bytes
        // BitVec: ~21 bytes (166 bits / 8 + overhead)
        let bitvec_bytes = std::mem::size_of_val(value.as_binary());
        println!("BitVec size: {} bytes", bitvec_bytes);

        // Verify conversion still works
        let mut vec = Vec::new();
        value.to_floats(&mut vec);
        assert_eq!(vec.len(), 166);
    }

    // Example Descriptor implementation for testing
    struct ExampleDescriptor;

    impl Descriptor for ExampleDescriptor {
        fn name(&self) -> &str {
            "Example"
        }

        fn calculate(&self, mol: &mut Molecule, ctx: &mut crate::desc::context::CalculationContext) -> DescriptorValue {
            DescriptorValue::Int(mol.num_atoms() as u32)
        }
    }

    #[test]
    fn test_descriptor_trait() {
        use crate::io::smiles::parse_smiles;

        let mut mol = parse_smiles("CCO").unwrap();
        let desc = ExampleDescriptor;
        let value = desc.calculate(&mut mol, &mut crate::desc::context::CalculationContext::new());

        assert_eq!(desc.name(), "Example");
        assert_eq!(value.as_int(), 3); // 3 heavy atoms
    }

    #[test]
    fn test_build_ml_vector() {
        use crate::io::smiles::parse_smiles;

        let mut mol = parse_smiles("CC").unwrap();
        let descriptors: Vec<Box<dyn Descriptor>> = vec![Box::new(ExampleDescriptor)];

        let vec = build_ml_vector(&mut mol, &descriptors);
        assert_eq!(vec, vec![2.0]); // 2 heavy atoms
    }
}
