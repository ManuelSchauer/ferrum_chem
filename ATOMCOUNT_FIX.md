# AtomCount Descriptor Fix

**Date**: Feb 4, 2026  
**Issue**: AtomCount was only returning heavy atoms, not total atoms including hydrogens  
**Status**: ✅ Fixed

## Problem

The `AtomCount` descriptor was using `mol.num_atoms()` which only counts explicit atoms in the molecular graph (heavy atoms). This did not include implicit hydrogens, making it identical to `HeavyAtomCount`.

**Before Fix**:
- Ethanol (CCO / C2H6O): AtomCount = 3 (only C, C, O)
- Benzene (c1ccccc1 / C6H6): AtomCount = 6 (only carbons)

## Solution

Modified `AtomCount` to sum:
1. Heavy atom count: `mol.num_atoms()`
2. Total hydrogen count: Sum of `atom.total_h()` for all atoms

**After Fix**:
- Ethanol (CCO / C2H6O): AtomCount = 9 (2C + 1O + 6H) ✅
- Benzene (c1ccccc1 / C6H6): AtomCount = 12 (6C + 6H) ✅
- Methane (C / CH4): AtomCount = 5 (1C + 4H) ✅

## Implementation

```rust
impl Descriptor for AtomCount {
    fn calculate(&self, mol: &Molecule, output: &mut [Real]) {
        // Count heavy atoms + all hydrogens (explicit and implicit)
        let heavy_count = mol.num_atoms();
        let hydrogen_count: u32 = mol.atoms.iter()
            .map(|atom| atom.total_h() as u32)
            .sum();
        
        output[0] = (heavy_count as u32 + hydrogen_count) as Real;
    }
}
```

## Verification

Added 2 new tests:
- `test_atom_count_methane()` - Verifies CH4 = 5 atoms
- `test_atom_count_benzene()` - Verifies C6H6 = 12 atoms

Updated existing test:
- `test_atom_count()` - Now expects 9 atoms for ethanol (was 3)

## Descriptor Distinction

Now properly differentiated:

| Descriptor | Description | Example (CCO) |
|------------|-------------|---------------|
| **AtomCount** | Total atoms (heavy + H) | 9 (2C + 1O + 6H) |
| **HeavyAtomCount** | Only non-hydrogen atoms | 3 (2C + 1O) |
| **SingleAtomCount("C")** | Specific element only | 2 (carbons) |

## Test Results

- **Before fix**: 94 tests passing
- **After fix**: 96 tests passing (2 new tests added)
- All existing tests still pass ✅

## Files Modified

- `crates/ferrum_core/src/desc/atom_count.rs`
  - Updated `AtomCount::calculate()` implementation
  - Updated documentation
  - Added 2 new unit tests
  - Modified 1 existing test expectation

## Impact

This fix ensures that:
1. ✅ AtomCount returns chemically correct total atom counts
2. ✅ Molecular formulas match expected values (e.g., C6H6 = 12 atoms)
3. ✅ Clear distinction between AtomCount and HeavyAtomCount
4. ✅ Users get the expected behavior when counting total atoms

## Validated With

- Unit tests (9 tests in atom_count module)
- descriptor_demo.rs (shows Toluene C7H8 = 15 atoms)
- csv_demo.rs (shows Benzene C6H6 = 12 atoms)

All outputs now show chemically correct atom counts! 🎉
