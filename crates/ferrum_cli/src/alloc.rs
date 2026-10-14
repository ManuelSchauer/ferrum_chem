//! Global allocator of the CLI: mimalloc, with very large blocks from the
//! system allocator.
//!
//! mimalloc is the fastest allocator for the many small, short-lived
//! allocations of descriptor calculation. Its arena, however, keeps the
//! address range of freed multi-megabyte blocks (e.g. the memo tables a few
//! large molecules need) for a while and macOS still counts purged pages as
//! resident, so a handful of such blocks raised the peak RSS of a 14-thread
//! run by 100-200 MB. Blocks of [`LARGE`] bytes and more are rare, so they go
//! to the system allocator, which maps them on demand and returns them to the
//! OS when freed.

use std::alloc::{GlobalAlloc, Layout, System};

use mimalloc::MiMalloc;

/// Smallest block served by the system allocator.
pub const LARGE: usize = 10 << 10;

/// mimalloc for blocks below [`LARGE`] bytes, the system allocator above.
pub struct Allocator;

#[inline]
fn large(size: usize) -> bool {
    size >= LARGE
}

// SAFETY: every block is freed (and resized) by the allocator that returned
// it: `dealloc` and `realloc` receive the layout the block was allocated with,
// and its size selects the same allocator as at allocation.
unsafe impl GlobalAlloc for Allocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { if large(layout.size()) { System.alloc(layout) } else { MiMalloc.alloc(layout) } }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        unsafe { if large(layout.size()) { System.alloc_zeroed(layout) } else { MiMalloc.alloc_zeroed(layout) } }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { if large(layout.size()) { System.dealloc(ptr, layout) } else { MiMalloc.dealloc(ptr, layout) } }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe {
            match (large(layout.size()), large(new_size)) {
                (false, false) => MiMalloc.realloc(ptr, layout, new_size),
                (true, true) => System.realloc(ptr, layout, new_size),
                // crossing the threshold: move the block to the other allocator
                _ => {
                    let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
                    let new = self.alloc(new_layout);
                    if !new.is_null() {
                        std::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
                        self.dealloc(ptr, layout);
                    }
                    new
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_move_between_allocators_on_realloc() {
        let a = Allocator;
        unsafe {
            let layout = Layout::from_size_align(1000, 16).unwrap();
            let p = a.alloc(layout);
            for i in 0..1000 {
                *p.add(i) = i as u8;
            }
            // small -> large -> larger -> small
            let p = a.realloc(p, layout, 3 * LARGE);
            let p = a.realloc(p, Layout::from_size_align(3 * LARGE, 16).unwrap(), 5 * LARGE);
            let p = a.realloc(p, Layout::from_size_align(5 * LARGE, 16).unwrap(), 500);
            assert_eq!(p as usize % 16, 0);
            for i in 0..500 {
                assert_eq!(*p.add(i), i as u8);
            }
            a.dealloc(p, Layout::from_size_align(500, 16).unwrap());
            let z = a.alloc_zeroed(Layout::from_size_align(2 * LARGE, 8).unwrap());
            assert!((0..2 * LARGE).step_by(4096).all(|i| *z.add(i) == 0));
            a.dealloc(z, Layout::from_size_align(2 * LARGE, 8).unwrap());
        }
    }
}
