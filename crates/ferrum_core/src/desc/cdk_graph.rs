//! Helpers shared by the CDK topological descriptors (Wiener numbers,
//! Petitjean number, eccentric connectivity index, MDE).
//!
//! CDK computes topological distances with `PathTools.computeFloydAPSP`, which
//! leaves atom pairs in different components at the sentinel distance
//! 999 999 999. That value leaks into several CDK descriptors of disconnected
//! molecules, so the helpers here reproduce it on top of the cached
//! breadth-first distance matrix (`DIST_INF` = unreachable).

use crate::algo::paths::DIST_INF;

/// Distance `PathTools.computeFloydAPSP` reports for unreachable atom pairs.
pub(crate) const CDK_UNREACHABLE: i32 = 999_999_999;

/// CDK topological distance for an entry of the cached distance matrix.
#[inline]
pub(crate) fn cdk_distance(d: u16) -> i32 {
    if d == DIST_INF {
        CDK_UNREACHABLE
    } else {
        d as i32
    }
}

/// CDK eccentricity of every atom: the largest CDK distance in its row of the
/// `n x n` distance matrix (0 for a lone atom, 999 999 999 for any atom of a
/// disconnected molecule).
pub(crate) fn cdk_eccentricities(dist: &[u16], n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| dist[i * n..(i + 1) * n].iter().map(|&d| cdk_distance(d)).max().unwrap_or(-1))
        .collect()
}
