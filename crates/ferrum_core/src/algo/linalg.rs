//! Small dense linear-algebra helpers.
//!
//! - [`symmetric_eigenvalues`]: full spectrum (EISPACK `tred1` + QL), kept as
//!   the reference implementation.
//! - [`extreme_eigenvalues`] / [`extreme_eigenvalues_pair`] /
//!   [`burden_extremes`]: only the smallest and largest eigenvalue, as needed
//!   by the BCUT descriptors. Reduction to tridiagonal form, then a Laguerre
//!   iteration on the tridiagonal matrix whose result is certified by Sturm
//!   counts (bisection fallback). The reduction is Householder's, with the
//!   rank-2 update of each step deferred and fused into the next step's
//!   matrix-vector product, or for large Burden matrices (sparse molecular
//!   graph plus a constant fill) a band reduction by plane rotations
//!   (`lanes::band`). Every step is backward stable, so the extremes agree
//!   with full dense solvers (LAPACK, JAMA, Eigen, numpy) to
//!   `O(n * eps * ||A||)`. Several same-size problems run in lock step, one
//!   per SIMD lane (`lanes`: two to eight problems on `f2`, NEON on aarch64).

mod f2;
pub(crate) mod lanes;

use lanes::tri;

/// Eigenvalues of a real symmetric `n x n` matrix (row-major), ascending.
///
/// Householder reduction to tridiagonal form followed by the implicit QL
/// algorithm (eigenvalues only). `a` is used as workspace and overwritten.
pub fn symmetric_eigenvalues(n: usize, a: &mut [f64]) -> Vec<f64> {
    assert_eq!(a.len(), n * n);
    if n == 0 {
        return Vec::new();
    }
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    tridiagonalize(n, a, &mut d, &mut e);
    tql1(n, &mut d, &mut e);
    d.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    d
}

/// Reusable scratch buffers for [`extreme_eigenvalues`] /
/// [`extreme_eigenvalues_pair`] / [`burden_extremes`].
#[derive(Default)]
pub struct EigenWorkspace {
    d: Vec<f64>,
    e: Vec<f64>,
    u: Vec<f64>,
    p: Vec<f64>,
    pq: Vec<f64>,
    /// matrix storage: lane-interleaved packed lower triangles
    a: Vec<f64>,
}

impl EigenWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    fn reset_vectors(&mut self, len: usize) {
        for v in [
            &mut self.d,
            &mut self.e,
            &mut self.u,
            &mut self.p,
            &mut self.pq,
        ] {
            v.clear();
            v.resize(len, 0.0);
        }
    }

    /// Drops very large buffers (kept workspaces are thread-local).
    fn trim(&mut self) {
        if self.a.capacity() > 1 << 20 {
            *self = Self::default();
        }
    }
}

/// Smallest and largest eigenvalue `(min, max)` of a real symmetric `n x n`
/// matrix (row-major). Only the lower triangle (`a[i * n + j]`, `j <= i`) is
/// read; `a` is left untouched.
///
/// Householder reduction to tridiagonal form (same reflectors as EISPACK
/// `tred1`, vectorised over contiguous rows of the lower triangle) followed by
/// a Laguerre iteration on the tridiagonal matrix whose result is certified
/// by Sturm counts (bisection fallback, LAPACK `dstebz` style). Both steps are
/// backward stable, so the result agrees with a full dense solver (QL,
/// LAPACK, JAMA, Eigen) to `O(eps * ||A||)`.
pub fn extreme_eigenvalues(ws: &mut EigenWorkspace, n: usize, a: &[f64]) -> (f64, f64) {
    assert_eq!(a.len(), n * n);
    let mut out = [(f64::NAN, f64::NAN)];
    solve_dense::<1>(ws, n, &[a], &mut out);
    out[0]
}

/// [`extreme_eigenvalues`] for two symmetric matrices of the same size,
/// solved in lock step (one per SIMD lane); `[(min_a, max_a), (min_b, max_b)]`.
/// Only the lower triangles are read; `a` and `b` are left untouched.
pub fn extreme_eigenvalues_pair(
    ws: &mut EigenWorkspace,
    n: usize,
    a: &[f64],
    b: &[f64],
) -> [(f64, f64); 2] {
    assert!(a.len() == n * n && b.len() == n * n);
    let mut out = [(f64::NAN, f64::NAN); 2];
    solve_dense::<1>(ws, n, &[a, b], &mut out);
    out
}

/// Extremes of up to `2 * V` dense matrices (lower triangles read) into `out`.
fn solve_dense<const V: usize>(ws: &mut EigenWorkspace, n: usize, mats: &[&[f64]], out: &mut [(f64, f64)]) {
    if n == 0 {
        out.fill((f64::NAN, f64::NAN));
        return;
    }
    let mut buf = std::mem::take(&mut ws.a);
    buf.resize(2 * V * tri(n), 0.0);
    lanes::assemble_dense::<V>(n, mats, &mut buf);
    let bounds: [SpectrumBounds; lanes::MAX_LANES] =
        std::array::from_fn(|m| gershgorin_bounds(n, mats[m.min(mats.len() - 1)], &mut ws.d));
    solve_lanes::<V>(ws, n, &mut buf, &bounds, out);
    ws.a = buf;
}

/// Known bounds on the spectrum of a symmetric matrix: all eigenvalues lie
/// in `[lo, hi]`, and `min <= min_ub`, `max >= max_lb` (e.g. Gershgorin and
/// principal submatrices).
#[derive(Clone, Copy, Debug)]
struct SpectrumBounds {
    lo: f64,
    hi: f64,
    min_ub: f64,
    max_lb: f64,
}

impl SpectrumBounds {
    const EMPTY: Self = Self {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
        min_ub: f64::INFINITY,
        max_lb: f64::NEG_INFINITY,
    };
}

/// Extremes of the `2 * V` lane-interleaved packed matrices in `a`
/// (overwritten) into `out[..min(out.len(), 2 * V)]`; `bounds`: bounds on
/// the spectra (per lane).
fn solve_lanes<const V: usize>(
    ws: &mut EigenWorkspace,
    n: usize,
    a: &mut [f64],
    bounds: &[SpectrumBounds],
    out: &mut [(f64, f64)],
) {
    let w = 2 * V;
    let k = out.len().min(w);
    if n == 1 {
        for (m, o) in out[..k].iter_mut().enumerate() {
            *o = (a[m], a[m]);
        }
        return;
    }
    ws.reset_vectors(w * n);
    lanes::tridiagonalize::<V>(n, a, &mut ws.d, &mut ws.e, &mut ws.u, &mut ws.p, &mut ws.pq);
    let mut res = [(f64::NAN, f64::NAN); lanes::MAX_LANES];
    lanes::tridiagonal_extremes::<V>(n, &ws.d, &ws.e, &mut ws.u, bounds, &mut res);
    out[..k].copy_from_slice(&res[..k]);
}

/// Extremes of up to `2 * V` Burden matrices (`jobs`, all `n x n` with
/// `n >= 3`) through band reduction (`lanes::band`; `inv[old] = new` atom
/// order with half-bandwidth `b`) into `out`; `bounds`: bounds on the
/// spectra (per lane), `buf` is scratch.
#[allow(clippy::too_many_arguments)]
fn solve_band<const V: usize>(
    ws: &mut EigenWorkspace,
    n: usize,
    b: usize,
    inv: &[u32],
    jobs: &[BurdenMatrix],
    buf: &mut Vec<f64>,
    bounds: &[SpectrumBounds],
    out: &mut [(f64, f64)],
) {
    let w = 2 * V;
    buf.resize(n * (b + 2) * w, 0.0);
    lanes::band::assemble::<V>(n, b, inv, jobs, buf);
    ws.reset_vectors(w * n);
    lanes::band::tridiagonalize::<V>(n, b, buf, &mut ws.u, &mut ws.d, &mut ws.e);
    // Q^T A Q = T + f n e_0 e_0^T
    for m in 0..w {
        ws.d[m] += jobs[m.min(jobs.len() - 1)].fill * n as f64;
    }
    let mut res = [(f64::NAN, f64::NAN); MAX_GROUP];
    if w <= lanes::MAX_LANES {
        lanes::tridiagonal_extremes::<V>(n, &ws.d, &ws.e, &mut ws.u, bounds, &mut res);
    } else {
        // four lanes at a time (de-interleaved copies)
        let l = lanes::MAX_LANES;
        for part in 0..w / l {
            for i in 0..n {
                for m in 0..l {
                    ws.p[l * i + m] = ws.d[w * i + l * part + m];
                    ws.pq[l * i + m] = ws.e[w * i + l * part + m];
                }
            }
            lanes::tridiagonal_extremes::<2>(n, &ws.p, &ws.pq, &mut ws.u, &bounds[l * part..], &mut res[l * part..]);
        }
    }
    let k = out.len().min(w);
    out[..k].copy_from_slice(&res[..k]);
}

/// Spectrum bounds of a Burden matrix from its description (O(n + entries);
/// `r` is scratch). Gershgorin: each row sum is bounded by
/// `|fill| * (n - 1) + sum |w|` over the listed entries of that row (valid
/// even for repeated pairs). Inner bounds (Cauchy interlacing): the extreme
/// eigenvalues of the 1 x 1 and the bonded 2 x 2 principal submatrices.
fn burden_bounds(n: usize, m: &BurdenMatrix, r: &mut Vec<f64>) -> SpectrumBounds {
    r.clear();
    r.resize(n, m.fill.abs() * (n as f64 - 1.0));
    let mut b = SpectrumBounds::EMPTY;
    let diag = &m.diag[..n];
    for &(i, j, w) in m.entries {
        if i != j {
            let (i, j) = (i as usize, j as usize);
            r[i] += w.abs();
            r[j] += w.abs();
            let (mid, half) = (0.5 * (diag[i] + diag[j]), 0.5 * (diag[i] - diag[j]));
            let rad = half.hypot(w);
            b.min_ub = b.min_ub.min(mid - rad);
            b.max_lb = b.max_lb.max(mid + rad);
        }
    }
    for (ri, &d) in r.iter().zip(diag) {
        b.lo = b.lo.min(d - ri);
        b.hi = b.hi.max(d + ri);
        b.min_ub = b.min_ub.min(d);
        b.max_lb = b.max_lb.max(d);
    }
    b
}

/// Gershgorin interval `(lo, hi)` of the symmetric `n x n` row-major matrix
/// whose lower triangle is `a` (`r` is scratch). It encloses the spectrum
/// and is usually much tighter than the Gershgorin interval of the
/// tridiagonal form (Burden matrices are strongly diagonally dominant),
/// which saves iterations in the Laguerre iteration. Inner bounds: the
/// extreme diagonal elements.
fn gershgorin_bounds(n: usize, a: &[f64], r: &mut Vec<f64>) -> SpectrumBounds {
    r.clear();
    r.resize(n, 0.0);
    for i in 0..n {
        let mut ri = 0.0;
        for j in 0..i {
            let v = a[i * n + j].abs();
            ri += v;
            r[j] += v;
        }
        r[i] += ri;
    }
    let mut b = SpectrumBounds::EMPTY;
    for i in 0..n {
        let d = a[i * n + i];
        b.lo = b.lo.min(d - r[i]);
        b.hi = b.hi.max(d + r[i]);
        b.min_ub = b.min_ub.min(d);
        b.max_lb = b.max_lb.max(d);
    }
    b
}

/// A Burden-type symmetric matrix: `fill` everywhere off the diagonal except
/// for the listed `(i, j, weight)` entries (each unordered pair once; later
/// entries win), and `diag` on the diagonal.
#[derive(Clone, Copy)]
pub struct BurdenMatrix<'a> {
    pub fill: f64,
    pub entries: &'a [(u32, u32, f64)],
    pub diag: &'a [f64],
}

thread_local! {
    static BURDEN_WS: std::cell::RefCell<EigenWorkspace> = std::cell::RefCell::new(EigenWorkspace::new());
}

/// Largest `n` for which the dense reduction runs four matrices per pass
/// (beyond it two per pass are as fast: the fused kernel's working set
/// outgrows the L1 cache).
const DENSE_QUAD_MAX_N: usize = 56;

/// Most matrices per band-reduction pass (`V = 4`).
const MAX_GROUP: usize = 8;

/// Smallest `n` worth a band reduction.
const BAND_MIN_N: usize = 48;

/// Band reduction beats the dense one for `n >= factor * (b + 2)`, with the
/// factor depending on the matrices per pass (`1 << k` for `k` = 1, 2, 3):
/// with eight lanes the rotations are throughput-bound, with fewer the
/// latency of each rotation's dependency chain dominates.
const BAND_FACTOR: [usize; 4] = [usize::MAX, 19, 12, 8];

/// Atom order for the band reduction of `jobs` (`inv[old] = new`) and its
/// half-bandwidth `b`, for `n` large enough that it may beat the dense
/// reduction (see [`BAND_FACTOR`]).
fn band_order(n: usize, jobs: &[BurdenMatrix]) -> Option<(Vec<u32>, usize)> {
    if n < BAND_MIN_N {
        return None;
    }
    let edges: Vec<(u32, u32)> = jobs.iter().flat_map(|j| j.entries.iter().map(|&(a, b, _)| (a, b))).collect();
    let (order, b) = lanes::band::rcm_order(n, &edges);
    if n < BAND_FACTOR[3] * (b + 2) {
        return None;
    }
    let mut inv = vec![0u32; n];
    for (k, &o) in order.iter().enumerate() {
        inv[o as usize] = k as u32;
    }
    Some((inv, b))
}

/// How a group of matrices is reduced (the suffix is `V`: `2 * V` matrices
/// per pass).
#[derive(Clone, Copy)]
enum Method {
    /// dense Householder reduction ([`solve_lanes`])
    Dense1,
    Dense2,
    /// band reduction ([`solve_band`])
    Band1,
    Band2,
    Band4,
}

/// Extreme eigenvalues `(min, max)` of each Burden matrix (all `n x n`;
/// `(NaN, NaN)` for a matrix with a non-finite element).
///
/// Small or densely connected molecules: the matrices are assembled
/// straight into the solver's (thread-local) workspace and reduced by
/// Householder transformations, four at a time in lock step (`lanes`,
/// `V = 2`; two for large `n` or a remainder of one or two). Large
/// molecules with a narrow band after reordering: band reduction
/// (`lanes::band`), up to eight matrices per pass. Unused lanes repeat the
/// last matrix.
pub fn burden_extremes(n: usize, jobs: &[BurdenMatrix]) -> Vec<(f64, f64)> {
    if n == 0 {
        return vec![(f64::NAN, f64::NAN); jobs.len()];
    }
    BURDEN_WS.with(|ws| {
        let mut ws = ws.borrow_mut();
        let ws = &mut *ws;
        let mut out = vec![(f64::NAN, f64::NAN); jobs.len()];
        let band = if n >= 3 { band_order(n, jobs) } else { None };
        let band_ok = |k: usize| band.as_ref().is_some_and(|(_, b)| n >= BAND_FACTOR[k] * (b + 2));
        let mut start = 0;
        while start < jobs.len() {
            let rest = jobs.len() - start;
            let (take, method) = if rest > 4 && band_ok(3) {
                (rest.min(MAX_GROUP), Method::Band4)
            } else if rest > 2 && band_ok(2) {
                (rest.min(4), Method::Band2)
            } else if rest > 2 && n <= DENSE_QUAD_MAX_N {
                (rest.min(4), Method::Dense2)
            } else if rest <= 2 && band_ok(1) {
                (rest, Method::Band1)
            } else {
                (rest.min(2), Method::Dense1)
            };
            let group = &jobs[start..start + take];
            let mut bounds = [SpectrumBounds::EMPTY; MAX_GROUP];
            for (m, job) in group.iter().enumerate() {
                bounds[m] = burden_bounds(n, job, &mut ws.d);
            }
            // unused lanes repeat the last matrix
            let last = bounds[take - 1];
            bounds[take..].fill(last);
            let mut buf = std::mem::take(&mut ws.a);
            let res = &mut out[start..start + take];
            let (inv, b) = band.as_ref().map_or((&[][..], 0), |(inv, b)| (&inv[..], *b));
            match method {
                Method::Band1 => solve_band::<1>(ws, n, b, inv, group, &mut buf, &bounds, res),
                Method::Band2 => solve_band::<2>(ws, n, b, inv, group, &mut buf, &bounds, res),
                Method::Band4 => solve_band::<4>(ws, n, b, inv, group, &mut buf, &bounds, res),
                // Every packed entry is written, so stale contents need no clearing.
                Method::Dense2 => {
                    buf.resize(4 * tri(n), 0.0);
                    lanes::assemble_burden::<2>(n, group, &mut buf);
                    solve_lanes::<2>(ws, n, &mut buf, &bounds, res);
                }
                Method::Dense1 => {
                    buf.resize(2 * tri(n), 0.0);
                    lanes::assemble_burden::<1>(n, group, &mut buf);
                    solve_lanes::<1>(ws, n, &mut buf, &bounds, res);
                }
            }
            ws.a = buf;
            start += take;
        }
        ws.trim();
        // a non-finite entry makes the whole spectrum undefined
        for (o, m) in out.iter_mut().zip(jobs) {
            let finite = m.fill.is_finite()
                && m.diag[..n].iter().all(|v| v.is_finite())
                && m.entries.iter().all(|e| e.2.is_finite());
            if !finite {
                *o = (f64::NAN, f64::NAN);
            }
        }
        out
    })
}

/// `a * b + c`, fused where the target has a native FMA instruction (fused
/// and unfused results differ only by rounding, which these backward-stable
/// kernels tolerate).
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}

/// Laguerre iterations per extreme eigenvalue before the multisection fallback.
const MAX_IT: usize = 40;

/// Iteration state for one extreme eigenvalue of a symmetric tridiagonal
/// matrix: `side` 0 = minimum, 1 = maximum. `[lo, hi]` always brackets the
/// eigenvalue (certified by Sturm counts); `x` is the next Laguerre point and
/// `probes` the next two count-only points.
#[derive(Clone, Copy)]
struct Extreme {
    side: usize,
    n: usize,
    tnorm: f64,
    lo: f64,
    hi: f64,
    x: f64,
    probes: [f64; 2],
    done: bool,
    result: f64,
}

impl Extreme {
    /// `[lo, hi]`: initial bracket of the eigenvalue (its outer end must lie
    /// outside the spectrum); `tnorm`: norm estimate.
    fn new(side: usize, n: usize, lo: f64, hi: f64, tnorm: f64) -> Self {
        let (lo, hi) = (lo.min(hi), hi.max(lo));
        // First-round probes: the extreme eigenvalue usually lies close to
        // its inner bound, so test points near it; one found outside the
        // spectrum becomes the next (much closer) Laguerre start.
        let w = hi - lo;
        let probes = if side == 0 {
            [hi - w / 16.0, hi - w / 4.0]
        } else {
            [lo + w / 16.0, lo + w / 4.0]
        };
        Self {
            side,
            n,
            tnorm,
            lo,
            hi,
            x: if side == 0 { lo } else { hi },
            probes,
            done: false,
            result: f64::NAN,
        }
    }

    /// Takes the Sturm count `c` (eigenvalues below `x`), the counts at the
    /// two probes and the Laguerre sums `g = p'/p`, `h = (p'/p)^2 - p''/p` at
    /// `x`, tightens the bracket and picks the next points.
    ///
    /// Laguerre's method on a polynomial with only real roots never steps
    /// over a root: from outside the spectrum it moves monotonically onto the
    /// extreme eigenvalue, and from a point with exactly one eigenvalue
    /// between it and the outside (an overshoot by rounding) it moves back
    /// onto that eigenvalue. Elsewhere the bracket is bisected.
    fn update(&mut self, c: f64, probe_counts: [f64; 2], g: f64, h: f64) {
        if self.done {
            return;
        }
        let nf = self.n as f64;
        let pts = [
            (self.x, c),
            (self.probes[0], probe_counts[0]),
            (self.probes[1], probe_counts[1]),
        ];
        for (sg, cc) in pts {
            // the point lies below the wanted eigenvalue
            let below = if self.side == 0 { cc == 0.0 } else { cc < nf };
            if below {
                self.lo = self.lo.max(sg);
            } else {
                self.hi = self.hi.min(sg);
            }
        }
        // Counts are exact for a matrix within a few ulps (relative,
        // elementwise) of T, i.e. reliable to a few eps * ||T||.
        let eps = f64::EPSILON;
        let tol = eps * (4.0 * self.lo.abs().max(self.hi.abs()) + 8.0 * self.tnorm);
        if self.hi - self.lo <= tol {
            self.result = 0.5 * (self.lo + self.hi);
            self.done = true;
            return;
        }
        // `Some(true)`: step up (denominator G - S), `Some(false)`: step down.
        let up = match self.side {
            0 if c == 0.0 => Some(true),
            0 if c == 1.0 => Some(false),
            1 if c == nf => Some(false),
            1 if c == nf - 1.0 => Some(true),
            _ => None,
        };
        let mut xn = f64::NAN;
        if let Some(up) = up {
            if g.is_finite() && h.is_finite() {
                let s = ((nf - 1.0) * fma(nf, h, -g * g)).max(0.0).sqrt();
                xn = self.x - nf / if up { g - s } else { g + s };
            }
            // From outside: a probe certified outside the spectrum and closer
            // to the eigenvalue (the bracket end) is the better next point.
            let from_outside = if self.side == 0 { c == 0.0 } else { c == nf };
            if from_outside {
                xn = if self.side == 0 {
                    xn.max(self.lo)
                } else {
                    xn.min(self.hi)
                };
            }
        }
        let inside = if self.side == 0 {
            xn >= self.lo && xn < self.hi
        } else {
            xn > self.lo && xn <= self.hi
        };
        if !inside {
            xn = 0.5 * (self.lo + self.hi);
        }
        self.x = xn;
        self.probes = [xn - 0.5 * tol, xn + 0.5 * tol];
    }
}

/// Sturm-count multisection (LAPACK `dstebz` style). `br[0]` brackets the
/// smallest eigenvalue (`count(a) = 0`, `count(b) >= 1`), `br[1]` the largest
/// (`count(a) < n`, `count(b) = n`); only lanes with `need` are refined.
fn multisection(
    n: usize,
    d: &[f64],
    e2: &[f64],
    pivmin: f64,
    br: [(f64, f64); 2],
    need: [bool; 2],
) -> [f64; 2] {
    const K: usize = 4; // interior points per bracket and pass
    let eps = f64::EPSILON;
    let [(mut a0, mut b0), (mut a1, mut b1)] = br;
    if a0 > b0 {
        std::mem::swap(&mut a0, &mut b0);
    }
    if a1 > b1 {
        std::mem::swap(&mut a1, &mut b1);
    }
    let atol = 2.0 * pivmin;
    for _ in 0..200 {
        let done0 = !need[0] || b0 - a0 <= atol + 2.0 * eps * a0.abs().max(b0.abs());
        let done1 = !need[1] || b1 - a1 <= atol + 2.0 * eps * a1.abs().max(b1.abs());
        if done0 && done1 {
            break;
        }
        let mut sig = [0.0f64; 2 * K];
        let (w0, w1) = ((b0 - a0) / (K + 1) as f64, (b1 - a1) / (K + 1) as f64);
        for t in 0..K {
            sig[t] = a0 + w0 * (t + 1) as f64;
            sig[K + t] = a1 + w1 * (t + 1) as f64;
        }
        let mut q = [0.0f64; 2 * K];
        let mut cnt = [0.0f64; 2 * K];
        for t in 0..2 * K {
            let v = d[0] - sig[t];
            let v = if v.abs() < pivmin { -pivmin } else { v };
            q[t] = v;
            cnt[t] = if v < 0.0 { 1.0 } else { 0.0 };
        }
        for i in 1..n {
            let (di, ei2) = (d[i], e2[i]);
            for t in 0..2 * K {
                let v = (di - sig[t]) - ei2 / q[t];
                let v = if v.abs() < pivmin { -pivmin } else { v };
                q[t] = v;
                cnt[t] += if v < 0.0 { 1.0 } else { 0.0 };
            }
        }
        if !done0 {
            // first point with count >= 1
            let mut na = a0;
            let mut nb = b0;
            for t in 0..K {
                if cnt[t] >= 1.0 {
                    nb = sig[t];
                    break;
                }
                na = sig[t];
            }
            if na == a0 && nb == b0 {
                // no progress possible (interval below resolution)
                a0 = 0.5 * (a0 + b0);
                b0 = a0;
            } else {
                a0 = na;
                b0 = nb;
            }
        }
        if !done1 {
            let nn = n as f64;
            let mut na = a1;
            let mut nb = b1;
            for t in 0..K {
                let s = sig[K + t];
                if cnt[K + t] >= nn {
                    nb = s;
                    break;
                }
                na = s;
            }
            if na == a1 && nb == b1 {
                a1 = 0.5 * (a1 + b1);
                b1 = a1;
            } else {
                a1 = na;
                b1 = nb;
            }
        }
    }
    [0.5 * (a0 + b0), 0.5 * (a1 + b1)]
}

/// Householder tridiagonalization (EISPACK `tred1`, eigenvalues only).
fn tridiagonalize(n: usize, a: &mut [f64], d: &mut [f64], e: &mut [f64]) {
    let idx = |i: usize, j: usize| i * n + j;
    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0;
        if l > 0 {
            let scale: f64 = (0..=l).map(|k| a[idx(i, k)].abs()).sum();
            if scale == 0.0 {
                e[i] = a[idx(i, l)];
            } else {
                for k in 0..=l {
                    a[idx(i, k)] /= scale;
                    h += a[idx(i, k)] * a[idx(i, k)];
                }
                let f = a[idx(i, l)];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                a[idx(i, l)] = f - g;
                let mut f_acc = 0.0;
                for j in 0..=l {
                    let mut g = 0.0;
                    for k in 0..=j {
                        g += a[idx(j, k)] * a[idx(i, k)];
                    }
                    for k in j + 1..=l {
                        g += a[idx(k, j)] * a[idx(i, k)];
                    }
                    e[j] = g / h;
                    f_acc += e[j] * a[idx(i, j)];
                }
                let hh = f_acc / (h + h);
                for j in 0..=l {
                    let f = a[idx(i, j)];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 0..=j {
                        a[idx(j, k)] -= f * e[k] + g * a[idx(i, k)];
                    }
                }
            }
        } else {
            e[i] = a[idx(i, l)];
        }
        d[i] = h;
    }
    for i in 0..n {
        d[i] = a[idx(i, i)];
    }
}

/// Implicit QL on a symmetric tridiagonal matrix (diagonal `d`, sub-diagonal
/// `e[1..]`); eigenvalues are left in `d`.
fn tql1(n: usize, d: &mut [f64], e: &mut [f64]) {
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 60 {
                break;
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r.abs() } else { -r.abs() });
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut i = m;
            let mut underflow = false;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            if underflow {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigenvalues_of_small_matrices() {
        let mut a = vec![2.0, 1.0, 1.0, 2.0];
        let ev = symmetric_eigenvalues(2, &mut a);
        assert!((ev[0] - 1.0).abs() < 1e-12 && (ev[1] - 3.0).abs() < 1e-12);

        // path graph P3 adjacency: eigenvalues -sqrt(2), 0, sqrt(2)
        let mut a = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let ev = symmetric_eigenvalues(3, &mut a);
        let s2 = 2f64.sqrt();
        assert!((ev[0] + s2).abs() < 1e-12 && ev[1].abs() < 1e-12 && (ev[2] - s2).abs() < 1e-12);
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn check(n: usize, a: &[f64], ws: &mut EigenWorkspace) {
        // pair path: (a, a) and (a, -a) must give the single-path results
        let neg: Vec<f64> = a.iter().map(|v| -v).collect();
        let single = extreme_eigenvalues(ws, n, a);
        let [p0, p1] = extreme_eigenvalues_pair(ws, n, a, &neg);
        let sc = single.0.abs().max(single.1.abs()).max(1.0);
        assert!(
            (p0.0 - single.0).abs() <= 1e-13 * sc && (p0.1 - single.1).abs() <= 1e-13 * sc,
            "n={n} pair lane 0"
        );
        assert!(
            (p1.0 + single.1).abs() <= 1e-13 * sc && (p1.1 + single.0).abs() <= 1e-13 * sc,
            "n={n} pair lane 1"
        );
        let mut m = a.to_vec();
        let full = symmetric_eigenvalues(n, &mut m);
        let (lo, hi) = extreme_eigenvalues(ws, n, a);
        let norm = full[0].abs().max(full[n - 1].abs()).max(1.0);
        assert!(
            (lo - full[0]).abs() <= 1e-13 * norm,
            "n={n} lo {lo} vs {}",
            full[0]
        );
        assert!(
            (hi - full[n - 1]).abs() <= 1e-13 * norm,
            "n={n} hi {hi} vs {}",
            full[n - 1]
        );
    }

    #[test]
    fn extreme_eigenvalues_match_full_solver() {
        let mut ws = EigenWorkspace::new();
        let mut st = 42u64;
        for n in 1..80 {
            for kind in 0..4 {
                // Burden-like: 0.001 fill, sparse "bonds", property diagonal
                let mut a = vec![0.001f64; n * n];
                for i in 1..n {
                    let j = (lcg(&mut st) * i as f64) as usize;
                    let w = [0.1, 0.15, 0.2, 0.3, 1.0, 0.7][(lcg(&mut st) * 6.0) as usize];
                    a[i * n + j] = w;
                    a[j * n + i] = w;
                }
                for i in 0..n {
                    a[i * n + i] = match kind {
                        0 => 12.0, // constant: heavy degeneracy
                        1 => [12.011, 15.999, 14.007, 32.06][(lcg(&mut st) * 4.0) as usize],
                        2 => lcg(&mut st) * 0.8 - 0.4, // charges
                        _ => 0.0,
                    };
                }
                check(n, &a, &mut ws);
            }
            // fully random dense symmetric
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..=i {
                    let v = lcg(&mut st) * 2.0 - 1.0;
                    a[i * n + j] = v;
                    a[j * n + i] = v;
                }
            }
            check(n, &a, &mut ws);
        }
        // zero matrix and diagonal matrix
        check(5, &[0.0; 25], &mut ws);
        let mut a = vec![0.0; 16];
        for i in 0..4 {
            a[i * 4 + i] = i as f64 - 1.5;
        }
        check(4, &a, &mut ws);
    }

    /// Bonds of a molecule-like graph on `n` atoms, a tree plus a few ring
    /// closures, each unordered pair once (the [`BurdenMatrix`] contract):
    /// `local` grows a chain with short branches and small rings (narrow band
    /// after reordering), otherwise parents and ring closures are random.
    fn molecule_like(n: usize, local: bool, st: &mut u64) -> Vec<(u32, u32, f64)> {
        let mut entries: Vec<(u32, u32, f64)> = Vec::new();
        for i in 1..n {
            let j = if local { i - 1 - ((lcg(st) * 3.0) as usize).min(i - 1) } else { (lcg(st) * i as f64) as usize };
            entries.push((i as u32, j as u32, [0.1, 0.15, 0.2, 0.3][(lcg(st) * 4.0) as usize]));
        }
        for _ in 0..n / 6 {
            let i = (lcg(st) * n as f64) as u32;
            let j = if local { i.saturating_sub(5) } else { (lcg(st) * n as f64) as u32 };
            if i != j && !entries.iter().any(|&(a, b, _)| (a, b) == (i, j) || (a, b) == (j, i)) {
                entries.push((i, j, 0.15));
            }
        }
        entries
    }

    /// Diagonals of `count` Burden matrices: masses, charges or constant.
    fn burden_diags(n: usize, count: usize, kind: usize, st: &mut u64) -> Vec<Vec<f64>> {
        (0..count)
            .map(|_| {
                (0..n)
                    .map(|_| match kind {
                        0 => [12.011, 15.999, 14.007, 32.06][(lcg(st) * 4.0) as usize],
                        1 => lcg(st) * 0.8 - 0.4,
                        _ => 1.0,
                    })
                    .collect()
            })
            .collect()
    }

    /// `(min, max)` eigenvalue of a Burden matrix by the full QL solver.
    fn burden_reference(n: usize, job: &BurdenMatrix) -> (f64, f64) {
        let mut a = vec![job.fill; n * n];
        for &(i, j, w) in job.entries {
            a[i as usize * n + j as usize] = w;
            a[j as usize * n + i as usize] = w;
        }
        for i in 0..n {
            a[i * n + i] = job.diag[i];
        }
        let full = symmetric_eigenvalues(n, &mut a);
        (full[0], full[n - 1])
    }

    fn assert_extremes(got: (f64, f64), want: (f64, f64), what: &str) {
        let norm = want.0.abs().max(want.1.abs()).max(1.0);
        assert!(
            (got.0 - want.0).abs() <= 1e-13 * norm && (got.1 - want.1).abs() <= 1e-13 * norm,
            "{what}: {got:?} vs {want:?}"
        );
    }

    #[test]
    fn band_reduction_matches_full_solver() {
        let mut st = 7u64;
        let mut ws = EigenWorkspace::new();
        for n in 3..70 {
            for kind in 0..3 {
                let entries = molecule_like(n, kind != 1, &mut st);
                let diags = burden_diags(n, 8, kind, &mut st);
                let jobs: Vec<BurdenMatrix> =
                    diags.iter().map(|d| BurdenMatrix { fill: 0.001, entries: &entries, diag: d }).collect();
                let edges: Vec<(u32, u32)> = entries.iter().map(|&(a, b, _)| (a, b)).collect();
                let (order, b) = lanes::band::rcm_order(n, &edges);
                let mut inv = vec![0u32; n];
                for (k, &o) in order.iter().enumerate() {
                    inv[o as usize] = k as u32;
                }
                let bounds: Vec<SpectrumBounds> = jobs.iter().map(|j| burden_bounds(n, j, &mut ws.d)).collect();
                // eight lanes, three (padded) lanes and one (padded) lane
                let mut out = [(0.0, 0.0); 12];
                let mut buf = Vec::new();
                solve_band::<4>(&mut ws, n, b, &inv, &jobs, &mut buf, &bounds, &mut out[..8]);
                solve_band::<2>(&mut ws, n, b, &inv, &jobs[..3], &mut buf, &bounds, &mut out[8..11]);
                solve_band::<1>(&mut ws, n, b, &inv, &jobs[..1], &mut buf, &bounds, &mut out[11..]);
                for (m, job) in jobs.iter().chain(&jobs[..3]).chain(&jobs[..1]).enumerate() {
                    assert_extremes(out[m], burden_reference(n, job), &format!("n={n} kind={kind} b={b} lane {m}"));
                }
            }
        }
    }

    #[test]
    fn burden_extremes_match_full_solver() {
        // every reduction path: dense with four / two lanes, band reduction
        // with eight / four / two lanes (large n), padded lanes
        let mut st = 11u64;
        for n in [1, 2, 7, 33, 60, 90, 130, 200] {
            let entries = molecule_like(n, true, &mut st);
            for count in [1, 2, 3, 4, 5, 8, 15] {
                if n >= 130 && count > 5 {
                    continue;
                }
                let kind = count % 3;
                let diags = burden_diags(n, count, kind, &mut st);
                let jobs: Vec<BurdenMatrix> =
                    diags.iter().map(|d| BurdenMatrix { fill: 0.001, entries: &entries, diag: d }).collect();
                for (k, (got, job)) in burden_extremes(n, &jobs).into_iter().zip(&jobs).enumerate() {
                    assert_extremes(got, burden_reference(n, job), &format!("n={n} count={count} job {k}"));
                }
            }
        }
    }
}
