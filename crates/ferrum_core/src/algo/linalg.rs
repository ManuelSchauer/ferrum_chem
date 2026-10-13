//! Small dense linear-algebra helpers.
//!
//! - [`symmetric_eigenvalues`]: full spectrum (EISPACK `tred1` + QL), kept as
//!   the reference implementation.
//! - [`extreme_eigenvalues`] / [`extreme_eigenvalues_pair`] /
//!   [`burden_extremes`]: only the smallest and largest eigenvalue, as needed
//!   by the BCUT descriptors. Householder tridiagonalization with the rank-2
//!   update of each step deferred and fused into the next step's
//!   matrix-vector product, then a Laguerre iteration on the tridiagonal
//!   matrix whose result is certified by Sturm counts (bisection fallback).
//!   Every step is backward stable, so the extremes agree with full dense
//!   solvers (LAPACK, JAMA, Eigen, numpy) to `O(n * eps * ||A||)`.
//!   `_pair` / `burden_extremes` run two same-size problems in lock step, one
//!   per SIMD lane (`f2`, NEON on aarch64), see `pair`.

mod f2;
mod pair;

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
/// [`extreme_eigenvalues_pair`].
#[derive(Default)]
pub struct EigenWorkspace {
    d: Vec<f64>,
    e: Vec<f64>,
    u: Vec<f64>,
    p: Vec<f64>,
    pu: Vec<f64>,
    pq: Vec<f64>,
    /// matrix storage: one `n x n` matrix, or two lane-interleaved ones
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
            &mut self.pu,
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
/// read; `a` is used as workspace and overwritten.
///
/// Householder reduction to tridiagonal form (same reflectors as EISPACK
/// `tred1`, vectorised over contiguous rows of the lower triangle) followed by
/// a Laguerre iteration on the tridiagonal matrix whose result is certified
/// by Sturm counts (bisection fallback, LAPACK `dstebz` style). Both steps are
/// backward stable, so the result agrees with a full dense solver (QL,
/// LAPACK, JAMA, Eigen) to `O(eps * ||A||)`.
pub fn extreme_eigenvalues(ws: &mut EigenWorkspace, n: usize, a: &mut [f64]) -> (f64, f64) {
    assert_eq!(a.len(), n * n);
    let bounds = if n > 1 {
        gershgorin_bounds(n, a, 1, 0, &mut ws.d)
    } else {
        SpectrumBounds::NONE
    };
    solve_single(ws, n, a, bounds)
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
    const NONE: Self = Self {
        lo: f64::NEG_INFINITY,
        hi: f64::INFINITY,
        min_ub: f64::INFINITY,
        max_lb: f64::NEG_INFINITY,
    };
}

/// [`extreme_eigenvalues`] with known `bounds` on the spectrum.
fn solve_single(
    ws: &mut EigenWorkspace,
    n: usize,
    a: &mut [f64],
    bounds: SpectrumBounds,
) -> (f64, f64) {
    match n {
        0 => return (f64::NAN, f64::NAN),
        1 => return (a[0], a[0]),
        _ => {}
    }
    ws.reset_vectors(n);
    tridiagonalize_lower(
        n,
        a,
        &mut ws.d,
        &mut ws.e,
        [&mut ws.u, &mut ws.p, &mut ws.pu, &mut ws.pq],
    );
    // reuse `u` for the squared off-diagonal
    tridiagonal_extremes(n, &ws.d, &ws.e, &mut ws.u, bounds)
}

/// [`extreme_eigenvalues`] for two symmetric matrices of the same size,
/// solved in lock step (one per SIMD lane); `[(min_a, max_a), (min_b, max_b)]`.
/// Only the lower triangles are read; `a` and `b` are left untouched. The
/// per-matrix arithmetic is that of the single-matrix path (up to summation
/// order).
pub fn extreme_eigenvalues_pair(
    ws: &mut EigenWorkspace,
    n: usize,
    a: &[f64],
    b: &[f64],
) -> [(f64, f64); 2] {
    assert!(a.len() == n * n && b.len() == n * n);
    let mut buf = std::mem::take(&mut ws.a);
    buf.clear();
    buf.resize(2 * n * n, 0.0);
    for i in 0..n {
        let row = &mut buf[2 * i * n..2 * i * n + 2 * (i + 1)];
        for ((dst, &x), &y) in row
            .as_chunks_mut::<2>().0.iter_mut()
            .zip(&a[i * n..i * n + i + 1])
            .zip(&b[i * n..i * n + i + 1])
        {
            dst[0] = x;
            dst[1] = y;
        }
    }
    let bounds = if n > 1 {
        [
            gershgorin_bounds(n, &buf, 2, 0, &mut ws.d),
            gershgorin_bounds(n, &buf, 2, 1, &mut ws.d),
        ]
    } else {
        [SpectrumBounds::NONE; 2]
    };
    let r = solve_pair(ws, n, &mut buf, bounds);
    ws.a = buf;
    r
}

/// Extremes of the two lane-interleaved matrices in `a2` (overwritten);
/// `bounds`: bounds on the two spectra.
fn solve_pair(
    ws: &mut EigenWorkspace,
    n: usize,
    a2: &mut [f64],
    bounds: [SpectrumBounds; 2],
) -> [(f64, f64); 2] {
    match n {
        0 => return [(f64::NAN, f64::NAN); 2],
        1 => return [(a2[0], a2[0]), (a2[1], a2[1])],
        _ => {}
    }
    ws.reset_vectors(2 * n);
    pair::tridiagonalize_pair(
        n, a2, &mut ws.d, &mut ws.e, &mut ws.u, &mut ws.p, &mut ws.pu, &mut ws.pq,
    );
    pair::tridiagonal_extremes_pair(n, &ws.d, &ws.e, &mut ws.u, bounds)
}

/// Spectrum bounds of a Burden matrix from its description (O(n + entries)).
/// Gershgorin: each row sum is bounded by `|fill| * (n - 1) + sum |w|` over
/// the listed entries of that row (valid even for repeated pairs). Inner
/// bounds (Cauchy interlacing): the extreme eigenvalues of the 1 x 1 and the
/// bonded 2 x 2 principal submatrices.
fn burden_bounds(n: usize, m: &BurdenMatrix) -> SpectrumBounds {
    let mut r = vec![m.fill.abs() * (n as f64 - 1.0); n];
    let mut b = SpectrumBounds {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
        min_ub: f64::INFINITY,
        max_lb: f64::NEG_INFINITY,
    };
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

/// Gershgorin interval `(lo, hi)` of the symmetric matrix whose lower
/// triangle element `(i, j)` is `a[lanes * (i * n + j) + lane]`. It encloses
/// the spectrum and is usually much tighter than the Gershgorin interval of
/// the tridiagonal form (Burden matrices are strongly diagonally dominant),
/// which saves iterations in `tridiagonal_extremes`. Inner bounds: the
/// extreme diagonal elements.
fn gershgorin_bounds(
    n: usize,
    a: &[f64],
    lanes: usize,
    lane: usize,
    r: &mut Vec<f64>,
) -> SpectrumBounds {
    r.clear();
    r.resize(n, 0.0);
    for i in 0..n {
        let mut ri = 0.0;
        for j in 0..i {
            let v = a[lanes * (i * n + j) + lane].abs();
            ri += v;
            r[j] += v;
        }
        r[i] += ri;
    }
    let mut b = SpectrumBounds {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
        min_ub: f64::INFINITY,
        max_lb: f64::NEG_INFINITY,
    };
    for i in 0..n {
        let d = a[lanes * (i * n + i) + lane];
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

/// Extreme eigenvalues `(min, max)` of each Burden matrix (all `n x n`;
/// `(NaN, NaN)` for a matrix with a non-finite element).
/// Matrices are assembled straight into the solver's (thread-local)
/// workspace and solved two at a time in lock step
/// ([`extreme_eigenvalues_pair`]), an odd one alone ([`extreme_eigenvalues`]).
pub fn burden_extremes(n: usize, jobs: &[BurdenMatrix]) -> Vec<(f64, f64)> {
    BURDEN_WS.with(|ws| {
        let mut ws = ws.borrow_mut();
        let ws = &mut *ws;
        let mut out = Vec::with_capacity(jobs.len());
        for chunk in jobs.chunks(2) {
            let mut buf = std::mem::take(&mut ws.a);
            // Every lower-triangle entry is written below and the upper part
            // is never read, so stale contents need no clearing.
            let len = if chunk.len() == 2 { 2 * n * n } else { n * n };
            if buf.len() < len {
                buf.resize(len, 0.0);
            }
            buf.truncate(len);
            if let [m0, m1] = chunk {
                for i in 0..n {
                    let row = &mut buf[2 * i * n..2 * i * n + 2 * (i + 1)];
                    for c in row.as_chunks_mut::<2>().0 {
                        c[0] = m0.fill;
                        c[1] = m1.fill;
                    }
                }
                for (lane, m) in [m0, m1].into_iter().enumerate() {
                    for &(i, j, w) in m.entries {
                        let (i, j) = (i.max(j) as usize, i.min(j) as usize);
                        buf[2 * (i * n + j) + lane] = w;
                    }
                    for (i, &v) in m.diag[..n].iter().enumerate() {
                        buf[2 * (i * n + i) + lane] = v;
                    }
                }
                out.extend(solve_pair(
                    ws,
                    n,
                    &mut buf,
                    [burden_bounds(n, m0), burden_bounds(n, m1)],
                ));
            } else {
                let m = &chunk[0];
                for i in 0..n {
                    buf[i * n..i * n + i].fill(m.fill);
                }
                for &(i, j, w) in m.entries {
                    let (i, j) = (i.max(j) as usize, i.min(j) as usize);
                    buf[i * n + j] = w;
                }
                for (i, &v) in m.diag[..n].iter().enumerate() {
                    buf[i * n + i] = v;
                }
                out.push(solve_single(ws, n, &mut buf, burden_bounds(n, m)));
            }
            ws.a = buf;
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

/// Scalars of one row in the fused update/matrix-vector kernels:
/// `pu[j]`, `pq[j]` (pending rank-2 update) and `u[j]` (current reflector).
#[derive(Clone, Copy)]
struct RowCoef {
    puj: f64,
    pqj: f64,
    uj: f64,
}

/// Columns `0..len` of one or two rows of the lower triangle: apply the
/// pending rank-2 update `r[k] -= puj * pq[k] + pqj * pu[k]`, accumulate
/// `s = sum(r[k] * u[k])` per row and `p[k] += sum_rows(r[k] * uj)`.
/// Returns the row dot products. Portable scalar version.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fused_rows<const R: usize>(
    rows: [&mut [f64]; R],
    c: [RowCoef; R],
    len: usize,
    pu: &[f64],
    pq: &[f64],
    u: &[f64],
    p: &mut [f64],
) -> [f64; R] {
    let mut s = [0.0f64; R];
    for k in 0..len {
        let (a, b, uk) = (pu[k], pq[k], u[k]);
        let mut pk = p[k];
        for r in 0..R {
            let v = fma(-c[r].pqj, a, fma(-c[r].puj, b, rows[r][k]));
            rows[r][k] = v;
            s[r] = fma(v, uk, s[r]);
            pk = fma(v, c[r].uj, pk);
        }
        p[k] = pk;
    }
    s
}

/// NEON version of the fused kernel (two `f64` lanes, two-way unrolled).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fused_rows<const R: usize>(
    mut rows: [&mut [f64]; R],
    c: [RowCoef; R],
    len: usize,
    pu: &[f64],
    pq: &[f64],
    u: &[f64],
    p: &mut [f64],
) -> [f64; R] {
    use std::arch::aarch64::*;
    assert!(pu.len() >= len && pq.len() >= len && u.len() >= len && p.len() >= len);
    for r in rows.iter() {
        assert!(r.len() >= len);
    }
    let mut s = [0.0f64; R];
    let main = len & !3;
    // SAFETY: all pointers are in bounds (`k + 3 < main <= len` checked above).
    unsafe {
        let rp: [*mut f64; R] = std::array::from_fn(|r| rows[r].as_mut_ptr());
        let (pup, pqp, up, pp) = (pu.as_ptr(), pq.as_ptr(), u.as_ptr(), p.as_mut_ptr());
        let mut acc0 = [vdupq_n_f64(0.0); R];
        let mut acc1 = [vdupq_n_f64(0.0); R];
        let npuj: [float64x2_t; R] = std::array::from_fn(|r| vdupq_n_f64(-c[r].puj));
        let npqj: [float64x2_t; R] = std::array::from_fn(|r| vdupq_n_f64(-c[r].pqj));
        let ujv: [float64x2_t; R] = std::array::from_fn(|r| vdupq_n_f64(c[r].uj));
        let mut k = 0;
        while k < main {
            let a0 = vld1q_f64(pup.add(k));
            let a1 = vld1q_f64(pup.add(k + 2));
            let b0 = vld1q_f64(pqp.add(k));
            let b1 = vld1q_f64(pqp.add(k + 2));
            let u0 = vld1q_f64(up.add(k));
            let u1 = vld1q_f64(up.add(k + 2));
            let mut p0 = vld1q_f64(pp.add(k));
            let mut p1 = vld1q_f64(pp.add(k + 2));
            for r in 0..R {
                let x0 = vld1q_f64(rp[r].add(k));
                let x1 = vld1q_f64(rp[r].add(k + 2));
                let v0 = vfmaq_f64(vfmaq_f64(x0, npuj[r], b0), npqj[r], a0);
                let v1 = vfmaq_f64(vfmaq_f64(x1, npuj[r], b1), npqj[r], a1);
                vst1q_f64(rp[r].add(k), v0);
                vst1q_f64(rp[r].add(k + 2), v1);
                acc0[r] = vfmaq_f64(acc0[r], v0, u0);
                acc1[r] = vfmaq_f64(acc1[r], v1, u1);
                p0 = vfmaq_f64(p0, v0, ujv[r]);
                p1 = vfmaq_f64(p1, v1, ujv[r]);
            }
            vst1q_f64(pp.add(k), p0);
            vst1q_f64(pp.add(k + 2), p1);
            k += 4;
        }
        for r in 0..R {
            s[r] = vaddvq_f64(vaddq_f64(acc0[r], acc1[r]));
        }
    }
    for k in main..len {
        let (a, b, uk) = (pu[k], pq[k], u[k]);
        let mut pk = p[k];
        for r in 0..R {
            let v = fma(-c[r].pqj, a, fma(-c[r].puj, b, rows[r][k]));
            rows[r][k] = v;
            s[r] = fma(v, uk, s[r]);
            pk = fma(v, c[r].uj, pk);
        }
        p[k] = pk;
    }
    s
}

/// `row[k] -= uj * q[k] + qj * u[k]`.
#[inline(always)]
fn rank2_row(row: &mut [f64], u: &[f64], q: &[f64], uj: f64, qj: f64) {
    let len = row.len();
    for ((r, a), b) in row.iter_mut().zip(&u[..len]).zip(&q[..len]) {
        *r = fma(-qj, *a, fma(-uj, *b, *r));
    }
}

/// Householder tridiagonalization working on contiguous rows of the lower
/// triangle (EISPACK `tred1` reflectors, last row first). The rank-2 update of
/// each step is deferred and fused with the matrix-vector product of the next
/// step, so the trailing matrix is streamed once per step. On return `d` holds
/// the diagonal and `e[1..]` the sub-diagonal.
#[inline(never)]
fn tridiagonalize_lower(
    n: usize,
    a: &mut [f64],
    d: &mut [f64],
    e: &mut [f64],
    bufs: [&mut Vec<f64>; 4],
) {
    let [u, p, pu, pq] = bufs;
    // Rows 0..i of the remaining block still need `A -= pu pq^T + pq pu^T`
    // (initially zero).
    pu.fill(0.0);
    pq.fill(0.0);
    for i in (1..n).rev() {
        let l = i - 1;
        let (head, tail) = a.split_at_mut(i * n);
        let x = &tail[..i];
        let scale: f64 = if l == 0 {
            0.0
        } else {
            x.iter().map(|v| v.abs()).sum()
        };
        if scale == 0.0 {
            e[i] = x[l];
            for j in 0..i {
                let (puj, pqj) = (pu[j], pq[j]);
                rank2_row(&mut head[j * n..j * n + j + 1], pu, pq, puj, pqj);
            }
            pu[..i].fill(0.0);
            pq[..i].fill(0.0);
            continue;
        }
        let inv_scale = 1.0 / scale;
        let mut h = 0.0;
        for (uk, &xk) in u[..i].iter_mut().zip(x) {
            let v = xk * inv_scale;
            *uk = v;
            h = fma(v, v, h);
        }
        let f = u[l];
        let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
        e[i] = scale * g;
        h -= f * g;
        u[l] = f - g;

        // p = A u over the leading i x i block (symmetric, lower triangle),
        // applying the previous step's deferred update on the way.
        p[..i].fill(0.0);
        let coef = |j: usize| RowCoef {
            puj: pu[j],
            pqj: pq[j],
            uj: u[j],
        };
        let upd = |x: f64, j: usize, k: usize| fma(-pq[j], pu[k], fma(-pu[j], pq[k], x));
        let mut j = 0;
        while j + 1 < i {
            // rows j and j + 1 share columns 0..j
            let (r0, r1) = head[j * n..].split_at_mut(n);
            let (r0, r1) = (&mut r0[..j + 1], &mut r1[..j + 2]);
            let (c0, c1) = (coef(j), coef(j + 1));
            let [s0, mut s1] = {
                let (a0, _) = r0.split_at_mut(j);
                let (a1, _) = r1.split_at_mut(j);
                fused_rows([a0, a1], [c0, c1], j, pu, pq, u, p)
            };
            let d0 = upd(r0[j], j, j);
            r0[j] = d0;
            let o1 = upd(r1[j], j + 1, j);
            r1[j] = o1;
            let d1 = upd(r1[j + 1], j + 1, j + 1);
            r1[j + 1] = d1;
            s1 = fma(o1, u[j], s1);
            p[j] += fma(o1, c1.uj, fma(d0, c0.uj, s0));
            p[j + 1] += fma(d1, c1.uj, s1);
            j += 2;
        }
        if j < i {
            let r0 = &mut head[j * n..j * n + j + 1];
            let c0 = coef(j);
            let [s0] = fused_rows([&mut r0[..j]], [c0], j, pu, pq, u, p);
            let d0 = upd(r0[j], j, j);
            r0[j] = d0;
            p[j] += fma(d0, c0.uj, s0);
        }
        let inv_h = 1.0 / h;
        let mut f_acc = 0.0;
        for (pk, &uk) in p[..i].iter_mut().zip(&u[..i]) {
            *pk *= inv_h;
            f_acc = fma(*pk, uk, f_acc);
        }
        let hh = f_acc / (h + h);
        for (pk, &uk) in p[..i].iter_mut().zip(&u[..i]) {
            *pk = fma(-hh, uk, *pk);
        }
        // Row l is the next step's Householder source: update it now; rows
        // 0..l are updated lazily in the next step.
        let (ul, ql) = (u[l], p[l]);
        rank2_row(&mut head[l * n..l * n + l + 1], u, p, ul, ql);
        std::mem::swap(u, pu);
        std::mem::swap(p, pq);
    }
    for i in 0..n {
        d[i] = a[i * n + i];
    }
}

/// Smallest and largest eigenvalue of the symmetric tridiagonal matrix with
/// diagonal `d` and sub-diagonal `e[1..]`. `e2` is scratch; `bounds` is an
/// enclosure of the spectrum (e.g. Gershgorin of the original matrix).
///
/// Laguerre's method on the characteristic polynomial, started at the
/// Gershgorin bounds: for a polynomial with only real roots it converges
/// monotonically from outside the spectrum (cubically for a simple root), so
/// every iterate stays a strict bound and all LDL^T pivots keep one sign (no
/// zero-pivot hazards). If an iterate ever crosses the extreme eigenvalue
/// (rounding) or convergence is slow (multiple root), Sturm-count
/// multisection finishes from the bracket established so far.
#[inline(never)]
fn tridiagonal_extremes(
    n: usize,
    d: &[f64],
    e: &[f64],
    e2: &mut [f64],
    bounds: SpectrumBounds,
) -> (f64, f64) {
    let mut gl = f64::INFINITY;
    let mut gu = f64::NEG_INFINITY;
    let mut max_e2: f64 = 0.0;
    e2[0] = 0.0;
    for i in 0..n {
        let mut r = 0.0;
        if i > 0 {
            r += e[i].abs();
            e2[i] = e[i] * e[i];
            max_e2 = max_e2.max(e2[i]);
        }
        if i + 1 < n {
            r += e[i + 1].abs();
        }
        gl = gl.min(d[i] - r);
        gu = gu.max(d[i] + r);
    }
    let pivmin = f64::MIN_POSITIVE * max_e2.max(1.0);
    let tnorm = gl.abs().max(gu.abs());
    let eps = f64::EPSILON;
    let fudge = 2.1 * tnorm * eps * n as f64 + 4.2 * pivmin;
    // tighter enclosure from the original matrix (same spectrum up to rounding)
    gl = gl.max(bounds.lo) - fudge;
    gu = gu.min(bounds.hi) + fudge;
    let (min_ub, max_lb) = (bounds.min_ub + fudge, bounds.max_lb - fudge);

    let mut ext = [
        Extreme::new(0, n, gl, min_ub, tnorm),
        Extreme::new(1, n, max_lb, gu, tnorm),
    ];
    for _ in 0..MAX_IT {
        // Laguerre lanes: LDL^T pivots q of T - x with derivative recurrences,
        // G = p'/p and H = (p'/p)^2 - p''/p of p(x) = det(T - x).
        // Probe lanes: Sturm counts at x -/+ probe (bracket certification).
        let x = [ext[0].x, ext[1].x];
        let sig = [
            ext[0].probes[0],
            ext[0].probes[1],
            ext[1].probes[0],
            ext[1].probes[1],
        ];
        let (cnt, g, h, pc) = sweep(n, d, e2, x, sig, pivmin);
        for l in 0..2 {
            ext[l].update(
                cnt[l] as f64,
                [pc[2 * l] as f64, pc[2 * l + 1] as f64],
                g[l],
                h[l],
            );
        }
        if ext[0].done && ext[1].done {
            break;
        }
    }
    if !(ext[0].done && ext[1].done) {
        let need = [!ext[0].done, !ext[1].done];
        let ms = multisection(
            n,
            d,
            e2,
            pivmin,
            [(ext[0].lo, ext[0].hi), (ext[1].lo, ext[1].hi)],
            need,
        );
        for l in 0..2 {
            if need[l] {
                ext[l].result = ms[l];
            }
        }
    }
    (ext[0].result, ext[1].result)
}

/// One pass over the tridiagonal matrix: at the Laguerre points `x` the
/// Sturm counts and `G = p'/p`, `H = (p'/p)^2 - p''/p` of `p(x) = det(T - x)`
/// (LDL^T pivots with derivative recurrences), and the Sturm counts at the
/// four probe points `sig`.
#[inline(always)]
#[allow(clippy::type_complexity)]
fn sweep(
    n: usize,
    d: &[f64],
    e2: &[f64],
    x: [f64; 2],
    sig: [f64; 4],
    pivmin: f64,
) -> ([usize; 2], [f64; 2], [f64; 2], [usize; 4]) {
    let mut cnt = [0usize; 2];
    let mut pc = [0usize; 4];
    let mut pq = [0.0f64; 4];
    for k in 0..4 {
        let v = d[0] - sig[k];
        let v = if v.abs() < pivmin { -pivmin } else { v };
        pq[k] = v;
        pc[k] = (v < 0.0) as usize;
    }
    let mut qpp = [0.0f64; 2];
    let mut t = [0.0f64; 2];
    let mut r = [0.0f64; 2];
    let mut g = [0.0f64; 2];
    let mut h = [0.0f64; 2];
    for l in 0..2 {
        let v = d[0] - x[l];
        let v = if v.abs() < pivmin { -pivmin } else { v };
        cnt[l] = (v < 0.0) as usize;
        t[l] = 1.0 / v;
        r[l] = -t[l];
        g[l] = r[l];
        h[l] = r[l] * r[l];
    }
    for i in 1..n {
        let (di, ei2) = (d[i], e2[i]);
        for l in 0..2 {
            let et = ei2 * t[l];
            let qn = (di - x[l]) - et;
            let qn = if qn.abs() < pivmin { -pivmin } else { qn };
            let qpn = fma(et, r[l], -1.0);
            let qppn = et * fma(qpp[l], t[l], -2.0 * r[l] * r[l]);
            cnt[l] += (qn < 0.0) as usize;
            qpp[l] = qppn;
            t[l] = 1.0 / qn;
            r[l] = qpn * t[l];
            g[l] += r[l];
            h[l] = fma(r[l], r[l], fma(-qppn, t[l], h[l]));
        }
        for k in 0..4 {
            let v = (di - sig[k]) - ei2 / pq[k];
            let v = if v.abs() < pivmin { -pivmin } else { v };
            pq[k] = v;
            pc[k] += (v < 0.0) as usize;
        }
    }
    (cnt, g, h, pc)
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
        let mut m = a.to_vec();
        let single = extreme_eigenvalues(ws, n, &mut m);
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
        let mut m = a.to_vec();
        let (lo, hi) = extreme_eigenvalues(ws, n, &mut m);
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
}
