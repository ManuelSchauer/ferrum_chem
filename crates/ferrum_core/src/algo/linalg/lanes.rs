//! `2 * V` same-size symmetric eigenvalue problems solved in lock step, one
//! per SIMD lane ([`FV`]: `V` two-lane [`F2`] vectors): the dense Householder
//! reduction to tridiagonal form and the Laguerre / Sturm iteration for the
//! extreme eigenvalues of the tridiagonal matrices (the band reduction of
//! large Burden matrices is in [`band`]). Running several problems side by
//! side removes the short-row tails and horizontal reductions that dominate
//! scalar kernels for molecule-sized matrices; the per-lane arithmetic does
//! not depend on `V`.
//!
//! Layout: packed lower triangle, element `(i, j)` (`j <= i`) of problem `l`
//! at `2 * V * (tri(i) + j) + l` with `tri(i) = i * (i + 1) / 2`; vectors are
//! interleaved the same way (`v[2 * V * k + l]`).

// The lane loops index `chunks_exact` slices on purpose (tuned kernel).
#![allow(clippy::chunks_exact_to_as_chunks)]

use super::f2::F2;
use super::{Extreme, MAX_IT, SpectrumBounds, multisection};

pub(super) mod band;

/// Offset of row `i` in a packed lower triangle (in elements).
#[inline(always)]
pub(crate) fn tri(i: usize) -> usize {
    i * (i + 1) / 2
}

/// `2 * V` `f64` lanes; every operation is the per-lane IEEE operation.
#[derive(Clone, Copy)]
pub(super) struct FV<const V: usize>([F2; V]);

impl<const V: usize> FV<V> {
    #[inline(always)]
    fn splat(v: f64) -> Self {
        FV([F2::splat(v); V])
    }
    /// Lane `l` set to `f(l)`.
    #[inline(always)]
    fn from_fn(f: impl Fn(usize) -> f64) -> Self {
        FV(std::array::from_fn(|k| F2::new(f(2 * k), f(2 * k + 1))))
    }
    #[inline(always)]
    fn lane(self, l: usize) -> f64 {
        let v = self.0[l / 2];
        if l.is_multiple_of(2) { v.lane0() } else { v.lane1() }
    }
    /// # Safety
    /// `p` must be valid for reading `2 * V` `f64`.
    #[inline(always)]
    unsafe fn load_ptr(p: *const f64) -> Self {
        FV(std::array::from_fn(|k| unsafe { F2::load_ptr(p.add(2 * k)) }))
    }
    /// # Safety
    /// `p` must be valid for writing `2 * V` `f64`.
    #[inline(always)]
    unsafe fn store_ptr(self, p: *mut f64) {
        for k in 0..V {
            unsafe { self.0[k].store_ptr(p.add(2 * k)) }
        }
    }
    #[inline(always)]
    fn load(s: &[f64]) -> Self {
        assert!(s.len() >= 2 * V);
        // SAFETY: 2 * V elements in bounds (checked above)
        unsafe { Self::load_ptr(s.as_ptr()) }
    }
    #[inline(always)]
    fn store(self, s: &mut [f64]) {
        assert!(s.len() >= 2 * V);
        // SAFETY: 2 * V elements in bounds (checked above)
        unsafe { self.store_ptr(s.as_mut_ptr()) }
    }
    #[inline(always)]
    fn map(self, f: impl Fn(F2) -> F2) -> Self {
        FV(std::array::from_fn(|k| f(self.0[k])))
    }
    #[inline(always)]
    fn zip(self, o: Self, f: impl Fn(F2, F2) -> F2) -> Self {
        FV(std::array::from_fn(|k| f(self.0[k], o.0[k])))
    }
    #[inline(always)]
    fn zip3(self, b: Self, c: Self, f: impl Fn(F2, F2, F2) -> F2) -> Self {
        FV(std::array::from_fn(|k| f(self.0[k], b.0[k], c.0[k])))
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        self.zip(o, F2::add)
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        self.zip(o, F2::sub)
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        self.zip(o, F2::mul)
    }
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        self.zip(o, F2::div)
    }
    /// `self * b + c` (fused).
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        self.zip3(b, c, F2::mul_add)
    }
    /// `c - self * b` (fused).
    #[inline(always)]
    fn neg_mul_add(self, b: Self, c: Self) -> Self {
        self.zip3(b, c, F2::neg_mul_add)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        self.map(F2::abs)
    }
    #[inline(always)]
    fn neg(self) -> Self {
        self.map(F2::neg)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        self.map(F2::sqrt)
    }
    /// Per lane `if self == 0 { a } else { b }`.
    #[inline(always)]
    fn select_eq_zero(self, a: Self, b: Self) -> Self {
        self.zip3(a, b, F2::select_eq_zero)
    }
    /// Per lane `if self >= 0 { a } else { b }`.
    #[inline(always)]
    fn select_ge_zero(self, a: Self, b: Self) -> Self {
        self.zip3(a, b, F2::select_ge_zero)
    }
    /// Per lane `if |self| < lim { r } else { self }`.
    #[inline(always)]
    fn replace_small(self, lim: Self, r: Self) -> Self {
        self.zip3(lim, r, F2::replace_small)
    }
    /// Per lane `1.0` if `self < 0` else `0.0`.
    #[inline(always)]
    fn neg_indicator(self) -> Self {
        self.map(F2::neg_indicator)
    }
}

/// Packs `2 * V` Burden matrices into `a` (lane `m` holds `jobs[m]`, or the
/// last job for the lanes beyond `jobs.len()`).
pub(super) fn assemble_burden<const V: usize>(n: usize, jobs: &[super::BurdenMatrix], a: &mut [f64]) {
    let w = 2 * V;
    let job = |m: usize| &jobs[m.min(jobs.len() - 1)];
    let fill = FV::<V>::from_fn(|m| job(m).fill);
    for c in a[..w * tri(n)].chunks_exact_mut(w) {
        fill.store(c);
    }
    for m in 0..w {
        let jb = job(m);
        for &(i, j, x) in jb.entries {
            let (i, j) = (i.max(j) as usize, i.min(j) as usize);
            a[w * (tri(i) + j) + m] = x;
        }
        for (i, &v) in jb.diag[..n].iter().enumerate() {
            a[w * (tri(i) + i) + m] = v;
        }
    }
}

/// Packs the lower triangles of `2 * V` dense row-major `n x n` matrices into
/// `a` (lane `m` holds `mats[m]`, or the last matrix beyond `mats.len()`).
pub(super) fn assemble_dense<const V: usize>(n: usize, mats: &[&[f64]], a: &mut [f64]) {
    let w = 2 * V;
    for m in 0..w {
        let src = mats[m.min(mats.len() - 1)];
        for i in 0..n {
            for j in 0..=i {
                a[w * (tri(i) + j) + m] = src[i * n + j];
            }
        }
    }
}

/// Fused deferred-update + matrix-vector kernel for the `R` consecutive rows
/// `j .. j + R` of the lower triangle: applies `A -= pu pq^T + pq pu^T` to
/// them, accumulates `p += A u` (symmetric: row and column contributions) and
/// `uau += u_r * (2 * s_r + a_rr * u_r)` for each row `r`, `s_r` being the
/// row part `sum_{k < r} a_rk u_k` (summed over all rows: `u^T A u`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fused_rows<const V: usize, const R: usize>(
    head: &mut [f64],
    j: usize,
    u: &[f64],
    p: &mut [f64],
    pu: &[f64],
    pq: &[f64],
    uau: &mut FV<V>,
) {
    let w = 2 * V;
    let last = j + R - 1;
    assert!(w * (tri(last) + last + 1) <= head.len());
    let need = w * (j + R);
    assert!(u.len() >= need && p.len() >= need && pu.len() >= need && pq.len() >= need);
    let zero = FV::<V>::splat(0.0);
    // SAFETY: rows j..j+R hold columns 0..=row (checked: the last element of
    // the last row is in bounds); vector indices are < need (checked above).
    unsafe {
        let base = head.as_mut_ptr();
        let rp: [*mut f64; R] = std::array::from_fn(|r| base.add(w * tri(j + r)));
        let (up, pp, pup, pqp) = (u.as_ptr(), p.as_mut_ptr(), pu.as_ptr(), pq.as_ptr());
        let puj: [FV<V>; R] = std::array::from_fn(|r| FV::load_ptr(pup.add(w * (j + r))));
        let pqj: [FV<V>; R] = std::array::from_fn(|r| FV::load_ptr(pqp.add(w * (j + r))));
        let uj: [FV<V>; R] = std::array::from_fn(|r| FV::load_ptr(up.add(w * (j + r))));
        // columns shared by all R rows (two columns per iteration with
        // separate row sums, which keeps the FMA latency chains short)
        let col = |k: usize, s: &mut [FV<V>; R]| {
            let o = w * k;
            let a = FV::load_ptr(pup.add(o));
            let b = FV::load_ptr(pqp.add(o));
            let uu = FV::load_ptr(up.add(o));
            let mut pk = FV::load_ptr(pp.add(o));
            for r in 0..R {
                let v = pqj[r].neg_mul_add(a, puj[r].neg_mul_add(b, FV::load_ptr(rp[r].add(o))));
                v.store_ptr(rp[r].add(o));
                s[r] = v.mul_add(uu, s[r]);
                pk = v.mul_add(uj[r], pk);
            }
            pk.store_ptr(pp.add(o));
        };
        let mut s = [zero; R];
        let mut s2 = [zero; R];
        let mut k = 0;
        while k + 2 <= j {
            col(k, &mut s);
            col(k + 1, &mut s2);
            k += 2;
        }
        if k < j {
            col(k, &mut s);
        }
        for r in 0..R {
            s[r] = s[r].add(s2[r]);
        }
        // the R x R corner (off-diagonal part), then the diagonal
        for r in 1..R {
            for c in 0..r {
                let o = w * (j + c);
                let v = pqj[r].neg_mul_add(puj[c], puj[r].neg_mul_add(pqj[c], FV::load_ptr(rp[r].add(o))));
                v.store_ptr(rp[r].add(o));
                s[r] = v.mul_add(uj[c], s[r]);
                v.mul_add(uj[r], FV::load_ptr(pp.add(o))).store_ptr(pp.add(o));
            }
        }
        for r in 0..R {
            let o = w * (j + r);
            let v = pqj[r].neg_mul_add(puj[r], puj[r].neg_mul_add(pqj[r], FV::load_ptr(rp[r].add(o))));
            v.store_ptr(rp[r].add(o));
            let t = v.mul_add(uj[r], s[r]);
            FV::load_ptr(pp.add(o)).add(t).store_ptr(pp.add(o));
            *uau = uj[r].mul_add(t.add(s[r]), *uau);
        }
    }
}

/// Sum of four partial sums (the reductions use four accumulators to stay
/// off the add latency chain).
#[inline(always)]
fn sum4<const V: usize>(acc: [FV<V>; 4]) -> FV<V> {
    acc[0].add(acc[2]).add(acc[1].add(acc[3]))
}

/// Squared norms `sum x_k^2` of `2 * V` interleaved vectors.
#[inline(always)]
fn sum_squares<const V: usize>(x: &[f64]) -> FV<V> {
    let mut acc = [FV::<V>::splat(0.0); 4];
    for (k, c) in x.chunks_exact(2 * V).enumerate() {
        let v = FV::load(c);
        acc[k & 3] = v.mul_add(v, acc[k & 3]);
    }
    sum4(acc)
}

/// Whether every lane of a squared row norm `h` is zero or far from over-
/// and underflow (no scaling needed).
#[inline(always)]
fn norm_in_safe_range<const V: usize>(h: FV<V>) -> bool {
    const SMALL: f64 = 1.0e-150;
    const BIG: f64 = 1.0e150;
    (0..2 * V).all(|m| {
        let v = h.lane(m);
        v == 0.0 || (SMALL..=BIG).contains(&v)
    })
}

/// Householder tridiagonalization (EISPACK `tred1` reflectors, last row
/// first) of `2 * V` lane-interleaved packed lower triangles. The rank-2
/// update of each step is deferred and fused with the matrix-vector product
/// of the next step, so the trailing matrix is streamed once per step; the
/// Householder vector of step `i` overwrites row `i` (no longer needed), and
/// `u^T A u` and the next row's norm come out of the fused passes. On return
/// `d` / `e[2 * V..]` hold the interleaved diagonals / sub-diagonals. `zeros`,
/// `p` and `pq` are scratch vectors of length `>= 2 * V * n`.
#[inline(never)]
pub(super) fn tridiagonalize<const V: usize>(
    n: usize,
    a: &mut [f64],
    d: &mut [f64],
    e: &mut [f64],
    zeros: &mut [f64],
    p: &mut Vec<f64>,
    pq: &mut Vec<f64>,
) {
    tridiagonalize_impl::<V>(n, a, d, e, zeros, p, pq, None);
}

/// [`tridiagonalize`] that also keeps the reflectors for
/// [`back_transform`]: `P_i = I - u_i u_i^T / h_i` (`i = 2 .. n`), `u_i` is
/// left in row `i` of `a` (columns `0..i`) and `h_i` in `h[2 * V * i..]`
/// (`T = P_2 ... P_{n-1} A P_{n-1} ... P_2`). The reduction itself is the
/// same.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tridiagonalize_keep<const V: usize>(
    n: usize,
    a: &mut [f64],
    d: &mut [f64],
    e: &mut [f64],
    h: &mut [f64],
    zeros: &mut [f64],
    p: &mut Vec<f64>,
    pq: &mut Vec<f64>,
) {
    assert!(h.len() >= 2 * V * n);
    tridiagonalize_impl::<V>(n, a, d, e, zeros, p, pq, Some(h));
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tridiagonalize_impl<const V: usize>(
    n: usize,
    a: &mut [f64],
    d: &mut [f64],
    e: &mut [f64],
    zeros: &mut [f64],
    p: &mut Vec<f64>,
    pq: &mut Vec<f64>,
    mut keep: Option<&mut [f64]>,
) {
    let w = 2 * V;
    assert!(a.len() >= w * tri(n) && d.len() >= w * n && e.len() >= w * n);
    assert!(zeros.len() >= w * n && p.len() >= w * n && pq.len() >= w * n);
    let zero = FV::<V>::splat(0.0);
    let one = FV::<V>::splat(1.0);
    zeros[..w * n].fill(0.0);
    pq[..w * n].fill(0.0);
    // squared norm of the next Householder source (row i, columns 0..i)
    let mut h_next = if n > 1 { sum_squares::<V>(&a[w * tri(n - 1)..w * (tri(n - 1) + n - 1)]) } else { zero };
    for i in (1..n).rev() {
        let l = i - 1;
        let (head, tail) = a.split_at_mut(w * tri(i));
        if l == 0 {
            FV::<V>::load(&tail[0..]).store(&mut e[w * i..]);
            // pending update of the remaining 1 x 1 block
            let pu: &[f64] = if i + 1 < n { &tail[w * (i + 1)..] } else { &zeros[..] };
            let (a0, b0) = (FV::<V>::load(&pu[0..]), FV::<V>::load(&pq[0..]));
            let v = b0.neg_mul_add(a0, a0.neg_mul_add(b0, FV::load(&head[0..])));
            v.store(&mut head[0..]);
            continue;
        }
        // Reflector from the unscaled row unless a lane's norm is near over-
        // or underflow; then EISPACK's 1-norm scaling (same reflector).
        let mut h = h_next;
        let mut scale = one;
        if !norm_in_safe_range(h) {
            let x = &mut tail[..w * i];
            let mut acc = [zero; 4];
            for (k, c) in x.chunks_exact(w).enumerate() {
                acc[k & 3] = acc[k & 3].add(FV::load(c).abs());
            }
            scale = sum4(acc);
            let inv_scale = scale.select_eq_zero(zero, one.div(scale));
            for c in x.chunks_exact_mut(w) {
                FV::load(c).mul(inv_scale).store(c);
            }
            h = sum_squares::<V>(x);
        }
        let f = FV::<V>::load(&tail[w * l..]);
        let sh = h.sqrt();
        let g = f.select_ge_zero(sh.neg(), sh);
        scale.mul(g).store(&mut e[w * i..]);
        let h = f.neg_mul_add(g, h);
        // A lane whose row is entirely zero needs no reflector: u = 0 (here
        // h = 0) makes the step a no-op for that lane (e = 0 = x[l]).
        let h = h.select_eq_zero(one, h);
        if let Some(keep) = keep.as_deref_mut() {
            h.store(&mut keep[w * i..]);
        }
        f.sub(g).store(&mut tail[w * l..]);
        let tail: &[f64] = tail;
        let u = &tail[..w * i];
        let pu: &[f64] = if i + 1 < n { &tail[w * (i + 1)..w * (2 * i + 1)] } else { &zeros[..w * i] };

        p[..w * i].fill(0.0);
        let mut uau = zero;
        // rows in blocks of four (two lanes) or two (four lanes): as many as
        // the registers hold
        let mut j = 0;
        if V == 1 {
            while j + 4 <= i {
                fused_rows::<V, 4>(head, j, u, p, pu, pq, &mut uau);
                j += 4;
            }
        } else {
            while j + 2 <= i {
                fused_rows::<V, 2>(head, j, u, p, pu, pq, &mut uau);
                j += 2;
            }
        }
        match i - j {
            3 => fused_rows::<V, 3>(head, j, u, p, pu, pq, &mut uau),
            2 => fused_rows::<V, 2>(head, j, u, p, pu, pq, &mut uau),
            1 => fused_rows::<V, 1>(head, j, u, p, pu, pq, &mut uau),
            _ => {}
        }
        // p = A u / h, hh = u^T p / (2 h), q = p - hh u
        let inv_h = one.div(h);
        let hh = uau.mul(inv_h).div(h.add(h));
        let ul = FV::<V>::load(&u[w * l..]);
        let ql = hh.neg_mul_add(ul, FV::load(&p[w * l..]).mul(inv_h));
        // q (stored over p), row l (the next Householder source) updated
        // eagerly, and its squared norm (without the diagonal element)
        let row = &mut head[w * tri(l)..w * (tri(l) + i)];
        let mut acc = [zero; 4];
        for (k, ((pc, uc), rc)) in p[..w * i]
            .chunks_exact_mut(w)
            .zip(u.chunks_exact(w))
            .zip(row.chunks_exact_mut(w))
            .enumerate()
        {
            let uk = FV::load(uc);
            let qk = hh.neg_mul_add(uk, FV::load(pc).mul(inv_h));
            qk.store(pc);
            let r = ql.neg_mul_add(uk, ul.neg_mul_add(qk, FV::load(rc)));
            r.store(rc);
            if k < l {
                acc[k & 3] = r.mul_add(r, acc[k & 3]);
            }
        }
        h_next = sum4(acc);
        std::mem::swap(p, pq);
    }
    for i in 0..n {
        FV::<V>::load(&a[w * (tri(i) + i)..]).store(&mut d[w * i..]);
    }
}

/// `x <- P_{n-1} ... P_2 x` for `2 * V` interleaved vectors `x` with the
/// reflectors kept by [`tridiagonalize_keep`] (`a`, `h`): maps eigenvectors
/// of the tridiagonal matrices to eigenvectors of the original ones.
#[inline(never)]
pub(crate) fn back_transform<const V: usize>(n: usize, a: &[f64], h: &[f64], x: &mut [f64]) {
    let w = 2 * V;
    assert!(a.len() >= w * tri(n) && h.len() >= w * n && x.len() >= w * n);
    let zero = FV::<V>::splat(0.0);
    for i in 2..n {
        let u = &a[w * tri(i)..w * (tri(i) + i)];
        let xs = &mut x[..w * i];
        let mut acc = [zero; 4];
        for (k, (uc, xc)) in u.chunks_exact(w).zip(xs.chunks_exact(w)).enumerate() {
            acc[k & 3] = FV::load(uc).mul_add(FV::load(xc), acc[k & 3]);
        }
        let f = sum4(acc).div(FV::load(&h[w * i..]));
        for (uc, xc) in u.chunks_exact(w).zip(xs.chunks_exact_mut(w)) {
            f.neg_mul_add(FV::load(uc), FV::load(xc)).store(xc);
        }
    }
}

/// Eigenvectors of `2 * V` interleaved symmetric tridiagonal matrices
/// (diagonal `d`, sub-diagonal `e[2 * V * i]` coupling `i - 1` and `i`) for
/// the eigenvalues `lambda` (one per lane) by inverse iteration, unit length,
/// interleaved into `x`: LAPACK `dstein` (LU of `T - lambda I` with partial
/// pivoting as in `dgttrf`, here by per-lane selects; tiny pivots replaced;
/// three solves from a pseudo-random start, each followed by a rescaling).
/// `s` is scratch of length `>= 5 * 2 * V * n`.
#[inline(never)]
pub(crate) fn inverse_iteration<const V: usize>(
    n: usize,
    d: &[f64],
    e: &[f64],
    lambda: &[f64],
    x: &mut [f64],
    s: &mut [f64],
) {
    let w = 2 * V;
    assert!(d.len() >= w * n && e.len() >= w * n && lambda.len() >= w && x.len() >= w * n && s.len() >= 5 * w * n);
    let one = FV::<V>::splat(1.0);
    let zero = FV::<V>::splat(0.0);
    if n == 1 {
        one.store(x);
        return;
    }
    let (dd, rest) = s.split_at_mut(w * n);
    let (dl, rest) = rest.split_at_mut(w * n);
    let (du, rest) = rest.split_at_mut(w * n);
    let (du2, rest) = rest.split_at_mut(w * n);
    let pv = &mut rest[..w * n];
    let at = |i: usize| w * i..w * (i + 1);
    let lam = FV::<V>::load(lambda);
    // norm (max |d_i| + |e_i|, as in the scalar code) and the LU input
    let mut norm = FV::<V>::splat(f64::MIN_POSITIVE);
    for i in 0..n {
        let di = FV::<V>::load(&d[w * i..]);
        let ei = if i + 1 < n { FV::load(&e[w * (i + 1)..]) } else { zero };
        let r = di.abs().add(ei.abs());
        norm = r.sub(norm).select_ge_zero(r, norm);
        di.sub(lam).store(&mut dd[at(i)]);
        ei.store(&mut dl[at(i)]);
        ei.store(&mut du[at(i)]);
        zero.store(&mut du2[at(i)]);
    }
    // dgttrf; pv[i] >= 0: no row interchange at step i
    for i in 0..n - 1 {
        let (a, b) = (FV::<V>::load(&dd[at(i)]), FV::load(&dl[at(i)]));
        let (c, dn) = (FV::<V>::load(&du[at(i)]), FV::load(&dd[at(i + 1)]));
        let t = a.abs().sub(b.abs());
        let num = t.select_ge_zero(b, a);
        let den = t.select_ge_zero(a, b);
        let fact = den.select_eq_zero(zero, num.div(den));
        t.select_ge_zero(a, b).store(&mut dd[at(i)]);
        fact.store(&mut dl[at(i)]);
        t.select_ge_zero(c, dn).store(&mut du[at(i)]);
        t.select_ge_zero(fact.neg_mul_add(c, dn), fact.neg_mul_add(dn, c)).store(&mut dd[at(i + 1)]);
        if i + 2 < n {
            let c2 = FV::<V>::load(&du[at(i + 1)]);
            t.select_ge_zero(zero, c2).store(&mut du2[at(i)]);
            t.select_ge_zero(c2, fact.neg().mul(c2)).store(&mut du[at(i + 1)]);
        }
        t.store(&mut pv[at(i)]);
    }
    // a singular U (lambda exact): perturb its zero pivots; then reciprocals
    let tiny = norm.mul(FV::splat(f64::EPSILON));
    for i in 0..n {
        let v = FV::<V>::load(&dd[at(i)]);
        let v = v.replace_small(tiny, v.select_ge_zero(tiny, tiny.neg()));
        one.div(v).store(&mut dd[at(i)]);
    }
    // deterministic pseudo-random start in (-1, 1)
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in 0..n {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        FV::<V>::splat(((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0).store(&mut x[at(i)]);
    }
    for _ in 0..3 {
        // dgtts2: L, then U
        for i in 0..n - 1 {
            let (a, b) = (FV::<V>::load(&x[at(i)]), FV::load(&x[at(i + 1)]));
            let (t, l) = (FV::<V>::load(&pv[at(i)]), FV::load(&dl[at(i)]));
            t.select_ge_zero(a, b).store(&mut x[at(i)]);
            t.select_ge_zero(l.neg_mul_add(a, b), l.neg_mul_add(b, a)).store(&mut x[at(i + 1)]);
        }
        let last = FV::<V>::load(&x[at(n - 1)]).mul(FV::load(&dd[at(n - 1)]));
        last.store(&mut x[at(n - 1)]);
        let mut v1 = last;
        let mut v2 = FV::<V>::load(&du[at(n - 2)]).neg_mul_add(last, FV::load(&x[at(n - 2)]));
        v2 = v2.mul(FV::load(&dd[at(n - 2)]));
        v2.store(&mut x[at(n - 2)]);
        for i in (0..n - 2).rev() {
            let r = FV::<V>::load(&du[at(i)]).neg_mul_add(v2, FV::load(&x[at(i)]));
            let r = FV::<V>::load(&du2[at(i)]).neg_mul_add(v1, r).mul(FV::load(&dd[at(i)]));
            r.store(&mut x[at(i)]);
            (v1, v2) = (v2, r);
        }
        let mut m = zero;
        for i in 0..n {
            let a = FV::<V>::load(&x[at(i)]).abs();
            m = a.sub(m).select_ge_zero(a, m);
        }
        let inv = one.div(m);
        for c in x[..w * n].chunks_exact_mut(w) {
            FV::load(c).mul(inv).store(c);
        }
    }
    let inv = one.div(sum_squares::<V>(&x[..w * n]).sqrt());
    for c in x[..w * n].chunks_exact_mut(w) {
        FV::load(c).mul(inv).store(c);
    }
}

/// One pass over `2 * V` interleaved tridiagonal matrices: at the `NL`
/// Laguerre points `xs` the Sturm counts and `G = p'/p`,
/// `H = (p'/p)^2 - p''/p` of `p(x) = det(T - x)` (LDL^T pivots with
/// derivative recurrences), and the Sturm counts at the `NP` probe points
/// `sig`.
#[inline(always)]
#[allow(clippy::type_complexity)]
fn sweep<const V: usize, const NL: usize, const NP: usize>(
    n: usize,
    d: &[f64],
    e2: &[f64],
    xs: [FV<V>; NL],
    sig: [FV<V>; NP],
    piv: FV<V>,
) -> ([FV<V>; NL], [FV<V>; NL], [FV<V>; NL], [FV<V>; NP]) {
    let w = 2 * V;
    let one = FV::<V>::splat(1.0);
    let zero = FV::<V>::splat(0.0);
    let npiv = piv.neg();
    let d0 = FV::<V>::load(&d[0..]);
    let mut pq = [zero; NP];
    let mut pc = [zero; NP];
    for k in 0..NP {
        let v = d0.sub(sig[k]).replace_small(piv, npiv);
        pq[k] = v;
        pc[k] = v.neg_indicator();
    }
    let mut cnt = [zero; NL];
    let mut qpp = [zero; NL];
    let mut t = [zero; NL];
    let mut r = [zero; NL];
    let mut g = [zero; NL];
    let mut h = [zero; NL];
    for s in 0..NL {
        let v = d0.sub(xs[s]).replace_small(piv, npiv);
        cnt[s] = v.neg_indicator();
        t[s] = one.div(v);
        r[s] = t[s].neg();
        g[s] = r[s];
        h[s] = r[s].mul(r[s]);
    }
    for i in 1..n {
        let di = FV::<V>::load(&d[w * i..]);
        let ei2 = FV::<V>::load(&e2[w * i..]);
        for s in 0..NL {
            let et = ei2.mul(t[s]);
            let qn = di.sub(xs[s]).sub(et).replace_small(piv, npiv);
            let qpn = et.mul_add(r[s], one.neg());
            let rr2 = r[s].mul(r[s]).mul(FV::splat(-2.0));
            let qppn = et.mul(qpp[s].mul_add(t[s], rr2));
            cnt[s] = cnt[s].add(qn.neg_indicator());
            qpp[s] = qppn;
            t[s] = one.div(qn);
            r[s] = qpn.mul(t[s]);
            g[s] = g[s].add(r[s]);
            h[s] = r[s].mul_add(r[s], qppn.neg_mul_add(t[s], h[s]));
        }
        for k in 0..NP {
            let v = di.sub(sig[k]).sub(ei2.div(pq[k])).replace_small(piv, npiv);
            pq[k] = v;
            pc[k] = pc[k].add(v.neg_indicator());
        }
    }
    (cnt, g, h, pc)
}

/// Most lanes handled by [`tridiagonal_extremes`].
pub(super) const MAX_LANES: usize = 4;

/// `(min, max)` eigenvalue of `2 * V` interleaved symmetric tridiagonal
/// matrices (`d`, sub-diagonal `e[2 * V..]`) into `out[..2 * V]`; `bounds`:
/// enclosures of the spectra (e.g. Gershgorin of the original matrices), `e2`
/// is scratch.
///
/// Laguerre's method on the characteristic polynomial, started at the outer
/// bounds: for a polynomial with only real roots it converges monotonically
/// from outside the spectrum (cubically for a simple root), so every iterate
/// stays a strict bound and all LDL^T pivots keep one sign (no zero-pivot
/// hazards). Each sweep also takes Sturm counts at two probe points per
/// extreme ([`Extreme`]); if an iterate ever crosses the extreme eigenvalue
/// (rounding) or convergence is slow (multiple root), Sturm-count
/// multisection finishes from the bracket established so far.
#[inline(never)]
pub(super) fn tridiagonal_extremes<const V: usize>(
    n: usize,
    d: &[f64],
    e: &[f64],
    e2: &mut [f64],
    bounds: &[SpectrumBounds],
    out: &mut [(f64, f64)],
) {
    let w = 2 * V;
    assert!(w <= MAX_LANES && bounds.len() >= w && out.len() >= w);
    let eps = f64::EPSILON;
    let mut gl = [f64::INFINITY; MAX_LANES];
    let mut gu = [f64::NEG_INFINITY; MAX_LANES];
    let mut max_e2 = [0.0f64; MAX_LANES];
    e2[..w].fill(0.0);
    for i in 0..n {
        for m in 0..w {
            let mut r = 0.0;
            if i > 0 {
                let v = e[w * i + m];
                r += v.abs();
                e2[w * i + m] = v * v;
                max_e2[m] = max_e2[m].max(v * v);
            }
            if i + 1 < n {
                r += e[w * (i + 1) + m].abs();
            }
            gl[m] = gl[m].min(d[w * i + m] - r);
            gu[m] = gu[m].max(d[w * i + m] + r);
        }
    }
    let mut pivmin = [0.0f64; MAX_LANES];
    let mut tnorm = [0.0f64; MAX_LANES];
    let mut inner = [(0.0f64, 0.0f64); MAX_LANES];
    for m in 0..w {
        pivmin[m] = f64::MIN_POSITIVE * max_e2[m].max(1.0);
        tnorm[m] = gl[m].abs().max(gu[m].abs());
        let fudge = 2.1 * tnorm[m] * eps * n as f64 + 4.2 * pivmin[m];
        gl[m] = gl[m].max(bounds[m].lo) - fudge;
        gu[m] = gu[m].min(bounds[m].hi) + fudge;
        inner[m] = (bounds[m].min_ub + fudge, bounds[m].max_lb - fudge);
    }

    let piv = FV::<V>::from_fn(|m| pivmin[m]);
    // [matrix][side]: side 0 = minimum, 1 = maximum
    let mut ext: [[Extreme; 2]; MAX_LANES] = std::array::from_fn(|m| {
        let m = m.min(w - 1);
        [
            Extreme::new(0, n, gl[m], inner[m].0, tnorm[m]),
            Extreme::new(1, n, inner[m].1, gu[m], tnorm[m]),
        ]
    });
    for _ in 0..MAX_IT {
        // chains: Laguerre at x[.][side]; Sturm counts at x -/+ probe
        let xs = [0, 1].map(|s| FV::<V>::from_fn(|m| ext[m][s].x));
        let sig = [(0, 0), (0, 1), (1, 0), (1, 1)].map(|(s, k)| FV::<V>::from_fn(|m| ext[m][s].probes[k]));
        let active = [0, 1].map(|s| ext[..w].iter().any(|e| !e[s].done));
        let (cnt, g, h, pc) = if active[0] && active[1] {
            sweep::<V, 2, 4>(n, d, e2, xs, sig, piv)
        } else {
            // one side has converged in every lane: sweep the other only
            // (the values reported for the converged side are not used)
            let s = usize::from(!active[0]);
            let (c, g, h, p) = sweep::<V, 1, 2>(n, d, e2, [xs[s]], [sig[2 * s], sig[2 * s + 1]], piv);
            ([c[0]; 2], [g[0]; 2], [h[0]; 2], [p[0], p[1], p[0], p[1]])
        };
        for (m, sides) in ext[..w].iter_mut().enumerate() {
            for (s, e) in sides.iter_mut().enumerate() {
                let probes = [pc[2 * s].lane(m), pc[2 * s + 1].lane(m)];
                e.update(cnt[s].lane(m), probes, g[s].lane(m), h[s].lane(m));
            }
        }
        if ext[..w].iter().all(|e| e[0].done && e[1].done) {
            break;
        }
    }
    for m in 0..w {
        let [e0, e1] = ext[m];
        let mut res = [e0.result, e1.result];
        if !(e0.done && e1.done) {
            let dm: Vec<f64> = (0..n).map(|i| d[w * i + m]).collect();
            let em: Vec<f64> = (0..n).map(|i| e2[w * i + m]).collect();
            let need = [!e0.done, !e1.done];
            let ms = multisection(n, &dm, &em, pivmin[m], [(e0.lo, e0.hi), (e1.lo, e1.hi)], need);
            for s in 0..2 {
                if need[s] {
                    res[s] = ms[s];
                }
            }
        }
        out[m] = (res[0], res[1]);
    }
}
