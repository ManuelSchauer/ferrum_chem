//! Two same-size symmetric eigenvalue problems solved in lock step, one per
//! SIMD lane ([`F2`]). The arithmetic per lane is exactly that of the scalar
//! path in `linalg.rs` (same reflectors, same Laguerre / Sturm iteration);
//! running two problems side by side removes the short-row tails and
//! horizontal reductions that dominate the scalar kernels for molecule-sized
//! matrices.
//!
//! Layout: element `(i, j)` of problem `l` lives at `2 * (i * n + j) + l`;
//! vectors are interleaved the same way (`v[2 * k + l]`).

// The lane loops index `chunks_exact(2)` slices on purpose (tuned kernel).
#![allow(clippy::chunks_exact_to_as_chunks)]

use super::f2::F2;
use super::{Extreme, MAX_IT, SpectrumBounds, multisection};

/// Fused deferred-update + matrix-vector kernel for the `R` consecutive rows
/// `j .. j + R` of the lower triangle: applies `A -= pu pq^T + pq pu^T` to
/// them and accumulates `p += A u` (symmetric: row and column contributions).
#[inline(always)]
fn fused_rows<const R: usize>(
    head: &mut [f64],
    n: usize,
    j: usize,
    u: &[f64],
    p: &mut [f64],
    pu: &[f64],
    pq: &[f64],
) {
    let last = j + R - 1;
    assert!(2 * (last * n + last + 1) <= head.len());
    let need = 2 * (j + R);
    assert!(u.len() >= need && p.len() >= need && pu.len() >= need && pq.len() >= need);
    let zero = F2::splat(0.0);
    // SAFETY: rows j..j+R hold columns 0..=row (checked: the last element of
    // the last row is in bounds and rows are n >= j + R apart); vector
    // indices are < need (checked above).
    unsafe {
        let base = head.as_mut_ptr();
        let rp: [*mut f64; R] = std::array::from_fn(|r| base.add(2 * (j + r) * n));
        let (up, pp, pup, pqp) = (u.as_ptr(), p.as_mut_ptr(), pu.as_ptr(), pq.as_ptr());
        let puj: [F2; R] = std::array::from_fn(|r| F2::load_ptr(pup.add(2 * (j + r))));
        let pqj: [F2; R] = std::array::from_fn(|r| F2::load_ptr(pqp.add(2 * (j + r))));
        let uj: [F2; R] = std::array::from_fn(|r| F2::load_ptr(up.add(2 * (j + r))));
        let mut s = [zero; R];
        // columns shared by all R rows
        for k in 0..j {
            let o = 2 * k;
            let a = F2::load_ptr(pup.add(o));
            let b = F2::load_ptr(pqp.add(o));
            let uu = F2::load_ptr(up.add(o));
            let mut pk = F2::load_ptr(pp.add(o));
            for r in 0..R {
                let v = pqj[r].neg_mul_add(a, puj[r].neg_mul_add(b, F2::load_ptr(rp[r].add(o))));
                v.store_ptr(rp[r].add(o));
                s[r] = v.mul_add(uu, s[r]);
                pk = v.mul_add(uj[r], pk);
            }
            pk.store_ptr(pp.add(o));
        }
        // the R x R corner (off-diagonal part), then the diagonal
        for r in 1..R {
            for c in 0..r {
                let o = 2 * (j + c);
                let v = pqj[r].neg_mul_add(
                    puj[c],
                    puj[r].neg_mul_add(pqj[c], F2::load_ptr(rp[r].add(o))),
                );
                v.store_ptr(rp[r].add(o));
                s[r] = v.mul_add(uj[c], s[r]);
                v.mul_add(uj[r], F2::load_ptr(pp.add(o)))
                    .store_ptr(pp.add(o));
            }
        }
        for r in 0..R {
            let o = 2 * (j + r);
            let v = pqj[r].neg_mul_add(
                puj[r],
                puj[r].neg_mul_add(pqj[r], F2::load_ptr(rp[r].add(o))),
            );
            v.store_ptr(rp[r].add(o));
            F2::load_ptr(pp.add(o))
                .add(v.mul_add(uj[r], s[r]))
                .store_ptr(pp.add(o));
        }
    }
}

/// Householder tridiagonalization of two interleaved matrices (lower
/// triangle; see `tridiagonalize_lower` for the scalar original). On return
/// `d` / `e[2..]` hold the interleaved diagonals / sub-diagonals.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tridiagonalize_pair(
    n: usize,
    a: &mut [f64],
    d: &mut [f64],
    e: &mut [f64],
    u: &mut Vec<f64>,
    p: &mut Vec<f64>,
    pu: &mut Vec<f64>,
    pq: &mut Vec<f64>,
) {
    assert!(a.len() >= 2 * n * n && d.len() >= 2 * n && e.len() >= 2 * n);
    let zero = F2::splat(0.0);
    let one = F2::splat(1.0);
    pu[..2 * n].fill(0.0);
    pq[..2 * n].fill(0.0);
    for i in (1..n).rev() {
        let l = i - 1;
        let (head, tail) = a.split_at_mut(2 * i * n);
        let x = &tail[..2 * i];
        if l == 0 {
            F2::load(&x[0..]).store(&mut e[2 * i..]);
            // pending update of the remaining 1 x 1 block
            let (a0, b0) = (F2::load(&pu[0..]), F2::load(&pq[0..]));
            let v = b0.neg_mul_add(a0, a0.neg_mul_add(b0, F2::load(&head[0..])));
            v.store(&mut head[0..]);
            continue;
        }
        // (reductions use four partial sums to stay off the add latency chain)
        let mut acc = [zero; 4];
        for (k, c) in x.as_chunks::<2>().0.iter().enumerate() {
            acc[k & 3] = acc[k & 3].add(F2::load(c).abs());
        }
        let scale = acc[0].add(acc[2]).add(acc[1].add(acc[3]));
        // A lane whose row is entirely zero needs no reflector: u = 0 makes
        // the step a no-op for that lane (e = 0 = x[l]).
        let inv_scale = scale.select_eq_zero(zero, one.div(scale));
        let mut acc = [zero; 4];
        for (k, (c, uc)) in x
            .as_chunks::<2>().0.iter()
            .zip(u[..2 * i].chunks_exact_mut(2))
            .enumerate()
        {
            let v = F2::load(c).mul(inv_scale);
            v.store(uc);
            acc[k & 3] = v.mul_add(v, acc[k & 3]);
        }
        let h = acc[0].add(acc[2]).add(acc[1].add(acc[3]));
        let f = F2::load(&u[2 * l..]);
        let sh = h.sqrt();
        let g = f.select_ge_zero(sh.neg(), sh);
        scale.mul(g).store(&mut e[2 * i..]);
        let h = f.neg_mul_add(g, h);
        let h = scale.select_eq_zero(one, h);
        f.sub(g).store(&mut u[2 * l..]);

        p[..2 * i].fill(0.0);
        let mut j = 0;
        while j + 4 <= i {
            fused_rows::<4>(head, n, j, u, p, pu, pq);
            j += 4;
        }
        match i - j {
            3 => fused_rows::<3>(head, n, j, u, p, pu, pq),
            2 => fused_rows::<2>(head, n, j, u, p, pu, pq),
            1 => fused_rows::<1>(head, n, j, u, p, pu, pq),
            _ => {}
        }
        let inv_h = one.div(h);
        let mut acc = [zero; 4];
        for (k, (pc, uc)) in p[..2 * i]
            .as_chunks_mut::<2>().0.iter_mut()
            .zip(u[..2 * i].chunks_exact(2))
            .enumerate()
        {
            let pk = F2::load(pc).mul(inv_h);
            pk.store(pc);
            acc[k & 3] = pk.mul_add(F2::load(uc), acc[k & 3]);
        }
        let f_acc = acc[0].add(acc[2]).add(acc[1].add(acc[3]));
        let hh = f_acc.div(h.add(h));
        let (ul, ql) = (
            F2::load(&u[2 * l..]),
            hh.neg_mul_add(F2::load(&u[2 * l..]), F2::load(&p[2 * l..])),
        );
        // q = p - hh u, and row l (the next Householder source) updated eagerly
        let row = &mut head[2 * l * n..2 * l * n + 2 * i];
        for ((pc, uc), rc) in p[..2 * i]
            .as_chunks_mut::<2>().0.iter_mut()
            .zip(u[..2 * i].chunks_exact(2))
            .zip(row.chunks_exact_mut(2))
        {
            let uk = F2::load(uc);
            let qk = hh.neg_mul_add(uk, F2::load(pc));
            qk.store(pc);
            ql.neg_mul_add(uk, ul.neg_mul_add(qk, F2::load(rc)))
                .store(rc);
        }
        std::mem::swap(u, pu);
        std::mem::swap(p, pq);
    }
    for i in 0..n {
        F2::load(&a[2 * (i * n + i)..]).store(&mut d[2 * i..]);
    }
}

/// One pass over two interleaved tridiagonal matrices: at the Laguerre points
/// `xs[side]` the Sturm counts and `G = p'/p`, `H = (p'/p)^2 - p''/p` of
/// `p(x) = det(T - x)` (LDL^T pivots with derivative recurrences), and the
/// Sturm counts at the four probe points `sig`.
#[inline(always)]
#[allow(clippy::type_complexity)]
fn sweep(
    n: usize,
    d: &[f64],
    e2: &[f64],
    xs: [F2; 2],
    sig: [F2; 4],
    piv: F2,
) -> ([F2; 2], [F2; 2], [F2; 2], [F2; 4]) {
    let one = F2::splat(1.0);
    let zero = F2::splat(0.0);
    let npiv = piv.neg();
    let d0 = F2::load(&d[0..]);
    let mut pq = [zero; 4];
    let mut pc = [zero; 4];
    for k in 0..4 {
        let v = d0.sub(sig[k]).replace_small(piv, npiv);
        pq[k] = v;
        pc[k] = v.neg_indicator();
    }
    let mut cnt = [zero; 2];
    let mut qpp = [zero; 2];
    let mut t = [zero; 2];
    let mut r = [zero; 2];
    let mut g = [zero; 2];
    let mut h = [zero; 2];
    for s in 0..2 {
        let v = d0.sub(xs[s]).replace_small(piv, npiv);
        cnt[s] = v.neg_indicator();
        t[s] = one.div(v);
        r[s] = t[s].neg();
        g[s] = r[s];
        h[s] = r[s].mul(r[s]);
    }
    for i in 1..n {
        let di = F2::load(&d[2 * i..]);
        let ei2 = F2::load(&e2[2 * i..]);
        for s in 0..2 {
            let et = ei2.mul(t[s]);
            let qn = di.sub(xs[s]).sub(et).replace_small(piv, npiv);
            let qpn = et.mul_add(r[s], one.neg());
            let rr2 = r[s].mul(r[s]).mul(F2::splat(-2.0));
            let qppn = et.mul(qpp[s].mul_add(t[s], rr2));
            cnt[s] = cnt[s].add(qn.neg_indicator());
            qpp[s] = qppn;
            t[s] = one.div(qn);
            r[s] = qpn.mul(t[s]);
            g[s] = g[s].add(r[s]);
            h[s] = r[s].mul_add(r[s], qppn.neg_mul_add(t[s], h[s]));
        }
        for k in 0..4 {
            let v = di.sub(sig[k]).sub(ei2.div(pq[k])).replace_small(piv, npiv);
            pq[k] = v;
            pc[k] = pc[k].add(v.neg_indicator());
        }
    }
    (cnt, g, h, pc)
}

/// `(min, max)` eigenvalue of two interleaved symmetric tridiagonal matrices
/// (`d`, sub-diagonal `e[2..]`); Laguerre iteration certified by Sturm counts,
/// lane-parallel version of `tridiagonal_extremes`. `e2` is scratch.
#[inline(never)]
pub(super) fn tridiagonal_extremes_pair(
    n: usize,
    d: &[f64],
    e: &[f64],
    e2: &mut [f64],
    bounds: [SpectrumBounds; 2],
) -> [(f64, f64); 2] {
    let eps = f64::EPSILON;
    let mut gl = [f64::INFINITY; 2];
    let mut gu = [f64::NEG_INFINITY; 2];
    let mut max_e2 = [0.0f64; 2];
    e2[0] = 0.0;
    e2[1] = 0.0;
    for i in 0..n {
        for m in 0..2 {
            let mut r = 0.0;
            if i > 0 {
                let v = e[2 * i + m];
                r += v.abs();
                e2[2 * i + m] = v * v;
                max_e2[m] = max_e2[m].max(v * v);
            }
            if i + 1 < n {
                r += e[2 * i + 2 + m].abs();
            }
            gl[m] = gl[m].min(d[2 * i + m] - r);
            gu[m] = gu[m].max(d[2 * i + m] + r);
        }
    }
    let mut pivmin = [0.0f64; 2];
    let mut tnorm = [0.0f64; 2];
    let mut inner = [(0.0f64, 0.0f64); 2];
    for m in 0..2 {
        pivmin[m] = f64::MIN_POSITIVE * max_e2[m].max(1.0);
        tnorm[m] = gl[m].abs().max(gu[m].abs());
        let fudge = 2.1 * tnorm[m] * eps * n as f64 + 4.2 * pivmin[m];
        gl[m] = gl[m].max(bounds[m].lo) - fudge;
        gu[m] = gu[m].min(bounds[m].hi) + fudge;
        inner[m] = (bounds[m].min_ub + fudge, bounds[m].max_lb - fudge);
    }

    let piv = F2::new(pivmin[0], pivmin[1]);
    // [matrix][side]: side 0 = minimum, 1 = maximum
    let mut ext = [0, 1].map(|m| {
        [
            Extreme::new(0, n, gl[m], inner[m].0, tnorm[m]),
            Extreme::new(1, n, inner[m].1, gu[m], tnorm[m]),
        ]
    });
    for _ in 0..MAX_IT {
        // chains: Laguerre at x[.][side]; Sturm counts at x -/+ probe
        let xs = [
            F2::new(ext[0][0].x, ext[1][0].x),
            F2::new(ext[0][1].x, ext[1][1].x),
        ];
        let sig = [
            F2::new(ext[0][0].probes[0], ext[1][0].probes[0]),
            F2::new(ext[0][0].probes[1], ext[1][0].probes[1]),
            F2::new(ext[0][1].probes[0], ext[1][1].probes[0]),
            F2::new(ext[0][1].probes[1], ext[1][1].probes[1]),
        ];
        let (cnt, g, h, pc) = sweep(n, d, e2, xs, sig, piv);
        let lane = |v: F2, m: usize| if m == 0 { v.lane0() } else { v.lane1() };
        for (m, sides) in ext.iter_mut().enumerate() {
            for (s, e) in sides.iter_mut().enumerate() {
                let probes = [lane(pc[2 * s], m), lane(pc[2 * s + 1], m)];
                e.update(lane(cnt[s], m), probes, lane(g[s], m), lane(h[s], m));
            }
        }
        if ext.iter().all(|e| e[0].done && e[1].done) {
            break;
        }
    }
    let mut out = [(0.0, 0.0); 2];
    for m in 0..2 {
        let [e0, e1] = ext[m];
        let mut res = [e0.result, e1.result];
        if !(e0.done && e1.done) {
            let dm: Vec<f64> = (0..n).map(|i| d[2 * i + m]).collect();
            let em: Vec<f64> = (0..n).map(|i| e2[2 * i + m]).collect();
            let need = [!e0.done, !e1.done];
            let ms = multisection(
                n,
                &dm,
                &em,
                pivmin[m],
                [(e0.lo, e0.hi), (e1.lo, e1.hi)],
                need,
            );
            for s in 0..2 {
                if need[s] {
                    res[s] = ms[s];
                }
            }
        }
        out[m] = (res[0], res[1]);
    }
    out
}
