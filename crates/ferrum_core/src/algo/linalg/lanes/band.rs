//! Tridiagonalization of Burden matrices through their banded part.
//!
//! A Burden matrix is `A = S + f * 1 1^T` with `S` as sparse as the
//! molecular graph (`S_ij = w_ij - f` on bonds, `S_ii = d_i - f`, zero
//! elsewhere). After a reverse Cuthill-McKee ordering of the atoms `S` is a
//! band matrix of small half-bandwidth `b`, so instead of a dense Householder
//! reduction (`4/3 n^3` flops) two sequences of plane rotations are used:
//!
//! 1. rotations in planes `(k, k + 1)`, `k = n - 2 .. 0`, that turn `1` into
//!    `sqrt(n) e_0`; the two entries each one pushes out of the band are
//!    chased off the matrix (down and up, `b` rows per rotation);
//! 2. Schwarz's band reduction of the result to tridiagonal form, whose
//!    rotations never touch index 0 (bulges chased down).
//!
//! With `Q` the product of all rotations, `Q^T 1 = sqrt(n) e_0`, so
//! `Q^T A Q = T + f n e_0 e_0^T` with `T = Q^T S Q` tridiagonal: about
//! `n^2 / 2` rotations of `O(b)` work. Orthogonal transformations are
//! backward stable, so the extremes agree with the dense reduction to
//! `O(n * eps * ||A||)`.

use super::FV;

/// Lane-interleaved lower band storage of `2 * V` symmetric matrices:
/// element `(i, j)`, `j <= i <= j + bb`, at `(j * (bb + 1) + i - j) * 2 * V`
/// (`bb = b + 1` leaves room for one bulge diagonal).
struct Band<'a, const V: usize> {
    a: &'a mut [f64],
    n: usize,
    b: usize,
}

impl<const V: usize> Band<'_, V> {
    #[inline(always)]
    fn idx(&self, i: usize, j: usize) -> usize {
        debug_assert!(j <= i && i - j <= self.b + 1 && i < self.n);
        (j * (self.b + 2) + i - j) * 2 * V
    }
    #[inline(always)]
    fn get(&self, i: usize, j: usize) -> FV<V> {
        FV::load(&self.a[self.idx(i, j)..])
    }
    #[inline(always)]
    fn set(&mut self, i: usize, j: usize, v: FV<V>) {
        let k = self.idx(i, j);
        v.store(&mut self.a[k..]);
    }

    /// Applies the rotation `(c, s)` in plane `(p, p + 1)`,
    /// `x_p' = c x_p + s x_q`, `x_q' = c x_q - s x_p`, as a similarity
    /// transformation: the row pairs in columns `lo..p`, the 2 x 2 diagonal
    /// block and the column pairs in rows `p + 2 ..= hi`.
    #[inline(always)]
    fn rotate(&mut self, p: usize, c: FV<V>, s: FV<V>, lo: usize, hi: usize) {
        let q = p + 1;
        let w = 2 * V;
        let cs = (self.b + 2) * w;
        // every touched element lies inside the stored band of the matrix
        assert!(lo <= p && q - lo <= self.b + 1 && hi < self.n && hi <= q + self.b);
        assert!(self.a.len() >= self.n * cs);
        // SAFETY: (i, j) with j <= i < n and i - j <= b + 1 is at
        // j * cs + (i - j) * w < n * cs (checked above for every element)
        unsafe {
            let base = self.a.as_mut_ptr();
            // row pairs (p, k), (q, k): adjacent, one column apart per k
            let mut ptr = base.add(lo * cs + (p - lo) * w);
            for _ in lo..p {
                let (x, y) = rot(c, s, FV::load_ptr(ptr), FV::load_ptr(ptr.add(w)));
                x.store_ptr(ptr);
                y.store_ptr(ptr.add(w));
                ptr = ptr.add(cs - w);
            }
            let (pp, qq) = (base.add(p * cs), base.add(q * cs));
            let (a, bq, dq) = (FV::load_ptr(pp), FV::load_ptr(pp.add(w)), FV::load_ptr(qq));
            // R = G M (by columns), then G M G^T = R G^T (by rows)
            let (r11, r21) = rot(c, s, a, bq);
            let (r12, r22) = rot(c, s, bq, dq);
            let (m11, _) = rot(c, s, r11, r12);
            let (m21, m22) = rot(c, s, r21, r22);
            m11.store_ptr(pp);
            m21.store_ptr(pp.add(w));
            m22.store_ptr(qq);
            // column pairs (i, p), (i, q), i = q + 1 ..= hi
            for t in 0..hi - q {
                let (xp, yp) = (pp.add((t + 2) * w), qq.add((t + 1) * w));
                let (x, y) = rot(c, s, FV::load_ptr(xp), FV::load_ptr(yp));
                x.store_ptr(xp);
                y.store_ptr(yp);
            }
        }
    }
}

/// `(c x + s y, c y - s x)`.
#[inline(always)]
fn rot<const V: usize>(c: FV<V>, s: FV<V>, x: FV<V>, y: FV<V>) -> (FV<V>, FV<V>) {
    (s.mul_add(y, c.mul(x)), s.neg_mul_add(x, c.mul(y)))
}

/// Rotation `(c, s)` with `(c x + s y, c y - s x) = (r, 0)`, and `r`
/// (`c = 1, s = 0` where `x = y = 0`).
#[inline(always)]
fn givens<const V: usize>(x: FV<V>, y: FV<V>) -> (FV<V>, FV<V>, FV<V>) {
    let r = x.mul_add(x, y.mul(y)).sqrt();
    let zero = FV::<V>::splat(0.0);
    let one = FV::<V>::splat(1.0);
    let inv = r.select_eq_zero(zero, one.div(r));
    (r.select_eq_zero(one, x.mul(inv)), y.mul(inv), r)
}

/// Chases the bulge at `(r, r - b - 1)` down and off the matrix.
#[inline(always)]
fn chase_down<const V: usize>(m: &mut Band<'_, V>, mut r: usize) {
    let (n, b) = (m.n, m.b);
    while r < n {
        let (p, c0) = (r - 1, r - b - 1);
        let (c, s, rr) = givens(m.get(p, c0), m.get(r, c0));
        m.set(p, c0, rr);
        m.set(r, c0, FV::splat(0.0));
        m.rotate(p, c, s, c0 + 1, (r + b).min(n - 1));
        r += b;
    }
}

/// Chases the bulge at `(c0 + b + 1, c0)` up and off the matrix (rotations
/// in planes `(c0, c0 + 1)`, also applied to `v`).
#[inline(always)]
fn chase_up<const V: usize>(m: &mut Band<'_, V>, v: &mut [f64], mut c0: usize) {
    let b = m.b;
    let w = 2 * V;
    loop {
        let (p, q, r0) = (c0, c0 + 1, c0 + b + 1);
        let (x, y) = (m.get(r0, p), m.get(r0, q));
        // (x, y) -> (0, r)
        let (c, s, rr) = givens(y, x.neg());
        m.set(r0, p, FV::splat(0.0));
        m.set(r0, q, rr);
        m.rotate(p, c, s, p.saturating_sub(b), r0 - 1);
        let (vp, vq) = rot(c, s, FV::load(&v[w * p..]), FV::load(&v[w * q..]));
        vp.store(&mut v[w * p..]);
        vq.store(&mut v[w * q..]);
        if p < b {
            break;
        }
        c0 = p - b;
    }
}

/// Reduces the `2 * V` banded matrices in `a` (half-bandwidth `b`, see
/// [`Band`]) to tridiagonal form `T` with `Q^T 1 = sqrt(n) e_0` (see the
/// module documentation): diagonal into `d`, sub-diagonal into `e[2 * V..]`
/// (lane-interleaved). `v` is scratch of length `2 * V * n`.
pub(in super::super) fn tridiagonalize<const V: usize>(
    n: usize,
    b: usize,
    a: &mut [f64],
    v: &mut [f64],
    d: &mut [f64],
    e: &mut [f64],
) {
    let w = 2 * V;
    assert!(b >= 2 && n >= 2 && a.len() >= n * (b + 2) * w && v.len() >= w * n);
    assert!(d.len() >= w * n && e.len() >= w * n);
    let mut m = Band::<V> { a, n, b };
    v[..w * n].fill(1.0);
    // 1. 1 -> sqrt(n) e_0
    for k in (0..n - 1).rev() {
        let (c, s, rr) = givens(FV::load(&v[w * k..]), FV::load(&v[w * (k + 1)..]));
        rr.store(&mut v[w * k..]);
        FV::<V>::splat(0.0).store(&mut v[w * (k + 1)..]);
        m.rotate(k, c, s, k.saturating_sub(b), (k + 1 + b).min(n - 1));
        if k + 1 + b < n {
            chase_down(&mut m, k + 1 + b);
        }
        if k >= b {
            chase_up(&mut m, v, k - b);
        }
    }
    // 2. band reduction (Schwarz), column by column from the outside in
    for j in 0..n.saturating_sub(2) {
        for dd in (2..=b.min(n - 1 - j)).rev() {
            let (p, q) = (j + dd - 1, j + dd);
            let (c, s, rr) = givens(m.get(p, j), m.get(q, j));
            m.set(p, j, rr);
            m.set(q, j, FV::splat(0.0));
            m.rotate(p, c, s, j + 1, (q + b).min(n - 1));
            if q + b < n {
                chase_down(&mut m, q + b);
            }
        }
    }
    for i in 0..n {
        m.get(i, i).store(&mut d[w * i..]);
        if i > 0 {
            m.get(i, i - 1).store(&mut e[w * i..]);
        }
    }
}

/// Packs `2 * V` Burden matrices minus their fill (`S = A - f 1 1^T`) into
/// band storage for [`tridiagonalize`] (half-bandwidth `b` of the ordering
/// `inv[old] = new`; lane `m` holds `jobs[m]`, or the last job beyond
/// `jobs.len()`).
pub(in super::super) fn assemble<const V: usize>(
    n: usize,
    b: usize,
    inv: &[u32],
    jobs: &[super::super::BurdenMatrix],
    a: &mut [f64],
) {
    let w = 2 * V;
    let len = n * (b + 2) * w;
    a[..len].fill(0.0);
    let at = |i: usize, j: usize| {
        let (i, j) = (i.max(j), i.min(j));
        (j * (b + 2) + i - j) * w
    };
    for m in 0..w {
        let job = &jobs[m.min(jobs.len() - 1)];
        for &(i, j, x) in job.entries {
            if i != j {
                a[at(inv[i as usize] as usize, inv[j as usize] as usize) + m] = x - job.fill;
            }
        }
        for (i, &x) in job.diag[..n].iter().enumerate() {
            let k = inv[i] as usize;
            a[at(k, k) + m] = x - job.fill;
        }
    }
}

/// Reverse Cuthill-McKee ordering of the graph with `n` vertices and the
/// given edges: `order[new] = old`, and the half-bandwidth of the reordered
/// adjacency (at least 2: with `b = 1` the two bulges of a first-stage
/// rotation would meet).
pub(in super::super) fn rcm_order(n: usize, edges: &[(u32, u32)]) -> (Vec<u32>, usize) {
    let mut deg = vec![0u32; n + 1];
    for &(i, j) in edges {
        if i != j {
            deg[i as usize] += 1;
            deg[j as usize] += 1;
        }
    }
    // CSR adjacency
    let mut start = vec![0u32; n + 1];
    for i in 0..n {
        start[i + 1] = start[i] + deg[i];
    }
    let mut fill = start.clone();
    let mut adj = vec![0u32; start[n] as usize];
    for &(i, j) in edges {
        if i != j {
            adj[fill[i as usize] as usize] = j;
            fill[i as usize] += 1;
            adj[fill[j as usize] as usize] = i;
            fill[j as usize] += 1;
        }
    }
    let nb = |i: usize| &adj[start[i] as usize..start[i + 1] as usize];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut level = vec![u32::MAX; n];
    let mut queue: Vec<u32> = Vec::new();
    let mut scratch: Vec<u32> = Vec::new();
    while order.len() < n {
        // start: an unvisited vertex of minimum degree, moved to a far end
        // of its component (one George-Liu pseudo-peripheral step: the
        // minimum-degree vertex of the last BFS level)
        let first = (0..n).filter(|&i| !seen[i]).min_by_key(|&i| (deg[i], i)).unwrap_or(0);
        queue.clear();
        queue.push(first as u32);
        level[first] = 0;
        let mut h = 0;
        while h < queue.len() {
            let x = queue[h] as usize;
            h += 1;
            for &y in nb(x) {
                if level[y as usize] == u32::MAX {
                    level[y as usize] = level[x] + 1;
                    queue.push(y);
                }
            }
        }
        let last = level[queue[queue.len() - 1] as usize];
        let root = queue
            .iter()
            .map(|&x| x as usize)
            .filter(|&x| level[x] == last)
            .min_by_key(|&x| (deg[x], x))
            .unwrap_or(first);
        let base = order.len();
        order.push(root as u32);
        seen[root] = true;
        let mut h = base;
        while h < order.len() {
            let x = order[h] as usize;
            h += 1;
            scratch.clear();
            scratch.extend(nb(x).iter().copied().filter(|&y| !seen[y as usize]));
            scratch.sort_unstable_by_key(|&y| (deg[y as usize], y));
            scratch.dedup();
            for &y in &scratch {
                if !seen[y as usize] {
                    seen[y as usize] = true;
                    order.push(y);
                }
            }
        }
    }
    order.reverse();
    let mut inv = vec![0u32; n];
    for (k, &o) in order.iter().enumerate() {
        inv[o as usize] = k as u32;
    }
    let b = edges
        .iter()
        .map(|&(i, j)| (inv[i as usize] as i64 - inv[j as usize] as i64).unsigned_abs() as usize)
        .max()
        .unwrap_or(0)
        .max(2);
    (order, b)
}
