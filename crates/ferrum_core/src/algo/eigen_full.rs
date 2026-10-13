//! Dense real symmetric eigen-decomposition at LAPACK accuracy.
//!
//! - [`LaneEigen`] / [`SymEigen`] / [`solve_batch`]: every eigenvalue plus
//!   the unit eigenvector of the largest one, with reusable buffers, for two
//!   or four same-size matrices at a time ([`LaneEigen`]) or for one
//!   ([`SymEigen`]). Householder reduction to tridiagonal form, each rank-2
//!   update fused into the next step's matrix-vector product, reflectors
//!   kept: with one matrix per SIMD lane from the last row up
//!   (`linalg::lanes`, EISPACK `tred1` reflectors) or scalar from the first
//!   column on (`dsytd2`). Then implicit QL on the tridiagonal matrices for
//!   the eigenvalues (EISPACK `tql1`; the sweeps of all matrices of a call
//!   run interleaved), inverse iteration on the tridiagonal matrix for the
//!   eigenvector (LU with partial pivoting, LAPACK `dgttrf` / `dgtts2`, as in
//!   `dstein`; for [`LaneEigen`] one matrix per lane) and back-transformation
//!   with the kept reflectors. This costs about a third of a full
//!   decomposition with vectors.
//! - [`symmetric_eigen_full`]: all eigenvalues and eigenvectors (reflectors
//!   accumulated into `Q`, implicit QL with rotations applied to `Q`, EISPACK
//!   `tql2`).
//!
//! Every step is backward stable, so eigenvalues agree with `numpy.linalg.eigh`
//! (LAPACK `dsyevd`) to `O(n * eps * ||A||)` and an eigenvector of a simple
//! eigenvalue to `O(n * eps * ||A|| / gap)` (up to its sign).

use super::linalg::lanes::{self, tri};

/// Unit roundoff of `f64` (EISPACK's `eps`).
const EPS: f64 = f64::EPSILON;

/// Reusable workspace for the decomposition of `2` or `4` symmetric
/// `n x n` matrices in lock step (one per SIMD lane): eigenvalues and the
/// eigenvector of the largest eigenvalue of each.
///
/// The matrices are written into lane-interleaved packed lower triangles
/// ([`LaneEigen::packed_mut`]). The Householder reduction runs from the last
/// row up (EISPACK `tred1`), i.e. it starts from row `n - 1`; LAPACK's
/// `dsytrd` with a lower triangle (`numpy.linalg.eigh`) starts from column
/// `0`, which a caller can mirror by passing the matrices in reverse index
/// order.
///
/// ```
/// use ferrum_core::algo::eigen_full::{LaneEigen, solve_lanes};
/// let mut eig = LaneEigen::new();
/// // lane 0: [[2, 1], [1, 2]]; lane 1: [[1, 0], [0, -1]]
/// eig.packed_mut(2, 2, 2).copy_from_slice(&[2.0, 1.0, 1.0, 0.0, 2.0, -1.0]);
/// solve_lanes(std::slice::from_mut(&mut eig));
/// assert!(eig.is_ok(0) && eig.is_ok(1));
/// let (lo, hi) = eig.extremes(0);
/// assert!((lo - 1.0).abs() < 1e-15 && (hi - 3.0).abs() < 1e-15);
/// assert!((eig.max_vector(0)[0].abs() - 0.5f64.sqrt()).abs() < 1e-15);
/// assert_eq!(eig.extremes(1), (-1.0, 1.0));
/// ```
#[derive(Default)]
pub struct LaneEigen {
    n: usize,
    /// lanes per pass (2 or 4) and lanes holding a matrix
    lanes: usize,
    used: usize,
    /// packed matrices (layout in [`LaneEigen::packed_mut`]); overwritten by
    /// the reflectors
    a: Vec<f64>,
    /// reflector scales, interleaved
    h: Vec<f64>,
    /// tridiagonal matrices, interleaved: diagonal, sub-diagonal (`te[w * i]`
    /// couples `i - 1` and `i`)
    td: Vec<f64>,
    te: Vec<f64>,
    // scratch of the reduction
    s0: Vec<f64>,
    s1: Vec<f64>,
    s2: Vec<f64>,
    /// per lane (`lane * n ..`): eigenvalues (in no particular order) and the
    /// unit eigenvector of the largest one
    w: Vec<f64>,
    v: Vec<f64>,
    /// per lane: QL off-diagonal (`q[i]` couples `i` and `i + 1`)
    q: Vec<f64>,
    /// interleaved eigenvectors during the back-transformation
    x: Vec<f64>,
    /// scratch of the inverse iteration
    ivs: Vec<f64>,
    ok: [bool; MAX_LANES],
    /// smallest and largest eigenvalue
    ext: [(f64, f64); MAX_LANES],
}

/// Most matrices per [`LaneEigen`].
pub const MAX_LANES: usize = 4;

/// Scratch of the inverse iteration.
#[derive(Default)]
struct InverseIteration {
    d: Vec<f64>,
    e: Vec<f64>,
    dd: Vec<f64>,
    dl: Vec<f64>,
    du: Vec<f64>,
    du2: Vec<f64>,
    piv: Vec<bool>,
    y: Vec<f64>,
}

impl LaneEigen {
    pub fn new() -> Self {
        Self::default()
    }

    /// Storage for `used <= lanes` symmetric `n x n` matrices decomposed in
    /// lock step (`lanes` is 2 or 4): element `(i, j)`, `j <= i`, of matrix
    /// `l` at `lanes * (i * (i + 1) / 2 + j) + l`. Every element of the
    /// `used` lanes must be written before [`solve_lanes`]; the other lanes
    /// are ignored. The previous contents are unspecified.
    pub fn packed_mut(&mut self, n: usize, lanes: usize, used: usize) -> &mut [f64] {
        assert!((lanes == 2 || lanes == 4) && used <= lanes);
        self.n = n;
        self.lanes = lanes;
        self.used = used;
        self.a.resize(lanes * tri(n), 0.0);
        &mut self.a[..lanes * tri(n)]
    }

    /// Number of lanes holding a matrix.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Whether the decomposition of lane `l` succeeded (finite matrix, QL
    /// converged).
    pub fn is_ok(&self, l: usize) -> bool {
        self.ok[l]
    }

    /// Eigenvalues of lane `l`, in no particular order.
    pub fn values(&self, l: usize) -> &[f64] {
        &self.w[l * self.n..(l + 1) * self.n]
    }

    /// Smallest and largest eigenvalue of lane `l`.
    pub fn extremes(&self, l: usize) -> (f64, f64) {
        self.ext[l]
    }

    /// Unit eigenvector of the largest eigenvalue of lane `l` (arbitrary sign).
    pub fn max_vector(&self, l: usize) -> &[f64] {
        &self.v[l * self.n..(l + 1) * self.n]
    }

    /// Tridiagonal reduction (first part of a decomposition); leaves the
    /// tridiagonal matrix of each lane in `w` / `q` for the QL iteration.
    fn reduce(&mut self) {
        let (n, w) = (self.n, self.lanes);
        for buf in [&mut self.h, &mut self.td, &mut self.te, &mut self.s0, &mut self.s1, &mut self.s2, &mut self.x] {
            buf.clear();
            buf.resize(w * n, 0.0);
        }
        for buf in [&mut self.w, &mut self.v, &mut self.q] {
            buf.clear();
            buf.resize(w * n, 0.0);
        }
        let a = &mut self.a[..w * tri(n)];
        // a non-finite matrix fails; it and the unused lanes are zeroed so
        // that they cannot influence the others (the reduction scales a pass
        // when any lane's row norm is out of range)
        let mut finite = [true; MAX_LANES];
        for c in a.chunks_exact(w) {
            for (f, x) in finite.iter_mut().zip(c) {
                *f &= x.is_finite();
            }
        }
        for l in 0..w {
            self.ok[l] = l < self.used && finite[l];
            if !self.ok[l] {
                a.iter_mut().skip(l).step_by(w).for_each(|x| *x = 0.0);
            }
        }
        if n == 0 {
            return;
        }
        match w {
            2 => lanes::tridiagonalize_keep::<1>(
                n,
                a,
                &mut self.td,
                &mut self.te,
                &mut self.h,
                &mut self.s0,
                &mut self.s1,
                &mut self.s2,
            ),
            _ => lanes::tridiagonalize_keep::<2>(
                n,
                a,
                &mut self.td,
                &mut self.te,
                &mut self.h,
                &mut self.s0,
                &mut self.s1,
                &mut self.s2,
            ),
        }
        for l in 0..w {
            let (d, q) = (&mut self.w[l * n..(l + 1) * n], &mut self.q[l * n..(l + 1) * n]);
            for i in 0..n {
                d[i] = self.td[w * i + l];
                q[i] = if i + 1 < n { self.te[w * (i + 1) + l] } else { 0.0 };
            }
        }
    }

    /// Eigenvectors of the largest eigenvalues once `w` holds the eigenvalues.
    fn finish(&mut self) {
        let (n, w) = (self.n, self.lanes);
        if n == 0 {
            return;
        }
        let mut lambda = [0.0; MAX_LANES];
        for l in 0..w {
            if self.ok[l] {
                let vals = &self.w[l * n..(l + 1) * n];
                let lo = vals.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                self.ext[l] = (lo, hi);
                lambda[l] = hi;
            }
        }
        self.ivs.resize(5 * w * n, 0.0);
        match w {
            2 => {
                lanes::inverse_iteration::<1>(n, &self.td, &self.te, &lambda, &mut self.x, &mut self.ivs);
                lanes::back_transform::<1>(n, &self.a, &self.h, &mut self.x);
            }
            _ => {
                lanes::inverse_iteration::<2>(n, &self.td, &self.te, &lambda, &mut self.x, &mut self.ivs);
                lanes::back_transform::<2>(n, &self.a, &self.h, &mut self.x);
            }
        }
        for l in 0..w {
            if self.ok[l] {
                for (i, v) in self.v[l * n..(l + 1) * n].iter_mut().enumerate() {
                    *v = self.x[w * i + l];
                }
            }
        }
    }
}

/// Reusable workspace: eigenvalues and the eigenvector of the largest
/// eigenvalue of one symmetric matrix, by a scalar Householder reduction
/// that starts from column 0 (LAPACK `dsytd2` on the lower triangle, as
/// `numpy.linalg.eigh`; each rank-2 update fused into the next step's
/// matrix-vector product, reflectors kept in place).
///
/// ```
/// use ferrum_core::algo::eigen_full::SymEigen;
/// let mut eig = SymEigen::new();
/// eig.matrix_mut(2).copy_from_slice(&[2.0, 1.0, 1.0, 2.0]);
/// assert!(eig.solve());
/// assert!((eig.values()[0] - 1.0).abs() < 1e-15 && (eig.values()[1] - 3.0).abs() < 1e-15);
/// assert!((eig.max_vector()[0].abs() - 0.5f64.sqrt()).abs() < 1e-15);
/// ```
#[derive(Default)]
pub struct SymEigen {
    n: usize,
    /// `n x n` row-major matrix; overwritten by the Householder vectors
    a: Vec<f64>,
    tau: Vec<f64>,
    /// tridiagonal matrix: diagonal and sub-diagonal (`e[n-1] = 0`)
    d: Vec<f64>,
    e: Vec<f64>,
    /// eigenvalues, ascending
    w: Vec<f64>,
    /// unit eigenvector of the largest eigenvalue
    v: Vec<f64>,
    /// QL off-diagonal
    p: Vec<f64>,
    scratch: Vec<f64>,
    ok: bool,
    iv: InverseIteration,
}

impl SymEigen {
    pub fn new() -> Self {
        Self::default()
    }

    /// The `n x n` row-major input matrix, to be filled (both triangles) before
    /// [`SymEigen::solve`]. Its previous contents are unspecified.
    pub fn matrix_mut(&mut self, n: usize) -> &mut [f64] {
        self.n = n;
        self.a.resize(n * n, 0.0);
        &mut self.a[..n * n]
    }

    /// Decomposes the matrix set by [`SymEigen::matrix_mut`] (which is
    /// overwritten). Returns `false` when the matrix holds a non-finite value
    /// or the QL iteration does not converge.
    pub fn solve(&mut self) -> bool {
        solve_batch(std::slice::from_mut(self), &mut []);
        self.ok
    }

    /// Whether the last decomposition succeeded.
    pub fn is_ok(&self) -> bool {
        self.ok
    }

    /// Eigenvalues in ascending order.
    pub fn values(&self) -> &[f64] {
        &self.w
    }

    /// Unit eigenvector of the largest eigenvalue (arbitrary sign).
    pub fn max_vector(&self) -> &[f64] {
        &self.v
    }

    /// Tridiagonal reduction (first part of a decomposition).
    fn reduce(&mut self) {
        let n = self.n;
        for buf in [&mut self.tau, &mut self.d, &mut self.e, &mut self.w, &mut self.v, &mut self.p] {
            buf.clear();
            buf.resize(n, 0.0);
        }
        self.ok = n == 0 || self.a[..n * n].iter().all(|x| x.is_finite());
        if n == 0 || !self.ok {
            return;
        }
        self.scratch.resize(4 * n, 0.0);
        tridiagonalize(n, &mut self.a, &mut self.tau, &mut self.d, &mut self.e, &mut self.scratch);
        self.w.copy_from_slice(&self.d);
        self.p.copy_from_slice(&self.e);
    }

    /// Eigenvector of the largest eigenvalue once `w` holds the eigenvalues.
    fn finish(&mut self) {
        let n = self.n;
        if n == 0 || !self.ok {
            return;
        }
        self.w.sort_by(f64::total_cmp);
        let lambda = self.w[n - 1];
        let iv = &mut self.iv;
        iv.d.clear();
        iv.d.extend_from_slice(&self.d);
        iv.e.clear();
        iv.e.extend_from_slice(&self.e);
        iv.y.resize(n, 0.0);
        iv.vector(lambda);
        self.v.copy_from_slice(&iv.y[..n]);
        apply_reflectors(n, &self.a, &self.tau, &mut self.v);
    }
}

/// [`solve_batch`] without scalar workspaces.
pub fn solve_lanes(batch: &mut [LaneEigen]) {
    solve_batch(&mut [], batch);
}

/// Decomposes the matrices of several workspaces (each set by
/// [`SymEigen::matrix_mut`] / [`LaneEigen::packed_mut`]);
/// [`SymEigen::is_ok`] / [`LaneEigen::is_ok`] tell which succeeded.
///
/// The implicit QL sweeps are sequential chains of divisions and square
/// roots; running the sweeps of all matrices interleaved lets the CPU
/// overlap them, which roughly halves the eigenvalue cost.
pub fn solve_batch(singles: &mut [SymEigen], batch: &mut [LaneEigen]) {
    for s in singles.iter_mut() {
        s.reduce();
    }
    for b in batch.iter_mut() {
        b.reduce();
    }
    let mut probs: Vec<QlProblem> = Vec::new();
    for s in singles.iter_mut() {
        if s.ok && s.n > 0 {
            probs.push((&mut s.w[..], &mut s.p[..], &mut s.ok, true));
        }
    }
    for b in batch.iter_mut() {
        let n = b.n;
        if n == 0 {
            continue;
        }
        for ((d, e), ok) in b.w.chunks_exact_mut(n).zip(b.q.chunks_exact_mut(n)).zip(b.ok.iter_mut()) {
            if *ok {
                probs.push((d, e, ok, false));
            }
        }
    }
    ql_interleaved(&mut probs);
    drop(probs);
    for s in singles.iter_mut() {
        s.finish();
    }
    for b in batch.iter_mut() {
        b.finish();
    }
}

/// A tridiagonal matrix for [`ql_interleaved`]: diagonal, off-diagonal,
/// success flag, and whether its rotations use [`Ql::rsqrt`].
type QlProblem<'a> = (&'a mut [f64], &'a mut [f64], &'a mut bool, bool);

/// Implicit QL on every tridiagonal matrix `(d, e)` of `probs` (eigenvalues
/// left in `d`, unsorted); the flag is cleared where the iteration does not
/// converge.
fn ql_interleaved(probs: &mut [QlProblem]) {
    for chunk in probs.chunks_mut(QL_LANES) {
        let mut st = [Ql::default(); QL_LANES];
        for (s, p) in st.iter_mut().zip(chunk.iter()) {
            s.done = p.0.is_empty();
            s.rsqrt = p.3;
        }
        loop {
            let mut active = false;
            for (s, (d, e, ok, _)) in st.iter_mut().zip(chunk.iter_mut()) {
                if s.done {
                    continue;
                }
                let (d, e) = (&mut d[..], &mut e[..]);
                if s.i == s.l {
                    if s.sweeping {
                        s.end_sweep(d, e);
                    }
                    if !s.next_sweep(d, e) {
                        if s.failed {
                            **ok = false;
                        }
                        continue;
                    }
                }
                s.step(d, e);
                active = true;
            }
            if !active {
                break;
            }
        }
    }
}

/// Number of QL iterations run interleaved.
const QL_LANES: usize = 16;

/// State of the implicit QL iteration (EISPACK `tql1` as in JAMA) on one
/// symmetric tridiagonal matrix (`d` diagonal, `e[i]` coupling `i` and `i+1`,
/// `e[n-1] = 0`), resumable after every rotation so that several matrices can
/// be iterated interleaved. Eigenvalues are left in `d` (unsorted).
#[derive(Clone, Copy, Default)]
struct Ql {
    /// current eigenvalue index, end of its unreduced block
    l: usize,
    m: usize,
    /// rotation index of the running sweep (`l` when none is running)
    i: usize,
    f: f64,
    tst1: f64,
    iter: u32,
    in_l: bool,
    sweeping: bool,
    done: bool,
    failed: bool,
    /// rotations from a Newton-refined reciprocal square root (fewer slow
    /// operations, more operations in all) instead of `sqrt` and a division
    rsqrt: bool,
    c: f64,
    c2: f64,
    c3: f64,
    s: f64,
    s2: f64,
    p: f64,
    el1: f64,
    dl1: f64,
}

impl Ql {
    /// Deflates converged eigenvalues and sets up the next sweep (Wilkinson
    /// shift); `false` once all eigenvalues are found (or on failure).
    fn next_sweep(&mut self, d: &mut [f64], e: &mut [f64]) -> bool {
        let n = d.len();
        self.sweeping = false;
        let mut l = self.l;
        if self.in_l && e[l].abs() <= EPS * self.tst1 {
            d[l] += self.f;
            e[l] = 0.0;
            l += 1;
            self.in_l = false;
        }
        while !self.in_l {
            if l == n {
                self.l = l;
                self.i = l;
                self.done = true;
                return false;
            }
            self.tst1 = self.tst1.max(d[l].abs() + e[l].abs());
            let mut m = l;
            while m < n - 1 && e[m].abs() > EPS * self.tst1 {
                m += 1;
            }
            if m == l {
                d[l] += self.f;
                e[l] = 0.0;
                l += 1;
            } else {
                self.m = m;
                self.in_l = true;
                self.iter = 0;
            }
        }
        self.l = l;
        self.iter += 1;
        if self.iter > 60 {
            self.done = true;
            self.failed = true;
            return false;
        }
        let m = self.m;
        let g = d[l];
        let mut p = (d[l + 1] - g) / (2.0 * e[l]);
        let mut r = pythag(p, 1.0);
        if p < 0.0 {
            r = -r;
        }
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        self.dl1 = d[l + 1];
        let h = g - d[l];
        for x in &mut d[l + 2..] {
            *x -= h;
        }
        self.f += h;
        p = d[m];
        self.p = p;
        (self.c, self.c2, self.c3) = (1.0, 1.0, 1.0);
        self.el1 = e[l + 1];
        (self.s, self.s2) = (0.0, 0.0);
        self.i = m;
        self.sweeping = true;
        true
    }

    /// One Givens rotation of the running sweep.
    #[inline(always)]
    fn step(&mut self, d: &mut [f64], e: &mut [f64]) {
        self.i -= 1;
        let i = self.i;
        self.c3 = self.c2;
        self.c2 = self.c;
        self.s2 = self.s;
        let (c, s, p) = (self.c, self.s, self.p);
        let g = c * e[i];
        let h = c * p;
        // r = sqrt(p² + e²), c = p / r, s = e / r
        let ei = e[i];
        let x = fma(p, p, ei * ei);
        let (r, inv) = if x.is_finite() && x >= f64::MIN_POSITIVE {
            if self.rsqrt {
                let inv = rsqrt(x);
                (x * inv, inv)
            } else {
                let r = x.sqrt();
                (r, 1.0 / r)
            }
        } else {
            let r = p.hypot(ei);
            (r, 1.0 / r)
        };
        e[i + 1] = s * r;
        let s = ei * inv;
        let c = p * inv;
        self.p = c * d[i] - s * g;
        d[i + 1] = h + s * (c * g + s * d[i]);
        (self.c, self.s) = (c, s);
    }

    /// Closes a sweep.
    fn end_sweep(&mut self, d: &mut [f64], e: &mut [f64]) {
        let l = self.l;
        let p = -self.s * self.s2 * self.c3 * self.el1 * e[l] / self.dl1;
        e[l] = self.s * p;
        d[l] = self.c * p;
    }
}

impl InverseIteration {
    /// Eigenvector of the tridiagonal matrix (`d`, `e[i]` coupling `i` and
    /// `i + 1`) for the eigenvalue `lambda` by inverse iteration (LAPACK
    /// `dstein`: LU of `T - lambda I` with partial pivoting, tiny pivots
    /// replaced, pseudo-random start), unit length, into `y`.
    fn vector(&mut self, lambda: f64) {
        let n = self.d.len();
        let v = &mut self.y[..n];
        if n == 1 {
            v[0] = 1.0;
            return;
        }
        let norm = self.d.iter().zip(&self.e).map(|(d, e)| d.abs() + e.abs()).fold(0.0f64, f64::max).max(f64::MIN_POSITIVE);
        let (dd, dl, du, du2, piv) = (&mut self.dd, &mut self.dl, &mut self.du, &mut self.du2, &mut self.piv);
        dd.clear();
        dd.extend(self.d.iter().map(|&d| d - lambda));
        dl.clear();
        dl.extend_from_slice(&self.e[..n - 1]);
        du.clear();
        du.extend_from_slice(&self.e[..n - 1]);
        du2.clear();
        du2.resize(n.saturating_sub(2), 0.0);
        piv.clear();
        piv.resize(n - 1, false);
        // dgttrf
        for i in 0..n - 1 {
            if dd[i].abs() >= dl[i].abs() {
                if dd[i] != 0.0 {
                    let fact = dl[i] / dd[i];
                    dl[i] = fact;
                    dd[i + 1] -= fact * du[i];
                }
            } else {
                let fact = dd[i] / dl[i];
                dd[i] = dl[i];
                dl[i] = fact;
                let temp = du[i];
                du[i] = dd[i + 1];
                dd[i + 1] = temp - fact * dd[i + 1];
                if i + 2 < n {
                    du2[i] = du[i + 1];
                    du[i + 1] *= -fact;
                }
                piv[i] = true;
            }
        }
        // a singular U (lambda exact): perturb its zero pivots
        let tiny = EPS * norm;
        for x in dd.iter_mut() {
            if x.abs() < tiny {
                *x = if *x < 0.0 { -tiny } else { tiny };
            }
        }
        // deterministic pseudo-random start in (-1, 1)
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        for x in v.iter_mut() {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            *x = ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        }
        for _ in 0..3 {
            // dgtts2: L, then U
            for i in 0..n - 1 {
                if piv[i] {
                    let temp = v[i];
                    v[i] = v[i + 1];
                    v[i + 1] = temp - dl[i] * v[i];
                } else {
                    v[i + 1] -= dl[i] * v[i];
                }
            }
            v[n - 1] /= dd[n - 1];
            v[n - 2] = (v[n - 2] - du[n - 2] * v[n - 1]) / dd[n - 2];
            for i in (0..n - 2).rev() {
                v[i] = (v[i] - du[i] * v[i + 1] - du2[i] * v[i + 2]) / dd[i];
            }
            let scale = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let inv = 1.0 / scale;
            v.iter_mut().for_each(|x| *x *= inv);
        }
        let inv = 1.0 / v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x *= inv);
    }
}

/// Householder reduction of the symmetric `n x n` row-major matrix `a` (both
/// triangles stored) to tridiagonal form `T = Qᵀ A Q`, `Q = H_0 ... H_{n-2}`,
/// `H_k = I - tau_k u_k u_kᵀ` (LAPACK `dsytd2`). On return `d` / `e` hold the
/// diagonal and sub-diagonal of `T` (`e[n-1] = 0`) and row `k` of `a` holds
/// `u_k` in columns `k+1..n` (`u_k[k+1] = 1`, zero before).
///
/// The rank-2 update `B -= u wᵀ + w uᵀ` of each step is deferred and fused
/// into the next step's product `B u`, so the trailing block is swept once
/// per step. `scratch` needs `4 n` values.
fn tridiagonalize(n: usize, a: &mut [f64], tau: &mut [f64], d: &mut [f64], e: &mut [f64], scratch: &mut [f64]) {
    let (up, rest) = scratch.split_at_mut(n);
    let (wp, rest) = rest.split_at_mut(n);
    let (un, rest) = rest.split_at_mut(n);
    let (mut up, mut wp, mut un, mut p) = (up, wp, un, &mut rest[..n]);
    // `up` / `wp`: pending update of the trailing block (absolute indices)
    let mut pending = false;
    for k in 0..n.saturating_sub(1) {
        if pending {
            let (uk, wk) = (up[k], wp[k]);
            rank2_row(&mut a[k * n + k..k * n + n], uk, wk, &wp[k..], &up[k..]);
        }
        d[k] = a[k * n + k];
        // column k below the diagonal == row k right of it (symmetric storage)
        let (head, tail) = a.split_at_mut((k + 1) * n);
        let row = &mut head[k * n + k + 1..k * n + n];
        let alpha = row[0];
        let xnorm = dot(&row[1..], &row[1..]).sqrt();
        if xnorm == 0.0 {
            // H_k = I; the pending update stays pending for the rows below
            tau[k] = 0.0;
            e[k] = alpha;
            row[0] = 1.0;
            row[1..].fill(0.0);
            continue;
        }
        let beta = -alpha.signum() * alpha.hypot(xnorm);
        let t = (beta - alpha) / beta;
        let scale = 1.0 / (alpha - beta);
        row[0] = 1.0;
        row[1..].iter_mut().for_each(|x| *x *= scale);
        tau[k] = t;
        e[k] = beta;
        let u = &*row;
        un[k + 1..].copy_from_slice(u);
        // trailing block B = a[k+1.., k+1..], brought up to date on the fly:
        // p = tau B u
        for i in k + 1..n {
            let b = &mut tail[(i - k - 1) * n + k + 1..(i - k) * n];
            let s = if pending { rank2_row_dot(b, up[i], wp[i], &wp[k + 1..], &up[k + 1..], u) } else { dot(b, u) };
            p[i] = t * s;
        }
        // w = p - (tau / 2)(pᵀu) u: the pending update of the next step
        let kk = -0.5 * t * dot(&p[k + 1..], u);
        for (pi, &ui) in p[k + 1..].iter_mut().zip(u) {
            *pi += kk * ui;
        }
        std::mem::swap(&mut up, &mut un);
        std::mem::swap(&mut wp, &mut p);
        pending = true;
    }
    let last = (n - 1) * n + n - 1;
    if pending {
        a[last] -= 2.0 * up[n - 1] * wp[n - 1];
    }
    d[n - 1] = a[last];
    e[n - 1] = 0.0;
}

/// `sqrt(a² + b²)`; the plain formula (much faster than `hypot`) unless it
/// would overflow or underflow.
#[inline(always)]
fn pythag(a: f64, b: f64) -> f64 {
    let s = a * a + b * b;
    if s.is_finite() && s >= f64::MIN_POSITIVE { s.sqrt() } else { a.hypot(b) }
}

/// `1 / sqrt(x)` for a normal positive `x` to about one ulp: hardware estimate
/// (aarch64 `frsqrte`, 8 bits) or bit trick, refined by Newton steps that use
/// only multiplications and FMAs (fully pipelined, unlike `sqrt` and `/`).
#[inline(always)]
fn rsqrt(x: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    let (mut y, steps) = {
        // SAFETY: NEON is part of the aarch64 baseline.
        (unsafe { std::arch::aarch64::vrsqrted_f64(x) }, 3)
    };
    #[cfg(not(target_arch = "aarch64"))]
    let (mut y, steps) = (f64::from_bits(0x5FE6_EB50_C7B5_37A9 - (x.to_bits() >> 1)), 4);
    for _ in 0..steps {
        let h = fma(-(x * y), y, 1.0);
        y = fma(0.5 * y, h, y);
    }
    y
}

/// `x * y + z`, fused where the hardware has FMA.
#[inline(always)]
fn fma(x: f64, y: f64, z: f64) -> f64 {
    if cfg!(any(target_arch = "aarch64", target_feature = "fma")) { x.mul_add(y, z) } else { x * y + z }
}

/// `row -= ui w + wi u`.
#[inline]
fn rank2_row(row: &mut [f64], ui: f64, wi: f64, w: &[f64], u: &[f64]) {
    let (nui, nwi) = (-ui, -wi);
    for ((x, &wj), &uj) in row.iter_mut().zip(w).zip(u) {
        *x = fma(nui, wj, fma(nwi, uj, *x));
    }
}

/// `row -= ui w + wi u`, returning the updated `row · v`.
#[inline]
fn rank2_row_dot(row: &mut [f64], ui: f64, wi: f64, w: &[f64], u: &[f64], v: &[f64]) -> f64 {
    let (nui, nwi) = (-ui, -wi);
    let len = row.len();
    let (w, u, v) = (&w[..len], &u[..len], &v[..len]);
    let mut acc = [0.0f64; 4];
    let (rc, rr) = row.as_chunks_mut::<4>();
    let c = rc.len() * 4;
    let (wc, uc, vc) = (w.as_chunks::<4>().0, u.as_chunks::<4>().0, v.as_chunks::<4>().0);
    for (((x, wj), uj), vj) in rc.iter_mut().zip(wc).zip(uc).zip(vc) {
        for l in 0..4 {
            let y = fma(nui, wj[l], fma(nwi, uj[l], x[l]));
            x[l] = y;
            acc[l] = fma(y, vj[l], acc[l]);
        }
    }
    let mut s = (acc[0] + acc[2]) + (acc[1] + acc[3]);
    for (((x, &wj), &uj), &vj) in rr.iter_mut().zip(&w[c..]).zip(&u[c..]).zip(&v[c..]) {
        let y = fma(nui, wj, fma(nwi, uj, *x));
        *x = y;
        s = fma(y, vj, s);
    }
    s
}

/// Dot product with four accumulators (vectorizes).
#[inline]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let (xc, xr) = x.as_chunks::<4>();
    let (yc, yr) = y[..x.len()].as_chunks::<4>();
    for (a, b) in xc.iter().zip(yc) {
        for l in 0..4 {
            acc[l] = fma(a[l], b[l], acc[l]);
        }
    }
    let mut s = (acc[0] + acc[2]) + (acc[1] + acc[3]);
    for (a, b) in xr.iter().zip(yr) {
        s += a * b;
    }
    s
}

/// `v <- Q v` with the reflectors stored by [`tridiagonalize`].
fn apply_reflectors(n: usize, a: &[f64], tau: &[f64], v: &mut [f64]) {
    for k in (0..n.saturating_sub(1)).rev() {
        let t = tau[k];
        if t == 0.0 {
            continue;
        }
        let u = &a[k * n + k + 1..k * n + n];
        let x = &mut v[k + 1..];
        let s = t * dot(u, x);
        x.iter_mut().zip(u).for_each(|(xi, &ui)| *xi -= s * ui);
    }
}

/// All eigenvalues (ascending, into `w`) and eigenvectors (column `j` of the
/// row-major `n x n` matrix `z` belongs to `w[j]`) of the symmetric row-major
/// matrix `a`, which is overwritten. Returns `false` on non-finite input or
/// when the QL iteration does not converge.
pub fn symmetric_eigen_full(n: usize, a: &mut [f64], w: &mut Vec<f64>, z: &mut Vec<f64>) -> bool {
    assert_eq!(a.len(), n * n);
    w.clear();
    w.resize(n, 0.0);
    z.clear();
    z.resize(n * n, 0.0);
    if n == 0 {
        return true;
    }
    if a.iter().any(|x| !x.is_finite()) {
        return false;
    }
    let mut tau = vec![0.0; n];
    let mut e = vec![0.0; n];
    let mut p = vec![0.0; 4 * n];
    tridiagonalize(n, a, &mut tau, w, &mut e, &mut p);
    // Q = H_0 ... H_{n-2}: apply to the unit vectors, column by column (stored
    // as rows of `qt`, i.e. Qᵀ)
    let mut qt = vec![0.0; n * n];
    for j in 0..n {
        let col = &mut qt[j * n..(j + 1) * n];
        col[j] = 1.0;
        apply_reflectors(n, a, &tau, col);
    }
    // tql2 with the rotations applied to the columns of Q (rows of qt)
    let d = w;
    let mut f = 0.0f64;
    let mut tst1 = 0.0f64;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n - 1 && e[m].abs() > EPS * tst1 {
            m += 1;
        }
        if m > l {
            let mut iter = 0;
            loop {
                iter += 1;
                if iter > 60 {
                    return false;
                }
                let g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = pythag(p, 1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let h = g - d[l];
                for x in &mut d[l + 2..] {
                    *x -= h;
                }
                f += h;
                p = d[m];
                let (mut c, mut c2, mut c3) = (1.0f64, 1.0f64, 1.0f64);
                let el1 = e[l + 1];
                let (mut s, mut s2) = (0.0f64, 0.0f64);
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c * e[i];
                    let h = c * p;
                    r = pythag(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    let (lo, hi) = qt.split_at_mut((i + 1) * n);
                    let qi = &mut lo[i * n..];
                    let qi1 = &mut hi[..n];
                    for (x, y) in qi.iter_mut().zip(qi1.iter_mut()) {
                        let hh = *y;
                        *y = s * *x + c * hh;
                        *x = c * *x - s * hh;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= EPS * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[i].total_cmp(&d[j]));
    let vals: Vec<f64> = order.iter().map(|&i| d[i]).collect();
    d.copy_from_slice(&vals);
    for (j, &src) in order.iter().enumerate() {
        for i in 0..n {
            z[i * n + j] = qt[src * n + i];
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic symmetric test matrix.
    fn sym(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let x = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
                a[i * n + j] = x;
                a[j * n + i] = x;
            }
        }
        a
    }

    fn residual(n: usize, a: &[f64], lambda: f64, v: &[f64]) -> f64 {
        (0..n).map(|i| ((0..n).map(|j| a[i * n + j] * v[j]).sum::<f64>() - lambda * v[i]).abs()).fold(0.0, f64::max)
    }

    /// Packs the row-major matrices `mats` into `eig` (`lanes` lanes).
    fn pack(eig: &mut LaneEigen, n: usize, lanes: usize, mats: &[Vec<f64>]) {
        let a = eig.packed_mut(n, lanes, mats.len());
        for (l, m) in mats.iter().enumerate() {
            for i in 0..n {
                for j in 0..=i {
                    a[lanes * (tri(i) + j) + l] = m[i * n + j];
                }
            }
        }
    }

    #[test]
    fn full_decomposition_is_accurate() {
        for &n in &[1usize, 2, 3, 7, 30, 65] {
            let a = sym(n, n as u64);
            let mut w = Vec::new();
            let mut z = Vec::new();
            assert!(symmetric_eigen_full(n, &mut a.clone(), &mut w, &mut z));
            assert!(w.windows(2).all(|x| x[0] <= x[1]));
            for j in 0..n {
                let v: Vec<f64> = (0..n).map(|i| z[i * n + j]).collect();
                assert!(residual(n, &a, w[j], &v) < 1e-13 * n as f64, "n={n} j={j}");
            }
            // orthonormal
            for j in 0..n {
                for k in 0..n {
                    let s: f64 = (0..n).map(|i| z[i * n + j] * z[i * n + k]).sum();
                    assert!((s - if j == k { 1.0 } else { 0.0 }).abs() < 1e-13);
                }
            }
        }
    }

    #[test]
    fn lanes_match_full_decomposition() {
        let mut eig = LaneEigen::new();
        for &n in &[1usize, 2, 3, 5, 16, 40, 100] {
            for lanes in [2, 4] {
                let mats: Vec<Vec<f64>> = (0..lanes).map(|l| sym(n, 7 + (n * lanes + l) as u64)).collect();
                pack(&mut eig, n, lanes, &mats);
                solve_lanes(std::slice::from_mut(&mut eig));
                for (l, a) in mats.iter().enumerate() {
                    assert!(eig.is_ok(l));
                    let mut w = Vec::new();
                    let mut z = Vec::new();
                    symmetric_eigen_full(n, &mut a.clone(), &mut w, &mut z);
                    let mut got = eig.values(l).to_vec();
                    got.sort_by(f64::total_cmp);
                    assert_eq!(eig.extremes(l), (got[0], got[n - 1]));
                    for (x, y) in got.iter().zip(&w) {
                        assert!((x - y).abs() < 1e-13 * n as f64, "n={n} lanes={lanes} l={l}");
                    }
                    let v = eig.max_vector(l);
                    assert!(residual(n, a, eig.extremes(l).1, v) < 1e-13 * n as f64);
                    let s: f64 = (0..n).map(|i| v[i] * z[i * n + n - 1]).sum();
                    assert!((s.abs() - 1.0).abs() < 1e-12, "n={n} lanes={lanes} l={l}");
                }
            }
        }
    }

    #[test]
    fn batch_matches_single() {
        // several workspaces of different sizes in one call, a non-finite
        // matrix, unused lanes and an empty matrix
        let sizes = [3usize, 50, 1, 17, 0, 9, 33, 2, 64];
        let mut batch: Vec<LaneEigen> = sizes.iter().map(|_| LaneEigen::new()).collect();
        let mats = |k: usize, n: usize| -> Vec<Vec<f64>> { (0..3).map(|l| sym(n, 100 + 4 * k as u64 + l)).collect() };
        for (k, (eig, &n)) in batch.iter_mut().zip(&sizes).enumerate() {
            let lanes = if k % 2 == 0 { 4 } else { 2 };
            let m = mats(k, n);
            pack(eig, n, lanes, &m[..lanes.min(3)]);
        }
        batch[5].packed_mut(9, 2, 2)[7] = f64::INFINITY; // lane 1 of the 9 x 9 pair
        solve_lanes(&mut batch);
        let mut single = LaneEigen::new();
        for (k, (eig, &n)) in batch.iter().zip(&sizes).enumerate() {
            for l in 0..eig.used() {
                if k == 5 && l == 1 {
                    assert!(!eig.is_ok(l));
                    continue;
                }
                assert!(eig.is_ok(l), "k={k} l={l}");
                let lanes = if k % 2 == 0 { 4 } else { 2 };
                pack(&mut single, n, lanes, &mats(k, n)[..lanes.min(3)]);
                solve_lanes(std::slice::from_mut(&mut single));
                assert!(single.is_ok(l));
                assert_eq!(eig.values(l), single.values(l));
                assert_eq!(eig.max_vector(l), single.max_vector(l));
            }
        }
    }

    #[test]
    fn scalar_workspace_matches_full_decomposition() {
        // alone and together with lane workspaces in one call
        let mut eig = SymEigen::new();
        let mut alone = SymEigen::new();
        let mut lanes = LaneEigen::new();
        for &n in &[1usize, 2, 5, 16, 40, 100] {
            let a = sym(n, 7 + n as u64);
            alone.matrix_mut(n).copy_from_slice(&a);
            assert!(alone.solve());
            eig.matrix_mut(n).copy_from_slice(&a);
            pack(&mut lanes, n, 2, &[sym(n, 1), sym(n, 2)]);
            solve_batch(std::slice::from_mut(&mut eig), std::slice::from_mut(&mut lanes));
            assert!(eig.is_ok() && lanes.is_ok(0) && lanes.is_ok(1));
            assert_eq!(eig.values(), alone.values());
            assert_eq!(eig.max_vector(), alone.max_vector());
            let mut w = Vec::new();
            let mut z = Vec::new();
            symmetric_eigen_full(n, &mut a.clone(), &mut w, &mut z);
            for (x, y) in eig.values().iter().zip(&w) {
                assert!((x - y).abs() < 1e-13 * n as f64);
            }
            let v = eig.max_vector();
            assert!(residual(n, &a, eig.values()[n - 1], v) < 1e-13 * n as f64);
            let s: f64 = (0..n).map(|i| v[i] * z[i * n + n - 1]).sum();
            assert!((s.abs() - 1.0).abs() < 1e-12, "n={n}");
        }
    }

    #[test]
    fn path_graph_spectrum() {
        // P4: eigenvalues 2 cos(k pi / 5)
        let n = 4;
        let mut a = vec![0.0; n * n];
        for i in 0..n - 1 {
            a[i * n + i + 1] = 1.0;
            a[(i + 1) * n + i] = 1.0;
        }
        let mut eig = LaneEigen::new();
        pack(&mut eig, n, 2, &[a]);
        solve_lanes(std::slice::from_mut(&mut eig));
        assert!(eig.is_ok(0) && !eig.is_ok(1));
        let mut got = eig.values(0).to_vec();
        got.sort_by(f64::total_cmp);
        for (k, x) in got.iter().rev().enumerate() {
            let exact = 2.0 * ((k + 1) as f64 * std::f64::consts::PI / 5.0).cos();
            assert!((x - exact).abs() < 4e-15);
        }
        // Perron vector: sin(k pi / 5), normalized
        let v = eig.max_vector(0);
        let sign = v[0].signum();
        let norm = (1..=4).map(|k| ((k as f64) * std::f64::consts::PI / 5.0).sin().powi(2)).sum::<f64>().sqrt();
        for (i, x) in v.iter().enumerate() {
            let exact = ((i + 1) as f64 * std::f64::consts::PI / 5.0).sin() / norm;
            assert!((sign * x - exact).abs() < 1e-15);
        }
    }

    #[test]
    fn rejects_non_finite() {
        let mut eig = LaneEigen::new();
        eig.packed_mut(2, 2, 2).copy_from_slice(&[1.0, 1.0, f64::NAN, 0.5, 1.0, 1.0]);
        solve_lanes(std::slice::from_mut(&mut eig));
        assert!(!eig.is_ok(0) && eig.is_ok(1));
        assert!((eig.extremes(1).1 - 1.5).abs() < 1e-15);
    }
}
