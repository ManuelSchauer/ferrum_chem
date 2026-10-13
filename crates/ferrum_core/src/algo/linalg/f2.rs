//! Two-lane `f64` vector used to run two independent eigenvalue problems of
//! the same size in lock step (one problem per lane). NEON on aarch64, plain
//! arrays elsewhere. Every lane operation is the IEEE operation of the scalar
//! code (FMA where the target has it).

#[cfg(target_arch = "aarch64")]
mod imp {
    use std::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct F2(float64x2_t);

    #[allow(unused_unsafe)]
    impl F2 {
        #[inline(always)]
        pub fn splat(v: f64) -> Self {
            F2(unsafe { vdupq_n_f64(v) })
        }
        #[inline(always)]
        pub fn new(a: f64, b: f64) -> Self {
            F2(unsafe { vsetq_lane_f64::<1>(b, vdupq_n_f64(a)) })
        }
        #[inline(always)]
        pub fn lane0(self) -> f64 {
            unsafe { vgetq_lane_f64::<0>(self.0) }
        }
        #[inline(always)]
        pub fn lane1(self) -> f64 {
            unsafe { vgetq_lane_f64::<1>(self.0) }
        }
        /// Loads `s[0..2]`.
        #[inline(always)]
        pub fn load(s: &[f64]) -> Self {
            assert!(s.len() >= 2);
            // SAFETY: two elements in bounds (checked above)
            F2(unsafe { vld1q_f64(s.as_ptr()) })
        }
        /// Stores into `s[0..2]`.
        #[inline(always)]
        pub fn store(self, s: &mut [f64]) {
            assert!(s.len() >= 2);
            // SAFETY: two elements in bounds (checked above)
            unsafe { vst1q_f64(s.as_mut_ptr(), self.0) }
        }
        /// # Safety
        /// `p` must be valid for reading two `f64`.
        #[inline(always)]
        pub unsafe fn load_ptr(p: *const f64) -> Self {
            F2(unsafe { vld1q_f64(p) })
        }
        /// # Safety
        /// `p` must be valid for writing two `f64`.
        #[inline(always)]
        pub unsafe fn store_ptr(self, p: *mut f64) {
            unsafe { vst1q_f64(p, self.0) }
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            F2(unsafe { vaddq_f64(self.0, o.0) })
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            F2(unsafe { vsubq_f64(self.0, o.0) })
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            F2(unsafe { vmulq_f64(self.0, o.0) })
        }
        #[inline(always)]
        pub fn div(self, o: Self) -> Self {
            F2(unsafe { vdivq_f64(self.0, o.0) })
        }
        /// `self * b + c` (fused).
        #[inline(always)]
        pub fn mul_add(self, b: Self, c: Self) -> Self {
            F2(unsafe { vfmaq_f64(c.0, self.0, b.0) })
        }
        /// `c - self * b` (fused).
        #[inline(always)]
        pub fn neg_mul_add(self, b: Self, c: Self) -> Self {
            F2(unsafe { vfmsq_f64(c.0, self.0, b.0) })
        }
        #[inline(always)]
        pub fn abs(self) -> Self {
            F2(unsafe { vabsq_f64(self.0) })
        }
        #[inline(always)]
        pub fn neg(self) -> Self {
            F2(unsafe { vnegq_f64(self.0) })
        }
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            F2(unsafe { vsqrtq_f64(self.0) })
        }
        /// Per lane `if self == 0 { a } else { b }`.
        #[inline(always)]
        pub fn select_eq_zero(self, a: Self, b: Self) -> Self {
            F2(unsafe { vbslq_f64(vceqzq_f64(self.0), a.0, b.0) })
        }
        /// Per lane `if self >= 0 { a } else { b }`.
        #[inline(always)]
        pub fn select_ge_zero(self, a: Self, b: Self) -> Self {
            F2(unsafe { vbslq_f64(vcgezq_f64(self.0), a.0, b.0) })
        }
        /// Per lane `if |self| < lim { r } else { self }`.
        #[inline(always)]
        pub fn replace_small(self, lim: Self, r: Self) -> Self {
            F2(unsafe { vbslq_f64(vcaltq_f64(self.0, lim.0), r.0, self.0) })
        }
        /// Per lane `1.0` if `self < 0` else `0.0`.
        #[inline(always)]
        pub fn neg_indicator(self) -> Self {
            F2(unsafe {
                vreinterpretq_f64_u64(vandq_u64(
                    vcltzq_f64(self.0),
                    vreinterpretq_u64_f64(vdupq_n_f64(1.0)),
                ))
            })
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod imp {
    #[derive(Clone, Copy)]
    pub struct F2([f64; 2]);

    #[inline(always)]
    fn fma(a: f64, b: f64, c: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            a.mul_add(b, c)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            a * b + c
        }
    }

    impl F2 {
        #[inline(always)]
        fn map(self, f: impl Fn(f64) -> f64) -> Self {
            F2([f(self.0[0]), f(self.0[1])])
        }
        #[inline(always)]
        fn zip(self, o: Self, f: impl Fn(f64, f64) -> f64) -> Self {
            F2([f(self.0[0], o.0[0]), f(self.0[1], o.0[1])])
        }
        #[inline(always)]
        pub fn splat(v: f64) -> Self {
            F2([v, v])
        }
        #[inline(always)]
        pub fn new(a: f64, b: f64) -> Self {
            F2([a, b])
        }
        #[inline(always)]
        pub fn lane0(self) -> f64 {
            self.0[0]
        }
        #[inline(always)]
        pub fn lane1(self) -> f64 {
            self.0[1]
        }
        #[inline(always)]
        pub fn load(s: &[f64]) -> Self {
            F2([s[0], s[1]])
        }
        #[inline(always)]
        pub fn store(self, s: &mut [f64]) {
            s[0] = self.0[0];
            s[1] = self.0[1];
        }
        /// # Safety
        /// `p` must be valid for reading two `f64`.
        #[inline(always)]
        pub unsafe fn load_ptr(p: *const f64) -> Self {
            unsafe { F2([*p, *p.add(1)]) }
        }
        /// # Safety
        /// `p` must be valid for writing two `f64`.
        #[inline(always)]
        pub unsafe fn store_ptr(self, p: *mut f64) {
            unsafe {
                *p = self.0[0];
                *p.add(1) = self.0[1];
            }
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            self.zip(o, |a, b| a + b)
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            self.zip(o, |a, b| a - b)
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            self.zip(o, |a, b| a * b)
        }
        #[inline(always)]
        pub fn div(self, o: Self) -> Self {
            self.zip(o, |a, b| a / b)
        }
        #[inline(always)]
        pub fn mul_add(self, b: Self, c: Self) -> Self {
            F2([
                fma(self.0[0], b.0[0], c.0[0]),
                fma(self.0[1], b.0[1], c.0[1]),
            ])
        }
        #[inline(always)]
        pub fn neg_mul_add(self, b: Self, c: Self) -> Self {
            F2([
                fma(-self.0[0], b.0[0], c.0[0]),
                fma(-self.0[1], b.0[1], c.0[1]),
            ])
        }
        #[inline(always)]
        pub fn abs(self) -> Self {
            self.map(f64::abs)
        }
        #[inline(always)]
        pub fn neg(self) -> Self {
            self.map(|v| -v)
        }
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            self.map(f64::sqrt)
        }
        #[inline(always)]
        pub fn select_eq_zero(self, a: Self, b: Self) -> Self {
            F2([
                if self.0[0] == 0.0 { a.0[0] } else { b.0[0] },
                if self.0[1] == 0.0 { a.0[1] } else { b.0[1] },
            ])
        }
        #[inline(always)]
        pub fn select_ge_zero(self, a: Self, b: Self) -> Self {
            F2([
                if self.0[0] >= 0.0 { a.0[0] } else { b.0[0] },
                if self.0[1] >= 0.0 { a.0[1] } else { b.0[1] },
            ])
        }
        #[inline(always)]
        pub fn replace_small(self, lim: Self, r: Self) -> Self {
            F2([
                if self.0[0].abs() < lim.0[0] {
                    r.0[0]
                } else {
                    self.0[0]
                },
                if self.0[1].abs() < lim.0[1] {
                    r.0[1]
                } else {
                    self.0[1]
                },
            ])
        }
        #[inline(always)]
        pub fn neg_indicator(self) -> Self {
            self.map(|v| if v < 0.0 { 1.0 } else { 0.0 })
        }
    }
}

pub use imp::F2;
