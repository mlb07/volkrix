#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn dot_u8_i8(input: &[u8], weights: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_u8_i8_avx2(input, weights) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // All network activation paths clamp to 0..=127, allowing the signed
        // ARM dot-product instruction to represent the u8 input exactly.
        debug_assert!(input.iter().all(|&value| value <= i8::MAX as u8));
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return unsafe { dot_u8_i8_dotprod(input, weights) };
        }
    }
    dot_u8_i8_scalar(input, weights)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_u8_i8_dotprod(input: &[u8], weights: &[i8]) -> i32 {
    let n = input.len().min(weights.len());
    let mut accumulators = [vdupq_n_s32(0); 4];
    let mut i = 0;
    while i + 64 <= n {
        for (lane, accumulator) in accumulators.iter_mut().enumerate() {
            let offset = i + lane * 16;
            let values = vreinterpretq_s8_u8(vld1q_u8(input.as_ptr().add(offset)));
            let row = vld1q_s8(weights.as_ptr().add(offset));
            std::arch::asm!(
                "sdot {acc:v}.4s, {values:v}.16b, {row:v}.16b",
                acc = inout(vreg) *accumulator,
                values = in(vreg) values,
                row = in(vreg) row,
                options(pure, nomem, nostack),
            );
        }
        i += 64;
    }
    let combined = vaddq_s32(
        vaddq_s32(accumulators[0], accumulators[1]),
        vaddq_s32(accumulators[2], accumulators[3]),
    );
    let mut combined = combined;
    // The first layer is normally wide enough for the four-way loop above,
    // while the later SFNN layers are only 16--32 inputs wide.  Keep those
    // layers on DotProd too instead of dropping their entire calculation to
    // the scalar tail.
    while i + 16 <= n {
        let values = vreinterpretq_s8_u8(vld1q_u8(input.as_ptr().add(i)));
        let row = vld1q_s8(weights.as_ptr().add(i));
        std::arch::asm!(
            "sdot {acc:v}.4s, {values:v}.16b, {row:v}.16b",
            acc = inout(vreg) combined,
            values = in(vreg) values,
            row = in(vreg) row,
            options(pure, nomem, nostack),
        );
        i += 16;
    }
    let mut sum = vaddvq_s32(combined);
    while i < n {
        sum += input[i] as i32 * weights[i] as i32;
        i += 1;
    }
    sum
}

fn dot_u8_i8_scalar(input: &[u8], weights: &[i8]) -> i32 {
    let mut sum = 0i32;
    for (a, w) in input.iter().zip(weights) {
        sum += *a as i32 * *w as i32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_u8_i8_avx2(input: &[u8], weights: &[i8]) -> i32 {
    let n = input.len();
    let ones = _mm256_set1_epi16(1);
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();
    let inp = input.as_ptr();
    let wgt = weights.as_ptr();

    let mut i = 0;
    while i + 128 <= n {
        macro_rules! block {
            ($acc:ident, $off:expr) => {{
                let a = _mm256_loadu_si256(inp.add(i + $off) as *const __m256i);
                let w = _mm256_loadu_si256(wgt.add(i + $off) as *const __m256i);
                let wide = _mm256_madd_epi16(_mm256_maddubs_epi16(a, w), ones);
                $acc = _mm256_add_epi32($acc, wide);
            }};
        }
        block!(acc0, 0);
        block!(acc1, 32);
        block!(acc2, 64);
        block!(acc3, 96);
        i += 128;
    }
    while i + 32 <= n {
        let a = _mm256_loadu_si256(inp.add(i) as *const __m256i);
        let w = _mm256_loadu_si256(wgt.add(i) as *const __m256i);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(a, w), ones));
        i += 32;
    }

    let acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256(acc, 1);
    let mut s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    let mut sum = _mm_cvtsi128_si32(s);
    while i < n {
        sum += input[i] as i32 * weights[i] as i32;
        i += 1;
    }
    sum
}

/// Computes `out[j] = (clamp(a[j],0,hi) * clamp(b[j],0,hi)) >> shift` as `u8`,
/// the pairwise feature-transformer activation used by SFNNv5 networks.
pub fn pairwise_clip_mul(a: &[i16], b: &[i16], out: &mut [u8], hi: i16, shift: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { pairwise_clip_mul_avx2(a, b, out, hi, shift) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { pairwise_clip_mul_neon(a, b, out, hi, shift) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    pairwise_clip_mul_scalar(a, b, out, hi, shift)
}

#[cfg(target_arch = "aarch64")]
unsafe fn pairwise_clip_mul_neon(a: &[i16], b: &[i16], out: &mut [u8], hi: i16, shift: i32) {
    let n = out.len().min(a.len()).min(b.len());
    let zero = vdupq_n_s16(0);
    let upper = vdupq_n_s16(hi);
    let shifts = vdupq_n_s16(-(shift as i16));
    let mut i = 0;
    while i + 16 <= n {
        let a0 = vminq_s16(vmaxq_s16(vld1q_s16(a.as_ptr().add(i)), zero), upper);
        let a1 = vminq_s16(vmaxq_s16(vld1q_s16(a.as_ptr().add(i + 8)), zero), upper);
        let b0 = vminq_s16(vmaxq_s16(vld1q_s16(b.as_ptr().add(i)), zero), upper);
        let b1 = vminq_s16(vmaxq_s16(vld1q_s16(b.as_ptr().add(i + 8)), zero), upper);
        let product0 = vshlq_u16(
            vmulq_u16(vreinterpretq_u16_s16(a0), vreinterpretq_u16_s16(b0)),
            shifts,
        );
        let product1 = vshlq_u16(
            vmulq_u16(vreinterpretq_u16_s16(a1), vreinterpretq_u16_s16(b1)),
            shifts,
        );
        vst1q_u8(
            out.as_mut_ptr().add(i),
            vcombine_u8(vqmovn_u16(product0), vqmovn_u16(product1)),
        );
        i += 16;
    }
    pairwise_clip_mul_scalar(&a[i..n], &b[i..n], &mut out[i..n], hi, shift);
}

fn pairwise_clip_mul_scalar(a: &[i16], b: &[i16], out: &mut [u8], hi: i16, shift: i32) {
    let hi = hi as i32;
    for j in 0..out.len() {
        let s0 = (a[j] as i32).clamp(0, hi);
        let s1 = (b[j] as i32).clamp(0, hi);
        out[j] = ((s0 * s1) as u32 >> shift) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pairwise_clip_mul_avx2(a: &[i16], b: &[i16], out: &mut [u8], hi: i16, shift: i32) {
    let m = out.len();
    let lo = _mm256_setzero_si256();
    let hiv = _mm256_set1_epi16(hi);
    let cnt = _mm_cvtsi32_si128(shift);
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let op = out.as_mut_ptr();

    let clip_mul = |off: usize| -> __m256i {
        let mut x = _mm256_loadu_si256(ap.add(off) as *const __m256i);
        let mut y = _mm256_loadu_si256(bp.add(off) as *const __m256i);
        x = _mm256_min_epi16(_mm256_max_epi16(x, lo), hiv);
        y = _mm256_min_epi16(_mm256_max_epi16(y, lo), hiv);
        _mm256_srl_epi16(_mm256_mullo_epi16(x, y), cnt)
    };

    let mut j = 0;
    while j + 32 <= m {
        let r0 = clip_mul(j);
        let r1 = clip_mul(j + 16);
        let packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(r0, r1), 0xD8);
        _mm256_storeu_si256(op.add(j) as *mut __m256i, packed);
        j += 32;
    }
    let hi = hi as i32;
    while j < m {
        let s0 = (a[j] as i32).clamp(0, hi);
        let s1 = (b[j] as i32).clamp(0, hi);
        out[j] = ((s0 * s1) as u32 >> shift) as u8;
        j += 1;
    }
}

/// Computes `out[j] = clamp(a[j], 0, 127)` as `u8`, the clipped-ReLU
/// feature-transformer activation used by HalfKP and HalfKAv2 networks.
pub fn clip_u8(a: &[i16], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { clip_u8_avx2(a, out) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { clip_u8_neon(a, out) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (o, &x) in out.iter_mut().zip(a) {
            *o = (x as i32).clamp(0, 127) as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn clip_u8_neon(a: &[i16], out: &mut [u8]) {
    let n = out.len().min(a.len());
    let zero = vdupq_n_s16(0);
    let upper = vdupq_n_s16(127);
    let mut i = 0;
    while i + 16 <= n {
        let lo = vminq_s16(vmaxq_s16(vld1q_s16(a.as_ptr().add(i)), zero), upper);
        let hi = vminq_s16(vmaxq_s16(vld1q_s16(a.as_ptr().add(i + 8)), zero), upper);
        vst1q_u8(
            out.as_mut_ptr().add(i),
            vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi)),
        );
        i += 16;
    }
    for (output, &value) in out[i..n].iter_mut().zip(&a[i..n]) {
        *output = (value as i32).clamp(0, 127) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn clip_u8_avx2(a: &[i16], out: &mut [u8]) {
    let m = out.len();
    let lo = _mm256_setzero_si256();
    let hiv = _mm256_set1_epi16(127);
    let ap = a.as_ptr();
    let op = out.as_mut_ptr();

    let clip = |off: usize| -> __m256i {
        let x = _mm256_loadu_si256(ap.add(off) as *const __m256i);
        _mm256_min_epi16(_mm256_max_epi16(x, lo), hiv)
    };

    let mut j = 0;
    while j + 32 <= m {
        let packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(clip(j), clip(j + 16)), 0xD8);
        _mm256_storeu_si256(op.add(j) as *mut __m256i, packed);
        j += 32;
    }
    while j < m {
        *out.get_unchecked_mut(j) = (*a.get_unchecked(j) as i32).clamp(0, 127) as u8;
        j += 1;
    }
}

pub fn add_i8_i16(acc: &mut [i16], w: &[i8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { add_i8_i16_avx2(acc, w) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { add_sub_i8_i16_neon(acc, w, true) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (a, &wi) in acc.iter_mut().zip(w) {
            *a = a.wrapping_add(wi as i16);
        }
    }
}

pub fn sub_i8_i16(acc: &mut [i16], w: &[i8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { sub_i8_i16_avx2(acc, w) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { add_sub_i8_i16_neon(acc, w, false) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (a, &wi) in acc.iter_mut().zip(w) {
            *a = a.wrapping_sub(wi as i16);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn add_sub_i8_i16_neon(acc: &mut [i16], w: &[i8], add: bool) {
    let n = acc.len().min(w.len());
    let mut i = 0;
    while i + 16 <= n {
        let weights = vld1q_s8(w.as_ptr().add(i));
        let weight_lo = vmovl_s8(vget_low_s8(weights));
        let weight_hi = vmovl_high_s8(weights);
        let acc_lo = vld1q_s16(acc.as_ptr().add(i));
        let acc_hi = vld1q_s16(acc.as_ptr().add(i + 8));
        let result_lo = if add {
            vaddq_s16(acc_lo, weight_lo)
        } else {
            vsubq_s16(acc_lo, weight_lo)
        };
        let result_hi = if add {
            vaddq_s16(acc_hi, weight_hi)
        } else {
            vsubq_s16(acc_hi, weight_hi)
        };
        vst1q_s16(acc.as_mut_ptr().add(i), result_lo);
        vst1q_s16(acc.as_mut_ptr().add(i + 8), result_hi);
        i += 16;
    }
    while i < n {
        acc[i] = if add {
            acc[i].wrapping_add(w[i] as i16)
        } else {
            acc[i].wrapping_sub(w[i] as i16)
        };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_i8_i16_avx2(acc: &mut [i16], w: &[i8]) {
    let n = acc.len();
    let ap = acc.as_mut_ptr();
    let wp = w.as_ptr();
    let mut i = 0;
    while i + 32 <= n {
        let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
        let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16) as *const __m128i));
        let a0 = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(ap.add(i + 16) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_add_epi16(a0, w0));
        _mm256_storeu_si256(ap.add(i + 16) as *mut __m256i, _mm256_add_epi16(a1, w1));
        i += 32;
    }
    while i < n {
        acc[i] = acc[i].wrapping_add(w[i] as i16);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sub_i8_i16_avx2(acc: &mut [i16], w: &[i8]) {
    let n = acc.len();
    let ap = acc.as_mut_ptr();
    let wp = w.as_ptr();
    let mut i = 0;
    while i + 32 <= n {
        let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
        let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16) as *const __m128i));
        let a0 = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(ap.add(i + 16) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_sub_epi16(a0, w0));
        _mm256_storeu_si256(ap.add(i + 16) as *mut __m256i, _mm256_sub_epi16(a1, w1));
        i += 32;
    }
    while i < n {
        acc[i] = acc[i].wrapping_sub(w[i] as i16);
        i += 1;
    }
}

pub fn add_i16(acc: &mut [i16], w: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { add_i16_avx2(acc, w) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { add_sub_i16_neon(acc, w, true) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (a, &wi) in acc.iter_mut().zip(w) {
            *a = a.wrapping_add(wi);
        }
    }
}

pub fn sub_i16(acc: &mut [i16], w: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { sub_i16_avx2(acc, w) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { add_sub_i16_neon(acc, w, false) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (a, &wi) in acc.iter_mut().zip(w) {
            *a = a.wrapping_sub(wi);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn add_sub_i16_neon(acc: &mut [i16], w: &[i16], add: bool) {
    let n = acc.len().min(w.len());
    let mut i = 0;
    while i + 16 <= n {
        for offset in [0, 8] {
            let left = vld1q_s16(acc.as_ptr().add(i + offset));
            let right = vld1q_s16(w.as_ptr().add(i + offset));
            let result = if add {
                vaddq_s16(left, right)
            } else {
                vsubq_s16(left, right)
            };
            vst1q_s16(acc.as_mut_ptr().add(i + offset), result);
        }
        i += 16;
    }
    while i < n {
        acc[i] = if add {
            acc[i].wrapping_add(w[i])
        } else {
            acc[i].wrapping_sub(w[i])
        };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_i16_avx2(acc: &mut [i16], w: &[i16]) {
    let n = acc.len();
    let ap = acc.as_mut_ptr();
    let wp = w.as_ptr();
    let mut i = 0;
    while i + 32 <= n {
        let a0 = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(ap.add(i + 16) as *const __m256i);
        let b0 = _mm256_loadu_si256(wp.add(i) as *const __m256i);
        let b1 = _mm256_loadu_si256(wp.add(i + 16) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_add_epi16(a0, b0));
        _mm256_storeu_si256(ap.add(i + 16) as *mut __m256i, _mm256_add_epi16(a1, b1));
        i += 32;
    }
    while i + 16 <= n {
        let a = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let b = _mm256_loadu_si256(wp.add(i) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_add_epi16(a, b));
        i += 16;
    }
    while i < n {
        acc[i] = acc[i].wrapping_add(w[i]);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sub_i16_avx2(acc: &mut [i16], w: &[i16]) {
    let n = acc.len();
    let ap = acc.as_mut_ptr();
    let wp = w.as_ptr();
    let mut i = 0;
    while i + 32 <= n {
        let a0 = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(ap.add(i + 16) as *const __m256i);
        let b0 = _mm256_loadu_si256(wp.add(i) as *const __m256i);
        let b1 = _mm256_loadu_si256(wp.add(i + 16) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_sub_epi16(a0, b0));
        _mm256_storeu_si256(ap.add(i + 16) as *mut __m256i, _mm256_sub_epi16(a1, b1));
        i += 32;
    }
    while i + 16 <= n {
        let a = _mm256_loadu_si256(ap.add(i) as *const __m256i);
        let b = _mm256_loadu_si256(wp.add(i) as *const __m256i);
        _mm256_storeu_si256(ap.add(i) as *mut __m256i, _mm256_sub_epi16(a, b));
        i += 16;
    }
    while i < n {
        acc[i] = acc[i].wrapping_sub(w[i]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn optimized_kernels_are_bit_exact_with_scalar_reference() {
        let mut seed = 0x8f31_2d69_aa55_1047;
        for len in [1, 7, 15, 16, 31, 32, 63, 64, 127, 128, 1024] {
            let input = (0..len)
                .map(|_| (next(&mut seed) & 0x7f) as u8)
                .collect::<Vec<_>>();
            let weights8 = (0..len).map(|_| next(&mut seed) as i8).collect::<Vec<_>>();
            assert_eq!(
                dot_u8_i8(&input, &weights8),
                dot_u8_i8_scalar(&input, &weights8)
            );

            let left = (0..len).map(|_| next(&mut seed) as i16).collect::<Vec<_>>();
            let right = (0..len).map(|_| next(&mut seed) as i16).collect::<Vec<_>>();
            let mut expected_activation = vec![0; len];
            let mut actual_activation = vec![0; len];
            pairwise_clip_mul_scalar(&left, &right, &mut expected_activation, 255, 9);
            pairwise_clip_mul(&left, &right, &mut actual_activation, 255, 9);
            assert_eq!(actual_activation, expected_activation);

            let mut expected_clip = left
                .iter()
                .map(|&value| (value as i32).clamp(0, 127) as u8)
                .collect::<Vec<_>>();
            let mut actual_clip = vec![0; len];
            clip_u8(&left, &mut actual_clip);
            assert_eq!(actual_clip, expected_clip);
            expected_clip.clear();

            let initial = (0..len).map(|_| next(&mut seed) as i16).collect::<Vec<_>>();
            let mut expected = initial.clone();
            let mut actual = initial.clone();
            for (lane, &weight) in expected.iter_mut().zip(&weights8) {
                *lane = lane.wrapping_add(weight as i16);
            }
            add_i8_i16(&mut actual, &weights8);
            assert_eq!(actual, expected);
            sub_i8_i16(&mut actual, &weights8);
            assert_eq!(actual, initial);

            let weights16 = (0..len).map(|_| next(&mut seed) as i16).collect::<Vec<_>>();
            expected.clone_from(&initial);
            actual.clone_from(&initial);
            for (lane, &weight) in expected.iter_mut().zip(&weights16) {
                *lane = lane.wrapping_add(weight);
            }
            add_i16(&mut actual, &weights16);
            assert_eq!(actual, expected);
            sub_i16(&mut actual, &weights16);
            assert_eq!(actual, initial);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "manual Apple-Silicon DotProd microprofile"]
    fn aarch64_small_dotprod_profile_report() {
        use std::{hint::black_box, time::Instant};

        let mut seed = 0x51a7_0f93_6d2c_b841;
        for len in [16usize, 30, 32] {
            let input = (0..len)
                .map(|_| (next(&mut seed) & 0x7f) as u8)
                .collect::<Vec<_>>();
            let weights = (0..len).map(|_| next(&mut seed) as i8).collect::<Vec<_>>();
            let iterations = 2_000_000u64;

            let scalar_started = Instant::now();
            let mut scalar_checksum = 0i64;
            for _ in 0..iterations {
                scalar_checksum = scalar_checksum.wrapping_add(i64::from(dot_u8_i8_scalar(
                    black_box(&input),
                    black_box(&weights),
                )));
            }
            let scalar_ns = scalar_started.elapsed().as_nanos();

            let dotprod_started = Instant::now();
            let mut dotprod_checksum = 0i64;
            for _ in 0..iterations {
                dotprod_checksum = dotprod_checksum.wrapping_add(i64::from(dot_u8_i8(
                    black_box(&input),
                    black_box(&weights),
                )));
            }
            let dotprod_ns = dotprod_started.elapsed().as_nanos();

            assert_eq!(dotprod_checksum, scalar_checksum);
            println!(
                "aarch64_small_dotprod len {len} iterations {iterations} scalar_ns {scalar_ns} dotprod_ns {dotprod_ns} speedup_x {:.3} checksum {dotprod_checksum}",
                scalar_ns as f64 / dotprod_ns as f64,
            );
        }
    }
}
