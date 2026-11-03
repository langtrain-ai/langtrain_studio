//! Quantized weight matrices and the int8 matmul kernels.
//!
//! Weights are repacked at load into struct-of-arrays form (aligned int8 or
//! packed nibbles + f32 block scales). Activations are quantized to int8 per
//! 32-element block, so every dot product is an integer dot plus one scale
//! multiply per block, the same scheme as ggml's Q8_0 x Q8_0 path.

use anyhow::{bail, Result};
use half::f16;
use rayon::prelude::*;

use crate::gguf::{DType, TensorInfo};

pub const QK: usize = 32;

#[derive(Debug)]
pub enum Weights {
    /// int8 values, one per element.
    Q8 { qs: Vec<i8> },
    /// 4-bit values (offset by 8), two per byte, Q4_0 nibble order.
    Q4 { qs: Vec<u8> },
}

/// A `rows x cols` weight matrix (ggml: `dims = [cols, rows]`).
#[derive(Debug)]
pub struct QMatrix {
    pub rows: usize,
    pub cols: usize,
    /// One scale per 32-element block, row-major.
    pub scales: Vec<f32>,
    pub w: Weights,
}

fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// Read a tensor as plain f32 values (norms, biases, embeddings).
pub fn to_f32(info: &TensorInfo, data: &[u8]) -> Result<Vec<f32>> {
    let n = info.n_elements();
    let mut out = vec![0f32; n];
    match info.dtype {
        DType::F32 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(4)) {
                *o = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        DType::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f16_at(data, i * 2);
            }
        }
        DType::Q8_0 => {
            for (b, blk) in data.chunks_exact(34).enumerate() {
                let d = f16_at(blk, 0);
                for j in 0..QK {
                    out[b * QK + j] = d * blk[2 + j] as i8 as f32;
                }
            }
        }
        DType::Q4_0 => {
            for (b, blk) in data.chunks_exact(18).enumerate() {
                let d = f16_at(blk, 0);
                for j in 0..16 {
                    out[b * QK + j] = d * ((blk[2 + j] & 0x0f) as i32 - 8) as f32;
                    out[b * QK + j + 16] = d * ((blk[2 + j] >> 4) as i32 - 8) as f32;
                }
            }
        }
        DType::Other(t) => bail!("unsupported tensor type {t}"),
    }
    Ok(out)
}

impl QMatrix {
    pub fn from_gguf(info: &TensorInfo, data: &[u8]) -> Result<Self> {
        if info.dims.len() != 2 {
            bail!("expected a 2-D weight, got dims {:?}", info.dims);
        }
        let (cols, rows) = (info.dims[0], info.dims[1]);
        if cols % QK != 0 {
            bail!("row length {cols} is not a multiple of {QK}");
        }
        let nb = rows * cols / QK;
        Ok(match info.dtype {
            DType::Q8_0 => {
                let mut scales = Vec::with_capacity(nb);
                let mut qs = Vec::with_capacity(rows * cols);
                for blk in data.chunks_exact(34) {
                    scales.push(f16_at(blk, 0));
                    qs.extend(blk[2..].iter().map(|&b| b as i8));
                }
                QMatrix { rows, cols, scales, w: Weights::Q8 { qs } }
            }
            DType::Q4_0 => {
                let mut scales = Vec::with_capacity(nb);
                let mut qs = Vec::with_capacity(rows * cols / 2);
                for blk in data.chunks_exact(18) {
                    scales.push(f16_at(blk, 0));
                    qs.extend_from_slice(&blk[2..]);
                }
                QMatrix { rows, cols, scales, w: Weights::Q4 { qs } }
            }
            // Float weights are requantized to Q8_0 (error ~0.4% per block).
            DType::F32 | DType::F16 => Self::from_f32(&to_f32(info, data)?, rows, cols),
            DType::Other(t) => bail!(
                "tensor type {t} is not supported by the native engine yet (use Q8_0, Q4_0, F16 or F32, or the llama.cpp backend)"
            ),
        })
    }

    pub fn from_f32(v: &[f32], rows: usize, cols: usize) -> Self {
        let mut scales = Vec::with_capacity(v.len() / QK);
        let mut qs = vec![0i8; v.len()];
        for (b, chunk) in v.chunks_exact(QK).enumerate() {
            scales.push(quantize_block(chunk, &mut qs[b * QK..(b + 1) * QK]));
        }
        QMatrix { rows, cols, scales, w: Weights::Q8 { qs } }
    }

    pub fn bytes(&self) -> usize {
        self.scales.len() * 4
            + match &self.w {
                Weights::Q8 { qs } => qs.len(),
                Weights::Q4 { qs } => qs.len(),
            }
    }

    /// Dot product of row `r` with one quantized activation row.
    #[inline]
    fn row_dot(&self, r: usize, x: &QActs, t: usize) -> f32 {
        let nb = self.cols / QK;
        let ws = &self.scales[r * nb..(r + 1) * nb];
        let xs = &x.scales[t * nb..(t + 1) * nb];
        let xq = &x.qs[t * self.cols..(t + 1) * self.cols];
        let mut sum = 0f32;
        match &self.w {
            Weights::Q8 { qs } => {
                let wq = &qs[r * self.cols..(r + 1) * self.cols];
                for b in 0..nb {
                    let s = dot32(&wq[b * QK..(b + 1) * QK], &xq[b * QK..(b + 1) * QK]);
                    sum += ws[b] * xs[b] * s as f32;
                }
            }
            Weights::Q4 { qs } => {
                let wq = &qs[r * self.cols / 2..(r + 1) * self.cols / 2];
                let mut tmp = [0i8; QK];
                for b in 0..nb {
                    let p = &wq[b * 16..(b + 1) * 16];
                    for j in 0..16 {
                        tmp[j] = (p[j] & 0x0f) as i8 - 8;
                        tmp[j + 16] = (p[j] >> 4) as i8 - 8;
                    }
                    let s = dot32(&tmp, &xq[b * QK..(b + 1) * QK]);
                    sum += ws[b] * xs[b] * s as f32;
                }
            }
        }
        sum
    }

    /// `out[t][r] = W[r] . x[t]` for all `t < x.n`. `out` is `[x.n][rows]`.
    pub fn matmul(&self, x: &QActs, out: &mut [f32]) {
        assert_eq!(x.cols, self.cols);
        let n = x.n;
        let rows = self.rows;
        debug_assert_eq!(out.len(), n * rows);
        if n == 1 {
            out.par_chunks_mut(ROW_CHUNK).enumerate().for_each(|(c, o)| {
                for (i, v) in o.iter_mut().enumerate() {
                    *v = self.row_dot(c * ROW_CHUNK + i, x, 0);
                }
            });
            return;
        }
        // Batched: each weight row is read once and reused for every token,
        // which is what makes batched decode memory-bandwidth efficient.
        let mut tmp = vec![0f32; rows * n];
        tmp.par_chunks_mut(ROW_CHUNK * n).enumerate().for_each(|(c, o)| {
            for (i, row_out) in o.chunks_mut(n).enumerate() {
                let r = c * ROW_CHUNK + i;
                for (t, v) in row_out.iter_mut().enumerate() {
                    *v = self.row_dot(r, x, t);
                }
            }
        });
        for r in 0..rows {
            for t in 0..n {
                out[t * rows + r] = tmp[r * n + t];
            }
        }
    }
}

const ROW_CHUNK: usize = 32;

/// Activations quantized to int8 per 32-element block.
#[derive(Debug, Default)]
pub struct QActs {
    pub n: usize,
    pub cols: usize,
    pub qs: Vec<i8>,
    pub scales: Vec<f32>,
}

impl QActs {
    /// Quantize `n` rows of `cols` floats.
    pub fn quantize(&mut self, x: &[f32], n: usize, cols: usize) {
        self.n = n;
        self.cols = cols;
        self.qs.resize(n * cols, 0);
        self.scales.resize(n * cols / QK, 0.0);
        for (b, chunk) in x[..n * cols].chunks_exact(QK).enumerate() {
            self.scales[b] = quantize_block(chunk, &mut self.qs[b * QK..(b + 1) * QK]);
        }
    }
}

fn quantize_block(x: &[f32], q: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
    for (o, v) in q.iter_mut().zip(x) {
        *o = (v * id).round() as i8;
    }
    d
}

/// Integer dot product of two 32-element int8 blocks.
#[inline]
pub fn dot32(a: &[i8], b: &[i8]) -> i32 {
    debug_assert!(a.len() == QK && b.len() == QK);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64; slices are exactly 32 long.
        unsafe { dot32_neon(a.as_ptr(), b.as_ptr()) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if avx2() {
            // SAFETY: AVX2 support checked at runtime.
            return unsafe { dot32_avx2(a.as_ptr(), b.as_ptr()) };
        }
        dot32_scalar(a, b)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    dot32_scalar(a, b)
}

#[inline]
pub fn dot32_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot32_neon(a: *const i8, b: *const i8) -> i32 {
    use std::arch::aarch64::*;
    let a0 = vld1q_s8(a);
    let a1 = vld1q_s8(a.add(16));
    let b0 = vld1q_s8(b);
    let b1 = vld1q_s8(b.add(16));
    // Widening i8 x i8 -> i16 products, pairwise-accumulated into i32 lanes.
    let mut acc = vpaddlq_s16(vmull_s8(vget_low_s8(a0), vget_low_s8(b0)));
    acc = vpadalq_s16(acc, vmull_high_s8(a0, b0));
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(a1), vget_low_s8(b1)));
    acc = vpadalq_s16(acc, vmull_high_s8(a1, b1));
    vaddvq_s32(acc)
}

#[cfg(target_arch = "x86_64")]
fn avx2() -> bool {
    use std::sync::OnceLock;
    static HAS: OnceLock<bool> = OnceLock::new();
    *HAS.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot32_avx2(a: *const i8, b: *const i8) -> i32 {
    use std::arch::x86_64::*;
    let va = _mm256_loadu_si256(a as *const __m256i);
    let vb = _mm256_loadu_si256(b as *const __m256i);
    // maddubs needs one unsigned operand: |a| * (b * sign(a)).
    let ax = _mm256_sign_epi8(va, va);
    let sy = _mm256_sign_epi8(vb, va);
    let p16 = _mm256_maddubs_epi16(ax, sy);
    let p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
    let s = _mm_add_epi32(_mm256_castsi256_si128(p32), _mm256_extracti128_si256(p32, 1));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    #[test]
    fn simd_dot_matches_scalar() {
        let mut s = 1u64;
        for _ in 0..500 {
            let a: Vec<i8> = (0..32).map(|_| (lcg(&mut s) * 255.0) as i8).collect();
            let b: Vec<i8> = (0..32).map(|_| (lcg(&mut s) * 255.0) as i8).collect();
            assert_eq!(dot32(&a, &b), dot32_scalar(&a, &b));
        }
        let max = vec![-128i8; 32];
        assert_eq!(dot32(&max, &max), dot32_scalar(&max, &max));
    }

    #[test]
    fn matmul_matches_float_reference() {
        let (rows, cols, n) = (70, 128, 5);
        let mut s = 7u64;
        let w: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut s)).collect();
        let x: Vec<f32> = (0..n * cols).map(|_| lcg(&mut s)).collect();
        let m = QMatrix::from_f32(&w, rows, cols);
        let mut qa = QActs::default();
        qa.quantize(&x, n, cols);
        let mut out = vec![0f32; n * rows];
        m.matmul(&qa, &mut out);
        for t in 0..n {
            for r in 0..rows {
                let want: f32 = (0..cols).map(|c| w[r * cols + c] * x[t * cols + c]).sum();
                let got = out[t * rows + r];
                assert!((got - want).abs() < 0.05, "t{t} r{r}: {got} vs {want}");
            }
        }
        // Batched and single-token paths agree exactly.
        let mut one = QActs::default();
        one.quantize(&x[cols..2 * cols], 1, cols);
        let mut o1 = vec![0f32; rows];
        m.matmul(&one, &mut o1);
        assert_eq!(&o1[..], &out[rows..2 * rows]);
    }
}
