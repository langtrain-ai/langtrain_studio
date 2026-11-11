//! Quantized weight matrices and the int8 matmul kernels.
//!
//! Weights are repacked at load into struct-of-arrays form with f32 scales.
//! Activations are quantized to int8 per 32-element block, so every dot
//! product is an integer dot plus a few scale multiplies per block, the same
//! scheme as ggml's `vec_dot` paths.
//!
//! Supported GGUF types: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and the
//! K-quants Q4_K, Q5_K and Q6_K. Together these cover the popular `Q4_K_M`,
//! `Q5_K_M`, `Q6_K` and `Q8_0` files, including models whose row length is
//! not a multiple of 256, where llama.cpp substitutes Q5_0/Q8_0 blocks. K-quants keep their bit width in memory; their per-block minimum
//! folds into the int8 dot as `d * sum(q*x) - m * sum(x)`.

use anyhow::{bail, Result};
use half::f16;

use crate::gguf::{DType, TensorInfo};

pub const QK: usize = 32;
const QK_K: usize = 256;

#[derive(Debug)]
pub enum Weights {
    /// int8 values, one per element; one scale per 32.
    Q8 { qs: Vec<i8> },
    /// Signed 4-bit (value - 8), two per byte: byte i holds values i and i+16.
    Q4 { qs: Vec<u8> },
    /// Unsigned 4-bit with a per-block minimum (from Q4_K).
    Q4M { qs: Vec<u8>, mins: Vec<f32> },
    /// Unsigned 5-bit with a per-block minimum (from Q5_K): 4-bit nibbles
    /// plus one high-bit plane (4 bytes per 32 values).
    Q5M {
        qs: Vec<u8>,
        qh: Vec<u8>,
        mins: Vec<f32>,
    },
    /// Signed 6-bit stored as int8, one scale per 16 values (from Q6_K).
    Q6 { qs: Vec<i8> },
}

/// A `rows x cols` weight matrix (ggml: `dims = [cols, rows]`).
#[derive(Debug)]
pub struct QMatrix {
    pub rows: usize,
    pub cols: usize,
    /// Block scales, row-major: one per 32 values (one per 16 for `Q6`).
    pub scales: Vec<f32>,
    pub w: Weights,
}

fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

// ---------- K-quant superblock decoders (layouts from ggml-quants.c) ----------

/// The 6-bit scale and min of sub-block `j` from a K-quant `scales[12]` field.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Q4_K (144 bytes): value = scale[s] * q - min[s] for 8 sub-blocks of 32.
fn decode_q4k(b: &[u8], q: &mut [u8; QK_K], sc: &mut [f32; 8], mn: &mut [f32; 8]) {
    let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
    let scales = &b[4..16];
    let qs = &b[16..144];
    for g in 0..4 {
        for half in 0..2 {
            let s = 2 * g + half;
            let (a, m) = scale_min_k4(s, scales);
            sc[s] = d * a as f32;
            mn[s] = dmin * m as f32;
            for l in 0..32 {
                let byte = qs[g * 32 + l];
                q[s * 32 + l] = if half == 0 { byte & 0x0f } else { byte >> 4 };
            }
        }
    }
}

/// Q5_K (176 bytes): like Q4_K with a fifth bit from `qh`.
fn decode_q5k(b: &[u8], q: &mut [u8; QK_K], sc: &mut [f32; 8], mn: &mut [f32; 8]) {
    let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
    let scales = &b[4..16];
    let qh = &b[16..48];
    let qs = &b[48..176];
    for g in 0..4 {
        for half in 0..2 {
            let s = 2 * g + half;
            let (a, m) = scale_min_k4(s, scales);
            sc[s] = d * a as f32;
            mn[s] = dmin * m as f32;
            let bit = 1u8 << s; // u1 = 1 << 2g, u2 = 2 << 2g
            for l in 0..32 {
                let byte = qs[g * 32 + l];
                let lo = if half == 0 { byte & 0x0f } else { byte >> 4 };
                q[s * 32 + l] = lo + if qh[l] & bit != 0 { 16 } else { 0 };
            }
        }
    }
}

/// Q6_K (210 bytes): value = scale[s] * q for 16 sub-blocks of 16, q in [-32, 31].
fn decode_q6k(b: &[u8], q: &mut [i8; QK_K], sc: &mut [f32; 16]) {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let scales = &b[192..208];
    let d = f16_at(b, 208);
    for (s, v) in sc.iter_mut().enumerate() {
        *v = d * scales[s] as i8 as f32;
    }
    for n in 0..2 {
        let (ql, qh) = (&ql[n * 64..], &qh[n * 32..]);
        for l in 0..32 {
            let base = n * 128;
            q[base + l] = ((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) as i8 - 32;
            q[base + l + 32] = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
            q[base + l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
            q[base + l + 96] = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
        }
    }
}

/// Q4_1 / Q5_0 / Q5_1 block -> (scale, min, q) with value = scale * q - min.
fn decode_legacy(dtype: DType, b: &[u8], q: &mut [u8; QK]) -> (f32, f32) {
    let (d, min, qh, qs) = match dtype {
        DType::Q4_1 => (f16_at(b, 0), -f16_at(b, 2), None, &b[4..20]),
        DType::Q5_0 => {
            let d = f16_at(b, 0);
            (
                d,
                16.0 * d,
                Some(u32::from_le_bytes(b[2..6].try_into().unwrap())),
                &b[6..22],
            )
        }
        DType::Q5_1 => (
            f16_at(b, 0),
            -f16_at(b, 2),
            Some(u32::from_le_bytes(b[4..8].try_into().unwrap())),
            &b[8..24],
        ),
        _ => unreachable!("not a legacy min type"),
    };
    for j in 0..16 {
        let (mut lo, mut hi) = (qs[j] & 0x0f, qs[j] >> 4);
        if let Some(h) = qh {
            lo |= (((h >> j) & 1) << 4) as u8;
            hi |= (((h >> (j + 16)) & 1) << 4) as u8;
        }
        q[j] = lo;
        q[j + 16] = hi;
    }
    (d, min)
}

fn legacy_size(dtype: DType) -> usize {
    match dtype {
        DType::Q4_1 => 20,
        DType::Q5_0 => 22,
        DType::Q5_1 => 24,
        _ => unreachable!(),
    }
}

/// Read a tensor as plain f32 values (norms, biases, embeddings, LoRA).
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
        DType::Q4_1 | DType::Q5_0 | DType::Q5_1 => {
            let mut q = [0u8; QK];
            for (b, blk) in data.chunks_exact(legacy_size(info.dtype)).enumerate() {
                let (d, m) = decode_legacy(info.dtype, blk, &mut q);
                for j in 0..QK {
                    out[b * QK + j] = d * q[j] as f32 - m;
                }
            }
        }
        DType::Q4K | DType::Q5K => {
            let (size, q5) = if info.dtype == DType::Q4K {
                (144, false)
            } else {
                (176, true)
            };
            let (mut q, mut sc, mut mn) = ([0u8; QK_K], [0f32; 8], [0f32; 8]);
            for (b, blk) in data.chunks_exact(size).enumerate() {
                if q5 {
                    decode_q5k(blk, &mut q, &mut sc, &mut mn)
                } else {
                    decode_q4k(blk, &mut q, &mut sc, &mut mn)
                }
                for i in 0..QK_K {
                    out[b * QK_K + i] = sc[i / 32] * q[i] as f32 - mn[i / 32];
                }
            }
        }
        DType::Q6K => {
            let (mut q, mut sc) = ([0i8; QK_K], [0f32; 16]);
            for (b, blk) in data.chunks_exact(210).enumerate() {
                decode_q6k(blk, &mut q, &mut sc);
                for i in 0..QK_K {
                    out[b * QK_K + i] = sc[i / 16] * q[i] as f32;
                }
            }
        }
        DType::Other(t) => bail!("unsupported tensor type {t}"),
    }
    Ok(out)
}

/// Pack 32 values (each < 16) into 16 bytes: byte i = v[i] | v[i+16] << 4.
fn pack_nibbles(v: &[u8], out: &mut Vec<u8>) {
    for i in 0..16 {
        out.push((v[i] & 0x0f) | ((v[i + 16] & 0x0f) << 4));
    }
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
        let n = rows * cols;
        let nb = n / QK;
        Ok(match info.dtype {
            DType::Q8_0 => {
                let mut scales = Vec::with_capacity(nb);
                let mut qs = Vec::with_capacity(n);
                for blk in data.chunks_exact(34) {
                    scales.push(f16_at(blk, 0));
                    qs.extend(blk[2..].iter().map(|&b| b as i8));
                }
                QMatrix { rows, cols, scales, w: Weights::Q8 { qs } }
            }
            DType::Q4_0 => {
                let mut scales = Vec::with_capacity(nb);
                let mut qs = Vec::with_capacity(n / 2);
                for blk in data.chunks_exact(18) {
                    scales.push(f16_at(blk, 0));
                    qs.extend_from_slice(&blk[2..]);
                }
                QMatrix { rows, cols, scales, w: Weights::Q4 { qs } }
            }
            DType::Q4_1 | DType::Q5_0 | DType::Q5_1 => {
                let q5 = info.dtype != DType::Q4_1;
                let mut scales = Vec::with_capacity(nb);
                let mut mins = Vec::with_capacity(nb);
                let mut qs = Vec::with_capacity(n / 2);
                let mut qh = Vec::with_capacity(if q5 { n / 8 } else { 0 });
                let mut q = [0u8; QK];
                for blk in data.chunks_exact(legacy_size(info.dtype)) {
                    let (d, m) = decode_legacy(info.dtype, blk, &mut q);
                    scales.push(d);
                    mins.push(m);
                    pack_nibbles(&q, &mut qs);
                    if q5 {
                        let bits: u32 = q.iter().enumerate().map(|(i, &x)| u32::from(x >> 4) << i).sum();
                        qh.extend_from_slice(&bits.to_le_bytes());
                    }
                }
                let w = if q5 { Weights::Q5M { qs, qh, mins } } else { Weights::Q4M { qs, mins } };
                QMatrix { rows, cols, scales, w }
            }
            DType::Q4K | DType::Q5K => {
                let q5 = info.dtype == DType::Q5K;
                let mut scales = Vec::with_capacity(nb);
                let mut mins = Vec::with_capacity(nb);
                let mut qs = Vec::with_capacity(n / 2);
                let mut qh = Vec::with_capacity(if q5 { n / 8 } else { 0 });
                let (mut q, mut sc, mut mn) = ([0u8; QK_K], [0f32; 8], [0f32; 8]);
                for blk in data.chunks_exact(if q5 { 176 } else { 144 }) {
                    if q5 { decode_q5k(blk, &mut q, &mut sc, &mut mn) } else { decode_q4k(blk, &mut q, &mut sc, &mut mn) }
                    for s in 0..8 {
                        let v = &q[s * 32..(s + 1) * 32];
                        scales.push(sc[s]);
                        mins.push(mn[s]);
                        pack_nibbles(v, &mut qs);
                        if q5 {
                            let bits: u32 = v.iter().enumerate().map(|(i, &x)| u32::from(x >> 4) << i).sum();
                            qh.extend_from_slice(&bits.to_le_bytes());
                        }
                    }
                }
                let w = if q5 { Weights::Q5M { qs, qh, mins } } else { Weights::Q4M { qs, mins } };
                QMatrix { rows, cols, scales, w }
            }
            DType::Q6K => {
                let mut scales = Vec::with_capacity(n / 16);
                let mut qs = Vec::with_capacity(n);
                let (mut q, mut sc) = ([0i8; QK_K], [0f32; 16]);
                for blk in data.chunks_exact(210) {
                    decode_q6k(blk, &mut q, &mut sc);
                    scales.extend_from_slice(&sc);
                    qs.extend_from_slice(&q);
                }
                QMatrix { rows, cols, scales, w: Weights::Q6 { qs } }
            }
            // Float weights are requantized to Q8_0 (error ~0.4% per block).
            DType::F32 | DType::F16 => Self::from_f32(&to_f32(info, data)?, rows, cols),
            DType::Other(t) => bail!(
                "tensor type {t} is not supported by the native engine yet (supported: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, F16, F32)"
            ),
        })
    }

    pub fn from_f32(v: &[f32], rows: usize, cols: usize) -> Self {
        let mut scales = Vec::with_capacity(v.len() / QK);
        let mut qs = vec![0i8; v.len()];
        for (b, chunk) in v.chunks_exact(QK).enumerate() {
            scales.push(quantize_block(chunk, &mut qs[b * QK..(b + 1) * QK]));
        }
        QMatrix {
            rows,
            cols,
            scales,
            w: Weights::Q8 { qs },
        }
    }

    pub fn bytes(&self) -> usize {
        self.scales.len() * 4
            + match &self.w {
                Weights::Q8 { qs } | Weights::Q6 { qs } => qs.len(),
                Weights::Q4 { qs } => qs.len(),
                Weights::Q4M { qs, mins } => qs.len() + mins.len() * 4,
                Weights::Q5M { qs, qh, mins } => qs.len() + qh.len() + mins.len() * 4,
            }
    }

    /// Unpack the 32 integer values of block `b` (absolute index) into `out`.
    #[inline]
    fn unpack_block(&self, b: usize, out: &mut [i8; QK]) {
        match &self.w {
            Weights::Q8 { qs } | Weights::Q6 { qs } => {
                for (o, &v) in out.iter_mut().zip(&qs[b * QK..(b + 1) * QK]) {
                    *o = v;
                }
            }
            Weights::Q4 { qs } => {
                let p = &qs[b * 16..(b + 1) * 16];
                for j in 0..16 {
                    out[j] = (p[j] & 0x0f) as i8 - 8;
                    out[j + 16] = (p[j] >> 4) as i8 - 8;
                }
            }
            Weights::Q4M { qs, .. } => {
                let p = &qs[b * 16..(b + 1) * 16];
                for j in 0..16 {
                    out[j] = (p[j] & 0x0f) as i8;
                    out[j + 16] = (p[j] >> 4) as i8;
                }
            }
            Weights::Q5M { qs, qh, .. } => {
                let p = &qs[b * 16..(b + 1) * 16];
                let h = u32::from_le_bytes(qh[b * 4..b * 4 + 4].try_into().unwrap());
                for j in 0..16 {
                    out[j] = ((p[j] & 0x0f) | (((h >> j) & 1) << 4) as u8) as i8;
                    out[j + 16] = ((p[j] >> 4) | (((h >> (j + 16)) & 1) << 4) as u8) as i8;
                }
            }
        }
    }

    /// Dot product of row `r` with activation row `t`.
    #[inline]
    fn row_dot(&self, r: usize, x: &QActs, t: usize) -> f32 {
        // One kernel for single-token and batched rows, so a request's output
        // never depends on how many other requests share the batch.
        #[cfg(target_arch = "aarch64")]
        if dotprod() {
            let mut v = [0f32; 1];
            // SAFETY: dotprod detected; `r` and `t` are in range.
            unsafe { neon::row_dots(self, r, x, t, 1, &mut v) };
            return v[0];
        }
        let nb = self.cols / QK;
        let xs = &x.scales[t * nb..(t + 1) * nb];
        let xq = &x.qs[t * self.cols..(t + 1) * self.cols];
        let row0 = r * nb;
        let mut sum = 0f32;
        match &self.w {
            Weights::Q8 { qs } => {
                let wq = &qs[r * self.cols..(r + 1) * self.cols];
                let ws = &self.scales[row0..row0 + nb];
                for b in 0..nb {
                    let s = dot32(&wq[b * QK..(b + 1) * QK], &xq[b * QK..(b + 1) * QK]);
                    sum += ws[b] * xs[b] * s as f32;
                }
            }
            Weights::Q6 { qs } => {
                let wq = &qs[r * self.cols..(r + 1) * self.cols];
                let ws = &self.scales[2 * row0..2 * (row0 + nb)];
                for b in 0..nb {
                    let (w, a) = (&wq[b * QK..(b + 1) * QK], &xq[b * QK..(b + 1) * QK]);
                    let lo = dot16(&w[..16], &a[..16]) as f32;
                    let hi = dot16(&w[16..], &a[16..]) as f32;
                    sum += xs[b] * (ws[2 * b] * lo + ws[2 * b + 1] * hi);
                }
            }
            Weights::Q4M { mins, .. } | Weights::Q5M { mins, .. } => {
                let ws = &self.scales[row0..row0 + nb];
                let ms = &mins[row0..row0 + nb];
                let xsum = &x.sums[t * nb..(t + 1) * nb];
                let mut tmp = [0i8; QK];
                for b in 0..nb {
                    self.unpack_block(row0 + b, &mut tmp);
                    let s = dot32(&tmp, &xq[b * QK..(b + 1) * QK]);
                    sum += xs[b] * (ws[b] * s as f32 - ms[b] * xsum[b] as f32);
                }
            }
            Weights::Q4 { .. } => {
                let ws = &self.scales[row0..row0 + nb];
                let mut tmp = [0i8; QK];
                for b in 0..nb {
                    self.unpack_block(row0 + b, &mut tmp);
                    let s = dot32(&tmp, &xq[b * QK..(b + 1) * QK]);
                    sum += ws[b] * xs[b] * s as f32;
                }
            }
        }
        sum
    }

    /// Dequantize row `r` (used for embedding lookups).
    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        let nb = self.cols / QK;
        let mut tmp = [0i8; QK];
        for b in 0..nb {
            let bi = r * nb + b;
            self.unpack_block(bi, &mut tmp);
            let o = &mut out[b * QK..(b + 1) * QK];
            match &self.w {
                Weights::Q6 { .. } => {
                    for (j, v) in o.iter_mut().enumerate() {
                        *v = self.scales[2 * bi + j / 16] * tmp[j] as f32;
                    }
                }
                Weights::Q4M { mins, .. } | Weights::Q5M { mins, .. } => {
                    for (v, &q) in o.iter_mut().zip(&tmp) {
                        *v = self.scales[bi] * q as f32 - mins[bi];
                    }
                }
                Weights::Q8 { .. } | Weights::Q4 { .. } => {
                    for (v, &q) in o.iter_mut().zip(&tmp) {
                        *v = self.scales[bi] * q as f32;
                    }
                }
            }
        }
    }

    /// `out[t][r] = W[r] . x[t]` for all `t < x.n`. `out` is `[x.n][rows]`.
    pub fn matmul(&self, x: &QActs, out: &mut [f32]) {
        assert_eq!(x.cols, self.cols);
        let n = x.n;
        let rows = self.rows;
        debug_assert_eq!(out.len(), n * rows);
        if n == 1 {
            let chunk = row_chunk(rows);
            crate::pool::for_chunks(out, chunk, |c, o| {
                for (i, v) in o.iter_mut().enumerate() {
                    *v = self.row_dot(c * chunk + i, x, 0);
                }
            });
            return;
        }
        // Batched: each weight row is read once and reused for every token,
        // which is what makes batched decode memory-bandwidth efficient.
        let mut tmp = vec![0f32; rows * n];
        crate::pool::for_chunks(&mut tmp, ROW_CHUNK * n, |c, o| {
            for (i, row_out) in o.chunks_mut(n).enumerate() {
                let r = c * ROW_CHUNK + i;
                #[cfg(target_arch = "aarch64")]
                if dotprod() {
                    // Unpack each weight block once and dot it with 4 tokens.
                    for t0 in (0..n).step_by(4) {
                        let cnt = (n - t0).min(4);
                        // SAFETY: dotprod detected; `r`, `t0..t0 + cnt` are in range.
                        unsafe { neon::row_dots(self, r, x, t0, cnt, &mut row_out[t0..t0 + cnt]) };
                    }
                    continue;
                }
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

#[cfg(target_arch = "aarch64")]
mod neon {
    //! Row kernels for aarch64 with the dotprod extension. One generic body
    //! is specialized per weight format (`Load`) and per token count, so the
    //! format check sits outside the hot loop and every token count sums in
    //! the same order: single-token and batched results are bit-identical.

    use std::arch::aarch64::*;

    use super::{QActs, QMatrix, Weights, QK};

    /// One 32-value block: (low 16, high 16, scale per half, minimum).
    pub(super) trait Load {
        const HAS_MIN: bool;
        /// The two halves have different scales (Q6); otherwise they are
        /// summed in one `sdot` chain with a single scale multiply.
        const SPLIT: bool = false;
        unsafe fn load(&self, bi: usize) -> (int8x16_t, int8x16_t, f32, f32, f32);
    }

    pub(super) struct Q8<'a>(pub &'a [i8], pub &'a [f32]);
    pub(super) struct Q6<'a>(pub &'a [i8], pub &'a [f32]);
    pub(super) struct Q4<'a>(pub &'a [u8], pub &'a [f32]);
    pub(super) struct Q4M<'a>(pub &'a [u8], pub &'a [f32], pub &'a [f32]);
    pub(super) struct Q5M<'a>(pub &'a [u8], pub &'a [u8], pub &'a [f32], pub &'a [f32]);

    impl Load for Q8<'_> {
        const HAS_MIN: bool = false;
        #[inline(always)]
        unsafe fn load(&self, bi: usize) -> (int8x16_t, int8x16_t, f32, f32, f32) {
            let p = self.0.as_ptr().add(bi * QK);
            let s = *self.1.get_unchecked(bi);
            (vld1q_s8(p), vld1q_s8(p.add(16)), s, s, 0.0)
        }
    }

    impl Load for Q6<'_> {
        const HAS_MIN: bool = false;
        const SPLIT: bool = true;
        #[inline(always)]
        unsafe fn load(&self, bi: usize) -> (int8x16_t, int8x16_t, f32, f32, f32) {
            let p = self.0.as_ptr().add(bi * QK);
            (vld1q_s8(p), vld1q_s8(p.add(16)), *self.1.get_unchecked(2 * bi), *self.1.get_unchecked(2 * bi + 1), 0.0)
        }
    }

    impl Load for Q4<'_> {
        const HAS_MIN: bool = false;
        #[inline(always)]
        unsafe fn load(&self, bi: usize) -> (int8x16_t, int8x16_t, f32, f32, f32) {
            let p = vld1q_u8(self.0.as_ptr().add(bi * 16));
            let eight = vdupq_n_s8(8);
            let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p, vdupq_n_u8(0x0f))), eight);
            let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p)), eight);
            let s = *self.1.get_unchecked(bi);
            (lo, hi, s, s, 0.0)
        }
    }

    impl Load for Q4M<'_> {
        const HAS_MIN: bool = true;
        #[inline(always)]
        unsafe fn load(&self, bi: usize) -> (int8x16_t, int8x16_t, f32, f32, f32) {
            let p = vld1q_u8(self.0.as_ptr().add(bi * 16));
            let lo = vreinterpretq_s8_u8(vandq_u8(p, vdupq_n_u8(0x0f)));
            let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(p));
            let s = *self.1.get_unchecked(bi);
            (lo, hi, s, s, *self.2.get_unchecked(bi))
        }
    }

    impl Load for Q5M<'_> {
        const HAS_MIN: bool = true;
        #[inline(always)]
        unsafe fn load(&self, bi: usize) -> (int8x16_t, int8x16_t, f32, f32, f32) {
            // Lane j takes bit j of the 32-bit high-bit word.
            const IDX_LO: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
            const IDX_HI: [u8; 16] = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3];
            const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let p = vld1q_u8(self.0.as_ptr().add(bi * 16));
            let word = std::ptr::read_unaligned(self.1.as_ptr().add(bi * 4) as *const u32);
            let h = vreinterpretq_u8_u32(vdupq_n_u32(word));
            let (bits, b16) = (vld1q_u8(BITS.as_ptr()), vdupq_n_u8(0x10));
            let lo = vorrq_u8(vandq_u8(p, vdupq_n_u8(0x0f)), vandq_u8(vtstq_u8(vqtbl1q_u8(h, vld1q_u8(IDX_LO.as_ptr())), bits), b16));
            let hi = vorrq_u8(vshrq_n_u8::<4>(p), vandq_u8(vtstq_u8(vqtbl1q_u8(h, vld1q_u8(IDX_HI.as_ptr())), bits), b16));
            let s = *self.2.get_unchecked(bi);
            (vreinterpretq_s8_u8(lo), vreinterpretq_s8_u8(hi), s, s, *self.3.get_unchecked(bi))
        }
    }

    /// Blocks `row0..row0 + nb` against tokens `t0..t0 + N`. Even and odd
    /// blocks use separate accumulators for instruction-level parallelism;
    /// the loop is unrolled by two so both stay in registers.
    #[target_feature(enable = "dotprod")]
    unsafe fn kernel<L: Load, const N: usize>(l: &L, row0: usize, nb: usize, x: &QActs, t0: usize) -> [f32; N] {
        let mut even = [vdupq_n_f32(0.0); N];
        let mut odd = [vdupq_n_f32(0.0); N];
        let mut min_acc = [0f32; N];
        let cols = nb * QK;
        let xq = x.qs.as_ptr().add(t0 * cols);
        let xs = x.scales.as_ptr().add(t0 * nb);
        let xsum = x.sums.as_ptr().add(t0 * nb);

        #[inline(always)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn block<L: Load, const N: usize>(
            l: &L,
            bi: usize,
            b: usize,
            nb: usize,
            cols: usize,
            xq: *const i8,
            xs: *const f32,
            xsum: *const i32,
            acc: &mut [float32x4_t; N],
            min_acc: &mut [f32; N],
        ) {
            let zero = vdupq_n_s32(0);
            let (w0, w1, s0, s1, mn) = l.load(bi);
            for j in 0..N {
                let xb = xq.add(j * cols + b * QK);
                let xsb = *xs.add(j * nb + b);
                let (x0, x1) = (vld1q_s8(xb), vld1q_s8(xb.add(16)));
                if L::SPLIT {
                    acc[j] = vmlaq_n_f32(acc[j], vcvtq_f32_s32(vdotq_s32(zero, w0, x0)), s0 * xsb);
                    acc[j] = vmlaq_n_f32(acc[j], vcvtq_f32_s32(vdotq_s32(zero, w1, x1)), s1 * xsb);
                } else {
                    let d = vdotq_s32(vdotq_s32(zero, w0, x0), w1, x1);
                    acc[j] = vmlaq_n_f32(acc[j], vcvtq_f32_s32(d), s0 * xsb);
                }
                if L::HAS_MIN {
                    min_acc[j] += mn * xsb * *xsum.add(j * nb + b) as f32;
                }
            }
        }

        let mut b = 0;
        while b + 1 < nb {
            block::<L, N>(l, row0 + b, b, nb, cols, xq, xs, xsum, &mut even, &mut min_acc);
            block::<L, N>(l, row0 + b + 1, b + 1, nb, cols, xq, xs, xsum, &mut odd, &mut min_acc);
            b += 2;
        }
        if b < nb {
            block::<L, N>(l, row0 + b, b, nb, cols, xq, xs, xsum, &mut even, &mut min_acc);
        }
        let mut out = [0f32; N];
        for j in 0..N {
            out[j] = vaddvq_f32(vaddq_f32(even[j], odd[j])) - min_acc[j];
        }
        out
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dispatch<L: Load>(l: &L, row0: usize, nb: usize, x: &QActs, t0: usize, cnt: usize, out: &mut [f32]) {
        match cnt {
            1 => out[..1].copy_from_slice(&kernel::<L, 1>(l, row0, nb, x, t0)),
            2 => out[..2].copy_from_slice(&kernel::<L, 2>(l, row0, nb, x, t0)),
            3 => out[..3].copy_from_slice(&kernel::<L, 3>(l, row0, nb, x, t0)),
            _ => out[..4].copy_from_slice(&kernel::<L, 4>(l, row0, nb, x, t0)),
        }
    }

    /// Row `r` against tokens `t0..t0 + cnt` (cnt <= 4), results into `out`.
    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn row_dots(m: &QMatrix, r: usize, x: &QActs, t0: usize, cnt: usize, out: &mut [f32]) {
        let nb = m.cols / QK;
        let row0 = r * nb;
        match &m.w {
            Weights::Q8 { qs } => dispatch(&Q8(qs, &m.scales), row0, nb, x, t0, cnt, out),
            Weights::Q6 { qs } => dispatch(&Q6(qs, &m.scales), row0, nb, x, t0, cnt, out),
            Weights::Q4 { qs } => dispatch(&Q4(qs, &m.scales), row0, nb, x, t0, cnt, out),
            Weights::Q4M { qs, mins } => dispatch(&Q4M(qs, &m.scales, mins), row0, nb, x, t0, cnt, out),
            Weights::Q5M { qs, qh, mins } => dispatch(&Q5M(qs, qh, &m.scales, mins), row0, nb, x, t0, cnt, out),
        }
    }
}

/// Rows per task: about 8 tasks per thread, at least 32 rows each.
fn row_chunk(rows: usize) -> usize {
    let threads = crate::pool::global().threads();
    (rows / (threads * 8)).max(ROW_CHUNK)
}

/// Several single-token matmuls on the same input in one parallel dispatch
/// (Q/K/V, or gate/up), so the pool synchronizes once instead of per matrix.
pub fn matmul_many(x: &QActs, jobs: &mut [(&QMatrix, &mut [f32])]) {
    if x.n != 1 {
        for (m, out) in jobs.iter_mut() {
            m.matmul(x, out);
        }
        return;
    }
    // Task list: (job, first row, last row).
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (j, (m, out)) in jobs.iter().enumerate() {
        assert_eq!(x.cols, m.cols);
        assert_eq!(out.len(), m.rows);
        let chunk = row_chunk(m.rows);
        tasks.extend(
            (0..m.rows)
                .step_by(chunk)
                .map(|r| (j, r, (r + chunk).min(m.rows))),
        );
    }
    let outs: Vec<(usize, usize)> = jobs
        .iter_mut()
        .map(|(_, o)| (o.as_mut_ptr() as usize, o.len()))
        .collect();
    let mats: Vec<&QMatrix> = jobs.iter().map(|(m, _)| *m).collect();
    crate::pool::global().run(tasks.len(), &|t| {
        let (j, r0, r1) = tasks[t];
        let (ptr, len) = outs[j];
        // SAFETY: each task writes a disjoint row range of one output buffer,
        // and the buffers outlive `run`.
        let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, len) };
        for (r, v) in out.iter_mut().enumerate().take(r1).skip(r0) {
            *v = mats[j].row_dot(r, x, 0);
        }
    });
}

/// Activations quantized to int8 per 32-element block.
#[derive(Debug, Default)]
pub struct QActs {
    pub n: usize,
    pub cols: usize,
    pub qs: Vec<i8>,
    pub scales: Vec<f32>,
    /// Sum of the int8 values in each block (for K-quant minimums).
    pub sums: Vec<i32>,
}

impl QActs {
    /// Quantize `n` rows of `cols` floats.
    pub fn quantize(&mut self, x: &[f32], n: usize, cols: usize) {
        self.n = n;
        self.cols = cols;
        self.qs.resize(n * cols, 0);
        self.scales.resize(n * cols / QK, 0.0);
        self.sums.resize(n * cols / QK, 0);
        for (b, chunk) in x[..n * cols].chunks_exact(QK).enumerate() {
            let q = &mut self.qs[b * QK..(b + 1) * QK];
            self.scales[b] = quantize_block(chunk, q);
            self.sums[b] = q.iter().map(|&v| v as i32).sum();
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
        if dotprod() {
            // SAFETY: the dotprod extension was detected at runtime.
            return unsafe { dot32_sdot(a.as_ptr(), b.as_ptr()) };
        }
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

/// Integer dot product of two 16-element int8 slices.
#[inline]
pub fn dot16(a: &[i8], b: &[i8]) -> i32 {
    debug_assert!(a.len() == 16 && b.len() == 16);
    #[cfg(target_arch = "aarch64")]
    {
        if dotprod() {
            // SAFETY: the dotprod extension was detected at runtime.
            return unsafe { dot16_sdot(a.as_ptr(), b.as_ptr()) };
        }
        // SAFETY: NEON is mandatory on aarch64; slices are exactly 16 long.
        unsafe {
            use std::arch::aarch64::*;
            let (va, vb) = (vld1q_s8(a.as_ptr()), vld1q_s8(b.as_ptr()));
            let acc = vpaddlq_s16(vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
            vaddvq_s32(vpadalq_s16(acc, vmull_high_s8(va, vb)))
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    dot32_scalar(a, b)
}

#[inline]
pub fn dot32_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

#[cfg(target_arch = "aarch64")]
fn dotprod() -> bool {
    use std::sync::OnceLock;
    static HAS: OnceLock<bool> = OnceLock::new();
    *HAS.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
}

/// `sdot`: four int8 products summed into each i32 lane in one instruction
/// (ARMv8.2+, every Apple Silicon and most current ARM servers).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot32_sdot(a: *const i8, b: *const i8) -> i32 {
    use std::arch::aarch64::*;
    let acc = vdotq_s32(vdupq_n_s32(0), vld1q_s8(a), vld1q_s8(b));
    vaddvq_s32(vdotq_s32(acc, vld1q_s8(a.add(16)), vld1q_s8(b.add(16))))
}




#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot16_sdot(a: *const i8, b: *const i8) -> i32 {
    use std::arch::aarch64::*;
    vaddvq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(a), vld1q_s8(b)))
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
    // Widen to i16 and use madd (i16*i16 pairs summed into i32): exact for
    // every input. The usual sign_epi8 + maddubs trick is wrong for -128,
    // which can't be negated in 8 bits (-128 * -128 came out as -16384).
    let a_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(a as *const __m128i));
    let a_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.add(16) as *const __m128i));
    let b_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128(b as *const __m128i));
    let b_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.add(16) as *const __m128i));
    let p32 = _mm256_add_epi32(_mm256_madd_epi16(a_lo, b_lo), _mm256_madd_epi16(a_hi, b_hi));
    let s = _mm_add_epi32(
        _mm256_castsi256_si128(p32),
        _mm256_extracti128_si256(p32, 1),
    );
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    fn random_bytes(n: usize, seed: &mut u64) -> Vec<u8> {
        (0..n).map(|_| ((lcg(seed) + 0.5) * 256.0) as u8).collect()
    }

    /// Random K-quant blocks with small, finite f16 scales.
    fn random_blocks(dtype: DType, n_blocks: usize, seed: &mut u64) -> Vec<u8> {
        let (size, scale_at): (usize, &[usize]) = match dtype {
            DType::Q4_1 => (20, &[0, 2]),
            DType::Q5_0 => (22, &[0]),
            DType::Q5_1 => (24, &[0, 2]),
            DType::Q4K => (144, &[0, 2]),
            DType::Q5K => (176, &[0, 2]),
            DType::Q6K => (210, &[208]),
            _ => unreachable!(),
        };
        let mut data = random_bytes(size * n_blocks, seed);
        for blk in data.chunks_exact_mut(size) {
            for &at in scale_at {
                let v = f16::from_f32(0.001 + (lcg(seed) + 0.5) * 0.01).to_le_bytes();
                blk[at..at + 2].copy_from_slice(&v);
            }
        }
        data
    }

    #[test]
    fn simd_dot_matches_scalar() {
        let mut s = 1u64;
        for _ in 0..500 {
            let a: Vec<i8> = (0..32).map(|_| (lcg(&mut s) * 255.0) as i8).collect();
            let b: Vec<i8> = (0..32).map(|_| (lcg(&mut s) * 255.0) as i8).collect();
            assert_eq!(dot32(&a, &b), dot32_scalar(&a, &b));
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is mandatory on aarch64; this checks the fallback path too.
            assert_eq!(
                unsafe { dot32_neon(a.as_ptr(), b.as_ptr()) },
                dot32_scalar(&a, &b)
            );
            assert_eq!(dot16(&a[..16], &b[..16]), dot32_scalar(&a[..16], &b[..16]));
        }
        // Edge values: -128 can't be negated in 8 bits, which broke the old
        // AVX2 sign trick. Random inputs above never produce it.
        let edges = [i8::MIN, -127, -1, 0, 1, i8::MAX];
        for &x in &edges {
            for &y in &edges {
                let (a, b) = (vec![x; 32], vec![y; 32]);
                assert_eq!(dot32(&a, &b), dot32_scalar(&a, &b), "dot32 of {x} and {y}");
                assert_eq!(dot16(&a[..16], &b[..16]), dot32_scalar(&a[..16], &b[..16]), "dot16 of {x} and {y}");
            }
        }
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

    /// Repacked kernels must agree with straight dequantization.
    #[test]
    fn repacked_types_match_dequantized_reference() {
        let (rows, cols, n) = (6, 512, 3);
        for dtype in [
            DType::Q4K,
            DType::Q5K,
            DType::Q6K,
            DType::Q4_1,
            DType::Q5_0,
            DType::Q5_1,
        ] {
            let mut s = 11u64;
            let n_blocks = rows * cols
                / if matches!(dtype, DType::Q4K | DType::Q5K | DType::Q6K) {
                    QK_K
                } else {
                    QK
                };
            let data = random_blocks(dtype, n_blocks, &mut s);
            let info = TensorInfo {
                dims: vec![cols, rows],
                dtype,
                offset: 0,
            };
            let dense = to_f32(&info, &data).unwrap();
            let m = QMatrix::from_gguf(&info, &data).unwrap();

            let mut row = vec![0f32; cols];
            for r in 0..rows {
                m.dequant_row(r, &mut row);
                for c in 0..cols {
                    assert!(
                        (row[c] - dense[r * cols + c]).abs() < 1e-5,
                        "{dtype:?} r{r} c{c}"
                    );
                }
            }

            let x: Vec<f32> = (0..n * cols).map(|_| lcg(&mut s)).collect();
            let mut qa = QActs::default();
            qa.quantize(&x, n, cols);
            let mut out = vec![0f32; n * rows];
            m.matmul(&qa, &mut out);
            // Reference uses the same int8 activations, so only float rounding remains.
            let xd: Vec<f32> = (0..n * cols)
                .map(|i| qa.scales[i / QK] * qa.qs[i] as f32)
                .collect();
            for t in 0..n {
                for r in 0..rows {
                    let want: f32 = (0..cols)
                        .map(|c| dense[r * cols + c] * xd[t * cols + c])
                        .sum();
                    let got = out[t * rows + r];
                    let tol = 1e-4
                        * (0..cols)
                            .map(|c| (dense[r * cols + c] * xd[t * cols + c]).abs())
                            .sum::<f32>()
                            .max(1.0);
                    assert!(
                        (got - want).abs() < tol,
                        "{dtype:?} t{t} r{r}: {got} vs {want}"
                    );
                }
            }
        }
    }

    /// The 4-token batched kernel must agree with the single-token path.
    #[test]
    fn batched_matches_single_token_for_every_type() {
        let (rows, cols, n) = (40, 512, 7);
        for dtype in [DType::Q4K, DType::Q5K, DType::Q6K, DType::Q4_1, DType::Q5_0, DType::Q5_1] {
            let mut s = 3u64;
            let n_blocks = rows * cols / if matches!(dtype, DType::Q4K | DType::Q5K | DType::Q6K) { QK_K } else { QK };
            let data = random_blocks(dtype, n_blocks, &mut s);
            let m = QMatrix::from_gguf(&TensorInfo { dims: vec![cols, rows], dtype, offset: 0 }, &data).unwrap();
            check_batched(&m, n, &mut s);
        }
        let mut s = 5u64;
        let w: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut s)).collect();
        check_batched(&QMatrix::from_f32(&w, rows, cols), n, &mut s);
        let q4: Vec<u8> = random_bytes(rows * cols / QK * 18, &mut s);
        let mut q4 = q4;
        for blk in q4.chunks_exact_mut(18) {
            blk[..2].copy_from_slice(&f16::from_f32(0.01).to_le_bytes());
        }
        check_batched(&QMatrix::from_gguf(&TensorInfo { dims: vec![cols, rows], dtype: DType::Q4_0, offset: 0 }, &q4).unwrap(), n, &mut s);
    }

    fn check_batched(m: &QMatrix, n: usize, s: &mut u64) {
        let x: Vec<f32> = (0..n * m.cols).map(|_| lcg(s)).collect();
        let mut qa = QActs::default();
        qa.quantize(&x, n, m.cols);
        let mut batched = vec![0f32; n * m.rows];
        m.matmul(&qa, &mut batched);
        for t in 0..n {
            let mut one = QActs::default();
            one.quantize(&x[t * m.cols..(t + 1) * m.cols], 1, m.cols);
            let mut single = vec![0f32; m.rows];
            m.matmul(&one, &mut single);
            for r in 0..m.rows {
                let (a, b) = (batched[t * m.rows + r], single[r]);
                assert!((a - b).abs() <= 1e-4 * b.abs().max(1.0), "t{t} r{r}: {a} vs {b}");
            }
        }
    }

    /// Hand-checked Q4_K block: scale/min unpacking for both halves of `scales`.
    #[test]
    fn q4k_scale_unpacking() {
        let mut scales = [0u8; 12];
        scales[0] = 0b11_000101; // sc0 = 5, high bits of sc4 = 3
        scales[4] = 0b10_000111; // m0 = 7, high bits of m4 = 2
        scales[8] = 0x9a; // sc4 low = 0xa, m4 low = 0x9
        assert_eq!(scale_min_k4(0, &scales), (5, 7));
        assert_eq!(scale_min_k4(4, &scales), (0xa | (3 << 4), 0x9 | (2 << 4)));
    }
}
