//! Llama-family transformer forward pass (Llama 2/3, Mistral, SmolLM, Qwen2).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;

use crate::gguf::{Gguf, Value};
use crate::kv::KvCache;
use crate::quant::{to_f32, QActs, QMatrix};

#[derive(Debug, Clone)]
pub struct Config {
    pub arch: String,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub rope_dim: usize,
    /// NeoX rotates `(i, i + d/2)` pairs; the original layout rotates `(2i, 2i+1)`.
    pub rope_neox: bool,
    pub ctx_train: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Proj {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl Proj {
    pub const ALL: [Proj; 7] = [Proj::Q, Proj::K, Proj::V, Proj::O, Proj::Gate, Proj::Up, Proj::Down];
    pub fn gguf_name(self) -> &'static str {
        match self {
            Proj::Q => "attn_q",
            Proj::K => "attn_k",
            Proj::V => "attn_v",
            Proj::O => "attn_output",
            Proj::Gate => "ffn_gate",
            Proj::Up => "ffn_up",
            Proj::Down => "ffn_down",
        }
    }
}

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    w: HashMap<Proj, QMatrix>,
    bias: HashMap<Proj, Vec<f32>>,
}

/// A LoRA delta `scale * B @ A` for one projection.
pub struct LoraPair {
    a: Vec<f32>, // [rank][in]
    b: Vec<f32>, // [out][rank]
    rank: usize,
    scale: f32,
}

pub struct Lora {
    pairs: HashMap<(usize, Proj), LoraPair>,
}

impl Lora {
    pub fn load(path: &Path, cfg: &Config) -> Result<Self> {
        let g = Gguf::open(path)?;
        let alpha = g.get("adapter.lora.alpha").and_then(Value::as_f64);
        let mut pairs = HashMap::new();
        for l in 0..cfg.n_layers {
            for p in Proj::ALL {
                let base = format!("blk.{l}.{}.weight", p.gguf_name());
                let (Some((ai, ad)), Some((bi, bd))) = (g.tensor(&format!("{base}.lora_a")), g.tensor(&format!("{base}.lora_b"))) else {
                    continue;
                };
                let rank = ai.dims[1];
                if bi.dims[0] != rank {
                    bail!("{base}: lora_a/lora_b rank mismatch");
                }
                let scale = alpha.map(|a| a as f32 / rank as f32).unwrap_or(1.0);
                pairs.insert((l, p), LoraPair { a: to_f32(ai, ad)?, b: to_f32(bi, bd)?, rank, scale });
            }
        }
        if pairs.is_empty() {
            bail!("{} contains no LoRA tensors", path.display());
        }
        Ok(Self { pairs })
    }

    /// `y[t] += scale * B (A x[t])` for `n` rows.
    fn apply(&self, layer: usize, p: Proj, x: &[f32], n: usize, in_dim: usize, y: &mut [f32], out_dim: usize) {
        let Some(lp) = self.pairs.get(&(layer, p)) else { return };
        for t in 0..n {
            let xt = &x[t * in_dim..(t + 1) * in_dim];
            let mid: Vec<f32> = (0..lp.rank).map(|r| dot(&lp.a[r * in_dim..(r + 1) * in_dim], xt)).collect();
            let yt = &mut y[t * out_dim..(t + 1) * out_dim];
            for (o, yo) in yt.iter_mut().enumerate() {
                *yo += lp.scale * dot(&lp.b[o * lp.rank..(o + 1) * lp.rank], &mid);
            }
        }
    }
}

pub struct Model {
    pub cfg: Config,
    embed: QMatrix,
    output_norm: Vec<f32>,
    output: Option<QMatrix>,
    layers: Vec<Layer>,
    inv_freq: Vec<f32>,
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let dim = w.len();
    for (xr, or) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let ms = xr.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for ((o, v), g) in or.iter_mut().zip(xr).zip(w) {
            *o = v * inv * g;
        }
    }
}

impl Model {
    pub fn load(g: &Gguf) -> Result<Self> {
        let arch = g.architecture()?.to_string();
        let rope_neox = match arch.as_str() {
            "llama" | "mistral" | "granite" => false,
            "qwen2" => true,
            other => bail!("architecture '{other}' is not supported by the native engine yet (llama, mistral, qwen2 are)"),
        };
        let k = |s: &str| format!("{arch}.{s}");
        let dim = g.u64(&k("embedding_length"))? as usize;
        let n_heads = g.u64(&k("attention.head_count"))? as usize;
        let n_kv_heads = g.get(&k("attention.head_count_kv")).and_then(Value::as_u64).unwrap_or(n_heads as u64) as usize;
        let head_dim = g.get(&k("attention.key_length")).and_then(Value::as_u64).map(|v| v as usize).unwrap_or(dim / n_heads);
        let t = |name: &str| g.tensor(name).ok_or_else(|| anyhow!("missing tensor {name}"));
        let (embed_info, _) = t("token_embd.weight")?;
        let cfg = Config {
            dim,
            n_layers: g.u64(&k("block_count"))? as usize,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim: g.u64(&k("feed_forward_length"))? as usize,
            vocab: embed_info.dims[1],
            eps: g.f64_or(&k("attention.layer_norm_rms_epsilon"), 1e-5) as f32,
            rope_base: g.f64_or(&k("rope.freq_base"), 10_000.0) as f32,
            rope_dim: g.get(&k("rope.dimension_count")).and_then(Value::as_u64).map(|v| v as usize).unwrap_or(head_dim),
            rope_neox,
            ctx_train: g.get(&k("context_length")).and_then(Value::as_u64).unwrap_or(4096) as usize,
            arch,
        };

        let mat = |name: &str| -> Result<QMatrix> {
            let (i, d) = t(name)?;
            QMatrix::from_gguf(i, d).with_context(|| name.to_string())
        };
        let vecf = |name: &str| -> Result<Vec<f32>> {
            let (i, d) = t(name)?;
            to_f32(i, d)
        };

        let layers = (0..cfg.n_layers)
            .into_par_iter()
            .map(|l| -> Result<Layer> {
                let mut w = HashMap::new();
                let mut bias = HashMap::new();
                for p in Proj::ALL {
                    let base = format!("blk.{l}.{}", p.gguf_name());
                    w.insert(p, mat(&format!("{base}.weight"))?);
                    if g.tensor(&format!("{base}.bias")).is_some() {
                        bias.insert(p, vecf(&format!("{base}.bias"))?);
                    }
                }
                Ok(Layer {
                    attn_norm: vecf(&format!("blk.{l}.attn_norm.weight"))?,
                    ffn_norm: vecf(&format!("blk.{l}.ffn_norm.weight"))?,
                    w,
                    bias,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let inv_freq = (0..cfg.rope_dim / 2).map(|i| cfg.rope_base.powf(-2.0 * i as f32 / cfg.rope_dim as f32)).collect();
        Ok(Self {
            embed: mat("token_embd.weight")?,
            output_norm: vecf("output_norm.weight")?,
            // Tied embeddings when there is no separate output matrix.
            output: g.tensor("output.weight").map(|(i, d)| QMatrix::from_gguf(i, d)).transpose()?,
            layers,
            inv_freq,
            cfg,
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.embed.bytes()
            + self.output.as_ref().map(QMatrix::bytes).unwrap_or(0)
            + self.layers.iter().flat_map(|l| l.w.values()).map(QMatrix::bytes).sum::<usize>()
    }

    fn rope(&self, x: &mut [f32], n_heads: usize, pos: usize) {
        let hd = self.cfg.head_dim;
        let half = self.cfg.rope_dim / 2;
        for h in 0..n_heads {
            let v = &mut x[h * hd..(h + 1) * hd];
            for (i, f) in self.inv_freq.iter().enumerate() {
                let (sin, cos) = (pos as f32 * f).sin_cos();
                let (a, b) = if self.cfg.rope_neox { (i, i + half) } else { (2 * i, 2 * i + 1) };
                let (x0, x1) = (v[a], v[b]);
                v[a] = x0 * cos - x1 * sin;
                v[b] = x0 * sin + x1 * cos;
            }
        }
    }

    fn proj(&self, layer: usize, p: Proj, x: &[f32], xq: &QActs, out: &mut [f32], lora: Option<&Lora>) {
        let l = &self.layers[layer];
        let w = &l.w[&p];
        w.matmul(xq, out);
        if let Some(b) = l.bias.get(&p) {
            for row in out.chunks_exact_mut(w.rows) {
                for (o, bv) in row.iter_mut().zip(b) {
                    *o += bv;
                }
            }
        }
        if let Some(lora) = lora {
            lora.apply(layer, p, x, xq.n, w.cols, out, w.rows);
        }
    }

    /// Run `tokens` (with their sequence ids and positions) through the model.
    /// Returns logits for each index in `want_logits`, in order.
    pub fn forward(
        &self,
        kv: &mut KvCache,
        tokens: &[i32],
        seqs: &[u32],
        positions: &[usize],
        want_logits: &[usize],
        lora: Option<&Lora>,
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let n = tokens.len();
        let (dim, hd) = (c.dim, c.head_dim);
        let (q_dim, kv_dim) = (c.n_heads * hd, c.n_kv_heads * hd);

        // Claim KV cells first so copy-on-write happens before any layer writes.
        let cells = (0..n).map(|i| kv.append(seqs[i], positions[i])).collect::<Result<Vec<_>>>()?;

        let mut x = vec![0f32; n * dim];
        let mut emb_row = vec![0f32; dim];
        for (i, &t) in tokens.iter().enumerate() {
            self.embed_row(t as usize, &mut emb_row)?;
            x[i * dim..(i + 1) * dim].copy_from_slice(&emb_row);
        }

        let mut h = vec![0f32; n * dim];
        let mut q = vec![0f32; n * q_dim];
        let mut k = vec![0f32; n * kv_dim];
        let mut v = vec![0f32; n * kv_dim];
        let mut att = vec![0f32; n * q_dim];
        let mut o = vec![0f32; n * dim];
        let mut gate = vec![0f32; n * c.ffn_dim];
        let mut up = vec![0f32; n * c.ffn_dim];
        let mut xq = QActs::default();
        let group = c.n_heads / c.n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();

        for (li, layer) in self.layers.iter().enumerate() {
            rms_norm(&x, &layer.attn_norm, c.eps, &mut h);
            xq.quantize(&h, n, dim);
            self.proj(li, Proj::Q, &h, &xq, &mut q, lora);
            self.proj(li, Proj::K, &h, &xq, &mut k, lora);
            self.proj(li, Proj::V, &h, &xq, &mut v, lora);
            for i in 0..n {
                self.rope(&mut q[i * q_dim..(i + 1) * q_dim], c.n_heads, positions[i]);
                self.rope(&mut k[i * kv_dim..(i + 1) * kv_dim], c.n_kv_heads, positions[i]);
                let (b, s) = cells[i];
                kv.write(li, b, s, &k[i * kv_dim..(i + 1) * kv_dim], &v[i * kv_dim..(i + 1) * kv_dim]);
            }

            // Causal attention over each token's own sequence, one task per (token, head).
            let kv_ref = &*kv;
            att.par_chunks_mut(hd).enumerate().for_each(|(idx, out)| {
                let (i, head) = (idx / c.n_heads, idx % c.n_heads);
                let kvh = head / group;
                let qv = &q[i * q_dim + head * hd..i * q_dim + (head + 1) * hd];
                let len = positions[i] + 1;
                let mut scores: Vec<f32> = kv_ref
                    .rows(li, seqs[i], len)
                    .map(|(kr, _)| {
                        let kr = &kr[kvh * hd..(kvh + 1) * hd];
                        qv.iter().zip(kr).map(|(a, b)| a * b.to_f32()).sum::<f32>() * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                out.fill(0.0);
                for ((_, vr), s) in kv_ref.rows(li, seqs[i], len).zip(&scores) {
                    let w = s / sum;
                    for (o, vv) in out.iter_mut().zip(&vr[kvh * hd..(kvh + 1) * hd]) {
                        *o += w * vv.to_f32();
                    }
                }
            });

            xq.quantize(&att, n, q_dim);
            self.proj(li, Proj::O, &att, &xq, &mut o, lora);
            for (xv, ov) in x.iter_mut().zip(&o) {
                *xv += ov;
            }

            rms_norm(&x, &layer.ffn_norm, c.eps, &mut h);
            xq.quantize(&h, n, dim);
            self.proj(li, Proj::Gate, &h, &xq, &mut gate, lora);
            self.proj(li, Proj::Up, &h, &xq, &mut up, lora);
            for (g, u) in gate.iter_mut().zip(&up) {
                *g = *g / (1.0 + (-*g).exp()) * u; // SiLU(g) * u
            }
            xq.quantize(&gate, n, c.ffn_dim);
            self.proj(li, Proj::Down, &gate, &xq, &mut o, lora);
            for (xv, ov) in x.iter_mut().zip(&o) {
                *xv += ov;
            }
        }

        // Only rows that need logits go through the (large) output matrix.
        let m = want_logits.len();
        if m == 0 {
            return Ok(Vec::new());
        }
        let mut sel = vec![0f32; m * dim];
        for (j, &i) in want_logits.iter().enumerate() {
            sel[j * dim..(j + 1) * dim].copy_from_slice(&x[i * dim..(i + 1) * dim]);
        }
        let mut normed = vec![0f32; m * dim];
        rms_norm(&sel, &self.output_norm, c.eps, &mut normed);
        xq.quantize(&normed, m, dim);
        let out_m = self.output.as_ref().unwrap_or(&self.embed);
        let mut logits = vec![0f32; m * out_m.rows];
        out_m.matmul(&xq, &mut logits);
        Ok(logits)
    }

    fn embed_row(&self, t: usize, out: &mut [f32]) -> Result<()> {
        if t >= self.embed.rows {
            bail!("token id {t} out of range");
        }
        // Dequantize one row of the embedding table.
        let cols = self.embed.cols;
        let nb = cols / crate::quant::QK;
        let scales = &self.embed.scales[t * nb..(t + 1) * nb];
        match &self.embed.w {
            crate::quant::Weights::Q8 { qs } => {
                for (j, o) in out.iter_mut().enumerate() {
                    *o = scales[j / 32] * qs[t * cols + j] as f32;
                }
            }
            crate::quant::Weights::Q4 { qs } => {
                let row = &qs[t * cols / 2..(t + 1) * cols / 2];
                for b in 0..nb {
                    for j in 0..16 {
                        out[b * 32 + j] = scales[b] * ((row[b * 16 + j] & 0x0f) as i32 - 8) as f32;
                        out[b * 32 + j + 16] = scales[b] * ((row[b * 16 + j] >> 4) as i32 - 8) as f32;
                    }
                }
            }
        }
        Ok(())
    }
}
