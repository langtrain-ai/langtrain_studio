//! Paged KV cache.
//!
//! KV lives in fixed 16-token blocks shared by all sequences. Each sequence
//! has a block table. Copying a prefix (`share`) only bumps reference counts;
//! a shared block is copied the first time a sequence writes into it. Memory
//! is f16, half of f32.

use std::collections::HashMap;

use anyhow::{bail, Result};
use half::f16;

pub const BLOCK: usize = 16;

pub struct KvCache {
    n_layers: usize,
    /// Floats per token per layer for K (and for V): `n_kv_heads * head_dim`.
    pub kv_dim: usize,
    n_blocks: usize,
    /// `[layer][block][slot][kv_dim]`, flattened.
    k: Vec<f16>,
    v: Vec<f16>,
    refs: Vec<u32>,
    free: Vec<usize>,
    seqs: HashMap<u32, Seq>,
}

#[derive(Default, Clone)]
struct Seq {
    blocks: Vec<usize>,
    len: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, kv_dim: usize, n_tokens: usize) -> Self {
        let n_blocks = n_tokens.div_ceil(BLOCK);
        let size = n_layers * n_blocks * BLOCK * kv_dim;
        Self {
            n_layers,
            kv_dim,
            n_blocks,
            k: vec![f16::ZERO; size],
            v: vec![f16::ZERO; size],
            refs: vec![0; n_blocks],
            free: (0..n_blocks).rev().collect(),
            seqs: HashMap::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.n_blocks * BLOCK
    }

    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * 2
    }

    pub fn len(&self, seq: u32) -> usize {
        self.seqs.get(&seq).map(|s| s.len).unwrap_or(0)
    }

    #[inline]
    fn offset(&self, layer: usize, block: usize, slot: usize) -> usize {
        ((layer * self.n_blocks + block) * BLOCK + slot) * self.kv_dim
    }

    fn alloc(&mut self) -> Result<usize> {
        let b = self.free.pop().ok_or_else(|| anyhow::anyhow!("KV cache full"))?;
        self.refs[b] = 1;
        Ok(b)
    }

    fn release(&mut self, b: usize) {
        self.refs[b] -= 1;
        if self.refs[b] == 0 {
            self.free.push(b);
        }
    }

    /// Reserve the cell for `pos` of `seq` (must equal the current length)
    /// and return `(block, slot)` to write K/V into.
    pub fn append(&mut self, seq: u32, pos: usize) -> Result<(usize, usize)> {
        let s = self.seqs.entry(seq).or_default().clone();
        if pos != s.len {
            bail!("seq {seq}: position {pos} is not contiguous (length {})", s.len);
        }
        let slot = pos % BLOCK;
        let mut blocks = s.blocks;
        if slot == 0 {
            blocks.push(self.alloc()?);
        } else {
            let last = *blocks.last().expect("non-empty");
            if self.refs[last] > 1 {
                // Copy-on-write: this sequence is about to diverge from a shared prefix.
                let nb = self.alloc()?;
                for l in 0..self.n_layers {
                    let (src, dst) = (self.offset(l, last, 0), self.offset(l, nb, 0));
                    let n = BLOCK * self.kv_dim;
                    self.k.copy_within(src..src + n, dst);
                    self.v.copy_within(src..src + n, dst);
                }
                self.release(last);
                *blocks.last_mut().unwrap() = nb;
            }
        }
        let block = *blocks.last().unwrap();
        self.seqs.insert(seq, Seq { blocks, len: pos + 1 });
        Ok((block, slot))
    }

    pub fn write(&mut self, layer: usize, block: usize, slot: usize, k: &[f32], v: &[f32]) {
        let o = self.offset(layer, block, slot);
        for (d, s) in self.k[o..o + self.kv_dim].iter_mut().zip(k) {
            *d = f16::from_f32(*s);
        }
        for (d, s) in self.v[o..o + self.kv_dim].iter_mut().zip(v) {
            *d = f16::from_f32(*s);
        }
    }

    /// Visit K and V rows for positions `0..n` of `seq` in `layer`.
    #[inline]
    pub fn rows(&self, layer: usize, seq: u32, n: usize) -> impl Iterator<Item = (&[f16], &[f16])> + '_ {
        let blocks: &[usize] = self.seqs.get(&seq).map(|s| s.blocks.as_slice()).unwrap_or(&[]);
        (0..n).map(move |p| {
            let o = self.offset(layer, blocks[p / BLOCK], p % BLOCK);
            (&self.k[o..o + self.kv_dim], &self.v[o..o + self.kv_dim])
        })
    }

    /// Drop positions `from..` of `seq`.
    pub fn truncate(&mut self, seq: u32, from: usize) {
        let Some(mut s) = self.seqs.remove(&seq) else { return };
        if from >= s.len {
            self.seqs.insert(seq, s);
            return;
        }
        let keep = from.div_ceil(BLOCK);
        for b in s.blocks.split_off(keep) {
            self.release(b);
        }
        s.len = from;
        if from > 0 {
            self.seqs.insert(seq, s);
        }
    }

    /// Make `dst` start with the first `len` tokens of `src`, sharing blocks.
    pub fn share(&mut self, src: u32, dst: u32, len: usize) {
        self.truncate(dst, 0);
        let Some(s) = self.seqs.get(&src) else { return };
        let len = len.min(s.len);
        let blocks: Vec<usize> = s.blocks[..len.div_ceil(BLOCK)].to_vec();
        for &b in &blocks {
            self.refs[b] += 1;
        }
        if len > 0 {
            self.seqs.insert(dst, Seq { blocks, len });
        }
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(kv: &mut KvCache, seq: u32, n: usize, base: f32) {
        for p in kv.len(seq)..n {
            let (b, s) = kv.append(seq, p).unwrap();
            let x = vec![base + p as f32; kv.kv_dim];
            kv.write(0, b, s, &x, &x);
        }
    }

    #[test]
    fn share_then_diverge_copies_on_write() {
        let mut kv = KvCache::new(1, 4, 256);
        fill(&mut kv, 0, 20, 0.0);
        kv.share(0, 1, 20);
        let free_before = kv.free_blocks();
        // Seq 1 writes into the shared partial block: it must get its own copy.
        let (b, s) = kv.append(1, 20).unwrap();
        kv.write(0, b, s, &[1000.0; 4], &[1000.0; 4]);
        assert_eq!(kv.free_blocks(), free_before - 1);
        let a: Vec<f32> = kv.rows(0, 0, 20).map(|(k, _)| k[0].to_f32()).collect();
        let c: Vec<f32> = kv.rows(0, 1, 21).map(|(k, _)| k[0].to_f32()).collect();
        assert_eq!(&a[..], &c[..20]);
        assert_eq!(c[20], 1000.0);
        assert_eq!(kv.len(0), 20);
    }

    #[test]
    fn truncate_frees_blocks() {
        let mut kv = KvCache::new(1, 4, 64);
        fill(&mut kv, 3, 40, 0.0);
        assert_eq!(kv.free_blocks(), 1);
        kv.truncate(3, 10);
        assert_eq!(kv.free_blocks(), 3);
        kv.truncate(3, 0);
        assert_eq!(kv.free_blocks(), 4);
        assert!(kv.append(3, 5).is_err());
    }
}
