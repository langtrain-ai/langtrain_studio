use serde::{Deserialize, Serialize};

use crate::backend::Token;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingParams {
    pub max_tokens: Option<usize>,
    /// `<= 0` means greedy decoding.
    pub temperature: f32,
    /// `0` disables top-k.
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: None,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
            stop: Vec::new(),
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self { temperature: 0.0, ..Self::default() }
    }
}

/// SplitMix64: tiny, fast, and good enough for token sampling.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: Rng,
    candidates: Vec<(Token, f32)>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let seed = params.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x5EED)
        });
        Self { params, rng: Rng(seed), candidates: Vec::new() }
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Sample the next token from `logits`, given previously generated `history`.
    pub fn sample(&mut self, logits: &[f32], history: &[Token]) -> Token {
        let p = &self.params;
        self.candidates.clear();
        self.candidates.extend(logits.iter().enumerate().map(|(i, &l)| (i as Token, l)));
        apply_penalties(&mut self.candidates, history, p);

        if p.temperature <= 0.0 {
            return argmax(&self.candidates);
        }

        // Top-k first: it shrinks a 150k-entry vocab to a handful before sorting.
        let c = &mut self.candidates;
        if p.top_k > 0 && p.top_k < c.len() {
            c.select_nth_unstable_by(p.top_k - 1, |a, b| b.1.total_cmp(&a.1));
            c.truncate(p.top_k);
        }
        c.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let max = c[0].1;
        let mut sum = 0.0f32;
        for (_, l) in c.iter_mut() {
            *l = ((*l - max) / p.temperature).exp();
            sum += *l;
        }
        for (_, l) in c.iter_mut() {
            *l /= sum;
        }

        if p.min_p > 0.0 {
            let floor = c[0].1 * p.min_p;
            let keep = c.iter().take_while(|(_, pr)| *pr >= floor).count().max(1);
            c.truncate(keep);
        }
        if p.top_p < 1.0 {
            let mut cum = 0.0;
            let mut keep = c.len();
            for (i, (_, pr)) in c.iter().enumerate() {
                cum += pr;
                if cum >= p.top_p {
                    keep = i + 1;
                    break;
                }
            }
            c.truncate(keep);
        }

        let total: f32 = c.iter().map(|(_, pr)| pr).sum();
        let mut r = self.rng.next_f32() * total;
        for &(tok, pr) in c.iter() {
            r -= pr;
            if r <= 0.0 {
                return tok;
            }
        }
        c[c.len() - 1].0
    }
}

fn apply_penalties(c: &mut [(Token, f32)], history: &[Token], p: &SamplingParams) {
    let no_penalty = p.repeat_penalty == 1.0 && p.presence_penalty == 0.0 && p.frequency_penalty == 0.0;
    if no_penalty || history.is_empty() {
        return;
    }
    let window = &history[history.len().saturating_sub(p.repeat_last_n.max(1))..];
    let mut counts: std::collections::HashMap<Token, u32> = std::collections::HashMap::new();
    for &t in window {
        *counts.entry(t).or_default() += 1;
    }
    for (tok, n) in counts {
        let Some(entry) = c.get_mut(tok as usize) else { continue };
        let l = &mut entry.1;
        if p.repeat_penalty != 1.0 {
            *l = if *l > 0.0 { *l / p.repeat_penalty } else { *l * p.repeat_penalty };
        }
        *l -= p.presence_penalty + p.frequency_penalty * n as f32;
    }
}

fn argmax(c: &[(Token, f32)]) -> Token {
    c.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|x| x.0).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let mut s = Sampler::new(SamplingParams::greedy());
        assert_eq!(s.sample(&[0.1, 3.0, 2.0], &[]), 1);
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let params = SamplingParams { seed: Some(7), temperature: 1.0, ..Default::default() };
        let logits: Vec<f32> = (0..100).map(|i| (i % 7) as f32).collect();
        let a: Vec<_> = { let mut s = Sampler::new(params.clone()); (0..20).map(|_| s.sample(&logits, &[])).collect() };
        let b: Vec<_> = { let mut s = Sampler::new(params); (0..20).map(|_| s.sample(&logits, &[])).collect() };
        assert_eq!(a, b);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let params = SamplingParams { top_k: 1, temperature: 1.5, ..Default::default() };
        let mut s = Sampler::new(params);
        for _ in 0..50 {
            assert_eq!(s.sample(&[1.0, 5.0, 4.9, 0.0], &[]), 1);
        }
    }

    #[test]
    fn repeat_penalty_changes_greedy_choice() {
        let params = SamplingParams { repeat_penalty: 2.0, ..SamplingParams::greedy() };
        let mut s = Sampler::new(params);
        assert_eq!(s.sample(&[0.0, 3.0, 2.0], &[1]), 2);
    }
}
