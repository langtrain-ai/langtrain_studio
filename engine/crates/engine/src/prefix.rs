//! Slot-level prefix cache.
//!
//! Every KV slot remembers the tokens it holds after its request finishes.
//! A new request is placed on the free slot that shares the longest prefix
//! with it, so only the new suffix is prefilled. If a *different* slot (busy
//! or not) holds a much longer matching prefix, its KV is copied over; in
//! llama.cpp's unified cache that copy only tags existing cells, so it is
//! nearly free. KV computed under one LoRA adapter is never reused for another.

use crate::backend::Token;

/// Copying another slot's KV is only worth it above this many extra tokens.
const MIN_COPY_GAIN: usize = 16;

#[derive(Debug, Clone, Default)]
pub struct Slot {
    /// Tokens currently materialized in this slot's KV cache.
    pub tokens: Vec<Token>,
    pub adapter: Option<String>,
    pub busy: bool,
    pub last_used: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    pub slot: usize,
    /// Prompt tokens already present in KV after any copy.
    pub reuse: usize,
    /// `(source slot, length)` to copy into `slot` before prefilling.
    pub copy_from: Option<(usize, usize)>,
}

#[derive(Debug)]
pub struct SlotCache {
    pub slots: Vec<Slot>,
    clock: u64,
}

pub fn common_prefix(a: &[Token], b: &[Token]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl SlotCache {
    pub fn new(n_slots: usize) -> Self {
        Self { slots: vec![Slot::default(); n_slots], clock: 0 }
    }

    pub fn has_free(&self) -> bool {
        self.slots.iter().any(|s| !s.busy)
    }

    /// KV cells held by idle slots (cached prefixes that can be evicted).
    pub fn idle_cells(&self) -> usize {
        self.slots.iter().filter(|s| !s.busy).map(|s| s.tokens.len()).sum()
    }

    /// Choose where to run `prompt`. Returns `None` when every slot is busy.
    pub fn place(&self, prompt: &[Token], adapter: Option<&str>) -> Option<Placement> {
        // At least one prompt token must be evaluated to get logits.
        let cap = prompt.len().saturating_sub(1);
        let matches = |s: &Slot| {
            if s.adapter.as_deref() == adapter {
                common_prefix(&s.tokens, prompt).min(cap)
            } else {
                0
            }
        };

        let (slot, own) = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.busy)
            // Longest match wins; among equals, evict the least recently used.
            .max_by(|(_, a), (_, b)| matches(a).cmp(&matches(b)).then(b.last_used.cmp(&a.last_used)))
            .map(|(i, s)| (i, matches(s)))?;

        let best_other = self
            .slots
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != slot)
            .map(|(i, s)| (i, matches(s)))
            .max_by_key(|&(_, n)| n);

        match best_other {
            Some((src, n)) if n >= own + MIN_COPY_GAIN => {
                Some(Placement { slot, reuse: n, copy_from: Some((src, n)) })
            }
            _ => Some(Placement { slot, reuse: own, copy_from: None }),
        }
    }

    pub fn acquire(&mut self, p: &Placement, prompt: &[Token], adapter: Option<&str>) {
        self.clock += 1;
        let s = &mut self.slots[p.slot];
        s.busy = true;
        s.last_used = self.clock;
        s.adapter = adapter.map(str::to_owned);
        s.tokens.clear();
        s.tokens.extend_from_slice(&prompt[..p.reuse]);
    }

    /// Mark a slot idle, recording exactly the tokens its KV now holds.
    pub fn release(&mut self, slot: usize, kv_tokens: &[Token]) {
        self.clock += 1;
        let s = &mut self.slots[slot];
        s.busy = false;
        s.last_used = self.clock;
        s.tokens.clear();
        s.tokens.extend_from_slice(kv_tokens);
    }

    /// Evict the least recently used idle slot that holds KV. Returns the slot evicted.
    pub fn evict_lru(&mut self) -> Option<usize> {
        let idx = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.busy && !s.tokens.is_empty())
            .min_by_key(|(_, s)| s.last_used)
            .map(|(i, _)| i)?;
        self.slots[idx].tokens.clear();
        Some(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: usize) -> Vec<Token> {
        (0..n as Token).collect()
    }

    #[test]
    fn prefers_slot_with_longest_prefix() {
        let mut c = SlotCache::new(3);
        c.release(1, &toks(50));
        let mut prompt = toks(50);
        prompt.extend([999, 998]);
        let p = c.place(&prompt, None).unwrap();
        assert_eq!(p, Placement { slot: 1, reuse: 50, copy_from: None });
    }

    #[test]
    fn full_match_still_evaluates_last_token() {
        let mut c = SlotCache::new(1);
        c.release(0, &toks(10));
        assert_eq!(c.place(&toks(10), None).unwrap().reuse, 9);
    }

    #[test]
    fn copies_from_busy_slot_when_worth_it() {
        let mut c = SlotCache::new(2);
        let prompt = toks(100);
        let p0 = c.place(&prompt, None).unwrap();
        c.acquire(&p0, &prompt, None);
        c.slots[p0.slot].tokens = toks(100);
        let p1 = c.place(&prompt, None).unwrap();
        assert_ne!(p1.slot, p0.slot);
        assert_eq!(p1.copy_from, Some((p0.slot, 99)));
    }

    #[test]
    fn adapters_do_not_share_kv() {
        let mut c = SlotCache::new(1);
        c.release(0, &toks(40));
        c.slots[0].adapter = Some("support".into());
        assert_eq!(c.place(&toks(40), None).unwrap().reuse, 0);
        assert_eq!(c.place(&toks(40), Some("support")).unwrap().reuse, 39);
    }

    #[test]
    fn evicts_least_recently_used() {
        let mut c = SlotCache::new(2);
        c.release(0, &toks(5));
        c.release(1, &toks(5));
        assert_eq!(c.evict_lru(), Some(0));
        assert_eq!(c.idle_cells(), 5);
    }
}
