//! Byte-level BPE tokenizer built from GGUF metadata (`tokenizer.ggml.*`),
//! plus Jinja chat templates rendered with minijinja.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use fancy_regex::Regex;
use langtrain_engine::ChatMessage;

use crate::gguf::{Gguf, Value};

const TOKEN_TYPE_CONTROL: u64 = 3;
const TOKEN_TYPE_USER_DEFINED: u64 = 4;

const GPT2_RE: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const LLAMA3_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    tokens: Vec<String>,
    vocab: HashMap<String, i32>,
    merges: HashMap<(String, String), usize>,
    /// Special tokens matched literally before BPE, longest first.
    specials: Vec<(String, i32)>,
    special_ids: Vec<bool>,
    pre: Vec<Regex>,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    pub bos: Option<i32>,
    pub add_bos: bool,
    eog: Vec<i32>,
    template: Option<String>,
    bos_text: String,
    eos_text: String,
}

/// GPT-2's reversible byte -> printable-char mapping.
fn bytes_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (b'!' as u32..=b'~' as u32).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    map
}

fn str_array(g: &Gguf, key: &str) -> Result<Vec<String>> {
    g.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing {key}"))?
        .iter()
        .map(|v| v.as_str().map(str::to_owned).ok_or_else(|| anyhow!("{key}: expected strings")))
        .collect()
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let model = g.str("tokenizer.ggml.model").unwrap_or("");
        if model != "gpt2" {
            bail!("tokenizer '{model}' is not supported by the native engine yet (byte-level BPE models such as Llama 3, Qwen and SmolLM are)");
        }
        let tokens = str_array(g, "tokenizer.ggml.tokens")?;
        let types: Vec<u64> = g
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1)).collect())
            .unwrap_or_else(|| vec![1; tokens.len()]);
        let merges = str_array(g, "tokenizer.ggml.merges")?
            .into_iter()
            .enumerate()
            .filter_map(|(rank, m)| m.split_once(' ').map(|(a, b)| ((a.to_string(), b.to_string()), rank)))
            .collect();
        let vocab: HashMap<String, i32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as i32)).collect();

        let mut special_ids = vec![false; tokens.len()];
        let mut specials: Vec<(String, i32)> = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            let ty = types.get(i).copied().unwrap_or(1);
            if ty == TOKEN_TYPE_CONTROL || ty == TOKEN_TYPE_USER_DEFINED {
                special_ids[i] = ty == TOKEN_TYPE_CONTROL;
                if !t.is_empty() {
                    specials.push((t.clone(), i as i32));
                }
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let pre_type = g.str("tokenizer.ggml.pre").unwrap_or("default");
        let pre = match pre_type {
            "llama3" | "llama-bpe" | "falcon3" | "pixtral" => vec![LLAMA3_RE],
            "qwen2" | "deepseek-r1-qwen" => vec![QWEN2_RE],
            // Digits are split one by one before the GPT-2 pattern.
            "smollm" | "starcoder" | "refact" | "command-r" | "codeshell" => vec![r"\p{N}", GPT2_RE],
            _ => vec![GPT2_RE],
        }
        .into_iter()
        .map(|r| Regex::new(r).map_err(|e| anyhow!("pre-tokenizer regex: {e}")))
        .collect::<Result<Vec<_>>>()?;

        let id = |k: &str| g.get(k).and_then(Value::as_u64).map(|v| v as i32);
        let bos = id("tokenizer.ggml.bos_token_id");
        let eos = id("tokenizer.ggml.eos_token_id");
        let mut eog: Vec<i32> = [eos, id("tokenizer.ggml.eot_token_id"), id("tokenizer.ggml.eom_token_id")].into_iter().flatten().collect();
        // Chat end markers that some GGUFs do not list as EOS.
        for name in ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "<end_of_turn>", "<|end|>", "<|eom_id|>"] {
            if let Some(&i) = vocab.get(name) {
                eog.push(i);
            }
        }
        eog.sort_unstable();
        eog.dedup();

        let byte_enc = bytes_to_unicode();
        let text_of = |i: Option<i32>| i.and_then(|i| tokens.get(i as usize)).cloned().unwrap_or_default();
        Ok(Self {
            bos_text: text_of(bos),
            eos_text: text_of(eos),
            byte_dec: byte_enc.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect(),
            byte_enc,
            add_bos: g.get("tokenizer.ggml.add_bos_token").and_then(Value::as_bool).unwrap_or(false),
            template: g.str("tokenizer.chat_template").map(str::to_owned),
            tokens,
            vocab,
            merges,
            specials,
            special_ids,
            pre,
            bos,
            eog,
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_eog(&self, t: i32) -> bool {
        self.eog.binary_search(&t).is_ok()
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Vec<i32> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            if let Some(b) = self.bos {
                out.push(b);
            }
        }
        // Split out special tokens, then BPE each plain segment.
        let mut rest = text;
        while !rest.is_empty() {
            let hit = self
                .specials
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|pos| (pos, s.len(), *id)))
                .min_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
            match hit {
                Some((pos, len, id)) => {
                    self.encode_plain(&rest[..pos], &mut out);
                    out.push(id);
                    rest = &rest[pos + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut pieces = vec![text];
        for re in &self.pre {
            let mut next = Vec::new();
            for p in pieces {
                // Keep both matches and the gaps between them.
                let mut last = 0;
                for m in re.find_iter(p).flatten() {
                    if m.start() > last {
                        next.push(&p[last..m.start()]);
                    }
                    next.push(m.as_str());
                    last = m.end();
                }
                if last < p.len() {
                    next.push(&p[last..]);
                }
            }
            pieces = next;
        }
        pieces
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<i32>) {
        for word in self.pre_tokenize(text) {
            let mapped: String = word.bytes().map(|b| self.byte_enc[b as usize]).collect();
            if let Some(&id) = self.vocab.get(&mapped) {
                out.push(id);
                continue;
            }
            let mut syms: Vec<String> = mapped.chars().map(String::from).collect();
            // Repeatedly apply the lowest-rank merge.
            loop {
                let best = syms
                    .windows(2)
                    .enumerate()
                    .filter_map(|(i, w)| self.merges.get(&(w[0].clone(), w[1].clone())).map(|&r| (r, i)))
                    .min();
                let Some((_, i)) = best else { break };
                let merged = format!("{}{}", syms[i], syms[i + 1]);
                syms[i] = merged;
                syms.remove(i + 1);
            }
            for s in syms {
                match self.vocab.get(&s) {
                    Some(&id) => out.push(id),
                    // Unknown symbol: fall back to single-byte tokens.
                    None => out.extend(s.chars().filter_map(|c| self.vocab.get(&c.to_string()).copied())),
                }
            }
        }
    }

    /// Bytes for one token. Control tokens render as nothing.
    pub fn decode_token(&self, t: i32) -> Vec<u8> {
        let i = t as usize;
        if i >= self.tokens.len() || self.special_ids[i] {
            return Vec::new();
        }
        let s = &self.tokens[i];
        if self.specials.iter().any(|(sp, id)| *id == t && sp == s) {
            return s.as_bytes().to_vec(); // user-defined tokens are literal text
        }
        s.chars().map(|c| self.byte_dec.get(&c).copied().unwrap_or(b'?')).collect()
    }

    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        let Some(tmpl) = &self.template else {
            let mut s: String = messages.iter().map(|m| format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content)).collect();
            s.push_str("<|im_start|>assistant\n");
            return Ok(s);
        };
        let mut env = minijinja::Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        env.add_template("chat", tmpl).map_err(|e| anyhow!("chat template: {e}"))?;
        let msgs: Vec<_> = messages.iter().map(|m| minijinja::context! { role => &m.role, content => &m.content }).collect();
        env.get_template("chat")?
            .render(minijinja::context! {
                messages => msgs,
                add_generation_prompt => true,
                bos_token => &self.bos_text,
                eos_token => &self.eos_text,
            })
            .map_err(|e| anyhow!("chat template: {e}"))
    }
}
