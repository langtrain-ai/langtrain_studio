//! Zero-copy GGUF reader: metadata is parsed, tensor data stays memory-mapped.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;

#[derive(Debug, Clone)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U64(v) => Some(v),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F64(v) => Some(v),
            Value::U64(v) => Some(v as f64),
            Value::I64(v) => Some(v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// ggml tensor types this engine understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Other(u32),
}

impl DType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => DType::F32,
            1 => DType::F16,
            2 => DType::Q4_0,
            3 => DType::Q4_1,
            6 => DType::Q5_0,
            7 => DType::Q5_1,
            8 => DType::Q8_0,
            12 => DType::Q4K,
            13 => DType::Q5K,
            14 => DType::Q6K,
            o => DType::Other(o),
        }
    }

    /// Bytes for `n` elements (`n` must be a multiple of the block size for quantized types).
    pub fn size_of(self, n: usize) -> Option<usize> {
        Some(match self {
            DType::F32 => n * 4,
            DType::F16 => n * 2,
            DType::Q4_0 => n / 32 * 18,
            DType::Q4_1 => n / 32 * 20,
            DType::Q5_0 => n / 32 * 22,
            DType::Q5_1 => n / 32 * 24,
            DType::Q8_0 => n / 32 * 34,
            DType::Q4K => n / 256 * 144,
            DType::Q5K => n / 256 * 176,
            DType::Q6K => n / 256 * 210,
            DType::Other(_) => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// ggml order: `dims[0]` is the contiguous (row) dimension.
    pub dims: Vec<usize>,
    pub dtype: DType,
    pub offset: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    mmap: Mmap,
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    data_start: usize,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len()).ok_or_else(|| anyhow!("truncated GGUF"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U64(self.take(1)?[0] as u64),
            1 => Value::I64(self.take(1)?[0] as i8 as i64),
            2 => Value::U64(u16::from_le_bytes(self.take(2)?.try_into()?) as u64),
            3 => Value::I64(i16::from_le_bytes(self.take(2)?.try_into()?) as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(self.u32()? as i32 as i64),
            6 => Value::F64(f32::from_le_bytes(self.take(4)?.try_into()?) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let elem = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(elem)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into()?)),
            t => bail!("unknown GGUF value type {t}"),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the model file is treated as read-only for the process lifetime.
        let mmap = unsafe { Mmap::map(&file)? };
        let mut r = Reader { buf: &mmap, pos: 0 };
        if r.take(4)? != b"GGUF" {
            bail!("{} is not a GGUF file", path.display());
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version}");
        }
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;

        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            metadata.insert(key, r.value(ty)?);
        }

        let mut tensors = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let dims = (0..n_dims).map(|_| r.u64().map(|d| d as usize)).collect::<Result<Vec<_>>>()?;
            let dtype = DType::from_u32(r.u32()?);
            let offset = r.u64()? as usize;
            tensors.insert(name, TensorInfo { dims, dtype, offset });
        }

        let align = metadata.get("general.alignment").and_then(Value::as_u64).unwrap_or(32) as usize;
        let data_start = r.pos.div_ceil(align) * align;
        let g = Self { mmap, metadata, tensors, data_start };
        for (name, t) in &g.tensors {
            if let Some(n) = t.dtype.size_of(t.n_elements()) {
                if g.data_start + t.offset + n > g.mmap.len() {
                    bail!("tensor {name} extends past end of file");
                }
            }
        }
        Ok(g)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        self.get(key).and_then(Value::as_u64).ok_or_else(|| anyhow!("missing metadata {key}"))
    }

    pub fn f64_or(&self, key: &str, default: f64) -> f64 {
        self.get(key).and_then(Value::as_f64).unwrap_or(default)
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    pub fn architecture(&self) -> Result<&str> {
        self.str("general.architecture").ok_or_else(|| anyhow!("missing general.architecture"))
    }

    pub fn tensor(&self, name: &str) -> Option<(&TensorInfo, &[u8])> {
        let t = self.tensors.get(name)?;
        let n = t.dtype.size_of(t.n_elements())?;
        let start = self.data_start + t.offset;
        Some((t, &self.mmap[start..start + n]))
    }
}
