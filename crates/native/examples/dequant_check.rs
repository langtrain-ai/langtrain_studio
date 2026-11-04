//! Print a fingerprint of every 2-D tensor's dequantized values, for
//! comparison with gguf-py's reference dequantizer:
//! `cargo run --release -p langtrain-native --example dequant_check -- model.gguf`

use langtrain_native::gguf::Gguf;
use langtrain_native::quant::{to_f32, QMatrix};

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dequant_check <model.gguf>");
    let g = Gguf::open(std::path::Path::new(&path))?;
    let mut names: Vec<&String> = g.tensors.keys().collect();
    names.sort();
    for name in names {
        let (info, data) = g.tensor(name).unwrap();
        if info.dims.len() != 2 {
            continue;
        }
        let v = to_f32(info, data)?;
        // The repacked matrix must reproduce the same values row by row.
        let m = QMatrix::from_gguf(info, data)?;
        let mut row = vec![0f32; m.cols];
        let mut max_diff = 0f32;
        for r in (0..m.rows).step_by((m.rows / 7).max(1)) {
            m.dequant_row(r, &mut row);
            for (a, b) in row.iter().zip(&v[r * m.cols..(r + 1) * m.cols]) {
                max_diff = max_diff.max((a - b).abs());
            }
        }
        let sum: f64 = v.iter().map(|&x| x as f64).sum();
        let abs: f64 = v.iter().map(|&x| x.abs() as f64).sum();
        println!(
            "{name} {:?} sum={sum:.6} abs={abs:.6} v0={:.6} vlast={:.6} repack_diff={max_diff:e}",
            info.dtype,
            v[0],
            v[v.len() - 1]
        );
    }
    Ok(())
}
