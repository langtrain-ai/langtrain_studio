//! Single-token matmul throughput per weight format:
//! `cargo run --release -p langtrain-native --example kernel_bench`

use std::time::Instant;

use langtrain_native::gguf::{DType, TensorInfo};
use langtrain_native::quant::{QActs, QMatrix};

fn main() {
    let threads = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    langtrain_native::pool::init(threads);
    // A SmolLM2-sized FFN matrix and a larger one.
    for (rows, cols) in [(1536usize, 576usize), (4096, 4096)] {
        for (name, dtype, block, bytes) in [
            ("Q8_0", DType::Q8_0, 32, 34),
            ("Q5_0", DType::Q5_0, 32, 22),
            ("Q4_K", DType::Q4K, 256, 144),
        ] {
            if cols % block != 0 {
                continue;
            }
            let n_bytes = rows * cols / block * bytes;
            let data: Vec<u8> = (0..n_bytes)
                .map(|i| (i * 7919 % 251) as u8 & 0x3f)
                .collect();
            let info = TensorInfo {
                dims: vec![cols, rows],
                dtype,
                offset: 0,
            };
            let m = QMatrix::from_gguf(&info, &data).unwrap();
            let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
            let mut qa = QActs::default();
            qa.quantize(&x, 1, cols);
            let mut out = vec![0f32; rows];
            for _ in 0..20 {
                m.matmul(&qa, &mut out);
            }
            let iters = (2e9 / (rows * cols) as f64) as usize + 10;
            let t = Instant::now();
            for _ in 0..iters {
                m.matmul(&qa, &mut out);
            }
            let s = t.elapsed().as_secs_f64() / iters as f64;
            println!(
                "{rows}x{cols} {name}: {:7.1} us/matmul  {:6.1} GB/s (repacked {:.1} MB)  {:6.1} Gop/s",
                s * 1e6,
                m.bytes() as f64 / s / 1e9,
                m.bytes() as f64 / 1e6,
                (rows * cols) as f64 * 2.0 / s / 1e9
            );
        }
    }
}
