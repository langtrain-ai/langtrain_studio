//! A low-latency thread pool for the per-token hot path.
//!
//! A decode step runs a few hundred short parallel loops (one per matmul and
//! attention pass). A general-purpose pool parks idle threads between them,
//! and waking them costs about as much as a small matmul. Here workers spin
//! for a short while after each job, so back-to-back jobs start in
//! microseconds, and park only when the engine goes idle.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How long an idle worker spins before parking.
const SPIN: Duration = Duration::from_micros(500);

type Task = dyn Fn(usize) + Sync;

/// One parallel loop. Counters live with the job, so a worker that picks it
/// up late finds `next >= n` and never calls a closure that has returned.
struct Job {
    f: *const Task,
    n: usize,
    next: AtomicUsize,
    pending: AtomicUsize,
}

struct Shared {
    /// Bumped once per job; workers watch it to see new work.
    generation: AtomicUsize,
    job: Mutex<Option<Arc<Job>>>,
    sleeping: Mutex<usize>,
    wake: Condvar,
    shutdown: AtomicBool,
}

// SAFETY: `f` is only called for a claimed index below `n`, and every such
// call finishes before `pending` reaches zero, which `run` waits for.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

pub struct Pool {
    shared: Arc<Shared>,
    threads: usize,
    /// Serializes callers: one job at a time.
    gate: Mutex<()>,
}

impl Pool {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            generation: AtomicUsize::new(0),
            job: Mutex::new(None),
            sleeping: Mutex::new(0),
            wake: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });
        for i in 1..threads {
            let s = shared.clone();
            std::thread::Builder::new()
                .name(format!("lt-hot-{i}"))
                .spawn(move || worker(s))
                .expect("spawn worker");
        }
        Self {
            shared,
            threads,
            gate: Mutex::new(()),
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `f(i)` for every `i < n`, across all threads including the caller.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        if n == 1 || self.threads == 1 {
            (0..n).for_each(f);
            return;
        }
        let _g = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        let s = &*self.shared;
        // SAFETY: erase the borrow's lifetime; we wait below until every task
        // has finished, so no worker can touch `f` after this call returns.
        let ptr: *const Task =
            unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static Task>(f) };
        let job = Arc::new(Job {
            f: ptr,
            n,
            next: AtomicUsize::new(0),
            pending: AtomicUsize::new(n),
        });
        *s.job.lock().unwrap() = Some(job.clone());
        s.generation.fetch_add(1, Ordering::AcqRel);
        if *s.sleeping.lock().unwrap() > 0 {
            s.wake.notify_all();
        }
        work(&job);
        while job.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        *s.job.lock().unwrap() = None;
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        let _g = self.shared.sleeping.lock().unwrap();
        self.shared.wake.notify_all();
    }
}

fn work(job: &Job) {
    loop {
        let i = job.next.fetch_add(1, Ordering::Relaxed);
        if i >= job.n {
            return;
        }
        // SAFETY: see `Job`: index `i` is claimed and below `n`, so `run` is
        // still waiting and the closure is alive.
        unsafe { (*job.f)(i) };
        job.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

fn worker(s: Arc<Shared>) {
    let mut seen = s.generation.load(Ordering::Acquire);
    loop {
        // Spin briefly for the next job, then park.
        let start = Instant::now();
        loop {
            if s.shutdown.load(Ordering::Acquire) {
                return;
            }
            let g = s.generation.load(Ordering::Acquire);
            if g != seen {
                seen = g;
                break;
            }
            if start.elapsed() > SPIN {
                let mut sleeping = s.sleeping.lock().unwrap();
                *sleeping += 1;
                while s.generation.load(Ordering::Acquire) == seen
                    && !s.shutdown.load(Ordering::Acquire)
                {
                    sleeping = s
                        .wake
                        .wait_timeout(sleeping, Duration::from_millis(50))
                        .unwrap()
                        .0;
                }
                *sleeping -= 1;
                continue;
            }
            std::hint::spin_loop();
        }
        let job = s.job.lock().unwrap().clone();
        if let Some(job) = job {
            work(&job);
        }
    }
}

static GLOBAL: OnceLock<Pool> = OnceLock::new();

/// Size the shared hot-path pool (first call wins).
pub fn init(threads: usize) {
    let _ = GLOBAL.set(Pool::new(threads));
}

/// The shared hot-path pool, defaulting to the available parallelism.
pub fn global() -> &'static Pool {
    GLOBAL.get_or_init(|| {
        Pool::new(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        )
    })
}

/// Split `out` into `chunk`-sized pieces and fill them in parallel.
pub fn for_chunks(out: &mut [f32], chunk: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let n = out.len().div_ceil(chunk);
    let base = SendPtr(out.as_mut_ptr());
    let len = out.len();
    global().run(n, &|c| {
        let start = c * chunk;
        let end = (start + chunk).min(len);
        // SAFETY: chunk ranges are disjoint, and `out` outlives `run`.
        let slice = unsafe { std::slice::from_raw_parts_mut(base.get().add(start), end - start) };
        f(c, slice);
    });
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
// SAFETY: used only to hand out disjoint sub-slices inside `for_chunks`.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn runs_every_task_exactly_once() {
        let pool = Pool::new(4);
        for n in [1, 3, 64, 1000] {
            let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
            pool.run(n, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n={n}");
        }
    }

    #[test]
    fn survives_idle_periods_and_concurrent_callers() {
        let pool = Arc::new(Pool::new(3));
        std::thread::sleep(Duration::from_millis(5)); // workers park
        let total = Arc::new(AtomicU64::new(0));
        let callers: Vec<_> = (0..4)
            .map(|_| {
                let (pool, total) = (pool.clone(), total.clone());
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        pool.run(17, &|i| {
                            total.fetch_add(i as u64, Ordering::Relaxed);
                        });
                    }
                })
            })
            .collect();
        for c in callers {
            c.join().unwrap();
        }
        assert_eq!(total.load(Ordering::Relaxed), 4 * 50 * (0..17).sum::<u64>());
    }

    #[test]
    fn for_chunks_covers_the_slice() {
        let mut v = vec![0f32; 1001];
        for_chunks(&mut v, 64, |c, s| s.iter_mut().for_each(|x| *x = c as f32));
        assert!(v.iter().enumerate().all(|(i, &x)| x == (i / 64) as f32));
    }
}
