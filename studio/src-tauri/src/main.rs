//! Langtrain Studio desktop shell.
//!
//! Starts the bundled `langtrain serve` sidecar, waits for it to answer, and
//! points the window at the web UI it serves (the `langtrain-web` crate).
//! Closing the window stops the server.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpStream;
use std::time::{Duration, Instant};

use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_shell::ShellExt;

const PORT: u16 = 11435;

fn wait_for_server(timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if TcpStream::connect_timeout(&([127, 0, 0, 1], PORT).into(), Duration::from_millis(200)).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Reuse a server the user already runs; otherwise start ours.
            if !wait_for_server(Duration::from_millis(300)) {
                let (_rx, child) = app
                    .shell()
                    .sidecar("langtrain")?
                    .args(["serve", "--port", &PORT.to_string()])
                    .spawn()?;
                app.manage(std::sync::Mutex::new(Some(child)));
                if !wait_for_server(Duration::from_secs(20)) {
                    return Err("Langtrain server did not start".into());
                }
            }
            let url = format!("http://127.0.0.1:{PORT}/").parse()?;
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("Langtrain Studio")
                .inner_size(1200.0, 800.0)
                .min_inner_size(720.0, 480.0)
                .build()?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.app_handle().try_state::<std::sync::Mutex<Option<tauri_plugin_shell::process::CommandChild>>>();
                if let Some(child) = state.and_then(|s| s.lock().ok().and_then(|mut c| c.take())) {
                    let _ = child.kill();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Langtrain Studio");
}
