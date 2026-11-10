mod agent;
mod hardware;
mod inference;
mod langtrain_cloud;
mod model_manager;
mod subscription;

use inference::InferenceState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(InferenceState::default())
        .invoke_handler(tauri::generate_handler![
            // Hardware detection
            hardware::detect_hardware,
            // Model management
            model_manager::list_local_models,
            model_manager::delete_local_model,
            model_manager::download_model,
            model_manager::get_models_dir,
            // Local inference
            inference::start_inference_server,
            inference::stop_inference_server,
            inference::inference_server_status,
            // Subscription
            subscription::check_pro_access,
            // Agent — file system + shell
            agent::read_file,
            agent::write_file,
            agent::create_file,
            agent::list_directory,
            agent::delete_path,
            agent::execute_shell,
            agent::search_in_files,
            agent::get_file_info,
            agent::move_or_rename,
            agent::get_working_directory,
            // Langtrain Cloud
            langtrain_cloud::list_cloud_models,
            langtrain_cloud::download_cloud_model,
            langtrain_cloud::check_cloud_connection,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
