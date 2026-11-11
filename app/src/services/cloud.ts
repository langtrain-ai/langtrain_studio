import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

export interface CloudModel {
  id: string;
  name: string;
  display_name: string;
  base_model: string;
  created_at: string;
  status: string;
  size_gb: number;
  download_url: string;
  quantization: string;
  task_type: string;
}

export interface DownloadProgress {
  model_id: string;
  filename: string;
  downloaded_bytes: number;
  total_bytes: number;
  percent: number;
  done: boolean;
  error?: string;
}

export async function listCloudModels(apiKey: string): Promise<CloudModel[]> {
  return invoke<CloudModel[]>('list_cloud_models', { apiKey });
}

export async function downloadCloudModel(
  apiKey: string,
  modelId: string,
  downloadUrl: string,
  filename: string,
  onProgress: (p: DownloadProgress) => void
): Promise<void> {
  const unlisten = await listen<DownloadProgress>('model_download_progress', (event) => {
    if (event.payload.model_id === modelId) {
      onProgress(event.payload);
    }
  });

  try {
    await invoke('download_cloud_model', { apiKey, modelId, downloadUrl, filename });
  } finally {
    unlisten();
  }
}

export async function checkCloudConnection(apiKey: string): Promise<{ connected: boolean; email?: string; plan?: string }> {
  try {
    return await invoke<{ connected: boolean; email?: string; plan?: string }>('check_cloud_connection', { apiKey });
  } catch {
    return { connected: false };
  }
}
