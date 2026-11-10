/**
 * Local Services — TypeScript wrappers for Tauri Rust commands
 * 
 * Provides: HardwareService, ModelManagerService, InferenceService, SubscriptionGate
 */

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

// ==================== Types ====================

export interface SystemProfile {
    cpu_name: string;
    cpu_cores: number;
    cpu_arch: string;
    ram_total_gb: number;
    ram_available_gb: number;
    gpu_name: string;
    gpu_vram_gb: number;
    gpu_platform: 'nvidia' | 'apple_silicon' | 'amd' | 'cpu';
    has_gpu: boolean;
    models_dir: string;
    disk_free_gb: number;
    can_run_inference: boolean;
    can_finetune: boolean;
    inference_note: string;
    finetune_note: string;
}

export interface LocalModel {
    name: string;
    filename: string;
    path: string;
    size_bytes: number;
    size_gb: number;
    quantization: string;
    parameters: string;
}

export interface DownloadProgress {
    model_name: string;
    downloaded_bytes: number;
    total_bytes: number;
    percent: number;
    speed_mbps: number;
}

export interface ServerStatus {
    running: boolean;
    model: string | null;
    port: number;
    endpoint: string | null;
}

export interface ProAccessResult {
    allowed: boolean;
    plan: string;
    source: 'live' | 'cached' | 'denied';
    message: string;
    days_until_expiry: number | null;
}

// ==================== Hardware Service ====================

export const HardwareService = {
    async detect(): Promise<SystemProfile> {
        return invoke<SystemProfile>('detect_hardware');
    },
};

// ==================== Model Manager Service ====================

export const ModelManagerService = {
    async list(): Promise<LocalModel[]> {
        return invoke<LocalModel[]>('list_local_models');
    },

    async delete(filename: string): Promise<void> {
        return invoke('delete_local_model', { filename });
    },

    async download(url: string, filename: string): Promise<LocalModel> {
        return invoke<LocalModel>('download_model', { url, filename });
    },

    async getModelsDir(): Promise<string> {
        return invoke<string>('get_models_dir');
    },

    onProgress(callback: (progress: DownloadProgress) => void) {
        return listen<DownloadProgress>('download-progress', (event) => {
            callback(event.payload);
        });
    },

    onComplete(callback: (model: LocalModel) => void) {
        return listen<LocalModel>('download-complete', (event) => {
            callback(event.payload);
        });
    },
};

// ==================== Inference Service ====================

export const InferenceService = {
    async start(
        modelPath: string,
        options?: { port?: number; contextSize?: number; gpuLayers?: number }
    ): Promise<ServerStatus> {
        return invoke<ServerStatus>('start_inference_server', {
            modelPath,
            port: options?.port,
            contextSize: options?.contextSize,
            gpuLayers: options?.gpuLayers,
        });
    },

    async stop(): Promise<ServerStatus> {
        return invoke<ServerStatus>('stop_inference_server');
    },

    async status(): Promise<ServerStatus> {
        return invoke<ServerStatus>('inference_server_status');
    },

    /**
     * Send a chat completion to the local inference server.
     * Uses the OpenAI-compatible endpoint exposed by llama.cpp.
     */
    async chat(
        messages: Array<{ role: string; content: string }>,
        options?: { temperature?: number; maxTokens?: number; port?: number }
    ): Promise<{ content: string; model: string; tokens: number }> {
        const port = options?.port ?? 11435;
        const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages,
                temperature: options?.temperature ?? 0.7,
                max_tokens: options?.maxTokens ?? 2048,
                stream: false,
            }),
        });

        if (!response.ok) {
            throw new Error(`Local inference failed: ${response.statusText}`);
        }

        const data = await response.json();
        const choice = data.choices?.[0];

        return {
            content: choice?.message?.content ?? '',
            model: data.model ?? 'local',
            tokens: data.usage?.total_tokens ?? 0,
        };
    },
};

// ==================== Subscription Gate ====================

export const SubscriptionGate = {
    async checkAccess(authToken?: string): Promise<ProAccessResult> {
        return invoke<ProAccessResult>('check_pro_access', {
            authToken: authToken ?? null,
        });
    },
};

// ==================== Model Catalog ====================
// Pre-defined models available for download

export interface CatalogModel {
    id: string;
    name: string;
    parameters: string;
    quantization: string;
    size_gb: number;
    description: string;
    download_url: string;
    filename: string;
    min_ram_gb: number;
    recommended_for: string[];
}

export const MODEL_CATALOG: CatalogModel[] = [
    {
        id: 'tinyllama-1.1b-q4',
        name: 'TinyLlama 1.1B',
        parameters: '1.1B',
        quantization: 'Q4_K_M',
        size_gb: 0.67,
        description: 'Fast, lightweight model for basic tasks and testing',
        download_url: 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
        filename: 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
        min_ram_gb: 4,
        recommended_for: ['testing', 'simple-tasks'],
    },
    {
        id: 'llama-3.2-3b-q4',
        name: 'Llama 3.2 3B',
        parameters: '3B',
        quantization: 'Q4_K_M',
        size_gb: 2.0,
        description: 'Compact model with strong reasoning for its size',
        download_url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
        filename: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf',
        min_ram_gb: 6,
        recommended_for: ['general', 'coding', 'chat'],
    },
    {
        id: 'llama-3.1-8b-q4',
        name: 'Llama 3.1 8B',
        parameters: '8B',
        quantization: 'Q4_K_M',
        size_gb: 4.9,
        description: 'Best balance of quality and speed for most tasks',
        download_url: 'https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        filename: 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        min_ram_gb: 8,
        recommended_for: ['general', 'coding', 'analysis', 'agents'],
    },
    {
        id: 'qwen2.5-7b-q4',
        name: 'Qwen 2.5 7B',
        parameters: '7B',
        quantization: 'Q4_K_M',
        size_gb: 4.7,
        description: 'Strong multilingual model with excellent coding ability',
        download_url: 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf',
        filename: 'qwen2.5-7b-instruct-q4_k_m.gguf',
        min_ram_gb: 8,
        recommended_for: ['coding', 'multilingual', 'analysis'],
    },
    {
        id: 'mistral-7b-q4',
        name: 'Mistral 7B v0.3',
        parameters: '7B',
        quantization: 'Q4_K_M',
        size_gb: 4.4,
        description: 'Fast, capable model with 32K context window',
        download_url: 'https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf',
        filename: 'Mistral-7B-Instruct-v0.3-Q4_K_M.gguf',
        min_ram_gb: 8,
        recommended_for: ['general', 'creative', 'analysis'],
    },
];
