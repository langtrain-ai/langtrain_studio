/**
 * LocalModelsView — Manage offline GGUF models
 * 
 * Features:
 * - System hardware profile display
 * - Model catalog with download from HuggingFace
 * - Installed models with run/delete
 * - Inference server status
 * - Pro-only gating
 */

import { useState, useEffect, useCallback } from 'react';
import {
    Download, Trash2, Play, Square, HardDrive, Cpu, MonitorSmartphone,
    Zap, AlertTriangle, CheckCircle, XCircle, Loader2, Crown
} from 'lucide-react';
import {
    HardwareService, ModelManagerService, InferenceService, SubscriptionGate,
    MODEL_CATALOG,
    type SystemProfile, type LocalModel, type DownloadProgress, type ServerStatus, type ProAccessResult
} from '../../services/local';
import './LocalModelsView.css';

export function LocalModelsView() {
    const [hardware, setHardware] = useState<SystemProfile | null>(null);
    const [models, setModels] = useState<LocalModel[]>([]);
    const [server, setServer] = useState<ServerStatus | null>(null);
    const [proAccess, setProAccess] = useState<ProAccessResult | null>(null);
    const [downloading, setDownloading] = useState<string | null>(null);
    const [progress, setProgress] = useState<DownloadProgress | null>(null);
    const [loading, setLoading] = useState(true);
    const [tab, setTab] = useState<'installed' | 'catalog' | 'hardware'>('catalog');

    // Load all data
    const loadData = useCallback(async () => {
        setLoading(true);
        try {
            const [hw, modelList, status, access] = await Promise.all([
                HardwareService.detect(),
                ModelManagerService.list(),
                InferenceService.status(),
                SubscriptionGate.checkAccess(),
            ]);
            setHardware(hw);
            setModels(modelList);
            setServer(status);
            setProAccess(access);
        } catch (e) {
            console.error('Failed to load local data:', e);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadData();

        // Listen for download progress
        const unlisten1 = ModelManagerService.onProgress((p) => setProgress(p));
        const unlisten2 = ModelManagerService.onComplete(() => {
            setDownloading(null);
            setProgress(null);
            loadData();
        });

        return () => {
            unlisten1.then((fn) => fn());
            unlisten2.then((fn) => fn());
        };
    }, [loadData]);

    // Actions
    const handleDownload = async (url: string, filename: string) => {
        if (!proAccess?.allowed) return;
        setDownloading(filename);
        try {
            await ModelManagerService.download(url, filename);
        } catch (e: any) {
            console.error('Download failed:', e);
            setDownloading(null);
        }
    };

    const handleDelete = async (filename: string) => {
        if (!confirm(`Delete ${filename}? This cannot be undone.`)) return;
        try {
            await ModelManagerService.delete(filename);
            await loadData();
        } catch (e: any) {
            console.error('Delete failed:', e);
        }
    };

    const handleStart = async (modelPath: string) => {
        try {
            const status = await InferenceService.start(modelPath);
            setServer(status);
        } catch (e: any) {
            console.error('Failed to start server:', e);
        }
    };

    const handleStop = async () => {
        try {
            const status = await InferenceService.stop();
            setServer(status);
        } catch (e: any) {
            console.error('Failed to stop server:', e);
        }
    };

    if (loading) {
        return (
            <div className="local-models-loading">
                <Loader2 className="spinner" size={24} />
                <span>Detecting hardware...</span>
            </div>
        );
    }

    // Pro gate overlay
    if (proAccess && !proAccess.allowed) {
        return (
            <div className="local-models-page">
                <div className="pro-gate-overlay">
                    <Crown size={48} className="pro-icon" />
                    <h2>Pro Feature</h2>
                    <p>{proAccess.message}</p>
                    <a href="https://app.langtrain.xyz/billing" target="_blank" rel="noopener" className="upgrade-btn">
                        Upgrade to Pro
                    </a>
                </div>
            </div>
        );
    }

    const installedFilenames = new Set(models.map((m) => m.filename));

    return (
        <div className="local-models-page">
            {/* Header */}
            <div className="lm-header">
                <div>
                    <h1>Local Models</h1>
                    <p className="subtitle">Run AI models offline — no internet required</p>
                </div>
                {server?.running && (
                    <div className="server-badge running">
                        <Zap size={14} />
                        <span>Server running on :{server.port}</span>
                        <button onClick={handleStop} className="stop-btn">
                            <Square size={12} /> Stop
                        </button>
                    </div>
                )}
            </div>

            {/* Tabs */}
            <div className="lm-tabs">
                <button className={tab === 'catalog' ? 'active' : ''} onClick={() => setTab('catalog')}>
                    <Download size={16} /> Model Catalog
                </button>
                <button className={tab === 'installed' ? 'active' : ''} onClick={() => setTab('installed')}>
                    <HardDrive size={16} /> Installed ({models.length})
                </button>
                <button className={tab === 'hardware' ? 'active' : ''} onClick={() => setTab('hardware')}>
                    <Cpu size={16} /> System
                </button>
            </div>

            {/* Download Progress */}
            {downloading && progress && (
                <div className="download-banner">
                    <Loader2 className="spinner" size={16} />
                    <span>Downloading {progress.model_name}</span>
                    <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
                    </div>
                    <span className="progress-text">
                        {progress.percent}% • {progress.speed_mbps} MB/s
                    </span>
                </div>
            )}

            {/* Catalog Tab */}
            {tab === 'catalog' && (
                <div className="models-grid">
                    {MODEL_CATALOG.map((model) => {
                        const isInstalled = installedFilenames.has(model.filename);
                        const isDownloading = downloading === model.filename;
                        const meetsReqs = hardware && hardware.ram_total_gb >= model.min_ram_gb;

                        return (
                            <div key={model.id} className={`model-card ${isInstalled ? 'installed' : ''}`}>
                                <div className="model-card-header">
                                    <h3>{model.name}</h3>
                                    <span className="param-badge">{model.parameters}</span>
                                </div>
                                <p className="model-desc">{model.description}</p>
                                <div className="model-meta">
                                    <span>{model.quantization}</span>
                                    <span>{model.size_gb} GB</span>
                                    <span>Min {model.min_ram_gb}GB RAM</span>
                                </div>
                                <div className="model-tags">
                                    {model.recommended_for.map((tag) => (
                                        <span key={tag} className="tag">{tag}</span>
                                    ))}
                                </div>
                                <div className="model-actions">
                                    {isInstalled ? (
                                        <span className="installed-badge">
                                            <CheckCircle size={14} /> Installed
                                        </span>
                                    ) : isDownloading ? (
                                        <button disabled className="btn-download">
                                            <Loader2 className="spinner" size={14} /> Downloading...
                                        </button>
                                    ) : (
                                        <button
                                            onClick={() => handleDownload(model.download_url, model.filename)}
                                            className="btn-download"
                                            disabled={!meetsReqs}
                                        >
                                            {meetsReqs ? (
                                                <><Download size={14} /> Download</>
                                            ) : (
                                                <><AlertTriangle size={14} /> Insufficient RAM</>
                                            )}
                                        </button>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Installed Tab */}
            {tab === 'installed' && (
                <div className="installed-list">
                    {models.length === 0 ? (
                        <div className="empty-state">
                            <HardDrive size={48} />
                            <h3>No models installed</h3>
                            <p>Download a model from the catalog to get started</p>
                            <button onClick={() => setTab('catalog')} className="btn-primary">
                                Browse Catalog
                            </button>
                        </div>
                    ) : (
                        models.map((model) => (
                            <div key={model.filename} className="installed-row">
                                <div className="installed-info">
                                    <h4>{model.name}</h4>
                                    <div className="installed-meta">
                                        <span>{model.parameters}</span>
                                        <span>{model.quantization}</span>
                                        <span>{model.size_gb} GB</span>
                                    </div>
                                </div>
                                <div className="installed-actions">
                                    {server?.running && server.model === model.path ? (
                                        <button onClick={handleStop} className="btn-running">
                                            <Square size={14} /> Stop
                                        </button>
                                    ) : (
                                        <button
                                            onClick={() => handleStart(model.path)}
                                            className="btn-run"
                                            disabled={server?.running === true}
                                        >
                                            <Play size={14} /> Run
                                        </button>
                                    )}
                                    <button onClick={() => handleDelete(model.filename)} className="btn-delete">
                                        <Trash2 size={14} />
                                    </button>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            )}

            {/* Hardware Tab */}
            {tab === 'hardware' && hardware && (
                <div className="hardware-panel">
                    <div className="hw-grid">
                        <div className="hw-card">
                            <Cpu size={24} />
                            <h4>CPU</h4>
                            <p className="hw-value">{hardware.cpu_name}</p>
                            <p className="hw-detail">{hardware.cpu_cores} cores • {hardware.cpu_arch}</p>
                        </div>
                        <div className="hw-card">
                            <MonitorSmartphone size={24} />
                            <h4>Memory</h4>
                            <p className="hw-value">{hardware.ram_total_gb} GB</p>
                            <p className="hw-detail">{hardware.ram_available_gb} GB available</p>
                        </div>
                        <div className="hw-card">
                            <Zap size={24} />
                            <h4>GPU</h4>
                            <p className="hw-value">{hardware.gpu_name}</p>
                            <p className="hw-detail">
                                {hardware.has_gpu
                                    ? `${hardware.gpu_vram_gb} GB VRAM • ${hardware.gpu_platform}`
                                    : 'No dedicated GPU'}
                            </p>
                        </div>
                        <div className="hw-card">
                            <HardDrive size={24} />
                            <h4>Storage</h4>
                            <p className="hw-value">{hardware.disk_free_gb} GB free</p>
                            <p className="hw-detail">{hardware.models_dir}</p>
                        </div>
                    </div>

                    <div className="capability-cards">
                        <div className={`cap-card ${hardware.can_run_inference ? 'capable' : 'limited'}`}>
                            {hardware.can_run_inference ? <CheckCircle size={20} /> : <XCircle size={20} />}
                            <div>
                                <h4>Local Inference</h4>
                                <p>{hardware.inference_note}</p>
                            </div>
                        </div>
                        <div className={`cap-card ${hardware.can_finetune ? 'capable' : 'limited'}`}>
                            {hardware.can_finetune ? <CheckCircle size={20} /> : <XCircle size={20} />}
                            <div>
                                <h4>Local Fine-Tuning</h4>
                                <p>{hardware.finetune_note}</p>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
