/**
 * Models View Component
 * Unified Model Registry: Manage Local Vault and Remote Hub models like Ollama.
 */

import { useState, useEffect } from 'react';
import {
    Box,
    Download,
    Search,
    Filter,
    HardDrive,
    Network,
    Cpu,
    Layers,
    Zap,
    Play,
    Trash2,
    Database,
    Clock
} from 'lucide-react';
import { api, ModelInfo } from '../../services/api';
import './ModelsView.css';

// ============================================================================
// Types & Mock Data for Local Registry
// ============================================================================

interface LocalModel {
    id: string;
    registryTag: string; // e.g. "llama3:8b-instruct-q4_K_M"
    format: 'GGUF' | 'Safetensors' | 'PyTorch';
    quantization: string;
    sizeBytes: number;
    parameters: number;
    addedAt: string;
    status: 'idle' | 'serving';
}

const MOCK_LOCAL_VAULT: LocalModel[] = [
    {
        id: 'loc-1',
        registryTag: 'llama3:8b-instruct-q4_K_M',
        format: 'GGUF',
        quantization: '4-bit INT',
        sizeBytes: 4.7 * 1024 * 1024 * 1024,
        parameters: 8e9,
        addedAt: '2 days ago',
        status: 'idle'
    },
    {
        id: 'loc-2',
        registryTag: 'mixtral:8x7b-instruct-v0.1-q5_K_M',
        format: 'GGUF',
        quantization: '5-bit INT',
        sizeBytes: 32.2 * 1024 * 1024 * 1024,
        parameters: 47e9,
        addedAt: '1 week ago',
        status: 'serving'
    },
    {
        id: 'loc-3',
        registryTag: 'finetuned-customer-support.safetensors',
        format: 'Safetensors',
        quantization: 'FP16',
        sizeBytes: 15.0 * 1024 * 1024 * 1024,
        parameters: 7e9,
        addedAt: 'Just now',
        status: 'idle'
    }
];

// ============================================================================
// Formatting
// ============================================================================

function formatParams(params?: number): string {
    if (!params) return 'N/A';
    if (params >= 1e9) return `${(params / 1e9).toFixed(1)}B`;
    if (params >= 1e6) return `${(params / 1e6).toFixed(0)}M`;
    return params.toString();
}

function formatSize(bytes?: number): string {
    if (!bytes) return 'N/A';
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)} GB`;
}

// ============================================================================
// Main Component
// ============================================================================

export function ModelsView() {
    const [activeTab, setActiveTab] = useState<'vault'|'hub'>('vault');
    const [hubModels, setHubModels] = useState<ModelInfo[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        if (activeTab === 'hub') {
            loadHubModels();
        } else {
            setIsLoading(false); // Mock instant load for local
        }
    }, [activeTab]);

    async function loadHubModels() {
        try {
            setIsLoading(true);
            const response = await api.listModels();
            setHubModels(response.data);
        } catch (err) {
            console.error('Failed to load models:', err);
            setHubModels([]);
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <div className="models-view" style={{ backgroundColor: '#09090b', minHeight: '100%', padding: '32px' }}>
            {/* Standard Header with Registry Switcher */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '32px' }}>
                <div>
                    <h1 style={{ fontSize: '28px', fontWeight: 700, color: 'white', margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <Database size={28} className="text-accent-purple" />
                        Model Registry
                    </h1>
                    <p style={{ color: '#a1a1aa', margin: 0, fontSize: '14px' }}>Manage local foundation weights, adapters, and pull from standard inference Hubs.</p>
                </div>
                
                {/* Custom Tech Tab Switcher */}
                <div style={{ display: 'flex', background: '#18181b', padding: '4px', borderRadius: '8px', border: '1px solid #27272a' }}>
                    <button 
                        onClick={() => setActiveTab('vault')}
                        style={{
                            display: 'flex', alignItems: 'center', gap: '8px',
                            padding: '8px 16px', borderRadius: '6px', fontSize: '13px', fontWeight: 600, border: 'none', cursor: 'pointer',
                            background: activeTab === 'vault' ? '#27272a' : 'transparent',
                            color: activeTab === 'vault' ? '#fff' : '#a1a1aa',
                            boxShadow: activeTab === 'vault' ? '0 1px 3px rgba(0,0,0,0.3)' : 'none'
                        }}
                    >
                        <HardDrive size={16} /> Local Vault
                    </button>
                    <button 
                        onClick={() => setActiveTab('hub')}
                        style={{
                            display: 'flex', alignItems: 'center', gap: '8px',
                            padding: '8px 16px', borderRadius: '6px', fontSize: '13px', fontWeight: 600, border: 'none', cursor: 'pointer',
                            background: activeTab === 'hub' ? '#27272a' : 'transparent',
                            color: activeTab === 'hub' ? '#fff' : '#a1a1aa',
                            boxShadow: activeTab === 'hub' ? '0 1px 3px rgba(0,0,0,0.3)' : 'none'
                        }}
                    >
                        <Network size={16} /> Remote Hub
                    </button>
                </div>
            </div>

            {/* Toolbar */}
            <div style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '12px', background: '#18181b', padding: '0 16px', borderRadius: '8px', border: '1px solid #27272a' }}>
                    <Search size={18} color="#a1a1aa" />
                    <input
                        type="text"
                        placeholder="Search models or tags (e.g., llama3, gguf, safetensors)..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        style={{ background: 'transparent', border: 'none', outline: 'none', color: '#fff', fontSize: '14px', width: '100%', padding: '12px 0' }}
                    />
                </div>
                <button style={{ display: 'flex', alignItems: 'center', gap: '8px', background: '#18181b', border: '1px solid #27272a', padding: '0 20px', borderRadius: '8px', color: '#a1a1aa', cursor: 'pointer' }}>
                    <Filter size={16} /> Filters
                </button>
            </div>

            {/* Tab content rendering */}
            {activeTab === 'vault' ? (
                <div style={{ display: 'grid', gap: '16px' }}>
                    {MOCK_LOCAL_VAULT.map(localInfo => (
                        <div key={localInfo.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#18181b', padding: '24px', borderRadius: '12px', border: '1px solid #27272a' }}>
                            {/* Left Meta Group */}
                            <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
                                <div style={{ width: '48px', height: '48px', borderRadius: '8px', background: 'rgba(168, 85, 247, 0.1)', border: '1px solid rgba(168, 85, 247, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <Box size={24} color="#a855f7" />
                                </div>
                                <div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                                        <h3 style={{ margin: 0, fontSize: '18px', color: '#fff', fontFamily: 'monospace' }}>{localInfo.registryTag}</h3>
                                        <span style={{ fontSize: '11px', fontWeight: 600, padding: '2px 8px', borderRadius: '12px', background: '#27272a', color: '#a1a1aa', border: '1px solid #3f3f46' }}>
                                            {localInfo.format}
                                        </span>
                                        {localInfo.status === 'serving' && (
                                            <span style={{ fontSize: '11px', fontWeight: 600, padding: '2px 8px', borderRadius: '12px', background: 'rgba(0,255,133,0.1)', color: '#00FF85', border: '1px solid rgba(0,255,133,0.2)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                <span style={{ width: '6px', height: '6px', background: '#00FF85', borderRadius: '50%' }}></span> Serving
                                            </span>
                                        )}
                                    </div>
                                    <div style={{ display: 'flex', gap: '24px', color: '#a1a1aa', fontSize: '13px' }}>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Cpu size={14}/> {formatParams(localInfo.parameters)} Params</span>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Layers size={14}/> {localInfo.quantization}</span>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><HardDrive size={14}/> {formatSize(localInfo.sizeBytes)}</span>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Clock size={14}/> Added {localInfo.addedAt}</span>
                                    </div>
                                </div>
                            </div>
                            
                            {/* Vault action tools */}
                            <div style={{ display: 'flex', gap: '12px' }}>
                                <button style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: '#a1a1aa', background: 'transparent', border: 'none', padding: '8px 12px', borderRadius: '6px' }} className="hover:bg-zinc-800">
                                    <Trash2 size={16} />
                                </button>
                                {localInfo.status === 'serving' ? (
                                    <button style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', background: 'transparent', color: '#fff', border: '1px solid #3f3f46', padding: '8px 16px', borderRadius: '6px', fontWeight: 500 }}>
                                        Manage API
                                    </button>
                                ) : (
                                    <button style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', background: '#fff', color: '#000', border: 'none', padding: '8px 20px', borderRadius: '6px', fontWeight: 600 }}>
                                        <Play size={14} fill="currentColor" /> Serve (vLLM)
                                    </button>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: '20px' }}>
                    {isLoading ? (
                        <div style={{ color: '#a1a1aa' }}>Fetching remote models via Registry API...</div>
                    ) : (
                        hubModels.map(model => (
                            <div key={model.id} style={{ display: 'flex', flexDirection: 'column', background: '#18181b', padding: '24px', borderRadius: '12px', border: '1px solid #27272a' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                                    <h3 style={{ margin: 0, fontSize: '18px', color: '#fff', fontWeight: 600 }}>{model.name}</h3>
                                    {model.supportsFinetuning && <Zap size={16} color="#fbbf24" />}
                                </div>
                                <p style={{ color: '#a1a1aa', fontSize: '14px', lineHeight: 1.5, marginBottom: '24px', flex: 1 }}>{model.description}</p>
                                
                                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#a1a1aa', fontSize: '12px', marginBottom: '24px', background: '#27272a', padding: '12px', borderRadius: '8px' }}>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}><span>Params</span><strong style={{ color: '#fff' }}>{formatParams(model.parameters)}</strong></div>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}><span>Context</span><strong style={{ color: '#fff' }}>{model.contextLength?.toLocaleString() || '4k'}</strong></div>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}><span>Download</span><strong style={{ color: '#fff' }}>{formatSize(model.sizeBytes)}</strong></div>
                                </div>
                                <button style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', background: 'transparent', color: '#fff', border: '1px solid #3f3f46', padding: '10px', borderRadius: '6px', fontWeight: 600, cursor: 'pointer' }}>
                                    <Download size={16} /> Pull Model
                                </button>
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    );
}
