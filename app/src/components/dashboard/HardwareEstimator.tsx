/**
 * HardwareEstimator
 * 
 * Fetches real hardware specs from the /hardware/detect backend endpoint.
 * Falls back to mock profiles when the API is unreachable (offline mode).
 */

import { useState, useEffect } from 'react';
import { Cpu, MemoryStick, Zap, BrainCircuit, Check, X, Activity, RefreshCw, Wifi, WifiOff } from 'lucide-react';
import { detectHardware, HardwareInfo } from '../../services/api';

// ── Model compatibility matrix ──────────────────────────────────────────────

const MODELS = [
    { name: 'Llama 3 8B',    parameters: 8,  baseVRAM: 16,  q4VRAM: 6  },
    { name: 'Mistral Nemo',  parameters: 12, baseVRAM: 24,  q4VRAM: 8  },
    { name: 'Mixtral 8x7B', parameters: 47, baseVRAM: 90,  q4VRAM: 32 },
    { name: 'Llama 3 70B',  parameters: 70, baseVRAM: 140, q4VRAM: 40 },
    { name: 'Mistral 7B',   parameters: 7,  baseVRAM: 14,  q4VRAM: 5  },
    { name: 'Qwen 2 72B',   parameters: 72, baseVRAM: 144, q4VRAM: 42 },
];

// Offline fallback profiles keyed by platform
const FALLBACK_PROFILES: Record<string, HardwareInfo> = {
    nvidia: {
        gpu_name: 'NVIDIA RTX 4090',
        vram_mb: 24576,
        vram_gb: 24,
        tflops_fp16: 82.6,
        bandwidth_gbps: 1008,
        compute_capability: '8.9',
        driver_version: null,
        platform: 'nvidia',
        has_gpu: true,
    },
    apple_silicon: {
        gpu_name: 'Apple M3 Max',
        vram_mb: 131072,
        vram_gb: 128,
        tflops_fp16: 40.5,
        bandwidth_gbps: 400,
        compute_capability: null,
        driver_version: null,
        platform: 'apple_silicon',
        has_gpu: true,
    },
    cpu: {
        gpu_name: 'CPU Only',
        vram_mb: 0,
        vram_gb: 0,
        tflops_fp16: null,
        bandwidth_gbps: null,
        compute_capability: null,
        driver_version: null,
        platform: 'cpu',
        has_gpu: false,
    },
};

// Platform display helpers
function platformLabel(platform: string) {
    if (platform === 'nvidia') return 'NVIDIA CUDA';
    if (platform === 'apple_silicon') return 'Apple Silicon';
    return 'CPU';
}

function platformColor(platform: string) {
    if (platform === 'nvidia') return '#76b900'; // NVIDIA green
    if (platform === 'apple_silicon') return '#A2AAAD'; // Apple grey
    return '#a1a1aa';
}

// ── Main Component ───────────────────────────────────────────────────────────

export const HardwareEstimator: React.FC = () => {
    const [hw, setHw] = useState<HardwareInfo | null>(null);
    const [loading, setLoading] = useState(true);
    const [offline, setOffline] = useState(false);

    async function loadHardware() {
        setLoading(true);
        setOffline(false);
        try {
            const data = await detectHardware();
            setHw(data);
        } catch {
            // API unreachable — use offline fallback
            setOffline(true);
            // Try to guess platform from user agent
            const ua = navigator.userAgent.toLowerCase();
            const platform = ua.includes('mac') ? 'apple_silicon' : 'nvidia';
            setHw(FALLBACK_PROFILES[platform]);
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => { loadHardware(); }, []);

    if (loading) {
        return (
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: '#a1a1aa', padding: '24px' }}>
                <RefreshCw size={16} style={{ animation: 'spin 1s linear infinite' }} />
                <span style={{ fontSize: '14px' }}>Detecting local hardware...</span>
            </div>
        );
    }

    if (!hw) return null;

    const vramGB = hw.vram_gb;

    return (
        <div style={{ backgroundColor: '#111111', border: '1px solid #1f1f1f', borderRadius: '8px', padding: '24px', color: '#E0E0E0', fontFamily: 'system-ui, sans-serif' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                <div>
                    <h2 style={{ margin: 0, fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <BrainCircuit size={20} style={{ color: '#a855f7' }} />
                        LLMFit Hardware Estimator
                    </h2>
                    <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: '#888888' }}>
                        Live hardware analysis — model compatibility based on your actual VRAM.
                    </p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    {offline ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#fbbf24' }}>
                            <WifiOff size={12} /> Offline (estimated)
                        </div>
                    ) : (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#00FF85' }}>
                            <Wifi size={12} /> Live Data
                        </div>
                    )}
                    <button
                        onClick={loadHardware}
                        style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px', padding: '6px 12px', color: '#a1a1aa', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}
                    >
                        <RefreshCw size={12} /> Refresh
                    </button>
                </div>
            </div>

            {/* Hardware Stats Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '32px' }}>
                <StatCard
                    icon={<Cpu size={16} />}
                    label="Architecture"
                    value={hw.gpu_name}
                    sub={platformLabel(hw.platform)}
                    color={platformColor(hw.platform)}
                />
                <StatCard
                    icon={<MemoryStick size={16} />}
                    label="VRAM"
                    value={hw.has_gpu ? `${vramGB} GB` : 'N/A'}
                    sub={hw.vram_mb > 0 ? `${hw.vram_mb.toLocaleString()} MB` : 'CPU training only'}
                    color="#00FF85"
                />
                <StatCard
                    icon={<Zap size={16} />}
                    label="Compute (FP16)"
                    value={hw.tflops_fp16 ? `${hw.tflops_fp16} TF` : 'N/A'}
                    sub="TFLOPS"
                    color="#FFD600"
                />
                <StatCard
                    icon={<Activity size={16} />}
                    label="Bandwidth"
                    value={hw.bandwidth_gbps ? `${hw.bandwidth_gbps} GB/s` : 'N/A'}
                    sub={hw.compute_capability ? `CC ${hw.compute_capability}` : (hw.driver_version || '—')}
                    color="#00C8FF"
                />
            </div>

            {/* Compatibility Matrix */}
            <h3 style={{ fontSize: '12px', marginBottom: '12px', color: '#666', borderBottom: '1px solid #1f1f1f', paddingBottom: '8px', letterSpacing: '1px' }}>
                LOCAL COMPATIBILITY MATRIX — {vramGB} GB VRAM
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {MODELS.map(model => {
                    const fp16Fits = vramGB >= model.baseVRAM;
                    const q4Fits = vramGB >= model.q4VRAM;
                    return (
                        <div key={model.name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
                            <div style={{ width: '200px' }}>
                                <div style={{ fontSize: '14px', fontWeight: 'bold' }}>{model.name}</div>
                                <div style={{ fontSize: '11px', color: '#666' }}>{model.parameters}B Params</div>
                            </div>

                            <div style={{ flex: 1, display: 'flex', justifyContent: 'space-around' }}>
                                <CompatBadge label="FP16 Weights" fits={fp16Fits} vram={model.baseVRAM} />
                                <CompatBadge label="INT4 Quantized" fits={q4Fits} vram={model.q4VRAM} />
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

// ── Sub-components ───────────────────────────────────────────────────────────

function StatCard({ icon, label, value, sub, color }: { icon: React.ReactNode; label: string; value: string; sub: string; color: string }) {
    return (
        <div style={{ padding: '16px', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color, fontSize: '12px', marginBottom: '10px' }}>
                {icon} {label}
            </div>
            <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '4px' }}>{value}</div>
            <div style={{ fontSize: '11px', color: '#555' }}>{sub}</div>
        </div>
    );
}

function CompatBadge({ label, fits, vram }: { label: string; fits: boolean; vram: number }) {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px', width: '120px' }}>
            <span style={{ fontSize: '10px', color: '#555' }}>{label}</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: fits ? '#00FF85' : '#FF4C4C', fontSize: '12px', fontWeight: 'bold' }}>
                {fits ? <Check size={14} /> : <X size={14} />}
                {fits ? `FIT (${vram}g)` : `OOM (${vram}g)`}
            </div>
        </div>
    );
}
