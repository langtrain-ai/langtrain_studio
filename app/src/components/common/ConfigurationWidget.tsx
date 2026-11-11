import { Layers, Activity, Zap, Beaker } from 'lucide-react';
import './ConfigurationWidget.css';


export interface TrainingConfig {
    mode: 'local' | 'cloud' | 'hybrid';
    learningRate: number;
    epochs: number;
    batchSize: number;
    useMixedPrecision: boolean;
    loraRank: number;
    loraAlpha: number;
}

interface ConfigurationWidgetProps {
    config: TrainingConfig;
    setConfig: (config: TrainingConfig) => void;
    onConfirm: () => void;
}

export function ConfigurationWidget({ config, setConfig, onConfirm }: ConfigurationWidgetProps) {
    const handleChange = (key: keyof TrainingConfig, value: any) => {
        setConfig({ ...config, [key]: value });
    };

    return (
        <div className="config-widget">
            <div className="config-widget__header">
                <div className="config-widget__title">
                    <Beaker size={16} className="text-accent-cyan" />
                    <span>Training Configuration</span>
                </div>
                <span className="config-widget__subtitle">Advanced</span>
            </div>

            <div className="config-widget__content">
                {/* Compute Mode Selection */}
                <div className="config-section">
                    <div className="config-section__label">
                        <Activity size={14} /> Compute Backend
                    </div>
                    <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                        <button
                            className={`tab-button ${config.mode === 'cloud' ? 'tab-button--active' : ''}`}
                            onClick={() => handleChange('mode', 'cloud')}
                            style={{ flex: 1, padding: '8px', justifyContent: 'center' }}
                        >
                            Cloud
                        </button>
                        <button
                            className={`tab-button ${config.mode === 'local' ? 'tab-button--active' : ''}`}
                            onClick={() => handleChange('mode', 'local')}
                            style={{ flex: 1, padding: '8px', justifyContent: 'center' }}
                        >
                            Local
                        </button>
                        <button
                            className={`tab-button ${config.mode === 'hybrid' ? 'tab-button--active' : ''}`}
                            onClick={() => handleChange('mode', 'hybrid')}
                            style={{ flex: 1, padding: '8px', justifyContent: 'center', borderColor: config.mode === 'hybrid' ? 'var(--accent-purple)' : undefined, color: config.mode === 'hybrid' ? 'var(--accent-purple)' : undefined }}
                        >
                            Hybrid
                        </button>
                    </div>

                    {/* Hardware Detection (Local & Hybrid) */}
                    {(config.mode === 'local' || config.mode === 'hybrid') && (
                        <div style={{
                            padding: '12px',
                            backgroundColor: 'rgba(255,255,255,0.03)',
                            border: config.mode === 'hybrid' ? '1px dashed var(--accent-purple)' : '1px solid var(--border-color)',
                            borderRadius: '8px',
                            display: 'flex',
                            gap: '12px',
                            marginBottom: '16px',
                            alignItems: 'center'
                        }}>
                            <img 
                                src="/assets/hardware/nvidia-consumer.jpg" 
                                alt="Detected Local Hardware" 
                                style={{ width: '64px', height: '64px', objectFit: 'cover', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.1)' }} 
                            />
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>NVIDIA RTX 4090 Detected</div>
                                <div style={{ fontSize: '12px', color: 'var(--accent-green)', fontWeight: 500 }}>24 GB VRAM Available</div>
                                <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>Compute Capability 8.9</div>
                            </div>
                        </div>
                    )}
                    
                    {config.mode === 'hybrid' && (
                        <div style={{ padding: '12px', backgroundColor: 'rgba(168, 85, 247, 0.05)', border: '1px solid rgba(168, 85, 247, 0.2)', borderRadius: '8px', marginBottom: '16px' }}>
                            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-purple)', marginBottom: '8px' }}>🚀 HYBRID ARCHITECTURE TARGET</div>
                            <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                                Base model weights will be federated on Cloud GPUs.
                                LoRA adapter computations will be evaluated securely against Local VRAM.
                            </p>
                        </div>
                    )}
                </div>

                {/* LoRA Settings */}
                <div className="config-section">
                    <div className="config-section__label">
                        <Layers size={14} /> Adapter Config
                    </div>
                    <div className="config-grid">
                        <div className="config-field">
                            <label>LoRA Rank (r)</label>
                            <input
                                type="number"
                                className="config-input"
                                value={config.loraRank}
                                onChange={(e) => handleChange("loraRank", parseInt(e.target.value) || 0)}
                            />
                        </div>
                        <div className="config-field">
                            <label>LoRA Alpha</label>
                            <input
                                type="number"
                                className="config-input"
                                value={config.loraAlpha}
                                onChange={(e) => handleChange("loraAlpha", parseInt(e.target.value) || 0)}
                            />
                        </div>
                    </div>
                </div>

                {/* Training Loop */}
                <div className="config-section">
                    <div className="config-section__label">
                        <Activity size={14} /> Training Loop
                    </div>

                    <div className="config-slider-container">
                        <div className="config-slider-header">
                            <label>Learning Rate</label>
                            <span className="config-value">{config.learningRate}</span>
                        </div>
                        <input
                            type="range"
                            className="config-slider"
                            min="1"
                            max="50"
                            value={config.learningRate * 100000}
                            onChange={(e) => handleChange("learningRate", parseInt(e.target.value) / 100000)}
                        />
                    </div>

                    <div className="config-grid">
                        <div className="config-field">
                            <label>Epochs</label>
                            <input
                                type="number"
                                className="config-input"
                                value={config.epochs}
                                onChange={(e) => handleChange("epochs", parseInt(e.target.value) || 1)}
                            />
                        </div>
                        <div className="config-field">
                            <label>Batch Size</label>
                            <input
                                type="number"
                                className="config-input"
                                value={config.batchSize}
                                onChange={(e) => handleChange("batchSize", parseInt(e.target.value) || 1)}
                            />
                        </div>
                    </div>
                </div>

                {/* Optimization */}
                <div className="config-switch">
                    <div className="switch-label">
                        <Zap size={16} style={{ color: 'var(--accent-purple)' }} />
                        <div>
                            <div className="switch-text">Mixed Precision (FP16)</div>
                            <span className="switch-subtext">Faster training, less memory</span>
                        </div>
                    </div>
                    <input
                        type="checkbox"
                        className="switch-input"
                        checked={config.useMixedPrecision}
                        onChange={(e) => handleChange("useMixedPrecision", e.target.checked)}
                    />
                </div>

                <div className="config-actions">
                    <button className="btn-apply" onClick={onConfirm}>
                        Apply Configuration
                    </button>
                </div>

            </div>
        </div>
    );
}

export const defaultConfig: TrainingConfig = {
    mode: 'local',
    learningRate: 2e-4,
    epochs: 3,
    batchSize: 4,
    useMixedPrecision: true,
    loraRank: 16,
    loraAlpha: 32
};
