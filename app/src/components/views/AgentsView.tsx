/**
 * Agents View Component
 * Agent configuration and management for desktop
 */

import { useState } from 'react';
import {
    Code2,
    FileText,
    MessageSquare,
    BarChart3,
    PenTool,
    Search,
    Languages,
    Database,
    Settings,
    Zap,
    Brain,
    Sparkles,
    X,
    ChevronRight
} from 'lucide-react';
import './AgentsView.css';

// Predefined Agents (matching web app)
const PREDEFINED_AGENTS = [
    { id: 'code-assistant', name: 'Code Assistant', description: 'AI-powered coding assistant', icon: Code2, category: 'Development' },
    { id: 'text-summarizer', name: 'Text Summarizer', description: 'Summarize documents and articles', icon: FileText, category: 'Productivity' },
    { id: 'chat-agent', name: 'Chat Agent', description: 'Versatile conversational AI', icon: MessageSquare, category: 'Communication' },
    { id: 'data-analyzer', name: 'Data Analyzer', description: 'Analyze datasets and generate insights', icon: BarChart3, category: 'Analytics' },
    { id: 'content-writer', name: 'Content Writer', description: 'Generate blog posts and marketing copy', icon: PenTool, category: 'Content' },
    { id: 'research-assistant', name: 'Research Assistant', description: 'Help with research tasks', icon: Search, category: 'Research' },
    { id: 'translator', name: 'Language Translator', description: 'Translate between languages', icon: Languages, category: 'Productivity' },
    { id: 'sql-assistant', name: 'SQL Assistant', description: 'Generate and optimize SQL queries', icon: Database, category: 'Development' },
];

const OPEN_SOURCE_MODELS = [
    { id: 'llama-3.1-8b-instruct', name: 'Llama 3.1 8B Instruct' },
    { id: 'llama-3.1-70b-instruct', name: 'Llama 3.1 70B Instruct' },
    { id: 'mistral-7b-v0.3', name: 'Mistral 7B v0.3' },
    { id: 'mixtral-8x7b-v0.1', name: 'Mixtral 8x7B v0.1' },
    { id: 'qwen-2.5-72b', name: 'Qwen 2.5 72B' },
];

const PLANNING_STRATEGIES = [
    { id: 'standard', label: 'Standard ReAct', description: 'Balanced reasoning and action loop', icon: Brain },
    { id: 'researcher', label: 'Researcher', description: 'Search-first with high reflection', icon: Search },
    { id: 'direct', label: 'Direct Responder', description: 'Fast answers, minimal steps', icon: Zap },
    { id: 'creative', label: 'Creative', description: 'Expansive thinking, high temperature', icon: Sparkles },
];

interface AgentConfig {
    strategy: string;
    memory_enabled: boolean;
    reflection_enabled: boolean;
    voice_enabled: boolean;
    max_iterations: number;
    temperature: number;
    max_tokens: number;
    system_prompt: string;
    model: string;
}

interface Agent {
    id: string;
    name: string;
    description: string;
    icon: React.ComponentType<{ size?: number; className?: string }>;
    category: string;
}

export function AgentsView() {
    const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
    const [config, setConfig] = useState<AgentConfig>({
        strategy: 'standard',
        memory_enabled: true,
        reflection_enabled: true,
        voice_enabled: false,
        max_iterations: 10,
        temperature: 0.7,
        max_tokens: 2048,
        system_prompt: '',
        model: 'llama-3.1-8b-instruct'
    });

    const handleSave = () => {
        console.log('Saving agent config:', config);
        // TODO: Save to backend or local storage
    };

    return (
        <div className="agents-view">
            {/* Main Content - Agent Grid */}
            <div className="agents-view__main">
                <div className="agents-header">
                    <h1>Agent Studio</h1>
                    <p className="agents-subtitle">Select an agent to manage settings</p>
                </div>

                <div className="agents-grid">
                    {PREDEFINED_AGENTS.map(agent => {
                        const Icon = agent.icon;
                        return (
                            <button
                                key={agent.id}
                                className={`agent-card ${selectedAgent?.id === agent.id ? 'agent-card--selected' : ''}`}
                                onClick={() => setSelectedAgent(agent)}
                            >
                                <div className="agent-card__icon">
                                    <Icon size={24} />
                                </div>
                                <div className="agent-card__content">
                                    <span className="agent-card__name">{agent.name}</span>
                                    <span className="agent-card__description">{agent.description}</span>
                                </div>
                                <ChevronRight size={16} className="agent-card__arrow" />
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Configuration Sidebar */}
            {selectedAgent && (
                <div className="agents-view__sidebar">
                    <div className="sidebar-header">
                        <div className="sidebar-header__title">
                            <Settings size={16} className="sidebar-header__icon" />
                            <span>Settings</span>
                        </div>
                        <button className="icon-button" onClick={() => setSelectedAgent(null)}>
                            <X size={16} />
                        </button>
                    </div>

                    <div className="sidebar-content">
                        {/* Agent Info */}
                        <div className="config-section config-section--agent-info">
                            <div className="agent-info__name">{selectedAgent.name}</div>
                            <div className="agent-info__description">{selectedAgent.description}</div>
                        </div>

                        {/* Planning Strategy */}
                        <div className="config-section">
                            <div className="section-label">Planning Strategy</div>
                            <div className="strategy-grid">
                                {PLANNING_STRATEGIES.map(strategy => {
                                    const Icon = strategy.icon;
                                    return (
                                        <button
                                            key={strategy.id}
                                            className={`strategy-card ${config.strategy === strategy.id ? 'strategy-card--selected' : ''}`}
                                            onClick={() => setConfig({ ...config, strategy: strategy.id })}
                                        >
                                            <Icon size={14} className="strategy-card__icon" />
                                            <span className="strategy-card__label">{strategy.label}</span>
                                        </button>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Capabilities */}
                        <div className="config-section">
                            <div className="section-label">Capabilities</div>
                            <div className="capability-toggles">
                                <label className="toggle-row">
                                    <div className="toggle-row__info">
                                        <Brain size={14} className="toggle-icon toggle-icon--purple" />
                                        <span>Long-term Memory</span>
                                    </div>
                                    <input
                                        type="checkbox"
                                        checked={config.memory_enabled}
                                        onChange={(e) => setConfig({ ...config, memory_enabled: e.target.checked })}
                                    />
                                </label>
                                <label className="toggle-row">
                                    <div className="toggle-row__info">
                                        <Sparkles size={14} className="toggle-icon toggle-icon--yellow" />
                                        <span>Self Reflection</span>
                                    </div>
                                    <input
                                        type="checkbox"
                                        checked={config.reflection_enabled}
                                        onChange={(e) => setConfig({ ...config, reflection_enabled: e.target.checked })}
                                    />
                                </label>
                                <label className="toggle-row">
                                    <div className="toggle-row__info">
                                        <Zap size={14} className="toggle-icon toggle-icon--rose" />
                                        <span>Voice Capabilities</span>
                                    </div>
                                    <input
                                        type="checkbox"
                                        checked={config.voice_enabled}
                                        onChange={(e) => setConfig({ ...config, voice_enabled: e.target.checked })}
                                    />
                                </label>
                            </div>
                        </div>

                        {/* Model Settings */}
                        <div className="config-section">
                            <div className="section-label">Model Settings</div>

                            <div className="input-group">
                                <label className="input-label">System Prompt</label>
                                <textarea
                                    value={config.system_prompt}
                                    onChange={(e) => setConfig({ ...config, system_prompt: e.target.value })}
                                    placeholder="You are a helpful AI assistant..."
                                    className="textarea-input"
                                    rows={3}
                                />
                            </div>

                            <div className="input-group">
                                <label className="input-label">Base Model</label>
                                <select
                                    value={config.model}
                                    onChange={(e) => setConfig({ ...config, model: e.target.value })}
                                    className="select-input"
                                >
                                    {OPEN_SOURCE_MODELS.map(model => (
                                        <option key={model.id} value={model.id}>
                                            {model.name}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div className="input-group">
                                <div className="slider-header">
                                    <label className="input-label">Temperature</label>
                                    <span className="slider-value">{config.temperature}</span>
                                </div>
                                <input
                                    type="range"
                                    min="0"
                                    max="1"
                                    step="0.1"
                                    value={config.temperature}
                                    onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
                                    className="slider-input"
                                />
                            </div>

                            <div className="input-group">
                                <div className="slider-header">
                                    <label className="input-label">Max Tokens</label>
                                    <span className="slider-value">{config.max_tokens}</span>
                                </div>
                                <input
                                    type="range"
                                    min="128"
                                    max="4096"
                                    step="128"
                                    value={config.max_tokens}
                                    onChange={(e) => setConfig({ ...config, max_tokens: parseInt(e.target.value) })}
                                    className="slider-input"
                                />
                            </div>
                        </div>

                        {/* Execution Limits */}
                        <div className="config-section">
                            <div className="section-label">Execution Limits</div>
                            <div className="input-group">
                                <div className="slider-header">
                                    <label className="input-label">Max Steps</label>
                                    <span className="slider-value">{config.max_iterations}</span>
                                </div>
                                <input
                                    type="range"
                                    min="1"
                                    max="50"
                                    step="1"
                                    value={config.max_iterations}
                                    onChange={(e) => setConfig({ ...config, max_iterations: parseInt(e.target.value) })}
                                    className="slider-input"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Footer */}
                    <div className="sidebar-footer">
                        <button className="primary-button" onClick={handleSave}>
                            Save Changes
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
