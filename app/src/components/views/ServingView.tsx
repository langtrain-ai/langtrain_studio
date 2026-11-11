/**
 * Serving View Component
 * Highly technical vLLM Engine orchestrator with live telemetry
 * and a built-in Inference Playground.
 */

import { useState, useEffect, useRef } from 'react';
import { Play, Square, Activity, Server, Radio, Database, Terminal, Cpu, Clock, Layers, Send, Copy, Check as CheckIcon } from 'lucide-react';

const MOCK_LOGS = [
    "[INFO] Initializing vLLM Engine v0.4.0...",
    "[INFO] Model arch: MixtralForCausalLM",
    "[INFO] Loading weights (Safetensors, 32.2 GB)...",
    "[INFO] Allocating KV Cache: 15.0 GB limit detected (PageAttention v2)",
    "[INFO] engine_use_ray=False, worker_use_ray=False",
    "[INFO] Started HTTP server on port 8000 (0.0.0.0)",
    "[INFO] Available routes: POST /v1/chat/completions, GET /v1/models",
    "[DEBUG] Batch size: 256 | Max seq len: 8192",
    "[DEBUG] PageAttention blocks initialized: 12,044 (block_size=16)",
    "[DEBUG] Scheduler: ContinuousBatching | Preemption mode: recompute",
];

export function ServingView() {
    const [engineState, setEngineState] = useState<'stopped' | 'starting' | 'running'>('stopped');
    const [logs, setLogs] = useState<string[]>([]);
    const [throughput, setThroughput] = useState(0);
    const [kvUtilization, setKvUtilization] = useState(0);
    const [activeRequests, setActiveRequests] = useState(0);
    const [copied, setCopied] = useState(false);

    // Inference playground state
    const [prompt, setPrompt] = useState('');
    const [playgroundMessages, setPlaygroundMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([]);
    const [inferring, setInferring] = useState(false);
    const logsEndRef = useRef<HTMLDivElement>(null);
    const chatEndRef = useRef<HTMLDivElement>(null);

    const handleToggleEngine = () => {
        if (engineState === 'stopped') {
            setEngineState('starting');
            setLogs(["[INFO] Booting ASGI server on 127.0.0.1:8000..."]);
            setTimeout(() => {
                setEngineState('running');
                setLogs(MOCK_LOGS);
                setThroughput(42);
                setKvUtilization(38);
                setActiveRequests(4);
            }, 2000);
        } else {
            setEngineState('stopped');
            setLogs(prev => [...prev, "[WARN] Graceful shutdown initiated...", "[INFO] Server stopped."]);
            setTimeout(() => { setThroughput(0); setKvUtilization(0); setActiveRequests(0); }, 500);
        }
    };

    useEffect(() => {
        if (engineState !== 'running') return;
        const interval = setInterval(() => {
            setThroughput(40 + Math.random() * 15);
            setKvUtilization(prev => Math.max(10, Math.min(prev + (Math.random() * 2 - 1), 95)));
            setActiveRequests(Math.floor(Math.random() * 8));
        }, 1200);
        return () => clearInterval(interval);
    }, [engineState]);

    useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs]);
    useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [playgroundMessages]);

    async function sendPlaygroundMessage() {
        if (!prompt.trim() || !engineState.includes('running')) return;

        const userMsg = { role: 'user' as const, content: prompt };
        setPlaygroundMessages(prev => [...prev, userMsg]);
        setPrompt('');
        setInferring(true);

        // Simulate inference response
        await new Promise(r => setTimeout(r, 800 + Math.random() * 1200));
        const fakeReplies = [
            "I can help with that. Based on the context you've provided, here's my analysis...",
            "That's an interesting question. Let me break it down step by step.",
            "Sure! The key insight here is understanding how transformers handle long-context inputs.",
            "Great question. The short answer is yes, but the nuance matters significantly here.",
        ];
        setPlaygroundMessages(prev => [
            ...prev,
            { role: 'assistant', content: fakeReplies[Math.floor(Math.random() * fakeReplies.length)] }
        ]);
        setInferring(false);
    }

    const apiSnippet = `curl http://127.0.0.1:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "mixtral:8x7b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'`;

    function copyApiSnippet() {
        navigator.clipboard.writeText(apiSnippet);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    }

    return (
        <div style={{ backgroundColor: '#09090b', minHeight: '100%', color: '#fff', fontFamily: 'system-ui, sans-serif', overflow: 'auto' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', padding: '32px 32px 0' }}>
                <div>
                    <h1 style={{ fontSize: '28px', fontWeight: 700, margin: '0 0 6px 0', display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <Server size={28} style={{ color: '#fbbf24' }} />
                        Serving Engine (vLLM)
                    </h1>
                    <p style={{ color: '#a1a1aa', margin: 0, fontSize: '14px' }}>
                        PageAttention-based high-throughput inference — OpenAI-compatible API
                    </p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <code style={{ color: '#00FF85', fontSize: '13px', padding: '6px 12px', background: 'rgba(0,255,133,0.05)', border: '1px solid rgba(0,255,133,0.15)', borderRadius: '6px' }}>
                        http://127.0.0.1:8000
                    </code>
                    <button
                        onClick={handleToggleEngine}
                        disabled={engineState === 'starting'}
                        style={{
                            display: 'flex', alignItems: 'center', gap: '8px',
                            background: engineState === 'running' ? 'transparent' : '#fff',
                            color: engineState === 'running' ? '#ef4444' : '#000',
                            border: engineState === 'running' ? '1px solid #ef4444' : 'none',
                            padding: '8px 20px', borderRadius: '6px', fontWeight: 600, cursor: engineState === 'starting' ? 'not-allowed' : 'pointer',
                            opacity: engineState === 'starting' ? 0.6 : 1
                        }}
                    >
                        {engineState === 'running' ? <><Square size={14} fill="currentColor" /> Stop</> :
                         engineState === 'starting' ? <><Activity size={14} /> Booting...</> :
                         <><Play size={14} fill="currentColor" /> Start vLLM</>}
                    </button>
                </div>
            </div>

            {/* Live Metrics Bar */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', padding: '24px 32px' }}>
                <MetricCard title="GENERATION THROUGHPUT" icon={<Activity size={14} />} color="#00C8FF"
                    value={`${throughput.toFixed(1)}`} unit="tok/s" />
                <MetricCardBar title="KV CACHE UTILIZATION" icon={<Database size={14} />} color="#f43f5e"
                    value={kvUtilization} />
                <MetricCard title="ACTIVE REQUESTS" icon={<Radio size={14} />} color="#10b981"
                    value={`${activeRequests}`} unit="concurrent" />
            </div>

            {/* Main Two-Column Layout */}
            <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '24px', padding: '0 32px 32px' }}>
                {/* Left: Engine Details */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    {/* Model Info */}
                    <Panel title="ACTIVE MODEL">
                        <div style={{ display: 'flex', gap: '16px', alignItems: 'center', marginBottom: '20px' }}>
                            <div style={{ background: '#27272a', padding: '12px', borderRadius: '8px' }}>
                                <Database size={24} color="#a1a1aa" />
                            </div>
                            <div>
                                <div style={{ fontSize: '16px', fontWeight: 700, fontFamily: 'monospace' }}>mixtral:8x7b-instruct</div>
                                <div style={{ fontSize: '12px', color: '#a1a1aa', marginTop: '2px' }}>GGUF • 5-bit INT • 32.2 GB</div>
                            </div>
                        </div>
                        <TelRow icon={<Cpu size={12} />} label="Batch Limit" value={engineState === 'running' ? '256' : '0'} />
                        <TelRow icon={<Layers size={12} />} label="Max Seq Length" value="8,192 tokens" />
                        <TelRow icon={<Clock size={12} />} label="Scheduler" value="Continuous Batching" />
                    </Panel>

                    {/* OpenAI-compatible API curl */}
                    <Panel title="API ENDPOINT (OpenAI-compatible)">
                        <div style={{ position: 'relative' }}>
                            <pre style={{ background: '#000', padding: '12px', borderRadius: '6px', fontSize: '11px', color: '#a1a1aa', border: '1px solid #222', overflow: 'auto', margin: 0, lineHeight: 1.6 }}>
                                {apiSnippet}
                            </pre>
                            <button
                                onClick={copyApiSnippet}
                                style={{ position: 'absolute', top: '8px', right: '8px', background: copied ? 'rgba(0,255,133,0.1)' : 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '4px', padding: '4px 8px', cursor: 'pointer', color: copied ? '#00FF85' : '#a1a1aa', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px' }}
                            >
                                {copied ? <><CheckIcon size={12} /> Copied</> : <><Copy size={12} /> Copy</>}
                            </button>
                        </div>
                    </Panel>
                </div>

                {/* Right: Logs + Playground */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    {/* Terminal Logs */}
                    <Panel title="ENGINE LOGS" style={{ flex: '0 0 auto' }}>
                        <div style={{ height: '200px', overflowY: 'auto', fontFamily: 'monospace', fontSize: '12px', lineHeight: 1.6 }}>
                            {logs.length === 0
                                ? <span style={{ color: '#3f3f46', fontStyle: 'italic' }}>Engine stopped.</span>
                                : logs.map((log, i) => (
                                    <div key={i} style={{ color: log.includes('DEBUG') ? '#52525b' : log.includes('WARN') ? '#fbbf24' : '#e4e4e7' }}>
                                        <span style={{ color: '#3f3f46', marginRight: '8px' }}>{String(i + 1).padStart(3, '0')}</span>
                                        {log}
                                    </div>
                                ))
                            }
                            <div ref={logsEndRef} />
                        </div>
                    </Panel>

                    {/* Inference Playground */}
                    <Panel title="INFERENCE PLAYGROUND" style={{ flex: 1 }}>
                        {!engineState.includes('running') ? (
                            <div style={{ color: '#52525b', fontSize: '13px', padding: '16px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <Server size={14} /> Start the engine to use the playground
                            </div>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', height: '280px' }}>
                                {/* Messages */}
                                <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '12px' }}>
                                    {playgroundMessages.length === 0 && (
                                        <div style={{ color: '#52525b', fontSize: '13px', fontStyle: 'italic' }}>
                                            Send a message to test your model...
                                        </div>
                                    )}
                                    {playgroundMessages.map((msg, i) => (
                                        <div key={i} style={{
                                            alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                                            maxWidth: '80%',
                                            padding: '10px 14px',
                                            borderRadius: msg.role === 'user' ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
                                            background: msg.role === 'user' ? '#27272a' : 'rgba(168,85,247,0.1)',
                                            border: `1px solid ${msg.role === 'user' ? '#3f3f46' : 'rgba(168,85,247,0.2)'}`,
                                            fontSize: '13px',
                                            lineHeight: 1.5,
                                        }}>
                                            {msg.content}
                                        </div>
                                    ))}
                                    {inferring && (
                                        <div style={{ alignSelf: 'flex-start', padding: '10px 14px', borderRadius: '12px 12px 12px 4px', background: 'rgba(168,85,247,0.1)', border: '1px solid rgba(168,85,247,0.2)', fontSize: '13px', color: '#a1a1aa' }}>
                                            <span style={{ animation: 'pulse 1s infinite' }}>●●●</span>
                                        </div>
                                    )}
                                    <div ref={chatEndRef} />
                                </div>
                                {/* Input */}
                                <div style={{ display: 'flex', gap: '8px' }}>
                                    <input
                                        value={prompt}
                                        onChange={e => setPrompt(e.target.value)}
                                        onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendPlaygroundMessage()}
                                        placeholder="Type a message and press Enter..."
                                        style={{ flex: 1, background: '#18181b', border: '1px solid #27272a', borderRadius: '6px', padding: '10px 14px', color: '#fff', fontSize: '13px', outline: 'none' }}
                                    />
                                    <button
                                        onClick={sendPlaygroundMessage}
                                        disabled={!prompt.trim() || inferring}
                                        style={{ background: '#fff', color: '#000', border: 'none', borderRadius: '6px', padding: '10px 16px', cursor: prompt.trim() && !inferring ? 'pointer' : 'not-allowed', opacity: prompt.trim() && !inferring ? 1 : 0.4 }}
                                    >
                                        <Send size={16} />
                                    </button>
                                </div>
                            </div>
                        )}
                    </Panel>
                </div>
            </div>
        </div>
    );
}

// ── Sub-components ───────────────────────────────────────────────────────────

function MetricCard({ title, icon, color, value, unit }: { title: string; icon: React.ReactNode; color: string; value: string; unit: string }) {
    return (
        <div style={{ background: '#18181b', padding: '20px', borderRadius: '12px', border: '1px solid #27272a' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color, fontSize: '11px', fontWeight: 700, marginBottom: '12px' }}>
                {icon} {title}
            </div>
            <div style={{ fontSize: '28px', fontWeight: 700, fontFamily: 'monospace' }}>
                {value} <span style={{ fontSize: '13px', color: '#a1a1aa', fontWeight: 'normal' }}>{unit}</span>
            </div>
        </div>
    );
}

function MetricCardBar({ title, icon, color, value }: { title: string; icon: React.ReactNode; color: string; value: number }) {
    return (
        <div style={{ background: '#18181b', padding: '20px', borderRadius: '12px', border: '1px solid #27272a' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color, fontSize: '11px', fontWeight: 700, marginBottom: '12px' }}>
                {icon} {title}
            </div>
            <div style={{ fontSize: '28px', fontWeight: 700, fontFamily: 'monospace', marginBottom: '12px' }}>
                {value.toFixed(1)}<span style={{ color: '#a1a1aa' }}>%</span>
            </div>
            <div style={{ background: '#27272a', height: '4px', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ width: `${value}%`, height: '100%', background: color, transition: 'width 0.3s ease' }} />
            </div>
        </div>
    );
}

function TelRow({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
    return (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 12px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', marginBottom: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: '#a1a1aa' }}>{icon}{label}</div>
            <div style={{ fontSize: '13px', fontWeight: 600 }}>{value}</div>
        </div>
    );
}

function Panel({ title, children, style }: { title: string; children: React.ReactNode; style?: React.CSSProperties }) {
    return (
        <div style={{ background: '#18181b', borderRadius: '12px', border: '1px solid #27272a', padding: '20px', ...style }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', fontWeight: 700, color: '#52525b', marginBottom: '16px', letterSpacing: '1px' }}>
                <Terminal size={12} /> {title}
            </div>
            {children}
        </div>
    );
}
