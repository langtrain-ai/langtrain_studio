/**
 * CoWorkView — Split-pane workspace with local AI assistant
 * 
 * Layout: File Tree | Code Editor | Chat Panel
 * Connects to either local inference (llama.cpp) or remote API
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import {
    File, Send, Bot,
    User, Zap, Globe, Loader2
} from 'lucide-react';
import { InferenceService, SubscriptionGate, type ServerStatus, type ProAccessResult } from '../../services/local';
import { api } from '../../services/api';
import { Crown } from 'lucide-react';
import './CoWorkView.css';

interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: number;
    source?: 'local' | 'remote';
}


export function CoWorkView() {
    const [proAccess, setProAccess] = useState<ProAccessResult | null>(null);
    const [server, setServer] = useState<ServerStatus | null>(null);
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [sending, setSending] = useState(false);
    const [useLocal, setUseLocal] = useState(true);
    const [editorContent, setEditorContent] = useState('// Open a file or start typing...\n');
    const [activeFile] = useState<string | null>(null);
    const [systemPrompt] = useState(
        'You are a helpful AI coding assistant. Help the user with their code. Be concise and provide working code snippets.'
    );
    const chatEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        loadState();
    }, []);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const loadState = async () => {
        try {
            const [access, status] = await Promise.all([
                SubscriptionGate.checkAccess(),
                InferenceService.status(),
            ]);
            setProAccess(access);
            setServer(status);
            setUseLocal(status.running);
        } catch (e) {
            console.error('Failed to load state:', e);
        }
    };

    const sendMessage = useCallback(async () => {
        const trimmed = input.trim();
        if (!trimmed || sending) return;

        const userMsg: ChatMessage = {
            role: 'user',
            content: trimmed,
            timestamp: Date.now(),
        };

        setMessages((prev) => [...prev, userMsg]);
        setInput('');
        setSending(true);

        try {
            // Build message history
            const allMessages = [
                { role: 'system', content: systemPrompt },
                ...messages.map((m) => ({ role: m.role, content: m.content })),
                { role: 'user', content: trimmed },
            ];

            // Add editor context if there's content
            if (editorContent && editorContent !== '// Open a file or start typing...\n') {
                allMessages[0] = {
                    role: 'system',
                    content: `${systemPrompt}\n\nThe user is currently editing${activeFile ? ` "${activeFile}"` : ''} with this content:\n\`\`\`\n${editorContent.slice(0, 3000)}\n\`\`\``,
                };
            }

            let responseContent = '';
            let source: 'local' | 'remote' = 'local';

            if (useLocal && server?.running) {
                // Use local inference
                const result = await InferenceService.chat(allMessages, {
                    temperature: 0.7,
                    maxTokens: 2048,
                });
                responseContent = result.content;
                source = 'local';
            } else {
                // Use remote API
                const result: any = await api.chatCompletions({
                    model: 'default',
                    messages: allMessages,
                    temperature: 0.7,
                    maxTokens: 2048,
                });
                responseContent = result.choices?.[0]?.message?.content ?? 'No response';
                source = 'remote';
            }

            const assistantMsg: ChatMessage = {
                role: 'assistant',
                content: responseContent,
                timestamp: Date.now(),
                source,
            };

            setMessages((prev) => [...prev, assistantMsg]);
        } catch (e: any) {
            const errorMsg: ChatMessage = {
                role: 'assistant',
                content: `Error: ${e.message || 'Failed to get response'}`,
                timestamp: Date.now(),
            };
            setMessages((prev) => [...prev, errorMsg]);
        } finally {
            setSending(false);
        }
    }, [input, messages, sending, useLocal, server, editorContent, activeFile, systemPrompt]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    // Pro gate
    if (proAccess && !proAccess.allowed) {
        return (
            <div className="cowork-page">
                <div className="pro-gate-overlay">
                    <Crown size={48} className="pro-icon" />
                    <h2>Co-Work — Pro Feature</h2>
                    <p>Code alongside an AI assistant running entirely on your machine. Upgrade to Pro to unlock.</p>
                    <a href="https://app.langtrain.xyz/billing" target="_blank" rel="noopener" className="upgrade-btn">
                        Upgrade to Pro
                    </a>
                </div>
            </div>
        );
    }

    return (
        <div className="cowork-page">
            {/* Header */}
            <div className="cowork-header">
                <div className="cowork-title">
                    <Bot size={20} />
                    <h2>Co-Work</h2>
                </div>
                <div className="cowork-controls">
                    <button
                        className={`mode-toggle ${useLocal ? 'local' : 'remote'}`}
                        onClick={() => setUseLocal(!useLocal)}
                        title={useLocal ? 'Using local model' : 'Using remote API'}
                    >
                        {useLocal ? <Zap size={14} /> : <Globe size={14} />}
                        {useLocal ? 'Local' : 'Remote'}
                    </button>
                    {useLocal && !server?.running && (
                        <span className="mode-warning">⚠ No local model running</span>
                    )}
                </div>
            </div>

            {/* Main Layout */}
            <div className="cowork-layout">
                {/* Editor Pane */}
                <div className="editor-pane">
                    <div className="editor-tab-bar">
                        <div className="editor-tab active">
                            <File size={12} />
                            <span>{activeFile || 'Untitled'}</span>
                        </div>
                    </div>
                    <textarea
                        className="code-editor"
                        value={editorContent}
                        onChange={(e) => setEditorContent(e.target.value)}
                        spellCheck={false}
                        placeholder="Start typing code..."
                    />
                </div>

                {/* Chat Pane */}
                <div className="chat-pane">
                    <div className="chat-messages">
                        {messages.length === 0 && (
                            <div className="chat-empty">
                                <Bot size={32} />
                                <h4>AI Assistant</h4>
                                <p>
                                    {useLocal && server?.running
                                        ? 'Running locally — no data leaves your machine'
                                        : 'Ask questions about your code'}
                                </p>
                            </div>
                        )}
                        {messages.map((msg, i) => (
                            <div key={i} className={`chat-msg ${msg.role}`}>
                                <div className="msg-avatar">
                                    {msg.role === 'user' ? <User size={14} /> : <Bot size={14} />}
                                </div>
                                <div className="msg-content">
                                    <pre>{msg.content}</pre>
                                    {msg.source && (
                                        <span className="msg-source">
                                            {msg.source === 'local' ? '⚡ Local' : '🌐 Remote'}
                                        </span>
                                    )}
                                </div>
                            </div>
                        ))}
                        {sending && (
                            <div className="chat-msg assistant">
                                <div className="msg-avatar"><Bot size={14} /></div>
                                <div className="msg-content">
                                    <Loader2 className="spinner" size={16} />
                                </div>
                            </div>
                        )}
                        <div ref={chatEndRef} />
                    </div>

                    <div className="chat-input-area">
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask about your code... (Enter to send)"
                            rows={2}
                            disabled={sending}
                        />
                        <button onClick={sendMessage} disabled={sending || !input.trim()} className="send-btn">
                            <Send size={16} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
