import React, { useState, useRef, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { runAgentLoop, AgentMessage, ToolCall, ToolResult } from '../../services/agent';
import { listCloudModels, downloadCloudModel, CloudModel, DownloadProgress } from '../../services/cloud';
import './StudioView.css';

const LLAMA_SERVER = 'http://127.0.0.1:11435';

function useAvailableModels() {
  const [models, setModels] = useState<string[]>([]);
  useEffect(() => {
    fetch(`${LLAMA_SERVER}/v1/models`)
      .then((r) => r.json())
      .then((d) => setModels(d.data?.map((m: any) => m.id) || []))
      .catch(() => setModels(['llama3', 'mistral', 'phi3']));
  }, []);
  return models;
}

function FileTree({ workingDir, onOpen }: { workingDir: string; onOpen: () => void }) {
  const [files, setFiles] = useState<string[]>([]);

  useEffect(() => {
    if (!workingDir) return;
    invoke<string[]>('list_directory', { path: workingDir })
      .then(setFiles)
      .catch(() => setFiles([]));
  }, [workingDir]);

  return (
    <div className="studio-sidebar">
      <div className="studio-sidebar-header">
        <span className="studio-sidebar-title">Files</span>
        <button className="studio-open-btn" onClick={onOpen}>Open…</button>
      </div>
      <div className="studio-file-list">
        {files.length === 0 ? (
          <div style={{ padding: '20px 14px', fontSize: 11, color: '#444', textAlign: 'center' }}>
            {workingDir ? 'Empty folder' : 'Open a folder to start'}
          </div>
        ) : (
          files.map((f) => {
            const isDir = f.endsWith('/');
            const name = isDir ? f.slice(0, -1) : f;
            return (
              <div key={f} className={`studio-file-item ${isDir ? 'dir' : ''}`} title={f}>
                <span className="studio-file-icon">{isDir ? '📁' : '📄'}</span>
                {name.split('/').pop() || name}
              </div>
            );
          })
        )}
      </div>
      {workingDir && <div className="studio-working-dir" title={workingDir}>{workingDir}</div>}
    </div>
  );
}

function ToolCallBlock({ call }: { call: ToolCall }) {
  return (
    <div className="studio-tool-call">
      <div className="studio-tool-call-header">⚡ {call.name}</div>
      <div className="studio-tool-call-body">{JSON.stringify(call.parameters, null, 2)}</div>
    </div>
  );
}

function ToolResultBlock({ result }: { result: ToolResult }) {
  return (
    <div className="studio-tool-result">
      <div className={`studio-tool-result-header ${result.error ? 'error' : ''}`}>
        {result.error ? '✗' : '✓'} {result.toolName}
      </div>
      <div className="studio-tool-result-body">{result.result}</div>
    </div>
  );
}

function MessageView({ msg }: { msg: AgentMessage }) {
  const isUser = msg.role === 'user';
  const isToolResult = msg.role === 'tool_result';

  if (isToolResult && msg.toolResults) {
    return (
      <div className="studio-message">
        {msg.toolResults.map((r, i) => <ToolResultBlock key={i} result={r} />)}
      </div>
    );
  }

  const textWithoutToolCalls = msg.content.replace(
    /<tool_call>[\s\S]*?<\/tool_call>/g, ''
  ).trim();

  return (
    <div className={`studio-message studio-message-${isUser ? 'user' : 'assistant'}`}>
      <span className="studio-message-role">{isUser ? 'You' : 'Studio'}</span>
      {textWithoutToolCalls && (
        <div className="studio-message-body">{textWithoutToolCalls}</div>
      )}
      {msg.toolCalls?.map((call, i) => <ToolCallBlock key={i} call={call} />)}
    </div>
  );
}

function CloudModelsModal({
  apiKey,
  onClose,
}: {
  apiKey: string;
  onClose: () => void;
}) {
  const [models, setModels] = useState<CloudModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState<Record<string, DownloadProgress>>({});

  useEffect(() => {
    listCloudModels(apiKey)
      .then(setModels)
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, [apiKey]);

  const download = async (model: CloudModel) => {
    const filename = `${model.name}.gguf`;
    try {
      await downloadCloudModel(apiKey, model.id, model.download_url, filename, (p) => {
        setProgress((prev) => ({ ...prev, [model.id]: p }));
      });
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="studio-modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="studio-cloud-modal">
        <div className="studio-modal-header">
          <span className="studio-modal-title">Cloud Models</span>
          <button className="studio-modal-close" onClick={onClose}>×</button>
        </div>
        <div className="studio-modal-body">
          {loading ? (
            <div style={{ padding: 20, textAlign: 'center', color: '#555' }}>Loading…</div>
          ) : models.length === 0 ? (
            <div style={{ padding: 20, textAlign: 'center', color: '#555' }}>No GGUF models found. Fine-tune a model on Langtrain first.</div>
          ) : (
            models.map((m) => {
              const p = progress[m.id];
              return (
                <div key={m.id} className="studio-model-card">
                  <div className="studio-model-info">
                    <div className="studio-model-name">{m.display_name || m.name}</div>
                    <div className="studio-model-meta">
                      {m.base_model} · {m.size_gb?.toFixed(1)}GB · {m.quantization || 'Q4_K_M'}
                    </div>
                    {p && !p.done && (
                      <div className="studio-progress-bar">
                        <div className="studio-progress-fill" style={{ width: `${p.percent}%` }} />
                      </div>
                    )}
                    {p?.done && <div style={{ fontSize: 10, color: '#c8f135', marginTop: 4 }}>Downloaded ✓</div>}
                  </div>
                  <button
                    className="studio-download-btn"
                    disabled={p && !p.done}
                    onClick={() => download(m)}
                  >
                    {p && !p.done ? `${Math.round(p.percent)}%` : 'Download'}
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

export function StudioView() {
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [input, setInput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [workingDir, setWorkingDir] = useState('');
  const [showCloudModels, setShowCloudModels] = useState(false);
  const [streamBuffer, setStreamBuffer] = useState('');
  const availableModels = useAvailableModels();
  const [selectedModel, setSelectedModel] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (availableModels.length > 0 && !selectedModel) setSelectedModel(availableModels[0]);
  }, [availableModels]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamBuffer]);

  const openFolder = async () => {
    const dir = await open({ directory: true, multiple: false });
    if (typeof dir === 'string') setWorkingDir(dir);
  };

  const send = useCallback(async () => {
    if (!input.trim() || isRunning) return;
    const userMsg = input.trim();
    setInput('');
    setIsRunning(true);
    setStreamBuffer('');

    let cancelled = false;
    abortRef.current = () => { cancelled = true; };

    let currentBuffer = '';
    const currentToolCalls: ToolCall[] = [];
    const currentToolResults: ToolResult[] = [];

    try {
      await runAgentLoop(userMsg, messages, {
        model: selectedModel || 'llama3',
        workingDir: workingDir || '.',
        onToken: (t) => {
          if (cancelled) return;
          currentBuffer += t;
          setStreamBuffer(currentBuffer);
        },
        onToolCall: (call) => {
          if (cancelled) return;
          currentToolCalls.push(call);
          setStreamBuffer('');
          currentBuffer = '';
        },
        onToolResult: (result) => {
          if (cancelled) return;
          currentToolResults.push(result);
        },
        onComplete: (msgs) => {
          if (cancelled) return;
          setMessages(msgs);
          setStreamBuffer('');
        },
        onError: (err) => {
          if (cancelled) return;
          setMessages((prev) => [
            ...prev,
            { role: 'user', content: userMsg },
            { role: 'assistant', content: `Error: ${err}` },
          ]);
        },
      });
    } finally {
      setIsRunning(false);
      setStreamBuffer('');
      abortRef.current = null;
    }
  }, [input, isRunning, messages, selectedModel, workingDir]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const apiKey = localStorage.getItem('langtrain_api_key') || '';

  return (
    <div className="studio-root">
      <FileTree workingDir={workingDir} onOpen={openFolder} />

      <div className="studio-main">
        <div className="studio-header">
          <span className="studio-header-title">Langtrain Studio</span>
          <select
            className="studio-model-select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {availableModels.length === 0 && <option value="">No local models</option>}
            {availableModels.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          <button className="studio-cloud-btn" onClick={() => setShowCloudModels(true)}>
            ☁ Cloud Models
          </button>
        </div>

        <div className="studio-messages">
          {messages.length === 0 && !streamBuffer && (
            <div className="studio-empty">
              <div style={{ fontSize: 32 }}>◆</div>
              <div className="studio-empty-title">Langtrain Studio</div>
              <div className="studio-empty-sub">
                Ask me to read files, write code, run commands, or help you build.
                {!workingDir && ' Open a folder to get started.'}
              </div>
            </div>
          )}

          {messages.map((msg, i) => <MessageView key={i} msg={msg} />)}

          {streamBuffer && (
            <div className="studio-message studio-message-assistant">
              <span className="studio-message-role">Studio</span>
              <div className="studio-message-body">{streamBuffer}<span style={{ opacity: 0.5 }}>▋</span></div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="studio-input-area">
          <textarea
            className="studio-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isRunning ? 'Studio is thinking…' : 'Ask anything — Shift+Enter for newline'}
            disabled={isRunning}
            rows={1}
          />
          {isRunning ? (
            <button
              className="studio-stop-btn"
              onClick={() => { abortRef.current?.(); setIsRunning(false); setStreamBuffer(''); }}
            >
              Stop
            </button>
          ) : (
            <button
              className="studio-send-btn"
              onClick={send}
              disabled={!input.trim()}
            >
              Send
            </button>
          )}
        </div>
      </div>

      {showCloudModels && (
        <CloudModelsModal apiKey={apiKey} onClose={() => setShowCloudModels(false)} />
      )}
    </div>
  );
}

export default StudioView;
