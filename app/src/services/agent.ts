import { invoke } from '@tauri-apps/api/core';

const LLAMA_SERVER = 'http://127.0.0.1:11435';

export type MessageRole = 'user' | 'assistant' | 'tool_result';

export interface ToolCall {
  name: string;
  parameters: Record<string, unknown>;
}

export interface ToolResult {
  toolName: string;
  result: string;
  error?: boolean;
}

export interface AgentMessage {
  role: MessageRole;
  content: string;
  toolCalls?: ToolCall[];
  toolResults?: ToolResult[];
}

export interface AgentOptions {
  model: string;
  workingDir: string;
  systemPrompt?: string;
  onToken: (token: string) => void;
  onToolCall: (call: ToolCall) => void;
  onToolResult: (result: ToolResult) => void;
  onComplete: (messages: AgentMessage[]) => void;
  onError: (err: string) => void;
}

const TOOL_DESCRIPTIONS = `
You are an AI coding assistant with access to the local filesystem and shell.
Use XML tool calls to interact with the environment:

<tool_call><name>read_file</name><parameters>{"path": "..."}</parameters></tool_call>
<tool_call><name>write_file</name><parameters>{"path": "...", "content": "..."}</parameters></tool_call>
<tool_call><name>create_file</name><parameters>{"path": "...", "content": "..."}</parameters></tool_call>
<tool_call><name>list_directory</name><parameters>{"path": "..."}</parameters></tool_call>
<tool_call><name>execute_shell</name><parameters>{"command": "...", "working_dir": "..."}</parameters></tool_call>
<tool_call><name>search_in_files</name><parameters>{"directory": "...", "pattern": "...", "file_pattern": "..."}</parameters></tool_call>
<tool_call><name>get_file_info</name><parameters>{"path": "..."}</parameters></tool_call>
<tool_call><name>move_or_rename</name><parameters>{"from_path": "...", "to_path": "..."}</parameters></tool_call>
<tool_call><name>delete_path</name><parameters>{"path": "..."}</parameters></tool_call>
<tool_call><name>get_working_directory</name><parameters>{}</parameters></tool_call>
`;

function parseToolCalls(text: string): ToolCall[] {
  const calls: ToolCall[] = [];
  const regex = /<tool_call>\s*<name>([\w_]+)<\/name>\s*<parameters>([\s\S]*?)<\/parameters>\s*<\/tool_call>/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    try {
      calls.push({ name: match[1], parameters: JSON.parse(match[2]) });
    } catch {
      // skip malformed
    }
  }
  return calls;
}

async function executeTool(call: ToolCall): Promise<string> {
  const p = call.parameters as Record<string, string>;
  try {
    switch (call.name) {
      case 'read_file':
        return await invoke<string>('read_file', { path: p.path });
      case 'write_file':
        await invoke('write_file', { path: p.path, content: p.content });
        return `File written: ${p.path}`;
      case 'create_file':
        await invoke('create_file', { path: p.path, content: p.content || '' });
        return `File created: ${p.path}`;
      case 'list_directory':
        return JSON.stringify(await invoke<string[]>('list_directory', { path: p.path }), null, 2);
      case 'delete_path':
        await invoke('delete_path', { path: p.path });
        return `Deleted: ${p.path}`;
      case 'execute_shell':
        return await invoke<string>('execute_shell', { command: p.command, workingDir: p.working_dir });
      case 'search_in_files':
        return JSON.stringify(await invoke('search_in_files', { directory: p.directory, pattern: p.pattern, filePattern: p.file_pattern }), null, 2);
      case 'get_file_info':
        return JSON.stringify(await invoke('get_file_info', { path: p.path }), null, 2);
      case 'move_or_rename':
        await invoke('move_or_rename', { fromPath: p.from_path, toPath: p.to_path });
        return `Moved ${p.from_path} → ${p.to_path}`;
      case 'get_working_directory':
        return await invoke<string>('get_working_directory', {});
      default:
        return `Unknown tool: ${call.name}`;
    }
  } catch (e) {
    throw new Error(String(e));
  }
}

async function streamCompletion(
  messages: Array<{ role: string; content: string }>,
  model: string,
  onToken: (t: string) => void,
  signal: AbortSignal
): Promise<string> {
  const res = await fetch(`${LLAMA_SERVER}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages, stream: true, temperature: 0.7, max_tokens: 4096 }),
    signal,
  });

  if (!res.ok) throw new Error(`llama-server ${res.status}: ${await res.text()}`);

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let full = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    for (const line of chunk.split('\n')) {
      if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
      try {
        const delta = JSON.parse(line.slice(6)).choices?.[0]?.delta?.content ?? '';
        if (delta) { full += delta; onToken(delta); }
      } catch { /* skip */ }
    }
  }
  return full;
}

export async function runAgentLoop(
  userMessage: string,
  history: AgentMessage[],
  options: AgentOptions
): Promise<void> {
  const { model, workingDir, systemPrompt, onToken, onToolCall, onToolResult, onComplete, onError } = options;
  const abortController = new AbortController();

  const systemContent = [
    systemPrompt || 'You are a helpful AI coding assistant.',
    TOOL_DESCRIPTIONS,
    `Working directory: ${workingDir}`,
  ].join('\n\n');

  const apiMessages: Array<{ role: string; content: string }> = [
    { role: 'system', content: systemContent },
    ...history.map((m) => ({ role: m.role === 'tool_result' ? 'user' : m.role, content: m.content })),
    { role: 'user', content: userMessage },
  ];

  const newMessages: AgentMessage[] = [...history, { role: 'user', content: userMessage }];

  try {
    for (let iteration = 0; iteration < 10; iteration++) {
      const assistantText = await streamCompletion(apiMessages, model, onToken, abortController.signal);
      const toolCalls = parseToolCalls(assistantText);

      const assistantMsg: AgentMessage = {
        role: 'assistant',
        content: assistantText,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      };
      newMessages.push(assistantMsg);
      apiMessages.push({ role: 'assistant', content: assistantText });

      if (toolCalls.length === 0) break;

      const results: ToolResult[] = [];
      for (const call of toolCalls) {
        onToolCall(call);
        let result: ToolResult;
        try {
          const output = await executeTool(call);
          result = { toolName: call.name, result: output };
        } catch (e) {
          result = { toolName: call.name, result: String(e), error: true };
        }
        onToolResult(result);
        results.push(result);
      }

      const toolResultContent = results
        .map((r) => `<tool_result name="${r.toolName}"${r.error ? ' error="true"' : ''}>\n${r.result}\n</tool_result>`)
        .join('\n');

      const toolMsg: AgentMessage = { role: 'tool_result', content: toolResultContent, toolResults: results };
      newMessages.push(toolMsg);
      apiMessages.push({ role: 'user', content: toolResultContent });
    }

    onComplete(newMessages);
  } catch (e) {
    if ((e as Error).name !== 'AbortError') onError(String(e));
  }
}
