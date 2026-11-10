/**
 * Langtrain API Client
 * Ported from Swift LangtrainAPI.swift
 */

import { settings } from '../lib/settings';
import { secureStorage } from '../lib/storage';

// API Configuration
export const API_CONFIG = {
    get baseURL() {
        return settings.get().apiBaseUrl;
    },
    version: 'v1',
    webURL: 'https://app.langtrain.xyz',

    get apiURL() {
        return settings.getApiUrl(this.version);
    },

    get authURL() {
        return `${this.webURL}/api/auth/user`;
    },
};

// API Error types
export class APIError extends Error {
    constructor(
        public code: string,
        message: string,
        public status?: number
    ) {
        super(message);
        this.name = 'APIError';
    }

    static invalidURL() {
        return new APIError('INVALID_URL', 'Invalid API URL');
    }

    static invalidResponse() {
        return new APIError('INVALID_RESPONSE', 'Invalid response from server');
    }

    static unauthorized() {
        return new APIError('UNAUTHORIZED', 'Please log in to continue', 401);
    }

    static forbidden() {
        return new APIError('FORBIDDEN', "You don't have permission to access this resource", 403);
    }

    static notFound() {
        return new APIError('NOT_FOUND', 'Resource not found', 404);
    }

    static rateLimited() {
        return new APIError('RATE_LIMITED', 'Too many requests, please try again later', 429);
    }

    static serverError(message: string, status: number) {
        return new APIError('SERVER_ERROR', message, status);
    }
}

// Data Types
export interface UserProfile {
    id: string;
    email: string;
    name?: string;
    avatarUrl?: string;
}

export interface SubscriptionInfo {
    isActive: boolean;
    plan: 'free' | 'pro' | 'enterprise';
    expiresAt?: string;
    features: string[];
}

export interface ModelInfo {
    id: string;
    name: string;
    description?: string;
    parameters?: number;
    contextLength?: number;
    supportsFinetuning?: boolean;
    downloadUrl?: string;
    sizeBytes?: number;
}

export interface ModelsResponse {
    data: ModelInfo[];
}

export interface DatasetInfo {
    id: string;
    filename: string;
    purpose: string;
    bytes: number;
    createdAt: string;
}

export interface DatasetsResponse {
    data: DatasetInfo[];
}

export interface FineTuneJob {
    id: string;
    name: string;
    status: string;
    progress: number;
    config?: Record<string, unknown>;
    metrics?: Record<string, unknown>;
    errorMessage?: string;
    startedAt?: string;
    completedAt?: string;
    createdAt: string;
}

export interface FineTuneJobsResponse {
    data: FineTuneJob[];
}

export interface CreateFineTuneRequest {
    datasetId: string;
    name?: string;
    baseModel: string;
    modelId?: string;
    trainingMethod?: 'sft' | 'dpo' | 'rlhf' | 'lora' | 'qlora';
    hyperparameters?: {
        nEpochs?: number;
        batchSize?: number;
        learningRate?: number;
        loraRank?: number;
        loraAlpha?: number;
        maxSeqLength?: number;
    };
    useCloud?: boolean;
}

export interface AgentInfo {
    id: string;
    name: string;
    description?: string;
    modelId?: string;
    config?: Record<string, unknown>;
    isActive?: boolean;
    createdAt?: string;
}

export interface AgentsResponse {
    data: AgentInfo[];
}

export interface HealthResponse {
    status: string;
    version: string;
}

// Data Conversion Types
export interface ColumnMappingInfo {
    source: string;
    target: string;
    confidence: string;
}

export interface AnalyzeDatasetResponse {
    file_format: string;
    detected_schema: string;
    column_mappings: ColumnMappingInfo[];
    sample_count: number;
    columns_found: string[];
    confidence: number;
    suggestions: string[];
    warnings: string[];
}

export interface PreviewConversionResponse {
    success: boolean;
    target_schema: string;
    preview_lines: string[];
    total_rows: number;
    errors: string[];
    warnings: string[];
}

export interface ConvertToJSONLResponse {
    success: boolean;
    total_rows: number;
    converted_rows: number;
    skipped_rows: number;
    target_schema: string;
    errors: string[];
    warnings: string[];
    jsonl_content?: string;
    message?: string;
    size_bytes?: number;
}

export interface TargetSchemaInfo {
    id: string;
    name: string;
    description: string;
    fields: string[];
    use_case: string;
}

// Bias Rules Types
export interface BiasCondition {
    type: 'contains' | 'regex' | 'semantic' | 'ml_classifier';
    field: string;  // prompt, completion, text, any
    patterns: string[];
    context?: Record<string, unknown>;
    case_sensitive?: boolean;
}

export interface BiasAction {
    type: 'block' | 'warn' | 'modify' | 'downweight';
    message: string;
    replacement?: string;
    weight?: number;
}

export interface BiasRule {
    id: string;
    workspace_id: string;
    name: string;
    description?: string;
    condition: BiasCondition;
    action: BiasAction;
    threshold: number;
    enabled: boolean;
    created_by?: string;
    created_at?: string;
    updated_at?: string;
}

export interface BiasRuleCreate {
    name: string;
    description?: string;
    workspace_id: string;
    condition: BiasCondition;
    action: BiasAction;
    threshold?: number;
    enabled?: boolean;
}

export interface BiasValidationResult {
    valid: boolean;
    violations: BiasViolation[];
    samples_checked: number;
    samples_blocked: number;
    samples_warned: number;
    samples_modified: number;
    samples_downweighted: number;
}

export interface BiasViolation {
    rule_id: string;
    rule_name: string;
    field: string;
    matched_pattern: string;
    confidence: number;
    action: BiasAction;
    sample_index?: number;
}

export interface BiasImpactAnalysis {
    total_samples: number;
    affected_samples: number;
    blocked_samples: number;
    blocked_percentage: number;
    warned_samples: number;
    warned_percentage: number;
    modified_samples: number;
    downweighted_samples: number;
    violations_by_rule: Record<string, number>;
    rules_checked: number;
}

export interface UsageResponse {
    tokens: { used: number; limit: number; percentage: number; unlimited: boolean };
    finetuneJobs: { used: number; limit: number; percentage: number; unlimited: boolean };
    agentRuns: { used: number; limit: number; percentage: number; unlimited: boolean };
    period: { start: string; end?: string; lastUpdated?: string };
    limits: Record<string, number>;
}

export interface UsageResponse {
    tokens: { used: number; limit: number; percentage: number; unlimited: boolean };
    finetuneJobs: { used: number; limit: number; percentage: number; unlimited: boolean };
    agentRuns: { used: number; limit: number; percentage: number; unlimited: boolean };
    period: { start: string; end?: string; lastUpdated?: string };
    limits: Record<string, number>;
}

export interface PlanResponse {
    plan: {
        id?: string;
        name: string;
        code: string;
        billingPeriod: string;
        priceInr?: number;
    };
    limits: Record<string, number>;
    billingStatus: string;
    usageSummary: {
        tokensUsed: number;
        finetuneJobsStarted: number;
        agentRunsCount: number;
        periodStart: string;
        periodEnd?: string;
    };
}

export interface UsageSummary {
    periodDays: number;
    apiCalls: number;
    totalCost: number;
    startDate: string;
    endDate: string;
}

export interface Deployment {
    id: string;
    name: string;
    status: string;
    endpoint?: string;
    createdAt?: string;
}

// Aliases for Home API compatibility
export type TrainingJob = FineTuneJob;
export type Agent = AgentInfo;
export type Dataset = DatasetInfo;

// API Client Class
class APIClient {
    private static instance: APIClient;

    private constructor() { }

    static getInstance(): APIClient {
        if (!APIClient.instance) {
            APIClient.instance = new APIClient();
        }
        return APIClient.instance;
    }

    private async request<T>(
        endpoint: string,
        options: {
            method?: string;
            body?: unknown;
            requiresAuth?: boolean;
        } = {}
    ): Promise<T> {
        const { method = 'GET', body, requiresAuth = true } = options;

        const url = `${API_CONFIG.apiURL}/${endpoint}`;

        const headers: HeadersInit = {
            'Content-Type': 'application/json',
            'User-Agent': 'Langtrain-Studio-Desktop/1.0',
        };

        if (requiresAuth) {
            const token = secureStorage.getItem('langtrain_auth_token');
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }
        }

        const response = await fetch(url, {
            method,
            headers,
            body: body ? JSON.stringify(body) : undefined,
        });

        if (!response.ok) {
            switch (response.status) {
                case 401:
                    // Clear auth on unauthorized
                    secureStorage.removeItem('langtrain_auth_token');
                    throw APIError.unauthorized();
                case 403:
                    throw APIError.forbidden();
                case 404:
                    throw APIError.notFound();
                case 429:
                    throw APIError.rateLimited();
                default:
                    const errorData = await response.json().catch(() => ({}));
                    throw APIError.serverError(
                        errorData.detail || `Unknown error (status: ${response.status})`,
                        response.status
                    );
            }
        }

        return response.json();
    }

    // Authentication
    async getCurrentUser(): Promise<UserProfile> {
        return this.request('auth/me');
    }

    async validateSubscription(): Promise<SubscriptionInfo> {
        return this.request('subscription/status');
    }

    // Models
    async listModels(): Promise<ModelsResponse> {
        return this.request('models', { requiresAuth: false });
    }

    async getModel(id: string): Promise<ModelInfo> {
        return this.request(`models/${id}`, { requiresAuth: false });
    }

    // Datasets (Workspace Scoped)
    async listWorkspaceDatasets(workspaceId: string): Promise<DatasetsResponse> {
        return this.request(`files?workspace_id=${workspaceId}`);
    }

    async uploadDataset(
        file: File,
        purpose: string = 'fine-tune',
        workspaceId: string = 'default'
    ): Promise<DatasetInfo> {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('purpose', purpose);
        formData.append('workspace_id', workspaceId);

        const url = `${API_CONFIG.apiURL}/files`;
        const token = secureStorage.getItem('langtrain_auth_token');

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'User-Agent': 'Langtrain-Studio-Desktop/1.0',
            },
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new APIError(
                'UPLOAD_FAILED',
                errorData.message || 'Failed to upload dataset',
                response.status
            );
        }

        return response.json();
    }

    async deleteDataset(datasetId: string): Promise<void> {
        await this.request(`files/${datasetId}`, { method: 'DELETE' });
    }

    // Fine-tuning Jobs
    async createFineTuningJob(request: CreateFineTuneRequest): Promise<FineTuneJob> {
        return this.request('fine-tuning/jobs', {
            method: 'POST',
            body: request,
        });
    }

    async getFineTuningJob(id: string): Promise<FineTuneJob> {
        return this.request(`fine-tuning/jobs/${id}`);
    }

    async listFineTuningJobs(workspaceId?: string): Promise<FineTuneJobsResponse> {
        const endpoint = workspaceId
            ? `fine-tuning/jobs?workspace_id=${workspaceId}`
            : 'fine-tuning/jobs';
        return this.request(endpoint);
    }

    async cancelFineTuningJob(id: string): Promise<FineTuneJob> {
        return this.request(`fine-tuning/jobs/${id}/cancel`, { method: 'POST' });
    }

    // Agents (Workspace Scoped)
    async listWorkspaceAgents(workspaceId: string): Promise<AgentsResponse> {
        return this.request(`workspaces/${workspaceId}/agents`);
    }

    async createAgent(workspaceId: string, agent: Partial<AgentInfo>): Promise<AgentInfo> {
        return this.request(`workspaces/${workspaceId}/agents`, {
            method: 'POST',
            body: agent,
        });
    }

    async updateAgent(id: string, agent: Partial<AgentInfo>): Promise<AgentInfo> {
        return this.request(`agents/${id}`, {
            method: 'PUT',
            body: agent,
        });
    }

    async deleteAgent(id: string): Promise<void> {
        await this.request(`agents/${id}`, { method: 'DELETE' });
    }

    async runAgent(id: string, messages: Array<{ role: string; content: string }>): Promise<unknown> {
        return this.request(`agents/${id}/runs`, {
            method: 'POST',
            body: { input: { messages } },
        });
    }

    // Chat Completions (Inference)
    async chatCompletions(request: {
        model: string;
        messages: Array<{ role: string; content: string }>;
        temperature?: number;
        maxTokens?: number;
    }): Promise<unknown> {
        return this.request('chat/completions', {
            method: 'POST',
            body: request,
        });
    }

    // Hub Models (for model browser)
    async getHubModels(): Promise<{ models: ModelInfo[] }> {
        return this.request('../hub/', { requiresAuth: false });
    }

    async searchHubModels(query: string): Promise<{ models: ModelInfo[] }> {
        return this.request(`hub/search?q=${encodeURIComponent(query)}`, { requiresAuth: false });
    }

    // Health Check
    async healthCheck(): Promise<HealthResponse> {
        return this.request('../health', { requiresAuth: false });
    }

    // Billing & Usage
    async getUsage(workspaceId: string): Promise<UsageResponse> {
        return this.request(`billing/usage?workspace_id=${workspaceId}`);
    }

    async getPlan(workspaceId: string): Promise<PlanResponse> {
        return this.request(`billing/plan?workspace_id=${workspaceId}`);
    }

    // ==========================================
    // Data Conversion (Pattern Detection + JSONL)
    // ==========================================

    /**
     * Analyze uploaded file to detect data patterns
     */
    async analyzeDataset(file: File): Promise<AnalyzeDatasetResponse> {
        const formData = new FormData();
        formData.append('file', file);

        const url = `${API_CONFIG.apiURL}/convert/analyze`;
        const token = secureStorage.getItem('langtrain_auth_token');

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'User-Agent': 'Langtrain-Studio-Desktop/1.0',
            },
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new APIError(
                'ANALYSIS_FAILED',
                errorData.detail || 'Failed to analyze dataset',
                response.status
            );
        }

        return response.json();
    }

    /**
     * Preview conversion of file to JSONL
     */
    async previewConversion(
        file: File,
        options?: { targetSchema?: string; previewRows?: number }
    ): Promise<PreviewConversionResponse> {
        const formData = new FormData();
        formData.append('file', file);
        if (options?.targetSchema) {
            formData.append('target_schema', options.targetSchema);
        }
        if (options?.previewRows) {
            formData.append('preview_rows', options.previewRows.toString());
        }

        const url = `${API_CONFIG.apiURL}/convert/preview`;
        const token = secureStorage.getItem('langtrain_auth_token');

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'User-Agent': 'Langtrain-Studio-Desktop/1.0',
            },
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new APIError(
                'PREVIEW_FAILED',
                errorData.detail || 'Failed to preview conversion',
                response.status
            );
        }

        return response.json();
    }

    /**
     * Execute full conversion to JSONL
     */
    async convertToJSONL(
        file: File,
        options?: { targetSchema?: string; customMappings?: Record<string, string> }
    ): Promise<ConvertToJSONLResponse> {
        const formData = new FormData();
        formData.append('file', file);
        if (options?.targetSchema) {
            formData.append('target_schema', options.targetSchema);
        }
        if (options?.customMappings) {
            formData.append('custom_mappings', JSON.stringify(options.customMappings));
        }

        const url = `${API_CONFIG.apiURL}/convert/execute`;
        const token = secureStorage.getItem('langtrain_auth_token');

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'User-Agent': 'Langtrain-Studio-Desktop/1.0',
            },
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new APIError(
                'CONVERSION_FAILED',
                errorData.detail || 'Failed to convert dataset',
                response.status
            );
        }

        return response.json();
    }

    /**
     * Get available target schemas for conversion
     */
    async getConversionSchemas(): Promise<{ schemas: TargetSchemaInfo[] }> {
        return this.request('convert/schemas', { requiresAuth: false });
    }

    // ==========================================
    // Bias Rules
    // ==========================================

    /**
     * Create a new bias rule
     */
    async createBiasRule(rule: BiasRuleCreate): Promise<BiasRule> {
        return this.request('bias-rules/', {
            method: 'POST',
            body: rule,
        });
    }

    /**
     * List bias rules for a workspace
     */
    async listBiasRules(workspaceId: string, enabledOnly: boolean = false): Promise<BiasRule[]> {
        const params = new URLSearchParams({ workspace_id: workspaceId });
        if (enabledOnly) params.append('enabled_only', 'true');
        return this.request(`bias-rules/?${params.toString()}`);
    }

    /**
     * Get a single bias rule
     */
    async getBiasRule(ruleId: string): Promise<BiasRule> {
        return this.request(`bias-rules/${ruleId}`);
    }

    /**
     * Update a bias rule
     */
    async updateBiasRule(ruleId: string, updates: Partial<BiasRuleCreate>): Promise<BiasRule> {
        return this.request(`bias-rules/${ruleId}`, {
            method: 'PUT',
            body: updates,
        });
    }

    /**
     * Delete a bias rule
     */
    async deleteBiasRule(ruleId: string): Promise<{ success: boolean }> {
        return this.request(`bias-rules/${ruleId}`, { method: 'DELETE' });
    }

    /**
     * Validate text against bias rules
     */
    async validateTextWithBiasRules(
        text: string,
        ruleIds: string[]
    ): Promise<BiasValidationResult> {
        return this.request('bias-rules/validate/text', {
            method: 'POST',
            body: { text, rule_ids: ruleIds },
        });
    }

    /**
     * Validate file against bias rules
     */
    async validateFileWithBiasRules(
        file: File,
        ruleIds: string[]
    ): Promise<BiasValidationResult & { filename: string }> {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('rule_ids', JSON.stringify(ruleIds));

        const url = `${API_CONFIG.apiURL}/bias-rules/validate/file`;
        const token = secureStorage.getItem('langtrain_auth_token');

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'User-Agent': 'Langtrain-Studio-Desktop/1.0',
            },
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new APIError(
                'VALIDATION_FAILED',
                errorData.detail || 'Failed to validate file',
                response.status
            );
        }

        return response.json();
    }

    /**
     * Analyze impact of bias rules on a file
     */
    async analyzeBiasImpact(
        file: File,
        ruleIds: string[]
    ): Promise<BiasImpactAnalysis & { filename: string }> {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('rule_ids', JSON.stringify(ruleIds));

        const url = `${API_CONFIG.apiURL}/bias-rules/analyze/file`;
        const token = secureStorage.getItem('langtrain_auth_token');

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'User-Agent': 'Langtrain-Studio-Desktop/1.0',
            },
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new APIError(
                'ANALYSIS_FAILED',
                errorData.detail || 'Failed to analyze bias impact',
                response.status
            );
        }

        return response.json();
    }

    // Training
    async listTrainingJobs(params?: { limit?: number; offset?: number; status?: string }): Promise<{ data: TrainingJob[] }> {
        const query = new URLSearchParams();
        if (params?.limit) query.set('limit', String(params.limit));
        if (params?.offset) query.set('offset', String(params.offset));
        if (params?.status) query.set('status', params.status);
        return this.request(`training?${query.toString()}`);
    }

    async getTrainingJob(jobId: string): Promise<TrainingJob> {
        return this.request(`training/${jobId}`);
    }

    // Agents
    async listAgents(params?: { limit?: number; offset?: number }): Promise<{ data: Agent[] }> {
        const query = new URLSearchParams();
        if (params?.limit) query.set('limit', String(params.limit));
        if (params?.offset) query.set('offset', String(params.offset));
        return this.request(`agents?${query.toString()}`);
    }

    async getAgent(agentId: string): Promise<Agent> {
        return this.request(`agents/${agentId}`);
    }

    // Datasets
    async listDatasets(params?: { limit?: number; offset?: number }): Promise<{ data: Dataset[] }> {
        const query = new URLSearchParams();
        if (params?.limit) query.set('limit', String(params.limit));
        if (params?.offset) query.set('offset', String(params.offset));
        return this.request(`datasets?${query.toString()}`);
    }

    async getDataset(datasetId: string): Promise<Dataset> {
        return this.request(`datasets/${datasetId}`);
    }

    async getDatasetProfile(datasetId: string): Promise<any> {
        return this.request(`files/${datasetId}/profile`);
    }

    // Deployments
    async listDeployments(params?: { limit?: number; offset?: number }): Promise<{ data: Deployment[] }> {
        const query = new URLSearchParams();
        if (params?.limit) query.set('limit', String(params.limit));
        if (params?.offset) query.set('offset', String(params.offset));
        return this.request(`deployments?${query.toString()}`);
    }

    async getDeployment(deploymentId: string): Promise<Deployment> {
        return this.request(`deployments/${deploymentId}`);
    }

    // Usage
    async getUsageSummary(days: number = 30): Promise<UsageSummary> {
        return this.request(`usage?days=${days}`);
    }

    async getTransactions(limit: number = 20): Promise<{ data: Array<{ id: string; amount: number; status: string; createdAt?: string }> }> {
        return this.request(`usage/transactions?limit=${limit}`);
    }

    // Settings
    async getProfile(): Promise<UserProfile> {
        return this.request('settings/profile');
    }

    async updateProfile(data: { fullName?: string; avatarUrl?: string }): Promise<{ success: boolean }> {
        return this.request('settings/profile', { method: 'PATCH', body: data });
    }
}

export const api = APIClient.getInstance();

// ---------------------------------------------------------------------------
// Hardware & Local Training helpers (call the new server endpoints)
// ---------------------------------------------------------------------------

export interface HardwareInfo {
    gpu_name: string;
    vram_mb: number;
    vram_gb: number;
    tflops_fp16: number | null;
    bandwidth_gbps: number | null;
    compute_capability: string | null;
    driver_version: string | null;
    platform: 'nvidia' | 'apple_silicon' | 'cpu';
    has_gpu: boolean;
}

export interface LocalJobResponse {
    job_id: string;
    status: string;
    mode: string;
    hardware: HardwareInfo;
    stream_url: string;
    log_channel: string;
}

export async function detectHardware(): Promise<HardwareInfo> {
    const url = `${API_CONFIG.apiURL}/hardware/detect`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Hardware detection failed');
    return res.json();
}

export async function startLocalJob(opts: {
    base_model: string;
    dataset_path: string;
    config?: Record<string, unknown>;
    callback_url?: string;
}): Promise<LocalJobResponse> {
    const url = `${API_CONFIG.apiURL}/training/jobs/local`;
    const token = localStorage.getItem('langtrain_auth_token');
    const res = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(opts),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail?.message || 'Failed to start local job');
    }
    return res.json();
}

export function openJobLogStream(jobId: string): WebSocket {
    const base = API_CONFIG.apiURL.replace(/^http/, 'ws');
    return new WebSocket(`${base}/training/jobs/local/${jobId}/stream`);
}
