import React, { useState, useEffect } from "react";
import {
    Shield,
    Plus,
    Trash2,
    AlertTriangle,
    CheckCircle2,
    Ban,
    AlertCircle,
    Edit3,
    ChevronDown,
    ChevronUp,
    BarChart3,
    Eye,
    X,
    Loader2,
} from "lucide-react";
import { api, BiasRule, BiasRuleCreate, BiasImpactAnalysis } from "../../services/api";

interface GuardrailsViewProps {
    workspaceId: string;
    datasetFile?: File;
    onRulesChange?: (rules: BiasRule[]) => void;
    onAnalysisComplete?: (analysis: BiasImpactAnalysis) => void;
}

const ACTION_ICONS: Record<string, React.ReactNode> = {
    block: <Ban className="w-4 h-4 text-red-400" />,
    warn: <AlertTriangle className="w-4 h-4 text-yellow-400" />,
    modify: <Edit3 className="w-4 h-4 text-blue-400" />,
    downweight: <BarChart3 className="w-4 h-4 text-purple-400" />,
};

const ACTION_COLORS: Record<string, string> = {
    block: "border-red-500/30 bg-red-500/10",
    warn: "border-yellow-500/30 bg-yellow-500/10",
    modify: "border-blue-500/30 bg-blue-500/10",
    downweight: "border-purple-500/30 bg-purple-500/10",
};

export function GuardrailsView({
    workspaceId,
    datasetFile,
    onRulesChange,
    onAnalysisComplete,
}: GuardrailsViewProps) {
    const [rules, setRules] = useState<BiasRule[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isCreating, setIsCreating] = useState(false);
    const [expandedRuleId, setExpandedRuleId] = useState<string | null>(null);
    const [analysis, setAnalysis] = useState<BiasImpactAnalysis | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // New rule form
    const [newRuleName, setNewRuleName] = useState("");
    const [newRuleDescription, setNewRuleDescription] = useState("");
    const [matchType, setMatchType] = useState<"contains" | "regex" | "semantic">("contains");
    const [field, setField] = useState<"prompt" | "completion" | "text" | "any">("any");
    const [actionType, setActionType] = useState<"block" | "warn" | "modify" | "downweight">("warn");
    const [threshold, setThreshold] = useState(80);
    const [patterns, setPatterns] = useState<string[]>([]);
    const [patternInput, setPatternInput] = useState("");
    const [actionMessage, setActionMessage] = useState("Bias rule triggered");

    useEffect(() => {
        fetchRules();
    }, [workspaceId]);

    const fetchRules = async () => {
        try {
            setIsLoading(true);
            setError(null);
            const data = await api.listBiasRules(workspaceId);
            setRules(data);
            onRulesChange?.(data);
        } catch (err: any) {
            setError(err.message || "Failed to fetch rules");
        } finally {
            setIsLoading(false);
        }
    };

    const createRule = async () => {
        if (!newRuleName || patterns.length === 0) return;

        const ruleData: BiasRuleCreate = {
            name: newRuleName,
            description: newRuleDescription,
            workspace_id: workspaceId,
            condition: {
                type: matchType,
                field: field,
                patterns: patterns,
                case_sensitive: false,
            },
            action: {
                type: actionType,
                message: actionMessage,
            },
            threshold: threshold / 100,
            enabled: true,
        };

        try {
            setIsCreating(true);
            const created = await api.createBiasRule(ruleData);
            const updated = [...rules, created];
            setRules(updated);
            onRulesChange?.(updated);
            resetForm();
        } catch (err: any) {
            setError(err.message || "Failed to create rule");
        } finally {
            setIsCreating(false);
        }
    };

    const deleteRule = async (ruleId: string) => {
        try {
            await api.deleteBiasRule(ruleId);
            const updated = rules.filter((r) => r.id !== ruleId);
            setRules(updated);
            onRulesChange?.(updated);
        } catch (err: any) {
            setError(err.message || "Failed to delete rule");
        }
    };

    const toggleRule = async (ruleId: string, enabled: boolean) => {
        try {
            await api.updateBiasRule(ruleId, { enabled });
            const updated = rules.map((r) => (r.id === ruleId ? { ...r, enabled } : r));
            setRules(updated);
            onRulesChange?.(updated);
        } catch (err: any) {
            setError(err.message || "Failed to toggle rule");
        }
    };

    const analyzeImpact = async () => {
        if (!datasetFile || rules.length === 0) return;

        try {
            setIsAnalyzing(true);
            const enabledRuleIds = rules.filter((r) => r.enabled).map((r) => r.id);
            const result = await api.analyzeBiasImpact(datasetFile, enabledRuleIds);
            setAnalysis(result);
            onAnalysisComplete?.(result);
        } catch (err: any) {
            setError(err.message || "Failed to analyze");
        } finally {
            setIsAnalyzing(false);
        }
    };

    const addPattern = () => {
        if (patternInput.trim()) {
            setPatterns([...patterns, patternInput.trim()]);
            setPatternInput("");
        }
    };

    const removePattern = (index: number) => {
        setPatterns(patterns.filter((_, i) => i !== index));
    };

    const resetForm = () => {
        setNewRuleName("");
        setNewRuleDescription("");
        setPatterns([]);
        setPatternInput("");
        setActionMessage("Bias rule triggered");
        setThreshold(80);
        setMatchType("contains");
        setField("any");
        setActionType("warn");
    };

    if (isLoading) {
        return (
            <div className="flex items-center justify-center p-8">
                <Loader2 className="w-6 h-6 animate-spin text-accent" />
                <span className="ml-2 text-text-secondary">Loading bias rules...</span>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Shield className="w-5 h-5 text-accent" />
                    <h3 className="text-lg font-semibold">Bias Rules & Thresholds</h3>
                </div>
                {datasetFile && rules.length > 0 && (
                    <button
                        onClick={analyzeImpact}
                        disabled={isAnalyzing}
                        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-accent/20 text-accent hover:bg-accent/30 text-sm"
                    >
                        {isAnalyzing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Eye className="w-4 h-4" />}
                        Analyze Impact
                    </button>
                )}
            </div>

            {/* Error */}
            {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
                    {error}
                    <button onClick={() => setError(null)} className="ml-2 hover:text-red-300">
                        <X className="w-4 h-4 inline" />
                    </button>
                </div>
            )}

            {/* Impact Analysis */}
            {analysis && (
                <div className="p-4 rounded-lg bg-surface border border-border">
                    <div className="flex justify-between items-center mb-3">
                        <h4 className="font-medium flex items-center gap-2">
                            <BarChart3 className="w-4 h-4 text-accent" />
                            Impact Analysis
                        </h4>
                        <button onClick={() => setAnalysis(null)} className="text-text-secondary hover:text-text-primary">
                            <X className="w-4 h-4" />
                        </button>
                    </div>
                    <div className="grid grid-cols-4 gap-4 text-center">
                        <div>
                            <div className="text-xl font-bold">{analysis.total_samples}</div>
                            <div className="text-xs text-text-secondary">Total</div>
                        </div>
                        <div>
                            <div className="text-xl font-bold text-red-400">{analysis.blocked_samples}</div>
                            <div className="text-xs text-text-secondary">Blocked ({analysis.blocked_percentage.toFixed(1)}%)</div>
                        </div>
                        <div>
                            <div className="text-xl font-bold text-yellow-400">{analysis.warned_samples}</div>
                            <div className="text-xs text-text-secondary">Warned ({analysis.warned_percentage.toFixed(1)}%)</div>
                        </div>
                        <div>
                            <div className="text-xl font-bold text-green-400">
                                {analysis.total_samples - analysis.affected_samples}
                            </div>
                            <div className="text-xs text-text-secondary">Clean</div>
                        </div>
                    </div>
                    {/* Visual bar */}
                    <div className="mt-3 h-2 bg-bg-tertiary rounded-full overflow-hidden flex">
                        <div
                            className="bg-green-500"
                            style={{ width: `${((analysis.total_samples - analysis.affected_samples) / analysis.total_samples) * 100}%` }}
                        />
                        <div className="bg-yellow-500" style={{ width: `${analysis.warned_percentage}%` }} />
                        <div className="bg-red-500" style={{ width: `${analysis.blocked_percentage}%` }} />
                    </div>
                </div>
            )}

            {/* Existing Rules */}
            {rules.length > 0 && (
                <div className="space-y-2">
                    <h4 className="text-sm text-text-secondary">Active Rules ({rules.length})</h4>
                    {rules.map((rule) => (
                        <div
                            key={rule.id}
                            className={`p-3 rounded-lg border ${rule.enabled ? ACTION_COLORS[rule.action.type] : "border-border/30 opacity-60"}`}
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    {ACTION_ICONS[rule.action.type]}
                                    <span className="font-medium">{rule.name}</span>
                                    <span className={`text-xs px-1.5 py-0.5 rounded ${rule.enabled ? "bg-green-500/20 text-green-400" : "bg-gray-500/20 text-gray-400"}`}>
                                        {rule.enabled ? "Active" : "Disabled"}
                                    </span>
                                </div>
                                <div className="flex items-center gap-1">
                                    <button onClick={() => toggleRule(rule.id, !rule.enabled)} className="p-1 hover:bg-surface rounded">
                                        {rule.enabled ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : <AlertCircle className="w-4 h-4 text-gray-400" />}
                                    </button>
                                    <button onClick={() => setExpandedRuleId(expandedRuleId === rule.id ? null : rule.id)} className="p-1 hover:bg-surface rounded">
                                        {expandedRuleId === rule.id ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                    </button>
                                    <button onClick={() => deleteRule(rule.id)} className="p-1 hover:bg-red-500/20 rounded">
                                        <Trash2 className="w-4 h-4 text-red-400" />
                                    </button>
                                </div>
                            </div>
                            {expandedRuleId === rule.id && (
                                <div className="mt-2 pt-2 border-t border-border/30 text-sm text-text-secondary">
                                    <div>Patterns: {rule.condition.patterns.join(", ")}</div>
                                    <div>Threshold: {(rule.threshold * 100).toFixed(0)}%</div>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {/* Create New Rule */}
            <div className="p-4 rounded-lg bg-surface border border-border">
                <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
                    <Plus className="w-4 h-4" />
                    Create New Rule
                </h4>

                <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-3">
                        <input
                            type="text"
                            value={newRuleName}
                            onChange={(e) => setNewRuleName(e.target.value)}
                            placeholder="Rule name *"
                            className="px-3 py-2 rounded-lg bg-bg border border-border text-sm"
                        />
                        <input
                            type="text"
                            value={newRuleDescription}
                            onChange={(e) => setNewRuleDescription(e.target.value)}
                            placeholder="Description"
                            className="px-3 py-2 rounded-lg bg-bg border border-border text-sm"
                        />
                    </div>

                    <div className="grid grid-cols-4 gap-3">
                        <select value={matchType} onChange={(e) => setMatchType(e.target.value as any)} className="px-3 py-2 rounded-lg bg-bg border border-border text-sm">
                            <option value="contains">Contains</option>
                            <option value="regex">Regex</option>
                            <option value="semantic">Semantic</option>
                        </select>
                        <select value={field} onChange={(e) => setField(e.target.value as any)} className="px-3 py-2 rounded-lg bg-bg border border-border text-sm">
                            <option value="any">Any Field</option>
                            <option value="prompt">Prompt</option>
                            <option value="completion">Completion</option>
                            <option value="text">Text</option>
                        </select>
                        <select value={actionType} onChange={(e) => setActionType(e.target.value as any)} className="px-3 py-2 rounded-lg bg-bg border border-border text-sm">
                            <option value="block">Block</option>
                            <option value="warn">Warn</option>
                            <option value="modify">Modify</option>
                            <option value="downweight">Downweight</option>
                        </select>
                        <div className="flex items-center gap-2">
                            <input type="range" min="0" max="100" value={threshold} onChange={(e) => setThreshold(parseInt(e.target.value))} className="flex-1" />
                            <span className="text-sm w-10">{threshold}%</span>
                        </div>
                    </div>

                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={patternInput}
                            onChange={(e) => setPatternInput(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && addPattern()}
                            placeholder="Add pattern (Enter to add)"
                            className="flex-1 px-3 py-2 rounded-lg bg-bg border border-border text-sm"
                        />
                        <button onClick={addPattern} className="px-3 py-2 rounded-lg bg-accent/20 text-accent">
                            <Plus className="w-4 h-4" />
                        </button>
                    </div>

                    {patterns.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                            {patterns.map((p, i) => (
                                <span key={i} className="inline-flex items-center gap-1 px-2 py-1 rounded bg-accent/20 text-accent text-sm">
                                    {p}
                                    <button onClick={() => removePattern(i)}>
                                        <X className="w-3 h-3" />
                                    </button>
                                </span>
                            ))}
                        </div>
                    )}

                    <div className="flex justify-end">
                        <button
                            onClick={createRule}
                            disabled={isCreating || !newRuleName || patterns.length === 0}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-white disabled:opacity-50"
                        >
                            {isCreating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
                            Create Rule
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default GuardrailsView;
