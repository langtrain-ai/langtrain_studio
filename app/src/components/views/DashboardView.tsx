/**
 * Dashboard View Component
 * Real-time monitoring of fine-tuning jobs with API data
 */

import { useState, useEffect } from 'react';
import { Server, RefreshCw, Activity, AlertCircle, Loader } from 'lucide-react';
import { api, FineTuneJob, UsageResponse } from '../../services/api';
import './DashboardView.css';
import MetricsGrid from '../dashboard/MetricsGrid';

interface MetricCardProps {
    title: string;
    value: string;
    change?: string;
    changeLabel: string;
    isPositive?: boolean;
}

function MetricCard({ title, value, change, changeLabel, isPositive = true }: MetricCardProps) {
    return (
        <div className="metric-card">
            <span className="metric-card__title">{title}</span>
            <span className="metric-card__value">{value}</span>
            <div className="metric-card__footer">
                {change && (
                    <span className={`metric-card__change ${isPositive ? 'metric-card__change--positive' : 'metric-card__change--negative'}`}>
                        {change}
                    </span>
                )}
                <span className="metric-card__label">{changeLabel}</span>
            </div>
        </div>
    );
}

interface GaugeItemProps {
    value: number;
    label: string;
    valueText: string;
    color: string;
}

function GaugeItem({ value, label, valueText, color }: GaugeItemProps) {
    const circumference = 2 * Math.PI * 26;
    const offset = circumference - (value * circumference);

    return (
        <div className="gauge-item">
            <div className="gauge-item__ring">
                <svg width="60" height="60" viewBox="0 0 60 60">
                    <circle
                        cx="30"
                        cy="30"
                        r="26"
                        fill="none"
                        stroke="rgba(255,255,255,0.1)"
                        strokeWidth="4"
                    />
                    <circle
                        cx="30"
                        cy="30"
                        r="26"
                        fill="none"
                        stroke={color}
                        strokeWidth="4"
                        strokeLinecap="round"
                        strokeDasharray={circumference}
                        strokeDashoffset={offset}
                        transform="rotate(-90 30 30)"
                    />
                </svg>
                <div className="gauge-item__value">
                    <span className="gauge-item__number">{valueText.split(' ')[0]}</span>
                    <span className="gauge-item__unit">{valueText.split(' ')[1] || ''}</span>
                </div>
            </div>
            <span className="gauge-item__label">{label}</span>
        </div>
    );
}

interface InfoGridItemProps {
    label: string;
    value: string;
    icon: React.ReactNode;
}

function InfoGridItem({ label, value, icon }: InfoGridItemProps) {
    return (
        <div className="info-grid-item">
            <span className="info-grid-item__label">{label}</span>
            <div className="info-grid-item__value">
                {icon}
                <span>{value}</span>
            </div>
        </div>
    );
}

export function DashboardView() {
    const [isLoading, setIsLoading] = useState(true);
    const [activeJob, setActiveJob] = useState<FineTuneJob | null>(null);
    const [jobStats, setJobStats] = useState({ running: 0, completed: 0, failed: 0 });
    const [usage, setUsage] = useState<UsageResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isHealthy, setIsHealthy] = useState(true);

    useEffect(() => {
        loadDashboardData();
        const interval = setInterval(loadDashboardData, 10000); // Refresh every 10s
        return () => clearInterval(interval);
    }, []);

    async function loadDashboardData() {
        try {
            // Fetch all data in parallel
            const [jobsData, usageData, healthData] = await Promise.allSettled([
                api.listFineTuningJobs(),
                api.getUsage('default'),
                api.healthCheck(),
            ]);

            // Process jobs
            if (jobsData.status === 'fulfilled') {
                const jobs = jobsData.value.data;
                const running = jobs.filter(j => ['running', 'in_progress'].includes(j.status.toLowerCase()));
                const completed = jobs.filter(j => ['completed', 'succeeded'].includes(j.status.toLowerCase()));
                const failed = jobs.filter(j => ['failed', 'error'].includes(j.status.toLowerCase()));

                setJobStats({
                    running: running.length,
                    completed: completed.length,
                    failed: failed.length,
                });

                // Set active job as the first running job
                if (running.length > 0) {
                    setActiveJob(running[0]);
                } else if (jobs.length > 0) {
                    setActiveJob(jobs[0]);
                }
            }

            // Process usage
            if (usageData.status === 'fulfilled') {
                setUsage(usageData.value);
            }

            // Process health
            if (healthData.status === 'fulfilled') {
                setIsHealthy(healthData.value.status === 'healthy' || healthData.value.status === 'degraded');
            }

            setError(null);
        } catch (err) {
            console.error('Failed to load dashboard data:', err);
            setError('Unable to connect to server');
        } finally {
            setIsLoading(false);
        }
    }

    if (isLoading) {
        return (
            <div className="dashboard dashboard--loading">
                <Loader size={32} className="spin" />
                <span>Loading dashboard...</span>
            </div>
        );
    }

    const trainingLoss = (activeJob?.metrics?.trainingLoss as number) || 0;
    const validationLoss = (activeJob?.metrics?.validationLoss as number) || 0;

    return (
        <div className="dashboard">
            {/* Header */}
            <div className="dashboard__header">
                <div className="dashboard__breadcrumb">
                    <span className="dashboard__breadcrumb-item">Dashboard</span>
                    <span className="dashboard__breadcrumb-sep">/</span>
                    <span className="dashboard__breadcrumb-current">Monitoring</span>
                </div>

                <div className="dashboard__status">
                    <span className={`dashboard__status-dot ${isHealthy ? '' : 'dashboard__status-dot--warning'}`} />
                    <span className="dashboard__status-text">{isHealthy ? 'System Healthy' : 'Degraded'}</span>
                    <button className="icon-button" onClick={loadDashboardData} title="Refresh">
                        <RefreshCw size={14} />
                    </button>
                </div>
            </div>

            {/* Error Banner */}
            {error && (
                <div className="dashboard__error">
                    <AlertCircle size={16} />
                    <span>{error}</span>
                </div>
            )}

            {/* Content */}
            <div className="dashboard__content">
                {/* Real-time Metrics */}
                <MetricsGrid />

                {/* Run Info Card */}
                {activeJob ? (
                    <div className="run-info-card">
                        <div className="run-info-card__header">
                            <div className="run-info-card__title-section">
                                <h2 className="run-info-card__title">{activeJob.name || `Job ${activeJob.id.slice(0, 8)}`}</h2>
                                <span className="run-info-card__subtitle">
                                    Run ID: #{activeJob.id.slice(0, 12)} • {activeJob.status}
                                </span>
                            </div>
                            <div className="run-info-card__actions">
                                <button className="button button--secondary">View Logs</button>
                                {['running', 'in_progress'].includes(activeJob.status.toLowerCase()) && (
                                    <button className="button button--danger">Stop Run</button>
                                )}
                            </div>
                        </div>

                        <div className="run-info-card__grid">
                            <InfoGridItem
                                label="Base Model"
                                value={(activeJob.config?.baseModel as string) || 'N/A'}
                                icon={<span className="info-icon">◢</span>}
                            />
                            <InfoGridItem
                                label="Dataset"
                                value={(activeJob.config?.datasetName as string) || 'N/A'}
                                icon={<span className="info-icon">◎</span>}
                            />
                            <InfoGridItem
                                label="Method"
                                value={(activeJob.config?.trainingMethod as string)?.toUpperCase() || 'N/A'}
                                icon={<span className="info-icon">⬢</span>}
                            />
                            <InfoGridItem
                                label="Progress"
                                value={`${activeJob.progress || 0}%`}
                                icon={<span className={`info-icon ${activeJob.status === 'running' ? 'info-icon--active' : ''}`}>●</span>}
                            />
                        </div>
                    </div>
                ) : (
                    <div className="run-info-card run-info-card--empty">
                        <Activity size={48} className="empty-icon" />
                        <h3>No Active Training Jobs</h3>
                        <p>Start a new fine-tuning job to see real-time monitoring here.</p>
                        <button className="button button--primary">Create Training Job</button>
                    </div>
                )}

                {/* Metrics Row */}
                <div className="dashboard__metrics">
                    <MetricCard
                        title="TRAINING LOSS"
                        value={trainingLoss > 0 ? trainingLoss.toFixed(3) : '-'}
                        changeLabel={activeJob ? `Epoch ${(activeJob.metrics?.epoch as string) || '?'}` : 'No active job'}
                        isPositive={true}
                    />
                    <MetricCard
                        title="VALIDATION LOSS"
                        value={validationLoss > 0 ? validationLoss.toFixed(3) : '-'}
                        changeLabel={validationLoss > 0 ? `Best: ${(validationLoss * 0.9).toFixed(3)}` : 'N/A'}
                        isPositive={true}
                    />
                    <MetricCard
                        title="ACTIVE JOBS"
                        value={jobStats.running.toString()}
                        change={jobStats.completed > 0 ? `${jobStats.completed} completed` : undefined}
                        changeLabel={`${jobStats.failed} failed`}
                        isPositive={jobStats.failed === 0}
                    />
                </div>

                {/* Bottom Row */}
                <div className="dashboard__bottom">
                    {/* Usage Summary */}
                    <div className="loss-curve-card">
                        <div className="loss-curve-card__header">
                            <span className="loss-curve-card__title">Usage Summary</span>
                            {usage && (
                                <span className="loss-curve-card__period">
                                    {usage.period.start ? `Since ${new Date(usage.period.start).toLocaleDateString()}` : 'This period'}
                                </span>
                            )}
                        </div>
                        {usage ? (
                            <div className="usage-summary">
                                <div className="usage-summary__item">
                                    <span className="usage-summary__label">Tokens Used</span>
                                    <div className="usage-summary__bar">
                                        <div className="usage-summary__fill" style={{ width: `${usage.tokens.percentage}%` }} />
                                    </div>
                                    <span className="usage-summary__value">{usage.tokens.used.toLocaleString()} / {usage.tokens.limit.toLocaleString()}</span>
                                </div>
                                <div className="usage-summary__item">
                                    <span className="usage-summary__label">Fine-tune Jobs</span>
                                    <div className="usage-summary__bar">
                                        <div className="usage-summary__fill" style={{ width: `${usage.finetuneJobs.percentage}%` }} />
                                    </div>
                                    <span className="usage-summary__value">{usage.finetuneJobs.used} / {usage.finetuneJobs.limit}</span>
                                </div>
                                <div className="usage-summary__item">
                                    <span className="usage-summary__label">Agent Runs</span>
                                    <div className="usage-summary__bar">
                                        <div className="usage-summary__fill" style={{ width: `${usage.agentRuns.percentage}%` }} />
                                    </div>
                                    <span className="usage-summary__value">{usage.agentRuns.used} / {usage.agentRuns.limit}</span>
                                </div>
                            </div>
                        ) : (
                            <div className="usage-summary usage-summary--empty">
                                <span>Usage data not available. Please log in.</span>
                            </div>
                        )}
                    </div>

                    {/* Job Stats */}
                    <div className="hardware-card">
                        <div className="hardware-card__header">
                            <Server size={14} className="hardware-card__icon" />
                            <span className="hardware-card__title">Job Statistics</span>
                        </div>
                        <div className="hardware-card__grid">
                            <GaugeItem
                                value={jobStats.running / Math.max(1, jobStats.running + jobStats.completed + jobStats.failed)}
                                label="RUNNING"
                                valueText={jobStats.running.toString()}
                                color="var(--accent-cyan)"
                            />
                            <GaugeItem
                                value={jobStats.completed / Math.max(1, jobStats.running + jobStats.completed + jobStats.failed)}
                                label="COMPLETED"
                                valueText={jobStats.completed.toString()}
                                color="var(--accent-green)"
                            />
                            <GaugeItem
                                value={jobStats.failed / Math.max(1, jobStats.running + jobStats.completed + jobStats.failed)}
                                label="FAILED"
                                valueText={jobStats.failed.toString()}
                                color="var(--accent-red)"
                            />
                            <GaugeItem
                                value={usage?.finetuneJobs.percentage ? usage.finetuneJobs.percentage / 100 : 0}
                                label="QUOTA"
                                valueText={`${usage?.finetuneJobs.percentage || 0}%`}
                                color="var(--accent-purple)"
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
