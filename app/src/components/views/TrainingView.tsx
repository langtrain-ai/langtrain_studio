/**
 * Training View Component
 * Monitor and manage training jobs
 */

import { useState, useEffect } from 'react';
import {
    Activity,
    StopCircle,
    RefreshCw,
    Terminal,
    Cpu,
    Loader,
    CheckCircle,
    TrendingDown,
    BrainCircuit
} from 'lucide-react';
import { api, FineTuneJob } from '../../services/api';
import { ConfigurationWidget, defaultConfig, TrainingConfig } from '../common/ConfigurationWidget';
import SystemPerformanceMonitor from '../dashboard/SystemPerformanceMonitor';
import { HardwareEstimator } from '../dashboard/HardwareEstimator';
import './TrainingView.css';

interface JobRowProps {
    job: FineTuneJob;
    onCancel: (id: string) => void;
    isSelected: boolean;
    onSelect: (job: FineTuneJob) => void;
}

function getStatusColor(status: string): string {
    switch (status.toLowerCase()) {
        case 'running':
        case 'in_progress':
            return 'var(--accent-cyan)';
        case 'completed':
        case 'succeeded':
            return 'var(--accent-green)';
        case 'failed':
        case 'error':
            return 'var(--accent-red)';
        case 'pending':
        case 'queued':
            return 'var(--accent-orange)';
        default:
            return 'var(--text-tertiary)';
    }
}

function JobRow({ job, onCancel, isSelected, onSelect }: JobRowProps) {
    const isRunning = ['running', 'in_progress'].includes(job.status.toLowerCase());

    return (
        <div
            className={`job-row ${isSelected ? 'job-row--selected' : ''}`}
            onClick={() => onSelect(job)}
        >
            <div className="job-row__status">
                <span
                    className="status-dot"
                    style={{ backgroundColor: getStatusColor(job.status) }}
                />
            </div>
            <div className="job-row__info">
                <span className="job-row__name">{job.name || `Job ${job.id.slice(0, 8)}`}</span>
                <span className="job-row__id">{job.id.slice(0, 12)}</span>
            </div>
            <div className="job-row__progress">
                {isRunning ? (
                    <div className="mini-progress">
                        <div
                            className="mini-progress__fill"
                            style={{ width: `${job.progress}%` }}
                        />
                    </div>
                ) : (
                    <span className="job-row__status-text">
                        {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                    </span>
                )}
            </div>
            <div className="job-row__actions">
                {isRunning && (
                    <button
                        className="icon-button icon-button--danger"
                        onClick={(e) => { e.stopPropagation(); onCancel(job.id); }}
                    >
                        <StopCircle size={14} />
                    </button>
                )}
            </div>
        </div>
    );
}

interface MetricTileProps {
    label: string;
    value: string;
    trend?: string;
    icon: React.ReactNode;
}

function MetricTile({ label, value, trend, icon }: MetricTileProps) {
    return (
        <div className="metric-tile">
            <div className="metric-tile__icon">{icon}</div>
            <div className="metric-tile__content">
                <span className="metric-tile__value">{value}</span>
                <span className="metric-tile__label">{label}</span>
            </div>
            {trend && <span className="metric-tile__trend">{trend}</span>}
        </div>
    );
}

export function TrainingView() {
    const [jobs, setJobs] = useState<FineTuneJob[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [selectedJob, setSelectedJob] = useState<FineTuneJob | null>(null);

    const [activeTab, setActiveTab] = useState<'metrics' | 'logs' | 'estimator'>('estimator');
    const [showConfig, setShowConfig] = useState(false);
    const [newJobConfig, setNewJobConfig] = useState<TrainingConfig>(defaultConfig);

    useEffect(() => {
        loadJobs();
        const interval = setInterval(loadJobs, 5000);
        return () => clearInterval(interval);
    }, []);

    async function loadJobs() {
        try {
            const response = await api.listFineTuningJobs();
            setJobs(response.data);
            if (response.data.length > 0 && !selectedJob) {
                setSelectedJob(response.data[0]);
            }
        } catch (err) {
            console.error('Failed to load jobs:', err);
            setJobs([]);
        } finally {
            setIsLoading(false);
        }
    }

    async function handleCancelJob(id: string) {
        try {
            await api.cancelFineTuningJob(id);
            loadJobs();
        } catch (err) {
            console.error('Failed to cancel job:', err);
        }
    }

    async function handleStartJob() {
        console.log("Starting job with config:", newJobConfig);
        // Here we would call api.createJob(newJobConfig)
        setShowConfig(false);
        // Simulate job creation
        setTimeout(loadJobs, 1000);
    }

    const runningJobs = jobs.filter(j => ['running', 'in_progress'].includes(j.status.toLowerCase()));
    const completedJobs = jobs.filter(j => ['completed', 'succeeded'].includes(j.status.toLowerCase()));

    return (
        <div className="training-view">
            {/* Left Panel - Job List */}
            <div className="training-view__sidebar">
                <div className="sidebar-header">
                    <h2>Training Jobs</h2>
                    <div style={{ display: 'flex', gap: '8px' }}>
                        <button className="icon-button" onClick={() => setShowConfig(true)} title="New Job">
                            <span style={{ fontSize: '18px', fontWeight: 'bold' }}>+</span>
                        </button>
                        <button className="icon-button" onClick={loadJobs}>
                            <RefreshCw size={14} />
                        </button>
                    </div>
                </div>

                <div className="sidebar-stats">
                    <div className="sidebar-stat">
                        <Loader size={14} className={runningJobs.length > 0 ? 'spin' : ''} />
                        <span>{runningJobs.length} Running</span>
                    </div>
                    <div className="sidebar-stat">
                        <CheckCircle size={14} />
                        <span>{completedJobs.length} Completed</span>
                    </div>
                </div>

                <div className="job-list">
                    {isLoading ? (
                        <div className="job-list__loading">Loading...</div>
                    ) : jobs.length === 0 ? (
                        <div className="job-list__empty">No training jobs</div>
                    ) : (
                        jobs.map(job => (
                            <JobRow
                                key={job.id}
                                job={job}
                                onCancel={handleCancelJob}
                                isSelected={selectedJob?.id === job.id}
                                onSelect={setSelectedJob}
                            />
                        ))
                    )}
                </div>
            </div>

            {/* Right Panel - Job Details */}
            <div className="training-view__main">
                {selectedJob ? (
                    <>
                        <div className="job-detail-header">
                            <div className="job-detail-header__info">
                                <h1>{selectedJob.name || `Job ${selectedJob.id.slice(0, 8)}`}</h1>
                                <span className="job-id">{selectedJob.id}</span>
                            </div>
                            <div className="job-detail-header__status">
                                <span
                                    className="status-badge"
                                    style={{
                                        backgroundColor: `${getStatusColor(selectedJob.status)}20`,
                                        color: getStatusColor(selectedJob.status)
                                    }}
                                >
                                    {selectedJob.status}
                                </span>
                            </div>
                        </div>

                        {/* Progress Bar */}
                        {['running', 'in_progress'].includes(selectedJob.status.toLowerCase()) && (
                            <div className="job-progress-section">
                                <div className="job-progress">
                                    <div
                                        className="job-progress__fill"
                                        style={{ width: `${selectedJob.progress}%` }}
                                    />
                                </div>
                                <span className="job-progress__label">{selectedJob.progress}% complete</span>
                            </div>
                        )}

                        {/* Metrics Grid */}
                        <div className="metrics-grid">
                            <MetricTile
                                label="Training Loss"
                                value={(selectedJob.metrics?.trainingLoss as number)?.toFixed(3) || '-'}
                                trend={selectedJob.metrics?.trainingLoss ? "-12%" : undefined}
                                icon={<TrendingDown size={16} />}
                            />
                            <MetricTile
                                label="Validation Loss"
                                value={(selectedJob.metrics?.validationLoss as number)?.toFixed(3) || '-'}
                                trend={selectedJob.metrics?.validationLoss ? "-5%" : undefined}
                                icon={<TrendingDown size={16} />}
                            />
                            <MetricTile
                                label="Epoch"
                                value={(selectedJob.metrics?.epoch as string) || '-'}
                                icon={<Activity size={16} />}
                            />
                            <MetricTile
                                label="GPU Utilization"
                                value={(selectedJob.metrics?.gpuUtilization as string) || '-'}
                                icon={<Cpu size={16} />}
                            />
                        </div>

                        {/* Tabs */}
                        <div className="detail-tabs">
                            <button
                                className={`tab-button ${activeTab === 'estimator' ? 'tab-button--active' : ''}`}
                                onClick={() => setActiveTab('estimator')}
                            >
                                <BrainCircuit size={14} />
                                Hardware Estimator
                            </button>
                            <button
                                className={`tab-button ${activeTab === 'metrics' ? 'tab-button--active' : ''}`}
                                onClick={() => setActiveTab('metrics')}
                            >
                                <Activity size={14} />
                                Metrics
                            </button>
                            <button
                                className={`tab-button ${activeTab === 'logs' ? 'tab-button--active' : ''}`}
                                onClick={() => setActiveTab('logs')}
                            >
                                <Terminal size={14} />
                                Logs
                            </button>
                        </div>

                        {/* Tab Content */}
                        <div className="detail-content" style={{ padding: activeTab === 'estimator' ? '16px' : undefined }}>
                            {activeTab === 'estimator' ? (
                                <HardwareEstimator />
                            ) : activeTab === 'metrics' ? (
                                <div className="metrics-chart" style={{ padding: '0', display: 'flex', flexDirection: 'column' }}>
                                    <SystemPerformanceMonitor />
                                </div>
                            ) : (
                                <div className="logs-container">
                                    <pre className="logs-output">
                                        {(selectedJob.metrics?.logs as string) || 'No logs available yet...'}
                                    </pre>
                                </div>
                            )}
                        </div>
                    </>
                ) : (
                    <div className="no-selection">
                        <Activity size={48} />
                        <h3>No job selected</h3>
                        <p>Select a training job from the list to view details</p>
                    </div>
                )}
            </div>

            {/* Modal Overlay */}
            {showConfig && (
                <div className="modal-overlay">
                    <div className="modal-content">
                        <button className="modal-close" onClick={() => setShowConfig(false)}>×</button>
                        <ConfigurationWidget
                            config={newJobConfig}
                            setConfig={setNewJobConfig}
                            onConfirm={handleStartJob}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
