/**
 * Analytics View Component
 * View training metrics and insights from real API data
 */

import { useState, useEffect } from 'react';
import {
    BarChart3,
    TrendingDown,
    Clock,
    Zap,
    Target,
    Activity,
    PieChart,
    ArrowUpRight,
    ArrowDownRight,
    Loader
} from 'lucide-react';
import { api, UsageResponse, FineTuneJob } from '../../services/api';
import './AnalyticsView.css';

interface StatCardProps {
    title: string;
    value: string;
    change?: string;
    changeType?: 'positive' | 'negative' | 'neutral';
    icon: React.ReactNode;
}

function StatCard({ title, value, change, changeType = 'neutral', icon }: StatCardProps) {
    return (
        <div className="analytics-stat-card">
            <div className="analytics-stat-card__icon">{icon}</div>
            <div className="analytics-stat-card__content">
                <span className="analytics-stat-card__title">{title}</span>
                <span className="analytics-stat-card__value">{value}</span>
                {change && (
                    <span className={`analytics-stat-card__change analytics-stat-card__change--${changeType}`}>
                        {changeType === 'positive' ? <ArrowUpRight size={12} /> :
                            changeType === 'negative' ? <ArrowDownRight size={12} /> : null}
                        {change}
                    </span>
                )}
            </div>
        </div>
    );
}

interface BarChartVizProps {
    data: number[];
}

function BarChartViz({ data }: BarChartVizProps) {
    const maxValue = Math.max(...data, 1);

    return (
        <div className="bar-chart">
            {data.map((value, index) => (
                <div key={index} className="bar-chart__column">
                    <div
                        className="bar-chart__bar"
                        style={{ height: `${(value / maxValue) * 100}%` }}
                    />
                    <span className="bar-chart__label">
                        {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][index]}
                    </span>
                </div>
            ))}
        </div>
    );
}

interface DonutChartProps {
    data: { label: string; value: number; color: string }[];
}

function DonutChart({ data }: DonutChartProps) {
    const total = data.reduce((acc, d) => acc + d.value, 0);
    if (total === 0) {
        return (
            <div className="donut-chart-container">
                <div className="chart-empty">No data available</div>
            </div>
        );
    }
    let currentAngle = -90;

    return (
        <div className="donut-chart-container">
            <svg className="donut-chart" viewBox="0 0 100 100">
                {data.map((item, index) => {
                    const angle = (item.value / total) * 360;
                    const startAngle = currentAngle;
                    currentAngle += angle;

                    const x1 = 50 + 35 * Math.cos((startAngle * Math.PI) / 180);
                    const y1 = 50 + 35 * Math.sin((startAngle * Math.PI) / 180);
                    const x2 = 50 + 35 * Math.cos(((startAngle + angle) * Math.PI) / 180);
                    const y2 = 50 + 35 * Math.sin(((startAngle + angle) * Math.PI) / 180);

                    const largeArcFlag = angle > 180 ? 1 : 0;

                    return (
                        <path
                            key={index}
                            d={`M 50 50 L ${x1} ${y1} A 35 35 0 ${largeArcFlag} 1 ${x2} ${y2} Z`}
                            fill={item.color}
                        />
                    );
                })}
                <circle cx="50" cy="50" r="22" fill="var(--bg-card)" />
            </svg>
            <div className="donut-chart__legend">
                {data.map((item, index) => (
                    <div key={index} className="legend-item">
                        <span className="legend-dot" style={{ backgroundColor: item.color }} />
                        <span className="legend-label">{item.label}</span>
                        <span className="legend-value">{item.value}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

export function AnalyticsView() {
    const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');
    const [isLoading, setIsLoading] = useState(true);
    const [usage, setUsage] = useState<UsageResponse | null>(null);
    const [jobs, setJobs] = useState<FineTuneJob[]>([]);

    useEffect(() => {
        loadAnalytics();
    }, [timeRange]);

    async function loadAnalytics() {
        try {
            setIsLoading(true);
            const [usageData, jobsData] = await Promise.all([
                api.getUsage('default'),
                api.listFineTuningJobs(),
            ]);
            setUsage(usageData);
            setJobs(jobsData.data);
        } catch (err) {
            console.error('Failed to load analytics:', err);
            setUsage(null);
            setJobs([]);
        } finally {
            setIsLoading(false);
        }
    }

    // Calculate job status distribution
    const jobStatusData = [
        { label: 'Completed', value: jobs.filter(j => ['completed', 'succeeded'].includes(j.status.toLowerCase())).length, color: 'var(--accent-green)' },
        { label: 'Running', value: jobs.filter(j => ['running', 'in_progress'].includes(j.status.toLowerCase())).length, color: 'var(--accent-cyan)' },
        { label: 'Failed', value: jobs.filter(j => ['failed', 'error'].includes(j.status.toLowerCase())).length, color: 'var(--accent-red)' },
        { label: 'Pending', value: jobs.filter(j => ['pending', 'queued'].includes(j.status.toLowerCase())).length, color: 'var(--accent-orange)' },
    ];

    // Calculate success rate
    const completedCount = jobStatusData[0].value;
    const failedCount = jobStatusData[2].value;
    const totalFinished = completedCount + failedCount;
    const successRate = totalFinished > 0 ? ((completedCount / totalFinished) * 100).toFixed(1) : '-';

    // Monthly job data (from actual jobs)
    const monthlyData = Array(12).fill(0);
    jobs.forEach(job => {
        const month = new Date(job.createdAt).getMonth();
        monthlyData[month]++;
    });

    if (isLoading) {
        return (
            <div className="analytics-view analytics-view--loading">
                <Loader size={32} className="spin" />
                <span>Loading analytics...</span>
            </div>
        );
    }

    return (
        <div className="analytics-view">
            {/* Header */}
            <div className="analytics-view__header">
                <div className="analytics-view__title-section">
                    <h1 className="analytics-view__title">Analytics</h1>
                    <p className="analytics-view__subtitle">View training metrics and insights</p>
                </div>
                <div className="analytics-view__actions">
                    <div className="time-range-selector">
                        <button
                            className={timeRange === '7d' ? 'active' : ''}
                            onClick={() => setTimeRange('7d')}
                        >
                            7 days
                        </button>
                        <button
                            className={timeRange === '30d' ? 'active' : ''}
                            onClick={() => setTimeRange('30d')}
                        >
                            30 days
                        </button>
                        <button
                            className={timeRange === '90d' ? 'active' : ''}
                            onClick={() => setTimeRange('90d')}
                        >
                            90 days
                        </button>
                    </div>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="analytics-view__stats">
                <StatCard
                    title="Total Jobs"
                    value={jobs.length.toString()}
                    icon={<Activity size={20} />}
                />
                <StatCard
                    title="Success Rate"
                    value={successRate !== '-' ? `${successRate}%` : '-'}
                    changeType={Number(successRate) > 80 ? 'positive' : 'neutral'}
                    icon={<Target size={20} />}
                />
                <StatCard
                    title="Fine-tune Jobs Used"
                    value={usage ? `${usage.finetuneJobs.used}/${usage.finetuneJobs.limit}` : '-'}
                    icon={<Clock size={20} />}
                />
                <StatCard
                    title="Tokens Used"
                    value={usage ? usage.tokens.used.toLocaleString() : '-'}
                    icon={<Zap size={20} />}
                />
            </div>

            {/* Charts Row */}
            <div className="analytics-view__charts">
                {/* Training Jobs Chart */}
                <div className="chart-card chart-card--wide">
                    <div className="chart-card__header">
                        <div className="chart-card__title">
                            <BarChart3 size={16} />
                            <span>Training Jobs by Month</span>
                        </div>
                        <span className="chart-card__subtitle">Jobs started per month</span>
                    </div>
                    <div className="chart-card__content">
                        <BarChartViz data={monthlyData} />
                    </div>
                </div>

                {/* Job Status Distribution */}
                <div className="chart-card">
                    <div className="chart-card__header">
                        <div className="chart-card__title">
                            <PieChart size={16} />
                            <span>Job Status</span>
                        </div>
                        <span className="chart-card__subtitle">Distribution by status</span>
                    </div>
                    <div className="chart-card__content">
                        <DonutChart data={jobStatusData} />
                    </div>
                </div>
            </div>

            {/* Usage Info */}
            <div className="analytics-view__bottom">
                <div className="chart-card chart-card--full">
                    <div className="chart-card__header">
                        <div className="chart-card__title">
                            <TrendingDown size={16} />
                            <span>Usage Summary</span>
                        </div>
                        <span className="chart-card__subtitle">
                            {usage?.period.start ? `Since ${new Date(usage.period.start).toLocaleDateString()}` : 'Current period'}
                        </span>
                    </div>
                    <div className="chart-card__content">
                        {usage ? (
                            <div className="usage-grid">
                                <div className="usage-item">
                                    <span className="usage-item__label">Tokens</span>
                                    <div className="usage-item__bar">
                                        <div
                                            className="usage-item__fill"
                                            style={{ width: `${usage.tokens.percentage}%` }}
                                        />
                                    </div>
                                    <span className="usage-item__value">{usage.tokens.percentage}%</span>
                                </div>
                                <div className="usage-item">
                                    <span className="usage-item__label">Fine-tune Jobs</span>
                                    <div className="usage-item__bar">
                                        <div
                                            className="usage-item__fill"
                                            style={{ width: `${usage.finetuneJobs.percentage}%` }}
                                        />
                                    </div>
                                    <span className="usage-item__value">{usage.finetuneJobs.percentage}%</span>
                                </div>
                                <div className="usage-item">
                                    <span className="usage-item__label">Agent Runs</span>
                                    <div className="usage-item__bar">
                                        <div
                                            className="usage-item__fill"
                                            style={{ width: `${usage.agentRuns.percentage}%` }}
                                        />
                                    </div>
                                    <span className="usage-item__value">{usage.agentRuns.percentage}%</span>
                                </div>
                            </div>
                        ) : (
                            <div className="chart-empty">
                                <span>Usage data not available. Please log in to view your usage.</span>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
