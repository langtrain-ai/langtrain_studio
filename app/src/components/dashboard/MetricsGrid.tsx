import { useEffect, useState } from 'react';
// import { Card } from '@/components/ui/card';
import { Activity, Server, Clock, BarChart3 } from 'lucide-react';

interface MetricsData {
    uptime_seconds: number;
    requests: {
        total: number;
        avg_latency_ms: number;
    };
    training: {
        jobs_completed: number;
        success_rate: number;
    };
    inference: {
        requests: number;
    };
}

import { API_CONFIG } from '../../services/api';

const MetricsGrid = () => {
    const [metrics, setMetrics] = useState<MetricsData | null>(null);

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                // Fetch metrics from the actual configure backend 
                const res = await fetch(`${API_CONFIG.baseURL}/metrics`);
                if (res.ok) {
                    const data = await res.json();
                    setMetrics(data);
                }
            } catch (e) {
                console.error("Failed to fetch metrics", e);
            }
        }
        fetchMetrics();
        const interval = setInterval(fetchMetrics, 5000);
        return () => clearInterval(interval);
    }, []);

    const formatUptime = (seconds: number) => {
        const hrs = Math.floor(seconds / 3600);
        return `${hrs}h`;
    };

    const cards = [
        { title: 'Requests', val: metrics?.requests.total.toLocaleString() ?? '-', icon: Activity },
        { title: 'Latency', val: metrics ? `${metrics.requests.avg_latency_ms}ms` : '-', icon: Clock },
        { title: 'Training', val: metrics?.training.jobs_completed.toLocaleString() ?? '-', icon: BarChart3 },
        { title: 'Uptime', val: metrics ? formatUptime(metrics.uptime_seconds) : '-', icon: Server },
    ];

    return (
        <div className="metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-md)', marginBottom: 'var(--space-lg)' }}>
            {cards.map((c, i) => (
                <div key={i} className="metric-card" style={{
                    padding: 'var(--space-md)',
                    background: 'var(--bg-glass)',
                    borderRadius: 'var(--radius-lg)',
                    border: '1px solid var(--border-color)',
                    backdropFilter: 'blur(12px)'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 'var(--space-sm)' }}>
                        <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-tertiary)' }}>{c.title}</span>
                        <c.icon size={16} color="var(--accent-cyan)" />
                    </div>
                    <div style={{ fontSize: 'var(--text-2xl)', fontWeight: 'bold', color: 'var(--text-primary)' }}>{c.val}</div>
                </div>
            ))}
        </div>
    );
};

export default MetricsGrid;
