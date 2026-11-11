/**
 * SystemPerformanceMonitor.tsx
 * F1-Style System Telemetry Component
 *
 * Real-time CPU/GPU performance monitoring with flat, industrial design
 * No gradients, no shadows - pure data visualization
 */

import React, { useState, useEffect } from 'react';

// ============================================================================
// Type Definitions
// ============================================================================

interface SystemMetrics {
  cpu: {
    temp: number;
    utilization: number;
    clock: number;
    power: number;
  };
  gpu: {
    temp: number;
    utilization: number;
    vram: number;
    power: number;
  };
}

type MetricMode = 'CPU' | 'GPU';
type MetricStatus = 'safe' | 'warning' | 'critical';

interface GaugeProps {
  label: string;
  value: number;
  max: number;
  unit: string;
  status: MetricStatus;
}

// ============================================================================
// Mock Data Hook (Replace with real telemetry)
// ============================================================================

function useSystemMetrics(): SystemMetrics {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    cpu: { temp: 72, utilization: 82, clock: 4.1, power: 95 },
    gpu: { temp: 78, utilization: 96, vram: 88, power: 270 }
  });

  useEffect(() => {
    // Simulate live updates every 2 seconds
    const interval = setInterval(() => {
      setMetrics({
        cpu: {
          temp: 72 + Math.random() * 8 - 2,
          utilization: 82 + Math.random() * 15 - 5,
          clock: 4.1 + Math.random() * 0.4 - 0.2,
          power: 95 + Math.random() * 15 - 8
        },
        gpu: {
          temp: 78 + Math.random() * 10 - 2,
          utilization: 96 + Math.random() * 4 - 2,
          vram: 88 + Math.random() * 8 - 2,
          power: 270 + Math.random() * 40 - 20
        }
      });
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return metrics;
}

// ============================================================================
// Status Calculation
// ============================================================================

function getMetricStatus(value: number, safeThreshold: number, criticalThreshold: number): MetricStatus {
  if (value < safeThreshold) return 'safe';
  if (value < criticalThreshold) return 'warning';
  return 'critical';
}

// ============================================================================
// Color Constants (F1 Palette)
// ============================================================================

const COLORS = {
  background: '#000000',
  panel: '#111111',
  border: '#1F1F1F',
  textPrimary: '#E0E0E0',
  textSecondary: '#888888',
  safe: '#00FF85',
  warning: '#FFD600',
  critical: '#FF4C4C',
  accent: '#00C8FF'
};

// ============================================================================
// Gauge Component
// ============================================================================

const GaugeMeter: React.FC<GaugeProps> = ({ label, value, max, unit, status }) => {
  const percentage = (value / max) * 100;
  const strokeDasharray = `${percentage * 2.36} ${236 - percentage * 2.36}`; // 0.75 * 2 * π * 50 = 236

  const statusColor = {
    safe: COLORS.safe,
    warning: COLORS.warning,
    critical: COLORS.critical
  }[status];

  return (
    <div style={{
      padding: '12px',
      backgroundColor: COLORS.background,
      border: `1px solid ${COLORS.border}`,
      display: 'flex',
      flexDirection: 'column',
      gap: '10px'
    }}>
      {/* Label with status indicator */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '4px'
      }}>
        <div style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          backgroundColor: statusColor
        }} />
        <span style={{
          fontSize: '9px',
          fontWeight: 500,
          color: COLORS.textSecondary,
          letterSpacing: '0.6px',
          textTransform: 'uppercase'
        }}>
          {label}
        </span>
      </div>

      {/* Gauge SVG */}
      <div style={{
        position: 'relative',
        width: '100px',
        height: '100px',
        margin: '0 auto'
      }}>
        <svg width="100" height="100" viewBox="0 0 100 100">
          {/* Background arc */}
          <circle
            cx="50"
            cy="50"
            r="37.5"
            fill="none"
            stroke={COLORS.border}
            strokeWidth="10"
            strokeDasharray="236 78.5" // 0.75 of circle
            strokeDashoffset="0"
            transform="rotate(135 50 50)"
            strokeLinecap="butt"
          />

          {/* Value arc */}
          <circle
            cx="50"
            cy="50"
            r="37.5"
            fill="none"
            stroke={statusColor}
            strokeWidth="10"
            strokeDasharray={strokeDasharray}
            strokeDashoffset="0"
            transform="rotate(135 50 50)"
            strokeLinecap="butt"
            style={{ transition: 'stroke-dasharray 0.3s linear' }}
          />
        </svg>

        {/* Center text */}
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center'
        }}>
          <div style={{
            fontSize: '24px',
            fontWeight: 'bold',
            fontFamily: 'SF Mono, monospace',
            color: COLORS.textPrimary
          }}>
            {Math.round(value)}
          </div>
          <div style={{
            fontSize: '10px',
            fontWeight: 500,
            color: COLORS.textSecondary
          }}>
            {unit}
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

const SystemPerformanceMonitor: React.FC = () => {
  const [mode, setMode] = useState<MetricMode>('GPU');
  const metrics = useSystemMetrics();

  // CPU Gauges
  const cpuGauges: GaugeProps[] = [
    {
      label: 'TEMP',
      value: metrics.cpu.temp,
      max: 100,
      unit: '°C',
      status: getMetricStatus(metrics.cpu.temp, 75, 85)
    },
    {
      label: 'UTIL',
      value: metrics.cpu.utilization,
      max: 100,
      unit: '%',
      status: getMetricStatus(metrics.cpu.utilization, 85, 90)
    },
    {
      label: 'CLOCK',
      value: metrics.cpu.clock,
      max: 5,
      unit: 'GHz',
      status: 'safe'
    },
    {
      label: 'POWER',
      value: metrics.cpu.power,
      max: 125,
      unit: 'W',
      status: getMetricStatus(metrics.cpu.power, 100, 115)
    }
  ];

  // GPU Gauges
  const gpuGauges: GaugeProps[] = [
    {
      label: 'TEMP',
      value: metrics.gpu.temp,
      max: 100,
      unit: '°C',
      status: getMetricStatus(metrics.gpu.temp, 80, 85)
    },
    {
      label: 'UTIL',
      value: metrics.gpu.utilization,
      max: 100,
      unit: '%',
      status: getMetricStatus(metrics.gpu.utilization, 90, 95)
    },
    {
      label: 'VRAM',
      value: metrics.gpu.vram,
      max: 100,
      unit: '%',
      status: getMetricStatus(metrics.gpu.vram, 85, 95)
    },
    {
      label: 'POWER',
      value: metrics.gpu.power,
      max: 350,
      unit: 'W',
      status: getMetricStatus(metrics.gpu.power, 300, 330)
    }
  ];

  const gauges = mode === 'CPU' ? cpuGauges : gpuGauges;

  return (
    <div style={{
      backgroundColor: COLORS.panel,
      border: `1px solid ${COLORS.border}`,
      padding: '16px',
      fontFamily: 'SF Pro Rounded, system-ui, sans-serif'
    }}>
      {/* Header */}
      <div style={{
        marginBottom: '16px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <h2 style={{
            margin: 0,
            fontSize: '11px',
            fontWeight: 600,
            color: COLORS.textSecondary,
            letterSpacing: '1px',
            textTransform: 'uppercase'
          }}>
            SYSTEM PERFORMANCE MONITOR
          </h2>
          <p style={{
            margin: '4px 0 0 0',
            fontSize: '10px',
            color: COLORS.textSecondary
          }}>
            Live hardware telemetry during fine-tuning
          </p>
        </div>

        {/* Mode Toggle */}
        <div style={{
          display: 'flex',
          border: `1px solid ${COLORS.border}`,
          backgroundColor: COLORS.background
        }}>
          <button
            onClick={() => setMode('CPU')}
            style={{
              padding: '8px 16px',
              fontSize: '11px',
              fontWeight: 'bold',
              color: mode === 'CPU' ? COLORS.accent : COLORS.textSecondary,
              backgroundColor: mode === 'CPU' ? COLORS.panel : 'transparent',
              border: 'none',
              cursor: 'pointer',
              textTransform: 'uppercase',
              letterSpacing: '0.8px',
              transition: 'all 0.2s'
            }}
          >
            CPU
          </button>
          <div style={{
            width: '1px',
            backgroundColor: COLORS.border
          }} />
          <button
            onClick={() => setMode('GPU')}
            style={{
              padding: '8px 16px',
              fontSize: '11px',
              fontWeight: 'bold',
              color: mode === 'GPU' ? COLORS.accent : COLORS.textSecondary,
              backgroundColor: mode === 'GPU' ? COLORS.panel : 'transparent',
              border: 'none',
              cursor: 'pointer',
              textTransform: 'uppercase',
              letterSpacing: '0.8px',
              transition: 'all 0.2s'
            }}
          >
            GPU
          </button>
        </div>
      </div>

      {/* Gauge Grid (2x2) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '16px',
        marginBottom: '16px'
      }}>
        {gauges.map((gauge, index) => (
          <GaugeMeter key={index} {...gauge} />
        ))}
      </div>

      {/* Bottom Stats Strip */}
      <div style={{
        borderTop: `1px solid ${COLORS.border}`,
        paddingTop: '12px',
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '10px'
      }}>
        <div>
          <span style={{ color: COLORS.textSecondary }}>LAST UPDATE:</span>{' '}
          <span style={{ color: COLORS.textPrimary, fontFamily: 'SF Mono, monospace' }}>
            {new Date().toLocaleTimeString()}
          </span>
        </div>
        <div style={{ display: 'flex', gap: '16px' }}>
          <div>
            <span style={{ color: COLORS.textSecondary }}>MODE:</span>{' '}
            <span style={{ color: COLORS.accent, fontWeight: 'bold' }}>{mode}</span>
          </div>
          <div>
            <span style={{ color: COLORS.textSecondary }}>STATUS:</span>{' '}
            <span style={{ color: COLORS.safe, fontWeight: 'bold' }}>OPERATIONAL</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemPerformanceMonitor;
