/**
 * Langtrain Studio Design System - TypeScript Theme
 * Ported from Swift DesignSystem.swift
 */

export const theme = {
    colors: {
        // Backgrounds
        bgPrimary: '#0D0D0F',
        bgSidebar: '#0F0F11',
        bgCard: '#17171A',
        bgElevated: '#1F1F22',
        border: '#27272A',

        // Text
        textPrimary: '#FFFFFF',
        textSecondary: '#A1A1AA',
        textTertiary: '#71717A',

        // Accents
        accentCyan: '#22D3EE',
        accentGreen: '#22C55E',
        accentRed: '#EF4444',
        accentPurple: '#A855F7',
        accentBlue: '#3B82F6',
        accentOrange: '#F97316',
    },

    spacing: {
        xs: '4px',
        sm: '8px',
        md: '16px',
        lg: '24px',
        xl: '32px',
        '2xl': '48px',
    },

    radius: {
        sm: '4px',
        md: '8px',
        lg: '12px',
        xl: '16px',
        full: '9999px',
    },

    fonts: {
        sans: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        mono: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
        display: "'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif",
    },

    fontSizes: {
        xs: '11px',
        sm: '13px',
        base: '14px',
        lg: '16px',
        xl: '20px',
        '2xl': '24px',
        '3xl': '32px',
    },
} as const;

export type Theme = typeof theme;

// Sidebar navigation items
export const sidebarItems = [
    { id: 'dashboard', title: 'Dashboard', icon: 'Home', path: '/' },
    { id: 'agents', title: 'Agents', icon: 'Bot', path: '/agents' },
    { id: 'projects', title: 'Projects', icon: 'Layers', path: '/projects' },
    { id: 'datasets', title: 'Datasets', icon: 'Database', path: '/datasets' },
    { id: 'models', title: 'Models', icon: 'Cpu', path: '/models' },
    { id: 'training', title: 'Training', icon: 'Terminal', path: '/training' },
    { id: 'analytics', title: 'Analytics', icon: 'LineChart', path: '/analytics' },
] as const;

export const systemItems = [
    { id: 'settings', title: 'Settings', icon: 'Settings', path: '/settings' },
    { id: 'documentation', title: 'Documentation', icon: 'BookOpen', path: '/documentation' },
] as const;

export type SidebarItemId = typeof sidebarItems[number]['id'] | typeof systemItems[number]['id'];
