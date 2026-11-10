export const sidebarItems = [
    { id: 'dashboard', title: 'Overview', icon: 'Home', path: '/' },
    { id: 'agents', title: 'Agents', icon: 'Bot', path: '/agents' },
    { id: 'projects', title: 'Projects', icon: 'Layers', path: '/projects' },
    { id: 'datasets', title: 'Datasets', icon: 'Database', path: '/datasets' },
    { id: 'curation', title: 'Data Curation', icon: 'Sparkles', path: '/curation' },
    { id: 'models', title: 'Model Registry', icon: 'Bot', path: '/models' },
    { id: 'serving', title: 'vLLM Serve', icon: 'Server', path: '/serving' },
    { id: 'training', title: 'Tuning Jobs', icon: 'Terminal', path: '/training' },
    { id: 'analytics', title: 'Analytics', icon: 'LineChart', path: '/analytics' },
] as const;

export const offlineItems = [
    { id: 'local-models', title: 'Local Models', icon: 'HardDrive', path: '/local-models' },
    { id: 'studio', title: 'Studio', icon: 'BrainCircuit', path: '/studio' },
    { id: 'co-work', title: 'Co-Work', icon: 'MessageSquare', path: '/co-work' },
] as const;

export const systemItems = [
    { id: 'settings', title: 'Settings', icon: 'Settings', path: '/settings' },
    { id: 'docs', title: 'Documentation', icon: 'BookOpen', path: '/documentation' },
] as const;

