
export interface AppSettings {
    apiBaseUrl: string;
    requestTimeoutMs: number;
    maxRetries: number;
    theme: 'system' | 'light' | 'dark';
}

const DEFAULT_SETTINGS: AppSettings = {
    apiBaseUrl: 'https://api.langtrain.xyz',
    requestTimeoutMs: 30000,
    maxRetries: 3,
    theme: 'dark'
};

const SETTINGS_KEY = 'langtrain_studio_settings';

export class SettingsManager {
    private static instance: SettingsManager;
    private settings: AppSettings;

    private constructor() {
        this.settings = this.loadSettings();
    }

    public static getInstance(): SettingsManager {
        if (!SettingsManager.instance) {
            SettingsManager.instance = new SettingsManager();
        }
        return SettingsManager.instance;
    }

    private loadSettings(): AppSettings {
        try {
            const stored = localStorage.getItem(SETTINGS_KEY);
            if (!stored) return DEFAULT_SETTINGS;

            return {
                ...DEFAULT_SETTINGS,
                ...JSON.parse(stored)
            };
        } catch (e) {
            console.error('Failed to load settings', e);
            return DEFAULT_SETTINGS;
        }
    }

    public get(): AppSettings {
        return { ...this.settings };
    }

    public getApiUrl(version: string = 'v1'): string {
        // Ensure proper /api/{version} base path matching langtrain-server mounting
        const base = this.settings.apiBaseUrl.replace(/\/+$/, "");
        if (base.endsWith(`/api/${version}`)) return base;
        if (base.endsWith(`/${version}`)) return base;
        if (base.endsWith('/api')) return `${base}/${version}`;
        return `${base}/api/${version}`;
    }

    public save(newSettings: Partial<AppSettings>): void {
        this.settings = {
            ...this.settings,
            ...newSettings
        };
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(this.settings));

        // Dispatch event for reactive updates if needed
        window.dispatchEvent(new CustomEvent('settings-changed', { detail: this.settings }));
    }

    public reset(): void {
        this.settings = DEFAULT_SETTINGS;
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(DEFAULT_SETTINGS));
        window.dispatchEvent(new CustomEvent('settings-changed', { detail: DEFAULT_SETTINGS }));
    }
}

export const settings = SettingsManager.getInstance();
