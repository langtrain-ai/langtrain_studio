/**
 * Settings View Component
 * Configure workspace settings
 */

import { useState, useEffect } from 'react';
import {
    Settings,
    User,
    Key,
    Bell,
    Palette,
    Server,
    HardDrive,
    Shield,
    Copy,
    Check,
    Moon,
    Sun,
    Monitor
} from 'lucide-react';
import { useAuth } from '../../services/auth';
import { settings } from '../../lib/settings';
import { secureStorage } from '../../lib/storage';
import './SettingsView.css';

type SettingsTab = 'profile' | 'api' | 'notifications' | 'appearance' | 'storage' | 'privacy';

interface TabButtonProps {
    id: SettingsTab;
    label: string;
    icon: React.ReactNode;
    active: boolean;
    onClick: (id: SettingsTab) => void;
}

function TabButton({ id, label, icon, active, onClick }: TabButtonProps) {
    return (
        <button
            className={`settings-tab-button ${active ? 'settings-tab-button--active' : ''}`}
            onClick={() => onClick(id)}
        >
            {icon}
            <span>{label}</span>
        </button>
    );
}

function ProfileSettings() {
    const { user, isAuthenticated } = useAuth();
    const [name, setName] = useState(user?.name || '');
    const [email, setEmail] = useState(user?.email || '');
    const [saved, setSaved] = useState(false);

    useEffect(() => {
        if (user) {
            setName(user.name || '');
            setEmail(user.email || '');
        }
    }, [user]);

    const handleSave = () => {
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    };

    return (
        <div className="settings-section">
            <h2>Profile Settings</h2>
            <p className="settings-description">
                {isAuthenticated ? `Signed in as ${user?.email}` : 'Manage your workspace profile'}
            </p>

            <div className="settings-form">
                <div className="form-group">
                    <label>Display Name</label>
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="Your name"
                    />
                </div>
                <div className="form-group">
                    <label>Email</label>
                    <input
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="your@email.com"
                        disabled={isAuthenticated}
                    />
                </div>
                <button className="button button--primary" onClick={handleSave}>
                    {saved ? 'Saved ✓' : 'Save Changes'}
                </button>
            </div>
        </div>
    );
}

function APISettings() {
    const token = secureStorage.getItem('langtrain_auth_token') || '';
    const [copied, setCopied] = useState(false);
    const [baseUrl, setBaseUrl] = useState(settings.get().apiBaseUrl);
    const [savedEndpoint, setSavedEndpoint] = useState(false);

    function handleCopy() {
        if (token) {
            navigator.clipboard.writeText(token);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    }

    function handleSaveEndpoint() {
        settings.save({ apiBaseUrl: baseUrl.trim() });
        setSavedEndpoint(true);
        setTimeout(() => setSavedEndpoint(false), 2000);
    }

    return (
        <div className="settings-section">
            <h2>API Settings</h2>
            <p className="settings-description">Manage your API credentials and service endpoints</p>

            <div className="settings-card">
                <div className="settings-card__header">
                    <Key size={16} />
                    <span>Active Session / API Token</span>
                </div>
                <div className="api-key-display">
                    <code>{token ? `${token.slice(0, 16)}••••••••` : 'Not authenticated — log in via Authenticator'}</code>
                    {token && (
                        <button className="icon-button" onClick={handleCopy} title="Copy token">
                            {copied ? <Check size={14} /> : <Copy size={14} />}
                        </button>
                    )}
                </div>
                <p className="settings-hint">Used for authenticating Studio requests to the Langtrain backend.</p>
            </div>

            <div className="settings-card">
                <div className="settings-card__header">
                    <Server size={16} />
                    <span>Backend API Base URL</span>
                </div>
                <div style={{ display: 'flex', gap: '8px', marginTop: '12px', alignItems: 'center' }}>
                    <input
                        type="text"
                        value={baseUrl}
                        onChange={(e) => setBaseUrl(e.target.value)}
                        placeholder="https://api.langtrain.xyz"
                        style={{
                            flex: 1,
                            padding: '8px 12px',
                            borderRadius: '6px',
                            background: 'var(--bg-secondary)',
                            border: '1px solid var(--border-color)',
                            color: 'var(--text-primary)',
                            fontFamily: 'monospace',
                            fontSize: '13px'
                        }}
                    />
                    <button className="button button--secondary" onClick={handleSaveEndpoint}>
                        {savedEndpoint ? 'Saved ✓' : 'Save'}
                    </button>
                </div>
                <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                    <button
                        className="button button--ghost"
                        style={{ fontSize: '11px', padding: '4px 8px' }}
                        onClick={() => {
                            setBaseUrl('https://api.langtrain.xyz');
                            settings.save({ apiBaseUrl: 'https://api.langtrain.xyz' });
                        }}
                    >
                        Production (Cloud)
                    </button>
                    <button
                        className="button button--ghost"
                        style={{ fontSize: '11px', padding: '4px 8px' }}
                        onClick={() => {
                            setBaseUrl('http://localhost:8000');
                            settings.save({ apiBaseUrl: 'http://localhost:8000' });
                        }}
                    >
                        Local Server (:8000)
                    </button>
                </div>
                <p className="settings-hint" style={{ marginTop: '10px' }}>
                    Resolved API target: <code>{settings.getApiUrl()}</code>
                </p>
            </div>
        </div>
    );
}

function NotificationSettings() {
    const [emailNotifs, setEmailNotifs] = useState(true);
    const [pushNotifs, setPushNotifs] = useState(false);
    const [jobComplete, setJobComplete] = useState(true);
    const [jobFailed, setJobFailed] = useState(true);

    return (
        <div className="settings-section">
            <h2>Notification Settings</h2>
            <p className="settings-description">Configure how you receive notifications</p>

            <div className="settings-list">
                <div className="settings-toggle-item">
                    <div className="settings-toggle-item__info">
                        <span className="settings-toggle-item__label">Email Notifications</span>
                        <span className="settings-toggle-item__description">Receive notifications via email</span>
                    </div>
                    <label className="toggle">
                        <input
                            type="checkbox"
                            checked={emailNotifs}
                            onChange={(e) => setEmailNotifs(e.target.checked)}
                        />
                        <span className="toggle__slider" />
                    </label>
                </div>

                <div className="settings-toggle-item">
                    <div className="settings-toggle-item__info">
                        <span className="settings-toggle-item__label">Push Notifications</span>
                        <span className="settings-toggle-item__description">Get desktop push notifications</span>
                    </div>
                    <label className="toggle">
                        <input
                            type="checkbox"
                            checked={pushNotifs}
                            onChange={(e) => setPushNotifs(e.target.checked)}
                        />
                        <span className="toggle__slider" />
                    </label>
                </div>

                <div className="settings-toggle-item">
                    <div className="settings-toggle-item__info">
                        <span className="settings-toggle-item__label">Job Completed</span>
                        <span className="settings-toggle-item__description">Notify when a training job completes</span>
                    </div>
                    <label className="toggle">
                        <input
                            type="checkbox"
                            checked={jobComplete}
                            onChange={(e) => setJobComplete(e.target.checked)}
                        />
                        <span className="toggle__slider" />
                    </label>
                </div>

                <div className="settings-toggle-item">
                    <div className="settings-toggle-item__info">
                        <span className="settings-toggle-item__label">Job Failed</span>
                        <span className="settings-toggle-item__description">Notify when a training job fails</span>
                    </div>
                    <label className="toggle">
                        <input
                            type="checkbox"
                            checked={jobFailed}
                            onChange={(e) => setJobFailed(e.target.checked)}
                        />
                        <span className="toggle__slider" />
                    </label>
                </div>
            </div>
        </div>
    );
}

function AppearanceSettings() {
    const [theme, setTheme] = useState<'dark' | 'light' | 'system'>('dark');

    return (
        <div className="settings-section">
            <h2>Appearance</h2>
            <p className="settings-description">Customize the look and feel</p>

            <div className="theme-selector">
                <button
                    className={`theme-option ${theme === 'dark' ? 'theme-option--active' : ''}`}
                    onClick={() => setTheme('dark')}
                >
                    <Moon size={24} />
                    <span>Dark</span>
                </button>
                <button
                    className={`theme-option ${theme === 'light' ? 'theme-option--active' : ''}`}
                    onClick={() => setTheme('light')}
                >
                    <Sun size={24} />
                    <span>Light</span>
                </button>
                <button
                    className={`theme-option ${theme === 'system' ? 'theme-option--active' : ''}`}
                    onClick={() => setTheme('system')}
                >
                    <Monitor size={24} />
                    <span>System</span>
                </button>
            </div>
        </div>
    );
}

function StorageSettings() {
    const usedStorage = 2.4;
    const totalStorage = 10;
    const percentage = (usedStorage / totalStorage) * 100;

    return (
        <div className="settings-section">
            <h2>Storage</h2>
            <p className="settings-description">Manage your storage usage</p>

            <div className="settings-card">
                <div className="storage-header">
                    <HardDrive size={20} />
                    <span>Storage Usage</span>
                </div>
                <div className="storage-bar">
                    <div className="storage-bar__fill" style={{ width: `${percentage}%` }} />
                </div>
                <div className="storage-info">
                    <span>{usedStorage} GB used of {totalStorage} GB</span>
                    <span>{(100 - percentage).toFixed(0)}% free</span>
                </div>

                <div className="storage-breakdown">
                    <div className="storage-item">
                        <span className="storage-item__color" style={{ backgroundColor: 'var(--accent-cyan)' }} />
                        <span className="storage-item__label">Models</span>
                        <span className="storage-item__value">1.8 GB</span>
                    </div>
                    <div className="storage-item">
                        <span className="storage-item__color" style={{ backgroundColor: 'var(--accent-purple)' }} />
                        <span className="storage-item__label">Datasets</span>
                        <span className="storage-item__value">0.5 GB</span>
                    </div>
                    <div className="storage-item">
                        <span className="storage-item__color" style={{ backgroundColor: 'var(--accent-orange)' }} />
                        <span className="storage-item__label">Checkpoints</span>
                        <span className="storage-item__value">0.1 GB</span>
                    </div>
                </div>
            </div>

            <button className="button button--secondary">Clear Cache</button>
        </div>
    );
}

function PrivacySettings() {
    const [analytics, setAnalytics] = useState(true);
    const [crashReports, setCrashReports] = useState(true);

    return (
        <div className="settings-section">
            <h2>Privacy & Security</h2>
            <p className="settings-description">Manage your privacy preferences</p>

            <div className="settings-list">
                <div className="settings-toggle-item">
                    <div className="settings-toggle-item__info">
                        <span className="settings-toggle-item__label">Usage Analytics</span>
                        <span className="settings-toggle-item__description">Help us improve by sending anonymous usage data</span>
                    </div>
                    <label className="toggle">
                        <input
                            type="checkbox"
                            checked={analytics}
                            onChange={(e) => setAnalytics(e.target.checked)}
                        />
                        <span className="toggle__slider" />
                    </label>
                </div>

                <div className="settings-toggle-item">
                    <div className="settings-toggle-item__info">
                        <span className="settings-toggle-item__label">Crash Reports</span>
                        <span className="settings-toggle-item__description">Automatically send crash reports</span>
                    </div>
                    <label className="toggle">
                        <input
                            type="checkbox"
                            checked={crashReports}
                            onChange={(e) => setCrashReports(e.target.checked)}
                        />
                        <span className="toggle__slider" />
                    </label>
                </div>
            </div>

            <div className="settings-card settings-card--danger">
                <div className="settings-card__header">
                    <Shield size={16} />
                    <span>Danger Zone</span>
                </div>
                <p className="settings-hint">These actions are irreversible. Please proceed with caution.</p>
                <div className="danger-actions">
                    <button className="button button--danger">Delete All Data</button>
                    <button className="button button--danger">Delete Account</button>
                </div>
            </div>
        </div>
    );
}

export function SettingsView() {
    const [activeTab, setActiveTab] = useState<SettingsTab>('profile');

    const tabs = [
        { id: 'profile' as SettingsTab, label: 'Profile', icon: <User size={16} /> },
        { id: 'api' as SettingsTab, label: 'API', icon: <Key size={16} /> },
        { id: 'notifications' as SettingsTab, label: 'Notifications', icon: <Bell size={16} /> },
        { id: 'appearance' as SettingsTab, label: 'Appearance', icon: <Palette size={16} /> },
        { id: 'storage' as SettingsTab, label: 'Storage', icon: <HardDrive size={16} /> },
        { id: 'privacy' as SettingsTab, label: 'Privacy', icon: <Shield size={16} /> },
    ];

    function renderContent() {
        switch (activeTab) {
            case 'profile': return <ProfileSettings />;
            case 'api': return <APISettings />;
            case 'notifications': return <NotificationSettings />;
            case 'appearance': return <AppearanceSettings />;
            case 'storage': return <StorageSettings />;
            case 'privacy': return <PrivacySettings />;
        }
    }

    return (
        <div className="settings-view">
            {/* Sidebar */}
            <div className="settings-view__sidebar">
                <div className="settings-view__header">
                    <Settings size={20} />
                    <h1>Settings</h1>
                </div>
                <nav className="settings-nav">
                    {tabs.map(tab => (
                        <TabButton
                            key={tab.id}
                            id={tab.id}
                            label={tab.label}
                            icon={tab.icon}
                            active={activeTab === tab.id}
                            onClick={setActiveTab}
                        />
                    ))}
                </nav>
            </div>

            {/* Content */}
            <div className="settings-view__content">
                {renderContent()}
            </div>
        </div>
    );
}
