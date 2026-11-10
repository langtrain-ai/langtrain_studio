/**
 * Sidebar Component
 * Ported from Swift SidebarView.swift
 */

import { NavLink } from 'react-router-dom';
import {
    Home,
    Layers,
    Database,
    Cpu,
    Terminal,
    LineChart,
    Settings,
    Users,
    BookOpen,
    LogOut,
    Bot,
    Sparkles,
    Server,
    HardDrive,
    BrainCircuit,
    type LucideIcon,
} from 'lucide-react';
import { sidebarItems, systemItems, offlineItems } from '../../lib/tokens';
import { useAuth } from '../../services/auth';
import './Sidebar.css';

// Icon mapping
const iconMap: Record<string, LucideIcon> = {
    Home,
    Layers,
    Database,
    Cpu,
    Terminal,
    LineChart,
    Settings,
    BookOpen,
    Bot,
    Sparkles,
    Server,
    HardDrive,
    BrainCircuit,
};

interface SidebarItemProps {
    id: string;
    title: string;
    icon: string;
    path: string;
}

function SidebarItem({ title, icon, path }: SidebarItemProps) {
    const Icon = iconMap[icon] || Home;

    return (
        <NavLink
            to={path}
            className={({ isActive }) =>
                `sidebar-item ${isActive ? 'sidebar-item--active' : ''}`
            }
        >
            <Icon size={16} className="sidebar-item__icon" />
            <span className="sidebar-item__title">{title}</span>
        </NavLink>
    );
}

export function Sidebar() {
    const { user, logout } = useAuth();

    return (
        <aside className="sidebar">
            {/* Header */}
            <div className="sidebar__header">
                <div className="sidebar__logo">
                    <img
                        src="/langtrain-logo.svg"
                        alt="Langtrain"
                        className="sidebar__logo-image"
                        onError={(e) => {
                            // Fallback to text if logo doesn't exist
                            e.currentTarget.style.display = 'none';
                        }}
                    />
                    <div className="sidebar__logo-text">
                        <span className="sidebar__logo-title">Langtrain</span>
                        <span className="sidebar__logo-subtitle">STUDIO</span>
                    </div>
                </div>
            </div>

            {/* Navigation */}
            <nav className="sidebar__nav">
                <div className="sidebar__section">
                    {sidebarItems.map((item) => (
                        <SidebarItem key={item.id} {...item} />
                    ))}
                </div>

                <div className="sidebar__divider" />

                <div className="sidebar__section">
                    <span className="sidebar__section-title">SYSTEM</span>
                    {systemItems.map((item) => (
                        <SidebarItem key={item.id} {...item} />
                    ))}
                </div>

                <div className="sidebar__divider" />

                <div className="sidebar__section">
                    <span className="sidebar__section-title">OFFLINE</span>
                    {offlineItems.map((item) => (
                        <SidebarItem key={item.id} {...item} />
                    ))}
                </div>
            </nav>

            {/* Footer - User Info & Logout */}
            <div className="sidebar__footer">
                <div className="sidebar__user">
                    <div className="sidebar__user-avatar">
                        <Users size={14} />
                    </div>
                    <div className="sidebar__user-info">
                        <span className="sidebar__user-name">Pro Workspace</span>
                        <span className="sidebar__user-email">
                            {user?.email || 'Guest'}
                        </span>
                    </div>
                </div>

                <button
                    className="sidebar-logout"
                    onClick={logout}
                    title="Sign Out"
                >
                    <LogOut size={16} />
                </button>
            </div>
        </aside>
    );
}
