/**
 * Projects View Component
 * Displays and manages fine-tuning projects
 */

import { useState, useEffect } from 'react';
import {
    FolderOpen,
    Plus,
    MoreVertical,
    Pause,
    Trash2,
    Clock,
    CheckCircle,
    AlertCircle,
    Loader
} from 'lucide-react';
import { api, FineTuneJob } from '../../services/api';
import './ProjectsView.css';

type ProjectStatus = 'running' | 'completed' | 'failed' | 'pending' | 'cancelled';

interface ProjectCardProps {
    project: FineTuneJob;
    onCancel: (id: string) => void;
}

function getStatusIcon(status: string) {
    switch (status.toLowerCase()) {
        case 'running':
        case 'in_progress':
            return <Loader size={14} className="status-icon status-icon--running" />;
        case 'completed':
        case 'succeeded':
            return <CheckCircle size={14} className="status-icon status-icon--completed" />;
        case 'failed':
        case 'error':
            return <AlertCircle size={14} className="status-icon status-icon--failed" />;
        case 'pending':
        case 'queued':
            return <Clock size={14} className="status-icon status-icon--pending" />;
        default:
            return <Clock size={14} className="status-icon status-icon--pending" />;
    }
}

function getStatusLabel(status: string): string {
    switch (status.toLowerCase()) {
        case 'in_progress': return 'Running';
        case 'succeeded': return 'Completed';
        case 'error': return 'Failed';
        case 'queued': return 'Pending';
        default: return status.charAt(0).toUpperCase() + status.slice(1);
    }
}

function formatDate(dateString?: string): string {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function ProjectCard({ project, onCancel }: ProjectCardProps) {
    const [showMenu, setShowMenu] = useState(false);
    const isRunning = ['running', 'in_progress', 'pending', 'queued'].includes(project.status.toLowerCase());

    return (
        <div className="project-card">
            <div className="project-card__header">
                <div className="project-card__icon">
                    <FolderOpen size={20} />
                </div>
                <div className="project-card__title-section">
                    <h3 className="project-card__title">{project.name || `Job ${project.id.slice(0, 8)}`}</h3>
                    <span className="project-card__id">ID: {project.id.slice(0, 12)}</span>
                </div>
                <div className="project-card__menu">
                    <button
                        className="icon-button"
                        onClick={() => setShowMenu(!showMenu)}
                    >
                        <MoreVertical size={16} />
                    </button>
                    {showMenu && (
                        <div className="project-card__dropdown">
                            {isRunning && (
                                <button onClick={() => { onCancel(project.id); setShowMenu(false); }}>
                                    <Pause size={14} />
                                    Cancel Job
                                </button>
                            )}
                            <button className="danger">
                                <Trash2 size={14} />
                                Delete
                            </button>
                        </div>
                    )}
                </div>
            </div>

            <div className="project-card__body">
                <div className="project-card__status">
                    {getStatusIcon(project.status)}
                    <span>{getStatusLabel(project.status)}</span>
                </div>

                {isRunning && project.progress > 0 && (
                    <div className="project-card__progress">
                        <div className="progress-bar">
                            <div
                                className="progress-bar__fill"
                                style={{ width: `${project.progress}%` }}
                            />
                        </div>
                        <span className="progress-label">{project.progress}%</span>
                    </div>
                )}

                <div className="project-card__meta">
                    <div className="meta-item">
                        <span className="meta-label">Created</span>
                        <span className="meta-value">{formatDate(project.createdAt)}</span>
                    </div>
                    {project.completedAt && (
                        <div className="meta-item">
                            <span className="meta-label">Completed</span>
                            <span className="meta-value">{formatDate(project.completedAt)}</span>
                        </div>
                    )}
                </div>

                {project.errorMessage && (
                    <div className="project-card__error">
                        <AlertCircle size={12} />
                        <span>{project.errorMessage}</span>
                    </div>
                )}
            </div>
        </div>
    );
}

export function ProjectsView() {
    const [projects, setProjects] = useState<FineTuneJob[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [filter, setFilter] = useState<'all' | ProjectStatus>('all');

    useEffect(() => {
        loadProjects();
    }, []);

    async function loadProjects() {
        try {
            setIsLoading(true);
            setError(null);
            const response = await api.listFineTuningJobs();
            setProjects(response.data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load projects');
        } finally {
            setIsLoading(false);
        }
    }

    async function handleCancelJob(id: string) {
        try {
            await api.cancelFineTuningJob(id);
            loadProjects();
        } catch (err) {
            console.error('Failed to cancel job:', err);
        }
    }

    const filteredProjects = projects.filter(p => {
        if (filter === 'all') return true;
        return p.status.toLowerCase().includes(filter);
    });

    const stats = {
        total: projects.length,
        running: projects.filter(p => ['running', 'in_progress'].includes(p.status.toLowerCase())).length,
        completed: projects.filter(p => ['completed', 'succeeded'].includes(p.status.toLowerCase())).length,
        failed: projects.filter(p => ['failed', 'error'].includes(p.status.toLowerCase())).length,
    };

    return (
        <div className="projects-view">
            {/* Header */}
            <div className="projects-view__header">
                <div className="projects-view__title-section">
                    <h1 className="projects-view__title">Projects</h1>
                    <p className="projects-view__subtitle">Manage your fine-tuning projects</p>
                </div>
                <button className="button button--primary">
                    <Plus size={16} />
                    New Project
                </button>
            </div>

            {/* Stats */}
            <div className="projects-view__stats">
                <div className="stat-card" onClick={() => setFilter('all')}>
                    <span className="stat-card__value">{stats.total}</span>
                    <span className="stat-card__label">Total</span>
                </div>
                <div className="stat-card stat-card--running" onClick={() => setFilter('running')}>
                    <span className="stat-card__value">{stats.running}</span>
                    <span className="stat-card__label">Running</span>
                </div>
                <div className="stat-card stat-card--completed" onClick={() => setFilter('completed')}>
                    <span className="stat-card__value">{stats.completed}</span>
                    <span className="stat-card__label">Completed</span>
                </div>
                <div className="stat-card stat-card--failed" onClick={() => setFilter('failed')}>
                    <span className="stat-card__value">{stats.failed}</span>
                    <span className="stat-card__label">Failed</span>
                </div>
            </div>

            {/* Content */}
            <div className="projects-view__content">
                {isLoading ? (
                    <div className="projects-view__loading">
                        <Loader size={32} className="spin" />
                        <span>Loading projects...</span>
                    </div>
                ) : error ? (
                    <div className="projects-view__error">
                        <AlertCircle size={32} />
                        <span>{error}</span>
                        <button className="button button--secondary" onClick={loadProjects}>
                            Retry
                        </button>
                    </div>
                ) : filteredProjects.length === 0 ? (
                    <div className="projects-view__empty">
                        <FolderOpen size={48} className="empty-icon" />
                        <h3>No projects found</h3>
                        <p>Create your first fine-tuning project to get started</p>
                        <button className="button button--primary">
                            <Plus size={16} />
                            Create Project
                        </button>
                    </div>
                ) : (
                    <div className="projects-view__grid">
                        {filteredProjects.map(project => (
                            <ProjectCard
                                key={project.id}
                                project={project}
                                onCancel={handleCancelJob}
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
