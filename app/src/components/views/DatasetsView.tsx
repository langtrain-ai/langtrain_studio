/**
 * Datasets View Component
 * Displays and manages training datasets with real file upload
 */

import { useState, useEffect, useRef } from 'react';
import {
    Database,
    Upload,
    Trash2,
    FileText,
    HardDrive,
    Calendar,
    MoreVertical,
    Search,
    Filter,
    Download,
    Eye,
    Loader,
    RefreshCw
} from 'lucide-react';
import { api, DatasetInfo } from '../../services/api';
import { useToast } from '../common';
import './DatasetsView.css';

function formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatDate(dateString: string): string {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    });
}

function getPurposeLabel(purpose: string): string {
    return purpose.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
}

interface DatasetRowProps {
    dataset: DatasetInfo;
    onDelete: (id: string) => void;
    isDeleting?: boolean;
}

function DatasetRow({ dataset, onDelete, isDeleting }: DatasetRowProps) {
    const [showMenu, setShowMenu] = useState(false);

    return (
        <div className={`dataset-row ${isDeleting ? 'dataset-row--deleting' : ''}`}>
            <div className="dataset-row__icon">
                <FileText size={18} />
            </div>
            <div className="dataset-row__info">
                <span className="dataset-row__name">{dataset.filename}</span>
                <span className="dataset-row__id">{dataset.id.slice(0, 12)}</span>
            </div>
            <div className="dataset-row__purpose">
                <span className="purpose-badge">{getPurposeLabel(dataset.purpose)}</span>
            </div>
            <div className="dataset-row__size">
                <HardDrive size={12} />
                <span>{formatBytes(dataset.bytes)}</span>
            </div>
            <div className="dataset-row__date">
                <Calendar size={12} />
                <span>{formatDate(dataset.createdAt)}</span>
            </div>
            <div className="dataset-row__actions">
                <button
                    className="icon-button"
                    onClick={() => setShowMenu(!showMenu)}
                    disabled={isDeleting}
                >
                    {isDeleting ? <Loader size={16} className="spin" /> : <MoreVertical size={16} />}
                </button>
                {showMenu && (
                    <div className="dataset-row__dropdown">
                        <button>
                            <Eye size={14} />
                            Preview
                        </button>
                        <button>
                            <Download size={14} />
                            Download
                        </button>
                        <button className="danger" onClick={() => { onDelete(dataset.id); setShowMenu(false); }}>
                            <Trash2 size={14} />
                            Delete
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}

export function DatasetsView() {
    const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [deletingId, setDeletingId] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const toast = useToast();

    useEffect(() => {
        loadDatasets();
    }, []);

    async function loadDatasets() {
        try {
            setIsLoading(true);
            const response = await api.listDatasets();
            setDatasets(response.data);
        } catch (err) {
            console.error('Failed to load datasets:', err instanceof Error ? err.message : 'Unknown error');
            setDatasets([]);
            toast.error('Failed to load datasets. Please try again.');
        } finally {
            setIsLoading(false);
        }
    }

    async function handleDelete(id: string) {
        try {
            setDeletingId(id);
            await api.deleteDataset(id);
            setDatasets(prev => prev.filter(d => d.id !== id));
            toast.success('Dataset deleted successfully');
        } catch (err) {
            console.error('Failed to delete dataset:', err);
            toast.error('Failed to delete dataset');
        } finally {
            setDeletingId(null);
        }
    }

    async function handleUpload(files: File[]) {
        if (files.length === 0) return;

        const validExtensions = ['.jsonl', '.csv', '.parquet', '.json'];
        const validFiles = files.filter(f =>
            validExtensions.some(ext => f.name.toLowerCase().endsWith(ext))
        );

        if (validFiles.length === 0) {
            toast.error('Please upload JSONL, CSV, or Parquet files');
            return;
        }

        setIsUploading(true);
        setUploadProgress(0);

        for (let i = 0; i < validFiles.length; i++) {
            const file = validFiles[i];
            try {
                setUploadProgress(((i) / validFiles.length) * 100);
                const newDataset = await api.uploadDataset(file);
                setDatasets(prev => [newDataset, ...prev]);
                toast.success(`Uploaded ${file.name}`);
            } catch (err) {
                console.error(`Failed to upload ${file.name}:`, err);
                toast.error(`Failed to upload ${file.name}`);
            }
        }

        setUploadProgress(100);
        setIsUploading(false);
    }

    function handleDragOver(e: React.DragEvent) {
        e.preventDefault();
        setIsDragging(true);
    }

    function handleDragLeave() {
        setIsDragging(false);
    }

    function handleDrop(e: React.DragEvent) {
        e.preventDefault();
        setIsDragging(false);
        const files = Array.from(e.dataTransfer.files);
        handleUpload(files);
    }

    function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
        if (e.target.files) {
            const files = Array.from(e.target.files);
            handleUpload(files);
        }
    }

    const filteredDatasets = datasets.filter(d =>
        d.filename.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const totalSize = datasets.reduce((acc, d) => acc + d.bytes, 0);

    return (
        <div className="datasets-view">
            {/* Hidden file input */}
            <input
                ref={fileInputRef}
                type="file"
                accept=".jsonl,.csv,.parquet,.json"
                multiple
                style={{ display: 'none' }}
                onChange={handleFileSelect}
            />

            {/* Header */}
            <div className="datasets-view__header">
                <div className="datasets-view__title-section">
                    <h1 className="datasets-view__title">Datasets</h1>
                    <p className="datasets-view__subtitle">Upload and manage training datasets</p>
                </div>
                <div className="datasets-view__header-actions">
                    <button className="icon-button" onClick={loadDatasets} title="Refresh">
                        <RefreshCw size={16} />
                    </button>
                    <button
                        className="button button--primary"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isUploading}
                    >
                        <Upload size={16} />
                        Upload Dataset
                    </button>
                </div>
            </div>

            {/* Stats */}
            <div className="datasets-view__stats">
                <div className="stat-card">
                    <Database size={20} className="stat-card__icon" />
                    <div className="stat-card__content">
                        <span className="stat-card__value">{datasets.length}</span>
                        <span className="stat-card__label">Datasets</span>
                    </div>
                </div>
                <div className="stat-card">
                    <HardDrive size={20} className="stat-card__icon" />
                    <div className="stat-card__content">
                        <span className="stat-card__value">{formatBytes(totalSize)}</span>
                        <span className="stat-card__label">Total Size</span>
                    </div>
                </div>
            </div>

            {/* Upload Zone */}
            <div
                className={`datasets-view__upload-zone ${isDragging ? 'datasets-view__upload-zone--active' : ''} ${isUploading ? 'datasets-view__upload-zone--uploading' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !isUploading && fileInputRef.current?.click()}
            >
                {isUploading ? (
                    <>
                        <Loader size={32} className="upload-icon spin" />
                        <span className="upload-text">Uploading...</span>
                        <div className="upload-progress">
                            <div className="upload-progress__bar" style={{ width: `${uploadProgress}%` }} />
                        </div>
                    </>
                ) : (
                    <>
                        <Upload size={32} className="upload-icon" />
                        <span className="upload-text">Drag and drop files here, or click to browse</span>
                        <span className="upload-hint">Supports JSONL, CSV, and Parquet files</span>
                    </>
                )}
            </div>

            {/* Search & Filter */}
            <div className="datasets-view__toolbar">
                <div className="search-box">
                    <Search size={16} />
                    <input
                        type="text"
                        placeholder="Search datasets..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                    />
                </div>
                <button className="icon-button">
                    <Filter size={16} />
                </button>
            </div>

            {/* Dataset List */}
            <div className="datasets-view__content">
                {isLoading ? (
                    <div className="datasets-view__loading">
                        <Loader size={24} className="spin" />
                        <span>Loading datasets...</span>
                    </div>
                ) : filteredDatasets.length === 0 ? (
                    <div className="datasets-view__empty">
                        <Database size={48} className="empty-icon" />
                        <h3>No datasets found</h3>
                        <p>Upload your first dataset to get started</p>
                    </div>
                ) : (
                    <div className="datasets-view__table">
                        <div className="datasets-view__table-header">
                            <span>Name</span>
                            <span>Purpose</span>
                            <span>Size</span>
                            <span>Created</span>
                            <span></span>
                        </div>
                        {filteredDatasets.map(dataset => (
                            <DatasetRow
                                key={dataset.id}
                                dataset={dataset}
                                onDelete={handleDelete}
                                isDeleting={deletingId === dataset.id}
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
