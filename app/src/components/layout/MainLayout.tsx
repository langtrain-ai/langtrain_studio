/**
 * Main Layout Component
 * Container with sidebar and content area
 */

import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import './MainLayout.css';

export function MainLayout() {
    return (
        <div className="main-layout">
            <Sidebar />
            <main className="main-layout__content">
                <Outlet />
            </main>
        </div>
    );
}
