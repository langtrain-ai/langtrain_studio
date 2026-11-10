/**
 * Langtrain Studio Desktop
 * Main Application Entry Point
 */

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { MainLayout } from './components/layout';
import {
  LoginView,
  DashboardView,
  ProjectsView,
  DatasetsView,
  ModelsView,
  ServingView,
  TrainingView,
  AnalyticsView,
  SettingsView,
  AgentsView,
  CurationView,
  LocalModelsView,
  CoWorkView,
  StudioView,
} from './components/views';
import { useAuth } from './services/auth';
import './styles/theme.css';

// Protected Route wrapper
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner" />
        <span>Loading...</span>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

// App Routes
function AppRoutes() {
  const { isAuthenticated } = useAuth();

  return (
    <Routes>
      {/* Public route */}
      <Route
        path="/login"
        element={isAuthenticated ? <Navigate to="/" replace /> : <LoginView />}
      />

      {/* Protected routes */}
      <Route
        element={
          <ProtectedRoute>
            <MainLayout />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<DashboardView />} />
        <Route path="/agents" element={<AgentsView />} />
        <Route path="/projects" element={<ProjectsView />} />
        <Route path="/datasets" element={<DatasetsView />} />
        <Route path="/curation" element={<CurationView />} />
        <Route path="/models" element={<ModelsView />} />
        <Route path="/serving" element={<ServingView />} />
        <Route path="/training" element={<TrainingView />} />
        <Route path="/analytics" element={<AnalyticsView />} />
        <Route path="/settings" element={<SettingsView />} />
        <Route path="/local-models" element={<LocalModelsView />} />
        <Route path="/studio" element={<StudioView />} />
        <Route path="/co-work" element={<CoWorkView />} />
      </Route>

      {/* Catch-all redirect */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}

export default App;
