import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from './components/Layout/Layout';
import Dashboard from './components/Dashboard/Dashboard';
import ChatInterface from './components/Chat/ChatInterface';
import EnhancedChatInterface from './components/Chat/EnhancedChatInterface';
import StrategiesPage from './components/Strategies/StrategiesPage';
import RecommendationsPage from './components/Recommendations/RecommendationsPage';
import SettingsPage from './components/Settings/SettingsPage';
import TestDashboard from './components/Testing/TestDashboard';
import { useUI } from './store';

// 创建QueryClient实例
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// 主应用组件
function AppContent() {
  const { currentView } = useUI();

  // 根据当前视图渲染对应组件
  const renderCurrentView = () => {
    switch (currentView) {
      case 'dashboard':
        return <Dashboard />;
      case 'chat':
        return <EnhancedChatInterface />;
      case 'strategies':
        return <StrategiesPage />;
      case 'recommendations':
        return <RecommendationsPage />;
      case 'testing':
        return <TestDashboard />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <Layout>
      {renderCurrentView()}
    </Layout>
  );
}

// 主App组件
export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}
