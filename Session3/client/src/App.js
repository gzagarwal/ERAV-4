import React, { useState } from 'react';
import { Toaster } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import PaperAnalysis from './components/PaperAnalysis';
import QuestionInterface from './components/QuestionInterface';
import CodeGenerator from './components/CodeGenerator';
import ParagraphAnalyzer from './components/ParagraphAnalyzer';
import TechnicalAnalysis from './components/TechnicalAnalysis';
import LoadingSpinner from './components/LoadingSpinner';

function App() {
  const [currentPaper, setCurrentPaper] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');

  const handlePaperUpload = (paperData) => {
    setCurrentPaper(paperData);
    setActiveTab('analysis');
  };

  const tabs = [
    { id: 'upload', label: 'Upload Paper', icon: '📄' },
    { id: 'analysis', label: 'Analysis', icon: '🔍', disabled: !currentPaper },
    { id: 'questions', label: 'Ask Questions', icon: '❓', disabled: !currentPaper },
    { id: 'code', label: 'Generate Code', icon: '💻', disabled: !currentPaper },
    { id: 'paragraph', label: 'Analyze Paragraph', icon: '📝', disabled: !currentPaper },
    { id: 'technical', label: 'Technical Deep Dive', icon: '🔬', disabled: !currentPaper },
  ];

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'upload':
        return <FileUpload onUpload={handlePaperUpload} isLoading={isLoading} setIsLoading={setIsLoading} />;
      case 'analysis':
        return <PaperAnalysis paper={currentPaper} />;
      case 'questions':
        return <QuestionInterface paper={currentPaper} />;
      case 'code':
        return <CodeGenerator paper={currentPaper} />;
      case 'paragraph':
        return <ParagraphAnalyzer paper={currentPaper} />;
      case 'technical':
        return <TechnicalAnalysis paper={currentPaper} />;
      default:
        return <FileUpload onUpload={handlePaperUpload} isLoading={isLoading} setIsLoading={setIsLoading} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: 3000,
            iconTheme: {
              primary: '#4ade80',
              secondary: '#fff',
            },
          },
          error: {
            duration: 5000,
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />
      
      <Header />
      
      <div className="container mx-auto px-4 py-8">
        {/* Tab Navigation */}
        <div className="mb-8">
          <div className="flex flex-wrap gap-2 justify-center">
            {tabs.map((tab) => (
              <motion.button
                key={tab.id}
                onClick={() => !tab.disabled && setActiveTab(tab.id)}
                disabled={tab.disabled}
                className={`
                  flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all duration-200
                  ${activeTab === tab.id 
                    ? 'bg-white text-indigo-600 shadow-lg border-2 border-indigo-200' 
                    : tab.disabled 
                      ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                      : 'bg-white/70 text-gray-600 hover:bg-white hover:text-indigo-600 hover:shadow-md'
                  }
                `}
                whileHover={!tab.disabled ? { scale: 1.05 } : {}}
                whileTap={!tab.disabled ? { scale: 0.95 } : {}}
              >
                <span className="text-lg">{tab.icon}</span>
                <span className="hidden sm:inline">{tab.label}</span>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="max-w-6xl mx-auto"
          >
            {isLoading && <LoadingSpinner />}
            {renderActiveTab()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

export default App;
