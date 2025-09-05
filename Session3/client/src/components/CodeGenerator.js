import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Code, Copy, Download, Play, Check, FileCode } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import axios from 'axios';
import toast from 'react-hot-toast';

const CodeGenerator = ({ paper }) => {
  const [algorithm, setAlgorithm] = useState('');
  const [language, setLanguage] = useState('python');
  const [generatedCode, setGeneratedCode] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const languages = [
    { value: 'python', label: 'Python', icon: '🐍' },
    { value: 'java', label: 'Java', icon: '☕' },
    { value: 'javascript', label: 'JavaScript', icon: '🟨' },
    { value: 'go', label: 'Go', icon: '🐹' },
    { value: 'cpp', label: 'C++', icon: '⚡' },
    { value: 'rust', label: 'Rust', icon: '🦀' },
    { value: 'typescript', label: 'TypeScript', icon: '🔷' },
    { value: 'csharp', label: 'C#', icon: '🔷' }
  ];

  const predefinedAlgorithms = [
    "Implement the main algorithm described in the paper",
    "Create a data preprocessing pipeline based on the methodology",
    "Implement the evaluation metrics used in the paper",
    "Build the neural network architecture described",
    "Create the optimization algorithm mentioned",
    "Implement the feature extraction method",
    "Build the classification/regression model",
    "Create the data visualization functions"
  ];

  const generateCode = async (algorithmText = algorithm) => {
    if (!algorithmText.trim()) {
      toast.error('Please describe the algorithm to implement');
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post('/api/generate-code', {
        algorithm: algorithmText,
        language: language,
        paper_text: paper.text
      });

      if (response.data.success) {
        setGeneratedCode(response.data.code);
        toast.success('Code generated successfully!');
      } else {
        throw new Error(response.data.message || 'Failed to generate code');
      }
    } catch (error) {
      console.error('Error generating code:', error);
      toast.error(error.response?.data?.detail || 'Failed to generate code');
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast.success('Code copied to clipboard!');
    } catch (err) {
      toast.error('Failed to copy to clipboard');
    }
  };

  const downloadCode = () => {
    const extension = language === 'javascript' ? 'js' : 
                     language === 'typescript' ? 'ts' :
                     language === 'cpp' ? 'cpp' :
                     language === 'csharp' ? 'cs' : language;
    
    const blob = new Blob([generatedCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `algorithm_implementation.${extension}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Code downloaded successfully!');
  };

  if (!paper) {
    return (
      <div className="text-center py-12">
        <Code className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-gray-600">No paper uploaded</h3>
        <p className="text-gray-500">Please upload a research paper to generate code</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="max-w-6xl mx-auto space-y-8"
    >
      {/* Header */}
      <div className="text-center">
        <motion.h2 
          className="text-3xl font-bold text-gray-800 mb-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Generate Code Implementation
        </motion.h2>
        <motion.p 
          className="text-gray-600 text-lg"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          Generate code implementations based on algorithms and methods from your research paper
        </motion.p>
      </div>

      {/* Predefined Algorithms */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-white rounded-2xl shadow-lg p-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 mb-4">Quick Algorithm Templates</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {predefinedAlgorithms.map((alg, index) => (
            <motion.button
              key={index}
              onClick={() => generateCode(alg)}
              disabled={isLoading}
              className="text-left p-4 bg-gray-50 hover:bg-indigo-50 border border-gray-200 hover:border-indigo-300 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <p className="text-sm text-gray-700">{alg}</p>
            </motion.button>
          ))}
        </div>
      </motion.div>

      {/* Code Generation Form */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-white rounded-2xl shadow-lg p-6"
      >
        <h3 className="text-xl font-semibold text-gray-800 mb-4">Custom Code Generation</h3>
        <div className="space-y-6">
          {/* Language Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Programming Language
            </label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {languages.map((lang) => (
                <motion.button
                  key={lang.value}
                  onClick={() => setLanguage(lang.value)}
                  className={`
                    flex items-center gap-2 p-3 rounded-lg border-2 transition-all duration-200
                    ${language === lang.value 
                      ? 'border-indigo-500 bg-indigo-50 text-indigo-700' 
                      : 'border-gray-200 bg-white hover:border-gray-300 text-gray-700'
                    }
                  `}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span className="text-lg">{lang.icon}</span>
                  <span className="text-sm font-medium">{lang.label}</span>
                </motion.button>
              ))}
            </div>
          </div>

          {/* Algorithm Description */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Algorithm Description
            </label>
            <textarea
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
              placeholder="Describe the algorithm or method you want to implement based on the research paper..."
              className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
              rows={4}
              disabled={isLoading}
            />
          </div>

          {/* Generate Button */}
          <div className="flex justify-end">
            <motion.button
              onClick={() => generateCode()}
              disabled={isLoading || !algorithm.trim()}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg font-medium hover:from-indigo-600 hover:to-purple-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              ) : (
                <Play className="w-5 h-5" />
              )}
              <span>{isLoading ? 'Generating...' : 'Generate Code'}</span>
            </motion.button>
          </div>
        </div>
      </motion.div>

      {/* Generated Code */}
      {generatedCode && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-white rounded-2xl shadow-lg p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-semibold text-gray-800 flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-blue-600 rounded-lg flex items-center justify-center">
                <FileCode className="w-5 h-5 text-white" />
              </div>
              Generated Code ({languages.find(l => l.value === language)?.label})
            </h3>
            <div className="flex gap-2">
              <motion.button
                onClick={() => copyToClipboard(generatedCode)}
                className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {copied ? <Check className="w-4 h-4 text-green-600" /> : <Copy className="w-4 h-4" />}
                <span className="text-sm font-medium">Copy</span>
              </motion.button>
              <motion.button
                onClick={downloadCode}
                className="flex items-center gap-2 px-4 py-2 bg-indigo-100 hover:bg-indigo-200 text-indigo-700 rounded-lg transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Download className="w-4 h-4" />
                <span className="text-sm font-medium">Download</span>
              </motion.button>
            </div>
          </div>
          
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <SyntaxHighlighter
              language={language}
              style={vscDarkPlus}
              customStyle={{
                margin: 0,
                borderRadius: '0.5rem',
                fontSize: '0.875rem',
                lineHeight: '1.5'
              }}
              showLineNumbers={true}
              wrapLines={true}
            >
              {generatedCode}
            </SyntaxHighlighter>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default CodeGenerator;
