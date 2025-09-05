import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Search, Copy, Check, Target } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import toast from 'react-hot-toast';

const ParagraphAnalyzer = ({ paper }) => {
  const [selectedParagraph, setSelectedParagraph] = useState('');
  const [analysis, setAnalysis] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const analyzeParagraph = async () => {
    if (!selectedParagraph.trim()) {
      toast.error('Please select or enter a paragraph to analyze');
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post('/api/analyze-paragraph', {
        paragraph: selectedParagraph,
        paper_text: paper.text
      });

      if (response.data.success) {
        setAnalysis(response.data.analysis);
        toast.success('Paragraph analyzed successfully!');
      } else {
        throw new Error(response.data.message || 'Failed to analyze paragraph');
      }
    } catch (error) {
      console.error('Error analyzing paragraph:', error);
      toast.error(error.response?.data?.detail || 'Failed to analyze paragraph');
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast.success('Analysis copied to clipboard!');
    } catch (err) {
      toast.error('Failed to copy to clipboard');
    }
  };

  const extractParagraphs = (text) => {
    // Split text into paragraphs (double line breaks)
    const paragraphs = text.split('\n\n').filter(p => p.trim().length > 50);
    return paragraphs.slice(0, 20); // Limit to first 20 paragraphs
  };

  const selectParagraph = (paragraph) => {
    setSelectedParagraph(paragraph);
    setAnalysis(''); // Clear previous analysis
  };

  if (!paper) {
    return (
      <div className="text-center py-12">
        <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-gray-600">No paper uploaded</h3>
        <p className="text-gray-500">Please upload a research paper to analyze paragraphs</p>
      </div>
    );
  }

  const paragraphs = extractParagraphs(paper.text);

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
          Analyze Specific Paragraphs
        </motion.h2>
        <motion.p 
          className="text-gray-600 text-lg"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          Select any paragraph from your paper for detailed AI analysis and explanation
        </motion.p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Paragraph Selection */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
          className="space-y-6"
        >
          {/* Manual Input */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
                <FileText className="w-5 h-5 text-white" />
              </div>
              Enter or Paste Paragraph
            </h3>
            <div className="space-y-4">
              <textarea
                value={selectedParagraph}
                onChange={(e) => setSelectedParagraph(e.target.value)}
                placeholder="Paste or type the paragraph you want to analyze..."
                className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
                rows={6}
                disabled={isLoading}
              />
              <motion.button
                onClick={analyzeParagraph}
                disabled={isLoading || !selectedParagraph.trim()}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg font-medium hover:from-indigo-600 hover:to-purple-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {isLoading ? (
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <Search className="w-5 h-5" />
                )}
                <span>{isLoading ? 'Analyzing...' : 'Analyze Paragraph'}</span>
              </motion.button>
            </div>
          </div>

          {/* Quick Selection */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-blue-600 rounded-lg flex items-center justify-center">
                <Target className="w-5 h-5 text-white" />
              </div>
              Quick Select from Paper
            </h3>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {paragraphs.map((paragraph, index) => (
                <motion.button
                  key={index}
                  onClick={() => selectParagraph(paragraph)}
                  className="w-full text-left p-4 bg-gray-50 hover:bg-indigo-50 border border-gray-200 hover:border-indigo-300 rounded-lg transition-all duration-200"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                      <span className="text-xs font-medium text-indigo-600">{index + 1}</span>
                    </div>
                    <div>
                      <p className="text-sm text-gray-700 line-clamp-3">
                        {paragraph.substring(0, 200)}...
                      </p>
                      <p className="text-xs text-gray-500 mt-2">
                        {paragraph.length} characters
                      </p>
                    </div>
                  </div>
                </motion.button>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Analysis Results */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
          className="space-y-6"
        >
          {analysis && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-gray-800 flex items-center gap-3">
                  <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-blue-600 rounded-lg flex items-center justify-center">
                    <Search className="w-5 h-5 text-white" />
                  </div>
                  AI Analysis
                </h3>
                <motion.button
                  onClick={() => copyToClipboard(analysis)}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {copied ? <Check className="w-4 h-4 text-green-600" /> : <Copy className="w-4 h-4" />}
                  <span className="text-sm font-medium">Copy</span>
                </motion.button>
              </div>
              
              <div className="prose prose-lg max-w-none">
                <ReactMarkdown className="markdown-content">
                  {analysis}
                </ReactMarkdown>
              </div>
            </div>
          )}

          {/* Selected Paragraph Preview */}
          {selectedParagraph && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-3">
                <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-600 rounded-lg flex items-center justify-center">
                  <FileText className="w-5 h-5 text-white" />
                </div>
                Selected Paragraph
              </h3>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-sm text-gray-700 leading-relaxed">
                  {selectedParagraph}
                </p>
                <p className="text-xs text-gray-500 mt-3">
                  {selectedParagraph.length} characters • {selectedParagraph.split(' ').length} words
                </p>
              </div>
            </div>
          )}

          {/* Instructions */}
          {!analysis && !selectedParagraph && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">How to Use</h3>
              <div className="space-y-4 text-gray-600">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                    <span className="text-xs font-medium text-indigo-600">1</span>
                  </div>
                  <p>Select a paragraph from the paper using the quick selection buttons, or paste your own text</p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                    <span className="text-xs font-medium text-indigo-600">2</span>
                  </div>
                  <p>Click "Analyze Paragraph" to get AI-powered insights and explanations</p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                    <span className="text-xs font-medium text-indigo-600">3</span>
                  </div>
                  <p>Get detailed explanations of concepts, terms, and their significance in simple language</p>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </motion.div>
  );
};

export default ParagraphAnalyzer;
