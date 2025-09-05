import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Microscope, Copy, Check, Download, Brain, Calculator, FlaskConical, TrendingUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import toast from 'react-hot-toast';

const TechnicalAnalysis = ({ paper }) => {
  const [technicalAnalysis, setTechnicalAnalysis] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const generateTechnicalAnalysis = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post('/api/technical-analysis', {
        paper_text: paper.text
      });

      if (response.data.success) {
        setTechnicalAnalysis(response.data.technical_analysis);
        toast.success('Technical analysis generated successfully!');
      } else {
        throw new Error(response.data.message || 'Failed to generate technical analysis');
      }
    } catch (error) {
      console.error('Error generating technical analysis:', error);
      toast.error(error.response?.data?.detail || 'Failed to generate technical analysis');
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast.success('Technical analysis copied to clipboard!');
    } catch (err) {
      toast.error('Failed to copy to clipboard');
    }
  };

  const downloadAnalysis = () => {
    const blob = new Blob([technicalAnalysis], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${paper.filename}_technical_analysis.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Technical analysis downloaded successfully!');
  };

  if (!paper) {
    return (
      <div className="text-center py-12">
        <Microscope className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-gray-600">No paper uploaded</h3>
        <p className="text-gray-500">Please upload a research paper to get technical analysis</p>
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
          Technical Deep Dive Analysis
        </motion.h2>
        <motion.p 
          className="text-gray-600 text-lg"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          Get comprehensive technical analysis including methodology, mathematical foundations, and implementation insights
        </motion.p>
      </div>

      {/* Analysis Overview Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        <div className="bg-white rounded-xl shadow-sm p-6 text-center">
          <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
            <Brain className="w-6 h-6 text-blue-600" />
          </div>
          <h4 className="font-semibold text-gray-800 mb-2">Methodology</h4>
          <p className="text-sm text-gray-600">Research methods and techniques analysis</p>
        </div>
        
        <div className="bg-white rounded-xl shadow-sm p-6 text-center">
          <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
            <Calculator className="w-6 h-6 text-green-600" />
          </div>
          <h4 className="font-semibold text-gray-800 mb-2">Math Foundations</h4>
          <p className="text-sm text-gray-600">Mathematical concepts breakdown</p>
        </div>
        
        <div className="bg-white rounded-xl shadow-sm p-6 text-center">
          <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
            <FlaskConical className="w-6 h-6 text-purple-600" />
          </div>
          <h4 className="font-semibold text-gray-800 mb-2">Experimental Design</h4>
          <p className="text-sm text-gray-600">Setup and validation methods</p>
        </div>
        
        <div className="bg-white rounded-xl shadow-sm p-6 text-center">
          <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center mx-auto mb-4">
            <TrendingUp className="w-6 h-6 text-orange-600" />
          </div>
          <h4 className="font-semibold text-gray-800 mb-2">Results & Insights</h4>
          <p className="text-sm text-gray-600">Statistical significance analysis</p>
        </div>
      </motion.div>

      {/* Generate Analysis Button */}
      {!technicalAnalysis && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="text-center"
        >
          <motion.button
            onClick={generateTechnicalAnalysis}
            disabled={isLoading}
            className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-xl font-semibold hover:from-indigo-600 hover:to-purple-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed mx-auto"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isLoading ? (
              <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <Microscope className="w-6 h-6" />
            )}
            <span>{isLoading ? 'Generating Technical Analysis...' : 'Generate Technical Analysis'}</span>
          </motion.button>
          <p className="text-gray-500 mt-4 text-sm">
            This may take a few moments as we analyze the paper in detail
          </p>
        </motion.div>
      )}

      {/* Technical Analysis Results */}
      {technicalAnalysis && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-white rounded-2xl shadow-lg p-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-2xl font-bold text-gray-800 flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center">
                <Microscope className="w-6 h-6 text-white" />
              </div>
              Technical Analysis Report
            </h3>
            <div className="flex gap-2">
              <motion.button
                onClick={() => copyToClipboard(technicalAnalysis)}
                className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {copied ? <Check className="w-4 h-4 text-green-600" /> : <Copy className="w-4 h-4" />}
                <span className="text-sm font-medium">Copy</span>
              </motion.button>
              <motion.button
                onClick={downloadAnalysis}
                className="flex items-center gap-2 px-4 py-2 bg-indigo-100 hover:bg-indigo-200 text-indigo-700 rounded-lg transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Download className="w-4 h-4" />
                <span className="text-sm font-medium">Download</span>
              </motion.button>
            </div>
          </div>
          
          <div className="prose prose-lg max-w-none">
            <ReactMarkdown className="markdown-content">
              {technicalAnalysis}
            </ReactMarkdown>
          </div>
        </motion.div>
      )}

      {/* What's Included Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="bg-white rounded-2xl shadow-lg p-8"
      >
        <h3 className="text-2xl font-bold text-gray-800 mb-6">What's Included in Technical Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <Brain className="w-4 h-4 text-blue-600" />
              </div>
              <div>
                <h4 className="font-semibold text-gray-800">Methodology Deep Dive</h4>
                <p className="text-sm text-gray-600">Detailed explanation of research methods, algorithms, and techniques used</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <Calculator className="w-4 h-4 text-green-600" />
              </div>
              <div>
                <h4 className="font-semibold text-gray-800">Mathematical Foundations</h4>
                <p className="text-sm text-gray-600">Breakdown of mathematical concepts, formulas, and theoretical frameworks</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-purple-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <FlaskConical className="w-4 h-4 text-purple-600" />
              </div>
              <div>
                <h4 className="font-semibold text-gray-800">Experimental Design</h4>
                <p className="text-sm text-gray-600">Analysis of experimental setup, data collection, and validation methods</p>
              </div>
            </div>
          </div>
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <TrendingUp className="w-4 h-4 text-orange-600" />
              </div>
              <div>
                <h4 className="font-semibold text-gray-800">Results Interpretation</h4>
                <p className="text-sm text-gray-600">Detailed analysis of results and their statistical significance</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-red-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <Microscope className="w-4 h-4 text-red-600" />
              </div>
              <div>
                <h4 className="font-semibold text-gray-800">Limitations & Future Work</h4>
                <p className="text-sm text-gray-600">Discussion of paper's limitations and potential improvements</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <Brain className="w-4 h-4 text-indigo-600" />
              </div>
              <div>
                <h4 className="font-semibold text-gray-800">Implementation Insights</h4>
                <p className="text-sm text-gray-600">Guidance on implementing key algorithms and methods</p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default TechnicalAnalysis;
