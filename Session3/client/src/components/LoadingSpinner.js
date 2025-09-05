import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Sparkles } from 'lucide-react';

const LoadingSpinner = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
    >
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.8, opacity: 0 }}
        className="bg-white rounded-2xl p-8 shadow-2xl max-w-md mx-4 text-center"
      >
        <div className="relative mb-6">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            className="w-16 h-16 mx-auto"
          >
            <div className="w-full h-full border-4 border-indigo-200 border-t-indigo-600 rounded-full"></div>
          </motion.div>
          <motion.div
            animate={{ 
              scale: [1, 1.2, 1],
              opacity: [0.7, 1, 0.7]
            }}
            transition={{ 
              duration: 1.5, 
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute inset-0 flex items-center justify-center"
          >
            <Brain className="w-8 h-8 text-indigo-600" />
          </motion.div>
        </div>
        
        <motion.h3 
          className="text-xl font-semibold text-gray-800 mb-2"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        >
          AI is analyzing your paper
        </motion.h3>
        
        <motion.p 
          className="text-gray-600 mb-4"
          animate={{ opacity: [0.7, 1, 0.7] }}
          transition={{ duration: 1.5, repeat: Infinity, delay: 0.5 }}
        >
          This may take a few moments...
        </motion.p>
        
        <div className="flex items-center justify-center gap-2">
          <motion.div
            animate={{ opacity: [0, 1, 0] }}
            transition={{ duration: 1, repeat: Infinity, delay: 0 }}
            className="w-2 h-2 bg-indigo-600 rounded-full"
          />
          <motion.div
            animate={{ opacity: [0, 1, 0] }}
            transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
            className="w-2 h-2 bg-indigo-600 rounded-full"
          />
          <motion.div
            animate={{ opacity: [0, 1, 0] }}
            transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
            className="w-2 h-2 bg-indigo-600 rounded-full"
          />
        </div>
        
        <motion.div
          animate={{ y: [0, -5, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="mt-4 flex items-center justify-center gap-2 text-indigo-600"
        >
          <Sparkles className="w-4 h-4" />
          <span className="text-sm font-medium">Powered by Gemini AI</span>
          <Sparkles className="w-4 h-4" />
        </motion.div>
      </motion.div>
    </motion.div>
  );
};

export default LoadingSpinner;
