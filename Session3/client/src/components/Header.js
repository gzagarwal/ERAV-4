import React from 'react';
import { motion } from 'framer-motion';
import { BookOpen, Brain, Sparkles } from 'lucide-react';

const Header = () => {
  return (
    <motion.header 
      className="bg-white/80 backdrop-blur-md border-b border-white/20 shadow-sm"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-center">
          <motion.div 
            className="flex items-center gap-4"
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.2 }}
          >
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl blur opacity-75"></div>
              <div className="relative bg-gradient-to-r from-indigo-500 to-purple-600 p-3 rounded-xl">
                <BookOpen className="w-8 h-8 text-white" />
              </div>
            </div>
            
            <div className="text-center">
              <motion.h1 
                className="text-3xl md:text-4xl font-bold gradient-text"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.6 }}
              >
                Research Paper Analyzer
              </motion.h1>
              <motion.p 
                className="text-gray-600 mt-1 flex items-center justify-center gap-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4, duration: 0.6 }}
              >
                <Brain className="w-4 h-4" />
                Powered by Gemini AI
                <Sparkles className="w-4 h-4 text-yellow-500" />
              </motion.p>
            </div>
          </motion.div>
        </div>
        
        <motion.div 
          className="mt-4 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.6 }}
        >
          <p className="text-gray-600 max-w-2xl mx-auto">
            Upload research papers and get instant AI-powered analysis, summaries, code implementations, 
            and detailed explanations in simple, understandable language.
          </p>
        </motion.div>
      </div>
    </motion.header>
  );
};

export default Header;
