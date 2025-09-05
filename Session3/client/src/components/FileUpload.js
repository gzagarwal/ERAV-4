import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { Upload, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import axios from 'axios';

const FileUpload = ({ onUpload, isLoading, setIsLoading }) => {
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    if (file.type !== 'application/pdf') {
      toast.error('Please upload a PDF file');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      toast.error('File size must be less than 10MB');
      return;
    }

    setIsLoading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('pdf', file);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const response = await axios.post('/api/upload-pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(Math.min(percentCompleted, 90));
        },
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (response.data.success) {
        toast.success('Paper uploaded and analyzed successfully!');
        onUpload(response.data);
      } else {
        throw new Error(response.data.message || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error(error.response?.data?.detail || 'Failed to upload and analyze the paper');
    } finally {
      setIsLoading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  }, [onUpload, setIsLoading]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: false,
    disabled: isLoading
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="max-w-4xl mx-auto"
    >
      <div className="text-center mb-8">
        <motion.h2 
          className="text-3xl font-bold text-gray-800 mb-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Upload Your Research Paper
        </motion.h2>
        <motion.p 
          className="text-gray-600 text-lg"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          Drag and drop your PDF file or click to browse. We'll analyze it with AI and provide insights.
        </motion.p>
      </div>

      <motion.div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300
          ${isDragActive && !isDragReject 
            ? 'border-indigo-500 bg-indigo-50' 
            : isDragReject 
              ? 'border-red-500 bg-red-50' 
              : 'border-gray-300 bg-white hover:border-indigo-400 hover:bg-indigo-50'
          }
          ${isLoading ? 'pointer-events-none opacity-75' : ''}
        `}
        whileHover={!isLoading ? { scale: 1.02 } : {}}
        whileTap={!isLoading ? { scale: 0.98 } : {}}
      >
        <input {...getInputProps()} />
        
        <div className="space-y-6">
          <motion.div
            className="mx-auto w-20 h-20 rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 flex items-center justify-center"
            animate={isLoading ? { rotate: 360 } : {}}
            transition={{ duration: 2, repeat: isLoading ? Infinity : 0, ease: "linear" }}
          >
            {isLoading ? (
              <div className="w-8 h-8 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : isDragActive ? (
              <Upload className="w-10 h-10 text-white" />
            ) : (
              <FileText className="w-10 h-10 text-white" />
            )}
          </motion.div>

          <div>
            {isLoading ? (
              <div className="space-y-4">
                <h3 className="text-xl font-semibold text-gray-700">Analyzing your paper...</h3>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <motion.div
                    className="bg-gradient-to-r from-indigo-500 to-purple-600 h-3 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${uploadProgress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                <p className="text-sm text-gray-500">{uploadProgress}% complete</p>
              </div>
            ) : isDragActive ? (
              <div>
                <h3 className="text-xl font-semibold text-indigo-600 mb-2">Drop your PDF here</h3>
                <p className="text-gray-600">Release to upload and analyze</p>
              </div>
            ) : isDragReject ? (
              <div>
                <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-red-600 mb-2">Invalid file type</h3>
                <p className="text-gray-600">Please upload a PDF file</p>
              </div>
            ) : (
              <div>
                <h3 className="text-xl font-semibold text-gray-700 mb-2">
                  Choose a PDF file or drag it here
                </h3>
                <p className="text-gray-600 mb-4">
                  Supported format: PDF (max 10MB)
                </p>
                <motion.button
                  className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg font-medium hover:from-indigo-600 hover:to-purple-700 transition-all duration-200"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Browse Files
                </motion.button>
              </div>
            )}
          </div>
        </div>
      </motion.div>

      <motion.div 
        className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        <div className="text-center p-6 bg-white rounded-xl shadow-sm">
          <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
            <CheckCircle className="w-6 h-6 text-blue-600" />
          </div>
          <h4 className="font-semibold text-gray-800 mb-2">AI-Powered Analysis</h4>
          <p className="text-sm text-gray-600">Get comprehensive summaries and insights using advanced AI</p>
        </div>
        
        <div className="text-center p-6 bg-white rounded-xl shadow-sm">
          <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
            <FileText className="w-6 h-6 text-green-600" />
          </div>
          <h4 className="font-semibold text-gray-800 mb-2">Multiple Formats</h4>
          <p className="text-sm text-gray-600">Support for various research paper formats and structures</p>
        </div>
        
        <div className="text-center p-6 bg-white rounded-xl shadow-sm">
          <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
            <Upload className="w-6 h-6 text-purple-600" />
          </div>
          <h4 className="font-semibold text-gray-800 mb-2">Easy Upload</h4>
          <p className="text-sm text-gray-600">Simple drag-and-drop interface for quick file processing</p>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default FileUpload;
