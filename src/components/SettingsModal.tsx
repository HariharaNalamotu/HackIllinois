import React, { useState, useEffect } from 'react';
import { X, Eye, EyeOff, Check, AlertCircle } from 'lucide-react';

const API_KEY_STORAGE_KEY = 'ml-workflow-openai-api-key';
const TAVILY_KEY_STORAGE_KEY = 'ml-workflow-tavily-api-key';
const BACKEND_URL_STORAGE_KEY = 'ml-workflow-backend-url';

export const getStoredApiKey = (): string => {
  return localStorage.getItem(API_KEY_STORAGE_KEY) || '';
};

export const getStoredTavilyKey = (): string => {
  return localStorage.getItem(TAVILY_KEY_STORAGE_KEY) || '';
};

export const getStoredBackendUrl = (): string => {
  return localStorage.getItem(BACKEND_URL_STORAGE_KEY) || 'http://localhost:8787';
};

interface SettingsModalProps {
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ onClose }) => {
  const [apiKey, setApiKey] = useState('');
  const [tavilyKey, setTavilyKey] = useState('');
  const [backendUrl, setBackendUrl] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [showTavilyKey, setShowTavilyKey] = useState(false);
  const [status, setStatus] = useState<'idle' | 'checking' | 'connected' | 'error'>('idle');
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setApiKey(getStoredApiKey());
    setTavilyKey(getStoredTavilyKey());
    setBackendUrl(getStoredBackendUrl());
  }, []);

  const handleSave = () => {
    localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
    localStorage.setItem(TAVILY_KEY_STORAGE_KEY, tavilyKey);
    localStorage.setItem(BACKEND_URL_STORAGE_KEY, backendUrl);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleCheckConnection = async () => {
    setStatus('checking');
    try {
      const res = await fetch(`${backendUrl}/api/health`);
      if (res.ok) {
        setStatus('connected');
      } else {
        setStatus('error');
      }
    } catch {
      setStatus('error');
    }
    setTimeout(() => setStatus('idle'), 3000);
  };

  const maskedKey = apiKey ? apiKey.slice(0, 7) + '...' + apiKey.slice(-4) : '';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-[#12121a] border border-[#2a2a38] rounded-xl w-[480px] shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#22222e]">
          <h2 className="text-gray-200 font-semibold">Settings</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-[#1a1a24] rounded-lg text-gray-400 hover:text-gray-200 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* API Key */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              OpenAI API Key
            </label>
            <div className="relative">
              <input
                type={showKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-..."
                className="w-full bg-[#1a1a24] border border-[#2a2a38] rounded-lg px-4 py-3 pr-12 text-gray-200 focus:outline-none focus:border-[#00d4ff] font-mono text-sm"
              />
              <button
                onClick={() => setShowKey(!showKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
              >
                {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              Stored locally in your browser. Never sent to our servers.
            </p>
          </div>

          {/* Tavily API Key */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Tavily API Key
            </label>
            <div className="relative">
              <input
                type={showTavilyKey ? 'text' : 'password'}
                value={tavilyKey}
                onChange={(e) => setTavilyKey(e.target.value)}
                placeholder="tvly-..."
                className="w-full bg-[#1a1a24] border border-[#2a2a38] rounded-lg px-4 py-3 pr-12 text-gray-200 focus:outline-none focus:border-[#00d4ff] font-mono text-sm"
              />
              <button
                onClick={() => setShowTavilyKey(!showTavilyKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
              >
                {showTavilyKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              Optional. Enables web search for better tool code generation.
            </p>
          </div>

          {/* Backend URL */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Backend URL
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={backendUrl}
                onChange={(e) => setBackendUrl(e.target.value)}
                placeholder="http://localhost:8787"
                className="flex-1 bg-[#1a1a24] border border-[#2a2a38] rounded-lg px-4 py-3 text-gray-200 focus:outline-none focus:border-[#00d4ff] font-mono text-sm"
              />
              <button
                onClick={handleCheckConnection}
                disabled={status === 'checking'}
                className="px-3 py-2 bg-[#1a1a24] border border-[#2a2a38] rounded-lg text-sm text-gray-400 hover:text-gray-200 hover:border-[#00d4ff] transition-colors"
              >
                {status === 'checking' ? 'Checking...' : 'Test'}
              </button>
            </div>
            {status === 'connected' && (
              <p className="text-xs text-green-400 mt-1 flex items-center gap-1">
                <Check className="w-3 h-3" /> Connected
              </p>
            )}
            {status === 'error' && (
              <p className="text-xs text-red-400 mt-1 flex items-center gap-1">
                <AlertCircle className="w-3 h-3" /> Cannot connect to backend
              </p>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-4 border-t border-[#22222e]">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-400 hover:text-gray-200 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-6 py-2 bg-[#00d4ff] text-[#0a0a0f] rounded-lg font-medium hover:bg-[#00b8d4] transition-colors"
          >
            {saved ? 'Saved!' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
};
