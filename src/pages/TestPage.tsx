import React, { useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Send, Image, Mic, FileAudio, X } from 'lucide-react';
import { useWorkflowsStore } from '../store/workflowsStore';
import { InputNodeType } from '../store/workflowStore';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

// Chat Interface for Text/Agentic LLM workflows
const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    // Simulate response
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'This is a simulated response. In production, this would be connected to your trained model.',
      };
      setMessages((prev) => [...prev, assistantMessage]);
    }, 1000);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <Send className="w-12 h-12 mb-4 opacity-50" />
            <p>Start a conversation to test your model</p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[70%] px-4 py-3 rounded-lg ${
                  msg.role === 'user'
                    ? 'bg-[#00d4ff] text-[#0a0a0f]'
                    : 'bg-[#1a1a24] text-gray-200 border border-[#2a2a38]'
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Input */}
      <div className="p-4 border-t border-[#22222e]">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Type a message..."
            className="flex-1 bg-[#1a1a24] border border-[#2a2a38] rounded-lg px-4 py-3 text-gray-200 focus:outline-none focus:border-[#00d4ff]"
          />
          <button
            onClick={handleSend}
            className="px-6 py-3 bg-[#00d4ff] text-[#0a0a0f] rounded-lg font-medium hover:bg-[#00b8d4] transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

// Visual Interface for CV workflows
const VisualInterface: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(file);
      setResult(null);
    }
  };

  const handleAnalyze = () => {
    if (!selectedFile) return;
    // Simulate analysis
    setTimeout(() => {
      setResult('Detected: Person (95%), Car (87%), Tree (72%)');
    }, 1500);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="flex flex-col items-center justify-center h-full p-8">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,video/*"
        onChange={handleFileSelect}
        className="hidden"
      />

      {!preview ? (
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex flex-col items-center justify-center w-96 h-64 border-2 border-dashed border-[#2a2a38] rounded-xl hover:border-[#00d4ff] transition-colors cursor-pointer"
        >
          <Image className="w-16 h-16 text-gray-600 mb-4" />
          <p className="text-gray-400 mb-2">Click to upload an image or video</p>
          <p className="text-gray-600 text-sm">Supports JPG, PNG, MP4, WebM</p>
        </button>
      ) : (
        <div className="flex flex-col items-center">
          <div className="relative mb-6">
            <img
              src={preview}
              alt="Preview"
              className="max-w-lg max-h-80 rounded-lg border border-[#2a2a38]"
            />
            <button
              onClick={handleClear}
              className="absolute -top-3 -right-3 p-1 bg-red-500 rounded-full text-white hover:bg-red-600"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {result ? (
            <div className="text-center">
              <p className="text-2xl font-bold text-[#22c55e] mb-4 animate-pulse">
                {result}
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-[#00d4ff] hover:underline"
              >
                Try another image
              </button>
            </div>
          ) : (
            <button
              onClick={handleAnalyze}
              className="px-8 py-3 bg-[#00d4ff] text-[#0a0a0f] rounded-lg font-medium hover:bg-[#00b8d4] transition-colors"
            >
              Analyze
            </button>
          )}
        </div>
      )}
    </div>
  );
};

// Audio Interface for Audio/Voice workflows
const AudioInterface: React.FC<{ isVoice?: boolean }> = ({ isVoice = false }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
    }
  };

  const handleAnalyze = () => {
    if (!selectedFile) return;
    // Simulate analysis
    setTimeout(() => {
      if (isVoice) {
        setResult('Transcription: "Hello, this is a test of the voice recognition system."');
      } else {
        setResult('Detected: Speech (92%), Music (45%), Background Noise (23%)');
      }
    }, 1500);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setResult(null);
  };

  return (
    <div className="flex flex-col items-center justify-center h-full p-8">
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileSelect}
        className="hidden"
      />

      {!selectedFile ? (
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex flex-col items-center justify-center w-96 h-64 border-2 border-dashed border-[#2a2a38] rounded-xl hover:border-[#00d4ff] transition-colors cursor-pointer"
        >
          {isVoice ? (
            <Mic className="w-16 h-16 text-gray-600 mb-4" />
          ) : (
            <FileAudio className="w-16 h-16 text-gray-600 mb-4" />
          )}
          <p className="text-gray-400 mb-2">Click to upload an audio file</p>
          <p className="text-gray-600 text-sm">Supports MP3, WAV, OGG, M4A</p>
        </button>
      ) : (
        <div className="flex flex-col items-center">
          <div className="relative mb-6 p-8 bg-[#1a1a24] border border-[#2a2a38] rounded-xl">
            <div className="flex items-center gap-4">
              {isVoice ? (
                <Mic className="w-12 h-12 text-[#f97316]" />
              ) : (
                <FileAudio className="w-12 h-12 text-[#22c55e]" />
              )}
              <div>
                <p className="text-gray-200 font-medium">{selectedFile.name}</p>
                <p className="text-gray-500 text-sm">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <button
                onClick={handleClear}
                className="ml-4 p-1 text-gray-500 hover:text-red-400"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {result ? (
            <div className="text-center max-w-lg">
              <p className="text-2xl font-bold text-[#22c55e] mb-4">
                {result}
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-[#00d4ff] hover:underline"
              >
                Try another file
              </button>
            </div>
          ) : (
            <button
              onClick={handleAnalyze}
              className="px-8 py-3 bg-[#00d4ff] text-[#0a0a0f] rounded-lg font-medium hover:bg-[#00b8d4] transition-colors"
            >
              Analyze
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export const TestPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const workflows = useWorkflowsStore((state) => state.workflows);
  const workflow = workflows.find((w) => w.id === id);

  if (!workflow) {
    return (
      <div className="h-screen w-screen flex flex-col items-center justify-center bg-[#0a0a0f]">
        <p className="text-gray-400 mb-4">Workflow not found</p>
        <button
          onClick={() => navigate('/')}
          className="text-[#00d4ff] hover:underline"
        >
          Go back home
        </button>
      </div>
    );
  }

  // Determine the primary input type
  const inputTypes: InputNodeType[] = ['textRetrieval', 'agenticLLM', 'visualData', 'audioData', 'voiceInput'];
  const workflowInputTypes = workflow.nodes
    .filter((node) => inputTypes.includes(node.data.type as InputNodeType))
    .map((node) => node.data.type as InputNodeType);

  // Prioritize interface based on input types
  let InterfaceComponent = ChatInterface;
  let interfaceLabel = 'Chat';

  if (workflowInputTypes.includes('visualData')) {
    InterfaceComponent = VisualInterface;
    interfaceLabel = 'Visual Analysis';
  } else if (workflowInputTypes.includes('audioData')) {
    InterfaceComponent = () => <AudioInterface isVoice={false} />;
    interfaceLabel = 'Audio Analysis';
  } else if (workflowInputTypes.includes('voiceInput')) {
    InterfaceComponent = () => <AudioInterface isVoice={true} />;
    interfaceLabel = 'Voice Processing';
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0a0a0f]">
      {/* Header */}
      <header className="h-12 bg-[#12121a] border-b border-[#22222e] flex items-center px-4 gap-4">
        <button
          onClick={() => navigate('/')}
          className="p-2 hover:bg-[#1a1a24] rounded-lg text-gray-400 hover:text-gray-200 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex items-center gap-2">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="8" cy="8" r="3" stroke="#00d4ff" strokeWidth="1.5" />
            <circle cx="16" cy="16" r="3" stroke="#00d4ff" strokeWidth="1.5" />
            <path d="M10.5 9.5L13.5 14.5" stroke="#00d4ff" strokeWidth="1.5" />
          </svg>
          <h1 className="text-gray-200 font-semibold">Inference: {workflow.name}</h1>
        </div>
        <div className="ml-auto px-3 py-1 bg-[#1a1a24] rounded-full text-sm text-gray-400">
          {interfaceLabel}
        </div>
      </header>

      {/* Interface */}
      <div className="flex-1 overflow-hidden">
        <InterfaceComponent />
      </div>
    </div>
  );
};
