import React, { useState, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Send, Image, Mic, FileAudio, X, ThumbsUp, ThumbsDown, ChevronDown, ChevronRight, Bot } from 'lucide-react';
import { useWorkflowsStore } from '../store/workflowsStore';
import { InputNodeType } from '../store/workflowStore';
import { chatStream, submitFeedback, RLAIFScore, ChatMessage, WorkflowConfig } from '../services/api';

interface SubAgentStep {
  agent: string;
  content: string;
}

interface ToolCallInfo {
  name: string;
  arguments: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  rlaifScore?: RLAIFScore;
  feedbackGiven?: 'up' | 'down';
  toolCalls?: ToolCallInfo[];
  subAgentSteps?: SubAgentStep[];
}

// Quality badge color based on RLAIF overall score
const getQualityColor = (score: number): string => {
  if (score >= 8) return '#22c55e'; // green
  if (score >= 5) return '#fbbf24'; // yellow
  return '#ef4444'; // red
};

const getQualityLabel = (score: number): string => {
  if (score >= 8) return 'High Quality';
  if (score >= 5) return 'Moderate';
  return 'Low Quality';
};

// RLHF Feedback Buttons
const FeedbackButtons: React.FC<{
  messageId: string;
  feedbackGiven?: 'up' | 'down';
  onFeedback: (messageId: string, rating: 'up' | 'down') => void;
}> = ({ messageId, feedbackGiven, onFeedback }) => {
  const [showTextInput, setShowTextInput] = useState(false);
  const [feedbackText, setFeedbackText] = useState('');

  const handleFeedback = (rating: 'up' | 'down') => {
    onFeedback(messageId, rating);
    if (rating === 'down') {
      setShowTextInput(true);
    }
  };

  const handleSubmitText = () => {
    if (feedbackText.trim()) {
      submitFeedback({ messageId, rating: feedbackGiven || 'down', feedback: feedbackText });
      setShowTextInput(false);
      setFeedbackText('');
    }
  };

  return (
    <div className="mt-2">
      <div className="flex items-center gap-1">
        <button
          onClick={() => handleFeedback('up')}
          className={`p-1.5 rounded transition-colors ${
            feedbackGiven === 'up'
              ? 'text-green-400 bg-green-400/10'
              : 'text-gray-600 hover:text-green-400 hover:bg-green-400/10'
          }`}
          title="Good response"
        >
          <ThumbsUp className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={() => handleFeedback('down')}
          className={`p-1.5 rounded transition-colors ${
            feedbackGiven === 'down'
              ? 'text-red-400 bg-red-400/10'
              : 'text-gray-600 hover:text-red-400 hover:bg-red-400/10'
          }`}
          title="Bad response"
        >
          <ThumbsDown className="w-3.5 h-3.5" />
        </button>
      </div>
      {showTextInput && (
        <div className="mt-2 flex gap-2">
          <input
            type="text"
            value={feedbackText}
            onChange={(e) => setFeedbackText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSubmitText()}
            placeholder="What was wrong?"
            className="flex-1 bg-[#0a0a0f] border border-[#2a2a38] rounded px-3 py-1.5 text-xs text-gray-300 focus:outline-none focus:border-[#00d4ff]"
          />
          <button
            onClick={handleSubmitText}
            className="px-3 py-1.5 bg-[#2a2a38] rounded text-xs text-gray-300 hover:bg-[#3a3a48]"
          >
            Send
          </button>
        </div>
      )}
    </div>
  );
};

// Sub-agent steps collapsible block
const SubAgentBlock: React.FC<{ steps: SubAgentStep[] }> = ({ steps }) => {
  const [expanded, setExpanded] = useState(false);

  if (steps.length === 0) return null;

  return (
    <div className="mt-2 border border-[#2a2a38] rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-400 hover:bg-[#22222e] transition-colors"
      >
        <Bot className="w-3.5 h-3.5 text-[#fb923c]" />
        <span>{steps.length} sub-agent step{steps.length !== 1 ? 's' : ''}</span>
        {expanded ? <ChevronDown className="w-3.5 h-3.5 ml-auto" /> : <ChevronRight className="w-3.5 h-3.5 ml-auto" />}
      </button>
      {expanded && (
        <div className="border-t border-[#2a2a38] p-3 space-y-2">
          {steps.map((step, i) => (
            <div key={i} className="text-xs">
              <span className="text-[#fb923c] font-medium">{step.agent}:</span>
              <span className="text-gray-400 ml-2">{step.content}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Chat Interface for Text/Agentic LLM workflows
const ChatInterface: React.FC<{ workflowConfig: WorkflowConfig }> = ({ workflowConfig }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleFeedback = (messageId: string, rating: 'up' | 'down') => {
    setMessages((prev) =>
      prev.map((m) => (m.id === messageId ? { ...m, feedbackGiven: rating } : m))
    );
    submitFeedback({ messageId, rating });
  };

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput('');
    setIsStreaming(true);

    const assistantId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      toolCalls: [],
      subAgentSteps: [],
    };
    setMessages((prev) => [...prev, assistantMessage]);

    const chatMessages: ChatMessage[] = updatedMessages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    await chatStream(chatMessages, workflowConfig, {
      onToken: (token) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: m.content + token } : m
          )
        );
      },
      onToolCall: (toolCall) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, toolCalls: [...(m.toolCalls || []), toolCall] }
              : m
          )
        );
      },
      onSubAgent: (step) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, subAgentSteps: [...(m.subAgentSteps || []), step] }
              : m
          )
        );
      },
      onRlaifScore: (score) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, rlaifScore: score } : m
          )
        );
      },
      onDone: () => {
        setIsStreaming(false);
      },
      onError: (error) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: m.content || `Error: ${error}` }
              : m
          )
        );
        setIsStreaming(false);
      },
    });
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
              <div className={`max-w-[70%] ${msg.role === 'user' ? '' : ''}`}>
                <div
                  className={`px-4 py-3 rounded-lg ${
                    msg.role === 'user'
                      ? 'bg-[#00d4ff] text-[#0a0a0f]'
                      : 'bg-[#1a1a24] text-gray-200 border border-[#2a2a38]'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <div className="flex-1 whitespace-pre-wrap">{msg.content}</div>
                    {msg.rlaifScore && (
                      <div
                        className="flex-shrink-0 px-2 py-0.5 rounded-full text-[10px] font-medium"
                        style={{
                          backgroundColor: getQualityColor(msg.rlaifScore.overall) + '20',
                          color: getQualityColor(msg.rlaifScore.overall),
                        }}
                        title={`Helpfulness: ${msg.rlaifScore.helpfulness}/10, Accuracy: ${msg.rlaifScore.accuracy}/10, Safety: ${msg.rlaifScore.safety}/10`}
                      >
                        {getQualityLabel(msg.rlaifScore.overall)} ({msg.rlaifScore.overall}/10)
                      </div>
                    )}
                  </div>

                  {/* Tool calls display */}
                  {msg.toolCalls && msg.toolCalls.length > 0 && (
                    <div className="mt-2 space-y-1">
                      {msg.toolCalls.map((tc, i) => (
                        <div key={i} className="text-xs bg-[#0a0a0f] rounded px-2 py-1 font-mono">
                          <span className="text-[#ffd700]">{tc.name}</span>
                          <span className="text-gray-500">({tc.arguments})</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Sub-agent steps */}
                {msg.subAgentSteps && msg.subAgentSteps.length > 0 && (
                  <SubAgentBlock steps={msg.subAgentSteps} />
                )}

                {/* RLHF Feedback */}
                {msg.role === 'assistant' && msg.content && (
                  <FeedbackButtons
                    messageId={msg.id}
                    feedbackGiven={msg.feedbackGiven}
                    onFeedback={handleFeedback}
                  />
                )}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
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
            disabled={isStreaming}
            className="flex-1 bg-[#1a1a24] border border-[#2a2a38] rounded-lg px-4 py-3 text-gray-200 focus:outline-none focus:border-[#00d4ff] disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={isStreaming}
            className="px-6 py-3 bg-[#00d4ff] text-[#0a0a0f] rounded-lg font-medium hover:bg-[#00b8d4] transition-colors disabled:opacity-50"
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

  // Build workflow config for API calls
  const workflowConfig: WorkflowConfig = {
    nodes: workflow.nodes,
    edges: workflow.edges,
  };

  // Determine the primary input type
  const inputTypes: InputNodeType[] = ['textInput', 'imageInput', 'audioInput', 'spreadsheetInput'];
  const workflowInputTypes = workflow.nodes
    .filter((node) => inputTypes.includes(node.data.type as InputNodeType))
    .map((node) => node.data.type as InputNodeType);

  // Prioritize interface based on input types
  let InterfaceComponent: React.FC = () => <ChatInterface workflowConfig={workflowConfig} />;
  let interfaceLabel = 'Chat';

  if (workflowInputTypes.includes('imageInput')) {
    InterfaceComponent = VisualInterface;
    interfaceLabel = 'Visual Analysis';
  } else if (workflowInputTypes.includes('audioInput')) {
    InterfaceComponent = () => <AudioInterface isVoice={false} />;
    interfaceLabel = 'Audio Analysis';
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0a0a0f]">
      {/* Header */}
      <header className="h-12 bg-[#12121a] border-b border-[#22222e] flex items-center px-4 gap-4">
        <button
          onClick={() => navigate(`/editor/${id}`)}
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
