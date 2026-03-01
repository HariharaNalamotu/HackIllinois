import React, { useState, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Send, ThumbsUp, ThumbsDown, Brain, CheckCircle } from 'lucide-react';
import { useWorkflowsStore } from '../store/workflowsStore';
import { chatStream, submitFeedback, ChatMessage, WorkflowConfig, RLAIFScore } from '../services/api';
import { getStoredApiKey, getStoredBackendUrl } from '../components/SettingsModal';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  feedbackGiven?: 'up' | 'down';
  rlaifScore?: RLAIFScore;
}

interface RLHFSessionState {
  iterations: number;
  currentIteration: number;
  status: 'active' | 'completed';
}

export const RLHFPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const workflows = useWorkflowsStore((state) => state.workflows);
  const workflow = workflows.find((w) => w.id === id);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [session, setSession] = useState<RLHFSessionState | null>(null);
  const [feedbackText, setFeedbackText] = useState<Record<string, string>>({});
  const [showTextFor, setShowTextFor] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Fetch session state on mount
  useEffect(() => {
    const fetchSession = async () => {
      try {
        const backendUrl = getStoredBackendUrl();
        const res = await fetch(`${backendUrl}/api/feedback`);
        if (res.ok) {
          const data = await res.json() as any;
          if (data.session) {
            setSession({
              iterations: data.session.iterations,
              currentIteration: data.session.currentIteration,
              status: data.session.status,
            });
          }
        }
      } catch {}
    };
    fetchSession();
  }, []);

  if (!workflow) {
    return (
      <div className="h-screen w-screen flex flex-col items-center justify-center bg-[#0a0a0f]">
        <p className="text-gray-400 mb-4">Workflow not found</p>
        <button onClick={() => navigate('/')} className="text-[#00d4ff] hover:underline">
          Go back home
        </button>
      </div>
    );
  }

  const workflowConfig: WorkflowConfig = { nodes: workflow.nodes, edges: workflow.edges };
  const feedbackCount = messages.filter((m) => m.role === 'assistant' && m.feedbackGiven).length;
  const totalIterations = session?.iterations || 3;
  const isComplete = feedbackCount >= totalIterations;
  const progressPercent = Math.min(100, Math.round((feedbackCount / totalIterations) * 100));

  const getConversationSnippet = (messageId: string): string => {
    const msgIndex = messages.findIndex((m) => m.id === messageId);
    const assistantMsg = messages[msgIndex];
    const userMsg = messages.slice(0, msgIndex).reverse().find((m) => m.role === 'user');
    return userMsg
      ? `User: ${userMsg.content}\nAssistant: ${assistantMsg?.content || ''}`
      : assistantMsg?.content || '';
  };

  const handleFeedback = async (messageId: string, rating: 'up' | 'down') => {
    const snippet = getConversationSnippet(messageId);

    setMessages((prev) =>
      prev.map((m) => (m.id === messageId ? { ...m, feedbackGiven: rating } : m))
    );

    const result = await submitFeedback({ messageId, rating, conversationSnippet: snippet });

    if (rating === 'down') {
      setShowTextFor(messageId);
    }

    // Update session from response
    try {
      const backendUrl = getStoredBackendUrl();
      const res = await fetch(`${backendUrl}/api/feedback`);
      if (res.ok) {
        const data = await res.json() as any;
        if (data.session) {
          setSession({
            iterations: data.session.iterations,
            currentIteration: data.session.currentIteration,
            status: data.session.status,
          });
        }
      }
    } catch {}
  };

  const handleSubmitTextFeedback = async (messageId: string) => {
    const text = feedbackText[messageId];
    if (!text?.trim()) return;

    const snippet = getConversationSnippet(messageId);
    await submitFeedback({
      messageId,
      rating: 'down',
      feedback: text,
      conversationSnippet: snippet,
    });

    setShowTextFor(null);
    setFeedbackText((prev) => ({ ...prev, [messageId]: '' }));
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
    setMessages((prev) => [...prev, { id: assistantId, role: 'assistant', content: '' }]);

    const chatMessages: ChatMessage[] = updatedMessages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    await chatStream(chatMessages, workflowConfig, {
      onToken: (token) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + token } : m))
        );
      },
      onToolCall: () => {},
      onSubAgent: () => {},
      onRlaifScore: (score) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, rlaifScore: score } : m))
        );
      },
      onDone: () => setIsStreaming(false),
      onError: (error) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: m.content || `Error: ${error}` } : m
          )
        );
        setIsStreaming(false);
      },
    });
  };

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
          <Brain className="w-5 h-5 text-[#f472b6]" />
          <h1 className="text-gray-200 font-semibold">RLHF Training: {workflow.name}</h1>
        </div>
        <div className="ml-auto flex items-center gap-3">
          <div className="text-xs text-gray-400">
            Feedback: {feedbackCount} / {totalIterations}
          </div>
          <div className="w-32 h-2 bg-[#1a1a24] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${progressPercent}%`,
                backgroundColor: isComplete ? '#22c55e' : '#f472b6',
              }}
            />
          </div>
          {isComplete && (
            <span className="text-xs text-green-400 flex items-center gap-1">
              <CheckCircle className="w-3.5 h-3.5" /> Complete
            </span>
          )}
        </div>
      </header>

      {/* Instructions banner */}
      {feedbackCount === 0 && (
        <div className="bg-[#f472b6]/10 border-b border-[#f472b6]/20 px-6 py-3">
          <p className="text-sm text-[#f472b6]">
            Chat with the model and rate each response with thumbs up/down. Your feedback directly
            shapes how the model responds in future messages. Provide {totalIterations} rounds of feedback
            to complete training.
          </p>
        </div>
      )}

      {/* Completion banner */}
      {isComplete && (
        <div className="bg-[#22c55e]/10 border-b border-[#22c55e]/20 px-6 py-3 flex items-center justify-between">
          <p className="text-sm text-green-400">
            RLHF training complete! The model has been tuned with your feedback.
          </p>
          <button
            onClick={() => navigate(`/editor/${id}`)}
            className="px-4 py-1.5 bg-[#22c55e] text-[#0a0a0f] rounded-lg text-sm font-medium hover:bg-[#16a34a] transition-colors"
          >
            Back to Editor
          </button>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <Brain className="w-12 h-12 mb-4 opacity-50 text-[#f472b6]" />
            <p>Send a message to begin RLHF training</p>
            <p className="text-xs mt-1 text-gray-600">Rate responses to teach the model your preferences</p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className="max-w-[70%]">
                <div
                  className={`px-4 py-3 rounded-lg ${
                    msg.role === 'user'
                      ? 'bg-[#00d4ff] text-[#0a0a0f]'
                      : 'bg-[#1a1a24] text-gray-200 border border-[#2a2a38]'
                  }`}
                >
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                </div>

                {/* Feedback UI for assistant messages */}
                {msg.role === 'assistant' && msg.content && !isStreaming && (
                  <div className="mt-2">
                    {!msg.feedbackGiven ? (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">Rate this response:</span>
                        <button
                          onClick={() => handleFeedback(msg.id, 'up')}
                          className="p-2 rounded-lg text-gray-500 hover:text-green-400 hover:bg-green-400/10 transition-colors"
                          title="Good response"
                        >
                          <ThumbsUp className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleFeedback(msg.id, 'down')}
                          className="p-2 rounded-lg text-gray-500 hover:text-red-400 hover:bg-red-400/10 transition-colors"
                          title="Bad response"
                        >
                          <ThumbsDown className="w-4 h-4" />
                        </button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <span className={`text-xs flex items-center gap-1 ${
                          msg.feedbackGiven === 'up' ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {msg.feedbackGiven === 'up' ? (
                            <><ThumbsUp className="w-3.5 h-3.5" /> Positive feedback recorded</>
                          ) : (
                            <><ThumbsDown className="w-3.5 h-3.5" /> Negative feedback recorded</>
                          )}
                        </span>
                      </div>
                    )}

                    {/* Text feedback for negative ratings */}
                    {showTextFor === msg.id && (
                      <div className="mt-2 flex gap-2">
                        <input
                          type="text"
                          value={feedbackText[msg.id] || ''}
                          onChange={(e) => setFeedbackText((prev) => ({ ...prev, [msg.id]: e.target.value }))}
                          onKeyDown={(e) => e.key === 'Enter' && handleSubmitTextFeedback(msg.id)}
                          placeholder="What's wrong with this response?"
                          className="flex-1 bg-[#0a0a0f] border border-[#2a2a38] rounded-lg px-3 py-2 text-sm text-gray-300 focus:outline-none focus:border-[#f472b6]"
                        />
                        <button
                          onClick={() => handleSubmitTextFeedback(msg.id)}
                          className="px-4 py-2 bg-[#f472b6] text-[#0a0a0f] rounded-lg text-sm font-medium hover:bg-[#ec4899] transition-colors"
                        >
                          Submit
                        </button>
                      </div>
                    )}
                  </div>
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
            placeholder={isComplete ? 'Training complete — you can keep chatting or go back to editor' : 'Type a message to train the model...'}
            disabled={isStreaming}
            className="flex-1 bg-[#1a1a24] border border-[#2a2a38] rounded-lg px-4 py-3 text-gray-200 focus:outline-none focus:border-[#f472b6] disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={isStreaming}
            className="px-6 py-3 bg-[#f472b6] text-[#0a0a0f] rounded-lg font-medium hover:bg-[#ec4899] transition-colors disabled:opacity-50"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};
