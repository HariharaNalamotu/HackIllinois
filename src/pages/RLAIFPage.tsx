import React, { useState, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Activity, Play, Square, CheckCircle, Zap, MessageSquare, Bot, BarChart3 } from 'lucide-react';
import { useWorkflowsStore } from '../store/workflowsStore';
import { getStoredBackendUrl, getStoredApiKey } from '../components/SettingsModal';
import type { RLAIFScore } from '../services/api';

interface IterationEntry {
  iteration: number;
  prompt: string;
  response: string;
  score: RLAIFScore | null;
  status: 'generating_prompt' | 'getting_response' | 'evaluating' | 'complete' | 'error';
  error?: string;
}

function ScoreBadge({ label, value }: { label: string; value: number }) {
  const color =
    value >= 8 ? 'text-green-400 bg-green-400/10' :
    value >= 5 ? 'text-yellow-400 bg-yellow-400/10' :
    'text-red-400 bg-red-400/10';

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${color}`}>
      {label}: {value}/10
    </span>
  );
}

function PhaseIndicator({ phase }: { phase: string }) {
  const labels: Record<string, string> = {
    generating_prompt: 'AI generating test prompt...',
    getting_response: 'Getting model response...',
    evaluating: 'AI evaluating response...',
    complete: 'Complete',
    error: 'Error',
  };

  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-gray-400">
      {phase !== 'complete' && phase !== 'error' && (
        <span className="w-2 h-2 rounded-full bg-[#a78bfa] animate-pulse" />
      )}
      {labels[phase] || phase}
    </span>
  );
}

export const RLAIFPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const workflows = useWorkflowsStore((state) => state.workflows);
  const workflow = workflows.find((w) => w.id === id);

  const [isRunning, setIsRunning] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [iterations, setIterations] = useState<IterationEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sessionInfo, setSessionInfo] = useState<{
    evaluatorModel?: string;
    resolvedModel?: string;
    targetModel?: string;
    iterations?: number;
  } | null>(null);
  const [summary, setSummary] = useState<{
    totalEvaluations: number;
    averageScore: number;
    goodPatterns: number;
  } | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [iterations]);

  const startTraining = async () => {
    if (!workflow) return;

    setIsRunning(true);
    setIsComplete(false);
    setIterations([]);
    setError(null);
    setSummary(null);
    setSessionInfo(null);

    const backendUrl = getStoredBackendUrl();
    const apiKey = getStoredApiKey();
    const abortController = new AbortController();
    abortRef.current = abortController;

    try {
      const res = await fetch(`${backendUrl}/api/train/rlaif`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { 'X-API-Key': apiKey } : {}),
        },
        body: JSON.stringify({
          workflowConfig: {
            nodes: workflow.nodes,
            edges: workflow.edges,
          },
        }),
        signal: abortController.signal,
      });

      if (!res.ok) {
        // Non-SSE error responses — read as text first, then try JSON
        const errText = await res.text().catch(() => '');
        let errMsg = `Failed (${res.status})`;
        try {
          const errData = JSON.parse(errText);
          errMsg = errData.error || errMsg;
        } catch {
          if (errText) errMsg = errText.slice(0, 200);
        }
        setError(errMsg);
        setIsRunning(false);
        return;
      }

      // It's now an SSE stream
      const reader = res.body?.getReader();
      if (!reader) {
        setError('No response body');
        setIsRunning(false);
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') break;

          try {
            const parsed = JSON.parse(data);
            handleSSEEvent(parsed);
          } catch {
            // skip malformed
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err.message || 'Failed to connect to backend');
      }
    } finally {
      setIsRunning(false);
      abortRef.current = null;
    }
  };

  const stopTraining = async () => {
    // Abort the fetch stream
    abortRef.current?.abort();

    // Also tell the backend to stop
    const backendUrl = getStoredBackendUrl();
    const apiKey = getStoredApiKey();
    try {
      await fetch(`${backendUrl}/api/train/rlaif/stop`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { 'X-API-Key': apiKey } : {}),
        },
      });
    } catch {
      // Best effort
    }

    setIsRunning(false);
  };

  const handleSSEEvent = (event: any) => {
    switch (event.type) {
      case 'session_started':
        setSessionInfo({
          evaluatorModel: event.evaluatorModel,
          resolvedModel: event.resolvedModel,
          targetModel: event.targetModel,
          iterations: event.iterations,
        });
        break;

      case 'iteration_start':
        setIterations((prev) => [
          ...prev,
          {
            iteration: event.iteration,
            prompt: '',
            response: '',
            score: null,
            status: event.phase || 'generating_prompt',
          },
        ]);
        break;

      case 'prompt_generated':
        setIterations((prev) =>
          prev.map((it) =>
            it.iteration === event.iteration
              ? { ...it, prompt: event.prompt, status: 'getting_response' }
              : it
          )
        );
        break;

      case 'phase':
        setIterations((prev) =>
          prev.map((it) =>
            it.iteration === event.iteration
              ? { ...it, status: event.phase }
              : it
          )
        );
        break;

      case 'response_received':
        setIterations((prev) =>
          prev.map((it) =>
            it.iteration === event.iteration
              ? { ...it, response: event.response, status: 'evaluating' }
              : it
          )
        );
        break;

      case 'evaluation_complete':
        setIterations((prev) =>
          prev.map((it) =>
            it.iteration === event.iteration
              ? { ...it, score: event.score, status: 'complete' }
              : it
          )
        );
        break;

      case 'iteration_error':
        setIterations((prev) =>
          prev.map((it) =>
            it.iteration === event.iteration
              ? { ...it, status: 'error', error: event.error }
              : it
          )
        );
        break;

      case 'training_complete':
        setSummary({
          totalEvaluations: event.totalEvaluations,
          averageScore: event.averageScore,
          goodPatterns: event.goodPatterns,
        });
        setIsComplete(true);
        break;

      case 'stopped':
        setIsComplete(true);
        break;

      case 'error':
        setError(event.content);
        break;
    }
  };

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

  const completedCount = iterations.filter((it) => it.status === 'complete').length;
  const totalIterations = sessionInfo?.iterations || 0;
  const progressPercent = totalIterations > 0
    ? Math.min(100, Math.round((completedCount / totalIterations) * 100))
    : 0;

  const avgScore = completedCount > 0
    ? Math.round(
        (iterations
          .filter((it) => it.score)
          .reduce((sum, it) => sum + (it.score?.overall || 0), 0) /
          completedCount) *
          10
      ) / 10
    : 0;

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
          <Activity className="w-5 h-5 text-[#a78bfa]" />
          <h1 className="text-gray-200 font-semibold">RLAIF Automated Training: {workflow.name}</h1>
        </div>
        <div className="ml-auto flex items-center gap-3">
          {isRunning && (
            <>
              <div className="text-xs text-gray-400">
                {completedCount} / {totalIterations}
              </div>
              <div className="w-32 h-2 bg-[#1a1a24] rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${progressPercent}%`,
                    backgroundColor: '#a78bfa',
                  }}
                />
              </div>
            </>
          )}
          {isComplete && (
            <span className="text-xs text-green-400 flex items-center gap-1">
              <CheckCircle className="w-3.5 h-3.5" /> Complete
            </span>
          )}
        </div>
      </header>

      {/* Session Info Bar */}
      {sessionInfo && (
        <div className="bg-[#a78bfa]/10 border-b border-[#a78bfa]/20 px-6 py-3 flex items-center gap-6 text-sm">
          <span className="text-[#a78bfa]">
            Evaluator: <span className="text-gray-300">{sessionInfo.resolvedModel}</span>
          </span>
          <span className="text-[#a78bfa]">
            Target: <span className="text-gray-300">{sessionInfo.targetModel}</span>
          </span>
          {completedCount > 0 && (
            <span className="text-[#a78bfa]">
              Avg Score:{' '}
              <span className={`font-semibold ${avgScore >= 7 ? 'text-green-400' : avgScore >= 4 ? 'text-yellow-400' : 'text-red-400'}`}>
                {avgScore}/10
              </span>
            </span>
          )}
        </div>
      )}

      {/* Summary Banner */}
      {isComplete && summary && (
        <div className="bg-[#22c55e]/10 border-b border-[#22c55e]/20 px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-6 text-sm">
            <span className="text-green-400 flex items-center gap-1.5">
              <BarChart3 className="w-4 h-4" />
              {summary.totalEvaluations} evaluations completed
            </span>
            <span className="text-green-400">
              Avg Score: <span className="font-bold">{summary.averageScore}/10</span>
            </span>
            <span className="text-green-400">
              {summary.goodPatterns} high-quality patterns stored
            </span>
          </div>
          <button
            onClick={() => navigate(`/editor/${id}`)}
            className="px-4 py-1.5 bg-[#22c55e] text-[#0a0a0f] rounded-lg text-sm font-medium hover:bg-[#16a34a] transition-colors"
          >
            Back to Editor
          </button>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {!isRunning && iterations.length === 0 && !isComplete ? (
          // Start screen
          <div className="flex flex-col items-center justify-center h-full">
            <div className="max-w-md text-center">
              <div className="w-20 h-20 rounded-2xl bg-[#a78bfa]/10 flex items-center justify-center mx-auto mb-6">
                <Zap className="w-10 h-10 text-[#a78bfa]" />
              </div>
              <h2 className="text-xl font-semibold text-gray-200 mb-3">Automated AI Evaluation</h2>
              <p className="text-gray-400 mb-2 text-sm">
                RLAIF runs fully automatically — no human prompting needed.
              </p>
              <p className="text-gray-500 text-xs mb-8">
                The evaluator AI will generate diverse test prompts, send them to your model,
                and score each response on helpfulness, accuracy, and safety.
                High-quality response patterns are stored to improve future outputs.
              </p>
              <button
                onClick={startTraining}
                className="px-8 py-3 bg-[#a78bfa] text-white rounded-lg font-medium hover:bg-[#9061f9] transition-colors flex items-center gap-2 mx-auto"
              >
                <Play className="w-5 h-5" />
                Start Automated Training
              </button>
            </div>
          </div>
        ) : error && !isRunning && iterations.length === 0 ? (
          // Error screen
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <p className="text-red-400 mb-2">{error}</p>
            <button onClick={startTraining} className="text-[#a78bfa] hover:underline text-sm">
              Retry
            </button>
          </div>
        ) : (
          // Live iteration log
          <div className="space-y-4 max-w-4xl mx-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-gray-300 font-semibold text-lg">Automated Evaluation Log</h2>
              {isRunning && (
                <button
                  onClick={stopTraining}
                  className="px-4 py-2 bg-red-500/10 text-red-400 border border-red-500/20 rounded-lg text-sm font-medium hover:bg-red-500/20 transition-colors flex items-center gap-2"
                >
                  <Square className="w-4 h-4" />
                  Stop Training
                </button>
              )}
              {!isRunning && isComplete && (
                <button
                  onClick={startTraining}
                  className="px-4 py-2 bg-[#a78bfa]/10 text-[#a78bfa] border border-[#a78bfa]/20 rounded-lg text-sm font-medium hover:bg-[#a78bfa]/20 transition-colors flex items-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Run Again
                </button>
              )}
            </div>

            {iterations.map((entry) => (
              <div
                key={entry.iteration}
                className={`bg-[#12121a] border rounded-lg p-4 transition-all ${
                  entry.status === 'complete'
                    ? 'border-[#22222e]'
                    : entry.status === 'error'
                    ? 'border-red-500/30'
                    : 'border-[#a78bfa]/30'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs text-gray-500 flex items-center gap-2">
                    <span className="font-medium text-gray-400">Iteration #{entry.iteration}</span>
                    <PhaseIndicator phase={entry.status} />
                  </span>
                  {entry.score && (
                    <div className="flex gap-2">
                      <ScoreBadge label="Help" value={entry.score.helpfulness} />
                      <ScoreBadge label="Acc" value={entry.score.accuracy} />
                      <ScoreBadge label="Safe" value={entry.score.safety} />
                      <span
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-bold ${
                          entry.score.overall >= 8
                            ? 'text-green-400 bg-green-400/15'
                            : entry.score.overall >= 5
                            ? 'text-yellow-400 bg-yellow-400/15'
                            : 'text-red-400 bg-red-400/15'
                        }`}
                      >
                        Overall: {entry.score.overall}/10
                      </span>
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  {entry.prompt && (
                    <div>
                      <span className="text-xs text-[#a78bfa] font-medium flex items-center gap-1">
                        <Zap className="w-3 h-3" /> AI-Generated Prompt:
                      </span>
                      <p className="text-sm text-gray-300 mt-0.5 whitespace-pre-wrap">{entry.prompt}</p>
                    </div>
                  )}
                  {entry.response && (
                    <div>
                      <span className="text-xs text-[#00d4ff] font-medium flex items-center gap-1">
                        <Bot className="w-3 h-3" /> Model Response:
                      </span>
                      <p className="text-sm text-gray-400 mt-0.5 whitespace-pre-wrap">
                        {entry.response.length > 500
                          ? entry.response.slice(0, 500) + '...'
                          : entry.response}
                      </p>
                    </div>
                  )}
                  {entry.error && (
                    <p className="text-xs text-red-400 mt-1">{entry.error}</p>
                  )}
                </div>
              </div>
            ))}

            {isRunning && iterations.length > 0 && (
              <div className="flex items-center justify-center py-4">
                <div className="animate-spin w-6 h-6 border-2 border-[#a78bfa] border-t-transparent rounded-full" />
              </div>
            )}

            <div ref={logEndRef} />
          </div>
        )}
      </div>
    </div>
  );
};
