import React, { useEffect, useRef, useState } from 'react';
import { X, CheckCircle, XCircle, Loader2, Zap } from 'lucide-react';
import { pollJobStatus, streamJobLogs } from '../services/api';
import { useWorkflowsStore } from '../store/workflowsStore';

interface Props {
  jobId: string;
  workflowId: string;
  onClose: () => void;
}

type Phase = 'submitting' | 'running' | 'done' | 'error';

export const TrainingModal: React.FC<Props> = ({ jobId, workflowId, onClose }) => {
  const [phase, setPhase]       = useState<Phase>('submitting');
  const [logs, setLogs]         = useState<string[]>([]);
  const [elapsed, setElapsed]   = useState(0);
  const [modelName, setModelName] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string>('');

  const addTrainedModel = useWorkflowsStore((s) => s.addTrainedModel);
  const logEndRef       = useRef<HTMLDivElement>(null);
  const startRef        = useRef<number>(Date.now());

  // Elapsed timer
  useEffect(() => {
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Start streaming + polling
  useEffect(() => {
    setPhase('running');

    const cleanup = streamJobLogs(
      jobId,
      (line) => setLogs((prev) => [...prev, line]),
      (result) => {
        const r = result as Record<string, unknown> | undefined;
        const name = (r?.output_name as string) || (r?.model_name as string) || jobId.slice(0, 8);
        setModelName(name);
        addTrainedModel(workflowId, name);
        setPhase('done');
      }
    );

    // Fallback polling every 5s in case SSE doesn't fire
    const pollId = setInterval(async () => {
      const status = await pollJobStatus(jobId);
      if (status.status === 'complete') {
        cleanup();
        clearInterval(pollId);
        const r = status.result as Record<string, unknown> | undefined;
        const name = (r?.output_name as string) || jobId.slice(0, 8);
        setModelName(name);
        addTrainedModel(workflowId, name);
        setPhase('done');
      } else if (status.status === 'error') {
        cleanup();
        clearInterval(pollId);
        setErrorMsg((status as any).error || 'Training failed');
        setPhase('error');
      }
    }, 5000);

    return () => {
      cleanup();
      clearInterval(pollId);
    };
  }, [jobId]);

  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="w-full max-w-2xl bg-[#12121a] rounded-xl border border-[#22222e] shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center gap-3 px-6 py-4 border-b border-[#22222e]">
          {phase === 'running' || phase === 'submitting'
            ? <Loader2 className="w-5 h-5 text-[#00d4ff] animate-spin" />
            : phase === 'done'
            ? <CheckCircle className="w-5 h-5 text-[#22c55e]" />
            : <XCircle className="w-5 h-5 text-[#ef4444]" />
          }
          <h2 className="text-gray-200 font-semibold flex-1">
            {phase === 'submitting' && 'Submitting training job…'}
            {phase === 'running'    && 'Training in progress'}
            {phase === 'done'       && 'Training complete!'}
            {phase === 'error'      && 'Training failed'}
          </h2>
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Zap className="w-4 h-4" />
            <span>{fmt(elapsed)}</span>
          </div>
          {(phase === 'done' || phase === 'error') && (
            <button onClick={onClose} className="p-1 hover:bg-[#1a1a24] rounded-lg text-gray-400">
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Log pane */}
        <div className="flex-1 p-4 overflow-y-auto h-80 bg-[#0a0a0f] font-mono text-xs text-gray-300 space-y-0.5">
          {logs.length === 0 && phase !== 'done' && phase !== 'error' && (
            <span className="text-gray-500">Waiting for GPU container to start…</span>
          )}
          {logs.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap leading-5">{line}</div>
          ))}
          <div ref={logEndRef} />
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-[#22222e]">
          {phase === 'done' && (
            <div className="flex items-center justify-between">
              <div>
                <p className="text-[#22c55e] font-medium">Model saved: <span className="font-mono">{modelName}</span></p>
                <p className="text-gray-500 text-sm mt-0.5">Available in your training workflow's model list</p>
              </div>
              <button
                onClick={onClose}
                className="px-4 py-2 bg-[#22c55e] text-[#0a0a0f] rounded-lg font-medium hover:bg-[#16a34a] transition-colors"
              >
                Done
              </button>
            </div>
          )}
          {phase === 'error' && (
            <div className="flex items-center justify-between">
              <p className="text-[#ef4444] text-sm">{errorMsg || 'An error occurred during training.'}</p>
              <button
                onClick={onClose}
                className="px-4 py-2 bg-[#1a1a24] text-gray-300 rounded-lg hover:bg-[#22222e] transition-colors"
              >
                Close
              </button>
            </div>
          )}
          {(phase === 'running' || phase === 'submitting') && (
            <p className="text-gray-500 text-sm">
              Training is running on a Modal GPU. This window can be left open to stream logs.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
