import React, { useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Upload, Loader2, Zap, Copy, CheckCircle } from 'lucide-react';
import { useWorkflowsStore } from '../store/workflowsStore';
import { runInference } from '../services/api';
import { buildInferPipelineFromTraining, buildPipelineSpec } from '../utils/pipelineBuilder';
import type { InputNodeType } from '../store/workflowStore';

import { getStoredBackendUrl } from '../components/SettingsModal';

const getWorkerBase = () => getStoredBackendUrl();

// ── Result display ─────────────────────────────────────────────────────────────

const ResultPane: React.FC<{
  result: unknown;
  llmResponse?: string;
  loading: boolean;
}> = ({ result, llmResponse, loading }) => {
  const [copied, setCopied] = useState(false);

  const resultStr = result !== null && result !== undefined
    ? (typeof result === 'string' ? result : JSON.stringify(result, null, 2))
    : null;

  const copy = async () => {
    if (!resultStr) return;
    await navigator.clipboard.writeText(resultStr);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Output</h3>
        {resultStr && (
          <button
            onClick={copy}
            className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-200 transition-colors"
          >
            {copied
              ? <CheckCircle className="w-3.5 h-3.5 text-[#22c55e]" />
              : <Copy className="w-3.5 h-3.5" />
            }
            {copied ? 'Copied' : 'Copy'}
          </button>
        )}
      </div>

      {loading ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-3 text-gray-500">
          <Loader2 className="w-8 h-8 animate-spin text-[#6366f1]" />
          <p className="text-sm">Running inference…</p>
        </div>
      ) : resultStr ? (
        <div className="flex-1 flex flex-col gap-4 overflow-y-auto">
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">Model Result</p>
            <pre className="bg-[#0a0a0f] border border-[#2a2a38] rounded-lg p-4 text-xs text-gray-200 font-mono overflow-auto whitespace-pre-wrap">
              {resultStr}
            </pre>
          </div>
          {llmResponse && (
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">LLM Response</p>
              <div className="bg-[#12121a] border border-[#2a2a38] rounded-lg p-4 text-sm text-gray-200 leading-relaxed">
                {llmResponse}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center text-gray-600">
          <Zap className="w-10 h-10 mb-3 opacity-30" />
          <p className="text-sm">Submit input to see results</p>
        </div>
      )}
    </div>
  );
};

// ── Text input pane ────────────────────────────────────────────────────────────

const TextInputPane: React.FC<{
  onSubmit: (data: string) => void;
  loading: boolean;
}> = ({ onSubmit, loading }) => {
  const [text, setText] = useState('');
  return (
    <div className="flex flex-col gap-3 h-full">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste your text here…"
        className="flex-1 bg-[#0a0a0f] border border-[#2a2a38] rounded-lg p-4 text-sm text-gray-200 font-mono resize-none focus:outline-none focus:border-[#6366f1] transition-colors"
      />
      <button
        onClick={() => text.trim() && onSubmit(text)}
        disabled={loading || !text.trim()}
        className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg font-medium bg-[#6366f1] text-white hover:bg-[#4f46e5] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
        Run Inference
      </button>
    </div>
  );
};

// ── File input pane ────────────────────────────────────────────────────────────

const FileInputPane: React.FC<{
  accept: string;
  label: string;
  multiple?: boolean;
  onSubmit: (files: File | File[]) => void;
  loading: boolean;
}> = ({ accept, label, multiple = false, onSubmit, loading }) => {
  const [files, setFiles] = useState<File[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFiles = (fl: FileList | null) => {
    if (!fl) return;
    setFiles(Array.from(fl));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="flex flex-col gap-3 h-full">
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
        className="flex-1 flex flex-col items-center justify-center gap-3 border-2 border-dashed border-[#2a2a38] rounded-lg cursor-pointer hover:border-[#6366f1] transition-colors"
      >
        <input
          ref={fileRef}
          type="file"
          accept={accept}
          multiple={multiple}
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <Upload className="w-8 h-8 text-gray-600" />
        {files.length > 0 ? (
          <p className="text-sm text-gray-300">{files.map((f) => f.name).join(', ')}</p>
        ) : (
          <p className="text-sm text-gray-500">{label}</p>
        )}
      </div>
      <button
        onClick={() => {
          if (!files.length) return;
          onSubmit(multiple ? files : files[0]);
        }}
        disabled={loading || files.length === 0}
        className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg font-medium bg-[#6366f1] text-white hover:bg-[#4f46e5] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
        Run Inference
      </button>
    </div>
  );
};

// ── Page ───────────────────────────────────────────────────────────────────────

export const InferencePage: React.FC = () => {
  const { id: workflowId } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const workflows = useWorkflowsStore((s) => s.workflows);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<unknown>(null);
  const [llmResponse, setLlmResponse] = useState<string | undefined>();
  const [error, setError] = useState<string | null>(null);

  const workflow = workflows.find((w) => w.id === workflowId);

  if (!workflow || !workflowId) {
    return (
      <div className="h-screen w-screen flex flex-col items-center justify-center bg-[#0a0a0f]">
        <p className="text-gray-400 mb-4">Workflow not found</p>
        <button onClick={() => navigate('/')} className="text-[#6366f1] hover:underline">
          Go home
        </button>
      </div>
    );
  }

  // Detect active input type from workflow nodes
  const inputTypes: InputNodeType[] = ['textInput', 'imageInput', 'audioInput', 'spreadsheetInput'];
  const activeInput = workflow.nodes
    .map((n) => n.data.type as string)
    .find((t) => inputTypes.includes(t as InputNodeType)) as InputNodeType | undefined;

  // For deployment workflows, use source training ID so Actian collection name matches
  const inferWorkflowId = workflow.sourceTrainingId || workflowId;
  const endpointUrl = `${getWorkerBase()}/api/deploy/${inferWorkflowId}`;

  // Extract LLM node config if present (sent to Worker for post-processing)
  const llmNode = workflow.nodes.find((n) => n.data.type === 'llmNode');
  const llmConfig = llmNode ? (llmNode.data.parameters as Record<string, unknown>) : undefined;

  const submit = async (inputData: string | File | File[]) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setLlmResponse(undefined);
    try {
      const edges = workflow.edges || [];
      const trainedModel = workflow.trainedModels?.[workflow.trainedModels.length - 1];

      let pipelineSpec;
      if (trainedModel) {
        // Has a trained model → transform training pipeline for inference
        pipelineSpec = buildInferPipelineFromTraining(workflow.nodes, edges, trainedModel);
      } else {
        // No trained model (standalone deployment) → build simple pass-through pipeline
        // Filter out llmNode (handled by Worker) and saveModel
        const pipelineNodes = workflow.nodes.filter(
          (n) => n.data.type !== 'llmNode' && n.data.type !== 'saveModel'
        );
        const excludeIds = new Set(
          workflow.nodes.filter((n) => n.data.type === 'llmNode' || n.data.type === 'saveModel').map((n) => n.id)
        );
        const pipelineEdges = edges.filter((e) => !excludeIds.has(e.source) && !excludeIds.has(e.target));
        const spec = buildPipelineSpec(pipelineNodes, pipelineEdges, 'infer');

        // Append infer_output if not already present
        if (!spec.nodes.some((n) => n.type === 'infer_output')) {
          const hasOutgoing = new Set(spec.edges.map((e) => e.from));
          const lastNode = spec.nodes.filter((n) => !hasOutgoing.has(n.id)).pop()
            ?? spec.nodes[spec.nodes.length - 1];
          const inferOutputId = '__infer_output__';
          spec.nodes.push({ id: inferOutputId, type: 'infer_output', params: {} });
          if (lastNode) spec.edges.push({ from: lastNode.id, to: inferOutputId });
        }
        pipelineSpec = spec;
      }

      const res = await runInference(inferWorkflowId, inputData, pipelineSpec, llmConfig);
      setResult(res.result);
      setLlmResponse(res.llmResponse);
    } catch (err: any) {
      setError(err.message || 'Inference failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0a0a0f]">
      {/* Header */}
      <header className="h-12 bg-[#12121a] border-b border-[#22222e] flex items-center px-4 gap-4 shrink-0">
        <button
          onClick={() => navigate(`/editor/${workflowId}`)}
          className="p-2 hover:bg-[#1a1a24] rounded-lg text-gray-400 hover:text-gray-200 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-[#6366f1]" />
          <h1 className="text-gray-200 font-semibold">{workflow.name}</h1>
          <span className="text-xs px-2 py-0.5 rounded-full bg-[#6366f1]/15 text-[#6366f1] border border-[#6366f1]/30 font-medium">
            Inference
          </span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-xs text-gray-600 font-mono truncate max-w-64">
            POST {endpointUrl}
          </span>
        </div>
      </header>

      {/* Main split-pane */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Input */}
        <div className="w-1/2 flex flex-col border-r border-[#22222e] p-6">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
            Input
            {activeInput && (
              <span className="ml-2 text-[10px] normal-case text-gray-600">
                ({activeInput.replace('Input', '')})
              </span>
            )}
          </h3>

          <div className="flex-1 min-h-0">
            {activeInput === 'textInput' ? (
              <TextInputPane onSubmit={submit} loading={loading} />
            ) : activeInput === 'imageInput' ? (
              <FileInputPane
                accept=".jpg,.jpeg,.png,.bmp,.webp"
                label="Drop an image or click to browse"
                onSubmit={submit}
                loading={loading}
              />
            ) : activeInput === 'audioInput' ? (
              <FileInputPane
                accept=".wav,.mp3,.flac,.ogg,.m4a"
                label="Drop an audio file or click to browse"
                onSubmit={submit}
                loading={loading}
              />
            ) : activeInput === 'spreadsheetInput' ? (
              <FileInputPane
                accept=".csv,.xlsx,.json"
                label="Drop a CSV/spreadsheet or click to browse"
                onSubmit={submit}
                loading={loading}
              />
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-gray-600 gap-3">
                <p className="text-sm">No input node found in this workflow.</p>
                <button
                  onClick={() => navigate(`/editor/${workflowId}`)}
                  className="text-xs text-[#6366f1] hover:underline"
                >
                  Go to editor →
                </button>
              </div>
            )}
          </div>

          {error && (
            <p className="mt-3 text-xs text-red-400 bg-red-400/10 border border-red-400/20 rounded-md px-3 py-2">
              {error}
            </p>
          )}
        </div>

        {/* Right: Output */}
        <div className="w-1/2 flex flex-col p-6">
          <ResultPane result={result} llmResponse={llmResponse} loading={loading} />
        </div>
      </div>
    </div>
  );
};
