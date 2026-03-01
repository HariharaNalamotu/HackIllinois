import React, { useRef, useCallback, useState } from 'react';
import { Settings, Trash2, Upload, FolderOpen, File, X, Info, CheckCircle, Copy, Globe } from 'lucide-react';
import { useParams } from 'react-router-dom';
import { useWorkflowStore, NodeType } from '../store/workflowStore';
import { useWorkflowsStore } from '../store/workflowsStore';
import {
  nodeDefinitions,
  EMBEDDING_MODELS,
  IMAGE_CLASSIFIER_MODELS,
  OBJECT_DETECT_MODELS,
  AUDIO_MODELS,
  CHUNKING_METHODS,
  TABULAR_MODEL_TYPES,
  AUDIO_TASKS,
  BOUNDING_BOX_FORMATS,
  FINE_TUNE_METHODS,
  POOLING_OPTIONS,
  LLM_PROVIDERS,
} from '../types/nodes';

import { getStoredBackendUrl } from './SettingsModal';

const getWorkerBase = () => getStoredBackendUrl();

// ── Reusable primitives ───────────────────────────────────────────────────────

const SelectField: React.FC<{
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}> = ({ label, value, options, onChange }) => (
  <div className="space-y-1">
    <label className="text-xs text-gray-400 uppercase tracking-wide">{label}</label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-[#00d4ff] transition-colors"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  </div>
);

const TextField: React.FC<{
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  multiline?: boolean;
}> = ({ label, value, onChange, placeholder, multiline }) => (
  <div className="space-y-1">
    <label className="text-xs text-gray-400 uppercase tracking-wide">{label}</label>
    {multiline ? (
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={3}
        className="w-full bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-[#00d4ff] transition-colors resize-none"
      />
    ) : (
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-[#00d4ff] transition-colors"
      />
    )}
  </div>
);

const NumberField: React.FC<{
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}> = ({ label, value, onChange, min, max, step = 1 }) => (
  <div className="space-y-1">
    <label className="text-xs text-gray-400 uppercase tracking-wide">{label}</label>
    <input
      type="number"
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
      min={min}
      max={max}
      step={step}
      className="w-full bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-[#00d4ff] transition-colors"
    />
  </div>
);

const Toggle: React.FC<{
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}> = ({ label, checked, onChange }) => (
  <div className="flex items-center justify-between">
    <span className="text-sm text-gray-300">{label}</span>
    <button
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        checked ? 'bg-[#00d4ff]' : 'bg-[#2a2a38]'
      }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          checked ? 'translate-x-6' : 'translate-x-1'
        }`}
      />
    </button>
  </div>
);

const SourceToggle: React.FC<{ value: string; onChange: (v: string) => void }> = ({
  value,
  onChange,
}) => (
  <div className="space-y-1">
    <label className="text-xs text-gray-400 uppercase tracking-wide">Data Source</label>
    <div className="flex rounded-md overflow-hidden border border-[#2a2a38]">
      {['upload', 'api'].map((src) => (
        <button
          key={src}
          onClick={() => onChange(src)}
          className={`flex-1 py-2 text-sm font-medium transition-colors ${
            value === src
              ? 'bg-[#00d4ff] text-[#0a0a0f]'
              : 'bg-[#1a1a24] text-gray-400 hover:text-gray-200'
          }`}
        >
          {src === 'upload' ? 'Upload' : 'API'}
        </button>
      ))}
    </div>
  </div>
);

const FolderUploadButton: React.FC<{ onFiles: (files: FileList) => void }> = ({ onFiles }) => {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={ref}
        type="file"
        // @ts-ignore
        webkitdirectory=""
        className="hidden"
        onChange={(e) => e.target.files && onFiles(e.target.files)}
      />
      <button
        onClick={() => ref.current?.click()}
        className="w-full flex items-center justify-center gap-2 bg-[#1a1a24] border border-dashed border-[#2a2a38] rounded-md px-3 py-3 text-sm text-gray-400 hover:border-[#00d4ff] hover:text-[#00d4ff] transition-colors"
      >
        <FolderOpen className="w-4 h-4" />
        Browse Folder
      </button>
    </>
  );
};

/** Convert a FileList to File[] for storage in Zustand. */
function fileListToArray(fl: FileList): globalThis.File[] {
  const arr: globalThis.File[] = [];
  for (let i = 0; i < fl.length; i++) arr.push(fl[i]);
  return arr;
}

/** Display label for uploadedFiles (could be File[], FileList, or legacy string). */
function filesLabel(v: unknown): string {
  if (Array.isArray(v)) return `${v.length} file(s) selected`;
  if (v instanceof FileList) return `${v.length} file(s) selected`;
  if (typeof v === 'string') return v;
  return 'File selected';
}

const Section: React.FC<{ title: string }> = ({ title }) => (
  <div className="border-t border-[#22222e] pt-4">
    <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">{title}</p>
  </div>
);

const OutputNameField: React.FC<{ value: string; onChange: (v: string) => void }> = ({
  value,
  onChange,
}) => (
  <TextField
    label="Output Model Name"
    value={value}
    onChange={onChange}
    placeholder="e.g., my-model-v1"
  />
);

// ── Image format auto-detection ───────────────────────────────────────────────

const detectImageFormat = (files: FileList): 'imagefolder' | 'boundingbox' | 'unknown' => {
  const names = Array.from(files).map((f) => (f as any).webkitRelativePath || f.name);
  const annotationExts = ['.json', '.xml', '.txt', '.csv'];
  const hasAnnotation = names.some((n) =>
    annotationExts.some((ext) => n.toLowerCase().endsWith(ext))
  );
  if (hasAnnotation) return 'boundingbox';
  const dirs = new Set(names.map((n) => n.split('/').slice(0, -1).join('/')).filter(Boolean));
  if (dirs.size > 1) return 'imagefolder';
  return 'unknown';
};

// ── Node-specific forms ───────────────────────────────────────────────────────

const TextInputForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  const fileRef = useRef<HTMLInputElement>(null);

  return (
    <div className="space-y-4">
      <SourceToggle
        value={p.dataSource as string}
        onChange={(v) => update(node.id, { dataSource: v })}
      />
      {p.dataSource === 'api' ? (
        <TextField
          label="API URL"
          value={p.apiUrl as string}
          onChange={(v) => update(node.id, { apiUrl: v })}
          placeholder="https://..."
        />
      ) : (
        <div className="space-y-2">
          <label className="text-xs text-gray-400 uppercase tracking-wide">Files</label>
          {p.uploadedFiles ? (
            <div className="flex items-center gap-2 bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2">
              <File className="w-4 h-4 text-[#00d4ff]" />
              <span className="text-sm text-gray-300 flex-1 truncate">
                {filesLabel(p.uploadedFiles)}
              </span>
              <button
                onClick={() => update(node.id, { uploadedFiles: null })}
                className="text-gray-500 hover:text-red-400"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".txt,.pdf,.md,.docx,.csv,.json"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.length)
                    update(node.id, {
                      uploadedFiles: fileListToArray(e.target.files),
                    });
                }}
              />
              <button
                onClick={() => fileRef.current?.click()}
                className="w-full flex items-center justify-center gap-2 bg-[#1a1a24] border border-dashed border-[#2a2a38] rounded-md px-3 py-3 text-sm text-gray-400 hover:border-[#00d4ff] hover:text-[#00d4ff] transition-colors"
              >
                <Upload className="w-4 h-4" />
                Upload Files (.txt, .pdf, .md, .docx)
              </button>
              <FolderUploadButton
                onFiles={(files) =>
                  update(node.id, { uploadedFiles: fileListToArray(files) })
                }
              />
            </>
          )}
        </div>
      )}
    </div>
  );
};

const formatBadgeMap: Record<string, { label: string; color: string }> = {
  imagefolder: { label: 'ImageFolder (classification)', color: '#a855f7' },
  boundingbox: { label: 'Bounding Box (detection)', color: '#f472b6' },
  unknown: { label: 'Unknown — please override', color: '#6b7280' },
};

const ImageInputForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;

  const handleFiles = useCallback(
    (files: FileList) => {
      const detected = detectImageFormat(files);
      update(node.id, {
        uploadedFiles: fileListToArray(files),
        detectedFormat: detected,
        imageFormat: detected,
      });
    },
    [node.id, update]
  );

  const fmt = (p.imageFormat as string) || 'unknown';
  const badge = formatBadgeMap[fmt] ?? formatBadgeMap.unknown;

  return (
    <div className="space-y-4">
      <SourceToggle
        value={p.dataSource as string}
        onChange={(v) => update(node.id, { dataSource: v })}
      />
      {p.dataSource === 'api' ? (
        <TextField
          label="API URL"
          value={p.apiUrl as string}
          onChange={(v) => update(node.id, { apiUrl: v })}
          placeholder="https://..."
        />
      ) : (
        <div className="space-y-3">
          {p.uploadedFiles ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2 bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2">
                <File className="w-4 h-4" style={{ color: badge.color }} />
                <span className="text-sm text-gray-300 flex-1 truncate">
                  {filesLabel(p.uploadedFiles)}
                </span>
                <button
                  onClick={() =>
                    update(node.id, {
                      uploadedFiles: null,
                      detectedFormat: null,
                      imageFormat: 'unknown',
                    })
                  }
                  className="text-gray-500 hover:text-red-400"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="flex items-center gap-2 px-1">
                <span className="text-xs text-gray-400">Auto-detected:</span>
                <span
                  className="text-xs font-medium px-2 py-0.5 rounded-full"
                  style={{ backgroundColor: badge.color + '20', color: badge.color }}
                >
                  {badge.label}
                </span>
              </div>
            </div>
          ) : (
            <FolderUploadButton onFiles={handleFiles} />
          )}
          <div className="space-y-1">
            <label className="text-xs text-gray-400 uppercase tracking-wide">
              Override Format
            </label>
            <select
              value={fmt}
              onChange={(e) => update(node.id, { imageFormat: e.target.value })}
              className="w-full bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-[#00d4ff] transition-colors"
            >
              <option value="unknown">Auto-detected</option>
              <option value="imagefolder">ImageFolder (classification)</option>
              <option value="boundingbox">Bounding Box (detection)</option>
            </select>
          </div>
        </div>
      )}
    </div>
  );
};

const AudioInputForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  const fileRef = useRef<HTMLInputElement>(null);

  return (
    <div className="space-y-4">
      <SourceToggle
        value={p.dataSource as string}
        onChange={(v) => update(node.id, { dataSource: v })}
      />
      {p.dataSource === 'api' ? (
        <TextField
          label="API URL"
          value={p.apiUrl as string}
          onChange={(v) => update(node.id, { apiUrl: v })}
          placeholder="https://..."
        />
      ) : (
        <div className="space-y-2">
          {p.uploadedFiles ? (
            <div className="flex items-center gap-2 bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2">
              <File className="w-4 h-4 text-[#22c55e]" />
              <span className="text-sm text-gray-300 flex-1 truncate">
                {filesLabel(p.uploadedFiles)}
              </span>
              <button
                onClick={() => update(node.id, { uploadedFiles: null })}
                className="text-gray-500 hover:text-red-400"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".wav,.mp3,.flac,.ogg,.m4a"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.length)
                    update(node.id, {
                      uploadedFiles: fileListToArray(e.target.files),
                    });
                }}
              />
              <button
                onClick={() => fileRef.current?.click()}
                className="w-full flex items-center justify-center gap-2 bg-[#1a1a24] border border-dashed border-[#2a2a38] rounded-md px-3 py-3 text-sm text-gray-400 hover:border-[#22c55e] hover:text-[#22c55e] transition-colors"
              >
                <Upload className="w-4 h-4" />
                Upload Audio (.wav, .mp3, .flac)
              </button>
              <FolderUploadButton
                onFiles={(files) =>
                  update(node.id, { uploadedFiles: fileListToArray(files) })
                }
              />
            </>
          )}
        </div>
      )}
    </div>
  );
};

const SpreadsheetInputForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  const fileRef = useRef<HTMLInputElement>(null);

  return (
    <div className="space-y-4">
      <SourceToggle
        value={p.dataSource as string}
        onChange={(v) => update(node.id, { dataSource: v })}
      />
      {p.dataSource === 'api' ? (
        <TextField
          label="API URL"
          value={p.apiUrl as string}
          onChange={(v) => update(node.id, { apiUrl: v })}
          placeholder="https://..."
        />
      ) : (
        <div className="space-y-2">
          {p.uploadedFiles ? (
            <div className="flex items-center gap-2 bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2">
              <File className="w-4 h-4 text-[#f97316]" />
              <span className="text-sm text-gray-300 flex-1 truncate">
                {filesLabel(p.uploadedFiles)}
              </span>
              <button
                onClick={() => update(node.id, { uploadedFiles: null })}
                className="text-gray-500 hover:text-red-400"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".csv,.xlsx,.xls,.json"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files?.length)
                    update(node.id, {
                      uploadedFiles: fileListToArray(e.target.files),
                    });
                }}
              />
              <button
                onClick={() => fileRef.current?.click()}
                className="w-full flex items-center justify-center gap-2 bg-[#1a1a24] border border-dashed border-[#2a2a38] rounded-md px-3 py-3 text-sm text-gray-400 hover:border-[#f97316] hover:text-[#f97316] transition-colors"
              >
                <Upload className="w-4 h-4" />
                Upload Spreadsheet (.csv, .xlsx, .json)
              </button>
            </>
          )}
        </div>
      )}
      <TextField
        label="Target Column"
        value={p.targetColumn as string}
        onChange={(v) => update(node.id, { targetColumn: v })}
        placeholder="e.g., label, price, category"
      />
    </div>
  );
};

const ChunkNodeForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  const showSizeOpts = ['fixed_size', 'sliding_window'].includes(p.method as string);

  return (
    <div className="space-y-4">
      <SelectField
        label="Chunking Method"
        value={p.method as string}
        options={CHUNKING_METHODS}
        onChange={(v) => update(node.id, { method: v })}
      />
      {p.method === 'auto' && (
        <div className="flex items-start gap-2 bg-[#1a1a24] border border-[#2a2a38] rounded-md p-3">
          <Info className="w-4 h-4 text-[#00d4ff] mt-0.5 flex-shrink-0" />
          <p className="text-xs text-gray-400">
            AI will analyze your data and automatically select the best chunking strategy at
            runtime.
          </p>
        </div>
      )}
      {showSizeOpts && (
        <>
          <NumberField
            label="Chunk Size (tokens)"
            value={p.chunkSize as number}
            onChange={(v) => update(node.id, { chunkSize: v })}
            min={64}
            max={4096}
            step={64}
          />
          <NumberField
            label="Overlap (tokens)"
            value={p.overlap as number}
            onChange={(v) => update(node.id, { overlap: v })}
            min={0}
            max={512}
            step={16}
          />
        </>
      )}
    </div>
  );
};

const EmbeddingModelForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;

  return (
    <div className="space-y-4">
      <SelectField
        label="Base Embedding Model"
        value={p.model as string}
        options={EMBEDDING_MODELS}
        onChange={(v) => update(node.id, { model: v })}
      />
      <Section title="Fine-tuning" />
      <SelectField
        label="Training Method"
        value={p.method as string}
        options={FINE_TUNE_METHODS}
        onChange={(v) => update(node.id, { method: v })}
      />
      <NumberField
        label="Epochs"
        value={p.epochs as number}
        onChange={(v) => update(node.id, { epochs: v })}
        min={1}
        max={100}
      />
      <NumberField
        label="Learning Rate"
        value={p.learningRate as number}
        onChange={(v) => update(node.id, { learningRate: v })}
        min={1e-6}
        max={0.1}
        step={1e-5}
      />
      <OutputNameField
        value={p.outputName as string}
        onChange={(v) => update(node.id, { outputName: v })}
      />
    </div>
  );
};

const ImageClassifierForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;

  return (
    <div className="space-y-4">
      <SelectField
        label="Pretrained Model"
        value={p.baseModel as string}
        options={IMAGE_CLASSIFIER_MODELS}
        onChange={(v) => update(node.id, { baseModel: v })}
      />
      <Toggle
        label="Transfer Learning"
        checked={p.transfer as boolean}
        onChange={(v) => update(node.id, { transfer: v })}
      />
      <Section title="Training" />
      <NumberField
        label="Num Classes"
        value={p.numClasses as number}
        onChange={(v) => update(node.id, { numClasses: v })}
        min={2}
        max={1000}
      />
      <NumberField
        label="Epochs"
        value={p.epochs as number}
        onChange={(v) => update(node.id, { epochs: v })}
        min={1}
        max={500}
      />
      <NumberField
        label="Batch Size"
        value={p.batchSize as number}
        onChange={(v) => update(node.id, { batchSize: v })}
        min={1}
        max={512}
      />
      <NumberField
        label="Learning Rate"
        value={p.learningRate as number}
        onChange={(v) => update(node.id, { learningRate: v })}
        min={1e-6}
        max={0.1}
        step={1e-5}
      />
      <OutputNameField
        value={p.outputName as string}
        onChange={(v) => update(node.id, { outputName: v })}
      />
    </div>
  );
};

const ImageCNNForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;

  return (
    <div className="space-y-4">
      <Section title="Architecture" />
      <NumberField
        label="Num Layers"
        value={p.numLayers as number}
        onChange={(v) => update(node.id, { numLayers: v })}
        min={1}
        max={8}
      />
      <TextField
        label="Filters (comma-separated)"
        value={p.filters as string}
        onChange={(v) => update(node.id, { filters: v })}
        placeholder="32,64,128"
      />
      <NumberField
        label="Kernel Size"
        value={p.kernelSize as number}
        onChange={(v) => update(node.id, { kernelSize: v })}
        min={1}
        max={11}
        step={2}
      />
      <SelectField
        label="Pooling"
        value={p.pooling as string}
        options={POOLING_OPTIONS}
        onChange={(v) => update(node.id, { pooling: v })}
      />
      <NumberField
        label="Num Classes"
        value={p.numClasses as number}
        onChange={(v) => update(node.id, { numClasses: v })}
        min={2}
        max={1000}
      />
      <Section title="Training" />
      <NumberField
        label="Epochs"
        value={p.epochs as number}
        onChange={(v) => update(node.id, { epochs: v })}
        min={1}
        max={500}
      />
      <NumberField
        label="Batch Size"
        value={p.batchSize as number}
        onChange={(v) => update(node.id, { batchSize: v })}
        min={1}
        max={512}
      />
      <NumberField
        label="Learning Rate"
        value={p.learningRate as number}
        onChange={(v) => update(node.id, { learningRate: v })}
        min={1e-6}
        max={0.1}
        step={1e-5}
      />
      <OutputNameField
        value={p.outputName as string}
        onChange={(v) => update(node.id, { outputName: v })}
      />
    </div>
  );
};

const ImageCAEForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;

  return (
    <div className="space-y-4">
      <Section title="Architecture" />
      <NumberField
        label="Num Layers"
        value={p.numLayers as number}
        onChange={(v) => update(node.id, { numLayers: v })}
        min={1}
        max={8}
      />
      <TextField
        label="Filters (comma-separated)"
        value={p.filters as string}
        onChange={(v) => update(node.id, { filters: v })}
        placeholder="32,64,128"
      />
      <NumberField
        label="Latent Dimension"
        value={p.latentDim as number}
        onChange={(v) => update(node.id, { latentDim: v })}
        min={16}
        max={2048}
      />
      <Section title="Training" />
      <NumberField
        label="Epochs"
        value={p.epochs as number}
        onChange={(v) => update(node.id, { epochs: v })}
        min={1}
        max={500}
      />
      <NumberField
        label="Batch Size"
        value={p.batchSize as number}
        onChange={(v) => update(node.id, { batchSize: v })}
        min={1}
        max={512}
      />
      <NumberField
        label="Learning Rate"
        value={p.learningRate as number}
        onChange={(v) => update(node.id, { learningRate: v })}
        min={1e-6}
        max={0.1}
        step={1e-5}
      />
      <OutputNameField
        value={p.outputName as string}
        onChange={(v) => update(node.id, { outputName: v })}
      />
    </div>
  );
};

const ObjectDetectorForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;

  return (
    <div className="space-y-4">
      <SelectField
        label="Base Model"
        value={p.baseModel as string}
        options={OBJECT_DETECT_MODELS}
        onChange={(v) => update(node.id, { baseModel: v })}
      />
      <SelectField
        label="Annotation Format"
        value={p.inputFormat as string}
        options={BOUNDING_BOX_FORMATS}
        onChange={(v) => update(node.id, { inputFormat: v })}
      />
      <Section title="Training" />
      <NumberField
        label="Num Classes"
        value={p.numClasses as number}
        onChange={(v) => update(node.id, { numClasses: v })}
        min={1}
        max={1000}
      />
      <NumberField
        label="Epochs"
        value={p.epochs as number}
        onChange={(v) => update(node.id, { epochs: v })}
        min={1}
        max={500}
      />
      <NumberField
        label="Batch Size"
        value={p.batchSize as number}
        onChange={(v) => update(node.id, { batchSize: v })}
        min={1}
        max={64}
      />
      <NumberField
        label="Learning Rate"
        value={p.learningRate as number}
        onChange={(v) => update(node.id, { learningRate: v })}
        min={1e-6}
        max={0.1}
        step={1e-6}
      />
      <OutputNameField
        value={p.outputName as string}
        onChange={(v) => update(node.id, { outputName: v })}
      />
    </div>
  );
};

const AudioSpeechModelForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;

  return (
    <div className="space-y-4">
      <SelectField
        label="Task"
        value={p.task as string}
        options={AUDIO_TASKS}
        onChange={(v) => update(node.id, { task: v })}
      />
      <SelectField
        label="Base Model"
        value={p.baseModel as string}
        options={AUDIO_MODELS}
        onChange={(v) => update(node.id, { baseModel: v })}
      />
      <Section title="Training" />
      {p.task === 'classification' && (
        <NumberField
          label="Num Classes"
          value={p.numClasses as number}
          onChange={(v) => update(node.id, { numClasses: v })}
          min={2}
          max={1000}
        />
      )}
      <NumberField
        label="Epochs"
        value={p.epochs as number}
        onChange={(v) => update(node.id, { epochs: v })}
        min={1}
        max={100}
      />
      <NumberField
        label="Learning Rate"
        value={p.learningRate as number}
        onChange={(v) => update(node.id, { learningRate: v })}
        min={1e-6}
        max={0.1}
        step={1e-5}
      />
      <OutputNameField
        value={p.outputName as string}
        onChange={(v) => update(node.id, { outputName: v })}
      />
    </div>
  );
};

const AudioCNNForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;

  return (
    <div className="space-y-4">
      <Section title="Architecture" />
      <NumberField
        label="Num Layers"
        value={p.numLayers as number}
        onChange={(v) => update(node.id, { numLayers: v })}
        min={1}
        max={8}
      />
      <TextField
        label="Filters (comma-separated)"
        value={p.filters as string}
        onChange={(v) => update(node.id, { filters: v })}
        placeholder="32,64,128"
      />
      <NumberField
        label="Kernel Size"
        value={p.kernelSize as number}
        onChange={(v) => update(node.id, { kernelSize: v })}
        min={1}
        max={11}
        step={2}
      />
      <NumberField
        label="Num Classes"
        value={p.numClasses as number}
        onChange={(v) => update(node.id, { numClasses: v })}
        min={2}
        max={1000}
      />
      <Section title="Audio Processing" />
      <NumberField
        label="Sample Rate (Hz)"
        value={p.sampleRate as number}
        onChange={(v) => update(node.id, { sampleRate: v })}
        min={8000}
        max={48000}
        step={1000}
      />
      <NumberField
        label="Mel Bands"
        value={p.nMels as number}
        onChange={(v) => update(node.id, { nMels: v })}
        min={40}
        max={256}
        step={8}
      />
      <Section title="Training" />
      <NumberField
        label="Epochs"
        value={p.epochs as number}
        onChange={(v) => update(node.id, { epochs: v })}
        min={1}
        max={500}
      />
      <NumberField
        label="Batch Size"
        value={p.batchSize as number}
        onChange={(v) => update(node.id, { batchSize: v })}
        min={1}
        max={512}
      />
      <NumberField
        label="Learning Rate"
        value={p.learningRate as number}
        onChange={(v) => update(node.id, { learningRate: v })}
        min={1e-6}
        max={0.1}
        step={1e-5}
      />
      <OutputNameField
        value={p.outputName as string}
        onChange={(v) => update(node.id, { outputName: v })}
      />
    </div>
  );
};

const TabularModelForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  const isRNN = ['lstm', 'gru', 'rnn'].includes(p.modelType as string);

  return (
    <div className="space-y-4">
      <SelectField
        label="Model Type"
        value={p.modelType as string}
        options={TABULAR_MODEL_TYPES}
        onChange={(v) => update(node.id, { modelType: v })}
      />
      <TextField
        label="Target Column"
        value={p.targetColumn as string}
        onChange={(v) => update(node.id, { targetColumn: v })}
        placeholder="e.g., label, price"
      />
      <Section title="Architecture" />
      <NumberField
        label="Num Layers"
        value={p.numLayers as number}
        onChange={(v) => update(node.id, { numLayers: v })}
        min={1}
        max={16}
      />
      <NumberField
        label="Hidden Dimension"
        value={p.hiddenDim as number}
        onChange={(v) => update(node.id, { hiddenDim: v })}
        min={16}
        max={2048}
      />
      {isRNN && (
        <Toggle
          label="Bidirectional"
          checked={p.bidirectional as boolean}
          onChange={(v) => update(node.id, { bidirectional: v })}
        />
      )}
      <Section title="Training" />
      <NumberField
        label="Epochs"
        value={p.numEpochs as number}
        onChange={(v) => update(node.id, { numEpochs: v })}
        min={1}
        max={500}
      />
      <NumberField
        label="Batch Size"
        value={p.batchSize as number}
        onChange={(v) => update(node.id, { batchSize: v })}
        min={1}
        max={512}
      />
      <NumberField
        label="Learning Rate"
        value={p.learningRate as number}
        onChange={(v) => update(node.id, { learningRate: v })}
        min={1e-6}
        max={0.1}
        step={1e-5}
      />
      <OutputNameField
        value={p.outputName as string}
        onChange={(v) => update(node.id, { outputName: v })}
      />
    </div>
  );
};

const DeployModelNodeForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  const workflows = useWorkflowsStore((s) => s.workflows);

  const inputTypeOptions = [
    { value: 'textInput',        label: 'Text' },
    { value: 'imageInput',       label: 'Image' },
    { value: 'audioInput',       label: 'Audio' },
    { value: 'spreadsheetInput', label: 'Tabular' },
  ];

  const customModels = workflows.flatMap((w) => w.trainedModels);

  const pretrainedByInput: Record<string, { value: string; label: string }[]> = {
    textInput:        EMBEDDING_MODELS,
    imageInput:       [...IMAGE_CLASSIFIER_MODELS, ...OBJECT_DETECT_MODELS],
    audioInput:       AUDIO_MODELS,
    spreadsheetInput: [],
  };

  const pretrainedModels = pretrainedByInput[p.inputType as string] ?? [];

  return (
    <div className="space-y-4">
      <SelectField
        label="Input Type"
        value={p.inputType as string}
        options={inputTypeOptions}
        onChange={(v) => update(node.id, { inputType: v, modelName: '' })}
      />
      <div className="space-y-1">
        <label className="text-xs font-medium text-gray-400">Model</label>
        <select
          value={p.modelName as string}
          onChange={(e) => update(node.id, { modelName: e.target.value })}
          className="w-full bg-[#1a1a24] border border-[#2a2a38] text-gray-200 text-xs rounded-md px-3 py-2 focus:outline-none focus:border-[#6366f1]"
        >
          <option value="">— Select model —</option>
          {customModels.length > 0 && (
            <optgroup label="Custom (your trained models)">
              {customModels.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </optgroup>
          )}
          {pretrainedModels.length > 0 && (
            <optgroup label="Pre-trained">
              {pretrainedModels.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </optgroup>
          )}
        </select>
      </div>
      {customModels.length === 0 && (
        <p className="text-xs text-gray-600 italic px-1">
          No trained models yet — run a training workflow first, or select a pretrained model above.
        </p>
      )}
    </div>
  );
};

const LLMNodeForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  const isOllama = (p.provider as string)?.startsWith('ollama');
  const isCustom  = p.provider === 'ollama_custom';

  return (
    <div className="space-y-4">
      <SelectField
        label="Provider"
        value={p.provider as string}
        options={LLM_PROVIDERS}
        onChange={(v) => update(node.id, { provider: v })}
      />
      {isCustom && (
        <TextField
          label="Model Name"
          value={p.model as string}
          onChange={(v) => update(node.id, { model: v })}
          placeholder="e.g., llama3.2"
        />
      )}
      {isOllama && (
        <TextField
          label="Ollama URL"
          value={p.ollamaUrl as string}
          onChange={(v) => update(node.id, { ollamaUrl: v })}
          placeholder="http://localhost:11434"
        />
      )}
      <TextField
        label="System Prompt"
        value={p.systemPrompt as string}
        onChange={(v) => update(node.id, { systemPrompt: v })}
        placeholder="Describe the model output in plain language."
        multiline
      />
      <div className="flex items-start gap-2 bg-[#1a1a24] border border-[#2a2a38] rounded-md p-3">
        <Info className="w-4 h-4 text-[#ec4899] mt-0.5 flex-shrink-0" />
        <p className="text-xs text-gray-400">
          The LLM receives raw model output as context and returns a natural language description.
        </p>
      </div>
    </div>
  );
};

const DeployOutputNodeForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  const { id: workflowId } = useParams<{ id: string }>();
  const [copied, setCopied] = useState(false);

  const endpointUrl = `${getWorkerBase()}/api/deploy/${workflowId ?? ':workflowId'}`;
  const curlSnippet = `curl -X POST ${endpointUrl} \\\n  -F "file=@/path/to/input"`;

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(curlSnippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <label className="text-xs text-gray-400 uppercase tracking-wide">Endpoint URL</label>
        <div className="flex items-center gap-2 bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2">
          <Globe className="w-4 h-4 text-[#14b8a6] shrink-0" />
          <span className="text-xs text-gray-300 font-mono break-all flex-1">{endpointUrl}</span>
        </div>
      </div>
      <SelectField
        label="Response Format"
        value={p.format as string}
        options={[
          { value: 'json', label: 'JSON' },
          { value: 'text', label: 'Plain Text' },
        ]}
        onChange={(v) => update(node.id, { format: v })}
      />
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-400 uppercase tracking-wide">cURL Example</label>
          <button
            onClick={copyToClipboard}
            className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-200 transition-colors"
          >
            {copied
              ? <CheckCircle className="w-3.5 h-3.5 text-[#22c55e]" />
              : <Copy className="w-3.5 h-3.5" />
            }
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
        <pre className="bg-[#0a0a0f] border border-[#2a2a38] rounded-md p-3 text-xs text-gray-300 font-mono overflow-x-auto whitespace-pre-wrap">
          {curlSnippet}
        </pre>
      </div>
    </div>
  );
};

// ── Chunk variant forms ───────────────────────────────────────────────────────

const ChunkSimpleForm: React.FC = () => (
  <div className="flex items-start gap-2 bg-[#1a1a24] border border-[#2a2a38] rounded-md p-3">
    <Info className="w-4 h-4 text-[#2dd4bf] mt-0.5 flex-shrink-0" />
    <p className="text-xs text-gray-400">
      This chunking strategy has no configurable parameters. Connect it to an embedding node to use it.
    </p>
  </div>
);

const ChunkSizeForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  return (
    <div className="space-y-4">
      <NumberField label="Chunk Size (tokens)" value={p.chunkSize as number} onChange={(v) => update(node.id, { chunkSize: v })} min={32} max={4096} />
      <NumberField label="Overlap (tokens)" value={p.overlap as number} onChange={(v) => update(node.id, { overlap: v })} min={0} max={512} />
    </div>
  );
};

// ── Embedding variant form (shared) ───────────────────────────────────────────

const EmbeddingVariantForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  return (
    <div className="space-y-4">
      <SelectField label="Training Method" value={p.method as string} options={FINE_TUNE_METHODS} onChange={(v) => update(node.id, { method: v })} />
      <NumberField label="Epochs" value={p.epochs as number} onChange={(v) => update(node.id, { epochs: v })} min={1} max={100} />
      <NumberField label="Learning Rate" value={p.learningRate as number} onChange={(v) => update(node.id, { learningRate: v })} min={1e-6} max={0.1} step={1e-5} />
      <OutputNameField value={p.outputName as string} onChange={(v) => update(node.id, { outputName: v })} />
    </div>
  );
};

// ── Image classifier variant form (shared) ────────────────────────────────────

const ImageClassifierVariantForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  return (
    <div className="space-y-4">
      <Toggle label="Transfer Learning" checked={p.transfer as boolean} onChange={(v) => update(node.id, { transfer: v })} />
      <Section title="Training" />
      <NumberField label="Num Classes" value={p.numClasses as number} onChange={(v) => update(node.id, { numClasses: v })} min={2} max={1000} />
      <NumberField label="Epochs" value={p.epochs as number} onChange={(v) => update(node.id, { epochs: v })} min={1} max={500} />
      <NumberField label="Batch Size" value={p.batchSize as number} onChange={(v) => update(node.id, { batchSize: v })} min={1} max={512} />
      <NumberField label="Learning Rate" value={p.learningRate as number} onChange={(v) => update(node.id, { learningRate: v })} min={1e-6} max={0.1} step={1e-5} />
      <OutputNameField value={p.outputName as string} onChange={(v) => update(node.id, { outputName: v })} />
    </div>
  );
};

// ── Object detector variant form (shared) ─────────────────────────────────────

const ObjectDetectorVariantForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  return (
    <div className="space-y-4">
      <SelectField label="Annotation Format" value={p.inputFormat as string} options={BOUNDING_BOX_FORMATS} onChange={(v) => update(node.id, { inputFormat: v })} />
      <Section title="Training" />
      <NumberField label="Num Classes" value={p.numClasses as number} onChange={(v) => update(node.id, { numClasses: v })} min={1} max={1000} />
      <NumberField label="Epochs" value={p.epochs as number} onChange={(v) => update(node.id, { epochs: v })} min={1} max={500} />
      <NumberField label="Batch Size" value={p.batchSize as number} onChange={(v) => update(node.id, { batchSize: v })} min={1} max={64} />
      <NumberField label="Learning Rate" value={p.learningRate as number} onChange={(v) => update(node.id, { learningRate: v })} min={1e-6} max={0.01} step={1e-6} />
      <OutputNameField value={p.outputName as string} onChange={(v) => update(node.id, { outputName: v })} />
    </div>
  );
};

// ── Audio speech variant forms ────────────────────────────────────────────────

const AudioWhisperForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  return (
    <div className="space-y-4">
      <div className="flex items-start gap-2 bg-[#1a1a24] border border-[#2a2a38] rounded-md p-3">
        <Info className="w-4 h-4 text-[#22c55e] mt-0.5 flex-shrink-0" />
        <p className="text-xs text-gray-400">Task: <span className="text-gray-200">Speech-to-Text transcription</span></p>
      </div>
      <NumberField label="Epochs" value={p.epochs as number} onChange={(v) => update(node.id, { epochs: v })} min={1} max={100} />
      <NumberField label="Learning Rate" value={p.learningRate as number} onChange={(v) => update(node.id, { learningRate: v })} min={1e-6} max={0.01} step={1e-6} />
      <OutputNameField value={p.outputName as string} onChange={(v) => update(node.id, { outputName: v })} />
    </div>
  );
};

const AudioWav2Vec2Form: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  return (
    <div className="space-y-4">
      <SelectField label="Task" value={p.task as string} options={AUDIO_TASKS.filter(t => t.value !== 'emotion_recognition')} onChange={(v) => update(node.id, { task: v })} />
      <NumberField label="Num Classes" value={p.numClasses as number} onChange={(v) => update(node.id, { numClasses: v })} min={2} max={1000} />
      <NumberField label="Epochs" value={p.epochs as number} onChange={(v) => update(node.id, { epochs: v })} min={1} max={100} />
      <NumberField label="Learning Rate" value={p.learningRate as number} onChange={(v) => update(node.id, { learningRate: v })} min={1e-6} max={0.01} step={1e-6} />
      <OutputNameField value={p.outputName as string} onChange={(v) => update(node.id, { outputName: v })} />
    </div>
  );
};

const AudioWav2Vec2EmotionForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  return (
    <div className="space-y-4">
      <div className="flex items-start gap-2 bg-[#1a1a24] border border-[#2a2a38] rounded-md p-3">
        <Info className="w-4 h-4 text-[#22c55e] mt-0.5 flex-shrink-0" />
        <p className="text-xs text-gray-400">Task: <span className="text-gray-200">Emotion Recognition</span> — predicts 7 emotion classes (neutral, happy, sad, angry, fearful, disgust, surprised).</p>
      </div>
      <NumberField label="Epochs" value={p.epochs as number} onChange={(v) => update(node.id, { epochs: v })} min={1} max={100} />
      <NumberField label="Learning Rate" value={p.learningRate as number} onChange={(v) => update(node.id, { learningRate: v })} min={1e-6} max={0.01} step={1e-6} />
      <OutputNameField value={p.outputName as string} onChange={(v) => update(node.id, { outputName: v })} />
    </div>
  );
};

// ── Tabular variant forms ─────────────────────────────────────────────────────

const TabularSequentialForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  return (
    <div className="space-y-4">
      <TextField label="Target Column" value={p.targetColumn as string} onChange={(v) => update(node.id, { targetColumn: v })} placeholder="e.g., label" />
      <Section title="Architecture" />
      <NumberField label="Num Layers" value={p.numLayers as number} onChange={(v) => update(node.id, { numLayers: v })} min={1} max={8} />
      <NumberField label="Hidden Dim" value={p.hiddenDim as number} onChange={(v) => update(node.id, { hiddenDim: v })} min={16} max={2048} />
      <Toggle label="Bidirectional" checked={p.bidirectional as boolean} onChange={(v) => update(node.id, { bidirectional: v })} />
      <Section title="Training" />
      <NumberField label="Epochs" value={p.numEpochs as number} onChange={(v) => update(node.id, { numEpochs: v })} min={1} max={500} />
      <NumberField label="Batch Size" value={p.batchSize as number} onChange={(v) => update(node.id, { batchSize: v })} min={1} max={512} />
      <NumberField label="Learning Rate" value={p.learningRate as number} onChange={(v) => update(node.id, { learningRate: v })} min={1e-6} max={0.1} step={1e-5} />
      <OutputNameField value={p.outputName as string} onChange={(v) => update(node.id, { outputName: v })} />
    </div>
  );
};

const TabularDenseForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  return (
    <div className="space-y-4">
      <TextField label="Target Column" value={p.targetColumn as string} onChange={(v) => update(node.id, { targetColumn: v })} placeholder="e.g., label" />
      <Section title="Architecture" />
      <NumberField label="Num Layers" value={p.numLayers as number} onChange={(v) => update(node.id, { numLayers: v })} min={1} max={16} />
      <NumberField label="Hidden Dim" value={p.hiddenDim as number} onChange={(v) => update(node.id, { hiddenDim: v })} min={16} max={4096} />
      <Section title="Training" />
      <NumberField label="Epochs" value={p.numEpochs as number} onChange={(v) => update(node.id, { numEpochs: v })} min={1} max={500} />
      <NumberField label="Batch Size" value={p.batchSize as number} onChange={(v) => update(node.id, { batchSize: v })} min={1} max={512} />
      <NumberField label="Learning Rate" value={p.learningRate as number} onChange={(v) => update(node.id, { learningRate: v })} min={1e-6} max={0.1} step={1e-5} />
      <OutputNameField value={p.outputName as string} onChange={(v) => update(node.id, { outputName: v })} />
    </div>
  );
};

const SaveModelForm: React.FC = () => {
  const node = useWorkflowStore((s) => s.selectedNode)!;
  const update = useWorkflowStore((s) => s.updateNodeParameters);
  const p = node.data.parameters;
  return (
    <div className="space-y-4">
      <TextField
        label="Model Name"
        value={(p.modelName as string) ?? ''}
        onChange={(v) => update(node.id, { modelName: v })}
        placeholder="e.g., my-text-classifier"
      />
      <div className="flex items-start gap-2 bg-[#1a1a24] border border-[#2a2a38] rounded-md p-3">
        <CheckCircle className="w-4 h-4 text-[#ef4444] mt-0.5 flex-shrink-0" />
        <p className="text-xs text-gray-400">
          The trained model will be saved with this name to Cloudflare R2 and the Modal Volume.
        </p>
      </div>
    </div>
  );
};

// ── Form selector ─────────────────────────────────────────────────────────────

const getFormComponent = (type: NodeType): React.FC | null => {
  const map: Partial<Record<NodeType, React.FC>> = {
    // Input nodes
    textInput: TextInputForm,
    imageInput: ImageInputForm,
    audioInput: AudioInputForm,
    spreadsheetInput: SpreadsheetInputForm,
    // Legacy combined nodes
    chunkNode: ChunkNodeForm,
    embeddingModel: EmbeddingModelForm,
    imageClassifier: ImageClassifierForm,
    imageCNN: ImageCNNForm,
    imageCAE: ImageCAEForm,
    objectDetector: ObjectDetectorForm,
    audioSpeechModel: AudioSpeechModelForm,
    audioCNN: AudioCNNForm,
    tabularModel: TabularModelForm,
    // Chunk variants
    chunkAuto: ChunkSimpleForm,
    chunkSentence: ChunkSimpleForm,
    chunkParagraph: ChunkSimpleForm,
    chunkSlidingWindow: ChunkSizeForm,
    chunkFixedSize: ChunkSizeForm,
    chunkMarkdown: ChunkSimpleForm,
    chunkRecursive: ChunkSimpleForm,
    chunkCode: ChunkSimpleForm,
    // Embedding variants
    embeddingMiniLM: EmbeddingVariantForm,
    embeddingMPNet: EmbeddingVariantForm,
    embeddingBGESmall: EmbeddingVariantForm,
    embeddingBGEBase: EmbeddingVariantForm,
    embeddingMultilingual: EmbeddingVariantForm,
    // Image classifier variants
    classifierResNet50: ImageClassifierVariantForm,
    classifierConvNeXt: ImageClassifierVariantForm,
    classifierResNet18: ImageClassifierVariantForm,
    // Object detector variants
    detectorYOLOS: ObjectDetectorVariantForm,
    detectorRTDETR: ObjectDetectorVariantForm,
    detectorDETR: ObjectDetectorVariantForm,
    // Audio speech variants
    audioWhisper: AudioWhisperForm,
    audioWav2Vec2: AudioWav2Vec2Form,
    audioWav2Vec2Emotion: AudioWav2Vec2EmotionForm,
    // Tabular variants
    tabularLSTM: TabularSequentialForm,
    tabularGRU: TabularSequentialForm,
    tabularRNN: TabularSequentialForm,
    tabularFFNN: TabularDenseForm,
    tabularDNN: TabularDenseForm,
    // Output nodes
    saveModel: SaveModelForm,
    deployModelNode: DeployModelNodeForm,
    llmNode: LLMNodeForm,
    deployOutputNode: DeployOutputNodeForm,
  };
  return map[type] ?? null;
};

// ── Panel ─────────────────────────────────────────────────────────────────────

export const PropertiesPanel: React.FC = () => {
  const selectedNode = useWorkflowStore((state) => state.selectedNode);
  const removeNode = useWorkflowStore((state) => state.removeNode);
  const setSelectedNode = useWorkflowStore((state) => state.setSelectedNode);

  const definition = selectedNode
    ? nodeDefinitions.find((n) => n.type === selectedNode.data.type)
    : null;

  const FormComponent = selectedNode ? getFormComponent(selectedNode.data.type) : null;
  const Icon = definition?.icon;

  const handleDelete = () => {
    if (selectedNode) {
      removeNode(selectedNode.id);
      setSelectedNode(null);
    }
  };

  return (
    <div className="w-80 bg-[#12121a] border-l border-[#22222e] flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-[#22222e] flex items-center justify-between">
        <h2 className="text-[#00d4ff] text-sm font-semibold uppercase tracking-wider">
          Properties
        </h2>
        <div className="w-8 h-8 flex items-center justify-center">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#ffd700" strokeWidth="1.5" />
            <path d="M2 17L12 22L22 17" stroke="#ffd700" strokeWidth="1.5" />
            <path d="M2 12L12 17L22 12" stroke="#ffd700" strokeWidth="1.5" />
          </svg>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {selectedNode ? (
          <div className="p-4 space-y-6">
            {/* Node header */}
            <div className="flex items-center gap-3 pb-4 border-b border-[#22222e]">
              {Icon && (
                <div
                  className="p-2 rounded-lg"
                  style={{
                    backgroundColor: definition?.color + '20',
                    color: definition?.color,
                  }}
                >
                  <Icon className="w-5 h-5" />
                </div>
              )}
              <div className="flex-1">
                <h3 className="text-sm font-medium text-gray-200">{selectedNode.data.label}</h3>
                <p className="text-xs text-gray-500">{definition?.description}</p>
              </div>
            </div>

            {/* Form */}
            {FormComponent && <FormComponent />}

            {/* Delete button */}
            <div className="pt-4 border-t border-[#22222e]">
              <button
                onClick={handleDelete}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 hover:bg-red-500/20 transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                Delete Node
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center p-8">
            <Settings className="w-12 h-12 text-gray-700 mb-4" />
            <p className="text-gray-500 text-sm">Select a node to edit its properties</p>
          </div>
        )}
      </div>
    </div>
  );
};
