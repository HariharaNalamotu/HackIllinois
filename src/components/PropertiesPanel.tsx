import React, { useRef } from 'react';
import { Settings, Trash2, Upload, Plus, X, FolderOpen, File } from 'lucide-react';
import { useWorkflowStore, NodeType, ToolParameter } from '../store/workflowStore';
import { nodeDefinitions, modelOptions, chunkingStrategyOptions, evalStrategyOptions, parameterTypeOptions, voiceTaskOptions } from '../types/nodes';
import { v4 as uuidv4 } from 'uuid';

// Reusable form components
interface SelectFieldProps {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}

const SelectField: React.FC<SelectFieldProps> = ({ label, value, options, onChange }) => (
  <div className="space-y-1">
    <label className="text-xs text-gray-400 uppercase tracking-wide">{label}</label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-[#00d4ff] transition-colors"
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  </div>
);

interface TextFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  multiline?: boolean;
}

const TextField: React.FC<TextFieldProps> = ({ label, value, onChange, placeholder, multiline }) => (
  <div className="space-y-1">
    <label className="text-xs text-gray-400 uppercase tracking-wide">{label}</label>
    {multiline ? (
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={4}
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

interface NumberFieldProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

const NumberField: React.FC<NumberFieldProps> = ({ label, value, onChange, min, max, step = 1 }) => (
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

interface CheckboxFieldProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

const CheckboxField: React.FC<CheckboxFieldProps> = ({ label, checked, onChange }) => (
  <label className="flex items-center gap-3 cursor-pointer group">
    <div
      className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
        checked ? 'bg-[#00d4ff] border-[#00d4ff]' : 'border-[#2a2a38] group-hover:border-[#00d4ff]'
      }`}
      onClick={() => onChange(!checked)}
    >
      {checked && (
        <svg className="w-3 h-3 text-[#0a0a0f]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
        </svg>
      )}
    </div>
    <span className="text-sm text-gray-300">{label}</span>
  </label>
);

interface FileUploadFieldProps {
  label: string;
  value: string | null;
  onChange: (files: FileList | null) => void;
  accept?: string;
  multiple?: boolean;
  allowFolder?: boolean;
}

const FileUploadField: React.FC<FileUploadFieldProps> = ({ label, value, onChange, accept, multiple, allowFolder }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="space-y-2">
      <label className="text-xs text-gray-400 uppercase tracking-wide">{label}</label>
      <div className="space-y-2">
        {value ? (
          <div className="flex items-center gap-2 bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2">
            <File className="w-4 h-4 text-[#00d4ff]" />
            <span className="text-sm text-gray-300 flex-1 truncate">{value}</span>
            <button
              onClick={() => onChange(null)}
              className="text-gray-500 hover:text-red-400 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <div className="flex gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept={accept}
              multiple={multiple}
              onChange={(e) => onChange(e.target.files)}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 flex items-center justify-center gap-2 bg-[#1a1a24] border border-dashed border-[#2a2a38] rounded-md px-3 py-3 text-sm text-gray-400 hover:border-[#00d4ff] hover:text-[#00d4ff] transition-colors"
            >
              <Upload className="w-4 h-4" />
              <span>{multiple ? 'Upload Files' : 'Upload File'}</span>
            </button>
            {allowFolder && (
              <>
                <input
                  ref={folderInputRef}
                  type="file"
                  // @ts-ignore - webkitdirectory is not in types
                  webkitdirectory=""
                  onChange={(e) => onChange(e.target.files)}
                  className="hidden"
                />
                <button
                  onClick={() => folderInputRef.current?.click()}
                  className="flex items-center justify-center gap-2 bg-[#1a1a24] border border-dashed border-[#2a2a38] rounded-md px-3 py-3 text-sm text-gray-400 hover:border-[#00d4ff] hover:text-[#00d4ff] transition-colors"
                >
                  <FolderOpen className="w-4 h-4" />
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// Tool Parameter Editor for Agent Tools
interface ToolParameterEditorProps {
  parameters: ToolParameter[];
  onChange: (parameters: ToolParameter[]) => void;
}

const ToolParameterEditor: React.FC<ToolParameterEditorProps> = ({ parameters, onChange }) => {
  const addParameter = () => {
    const newParam: ToolParameter = {
      id: uuidv4(),
      name: '',
      type: 'string',
      description: '',
      required: false,
    };
    onChange([...parameters, newParam]);
  };

  const updateParameter = (id: string, updates: Partial<ToolParameter>) => {
    onChange(parameters.map((p) => (p.id === id ? { ...p, ...updates } : p)));
  };

  const removeParameter = (id: string) => {
    onChange(parameters.filter((p) => p.id !== id));
  };

  const addObjectProperty = (paramId: string) => {
    const param = parameters.find((p) => p.id === paramId);
    if (param) {
      const props = param.objectProperties || [];
      updateParameter(paramId, { objectProperties: [...props, { key: '', value: '' }] });
    }
  };

  const updateObjectProperty = (paramId: string, index: number, key: string, value: string) => {
    const param = parameters.find((p) => p.id === paramId);
    if (param && param.objectProperties) {
      const newProps = [...param.objectProperties];
      newProps[index] = { key, value };
      updateParameter(paramId, { objectProperties: newProps });
    }
  };

  const removeObjectProperty = (paramId: string, index: number) => {
    const param = parameters.find((p) => p.id === paramId);
    if (param && param.objectProperties) {
      const newProps = param.objectProperties.filter((_, i) => i !== index);
      updateParameter(paramId, { objectProperties: newProps });
    }
  };

  const addArrayValue = (paramId: string) => {
    const param = parameters.find((p) => p.id === paramId);
    if (param) {
      const values = param.arrayValues || [];
      updateParameter(paramId, { arrayValues: [...values, ''] });
    }
  };

  const updateArrayValue = (paramId: string, index: number, value: string) => {
    const param = parameters.find((p) => p.id === paramId);
    if (param && param.arrayValues) {
      const newValues = [...param.arrayValues];
      newValues[index] = value;
      updateParameter(paramId, { arrayValues: newValues });
    }
  };

  const removeArrayValue = (paramId: string, index: number) => {
    const param = parameters.find((p) => p.id === paramId);
    if (param && param.arrayValues) {
      const newValues = param.arrayValues.filter((_, i) => i !== index);
      updateParameter(paramId, { arrayValues: newValues });
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-xs text-gray-400 uppercase tracking-wide">Function Parameters</label>
        <button
          onClick={addParameter}
          className="flex items-center gap-1 text-xs text-[#00d4ff] hover:text-[#00b8d4] transition-colors"
        >
          <Plus className="w-3 h-3" />
          Add Parameter
        </button>
      </div>

      {parameters.length === 0 ? (
        <p className="text-xs text-gray-600 italic">No parameters defined</p>
      ) : (
        <div className="space-y-4">
          {parameters.map((param) => (
            <div key={param.id} className="bg-[#12121a] border border-[#2a2a38] rounded-lg p-3 space-y-3">
              <div className="flex items-start justify-between">
                <input
                  type="text"
                  value={param.name}
                  onChange={(e) => updateParameter(param.id, { name: e.target.value })}
                  placeholder="Parameter name"
                  className="bg-transparent border-b border-[#2a2a38] text-sm text-gray-200 focus:outline-none focus:border-[#00d4ff] pb-1"
                />
                <button
                  onClick={() => removeParameter(param.id)}
                  className="text-gray-500 hover:text-red-400 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <select
                  value={param.type}
                  onChange={(e) => updateParameter(param.id, { type: e.target.value as ToolParameter['type'] })}
                  className="bg-[#1a1a24] border border-[#2a2a38] rounded px-2 py-1 text-xs text-gray-300"
                >
                  {parameterTypeOptions.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
                <label className="flex items-center gap-2 text-xs text-gray-400">
                  <input
                    type="checkbox"
                    checked={param.required}
                    onChange={(e) => updateParameter(param.id, { required: e.target.checked })}
                    className="rounded"
                  />
                  Required
                </label>
              </div>

              <textarea
                value={param.description}
                onChange={(e) => updateParameter(param.id, { description: e.target.value })}
                placeholder="Description"
                rows={2}
                className="w-full bg-[#1a1a24] border border-[#2a2a38] rounded px-2 py-1 text-xs text-gray-300 resize-none"
              />

              {/* Object properties */}
              {param.type === 'object' && (
                <div className="space-y-2 pl-3 border-l-2 border-[#2a2a38]">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">Object Properties</span>
                    <button
                      onClick={() => addObjectProperty(param.id)}
                      className="text-xs text-[#00d4ff]"
                    >
                      + Add
                    </button>
                  </div>
                  {(param.objectProperties || []).map((prop, idx) => (
                    <div key={idx} className="flex gap-2 items-center">
                      <input
                        type="text"
                        value={prop.key}
                        onChange={(e) => updateObjectProperty(param.id, idx, e.target.value, prop.value)}
                        placeholder="Key"
                        className="flex-1 bg-[#1a1a24] border border-[#2a2a38] rounded px-2 py-1 text-xs text-gray-300"
                      />
                      <input
                        type="text"
                        value={prop.value}
                        onChange={(e) => updateObjectProperty(param.id, idx, prop.key, e.target.value)}
                        placeholder="Value"
                        className="flex-1 bg-[#1a1a24] border border-[#2a2a38] rounded px-2 py-1 text-xs text-gray-300"
                      />
                      <button
                        onClick={() => removeObjectProperty(param.id, idx)}
                        className="text-gray-500 hover:text-red-400"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Array values */}
              {param.type === 'array' && (
                <div className="space-y-2 pl-3 border-l-2 border-[#2a2a38]">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">Array Values</span>
                    <button
                      onClick={() => addArrayValue(param.id)}
                      className="text-xs text-[#00d4ff]"
                    >
                      + Add
                    </button>
                  </div>
                  {(param.arrayValues || []).map((val, idx) => (
                    <div key={idx} className="flex gap-2 items-center">
                      <input
                        type="text"
                        value={val}
                        onChange={(e) => updateArrayValue(param.id, idx, e.target.value)}
                        placeholder={`Value ${idx + 1}`}
                        className="flex-1 bg-[#1a1a24] border border-[#2a2a38] rounded px-2 py-1 text-xs text-gray-300"
                      />
                      <button
                        onClick={() => removeArrayValue(param.id, idx)}
                        className="text-gray-500 hover:text-red-400"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Node-specific form components
const TextRetrievalForm: React.FC = () => {
  const selectedNode = useWorkflowStore((state) => state.selectedNode);
  const updateNodeParameters = useWorkflowStore((state) => state.updateNodeParameters);

  if (!selectedNode) return null;
  const params = selectedNode.data.parameters;

  return (
    <div className="space-y-4">
      <SelectField
        label="Base Decoder Model"
        value={params.baseDecoderModel}
        options={modelOptions}
        onChange={(v) => updateNodeParameters(selectedNode.id, { baseDecoderModel: v })}
      />
      <FileUploadField
        label="Vector DB Dataset"
        value={params.vectorDBDataset}
        onChange={(files) => updateNodeParameters(selectedNode.id, { vectorDBDataset: files?.[0]?.name || null })}
        multiple
        allowFolder
      />
      <SelectField
        label="Chunking Strategy"
        value={params.chunkingStrategy}
        options={chunkingStrategyOptions}
        onChange={(v) => updateNodeParameters(selectedNode.id, { chunkingStrategy: v })}
      />
      <NumberField
        label="Chunking Size"
        value={params.chunkingSize}
        onChange={(v) => updateNodeParameters(selectedNode.id, { chunkingSize: v })}
        min={64}
        max={4096}
        step={64}
      />
    </div>
  );
};

const AgenticLLMForm: React.FC = () => {
  const selectedNode = useWorkflowStore((state) => state.selectedNode);
  const updateNodeParameters = useWorkflowStore((state) => state.updateNodeParameters);

  if (!selectedNode) return null;
  const params = selectedNode.data.parameters;

  return (
    <div className="space-y-4">
      <SelectField
        label="Sub-Agent Model"
        value={params.subAgentModel}
        options={modelOptions}
        onChange={(v) => updateNodeParameters(selectedNode.id, { subAgentModel: v })}
      />
      <TextField
        label="Sub-Agent Prompt"
        value={params.subAgentPrompt}
        onChange={(v) => updateNodeParameters(selectedNode.id, { subAgentPrompt: v })}
        placeholder="Enter the prompt for the sub-agent..."
        multiline
      />
    </div>
  );
};

const VisualDataForm: React.FC = () => {
  const selectedNode = useWorkflowStore((state) => state.selectedNode);
  const updateNodeParameters = useWorkflowStore((state) => state.updateNodeParameters);

  if (!selectedNode) return null;
  const params = selectedNode.data.parameters;

  return (
    <div className="space-y-4">
      <FileUploadField
        label="Training Dataset"
        value={params.dataset}
        onChange={(files) => updateNodeParameters(selectedNode.id, { dataset: files?.[0]?.name || null })}
        accept="image/*,video/*"
        multiple
        allowFolder
      />
      <NumberField
        label="Epoch Count"
        value={params.epochCount}
        onChange={(v) => updateNodeParameters(selectedNode.id, { epochCount: v })}
        min={1}
        max={1000}
      />
      <NumberField
        label="Learning Rate"
        value={params.learningRate}
        onChange={(v) => updateNodeParameters(selectedNode.id, { learningRate: v })}
        min={0.00001}
        max={1}
        step={0.0001}
      />
      <NumberField
        label="Weight Decay"
        value={params.weightDecay}
        onChange={(v) => updateNodeParameters(selectedNode.id, { weightDecay: v })}
        min={0}
        max={1}
        step={0.001}
      />
      <SelectField
        label="Evaluation Strategy"
        value={params.evalStrategy}
        options={evalStrategyOptions}
        onChange={(v) => updateNodeParameters(selectedNode.id, { evalStrategy: v })}
      />
    </div>
  );
};

const AudioDataForm: React.FC = () => {
  const selectedNode = useWorkflowStore((state) => state.selectedNode);
  const updateNodeParameters = useWorkflowStore((state) => state.updateNodeParameters);

  if (!selectedNode) return null;
  const params = selectedNode.data.parameters;

  return (
    <div className="space-y-4">
      <FileUploadField
        label="Training Dataset"
        value={params.dataset}
        onChange={(files) => updateNodeParameters(selectedNode.id, { dataset: files?.[0]?.name || null })}
        accept="audio/*"
        multiple
        allowFolder
      />
      <NumberField
        label="Epoch Count"
        value={params.epochCount}
        onChange={(v) => updateNodeParameters(selectedNode.id, { epochCount: v })}
        min={1}
        max={1000}
      />
      <NumberField
        label="Learning Rate"
        value={params.learningRate}
        onChange={(v) => updateNodeParameters(selectedNode.id, { learningRate: v })}
        min={0.00001}
        max={1}
        step={0.0001}
      />
      <NumberField
        label="Weight Decay"
        value={params.weightDecay}
        onChange={(v) => updateNodeParameters(selectedNode.id, { weightDecay: v })}
        min={0}
        max={1}
        step={0.001}
      />
      <SelectField
        label="Evaluation Strategy"
        value={params.evalStrategy}
        options={evalStrategyOptions}
        onChange={(v) => updateNodeParameters(selectedNode.id, { evalStrategy: v })}
      />
    </div>
  );
};

const VoiceInputForm: React.FC = () => {
  const selectedNode = useWorkflowStore((state) => state.selectedNode);
  const updateNodeParameters = useWorkflowStore((state) => state.updateNodeParameters);

  if (!selectedNode) return null;
  const params = selectedNode.data.parameters;

  return (
    <div className="space-y-4">
      <SelectField
        label="Task"
        value={params.task}
        options={voiceTaskOptions}
        onChange={(v) => updateNodeParameters(selectedNode.id, { task: v })}
      />
      <FileUploadField
        label="Dataset"
        value={params.dataset}
        onChange={(files) => updateNodeParameters(selectedNode.id, { dataset: files?.[0]?.name || null })}
        accept="audio/*"
        multiple
        allowFolder
      />
    </div>
  );
};

const AgentToolForm: React.FC = () => {
  const selectedNode = useWorkflowStore((state) => state.selectedNode);
  const updateNodeParameters = useWorkflowStore((state) => state.updateNodeParameters);

  if (!selectedNode) return null;
  const params = selectedNode.data.parameters;

  return (
    <div className="space-y-4">
      <TextField
        label="Function Name"
        value={params.functionName}
        onChange={(v) => updateNodeParameters(selectedNode.id, { functionName: v })}
        placeholder="e.g., search_web"
      />
      <TextField
        label="Function Description"
        value={params.functionDescription}
        onChange={(v) => updateNodeParameters(selectedNode.id, { functionDescription: v })}
        placeholder="Describe what this function does and when to use it..."
        multiline
      />
      <ToolParameterEditor
        parameters={params.parameters || []}
        onChange={(p) => updateNodeParameters(selectedNode.id, { parameters: p })}
      />
    </div>
  );
};

const RLHFForm: React.FC = () => {
  const selectedNode = useWorkflowStore((state) => state.selectedNode);
  const updateNodeParameters = useWorkflowStore((state) => state.updateNodeParameters);

  if (!selectedNode) return null;
  const params = selectedNode.data.parameters;

  return (
    <div className="space-y-4">
      <NumberField
        label="Number of Iterations"
        value={params.iterations}
        onChange={(v) => updateNodeParameters(selectedNode.id, { iterations: v })}
        min={1}
        max={100}
      />
    </div>
  );
};

const RLAIFForm: React.FC = () => {
  const selectedNode = useWorkflowStore((state) => state.selectedNode);
  const updateNodeParameters = useWorkflowStore((state) => state.updateNodeParameters);

  if (!selectedNode) return null;
  const params = selectedNode.data.parameters;

  return (
    <div className="space-y-4">
      <SelectField
        label="Evaluator Model"
        value={params.evaluatorModel}
        options={modelOptions}
        onChange={(v) => updateNodeParameters(selectedNode.id, { evaluatorModel: v })}
      />
      <NumberField
        label="Number of Iterations"
        value={params.iterations}
        onChange={(v) => updateNodeParameters(selectedNode.id, { iterations: v })}
        min={1}
        max={100}
      />
    </div>
  );
};

const SubAgentForm: React.FC = () => {
  const selectedNode = useWorkflowStore((state) => state.selectedNode);
  const updateNodeParameters = useWorkflowStore((state) => state.updateNodeParameters);

  if (!selectedNode) return null;
  const params = selectedNode.data.parameters;

  return (
    <div className="space-y-4">
      <SelectField
        label="Sub-Agent Model"
        value={params.subAgentModel}
        options={modelOptions}
        onChange={(v) => updateNodeParameters(selectedNode.id, { subAgentModel: v })}
      />
      <TextField
        label="Sub-Agent Prompt"
        value={params.subAgentPrompt}
        onChange={(v) => updateNodeParameters(selectedNode.id, { subAgentPrompt: v })}
        placeholder="Enter the prompt for the sub-agent..."
        multiline
      />
    </div>
  );
};

const ChunkingOptimizationForm: React.FC = () => {
  const selectedNode = useWorkflowStore((state) => state.selectedNode);
  const updateNodeParameters = useWorkflowStore((state) => state.updateNodeParameters);

  if (!selectedNode) return null;
  const params = selectedNode.data.parameters;

  return (
    <div className="space-y-4">
      <CheckboxField
        label="Enable Auto-Optimization"
        checked={params.enableAutoOptimization}
        onChange={(v) => updateNodeParameters(selectedNode.id, { enableAutoOptimization: v })}
      />
      <div className="space-y-2">
        <label className="text-xs text-gray-400 uppercase tracking-wide">Test Strategies</label>
        <div className="space-y-2">
          {chunkingStrategyOptions.map((strategy) => (
            <CheckboxField
              key={strategy.value}
              label={strategy.label}
              checked={(params.testStrategies || []).includes(strategy.value)}
              onChange={(checked) => {
                const current = params.testStrategies || [];
                const updated = checked
                  ? [...current, strategy.value]
                  : current.filter((s: string) => s !== strategy.value);
                updateNodeParameters(selectedNode.id, { testStrategies: updated });
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

const HyperparamTuningForm: React.FC = () => {
  const selectedNode = useWorkflowStore((state) => state.selectedNode);
  const updateNodeParameters = useWorkflowStore((state) => state.updateNodeParameters);

  if (!selectedNode) return null;
  const params = selectedNode.data.parameters;

  return (
    <div className="space-y-4">
      <div className="space-y-3">
        <label className="text-xs text-gray-400 uppercase tracking-wide">Search Methods</label>
        <CheckboxField
          label="Grid Search"
          checked={params.enableGridSearch}
          onChange={(v) => updateNodeParameters(selectedNode.id, { enableGridSearch: v })}
        />
        <CheckboxField
          label="Random Search"
          checked={params.enableRandomSearch}
          onChange={(v) => updateNodeParameters(selectedNode.id, { enableRandomSearch: v })}
        />
        <CheckboxField
          label="Bayesian Optimization"
          checked={params.enableBayesian}
          onChange={(v) => updateNodeParameters(selectedNode.id, { enableBayesian: v })}
        />
      </div>
    </div>
  );
};

// Form selector based on node type
const getFormComponent = (type: NodeType): React.FC | null => {
  switch (type) {
    case 'textRetrieval':
      return TextRetrievalForm;
    case 'agenticLLM':
      return AgenticLLMForm;
    case 'visualData':
      return VisualDataForm;
    case 'audioData':
      return AudioDataForm;
    case 'voiceInput':
      return VoiceInputForm;
    case 'agentTool':
      return AgentToolForm;
    case 'rlhf':
      return RLHFForm;
    case 'rlaif':
      return RLAIFForm;
    case 'subAgent':
      return SubAgentForm;
    case 'chunkingOptimization':
      return ChunkingOptimizationForm;
    case 'visualHyperparamTuning':
    case 'audioHyperparamTuning':
    case 'voiceHyperparamTuning':
      return HyperparamTuningForm;
    default:
      return null;
  }
};

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
                  style={{ backgroundColor: definition?.color + '20', color: definition?.color }}
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
