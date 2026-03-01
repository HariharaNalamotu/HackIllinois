import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Pencil, Trash2, Brain, Rocket, ChevronDown, ChevronRight, Download, Zap } from 'lucide-react';
import { useWorkflowsStore, Workflow } from '../store/workflowsStore';
import { downloadModel } from '../services/api';

// ── Workflow card ─────────────────────────────────────────────────────────────

interface WorkflowCardProps {
  workflow: Workflow;
  onEdit: (id: string) => void;
  onRun: (id: string) => void;
  onDelete: (id: string, e: React.MouseEvent) => void;
}

const WorkflowCard: React.FC<WorkflowCardProps> = ({ workflow, onEdit, onRun, onDelete }) => {
  const [modelsOpen, setModelsOpen] = useState(false);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);

  const handleDownload = async (modelName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDownloadingModel(modelName);
    try {
      await downloadModel(modelName);
    } catch (err) {
      console.error('Download failed:', err);
    } finally {
      setDownloadingModel(null);
    }
  };

  return (
    <div className="bg-[#12121a] border border-[#22222e] rounded-lg overflow-hidden hover:border-[#2a2a38] transition-colors group">
      <div className="p-4 flex items-center justify-between">
        <div className="flex-1 cursor-pointer" onClick={() => onEdit(workflow.id)}>
          <div className="flex items-center gap-2 mb-1">
            <h3 className="text-gray-200 font-medium group-hover:text-[#00d4ff] transition-colors">
              {workflow.name}
            </h3>
            <span
              className={`text-[10px] px-2 py-0.5 rounded-full font-semibold uppercase tracking-wide ${
                workflow.type === 'training'
                  ? 'bg-blue-500/15 text-blue-400 border border-blue-500/30'
                  : 'bg-green-500/15 text-green-400 border border-green-500/30'
              }`}
            >
              {workflow.type}
            </span>
          </div>
          <p className="text-gray-500 text-sm">
            {workflow.nodes.length} node{workflow.nodes.length !== 1 ? 's' : ''}
            {workflow.trainedModels?.length
              ? ` · ${workflow.trainedModels.length} model${workflow.trainedModels.length !== 1 ? 's' : ''} saved`
              : ''}
            {' · '}Updated {new Date(workflow.updatedAt).toLocaleDateString()}
          </p>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => onEdit(workflow.id)}
            title="Edit workflow"
            className="p-2 rounded-lg bg-[#1a1a24] hover:bg-[#22222e] text-gray-400 hover:text-[#00d4ff] transition-colors"
          >
            <Pencil className="w-4 h-4" />
          </button>
          <button
            onClick={() => onRun(workflow.id)}
            title="Run inference"
            className="p-2 rounded-lg bg-[#1a1a24] hover:bg-[#22222e] text-gray-400 hover:text-[#6366f1] transition-colors"
          >
            <Zap className="w-4 h-4" />
          </button>
          {workflow.trainedModels?.length > 0 && (
            <button
              onClick={(e) => { e.stopPropagation(); setModelsOpen(!modelsOpen); }}
              title="Show trained models"
              className="p-2 rounded-lg bg-[#1a1a24] hover:bg-[#22222e] text-gray-400 hover:text-[#22c55e] transition-colors"
            >
              {modelsOpen
                ? <ChevronDown className="w-4 h-4" />
                : <ChevronRight className="w-4 h-4" />
              }
            </button>
          )}
          <button
            onClick={(e) => onDelete(workflow.id, e)}
            title="Delete"
            className="p-2 rounded-lg bg-[#1a1a24] hover:bg-[#22222e] text-gray-400 hover:text-red-400 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Trained models expandable list */}
      {modelsOpen && workflow.trainedModels?.length > 0 && (
        <div className="border-t border-[#22222e] px-4 py-3 space-y-1.5">
          <p className="text-xs text-gray-600 uppercase tracking-wider mb-2">Trained Models</p>
          {workflow.trainedModels.map((modelName) => (
            <div
              key={modelName}
              className="flex items-center justify-between bg-[#0a0a0f] border border-[#22222e] rounded-md px-3 py-2"
            >
              <span className="text-xs text-gray-300 font-mono truncate flex-1">{modelName}</span>
              <button
                onClick={(e) => handleDownload(modelName, e)}
                disabled={downloadingModel === modelName}
                className="flex items-center gap-1 text-xs text-[#22c55e] hover:text-white disabled:opacity-50 transition-colors ml-2 shrink-0"
                title="Download model"
              >
                <Download className="w-3.5 h-3.5" />
                {downloadingModel === modelName ? 'Downloading…' : 'Download'}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ── Section header ────────────────────────────────────────────────────────────

interface SectionProps {
  title: string;
  icon: React.ReactNode;
  accentColor: string;
  workflows: Workflow[];
  onEdit: (id: string) => void;
  onRun: (id: string) => void;
  onDelete: (id: string, e: React.MouseEvent) => void;
  newButton: React.ReactNode;
  inputRow?: React.ReactNode;
}

const Section: React.FC<SectionProps> = ({
  title, icon, accentColor, workflows,
  onEdit, onRun, onDelete, newButton, inputRow,
}) => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="mb-10">
      <div className="flex items-center justify-between mb-4">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center gap-2 text-gray-200 hover:text-white transition-colors"
        >
          <div style={{ color: accentColor }}>{icon}</div>
          <span className="text-lg font-semibold">{title}</span>
          <span className="text-gray-500 text-sm ml-1">({workflows.length})</span>
          {collapsed
            ? <ChevronRight className="w-4 h-4 text-gray-500 ml-1" />
            : <ChevronDown  className="w-4 h-4 text-gray-500 ml-1" />
          }
        </button>
        {newButton}
      </div>

      {!collapsed && (
        <>
          {inputRow}

          {workflows.length === 0 ? (
            <div className="text-center py-8 text-gray-600 border border-dashed border-[#22222e] rounded-lg">
              <p className="text-sm">No workflows yet</p>
            </div>
          ) : (
            <div className="space-y-2">
              {workflows.map((wf) => (
                <WorkflowCard
                  key={wf.id}
                  workflow={wf}
                  onEdit={onEdit}
                  onRun={onRun}
                  onDelete={onDelete}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};

// ── Home page ─────────────────────────────────────────────────────────────────

export const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const workflows = useWorkflowsStore((s) => s.workflows);
  const createWorkflow = useWorkflowsStore((s) => s.createWorkflow);
  const createDeploymentFromTraining = useWorkflowsStore((s) => s.createDeploymentFromTraining);
  const deleteWorkflow = useWorkflowsStore((s) => s.deleteWorkflow);
  const setCurrentWorkflow = useWorkflowsStore((s) => s.setCurrentWorkflow);

  const [showTrainingInput, setShowTrainingInput]     = useState(false);
  const [showDeploymentInput, setShowDeploymentInput] = useState(false);
  const [deploymentMode, setDeploymentMode]           = useState<'from-training' | 'from-scratch'>('from-scratch');
  const [trainingName, setTrainingName]               = useState('');
  const [deploymentName, setDeploymentName]           = useState('');
  const [selectedTrainingId, setSelectedTrainingId]    = useState('');

  const trainingWorkflows   = workflows.filter((w) => w.type === 'training');
  const deploymentWorkflows = workflows.filter((w) => w.type === 'deployment');
  const trainedWorkflows    = trainingWorkflows.filter((w) => w.trainedModels.length > 0);

  const handleCreateTraining = () => {
    if (!trainingName.trim()) return;
    const id = createWorkflow(trainingName.trim(), 'training');
    setTrainingName(''); setShowTrainingInput(false);
    navigate(`/editor/${id}`);
  };

  const handleCreateDeployment = () => {
    if (!deploymentName.trim()) return;
    let id: string | null;
    if (deploymentMode === 'from-training' && selectedTrainingId) {
      id = createDeploymentFromTraining(deploymentName.trim(), selectedTrainingId);
    } else {
      id = createWorkflow(deploymentName.trim(), 'deployment');
    }
    if (!id) return;
    setDeploymentName(''); setSelectedTrainingId(''); setShowDeploymentInput(false);
    navigate(`/editor/${id}`);
  };

  const handleEdit   = (id: string) => { setCurrentWorkflow(id); navigate(`/editor/${id}`); };
  const handleRun    = (id: string) => { setCurrentWorkflow(id); navigate(`/run/${id}`); };
  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm('Delete this workflow?')) deleteWorkflow(id);
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0a0a0f]">
      {/* Header */}
      <header className="h-12 bg-[#12121a] border-b border-[#22222e] flex items-center px-4 gap-4 shrink-0">
        <div className="flex items-center gap-2">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="8"  cy="8"  r="3" stroke="#00d4ff" strokeWidth="1.5" />
            <circle cx="16" cy="16" r="3" stroke="#00d4ff" strokeWidth="1.5" />
            <path d="M10.5 9.5L13.5 14.5" stroke="#00d4ff" strokeWidth="1.5" />
          </svg>
          <h1 className="text-gray-200 font-semibold">ML Workflow Studio</h1>
        </div>
        <div className="text-gray-500 text-sm">
          {workflows.length} workflow{workflows.length !== 1 ? 's' : ''}
        </div>
      </header>

      {/* Main */}
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-semibold text-gray-200 mb-8">Your Workflows</h2>

          {/* Training Workflows */}
          <Section
            title="Training Workflows"
            icon={<Brain className="w-5 h-5" />}
            accentColor="#00d4ff"
            workflows={trainingWorkflows}
            onEdit={handleEdit}
            onRun={handleRun}
            onDelete={handleDelete}
            newButton={
              <button
                onClick={() => { setShowDeploymentInput(false); setShowTrainingInput(true); }}
                className="flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-colors bg-[#00d4ff18] border-[#00d4ff55] text-[#00d4ff]"
              >
                <Plus className="w-4 h-4" />
                New Training
              </button>
            }
            inputRow={showTrainingInput ? (
              <div className="flex items-center gap-3 bg-[#12121a] border border-[#22222e] rounded-lg p-3 mb-3">
                <input
                  type="text"
                  value={trainingName}
                  onChange={(e) => setTrainingName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleCreateTraining()}
                  placeholder="Training workflow name…"
                  autoFocus
                  className="bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none w-64"
                />
                <button
                  onClick={handleCreateTraining}
                  className="px-4 py-2 rounded-md font-medium text-sm text-[#0a0a0f] bg-[#00d4ff] transition-colors"
                >
                  Create
                </button>
                <button onClick={() => { setShowTrainingInput(false); setTrainingName(''); }} className="px-4 py-2 text-gray-400 hover:text-gray-200 transition-colors text-sm">
                  Cancel
                </button>
              </div>
            ) : undefined}
          />

          {/* Deployment Workflows */}
          <Section
            title="Deployment Workflows"
            icon={<Rocket className="w-5 h-5" />}
            accentColor="#22c55e"
            workflows={deploymentWorkflows}
            onEdit={handleEdit}
            onRun={handleRun}
            onDelete={handleDelete}
            newButton={
              <button
                onClick={() => { setShowTrainingInput(false); setShowDeploymentInput(true); }}
                className="flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-colors bg-[#22c55e18] border-[#22c55e55] text-[#22c55e]"
              >
                <Plus className="w-4 h-4" />
                New Deployment
              </button>
            }
            inputRow={showDeploymentInput ? (
              <div className="bg-[#12121a] border border-[#22222e] rounded-lg p-3 mb-3 space-y-3">
                {/* Mode toggle */}
                <div className="flex gap-2">
                  <button
                    onClick={() => { setDeploymentMode('from-scratch'); setSelectedTrainingId(''); }}
                    className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                      deploymentMode === 'from-scratch'
                        ? 'bg-[#22c55e]/20 text-[#22c55e] border border-[#22c55e]/40'
                        : 'bg-[#1a1a24] text-gray-500 border border-[#2a2a38]'
                    }`}
                  >
                    From Scratch
                  </button>
                  <button
                    onClick={() => setDeploymentMode('from-training')}
                    disabled={trainedWorkflows.length === 0}
                    className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                      deploymentMode === 'from-training'
                        ? 'bg-[#22c55e]/20 text-[#22c55e] border border-[#22c55e]/40'
                        : 'bg-[#1a1a24] text-gray-500 border border-[#2a2a38]'
                    } disabled:opacity-40 disabled:cursor-not-allowed`}
                  >
                    From Training
                  </button>
                </div>
                <div className="flex items-center gap-3 flex-wrap">
                  {deploymentMode === 'from-training' && (
                    <select
                      value={selectedTrainingId}
                      onChange={(e) => setSelectedTrainingId(e.target.value)}
                      className="bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none"
                    >
                      <option value="">Select training workflow…</option>
                      {trainedWorkflows.map((w) => (
                        <option key={w.id} value={w.id}>
                          {w.name} ({w.trainedModels.length} model{w.trainedModels.length !== 1 ? 's' : ''})
                        </option>
                      ))}
                    </select>
                  )}
                  <input
                    type="text"
                    value={deploymentName}
                    onChange={(e) => setDeploymentName(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleCreateDeployment()}
                    placeholder="Deployment name…"
                    autoFocus
                    className="bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none w-48"
                  />
                  <button
                    onClick={handleCreateDeployment}
                    disabled={!deploymentName.trim() || (deploymentMode === 'from-training' && !selectedTrainingId)}
                    className="px-4 py-2 rounded-md font-medium text-sm text-[#0a0a0f] bg-[#22c55e] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Create
                  </button>
                  <button onClick={() => { setShowDeploymentInput(false); setDeploymentName(''); setSelectedTrainingId(''); }} className="px-4 py-2 text-gray-400 hover:text-gray-200 transition-colors text-sm">
                    Cancel
                  </button>
                </div>
              </div>
            ) : undefined}
          />
        </div>
      </div>
    </div>
  );
};
