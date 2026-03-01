import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Pencil, Play, Trash2 } from 'lucide-react';
import { useWorkflowsStore } from '../store/workflowsStore';

export const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const workflows = useWorkflowsStore((state) => state.workflows);
  const createWorkflow = useWorkflowsStore((state) => state.createWorkflow);
  const deleteWorkflow = useWorkflowsStore((state) => state.deleteWorkflow);
  const setCurrentWorkflow = useWorkflowsStore((state) => state.setCurrentWorkflow);

  const [showNewWorkflowInput, setShowNewWorkflowInput] = useState(false);
  const [newWorkflowName, setNewWorkflowName] = useState('');

  const handleCreateWorkflow = () => {
    if (newWorkflowName.trim()) {
      const id = createWorkflow(newWorkflowName.trim());
      setNewWorkflowName('');
      setShowNewWorkflowInput(false);
      navigate(`/editor/${id}`);
    }
  };

  const handleEditWorkflow = (id: string) => {
    setCurrentWorkflow(id);
    navigate(`/editor/${id}`);
  };

  const handleInferenceWorkflow = (id: string) => {
    setCurrentWorkflow(id);
    navigate(`/test/${id}`);
  };

  const handleDeleteWorkflow = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm('Are you sure you want to delete this workflow?')) {
      deleteWorkflow(id);
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0a0a0f]">
      {/* Header */}
      <header className="h-12 bg-[#12121a] border-b border-[#22222e] flex items-center px-4 gap-4">
        <div className="flex items-center gap-2">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="8" cy="8" r="3" stroke="#00d4ff" strokeWidth="1.5" />
            <circle cx="16" cy="16" r="3" stroke="#00d4ff" strokeWidth="1.5" />
            <path d="M10.5 9.5L13.5 14.5" stroke="#00d4ff" strokeWidth="1.5" />
          </svg>
          <h1 className="text-gray-200 font-semibold">ML Workflow Studio</h1>
        </div>
        <div className="text-gray-500 text-sm">
          {workflows.length} workflow{workflows.length !== 1 ? 's' : ''}
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-semibold text-gray-200 mb-6">Your Workflows</h2>

          {/* Workflow List */}
          <div className="space-y-3 mb-6">
            {workflows.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                <p className="mb-2">No workflows yet</p>
                <p className="text-sm">Create your first workflow to get started</p>
              </div>
            ) : (
              workflows.map((workflow) => (
                <div
                  key={workflow.id}
                  className="bg-[#12121a] border border-[#22222e] rounded-lg p-4 flex items-center justify-between hover:border-[#2a2a38] transition-colors"
                >
                  <div
                    className="flex-1 cursor-pointer"
                    onClick={() => handleEditWorkflow(workflow.id)}
                  >
                    <h3 className="text-gray-200 font-medium hover:text-[#00d4ff] transition-colors">{workflow.name}</h3>
                    <p className="text-gray-500 text-sm">
                      {workflow.nodes.length} node{workflow.nodes.length !== 1 ? 's' : ''} ·
                      Updated {new Date(workflow.updatedAt).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleEditWorkflow(workflow.id)}
                      className="group relative p-2 rounded-lg bg-[#1a1a24] hover:bg-[#22222e] text-gray-400 hover:text-[#00d4ff] transition-colors"
                    >
                      <Pencil className="w-4 h-4" />
                      <span className="absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-1 bg-[#22222e] text-xs text-gray-300 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                        Edit
                      </span>
                    </button>
                    <button
                      onClick={() => handleInferenceWorkflow(workflow.id)}
                      className="group relative p-2 rounded-lg bg-[#1a1a24] hover:bg-[#22222e] text-gray-400 hover:text-[#22c55e] transition-colors"
                    >
                      <Play className="w-4 h-4" />
                      <span className="absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-1 bg-[#22222e] text-xs text-gray-300 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                        Inference
                      </span>
                    </button>
                    <button
                      onClick={(e) => handleDeleteWorkflow(workflow.id, e)}
                      className="group relative p-2 rounded-lg bg-[#1a1a24] hover:bg-[#22222e] text-gray-400 hover:text-red-400 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                      <span className="absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-1 bg-[#22222e] text-xs text-gray-300 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                        Delete
                      </span>
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Add New Workflow */}
          <div className="flex justify-center">
            {showNewWorkflowInput ? (
              <div className="flex items-center gap-3 bg-[#12121a] border border-[#22222e] rounded-lg p-3">
                <input
                  type="text"
                  value={newWorkflowName}
                  onChange={(e) => setNewWorkflowName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleCreateWorkflow()}
                  placeholder="Workflow name..."
                  autoFocus
                  className="bg-[#1a1a24] border border-[#2a2a38] rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-[#00d4ff] w-64"
                />
                <button
                  onClick={handleCreateWorkflow}
                  className="px-4 py-2 bg-[#00d4ff] text-[#0a0a0f] rounded-md font-medium hover:bg-[#00b8d4] transition-colors"
                >
                  Create
                </button>
                <button
                  onClick={() => {
                    setShowNewWorkflowInput(false);
                    setNewWorkflowName('');
                  }}
                  className="px-4 py-2 text-gray-400 hover:text-gray-200 transition-colors"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowNewWorkflowInput(true)}
                className="flex items-center gap-2 px-6 py-3 bg-[#12121a] border border-[#22222e] rounded-lg text-gray-300 hover:border-[#00d4ff] hover:text-[#00d4ff] transition-colors"
              >
                <Plus className="w-5 h-5" />
                New Workflow
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
