import React, { useCallback, useRef, useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  ReactFlowProvider,
  useReactFlow,
  SelectionMode,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { ArrowLeft, Play, Settings, Zap } from 'lucide-react';

import { NodePalette, WorkflowPaletteMode } from '../components/NodePalette';
import { PropertiesPanel } from '../components/PropertiesPanel';
import { TrainingModal } from '../components/TrainingModal';
import { WorkflowNode } from '../components/WorkflowNode';
import { useWorkflowStore, NodeType } from '../store/workflowStore';
import { useWorkflowsStore } from '../store/workflowsStore';
import { SettingsModal } from '../components/SettingsModal';
import { buildPipelineSpec, collectFiles } from '../utils/pipelineBuilder';
import { submitTrainingJob } from '../services/api';

const nodeTypes = {
  workflowNode: WorkflowNode,
};

function WorkflowCanvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  const nodes = useWorkflowStore((state) => state.nodes);
  const edges = useWorkflowStore((state) => state.edges);
  const onNodesChange = useWorkflowStore((state) => state.onNodesChange);
  const onEdgesChange = useWorkflowStore((state) => state.onEdgesChange);
  const onConnect = useWorkflowStore((state) => state.onConnect);
  const addNode = useWorkflowStore((state) => state.addNode);
  const removeNode = useWorkflowStore((state) => state.removeNode);
  const setSelectedNode = useWorkflowStore((state) => state.setSelectedNode);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow') as NodeType;
      if (!type) return;

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      addNode(type, position);
    },
    [screenToFlowPosition, addNode]
  );

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  // Double-click on a node to delete it (but not the output node)
  const onNodeDoubleClick = useCallback(
    (_event: React.MouseEvent, node: any) => {
      if (node.data?.type === 'saveModel') return;
      removeNode(node.id);
    },
    [removeNode]
  );

  // Double-click on an edge to delete it
  const onEdgeDoubleClick = useCallback(
    (_event: React.MouseEvent, edge: any) => {
      onEdgesChange([{ id: edge.id, type: 'remove' }]);
    },
    [onEdgesChange]
  );

  return (
    <div ref={reactFlowWrapper} className="flex-1 h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDragOver={onDragOver}
        onDrop={onDrop}
        onPaneClick={onPaneClick}
        onNodeDoubleClick={onNodeDoubleClick}
        onEdgeDoubleClick={onEdgeDoubleClick}
        nodeTypes={nodeTypes}
        fitView
        snapToGrid
        snapGrid={[15, 15]}
        panOnDrag
        selectionKeyCode="Shift"
        multiSelectionKeyCode="Shift"
        selectionOnDrag
        selectionMode={SelectionMode.Partial}
        defaultEdgeOptions={{
          animated: true,
          style: { stroke: '#00d4ff', strokeWidth: 2 },
        }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="#2a2a38"
        />
        <Controls
          showZoom
          showFitView
          showInteractive={false}
        />
        <MiniMap
          nodeColor={(node) => {
            const type = node.data?.type;
            switch (type) {
              case 'textInput':       return '#00d4ff';
              case 'imageInput':      return '#a855f7';
              case 'audioInput':      return '#22c55e';
              case 'spreadsheetInput': return '#f97316';
              case 'saveModel':       return '#ef4444';
              default:                return '#ffd700';
            }
          }}
          maskColor="rgba(0, 0, 0, 0.8)"
          style={{
            backgroundColor: '#1a1a24',
          }}
        />
      </ReactFlow>
    </div>
  );
}

function EditorHeader({
  workflowName,
  workflowId,
  workflowType,
}: {
  workflowName: string;
  workflowId: string;
  workflowType: 'training' | 'deployment';
}) {
  const navigate = useNavigate();
  const nodes = useWorkflowStore((state) => state.nodes);
  const edges = useWorkflowStore((state) => state.edges);
  const getInputNodeTypes = useWorkflowStore((state) => state.getInputNodeTypes);
  const [showSettings, setShowSettings] = useState(false);
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);

  const activeInputNodes = getInputNodeTypes();

  const processingNodeTypes = [
    'chunkNode', 'embeddingModel',
    'imageClassifier', 'imageCNN', 'imageCAE', 'objectDetector',
    'audioSpeechModel', 'audioCNN', 'tabularModel',
  ];
  const hasProcessingNode = nodes.some((n) => processingNodeTypes.includes(n.data.type));
  const trainDisabled = activeInputNodes.length === 0 || !hasProcessingNode;

  const handleTrainModel = async () => {
    if (trainDisabled) return;
    setTrainError(null);
    try {
      const pipelineSpec = buildPipelineSpec(nodes, edges, 'train');
      const files = collectFiles(nodes);
      const { job_id } = await submitTrainingJob(workflowId, pipelineSpec, files);
      setTrainingJobId(job_id);
    } catch (err: any) {
      setTrainError(err.message || 'Failed to submit training job');
    }
  };

  return (
    <>
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
          <h1 className="text-gray-200 font-semibold">{workflowName}</h1>
        </div>
        <div className="text-gray-500 text-sm">
          {nodes.length} node{nodes.length !== 1 ? 's' : ''} · {edges.length} connection{edges.length !== 1 ? 's' : ''}
        </div>

        <div className="ml-auto flex items-center gap-2">
          {trainError && (
            <span className="text-xs text-red-400 max-w-48 truncate" title={trainError}>
              {trainError}
            </span>
          )}

          <button
            onClick={() => setShowSettings(true)}
            className="p-2 hover:bg-[#1a1a24] rounded-lg text-gray-400 hover:text-gray-200 transition-colors"
            title="Settings"
          >
            <Settings className="w-5 h-5" />
          </button>

          {workflowType === 'training' && (
            <button
              onClick={handleTrainModel}
              disabled={trainDisabled}
              title={
                trainDisabled
                  ? activeInputNodes.length === 0
                    ? 'Add an input node to enable training'
                    : 'Add at least one processing node to enable training'
                  : 'Run Training Pipeline on GPU'
              }
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                trainDisabled
                  ? 'bg-[#1a1a24] text-gray-600 cursor-not-allowed'
                  : 'bg-[#22c55e] text-[#0a0a0f] hover:bg-[#16a34a]'
              }`}
            >
              <Play className="w-4 h-4" />
              Run Training
            </button>
          )}

          {workflowType === 'deployment' && (
            <button
              onClick={() => navigate(`/run/${workflowId}`)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-[#6366f1] text-white hover:bg-[#4f46e5] transition-colors"
            >
              <Zap className="w-4 h-4" />
              Test Inference
            </button>
          )}
        </div>
      </header>

      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}

      {trainingJobId && (
        <TrainingModal
          jobId={trainingJobId}
          workflowId={workflowId}
          onClose={() => setTrainingJobId(null)}
        />
      )}
    </>
  );
}

function EditorContent({ workflowId }: { workflowId: string }) {
  const navigate = useNavigate();
  const workflows = useWorkflowsStore((state) => state.workflows);
  const updateWorkflow = useWorkflowsStore((state) => state.updateWorkflow);
  const workflow = workflows.find((w) => w.id === workflowId);

  const nodes = useWorkflowStore((state) => state.nodes);
  const edges = useWorkflowStore((state) => state.edges);
  const setNodes = useWorkflowStore((state) => state.setNodes);
  const setEdges = useWorkflowStore((state) => state.setEdges);

  // Load workflow data on mount
  useEffect(() => {
    if (workflow) {
      setNodes(workflow.nodes);
      setEdges(workflow.edges);
    }
  }, [workflowId]);

  // Save workflow data on changes
  useEffect(() => {
    if (workflow) {
      const timeout = setTimeout(() => {
        updateWorkflow(workflowId, nodes, edges);
      }, 500);
      return () => clearTimeout(timeout);
    }
  }, [nodes, edges, workflowId, workflow, updateWorkflow]);

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

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0a0a0f]">
      <EditorHeader
        workflowName={workflow.name}
        workflowId={workflowId}
        workflowType={workflow.type}
      />
      <div className="flex-1 flex overflow-hidden">
        <NodePalette mode={workflow.type as WorkflowPaletteMode} />
        <WorkflowCanvas />
        <PropertiesPanel />
      </div>
    </div>
  );
}

export const EditorPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();

  if (!id) {
    return null;
  }

  return (
    <ReactFlowProvider>
      <EditorContent workflowId={id} />
    </ReactFlowProvider>
  );
};
