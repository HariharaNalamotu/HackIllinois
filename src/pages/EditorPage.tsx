import React, { useCallback, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  ReactFlowProvider,
  useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { ArrowLeft } from 'lucide-react';

import { NodePalette } from '../components/NodePalette';
import { PropertiesPanel } from '../components/PropertiesPanel';
import { WorkflowNode } from '../components/WorkflowNode';
import { useWorkflowStore, NodeType } from '../store/workflowStore';
import { useWorkflowsStore } from '../store/workflowsStore';

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
        nodeTypes={nodeTypes}
        fitView
        snapToGrid
        snapGrid={[15, 15]}
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
              case 'textRetrieval':
                return '#00d4ff';
              case 'agenticLLM':
                return '#ff9500';
              case 'visualData':
                return '#a855f7';
              case 'audioData':
                return '#22c55e';
              case 'voiceInput':
                return '#f97316';
              default:
                return '#ffd700';
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

function EditorHeader({ workflowName }: { workflowName: string }) {
  const navigate = useNavigate();
  const nodes = useWorkflowStore((state) => state.nodes);
  const edges = useWorkflowStore((state) => state.edges);

  const handleTrainModel = () => {
    alert('Training would start here. This is a demo - in production, this would initiate the ML training pipeline.');
  };

  return (
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

      {/* Train Model Button */}
      <div className="ml-auto">
        <button
          onClick={handleTrainModel}
          className="flex items-center gap-2 px-4 py-2 bg-[#22c55e] text-[#0a0a0f] rounded-lg font-medium hover:bg-[#16a34a] transition-colors"
        >
          Train Model
        </button>
      </div>
    </header>
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
      <EditorHeader workflowName={workflow.name} />
      <div className="flex-1 flex overflow-hidden">
        <NodePalette />
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
