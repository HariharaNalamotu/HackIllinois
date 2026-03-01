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
import { ArrowLeft, Play, Settings, Loader2 } from 'lucide-react';

import { NodePalette } from '../components/NodePalette';
import { PropertiesPanel } from '../components/PropertiesPanel';
import { WorkflowNode } from '../components/WorkflowNode';
import { useWorkflowStore, NodeType, InputNodeType } from '../store/workflowStore';
import { useWorkflowsStore } from '../store/workflowsStore';
import { SettingsModal } from '../components/SettingsModal';

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
      if (node.data?.type === 'output') return;
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
              case 'output':
                return '#ef4444';
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

function EditorHeader({ workflowName, workflowId }: { workflowName: string; workflowId: string }) {
  const navigate = useNavigate();
  const nodes = useWorkflowStore((state) => state.nodes);
  const edges = useWorkflowStore((state) => state.edges);
  const getInputNodeTypes = useWorkflowStore((state) => state.getInputNodeTypes);
  const [showSettings, setShowSettings] = useState(false);

  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<string | null>(null);
  const [toolsGenerated, setToolsGenerated] = useState(false);

  const activeInputNodes = getInputNodeTypes();

  // Training-relevant nodes
  const trainingNodeTypes = ['rlhf', 'rlaif', 'hyperparamTuning'];
  const hasTrainingNodes = nodes.some((n) => trainingNodeTypes.includes(n.data.type));

  // LLM input nodes present
  const llmInputTypes: InputNodeType[] = ['agenticLLM', 'textRetrieval'];
  const hasLLMNodes = activeInputNodes.some((t) => llmInputTypes.includes(t));

  // Agent tools in an LLM workflow also enable training (code generation)
  const hasAgentTools = hasLLMNodes && nodes.some((n) => n.data.type === 'agentTool');

  // ML-only workflows (only visualData/audioData, no LLM nodes) need training first
  const mlOnlyInputTypes: InputNodeType[] = ['visualData', 'audioData', 'voiceInput'];
  const hasOnlyMLNodes = activeInputNodes.length > 0 && activeInputNodes.every((t) => mlOnlyInputTypes.includes(t));

  const trainDisabled = !hasTrainingNodes && !hasAgentTools;
  const needsToolGen = hasAgentTools && !toolsGenerated;
  const executeDisabled = hasOnlyMLNodes || activeInputNodes.length === 0 || needsToolGen;

  const handleTrainModel = async () => {
    if (trainDisabled || isTraining) return;

    setIsTraining(true);
    const statusParts: string[] = [];

    try {
      const { getStoredApiKey, getStoredTavilyKey, getStoredBackendUrl } = await import('../components/SettingsModal');
      const apiKey = getStoredApiKey();
      const tavilyKey = getStoredTavilyKey();
      const backendUrl = getStoredBackendUrl();

      if (!apiKey) {
        setTrainingStatus('Set your OpenAI API key in Settings first.');
        setTimeout(() => setTrainingStatus(null), 3000);
        setIsTraining(false);
        return;
      }

      // Check if any tool nodes likely need web search but no Tavily key is set
      if (hasAgentTools && !tavilyKey) {
        const webSearchKeywords = /\b(api|fetch|search|web|http|request|endpoint|url|scrape|crawl|query|lookup|retrieve|get data|external)\b/i;
        const toolNodes = nodes.filter((n) => n.data.type === 'agentTool' && n.data.parameters?.functionDescription);
        const needsSearch = toolNodes.some((n) => webSearchKeywords.test(n.data.parameters.functionDescription));
        if (needsSearch) {
          setTrainingStatus('Warning: Tool(s) may need web APIs but no Tavily key is set. Add one in Settings for better results.');
          await new Promise((r) => setTimeout(r, 3000));
        }
      }

      // Step 1: Generate tool code if agent tools are present
      if (hasAgentTools) {
        setTrainingStatus('Generating tool implementations...');
        try {
          const res = await fetch(`${backendUrl}/api/train`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-API-Key': apiKey,
              ...(tavilyKey ? { 'X-Tavily-Key': tavilyKey } : {}),
            },
            body: JSON.stringify({
              workflowConfig: { nodes, edges },
            }),
          });

          if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Training failed' }));
            statusParts.push(`Tool gen error: ${(err as any).error || 'Unknown'}`);
          } else {
            const result = await res.json() as { tools: { name: string; code: string }[] };
            const tools = result.tools || [];
            const succeeded = tools.filter((t) => !t.code.startsWith('// Error'));
            const failed = tools.filter((t) => t.code.startsWith('// Error'));
            if (succeeded.length > 0) {
              statusParts.push(`Generated: ${succeeded.map((t) => t.name).join(', ')}`);
              setToolsGenerated(true);
            }
            if (failed.length > 0) {
              statusParts.push(`Failed: ${failed.map((t) => t.name).join(', ')}`);
            }
          }
        } catch (err: any) {
          statusParts.push(`Tool gen error: ${err.message}`);
        }
      }

      // Step 2: Initialize RLHF/RLAIF/sub-agents
      const hasRLHF = nodes.some((n) => n.data.type === 'rlhf');
      const hasRLAIF = nodes.some((n) => n.data.type === 'rlaif');
      const hasSubAgents = nodes.some((n) => n.data.type === 'subAgent');
      const hasHyperparam = nodes.some((n) => n.data.type === 'hyperparamTuning');

      // RLHF initialization
      if (hasRLHF) {
        setTrainingStatus('Initializing RLHF session...');
        try {
          const res = await fetch(`${backendUrl}/api/train/rlhf`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
            body: JSON.stringify({ workflowConfig: { nodes, edges } }),
          });
          if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'RLHF init failed' }));
            statusParts.push(`RLHF error: ${(err as any).error || 'Unknown'}`);
          } else {
            const result = await res.json() as any;
            statusParts.push(result.session?.message || 'RLHF session started');
          }
        } catch (err: any) {
          statusParts.push(`RLHF error: ${err.message}`);
        }
      }

      // RLAIF — fully automated, runs from its own page (no init needed here)
      if (hasRLAIF) {
        statusParts.push('RLAIF ready — start automated training from the RLAIF page');
      }

      // Sub-agent registration and validation
      if (hasSubAgents) {
        setTrainingStatus('Registering sub-agents...');
        try {
          const res = await fetch(`${backendUrl}/api/train/subagents`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
            body: JSON.stringify({ workflowConfig: { nodes, edges } }),
          });
          if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Sub-agent init failed' }));
            statusParts.push(`Sub-agent error: ${(err as any).error || 'Unknown'}`);
          } else {
            const result = await res.json() as any;
            statusParts.push(result.message || 'Sub-agents registered');
          }
        } catch (err: any) {
          statusParts.push(`Sub-agent error: ${err.message}`);
        }
      }

      if (hasHyperparam) statusParts.push('Hyperparameter tuning initialized');

      const hasErrors = statusParts.some((s) => s.toLowerCase().includes('error'));
      setTrainingStatus(statusParts.join(' · ') || 'Training complete');

      // Redirect to RLHF training interface if RLHF was initialized successfully
      if (hasRLHF && !hasErrors) {
        setTrainingStatus('Redirecting to RLHF training...');
        setTimeout(() => navigate(`/rlhf/${workflowId}`), 1500);
        return;
      }

      // Redirect to RLAIF monitor if RLAIF was initialized successfully
      if (hasRLAIF && !hasErrors) {
        setTrainingStatus('Redirecting to RLAIF monitor...');
        setTimeout(() => navigate(`/rlaif/${workflowId}`), 1500);
        return;
      }
    } catch (err: any) {
      setTrainingStatus(`Error: ${err.message}`);
    }

    setTimeout(() => setTrainingStatus(null), 5000);
    setIsTraining(false);
  };

  const handleExecute = () => {
    if (executeDisabled) return;
    navigate(`/test/${workflowId}`);
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
          {/* Settings Button */}
          <button
            onClick={() => setShowSettings(true)}
            className="p-2 hover:bg-[#1a1a24] rounded-lg text-gray-400 hover:text-gray-200 transition-colors"
            title="Settings"
          >
            <Settings className="w-5 h-5" />
          </button>

          {/* Train Model Button */}
          <button
            onClick={handleTrainModel}
            disabled={trainDisabled || isTraining}
            title={trainDisabled ? 'Add agent tools, RLHF, RLAIF, or Hyperparameter Tuning nodes to enable training' : (hasAgentTools ? 'Generate tool implementations via Codex' : 'Train Model')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              trainDisabled
                ? 'bg-[#1a1a24] text-gray-600 cursor-not-allowed'
                : 'bg-[#22c55e] text-[#0a0a0f] hover:bg-[#16a34a]'
            }`}
          >
            {isTraining && <Loader2 className="w-4 h-4 animate-spin" />}
            {hasAgentTools && hasTrainingNodes ? 'Generate Tools & Train' : hasAgentTools ? 'Generate Tools' : 'Train Model'}
          </button>
          {trainingStatus && (
            <span className="text-xs text-gray-400 max-w-[200px] whitespace-normal break-words" title={trainingStatus}>
              {trainingStatus}
            </span>
          )}

          {/* Execute Button */}
          <button
            onClick={handleExecute}
            disabled={executeDisabled}
            title={executeDisabled ? (activeInputNodes.length === 0 ? 'Add input nodes first' : needsToolGen ? 'Generate tools first (Train Model)' : 'ML-only workflows need training first') : 'Execute Inference'}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              executeDisabled
                ? 'bg-[#1a1a24] text-gray-600 cursor-not-allowed'
                : 'bg-[#00d4ff] text-[#0a0a0f] hover:bg-[#00b8d4]'
            }`}
          >
            <Play className="w-4 h-4" />
            Execute
          </button>
        </div>
      </header>
      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}
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
  const addNode = useWorkflowStore((state) => state.addNode);
  const hasNodeOfType = useWorkflowStore((state) => state.hasNodeOfType);

  // Track latest nodes/edges in a ref so the unmount save always has current data
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  nodesRef.current = nodes;
  edgesRef.current = edges;

  // Guard: don't save until the workflow has been loaded at least once
  const hasLoadedRef = useRef(false);

  // Load workflow data on mount (and when switching workflows)
  useEffect(() => {
    hasLoadedRef.current = false;
    if (workflow) {
      setNodes(workflow.nodes);
      setEdges(workflow.edges);
      // Mark loaded after a tick so the save effect skips the current render's stale nodes
      requestAnimationFrame(() => { hasLoadedRef.current = true; });
    }
  }, [workflowId]);

  // Re-load if workflow becomes available after async persist rehydration
  useEffect(() => {
    if (workflow && !hasLoadedRef.current) {
      setNodes(workflow.nodes);
      setEdges(workflow.edges);
      requestAnimationFrame(() => { hasLoadedRef.current = true; });
    }
  }, [workflow?.updatedAt]);

  // Ensure an output node always exists
  useEffect(() => {
    if (hasLoadedRef.current && !hasNodeOfType('output')) {
      addNode('output', { x: 750, y: 300 });
    }
  }, [nodes.length]);

  // Save workflow data on changes (debounced), and flush on unmount
  useEffect(() => {
    if (!workflow || !hasLoadedRef.current) return;
    const timeout = setTimeout(() => {
      updateWorkflow(workflowId, nodes, edges);
    }, 500);
    return () => clearTimeout(timeout);
  }, [nodes, edges, workflowId, workflow, updateWorkflow]);

  // Flush save on unmount and on page refresh/close
  useEffect(() => {
    const flushSave = () => {
      if (!hasLoadedRef.current) return;
      updateWorkflow(workflowId, nodesRef.current, edgesRef.current);
    };
    window.addEventListener('beforeunload', flushSave);
    return () => {
      window.removeEventListener('beforeunload', flushSave);
      flushSave();
    };
  }, [workflowId, updateWorkflow]);

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
      <EditorHeader workflowName={workflow.name} workflowId={workflowId} />
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
