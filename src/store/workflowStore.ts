import { create } from 'zustand';
import { Node, Edge, Connection, addEdge, applyNodeChanges, applyEdgeChanges, NodeChange, EdgeChange } from '@xyflow/react';
import { v4 as uuidv4 } from 'uuid';

// Input node types
export type InputNodeType = 'textRetrieval' | 'agenticLLM' | 'visualData' | 'audioData' | 'voiceInput';

// Optimization node types
export type OptimizationNodeType =
  | 'agentTool'
  | 'rlhf'
  | 'rlaif'
  | 'subAgent'
  | 'chunkingOptimization'
  | 'visualHyperparamTuning'
  | 'audioHyperparamTuning'
  | 'voiceHyperparamTuning';

export type NodeType = InputNodeType | OptimizationNodeType;

// Tool parameter type for agent tools
export interface ToolParameter {
  id: string;
  name: string;
  type: 'string' | 'number' | 'integer' | 'boolean' | 'null' | 'object' | 'array';
  description: string;
  required: boolean;
  objectProperties?: { key: string; value: string }[];
  arrayValues?: string[];
}

export interface NodeData {
  label: string;
  type: NodeType;
  parameters: Record<string, any>;
  [key: string]: unknown;
}

export interface WorkflowNode extends Node {
  data: NodeData;
}

interface WorkflowState {
  nodes: WorkflowNode[];
  edges: Edge[];
  selectedNode: WorkflowNode | null;

  // Actions
  setNodes: (nodes: WorkflowNode[]) => void;
  setEdges: (edges: Edge[]) => void;
  addNode: (type: NodeType, position: { x: number; y: number }) => void;
  updateNodeData: (nodeId: string, data: Partial<NodeData>) => void;
  updateNodeParameters: (nodeId: string, parameters: Record<string, any>) => void;
  removeNode: (nodeId: string) => void;
  setSelectedNode: (node: WorkflowNode | null) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;

  // Helpers
  hasNodeOfType: (type: NodeType) => boolean;
  getInputNodeTypes: () => InputNodeType[];
}

const getDefaultParameters = (type: NodeType): Record<string, any> => {
  switch (type) {
    // Input nodes
    case 'textRetrieval':
      return {
        baseDecoderModel: 'gpt-5.2',
        vectorDBDataset: null,
        chunkingStrategy: 'recursive',
        chunkingSize: 512,
      };
    case 'agenticLLM':
      return {
        subAgentModel: 'gpt-5.2',
        subAgentPrompt: '',
      };
    case 'visualData':
      return {
        weightDecay: 0.01,
        epochCount: 10,
        dataset: null,
        learningRate: 0.001,
        evalStrategy: 'epoch',
      };
    case 'audioData':
      return {
        weightDecay: 0.01,
        epochCount: 10,
        dataset: null,
        learningRate: 0.001,
        evalStrategy: 'epoch',
      };
    case 'voiceInput':
      return {
        task: 'transcription',
        dataset: null,
      };

    // Optimization nodes for Agentic LLM
    case 'agentTool':
      return {
        functionName: '',
        functionDescription: '',
        parameters: [] as ToolParameter[],
      };
    case 'rlhf':
      return {
        iterations: 3,
      };
    case 'rlaif':
      return {
        evaluatorModel: 'gpt-5.2',
        iterations: 3,
      };
    case 'subAgent':
      return {
        subAgentModel: 'gpt-5.2',
        subAgentPrompt: '',
      };

    // Optimization nodes for Text Retrieval
    case 'chunkingOptimization':
      return {
        enableAutoOptimization: true,
        testStrategies: ['recursive', 'semantic', 'sentence'],
      };

    // Optimization nodes for Visual/Audio
    case 'visualHyperparamTuning':
      return {
        enableGridSearch: true,
        enableRandomSearch: false,
        enableBayesian: false,
      };
    case 'audioHyperparamTuning':
      return {
        enableGridSearch: true,
        enableRandomSearch: false,
        enableBayesian: false,
      };
    case 'voiceHyperparamTuning':
      return {
        enableGridSearch: true,
        enableRandomSearch: false,
        enableBayesian: false,
      };

    default:
      return {};
  }
};

const getNodeLabel = (type: NodeType): string => {
  switch (type) {
    // Input nodes
    case 'textRetrieval':
      return 'Text Retrieval (RAG)';
    case 'agenticLLM':
      return 'Agentic LLM Input';
    case 'visualData':
      return 'Visual Data (CV)';
    case 'audioData':
      return 'Audio (CNN)';
    case 'voiceInput':
      return 'Voice Input';

    // Optimization nodes
    case 'agentTool':
      return 'Agent Tool';
    case 'rlhf':
      return 'RLHF';
    case 'rlaif':
      return 'RLAIF';
    case 'subAgent':
      return 'Sub-Agent';
    case 'chunkingOptimization':
      return 'Chunking Optimization';
    case 'visualHyperparamTuning':
      return 'Hyperparameter Tuning';
    case 'audioHyperparamTuning':
      return 'Hyperparameter Tuning';
    case 'voiceHyperparamTuning':
      return 'Hyperparameter Tuning';

    default:
      return 'Unknown Node';
  }
};

export const useWorkflowStore = create<WorkflowState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNode: null,

  setNodes: (nodes) => {
    set({ nodes, selectedNode: null });
  },

  setEdges: (edges) => {
    set({ edges });
  },

  addNode: (type, position) => {
    const newNode: WorkflowNode = {
      id: uuidv4(),
      type: 'workflowNode',
      position,
      data: {
        label: getNodeLabel(type),
        type,
        parameters: getDefaultParameters(type),
      },
    };
    set((state) => ({
      nodes: [...state.nodes, newNode],
    }));
  },

  updateNodeData: (nodeId, data) => {
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...data } }
          : node
      ),
      selectedNode:
        state.selectedNode?.id === nodeId
          ? { ...state.selectedNode, data: { ...state.selectedNode.data, ...data } }
          : state.selectedNode,
    }));
  },

  updateNodeParameters: (nodeId, parameters) => {
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, parameters: { ...node.data.parameters, ...parameters } } }
          : node
      ),
      selectedNode:
        state.selectedNode?.id === nodeId
          ? {
              ...state.selectedNode,
              data: {
                ...state.selectedNode.data,
                parameters: { ...state.selectedNode.data.parameters, ...parameters }
              }
            }
          : state.selectedNode,
    }));
  },

  removeNode: (nodeId) => {
    set((state) => ({
      nodes: state.nodes.filter((node) => node.id !== nodeId),
      edges: state.edges.filter(
        (edge) => edge.source !== nodeId && edge.target !== nodeId
      ),
      selectedNode: state.selectedNode?.id === nodeId ? null : state.selectedNode,
    }));
  },

  setSelectedNode: (node) => {
    set({ selectedNode: node });
  },

  onNodesChange: (changes) => {
    set((state) => ({
      nodes: applyNodeChanges(changes, state.nodes) as WorkflowNode[],
    }));
  },

  onEdgesChange: (changes) => {
    set((state) => ({
      edges: applyEdgeChanges(changes, state.edges),
    }));
  },

  onConnect: (connection) => {
    set((state) => ({
      edges: addEdge(
        {
          ...connection,
          animated: true,
          style: { stroke: '#00d4ff', strokeWidth: 2 },
        },
        state.edges
      ),
    }));
  },

  hasNodeOfType: (type) => {
    return get().nodes.some((node) => node.data.type === type);
  },

  getInputNodeTypes: () => {
    const inputTypes: InputNodeType[] = ['textRetrieval', 'agenticLLM', 'visualData', 'audioData', 'voiceInput'];
    return get().nodes
      .filter((node) => inputTypes.includes(node.data.type as InputNodeType))
      .map((node) => node.data.type as InputNodeType);
  },
}));
