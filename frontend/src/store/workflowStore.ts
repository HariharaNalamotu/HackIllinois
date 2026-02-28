import { create } from 'zustand';
import { Node, Edge, Connection, addEdge, applyNodeChanges, applyEdgeChanges, NodeChange, EdgeChange } from '@xyflow/react';
import { v4 as uuidv4 } from 'uuid';

// ── Node type taxonomy ────────────────────────────────────────────────────────

/** The four mutually-exclusive input types */
export type InputNodeType = 'textInput' | 'imageInput' | 'audioInput' | 'spreadsheetInput';

/** Processing nodes — shown/enabled only when the relevant input is present */
export type ProcessNodeType =
  | 'chunkNode'        // text
  | 'embeddingModel'   // text
  | 'imageClassifier'  // image – ImageFolder
  | 'imageCNN'         // image – custom CNN
  | 'imageCAE'         // image – Convolutional AutoEncoder
  | 'objectDetector'   // image – BoundingBox
  | 'audioSpeechModel' // audio
  | 'audioCNN'         // audio – CNN on spectrograms
  | 'tabularModel'     // spreadsheet
  | 'deployModelNode'  // deployment: pick a trained/pretrained model
  | 'llmNode';         // deployment: LLM augmentation

/** Terminal output node */
export type OutputNodeType = 'saveModel' | 'deployOutputNode';

export type NodeType = InputNodeType | ProcessNodeType | OutputNodeType;

export interface NodeData {
  label: string;
  type: NodeType;
  parameters: Record<string, unknown>;
  [key: string]: unknown;
}

export interface WorkflowNode extends Node {
  data: NodeData;
}

interface WorkflowState {
  nodes: WorkflowNode[];
  edges: Edge[];
  selectedNode: WorkflowNode | null;

  setNodes: (nodes: WorkflowNode[]) => void;
  setEdges: (edges: Edge[]) => void;
  addNode: (type: NodeType, position: { x: number; y: number }) => void;
  updateNodeData: (nodeId: string, data: Partial<NodeData>) => void;
  updateNodeParameters: (nodeId: string, parameters: Record<string, unknown>) => void;
  removeNode: (nodeId: string) => void;
  setSelectedNode: (node: WorkflowNode | null) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;

  hasNodeOfType: (type: NodeType) => boolean;
  getInputNodeTypes: () => InputNodeType[];
  getActiveInputType: () => InputNodeType | null;
}

// ── Default parameters ────────────────────────────────────────────────────────

const getDefaultParameters = (type: NodeType): Record<string, unknown> => {
  switch (type) {
    case 'textInput':
      return { dataSource: 'upload', uploadedFiles: null, apiUrl: '' };
    case 'imageInput':
      return { dataSource: 'upload', imageFormat: 'unknown', uploadedFiles: null, apiUrl: '', detectedFormat: null };
    case 'audioInput':
      return { dataSource: 'upload', uploadedFiles: null, apiUrl: '' };
    case 'spreadsheetInput':
      return { dataSource: 'upload', uploadedFiles: null, apiUrl: '', targetColumn: '' };

    case 'chunkNode':
      return { method: 'auto', chunkSize: 512, overlap: 64 };
    case 'embeddingModel':
      return { model: 'all-MiniLM-L6-v2', fineTune: false, method: 'simcse', epochs: 3, learningRate: 3e-5, outputName: '' };

    case 'imageClassifier':
      return { baseModel: 'resnet-50', transfer: true, numClasses: 2, epochs: 10, batchSize: 32, learningRate: 1e-3, outputName: '' };
    case 'imageCNN':
      return { numLayers: 3, filters: '32,64,128', kernelSize: 3, pooling: 'max', numClasses: 2, epochs: 10, batchSize: 32, learningRate: 1e-3, outputName: '' };
    case 'imageCAE':
      return { numLayers: 3, filters: '32,64,128', latentDim: 256, epochs: 20, batchSize: 32, learningRate: 1e-3, outputName: '' };
    case 'objectDetector':
      return { baseModel: 'yolos-tiny', numClasses: 80, epochs: 10, batchSize: 8, learningRate: 5e-5, inputFormat: 'coco', outputName: '' };

    case 'audioSpeechModel':
      return { task: 'transcription', baseModel: 'whisper-tiny', epochs: 5, learningRate: 1e-4, numClasses: 2, outputName: '' };
    case 'audioCNN':
      return { numLayers: 3, filters: '32,64,128', kernelSize: 3, numClasses: 2, sampleRate: 16000, nMels: 80, epochs: 10, batchSize: 32, learningRate: 1e-3, outputName: '' };

    case 'tabularModel':
      return { modelType: 'lstm', targetColumn: '', numLayers: 2, hiddenDim: 128, numEpochs: 20, batchSize: 64, learningRate: 1e-3, bidirectional: false, outputName: '' };

    case 'deployModelNode':
      return { modelName: '', inputType: 'textInput' };

    case 'llmNode':
      return { provider: 'openai', model: 'gpt-4o-mini', systemPrompt: 'Describe the model output.', ollamaUrl: '' };

    case 'deployOutputNode':
      return { format: 'json' };

    case 'saveModel':
      return {};

    default:
      return {};
  }
};

const getNodeLabel = (type: NodeType): string => {
  const labels: Record<NodeType, string> = {
    textInput: 'Text Input',
    imageInput: 'Image Input',
    audioInput: 'Audio Input',
    spreadsheetInput: 'Spreadsheet Input',
    chunkNode: 'Chunk',
    embeddingModel: 'Embedding Model',
    imageClassifier: 'Image Classifier',
    imageCNN: 'Custom CNN',
    imageCAE: 'Conv. AutoEncoder',
    objectDetector: 'Object Detector',
    audioSpeechModel: 'Audio Speech Model',
    audioCNN: 'Audio CNN',
    tabularModel: 'Tabular Neural Net',
    deployModelNode: 'Deploy Model',
    llmNode: 'LLM Node',
    deployOutputNode: 'Deploy Output',
    saveModel: 'Save Model',
  };
  return labels[type] ?? 'Node';
};

const INPUT_TYPES: InputNodeType[] = ['textInput', 'imageInput', 'audioInput', 'spreadsheetInput'];

// ── Store ─────────────────────────────────────────────────────────────────────

export const useWorkflowStore = create<WorkflowState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNode: null,

  setNodes: (nodes) => set({ nodes, selectedNode: null }),
  setEdges: (edges) => set({ edges }),

  addNode: (type, position) => {
    const state = get();

    // Hard-block: only one input type at a time
    if (INPUT_TYPES.includes(type as InputNodeType)) {
      const existing = state.nodes.find((n) =>
        INPUT_TYPES.includes(n.data.type as InputNodeType)
      );
      if (existing && existing.data.type !== type) return;
    }

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
    set((s) => ({ nodes: [...s.nodes, newNode] }));
  },

  updateNodeData: (nodeId, data) => {
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === nodeId ? { ...n, data: { ...n.data, ...data } } : n
      ),
      selectedNode:
        s.selectedNode?.id === nodeId
          ? { ...s.selectedNode, data: { ...s.selectedNode.data, ...data } }
          : s.selectedNode,
    }));
  },

  updateNodeParameters: (nodeId, parameters) => {
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, parameters: { ...n.data.parameters, ...parameters } } }
          : n
      ),
      selectedNode:
        s.selectedNode?.id === nodeId
          ? {
              ...s.selectedNode,
              data: {
                ...s.selectedNode.data,
                parameters: { ...s.selectedNode.data.parameters, ...parameters },
              },
            }
          : s.selectedNode,
    }));
  },

  removeNode: (nodeId) => {
    const node = get().nodes.find((n) => n.id === nodeId);
    if (node?.data.type === 'saveModel') return;
    set((s) => ({
      nodes: s.nodes.filter((n) => n.id !== nodeId),
      edges: s.edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
      selectedNode: s.selectedNode?.id === nodeId ? null : s.selectedNode,
    }));
  },

  setSelectedNode: (node) => set({ selectedNode: node }),

  onNodesChange: (changes) => {
    const state = get();
    const filtered = changes.filter((c) => {
      if (c.type === 'remove') {
        const n = state.nodes.find((node) => node.id === c.id);
        return n?.data.type !== 'saveModel';
      }
      return true;
    });
    set((s) => ({ nodes: applyNodeChanges(filtered, s.nodes) as WorkflowNode[] }));
  },

  onEdgesChange: (changes) => {
    set((s) => ({ edges: applyEdgeChanges(changes, s.edges) }));
  },

  onConnect: (connection) => {
    set((s) => ({
      edges: addEdge(
        { ...connection, animated: true, style: { stroke: '#00d4ff', strokeWidth: 2 } },
        s.edges
      ),
    }));
  },

  hasNodeOfType: (type) => get().nodes.some((n) => n.data.type === type),

  getInputNodeTypes: () =>
    get()
      .nodes.filter((n) => INPUT_TYPES.includes(n.data.type as InputNodeType))
      .map((n) => n.data.type as InputNodeType),

  getActiveInputType: () => {
    const types = get()
      .nodes.filter((n) => INPUT_TYPES.includes(n.data.type as InputNodeType))
      .map((n) => n.data.type as InputNodeType);
    return types[0] ?? null;
  },
}));
