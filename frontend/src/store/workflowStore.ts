import { create } from 'zustand';
import { Node, Edge, Connection, addEdge, applyNodeChanges, applyEdgeChanges, NodeChange, EdgeChange } from '@xyflow/react';
import { v4 as uuidv4 } from 'uuid';

// ── Node type taxonomy ────────────────────────────────────────────────────────

/** The five mutually-exclusive input types */
export type InputNodeType = 'textInput' | 'imageInput' | 'audioInput' | 'spreadsheetInput' | 'agenticLLM';

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
  | 'llmNode'          // deployment: LLM augmentation
  // Agentic optimization nodes
  | 'agentTool'
  | 'rlhf'
  | 'rlaif'
  | 'subAgent'
  | 'chunkingOptimization'
  | 'hyperparamTuning'
  // Chunk variants
  | 'chunkAuto'
  | 'chunkSentence'
  | 'chunkParagraph'
  | 'chunkSlidingWindow'
  | 'chunkFixedSize'
  | 'chunkMarkdown'
  | 'chunkRecursive'
  | 'chunkCode'
  // Embedding variants
  | 'embeddingMiniLM'
  | 'embeddingMPNet'
  | 'embeddingBGESmall'
  | 'embeddingBGEBase'
  | 'embeddingMultilingual'
  // Image classifier variants
  | 'classifierResNet50'
  | 'classifierConvNeXt'
  | 'classifierResNet18'
  // Object detector variants
  | 'detectorYOLOS'
  | 'detectorRTDETR'
  | 'detectorDETR'
  // Audio speech variants
  | 'audioWhisper'
  | 'audioWav2Vec2'
  | 'audioWav2Vec2Emotion'
  // Tabular model variants
  | 'tabularLSTM'
  | 'tabularGRU'
  | 'tabularRNN'
  | 'tabularFFNN'
  | 'tabularDNN';

/** Terminal output node */
export type OutputNodeType = 'saveModel' | 'deployOutputNode';

export type NodeType = InputNodeType | ProcessNodeType | OutputNodeType;

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

    // Chunk variants
    case 'chunkAuto':
    case 'chunkSentence':
    case 'chunkParagraph':
    case 'chunkMarkdown':
    case 'chunkRecursive':
    case 'chunkCode':
      return {};
    case 'chunkSlidingWindow':
    case 'chunkFixedSize':
      return { chunkSize: 512, overlap: 64 };

    case 'embeddingModel':
      return { model: 'all-MiniLM-L6-v2', fineTune: true, method: 'simcse', epochs: 3, learningRate: 3e-5, outputName: '' };

    // Embedding variants
    case 'embeddingMiniLM':
    case 'embeddingMPNet':
    case 'embeddingBGESmall':
    case 'embeddingBGEBase':
    case 'embeddingMultilingual':
      return { fineTune: true, method: 'simcse', epochs: 3, learningRate: 3e-5, outputName: '' };

    case 'imageClassifier':
      return { baseModel: 'resnet-50', transfer: true, numClasses: 2, epochs: 10, batchSize: 32, learningRate: 1e-3, outputName: '' };
    case 'imageCNN':
      return { numLayers: 3, filters: '32,64,128', kernelSize: 3, pooling: 'max', numClasses: 2, epochs: 10, batchSize: 32, learningRate: 1e-3, outputName: '' };
    case 'imageCAE':
      return { numLayers: 3, filters: '32,64,128', latentDim: 256, epochs: 20, batchSize: 32, learningRate: 1e-3, outputName: '' };

    // Image classifier variants
    case 'classifierResNet50':
    case 'classifierConvNeXt':
    case 'classifierResNet18':
      return { transfer: true, numClasses: 2, epochs: 10, batchSize: 32, learningRate: 1e-3, outputName: '' };

    case 'objectDetector':
      return { baseModel: 'yolos-tiny', numClasses: 80, epochs: 10, batchSize: 8, learningRate: 5e-5, inputFormat: 'coco', outputName: '' };

    // Object detector variants
    case 'detectorYOLOS':
    case 'detectorRTDETR':
    case 'detectorDETR':
      return { numClasses: 80, epochs: 10, batchSize: 8, learningRate: 5e-5, inputFormat: 'coco', outputName: '' };

    case 'audioSpeechModel':
      return { task: 'transcription', baseModel: 'whisper-tiny', epochs: 5, learningRate: 1e-4, numClasses: 2, outputName: '' };
    case 'audioCNN':
      return { numLayers: 3, filters: '32,64,128', kernelSize: 3, numClasses: 2, sampleRate: 16000, nMels: 80, epochs: 10, batchSize: 32, learningRate: 1e-3, outputName: '' };

    // Audio speech variants
    case 'audioWhisper':
      return { epochs: 5, learningRate: 1e-4, outputName: '' };
    case 'audioWav2Vec2':
      return { task: 'classification', epochs: 5, learningRate: 1e-4, numClasses: 2, outputName: '' };
    case 'audioWav2Vec2Emotion':
      return { epochs: 5, learningRate: 1e-4, numClasses: 7, outputName: '' };

    case 'tabularModel':
      return { modelType: 'lstm', targetColumn: '', numLayers: 2, hiddenDim: 128, numEpochs: 20, batchSize: 64, learningRate: 1e-3, bidirectional: false, outputName: '' };

    // Tabular model variants
    case 'tabularLSTM':
    case 'tabularGRU':
    case 'tabularRNN':
      return { targetColumn: '', numLayers: 2, hiddenDim: 128, numEpochs: 20, batchSize: 64, learningRate: 1e-3, bidirectional: false, outputName: '' };
    case 'tabularFFNN':
    case 'tabularDNN':
      return { targetColumn: '', numLayers: 2, hiddenDim: 128, numEpochs: 20, batchSize: 64, learningRate: 1e-3, outputName: '' };

    // Agentic input
    case 'agenticLLM':
      return { subAgentModel: 'gpt-5.2', subAgentPrompt: '' };

    // Agentic optimization nodes
    case 'agentTool':
      return { functionName: '', functionDescription: '', parameters: [] as ToolParameter[] };
    case 'rlhf':
      return { iterations: 3 };
    case 'rlaif':
      return { evaluatorModel: 'gpt-5.2', iterations: 3 };
    case 'subAgent':
      return { subAgentModel: 'gpt-5.2', subAgentPrompt: '' };
    case 'chunkingOptimization':
      return { enableAutoOptimization: true, testStrategies: ['recursive', 'semantic', 'sentence'] };
    case 'hyperparamTuning':
      return { enableGridSearch: true, enableRandomSearch: false, enableBayesian: false };

    case 'deployModelNode':
      return { modelName: '', inputType: 'textInput' };

    case 'llmNode':
      return { provider: 'openai', model: 'gpt-4o-mini', systemPrompt: 'Describe the model output.', ollamaUrl: '' };

    case 'deployOutputNode':
      return { format: 'json' };

    case 'saveModel':
      return { modelName: '' };

    default:
      return {};
  }
};

const getNodeLabel = (type: NodeType): string => {
  const labels: Record<NodeType, string> = {
    // Inputs
    textInput: 'Text Input',
    imageInput: 'Image Input',
    audioInput: 'Audio Input',
    spreadsheetInput: 'Spreadsheet Input',
    // Legacy process nodes
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
    // Agentic
    agenticLLM: 'Agentic LLM Input',
    agentTool: 'Agent Tool',
    rlhf: 'RLHF',
    rlaif: 'RLAIF',
    subAgent: 'Sub-Agent',
    chunkingOptimization: 'Chunking Optimization',
    hyperparamTuning: 'Hyperparameter Tuning',
    // Outputs
    deployOutputNode: 'Deploy Output',
    saveModel: 'Save Model',
    // Chunk variants
    chunkAuto: 'Auto Chunk',
    chunkSentence: 'Sentence Chunk',
    chunkParagraph: 'Paragraph Chunk',
    chunkSlidingWindow: 'Sliding Window',
    chunkFixedSize: 'Fixed Size Chunk',
    chunkMarkdown: 'Markdown Chunk',
    chunkRecursive: 'Recursive Chunk',
    chunkCode: 'Code Chunk',
    // Embedding variants
    embeddingMiniLM: 'MiniLM L6 v2',
    embeddingMPNet: 'MPNet Base v2',
    embeddingBGESmall: 'BGE Small EN',
    embeddingBGEBase: 'BGE Base EN',
    embeddingMultilingual: 'Multilingual MiniLM',
    // Image classifier variants
    classifierResNet50: 'ResNet-50',
    classifierConvNeXt: 'ConvNeXt Tiny',
    classifierResNet18: 'ResNet-18',
    // Object detector variants
    detectorYOLOS: 'YOLO-S Tiny',
    detectorRTDETR: 'RT-DETR R18',
    detectorDETR: 'DETR ResNet-50',
    // Audio speech variants
    audioWhisper: 'Whisper Tiny',
    audioWav2Vec2: 'Wav2Vec2 Base',
    audioWav2Vec2Emotion: 'Wav2Vec2 Emotion',
    // Tabular model variants
    tabularLSTM: 'LSTM',
    tabularGRU: 'GRU',
    tabularRNN: 'Vanilla RNN',
    tabularFFNN: 'FFNN',
    tabularDNN: 'Deep NN',
  };
  return labels[type] ?? 'Node';
};

const INPUT_TYPES: InputNodeType[] = ['textInput', 'imageInput', 'audioInput', 'spreadsheetInput', 'agenticLLM'];

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
    set((s) => ({
      nodes: s.nodes.filter((n) => n.id !== nodeId),
      edges: s.edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
      selectedNode: s.selectedNode?.id === nodeId ? null : s.selectedNode,
    }));
  },

  setSelectedNode: (node) => set({ selectedNode: node }),

  onNodesChange: (changes) => {
    set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) as WorkflowNode[] }));
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
