import { NodeType, InputNodeType } from '../store/workflowStore';
import {
  FileText,
  Bot,
  Eye,
  AudioLines,
  Wrench,
  Brain,
  Sparkles,
  Users,
  SplitSquareHorizontal,
  SlidersHorizontal,
  Mic,
  LogOut,
  LucideIcon
} from 'lucide-react';

export interface NodeDefinition {
  type: NodeType;
  label: string;
  description: string;
  icon: LucideIcon;
  category: 'input' | 'optimization' | 'output';
  color: string;
  requiresInputNode?: InputNodeType; // Which input node enables this optimization
  requiresAnyInputNode?: InputNodeType[]; // Enabled when ANY of these input nodes are present
}

export const nodeDefinitions: NodeDefinition[] = [
  // ============ INPUT NODES ============
  {
    type: 'textRetrieval',
    label: 'Text Retrieval (RAG)',
    description: 'Generate text via RAG from a vector database for information retrieval',
    icon: FileText,
    category: 'input',
    color: '#00d4ff',
  },
  {
    type: 'agenticLLM',
    label: 'Agentic LLM Input',
    description: 'Multimodal input for agentic LLM orchestration workflows',
    icon: Bot,
    category: 'input',
    color: '#ff9500',
  },
  {
    type: 'visualData',
    label: 'Visual Data (CV)',
    description: 'Images & video input for computer vision model training',
    icon: Eye,
    category: 'input',
    color: '#a855f7',
  },
  {
    type: 'audioData',
    label: 'Audio (CNN)',
    description: 'Audio data for CNN-based audio processing models',
    icon: AudioLines,
    category: 'input',
    color: '#22c55e',
  },
  {
    type: 'voiceInput',
    label: 'Voice Input',
    description: 'Voice processing for transcription, emotion detection, and more',
    icon: Mic,
    category: 'input',
    color: '#f97316',
  },

  // ============ OPTIMIZATION NODES FOR AGENTIC LLM ============
  {
    type: 'agentTool',
    label: 'Agent Tools',
    description: 'Define callable tools for agentic workflow execution',
    icon: Wrench,
    category: 'optimization',
    color: '#ffd700',
    requiresInputNode: 'agenticLLM',
  },
  {
    type: 'rlhf',
    label: 'RLHF',
    description: 'Reinforcement learning with human feedback optimization',
    icon: Brain,
    category: 'optimization',
    color: '#f472b6',
    requiresInputNode: 'agenticLLM',
  },
  {
    type: 'rlaif',
    label: 'RLAIF',
    description: 'Reinforcement learning with AI feedback optimization',
    icon: Sparkles,
    category: 'optimization',
    color: '#818cf8',
    requiresInputNode: 'agenticLLM',
  },
  {
    type: 'subAgent',
    label: 'Sub-Agent',
    description: 'Add sub-agents for orchestrated agentic workflows',
    icon: Users,
    category: 'optimization',
    color: '#fb923c',
    requiresInputNode: 'agenticLLM',
  },

  // ============ OPTIMIZATION NODES FOR TEXT RETRIEVAL ============
  {
    type: 'chunkingOptimization',
    label: 'Chunking Optimization',
    description: 'Optimize chunking strategy for better retrieval performance',
    icon: SplitSquareHorizontal,
    category: 'optimization',
    color: '#2dd4bf',
    requiresInputNode: 'textRetrieval',
  },

  // ============ HYPERPARAMETER TUNING (TRADITIONAL ML ONLY) ============
  {
    type: 'hyperparamTuning',
    label: 'Hyperparameter Tuning',
    description: 'Automated hyperparameter optimization for traditional ML models',
    icon: SlidersHorizontal,
    category: 'optimization',
    color: '#c084fc',
    requiresAnyInputNode: ['visualData', 'audioData'],
  },

  // ============ OUTPUT NODE ============
  {
    type: 'output',
    label: 'Output',
    description: 'Final output endpoint for the workflow',
    icon: LogOut,
    category: 'output',
    color: '#ef4444',
  },
];

export const modelOptions = [
  { value: 'gpt-5.2', label: 'GPT-5.2' },
  { value: 'gpt-5-mini', label: 'GPT-5 Mini' },
  { value: 'gpt-5-nano', label: 'GPT-5 Nano' },
];

export const chunkingStrategyOptions = [
  { value: 'recursive', label: 'Recursive' },
  { value: 'character', label: 'Character' },
  { value: 'token', label: 'Token' },
  { value: 'sentence', label: 'Sentence' },
  { value: 'semantic', label: 'Semantic' },
];

export const evalStrategyOptions = [
  { value: 'epoch', label: 'Every Epoch' },
  { value: 'steps', label: 'Every N Steps' },
  { value: 'no', label: 'No Evaluation' },
];

export const parameterTypeOptions = [
  { value: 'string', label: 'String' },
  { value: 'number', label: 'Number' },
  { value: 'integer', label: 'Integer' },
  { value: 'boolean', label: 'Boolean' },
  { value: 'null', label: 'Null' },
  { value: 'object', label: 'Object' },
  { value: 'array', label: 'Array' },
];

export const voiceTaskOptions = [
  { value: 'transcription', label: 'Transcription' },
  { value: 'keyword_spotting', label: 'Keyword Spotting' },
  { value: 'emotion_detection', label: 'Emotion Detection' },
  { value: 'intent_classification', label: 'Intent Classification' },
  { value: 'gender_classification', label: 'Gender Classification' },
  { value: 'custom', label: 'Custom Task' },
];
