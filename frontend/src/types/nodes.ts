import { NodeType, InputNodeType, ProcessNodeType, OutputNodeType } from '../store/workflowStore';
import {
  FileText, Image, AudioLines, Table,
  SplitSquareHorizontal, Layers, Eye, Scan,
  Mic, Activity, BarChart3, Save,
  Bot, Cpu, Globe,
  LucideIcon
} from 'lucide-react';

// ── Static model catalogues ───────────────────────────────────────────────────

export const EMBEDDING_MODELS = [
  { value: 'all-MiniLM-L6-v2',                     label: 'MiniLM L6 v2  (fast, 90 MB)' },
  { value: 'all-mpnet-base-v2',                     label: 'MPNet Base v2  (420 MB)' },
  { value: 'bge-small-en-v1.5',                     label: 'BGE Small EN v1.5  (130 MB)' },
  { value: 'bge-base-en-v1.5',                      label: 'BGE Base EN v1.5  (440 MB)' },
  { value: 'paraphrase-multilingual-MiniLM-L12-v2', label: 'Multilingual MiniLM L12  (470 MB)' },
];

export const IMAGE_CLASSIFIER_MODELS = [
  { value: 'resnet-50',     label: 'ResNet-50  (transfer learning)' },
  { value: 'convnext-tiny', label: 'ConvNeXt Tiny  (transfer learning)' },
  { value: 'resnet-18',     label: 'ResNet-18  (lightweight)' },
];

export const OBJECT_DETECT_MODELS = [
  { value: 'yolos-tiny',     label: 'YOLO-S Tiny  (30 MB, fastest)' },
  { value: 'rtdetr-r18vd',   label: 'RT-DETR R18  (150 MB)' },
  { value: 'detr-resnet-50', label: 'DETR ResNet-50  (160 MB)' },
];

export const AUDIO_MODELS = [
  { value: 'whisper-tiny',    label: 'Whisper Tiny  – Speech-to-Text' },
  { value: 'wav2vec2-base',   label: 'Wav2Vec2 Base  – Classification' },
  { value: 'wav2vec2-emotion', label: 'Wav2Vec2 SUPERB  – Emotion Recognition' },
];

export const CHUNKING_METHODS = [
  { value: 'auto',             label: 'Auto  (AI-Assisted)' },
  { value: 'sentence',         label: 'Sentence' },
  { value: 'paragraph',        label: 'Paragraph' },
  { value: 'sliding_window',   label: 'Sliding Window' },
  { value: 'fixed_size',       label: 'Fixed Size' },
  { value: 'recursive',        label: 'Recursive' },
  { value: 'markdown_headers', label: 'Markdown Headers' },
  { value: 'html_sections',    label: 'HTML Sections' },
  { value: 'code_blocks',      label: 'Code Blocks' },
  { value: 'csv_rows',         label: 'CSV Rows' },
  { value: 'json_objects',     label: 'JSON Objects' },
  { value: 'page',             label: 'Page  (PDF)' },
];

export const TABULAR_MODEL_TYPES = [
  { value: 'lstm', label: 'LSTM' },
  { value: 'gru',  label: 'GRU' },
  { value: 'rnn',  label: 'Vanilla RNN' },
  { value: 'ffnn', label: 'FFNN  (Feedforward)' },
  { value: 'dnn',  label: 'DNN  (Deep Neural Net)' },
];

export const AUDIO_TASKS = [
  { value: 'transcription',       label: 'Speech-to-Text' },
  { value: 'summarization',       label: 'Speech Summarization' },
  { value: 'emotion_recognition', label: 'Emotion Recognition' },
  { value: 'classification',      label: 'Audio Classification' },
];

export const BOUNDING_BOX_FORMATS = [
  { value: 'coco',   label: 'COCO JSON' },
  { value: 'yolo',   label: 'YOLO TXT' },
  { value: 'pascal', label: 'Pascal VOC XML' },
  { value: 'csv',    label: 'CSV  (x1,y1,x2,y2,label)' },
];

export const LLM_PROVIDERS = [
  { value: 'openai',        label: 'OpenAI – GPT-4o-mini' },
  { value: 'openai_large',  label: 'OpenAI – GPT-4o' },
  { value: 'ollama_qwen',   label: 'Ollama – Qwen 2.5 (0.5B)' },
  { value: 'ollama_llama',  label: 'Ollama – LLaMA 3.2 (1B)' },
  { value: 'ollama_custom', label: 'Ollama – Custom Model' },
];

export const FINE_TUNE_METHODS = [
  { value: 'simcse', label: 'SimCSE  (contrastive)' },
  { value: 'mnrl',   label: 'MNRL  (multiple negatives)' },
  { value: 'lora',   label: 'LoRA  (parameter-efficient)' },
  { value: 'sft',    label: 'SFT  (supervised fine-tuning)' },
];

export const POOLING_OPTIONS = [
  { value: 'max', label: 'Max Pooling' },
  { value: 'avg', label: 'Average Pooling' },
];

// ── Node definition ───────────────────────────────────────────────────────────

export interface NodeDefinition {
  type: NodeType;
  label: string;
  description: string;
  icon: LucideIcon;
  category: 'input' | 'processing' | 'output';
  color: string;
  /** Processing nodes that require this input type to be active */
  requiredInput?: InputNodeType;
  /** Processing nodes that require the image format to be detected */
  requiredImageFormat?: 'imagefolder' | 'boundingbox';
}

export const nodeDefinitions: NodeDefinition[] = [
  // ── Input ─────────────────────────────────────────────────────────────────
  {
    type: 'textInput',
    label: 'Text Input',
    description: 'Upload .txt / .pdf / .md / .docx files or pull from an API',
    icon: FileText,
    category: 'input',
    color: '#00d4ff',
  },
  {
    type: 'imageInput',
    label: 'Image Input',
    description: 'Upload image folder (classification) or bounding-box dataset',
    icon: Image,
    category: 'input',
    color: '#a855f7',
  },
  {
    type: 'audioInput',
    label: 'Audio Input',
    description: 'Upload .wav / .mp3 / .flac files for speech or classification',
    icon: AudioLines,
    category: 'input',
    color: '#22c55e',
  },
  {
    type: 'spreadsheetInput',
    label: 'Spreadsheet Input',
    description: 'Upload .csv / .xlsx / .json tabular data',
    icon: Table,
    category: 'input',
    color: '#f97316',
  },

  // ── Processing: Text ──────────────────────────────────────────────────────
  {
    type: 'chunkNode',
    label: 'Chunk',
    description: 'Split text into training chunks (11 strategies + AI-assisted auto)',
    icon: SplitSquareHorizontal,
    category: 'processing',
    color: '#2dd4bf',
    requiredInput: 'textInput',
  },
  {
    type: 'embeddingModel',
    label: 'Embedding Model',
    description: 'Choose a static embedding model; optionally fine-tune with SimCSE/LoRA',
    icon: Layers,
    category: 'processing',
    color: '#00d4ff',
    requiredInput: 'textInput',
  },

  // ── Processing: Image – classification ────────────────────────────────────
  {
    type: 'imageClassifier',
    label: 'Image Classifier',
    description: 'Fine-tune ResNet / ConvNeXt for classification (ImageFolder format)',
    icon: Eye,
    category: 'processing',
    color: '#a855f7',
    requiredInput: 'imageInput',
    requiredImageFormat: 'imagefolder',
  },
  {
    type: 'imageCNN',
    label: 'Custom CNN',
    description: 'Build a CNN from scratch – configure layers, filters, epochs',
    icon: Layers,
    category: 'processing',
    color: '#c084fc',
    requiredInput: 'imageInput',
    requiredImageFormat: 'imagefolder',
  },
  {
    type: 'imageCAE',
    label: 'Conv. AutoEncoder',
    description: 'Convolutional AutoEncoder for unsupervised feature learning',
    icon: Scan,
    category: 'processing',
    color: '#e879f9',
    requiredInput: 'imageInput',
    requiredImageFormat: 'imagefolder',
  },

  // ── Processing: Image – detection ─────────────────────────────────────────
  {
    type: 'objectDetector',
    label: 'Object Detector',
    description: 'Fine-tune YOLO / RT-DETR / DETR on bounding-box annotations',
    icon: Scan,
    category: 'processing',
    color: '#f472b6',
    requiredInput: 'imageInput',
    requiredImageFormat: 'boundingbox',
  },

  // ── Processing: Audio ─────────────────────────────────────────────────────
  {
    type: 'audioSpeechModel',
    label: 'Audio Speech Model',
    description: 'Fine-tune Whisper / Wav2Vec2 for STT, summarization, or emotion',
    icon: Mic,
    category: 'processing',
    color: '#22c55e',
    requiredInput: 'audioInput',
  },
  {
    type: 'audioCNN',
    label: 'Audio CNN',
    description: 'CNN trained on log-mel spectrograms for audio classification',
    icon: Activity,
    category: 'processing',
    color: '#4ade80',
    requiredInput: 'audioInput',
  },

  // ── Processing: Tabular ───────────────────────────────────────────────────
  {
    type: 'tabularModel',
    label: 'Tabular Neural Net',
    description: 'LSTM / GRU / RNN / FFNN / DNN – configure layers and neurons',
    icon: BarChart3,
    category: 'processing',
    color: '#fb923c',
    requiredInput: 'spreadsheetInput',
  },

  // ── Output ────────────────────────────────────────────────────────────────
  {
    type: 'saveModel',
    label: 'Save Model',
    description: 'Persist trained model to Cloudflare R2 and the Modal Volume',
    icon: Save,
    category: 'output',
    color: '#ef4444',
  },

  // ── Deployment nodes ──────────────────────────────────────────────────────
  {
    type: 'deployModelNode',
    label: 'Deploy Model',
    description: 'Select a trained or pretrained model for inference',
    icon: Cpu,
    category: 'processing',
    color: '#6366f1',
  },
  {
    type: 'llmNode',
    label: 'LLM Node',
    description: 'Augment model output with an LLM (OpenAI or Ollama)',
    icon: Bot,
    category: 'processing',
    color: '#ec4899',
  },
  {
    type: 'deployOutputNode',
    label: 'Deploy Output',
    description: 'Expose a persistent REST endpoint for your inference pipeline',
    icon: Globe,
    category: 'output',
    color: '#14b8a6',
  },
];

// Legacy exports kept for compatibility
export const modelOptions = EMBEDDING_MODELS;
export const chunkingStrategyOptions = CHUNKING_METHODS;
export const evalStrategyOptions = [
  { value: 'epoch', label: 'Every Epoch' },
  { value: 'steps', label: 'Every N Steps' },
  { value: 'no',    label: 'No Evaluation' },
];
export const parameterTypeOptions = [
  { value: 'string',  label: 'String' },
  { value: 'number',  label: 'Number' },
  { value: 'integer', label: 'Integer' },
  { value: 'boolean', label: 'Boolean' },
  { value: 'null',    label: 'Null' },
  { value: 'object',  label: 'Object' },
  { value: 'array',   label: 'Array' },
];
export const voiceTaskOptions = AUDIO_TASKS;

// Types for imports that use old names
export type { InputNodeType, ProcessNodeType, OutputNodeType };
