/**
 * pipelineBuilder.ts
 *
 * Converts frontend WorkflowNode[] + Edge[] into a PipelineSpec JSON
 * that the backend pipeline executor understands.
 */

import type { Edge } from '@xyflow/react';
import type { WorkflowNode } from '../store/workflowStore';

// ── Node type mapping: frontend → backend ─────────────────────────────────────

const NODE_TYPE_MAP: Record<string, string> = {
  // Input nodes
  textInput:        'text_input',
  imageInput:       'image_input',
  audioInput:       'audio_input',
  spreadsheetInput: 'spreadsheet_input',
  // Legacy combined nodes
  chunkNode:        'chunk',
  embeddingModel:   'text_model',
  imageClassifier:  'cnn_model',
  imageCNN:         'cnn_model',
  imageCAE:         'image_cae',
  objectDetector:   'object_detect_model',
  audioSpeechModel: 'audio_model',
  audioCNN:         'audio_cnn',
  tabularModel:     'tabular_model',
  // Chunk variants → 'chunk'
  chunkAuto:          'chunk',
  chunkSentence:      'chunk',
  chunkParagraph:     'chunk',
  chunkSlidingWindow: 'chunk',
  chunkFixedSize:     'chunk',
  chunkMarkdown:      'chunk',
  chunkRecursive:     'chunk',
  chunkCode:          'chunk',
  // Embedding variants → 'text_model'
  embeddingMiniLM:       'text_model',
  embeddingMPNet:        'text_model',
  embeddingBGESmall:     'text_model',
  embeddingBGEBase:      'text_model',
  embeddingMultilingual: 'text_model',
  // Image classifier variants → 'cnn_model'
  classifierResNet50: 'cnn_model',
  classifierConvNeXt: 'cnn_model',
  classifierResNet18: 'cnn_model',
  // Object detector variants → 'object_detect_model'
  detectorYOLOS: 'object_detect_model',
  detectorRTDETR: 'object_detect_model',
  detectorDETR:  'object_detect_model',
  // Audio speech variants → 'audio_model'
  audioWhisper:        'audio_model',
  audioWav2Vec2:       'audio_model',
  audioWav2Vec2Emotion: 'audio_model',
  // Tabular model variants → 'tabular_model'
  tabularLSTM: 'tabular_model',
  tabularGRU:  'tabular_model',
  tabularRNN:  'tabular_model',
  tabularFFNN: 'tabular_model',
  tabularDNN:  'tabular_model',
  // Output nodes
  saveModel:        'model_save',
  deployModelNode:  'infer_output',
  llmNode:          'api_output',
  deployOutputNode: 'infer_output',
};

// ── camelCase → snake_case for param keys ─────────────────────────────────────

function toSnakeCase(key: string): string {
  return key.replace(/([A-Z])/g, '_$1').toLowerCase();
}

// ── Parameter translation per node type ──────────────────────────────────────

function translateParams(
  frontendType: string,
  params: Record<string, unknown>
): Record<string, unknown> {
  const out: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === '') continue;
    // Skip file blobs — they're sent as multipart files separately
    if (value instanceof FileList || value instanceof File) continue;
    if (Array.isArray(value) && value[0] instanceof File) continue;

    const snakeKey = toSnakeCase(key);
    out[snakeKey] = value;
  }

  // Node-type-specific renames / additions
  switch (frontendType) {
    // ── Legacy combined nodes ────────────────────────────────────────────────
    case 'embeddingModel':
      if (params.model) out['base_model'] = params.model;
      if (params.fineTune && params.method) out['method'] = params.method;
      else if (!params.fineTune) out['method'] = 'simcse';
      break;

    case 'imageClassifier':
      out['base_model'] = params.baseModel;
      out['transfer']   = params.transfer !== false;
      break;

    case 'imageCNN':
      out['base_model'] = 'none';
      out['num_layers'] = params.numLayers;
      break;

    case 'imageCAE':
      out['latent_dim'] = params.latentDim;
      break;

    case 'objectDetector':
      out['base_model']   = params.baseModel;
      out['input_format'] = params.inputFormat;
      break;

    case 'audioSpeechModel':
      out['base_model'] = params.baseModel;
      out['task']       = params.task;
      break;

    case 'tabularModel':
      out['model_type']    = params.modelType;
      out['target_column'] = params.targetColumn;
      out['hidden_dim']    = params.hiddenDim;
      out['num_layers']    = params.numLayers;
      out['num_epochs']    = params.numEpochs;
      out['bidirectional'] = params.bidirectional;
      break;

    // ── Chunk variants ────────────────────────────────────────────────────────
    case 'chunkAuto':          out['method'] = 'auto'; break;
    case 'chunkSentence':      out['method'] = 'sentence'; break;
    case 'chunkParagraph':     out['method'] = 'paragraph'; break;
    case 'chunkSlidingWindow': out['method'] = 'sliding_window'; break;
    case 'chunkFixedSize':     out['method'] = 'fixed_size'; break;
    case 'chunkMarkdown':      out['method'] = 'markdown_headers'; break;
    case 'chunkRecursive':     out['method'] = 'recursive'; break;
    case 'chunkCode':          out['method'] = 'code_blocks'; break;

    // ── Embedding variants ────────────────────────────────────────────────────
    case 'embeddingMiniLM':
      out['base_model'] = 'sentence-transformers/all-MiniLM-L6-v2';
      if (params.fineTune && params.method) out['method'] = params.method;
      break;
    case 'embeddingMPNet':
      out['base_model'] = 'sentence-transformers/all-mpnet-base-v2';
      if (params.fineTune && params.method) out['method'] = params.method;
      break;
    case 'embeddingBGESmall':
      out['base_model'] = 'BAAI/bge-small-en-v1.5';
      if (params.fineTune && params.method) out['method'] = params.method;
      break;
    case 'embeddingBGEBase':
      out['base_model'] = 'BAAI/bge-base-en-v1.5';
      if (params.fineTune && params.method) out['method'] = params.method;
      break;
    case 'embeddingMultilingual':
      out['base_model'] = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2';
      if (params.fineTune && params.method) out['method'] = params.method;
      break;

    // ── Image classifier variants ─────────────────────────────────────────────
    case 'classifierResNet50':
      out['base_model'] = 'resnet-50';
      out['transfer']   = params.transfer !== false;
      break;
    case 'classifierConvNeXt':
      out['base_model'] = 'convnext-tiny';
      out['transfer']   = params.transfer !== false;
      break;
    case 'classifierResNet18':
      out['base_model'] = 'resnet-18';
      out['transfer']   = params.transfer !== false;
      break;

    // ── Object detector variants ──────────────────────────────────────────────
    case 'detectorYOLOS':
      out['base_model']   = 'yolos-tiny';
      out['input_format'] = params.inputFormat ?? 'coco';
      break;
    case 'detectorRTDETR':
      out['base_model']   = 'rtdetr-r18vd';
      out['input_format'] = params.inputFormat ?? 'coco';
      break;
    case 'detectorDETR':
      out['base_model']   = 'detr-resnet-50';
      out['input_format'] = params.inputFormat ?? 'coco';
      break;

    // ── Audio speech variants ─────────────────────────────────────────────────
    case 'audioWhisper':
      out['base_model'] = 'whisper-tiny';
      out['task']       = 'transcription';
      break;
    case 'audioWav2Vec2':
      out['base_model'] = 'wav2vec2-base';
      out['task']       = params.task ?? 'classification';
      break;
    case 'audioWav2Vec2Emotion':
      out['base_model'] = 'wav2vec2-emotion';
      out['task']       = 'emotion_recognition';
      break;

    // ── Tabular model variants ────────────────────────────────────────────────
    case 'tabularLSTM':
      out['model_type']    = 'lstm';
      out['target_column'] = params.targetColumn;
      out['hidden_dim']    = params.hiddenDim;
      out['num_layers']    = params.numLayers;
      out['num_epochs']    = params.numEpochs;
      out['bidirectional'] = params.bidirectional;
      break;
    case 'tabularGRU':
      out['model_type']    = 'gru';
      out['target_column'] = params.targetColumn;
      out['hidden_dim']    = params.hiddenDim;
      out['num_layers']    = params.numLayers;
      out['num_epochs']    = params.numEpochs;
      out['bidirectional'] = params.bidirectional;
      break;
    case 'tabularRNN':
      out['model_type']    = 'rnn';
      out['target_column'] = params.targetColumn;
      out['hidden_dim']    = params.hiddenDim;
      out['num_layers']    = params.numLayers;
      out['num_epochs']    = params.numEpochs;
      out['bidirectional'] = params.bidirectional;
      break;
    case 'tabularFFNN':
      out['model_type']    = 'ffnn';
      out['target_column'] = params.targetColumn;
      out['hidden_dim']    = params.hiddenDim;
      out['num_layers']    = params.numLayers;
      out['num_epochs']    = params.numEpochs;
      break;
    case 'tabularDNN':
      out['model_type']    = 'dnn';
      out['target_column'] = params.targetColumn;
      out['hidden_dim']    = params.hiddenDim;
      out['num_layers']    = params.numLayers;
      out['num_epochs']    = params.numEpochs;
      break;
  }

  return out;
}

// ── Main builder ──────────────────────────────────────────────────────────────

export interface BackendNodeSpec {
  id: string;
  type: string;
  params: Record<string, unknown>;
}

export interface BackendEdgeSpec {
  from: string;
  to: string;
}

export interface PipelineSpec {
  pipeline_type: 'train' | 'infer';
  nodes: BackendNodeSpec[];
  edges: BackendEdgeSpec[];
}

export function buildPipelineSpec(
  nodes: WorkflowNode[],
  edges: Edge[],
  pipelineType: 'train' | 'infer' = 'train'
): PipelineSpec {
  const backendNodes: BackendNodeSpec[] = nodes.map((n) => ({
    id:     n.id,
    type:   NODE_TYPE_MAP[n.data.type] ?? n.data.type,
    params: translateParams(n.data.type, n.data.parameters),
  }));

  const backendEdges: BackendEdgeSpec[] = edges.map((e) => ({
    from: e.source,
    to:   e.target,
  }));

  return {
    pipeline_type: pipelineType,
    nodes: backendNodes,
    edges: backendEdges,
  };
}

/** Collect all uploaded File objects from input nodes keyed by node id. */
export function collectFiles(nodes: WorkflowNode[]): Array<{ nodeId: string; file: File }> {
  const result: Array<{ nodeId: string; file: File }> = [];
  for (const node of nodes) {
    const uploaded = node.data.parameters.uploadedFiles;
    if (!uploaded) continue;
    if (uploaded instanceof FileList) {
      for (let i = 0; i < uploaded.length; i++) {
        result.push({ nodeId: node.id, file: uploaded[i] });
      }
    } else if (uploaded instanceof File) {
      result.push({ nodeId: node.id, file: uploaded });
    } else if (Array.isArray(uploaded)) {
      for (const f of uploaded as File[]) {
        if (f instanceof File) result.push({ nodeId: node.id, file: f });
      }
    }
  }
  return result;
}
