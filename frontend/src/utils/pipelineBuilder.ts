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
  textInput:        'text_input',
  imageInput:       'image_input',
  audioInput:       'audio_input',
  spreadsheetInput: 'spreadsheet_input',
  chunkNode:        'chunk',
  embeddingModel:   'text_model',
  imageClassifier:  'cnn_model',
  imageCNN:         'cnn_model',
  imageCAE:         'image_cae',
  objectDetector:   'object_detect_model',
  audioSpeechModel: 'audio_model',
  audioCNN:         'audio_cnn',
  tabularModel:     'tabular_model',
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
    case 'embeddingModel':
      // method: frontend stores 'simcse'/'mnrl'/'lora'/'sft'; backend uses same
      if (params.model) out['base_model'] = params.model;
      if (params.fineTune && params.method) out['method'] = params.method;
      else if (!params.fineTune) out['method'] = 'simcse'; // no fine-tune → simcse pass-through
      break;

    case 'imageClassifier':
      out['base_model'] = params.baseModel;
      out['transfer']   = params.transfer !== false;
      break;

    case 'imageCNN':
      out['base_model'] = 'none'; // from-scratch
      out['num_layers'] = params.numLayers;
      break;

    case 'imageCAE':
      out['latent_dim'] = params.latentDim;
      break;

    case 'objectDetector':
      out['base_model']    = params.baseModel;
      out['input_format']  = params.inputFormat;
      break;

    case 'audioSpeechModel':
      out['base_model'] = params.baseModel;
      out['task']       = params.task;
      break;

    case 'tabularModel':
      out['model_type']     = params.modelType;
      out['target_column']  = params.targetColumn;
      out['hidden_dim']     = params.hiddenDim;
      out['num_layers']     = params.numLayers;
      out['num_epochs']     = params.numEpochs;
      out['bidirectional']  = params.bidirectional;
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
