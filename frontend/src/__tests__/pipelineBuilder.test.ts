import { describe, it, expect } from 'vitest';
import { buildPipelineSpec, collectFiles } from '../utils/pipelineBuilder';
import type { WorkflowNode } from '../store/workflowStore';

function makeNode(id: string, type: string, params: Record<string, unknown> = {}): WorkflowNode {
  return {
    id,
    type: 'workflowNode',
    position: { x: 0, y: 0 },
    data: { label: type, type: type as any, parameters: params },
  };
}

describe('buildPipelineSpec', () => {
  it('maps frontend node types to backend types', () => {
    const nodes: WorkflowNode[] = [
      makeNode('n1', 'textInput'),
      makeNode('n2', 'embeddingMiniLM', { fineTune: true, method: 'simcse' }),
      makeNode('n3', 'saveModel'),
    ];
    const edges = [
      { id: 'e1', source: 'n1', target: 'n2' },
      { id: 'e2', source: 'n2', target: 'n3' },
    ];

    const spec = buildPipelineSpec(nodes, edges, 'train');

    expect(spec.pipeline_type).toBe('train');
    expect(spec.nodes).toHaveLength(3);
    expect(spec.nodes[0]).toEqual({ id: 'n1', type: 'text_input', params: {} });
    expect(spec.nodes[1].type).toBe('text_model');
    expect(spec.nodes[1].params.base_model).toBe('sentence-transformers/all-MiniLM-L6-v2');
    expect(spec.nodes[1].params.method).toBe('simcse');
    expect(spec.nodes[2].type).toBe('model_save');

    expect(spec.edges).toEqual([
      { from: 'n1', to: 'n2' },
      { from: 'n2', to: 'n3' },
    ]);
  });

  it('preserves node IDs exactly', () => {
    const id = 'my-custom-uuid-123';
    const spec = buildPipelineSpec([makeNode(id, 'audioInput')], [], 'train');
    expect(spec.nodes[0].id).toBe(id);
  });

  it('maps all image classifier variants', () => {
    const variants = [
      { frontend: 'classifierResNet50', backendModel: 'resnet-50' },
      { frontend: 'classifierConvNeXt', backendModel: 'convnext-tiny' },
      { frontend: 'classifierResNet18', backendModel: 'resnet-18' },
    ];
    for (const { frontend, backendModel } of variants) {
      const spec = buildPipelineSpec([makeNode('x', frontend)], [], 'train');
      expect(spec.nodes[0].type).toBe('cnn_model');
      expect(spec.nodes[0].params.base_model).toBe(backendModel);
    }
  });

  it('maps chunk variants to chunk type with correct method param', () => {
    const cases = [
      { frontend: 'chunkAuto', method: 'auto' },
      { frontend: 'chunkSentence', method: 'sentence' },
      { frontend: 'chunkParagraph', method: 'paragraph' },
      { frontend: 'chunkSlidingWindow', method: 'sliding_window' },
      { frontend: 'chunkFixedSize', method: 'fixed_size' },
      { frontend: 'chunkMarkdown', method: 'markdown_headers' },
      { frontend: 'chunkRecursive', method: 'recursive' },
      { frontend: 'chunkCode', method: 'code_blocks' },
    ];
    for (const { frontend, method } of cases) {
      const spec = buildPipelineSpec([makeNode('x', frontend)], [], 'train');
      expect(spec.nodes[0].type).toBe('chunk');
      expect(spec.nodes[0].params.method).toBe(method);
    }
  });

  it('converts camelCase params to snake_case', () => {
    const node = makeNode('t', 'tabularLSTM', {
      targetColumn: 'price',
      hiddenDim: 256,
      numLayers: 3,
      numEpochs: 50,
      bidirectional: true,
    });
    const spec = buildPipelineSpec([node], [], 'train');
    const p = spec.nodes[0].params;
    expect(p.target_column).toBe('price');
    expect(p.hidden_dim).toBe(256);
    expect(p.num_layers).toBe(3);
    expect(p.model_type).toBe('lstm');
    expect(p.bidirectional).toBe(true);
  });

  it('skips null/empty params but keeps valid strings', () => {
    const node = makeNode('t', 'textInput', { dataSource: 'upload', uploadedFiles: null, apiUrl: '' });
    const spec = buildPipelineSpec([node], [], 'train');
    // null and '' are skipped, but 'upload' is a valid value
    expect(spec.nodes[0].params).toEqual({ data_source: 'upload' });
    expect(spec.nodes[0].params).not.toHaveProperty('uploaded_files');
    expect(spec.nodes[0].params).not.toHaveProperty('api_url');
  });
});

describe('collectFiles', () => {
  it('returns empty array when no files uploaded', () => {
    const node = makeNode('n1', 'textInput', { uploadedFiles: null });
    expect(collectFiles([node])).toEqual([]);
  });

  it('collects File[] from uploadedFiles', () => {
    const f1 = new File(['hello'], 'test.txt', { type: 'text/plain' });
    const f2 = new File(['world'], 'test2.txt', { type: 'text/plain' });
    const node = makeNode('n1', 'textInput', { uploadedFiles: [f1, f2] });
    const result = collectFiles([node]);
    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({ nodeId: 'n1', file: f1 });
    expect(result[1]).toEqual({ nodeId: 'n1', file: f2 });
  });

  it('collects single File from uploadedFiles', () => {
    const f = new File(['data'], 'data.pdf');
    const node = makeNode('n1', 'textInput', { uploadedFiles: f });
    const result = collectFiles([node]);
    expect(result).toHaveLength(1);
    expect(result[0].file).toBe(f);
  });

  it('REJECTS string uploadedFiles — this was the critical bug', () => {
    // This is the bug that caused "no usable text extracted from uploaded files"
    // uploadedFiles was set to a string like "3 file(s) selected" instead of File[]
    const node = makeNode('n1', 'textInput', { uploadedFiles: '3 file(s) selected' });
    const result = collectFiles([node]);
    // String is truthy so it passes the `if (!uploaded) continue` check,
    // but it's not FileList, File, or Array<File> so nothing gets pushed.
    expect(result).toEqual([]); // BUG: files silently lost when stored as string
  });

  it('collects files from multiple nodes', () => {
    const f1 = new File(['a'], 'a.txt');
    const f2 = new File(['b'], 'b.wav');
    const nodes = [
      makeNode('text-1', 'textInput', { uploadedFiles: [f1] }),
      makeNode('audio-1', 'audioInput', { uploadedFiles: [f2] }),
    ];
    const result = collectFiles(nodes);
    expect(result).toHaveLength(2);
    expect(result[0].nodeId).toBe('text-1');
    expect(result[1].nodeId).toBe('audio-1');
  });
});
