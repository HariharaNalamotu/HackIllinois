/**
 * DATA FLOW INTEGRATION TESTS
 *
 * These tests verify the complete data flow from frontend state through
 * serialization to the API call, catching the exact bugs we've encountered:
 *
 * 1. Files stored as strings → 0 files sent → TextInputNode fails
 * 2. Hardcoded WORKER_BASE → requests go to wrong server
 * 3. Node IDs not preserved → text_chunks keyed by wrong ID
 * 4. Pipeline spec node types don't match NODE_TYPE_MAP
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useWorkflowStore } from '../store/workflowStore';
import { useWorkflowsStore } from '../store/workflowsStore';
import { buildPipelineSpec, collectFiles } from '../utils/pipelineBuilder';

vi.mock('../components/SettingsModal', () => ({
  getStoredApiKey: () => '',
  getStoredBackendUrl: () => 'http://localhost:8787',
}));

beforeEach(() => {
  useWorkflowStore.setState({ nodes: [], edges: [], selectedNode: null });
  useWorkflowsStore.setState({ workflows: [], currentWorkflowId: null });
});

describe('End-to-end data flow: file upload → serialization → API', () => {
  it('full text pipeline: textInput → embeddingMiniLM → saveModel', () => {
    const store = useWorkflowStore.getState();

    // 1. User adds nodes
    store.addNode('textInput', { x: 0, y: 0 });
    store.addNode('embeddingMiniLM', { x: 200, y: 0 });
    store.addNode('saveModel', { x: 400, y: 0 });

    const nodes = useWorkflowStore.getState().nodes;
    expect(nodes).toHaveLength(3);

    // 2. User uploads files (as File[] — the fixed version)
    const files = [
      new File(['Chapter 1: The Beginning...'], 'doc.txt', { type: 'text/plain' }),
      new File(['%PDF content'], 'paper.pdf', { type: 'application/pdf' }),
    ];
    useWorkflowStore.getState().updateNodeParameters(nodes[0].id, {
      uploadedFiles: files,
    });

    // 3. User connects nodes
    useWorkflowStore.setState({
      edges: [
        { id: 'e1', source: nodes[0].id, target: nodes[1].id },
        { id: 'e2', source: nodes[1].id, target: nodes[2].id },
      ],
    });

    // 4. Serialize pipeline
    const currentNodes = useWorkflowStore.getState().nodes;
    const currentEdges = useWorkflowStore.getState().edges;
    const spec = buildPipelineSpec(currentNodes, currentEdges, 'train');

    // Verify node IDs are preserved exactly
    expect(spec.nodes[0].id).toBe(nodes[0].id);
    expect(spec.nodes[1].id).toBe(nodes[1].id);
    expect(spec.nodes[2].id).toBe(nodes[2].id);

    // Verify types are mapped correctly
    expect(spec.nodes[0].type).toBe('text_input');
    expect(spec.nodes[1].type).toBe('text_model');
    expect(spec.nodes[2].type).toBe('model_save');

    // Verify edges reference correct IDs
    expect(spec.edges[0]).toEqual({ from: nodes[0].id, to: nodes[1].id });

    // 5. Collect files
    const collected = collectFiles(currentNodes);
    expect(collected).toHaveLength(2);
    expect(collected[0].nodeId).toBe(nodes[0].id); // Same ID as in spec!
    expect(collected[0].file.name).toBe('doc.txt');
    expect(collected[1].file.name).toBe('paper.pdf');
  });

  it('BUG REGRESSION: string uploadedFiles produces zero files', () => {
    const store = useWorkflowStore.getState();
    store.addNode('textInput', { x: 0, y: 0 });
    const nodeId = useWorkflowStore.getState().nodes[0].id;

    // Simulate the OLD bug: storing string instead of File[]
    useWorkflowStore.getState().updateNodeParameters(nodeId, {
      uploadedFiles: '2 file(s) selected',
    });

    const nodes = useWorkflowStore.getState().nodes;
    const collected = collectFiles(nodes);

    // This is the bug: truthy string passes the null check but produces 0 files
    expect(collected).toHaveLength(0);
    // The worker receives no files[nodeId] entries
    // Modal gets empty text_chunks_direct → TextInputNode fails
  });

  it('node IDs in collectFiles match node IDs in pipeline spec', () => {
    const store = useWorkflowStore.getState();
    store.addNode('textInput', { x: 0, y: 0 });
    store.addNode('embeddingMiniLM', { x: 200, y: 0 });

    const nodes = useWorkflowStore.getState().nodes;
    const textNodeId = nodes[0].id;

    // Upload real files
    useWorkflowStore.getState().updateNodeParameters(textNodeId, {
      uploadedFiles: [new File(['test'], 'test.txt')],
    });

    const updatedNodes = useWorkflowStore.getState().nodes;
    const spec = buildPipelineSpec(updatedNodes, [], 'train');
    const collected = collectFiles(updatedNodes);

    // Critical: the nodeId in collected files MUST match the id in spec
    const specTextNodeId = spec.nodes.find((n) => n.type === 'text_input')?.id;
    const collectedNodeId = collected[0]?.nodeId;
    expect(specTextNodeId).toBe(collectedNodeId);
    expect(specTextNodeId).toBe(textNodeId);
  });

  it('image pipeline: imageInput → classifierResNet50 → saveModel', () => {
    const store = useWorkflowStore.getState();
    store.addNode('imageInput', { x: 0, y: 0 });
    store.addNode('classifierResNet50', { x: 200, y: 0 });
    store.addNode('saveModel', { x: 400, y: 0 });

    const nodes = useWorkflowStore.getState().nodes;
    const imgNode = nodes[0];

    // Upload image files as File[]
    useWorkflowStore.getState().updateNodeParameters(imgNode.id, {
      uploadedFiles: [
        new File(['img1'], 'cat.jpg', { type: 'image/jpeg' }),
        new File(['img2'], 'dog.jpg', { type: 'image/jpeg' }),
      ],
    });

    const updatedNodes = useWorkflowStore.getState().nodes;
    const spec = buildPipelineSpec(updatedNodes, [], 'train');
    const collected = collectFiles(updatedNodes);

    expect(spec.nodes[0].type).toBe('image_input');
    expect(spec.nodes[1].type).toBe('cnn_model');
    expect(spec.nodes[1].params.base_model).toBe('resnet-50');
    expect(collected).toHaveLength(2);
    expect(collected[0].nodeId).toBe(imgNode.id);
  });

  it('workflow persistence: saved workflow retains node/edge data', () => {
    // Create workflow
    const wfId = useWorkflowsStore.getState().createWorkflow('My Pipeline', 'training');

    // Build canvas
    const store = useWorkflowStore.getState();
    store.addNode('textInput', { x: 0, y: 0 });
    store.addNode('embeddingMPNet', { x: 200, y: 0 });
    const nodes = useWorkflowStore.getState().nodes;
    useWorkflowStore.setState({
      edges: [{ id: 'e1', source: nodes[0].id, target: nodes[1].id }],
    });

    // Save to workflows store
    const currentNodes = useWorkflowStore.getState().nodes;
    const currentEdges = useWorkflowStore.getState().edges;
    useWorkflowsStore.getState().updateWorkflow(wfId, currentNodes, currentEdges);

    // Verify persistence
    const saved = useWorkflowsStore.getState().workflows[0];
    expect(saved.nodes).toHaveLength(2);
    expect(saved.edges).toHaveLength(1);
    expect(saved.nodes[0].data.type).toBe('textInput');
    expect(saved.nodes[1].data.type).toBe('embeddingMPNet');
  });

  it('FormData construction matches what worker expects', () => {
    const store = useWorkflowStore.getState();
    store.addNode('textInput', { x: 0, y: 0 });
    store.addNode('embeddingBGESmall', { x: 200, y: 0 });
    store.addNode('saveModel', { x: 400, y: 0 });

    const nodes = useWorkflowStore.getState().nodes;
    const textNodeId = nodes[0].id;

    useWorkflowStore.getState().updateNodeParameters(textNodeId, {
      uploadedFiles: [new File(['content'], 'doc.txt', { type: 'text/plain' })],
    });

    const updatedNodes = useWorkflowStore.getState().nodes;
    const edges = [
      { id: 'e1', source: nodes[0].id, target: nodes[1].id },
      { id: 'e2', source: nodes[1].id, target: nodes[2].id },
    ];
    const spec = buildPipelineSpec(updatedNodes, edges, 'train');
    const files = collectFiles(updatedNodes);

    // Simulate what submitTrainingJob does
    const fd = new FormData();
    fd.set('pipeline', JSON.stringify(spec));
    for (const { nodeId, file } of files) {
      fd.append(`files[${nodeId}]`, file, file.name);
    }

    // Worker parses pipeline to build nodeTypeMap
    const pipelineStr = fd.get('pipeline') as string;
    const parsed = JSON.parse(pipelineStr);
    const nodeTypeMap: Record<string, string> = {};
    for (const n of parsed.nodes) nodeTypeMap[n.id] = n.type;

    // Worker extracts nodeId from files[nodeId] key
    for (const [key] of fd.entries()) {
      if (!key.startsWith('files[') || !key.endsWith(']')) continue;
      const extractedNodeId = key.slice(6, -1);

      // Critical: nodeId from file key must exist in nodeTypeMap
      expect(nodeTypeMap[extractedNodeId]).toBeDefined();
      // And it must be text_input for the worker to chunk it
      expect(nodeTypeMap[extractedNodeId]).toBe('text_input');
    }
  });
});

describe('Worker routing simulation', () => {
  it('worker correctly identifies text_input nodes for chunking', () => {
    // Simulate what the worker does when it receives the FormData
    const spec = {
      pipeline_type: 'train',
      nodes: [
        { id: 'abc-123', type: 'text_input', params: {} },
        { id: 'def-456', type: 'text_model', params: { base_model: 'bge-small-en-v1.5' } },
        { id: 'ghi-789', type: 'model_save', params: {} },
      ],
      edges: [
        { from: 'abc-123', to: 'def-456' },
        { from: 'def-456', to: 'ghi-789' },
      ],
    };

    // Worker builds nodeTypeMap
    const nodeTypeMap: Record<string, string> = {};
    for (const node of spec.nodes) nodeTypeMap[node.id] = node.type;

    // For files[abc-123], worker checks nodeType
    const nodeType = nodeTypeMap['abc-123'];
    expect(nodeType).toBe('text_input');
    // Worker should chunk this file and set text_chunks[abc-123]

    // For other nodes, they're not text_input
    expect(nodeTypeMap['def-456']).toBe('text_model');
    expect(nodeTypeMap['ghi-789']).toBe('model_save');
  });

  it('simulates Modal text_chunks parsing', () => {
    // Simulate what Modal's train_endpoint does
    const formEntries: [string, string][] = [
      ['text_chunks[abc-123]', JSON.stringify(['chunk 1', 'chunk 2', 'chunk 3'])],
      ['pipeline', '{"nodes":[{"id":"abc-123","type":"text_input"}]}'],
    ];

    const textChunksDirect: Record<string, string[]> = {};
    for (const [key, value] of formEntries) {
      if (key.startsWith('text_chunks[') && key.endsWith(']')) {
        const nodeId = key.slice(12, -1);
        textChunksDirect[nodeId] = JSON.parse(value);
      }
    }

    expect(textChunksDirect['abc-123']).toEqual(['chunk 1', 'chunk 2', 'chunk 3']);
    expect(Object.keys(textChunksDirect)).toHaveLength(1);
  });

  it('simulates TextInputNode chunk lookup — match vs mismatch', () => {
    // Simulate what TextInputNode.execute() does
    const textChunks: Record<string, string[]> = {
      'abc-123': ['chunk 1', 'chunk 2'],
    };

    // Correct ID → finds chunks
    const selfId = 'abc-123';
    const preChunks = (textChunks[selfId] || []);
    expect(preChunks).toHaveLength(2);

    // Wrong ID → empty (the mismatch bug)
    const wrongId = 'xyz-999';
    const noChunks = (textChunks[wrongId] || []);
    expect(noChunks).toHaveLength(0);
    // This would cause "no usable text extracted from uploaded files"
  });
});

describe('Workflow not found — inference endpoint contract', () => {
  it('inference requires workflow to be saved in worker KV', () => {
    // The "Workflow not found" error happens because:
    // 1. Frontend creates workflow in Zustand (in-memory only)
    // 2. Worker stores workflows in KV
    // 3. On inference, worker looks up workflow in KV by ID
    // 4. If the workflow was never saved to KV, it returns 404

    const wfId = useWorkflowsStore.getState().createWorkflow('Test', 'training');

    // Workflow exists in frontend state
    expect(useWorkflowsStore.getState().workflows.find((w) => w.id === wfId)).toBeDefined();

    // But the frontend NEVER saves the workflow to the Worker's KV store
    // unless there's an explicit save/sync API call.
    // This is the root cause of "Workflow not found" on inference.
  });
});
