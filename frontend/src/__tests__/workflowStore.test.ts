import { describe, it, expect, beforeEach } from 'vitest';
import { useWorkflowStore } from '../store/workflowStore';

// Reset store between tests
beforeEach(() => {
  useWorkflowStore.setState({ nodes: [], edges: [], selectedNode: null });
});

describe('workflowStore', () => {
  it('starts empty', () => {
    const { nodes, edges, selectedNode } = useWorkflowStore.getState();
    expect(nodes).toEqual([]);
    expect(edges).toEqual([]);
    expect(selectedNode).toBeNull();
  });

  it('adds a node with correct defaults', () => {
    useWorkflowStore.getState().addNode('textInput', { x: 100, y: 200 });
    const { nodes } = useWorkflowStore.getState();
    expect(nodes).toHaveLength(1);
    expect(nodes[0].data.type).toBe('textInput');
    expect(nodes[0].data.label).toBe('Text Input');
    expect(nodes[0].data.parameters.dataSource).toBe('upload');
    expect(nodes[0].data.parameters.uploadedFiles).toBeNull();
    expect(nodes[0].position).toEqual({ x: 100, y: 200 });
  });

  it('blocks adding a second input type', () => {
    useWorkflowStore.getState().addNode('textInput', { x: 0, y: 0 });
    useWorkflowStore.getState().addNode('imageInput', { x: 100, y: 0 });
    const { nodes } = useWorkflowStore.getState();
    // Should still be 1 — imageInput was blocked because textInput exists
    expect(nodes).toHaveLength(1);
    expect(nodes[0].data.type).toBe('textInput');
  });

  it('allows adding same input type twice', () => {
    useWorkflowStore.getState().addNode('textInput', { x: 0, y: 0 });
    useWorkflowStore.getState().addNode('textInput', { x: 100, y: 0 });
    const { nodes } = useWorkflowStore.getState();
    // Same input type is allowed (no conflict check for same type)
    expect(nodes).toHaveLength(2);
  });

  it('allows adding process nodes alongside input nodes', () => {
    useWorkflowStore.getState().addNode('textInput', { x: 0, y: 0 });
    useWorkflowStore.getState().addNode('embeddingMiniLM', { x: 200, y: 0 });
    useWorkflowStore.getState().addNode('saveModel', { x: 400, y: 0 });
    expect(useWorkflowStore.getState().nodes).toHaveLength(3);
  });

  it('removes a node and its connected edges', () => {
    const store = useWorkflowStore.getState();
    store.addNode('textInput', { x: 0, y: 0 });
    store.addNode('embeddingMiniLM', { x: 200, y: 0 });
    const nodes = useWorkflowStore.getState().nodes;
    const srcId = nodes[0].id;
    const tgtId = nodes[1].id;

    // Manually add an edge
    useWorkflowStore.setState({
      edges: [{ id: 'e1', source: srcId, target: tgtId }],
    });
    expect(useWorkflowStore.getState().edges).toHaveLength(1);

    useWorkflowStore.getState().removeNode(srcId);
    expect(useWorkflowStore.getState().nodes).toHaveLength(1);
    expect(useWorkflowStore.getState().edges).toHaveLength(0); // edge removed too
  });

  it('updates node parameters (preserving other params)', () => {
    useWorkflowStore.getState().addNode('textInput', { x: 0, y: 0 });
    const id = useWorkflowStore.getState().nodes[0].id;

    const mockFiles = [new File(['hello'], 'test.txt', { type: 'text/plain' })];
    useWorkflowStore.getState().updateNodeParameters(id, { uploadedFiles: mockFiles });

    const params = useWorkflowStore.getState().nodes[0].data.parameters;
    expect(params.uploadedFiles).toEqual(mockFiles);
    expect(params.dataSource).toBe('upload'); // original param preserved
  });

  it('getActiveInputType returns the first input node type', () => {
    expect(useWorkflowStore.getState().getActiveInputType()).toBeNull();
    useWorkflowStore.getState().addNode('imageInput', { x: 0, y: 0 });
    expect(useWorkflowStore.getState().getActiveInputType()).toBe('imageInput');
  });

  it('hasNodeOfType works correctly', () => {
    expect(useWorkflowStore.getState().hasNodeOfType('textInput')).toBe(false);
    useWorkflowStore.getState().addNode('textInput', { x: 0, y: 0 });
    expect(useWorkflowStore.getState().hasNodeOfType('textInput')).toBe(true);
    expect(useWorkflowStore.getState().hasNodeOfType('imageInput')).toBe(false);
  });

  it('updateNodeParameters updates selectedNode if it matches', () => {
    useWorkflowStore.getState().addNode('textInput', { x: 0, y: 0 });
    const node = useWorkflowStore.getState().nodes[0];
    useWorkflowStore.getState().setSelectedNode(node);

    useWorkflowStore.getState().updateNodeParameters(node.id, { apiUrl: 'http://test.com' });
    const selected = useWorkflowStore.getState().selectedNode;
    expect(selected?.data.parameters.apiUrl).toBe('http://test.com');
  });
});
