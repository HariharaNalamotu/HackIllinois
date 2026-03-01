import { describe, it, expect, beforeEach } from 'vitest';
import { useWorkflowsStore } from '../store/workflowsStore';

beforeEach(() => {
  useWorkflowsStore.setState({ workflows: [], currentWorkflowId: null });
});

describe('workflowsStore', () => {
  it('creates a workflow and sets it as current', () => {
    const id = useWorkflowsStore.getState().createWorkflow('Test WF', 'training');
    expect(id).toBeTruthy();
    const { workflows, currentWorkflowId } = useWorkflowsStore.getState();
    expect(workflows).toHaveLength(1);
    expect(workflows[0].name).toBe('Test WF');
    expect(workflows[0].type).toBe('training');
    expect(workflows[0].trainedModels).toEqual([]);
    expect(currentWorkflowId).toBe(id);
  });

  it('deletes a workflow and clears currentWorkflowId if it was current', () => {
    const id = useWorkflowsStore.getState().createWorkflow('WF', 'training');
    useWorkflowsStore.getState().deleteWorkflow(id);
    const { workflows, currentWorkflowId } = useWorkflowsStore.getState();
    expect(workflows).toHaveLength(0);
    expect(currentWorkflowId).toBeNull();
  });

  it('renames a workflow', () => {
    const id = useWorkflowsStore.getState().createWorkflow('Old Name', 'training');
    useWorkflowsStore.getState().renameWorkflow(id, 'New Name');
    expect(useWorkflowsStore.getState().workflows[0].name).toBe('New Name');
  });

  it('getCurrentWorkflow returns the current workflow', () => {
    const id = useWorkflowsStore.getState().createWorkflow('WF', 'training');
    const wf = useWorkflowsStore.getState().getCurrentWorkflow();
    expect(wf?.id).toBe(id);
    expect(wf?.name).toBe('WF');
  });

  it('addTrainedModel adds model name without duplicates', () => {
    const id = useWorkflowsStore.getState().createWorkflow('WF', 'training');
    useWorkflowsStore.getState().addTrainedModel(id, 'model-abc');
    useWorkflowsStore.getState().addTrainedModel(id, 'model-abc'); // duplicate
    useWorkflowsStore.getState().addTrainedModel(id, 'model-xyz');
    const wf = useWorkflowsStore.getState().workflows[0];
    expect(wf.trainedModels).toEqual(['model-abc', 'model-xyz']);
  });

  it('updateWorkflow persists nodes and edges', () => {
    const id = useWorkflowsStore.getState().createWorkflow('WF', 'training');
    const nodes = [{ id: 'n1', type: 'workflowNode', position: { x: 0, y: 0 }, data: { label: 'Test', type: 'textInput' as const, parameters: {} } }];
    const edges = [{ id: 'e1', source: 'n1', target: 'n2' }];
    useWorkflowsStore.getState().updateWorkflow(id, nodes as any, edges);
    const wf = useWorkflowsStore.getState().workflows[0];
    expect(wf.nodes).toHaveLength(1);
    expect(wf.edges).toHaveLength(1);
  });

  it('getWorkflowInputTypes returns input node types from workflow nodes', () => {
    const id = useWorkflowsStore.getState().createWorkflow('WF', 'training');
    const nodes = [
      { id: 'n1', type: 'workflowNode', position: { x: 0, y: 0 }, data: { label: 'Text', type: 'textInput', parameters: {} } },
      { id: 'n2', type: 'workflowNode', position: { x: 0, y: 0 }, data: { label: 'Embed', type: 'embeddingMiniLM', parameters: {} } },
    ];
    useWorkflowsStore.getState().updateWorkflow(id, nodes as any, []);
    const types = useWorkflowsStore.getState().getWorkflowInputTypes(id);
    expect(types).toEqual(['textInput']);
  });
});
