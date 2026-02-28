import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid';
import { WorkflowNode, InputNodeType } from './workflowStore';
import { Edge } from '@xyflow/react';

export interface Workflow {
  id: string;
  name: string;
  type: 'training' | 'deployment';
  nodes: WorkflowNode[];
  edges: Edge[];
  trainedModels: string[];   // output model names saved by this workflow
  createdAt: number;
  updatedAt: number;
}

interface WorkflowsState {
  workflows: Workflow[];
  currentWorkflowId: string | null;

  createWorkflow: (name: string, type: 'training' | 'deployment') => string;
  deleteWorkflow: (id: string) => void;
  renameWorkflow: (id: string, name: string) => void;
  setCurrentWorkflow: (id: string | null) => void;
  getCurrentWorkflow: () => Workflow | null;
  updateWorkflow: (id: string, nodes: WorkflowNode[], edges: Edge[]) => void;
  addTrainedModel: (workflowId: string, modelName: string) => void;
  getWorkflowInputTypes: (workflowId: string) => InputNodeType[];
}

export const useWorkflowsStore = create<WorkflowsState>()(
  (set, get) => ({
      workflows: [],
      currentWorkflowId: null,

      createWorkflow: (name, type) => {
        const id = uuidv4();
        const now = Date.now();
        const newWorkflow: Workflow = {
          id,
          name,
          type,
          nodes: [],
          edges: [],
          trainedModels: [],
          createdAt: now,
          updatedAt: now,
        };
        set((state) => ({
          workflows: [...state.workflows, newWorkflow],
          currentWorkflowId: id,
        }));
        return id;
      },

      deleteWorkflow: (id) => {
        set((state) => ({
          workflows: state.workflows.filter((w) => w.id !== id),
          currentWorkflowId: state.currentWorkflowId === id ? null : state.currentWorkflowId,
        }));
      },

      renameWorkflow: (id, name) => {
        set((state) => ({
          workflows: state.workflows.map((w) =>
            w.id === id ? { ...w, name, updatedAt: Date.now() } : w
          ),
        }));
      },

      setCurrentWorkflow: (id) => {
        set({ currentWorkflowId: id });
      },

      getCurrentWorkflow: () => {
        const { workflows, currentWorkflowId } = get();
        return workflows.find((w) => w.id === currentWorkflowId) || null;
      },

      updateWorkflow: (id, nodes, edges) => {
        set((state) => ({
          workflows: state.workflows.map((w) =>
            w.id === id ? { ...w, nodes, edges, updatedAt: Date.now() } : w
          ),
        }));
      },

      addTrainedModel: (workflowId, modelName) => {
        set((state) => ({
          workflows: state.workflows.map((w) =>
            w.id === workflowId
              ? { ...w, trainedModels: [...new Set([...w.trainedModels, modelName])] }
              : w
          ),
        }));
      },

      getWorkflowInputTypes: (workflowId) => {
        const workflow = get().workflows.find((w) => w.id === workflowId);
        if (!workflow) return [];
        const inputTypes: InputNodeType[] = ['textInput', 'imageInput', 'audioInput', 'spreadsheetInput'];
        return workflow.nodes
          .filter((node) => inputTypes.includes(node.data.type as InputNodeType))
          .map((node) => node.data.type as InputNodeType);
      },
  })
);
