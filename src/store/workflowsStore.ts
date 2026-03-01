import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { v4 as uuidv4 } from 'uuid';
import { WorkflowNode, InputNodeType } from './workflowStore';
import { Edge } from '@xyflow/react';

export interface Workflow {
  id: string;
  name: string;
  nodes: WorkflowNode[];
  edges: Edge[];
  createdAt: number;
  updatedAt: number;
}

interface WorkflowsState {
  workflows: Workflow[];
  currentWorkflowId: string | null;

  // Actions
  createWorkflow: (name: string) => string;
  deleteWorkflow: (id: string) => void;
  renameWorkflow: (id: string, name: string) => void;
  setCurrentWorkflow: (id: string | null) => void;
  getCurrentWorkflow: () => Workflow | null;
  updateWorkflow: (id: string, nodes: WorkflowNode[], edges: Edge[]) => void;

  // Helpers
  getWorkflowInputTypes: (workflowId: string) => InputNodeType[];
}

export const useWorkflowsStore = create<WorkflowsState>()(
  persist(
    (set, get) => ({
      workflows: [],
      currentWorkflowId: null,

      createWorkflow: (name) => {
        const id = uuidv4();
        const now = Date.now();
        const newWorkflow: Workflow = {
          id,
          name,
          nodes: [],
          edges: [],
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

      getWorkflowInputTypes: (workflowId) => {
        const workflow = get().workflows.find((w) => w.id === workflowId);
        if (!workflow) return [];

        const inputTypes: InputNodeType[] = ['textRetrieval', 'agenticLLM', 'visualData', 'audioData', 'voiceInput'];
        return workflow.nodes
          .filter((node) => inputTypes.includes(node.data.type as InputNodeType))
          .map((node) => node.data.type as InputNodeType);
      },
    }),
    {
      name: 'ml-workflows-storage',
    }
  )
);
