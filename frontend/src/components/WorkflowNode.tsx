import React, { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { GripHorizontal } from 'lucide-react';
import { NodeData, useWorkflowStore, InputNodeType } from '../store/workflowStore';
import { nodeDefinitions } from '../types/nodes';

type WorkflowNodeProps = NodeProps & {
  data: NodeData;
};

const INPUT_TYPES: InputNodeType[] = ['textInput', 'imageInput', 'audioInput', 'spreadsheetInput', 'agenticLLM'];

const AGENTIC_OPTIMIZATION_TYPES = new Set(['agentTool', 'subAgent']);
const NON_AGENTIC_OPTIMIZATION_TYPES = new Set(['chunkingOptimization', 'hyperparamTuning', 'rlhf', 'rlaif']);
const OPTIMIZATION_TYPES = new Set([...AGENTIC_OPTIMIZATION_TYPES, ...NON_AGENTIC_OPTIMIZATION_TYPES]);

const WorkflowNodeComponent: React.FC<WorkflowNodeProps> = ({ id, data, selected }) => {
  const setSelectedNode = useWorkflowStore((state) => state.setSelectedNode);
  const nodes = useWorkflowStore((state) => state.nodes);

  const definition = nodeDefinitions.find((n) => n.type === data.type);
  const Icon = definition?.icon;
  const color = definition?.color || '#00d4ff';

  const isInput = INPUT_TYPES.includes(data.type as InputNodeType);
  const isAgenticInput = data.type === 'agenticLLM';
  const isOutput = data.type === 'saveModel' || data.type === 'deployOutputNode';
  const isOptimization = OPTIMIZATION_TYPES.has(data.type);
  const isAgenticOpt = AGENTIC_OPTIMIZATION_TYPES.has(data.type);
  const isProcessing = !isInput && !isOutput;

  const handleClick = () => {
    const node = nodes.find((n) => n.id === id);
    if (node) setSelectedNode(node);
  };

  // Build a compact summary of key parameters to show on the node card
  const getDisplayParams = () => {
    const params = data.parameters;
    // Pick the most meaningful params for each node type (skip null/empty/false)
    const skip = new Set(['uploadedFiles', 'detectedFormat', 'apiUrl', 'outputName']);
    return Object.entries(params)
      .filter(([k, v]) => !skip.has(k) && v !== null && v !== '' && v !== false)
      .slice(0, 4)
      .map(([key, value]) => {
        const formattedKey = key
          .replace(/([A-Z])/g, ' $1')
          .replace(/^./, (s) => s.toUpperCase())
          .trim();
        let displayValue: string;
        if (typeof value === 'boolean') displayValue = value ? 'Yes' : 'No';
        else if (Array.isArray(value)) displayValue = `${value.length} items`;
        else if (typeof value === 'object') displayValue = 'Configured';
        else if (typeof value === 'number') displayValue = String(value);
        else if (typeof value === 'string' && value.length > 18) displayValue = value.slice(0, 18) + '…';
        else displayValue = String(value);
        return { key: formattedKey, value: displayValue };
      });
  };

  const displayParams = getDisplayParams();

  return (
    <div
      onClick={handleClick}
      className={`
        relative min-w-[220px] max-w-[280px] rounded-lg
        bg-[#1a1a24] border transition-all duration-200 cursor-pointer
        ${selected ? 'border-[#00d4ff] shadow-lg shadow-[#00d4ff]/20' : 'border-[#2a2a38]'}
      `}
    >
      {/* Header */}
      <div
        className="px-3 py-2 flex items-center gap-2 rounded-t-lg"
        style={{ backgroundColor: color + '20', borderBottom: `1px solid ${color}40` }}
      >
        <GripHorizontal className="w-4 h-4 text-gray-500 cursor-grab" />
        {Icon && <Icon className="w-4 h-4" style={{ color }} />}
        <span className="text-sm font-medium" style={{ color }}>
          {data.label}
        </span>
      </div>

      {/* Parameters */}
      {displayParams.length > 0 && (
        <div className="px-3 py-2 space-y-1">
          {displayParams.map(({ key, value }) => (
            <div key={key} className="flex justify-between text-xs gap-2">
              <span className="text-gray-500 whitespace-nowrap">{key}</span>
              <span className="text-gray-300 font-medium text-right">{value}</span>
            </div>
          ))}
        </div>
      )}

      {/* Agentic input: right + bottom handles */}
      {isInput && isAgenticInput && (
        <>
          <Handle
            type="source"
            position={Position.Right}
            id="output"
            className="!w-[8px] !h-[8px] !border-[1.5px] !border-[#1a1a24] !right-[-4px]"
            style={{ backgroundColor: color }}
          />
          <Handle
            type="source"
            position={Position.Bottom}
            id="opt-output"
            className="!w-[8px] !h-[8px] !border-[1.5px] !border-[#1a1a24] !bottom-[-4px]"
            style={{ backgroundColor: '#ffd700' }}
          />
        </>
      )}

      {/* Non-agentic input nodes: right handle only */}
      {isInput && !isAgenticInput && (
        <Handle
          type="source"
          position={Position.Right}
          id="output"
          className="!w-[8px] !h-[8px] !border-[1.5px] !border-[#1a1a24] !right-[-4px]"
          style={{ backgroundColor: color }}
        />
      )}

      {/* Agentic optimization nodes: top handle only */}
      {isProcessing && isAgenticOpt && (
        <Handle
          type="target"
          position={Position.Top}
          id="opt-input"
          className="!w-[8px] !h-[8px] !border-[1.5px] !border-[#1a1a24] !top-[-4px]"
          style={{ backgroundColor: '#ffd700' }}
        />
      )}

      {/* Non-agentic optimization nodes: left + right handles */}
      {isProcessing && isOptimization && !isAgenticOpt && (
        <>
          <Handle
            type="target"
            position={Position.Left}
            id="input"
            className="!w-[8px] !h-[8px] !bg-[#2a2a38] !border-[1.5px] !border-[#1a1a24] !left-[-4px]"
          />
          <Handle
            type="source"
            position={Position.Right}
            id="output"
            className="!w-[8px] !h-[8px] !border-[1.5px] !border-[#1a1a24] !right-[-4px]"
            style={{ backgroundColor: color }}
          />
        </>
      )}

      {/* Regular processing nodes: left + right handles */}
      {isProcessing && !isOptimization && (
        <>
          <Handle
            type="target"
            position={Position.Left}
            id="input"
            className="!w-[8px] !h-[8px] !bg-[#2a2a38] !border-[1.5px] !border-[#1a1a24] !left-[-4px]"
          />
          <Handle
            type="source"
            position={Position.Right}
            id="output"
            className="!w-[8px] !h-[8px] !border-[1.5px] !border-[#1a1a24] !right-[-4px]"
            style={{ backgroundColor: color }}
          />
        </>
      )}

      {/* Output nodes: left handle only */}
      {isOutput && (
        <Handle
          type="target"
          position={Position.Left}
          id="input"
          className="!w-[8px] !h-[8px] !bg-[#ef4444] !border-[1.5px] !border-[#1a1a24] !left-[-4px]"
        />
      )}
    </div>
  );
};

export const WorkflowNode = memo(WorkflowNodeComponent);
