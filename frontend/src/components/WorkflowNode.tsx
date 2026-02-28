import React, { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { GripHorizontal } from 'lucide-react';
import { NodeData, useWorkflowStore, InputNodeType } from '../store/workflowStore';
import { nodeDefinitions } from '../types/nodes';

type WorkflowNodeProps = NodeProps & {
  data: NodeData;
};

const INPUT_TYPES: InputNodeType[] = ['textInput', 'imageInput', 'audioInput', 'spreadsheetInput'];

const WorkflowNodeComponent: React.FC<WorkflowNodeProps> = ({ id, data, selected }) => {
  const setSelectedNode = useWorkflowStore((state) => state.setSelectedNode);
  const nodes = useWorkflowStore((state) => state.nodes);

  const definition = nodeDefinitions.find((n) => n.type === data.type);
  const Icon = definition?.icon;
  const color = definition?.color || '#00d4ff';

  const isInput = INPUT_TYPES.includes(data.type as InputNodeType);
  const isOutput = data.type === 'saveModel';
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
        min-w-[220px] max-w-[280px] rounded-lg overflow-hidden
        bg-[#1a1a24] border transition-all duration-200 cursor-pointer
        ${selected ? 'border-[#00d4ff] shadow-lg shadow-[#00d4ff]/20' : 'border-[#2a2a38]'}
      `}
    >
      {/* Header */}
      <div
        className="px-3 py-2 flex items-center gap-2"
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

      {/* Input nodes: source handle on the right */}
      {isInput && (
        <Handle
          type="source"
          position={Position.Right}
          id="output"
          className="!w-3 !h-3 !border-2 !border-[#1a1a24]"
          style={{ backgroundColor: color }}
        />
      )}

      {/* Processing nodes: target on left, source on right */}
      {isProcessing && (
        <>
          <Handle
            type="target"
            position={Position.Left}
            id="input"
            className="!w-3 !h-3 !bg-[#2a2a38] !border-2 !border-[#1a1a24]"
          />
          <Handle
            type="source"
            position={Position.Right}
            id="output"
            className="!w-3 !h-3 !border-2 !border-[#1a1a24]"
            style={{ backgroundColor: color }}
          />
        </>
      )}

      {/* Save Model node: only a target handle on the left */}
      {isOutput && (
        <Handle
          type="target"
          position={Position.Left}
          id="input"
          className="!w-3 !h-3 !bg-[#ef4444] !border-2 !border-[#1a1a24]"
        />
      )}
    </div>
  );
};

export const WorkflowNode = memo(WorkflowNodeComponent);
