import React, { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { GripHorizontal } from 'lucide-react';
import { NodeData, useWorkflowStore } from '../store/workflowStore';
import { nodeDefinitions } from '../types/nodes';

type WorkflowNodeProps = NodeProps & {
  data: NodeData;
};

const WorkflowNodeComponent: React.FC<WorkflowNodeProps> = ({ id, data, selected }) => {
  const setSelectedNode = useWorkflowStore((state) => state.setSelectedNode);
  const nodes = useWorkflowStore((state) => state.nodes);

  const definition = nodeDefinitions.find((n) => n.type === data.type);
  const Icon = definition?.icon;
  const color = definition?.color || '#00d4ff';

  const handleClick = () => {
    const node = nodes.find((n) => n.id === id);
    if (node) {
      setSelectedNode(node);
    }
  };

  // Get all display parameters
  const getDisplayParams = () => {
    const params = data.parameters;
    const entries = Object.entries(params);
    return entries.map(([key, value]) => {
      // Format the key nicely
      const formattedKey = key
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, (str) => str.toUpperCase())
        .trim();

      // Format the value
      let displayValue = value;
      if (value === null || value === undefined || value === '') {
        displayValue = '—';
      } else if (typeof value === 'boolean') {
        displayValue = value ? 'Yes' : 'No';
      } else if (Array.isArray(value)) {
        displayValue = `${value.length} items`;
      } else if (typeof value === 'object') {
        displayValue = 'Configured';
      } else if (typeof value === 'number') {
        displayValue = value.toString();
      } else if (typeof value === 'string' && value.length > 20) {
        displayValue = value.substring(0, 20) + '...';
      }

      return { key: formattedKey, value: displayValue };
    });
  };

  const displayParams = getDisplayParams();

  return (
    <div
      onClick={handleClick}
      className={`
        min-w-[220px] max-w-[280px] rounded-lg overflow-hidden
        bg-[#1a1a24] border transition-all duration-200
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
      <div className="px-3 py-2 space-y-1">
        {displayParams.map(({ key, value }) => (
          <div key={key} className="flex justify-between text-xs gap-2">
            <span className="text-gray-500 whitespace-nowrap">{key}</span>
            <span className="text-gray-300 font-medium text-right">{value}</span>
          </div>
        ))}
      </div>

      {/* Handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-3 !h-3 !bg-[#ffd700] !border-2 !border-[#1a1a24]"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!w-3 !h-3 !bg-[#ffd700] !border-2 !border-[#1a1a24]"
      />
    </div>
  );
};

export const WorkflowNode = memo(WorkflowNodeComponent);
