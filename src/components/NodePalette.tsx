import React, { useState } from 'react';
import { ChevronDown, ChevronRight, GripVertical, CircleDot, Settings2 } from 'lucide-react';
import { nodeDefinitions, NodeDefinition } from '../types/nodes';
import { useWorkflowStore, NodeType, InputNodeType } from '../store/workflowStore';

interface CategoryProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const Category: React.FC<CategoryProps> = ({ title, icon, children, defaultOpen = true }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="mb-4">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs font-semibold uppercase tracking-wider text-gray-400 hover:text-gray-200 transition-colors"
      >
        {icon}
        <span>{title}</span>
        {isOpen ? (
          <ChevronDown className="w-4 h-4 ml-auto" />
        ) : (
          <ChevronRight className="w-4 h-4 ml-auto" />
        )}
      </button>
      {isOpen && <div className="mt-1 space-y-1">{children}</div>}
    </div>
  );
};

interface PaletteNodeProps {
  definition: NodeDefinition;
  disabled?: boolean;
  disabledReason?: string;
}

const PaletteNode: React.FC<PaletteNodeProps> = ({ definition, disabled = false, disabledReason }) => {
  const addNode = useWorkflowStore((state) => state.addNode);
  const Icon = definition.icon;

  const handleDragStart = (e: React.DragEvent) => {
    if (disabled) {
      e.preventDefault();
      return;
    }
    e.dataTransfer.setData('application/reactflow', definition.type);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleClick = () => {
    if (disabled) return;
    addNode(definition.type as NodeType, { x: 400, y: 200 + Math.random() * 200 });
  };

  return (
    <div
      draggable={!disabled}
      onDragStart={handleDragStart}
      onClick={handleClick}
      title={disabled ? disabledReason : definition.description}
      className={`
        group flex items-center gap-3 px-3 py-3 mx-2 rounded-lg cursor-pointer
        transition-all duration-200 border border-transparent
        ${disabled
          ? 'opacity-40 cursor-not-allowed bg-[#12121a]'
          : 'hover:bg-[#22222e] hover:border-[#2a2a38] active:scale-[0.98]'
        }
      `}
      style={{
        borderLeftColor: disabled ? 'transparent' : definition.color,
        borderLeftWidth: '3px'
      }}
    >
      <div
        className={`
          p-2 rounded-lg transition-colors
          ${disabled ? 'bg-[#1a1a24]' : 'bg-[#1a1a24] group-hover:bg-[#22222e]'}
        `}
        style={{ color: disabled ? '#666' : definition.color }}
      >
        <Icon className="w-5 h-5" />
      </div>
      <div className="flex-1 min-w-0">
        <div className={`text-sm font-medium ${disabled ? 'text-gray-600' : 'text-gray-200'}`}>
          {definition.label}
        </div>
        <div className={`text-xs truncate ${disabled ? 'text-gray-700' : 'text-gray-500'}`}>
          {definition.description}
        </div>
      </div>
      {!disabled && (
        <GripVertical className="w-4 h-4 text-gray-600 opacity-0 group-hover:opacity-100 transition-opacity" />
      )}
    </div>
  );
};

const getInputNodeLabel = (type: InputNodeType): string => {
  switch (type) {
    case 'textRetrieval': return 'Text Retrieval (RAG)';
    case 'agenticLLM': return 'Agentic LLM Input';
    case 'visualData': return 'Visual Data (CV)';
    case 'audioData': return 'Audio (CNN)';
    case 'voiceInput': return 'Voice Input';
    default: return 'Unknown';
  }
};

export const NodePalette: React.FC = () => {
  const getInputNodeTypes = useWorkflowStore((state) => state.getInputNodeTypes);
  const activeInputNodes = getInputNodeTypes();

  const inputNodes = nodeDefinitions.filter((n) => n.category === 'input');
  const optimizationNodes = nodeDefinitions.filter((n) => n.category === 'optimization');
  // Output node is auto-added, not available in palette

  // Check if an optimization node is enabled based on active input nodes
  const isOptimizationEnabled = (node: NodeDefinition): boolean => {
    if (node.requiresAnyInputNode) {
      return node.requiresAnyInputNode.some((t) => activeInputNodes.includes(t));
    }
    if (!node.requiresInputNode) return true;
    return activeInputNodes.includes(node.requiresInputNode);
  };

  const getDisabledReason = (node: NodeDefinition): string => {
    if (node.requiresAnyInputNode) {
      const labels = node.requiresAnyInputNode.map(getInputNodeLabel).join(' or ');
      return `Add a ${labels} node to enable this`;
    }
    if (!node.requiresInputNode) return '';
    return `Add a "${getInputNodeLabel(node.requiresInputNode)}" node to enable this`;
  };

  return (
    <div className="w-72 bg-[#12121a] border-r border-[#22222e] flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-[#22222e]">
        <h2 className="text-[#00d4ff] text-sm font-semibold uppercase tracking-wider">
          Node Palette
        </h2>
        <p className="text-gray-500 text-xs mt-1">Drag or click to add</p>
      </div>

      {/* Node List */}
      <div className="flex-1 overflow-y-auto py-4">
        <Category
          title="Input Nodes"
          icon={<CircleDot className="w-4 h-4 text-[#00d4ff]" />}
        >
          {inputNodes.map((node) => (
            <PaletteNode key={node.type} definition={node} />
          ))}
        </Category>

        <Category
          title="Optimization"
          icon={<Settings2 className="w-4 h-4 text-[#ffd700]" />}
        >
          {optimizationNodes.map((node) => {
            const enabled = isOptimizationEnabled(node);
            return (
              <PaletteNode
                key={node.type}
                definition={node}
                disabled={!enabled}
                disabledReason={getDisabledReason(node)}
              />
            );
          })}
          {activeInputNodes.length === 0 && (
            <p className="text-xs text-gray-600 px-5 py-2 italic">
              Add an input node to unlock optimization options
            </p>
          )}
        </Category>
      </div>

      {/* Footer hint */}
      <div className="p-4 border-t border-[#22222e]">
        <p className="text-xs text-gray-500 text-center">
          Connect nodes by dragging from handles
        </p>
      </div>
    </div>
  );
};
