import React, { useState } from 'react';
import { ChevronDown, ChevronRight, GripVertical, CircleDot, Settings2, Save } from 'lucide-react';
import { nodeDefinitions, NodeDefinition } from '../types/nodes';
import { useWorkflowStore, NodeType, InputNodeType } from '../store/workflowStore';

export type WorkflowPaletteMode = 'training' | 'deployment';

// ── Collapsible category ──────────────────────────────────────────────────────

const Category: React.FC<{
  title: string; icon: React.ReactNode; children: React.ReactNode; defaultOpen?: boolean;
}> = ({ title, icon, children, defaultOpen = true }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mb-4">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs font-semibold uppercase tracking-wider text-gray-400 hover:text-gray-200 transition-colors"
      >
        {icon}
        <span>{title}</span>
        {open ? <ChevronDown className="w-4 h-4 ml-auto" /> : <ChevronRight className="w-4 h-4 ml-auto" />}
      </button>
      {open && <div className="mt-1 space-y-1">{children}</div>}
    </div>
  );
};

// ── Palette node item ─────────────────────────────────────────────────────────

const PaletteNode: React.FC<{
  definition: NodeDefinition;
  disabled?: boolean;
  disabledReason?: string;
}> = ({ definition, disabled = false, disabledReason }) => {
  const addNode = useWorkflowStore((s) => s.addNode);
  const Icon = definition.icon;

  const handleDragStart = (e: React.DragEvent) => {
    if (disabled) { e.preventDefault(); return; }
    e.dataTransfer.setData('application/reactflow', definition.type);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleClick = () => {
    if (disabled) return;
    addNode(definition.type as NodeType, { x: 250 + Math.random() * 300, y: 150 + Math.random() * 200 });
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
          ? 'opacity-35 cursor-not-allowed bg-[#12121a]'
          : 'hover:bg-[#22222e] hover:border-[#2a2a38] active:scale-[0.98]'
        }
      `}
      style={{ borderLeftColor: disabled ? 'transparent' : definition.color, borderLeftWidth: '3px' }}
    >
      <div
        className="p-2 rounded-lg"
        style={{ backgroundColor: disabled ? '#1a1a24' : definition.color + '22', color: disabled ? '#555' : definition.color }}
      >
        <Icon className="w-4 h-4" />
      </div>
      <div className="flex-1 min-w-0">
        <div className={`text-sm font-medium truncate ${disabled ? 'text-gray-600' : 'text-gray-200'}`}>
          {definition.label}
        </div>
        <div className={`text-xs truncate ${disabled ? 'text-gray-700' : 'text-gray-500'}`}>
          {disabled && disabledReason ? disabledReason : definition.description}
        </div>
      </div>
      {!disabled && <GripVertical className="w-4 h-4 text-gray-600 opacity-0 group-hover:opacity-100 shrink-0" />}
    </div>
  );
};

// ── Palette ───────────────────────────────────────────────────────────────────

export const NodePalette: React.FC<{ mode?: WorkflowPaletteMode }> = ({ mode = 'training' }) => {
  const getActiveInputType = useWorkflowStore((s) => s.getActiveInputType);
  const nodes = useWorkflowStore((s) => s.nodes);

  const activeInput: InputNodeType | null = getActiveInputType();

  // Detect image format from the imageInput node parameters
  const imageNode = nodes.find((n) => n.data.type === 'imageInput');
  const detectedImageFormat = imageNode
    ? (imageNode.data.parameters.imageFormat as string | undefined) ?? 'unknown'
    : 'unknown';

  const inputNodes      = nodeDefinitions.filter((n) => n.category === 'input');
  const processingNodes = nodeDefinitions.filter((n) => n.category === 'processing');
  const outputNodes     = nodeDefinitions.filter((n) => n.category === 'output');

  // ── Deployment mode ────────────────────────────────────────────────────────
  if (mode === 'deployment') {
    const deployProcessing = processingNodes.filter(
      (n) => n.type === 'deployModelNode' || n.type === 'llmNode'
    );
    const deployOutput = outputNodes.filter((n) => n.type === 'deployOutputNode');

    return (
      <div className="w-72 bg-[#12121a] border-r border-[#22222e] flex flex-col h-full">
        <div className="p-4 border-b border-[#22222e]">
          <h2 className="text-[#6366f1] text-sm font-semibold uppercase tracking-wider">Deployment Palette</h2>
          <p className="text-gray-500 text-xs mt-1">Build your inference pipeline</p>
        </div>
        <div className="flex-1 overflow-y-auto py-4">
          <Category title="Input" icon={<CircleDot className="w-4 h-4 text-[#00d4ff]" />}>
            {activeInput && (
              <p className="text-[10px] text-amber-500/80 px-3 pb-1 italic">
                Delete the active input node to switch type
              </p>
            )}
            {inputNodes.map((def) => (
              <PaletteNode
                key={def.type}
                definition={def}
                disabled={!!activeInput && def.type !== activeInput}
                disabledReason={activeInput ? `Remove the active ${activeInput.replace('Input', '')} input first` : ''}
              />
            ))}
          </Category>
          <Category title="Processing" icon={<Settings2 className="w-4 h-4 text-[#6366f1]" />}>
            {deployProcessing.map((def) => (
              <PaletteNode key={def.type} definition={def} />
            ))}
          </Category>
          <Category title="Output" icon={<Save className="w-4 h-4 text-[#14b8a6]" />}>
            {deployOutput.map((def) => (
              <PaletteNode key={def.type} definition={def} />
            ))}
          </Category>
        </div>
        <div className="p-4 border-t border-[#22222e]">
          <p className="text-xs text-gray-500 text-center">Connect nodes by dragging handles</p>
        </div>
      </div>
    );
  }

  // ── Training mode (default) ────────────────────────────────────────────────

  const isInputDisabled = (def: NodeDefinition): boolean => {
    if (!activeInput) return false;
    return def.type !== activeInput;
  };

  const getInputDisabledReason = (_def: NodeDefinition): string => {
    if (!activeInput) return '';
    return `Remove the active ${activeInput.replace('Input', '')} input node first`;
  };

  const isProcessingEnabled = (def: NodeDefinition): { enabled: boolean; reason: string } => {
    // Skip deployment-only nodes in training mode
    if (def.type === 'deployModelNode' || def.type === 'llmNode') return { enabled: false, reason: '' };
    if (!activeInput) return { enabled: false, reason: 'Add an input node first' };
    if (def.requiredInput && def.requiredInput !== activeInput)
      return { enabled: false, reason: `Requires a ${def.requiredInput.replace('Input', '')} input node` };

    if (def.requiredInput === 'imageInput' && def.requiredImageFormat) {
      if (detectedImageFormat === 'unknown')
        return { enabled: false, reason: 'Upload images first to detect format' };
      if (def.requiredImageFormat === 'imagefolder' && detectedImageFormat !== 'imagefolder')
        return { enabled: false, reason: 'Requires ImageFolder format (subdirectories of images)' };
      if (def.requiredImageFormat === 'boundingbox' && detectedImageFormat !== 'boundingbox')
        return { enabled: false, reason: 'Requires bounding-box annotation format' };
    }
    return { enabled: true, reason: '' };
  };

  // Filter out deployment-only nodes from training palette
  const trainingProcessing = processingNodes.filter(
    (n) => n.type !== 'deployModelNode' && n.type !== 'llmNode'
  );
  const trainingOutput = outputNodes.filter((n) => n.type !== 'deployOutputNode');

  const textNodes  = trainingProcessing.filter((n) => n.requiredInput === 'textInput');
  const imageNodes = trainingProcessing.filter((n) => n.requiredInput === 'imageInput');
  const audioNodes = trainingProcessing.filter((n) => n.requiredInput === 'audioInput');
  const tabNodes   = trainingProcessing.filter((n) => n.requiredInput === 'spreadsheetInput');

  const renderProcessingGroup = (groupNodes: NodeDefinition[], groupLabel: string) => (
    <>
      {groupNodes.length > 0 && activeInput && (
        <div className="px-3 pt-2 pb-1">
          <span className="text-[10px] uppercase tracking-widest text-gray-600 font-semibold">{groupLabel}</span>
        </div>
      )}
      {groupNodes.map((def) => {
        const { enabled, reason } = isProcessingEnabled(def);
        return (
          <PaletteNode
            key={def.type}
            definition={def}
            disabled={!enabled}
            disabledReason={reason}
          />
        );
      })}
    </>
  );

  return (
    <div className="w-72 bg-[#12121a] border-r border-[#22222e] flex flex-col h-full">
      <div className="p-4 border-b border-[#22222e]">
        <h2 className="text-[#00d4ff] text-sm font-semibold uppercase tracking-wider">Node Palette</h2>
        <p className="text-gray-500 text-xs mt-1">Drag or click to add nodes</p>
      </div>

      <div className="flex-1 overflow-y-auto py-4">
        <Category title="Input" icon={<CircleDot className="w-4 h-4 text-[#00d4ff]" />}>
          {activeInput && (
            <p className="text-[10px] text-amber-500/80 px-3 pb-1 italic">
              Delete the active input node to switch type
            </p>
          )}
          {inputNodes.map((def) => (
            <PaletteNode
              key={def.type}
              definition={def}
              disabled={isInputDisabled(def)}
              disabledReason={getInputDisabledReason(def)}
            />
          ))}
        </Category>

        <Category title="Processing" icon={<Settings2 className="w-4 h-4 text-[#ffd700]" />}>
          {!activeInput && (
            <p className="text-xs text-gray-600 px-5 py-2 italic">
              Add an input node to unlock processing options
            </p>
          )}
          {renderProcessingGroup(textNodes,  'Text')}
          {renderProcessingGroup(imageNodes, 'Image')}
          {renderProcessingGroup(audioNodes, 'Audio')}
          {renderProcessingGroup(tabNodes,   'Tabular')}
        </Category>

        <Category title="Output" icon={<Save className="w-4 h-4 text-[#ef4444]" />} defaultOpen>
          {trainingOutput.map((def) => (
            <PaletteNode key={def.type} definition={def} />
          ))}
        </Category>
      </div>

      <div className="p-4 border-t border-[#22222e]">
        <p className="text-xs text-gray-500 text-center">Connect nodes by dragging handles</p>
      </div>
    </div>
  );
};
