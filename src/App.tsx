import React, { useState, useRef } from 'react';
import { MessageSquare, AlertTriangle, AlertCircle, ChevronRight, ChevronLeft, Send, Plus, X, Settings } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface Warning {
  type: 'red' | 'warning';
  message: string;
  timestamp: string;
}

interface Node {
  id: string;
  x: number;
  y: number;
  text: string;
  color?: string;
}

interface Edge {
  id: string;
  from: string;
  to: string;
  label: string;
  type: 'depends' | 'blocks' | 'relates';
}

function App() {
  const [isChatOpen, setIsChatOpen] = useState(true);
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Hello! I\'m your Teams meeting assistant. How can I help you today?',
      timestamp: new Date().toLocaleTimeString()
    }
  ]);

  // Graph state
  const [nodes, setNodes] = useState<Node[]>([
    { id: '1', x: 100, y: 100, text: 'Project Timeline', color: '#6264A7' },
    { id: '2', x: 300, y: 100, text: 'Budget Approval', color: '#4CAF50' },
    { id: '3', x: 200, y: 200, text: 'Resource Allocation', color: '#FF9800' }
  ]);
  const [edges, setEdges] = useState<Edge[]>([
    { id: '1-2', from: '1', to: '2', label: 'depends on', type: 'depends' },
    { id: '2-3', from: '2', to: '3', label: 'blocks', type: 'blocks' }
  ]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [draggingNode, setDraggingNode] = useState<string | null>(null);
  const [newEdgeStart, setNewEdgeStart] = useState<string | null>(null);
  const [edgeType, setEdgeType] = useState<'depends' | 'blocks' | 'relates'>('depends');
  const [showGraphSettings, setShowGraphSettings] = useState(false);
  const graphRef = useRef<HTMLDivElement>(null);

  // Simulated warnings for the executive summary
  const [warnings] = useState<Warning[]>([
    {
      type: 'red',
      message: 'Potential compliance violation mentioned at 10:15',
      timestamp: '10:15 AM'
    },
    {
      type: 'warning',
      message: 'Unclear project timeline discussion',
      timestamp: '10:18 AM'
    }
  ]);

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const newMessage: Message = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages([...messages, newMessage]);
    setInputMessage('');

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: Message = {
        role: 'assistant',
        content: 'This is a simulated response. Connect to your LLM API endpoint for real responses.',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, aiResponse]);
    }, 1000);
  };

  const handleNodeMouseDown = (nodeId: string, e: React.MouseEvent) => {
    if (e.button === 0) { // Left click
      setDraggingNode(nodeId);
      setSelectedNode(nodeId);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (draggingNode && graphRef.current) {
      const rect = graphRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      setNodes(nodes.map(node => 
        node.id === draggingNode ? { ...node, x, y } : node
      ));
    }
  };

  const handleMouseUp = () => {
    setDraggingNode(null);
  };

  const addNode = () => {
    const colors = ['#6264A7', '#4CAF50', '#FF9800', '#E91E63', '#2196F3'];
    const newNode: Node = {
      id: String(nodes.length + 1),
      x: 200,
      y: 200,
      text: 'New Topic',
      color: colors[Math.floor(Math.random() * colors.length)]
    };
    setNodes([...nodes, newNode]);
  };

  const updateNodeText = (nodeId: string, text: string) => {
    setNodes(nodes.map(node =>
      node.id === nodeId ? { ...node, text } : node
    ));
  };

  const updateNodeColor = (nodeId: string, color: string) => {
    setNodes(nodes.map(node =>
      node.id === nodeId ? { ...node, color } : node
    ));
  };

  const deleteNode = (nodeId: string) => {
    setNodes(nodes.filter(node => node.id !== nodeId));
    setEdges(edges.filter(edge => edge.from !== nodeId && edge.to !== nodeId));
    setSelectedNode(null);
  };

  const handleNodeDoubleClick = (nodeId: string) => {
    if (!newEdgeStart) {
      setNewEdgeStart(nodeId);
    } else if (newEdgeStart !== nodeId) {
      const newEdge: Edge = {
        id: `${newEdgeStart}-${nodeId}`,
        from: newEdgeStart,
        to: nodeId,
        label: edgeType === 'depends' ? 'depends on' : edgeType === 'blocks' ? 'blocks' : 'relates to',
        type: edgeType
      };
      setEdges([...edges, newEdge]);
      setNewEdgeStart(null);
    }
  };

  return (
    <div className="flex h-screen bg-teams-dark text-gray-100">
      {/* Main Content Area */}
      <div className={`flex-1 transition-all duration-300 ${isChatOpen ? 'mr-96' : 'mr-0'}`}>
        <div className="p-6 space-y-6">
          {/* Executive Summary */}
          <div className="glass-panel rounded-xl shadow-lg p-6 relative led-glow">
            <h2 className="text-xl font-bold mb-6 text-teams-accent">Executive Summary</h2>
            
            {/* Red Flags Section */}
            <div className="mb-6">
              <h3 className="text-base font-semibold mb-4 flex items-center text-red-400">
                <AlertCircle className="mr-2" size={18} />
                Red Flags
              </h3>
              <div className="space-y-3">
                {warnings.filter(w => w.type === 'red').map((warning, index) => (
                  <div key={index} className="flex items-start p-3 bg-red-900/20 border border-red-700/30 rounded-lg">
                    <AlertCircle className="text-red-400 mr-3 mt-1" size={14} />
                    <div>
                      <p className="text-sm text-red-100">{warning.message}</p>
                      <p className="text-xs text-red-400 mt-1">{warning.timestamp}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Warnings Section */}
            <div>
              <h3 className="text-base font-semibold mb-4 flex items-center text-yellow-400">
                <AlertTriangle className="mr-2" size={18} />
                Warnings
              </h3>
              <div className="space-y-3">
                {warnings.filter(w => w.type === 'warning').map((warning, index) => (
                  <div key={index} className="flex items-start p-3 bg-yellow-900/20 border border-yellow-700/30 rounded-lg">
                    <AlertTriangle className="text-yellow-400 mr-3 mt-1" size={14} />
                    <div>
                      <p className="text-sm text-yellow-100">{warning.message}</p>
                      <p className="text-xs text-yellow-400 mt-1">{warning.timestamp}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Topic Graph */}
          <div className="glass-panel rounded-xl shadow-lg p-6 relative led-glow">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold text-teams-accent">Topic Dependencies</h2>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setShowGraphSettings(!showGraphSettings)}
                  className="flex items-center gap-2 bg-gray-800 px-3 py-2 rounded-lg hover:bg-gray-700 transition-colors"
                >
                  <Settings size={16} />
                  <span className="text-sm">Settings</span>
                </button>
                <button
                  onClick={addNode}
                  className="flex items-center gap-2 bg-teams-accent px-4 py-2 rounded-lg hover:bg-teams-hover transition-colors"
                >
                  <Plus size={16} />
                  <span className="text-sm">Add Topic</span>
                </button>
              </div>
            </div>

            {showGraphSettings && (
              <div className="mb-4 p-4 bg-gray-800 rounded-lg">
                <h3 className="text-sm font-semibold mb-3">Connection Type</h3>
                <div className="flex gap-3">
                  <button
                    onClick={() => setEdgeType('depends')}
                    className={`px-3 py-1 rounded text-sm ${
                      edgeType === 'depends' ? 'bg-blue-500' : 'bg-gray-700'
                    }`}
                  >
                    Depends
                  </button>
                  <button
                    onClick={() => setEdgeType('blocks')}
                    className={`px-3 py-1 rounded text-sm ${
                      edgeType === 'blocks' ? 'bg-red-500' : 'bg-gray-700'
                    }`}
                  >
                    Blocks
                  </button>
                  <button
                    onClick={() => setEdgeType('relates')}
                    className={`px-3 py-1 rounded text-sm ${
                      edgeType === 'relates' ? 'bg-green-500' : 'bg-gray-700'
                    }`}
                  >
                    Relates
                  </button>
                </div>
              </div>
            )}

            <div
              ref={graphRef}
              className="relative h-[400px] rounded-lg bg-gray-900/50 border border-gray-700/50"
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              {/* Edges */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                {edges.map(edge => {
                  const fromNode = nodes.find(n => n.id === edge.from);
                  const toNode = nodes.find(n => n.id === edge.to);
                  if (!fromNode || !toNode) return null;

                  const midX = (fromNode.x + toNode.x) / 2;
                  const midY = (fromNode.y + toNode.y) / 2;

                  return (
                    <g key={edge.id}>
                      <line
                        x1={fromNode.x}
                        y1={fromNode.y}
                        x2={toNode.x}
                        y2={toNode.y}
                        className={`node-connection ${edge.type}`}
                        strokeWidth="2"
                      />
                      <rect
                        x={midX - 40}
                        y={midY - 10}
                        width="80"
                        height="20"
                        className="fill-gray-800"
                        rx="4"
                      />
                      <text
                        x={midX}
                        y={midY + 5}
                        textAnchor="middle"
                        className="text-xs fill-gray-300"
                      >
                        {edge.label}
                      </text>
                    </g>
                  );
                })}
              </svg>

              {/* Nodes */}
              {nodes.map(node => (
                <div
                  key={node.id}
                  className={`absolute cursor-move p-3 rounded-lg shadow-lg transition-all duration-200 ${
                    selectedNode === node.id ? 'scale-105' : ''
                  }`}
                  style={{
                    left: node.x - 50,
                    top: node.y - 25,
                    width: '100px',
                    height: '50px',
                    backgroundColor: selectedNode === node.id ? '#6264A7' : node.color || '#2d2d2d',
                  }}
                  onMouseDown={(e) => handleNodeMouseDown(node.id, e)}
                  onDoubleClick={() => handleNodeDoubleClick(node.id)}
                >
                  {selectedNode === node.id ? (
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={node.text}
                        onChange={(e) => updateNodeText(node.id, e.target.value)}
                        className="w-full bg-transparent border-none outline-none text-white placeholder-gray-300 text-sm"
                        onClick={(e) => e.stopPropagation()}
                      />
                      <button
                        onClick={() => deleteNode(node.id)}
                        className="text-white hover:text-red-300"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ) : (
                    <div className="text-center text-sm">{node.text}</div>
                  )}
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-400 mt-3">
              Double-click nodes to connect them. Drag nodes to reposition.
            </p>
          </div>
        </div>
      </div>

      {/* Chat Panel */}
      <div 
        className={`fixed right-0 top-0 h-full w-96 glass-panel transform transition-transform duration-300 ${
          isChatOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="h-full flex flex-col">
          {/* Chat Header */}
          <div className="p-4 border-b border-gray-700/50 flex items-center justify-between bg-teams-accent rounded-t-xl">
            <div className="flex items-center text-white">
              <MessageSquare className="mr-2" size={18} />
              <h2 className="font-semibold text-sm">Meeting Assistant</h2>
            </div>
            <button
              onClick={() => setIsChatOpen(!isChatOpen)}
              className="text-white hover:bg-teams-hover p-1 rounded"
            >
              {isChatOpen ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
            </button>
          </div>

          {/* Messages Container */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    message.role === 'user'
                      ? 'bg-teams-accent text-white'
                      : 'bg-gray-800 text-gray-100'
                  }`}
                >
                  <p className="text-sm">{message.content}</p>
                  <p className={`text-xs mt-1 ${
                    message.role === 'user' ? 'text-gray-300' : 'text-gray-400'
                  }`}>
                    {message.timestamp}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Message Input */}
          <form onSubmit={handleSendMessage} className="p-4 border-t border-gray-700/50">
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Type your message..."
                className="flex-1 p-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-100 text-sm focus:outline-none focus:ring-2 focus:ring-teams-accent placeholder-gray-500"
              />
              <button
                type="submit"
                className="bg-teams-accent text-white p-2 rounded-lg hover:bg-teams-hover transition-colors"
              >
                <Send size={18} />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;