import React, { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';
import * as d3 from 'd3';
import NodeDetailsModal from './NodeDetailsModal';
import RefreshIcon from '@mui/icons-material/Refresh';
import SettingsIcon from '@mui/icons-material/Settings';
import AddIcon from '@mui/icons-material/Add';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import RestoreIcon from '@mui/icons-material/Restore';
import FullscreenIcon from '@mui/icons-material/Fullscreen';

const Container = styled.div`
  padding: 20px;
  background: #1e2130;
  border-radius: 8px;
  margin-bottom: 20px;
  position: relative;
`;

const SectionTitle = styled.h2`
  color: #a5a8b6;
  font-size: 24px;
  margin-bottom: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 10px;
`;

const Button = styled.button`
  background: #4c4f5a;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 6px;
  
  &:hover {
    background: #5a5d6a;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const ZoomControls = styled.div`
  position: absolute;
  bottom: 20px;
  right: 20px;
  display: flex;
  flex-direction: column;
  gap: 5px;
  background: rgba(30, 33, 48, 0.8);
  padding: 10px;
  border-radius: 5px;
  z-index: 100;
`;

const ZoomButton = styled.button`
  background: #4c4f5a;
  color: white;
  border: none;
  width: 40px;
  height: 40px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  
  &:hover {
    background: #5a5d6a;
  }
`;

const HelpText = styled.div`
  position: absolute;
  bottom: 20px;
  left: 20px;
  font-size: 12px;
  color: #a5a8b6;
  background: rgba(30, 33, 48, 0.8);
  padding: 8px 12px;
  border-radius: 4px;
  z-index: 100;
`;

const SettingsModal = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
`;

const SettingsContent = styled.div`
  background: #1e2130;
  padding: 24px;
  border-radius: 8px;
  max-width: 500px;
  width: 90%;
  color: white;
  position: relative;
  max-height: 90vh;
  overflow-y: auto;
`;

const CloseButton = styled.button`
  position: absolute;
  top: 16px;
  right: 16px;
  background: none;
  border: none;
  color: white;
  font-size: 24px;
  cursor: pointer;
  padding: 0;
  line-height: 1;
`;

const SettingsTitle = styled.h3`
  margin: 0 0 16px 0;
  color: #a5a8b6;
  font-size: 20px;
`;

const FormGroup = styled.div`
  margin-bottom: 16px;
`;

const Label = styled.label`
  display: block;
  margin-bottom: 8px;
  color: #a5a8b6;
`;

const Select = styled.select`
  width: 100%;
  padding: 8px;
  background: #2e3141;
  color: white;
  border: 1px solid #4c4f5a;
  border-radius: 4px;
  margin-bottom: 16px;
`;

const Input = styled.input`
  width: 100%;
  padding: 8px;
  background: #2e3141;
  color: white;
  border: 1px solid #4c4f5a;
  border-radius: 4px;
  margin-bottom: 16px;
`;

const Textarea = styled.textarea`
  width: 100%;
  padding: 8px;
  background: #2e3141;
  color: white;
  border: 1px solid #4c4f5a;
  border-radius: 4px;
  margin-bottom: 16px;
  min-height: 100px;
  font-family: inherit;
`;

const AddNodeModal = styled(SettingsModal)``;
const AddNodeContent = styled(SettingsContent)``;

const MindMap = ({ data: initialData, onRefresh }) => {
  const svgRef = useRef();
  const [selectedNode, setSelectedNode] = useState(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isAddNodeOpen, setIsAddNodeOpen] = useState(false);
  const [layoutSettings, setLayoutSettings] = useState({
    forceStrength: -300,
    linkDistance: 100,
    centeringForce: 0.1
  });
  const [newNode, setNewNode] = useState({
    name: '',
    type: 'summary',
    details: '',
    parent: ''
  });
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // Store zoom state
  const [currentZoom, setCurrentZoom] = useState(1);
  const zoomBehaviorRef = useRef(null);
  
  // Add local state for the mind map data so we can modify it directly
  const [data, setData] = useState(initialData);
  
  // Update internal data when props change
  useEffect(() => {
    setData(initialData);
  }, [initialData]);
  
  // Handle manual refresh with animation
  const handleRefresh = () => {
    if (onRefresh) {
      setIsRefreshing(true);
      onRefresh();
      // Disable refresh button for 2 seconds to prevent spam
      setTimeout(() => {
        setIsRefreshing(false);
      }, 2000);
    }
  };

  // Get a list of possible parent nodes for the "Add Node" modal
  const getParentOptions = () => {
    if (!data || !data.nodes) return [];
    
    // Get nodes with fewer than 6 connections to prevent overloading
    const nodeConnections = {};
    
    // Count connections for each node
    data.links.forEach(link => {
      if (!nodeConnections[link.source.id || link.source]) {
        nodeConnections[link.source.id || link.source] = 0;
      }
      nodeConnections[link.source.id || link.source]++;
    });
    
    // Only include nodes with fewer than 6 connections
    return data.nodes
      .filter(node => !nodeConnections[node.id] || nodeConnections[node.id] < 6)
      .map(node => ({
        id: node.id,
        name: node.name
      }));
  };
  
  // Handle adding a new node
  const handleAddNode = () => {
    if (newNode.name && newNode.parent) {
      // Create a new node with a unique ID
      const newId = `custom-${Date.now()}`;
      const newNodeObj = {
        id: newId,
        name: newNode.name,
        type: newNode.type,
        details: newNode.details || `${newNode.name} (user-added)`
      };
      
      // Create a new link
      const newLink = {
        source: newNode.parent,
        target: newId
      };
      
      // Update the local data state
      const updatedData = {
        nodes: [...data.nodes, newNodeObj],
        links: [...data.links, newLink]
      };
      
      // Update the state
      setData(updatedData);
      
      // Reset the form
      setNewNode({
        name: '',
        type: 'summary',
        details: '',
        parent: ''
      });
      
      // Close the modal
      setIsAddNodeOpen(false);
      
      // Force a redraw
      setTimeout(() => {
        createVisualization(updatedData);
      }, 100);
    }
  };
  
  // Apply settings changes
  const applySettings = () => {
    setIsSettingsOpen(false);
    // Force a redraw with the new settings
    if (data) {
      createVisualization(data);
    }
  };
  
  // Handle zoom control buttons
  const handleZoomIn = () => {
    if (zoomBehaviorRef.current) {
      const svg = d3.select(svgRef.current);
      const newZoom = Math.min(currentZoom * 1.3, 4); // Limit max zoom
      setCurrentZoom(newZoom);
      svg.transition().duration(300).call(zoomBehaviorRef.current.transform, 
        d3.zoomIdentity.scale(newZoom));
    }
  };

  const handleZoomOut = () => {
    if (zoomBehaviorRef.current) {
      const svg = d3.select(svgRef.current);
      const newZoom = Math.max(currentZoom * 0.7, 0.25); // Limit min zoom
      setCurrentZoom(newZoom);
      svg.transition().duration(300).call(zoomBehaviorRef.current.transform, 
        d3.zoomIdentity.scale(newZoom));
    }
  };

  const handleResetView = () => {
    if (zoomBehaviorRef.current) {
      const svg = d3.select(svgRef.current);
      setCurrentZoom(1);
      svg.transition().duration(500).call(zoomBehaviorRef.current.transform, 
        d3.zoomIdentity);
    }
  };
  
  const toggleFullscreen = () => {
    const container = svgRef.current.parentElement;
    
    if (!isFullscreen) {
      if (container.requestFullscreen) {
        container.requestFullscreen();
      } else if (container.webkitRequestFullscreen) {
        container.webkitRequestFullscreen();
      } else if (container.msRequestFullscreen) {
        container.msRequestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen();
      } else if (document.msExitFullscreen) {
        document.msExitFullscreen();
      }
    }
    
    setIsFullscreen(!isFullscreen);
  };
  
  // Create or update the D3 visualization
  const createVisualization = (visualData = data) => {
    if (!visualData || !svgRef.current) return;

    const width = 800;
    const height = 600;
    const containerWidth = svgRef.current.parentElement.clientWidth;
    const containerHeight = 600;

    // Clear previous SVG content
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current)
      .attr("width", containerWidth)
      .attr("height", containerHeight);
    
    // Create a group for our content that will be transformed for zoom/pan
    const g = svg.append("g");
    
    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.25, 4]) // Limit zoom level
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
        setCurrentZoom(event.transform.k);
      });
    
    zoomBehaviorRef.current = zoom;
    
    svg.call(zoom)
      .on("dblclick.zoom", null); // Disable double-click to zoom

    // Create force simulation with current settings
    const simulation = d3.forceSimulation(visualData.nodes)
      .force("link", d3.forceLink(visualData.links).id(d => d.id).distance(layoutSettings.linkDistance))
      .force("charge", d3.forceManyBody().strength(layoutSettings.forceStrength))
      .force("center", d3.forceCenter(containerWidth / 2, containerHeight / 2).strength(layoutSettings.centeringForce))
      // Add collision detection to prevent node overlap
      .force("collision", d3.forceCollide().radius(d => d.type === 'app' ? 40 : 30));

    // Create links
    const links = g.append("g")
      .selectAll("line")
      .data(visualData.links)
      .enter()
      .append("line")
      .attr("stroke", "#4a4a4a")
      .attr("stroke-width", 2);

    // Create nodes
    const nodes = g.append("g")
      .selectAll("g")
      .data(visualData.nodes)
      .enter()
      .append("g")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Add circles to nodes
    nodes.append("circle")
      .attr("r", d => d.type === 'app' ? 30 : 20)
      .attr("fill", d => {
        switch(d.type) {
          case 'app': return '#7c5cff';
          case 'summary': return '#4CAF50';
          case 'users': return '#2196F3';
          case 'redFlags': return '#f44336';
          case 'tasks': return '#FF9800';
          case 'terms': return '#9C27B0';
          default: return '#78909C';
        }
      });

    // Helper function to truncate text
    function truncateText(text, maxLength = 15) {
      if (!text) return '';
      return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }

    // Add labels to nodes
    nodes.append("text")
      .text(d => truncateText(d.name))
      .attr("text-anchor", "middle")
      .attr("dy", 30)
      .attr("fill", "white")
      .style("font-size", "12px")
      .style("pointer-events", "none"); // Make text not block clicks

    // Add small indicator for nodes with details
    nodes.filter(d => d.details)
      .append("text")
      .text("ⓘ")
      .attr("text-anchor", "middle")
      .attr("dy", -15)
      .attr("fill", "white")
      .attr("opacity", 0.7)
      .style("font-size", "10px")
      .style("pointer-events", "none");

    // Add click handler for nodes
    nodes.on("click", (event, d) => {
      setSelectedNode(d);
    });

    // Update positions on simulation tick
    simulation.on("tick", () => {
      links
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      nodes.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    // Drag functions
    function dragstarted(event) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
  };
  
  useEffect(() => {
    if (data) {
      createVisualization(data);
    }
    
    // Handle fullscreen change events
    const handleFullscreenChange = () => {
      setIsFullscreen(
        document.fullscreenElement || 
        document.webkitFullscreenElement || 
        document.msFullscreenElement
      );
    };
    
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('msfullscreenchange', handleFullscreenChange);
    
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('msfullscreenchange', handleFullscreenChange);
    };
  }, [data, layoutSettings]);

  return (
    <Container>
      <SectionTitle>
        Topic Dependencies
        <ButtonGroup>
          <Button onClick={handleRefresh} disabled={isRefreshing}>
            <RefreshIcon /> Refresh
          </Button>
          <Button onClick={() => setIsSettingsOpen(true)}>
            <SettingsIcon /> Settings
          </Button>
          <Button onClick={() => setIsAddNodeOpen(true)}>
            <AddIcon /> Add Topic
          </Button>
        </ButtonGroup>
      </SectionTitle>
      
      <div style={{ position: 'relative', width: '100%', height: '600px' }}>
        <svg ref={svgRef} style={{ width: '100%', height: '100%', background: '#1e2130' }}></svg>
        
        <ZoomControls>
          <ZoomButton onClick={handleZoomIn} title="Zoom In">
            <ZoomInIcon />
          </ZoomButton>
          <ZoomButton onClick={handleZoomOut} title="Zoom Out">
            <ZoomOutIcon />
          </ZoomButton>
          <ZoomButton onClick={handleResetView} title="Reset View">
            <RestoreIcon />
          </ZoomButton>
          <ZoomButton onClick={toggleFullscreen} title="Toggle Fullscreen">
            <FullscreenIcon />
          </ZoomButton>
        </ZoomControls>
        
        <HelpText>
          Drag to move nodes • Scroll to zoom • Click node for details
        </HelpText>
      </div>
      
      {/* Node details modal */}
      <NodeDetailsModal 
        node={selectedNode} 
        onClose={() => setSelectedNode(null)} 
      />
      
      {/* Settings modal */}
      {isSettingsOpen && (
        <SettingsModal onClick={() => setIsSettingsOpen(false)}>
          <SettingsContent onClick={e => e.stopPropagation()}>
            <CloseButton onClick={() => setIsSettingsOpen(false)}>&times;</CloseButton>
            <SettingsTitle>Mind Map Settings</SettingsTitle>
            
            <FormGroup>
              <Label>Force Strength</Label>
              <Input 
                type="range" 
                min="-500" 
                max="-100" 
                value={layoutSettings.forceStrength} 
                onChange={e => setLayoutSettings({...layoutSettings, forceStrength: parseInt(e.target.value)})}
              />
              <span>{layoutSettings.forceStrength}</span>
            </FormGroup>
            
            <FormGroup>
              <Label>Link Distance</Label>
              <Input 
                type="range" 
                min="50" 
                max="300" 
                value={layoutSettings.linkDistance} 
                onChange={e => setLayoutSettings({...layoutSettings, linkDistance: parseInt(e.target.value)})}
              />
              <span>{layoutSettings.linkDistance}</span>
            </FormGroup>
            
            <FormGroup>
              <Label>Centering Force</Label>
              <Input 
                type="range" 
                min="0" 
                max="1" 
                step="0.1" 
                value={layoutSettings.centeringForce} 
                onChange={e => setLayoutSettings({...layoutSettings, centeringForce: parseFloat(e.target.value)})}
              />
              <span>{layoutSettings.centeringForce}</span>
            </FormGroup>
            
            <Button onClick={applySettings}>Apply Settings</Button>
          </SettingsContent>
        </SettingsModal>
      )}
      
      {/* Add node modal */}
      {isAddNodeOpen && (
        <AddNodeModal onClick={() => setIsAddNodeOpen(false)}>
          <AddNodeContent onClick={e => e.stopPropagation()}>
            <CloseButton onClick={() => setIsAddNodeOpen(false)}>&times;</CloseButton>
            <SettingsTitle>Add New Topic</SettingsTitle>
            
            <FormGroup>
              <Label>Topic Name</Label>
              <Input 
                type="text" 
                value={newNode.name} 
                onChange={e => setNewNode({...newNode, name: e.target.value})}
                placeholder="Enter topic name"
              />
            </FormGroup>
            
            <FormGroup>
              <Label>Topic Type</Label>
              <Select 
                value={newNode.type} 
                onChange={e => setNewNode({...newNode, type: e.target.value})}
              >
                <option value="summary">Topic/Summary</option>
                <option value="users">Participant/User</option>
                <option value="tasks">Action Item/Task</option>
                <option value="redFlags">Decision/Red Flag</option>
                <option value="terms">Technical Term</option>
              </Select>
            </FormGroup>
            
            <FormGroup>
              <Label>Details</Label>
              <Textarea 
                value={newNode.details} 
                onChange={e => setNewNode({...newNode, details: e.target.value})}
                placeholder="Enter topic details"
              />
            </FormGroup>
            
            <FormGroup>
              <Label>Connect to</Label>
              <Select 
                value={newNode.parent} 
                onChange={e => setNewNode({...newNode, parent: e.target.value})}
              >
                <option value="">Select parent node</option>
                {getParentOptions().map(option => (
                  <option key={option.id} value={option.id}>{option.name}</option>
                ))}
              </Select>
            </FormGroup>
            
            <Button 
              onClick={handleAddNode}
              disabled={!newNode.name || !newNode.parent}
            >
              Add Topic
            </Button>
          </AddNodeContent>
        </AddNodeModal>
      )}
    </Container>
  );
};

export default MindMap; 