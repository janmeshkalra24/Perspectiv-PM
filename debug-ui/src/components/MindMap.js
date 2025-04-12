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
  padding: 24px;
  background: linear-gradient(145deg, #1a1d2e, #1e2235);
  border-radius: 16px;
  margin-bottom: 24px;
  position: relative;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.05);
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
  background: linear-gradient(145deg, #2a2d3e, #2e3245);
  color: #fff;
  border: none;
  padding: 10px 18px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.2s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  
  &:hover {
    background: linear-gradient(145deg, #2e3245, #2a2d3e);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  &:active {
    transform: translateY(0);
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
  background: linear-gradient(145deg, #2a2d3e, #2e3245);
  color: white;
  border: none;
  width: 44px;
  height: 44px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  
  &:hover {
    background: linear-gradient(145deg, #2e3245, #2a2d3e);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  &:active {
    transform: translateY(0);
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

const Tooltip = styled.div`
  font-size: 12px;
  color: #a5a8b6;
  margin: -8px 0 4px;
  font-style: italic;
`;

const ResetButton = styled(Button)`
  margin-left: 10px;
  background: linear-gradient(145deg, #2e3245, #2a2d3e);
  
  &:hover {
    background: linear-gradient(145deg, #2a2d3e, #2e3245);
  }
`;

const MindMap = ({ data: initialData, onRefresh }) => {
  const svgRef = useRef();
  const [selectedNode, setSelectedNode] = useState(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isAddNodeOpen, setIsAddNodeOpen] = useState(false);
  const [layoutSettings, setLayoutSettings] = useState({
    forceStrength: -400,
    linkDistance: 120,
    centeringForce: 0.15
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

    // Add utility functions for text processing
    function cleanMarkdown(text) {
      if (!text) return '';
      return text
        // Remove markdown bold syntax
        .replace(/\*\*(.*?)\*\*/g, '$1')
        // Remove markdown italic syntax
        .replace(/\*(.*?)\*/g, '$1')
        // Remove markdown code blocks
        .replace(/```[\s\S]*?```/g, '')
        // Remove markdown inline code
        .replace(/`([^`]+)`/g, '$1')
        // Remove markdown links
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        // Remove markdown headers
        .replace(/#{1,6}\s+/g, '')
        // Remove bullet points
        .replace(/^\s*[-*+]\s+/gm, '')
        // Remove numbered lists
        .replace(/^\s*\d+\.\s+/gm, '')
        // Remove excessive whitespace
        .replace(/\s+/g, ' ')
        .trim();
    }

    function formatNodeTitle(node) {
      if (!node.name || !node.details) {
        return truncateText(node.name, 25);
      }

      let title = '';
      // Enhanced formatting based on node type
      switch (node.type) {
        case 'tasks':
          const actionMatch = node.details.match(/(?:need to|should|must|will|todo:?)\s+([^,.!?]+)/i);
          title = actionMatch ? `📋 ${actionMatch[1]}` : `📋 ${node.name}`;
          break;
        
        case 'redFlags':
          const riskMatch = node.details.match(/(?:risk|blocker|issue)[s]?\s*[:]\s*([^,.!?]+)/i);
          title = riskMatch ? `⚠️ ${riskMatch[1]}` : `⚠️ ${node.name}`;
          break;
        
        case 'terms':
          const termMatch = node.details.match(/([^:]+):\s*([^,.!?]+)/);
          title = termMatch ? `📚 ${termMatch[1]}` : `📚 ${node.name}`;
          break;
        
        case 'summary':
          const summaryMatch = node.details.match(/(?:decided|agreed|concluded|determined)\s+([^,.!?]+)/i) ||
                             node.details.match(/([^,.!?]+(?:improves|enhances|enables|provides|supports)[^,.!?]+)/i);
          title = summaryMatch ? `💡 ${summaryMatch[1]}` : `💡 ${node.name}`;
          break;
        
        case 'app':
          title = `🔷 ${node.name}`;
          break;
          
        default:
          title = node.name;
      }
      
      return truncateText(cleanMarkdown(title), 30);
    }

    // Helper function to truncate text at word boundary
    function truncateText(text, maxLength = 25) {
      if (!text) return '';
      const cleaned = cleanMarkdown(text)
        // Remove any LLM prefixes like "Here is a detailed analysis..."
        .replace(/^(?:here is|here's|this is|i have|i've|i will|i'll|let me|let's)[^:]*:\s*/i, '')
        .trim();
      
      if (cleaned.length <= maxLength) return cleaned;
      
      // Try to find a good breakpoint near the maxLength
      const breakPoints = cleaned.substring(0, maxLength).split(/[\s,.;:-]/);
      breakPoints.pop(); // Remove last partial word/segment
      const truncated = breakPoints.join(' ').trim();
      
      return truncated.length > 0 ? truncated + '...' : cleaned.substring(0, maxLength - 3) + '...';
    }

    function formatNodeContent(content) {
      if (!content) return '';
      const parts = content.split('\n\n');
      // Take the first meaningful chunk of content
      const firstPart = parts.find(part => 
        part.length > 10 && !part.match(/^[#\s-*]/));
      return cleanMarkdown(firstPart || parts[0]);
    }

    function formatTooltipContent(content) {
      if (!content) return '';
      
      // Clean up markdown artifacts
      let cleanContent = content
        .replace(/```[a-z]*\n/g, '') // Remove code block markers
        .replace(/`/g, '') // Remove inline code markers
        .replace(/\*\*/g, '') // Remove bold markers
        .replace(/\n+/g, '<br/>') // Convert newlines to HTML breaks
        .replace(/- /g, '• ') // Convert markdown lists to bullet points
        .trim();
      
      // Add emoji indicators based on content
      if (cleanContent.match(/(?:risk|blocker|issue)[s]?:/i)) {
        cleanContent = '⚠️ ' + cleanContent;
      } else if (cleanContent.match(/(?:need to|should|must|will|todo):/i)) {
        cleanContent = '📋 ' + cleanContent;
      } else if (cleanContent.match(/(?:decided|agreed|concluded):/i)) {
        cleanContent = '✅ ' + cleanContent;
      }
      
      return `<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        ${cleanContent}
      </div>`;
    }

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
      .force("link", d3.forceLink(visualData.links).id(d => d.id).distance(d => {
        // Adjust distances based on node types
        if (d.source.type === 'app' || d.target.type === 'app') return layoutSettings.linkDistance * 1.2;
        if (d.source.id.startsWith('sprint-') || d.target.id.startsWith('sprint-')) return layoutSettings.linkDistance * 0.8;
        if (d.source.type === 'redFlags' || d.target.type === 'redFlags') return layoutSettings.linkDistance * 1.1;
        return layoutSettings.linkDistance;
      }))
      .force("charge", d3.forceManyBody()
        .strength(d => {
          // Adjust repulsion based on node type
          if (d.type === 'app') return layoutSettings.forceStrength * 1.5;
          if (d.type === 'summary') return layoutSettings.forceStrength * 1.2;
          if (d.type === 'redFlags') return layoutSettings.forceStrength * 1.1;
          return layoutSettings.forceStrength;
        }))
      .force("center", d3.forceCenter(containerWidth / 2, containerHeight / 2).strength(layoutSettings.centeringForce))
      .force("collision", d3.forceCollide().radius(d => {
        // Adjust collision radius based on node type
        if (d.type === 'app') return 50;
        if (d.type === 'summary') return 45;
        if (d.type === 'redFlags') return 42;
        return 40;
      }))
      .force("x", d3.forceX(containerWidth / 2).strength(0.08))
      .force("y", d3.forceY(containerHeight / 2).strength(0.08));

    // Create gradient definitions
    const defs = svg.append("defs");
    
    // Define gradients for each node type
    const gradients = {
      app: ["#7c4dff", "#651fff"],
      summary: ["#00e676", "#00c853"],
      users: ["#00b0ff", "#0091ea"],
      redFlags: ["#ff5252", "#ff1744"],
      tasks: ["#ffab40", "#ff9100"],
      terms: ["#e040fb", "#d500f9"],
      default: ["#78909c", "#546e7a"]
    };
    
    Object.entries(gradients).forEach(([type, [color1, color2]]) => {
      const gradient = defs.append("radialGradient")
        .attr("id", `gradient-${type}`)
        .attr("cx", "30%")
        .attr("cy", "30%")
        .attr("r", "70%");
        
      gradient.append("stop")
        .attr("offset", "0%")
        .attr("style", `stop-color: ${color1}; stop-opacity: 1`);
        
      gradient.append("stop")
        .attr("offset", "100%")
        .attr("style", `stop-color: ${color2}; stop-opacity: 1`);
    });

    // Create links with improved styling
    const links = g.append("g")
      .selectAll("line")
      .data(visualData.links)
      .enter()
      .append("line")
      .attr("stroke", d => {
        if (d.source.id.startsWith('sprint-') || d.target.id.startsWith('sprint-')) return "#4CAF50";
        if (d.source.id.startsWith('risk-') || d.target.id.startsWith('risk-')) return "#f44336";
        return "#4a4a4a";
      })
      .attr("stroke-width", d => {
        if (d.source.id.startsWith('sprint-') || d.target.id.startsWith('sprint-')) return 2.5;
        if (d.source.type === 'app' || d.target.type === 'app') return 2;
        return 1.5;
      })
      .attr("stroke-opacity", 0.6);

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

    // Add circles to nodes with enhanced styling
    nodes.append("circle")
      .attr("r", d => {
        switch(d.type) {
          case 'app': return 40;
          case 'summary': return 35;
          case 'tasks': return 32;
          case 'redFlags': return 32;
          case 'users': return 30;
          case 'terms': return 28;
          default: return 28;
        }
      })
      .attr("fill", d => `url(#gradient-${d.type || 'default'})`)
      .attr("stroke", d => {
        if (d.id.startsWith('sprint-')) return "#81c784";
        if (d.id.startsWith('goal-')) return "#4caf50";
        if (d.id.startsWith('risk-') || d.id.startsWith('dep-')) return "#ef5350";
        if (d.id.startsWith('blocker-')) return "#f44336";
        if (d.type === 'app') return "#7e57c2";
        if (d.type === 'summary') return "#42a5f5";
        if (d.type === 'tasks') return "#66bb6a";
        if (d.type === 'redFlags') return "#ef5350";
        return "rgba(255,255,255,0.1)";
      })
      .attr("stroke-width", d => {
        if (d.id.startsWith('sprint-') || d.id.startsWith('goal-')) return 3;
        if (d.type === 'app') return 2.5;
        if (d.type === 'redFlags') return 2;
        return 1.5;
      })
      .style("filter", "drop-shadow(0 4px 8px rgba(0,0,0,0.2))")
      .style("transition", "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)");

    // Add labels to nodes with improved styling
    nodes.append("text")
      .text(d => formatNodeTitle(d))
      .attr("text-anchor", "middle")
      .attr("dy", d => {
        switch(d.type) {
          case 'app': return 42;
          case 'summary': return 38;
          default: return 35;
        }
      })
      .attr("fill", "white")
      .style("font-size", d => {
        if (d.id.startsWith('sprint-') || 
            d.id === 'feature-status' ||
            d.id === 'dependencies-constraints' ||
            d.id === 'risks-resources' ||
            d.id === 'stakeholder-requests') {
          return "16px";
        }
        if (d.type === 'app') return "15px";
        return "14px";
      })
      .style("font-weight", d => {
        if (d.id.startsWith('sprint-') || 
            d.id === 'feature-status' ||
            d.id === 'dependencies-constraints' ||
            d.id === 'risks-resources' ||
            d.id === 'stakeholder-requests') {
          return "600";
        }
        if (d.type === 'app') return "500";
        return "400";
      })
      .style("letter-spacing", "0.03em")
      .style("text-shadow", "0 2px 4px rgba(0,0,0,0.3)")
      .style("pointer-events", "none");

    // Add icons/indicators for nodes with details
    nodes.filter(d => d.details)
      .append("text")
      .text(d => {
        // Use different icons based on node type
        if (d.id.startsWith('risk-')) return "⚠";
        if (d.id.startsWith('goal-')) return "🎯";
        if (d.id.startsWith('metric-')) return "📊";
        if (d.id.startsWith('feature-')) return "✨";
        if (d.id.startsWith('dep-')) return "🔄";
        if (d.id.startsWith('constraint-')) return "⛔";
        if (d.id.startsWith('resource-')) return "📋";
        if (d.id.startsWith('request-')) return "💬";
        return "ⓘ";
      })
      .attr("text-anchor", "middle")
      .attr("dy", -15)
      .attr("fill", "white")
      .attr("opacity", 0.9)
      .style("font-size", "12px")
      .style("pointer-events", "none");

    // Add hover effects with smoother transitions
    nodes.on("mouseover", function(event, d) {
      // Highlight connected nodes and links with smooth transition
      nodes.transition().duration(200)
        .style("opacity", n => {
          const isConnected = visualData.links.some(link => 
            (link.source.id === d.id && link.target.id === n.id) ||
            (link.target.id === d.id && link.source.id === n.id)
          );
          return isConnected || n.id === d.id ? 1 : 0.2;
        });
        
      links.transition().duration(200)
        .style("opacity", l => 
          l.source.id === d.id || l.target.id === d.id ? 1 : 0.1
        )
        .attr("stroke-width", l => {
          const baseWidth = l.source.id.startsWith('sprint-') || l.target.id.startsWith('sprint-') ? 2.5 :
                           l.source.type === 'app' || l.target.type === 'app' ? 2 : 1.5;
          return (l.source.id === d.id || l.target.id === d.id) ? baseWidth * 1.5 : baseWidth;
        });

      // Scale up the hovered node slightly
      d3.select(this).select("circle")
        .transition()
        .duration(200)
        .attr("r", d => {
          const baseRadius = d.type === 'app' ? 35 :
                           d.type === 'summary' ? 30 :
                           d.type === 'tasks' || d.type === 'redFlags' || d.type === 'users' ? 25 :
                           d.type === 'terms' ? 22 : 22;
          return baseRadius * 1.1;
        });

      // Show tooltip with enhanced styling
      if (d.details) {
        const tooltip = d3.select("body").append("div")
          .attr("class", "mindmap-tooltip")
          .style("position", "absolute")
          .style("background", "rgba(26, 29, 46, 0.98)")
          .style("color", "white")
          .style("padding", "16px")
          .style("border-radius", "12px")
          .style("font-size", "13px")
          .style("max-width", "320px")
          .style("pointer-events", "none")
          .style("z-index", 1000)
          .style("line-height", "1.5")
          .style("box-shadow", "0 8px 32px rgba(0,0,0,0.24)")
          .style("border", "1px solid rgba(255,255,255,0.08)")
          .style("backdrop-filter", "blur(12px)")
          .style("opacity", 0)
          .style("transform", "translateY(10px)");
        
        tooltip.html(formatTooltipContent(d.details))
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 10) + "px")
          .transition()
          .duration(200)
          .style("opacity", 1)
          .style("transform", "translateY(0)");
      }
    })
    .on("mouseout", function(event, d) {
      // Reset highlights with smooth transition
      nodes.transition().duration(200)
        .style("opacity", 1);
      
      links.transition().duration(200)
        .style("opacity", 0.6)
        .attr("stroke-width", l => 
          l.source.id.startsWith('sprint-') || l.target.id.startsWith('sprint-') ? 2.5 :
          l.source.type === 'app' || l.target.type === 'app' ? 2 : 1.5
        );

      // Scale down the node
      d3.select(this).select("circle")
        .transition()
        .duration(200)
        .attr("r", d => {
          return d.type === 'app' ? 35 :
                 d.type === 'summary' ? 30 :
                 d.type === 'tasks' || d.type === 'redFlags' || d.type === 'users' ? 25 :
                 d.type === 'terms' ? 22 : 22;
        });
      
      // Remove tooltip with fade out
      d3.selectAll(".mindmap-tooltip")
        .transition()
        .duration(200)
        .style("opacity", 0)
        .style("transform", "translateY(10px)")
        .remove();
    });

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
              <Label>Node Spacing (Force Strength)</Label>
              <Tooltip>Adjusts how far apart nodes push each other. More negative values create more space between nodes.</Tooltip>
              <Input 
                type="range" 
                min="-600" 
                max="-200" 
                value={layoutSettings.forceStrength} 
                onChange={e => setLayoutSettings({...layoutSettings, forceStrength: parseInt(e.target.value)})}
              />
              <span>{layoutSettings.forceStrength}</span>
            </FormGroup>
            
            <FormGroup>
              <Label>Connection Length</Label>
              <Tooltip>Controls the preferred length of connections between nodes. Higher values spread out the network.</Tooltip>
              <Input 
                type="range" 
                min="80" 
                max="200" 
                value={layoutSettings.linkDistance} 
                onChange={e => setLayoutSettings({...layoutSettings, linkDistance: parseInt(e.target.value)})}
              />
              <span>{layoutSettings.linkDistance}px</span>
            </FormGroup>
            
            <FormGroup>
              <Label>Center Gravity</Label>
              <Tooltip>Determines how strongly nodes are pulled toward the center. Higher values create a more compact layout.</Tooltip>
              <Input 
                type="range" 
                min="0" 
                max="0.3" 
                step="0.05" 
                value={layoutSettings.centeringForce} 
                onChange={e => setLayoutSettings({...layoutSettings, centeringForce: parseFloat(e.target.value)})}
              />
              <span>{layoutSettings.centeringForce}</span>
            </FormGroup>
            
            <Button onClick={applySettings}>Apply Settings</Button>
            <ResetButton onClick={() => {
              setLayoutSettings({
                forceStrength: -400,
                linkDistance: 120,
                centeringForce: 0.15
              });
              setTimeout(applySettings, 0);
            }}>Reset to Default</ResetButton>
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