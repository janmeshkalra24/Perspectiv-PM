import React from 'react';
import styled from 'styled-components';
import ReactMarkdown from 'react-markdown';

const ModalOverlay = styled.div`
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

const ModalContent = styled.div`
  background: #1e2130;
  padding: 24px;
  border-radius: 8px;
  max-width: 600px;
  width: 90%;
  color: white;
  position: relative;
  max-height: 80vh;
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

const Title = styled.h3`
  margin: 0 0 16px 0;
  color: #a5a8b6;
  font-size: 20px;
  padding-right: 20px;
`;

const NodeType = styled.div`
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  margin-bottom: 16px;
  font-weight: bold;
  color: white;
  background-color: ${props => {
    switch(props.type) {
      case 'app': return '#7c5cff';
      case 'summary': return '#4CAF50';
      case 'users': return '#2196F3';
      case 'redFlags': return '#f44336';
      case 'tasks': return '#FF9800';
      case 'terms': return '#9C27B0';
      default: return '#78909C';
    }
  }}
`;

const Details = styled.div`
  color: #e1e1e1;
  font-size: 14px;
  line-height: 1.6;
  margin-top: 12px;
  
  p {
    margin-bottom: 12px;
  }
  
  ul, ol {
    margin-left: 20px;
    margin-bottom: 12px;
  }
  
  strong {
    color: white;
    font-weight: bold;
  }
  
  h1, h2, h3, h4 {
    color: #a5a8b6;
    margin-top: 16px;
    margin-bottom: 8px;
  }
  
  code {
    background: rgba(255, 255, 255, 0.1);
    padding: 2px 4px;
    border-radius: 3px;
  }
  
  blockquote {
    border-left: 3px solid #a5a8b6;
    padding-left: 16px;
    margin-left: 0;
    color: #a5a8b6;
  }
`;

const NodeDetailsModal = ({ node, onClose }) => {
  if (!node) return null;
  
  // Get a friendly type name for display
  const getTypeName = (type) => {
    switch(type) {
      case 'app': return 'Category';
      case 'summary': return 'Summary';
      case 'users': return 'People';
      case 'redFlags': return 'Red Flag';
      case 'tasks': return 'Action Item';
      case 'terms': return 'Technical Term';
      default: return type;
    }
  };

  return (
    <ModalOverlay onClick={onClose}>
      <ModalContent onClick={e => e.stopPropagation()}>
        <CloseButton onClick={onClose}>&times;</CloseButton>
        <Title>{node.name}</Title>
        <NodeType type={node.type}>{getTypeName(node.type)}</NodeType>
        <Details>
          <ReactMarkdown>{node.details}</ReactMarkdown>
        </Details>
      </ModalContent>
    </ModalOverlay>
  );
};

export default NodeDetailsModal; 