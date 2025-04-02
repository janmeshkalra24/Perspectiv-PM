import React, { useState } from 'react';
import styled from 'styled-components';

const Container = styled.div`
  padding: 20px;
  background: #1e2130;
  border-radius: 8px;
  margin-bottom: 20px;
`;

const SectionTitle = styled.h2`
  color: #a5a8b6;
  font-size: 24px;
  margin-bottom: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Warning = styled.div`
  background: rgba(255, 204, 0, 0.1);
  border-left: 4px solid #ffd60a;
  padding: 12px 16px;
  border-radius: 4px;
  margin-bottom: 12px;
  
  &:first-of-type {
    margin-top: 12px;
  }
`;

const ActionTitle = styled.div`
  color: #ffffff;
  font-weight: 600;
  margin-bottom: 8px;
`;

const ActionTime = styled.div`
  color: #8e8e93;
  font-size: 12px;
  display: flex;
  align-items: center;
`;

const TimestampIcon = styled.span`
  margin-right: 5px;
  opacity: 0.7;
`;

const WarningsSection = ({ warnings }) => {
  // Format timestamp to be more readable
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return "Unknown time";
    
    // Check if the timestamp is already in MM:SS format
    if (/^\d+:\d+$/.test(timestamp)) {
      return timestamp;
    }
    
    // Check if it's a full time string like "10:15:20 AM"
    if (/\d+:\d+:\d+/.test(timestamp)) {
      return timestamp;
    }
    
    // Try to parse the timestamp as a number of seconds
    const seconds = parseFloat(timestamp);
    if (!isNaN(seconds)) {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = Math.floor(seconds % 60);
      return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    
    // Default to the original timestamp
    return timestamp;
  };

  return (
    <Container>
      <SectionTitle>Executive Summary</SectionTitle>
      
      {warnings.map((warning, index) => (
        <Warning key={index}>
          <ActionTitle>{warning.title || 'Open Action Item'}</ActionTitle>
          <ActionTime>
            <TimestampIcon>⏱</TimestampIcon> {formatTimestamp(warning.time)}
          </ActionTime>
          <div style={{ color: '#d1d1d6', marginTop: '8px' }}>
            {warning.details || warning.pm_insights || warning.description}
          </div>
        </Warning>
      ))}
    </Container>
  );
};

export default WarningsSection; 