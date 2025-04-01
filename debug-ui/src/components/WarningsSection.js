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

const Summary = styled.div`
  background: rgba(255, 255, 255, 0.05);
  padding: 12px 16px;
  border-radius: 4px;
  margin-bottom: 20px;
  color: #ffffff;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const SummaryStats = styled.div`
  display: flex;
  gap: 16px;
`;

const Stat = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const StatValue = styled.div`
  font-size: 24px;
  font-weight: bold;
  color: ${props => props.color || '#ffffff'};
`;

const StatLabel = styled.div`
  font-size: 12px;
  color: #a5a8b6;
`;

const FlagContainer = styled.div`
  margin-bottom: 16px;
`;

const RedFlag = styled.div`
  background: rgba(255, 69, 58, 0.1);
  border-left: 4px solid #ff453a;
  padding: 12px 16px;
  border-radius: 4px;
  margin-bottom: 12px;
  cursor: pointer;
`;

const Warning = styled.div`
  background: rgba(255, 204, 0, 0.1);
  border-left: 4px solid #ffd60a;
  padding: 12px 16px;
  border-radius: 4px;
  margin-bottom: 12px;
  cursor: pointer;
`;

const FlagTitle = styled.div`
  color: #ffffff;
  font-weight: 600;
  margin-bottom: 4px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const FlagTime = styled.div`
  color: #8e8e93;
  font-size: 12px;
  margin-bottom: ${props => props.expanded ? '8px' : '0'};
  display: flex;
  align-items: center;
`;

const TimestampIcon = styled.span`
  margin-right: 5px;
  opacity: 0.7;
`;

const FlagDetails = styled.div`
  color: #d1d1d6;
  font-size: 14px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  line-height: 1.5;
`;

const SolutionBox = styled.div`
  background: rgba(58, 129, 255, 0.1);
  border-left: 4px solid #3a81ff;
  padding: 12px;
  margin-top: 8px;
  border-radius: 4px;
`;

const SolutionTitle = styled.div`
  color: #3a81ff;
  font-weight: 600;
  margin-bottom: 4px;
  display: flex;
  align-items: center;
`;

const SolutionIcon = styled.span`
  margin-right: 6px;
`;

const TagContainer = styled.div`
  display: flex;
  gap: 8px;
  margin-top: 8px;
  flex-wrap: wrap;
`;

const Tag = styled.span`
  background: rgba(255, 255, 255, 0.1);
  color: #d1d1d6;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  white-space: nowrap;
`;

const SeverityIndicator = styled.span`
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: ${props => props.color};
  margin-right: 8px;
`;

const WarningsSection = ({ warnings, redFlags }) => {
  // State to track which items are expanded
  const [expandedItems, setExpandedItems] = useState({});

  // Toggle expanded state for an item
  const toggleExpanded = (id) => {
    setExpandedItems(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  // Generate suggested solutions based on the title
  const getSolution = (title) => {
    const solutions = {
      "Timeline Risk": "Review project timeline and milestones. Consider re-prioritizing tasks or requesting deadline extensions if necessary.",
      "Budget Concern": "Conduct budget review meeting. Identify areas for cost reduction or request additional funding with detailed justification.",
      "Resource Constraint": "Identify bottlenecks and consider resource reallocation or temporary staffing solutions.",
      "Critical Technical Issue": "Escalate to senior technical team members. Consider implementing a temporary workaround while developing a permanent solution.",
      "Compliance Risk": "Consult with legal team immediately. Document all compliance concerns and create action plan to address them.",
      "Security Risk": "Initiate security assessment. Consider limiting affected functionality until issue is resolved.",
      "Scope Management Issue": "Schedule scope review meeting. Prioritize requirements and consider splitting into multiple phases.",
      "Stakeholder Concern": "Schedule stakeholder alignment meeting. Identify key concerns and develop action plan to address them.",
      "Requirements Clarity Issue": "Schedule requirements refinement session with product owner and key stakeholders.",
      "Technical Debt Concern": "Allocate time in upcoming sprints to address technical debt. Consider a technical debt reduction plan.",
      "Testing Coverage Issue": "Increase test coverage by implementing automated testing. Consider pair programming for critical components.",
      "External Dependency": "Identify alternatives or workarounds. Establish clear communication channels with dependency providers.",
      "Communication Issue": "Implement regular sync meetings. Document and distribute key decisions and action items.",
      "Documentation Needed": "Allocate time for documentation. Consider implementing documentation-as-code practices.",
      "Skill or Training Gap": "Arrange training sessions or mentoring. Consider bringing in an expert consultant for knowledge transfer.",
      "Risk Management": "Develop comprehensive risk matrix with mitigation strategies. Review regularly in team meetings."
    };

    return solutions[title] || "Review the issue with the team and develop a specific action plan to address it.";
  };

  // Extract relevant tags based on the item details
  const getTags = (details) => {
    const tagPatterns = [
      { regex: /timeline|deadline|schedule|delay/i, tag: "Timeline" },
      { regex: /budget|cost|expense|funding/i, tag: "Budget" },
      { regex: /resource|staffing|capacity|bandwidth/i, tag: "Resources" },
      { regex: /technical|bug|issue|error|crash/i, tag: "Technical" },
      { regex: /compliance|legal|regulation|policy/i, tag: "Compliance" },
      { regex: /security|vulnerability|breach|risk/i, tag: "Security" },
      { regex: /scope|requirement|feature|change/i, tag: "Scope" },
      { regex: /stakeholder|client|customer|user/i, tag: "Stakeholders" },
      { regex: /test|quality|QA|coverage/i, tag: "Testing" },
      { regex: /dependency|external|integration|vendor/i, tag: "Dependencies" },
      { regex: /communication|meeting|alignment|team/i, tag: "Communication" },
      { regex: /documentation|doc|knowledge/i, tag: "Documentation" },
      { regex: /training|skill|experience|expertise/i, tag: "Skills" },
      { regex: /high priority|urgent|critical|immediate/i, tag: "High Priority" },
      { regex: /backend|api|database|server/i, tag: "Backend" },
      { regex: /frontend|ui|interface|user experience/i, tag: "Frontend" },
      { regex: /infrastructure|deployment|devops|cloud/i, tag: "Infrastructure" }
    ];

    const tags = [];
    if (!details) return tags;

    tagPatterns.forEach(pattern => {
      if (pattern.regex.test(details) && !tags.includes(pattern.tag)) {
        tags.push(pattern.tag);
      }
    });

    return tags;
  };

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

  // Get unique ID for each item to track expansion state
  const getItemId = (item, index, type) => `${type}-${index}-${item.title.replace(/\s+/g, '-')}`;

  return (
    <Container>
      <SectionTitle>Executive Summary</SectionTitle>
      
      <Summary>
        <div>
          This summary highlights key issues requiring attention, with {redFlags.length} critical items and {warnings.length} potential concerns.
        </div>
        <SummaryStats>
          <Stat>
            <StatValue color="#ff453a">{redFlags.length}</StatValue>
            <StatLabel>Critical</StatLabel>
          </Stat>
          <Stat>
            <StatValue color="#ffd60a">{warnings.length}</StatValue>
            <StatLabel>Warnings</StatLabel>
          </Stat>
        </SummaryStats>
      </Summary>
      
      {redFlags.length > 0 && (
        <FlagContainer>
          <h3 style={{ color: '#ff453a', marginBottom: '12px' }}>Critical Issues</h3>
          {redFlags.map((flag, index) => {
            const itemId = getItemId(flag, index, 'redFlag');
            const isExpanded = expandedItems[itemId];
            const tags = getTags(flag.details);
            
            return (
              <RedFlag key={itemId} onClick={() => toggleExpanded(itemId)}>
                <FlagTitle>
                  <div>
                    <SeverityIndicator color="#ff453a" />
                    {flag.title}
                  </div>
                  <span>{isExpanded ? '▼' : '▶'}</span>
                </FlagTitle>
                <FlagTime expanded={isExpanded}>
                  <TimestampIcon>⏱</TimestampIcon> {formatTimestamp(flag.time)}
                </FlagTime>
                
                {isExpanded && (
                  <>
                    <FlagDetails>
                      {flag.details}
                      
                      <SolutionBox>
                        <SolutionTitle>
                          <SolutionIcon>💡</SolutionIcon> Recommended Action
                        </SolutionTitle>
                        <div>{getSolution(flag.title)}</div>
                      </SolutionBox>
                      
                      {tags.length > 0 && (
                        <TagContainer>
                          {tags.map(tag => (
                            <Tag key={tag}>{tag}</Tag>
                          ))}
                        </TagContainer>
                      )}
                    </FlagDetails>
                  </>
                )}
              </RedFlag>
            );
          })}
        </FlagContainer>
      )}

      {warnings.length > 0 && (
        <FlagContainer>
          <h3 style={{ color: '#ffd60a', marginBottom: '12px' }}>Potential Concerns</h3>
          {warnings.map((warning, index) => {
            const itemId = getItemId(warning, index, 'warning');
            const isExpanded = expandedItems[itemId];
            const tags = getTags(warning.details);
            
            return (
              <Warning key={itemId} onClick={() => toggleExpanded(itemId)}>
                <FlagTitle>
                  <div>
                    <SeverityIndicator color="#ffd60a" />
                    {warning.title}
                  </div>
                  <span>{isExpanded ? '▼' : '▶'}</span>
                </FlagTitle>
                <FlagTime expanded={isExpanded}>
                  <TimestampIcon>⏱</TimestampIcon> {formatTimestamp(warning.time)}
                </FlagTime>
                
                {isExpanded && (
                  <>
                    <FlagDetails>
                      {warning.details}
                      
                      <SolutionBox>
                        <SolutionTitle>
                          <SolutionIcon>💡</SolutionIcon> Suggested Approach
                        </SolutionTitle>
                        <div>{getSolution(warning.title)}</div>
                      </SolutionBox>
                      
                      {tags.length > 0 && (
                        <TagContainer>
                          {tags.map(tag => (
                            <Tag key={tag}>{tag}</Tag>
                          ))}
                        </TagContainer>
                      )}
                    </FlagDetails>
                  </>
                )}
              </Warning>
            );
          })}
        </FlagContainer>
      )}
    </Container>
  );
};

export default WarningsSection; 