import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  Grid,
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider,
  CircularProgress,
} from '@mui/material';
import TimelineIcon from '@mui/icons-material/Timeline';
import CodeIcon from '@mui/icons-material/Code';
import GroupIcon from '@mui/icons-material/Group';
import PsychologyIcon from '@mui/icons-material/Psychology';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

function TeamMemberInsights() {
  const [insights, setInsights] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchInsights();
    // Set up polling for updates
    const interval = setInterval(fetchInsights, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchInsights = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/team_insights`);
      const data = await response.json();
      setInsights(data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching team insights:', error);
      setLoading(false);
    }
  };

  const getSkillColor = (skill) => {
    const colors = {
      technical: 'primary',
      communication: 'success',
      leadership: 'warning',
      problem_solving: 'error',
      collaboration: 'info',
    };
    return colors[skill.toLowerCase()] || 'default';
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Team Member Insights
      </Typography>
      
      <Grid container spacing={3}>
        {Object.entries(insights).map(([memberId, memberInsights]) => (
          <Grid item xs={12} md={6} key={memberId}>
            <Paper elevation={2} sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                {memberInsights.name}
              </Typography>
              
              <Box mb={2}>
                <Typography variant="subtitle2" color="textSecondary">
                  Mentioned {memberInsights.mentionCount} times in video frames
                </Typography>
              </Box>
              
              <Box mb={2}>
                <Typography variant="subtitle1" gutterBottom>
                  Skills & Expertise
                </Typography>
                <Box display="flex" flexWrap="wrap" gap={1}>
                  {memberInsights.skills.map((skill, index) => (
                    <Chip
                      key={index}
                      icon={<CodeIcon />}
                      label={skill}
                      color={getSkillColor(skill)}
                      size="small"
                    />
                  ))}
                </Box>
              </Box>
              
              <Box mb={2}>
                <Typography variant="subtitle1" gutterBottom>
                  Recent Activities
                </Typography>
                <List dense>
                  {memberInsights.recentActivities.map((activity, index) => (
                    <React.Fragment key={index}>
                      <ListItem>
                        <ListItemText
                          primary={activity.description}
                          secondary={new Date(activity.timestamp).toLocaleString()}
                        />
                      </ListItem>
                      {index < memberInsights.recentActivities.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </Box>
              
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Collaboration Patterns
                </Typography>
                <Box display="flex" flexWrap="wrap" gap={1}>
                  {memberInsights.collaborationPatterns.map((pattern, index) => (
                    <Chip
                      key={index}
                      icon={<GroupIcon />}
                      label={pattern}
                      variant="outlined"
                      size="small"
                    />
                  ))}
                </Box>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
}

export default TeamMemberInsights; 