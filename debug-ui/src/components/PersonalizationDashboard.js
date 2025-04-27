import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';

const PersonalizationDashboard = () => {
  const [profiles, setProfiles] = useState({});
  const [open, setOpen] = useState(false);
  const [newProfile, setNewProfile] = useState({ name: '', role: '', skills: '' });

  useEffect(() => {
    fetchProfiles();
  }, []);

  const fetchProfiles = async () => {
    try {
      const response = await fetch('/api/profiles');
      const data = await response.json();
      setProfiles(data);
    } catch (error) {
      console.error('Error fetching profiles:', error);
    }
  };

  const handleAddProfile = async () => {
    if (newProfile.name) {
      try {
        const response = await fetch('/api/profiles', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            name: newProfile.name,
            role: newProfile.role,
            skills: newProfile.skills.split(',').map(s => s.trim()).filter(Boolean),
          }),
        });
        if (response.ok) {
          fetchProfiles();
          handleClose();
        }
      } catch (error) {
        console.error('Error adding profile:', error);
      }
    }
  };

  const handleRemoveProfile = async (name) => {
    try {
      const response = await fetch(`/api/profiles/${encodeURIComponent(name)}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        fetchProfiles();
      }
    } catch (error) {
      console.error('Error removing profile:', error);
    }
  };

  const handleOpen = () => setOpen(true);
  const handleClose = () => {
    setOpen(false);
    setNewProfile({ name: '', role: '', skills: '' });
  };

  return (
    <Box sx={{ p: 3 }}>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Team Member Profiles</Typography>
          <Button variant="contained" onClick={handleOpen}>
            Add Team Member
          </Button>
        </Box>

        <Grid container spacing={3}>
          {Object.values(profiles).map((profile) => (
            <Grid item xs={12} md={6} key={profile.name}>
              <Paper sx={{ p: 2, height: '100%' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Box>
                    <Typography variant="h6">{profile.name}</Typography>
                    <Typography color="textSecondary">{profile.role}</Typography>
                    {profile.last_updated && (
                      <Typography variant="caption" color="textSecondary">
                        Last updated: {new Date(profile.last_updated * 1000).toLocaleString()}
                      </Typography>
                    )}
                  </Box>
                  <Button 
                    size="small" 
                    color="error" 
                    onClick={() => handleRemoveProfile(profile.name)}
                  >
                    Remove
                  </Button>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>Skills</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {profile.skills.map((skill, index) => (
                      <Chip key={index} label={skill} size="small" />
                    ))}
                  </Box>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>Recent Activities</Typography>
                  <List dense>
                    {profile.activities && profile.activities.length > 0 ? (
                      profile.activities.slice(-3).map((activity, index) => (
                        <ListItem key={index}>
                          <ListItemText 
                            primary={activity.activity}
                            secondary={new Date(activity.timestamp * 1000).toLocaleString()}
                          />
                        </ListItem>
                      ))
                    ) : (
                      <ListItem>
                        <ListItemText primary="No recent activities" />
                      </ListItem>
                    )}
                  </List>
                </Box>

                <Box>
                  <Typography variant="subtitle2" gutterBottom>Recent Mentions</Typography>
                  <List dense>
                    {profile.mentions && profile.mentions.length > 0 ? (
                      profile.mentions.slice(-3).map((mention, index) => (
                        <ListItem key={index}>
                          <ListItemText 
                            primary={mention.context}
                            secondary={new Date(mention.timestamp * 1000).toLocaleString()}
                          />
                        </ListItem>
                      ))
                    ) : (
                      <ListItem>
                        <ListItemText primary="No recent mentions" />
                      </ListItem>
                    )}
                  </List>
                </Box>
              </Paper>
            </Grid>
          ))}
        </Grid>

        <Dialog open={open} onClose={handleClose}>
          <DialogTitle>Add Team Member</DialogTitle>
          <DialogContent>
            <TextField
              autoFocus
              margin="dense"
              label="Name"
              fullWidth
              value={newProfile.name}
              onChange={(e) => setNewProfile({ ...newProfile, name: e.target.value })}
            />
            <TextField
              margin="dense"
              label="Role"
              fullWidth
              value={newProfile.role}
              onChange={(e) => setNewProfile({ ...newProfile, role: e.target.value })}
            />
            <TextField
              margin="dense"
              label="Skills (comma-separated)"
              fullWidth
              value={newProfile.skills}
              onChange={(e) => setNewProfile({ ...newProfile, skills: e.target.value })}
              helperText="Enter skills separated by commas"
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={handleClose}>Cancel</Button>
            <Button onClick={handleAddProfile} variant="contained">Add</Button>
          </DialogActions>
        </Dialog>
      </Paper>
    </Box>
  );
};

export default PersonalizationDashboard; 