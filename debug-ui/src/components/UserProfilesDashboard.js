import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  List,
  ListItem,
  ListItemText,
  Chip,
  CircularProgress,
  LinearProgress,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Divider,
  Badge,
  Tooltip,
  Alert,
  AlertTitle,
} from '@mui/material';
import { 
  Add as AddIcon, 
  Delete as DeleteIcon, 
  Refresh as RefreshIcon, 
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Warning as WarningIcon,
  AccessTime as AccessTimeIcon,
  Work as WorkIcon,
  Block as BlockIcon,
  Assessment as AssessmentIcon,
  ClearAll as ClearAllIcon,
} from '@mui/icons-material';

const API_BASE_URL = 'http://localhost:8000';

const UserProfilesDashboard = () => {
  const [profiles, setProfiles] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [openBlockerDialog, setOpenBlockerDialog] = useState(false);
  const [openDecisionDialog, setOpenDecisionDialog] = useState(false);
  const [openAddUserDialog, setOpenAddUserDialog] = useState(false);
  const [openClearProfilesDialog, setOpenClearProfilesDialog] = useState(false);
  const [newBlocker, setNewBlocker] = useState('');
  const [newDecision, setNewDecision] = useState({ description: '', status: 'pending' });
  const [selectedUserId, setSelectedUserId] = useState(null);
  const [updating, setUpdating] = useState(false);
  const [newUser, setNewUser] = useState({ user_id: '', name: '', role: '', skills: '' });

  // Polling intervals for real-time updates (in ms)
  const POLL_INTERVAL = 5000;
  const UPDATE_FROM_GEMINI_INTERVAL = 5000; // Auto-update from Gemini every 30 seconds

  useEffect(() => {
    fetchProfiles();
    
    // Set up polling for real-time updates
    const fetchInterval = setInterval(fetchProfiles, POLL_INTERVAL);
    
    // Set up automatic updates from Gemini
    const updateInterval = setInterval(autoUpdateProfilesFromGemini, UPDATE_FROM_GEMINI_INTERVAL);
    
    // Clean up on unmount
    return () => {
      clearInterval(fetchInterval);
      clearInterval(updateInterval);
    };
  }, []);

  const fetchProfiles = async () => {
    try {
      setError(null);
      // Add refresh=true query parameter to force the server to refresh profiles from disk
      const response = await fetch(`${API_BASE_URL}/profiles?refresh=true`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch profiles: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      setProfiles(data);
    } catch (error) {
      console.error('Error fetching profiles:', error);
      setError('Failed to load user profiles. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const handleAddBlocker = async () => {
    if (!newBlocker.trim() || !selectedUserId) return;
    
    try {
      const response = await fetch(`${API_BASE_URL}/profiles/${selectedUserId}/blockers`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ blocker: newBlocker }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to add blocker: ${response.status} ${response.statusText}`);
      }
      
      // Close dialog and refresh profiles
      setOpenBlockerDialog(false);
      setNewBlocker('');
      fetchProfiles();
    } catch (error) {
      console.error('Error adding blocker:', error);
      setError('Failed to add blocker. Please try again.');
    }
  };

  const handleRemoveBlocker = async (userId, blockerIndex) => {
    try {
      const response = await fetch(`${API_BASE_URL}/profiles/${userId}/blockers/${blockerIndex}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error(`Failed to remove blocker: ${response.status} ${response.statusText}`);
      }
      
      fetchProfiles();
    } catch (error) {
      console.error('Error removing blocker:', error);
      setError('Failed to remove blocker. Please try again.');
    }
  };

  const handleAddDecision = async () => {
    if (!newDecision.description.trim() || !selectedUserId) return;
    
    try {
      const response = await fetch(`${API_BASE_URL}/profiles/${selectedUserId}/decisions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newDecision),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to add decision: ${response.status} ${response.statusText}`);
      }
      
      // Close dialog and refresh profiles
      setOpenDecisionDialog(false);
      setNewDecision({ description: '', status: 'pending' });
      fetchProfiles();
    } catch (error) {
      console.error('Error adding decision:', error);
      setError('Failed to add decision. Please try again.');
    }
  };

  const handleUpdateDecisionStatus = async (userId, decisionIndex, newStatus) => {
    try {
      const response = await fetch(`${API_BASE_URL}/profiles/${userId}/decisions/${decisionIndex}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status: newStatus }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to update decision status: ${response.status} ${response.statusText}`);
      }
      
      fetchProfiles();
    } catch (error) {
      console.error('Error updating decision status:', error);
      setError('Failed to update decision status. Please try again.');
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Never';
    
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };

  const getWorkloadColor = (workload) => {
    switch (workload) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'success';
      default:
        return 'default';
    }
  };

  const getDecisionStatusColor = (status) => {
    switch (status) {
      case 'made':
        return 'success';
      case 'pending':
        return 'warning';
      default:
        return 'default';
    }
  };

  const handleOpenBlockerDialog = (userId) => {
    setSelectedUserId(userId);
    setOpenBlockerDialog(true);
  };

  const handleOpenDecisionDialog = (userId) => {
    setSelectedUserId(userId);
    setOpenDecisionDialog(true);
  };

  const handleCloseBlockerDialog = () => {
    setOpenBlockerDialog(false);
    setNewBlocker('');
  };

  const handleCloseDecisionDialog = () => {
    setOpenDecisionDialog(false);
    setNewDecision({ description: '', status: 'pending' });
  };

  const updateProfilesFromGemini = async (showFeedback = true) => {
    try {
      setUpdating(true);
      if (showFeedback) {
        setError(null);
      }
      
      const response = await fetch(`${API_BASE_URL}/update-profiles-from-gemini`, {
        method: 'POST',
      });
      
      const result = await response.json();
      
      if (response.ok) {
        // Check if we got a warning (no existing profiles)
        if (result.status === 'warning' && showFeedback) {
          setError({ severity: 'warning', message: result.message });
        } else if (showFeedback) {
          if (result.users && result.users.length > 0) {
            setError({ 
              severity: 'success', 
              message: `Successfully updated ${result.users.length} profiles from current frame.` 
            });
          } else {
            setError({ 
              severity: 'info', 
              message: "No users were detected in the current frame." 
            });
          }
        }
      } else if (showFeedback) {
        setError({ 
          severity: 'error', 
          message: result.message || "Failed to update profiles from current frame." 
        });
      }
      
      // Refresh profiles after update
      await fetchProfiles();
    } catch (error) {
      console.error('Error updating profiles from Gemini:', error);
      if (showFeedback) {
        setError({ 
          severity: 'error', 
          message: "Error updating profiles from current frame. Please try again."
        });
      }
    } finally {
      setUpdating(false);
    }
  };

  // Auto-refresh function that doesn't show feedback
  const autoUpdateProfilesFromGemini = () => {
    updateProfilesFromGemini(false);
  };

  const handleAddUser = async () => {
    if (!newUser.user_id.trim() || !newUser.name.trim()) {
      setError({ severity: 'warning', message: 'User ID and Name are required fields.' });
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      // Prepare skills array from comma-separated string
      let skills = [];
      if (newUser.skills) {
        skills = newUser.skills.split(',').map(s => s.trim()).filter(Boolean);
      }
      
      // Create a new user profile
      const response = await fetch(`${API_BASE_URL}/profiles`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: newUser.user_id,
          name: newUser.name,
          role: newUser.role || '',
          skills: skills,
          workload: 'medium',
          last_seen: Date.now() / 1000, // Current timestamp in seconds
          last_updated: Date.now() / 1000,
          blockers: [],
          decisions: [],
          activities: []
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to create profile: ${response.status} ${response.statusText}`);
      }
      
      // Close dialog and reset form
      setOpenAddUserDialog(false);
      setNewUser({ user_id: '', name: '', role: '', skills: '' });
      
      // Fetch profiles again to get updated list
      await fetchProfiles();
    } catch (error) {
      console.error('Error creating user profile:', error);
      setError('Failed to create user profile. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  const handleOpenAddUserDialog = () => {
    setOpenAddUserDialog(true);
  };
  
  const handleCloseAddUserDialog = () => {
    setOpenAddUserDialog(false);
    setNewUser({ user_id: '', name: '', role: '', skills: '' });
  };

  const handleOpenClearProfilesDialog = () => {
    setOpenClearProfilesDialog(true);
  };

  const handleCloseClearProfilesDialog = () => {
    setOpenClearProfilesDialog(false);
  };

  const handleClearAllProfiles = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Delete all profiles using the POST endpoint
      const response = await fetch(`${API_BASE_URL}/profiles/clear-all`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`Failed to clear profiles: ${response.status} ${response.statusText}`);
      }
      
      // Close dialog
      setOpenClearProfilesDialog(false);
      
      // Display success message
      setError({ 
        severity: 'success', 
        message: "All user profiles have been cleared successfully." 
      });
      
      // Fetch profiles again to get empty list
      await fetchProfiles();
    } catch (error) {
      console.error('Error clearing profiles:', error);
      setError({ 
        severity: 'error', 
        message: "Failed to clear user profiles. Please try again."
      });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ p: 3, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <CircularProgress sx={{ mb: 2 }} />
        <Typography>Loading profiles...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h5">User Profiles Dashboard</Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              startIcon={<AddIcon />}
              variant="contained"
              color="secondary"
              onClick={handleOpenAddUserDialog}
              disabled={loading || updating}
            >
              Add User
            </Button>
            <Button
              color="primary"
              variant="contained"
              startIcon={<RefreshIcon />}
              onClick={() => updateProfilesFromGemini(true)}
              disabled={loading || updating}
            >
              Refresh
            </Button>
            <Button
              startIcon={<ClearAllIcon />}
              color="error"
              variant="outlined"
              onClick={handleOpenClearProfilesDialog}
              disabled={loading || updating || Object.keys(profiles).length === 0}
            >
              Clear All Profiles
            </Button>
          </Box>
        </Box>

        {error && (
          <Alert severity={typeof error === 'object' ? error.severity : 'error'} sx={{ mb: 3 }}>
            <AlertTitle>{typeof error === 'object' ? error.severity.charAt(0).toUpperCase() + error.severity.slice(1) : 'Error'}</AlertTitle>
            {typeof error === 'object' ? error.message : error}
          </Alert>
        )}

        {updating && (
          <Box sx={{ width: '100%', mb: 3 }}>
            <Alert severity="info" sx={{ mb: 1 }}>
              Updating profiles from current frame...
            </Alert>
            <LinearProgress />
          </Box>
        )}

        {Object.keys(profiles).length === 0 ? (
          <Box>
            <Alert severity="info" sx={{ mb: 2 }}>
              No user profiles available. Add user profiles manually before analyzing video frames.
            </Alert>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={handleOpenAddUserDialog}
              startIcon={<AddIcon />}
            >
              Add User Profile
            </Button>
          </Box>
        ) : (
          <Grid container spacing={3}>
            {Object.entries(profiles).map(([userId, profile]) => (
              <Grid item xs={12} md={6} key={userId}>
                <Card elevation={3}>
                  <CardHeader
                    title={
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        {profile.name}
                        <Chip 
                          label={profile.workload || 'unknown'} 
                          color={getWorkloadColor(profile.workload)}
                          size="small"
                          icon={<WorkIcon />}
                          sx={{ ml: 1 }}
                        />
                      </Box>
                    }
                    subheader={
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          {profile.role || 'No role specified'}
                        </Typography>
                      </Box>
                    }
                  />
                  <Divider />
                  <CardContent>
                    <Grid container spacing={2}>
                      {/* Blockers Section */}
                      <Grid item xs={12}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                          <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center' }}>
                            <BlockIcon color="error" sx={{ mr: 1 }} />
                            Blockers
                          </Typography>
                          <IconButton 
                            size="small" 
                            color="primary"
                            onClick={() => handleOpenBlockerDialog(userId)}
                          >
                            <AddIcon />
                          </IconButton>
                        </Box>
                        {profile.blockers && profile.blockers.length > 0 ? (
                          <List dense>
                            {profile.blockers.map((blocker, index) => (
                              <ListItem 
                                key={index}
                                secondaryAction={
                                  <IconButton 
                                    edge="end" 
                                    size="small"
                                    onClick={() => handleRemoveBlocker(userId, index)}
                                  >
                                    <DeleteIcon fontSize="small" />
                                  </IconButton>
                                }
                              >
                                <ListItemText primary={blocker} />
                              </ListItem>
                            ))}
                          </List>
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            No blockers reported
                          </Typography>
                        )}
                      </Grid>

                      {/* Decisions Section */}
                      <Grid item xs={12}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                          <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center' }}>
                            <AssessmentIcon color="primary" sx={{ mr: 1 }} />
                            Decisions
                          </Typography>
                          <IconButton 
                            size="small" 
                            color="primary"
                            onClick={() => handleOpenDecisionDialog(userId)}
                          >
                            <AddIcon />
                          </IconButton>
                        </Box>
                        {profile.decisions && profile.decisions.length > 0 ? (
                          <List dense>
                            {profile.decisions.map((decision, index) => (
                              <ListItem 
                                key={index}
                                secondaryAction={
                                  decision.status === 'pending' ? (
                                    <Tooltip title="Mark as made">
                                      <IconButton 
                                        edge="end" 
                                        size="small"
                                        color="success"
                                        onClick={() => handleUpdateDecisionStatus(userId, index, 'made')}
                                      >
                                        <CheckCircleIcon fontSize="small" />
                                      </IconButton>
                                    </Tooltip>
                                  ) : (
                                    <Tooltip title="Mark as pending">
                                      <IconButton 
                                        edge="end" 
                                        size="small"
                                        color="warning"
                                        onClick={() => handleUpdateDecisionStatus(userId, index, 'pending')}
                                      >
                                        <AccessTimeIcon fontSize="small" />
                                      </IconButton>
                                    </Tooltip>
                                  )
                                }
                              >
                                <ListItemText 
                                  primary={
                                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                      {decision.status === 'made' ? (
                                        <CheckCircleIcon fontSize="small" color="success" sx={{ mr: 1 }} />
                                      ) : (
                                        <AccessTimeIcon fontSize="small" color="warning" sx={{ mr: 1 }} />
                                      )}
                                      {decision.description}
                                    </Box>
                                  }
                                  secondary={formatTimestamp(decision.timestamp)}
                                />
                              </ListItem>
                            ))}
                          </List>
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            No decisions reported
                          </Typography>
                        )}
                      </Grid>

                      {/* Recent Activities Section */}
                      <Grid item xs={12}>
                        <Typography variant="subtitle1" sx={{ mb: 1 }}>Recent Activities</Typography>
                        {profile.activities && profile.activities.length > 0 ? (
                          <List dense>
                            {profile.activities.slice(-3).map((activity, index) => (
                              <ListItem key={index}>
                                <ListItemText 
                                  primary={activity.activity}
                                  secondary={formatTimestamp(activity.timestamp)}
                                />
                              </ListItem>
                            ))}
                          </List>
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            No recent activities
                          </Typography>
                        )}
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Paper>

      {/* Add Blocker Dialog */}
      <Dialog open={openBlockerDialog} onClose={handleCloseBlockerDialog}>
        <DialogTitle>Add Blocker</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            id="blocker"
            label="Blocker Description"
            type="text"
            fullWidth
            variant="outlined"
            value={newBlocker}
            onChange={(e) => setNewBlocker(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseBlockerDialog}>Cancel</Button>
          <Button onClick={handleAddBlocker} variant="contained" disabled={!newBlocker.trim()}>Add</Button>
        </DialogActions>
      </Dialog>

      {/* Add Decision Dialog */}
      <Dialog open={openDecisionDialog} onClose={handleCloseDecisionDialog}>
        <DialogTitle>Add Decision</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            id="decision"
            label="Decision Description"
            type="text"
            fullWidth
            variant="outlined"
            value={newDecision.description}
            onChange={(e) => setNewDecision({ ...newDecision, description: e.target.value })}
            sx={{ mb: 2 }}
          />
          <FormControl fullWidth>
            <InputLabel id="decision-status-label">Status</InputLabel>
            <Select
              labelId="decision-status-label"
              id="decision-status"
              value={newDecision.status}
              label="Status"
              onChange={(e) => setNewDecision({ ...newDecision, status: e.target.value })}
            >
              <MenuItem value="pending">Pending</MenuItem>
              <MenuItem value="made">Made</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDecisionDialog}>Cancel</Button>
          <Button 
            onClick={handleAddDecision} 
            variant="contained" 
            disabled={!newDecision.description.trim()}
          >
            Add
          </Button>
        </DialogActions>
      </Dialog>

      {/* Add new dialog for adding users */}
      <Dialog open={openAddUserDialog} onClose={handleCloseAddUserDialog} maxWidth="sm" fullWidth>
        <DialogTitle>Add New User Profile</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Create a new user profile. User ID and Name are required. Gemini will use fuzzy matching to
            recognize this user in analyzed frames.
          </Typography>
          <TextField
            autoFocus
            margin="dense"
            id="user_id"
            label="User ID (required)"
            type="text"
            fullWidth
            variant="outlined"
            value={newUser.user_id}
            onChange={(e) => setNewUser({ ...newUser, user_id: e.target.value })}
            sx={{ mb: 2 }}
            required
            helperText="A unique identifier for this user"
          />
          <TextField
            margin="dense"
            id="name"
            label="Name (required)"
            type="text"
            fullWidth
            variant="outlined"
            value={newUser.name}
            onChange={(e) => setNewUser({ ...newUser, name: e.target.value })}
            sx={{ mb: 2 }}
            required
            helperText="The user's full name, which will be used for matching in frames"
          />
          <TextField
            margin="dense"
            id="role"
            label="Role"
            type="text"
            fullWidth
            variant="outlined"
            value={newUser.role}
            onChange={(e) => setNewUser({ ...newUser, role: e.target.value })}
            sx={{ mb: 2 }}
            helperText="The user's role (e.g., Product Manager, Developer)"
          />
          <TextField
            margin="dense"
            id="skills"
            label="Skills (comma-separated)"
            type="text"
            fullWidth
            variant="outlined"
            value={newUser.skills}
            onChange={(e) => setNewUser({ ...newUser, skills: e.target.value })}
            helperText="Enter skills separated by commas"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseAddUserDialog}>Cancel</Button>
          <Button 
            onClick={handleAddUser} 
            variant="contained" 
            disabled={!newUser.user_id.trim() || !newUser.name.trim()}
          >
            Add User
          </Button>
        </DialogActions>
      </Dialog>

      {/* Clear All Profiles Confirmation Dialog */}
      <Dialog open={openClearProfilesDialog} onClose={handleCloseClearProfilesDialog}>
        <DialogTitle>Clear All Profiles</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete all user profiles? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseClearProfilesDialog}>Cancel</Button>
          <Button 
            onClick={handleClearAllProfiles} 
            variant="contained" 
            color="error"
          >
            Clear All
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default UserProfilesDashboard; 