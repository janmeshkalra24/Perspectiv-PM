from typing import Any, Dict, Optional
import asyncio
import logging
from pathlib import Path
from collections import deque
from asyncio import Queue
import time
import os
import json

from .api.base import BaseAPIClient
from .sources.base import DataSource
from .context import ContextManager

logger = logging.getLogger(__name__)

class ScreenUnderstanding:
    """Main class for screen understanding and VQA system."""

    def __init__(
        self,
        model: BaseAPIClient,
        source: DataSource,
        max_context_frames: int = 10,
        max_context_tokens: int = 4000,
        context_file: str = "frame_context.json",
        max_queue_size: int = 50,
        profiles_file: str = "user_profiles.json"
    ):
        """Initialize the screen understanding system.
        
        Args:
            model: API client for visual understanding
            source: Data source for video frames
            max_context_frames: Maximum number of frames to keep in context
            max_context_tokens: Maximum number of tokens in context
            context_file: File to save/load context from
            max_queue_size: Maximum number of frames to queue for processing
            profiles_file: File to save/load user profiles from
        """
        self.model = model
        self.source = source
        self.context_manager = ContextManager(
            max_context_frames=max_context_frames,
            max_context_tokens=max_context_tokens,
            rate_limit_rpm=getattr(model, 'rate_limit_rpm', 30),
            context_file=context_file
        )
        self.profile_manager = ProfileManager(profiles_file=profiles_file)
        self._is_running = False
        self._current_frame = None
        self._frame_queue = Queue(maxsize=max_queue_size)
        self._processing_task = None
        self._processed_frames = set()  # Track processed frame indices

    async def _process_frames(self):
        """Process frames from the queue."""
        while self._is_running:
            try:
                frame_data = await self._frame_queue.get()
                frame_index = frame_data.get("frame_index")
                frame_key = frame_data.get("key", f"test:{frame_index}")  # Get key or construct it
                
                # Skip if already processed
                if frame_index in self._processed_frames:
                    logger.info(f"Skipping already processed frame {frame_index}")
                    self._frame_queue.task_done()
                    continue
                
                # Check rate limiting
                if not await self.context_manager.can_process():
                    logger.info("Rate limiting applied, waiting before processing next frame")
                    await asyncio.sleep(self.context_manager.request_delay)
                    # Put frame back in queue
                    await self._frame_queue.put(frame_data)
                    self._frame_queue.task_done()
                    continue
                
                logger.info(f"Processing frame {frame_index} with model...")
                
                # Process frame
                try:
                    # First get general description
                    result = await self.model.process_image(frame_data["image_data"])
                    
                    # Then extract PM-specific insights with structured prompts
                    pm_insights = await self.model.answer_question(
                        """Analyze this frame from a product management perspective. Provide ONLY the most critical insights in a concise format.

STRICT RULES:
1. Each item MUST be 100 characters or less
2. Each category MUST have at most 3 items
3. Use bullet points only for actual items
4. Skip any category that has no relevant items
5. NO explanatory text or filler words

Return ONLY this JSON format:
{
    "sprint_goals": [
        "Implement user auth by EOW",
        "Complete API docs"
    ],
    "key_metrics": [
        "API response time < 200ms",
        "Test coverage > 85%"
    ],
    "feature_status": [
        "Auth: 80% done, pending security review",
        "API docs: 20% complete"
    ],
    "dependencies": [
        "Auth service needs updated identity provider",
        "Mobile app blocked on API"
    ],
    "risks": [
        "Security review may delay auth release",
        "Limited backend capacity"
    ],
    "next_steps": [
        "Schedule security review",
        "Start API documentation"
    ],
    "decisions": [
        "Using OAuth2 for auth flow",
        "Postponing analytics to next sprint"
    ],
    "stakeholder_requests": [
        "Marketing needs user flows by Friday",
        "Support team requests better error messages"
    ]
}""",
                        {"image_data": frame_data["image_data"]}
                    )
                    
                    # Add PM insights to result
                    try:
                        result["pm_insights"] = pm_insights
                    except:
                        logger.warning(f"Could not parse PM insights for frame {frame_index}")
                        result["pm_insights"] = {}
                    
                    # Get existing profiles for fuzzy matching
                    existing_profiles = {}
                    profile_names = []
                    for user_id, profile in self.profile_manager.profiles.items():
                        profile_names.append({
                            "user_id": user_id,
                            "name": profile.name,
                            "role": profile.role
                        })
                    
                    # If no profiles exist yet, skip the user profile analysis
                    if not profile_names:
                        logger.warning("No user profiles exist for matching. Please create profiles manually first.")
                        result["user_profiles"] = {"users": []}
                    else:
                        # Analyze user profiles in the frame with fuzzy matching
                        user_profiles = await self.model.answer_question(
                            f"""Analyze this frame to detect and track ONLY the users from the provided list:

USER LIST FOR MATCHING:
{json.dumps(profile_names, indent=2)}

INSTRUCTIONS:
1. ONLY identify users from the above list.
2. Use fuzzy matching to identify users by name or role if exact matches aren't found.
3. For each identified user, assess:
   - Current workload (high/medium/low) if apparent
   - Any blockers they are facing
   - Any decisions being made or discussed
   - Activities they're engaged in
   - Skills mentioned
4. Keep all items brief and specific (max 100 characters each)

Return ONLY this JSON format:
{{
    "users": [
        {{
            "user_id": "string", // MUST be one from the provided list
            "workload": "string", // high, medium, or low
            "blockers": ["string"],
            "decisions": [
                {{
                    "description": "string",
                    "status": "string" // "pending" or "made"
                }}
            ],
            "activities": ["string"],
            "skills": ["string"]
        }}
    ]
}}

IMPORTANT:
- ONLY include users from the provided list that appear in the image
- Match each user based on their name or role using fuzzy matching
- If none of the users in the list appear in the image, return an empty users array""",
                            {"image_data": frame_data["image_data"]}
                        )
                    
                        # Update user profiles based on the analysis
                        try:
                            # Add debug logging
                            logger.info(f"User profiles response from Gemini with fuzzy matching: {json.dumps(user_profiles, indent=2)}")
                            
                            # Parse the user profiles data
                            if "users" in user_profiles and isinstance(user_profiles["users"], list):
                                logger.info(f"Found {len(user_profiles['users'])} users in the frame")
                                for user_data in user_profiles["users"]:
                                    user_id = user_data.get("user_id")
                                    if not user_id:
                                        logger.warning("User data missing user_id, skipping")
                                        continue
                                    
                                    # Verify this user exists in our profiles
                                    if user_id not in self.profile_manager.profiles:
                                        logger.warning(f"User {user_id} not found in existing profiles, skipping")
                                        continue
                                        
                                    # Get the profile
                                    profile = self.profile_manager.get_profile(user_id)
                                    logger.info(f"Processing user profile for user_id: {user_id}")
                                    
                                    # Update workload
                                    if "workload" in user_data and user_data["workload"]:
                                        profile.update_workload(user_data["workload"])
                                        
                                    # Update blockers
                                    if "blockers" in user_data and isinstance(user_data["blockers"], list):
                                        for blocker in user_data["blockers"]:
                                            profile.add_blocker(blocker)
                                            
                                    # Update decisions
                                    if "decisions" in user_data and isinstance(user_data["decisions"], list):
                                        for decision in user_data["decisions"]:
                                            description = decision.get("description", "")
                                            status = decision.get("status", "pending")
                                            if description:
                                                profile.add_decision(description, status)
                                                
                                    # Update activities
                                    if "activities" in user_data and isinstance(user_data["activities"], list):
                                        for activity in user_data["activities"]:
                                            profile.add_activity(activity)
                                    
                                    # Update skills
                                    if "skills" in user_data and isinstance(user_data["skills"], list):
                                        for skill in user_data["skills"]:
                                            if skill not in profile.skills:
                                                profile.skills.append(skill)
                                                
                                    # Update last seen timestamp
                                    timestamp = frame_data.get("metadata", {}).get("timestamp")
                                    profile.update_last_seen(timestamp)
                                    
                                    # Update the profile in the manager
                                    self.profile_manager.update_profile(profile)
                                    logger.info(f"Updated profile for user {user_id}")
                            else:
                                logger.warning("No users field in the Gemini response or it's not a list")
                                    
                            # Add user profiles to result
                            result["user_profiles"] = user_profiles
                        except Exception as e:
                            logger.warning(f"Error processing user profiles for frame {frame_index}: {e}")
                            result["user_profiles"] = {"users": []}
                    
                    logger.info(f"Model returned result for frame {frame_index}: {result.get('description', '')[:100]}...")
                    
                    # Add metadata to result
                    result.update({
                        "frame_index": frame_index,
                        "frame_key": frame_key,  # Add frame key for UI
                        "metadata": frame_data["metadata"]
                    })
                    
                    # Add to context
                    self.context_manager.add_frame_context(result)
                    logger.info(f"Added frame {frame_index} to context")
                    
                    # Mark as processed
                    self._processed_frames.add(frame_index)
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_index}: {e}", exc_info=True)
                
                finally:
                    self._frame_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in frame processing loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight error loop

    async def start(self) -> None:
        """Start processing frames from the source."""
        if self._is_running:
            return

        self._is_running = True
        self._processed_frames.clear()  # Clear processed frames set
        logger.info("Starting frame processing...")
        
        # Start processing task
        self._processing_task = asyncio.create_task(self._process_frames())
        
        try:
            while self._is_running:
                frame_data = await self.source.get_frame()
                if frame_data is None:
                    await asyncio.sleep(0.1)
                    continue

                frame_index = frame_data.get("frame_index", -1)
                logger.info(f"Received frame {frame_index} from source")
                
                # Store current frame
                self._current_frame = frame_data
                
                # Skip if already queued
                if frame_index in self._processed_frames:
                    logger.info(f"Skipping already processed frame {frame_index}")
                    continue
                
                # Add to processing queue
                try:
                    # Try to add to queue with a timeout
                    await asyncio.wait_for(
                        self._frame_queue.put(frame_data),
                        timeout=0.1
                    )
                    logger.info(f"Queued frame {frame_index} for processing")
                except asyncio.TimeoutError:
                    logger.warning(f"Queue full, skipping frame {frame_index}")
                
        except Exception as e:
            logger.error(f"Error in frame processing: {e}", exc_info=True)
            raise
        finally:
            self._is_running = False
            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass

    async def ask(self, question: str) -> str:
        """Ask a question about the current frame.
        
        Args:
            question: Question to answer
            
        Returns:
            Answer from the model
        """
        if not self._current_frame:
            raise ValueError("No frame available")
            
        # Get temporal context
        context_summary = self.context_manager.get_temporal_context()
        
        # Enhance question with context
        enhanced_question = f"""Question: {question}

Previous context from recent frames:
{context_summary}

Please answer the question based on the current frame, using the context from previous frames if relevant."""
        
        # Get answer
        return await self.model.answer_question(
            enhanced_question,
            {"image_data": self._current_frame["image_data"]}
        )

    async def stop(self) -> None:
        """Stop the system."""
        self._is_running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass 

class UserProfile:
    """Class for storing and managing user profile data in the screen understanding system."""

    def __init__(self, user_id: str, name: str = None, role: str = None):
        """Initialize a user profile.
        
        Args:
            user_id: Unique identifier for the user
            name: User's name (optional)
            role: User's role (optional)
        """
        self.user_id = user_id
        self.name = name or user_id
        self.role = role or ""
        self.last_seen = None
        self.last_updated = time.time()
        self.workload = "medium"  # Can be "high", "medium", or "low"
        self.blockers = []  # List of current blockers
        self.decisions = []  # List of decisions (made or pending)
        self.activities = []  # User's recent activities
        self.mentions = []  # Mentions of the user in contexts
        self.skills = []
        
    def update_workload(self, new_workload: str) -> None:
        """Update the user's workload level.
        
        Args:
            new_workload: New workload level (high/medium/low)
        """
        if new_workload.lower() in ["high", "medium", "low"]:
            self.workload = new_workload.lower()
            self._record_update()
            
    def add_blocker(self, blocker: str) -> None:
        """Add a new blocker for the user.
        
        Args:
            blocker: Description of the blocker
        """
        if blocker and blocker not in self.blockers:
            self.blockers.append(blocker)
            self._record_update()
            
    def remove_blocker(self, blocker: str) -> None:
        """Remove a blocker from the user's list.
        
        Args:
            blocker: Blocker to remove
        """
        if blocker in self.blockers:
            self.blockers.remove(blocker)
            self._record_update()
            
    def add_decision(self, decision: str, status: str = "pending") -> None:
        """Add a decision to the user's list.
        
        Args:
            decision: Description of the decision
            status: Status of the decision ("pending" or "made")
        """
        if decision:
            decision_entry = {
                "description": decision,
                "status": status,
                "timestamp": time.time()
            }
            self.decisions.append(decision_entry)
            self._record_update()
            
    def update_decision_status(self, decision_index: int, new_status: str) -> None:
        """Update the status of a decision.
        
        Args:
            decision_index: Index of the decision in the list
            new_status: New status of the decision
        """
        if 0 <= decision_index < len(self.decisions):
            self.decisions[decision_index]["status"] = new_status
            self._record_update()
            
    def add_activity(self, activity: str) -> None:
        """Add a new activity for the user.
        
        Args:
            activity: Description of the activity
        """
        if activity:
            activity_entry = {
                "activity": activity,
                "timestamp": time.time()
            }
            self.activities.append(activity_entry)
            self._record_update()
            
    def add_mention(self, context: str) -> None:
        """Add a mention of the user.
        
        Args:
            context: Context where the user was mentioned
        """
        if context:
            mention_entry = {
                "context": context,
                "timestamp": time.time()
            }
            self.mentions.append(mention_entry)
            self._record_update()
            
    def update_last_seen(self, timestamp=None) -> None:
        """Update when the user was last seen.
        
        Args:
            timestamp: Timestamp when the user was seen (defaults to now)
        """
        self.last_seen = timestamp or time.time()
        self._record_update()
        
    def _record_update(self) -> None:
        """Record that the profile was updated."""
        self.last_updated = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile to a dictionary.
        
        Returns:
            Dictionary representation of the profile
        """
        return {
            "user_id": self.user_id,
            "name": self.name,
            "role": self.role,
            "last_seen": self.last_seen,
            "last_updated": self.last_updated,
            "workload": self.workload,
            "blockers": self.blockers.copy(),
            "decisions": self.decisions.copy(),
            "activities": self.activities.copy(),
            "mentions": self.mentions.copy(),
            "skills": self.skills.copy()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create a profile from a dictionary.
        
        Args:
            data: Dictionary with profile data
            
        Returns:
            UserProfile instance
        """
        profile = cls(
            user_id=data.get("user_id", ""),
            name=data.get("name", ""),
            role=data.get("role", "")
        )
        profile.last_seen = data.get("last_seen")
        profile.last_updated = data.get("last_updated", time.time())
        profile.workload = data.get("workload", "medium")
        profile.blockers = data.get("blockers", [])
        profile.decisions = data.get("decisions", [])
        profile.activities = data.get("activities", [])
        profile.mentions = data.get("mentions", [])
        profile.skills = data.get("skills", [])
        return profile


class ProfileManager:
    """Class for managing multiple user profiles."""
    
    def __init__(self, profiles_file: str = "user_profiles.json"):
        """Initialize the profile manager.
        
        Args:
            profiles_file: File to save/load profiles from
        """
        self.profiles_file = profiles_file
        self.profiles = {}
        self._load_profiles()
        
    def get_profile(self, user_id: str) -> UserProfile:
        """Get a user profile by ID, creating it if it doesn't exist.
        
        Args:
            user_id: User ID to get profile for
            
        Returns:
            UserProfile instance
        """
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id)
        return self.profiles[user_id]
        
    def update_profile(self, profile: UserProfile) -> None:
        """Update a profile in the manager.
        
        Args:
            profile: Profile to update
        """
        logger.info(f"Updating profile for user_id: {profile.user_id}, name: {profile.name}")
        self.profiles[profile.user_id] = profile
        # Save profiles to disk immediately
        try:
            self._save_profiles()
            logger.info(f"Successfully saved updated profile for {profile.user_id} to {self.profiles_file}")
        except Exception as e:
            logger.error(f"Failed to save profile for {profile.user_id}: {e}", exc_info=True)
        
    def delete_profile(self, user_id: str) -> None:
        """Delete a profile from the manager.
        
        Args:
            user_id: ID of profile to delete
        """
        if user_id in self.profiles:
            del self.profiles[user_id]
            self._save_profiles()
            
    def create_test_users(self) -> None:
        """Create test user profiles if none exist.
        
        This is kept for backward compatibility but no longer creates test users.
        Manual user entry is now used instead.
        """
        logger.info("create_test_users called, but test users are disabled. Use manual user entry instead.")
        pass
            
    def analyze_user_presence(self, frame_data: Dict[str, Any]) -> None:
        """Analyze a frame to detect user presence and update profiles.
        
        Args:
            frame_data: Frame data including image and metadata
        """
        # This would be implemented to use model analysis
        pass
            
    def _load_profiles(self) -> None:
        """Load profiles from the profiles file."""
        try:
            if os.path.exists(self.profiles_file):
                with open(self.profiles_file, "r") as f:
                    data = json.load(f)
                    
                for user_id, profile_data in data.items():
                    self.profiles[user_id] = UserProfile.from_dict(profile_data)
                    
                logger.info(f"Loaded {len(self.profiles)} profiles from {self.profiles_file}")
                    
        except Exception as e:
            logger.error(f"Error loading profiles: {e}", exc_info=True)
            self.profiles = {}
            
    def _save_profiles(self) -> None:
        """Save profiles to the profiles file."""
        try:
            profiles_dict = {
                user_id: profile.to_dict() 
                for user_id, profile in self.profiles.items()
            }
                
            with open(self.profiles_file, "w") as f:
                json.dump(profiles_dict, f, indent=2)
                
            logger.info(f"Saved {len(self.profiles)} profiles to {self.profiles_file}")
                
        except Exception as e:
            logger.error(f"Error saving profiles: {e}", exc_info=True) 