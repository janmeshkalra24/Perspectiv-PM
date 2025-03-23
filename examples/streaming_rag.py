#!/usr/bin/env python3
"""
Streaming RAG example for the Perspectiv Knowledge Base.

This script demonstrates how to use the knowledge base for low-latency streaming RAG.
It simulates a live meeting transcript and performs incremental queries.
"""

import os
import sys
import time
import logging
from datetime import datetime
import threading
from typing import List

# Add the parent directory to the path so we can import the perspectiv package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perspectiv import KnowledgeBase

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sample transcript segments to simulate streaming
TRANSCRIPT_SEGMENTS = [
    """
Engineering Standup - 2023-06-01 (Starting)

John: Let's start with an update on the authentication system. We're having issues with the OAuth flow.
    """,
    
    """
Mary: I've been looking into that. The problem seems to be with the token refresh mechanism. 
The client is not properly handling expired tokens.

John: How long would it take to fix that?
    """,
    
    """
Mary: I think we can have a fix by the end of the week. It's not too complex, but we need to test it thoroughly.

David: What about the database migration? Are we still on track for that?
    """,
    
    """
John: Yes, we're planning to run the migration this weekend. We'll need to have a 2-hour downtime window.

David: Has the notification system been updated to inform users about the downtime?
    """,
    
    """
Mary: Not yet, I'll work on that today.

John: Great, any other blockers or dependencies we should be aware of?
    """,
    
    """
David: The frontend team is waiting for the new API endpoints for user preferences.

John: I'll follow up with them after this meeting.
    """
]

class StreamingTranscriptSimulator:
    """Simulates a streaming transcript from a meeting."""
    
    def __init__(self, segments: List[str], delay: float = 3.0):
        """
        Initialize the simulator.
        
        Args:
            segments: List of transcript segments
            delay: Delay in seconds between segments
        """
        self.segments = segments
        self.delay = delay
        self.current_transcript = ""
        self.is_running = False
        self.thread = None
        self.callbacks = []
    
    def start(self):
        """Start the simulation."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the simulation."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def add_callback(self, callback):
        """Add a callback to be called when new content is available."""
        self.callbacks.append(callback)
    
    def get_current_transcript(self):
        """Get the current transcript."""
        return self.current_transcript
    
    def _run(self):
        """Run the simulation."""
        for segment in self.segments:
            if not self.is_running:
                break
            
            self.current_transcript += segment
            for callback in self.callbacks:
                callback(segment, self.current_transcript)
            
            time.sleep(self.delay)


class StreamingRAG:
    """Demonstrates streaming RAG with the knowledge base."""
    
    def __init__(self, kb: KnowledgeBase, session_id: str = None):
        """
        Initialize the streaming RAG demo.
        
        Args:
            kb: Knowledge base instance
            session_id: Optional session ID
        """
        self.kb = kb
        self.session_id = session_id or f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.transcript_file = f"examples/streaming_transcript_{self.session_id}.txt"
        
        # Create an empty transcript file
        with open(self.transcript_file, "w") as f:
            f.write("")
    
    def on_new_transcript(self, segment: str, full_transcript: str):
        """
        Handle new transcript segment.
        
        Args:
            segment: New transcript segment
            full_transcript: Full transcript so far
        """
        # Update the transcript file
        with open(self.transcript_file, "w") as f:
            f.write(full_transcript)
        
        # Update the knowledge base
        print(f"\n--- New transcript segment received ---\n{segment}\n")
        
        # In a real-world scenario, you might want to do this less frequently
        # to avoid excessive updates, but for the demo we update on each segment
        status = self.kb.update_from_transcript(self.transcript_file, session_id=self.session_id)
        
        # Generate a relevant query based on the latest segment
        # In a real application, this could be more sophisticated
        query = self._generate_relevant_query(segment)
        
        if query:
            print(f"\n--- Auto-generated query: {query} ---")
            results = self.kb.query(query, session_id=self.session_id)
            
            # Print top result from each domain
            print("\nRelevant information to help understand the discussion:")
            for domain, domain_results in results["results"].items():
                if domain_results:
                    result = domain_results[0]
                    print(f"\n• Context from {domain}:")
                    print(f"  {result['content'][:200]}...")
    
    def _generate_relevant_query(self, segment: str) -> str:
        """
        Generate a relevant query based on the transcript segment.
        
        In a real application, this might involve NLP/LLM processing to extract
        relevant questions, but for the demo we'll use simple rules.
        
        Args:
            segment: Transcript segment

        Returns:
            Generated query
        """
        if "OAuth" in segment:
            return "What are common issues with OAuth authentication flows?"
        elif "token refresh" in segment:
            return "How to fix token refresh mechanisms in authentication?"
        elif "database migration" in segment:
            return "What's involved in database migration planning?"
        elif "downtime" in segment:
            return "How to minimize impact of downtime during migrations?"
        elif "notification system" in segment:
            return "Best practices for user notifications about system downtime"
        elif "dependencies" in segment or "blockers" in segment:
            return "How to track dependencies between development teams?"
        elif "frontend team" in segment:
            return "Coordination between frontend and backend teams for API changes"
        else:
            # If no specific trigger is found, return None to skip querying
            return None


def main():
    # Initialize the knowledge base
    print("Initializing knowledge base...")
    kb = KnowledgeBase(data_dir="./data")
    
    # Initialize empty indexes (they'll be updated as content arrives)
    kb.initialize()
    
    # Set up the streaming RAG demo
    streaming_rag = StreamingRAG(kb)
    
    # Set up the transcript simulator
    simulator = StreamingTranscriptSimulator(TRANSCRIPT_SEGMENTS, delay=5.0)
    simulator.add_callback(streaming_rag.on_new_transcript)
    
    # Start the simulation
    print("\nStarting simulated transcript stream. Press Ctrl+C to stop.\n")
    try:
        simulator.start()
        
        # Wait for the simulation to finish
        while simulator.is_running and simulator.thread.is_alive():
            time.sleep(1.0)
        
        print("\nSimulation completed!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down...")
    finally:
        simulator.stop()
        kb.close()


if __name__ == "__main__":
    main() 