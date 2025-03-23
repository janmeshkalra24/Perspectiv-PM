#!/usr/bin/env python3
"""
Basic usage example for the Perspectiv Knowledge Base.

This script demonstrates how to initialize the knowledge base, add documents,
query the knowledge base, and manage sessions.
"""

import os
import sys
import logging
from datetime import datetime

# Add the parent directory to the path so we can import the perspectiv package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perspectiv import KnowledgeBase

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Initialize the knowledge base
    print("Initializing knowledge base...")
    kb = KnowledgeBase(data_dir="./data")
    
    # Create a sample transcript
    print("Creating sample transcript...")
    transcript_path = "examples/sample_transcript.txt"
    with open(transcript_path, "w") as f:
        f.write("""
Engineering Standup - 2023-06-01

John: Let's start with an update on the authentication system. We're having issues with the OAuth flow.

Mary: I've been looking into that. The problem seems to be with the token refresh mechanism. 
The client is not properly handling expired tokens.

John: How long would it take to fix that?

Mary: I think we can have a fix by the end of the week. It's not too complex, but we need to test it thoroughly.

David: What about the database migration? Are we still on track for that?

John: Yes, we're planning to run the migration this weekend. We'll need to have a 2-hour downtime window.

David: Has the notification system been updated to inform users about the downtime?

Mary: Not yet, I'll work on that today.

John: Great, any other blockers or dependencies we should be aware of?

David: The frontend team is waiting for the new API endpoints for user preferences.

John: I'll follow up with them after this meeting.
""")
    
    # Create a session
    session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Creating session: {session_id}")
    
    # Update the knowledge base with the transcript
    print("Adding transcript to the knowledge base...")
    kb.update_from_transcript(transcript_path, session_id=session_id)
    
    # Initialize the knowledge base (builds indexes)
    print("Building indexes...")
    kb.initialize()
    
    # Query the knowledge base
    print("\nQuerying the knowledge base...\n")
    queries = [
        "What issues are we having with authentication?",
        "When is the database migration happening?",
        "What dependencies exist between teams?",
        "What is the timeline for fixing the OAuth issue?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = kb.query(query, session_id=session_id)
        
        # Print domain scores
        print("\nDomain Relevance:")
        for domain, score in results["domain_scores"]:
            print(f"  {domain}: {score:.4f}")
        
        # Print top results
        print("\nTop Results:")
        for domain, domain_results in results["results"].items():
            print(f"\n  From {domain}:")
            for result in domain_results[:2]:  # Show top 2 results per domain
                print(f"    Score: {result['score']:.4f}")
                print(f"    Content: {result['content'][:150]}...")
    
    # Get session history
    print("\nGetting session history...")
    history = kb.get_session_history(session_id)
    print(f"Session has {len(history['queries'])} queries and {len(history['transcripts'])} transcripts")
    
    # Clean up
    print("\nCleaning up...")
    kb.close()
    
    print("\nDone!")

if __name__ == "__main__":
    main() 