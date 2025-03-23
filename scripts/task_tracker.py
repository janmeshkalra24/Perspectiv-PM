#!/usr/bin/env python3
"""
Task and Dependency Tracker

This script tracks tasks, blockers, and dependencies from:
1. JIRA tickets
2. GitHub issues
3. Meeting transcripts
4. Team chat messages
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime, timedelta
import re

import requests
from bs4 import BeautifulSoup
from jira import JIRA
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskTracker:
    def __init__(
        self,
        output_dir: str,
        jira_url: str = None,
        jira_token: str = None,
        github_token: str = None,
        slack_token: str = None
    ):
        """
        Initialize the task tracker.
        
        Args:
            output_dir: Directory to save tracked content
            jira_url: JIRA instance URL
            jira_token: JIRA API token
            github_token: GitHub API token
            slack_token: Slack API token
        """
        self.output_dir = os.path.abspath(output_dir)
        self.jira_url = jira_url
        self.jira_token = jira_token
        self.github_token = github_token
        self.slack_token = slack_token
        
        # Initialize API clients
        self.jira = JIRA(server=jira_url, token_auth=jira_token) if jira_url and jira_token else None
        self.slack = WebClient(token=slack_token) if slack_token else None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def track_jira_tasks(
        self,
        project_keys: List[str],
        days_back: int = 30,
        status_filter: Optional[List[str]] = None
    ) -> List[str]:
        """
        Track tasks from JIRA projects.
        
        Args:
            project_keys: List of JIRA project keys
            days_back: Number of days to look back
            status_filter: Optional list of status values to filter by
        
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        if not self.jira:
            logger.warning("JIRA credentials not provided, skipping JIRA tracking")
            return saved_files
        
        try:
            for project_key in project_keys:
                # Build JQL query
                jql = f"project = {project_key}"
                if days_back:
                    date_filter = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                    jql += f" AND updated >= '{date_filter}'"
                if status_filter:
                    jql += f" AND status IN ({','.join(status_filter)})"
                
                # Search for issues
                issues = self.jira.search_issues(
                    jql,
                    maxResults=100,
                    fields="summary,description,status,priority,assignee,labels,issuelinks"
                )
                
                if issues:
                    file_path = os.path.join(self.output_dir, f"jira_{project_key}_tasks.md")
                    with open(file_path, 'w') as f:
                        f.write(f"# JIRA Tasks from {project_key}\n\n")
                        f.write(f"Generated on: {datetime.now()}\n\n")
                        
                        for issue in issues:
                            f.write(f"## {issue.key}: {issue.fields.summary}\n\n")
                            f.write(f"Status: {issue.fields.status}\n")
                            f.write(f"Priority: {issue.fields.priority}\n")
                            if issue.fields.assignee:
                                f.write(f"Assignee: {issue.fields.assignee.displayName}\n")
                            if issue.fields.labels:
                                f.write(f"Labels: {', '.join(issue.fields.labels)}\n")
                            
                            # Track dependencies
                            if issue.fields.issuelinks:
                                f.write("\nDependencies:\n")
                                for link in issue.fields.issuelinks:
                                    if hasattr(link, "outwardIssue"):
                                        f.write(f"- Depends on: {link.outwardIssue.key}\n")
                                    elif hasattr(link, "inwardIssue"):
                                        f.write(f"- Blocked by: {link.inwardIssue.key}\n")
                            
                            f.write(f"\nDescription:\n{issue.fields.description or 'No description'}\n\n")
                            f.write("---\n\n")
                    
                    saved_files.append(file_path)
            
        except Exception as e:
            logger.error(f"Error tracking JIRA tasks: {e}")
        
        return saved_files
    
    def track_github_issues(
        self,
        repos: List[str],
        days_back: int = 30,
        label_filter: Optional[List[str]] = None
    ) -> List[str]:
        """
        Track issues from GitHub repositories.
        
        Args:
            repos: List of repository names (format: owner/repo)
            days_back: Number of days to look back
            label_filter: Optional list of labels to filter by
        
        Returns:
            List of saved file paths
        """
        saved_files = []
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        
        for repo in repos:
            try:
                # Get issues
                since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
                issues_url = f"https://api.github.com/repos/{repo}/issues"
                params = {'since': since_date, 'state': 'all'}
                response = requests.get(issues_url, headers=headers, params=params)
                
                if response.status_code == 200:
                    issues = response.json()
                    
                    # Filter by labels if specified
                    if label_filter:
                        issues = [
                            issue for issue in issues
                            if any(label['name'] in label_filter for label in issue['labels'])
                        ]
                    
                    if issues:
                        file_path = os.path.join(self.output_dir, f"github_{repo.replace('/', '_')}_issues.md")
                        with open(file_path, 'w') as f:
                            f.write(f"# GitHub Issues from {repo}\n\n")
                            f.write(f"Generated on: {datetime.now()}\n\n")
                            
                            for issue in issues:
                                f.write(f"## {issue['title']}\n\n")
                                f.write(f"Status: {issue['state']}\n")
                                f.write(f"Created: {issue['created_at']}\n")
                                f.write(f"Updated: {issue['updated_at']}\n")
                                if issue['assignee']:
                                    f.write(f"Assignee: {issue['assignee']['login']}\n")
                                if issue['labels']:
                                    f.write(f"Labels: {', '.join(label['name'] for label in issue['labels'])}\n")
                                
                                f.write(f"\n{issue['body'] or 'No description'}\n\n")
                                f.write("---\n\n")
                        
                        saved_files.append(file_path)
            
            except Exception as e:
                logger.error(f"Error tracking GitHub issues for {repo}: {e}")
        
        return saved_files
    
    def track_slack_threads(self, channels: List[str]) -> List[str]:
        """
        Track discussions and decisions from Slack threads.
        
        Args:
            channels: List of channel IDs to track
            
        Returns:
            List of paths to tracked files
        """
        if not self.slack_token:
            logger.warning("Skipping Slack tracking - no token provided")
            return []
            
        tracked_files = []
        headers = {"Authorization": f"Bearer {self.slack_token}"}
        
        for channel in channels:
            try:
                # Get channel history
                url = "https://slack.com/api/conversations.history"
                params = {
                    "channel": channel,
                    "limit": 100,
                    "include_all_metadata": True
                }
                
                response = requests.get(url, headers=headers, params=params)
                if not response.ok:
                    error_data = response.json()
                    if error_data.get('error') == 'missing_scope':
                        needed_scopes = error_data.get('needed', '').split(',')
                        logger.error(f"Missing required Slack scopes: {', '.join(needed_scopes)}")
                        logger.error("Please update your Slack app permissions to include these scopes")
                        return []
                    else:
                        logger.error(f"Error accessing Slack channel {channel}: {response.text}")
                        continue
                
                messages = response.json().get('messages', [])
                
                # Process threads
                for msg in messages:
                    if msg.get('thread_ts'):
                        try:
                            # Get thread replies
                            thread_url = "https://slack.com/api/conversations.replies"
                            thread_params = {
                                "channel": channel,
                                "ts": msg['thread_ts'],
                                "limit": 100
                            }
                            
                            thread_response = requests.get(thread_url, headers=headers, params=thread_params)
                            if not thread_response.ok:
                                continue
                                
                            replies = thread_response.json().get('messages', [])
                            
                            # Save thread content
                            thread_content = [f"# Slack Thread from {channel}\n\n"]
                            thread_content.append(f"## Original Message\n{msg.get('text', '')}\n\n")
                            thread_content.append("## Replies\n")
                            
                            for reply in replies[1:]:  # Skip first message as it's the original
                                thread_content.append(f"- {reply.get('text', '')}\n")
                                
                            file_path = os.path.join(self.output_dir, f"slack_{channel}_{msg['thread_ts']}.md")
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(thread_content))
                                
                            tracked_files.append(file_path)
                            
                        except Exception as e:
                            logger.error(f"Error processing thread in channel {channel}: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Error tracking Slack channel {channel}: {e}")
                continue
                
        return tracked_files
    
    def analyze_dependencies(self) -> str:
        """
        Analyze dependencies across all tracked tasks.
        
        Returns:
            Path to the dependency analysis file
        """
        try:
            # Collect all task files
            task_files = []
            for file in os.listdir(self.output_dir):
                if file.endswith('.md'):
                    task_files.append(os.path.join(self.output_dir, file))
            
            # Analyze dependencies
            dependencies = []
            blockers = []
            
            for file_path in task_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    # Look for dependency indicators
                    dep_matches = re.finditer(r'Depends on: ([^\n]+)', content)
                    block_matches = re.finditer(r'Blocked by: ([^\n]+)', content)
                    
                    for match in dep_matches:
                        dependencies.append(match.group(1).strip())
                    
                    for match in block_matches:
                        blockers.append(match.group(1).strip())
            
            # Write analysis
            analysis_path = os.path.join(self.output_dir, 'dependency_analysis.md')
            with open(analysis_path, 'w') as f:
                f.write("# Task Dependency Analysis\n\n")
                f.write(f"Generated on: {datetime.now()}\n\n")
                
                f.write("## Dependencies\n\n")
                for dep in sorted(set(dependencies)):
                    f.write(f"- {dep}\n")
                
                f.write("\n## Blockers\n\n")
                for blocker in sorted(set(blockers)):
                    f.write(f"- {blocker}\n")
            
            return analysis_path
            
        except Exception as e:
            logger.error(f"Error analyzing dependencies: {e}")
            return ""

def main():
    parser = argparse.ArgumentParser(description="Track tasks and dependencies")
    parser.add_argument("--output-dir", required=True, help="Output directory for tracked tasks")
    parser.add_argument("--jira-url", help="JIRA instance URL")
    parser.add_argument("--jira-token", help="JIRA API token")
    parser.add_argument("--github-token", help="GitHub API token")
    parser.add_argument("--slack-token", help="Slack API token")
    args = parser.parse_args()
    
    tracker = TaskTracker(
        args.output_dir,
        jira_url=args.jira_url,
        jira_token=args.jira_token,
        github_token=args.github_token,
        slack_token=args.slack_token
    )
    
    # Example JIRA projects
    projects = ["PROJ", "ENG", "INFRA"]
    
    # Example GitHub repositories
    repos = [
        "organization/frontend",
        "organization/backend",
        "organization/infrastructure"
    ]
    
    # Example Slack channels
    channels = ["C1234567890", "C0987654321"]  # Replace with actual channel IDs
    
    # Run trackers
    logger.info("Tracking JIRA tasks...")
    jira_files = tracker.track_jira_tasks(projects)
    
    logger.info("Tracking GitHub issues...")
    github_files = tracker.track_github_issues(repos)
    
    logger.info("Tracking Slack threads...")
    slack_files = tracker.track_slack_threads(channels)
    
    # Analyze dependencies
    logger.info("Analyzing dependencies...")
    analysis_file = tracker.analyze_dependencies()
    
    # Print summary
    logger.info(f"Tracked {len(jira_files)} JIRA projects")
    logger.info(f"Tracked {len(github_files)} GitHub repositories")
    logger.info(f"Tracked {len(slack_files)} Slack channels")
    if analysis_file:
        logger.info(f"Dependency analysis saved to: {analysis_file}")

if __name__ == "__main__":
    main() 