#!/usr/bin/env python3
"""
Knowledge Base Population Script

This script helps users populate their knowledge base with content from various sources:
1. Technical documentation (from tech_scraper.py)
2. Product Management FAQs (from pm_faq_scraper.py)
3. Tasks and dependencies (from task_tracker.py)
"""

import os
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import shutil
import sys

# Add the parent directory to the path so we can import the perspectiv package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from tqdm import tqdm

from perspectiv.knowledge_base import KnowledgeBase
from tech_scraper import TechScraper
from pm_faq_scraper import PMScraper
from task_tracker import TaskTracker

# Set up logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# If detailed logging is enabled, add more information to the format
if os.getenv('DETAILED_LOGGING', '').lower() == 'true':
    for handler in logging.root.handlers:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))

# If log file is specified, add file handler
log_file = os.getenv('LOG_FILE')
if log_file:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.root.handlers[0].formatter)
    logging.root.addHandler(file_handler)

class KnowledgeBasePopulator:
    def __init__(
        self,
        kb_dir: str,
        temp_dir: str = None,
        config_file: str = None
    ):
        """
        Initialize the knowledge base populator.
        
        Args:
            kb_dir: Directory for the knowledge base
            temp_dir: Temporary directory for scraped content
            config_file: Optional configuration file path
        """
        self.kb_dir = os.path.abspath(kb_dir)
        self.temp_dir = os.path.abspath(temp_dir) if temp_dir else os.path.join(kb_dir, '.temp')
        self.config = self._load_config(config_file) if config_file else {}
        
        # Load scraping options from environment
        self.scrape_depth = os.getenv('SCRAPE_DEPTH', 'medium')
        self.max_docs = int(os.getenv('MAX_DOCS_PER_SOURCE', '100'))
        self.min_length = int(os.getenv('MIN_CONTENT_LENGTH', '500'))
        self.max_length = int(os.getenv('MAX_CONTENT_LENGTH', '50000'))
        self.language = os.getenv('LANGUAGE', 'en')
        self.min_date = os.getenv('MIN_DATE')
        self.exclude_archived = os.getenv('EXCLUDE_ARCHIVED', '').lower() == 'true'
        
        # Create directories
        os.makedirs(self.kb_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Create domain directories and add placeholder files
        self.domains = ['tech_decoding', 'product_manager_faqs', 'tasks_blockers_deps', 'cache_history']
        for domain in self.domains:
            domain_dir = os.path.join(self.kb_dir, domain)
            os.makedirs(domain_dir, exist_ok=True)
            placeholder_path = os.path.join(domain_dir, '.placeholder')
            if not os.path.exists(placeholder_path):
                with open(placeholder_path, 'w') as f:
                    f.write(f"Placeholder file for {domain} domain")
        
        # Initialize knowledge base
        self.kb = KnowledgeBase(self.kb_dir)
        
        # Initialize domain indexes
        logger.info("Initializing knowledge base and building indexes...")
        self._initialize_domain_indexes()
        
        # Check for local documents directory
        self.local_docs_dir = os.getenv('LOCAL_DOCS_DIR')
        if self.local_docs_dir and os.path.exists(self.local_docs_dir):
            logger.info(f"Found local documents directory: {self.local_docs_dir}")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading config file: {e}")
            return {}
    
    def _should_skip_source(self, source_name: str, required_env_vars: List[str]) -> bool:
        """Check if a source should be skipped based on missing environment variables."""
        for var in required_env_vars:
            if not os.getenv(var):
                logger.info(f"Skipping {source_name} scraping (missing {var})")
                return True
        return False
    
    def scrape_tech_docs(self) -> List[str]:
        """
        Scrape technical documentation.
        
        Returns:
            List of scraped file paths
        """
        output_dir = os.path.join(self.temp_dir, 'tech_docs')
        os.makedirs(output_dir, exist_ok=True)
        
        scraped_files = []
        
        # Initialize scraper with available tokens
        scraper = TechScraper(
            output_dir=output_dir,
            github_token=os.getenv('GITHUB_TOKEN'),
            stack_api_key=os.getenv('STACK_API_KEY')
        )
        
        # GitHub scraping
        if not self._should_skip_source('GitHub', ['GITHUB_TOKEN']):
            repos = self.config.get('tech_repos', [
                "facebook/react",
                "tensorflow/tensorflow"
            ])
            logger.info("Scraping GitHub documentation...")
            github_files = scraper.scrape_github_docs(repos)
            scraped_files.extend(github_files)
        
        # Stack Overflow scraping
        if not self._should_skip_source('Stack Overflow', ['STACK_API_KEY']):
            tags = self.config.get('stack_tags', [
                "python",
                "javascript"
            ])
            logger.info("Scraping Stack Overflow...")
            stack_files = scraper.scrape_stack_overflow(tags)
            scraped_files.extend(stack_files)
        
        # Blog scraping (no auth required)
        blogs = self.config.get('tech_blogs', [
            "https://engineering.fb.com",
            "https://netflixtechblog.com"
        ])
        logger.info("Scraping tech blogs...")
        blog_files = scraper.scrape_tech_blogs(blogs)
        scraped_files.extend(blog_files)
        
        # Local documents
        if self.local_docs_dir:
            logger.info(f"Processing local documents from {self.local_docs_dir}...")
            for root, _, files in os.walk(self.local_docs_dir):
                for file in files:
                    if file.endswith(('.md', '.txt', '.pdf', '.docx')):
                        src = os.path.join(root, file)
                        dst = os.path.join(output_dir, file)
                        shutil.copy2(src, dst)
                        scraped_files.append(dst)
        
        return scraped_files
    
    def scrape_pm_faqs(self) -> List[str]:
        """
        Scrape product management FAQs.
        
        Returns:
            List of scraped file paths
        """
        output_dir = os.path.join(self.temp_dir, 'pm_faqs')
        os.makedirs(output_dir, exist_ok=True)
        
        scraped_files = []
        
        # Initialize scraper with available tokens
        scraper = PMScraper(
            output_dir=output_dir,
            reddit_client_id=os.getenv('REDDIT_CLIENT_ID'),
            reddit_client_secret=os.getenv('REDDIT_CLIENT_SECRET')
        )
        
        # Blog scraping (no auth required)
        blogs = self.config.get('pm_blogs', [
            "https://www.mindtheproduct.com",
            "https://www.productschool.com/blog"
        ])
        logger.info("Scraping PM blogs...")
        blog_files = scraper.scrape_product_blogs(blogs)
        scraped_files.extend(blog_files)
        
        # Reddit scraping
        if not self._should_skip_source('Reddit', ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET']):
            logger.info("Scraping Reddit r/ProductManagement...")
            reddit_files = scraper.scrape_reddit_pm()
            scraped_files.extend(reddit_files)
        
        # FAQ scraping (no auth required)
        faqs = self.config.get('pm_faqs', [
            "https://www.productschool.com/blog/product-management-2/product-manager-faq"
        ])
        logger.info("Scraping PM FAQs...")
        faq_files = scraper.scrape_pm_faqs(faqs)
        scraped_files.extend(faq_files)
        
        return scraped_files
    
    def track_tasks(self) -> List[str]:
        """
        Track tasks and dependencies.
        
        Returns:
            List of tracked file paths
        """
        output_dir = os.path.join(self.temp_dir, 'tasks')
        os.makedirs(output_dir, exist_ok=True)
        
        scraped_files = []
        
        # Initialize tracker with available tokens
        tracker = TaskTracker(
            output_dir=output_dir,
            jira_url=os.getenv('JIRA_URL'),
            jira_token=os.getenv('JIRA_TOKEN'),
            github_token=os.getenv('GITHUB_TOKEN'),
            slack_token=os.getenv('SLACK_TOKEN')
        )
        
        # JIRA tracking
        if not self._should_skip_source('JIRA', ['JIRA_URL', 'JIRA_TOKEN']):
            projects = self.config.get('jira_projects', ["PROJ", "ENG"])
            logger.info("Tracking JIRA tasks...")
            jira_files = tracker.track_jira_tasks(projects)
            scraped_files.extend(jira_files)
        
        # GitHub issue tracking
        if not self._should_skip_source('GitHub Issues', ['GITHUB_TOKEN']):
            repos = self.config.get('task_repos', ["organization/frontend"])
            logger.info("Tracking GitHub issues...")
            github_files = tracker.track_github_issues(repos)
            scraped_files.extend(github_files)
        
        # Slack thread tracking
        if not self._should_skip_source('Slack', ['SLACK_TOKEN']):
            channels = self.config.get('slack_channels', {}).values()
            logger.info("Tracking Slack threads...")
            slack_files = tracker.track_slack_threads(list(channels))
            scraped_files.extend(slack_files)
        
        # Analyze dependencies if we have any files
        if scraped_files:
            logger.info("Analyzing dependencies...")
            analysis_file = tracker.analyze_dependencies()
            if analysis_file:
                scraped_files.append(analysis_file)
        
        return scraped_files
    
    def populate_knowledge_base(self, files: List[str], domain: str):
        """
        Add scraped content to the knowledge base.
        
        Args:
            files: List of file paths to add
            domain: Knowledge base domain to add content to
        """
        if not files:
            logger.warning(f"No files to add to domain '{domain}'")
            return
        
        logger.info(f"Adding {len(files)} files to knowledge base domain '{domain}'...")
        
        for file_path in tqdm(files):
            try:
                # Add to knowledge base using the retriever
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                    
                success = self.kb.retriever.add_document(file_path, domain)
                if not success:
                    logger.warning(f"Failed to add {file_path} to domain '{domain}'")
                
            except Exception as e:
                logger.error(f"Error adding file {file_path} to domain '{domain}': {e}")
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")

    def _initialize_domain_indexes(self):
        """Initialize indexes for all domains."""
        for domain in self.domains:
            domain_dir = os.path.join(self.kb_dir, domain)
            # Create a small test document to initialize the index
            test_doc_path = os.path.join(domain_dir, '.init_doc.txt')
            with open(test_doc_path, 'w') as f:
                f.write(f"Initialization document for {domain}")
            
            # Add document to initialize the index
            success = self.kb.retriever.add_document(test_doc_path, domain)
            if success:
                logger.info(f"Initialized index for domain: {domain}")
            else:
                logger.error(f"Failed to initialize index for domain: {domain}")
            
            # Clean up test document
            os.remove(test_doc_path)

def main():
    parser = argparse.ArgumentParser(description="Populate knowledge base with scraped content")
    parser.add_argument("--kb-dir", required=True, help="Knowledge base directory")
    parser.add_argument("--temp-dir", help="Temporary directory for scraped content")
    parser.add_argument("--config", help="Configuration file path")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize populator
    populator = KnowledgeBasePopulator(
        kb_dir=args.kb_dir,
        temp_dir=args.temp_dir,
        config_file=args.config
    )
    
    try:
        # Scrape and populate tech docs
        tech_files = populator.scrape_tech_docs()
        populator.populate_knowledge_base(tech_files, domain='tech_decoding')
        
        # Scrape and populate PM FAQs
        pm_files = populator.scrape_pm_faqs()
        populator.populate_knowledge_base(pm_files, domain='product_manager_faqs')
        
        # Track and populate tasks
        task_files = populator.track_tasks()
        populator.populate_knowledge_base(task_files, domain='tasks_blockers_deps')
        
        logger.info("Knowledge base population complete!")
        
    finally:
        # Clean up temporary files
        populator.cleanup()

if __name__ == "__main__":
    main() 