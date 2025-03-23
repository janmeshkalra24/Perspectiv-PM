#!/usr/bin/env python3
"""
Technical Documentation Scraper

This script scrapes technical documentation and engineering resources from:
1. GitHub repositories (READMEs, wikis, docs)
2. Engineering blogs
3. Stack Overflow top answers
4. Technical documentation sites
"""

import os
import json
import logging
from typing import List, Dict, Any
import argparse
from datetime import datetime
import re
import base64
import codecs

import requests
from bs4 import BeautifulSoup
import feedparser
from stackapi import StackAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechScraper:
    def __init__(
        self,
        output_dir: str,
        github_token: str = None,
        stack_api_key: str = None
    ):
        """
        Initialize the technical content scraper.
        
        Args:
            output_dir: Directory to save scraped content
            github_token: GitHub API token
            stack_api_key: Stack Exchange API key
        """
        self.output_dir = os.path.abspath(output_dir)
        self.github_token = github_token
        
        # Initialize Stack Exchange API client
        self.stack_api = StackAPI('stackoverflow', key=stack_api_key) if stack_api_key else None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def scrape_github_docs(self, repos: List[str]) -> List[str]:
        """
        Scrape documentation from GitHub repositories.
        
        Args:
            repos: List of repository names (e.g. ["facebook/react"])
            
        Returns:
            List of paths to scraped files
        """
        scraped_files = []
        
        for repo in repos:
            try:
                # Get repository documentation
                url = f"https://api.github.com/repos/{repo}/contents/docs"
                headers = {"Authorization": f"token {self.github_token}"} if self.github_token else {}
                
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    logger.warning(f"Error accessing {repo} docs: {response.status_code}")
                    continue
                
                files = response.json()
                if not isinstance(files, list):
                    logger.warning(f"No documentation found for {repo}")
                    continue
                
                for file in files:
                    if file['type'] != 'file' or not file['name'].endswith(('.md', '.txt')):
                        continue
                        
                    try:
                        # Get file content
                        content_response = requests.get(file['download_url'], headers=headers)
                        if content_response.status_code != 200:
                            continue
                            
                        # Handle both raw and base64 encoded content
                        content = content_response.text
                        if content_response.headers.get('content-transfer-encoding') == 'base64':
                            content = base64.b64decode(content).decode('utf-8')
                        
                        # Save to file
                        output_path = os.path.join(self.output_dir, f"github_{repo.replace('/', '_')}_{file['name']}")
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        scraped_files.append(output_path)
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file['name']} from {repo}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing GitHub repo {repo}: {e}")
                continue
                
        return scraped_files
    
    def scrape_stack_overflow(
        self,
        tags: List[str],
        min_score: int = 10,
        max_questions: int = 50
    ) -> List[str]:
        """
        Scrape top Stack Overflow answers.
        
        Args:
            tags: List of tags to search for
            min_score: Minimum score for questions/answers
            max_questions: Maximum number of questions to fetch
        
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        if not self.stack_api:
            logger.warning("Stack Exchange API key not provided, skipping Stack Overflow scraping")
            return saved_files
        
        try:
            # Get questions with specified tags
            questions = self.stack_api.fetch(
                'questions',
                tagged=';'.join(tags),
                sort='votes',
                min=min_score,
                pagesize=max_questions,
                filter='withbody'
            )
            
            for question in questions['items']:
                # Get answers for this question
                answers = self.stack_api.fetch(
                    'questions/{ids}/answers',
                    ids=[question['question_id']],
                    sort='votes',
                    filter='withbody'
                )
                
                if answers['items']:
                    file_path = os.path.join(self.output_dir, f"stackoverflow_{question['question_id']}.md")
                    with open(file_path, 'w') as f:
                        f.write(f"# {question['title']}\n\n")
                        f.write(f"Tags: {', '.join(question['tags'])}\n")
                        f.write(f"Score: {question['score']}\n")
                        f.write(f"URL: https://stackoverflow.com/q/{question['question_id']}\n\n")
                        f.write("## Question\n\n")
                        f.write(question['body'])
                        
                        f.write("\n\n## Top Answers\n\n")
                        for answer in answers['items'][:3]:  # Get top 3 answers
                            f.write(f"### Answer (Score: {answer['score']})\n\n")
                            f.write(answer['body'])
                            f.write("\n\n")
                    
                    saved_files.append(file_path)
            
        except Exception as e:
            logger.error(f"Error scraping Stack Overflow: {e}")
        
        return saved_files
    
    def scrape_tech_blogs(self, blog_urls: List[str]) -> List[str]:
        """
        Scrape technical blog posts.
        
        Args:
            blog_urls: List of blog URLs to scrape
        
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        for url in blog_urls:
            try:
                # Try to find RSS feed first
                feed_url = url.rstrip('/') + '/feed'
                feed = feedparser.parse(feed_url)
                
                if feed.entries:
                    # Process RSS feed entries
                    for entry in feed.entries[:10]:  # Get latest 10 posts
                        title = entry.title
                        content = entry.description
                        link = entry.link
                        date = entry.get('published', '')
                        
                        # Only save if it looks technical (basic heuristic)
                        if any(term in title.lower() or term in content.lower() 
                              for term in ['code', 'api', 'database', 'architecture', 'engineering']):
                            
                            file_path = os.path.join(self.output_dir, f"techblog_{title[:50]}.md")
                            with open(file_path, 'w') as f:
                                f.write(f"# {title}\n\n")
                                f.write(f"Date: {date}\n")
                                f.write(f"Source: {link}\n\n")
                                f.write(content)
                            
                            saved_files.append(file_path)
                else:
                    # Fallback to direct scraping
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    title = soup.title.string if soup.title else url.split('/')[-1]
                    article = soup.find('article') or soup.find('main') or soup.find('body')
                    content = article.get_text() if article else response.text
                    
                    file_path = os.path.join(self.output_dir, f"techblog_{title[:50]}.md")
                    with open(file_path, 'w') as f:
                        f.write(f"# {title}\n\nSource: {url}\n\n{content}")
                    
                    saved_files.append(file_path)
                
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
        
        return saved_files

def main():
    parser = argparse.ArgumentParser(description="Scrape technical documentation")
    parser.add_argument("--output-dir", required=True, help="Output directory for scraped content")
    parser.add_argument("--github-token", help="GitHub API token")
    parser.add_argument("--stack-api-key", help="Stack Exchange API key")
    args = parser.parse_args()
    
    scraper = TechScraper(
        args.output_dir,
        github_token=args.github_token,
        stack_api_key=args.stack_api_key
    )
    
    # Example GitHub repositories to scrape
    repos = [
        "facebook/react",
        "tensorflow/tensorflow",
        "kubernetes/kubernetes",
        "django/django"
    ]
    
    # Example tech blog URLs
    blogs = [
        "https://engineering.fb.com",
        "https://netflixtechblog.com",
        "https://engineering.linkedin.com/blog",
        "https://blog.google/technology"
    ]
    
    # Example Stack Overflow tags
    tags = [
        "python",
        "javascript",
        "docker",
        "kubernetes",
        "react",
        "machine-learning"
    ]
    
    # Run scrapers
    logger.info("Scraping GitHub documentation...")
    github_files = scraper.scrape_github_docs(repos)
    
    logger.info("Scraping Stack Overflow...")
    stack_files = scraper.scrape_stack_overflow(tags)
    
    logger.info("Scraping tech blogs...")
    blog_files = scraper.scrape_tech_blogs(blogs)
    
    # Print summary
    logger.info(f"Scraped {len(github_files)} GitHub documents")
    logger.info(f"Scraped {len(stack_files)} Stack Overflow Q&As")
    logger.info(f"Scraped {len(blog_files)} tech blog posts")

if __name__ == "__main__":
    main() 