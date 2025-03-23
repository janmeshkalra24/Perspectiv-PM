#!/usr/bin/env python3
"""
Product Manager FAQ Scraper

This script scrapes product management FAQs and best practices.
"""

import os
import json
import logging
from typing import List, Dict, Any
import argparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import praw
import feedparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PMScraper:
    def __init__(
        self, 
        output_dir: str,
        reddit_client_id: str = None,
        reddit_client_secret: str = None,
        reddit_user_agent: str = "PMScraper 1.0"
    ):
        """
        Initialize the PM content scraper.
        
        Args:
            output_dir: Directory to save scraped content
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: Reddit API user agent
        """
        self.output_dir = os.path.abspath(output_dir)
        
        # Initialize Reddit client if credentials provided
        self.reddit = None
        if reddit_client_id and reddit_client_secret:
            self.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def scrape_product_blogs(self, blog_urls: List[str]) -> List[str]:
        """
        Scrape PM blog posts and articles.
        
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
                        
                        file_path = os.path.join(self.output_dir, f"blog_{title[:50]}.md")
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
                    
                    file_path = os.path.join(self.output_dir, f"blog_{title[:50]}.md")
                    with open(file_path, 'w') as f:
                        f.write(f"# {title}\n\nSource: {url}\n\n{content}")
                    
                    saved_files.append(file_path)
                
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
        
        return saved_files
    
    def scrape_reddit_pm(self, max_posts: int = 100) -> List[str]:
        """
        Scrape top posts from r/ProductManagement.
        
        Args:
            max_posts: Maximum number of posts to scrape
        
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        if not self.reddit:
            logger.warning("Reddit API credentials not provided, skipping Reddit scraping")
            return saved_files
        
        try:
            subreddit = self.reddit.subreddit('ProductManagement')
            
            # Get top posts
            for post in subreddit.top(time_filter='all', limit=max_posts):
                file_path = os.path.join(self.output_dir, f"reddit_{post.id}.md")
                
                with open(file_path, 'w') as f:
                    f.write(f"# {post.title}\n\n")
                    f.write(f"Score: {post.score}\n")
                    f.write(f"Date: {datetime.fromtimestamp(post.created_utc)}\n")
                    f.write(f"URL: {post.url}\n\n")
                    f.write(post.selftext)
                    
                    # Get top comments
                    f.write("\n\n## Top Comments\n\n")
                    post.comments.replace_more(limit=0)
                    for comment in post.comments[:5]:  # Get top 5 comments
                        f.write(f"### Comment (Score: {comment.score})\n")
                        f.write(comment.body)
                        f.write("\n\n")
                
                saved_files.append(file_path)
                
        except Exception as e:
            logger.error(f"Error scraping Reddit: {e}")
        
        return saved_files
    
    def scrape_pm_faqs(self, faq_urls: List[str]) -> List[str]:
        """
        Scrape PM FAQs from various sources.
        
        Args:
            faq_urls: List of FAQ URLs to scrape
        
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        for url in faq_urls:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try to find FAQ sections (common patterns)
                faqs = []
                
                # Look for question-answer pairs
                questions = soup.find_all(['h2', 'h3', 'strong'])
                for q in questions:
                    # Try to find the answer (next sibling or parent's next sibling)
                    answer = q.find_next(['p', 'div']) or q.parent.find_next(['p', 'div'])
                    if answer:
                        faqs.append({
                            'question': q.get_text().strip(),
                            'answer': answer.get_text().strip()
                        })
                
                if faqs:
                    file_path = os.path.join(self.output_dir, f"faq_{url.split('/')[-1]}.md")
                    with open(file_path, 'w') as f:
                        f.write(f"# FAQs from {url}\n\n")
                        for faq in faqs:
                            f.write(f"## {faq['question']}\n\n")
                            f.write(f"{faq['answer']}\n\n")
                    
                    saved_files.append(file_path)
                
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
        
        return saved_files

def main():
    parser = argparse.ArgumentParser(description="Scrape PM knowledge")
    parser.add_argument("--output-dir", required=True, help="Output directory for scraped content")
    parser.add_argument("--reddit-client-id", help="Reddit API client ID")
    parser.add_argument("--reddit-client-secret", help="Reddit API client secret")
    args = parser.parse_args()
    
    scraper = PMScraper(
        args.output_dir,
        reddit_client_id=args.reddit_client_id,
        reddit_client_secret=args.reddit_client_secret
    )
    
    # Example PM blog URLs
    blogs = [
        "https://www.mindtheproduct.com",
        "https://www.productschool.com/blog",
        "https://www.productplan.com/blog",
        "https://www.romanpichler.com/blog"
    ]
    
    # Example FAQ URLs
    faqs = [
        "https://www.productschool.com/blog/product-management-2/product-manager-faq",
        "https://www.atlassian.com/agile/product-management/product-manager",
        "https://www.productplan.com/learn/product-manager-interview-questions"
    ]
    
    # Run scrapers
    logger.info("Scraping PM blogs...")
    blog_files = scraper.scrape_product_blogs(blogs)
    
    logger.info("Scraping Reddit r/ProductManagement...")
    reddit_files = scraper.scrape_reddit_pm()
    
    logger.info("Scraping PM FAQs...")
    faq_files = scraper.scrape_pm_faqs(faqs)
    
    # Print summary
    logger.info(f"Scraped {len(blog_files)} blog posts")
    logger.info(f"Scraped {len(reddit_files)} Reddit posts")
    logger.info(f"Scraped {len(faq_files)} FAQ pages")

if __name__ == "__main__":
    main() 