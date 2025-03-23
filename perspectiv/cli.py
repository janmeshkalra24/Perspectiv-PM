"""
Command Line Interface for the Perspectiv Knowledge Base.

This module provides a CLI for managing and visualizing the knowledge base.
"""

import os
import sys
import click
import rich
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.progress import track
from datetime import datetime
from typing import Optional

from .knowledge_base import KnowledgeBase

console = Console()

def format_timestamp(timestamp: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp

@click.group()
@click.option('--data-dir', default='./data', help='Knowledge base data directory')
@click.pass_context
def cli(ctx, data_dir):
    """Perspectiv Knowledge Base CLI"""
    ctx.ensure_object(dict)
    ctx.obj['kb'] = KnowledgeBase(data_dir=data_dir)
    ctx.obj['kb'].initialize()

@cli.command()
@click.pass_context
def status(ctx):
    """Show knowledge base status and statistics"""
    kb = ctx.obj['kb']
    stats = kb.get_stats()
    
    # Create status table
    table = Table(title="Knowledge Base Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Documents", str(stats['document_count']))
    table.add_row("Total Sessions", str(stats['sessions']))
    table.add_row("Total Tokens", str(stats['total_tokens']))
    
    console.print(table)
    
    # Show domain statistics
    domain_table = Table(title="Domain Statistics")
    domain_table.add_column("Domain", style="cyan")
    domain_table.add_column("Documents", style="green")
    domain_table.add_column("Last Updated", style="yellow")
    
    for domain, meta in kb.metadata['domains'].items():
        domain_table.add_row(
            domain,
            str(meta['document_count']),
            format_timestamp(meta['last_updated'])
        )
    
    console.print(domain_table)

@cli.command()
@click.argument('domain')
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_context
def add(ctx, domain: str, file_path: str):
    """Add a document to a domain"""
    kb = ctx.obj['kb']
    
    if domain not in kb.retriever.domains:
        console.print(f"[red]Error:[/red] Domain '{domain}' does not exist")
        return
    
    try:
        success = kb.retriever.add_document(file_path, domain)
        if success:
            console.print(f"[green]Successfully added[/green] {file_path} to domain '{domain}'")
        else:
            console.print(f"[red]Failed to add[/red] {file_path} to domain '{domain}'")
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")

@cli.command()
@click.argument('query')
@click.option('--domain', help='Specific domain to search in')
@click.option('--top-k', default=3, help='Number of results to show per domain')
@click.pass_context
def search(ctx, query: str, domain: Optional[str], top_k: int):
    """Search the knowledge base"""
    kb = ctx.obj['kb']
    
    domains = [domain] if domain else None
    results = kb.query(query, domains=domains, top_k=top_k)
    
    # Show domain relevance scores
    score_table = Table(title="Domain Relevance Scores")
    score_table.add_column("Domain", style="cyan")
    score_table.add_column("Score", style="green")
    
    for domain, score in results['domain_scores']:
        score_table.add_row(domain, f"{score:.4f}")
    
    console.print(score_table)
    
    # Show results from each domain
    for domain, domain_results in results['results'].items():
        if not domain_results:
            continue
            
        console.print(f"\n[cyan]Results from {domain}:[/cyan]")
        for i, result in enumerate(domain_results, 1):
            panel = Panel(
                f"{result['content'][:300]}...",
                title=f"[{i}] Score: {result['score']:.4f}",
                subtitle=f"Source: {result['metadata'].get('file_name', 'Unknown')}"
            )
            console.print(panel)

@cli.command()
@click.pass_context
def sessions(ctx):
    """List all sessions"""
    kb = ctx.obj['kb']
    sessions = kb.session_manager.list_sessions()
    
    if not sessions:
        console.print("[yellow]No sessions found[/yellow]")
        return
    
    table = Table(title="Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Last Updated", style="yellow")
    
    for session in sessions:
        table.add_row(
            session['id'],
            format_timestamp(session['created_at']),
            format_timestamp(session['last_updated'])
        )
    
    console.print(table)

@cli.command()
@click.argument('session_id')
@click.pass_context
def session(ctx, session_id: str):
    """Show details of a specific session"""
    kb = ctx.obj['kb']
    history = kb.get_session_history(session_id)
    
    if 'error' in history:
        console.print(f"[red]Error:[/red] {history['error']}")
        return
    
    # Session info
    console.print(Panel(
        f"Created: {format_timestamp(history['created_at'])}\n"
        f"Last Updated: {format_timestamp(history['last_updated'])}",
        title=f"Session: {session_id}"
    ))
    
    # Queries
    if history['queries']:
        query_table = Table(title="Queries")
        query_table.add_column("Query", style="cyan")
        query_table.add_column("Timestamp", style="green")
        query_table.add_column("Has Results", style="yellow")
        
        for query in history['queries']:
            query_table.add_row(
                query['query'],
                format_timestamp(query['timestamp']),
                "✓" if query['results'] else "✗"
            )
        
        console.print(query_table)
    
    # Transcripts
    if history['transcripts']:
        transcript_table = Table(title="Transcripts")
        transcript_table.add_column("File", style="cyan")
        transcript_table.add_column("Added", style="green")
        
        for transcript in history['transcripts']:
            transcript_table.add_row(
                os.path.basename(transcript['original_path']),
                format_timestamp(transcript['timestamp'])
            )
        
        console.print(transcript_table)

def main():
    cli(obj={}) 