#!/usr/bin/env python3
"""
arXiv Paper Scraper for Specific AI/ML Domains
Extracts papers from targeted domains and saves PDFs to artifacts directory
"""

import os
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import time
from datetime import datetime, timedelta
import re
from pathlib import Path
from pusher import GitHubRepositoryManager

class ArxivScraper:
    def __init__(self, artifacts_dir="artifacts"):
        self.base_url = "http://export.arxiv.org/api/query"
        self.artifacts_dir = Path(artifacts_dir)
        
        # Initialize GitHub Repository Manager
        self.github_manager = GitHubRepositoryManager(artifacts_dir=artifacts_dir)
        
        # Ensure artifacts directory exists
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            print(f"Artifacts directory ready: {self.artifacts_dir.absolute()}")
        except Exception as e:
            print(f"Error creating artifacts directory: {e}")
            raise
        
        # Domain mappings without keyword filtering
        self.domains = {
            "cs.AI": ["cs.AI"],
            "cs.LG": ["cs.LG"],
            "stat.ML": ["stat.ML"],
            "cs.CL": ["cs.CL"],
            "cs.CV": ["cs.CV"],
            "cs.RO": ["cs.RO"],
            "cs.NE": ["cs.NE"],
            "cs.MA": ["cs.MA"],
            "cs.SY": ["cs.SY", "eess.SY"],
            "cs.HC": ["cs.HC"]
        }
        
        # Initialize GitHub manager and load seen titles
        print(f"📚 Loaded {len(self.github_manager.seen_titles)} seen titles from GitHub")
    
    def is_pdf_seen(self, title):
        """Check if a PDF title has been seen before using GitHub repository"""
        return self.github_manager.is_title_seen(title.strip())
    
    def add_new_titles(self, new_titles):
        """Add new titles to GitHub repository"""
        if new_titles:
            self.github_manager.add_seen_titles(new_titles)
            print(f"📚 Added {len(new_titles)} new titles to GitHub repository")
            print(f"📚 Total seen titles: {len(self.github_manager.seen_titles)}")
    
    def build_query(self, categories, max_results=50, days_back=30):
        """Build arXiv API query for specific domain"""
        # Build category query
        cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        params = {
            'search_query': cat_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        return params    

    def fetch_papers(self, domain_name, categories, max_results=50):
        """Fetch papers from arXiv API for a specific domain"""
        params = self.build_query(categories, max_results)
        
        print(f"Searching {domain_name}...")
        response = requests.get(self.base_url, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching papers for {domain_name}: {response.status_code}")
            return []
        
        # Parse XML response
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        entries = root.findall('atom:entry', namespace)
        
        for entry in entries:
            paper = self.parse_paper_entry(entry, namespace)
            if paper:
                papers.append(paper)
        
        print(f"Found {len(papers)} papers for {domain_name}")
        return papers
    
    def fetch_papers_adaptive(self, domain_name, categories, target_new_papers):
        """Fetch papers with adaptive search - keeps searching until we find new papers or hit limit"""
        max_attempts = 3  # Maximum search attempts
        search_multiplier = 1
        
        for attempt in range(max_attempts):
            # Increase search size each attempt
            search_size = target_new_papers * search_multiplier
            papers = self.fetch_papers(domain_name, categories, search_size)
            
            if not papers:
                break
                
            # Check how many are new
            new_count = 0
            for paper in papers:
                if not self.is_pdf_seen(paper['title']):
                    new_count += 1
            
            # If we found enough new papers, return all papers
            if new_count >= target_new_papers or search_multiplier >= 4:
                print(f"📈 Adaptive search found {new_count} new papers (attempt {attempt + 1})")
                return papers
            
            # If all papers are seen, try searching more
            if new_count == 0:
                search_multiplier *= 2
                print(f"🔍 All {len(papers)} papers already seen, expanding search (attempt {attempt + 1})")
            else:
                # Found some new papers, return them
                return papers
        
        # Return whatever we found in the last attempt
        return papers if 'papers' in locals() else []
    
    def parse_paper_entry(self, entry, namespace):
        """Parse individual paper entry from XML"""
        try:
            paper = {}
            
            # Basic info
            paper['id'] = entry.find('atom:id', namespace).text
            paper['title'] = entry.find('atom:title', namespace).text.strip()
            paper['summary'] = entry.find('atom:summary', namespace).text.strip()
            paper['published'] = entry.find('atom:published', namespace).text
            
            # Authors
            authors = []
            for author in entry.findall('atom:author', namespace):
                name = author.find('atom:name', namespace)
                if name is not None:
                    authors.append(name.text)
            paper['authors'] = authors
            
            # Categories
            categories = []
            for category in entry.findall('atom:category', namespace):
                term = category.get('term')
                if term:
                    categories.append(term)
            paper['categories'] = categories
            
            # PDF link
            for link in entry.findall('atom:link', namespace):
                if link.get('type') == 'application/pdf':
                    paper['pdf_url'] = link.get('href')
                    break
            
            # Extract arXiv ID for filename
            arxiv_id = paper['id'].split('/')[-1]
            paper['arxiv_id'] = arxiv_id
            
            return paper
            
        except Exception as e:
            print(f"Error parsing paper entry: {e}")
            return None
    
    def download_pdf(self, paper, domain_name):
        """Download PDF for a paper"""
        if 'pdf_url' not in paper:
            print(f"No PDF URL for paper {paper['arxiv_id']}")
            return False
        
        # Store all PDFs in artifacts/pdfs directory
        pdfs_dir = self.artifacts_dir / "pdfs"
        pdfs_dir.mkdir(exist_ok=True)
        
        # Clean filename with domain prefix
        safe_title = re.sub(r'[^\w\s-]', '', paper['title'][:50])
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        filename = f"{domain_name}_{paper['arxiv_id']}_{safe_title}.pdf"
        filepath = pdfs_dir / filename
        
        # Skip if already exists
        if filepath.exists():
            print(f"PDF already exists: {filename}")
            return True
        
        try:
            print(f"Downloading: {filename}")
            response = requests.get(paper['pdf_url'], stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded: {filename}")
            return True
            
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False
    
    def save_consolidated_metadata(self, all_results):
        """Save all paper metadata to a single consolidated file"""
        metadata_file = self.artifacts_dir / "metadata.txt"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write("arXiv AI/ML Papers - Consolidated Metadata\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            total_papers = 0
            for domain_name, papers in all_results.items():
                if papers:
                    f.write(f"\n{'='*20} {domain_name} DOMAIN {'='*20}\n")
                    f.write(f"Found {len(papers)} papers in {domain_name}\n")
                    f.write("-" * 60 + "\n\n")
                    
                    for paper in papers:
                        f.write(f"Title: {paper['title']}\n")
                        f.write(f"arXiv ID: {paper['arxiv_id']}\n")
                        f.write(f"Authors: {', '.join(paper['authors'])}\n")
                        f.write(f"Categories: {', '.join(paper['categories'])}\n")
                        f.write(f"Published: {paper['published']}\n")
                        f.write(f"Summary: {paper['summary'][:200]}...\n")
                        f.write("-" * 30 + "\n\n")
                        total_papers += 1
            
            f.write(f"\n{'='*60}\n")
            f.write(f"TOTAL PAPERS PROCESSED: {total_papers}\n")
            f.write(f"{'='*60}\n")
    
    def scrape_domain_until_target(self, domain_name, categories, target_count, download_pdfs=True):
        """Keep searching a domain until we find the target number of new papers"""
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain_name}")
        print(f"🎯 Target: {target_count} new papers")
        print(f"{'='*60}")
        
        new_papers = []
        new_titles = []
        total_checked = 0
        seen_count = 0
        search_start = 0
        batch_size = 20  # Search in batches of 20
        max_total_search = 200  # Don't search more than 200 papers total
        
        while len(new_papers) < target_count and total_checked < max_total_search:
            print(f"🔍 Searching batch starting from position {search_start}...")
            
            # Fetch next batch of papers
            params = self.build_query(categories, batch_size)
            params['start'] = search_start
            
            response = requests.get(self.base_url, params=params)
            if response.status_code != 200:
                print(f"❌ Error fetching papers for {domain_name}: {response.status_code}")
                break
            
            # Parse XML response
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', namespace)
            
            if not entries:
                print(f"📭 No more papers found in {domain_name}")
                break
            
            # Process each paper individually
            for entry in entries:
                paper = self.parse_paper_entry(entry, namespace)
                if not paper:
                    continue
                
                total_checked += 1
                
                # Check if paper is already seen
                if self.is_pdf_seen(paper['title']):
                    print(f"⏭️  Paper {total_checked}: SEEN - {paper['title'][:50]}...")
                    seen_count += 1
                else:
                    print(f"✨ Paper {total_checked}: NEW - {paper['title'][:50]}...")
                    
                    # Download immediately if requested
                    if download_pdfs:
                        if self.download_pdf(paper, domain_name):
                            print(f"📥 Downloaded successfully")
                        else:
                            print(f"❌ Download failed")
                        time.sleep(1)  # Be respectful to arXiv servers
                    
                    new_papers.append(paper)
                    new_titles.append(paper['title'])
                    
                    print(f"📊 Progress: {len(new_papers)}/{target_count} new papers found")
                    
                    # Stop if we've reached our target
                    if len(new_papers) >= target_count:
                        break
            
            search_start += batch_size
            time.sleep(2)  # Rate limiting between batches
        
        # Update seen titles in GitHub immediately after this domain
        if new_titles:
            print(f"📚 Adding {len(new_titles)} new titles to GitHub repository...")
            self.add_new_titles(new_titles)
            print(f"✅ Updated GitHub with new titles from {domain_name}")
        
        print(f"\n📊 Domain {domain_name} Summary:")
        print(f"  🎯 Target: {target_count} new papers")
        print(f"  ✅ Found: {len(new_papers)} new papers")
        print(f"  ⏭️  Skipped: {seen_count} already seen papers")
        print(f"  🔍 Total checked: {total_checked} papers")
        
        if len(new_papers) < target_count:
            print(f"  ⚠️  Could only find {len(new_papers)}/{target_count} new papers in {domain_name}")
        else:
            print(f"  🎉 Successfully found all {target_count} new papers!")
        
        return new_papers

    def scrape_all_domains(self, max_results_per_domain=5, download_pdfs=True):
        """Scrape exactly the requested number of new papers from each domain"""
        print("Starting arXiv scraping for AI/ML domains...")
        print(f"🎯 Target: {max_results_per_domain} NEW papers per domain")
        print(f"📁 Artifacts will be saved to: {self.artifacts_dir.absolute()}")
        print(f"📚 Starting with {len(self.github_manager.seen_titles)} seen titles from GitHub")
        
        all_results = {}
        
        for domain_name, categories in self.domains.items():
            # Scrape this domain until we get the target number of new papers
            new_papers = self.scrape_domain_until_target(
                domain_name, categories, max_results_per_domain, download_pdfs
            )
            
            all_results[domain_name] = new_papers
            
            # Wait 30 seconds before moving to next domain (as requested)
            if domain_name != list(self.domains.keys())[-1]:  # Don't wait after last domain
                print(f"⏳ Waiting 30 seconds before processing next domain...")
                time.sleep(30)
        
        return all_results
    
    def generate_summary_report(self, results):
        """Generate a summary report of all scraped papers"""
        summary_file = self.artifacts_dir / "scraping_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("arXiv AI/ML Domain Scraping Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            total_papers = 0
            for domain_name, papers in results.items():
                paper_count = len(papers)
                total_papers += paper_count
                f.write(f"{domain_name}: {paper_count} papers\n")
            
            f.write(f"\nTotal papers found: {total_papers}\n")
            f.write(f"Papers saved to: {self.artifacts_dir.absolute()}\n")


def main():
    """Main execution function"""
    scraper = ArxivScraper()
    
    try:
        # Scrape all domains
        results = scraper.scrape_all_domains(
            max_results_per_domain=2,  # Only fetch 2 papers per domain
            download_pdfs=True
        )
        
        # Save consolidated metadata
        scraper.save_consolidated_metadata(results)
        
        # Generate summary
        scraper.generate_summary_report(results)
        
        print(f"\n{'='*60}")
        print("Scraping completed!")
        print(f"Check the '{scraper.artifacts_dir}' directory for PDFs and metadata")
        print(f"{'='*60}")
        
    finally:
        # Save seen titles but don't clean up artifacts (YAML workflow handles cleanup)
        print("\n🔄 Finalizing scraper - saving seen titles to GitHub...")
        scraper.github_manager.save_seen_titles()
        print("✅ Scraper completed and seen titles saved to GitHub!")


if __name__ == "__main__":
    main()
