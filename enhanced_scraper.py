#!/usr/bin/env python3
"""
Enhanced arXiv Paper Scraper for Advanced AI/ML Projects
Focuses on RAG, NLP, Agents, and CI/CD workflows
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

class EnhancedArxivScraper:
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
        
        # Enhanced domain mappings focused on AI/ML implementation
        self.domains = {
            "RAG_Systems": {
                "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.IR"],
                "keywords": [
                    "retrieval augmented generation", "rag", "retrieval-augmented",
                    "vector database", "embedding retrieval", "semantic search",
                    "knowledge retrieval", "document retrieval", "context retrieval",
                    "llm retrieval", "hybrid search", "dense retrieval", "sparse retrieval"
                ]
            },
            "Advanced_NLP": {
                "categories": ["cs.CL", "cs.AI", "cs.LG"],
                "keywords": [
                    "transformer", "attention mechanism", "bert", "gpt", "llama",
                    "natural language processing", "language model", "text generation",
                    "sentiment analysis", "named entity recognition", "question answering",
                    "text classification", "machine translation", "summarization",
                    "tokenization", "embedding", "fine-tuning", "prompt engineering"
                ]
            },
            "AI_Agents": {
                "categories": ["cs.AI", "cs.MA", "cs.LG", "cs.RO"],
                "keywords": [
                    "autonomous agent", "multi-agent", "agent-based", "intelligent agent",
                    "reinforcement learning", "policy gradient", "actor-critic",
                    "agent communication", "cooperative agents", "agent coordination",
                    "llm agent", "tool-using agent", "planning agent", "reasoning agent"
                ]
            },
            "MLOps_CICD": {
                "categories": ["cs.SE", "cs.LG", "cs.AI"],
                "keywords": [
                    "mlops", "machine learning operations", "model deployment",
                    "continuous integration", "continuous deployment", "ci/cd",
                    "model monitoring", "model versioning", "experiment tracking",
                    "automated testing", "pipeline automation", "devops", "kubernetes",
                    "docker", "model serving", "a/b testing", "feature store"
                ]
            },
            "Advanced_ML": {
                "categories": ["cs.LG", "cs.AI", "stat.ML"],
                "keywords": [
                    "deep learning", "neural network", "convolutional", "recurrent",
                    "generative adversarial", "variational autoencoder", "diffusion model",
                    "federated learning", "meta-learning", "few-shot learning",
                    "transfer learning", "self-supervised", "contrastive learning",
                    "graph neural network", "attention", "optimization"
                ]
            },
            "Computer_Vision": {
                "categories": ["cs.CV", "cs.AI", "cs.LG"],
                "keywords": [
                    "computer vision", "image processing", "object detection",
                    "image segmentation", "face recognition", "optical character recognition",
                    "image classification", "generative model", "style transfer",
                    "video analysis", "3d vision", "medical imaging", "autonomous driving"
                ]
            }
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
    
    def build_enhanced_query(self, domain_info, max_results=50):
        """Build enhanced arXiv API query with keyword filtering"""
        categories = domain_info["categories"]
        keywords = domain_info["keywords"]
        
        # Build category query
        cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        # Build keyword query for title and abstract
        keyword_query = " OR ".join([f'ti:"{kw}"' for kw in keywords[:5]])  # Limit to avoid URL length issues
        abstract_query = " OR ".join([f'abs:"{kw}"' for kw in keywords[:5]])
        
        # Combine queries
        full_query = f"({cat_query}) AND (({keyword_query}) OR ({abstract_query}))"
        
        params = {
            'search_query': full_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        return params
    
    def calculate_relevance_score(self, paper, domain_info):
        """Calculate relevance score based on keywords in title and abstract"""
        title = paper.get('title', '').lower()
        abstract = paper.get('summary', '').lower()
        keywords = domain_info["keywords"]
        
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in title:
                score += 3  # Title matches are more important
                matched_keywords.append(keyword)
            elif keyword_lower in abstract:
                score += 1  # Abstract matches
                matched_keywords.append(keyword)
        
        return score, matched_keywords
    
    def fetch_enhanced_papers(self, domain_name, domain_info, max_results=50):
        """Fetch papers with enhanced filtering and relevance scoring"""
        params = self.build_enhanced_query(domain_info, max_results)
        
        print(f"🔍 Searching {domain_name} with enhanced filtering...")
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
                # Calculate relevance score
                score, matched_keywords = self.calculate_relevance_score(paper, domain_info)
                paper['relevance_score'] = score
                paper['matched_keywords'] = matched_keywords
                paper['domain'] = domain_name
                
                # Only include papers with relevance score > 0
                if score > 0:
                    papers.append(paper)
        
        # Sort by relevance score (highest first)
        papers.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        print(f"Found {len(papers)} relevant papers for {domain_name}")
        return papers
    
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
        
        # Store PDFs organized by domain
        pdfs_dir = self.artifacts_dir / "pdfs" / domain_name
        pdfs_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean filename with relevance score
        safe_title = re.sub(r'[^\w\s-]', '', paper['title'][:50])
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        score = paper.get('relevance_score', 0)
        filename = f"score_{score}_{paper['arxiv_id']}_{safe_title}.pdf"
        filepath = pdfs_dir / filename
        
        # Skip if already exists
        if filepath.exists():
            print(f"PDF already exists: {filename}")
            return True
        
        try:
            print(f"📥 Downloading: {filename} (score: {score})")
            response = requests.get(paper['pdf_url'], stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return False
    
    def scrape_domain_until_target(self, domain_name, domain_info, target_count, download_pdfs=True):
        """Keep searching a domain until we find the target number of new papers"""
        print(f"\n{'='*60}")
        print(f"🎯 Processing domain: {domain_name}")
        print(f"🎯 Target: {target_count} new papers")
        print(f"🔍 Keywords: {', '.join(domain_info['keywords'][:5])}...")
        print(f"{'='*60}")
        
        new_papers = []
        new_titles = []
        total_checked = 0
        seen_count = 0
        search_start = 0
        batch_size = 30  # Larger batch for enhanced search
        max_total_search = 300  # Search more papers for better quality
        
        while len(new_papers) < target_count and total_checked < max_total_search:
            print(f"🔍 Searching batch starting from position {search_start}...")
            
            # Fetch next batch of papers with enhanced filtering
            domain_info_batch = domain_info.copy()
            papers = self.fetch_enhanced_papers(domain_name, domain_info_batch, batch_size)
            
            if not papers:
                print(f"📭 No more papers found in {domain_name}")
                break
            
            # Process each paper individually
            for paper in papers:
                total_checked += 1
                
                # Check if paper is already seen
                if self.is_pdf_seen(paper['title']):
                    print(f"⏭️  Paper {total_checked}: SEEN - {paper['title'][:50]}... (score: {paper['relevance_score']})")
                    seen_count += 1
                else:
                    print(f"✨ Paper {total_checked}: NEW - {paper['title'][:50]}... (score: {paper['relevance_score']})")
                    print(f"   📋 Keywords: {', '.join(paper['matched_keywords'][:3])}")
                    
                    # Download immediately if requested
                    if download_pdfs:
                        if self.download_pdf(paper, domain_name):
                            print(f"   📥 Downloaded successfully")
                        else:
                            print(f"   ❌ Download failed")
                        time.sleep(1)  # Be respectful to arXiv servers
                    
                    new_papers.append(paper)
                    new_titles.append(paper['title'])
                    
                    print(f"📊 Progress: {len(new_papers)}/{target_count} new papers found")
                    
                    # Stop if we've reached our target
                    if len(new_papers) >= target_count:
                        break
            
            search_start += batch_size
            time.sleep(3)  # Rate limiting between batches
        
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
        
        if new_papers:
            avg_score = sum(p['relevance_score'] for p in new_papers) / len(new_papers)
            print(f"  📈 Average relevance score: {avg_score:.1f}")
        
        if len(new_papers) < target_count:
            print(f"  ⚠️  Could only find {len(new_papers)}/{target_count} new papers in {domain_name}")
        else:
            print(f"  🎉 Successfully found all {target_count} new papers!")
        
        return new_papers

    def scrape_all_domains(self, max_results_per_domain=3, download_pdfs=True):
        """Scrape exactly the requested number of new papers from each enhanced domain"""
        print("🚀 Starting Enhanced arXiv scraping for Advanced AI/ML domains...")
        print(f"🎯 Target: {max_results_per_domain} NEW papers per domain")
        print(f"📁 Artifacts will be saved to: {self.artifacts_dir.absolute()}")
        print(f"📚 Starting with {len(self.github_manager.seen_titles)} seen titles from GitHub")
        
        all_results = {}
        
        for domain_name, domain_info in self.domains.items():
            # Scrape this domain until we get the target number of new papers
            new_papers = self.scrape_domain_until_target(
                domain_name, domain_info, max_results_per_domain, download_pdfs
            )
            
            all_results[domain_name] = new_papers
            
            # Wait between domains
            if domain_name != list(self.domains.keys())[-1]:
                print(f"⏳ Waiting 45 seconds before processing next domain...")
                time.sleep(45)
        
        return all_results
    
    def generate_enhanced_summary_report(self, results):
        """Generate an enhanced summary report of all scraped papers"""
        summary_file = self.artifacts_dir / "enhanced_scraping_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Enhanced arXiv AI/ML Domain Scraping Summary\n")
            f.write("Focus: RAG, NLP, Agents, MLOps/CI-CD, Advanced ML, Computer Vision\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            total_papers = 0
            total_score = 0
            
            for domain_name, papers in results.items():
                paper_count = len(papers)
                total_papers += paper_count
                
                if papers:
                    domain_score = sum(p['relevance_score'] for p in papers)
                    avg_score = domain_score / paper_count
                    total_score += domain_score
                    
                    f.write(f"\n{'='*20} {domain_name} DOMAIN {'='*20}\n")
                    f.write(f"Found {paper_count} papers in {domain_name}\n")
                    f.write(f"Average relevance score: {avg_score:.1f}\n")
                    f.write("-" * 60 + "\n\n")
                    
                    for paper in papers:
                        f.write(f"Title: {paper['title']}\n")
                        f.write(f"arXiv ID: {paper['arxiv_id']}\n")
                        f.write(f"Relevance Score: {paper['relevance_score']}\n")
                        f.write(f"Matched Keywords: {', '.join(paper['matched_keywords'])}\n")
                        f.write(f"Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}\n")
                        f.write(f"Categories: {', '.join(paper['categories'])}\n")
                        f.write(f"Published: {paper['published']}\n")
                        f.write(f"Summary: {paper['summary'][:200]}...\n")
                        f.write("-" * 30 + "\n\n")
            
            f.write(f"\n{'='*70}\n")
            f.write(f"TOTAL PAPERS PROCESSED: {total_papers}\n")
            if total_papers > 0:
                f.write(f"OVERALL AVERAGE RELEVANCE SCORE: {total_score/total_papers:.1f}\n")
            f.write(f"Papers saved to: {self.artifacts_dir.absolute()}\n")
            f.write(f"{'='*70}\n")


def main():
    """Main execution function"""
    scraper = EnhancedArxivScraper()
    
    try:
        # Scrape all enhanced domains
        results = scraper.scrape_all_domains(
            max_results_per_domain=3,  # Fetch 3 high-quality papers per domain
            download_pdfs=True
        )
        
        # Generate enhanced summary
        scraper.generate_enhanced_summary_report(results)
        
        print(f"\n{'='*70}")
        print("🎉 Enhanced scraping completed!")
        print(f"Check the '{scraper.artifacts_dir}' directory for PDFs and metadata")
        print("Focus areas: RAG Systems, Advanced NLP, AI Agents, MLOps/CI-CD")
        print(f"{'='*70}")
        
    finally:
        # Save seen titles
        print("\n🔄 Finalizing enhanced scraper - saving seen titles to GitHub...")
        scraper.github_manager.save_seen_titles()
        print("✅ Enhanced scraper completed and seen titles saved to GitHub!")


if __name__ == "__main__":
    main()