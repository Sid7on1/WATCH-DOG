#!/usr/bin/env python3
"""
Advanced Paper Content Extractor - Complete WATCHDOG_memory Integration
Fetches seen_titles.txt from repo, processes papers, pushes all artifacts back
"""

import os
import re
import json
import time
import asyncio
import aiohttp
import hashlib
import PyPDF2
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import arxiv
from tqdm import tqdm
import tiktoken
from dotenv import load_dotenv
import base64


@dataclass
class ExtractionConfig:
    """Configuration for complete paper extraction and WATCHDOG_memory integration."""
    # OpenRouter API settings
    openrouter_api_key: str
    model: str = "deepseek/deepseek-chat-v3-0324:free"  # FREE model, not cheaper
    temperature: float = 0.1
    max_tokens: int = 4000  # Higher for free model
    
    # Rate limiting with specified delays
    chunk_delay: float = 23.0  # 23 seconds between chunks
    paper_delay: float = 35.0  # 35 seconds between papers
    max_retries: int = 3
    
    # Processing limits
    max_tries_per_domain: int = 6  # Max 6 tries per domain
    min_relevant_papers: int = 1   # Process at least 1 relevant paper
    
    # WATCHDOG_memory integration
    github_token: str = ""
    github_username: str = "Sid7on1"
    watchdog_repo_name: str = "WATCHDOG_memory"
    seen_titles_url: str = "https://raw.githubusercontent.com/Sid7on1/WATCHDOG_memory/main/seen_titles.txt"
    
    # ArXiv settings
    days_back: int = 7
    target_domains: List[str] = field(default_factory=lambda: [
        "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "stat.ML"  # Expanded domains
    ])
    relevance_keywords: List[str] = field(default_factory=lambda: [
        # Core Transformer & LLM terms
        "transformer", "attention", "self-attention", "cross-attention", "multi-head attention",
        "language model", "LLM", "large language model", "BERT", "GPT", "T5", "LLAMA",
        
        # Architecture components
        "FFN", "feed forward", "feedforward", "MLP", "multi-layer perceptron",
        "layer norm", "layernorm", "normalization", "residual connection", "skip connection",
        "positional encoding", "embedding", "token", "tokenization",
        
        # Training & Optimization
        "gradient", "backpropagation", "optimization", "adam", "learning rate",
        "fine-tuning", "pre-training", "training", "loss function", "regularization",
        
        # Advanced concepts
        "vision language", "multimodal", "vision transformer", "ViT", "CLIP",
        "diffusion", "generative", "autoregressive", "encoder-decoder",
        "reasoning", "chain of thought", "in-context learning", "few-shot", "zero-shot",
        
        # AI Agents & Applications
        "agentic", "ai agent", "multi-agent", "autonomous", "planning", "tool use",
        "retrieval", "RAG", "knowledge", "memory", "instruction following",
        
        # Performance & Efficiency
        "efficient", "optimization", "compression", "quantization", "pruning",
        "scaling", "parameter efficient", "LoRA", "adapter", "distillation",
        
        # Emerging areas
        "mixture of experts", "MoE", "sparse", "routing", "gating",
        "reinforcement learning", "RLHF", "alignment", "safety", "neural architecture"
    ])
    
    # Output settings
    relevance_threshold: float = 0.3  # Very low threshold - include papers with potential
    verbose: bool = True


@dataclass
class ExtractedPaperContent:
    """Complete paper content for fix2.py processing."""
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    arxiv_id: str = ""
    pdf_url: str = ""
    categories: List[str] = field(default_factory=list)
    published: str = ""
    
    # Full extracted content
    full_text: str = ""
    llm_analysis: str = ""  # Raw LLM response (not JSON)
    relevance_score: float = 0.0
    relevance_reasoning: str = ""
    
    # Processing metadata
    extraction_method: str = ""
    processing_time: float = 0.0
    chunks_processed: int = 0
    token_count: int = 0


class AdvancedPaperExtractor:
    """Complete ArXiv paper extractor with WATCHDOG_memory integration."""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.session: Optional[aiohttp.ClientSession] = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Tracking
        self.seen_titles = set()
        self.processed_papers = []
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Output directories
        self.run_dir = Path(f"runs/{self.run_timestamp}")
        self.pdf_dir = self.run_dir / "pdfs"
        self.json_dir = self.run_dir / "jsons"
        self.text_dir = self.run_dir / "texts"
        
        # Create directories
        for dir_path in [self.run_dir, self.pdf_dir, self.json_dir, self.text_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("PaperExtractor")
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600),
            headers={
                "Authorization": f"Bearer {self.config.openrouter_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "AdvancedPaperExtractor/2.0"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def fetch_seen_titles_from_repo(self) -> set:
        """Fetch seen_titles.txt from WATCHDOG_memory repo."""
        print("📥 Fetching seen_titles.txt from WATCHDOG_memory repo...")
        
        try:
            async with self.session.get(self.config.seen_titles_url) as response:
                if response.status == 200:
                    content = await response.text()
                    titles = set(line.strip() for line in content.split('\n') if line.strip())
                    print(f"✅ Loaded {len(titles)} seen titles from WATCHDOG_memory repo")
                    return titles
                else:
                    print(f"⚠️  Could not fetch seen_titles.txt (status: {response.status})")
                    return set()
        except Exception as e:
            print(f"❌ Failed to fetch seen_titles.txt: {e}")
            return set()
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF with multiple methods."""
        stats = {"method": None, "pages": 0, "chars": 0, "time": 0}
        start_time = time.time()
        
        # Method 1: PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}\n")
            
            doc.close()
            
            if text_parts:
                full_text = "\n".join(text_parts)
                stats.update({
                    "method": "PyMuPDF",
                    "pages": len(doc),
                    "chars": len(full_text),
                    "time": time.time() - start_time
                })
                return full_text, stats
        except Exception as e:
            self.logger.warning(f"PyMuPDF failed: {e}")
        
        # Method 2: PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{text}\n")
                
                if text_parts:
                    full_text = "\n".join(text_parts)
                    stats.update({
                        "method": "PyPDF2",
                        "pages": len(pdf_reader.pages),
                        "chars": len(full_text),
                        "time": time.time() - start_time
                    })
                    return full_text, stats
        except Exception as e:
            self.logger.warning(f"PyPDF2 failed: {e}")
        
        raise Exception("All PDF extraction methods failed")
    
    async def call_llm(self, text: str, is_chunk: bool = False) -> Tuple[str, Optional[str]]:
        """Call FREE LLM model via OpenRouter."""
        system_prompt = """You are a research paper analyzer. Analyze the given research paper content and provide:

1. A comprehensive technical summary
2. Key algorithms and methods described
3. Important equations and formulas
4. Model architectures and components
5. Datasets and evaluation metrics used
6. Implementation details and insights
7. Main contributions and novelty

Provide a detailed analysis in plain text format. Be thorough and technical."""
        
        if is_chunk:
            user_prompt = f"""Analyze this section of a research paper:

{text}

Provide detailed technical analysis focusing on the content in this section."""
        else:
            user_prompt = f"""Analyze this complete research paper:

{text}

Provide comprehensive technical analysis covering all aspects of the paper."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0]['message']['content']
                            if content and content.strip():
                                return content.strip(), None
                            else:
                                return "", "Empty response from LLM"
                        else:
                            return "", "Invalid API response format"
                    else:
                        error_text = await response.text()
                        if "token" in error_text.lower() and "limit" in error_text.lower():
                            return "", "TOKEN_LIMIT_EXCEEDED"
                        return "", f"API error {response.status}: {error_text}"
                        
            except Exception as e:
                self.logger.error(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return "", f"Failed after {self.config.max_retries} attempts"
    
    def chunk_text(self, text: str, max_tokens: int = 3000) -> List[str]:
        """Split text into chunks by token count."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    async def analyze_paper_content(self, text: str) -> Tuple[str, int]:
        """Analyze paper content with LLM, chunking if needed."""
        print(f"📊 Analyzing paper content ({len(text):,} characters)...")
        
        # Try full text first
        token_count = len(self.tokenizer.encode(text))
        print(f"📈 Token count: {token_count:,}")
        
        if token_count <= 15000:  # Try full text first
            print("🔄 Attempting full text analysis...")
            analysis, error = await self.call_llm(text, is_chunk=False)
            
            if error != "TOKEN_LIMIT_EXCEEDED" and analysis:
                print("✅ Full text analysis successful!")
                return analysis, 1
            else:
                print(f"⚠️  Full text failed: {error}")
        
        # Chunk the text
        print("🔧 Chunking text for analysis...")
        chunks = self.chunk_text(text, max_tokens=3000)
        print(f"📦 Created {len(chunks)} chunks")
        
        chunk_analyses = []
        
        for i, chunk in enumerate(chunks):
            print(f"🔍 Analyzing chunk {i+1}/{len(chunks)}...")
            
            analysis, error = await self.call_llm(chunk, is_chunk=True)
            
            if analysis:
                chunk_analyses.append(f"=== Chunk {i+1} Analysis ===\n{analysis}")
                print(f"✅ Chunk {i+1} analyzed successfully")
            else:
                print(f"❌ Chunk {i+1} failed: {error}")
                chunk_analyses.append(f"=== Chunk {i+1} Analysis ===\nAnalysis failed: {error}")
            
            # 23 second delay between chunks
            if i < len(chunks) - 1:
                print(f"⏳ Waiting {self.config.chunk_delay} seconds before next chunk...")
                await asyncio.sleep(self.config.chunk_delay)
        
        # Combine all chunk analyses
        combined_analysis = "\n\n".join(chunk_analyses)
        print(f"📊 Chunking complete: {len(chunk_analyses)} chunks processed")
        
        return combined_analysis, len(chunks)
    
    def calculate_relevance_score(self, title: str, abstract: str, categories: List[str]) -> Tuple[float, str]:
        """Calculate relevance score based on keywords and domains."""
        content = f"{title.lower()} {abstract.lower()} {' '.join(categories).lower()}"
        
        # Count keyword matches
        keyword_matches = 0
        matched_keywords = []
        
        for keyword in self.config.relevance_keywords:
            if keyword.lower() in content:
                keyword_matches += 1
                matched_keywords.append(keyword)
        
        # Calculate score
        keyword_score = min(keyword_matches / len(self.config.relevance_keywords), 1.0)
        
        # Bonus for AI/ML terms
        ai_terms = ["neural", "learning", "algorithm", "model", "ai", "ml", "deep", "transformer"]
        ai_matches = sum(1 for term in ai_terms if term in content)
        ai_bonus = min(ai_matches * 0.1, 0.3)
        
        final_score = min(keyword_score + ai_bonus, 1.0)
        
        reasoning = f"Keywords matched: {matched_keywords} ({keyword_matches}/{len(self.config.relevance_keywords)}), AI terms: {ai_matches}, Final score: {final_score:.2f}"
        
        return final_score, reasoning
    
    async def download_pdf(self, url: str, filename: str) -> Path:
        """Download PDF from URL."""
        pdf_path = self.pdf_dir / filename
        
        if pdf_path.exists():
            print(f"📄 PDF already exists: {filename}")
            return pdf_path
        
        print(f"⬇️  Downloading PDF: {filename}")
        
        async with self.session.get(url) as response:
            if response.status == 200:
                with open(pdf_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                print(f"✅ Downloaded: {filename}")
                return pdf_path
            else:
                raise Exception(f"Failed to download PDF: HTTP {response.status}")
    
    async def fetch_papers_from_arxiv(self, domain: str, max_papers: int = 10) -> List[Dict[str, Any]]:
        """Fetch papers from ArXiv for a specific domain."""
        print(f"🔍 Searching ArXiv domain: {domain}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.days_back)
        
        search = arxiv.Search(
            query=f"cat:{domain}",
            max_results=max_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for paper in search.results():
            # Filter by date
            if paper.published.replace(tzinfo=None) < start_date:
                continue
            
            paper_info = {
                "id": paper.entry_id.split('/')[-1],
                "title": paper.title,
                "abstract": paper.summary,
                "authors": [str(author) for author in paper.authors],
                "published": paper.published.isoformat(),
                "categories": [str(cat) for cat in paper.categories],
                "pdf_url": paper.pdf_url
            }
            papers.append(paper_info)
        
        print(f"📚 Found {len(papers)} papers from {domain}")
        return papers 
   
    async def process_single_paper(self, paper_info: Dict[str, Any]) -> Optional[ExtractedPaperContent]:
        """Process a single paper completely."""
        title = paper_info["title"]
        
        # Check if already seen
        if title in self.seen_titles:
            print(f"⏭️  Skipping seen paper: {title[:60]}...")
            return None
        
        print(f"\n🔬 Processing paper: {title[:60]}...")
        start_time = time.time()
        
        try:
            # Step 1: Calculate relevance
            relevance_score, relevance_reasoning = self.calculate_relevance_score(
                title, paper_info["abstract"], paper_info["categories"]
            )
            
            print(f"📊 Relevance score: {relevance_score:.2f}")
            
            if relevance_score < self.config.relevance_threshold:
                print(f"❌ Paper not relevant (score: {relevance_score:.2f} < {self.config.relevance_threshold})")
                return None
            
            print(f"✅ Paper is relevant! Processing...")
            
            # Step 2: Download PDF
            pdf_filename = f"{paper_info['id']}.pdf"
            pdf_path = await self.download_pdf(paper_info["pdf_url"], pdf_filename)
            
            # Step 3: Extract text from PDF
            print("📄 Extracting text from PDF...")
            full_text, extraction_stats = self.extract_text_from_pdf(pdf_path)
            print(f"✅ Extracted {extraction_stats['chars']:,} characters using {extraction_stats['method']}")
            
            # Save raw text
            text_filename = f"{paper_info['id']}.txt"
            text_path = self.text_dir / text_filename
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            # Step 4: Analyze with LLM
            llm_analysis, chunks_processed = await self.analyze_paper_content(full_text)
            
            # Step 5: Create extracted content
            extracted_content = ExtractedPaperContent(
                title=title,
                abstract=paper_info["abstract"],
                authors=paper_info["authors"],
                arxiv_id=paper_info["id"],
                pdf_url=paper_info["pdf_url"],
                categories=paper_info["categories"],
                published=paper_info["published"],
                full_text=full_text,
                llm_analysis=llm_analysis,
                relevance_score=relevance_score,
                relevance_reasoning=relevance_reasoning,
                extraction_method=extraction_stats["method"],
                processing_time=time.time() - start_time,
                chunks_processed=chunks_processed,
                token_count=len(self.tokenizer.encode(full_text))
            )
            
            # Step 6: Save to JSON
            json_filename = f"{paper_info['id']}.json"
            json_path = self.json_dir / json_filename
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(extracted_content), f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved all artifacts for: {title[:60]}...")
            
            # Add to seen titles
            self.seen_titles.add(title)
            
            return extracted_content
            
        except Exception as e:
            print(f"❌ Failed to process paper: {e}")
            return None
    
    async def process_domain(self, domain: str) -> int:
        """Process papers from a specific domain."""
        print(f"\n🎯 Processing domain: {domain}")
        
        relevant_papers_found = 0
        tries = 0
        
        while relevant_papers_found == 0 and tries < self.config.max_tries_per_domain:
            tries += 1
            print(f"🔄 Domain {domain} - Attempt {tries}/{self.config.max_tries_per_domain}")
            
            # Fetch papers
            papers = await self.fetch_papers_from_arxiv(domain, max_papers=10)
            
            if not papers:
                print(f"⚠️  No papers found for domain {domain}")
                break
            
            # Process each paper
            for paper_info in papers:
                result = await self.process_single_paper(paper_info)
                
                if result:
                    relevant_papers_found += 1
                    self.processed_papers.append(result)
                    print(f"✅ Found relevant paper #{relevant_papers_found}")
                    
                    # If we found at least 1 relevant paper, we can stop
                    if relevant_papers_found >= self.config.min_relevant_papers:
                        break
                
                # 35 second delay between papers
                print(f"⏳ Waiting {self.config.paper_delay} seconds before next paper...")
                await asyncio.sleep(self.config.paper_delay)
            
            if relevant_papers_found >= self.config.min_relevant_papers:
                break
        
        print(f"📊 Domain {domain} complete: {relevant_papers_found} relevant papers found")
        return relevant_papers_found
    
    def save_local_seen_titles(self):
        """Save seen titles to local file."""
        seen_titles_path = self.run_dir / "seen_titles.txt"
        with open(seen_titles_path, 'w', encoding='utf-8') as f:
            for title in sorted(self.seen_titles):
                f.write(f"{title}\n")
        print(f"💾 Saved {len(self.seen_titles)} seen titles locally")
    
    async def push_artifacts_to_repo(self):
        """Push all artifacts to WATCHDOG_memory repo."""
        if not self.config.github_token:
            print("⚠️  No GitHub token provided, skipping repo push")
            return
        
        print(f"🚀 Pushing artifacts to WATCHDOG_memory repo...")
        
        try:
            # GitHub API headers
            headers = {
                "Authorization": f"token {self.config.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Get all files to upload
            files_to_upload = []
            
            # Collect all files from run directory
            for file_path in self.run_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(Path("."))
                    files_to_upload.append((str(relative_path), file_path))
            
            print(f"📁 Found {len(files_to_upload)} files to upload")
            
            # Upload each file
            for relative_path, file_path in files_to_upload:
                try:
                    # Read file content
                    if file_path.suffix.lower() == '.pdf':
                        with open(file_path, 'rb') as f:
                            content = base64.b64encode(f.read()).decode()
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = base64.b64encode(f.read().encode()).decode()
                    
                    # GitHub API URL
                    api_url = f"https://api.github.com/repos/{self.config.github_username}/{self.config.watchdog_repo_name}/contents/{relative_path}"
                    
                    # Create/update file
                    payload = {
                        "message": f"Add artifact: {relative_path}",
                        "content": content,
                        "branch": "main"
                    }
                    
                    async with self.session.put(api_url, headers=headers, json=payload) as response:
                        if response.status in [200, 201]:
                            print(f"✅ Uploaded: {relative_path}")
                        else:
                            print(f"❌ Failed to upload {relative_path}: {response.status}")
                
                except Exception as e:
                    print(f"❌ Error uploading {relative_path}: {e}")
            
            # Update main seen_titles.txt
            await self.update_main_seen_titles()
            
            print(f"🎉 All artifacts pushed to WATCHDOG_memory repo!")
            
        except Exception as e:
            print(f"❌ Failed to push artifacts: {e}")
    
    async def update_main_seen_titles(self):
        """Update the main seen_titles.txt in repo root."""
        try:
            # Combine all seen titles
            all_titles = sorted(self.seen_titles)
            content = '\n'.join(all_titles) + '\n'
            encoded_content = base64.b64encode(content.encode()).decode()
            
            headers = {
                "Authorization": f"token {self.config.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Get current file SHA
            api_url = f"https://api.github.com/repos/{self.config.github_username}/{self.config.watchdog_repo_name}/contents/seen_titles.txt"
            
            sha = None
            async with self.session.get(api_url, headers=headers) as response:
                if response.status == 200:
                    file_info = await response.json()
                    sha = file_info.get('sha')
            
            # Update file
            payload = {
                "message": f"Update seen_titles.txt - Added {len(self.processed_papers)} new papers",
                "content": encoded_content,
                "branch": "main"
            }
            
            if sha:
                payload["sha"] = sha
            
            async with self.session.put(api_url, headers=headers, json=payload) as response:
                if response.status in [200, 201]:
                    print(f"✅ Updated main seen_titles.txt with {len(all_titles)} titles")
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to update seen_titles.txt: {response.status} - {error_text}")
                    
        except Exception as e:
            print(f"❌ Failed to update main seen_titles.txt: {e}")
    
    async def run_complete_pipeline(self):
        """Run the complete paper extraction pipeline."""
        print("🚀 Starting Complete ArXiv Paper Extraction Pipeline")
        print("=" * 60)
        
        # Step 1: Fetch seen titles from repo
        self.seen_titles = await self.fetch_seen_titles_from_repo()
        
        # Step 2: Process each domain
        total_relevant_papers = 0
        
        for domain in self.config.target_domains:
            relevant_count = await self.process_domain(domain)
            total_relevant_papers += relevant_count
            
            if total_relevant_papers >= self.config.min_relevant_papers:
                print(f"✅ Minimum relevant papers ({self.config.min_relevant_papers}) reached!")
        
        # Step 3: Save local seen titles
        self.save_local_seen_titles()
        
        # Step 4: Generate summary
        print(f"\n📊 Pipeline Complete!")
        print(f"   📄 Total papers processed: {len(self.processed_papers)}")
        print(f"   ✅ Relevant papers found: {total_relevant_papers}")
        print(f"   📁 Run directory: {self.run_dir}")
        print(f"   🕒 Run timestamp: {self.run_timestamp}")
        
        # Step 5: Push all artifacts to WATCHDOG_memory repo
        await self.push_artifacts_to_repo()
        
        return {
            "total_processed": len(self.processed_papers),
            "relevant_papers": total_relevant_papers,
            "run_directory": str(self.run_dir),
            "timestamp": self.run_timestamp
        }


async def main():
    """Main function."""
    load_dotenv()
    
    print("🚀 Advanced Paper Content Extractor - WATCHDOG_memory Integration")
    print("=" * 70)
    
    # Get required environment variables
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN", "")
    
    if not openrouter_key:
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        return 1
    
    # Create configuration
    config = ExtractionConfig(
        openrouter_api_key=openrouter_key,
        github_token=github_token,
        verbose=True
    )
    
    print(f"📋 Configuration:")
    print(f"   🤖 Model: {config.model} (FREE)")
    print(f"   ⏱️  Chunk delay: {config.chunk_delay}s")
    print(f"   ⏱️  Paper delay: {config.paper_delay}s")
    print(f"   🎯 Min relevant papers: {config.min_relevant_papers}")
    print(f"   🔄 Max tries per domain: {config.max_tries_per_domain}")
    print(f"   📚 Domains: {', '.join(config.target_domains)}")
    print(f"   🔑 Keywords: {', '.join(config.relevance_keywords[:5])}...")
    print(f"   🔗 WATCHDOG_memory: {'✅' if github_token else '❌'}")
    print()
    
    try:
        async with AdvancedPaperExtractor(config) as extractor:
            results = await extractor.run_complete_pipeline()
            
            print(f"\n🎉 All Done!")
            print(f"📊 Results: {results}")
            return 0
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
