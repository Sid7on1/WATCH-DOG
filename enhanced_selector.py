#!/usr/bin/env python3
"""
Enhanced PDF Text Selector for Advanced AI/ML Projects
Focuses on RAG, NLP, Agents, and CI/CD implementation potential
"""

import os
import requests
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from pusher import GitHubRepositoryManager

# Load environment variables
load_dotenv()

def env(*names):
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return None

# Enhanced keyword sets for advanced AI/ML projects
ADVANCED_AI_KEYWORDS = {
    # RAG Systems
    "retrieval augmented generation", "rag", "vector database", "embedding retrieval",
    "semantic search", "knowledge retrieval", "document retrieval", "context retrieval",
    "llm retrieval", "hybrid search", "dense retrieval", "sparse retrieval",
    
    # Advanced NLP
    "transformer", "attention mechanism", "bert", "gpt", "llama", "language model",
    "text generation", "sentiment analysis", "named entity recognition", "question answering",
    "text classification", "machine translation", "summarization", "tokenization",
    "embedding", "fine-tuning", "prompt engineering", "instruction tuning",
    
    # AI Agents
    "autonomous agent", "multi-agent", "agent-based", "intelligent agent",
    "reinforcement learning", "policy gradient", "actor-critic", "agent communication",
    "cooperative agents", "agent coordination", "llm agent", "tool-using agent",
    "planning agent", "reasoning agent", "agent framework", "agent architecture",
    
    # MLOps/CI-CD
    "mlops", "machine learning operations", "model deployment", "continuous integration",
    "continuous deployment", "ci/cd", "model monitoring", "model versioning",
    "experiment tracking", "automated testing", "pipeline automation", "devops",
    "kubernetes", "docker", "model serving", "a/b testing", "feature store",
    
    # Advanced ML Techniques
    "deep learning", "neural network", "convolutional", "recurrent", "generative adversarial",
    "variational autoencoder", "diffusion model", "federated learning", "meta-learning",
    "few-shot learning", "transfer learning", "self-supervised", "contrastive learning",
    "graph neural network", "optimization", "distributed training"
}

IMPLEMENTATION_TERMS = {
    "algorithm", "method", "architecture", "implementation", "code", "pipeline",
    "framework", "module", "inference", "training loop", "loss function", "latency",
    "scalability", "performance", "optimization", "deployment", "production",
    "real-time", "batch processing", "api", "microservice", "containerization"
}

PROJECT_POTENTIAL_TERMS = {
    "open source", "github", "reproducible", "benchmark", "dataset", "evaluation",
    "metrics", "baseline", "state-of-the-art", "sota", "practical", "application",
    "industry", "commercial", "scalable", "efficient", "robust", "reliable"
}

def compute_enhanced_relevance_score(text: str) -> dict:
    """Compute enhanced relevance score with detailed breakdown"""
    t = text.lower()
    
    # Count different types of matches
    ai_matches = sum(1 for k in ADVANCED_AI_KEYWORDS if k in t)
    impl_matches = sum(1 for k in IMPLEMENTATION_TERMS if k in t)
    project_matches = sum(1 for k in PROJECT_POTENTIAL_TERMS if k in t)
    
    # Calculate weighted score
    total_score = (ai_matches * 2.0) + (impl_matches * 1.5) + (project_matches * 1.0)
    
    return {
        "total_score": total_score,
        "ai_keywords": ai_matches,
        "implementation_terms": impl_matches,
        "project_potential": project_matches,
        "is_highly_relevant": total_score >= 5.0,
        "is_implementable": impl_matches >= 2 and ai_matches >= 1
    }

class EnhancedPDFSelector:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.pdf_txts_dir = self.artifacts_dir / "pdf-txts"
        self.relevant_dir = self.artifacts_dir / "relevant"
        
        # Initialize GitHub Repository Manager
        self.github_manager = GitHubRepositoryManager(artifacts_dir=artifacts_dir)
        
        # Enhanced multi-API configuration for advanced analysis
        self.apis = {
            "openrouter": {
                "key": env("OPEN_API", "OPENROUTER_API_KEY", "OPENROUTER_API_TOKEN"),
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "models": [
                    "deepseek/deepseek-r1-0528:free",           # Best reasoning for analysis
                    "google/gemini-2.0-flash-exp:free",         # Fast + reliable
                    "qwen/qwen3-coder:free",                    # Technical analysis
                    "mistralai/mistral-nemo:free"               # High rate limits
                ]
            },
            "groq": {
                "key": env("groq_API", "GROQ_API"),
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "models": [
                    "llama-3.3-70b-versatile",                 # Best for analysis
                    "gemma2-9b-it",                            # Fast for selection
                    "meta-llama/llama-4-maverick-17b-128e-instruct"
                ]
            },
            "cohere": {
                "key": env("cohere_API", "COHERE_API", "COHERE_API_KEY"),
                "url": "https://api.cohere.ai/v2/chat",
                "models": ["command-r7b-12-2024"]
            }
        }
        
        # Current API selection
        self.current_api = "openrouter"
        self.current_model_index = 0
        
        # Enhanced configuration
        self.max_chunk_size = 30000  # Larger chunks for better context
        self.strict_threshold = float(os.getenv("SELECTOR_MIN_SCORE", "3.0"))  # Higher threshold
        
        # Ensure directories exist
        self.relevant_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("🚀 ENHANCED PDF SELECTOR - ADVANCED AI/ML FOCUS")
        print("="*80)
        print(f"📁 Source Directory    : {self.pdf_txts_dir}")
        print(f"📁 Relevant Papers     : {self.relevant_dir}")
        print(f"📚 Previously Seen PDFs: {len(self.github_manager.seen_titles)}")
        print(f"🎯 Focus Areas: RAG Systems, Advanced NLP, AI Agents, MLOps/CI-CD")
        print(f"🔍 Minimum Score Threshold: {self.strict_threshold}")
        print("="*80)
    
    def get_current_api_config(self):
        """Get current API configuration"""
        api_config = self.apis[self.current_api]
        current_model = api_config["models"][self.current_model_index]
        return api_config, current_model
    
    def switch_to_next_api(self):
        """Switch to next available API/model combination"""
        current_api_config = self.apis[self.current_api]
        if self.current_model_index < len(current_api_config["models"]) - 1:
            self.current_model_index += 1
            print(f"🔄 Switched to next model in {self.current_api}: {current_api_config['models'][self.current_model_index]}")
            return True
        
        api_names = list(self.apis.keys())
        current_api_index = api_names.index(self.current_api)
        
        for i in range(1, len(api_names)):
            next_api_index = (current_api_index + i) % len(api_names)
            next_api = api_names[next_api_index]
            
            if self.apis[next_api]["key"]:
                self.current_api = next_api
                self.current_model_index = 0
                print(f"🔄 Switched to API: {self.current_api} with model: {self.apis[next_api]['models'][0]}")
                return True
        
        print("❌ No more APIs available to switch to")
        return False
    
    def make_api_request(self, system_prompt, user_prompt, max_tokens=1500):
        """Make API request with current configuration"""
        api_config, current_model = self.get_current_api_config()
        
        if self.current_api == "cohere":
            return self.make_cohere_request(system_prompt, user_prompt, current_model, max_tokens)
        else:
            return self.make_openai_compatible_request(system_prompt, user_prompt, current_model, max_tokens)
    
    def make_openai_compatible_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to OpenAI-compatible APIs"""
        api_config, _ = self.get_current_api_config()
        
        headers = {
            "Authorization": f"Bearer {api_config['key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        response = requests.post(api_config["url"], headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def make_cohere_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to Cohere API"""
        try:
            # Try multiple import methods for Cohere
            try:
                from cohere import ClientV2
                client_class = ClientV2
            except ImportError:
                try:
                    from cohere import Client
                    client_class = Client
                except ImportError:
                    raise ImportError("Cohere library not available")
            
            api_config, _ = self.get_current_api_config()
            client = client_class(api_key=api_config['key'])
            
            # Combine system and user prompts for Cohere
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Try different API methods
            try:
                # Try v2 chat method
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                response = client.chat(
                    model=model,
                    messages=messages,
                    temperature=0.1
                )
                return response.message.content[0].text
            except:
                # Try v1 chat method
                response = client.chat(
                    model=model,
                    message=combined_prompt,
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                return response.text
            
        except ImportError:
            # Fallback to REST API
            api_config, _ = self.get_current_api_config()
            
            headers = {
                "Authorization": f"Bearer {api_config['key']}",
                "Content-Type": "application/json"
            }
            
            # Try v2 API format first
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1
            }
            
            try:
                response = requests.post(api_config["url"], headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result['message']['content'][0]['text']
            except:
                # Try v1 API format
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                payload = {
                    "model": model,
                    "message": combined_prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.1
                }
                response = requests.post("https://api.cohere.ai/v1/chat", headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result['text']
    
    def analyze_chunk_for_advanced_ai(self, chunk_text, pdf_name, chunk_num, total_chunks, max_retries=2):
        """Analyze chunk specifically for advanced AI/ML implementation potential"""
        
        system_prompt = """You are an expert AI/ML researcher and engineer specializing in advanced AI systems implementation.

MISSION: Identify papers with HIGH IMPLEMENTATION POTENTIAL for building advanced AI/ML projects.

PRIORITY FOCUS AREAS:
1. RAG (Retrieval-Augmented Generation) Systems
   - Vector databases, embedding retrieval, semantic search
   - Knowledge retrieval, document retrieval, context retrieval
   - Hybrid search, dense/sparse retrieval methods

2. Advanced NLP & Language Models
   - Transformer architectures, attention mechanisms
   - Fine-tuning, prompt engineering, instruction tuning
   - Text generation, classification, summarization
   - Multi-modal language models

3. AI Agents & Multi-Agent Systems
   - Autonomous agents, intelligent agent frameworks
   - Agent communication, coordination, cooperation
   - LLM-based agents, tool-using agents
   - Planning agents, reasoning agents

4. MLOps & CI/CD for AI
   - Model deployment, monitoring, versioning
   - Continuous integration/deployment for ML
   - Experiment tracking, automated testing
   - Containerization, orchestration, serving

5. Advanced ML Techniques
   - Deep learning architectures, optimization
   - Federated learning, meta-learning
   - Self-supervised learning, contrastive learning
   - Graph neural networks, diffusion models

EVALUATION CRITERIA:
- Does it provide CONCRETE algorithms/methods that can be implemented?
- Are there clear technical details for building systems?
- Does it include practical applications or use cases?
- Is it relevant to modern AI/ML development practices?
- Can it be turned into a working project/system?

RESPONSE FORMAT:
RELEVANT or NOT RELEVANT
Brief explanation focusing on implementation potential and specific techniques."""

        user_prompt = f"""Paper: {pdf_name}
Chunk: {chunk_num}/{total_chunks}

Content:
{chunk_text}

Analyze this chunk for advanced AI/ML implementation potential. Focus on concrete algorithms, architectures, and methods that can be built into working systems."""

        for attempt in range(max_retries + 1):
            try:
                api_config, current_model = self.get_current_api_config()
                print(f"🔍 Analyzing with {self.current_api}/{current_model} (attempt {attempt + 1})")
                
                analysis_text = self.make_api_request(system_prompt, user_prompt, 1500)
                
                # Parse the response
                is_relevant = analysis_text.lower().startswith("relevant")
                
                # Calculate enhanced relevance score
                score_info = compute_enhanced_relevance_score(chunk_text)
                
                analysis = {
                    "relevant": is_relevant and score_info["is_highly_relevant"],
                    "confidence": 0.9 if is_relevant and score_info["is_implementable"] else 0.3,
                    "reason": analysis_text,
                    "enhanced_score": score_info["total_score"],
                    "ai_keywords": score_info["ai_keywords"],
                    "implementation_terms": score_info["implementation_terms"],
                    "project_potential": score_info["project_potential"],
                    "is_implementable": score_info["is_implementable"],
                    "continue_reading": is_relevant or score_info["is_highly_relevant"]
                }
                
                print(f"📊 Enhanced Score: {score_info['total_score']:.1f} | AI: {score_info['ai_keywords']} | Impl: {score_info['implementation_terms']} | Proj: {score_info['project_potential']}")
                
                return analysis
                
            except Exception as e:
                print(f"❌ Error in analysis attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    if self.switch_to_next_api():
                        time.sleep(10)
                        continue
                    else:
                        break
                else:
                    break
        
        return None
    
    def process_pdf_with_enhanced_analysis(self, pdf_name):
        """Process a PDF with enhanced analysis for advanced AI/ML content"""
        pdf_text_dir = self.pdf_txts_dir / pdf_name.replace('.pdf', '')
        
        if not pdf_text_dir.exists():
            print(f"❌ Text directory not found: {pdf_text_dir}")
            return False
        
        # Get all chunk files
        chunk_files = sorted(pdf_text_dir.glob("chunk_*.txt"))
        
        if not chunk_files:
            print(f"❌ No chunk files found in {pdf_text_dir}")
            return False
        
        print(f"\n🔍 Enhanced analysis of {pdf_name} ({len(chunk_files)} chunks)")
        
        relevant_chunks = []
        total_enhanced_score = 0
        
        for i, chunk_file in enumerate(chunk_files, 1):
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_content = f.read()
                
                print(f"📄 Analyzing chunk {i}/{len(chunk_files)}...")
                
                analysis = self.analyze_chunk_for_advanced_ai(
                    chunk_content, pdf_name, i, len(chunk_files)
                )
                
                if analysis:
                    total_enhanced_score += analysis["enhanced_score"]
                    
                    if analysis["relevant"]:
                        relevant_chunks.append((i, chunk_content, analysis))
                        print(f"✅ Chunk {i}: RELEVANT (score: {analysis['enhanced_score']:.1f})")
                    else:
                        print(f"❌ Chunk {i}: NOT RELEVANT (score: {analysis['enhanced_score']:.1f})")
                    
                    # Continue reading decision
                    if not analysis["continue_reading"] and i >= 2:
                        print(f"⏹️  Stopping analysis - low relevance detected")
                        break
                else:
                    print(f"❌ Failed to analyze chunk {i}")
                
                time.sleep(12)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error processing chunk {i}: {e}")
                continue
        
        # Decision logic
        avg_score = total_enhanced_score / len(chunk_files) if chunk_files else 0
        is_relevant = len(relevant_chunks) > 0 and avg_score >= self.strict_threshold
        
        print(f"\n📊 Enhanced Analysis Summary for {pdf_name}:")
        print(f"   📈 Average Enhanced Score: {avg_score:.1f}")
        print(f"   ✅ Relevant Chunks: {len(relevant_chunks)}/{len(chunk_files)}")
        print(f"   🎯 Threshold: {self.strict_threshold}")
        print(f"   📋 Decision: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")
        
        if is_relevant:
            self.save_relevant_paper(pdf_name, relevant_chunks, avg_score)
            return True
        else:
            return False
    
    def save_relevant_paper(self, pdf_name, relevant_chunks, avg_score):
        """Save relevant paper chunks with enhanced metadata"""
        paper_dir = self.relevant_dir / pdf_name.replace('.pdf', '')
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each relevant chunk
        for chunk_num, chunk_content, analysis in relevant_chunks:
            chunk_file = paper_dir / f"chunk_{chunk_num}.txt"
            
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(f"PDF: {pdf_name}\n")
                f.write(f"Chunk: {chunk_num}\n")
                f.write(f"Enhanced Score: {analysis['enhanced_score']:.1f}\n")
                f.write(f"AI Keywords: {analysis['ai_keywords']}\n")
                f.write(f"Implementation Terms: {analysis['implementation_terms']}\n")
                f.write(f"Project Potential: {analysis['project_potential']}\n")
                f.write(f"Implementable: {analysis['is_implementable']}\n")
                f.write("=" * 50 + "\n\n")
                f.write(chunk_content)
        
        # Save metadata
        metadata = {
            "pdf_name": pdf_name,
            "total_chunks": len(relevant_chunks),
            "average_enhanced_score": avg_score,
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "focus_areas": "RAG, NLP, Agents, MLOps/CI-CD, Advanced ML",
            "chunks": [
                {
                    "chunk_num": chunk_num,
                    "enhanced_score": analysis["enhanced_score"],
                    "ai_keywords": analysis["ai_keywords"],
                    "implementation_terms": analysis["implementation_terms"],
                    "project_potential": analysis["project_potential"],
                    "is_implementable": analysis["is_implementable"]
                }
                for chunk_num, _, analysis in relevant_chunks
            ]
        }
        
        metadata_file = paper_dir / "enhanced_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved {len(relevant_chunks)} relevant chunks for {pdf_name}")
    
    def process_all_pdfs(self):
        """Process all PDFs with enhanced analysis"""
        if not self.pdf_txts_dir.exists():
            print(f"❌ PDF texts directory not found: {self.pdf_txts_dir}")
            return
        
        # Get all PDF directories (organized by domain)
        pdf_dirs = [d for d in self.pdf_txts_dir.iterdir() if d.is_dir()]
        
        if not pdf_dirs:
            print("❌ No PDF text directories found")
            return
        
        print(f"🔍 Found {len(pdf_dirs)} PDF directories to process")
        
        relevant_count = 0
        total_count = 0
        
        for pdf_dir in pdf_dirs:
            pdf_name = pdf_dir.name + '.pdf'
            total_count += 1
            
            print(f"\n{'='*60}")
            print(f"Processing {total_count}/{len(pdf_dirs)}: {pdf_name}")
            print(f"{'='*60}")
            
            if self.process_pdf_with_enhanced_analysis(pdf_name):
                relevant_count += 1
                print(f"✅ {pdf_name} marked as RELEVANT")
            else:
                print(f"❌ {pdf_name} marked as NOT RELEVANT")
        
        print(f"\n{'='*80}")
        print("🎉 ENHANCED SELECTION COMPLETE")
        print(f"{'='*80}")
        print(f"📊 Total PDFs processed: {total_count}")
        print(f"✅ Relevant papers found: {relevant_count}")
        print(f"📈 Relevance rate: {(relevant_count/total_count)*100:.1f}%")
        print(f"📁 Relevant papers saved to: {self.relevant_dir}")
        print(f"🎯 Focus: Advanced AI/ML implementation projects")
        print(f"{'='*80}")


def main():
    """Main execution function"""
    try:
        selector = EnhancedPDFSelector()
        selector.process_all_pdfs()
        
    except Exception as e:
        print(f"❌ Error initializing enhanced PDF selector: {e}")


if __name__ == "__main__":
    main()