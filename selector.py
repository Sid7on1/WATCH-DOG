#!/usr/bin/env python3
"""
Intelligent PDF Text Selector using Google Gemini 2.0 Flash
Analyzes PDF chunks for relevance to self-evolution and project development
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

# Import enhanced modules
try:
    from selector_config import SelectorConfig
    from performance_analytics import PerformanceAnalytics
    ENHANCED_FEATURES = True
    print("🚀 Enhanced features loaded: Configuration Management + Performance Analytics")
except ImportError as e:
    print(f"⚠️ Enhanced features not available: {e}")
    print("📝 Using default configuration")
    ENHANCED_FEATURES = False
    
    # Fallback configuration class
    class SelectorConfig:
        CHUNK_DELAY = 15
        PDF_DELAY = 30
        SUB_CHUNK_DELAY = 5
        API_RETRY_DELAY = 10
        ERROR_RETRY_DELAY = 30
        MAX_CHUNK_SIZE = 25000
        MAX_TOKENS = 1000
        TEMPERATURE = 0.1
        MAX_RETRIES = 2
        PROGRESS_INTERVAL = 5

class IntelligentPDFSelector:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.pdf_txts_dir = self.artifacts_dir / "pdf-txts"
        self.relevant_dir = self.artifacts_dir / "relevant"
        
        # Initialize GitHub Repository Manager
        self.github_manager = GitHubRepositoryManager(artifacts_dir=artifacts_dir)
        
        # Multi-API configuration - SELECTOR OPTIMIZED (Fast & High Rate Limits)
        self.apis = {
            "openrouter": {
                "key": os.getenv("OPEN_API"),
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "models": [
                    # FAST & HIGH RATE LIMIT MODELS (for PDF selection)
                    "google/gemini-2.0-flash-exp:free",         # Fast + reliable
                    "mistralai/mistral-nemo:free"               # High rate limits
                ]
            },
            "gemini": {
                "key": os.getenv("gemini_API"),
                "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                "models": ["gemini-2.5-flash"]  # Main model for selector (rare conditions)
            },
            "groq": {
                "key": os.getenv("groq_API"),
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "models": [
                    # SELECTOR OPTIMIZED (from reference.txt)
                    "gemma2-9b-it",                            # Fast for selection
                    "meta-llama/llama-4-maverick-17b-128e-instruct"  # High rate limits
                ]
            },
            "cohere": {
                "key": os.getenv("cohere_API"),
                "url": "https://api.cohere.ai/v2/chat",  # Updated to V2
                "models": [
                    # CORRECT MODEL (from list_all_models.py)
                    "command-r7b-12-2024"                      # For selector
                ]
            },
            "huggingface": {
                "key": os.getenv("HF_API"),
                "url": "https://api-inference.huggingface.co/models/{model}/v1/chat/completions",
                "models": [
                    # SELECTOR OPTIMIZED (Fast models available via API)
                    "microsoft/DialoGPT-medium",               # Fast conversational model
                    "Qwen/Qwen2.5-Coder-32B-Instruct"         # Coding-focused model
                ]
            }
        }
        
        # Current API selection
        self.current_api = "openrouter"
        self.current_model_index = 0
        
        # Rate limit tracking
        self.rate_limit_count = 0
        self.max_rate_limit_retries = 3
        self.process_stopped = False
        
        # Ensure directories exist
        self.relevant_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.artifacts_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration settings
        if ENHANCED_FEATURES:
            SelectorConfig.load_custom_config()
            self.config = SelectorConfig
            print(f"⚙️ Configuration loaded: {SelectorConfig.get_config_summary()}")
        else:
            self.config = SelectorConfig
        
        # Context length settings (from config)
        self.max_chunk_size = self.config.MAX_CHUNK_SIZE
        
        # Initialize performance analytics
        if ENHANCED_FEATURES:
            self.analytics = PerformanceAnalytics(self.artifacts_dir / "analytics")
            print("📊 Performance analytics initialized")
        else:
            self.analytics = None
        
        # Use GitHub repository for seen PDFs tracking
        print(f"📚 Using GitHub repository for seen PDFs: {len(self.github_manager.seen_titles)} titles loaded")
        
        print("\n" + "="*80)
        print("🚀 INTELLIGENT PDF SELECTOR - MULTI-API SYSTEM")
        print("="*80)
        print(f"📁 Source Directory    : {self.pdf_txts_dir}")
        print(f"📁 Relevant Papers     : {self.relevant_dir}")
        print(f"📁 Temp Directory      : {self.temp_dir}")
        print(f"📚 Previously Seen PDFs: {len(self.github_manager.seen_titles)}")
        print("\n🔧 AVAILABLE MODELS (Optimized for Fast Selection):")
        self.display_available_models()
        print(f"🎯 Starting API: {self.current_api.upper()}")
        print("="*80)
    
    def reset_process_state(self):
        """Reset process state for fresh runs"""
        self.process_stopped = False
        self.rate_limit_count = 0
        self.current_api = "openrouter"
        self.current_model_index = 0
        print("🔄 Process state reset for fresh run")
    
    def cleanup_temp_files(self):
        """Clean up temporary files to prevent disk space issues"""
        try:
            import shutil
            if self.temp_dir.exists():
                # Count files before cleanup
                temp_files = list(self.temp_dir.rglob("*"))
                file_count = len([f for f in temp_files if f.is_file()])
                
                if file_count > 0:
                    shutil.rmtree(self.temp_dir)
                    self.temp_dir.mkdir(parents=True, exist_ok=True)
                    print(f"🧹 Cleaned up {file_count} temporary files")
                else:
                    print("🧹 No temporary files to clean")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean temp files: {e}")
    
    def update_seen_pdfs(self, pdf_name):
        """Add PDF to seen list using GitHub repository"""
        title = self.extract_title_from_pdf_name(pdf_name)
        if not self.github_manager.is_title_seen(title):
            self.github_manager.add_seen_titles([title])
            print(f"📝 Added to GitHub seen PDFs: {title}")
    
    def get_system_status(self):
        """Get current system status for monitoring"""
        return {
            "current_api": self.current_api,
            "current_model": self.apis[self.current_api]["models"][self.current_model_index],
            "process_stopped": self.process_stopped,
            "rate_limit_count": self.rate_limit_count,
            "seen_pdfs_count": len(self.seen_pdfs),
            "temp_files_exist": self.temp_dir.exists() and any(self.temp_dir.iterdir())
        }
    
    def display_available_models(self):
        """Display all available models in a clean, readable format"""
        print()
        for api_name, api_config in self.apis.items():
            if api_config["key"]:
                print(f"   🔹 {api_name.upper():<12} ✅ Available")
                for i, model in enumerate(api_config["models"], 1):
                    # Clean model name display
                    model_display = model.replace(":free", "").replace("/", " / ")
                    print(f"      {i}. {model_display}")
                print()
            else:
                print(f"   🔸 {api_name.upper():<12} ❌ No API Key")
                print()
    
    def get_current_api_config(self):
        """Get current API configuration"""
        api_config = self.apis[self.current_api]
        current_model = api_config["models"][self.current_model_index]
        return api_config, current_model
    
    def switch_to_next_api(self):
        """Switch to next available API/model combination"""
        # Try next model in current API first
        current_api_config = self.apis[self.current_api]
        if self.current_model_index < len(current_api_config["models"]) - 1:
            self.current_model_index += 1
            print(f"🔄 Switched to next model in {self.current_api}: {current_api_config['models'][self.current_model_index]}")
            return True
        
        # Try next API
        api_names = list(self.apis.keys())
        current_api_index = api_names.index(self.current_api)
        
        for i in range(1, len(api_names)):
            next_api_index = (current_api_index + i) % len(api_names)
            next_api = api_names[next_api_index]
            
            if self.apis[next_api]["key"]:  # Check if API key exists
                self.current_api = next_api
                self.current_model_index = 0
                print(f"🔄 Switched to API: {self.current_api} with model: {self.apis[next_api]['models'][0]}")
                return True
        
        print("❌ No more APIs available to switch to")
        return False
    
    def make_api_request(self, system_prompt, user_prompt, max_tokens=1000):
        """Make API request with current configuration"""
        api_config, current_model = self.get_current_api_config()
        
        if self.current_api == "gemini":
            return self.make_gemini_request(system_prompt, user_prompt, current_model, max_tokens)
        elif self.current_api == "cohere":
            return self.make_cohere_request(system_prompt, user_prompt, current_model, max_tokens)
        elif self.current_api == "huggingface":
            return self.make_huggingface_request(system_prompt, user_prompt, current_model, max_tokens)
        else:
            return self.make_openai_compatible_request(system_prompt, user_prompt, current_model, max_tokens)
    
    def make_openai_compatible_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to OpenAI-compatible APIs (OpenRouter, Groq)"""
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
    
    def make_gemini_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to Google Gemini API using the new genai client"""
        try:
            from google import genai
            
            # Initialize client (gets API key from environment)
            client = genai.Client()
            
            # Combine system and user prompts for Gemini
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = client.models.generate_content(
                model=model,
                contents=combined_prompt
            )
            
            return response.text
            
        except ImportError:
            # Fallback to REST API if genai library not available
            api_config, _ = self.get_current_api_config()
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            headers = {
                "Content-Type": "application/json"
            }
            
            # Combine system and user prompts for Gemini
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": combined_prompt}]
                }],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.1
                }
            }
            
            response = requests.post(f"{url}?key={api_config['key']}", headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
    
    def make_cohere_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to Cohere API using the new ClientV2"""
        try:
            from cohere import ClientV2
            
            # Initialize Cohere ClientV2
            api_config, _ = self.get_current_api_config()
            client = ClientV2(api_key=api_config['key'])
            
            # Use the new chat_stream method with proper message format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = client.chat_stream(
                model=model,
                messages=messages,
                temperature=0.1
            )
            
            # Collect the streamed response
            full_response = ""
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'message'):
                    if hasattr(chunk.delta.message, 'content'):
                        if hasattr(chunk.delta.message.content, 'text'):
                            full_response += chunk.delta.message.content.text
            
            return full_response
            
        except ImportError:
            # Fallback to REST API if cohere library not available
            api_config, _ = self.get_current_api_config()
            
            headers = {
                "Authorization": f"Bearer {api_config['key']}",
                "Content-Type": "application/json"
            }
            
            # Use V2 API format
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
            return result['message']['content'][0]['text']
    
    def make_huggingface_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request using HuggingFace Transformers library (LOCAL) with improved error handling"""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"🤖 Loading local HuggingFace model: {model}")
            
            # Combine system and user prompts
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Check if model requires special handling
            trust_remote = "moonshotai" in model or "nvidia" in model or "deepseek" in model
            
            # Try to create pipeline with better error handling
            try:
                pipe = pipeline(
                    "text-generation", 
                    model=model,
                    trust_remote_code=trust_remote,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                # Generate response
                messages = [{"role": "user", "content": combined_prompt}]
                response = pipe(
                    messages, 
                    max_new_tokens=max_tokens, 
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=pipe.tokenizer.eos_token_id
                )
                
                # Extract generated text
                if isinstance(response, list) and len(response) > 0:
                    generated = response[0].get('generated_text', '')
                    # Clean up the response to remove the input prompt
                    if isinstance(generated, list) and len(generated) > 1:
                        return generated[-1].get('content', str(generated))
                    return str(generated)
                else:
                    return str(response)
                    
            except Exception as model_error:
                print(f"⚠️ Model loading failed: {model_error}")
                # Try fallback approach
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote)
                    model_obj = AutoModelForCausalLM.from_pretrained(
                        model, 
                        trust_remote_code=trust_remote,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    
                    inputs = tokenizer(combined_prompt, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model_obj.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=0.1,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Remove input prompt from response
                    if combined_prompt in response:
                        response = response.replace(combined_prompt, "").strip()
                    
                    return response
                    
                except Exception as fallback_error:
                    print(f"❌ Fallback approach also failed: {fallback_error}")
                    raise model_error
                
        except ImportError as import_error:
            print(f"❌ Required libraries not available: {import_error}")
            print("💡 Install with: pip install transformers torch")
            return "Error: Required libraries (transformers, torch) not installed"
            
        except Exception as e:
            print(f"❌ Unexpected error with local HuggingFace model {model}: {e}")
            print(f"💡 Consider switching to API-based models")
            return f"Error: {str(e)}"

    def load_seen_pdfs(self):
        """Load previously seen PDF titles from file"""
        seen_pdfs = set()
        if self.seen_pdfs_file.exists():
            try:
                with open(self.seen_pdfs_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        title = line.strip()
                        if title:
                            seen_pdfs.add(title)
                print(f"📚 Loaded {len(seen_pdfs)} previously seen PDFs for duplicate checking")
            except Exception as e:
                print(f"Error loading seen PDFs: {e}")
        else:
            print("📚 No previous seen-pdfs.txt found - will check all PDFs")
        return seen_pdfs
    
    def extract_title_from_pdf_name(self, pdf_name):
        """Extract the original paper title from PDF directory name"""
        # PDF directory names are like: cs.AI_2507.12345v1_Paper-Title
        # We need to extract the title part and reconstruct it
        try:
            parts = pdf_name.split('_', 2)  # Split into domain, arxiv_id, title
            if len(parts) >= 3:
                title_part = parts[2]
                # Replace hyphens with spaces and clean up
                title = title_part.replace('-', ' ').strip()
                return title
            else:
                return pdf_name
        except Exception as e:
            print(f"Error extracting title from {pdf_name}: {e}")
            return pdf_name
    
    def is_pdf_seen(self, pdf_name):
        """Check if a PDF has been seen before based on its title"""
        title = self.extract_title_from_pdf_name(pdf_name)
        return title in self.seen_pdfs
    
    def split_large_chunk(self, text, pdf_name, chunk_num):
        """Split a large chunk into smaller sub-chunks if it exceeds context limits"""
        if len(text) <= self.max_chunk_size:
            return [text]
        
        print(f"📄 Chunk {chunk_num} is too large ({len(text)} chars) - splitting into sub-chunks")
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        sub_chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                sub_chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            sub_chunks.append(current_chunk.strip())
        
        # Save sub-chunks to temp files for reference
        temp_pdf_dir = self.temp_dir / pdf_name
        temp_pdf_dir.mkdir(parents=True, exist_ok=True)
        
        for i, sub_chunk in enumerate(sub_chunks, 1):
            temp_file = temp_pdf_dir / f"chunk_{chunk_num}_sub_{i}.txt"
            temp_file.write_text(sub_chunk, encoding='utf-8')
        
        print(f"📄 Split into {len(sub_chunks)} sub-chunks (saved to temp directory)")
        return sub_chunks
    
    def analyze_sub_chunks(self, sub_chunks, pdf_name, chunk_num, total_chunks):
        """Analyze multiple sub-chunks and combine results"""
        print(f"🔍 Analyzing {len(sub_chunks)} sub-chunks for chunk {chunk_num}")
        
        sub_analyses = []
        relevant_count = 0
        total_confidence = 0
        all_concepts = []
        all_reasons = []
        
        for i, sub_chunk in enumerate(sub_chunks, 1):
            print(f"  📋 Processing sub-chunk {i}/{len(sub_chunks)}...")
            
            # Add sub-chunk identifier to the text
            sub_chunk_text = f"[Sub-chunk {i}/{len(sub_chunks)} of original chunk {chunk_num}]\n\n{sub_chunk}"
            
            analysis = self.analyze_chunk_relevance_direct(sub_chunk_text, pdf_name, f"{chunk_num}.{i}", total_chunks)
            
            if self.process_stopped:
                return None
            
            if analysis:
                sub_analyses.append(analysis)
                if analysis.get('relevant', False):
                    relevant_count += 1
                total_confidence += analysis.get('confidence', 0)
                all_concepts.extend(analysis.get('key_concepts', []))
                all_reasons.append(f"Sub-chunk {i}: {analysis.get('reason', '')}")
                
                # Small delay between sub-chunks
                if i < len(sub_chunks):
                    time.sleep(5)
        
        if not sub_analyses:
            return None
        
        # Combine results from all sub-chunks
        avg_confidence = total_confidence / len(sub_analyses)
        is_relevant = relevant_count > 0  # If any sub-chunk is relevant, consider the whole chunk relevant
        
        # Determine implementation potential
        impl_potentials = [a.get('implementation_potential', 'low') for a in sub_analyses]
        if 'high' in impl_potentials:
            impl_potential = 'high'
        elif 'medium' in impl_potentials:
            impl_potential = 'medium'
        else:
            impl_potential = 'low'
        
        # Determine if we should continue reading
        continue_reading = any(a.get('continue_reading', False) for a in sub_analyses)
        
        # Check if unsure (only for chunk 1)
        is_unsure = chunk_num == 1 and avg_confidence < 0.7 and relevant_count == 0
        
        combined_analysis = {
            "relevant": is_relevant,
            "confidence": avg_confidence,
            "reason": f"Combined analysis of {len(sub_chunks)} sub-chunks: {'; '.join(all_reasons)}",
            "key_concepts": list(set(all_concepts)),  # Remove duplicates
            "implementation_potential": impl_potential,
            "continue_reading": continue_reading,
            "unsure": is_unsure,
            "sub_chunk_count": len(sub_chunks),
            "relevant_sub_chunks": relevant_count
        }
        
        print(f"📊 Combined analysis: {relevant_count}/{len(sub_chunks)} sub-chunks relevant")
        return combined_analysis
    
    def analyze_chunk_relevance_direct(self, chunk_text, pdf_name, chunk_num, total_chunks, max_retries=2):
        """Direct analysis without context length handling (used for sub-chunks)"""
        system_prompt = """You are an AI research assistant helping to identify papers relevant for self-evolution and strong project development.

GOAL: The user wants to self-evolve and develop strong, innovative projects from research papers.

FOCUS AREAS:
- AI/ML techniques that can be implemented in projects
- Novel algorithms and methodologies
- Practical applications and implementations
- System architectures and frameworks
- Performance optimization techniques
- Innovative approaches to problem-solving
- Tools and technologies for development
- Research with clear implementation potential

EVALUATION CRITERIA:
1. Does this paper contain implementable techniques or algorithms?
2. Can the concepts be applied to build strong projects?
3. Does it provide practical insights for development?
4. Is it relevant to AI/ML project development?
5. Does it offer innovative approaches worth exploring?

RESPONSE FORMAT:
Simply respond with:
RELEVANT or NOT_RELEVANT
Followed by a brief explanation of why.

NO JSON NEEDED - Just plain text response."""

        user_prompt = f"""Paper: {pdf_name}
Chunk: {chunk_num}/{total_chunks}

Content:
{chunk_text}

Analyze this chunk for relevance to self-evolution and strong project development."""

        for attempt in range(max_retries + 1):
            try:
                api_config, current_model = self.get_current_api_config()
                print(f"🔍 Using {self.current_api} with {current_model}")
                
                # Track API call start time for analytics
                api_start_time = time.time()
                
                analysis_text = self.make_api_request(system_prompt, user_prompt, self.config.MAX_TOKENS)
                
                # Log API performance
                api_response_time = time.time() - api_start_time
                if self.analytics:
                    self.analytics.log_api_call(self.current_api, current_model, api_response_time, True)
                
                # Simple text analysis - NO JSON PARSING NEEDED
                analysis_text = analysis_text.strip().lower()
                
                # Determine relevance from simple text response
                is_relevant = "relevant" in analysis_text and "not_relevant" not in analysis_text
                
                # Clean terminal output
                status = "RELEVANT" if is_relevant else "NOT RELEVANT"
                print(f"\n📊 ANALYSIS RESULT:")
                print(f"   Status: {status}")
                print(f"   Response: {analysis_text[:200]}...")
                print(f"   API Used: {self.current_api.upper()} - {current_model}")
                print("-" * 60)
                
                # Return simple boolean result
                return is_relevant
                    
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                
                # Handle errors and try switching APIs
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    print(f"⚠️ Rate limit hit on {self.current_api}")
                    if self.switch_to_next_api():
                        time.sleep(10)
                        continue
                    else:
                        self.process_stopped = True
                        return None
                        
                elif "503" in error_msg or "busy" in error_msg.lower():
                    print(f"⚠️ Model busy on {self.current_api}")
                    if self.switch_to_next_api():
                        time.sleep(10)
                        continue
                
                if attempt < max_retries:
                    time.sleep(15)
                    continue
                else:
                    return None
                    
            except Exception as e:
                print(f"❌ Error with {self.current_api}: {e}")
                if self.switch_to_next_api() and attempt < max_retries:
                    time.sleep(15)
                    continue
                else:
                    return None
        
        return None
    
    def analyze_chunk_relevance(self, chunk_text, pdf_name, chunk_num, total_chunks, max_retries=2):
        """Analyze if a chunk is relevant for self-evolution and project development"""
        
        system_prompt = """You are an AI research assistant helping to identify papers relevant for self-evolution and strong project development.

GOAL: The user wants to self-evolve and develop strong, innovative projects from research papers.

FOCUS AREAS:
- AI/ML techniques that can be implemented in projects
- Novel algorithms and methodologies
- Practical applications and implementations
- System architectures and frameworks
- Performance optimization techniques
- Innovative approaches to problem-solving
- Tools and technologies for development
- Research with clear implementation potential

EVALUATION CRITERIA:
1. Does this paper contain implementable techniques or algorithms?
2. Can the concepts be applied to build strong projects?
3. Does it provide practical insights for development?
4. Is it relevant to AI/ML project development?
5. Does it offer innovative approaches worth exploring?

RESPONSE FORMAT:
Simply respond with:
RELEVANT or NOT_RELEVANT
Followed by a brief explanation of why.

NO JSON NEEDED - Just plain text response."""

        user_prompt = f"""Paper: {pdf_name}
Chunk: {chunk_num}/{total_chunks}

Content:
{chunk_text}

Analyze this chunk for relevance to self-evolution and strong project development."""


        
        for attempt in range(max_retries + 1):
            try:
                api_config, current_model = self.get_current_api_config()
                
                # Clean terminal output for analysis
                print(f"\n🔍 ANALYZING CHUNK {chunk_num}/{total_chunks}")
                print(f"   📄 Paper: {pdf_name}")
                print(f"   🤖 Using: {self.current_api.upper()} - {current_model.replace(':free', '').replace('/', ' / ')}")
                if attempt > 0:
                    print(f"   🔄 Attempt: {attempt + 1}")
                print("   " + "-" * 50)
                
                analysis_text = self.make_api_request(system_prompt, user_prompt, 1000)
                
                # Simple text analysis - NO JSON PARSING NEEDED
                analysis_text_clean = analysis_text.strip().lower()
                
                # Determine relevance from simple text response
                is_relevant = "relevant" in analysis_text_clean and "not_relevant" not in analysis_text_clean
                
                # Clean terminal output
                status = "RELEVANT" if is_relevant else "NOT RELEVANT"
                print(f"\n📊 ANALYSIS RESULT:")
                print(f"   Status: {status}")
                print(f"   Response: {analysis_text[:200]}...")
                print(f"   API Used: {self.current_api.upper()} - {current_model.replace(':free', '').replace('/', ' / ')}")
                print("-" * 60)
                
                # Return simple result for compatibility with existing code
                return {
                    "relevant": is_relevant,
                    "confidence": 0.8 if is_relevant else 0.2,
                    "reason": analysis_text[:100] + "..." if len(analysis_text) > 100 else analysis_text,
                    "key_concepts": [],
                    "implementation_potential": "medium" if is_relevant else "low",
                    "continue_reading": is_relevant,
                    "unsure": False
                }
                    
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                print(f"API request error: {error_msg}")
                
                # Check for rate limit errors and other API errors
                if "429" in error_msg or "rate limit" in error_msg.lower() or "404" in error_msg or "503" in error_msg:
                    print(f"⚠️ API error on {self.current_api}: {error_msg[:100]}...")
                    if self.switch_to_next_api():
                        print(f"⏳ Waiting 10 seconds before retry with new API...")
                        time.sleep(10)
                        continue
                    else:
                        print("🛑 All APIs exhausted - STOPPING")
                        self.process_stopped = True
                        return None
                
                # Check for context length exceeded errors
                elif "context_length_exceeded" in error_msg.lower() or "token count exceeds" in error_msg.lower() or "413" in error_msg:
                    print(f"📏 Context length exceeded - chunking input and retrying")
                    
                    # Split the chunk into smaller sub-chunks
                    sub_chunks = self.split_large_chunk(chunk_text, pdf_name, chunk_num)
                    
                    if len(sub_chunks) > 1:
                        # Process sub-chunks and combine results
                        combined_analysis = self.analyze_sub_chunks(sub_chunks, pdf_name, chunk_num, total_chunks)
                        
                        if combined_analysis:
                            status = "UNSURE" if combined_analysis.get('unsure') else ("RELEVANT" if combined_analysis.get('relevant') else "NOT RELEVANT")
                            print(f"📊 Combined Analysis: {status} (Confidence: {combined_analysis.get('confidence', 0):.2f})")
                            print(f"📊 Reason: {combined_analysis.get('reason', 'No reason provided')}")
                            return combined_analysis
                        else:
                            print(f"❌ Failed to analyze sub-chunks")
                            return None
                    else:
                        print(f"❌ Cannot split chunk further - content too large")
                        return {
                            "relevant": False,
                            "confidence": 0.0,
                            "reason": "Content too large for context window",
                            "key_concepts": [],
                            "implementation_potential": "low",
                            "continue_reading": False,
                            "unsure": False
                        }
                
                # Check for model busy errors
                elif "503" in error_msg or "busy" in error_msg.lower() or "overloaded" in error_msg.lower():
                    print(f"⚠️ Model busy on {self.current_api}")
                    if self.switch_to_next_api():
                        print(f"⏳ Waiting 10 seconds before retry with new API...")
                        time.sleep(10)
                        continue
                    else:
                        print(f"❌ All APIs are busy or unavailable")
                        if attempt < max_retries:
                            print(f"⏳ Waiting 30 seconds before retry...")
                            time.sleep(30)
                            continue
                
                # Other request errors
                else:
                    if attempt < max_retries:
                        print(f"⏳ Retrying in 30 seconds... (Attempt {attempt + 2}/{max_retries + 1})")
                        time.sleep(30)
                        continue
                    else:
                        print(f"❌ Failed after {max_retries + 1} attempts")
                        return None
                        
            except Exception as e:
                print(f"Unexpected error analyzing chunk: {e}")
                
                if attempt < max_retries:
                    print(f"⏳ Retrying in 30 seconds...")
                    time.sleep(30)
                    continue
                else:
                    return None
        
        return None
    
    def move_relevant_paper(self, pdf_dir, analysis_results):
        """Move relevant paper to the relevant folder with analysis summary"""
        try:
            # Create destination directory
            dest_dir = self.relevant_dir / pdf_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all chunk files
            chunk_files = list(pdf_dir.glob("chunk_*.txt"))
            for chunk_file in chunk_files:
                dest_file = dest_dir / chunk_file.name
                dest_file.write_text(chunk_file.read_text(encoding='utf-8'), encoding='utf-8')
            
            # Create analysis summary
            summary_file = dest_dir / "analysis_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2)
            
            print(f"✅ Moved relevant paper: {pdf_dir.name}")
            return True
            
        except Exception as e:
            print(f"Error moving paper {pdf_dir.name}: {e}")
            return False
    
    def process_pdf_directory(self, pdf_dir):
        """Process all chunks in a PDF directory"""
        chunk_files = sorted(pdf_dir.glob("chunk_*.txt"))
        
        if not chunk_files:
            print(f"No chunks found in {pdf_dir.name}")
            return False
        
        pdf_name = pdf_dir.name
        total_chunks = len(chunk_files)
        analysis_results = []
        
        # Start PDF processing analytics
        if self.analytics:
            self.analytics.log_pdf_start(pdf_name)
        
        print(f"\n{'='*60}")
        print(f"Processing PDF: {pdf_name}")
        print(f"Total chunks: {total_chunks}")
        print(f"{'='*60}")
        
        # Always start with chunk 1
        first_chunk = chunk_files[0]
        chunk_text = first_chunk.read_text(encoding='utf-8')
        
        # Analyze first chunk
        analysis = self.analyze_chunk_relevance(chunk_text, pdf_name, 1, total_chunks)
        
        if self.process_stopped:
            print(f"🛑 Process stopped due to rate limits - aborting")
            return False
        
        if not analysis:
            print(f"❌ Failed to analyze {pdf_name} - skipping")
            return False
        
        analysis_results.append(analysis)
        
        # Handle different scenarios for chunk 1
        if analysis.get('unsure', False):
            print(f"🤔 Model is unsure about chunk 1 - reading chunk 2 to make decision")
            
            # Read chunk 2 if available to help make decision
            if len(chunk_files) > 1:
                print(f"\nWaiting 15 seconds before reading chunk 2...")
                time.sleep(15)
                
                second_chunk = chunk_files[1]
                chunk_text = second_chunk.read_text(encoding='utf-8')
                
                # Analyze second chunk
                second_analysis = self.analyze_chunk_relevance(chunk_text, pdf_name, 2, total_chunks)
                
                if self.process_stopped:
                    print(f"🛑 Process stopped due to rate limits - aborting")
                    return False
                
                if not second_analysis:
                    print(f"❌ Failed to analyze chunk 2 - skipping paper")
                    return False
                
                analysis_results.append(second_analysis)
                
                # Make decision based on both chunks
                first_relevant = analysis.get('relevant', False)
                second_relevant = second_analysis.get('relevant', False)
                
                if first_relevant or second_relevant:
                    print(f"✅ Decision made: Paper is relevant based on chunks 1-2")
                    # Continue with remaining chunks if any
                    start_chunk_index = 2  # Start from chunk 3
                else:
                    print(f"❌ Decision made: Paper is not relevant based on chunks 1-2")
                    return False
            else:
                print(f"❌ Only one chunk available and model is unsure - skipping paper")
                return False
                
        elif not analysis.get('relevant', False) or not analysis.get('continue_reading', False):
            print(f"❌ Skipping {pdf_name} - not relevant or no need to continue")
            return False
        else:
            print(f"✅ First chunk is relevant - continuing with remaining chunks")
            start_chunk_index = 1  # Start from chunk 2
        
        # Process remaining chunks starting from the appropriate index
        for i in range(start_chunk_index, len(chunk_files)):
            chunk_file = chunk_files[i]
            chunk_num = i + 1
            
            print(f"\nWaiting 15 seconds before next chunk...")
            time.sleep(15)  # 15 second gap between chunks
            
            chunk_text = chunk_file.read_text(encoding='utf-8')
            analysis = self.analyze_chunk_relevance(chunk_text, pdf_name, chunk_num, total_chunks)
            
            if self.process_stopped:
                print(f"🛑 Process stopped due to rate limits - aborting")
                break
            
            if not analysis:
                print(f"❌ Failed to analyze chunk {chunk_num} - stopping")
                break
            
            analysis_results.append(analysis)
            
            # If we shouldn't continue reading, stop
            if not analysis.get('continue_reading', False):
                print(f"🛑 Stopping at chunk {chunk_num} - no need to continue")
                break
        
        # If any chunk was relevant, move the paper
        relevant_chunks = [a for a in analysis_results if a.get('relevant', False)]
        if relevant_chunks:
            self.move_relevant_paper(pdf_dir, analysis_results)
            return True
        else:
            print(f"❌ No relevant chunks found in {pdf_name}")
            return False
    
    def process_all_pdfs(self):
        """Process all PDF text directories with enhanced management"""
        # Reset process state for fresh run
        self.reset_process_state()
        
        # Clean up temporary files from previous runs
        self.cleanup_temp_files()
        
        if not self.pdf_txts_dir.exists():
            print(f"PDF texts directory not found: {self.pdf_txts_dir}")
            return
        
        # Get all PDF directories
        pdf_dirs = [d for d in self.pdf_txts_dir.iterdir() if d.is_dir()]
        
        if not pdf_dirs:
            print("No PDF text directories found")
            return
        
        print(f"Found {len(pdf_dirs)} PDF directories to analyze")
        print("Starting intelligent paper selection...")
        
        # Display current system status
        status = self.get_system_status()
        print(f"🔧 System Status: API={status['current_api']}, Model={status['current_model']}")
        
        # Filter out already seen PDFs
        new_pdf_dirs = []
        seen_count = 0
        
        for pdf_dir in pdf_dirs:
            if self.is_pdf_seen(pdf_dir.name):
                print(f"⏭️  Skipping seen PDF: {pdf_dir.name}")
                seen_count += 1
            else:
                new_pdf_dirs.append(pdf_dir)
        
        print(f"📊 Found {len(pdf_dirs)} PDFs, {seen_count} already seen, {len(new_pdf_dirs)} new to process")
        
        if not new_pdf_dirs:
            print("🔄 No new PDFs to process - all have been seen before")
            # Clean up and return
            self.cleanup_temp_files()
            return
        
        relevant_count = 0
        total_processed = 0
        failed_count = 0
        
        try:
            for pdf_dir in new_pdf_dirs:
                if self.process_stopped:
                    print(f"🛑 Process stopped due to rate limits - stopping all PDF processing")
                    break
                    
                if total_processed > 0:
                    print(f"\nWaiting 30 seconds before next PDF...")
                    time.sleep(30)  # 30 second gap between PDFs
                
                try:
                    is_relevant = self.process_pdf_directory(pdf_dir)
                    
                    if is_relevant:
                        relevant_count += 1
                        # Update seen PDFs list for successful processing
                        self.update_seen_pdfs(pdf_dir.name)
                    else:
                        # Still mark as seen even if not relevant
                        self.update_seen_pdfs(pdf_dir.name)
                    
                    total_processed += 1
                    
                except Exception as e:
                    print(f"❌ Error processing {pdf_dir.name}: {e}")
                    failed_count += 1
                    # Still mark as seen to avoid reprocessing
                    self.update_seen_pdfs(pdf_dir.name)
                    continue
                
                # Display progress every 5 PDFs
                if total_processed % 5 == 0:
                    progress = (total_processed / len(new_pdf_dirs)) * 100
                    print(f"📈 Progress: {total_processed}/{len(new_pdf_dirs)} ({progress:.1f}%) - {relevant_count} relevant found")
        
        except KeyboardInterrupt:
            print(f"\n⚠️ Process interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error during processing: {e}")
        
        finally:
            # Final cleanup
            self.cleanup_temp_files()
            
            # Final summary
            print(f"\n{'='*80}")
            print("🎯 INTELLIGENT SELECTION SUMMARY")
            print(f"{'='*80}")
            print(f"📊 Total PDFs analyzed: {total_processed}")
            print(f"✅ Relevant papers found: {relevant_count}")
            print(f"❌ Failed to process: {failed_count}")
            if total_processed > 0:
                print(f"📈 Relevance rate: {(relevant_count/total_processed)*100:.1f}%")
            print(f"📁 Relevant papers saved to: {self.relevant_dir}")
            print(f"📝 Updated seen PDFs: {len(self.seen_pdfs)} total")
            
            # System status
            final_status = self.get_system_status()
            print(f"🔧 Final API: {final_status['current_api']} - {final_status['current_model']}")
            print(f"🛑 Process stopped: {final_status['process_stopped']}")
            print(f"{'='*80}")
    
    def run_with_monitoring(self):
        """Run the selector with enhanced monitoring and error recovery"""
        try:
            print("🚀 Starting PDF Selector with Enhanced Monitoring")
            start_time = time.time()
            
            self.process_all_pdfs()
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️ Total processing time: {duration:.2f} seconds ({duration/60:.1f} minutes)")
            
        except Exception as e:
            print(f"❌ Critical error in PDF selector: {e}")
            print("🔄 Attempting recovery...")
            
            # Try to reset and continue
            self.reset_process_state()
            self.cleanup_temp_files()
            
            print("💡 Recovery complete. You can try running again.")
        
        finally:
            # Ensure cleanup always happens
            self.cleanup_temp_files()


def main():
    """Main execution function with enhanced monitoring"""
    try:
        selector = IntelligentPDFSelector()
        
        # Use the enhanced monitoring method
        selector.run_with_monitoring()
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error initializing PDF selector: {e}")
        print("💡 Try checking your API keys and internet connection")


if __name__ == "__main__":
    main()