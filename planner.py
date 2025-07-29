#!/usr/bin/env python3
"""
Intelligent Project Structure Planner
Analyzes relevant papers and creates detailed project plans in JSON format
"""

import os
import requests
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class IntelligentProjectPlanner:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.relevant_dir = self.artifacts_dir / "relevant"
        self.structures_dir = self.artifacts_dir / "structures"
        
        # Enhanced API key validation and management
        self.validate_api_keys()
        
        # Rate limiting intelligence with exponential backoff
        self.rate_limits = {}
        self.api_performance = {}
        self.backoff_multiplier = 2
        self.max_backoff = 300  # 5 minutes max
        
        # Multi-API configuration - PLANNING & STRATEGY OPTIMIZED (from reference.txt)
        self.apis = {
            "openrouter": {
                "key": os.getenv("OPEN_API"),
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "models": [
                    # STRATEGY & ANALYSIS REASONING MODELS (for planner and manager)
                    "moonshotai/kimi-k2:free",                  # Best for planning
                    "google/gemini-2.0-flash-exp:free",         # Strong planning
                    "deepseek/deepseek-r1-0528:free"            # Strategic reasoning
                ]
            },
            "gemini": {
                "key": os.getenv("gemini_API"),
                "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                "models": ["gemini-2.5-flash"]  # Main model for planner (rare conditions)
            },
            "groq": {
                "key": os.getenv("groq_API"),
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "models": [
                    # PLANNER & MANAGER MODELS (from reference.txt)
                    "llama-3.3-70b-versatile",                # Best for planning
                    "qwen/qwen3-32b"                           # Technical planning
                ]
            },
            "cohere": {
                "key": os.getenv("cohere_API"),
                "url": "https://api.cohere.ai/v2/chat",  # Updated to V2
                "models": [
                    # PLANNER OPTIMIZED (rare conditions - from reference.txt)
                    "c4ai-aya-expanse-32b"                     # Best for planning (rare conditions)
                ]
            },
            "huggingface": {
                "key": os.getenv("HF_API"),
                "url": "https://api-inference.huggingface.co/models/{model}/v1/chat/completions",
                "models": [
                    # PLANNER & MANAGER MODELS (from reference.txt)
                    "deepseek-ai/DeepSeek-R1",                 # Strategic reasoning
                    "moonshotai/Kimi-K2-Instruct",             # Planning expertise
                    "zai-org/GLM-4.5"                          # Management tasks
                ]
            }
        }
        
        # Start with most reliable API (Groq has good free tier)
        self.current_api = "groq"
        self.current_model_index = 0
        
        # Set current API key and model for compatibility
        api_config, current_model = self.get_current_api_config()
        self.api_key = api_config["key"]
        self.current_model = current_model
        
        # Rate limit tracking
        self.rate_limit_count = 0
        self.max_rate_limit_retries = 3
        self.process_stopped = False
        
        # Ensure directories exist
        self.structures_dir.mkdir(parents=True, exist_ok=True)
        
        # Project templates for fallback
        self.project_templates = {
            "transformer": {
                "files": [
                    {"name": "main_transformer.py", "purpose": "Main transformer model implementation", "priority": "high"},
                    {"name": "training.py", "purpose": "Training pipeline and optimization", "priority": "high"},
                    {"name": "dataset_downloader.py", "purpose": "HuggingFace dataset integration", "priority": "medium"},
                    {"name": "attention_layers.py", "purpose": "Custom attention mechanisms", "priority": "high"},
                    {"name": "positional_encoding.py", "purpose": "Positional encoding implementations", "priority": "medium"},
                    {"name": "tokenizer_utils.py", "purpose": "Tokenization utilities", "priority": "medium"},
                    {"name": "config.py", "purpose": "Model and training configuration", "priority": "high"},
                    {"name": "utils.py", "purpose": "Utility functions and helpers", "priority": "low"},
                    {"name": "evaluation.py", "purpose": "Model evaluation and metrics", "priority": "medium"},
                    {"name": "requirements.txt", "purpose": "Python dependencies", "priority": "high"},
                    {"name": "README.md", "purpose": "Project documentation", "priority": "medium"},
                    {"name": "setup.py", "purpose": "Package installation setup", "priority": "low"}
                ],
                "folders": ["data", "models", "logs", "checkpoints", "tests", "configs"]
            },
            "agent": {
                "files": [
                    {"name": "main_agent.py", "purpose": "Main agent implementation", "priority": "high"},
                    {"name": "environment.py", "purpose": "Environment setup and interaction", "priority": "high"},
                    {"name": "training.py", "purpose": "Agent training pipeline", "priority": "high"},
                    {"name": "policy.py", "purpose": "Policy network implementation", "priority": "high"},
                    {"name": "memory.py", "purpose": "Experience replay and memory", "priority": "medium"},
                    {"name": "reward_system.py", "purpose": "Reward calculation and shaping", "priority": "medium"},
                    {"name": "multi_agent_comm.py", "purpose": "Multi-agent communication", "priority": "medium"},
                    {"name": "config.py", "purpose": "Agent and environment configuration", "priority": "high"},
                    {"name": "utils.py", "purpose": "Utility functions", "priority": "low"},
                    {"name": "evaluation.py", "purpose": "Agent evaluation metrics", "priority": "medium"},
                    {"name": "requirements.txt", "purpose": "Python dependencies", "priority": "high"},
                    {"name": "README.md", "purpose": "Project documentation", "priority": "medium"},
                    {"name": "setup.py", "purpose": "Package installation setup", "priority": "low"}
                ],
                "folders": ["envs", "models", "logs", "checkpoints", "tests", "data", "policies"]
            },
            "computer_vision": {
                "files": [
                    {"name": "main_model.py", "purpose": "Main computer vision model", "priority": "high"},
                    {"name": "training.py", "purpose": "Training pipeline", "priority": "high"},
                    {"name": "data_loader.py", "purpose": "Image data loading and batching", "priority": "high"},
                    {"name": "preprocessing.py", "purpose": "Image preprocessing utilities", "priority": "medium"},
                    {"name": "augmentation.py", "purpose": "Data augmentation techniques", "priority": "medium"},
                    {"name": "feature_extraction.py", "purpose": "Feature extraction layers", "priority": "medium"},
                    {"name": "loss_functions.py", "purpose": "Custom loss functions", "priority": "medium"},
                    {"name": "evaluation.py", "purpose": "Model evaluation and metrics", "priority": "medium"},
                    {"name": "config.py", "purpose": "Model configuration", "priority": "high"},
                    {"name": "utils.py", "purpose": "Utility functions", "priority": "low"},
                    {"name": "requirements.txt", "purpose": "Python dependencies", "priority": "high"},
                    {"name": "README.md", "purpose": "Project documentation", "priority": "medium"},
                    {"name": "setup.py", "purpose": "Package installation setup", "priority": "low"}
                ],
                "folders": ["data", "models", "logs", "checkpoints", "tests", "images", "results"]
            },
            "nlp": {
                "files": [
                    {"name": "main_model.py", "purpose": "Main NLP model implementation", "priority": "high"},
                    {"name": "training.py", "purpose": "Training pipeline", "priority": "high"},
                    {"name": "tokenizer.py", "purpose": "Text tokenization utilities", "priority": "high"},
                    {"name": "data_processor.py", "purpose": "Text data processing", "priority": "high"},
                    {"name": "embeddings.py", "purpose": "Word/sentence embeddings", "priority": "medium"},
                    {"name": "language_model.py", "purpose": "Language model components", "priority": "medium"},
                    {"name": "evaluation.py", "purpose": "NLP evaluation metrics", "priority": "medium"},
                    {"name": "inference.py", "purpose": "Model inference pipeline", "priority": "medium"},
                    {"name": "config.py", "purpose": "Model configuration", "priority": "high"},
                    {"name": "utils.py", "purpose": "Utility functions", "priority": "low"},
                    {"name": "requirements.txt", "purpose": "Python dependencies", "priority": "high"},
                    {"name": "README.md", "purpose": "Project documentation", "priority": "medium"},
                    {"name": "setup.py", "purpose": "Package installation setup", "priority": "low"}
                ],
                "folders": ["data", "models", "logs", "checkpoints", "tests", "datasets", "embeddings"]
            },
            "reinforcement_learning": {
                "files": [
                    {"name": "main_rl.py", "purpose": "Main RL algorithm implementation", "priority": "high"},
                    {"name": "agent.py", "purpose": "RL agent implementation", "priority": "high"},
                    {"name": "environment.py", "purpose": "Environment wrapper", "priority": "high"},
                    {"name": "training.py", "purpose": "RL training loop", "priority": "high"},
                    {"name": "replay_buffer.py", "purpose": "Experience replay buffer", "priority": "medium"},
                    {"name": "networks.py", "purpose": "Neural network architectures", "priority": "high"},
                    {"name": "policy_gradient.py", "purpose": "Policy gradient methods", "priority": "medium"},
                    {"name": "value_functions.py", "purpose": "Value function approximation", "priority": "medium"},
                    {"name": "config.py", "purpose": "RL configuration", "priority": "high"},
                    {"name": "utils.py", "purpose": "Utility functions", "priority": "low"},
                    {"name": "requirements.txt", "purpose": "Python dependencies", "priority": "high"},
                    {"name": "README.md", "purpose": "Project documentation", "priority": "medium"},
                    {"name": "setup.py", "purpose": "Package installation setup", "priority": "low"}
                ],
                "folders": ["envs", "models", "logs", "checkpoints", "tests", "data", "policies"]
            },
            "optimization": {
                "files": [
                    {"name": "main_optimizer.py", "purpose": "Main optimization algorithm", "priority": "high"},
                    {"name": "algorithms.py", "purpose": "Optimization algorithms implementation", "priority": "high"},
                    {"name": "objective_functions.py", "purpose": "Objective function definitions", "priority": "high"},
                    {"name": "constraints.py", "purpose": "Constraint handling", "priority": "medium"},
                    {"name": "genetic_algorithm.py", "purpose": "Genetic algorithm implementation", "priority": "medium"},
                    {"name": "gradient_methods.py", "purpose": "Gradient-based optimization", "priority": "medium"},
                    {"name": "visualization.py", "purpose": "Results visualization", "priority": "medium"},
                    {"name": "benchmarks.py", "purpose": "Benchmarking utilities", "priority": "low"},
                    {"name": "config.py", "purpose": "Optimization configuration", "priority": "high"},
                    {"name": "utils.py", "purpose": "Utility functions", "priority": "low"},
                    {"name": "requirements.txt", "purpose": "Python dependencies", "priority": "high"},
                    {"name": "README.md", "purpose": "Project documentation", "priority": "medium"},
                    {"name": "setup.py", "purpose": "Package installation setup", "priority": "low"}
                ],
                "folders": ["data", "results", "logs", "tests", "benchmarks", "configs"]
            }
        }
        
        print(f"Intelligent Project Planner initialized with multi-API support")
        print(f"Available APIs: {list(self.apis.keys())}")
        print(f"Valid API keys: {self.get_valid_api_count()}")
        print(f"Current API: {self.current_api}")
        print(f"Source directory: {self.relevant_dir}")
        print(f"Plans will be stored in: artifacts/structures/[paper_name]/plan.json")
    
    def validate_api_keys(self):
        """Validate all API keys and track which ones are available"""
        self.valid_apis = []
        for api_name, config in self.apis.items():
            if config.get("key"):
                self.valid_apis.append(api_name)
                print(f"✅ {api_name}: API key found")
            else:
                print(f"⚠️ {api_name}: Missing API key")
        
        if not self.valid_apis:
            print("❌ No valid API keys found! Please check your .env file")
            raise ValueError("No valid API keys available")
    
    def get_valid_api_count(self):
        """Get count of valid APIs"""
        return f"{len(self.valid_apis)}/{len(self.apis)}"
    
    def is_rate_limited(self, api_name):
        """Check if API is currently rate limited"""
        if api_name in self.rate_limits:
            return time.time() < self.rate_limits[api_name]
        return False
    
    def set_rate_limit(self, api_name, backoff_seconds=60):
        """Set rate limit backoff for an API"""
        backoff_time = min(backoff_seconds * self.backoff_multiplier, self.max_backoff)
        self.rate_limits[api_name] = time.time() + backoff_time
        print(f"⏰ Rate limit set for {api_name}: {backoff_time}s backoff")
    
    def track_api_performance(self, api_name, success=True, response_time=0):
        """Track API performance metrics"""
        if api_name not in self.api_performance:
            self.api_performance[api_name] = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "avg_response_time": 0,
                "total_response_time": 0
            }
        
        stats = self.api_performance[api_name]
        stats["requests"] += 1
        stats["total_response_time"] += response_time
        stats["avg_response_time"] = stats["total_response_time"] / stats["requests"]
        
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
    
    def get_best_available_api(self):
        """Get the best performing available API that's not rate limited"""
        available_apis = [api for api in self.valid_apis if not self.is_rate_limited(api)]
        
        if not available_apis:
            return None
        
        # Sort by success rate, then by response time
        def api_score(api_name):
            if api_name not in self.api_performance:
                return (1.0, 0)  # New APIs get priority
            
            stats = self.api_performance[api_name]
            if stats["requests"] == 0:
                return (1.0, 0)
            
            success_rate = stats["successes"] / stats["requests"]
            response_time = stats["avg_response_time"]
            return (success_rate, -response_time)  # Negative for ascending sort
        
        return max(available_apis, key=api_score)
    
    def create_enhanced_fallback_plan(self, paper_content, paper_name):
        """Create enhanced fallback plan with paper content analysis"""
        print("🔄 Creating enhanced fallback plan with content analysis")
        
        # Enhanced project type detection using paper content
        content_lower = paper_content.lower()
        paper_lower = paper_name.lower()
        
        # Score different project types based on content
        type_scores = {
            "transformer": 0,
            "agent": 0,
            "computer_vision": 0,
            "nlp": 0,
            "reinforcement_learning": 0,
            "optimization": 0
        }
        
        # Transformer indicators
        transformer_keywords = ["transformer", "attention", "bert", "gpt", "encoder", "decoder", "self-attention", "multi-head"]
        type_scores["transformer"] = sum(1 for kw in transformer_keywords if kw in content_lower)
        
        # Agent indicators
        agent_keywords = ["agent", "multi-agent", "autonomous", "policy", "environment", "action", "reward"]
        type_scores["agent"] = sum(1 for kw in agent_keywords if kw in content_lower)
        
        # Computer vision indicators
        cv_keywords = ["vision", "image", "cnn", "convolution", "visual", "detection", "segmentation", "classification"]
        type_scores["computer_vision"] = sum(1 for kw in cv_keywords if kw in content_lower)
        
        # NLP indicators
        nlp_keywords = ["nlp", "language", "text", "linguistic", "tokenization", "embedding", "corpus"]
        type_scores["nlp"] = sum(1 for kw in nlp_keywords if kw in content_lower)
        
        # RL indicators
        rl_keywords = ["reinforcement", "q-learning", "policy gradient", "value function", "markov", "episode"]
        type_scores["reinforcement_learning"] = sum(1 for kw in rl_keywords if kw in content_lower)
        
        # Optimization indicators
        opt_keywords = ["optimization", "minimize", "maximize", "objective", "constraint", "gradient", "algorithm"]
        type_scores["optimization"] = sum(1 for kw in opt_keywords if kw in content_lower)
        
        # Select best matching type
        best_type = max(type_scores.items(), key=lambda x: x[1])
        project_type = best_type[0] if best_type[1] > 0 else "optimization"
        
        # Extract key algorithms from content
        key_algorithms = []
        algorithm_patterns = [
            r"(\w+)\s+algorithm", r"(\w+)\s+method", r"(\w+)\s+approach",
            r"(\w+)\s+technique", r"(\w+)\s+model", r"(\w+)\s+framework"
        ]
        
        import re
        for pattern in algorithm_patterns:
            matches = re.findall(pattern, content_lower)
            key_algorithms.extend([match.title() for match in matches[:3]])  # Limit to 3 per pattern
        
        # Remove duplicates and common words
        common_words = {"the", "and", "for", "with", "this", "that", "from", "using"}
        key_algorithms = list(set([alg for alg in key_algorithms if alg.lower() not in common_words]))[:10]
        
        if not key_algorithms:
            key_algorithms = ["content_based_analysis"]
        
        # Get template and enhance it
        template = self.project_templates.get(project_type, self.project_templates["optimization"])
        
        enhanced_plan = {
            "project_type": project_type,
            "project_name": f"enhanced_{paper_name.replace(' ', '_').replace('-', '_')}",
            "description": f"Enhanced AI project based on {paper_name} with content analysis (confidence: {best_type[1]} matches)",
            "key_algorithms": key_algorithms,
            "main_libraries": template.get("main_libraries", ["torch", "numpy", "pandas"]),
            "data_sources": ["content_derived", "paper_specific"],
            "model_architecture": f"Content-analyzed architecture for {project_type} - requires implementation",
            "training_requirements": {
                "gpu_required": True,
                "estimated_training_time": "requires analysis based on content",
                "memory_requirements": "8GB+ - content-based estimate",
                "distributed_training": False
            },
            "implementation_priority": "high" if best_type[1] > 3 else "medium",
            "complexity_level": "advanced" if best_type[1] > 5 else "intermediate",
            "production_ready": False,
            "required_files": template["files"],
            "required_folders": template["folders"],
            "special_requirements": [
                f"Content analysis confidence: {best_type[1]} keyword matches",
                "Enhanced fallback with paper content analysis",
                "Manual verification recommended"
            ],
            "paper_specific_notes": f"Enhanced template fallback with content analysis. Detected {len(key_algorithms)} potential algorithms. Project type confidence: {best_type[1]} matches.",
            "evaluation_metrics": ["accuracy", "loss", "content_specific_metrics"],
            "deployment_considerations": ["GPU requirements", "memory optimization", "content-specific requirements"],
            "fallback_used": True,
            "enhancement_level": "content_analyzed"
        }
        
        print(f"📋 Enhanced Fallback Project Type: {project_type} (confidence: {best_type[1]})")
        print(f"📋 Detected Algorithms: {len(key_algorithms)}")
        
        return enhanced_plan

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
    
    def make_api_request(self, system_prompt, user_prompt, max_tokens=3000):
        """Make API request with current configuration"""
        api_config, current_model = self.get_current_api_config()
        
        if self.current_api == "gemini":
            return self.make_gemini_request(system_prompt, user_prompt, current_model, max_tokens)
        elif self.current_api == "huggingface":
            return self.make_huggingface_request(system_prompt, user_prompt, current_model, max_tokens)
        else:
            return self.make_openai_compatible_request(system_prompt, user_prompt, current_model, max_tokens)
    
    def make_openai_compatible_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to OpenAI-compatible APIs (OpenRouter, Moonshot)"""
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
        
        response = requests.post(api_config["url"], headers=headers, json=payload, timeout=180)
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
            
            response = requests.post(f"{url}?key={api_config['key']}", headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
    
    def make_huggingface_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to HuggingFace Inference API"""
        api_config, _ = self.get_current_api_config()
        
        url = api_config["url"].format(model=model)
        headers = {
            "Authorization": f"Bearer {api_config['key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def analyze_paper_and_create_plan(self, paper_content, paper_name, max_retries=2):
        """Analyze paper content and create detailed project plan"""
        
        system_prompt = """You are an expert software architect and AI researcher who creates detailed project structure plans from research papers.

TASK: Analyze the research paper and create a comprehensive project structure plan with detailed file specifications.

ANALYSIS AREAS:
1. Paper Type Classification (transformer, agent, computer_vision, nlp, reinforcement_learning, optimization, etc.)
2. Key Algorithms and Techniques
3. Required Dependencies and Libraries
4. Data Requirements and Sources
5. Model Architecture Details
6. Training Requirements
7. Evaluation Methods
8. Production Deployment Considerations
9. SPECIFIC FILES NEEDED with detailed descriptions

RESPONSE FORMAT:
Respond with ONLY a JSON object:
{
    "project_type": "transformer|agent|computer_vision|nlp|reinforcement_learning|optimization|other",
    "project_name": "descriptive_project_name_based_on_paper",
    "description": "Detailed project description based on paper content",
    "key_algorithms": ["algorithm1", "algorithm2", "algorithm3"],
    "main_libraries": ["torch", "transformers", "numpy", "specific_libs"],
    "data_sources": ["huggingface_dataset", "custom_data", "specific_datasets"],
    "model_architecture": "Detailed description of the model architecture from paper",
    "training_requirements": {
        "gpu_required": true/false,
        "estimated_training_time": "specific_time_estimate",
        "memory_requirements": "specific_GB_estimate",
        "distributed_training": true/false
    },
    "implementation_priority": "high|medium|low",
    "complexity_level": "beginner|intermediate|advanced",
    "production_ready": true/false,
    "required_files": [
        {
            "filename": "specific_file.py",
            "purpose": "Detailed description of what this file should contain",
            "priority": "high|medium|low",
            "dependencies": ["list", "of", "required", "libraries"],
            "key_functions": ["function1", "function2", "function3"],
            "estimated_lines": 100,
            "complexity": "low|medium|high"
        }
    ],
    "required_folders": ["folder1", "folder2", "folder3"],
    "special_requirements": ["requirement1", "requirement2"],
    "paper_specific_notes": "Important implementation notes from the paper",
    "evaluation_metrics": ["metric1", "metric2", "metric3"],
    "deployment_considerations": ["consideration1", "consideration2"]
}

IMPORTANT: 
- Be very specific about required_files - include detailed purpose and key_functions for each file
- Focus on the paper's specific algorithms and novel contributions
- Include paper-specific implementation details
- Estimate realistic file sizes and complexity levels
- List specific dependencies for each file"""

        user_prompt = f"""Paper: {paper_name}

Content:
{paper_content}

Analyze this research paper and create a comprehensive, detailed project structure plan for implementation."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.current_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 3000,
            "temperature": 0.1
        }
        
        for attempt in range(max_retries + 1):
            try:
                # Use intelligent API selection
                best_api = self.get_best_available_api()
                if best_api and best_api != self.current_api:
                    self.current_api = best_api
                    self.current_model_index = 0
                    api_config, current_model = self.get_current_api_config()
                    self.api_key = api_config["key"]
                    self.current_model = current_model
                    print(f"🎯 Using best available API: {self.current_api}")
                
                api_config, current_model = self.get_current_api_config()
                print(f"🔍 Analyzing paper and creating detailed plan for {paper_name} using {self.current_api} with {current_model}..." + 
                      (f" (Attempt {attempt + 1})" if attempt > 0 else ""))
                
                start_time = time.time()
                analysis_text = self.make_api_request(system_prompt, user_prompt, 3000)
                response_time = time.time() - start_time
                
                # Track successful API performance
                self.track_api_performance(self.current_api, success=True, response_time=response_time)
                
                # Try to parse JSON response
                try:
                    json_start = analysis_text.find('{')
                    json_end = analysis_text.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = analysis_text[json_start:json_end]
                        plan = json.loads(json_str)
                    else:
                        raise ValueError("No JSON found in response")
                    
                    print(f"📋 Project Type: {plan.get('project_type', 'unknown')}")
                    print(f"📋 Project Name: {plan.get('project_name', 'unknown')}")
                    print(f"📋 Complexity: {plan.get('complexity_level', 'unknown')}")
                    print(f"📋 Files to create: {len(plan.get('required_files', []))}")
                    
                    return plan
                    
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing JSON response: {e}")
                    if attempt < max_retries:
                        print(f"Retrying in 30 seconds...")
                        time.sleep(30)
                        continue
                    
                    # Use template fallback
                    print("🔄 Using template fallback for project plan")
                    return self.create_fallback_plan(paper_name)
                    
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                print(f"API request error: {error_msg}")
                
                # Track failed API performance
                self.track_api_performance(self.current_api, success=False)
                
                # Enhanced error handling with intelligent rate limiting
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    print(f"⚠️ Rate limit hit on {self.current_api}")
                    self.set_rate_limit(self.current_api, 120)  # 2 minute backoff
                    
                    # Try to get best available API
                    best_api = self.get_best_available_api()
                    if best_api:
                        self.current_api = best_api
                        self.current_model_index = 0
                        api_config, current_model = self.get_current_api_config()
                        self.api_key = api_config["key"]
                        self.current_model = current_model
                        print(f"🔄 Switched to best available API: {self.current_api}")
                        time.sleep(5)
                        continue
                    else:
                        print("🛑 All APIs rate limited - using enhanced fallback")
                        return self.create_enhanced_fallback_plan(paper_content, paper_name)
                
                elif "503" in error_msg or "busy" in error_msg.lower():
                    print(f"⚠️ Service busy on {self.current_api}")
                    self.set_rate_limit(self.current_api, 60)  # 1 minute backoff
                    
                    best_api = self.get_best_available_api()
                    if best_api:
                        self.current_api = best_api
                        self.current_model_index = 0
                        api_config, current_model = self.get_current_api_config()
                        self.api_key = api_config["key"]
                        self.current_model = current_model
                        print(f"🔄 Switched to best available API: {self.current_api}")
                        time.sleep(10)
                        continue
                
                if attempt < max_retries:
                    print(f"⏳ Retrying in 30 seconds...")
                    time.sleep(30)
                    continue
                else:
                    return self.create_enhanced_fallback_plan(paper_content, paper_name)
                    
            except Exception as e:
                print(f"Unexpected error: {e}")
                if attempt < max_retries:
                    time.sleep(30)
                    continue
                else:
                    return self.create_fallback_plan(paper_name)
        
        return self.create_enhanced_fallback_plan(paper_content, paper_name)
    
    def print_performance_summary(self):
        """Print detailed API performance summary"""
        print(f"\n{'='*60}")
        print("API PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        if not self.api_performance:
            print("No API performance data available")
            return
        
        for api_name, stats in self.api_performance.items():
            if stats["requests"] > 0:
                success_rate = (stats["successes"] / stats["requests"]) * 100
                print(f"\n🔧 {api_name.upper()}:")
                print(f"   Requests: {stats['requests']}")
                print(f"   Success Rate: {success_rate:.1f}%")
                print(f"   Avg Response Time: {stats['avg_response_time']:.2f}s")
                print(f"   Status: {'Rate Limited' if self.is_rate_limited(api_name) else 'Available'}")
        
        # Show best performing API
        best_api = self.get_best_available_api()
        if best_api:
            print(f"\n🏆 Best Available API: {best_api}")
        
        print(f"{'='*60}")
    
    def create_fallback_plan(self, paper_name):
        """Create a fallback plan using templates when AI analysis fails"""
        print("🔄 Creating fallback plan using template structure")
        
        # Try to guess project type from paper name
        paper_lower = paper_name.lower()
        project_type = "optimization"  # Default fallback
        
        if any(word in paper_lower for word in ["transformer", "bert", "gpt", "attention"]):
            project_type = "transformer"
        elif any(word in paper_lower for word in ["agent", "multi-agent", "autonomous"]):
            project_type = "agent"
        elif any(word in paper_lower for word in ["vision", "cnn", "image", "visual"]):
            project_type = "computer_vision"
        elif any(word in paper_lower for word in ["nlp", "language", "text", "linguistic"]):
            project_type = "nlp"
        elif any(word in paper_lower for word in ["reinforcement", "rl", "policy", "reward"]):
            project_type = "reinforcement_learning"
        
        # Get template structure
        template = self.project_templates.get(project_type, self.project_templates["optimization"])
        
        fallback_plan = {
            "project_type": project_type,
            "project_name": f"project_from_{paper_name.replace(' ', '_').replace('-', '_')}",
            "description": f"AI project based on {paper_name} (template fallback)",
            "key_algorithms": ["template_based"],
            "main_libraries": ["torch", "numpy", "pandas"],
            "data_sources": ["custom_data"],
            "model_architecture": "Template-based architecture - requires manual specification",
            "training_requirements": {
                "gpu_required": True,
                "estimated_training_time": "unknown - requires analysis",
                "memory_requirements": "8GB+ - requires analysis",
                "distributed_training": False
            },
            "implementation_priority": "medium",
            "complexity_level": "intermediate",
            "production_ready": False,
            "required_files": template["files"],
            "required_folders": template["folders"],
            "special_requirements": ["Manual implementation required", "Paper analysis needed"],
            "paper_specific_notes": "Template fallback used - requires manual paper analysis",
            "evaluation_metrics": ["accuracy", "loss"],
            "deployment_considerations": ["GPU requirements", "memory optimization"],
            "fallback_used": True
        }
        
        print(f"📋 Fallback Project Type: {project_type}")
        print(f"📋 Fallback Project Name: {fallback_plan['project_name']}")
        
        return fallback_plan
    
    def save_project_plan(self, plan, paper_name):
        """Save project plan as JSON in artifacts/structures/[paper_name]/plan.json"""
        try:
            # Create paper directory in structures
            paper_dir = self.structures_dir / paper_name
            paper_dir.mkdir(parents=True, exist_ok=True)
            
            # Save plan as JSON
            plan_file = paper_dir / "plan.json"
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved project plan: {plan_file}")
            print(f"📊 Plan contains {len(plan.get('required_files', []))} files and {len(plan.get('required_folders', []))} folders")
            
            return plan_file
            
        except Exception as e:
            print(f"❌ Error saving project plan for {paper_name}: {e}")
            return None
    
    def process_relevant_papers(self):
        """Process all relevant papers and create project plans"""
        if not self.relevant_dir.exists():
            print(f"Relevant papers directory not found: {self.relevant_dir}")
            return
        
        # Get all relevant paper directories
        paper_dirs = [d for d in self.relevant_dir.iterdir() if d.is_dir()]
        
        if not paper_dirs:
            print("No relevant papers found to process")
            return
        
        print(f"Found {len(paper_dirs)} relevant papers to process")
        print("Starting intelligent project planning...")
        
        successful_plans = 0
        total_processed = 0
        
        for paper_dir in paper_dirs:
            if self.process_stopped:
                print("🛑 Process stopped - aborting project planning")
                break
            
            if total_processed > 0:
                print(f"\nWaiting 30 seconds before next paper...")
                time.sleep(30)  # Rate limiting between papers
            
            print(f"\n{'='*80}")
            print(f"Processing paper: {paper_dir.name}")
            print(f"{'='*80}")
            
            # Read paper content from chunks
            paper_content = self.read_paper_content(paper_dir)
            
            if not paper_content:
                print(f"❌ Could not read content from {paper_dir.name}")
                total_processed += 1
                continue
            
            # Analyze paper and create plan
            project_plan = self.analyze_paper_and_create_plan(paper_content, paper_dir.name)
            
            if self.process_stopped:
                break
            
            if not project_plan:
                print(f"❌ Failed to generate plan for {paper_dir.name}")
                total_processed += 1
                continue
            
            # Save project plan
            plan_file = self.save_project_plan(project_plan, paper_dir.name)
            
            if plan_file:
                successful_plans += 1
                print(f"✅ Successfully created plan for {paper_dir.name}")
            
            total_processed += 1
        
        print(f"\n{'='*80}")
        print("PROJECT PLANNING SUMMARY")
        print(f"{'='*80}")
        print(f"Total papers processed: {total_processed}")
        print(f"Successful plans created: {successful_plans}")
        if total_processed > 0:
            print(f"Success rate: {(successful_plans/total_processed)*100:.1f}%")
        else:
            print(f"Success rate: 0.0% (no papers processed)")
        print(f"Plans saved to: artifacts/structures/[paper_name]/plan.json")
        print(f"{'='*80}")
        
        # Show detailed API performance summary
        self.print_performance_summary()
    
    def read_paper_content(self, paper_dir):
        """Read and combine all chunks from a paper directory"""
        chunk_files = sorted(paper_dir.glob("chunk_*.txt"))
        
        if not chunk_files:
            return None
        
        content = ""
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_content = f.read()
                    # Remove chunk headers
                    lines = chunk_content.split('\n')
                    content_start = 0
                    for i, line in enumerate(lines):
                        if line.startswith('='):
                            content_start = i + 1
                            break
                    content += '\n'.join(lines[content_start:]) + '\n\n'
            except Exception as e:
                print(f"Error reading {chunk_file}: {e}")
                continue
        
        return content.strip()


def main():
    """Main execution function"""
    try:
        planner = IntelligentProjectPlanner()
        planner.process_relevant_papers()
        
    except Exception as e:
        print(f"Error initializing project planner: {e}")


if __name__ == "__main__":
    main()