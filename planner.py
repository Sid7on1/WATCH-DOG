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

# Assuming pusher.py and GitHubRepositoryManager are available in your environment
# If not, you might need to mock it or provide its implementation.
try:
    from pusher import GitHubRepositoryManager
except ImportError:
    print("Warning: GitHubRepositoryManager not found. Please ensure 'pusher.py' is available.")
    # Define a dummy class to allow the code to run for testing the planner logic
    class GitHubRepositoryManager:
        def __init__(self, artifacts_dir):
            print(f"Dummy GitHubRepositoryManager initialized for {artifacts_dir}")
        def upload_to_github(self, repo_name, file_path, commit_message):
            print(f"Dummy GitHub upload: {file_path} to {repo_name} with message: {commit_message}")
        def create_repository(self, repo_name):
            print(f"Dummy GitHub create repo: {repo_name}")


# Load environment variables from .env file
load_dotenv()

def env(*names):
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return None

def clean_project_name(raw_name: str) -> str:
    """Clean and shorten project names to avoid terrible naming like in screenshots"""
    import re
    
    # Remove common prefixes that make names too long
    prefixes_to_remove = [
        "enhanced_", "cs", "cscl_", "csne_", "cscv_", "csro_", "cslg_",
        "arxiv_", "paper_", "project_from_"
    ]
    
    name = raw_name.lower()
    for prefix in prefixes_to_remove:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    # Remove arXiv IDs and version numbers (like 250821741v1)
    name = re.sub(r'\d{8,}v?\d*_?', '', name)
    
    # Extract key meaningful words (avoid clinical, medical, etc.)
    meaningful_words = []
    words = name.replace('_', ' ').replace('-', ' ').split()
    
    # Skip non-AI/ML words
    skip_words = {
        'clinical', 'medical', 'patient', 'hospital', 'healthcare', 'therapy',
        'survey', 'credibility', 'radiation', 'oncology', 'kalman', 'filter',
        'control', 'theory', 'soft', 'inducement', 'steering', 'tabular',
        'uncertainty', 'the', 'of', 'and', 'or', 'in', 'on', 'at', 'to',
        'for', 'with', 'by', 'from', 'are', 'is', 'not', 'all', 'parameters',
        'created', 'equal', 'smart', 'isolat', 'unveiling', 'role', 'data',
        'benchmarking', 'gpt', 'measurab', 'quantifying', 'limits', 'reasoning',
        'systematic', 'efficient', 'fine', 'tuning', 'pretrained', 'natu'
    }
    
    for word in words:
        if len(word) > 2 and word not in skip_words and len(meaningful_words) < 3:
            meaningful_words.append(word)
    
    if not meaningful_words:
        # Fallback to AI/ML generic names
        return "ai_ml_project"
    
    # Create clean name with max 3 words
    clean_name = '_'.join(meaningful_words[:3])
    
    # Ensure it's not too long (max 30 chars)
    if len(clean_name) > 30:
        clean_name = clean_name[:30].rstrip('_')
    
    return clean_name

class IntelligentProjectPlanner:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.relevant_dir = self.artifacts_dir / "relevant"
        self.structures_dir = self.artifacts_dir / "structures"
        
        # Multi-API configuration - Define self.apis first
        self.apis = {
            "openrouter": {
                "key": env("OPEN_API", "OPENROUTER_API_KEY", "OPENROUTER_API_TOKEN"),
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "models": [
                    # STRATEGY & ANALYSIS REASONING MODELS (for planner and manager)
                    "moonshotai/kimi-k2:free",                  # Best for planning
                    "google/gemini-2.0-flash-exp:free",         # Strong planning
                    "deepseek/deepseek-r1-0528:free"            # Strategic reasoning
                ]
            },
            "gemini": {
                "key": env("gemini_API", "GEMINI_API_KEY"),
                "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                "models": ["gemini-2.5-flash"]  # Main model for planner (rare conditions)
            },
            "groq": {
                "key": env("groq_API", "GROQ_API"),
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "models": [
                    # PLANNER & MANAGER MODELS (from reference.txt)
                    "llama-3.3-70b-versatile",                # Best for planning
                    "qwen/qwen3-32b"                           # Technical planning
                ]
            },
            "cohere": {
                "key": env("cohere_API", "COHERE_API", "COHERE_API_KEY"),
                "url": "https://api.cohere.ai/v2/chat",  # Updated to V2
                "models": [
                    # PLANNER OPTIMIZED (rare conditions - from reference.txt)
                    "c4ai-aya-expanse-32b"                     # Best for planning (rare conditions)
                ]
            },
            "huggingface": {
                "key": env("HF_API", "HUGGINGFACE_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
                "url": "https://api-inference.huggingface.co/models/{model}/v1/chat/completions",
                "models": [
                    # PLANNER & MANAGER MODELS (from reference.txt)
                    "deepseek-ai/DeepSeek-R1",                 # Strategic reasoning
                    "moonshotai/Kimi-K2-Instruct",             # Planning expertise
                    "zai-org/GLM-4.5"                          # Management tasks
                ]
            }
        }
        
        # Validate API keys AFTER self.apis is populated
        self.validate_api_keys()
        
        # Initialize GitHub Repository Manager
        self.github_manager = GitHubRepositoryManager(artifacts_dir=artifacts_dir)
        
        # Rate limiting intelligence with exponential backoff
        self.rate_limits = {}
        self.api_performance = {}
        self.backoff_multiplier = 2
        self.max_backoff = 300  # 5 minutes max
        
        # Set initial current_api and current_model_index *before* calling get_current_api_config
        self.current_api = "groq" # Start with most reliable API (Groq has good free tier)
        self.current_model_index = 0
        
        # Now, get_current_api_config can safely be called as self.apis and self.current_api are set
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
            # Raising an error here stops the program if no APIs are available
            # This is generally good practice if the core functionality depends on APIs.
            raise ValueError("No valid API keys available to initialize IntelligentProjectPlanner.")
    
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
                return (1.0, 0)  # New APIs get priority (1.0 success rate, 0 response time)
            
            stats = self.api_performance[api_name]
            if stats["requests"] == 0:
                return (1.0, 0)
            
            success_rate = stats["successes"] / stats["requests"]
            response_time = stats["avg_response_time"]
            return (success_rate, -response_time)  # Negative for ascending sort on response time
        
        return max(available_apis, key=api_score)
    
    def create_enhanced_fallback_plan(self, paper_content, paper_name):
        """Create enhanced fallback plan with paper content analysis"""
        print("🔄 Creating enhanced fallback plan with content analysis")
        
        # Enhanced project type detection using paper content
        content_lower = paper_content.lower()
        paper_lower = paper_name.lower() # Not directly used for scoring, but good for reference
        
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
        transformer_keywords = ["transformer", "attention", "bert", "gpt", "encoder", "decoder", "self-attention", "multi-head", "seq2seq"]
        type_scores["transformer"] += sum(1 for kw in transformer_keywords if kw in content_lower)
        
        # Agent indicators
        agent_keywords = ["agent", "multi-agent", "autonomous", "policy", "environment", "action", "reward", "reinforcement learning", "dqn", "ppo", "actor-critic"]
        type_scores["agent"] += sum(1 for kw in agent_keywords if kw in content_lower)
        # Add reinforcement learning keywords to agent, as agents often use RL
        type_scores["agent"] += sum(1 for kw in ["reinforcement", "q-learning", "policy gradient", "value function", "markov", "episode", "rl"] if kw in content_lower)

        # Computer vision indicators
        cv_keywords = ["vision", "image", "cnn", "convolution", "visual", "detection", "segmentation", "classification", "gan", "resnet", "yolo", "pixel", "object"]
        type_scores["computer_vision"] += sum(1 for kw in cv_keywords if kw in content_lower)
        
        # NLP indicators
        nlp_keywords = ["nlp", "language", "text", "linguistic", "tokenization", "embedding", "corpus", "sentiment", "translation", "summarization", "neural machine translation", "lm"]
        type_scores["nlp"] += sum(1 for kw in nlp_keywords if kw in content_lower)
        
        # RL indicators (some overlap with agent, but more specific to RL algorithms)
        rl_keywords = ["reinforcement", "q-learning", "policy gradient", "value function", "markov decision process", "mcts", "ppo", "sac", "dqn"]
        type_scores["reinforcement_learning"] += sum(1 for kw in rl_keywords if kw in content_lower)
        
        # Optimization indicators
        opt_keywords = ["optimization", "minimize", "maximize", "objective", "constraint", "gradient", "algorithm", "stochastic gradient descent", "adam", "bfgs", "simulated annealing"]
        type_scores["optimization"] += sum(1 for kw in opt_keywords if kw in content_lower)
        
        # Select best matching type
        # If all scores are 0, default to a general "optimization" or a balanced template
        best_type_item = max(type_scores.items(), key=lambda x: x[1])
        project_type = best_type_item[0] if best_type_item[1] > 0 else "optimization" # Default if no keywords matched
        
        # Extract key algorithms from content
        key_algorithms = []
        algorithm_patterns = [
            r"(\b\w+(?:-\w+)*)\s+(?:algorithm|method|approach|technique|model|framework|network)\b",
            r"(\b\w+)\s+(?:learning|prediction|generation|segmentation|detection)\b"
        ]
        
        import re
        for pattern in algorithm_patterns:
            matches = re.findall(pattern, content_lower)
            # Filter out short, generic words and capitalize
            key_algorithms.extend([match.replace('_', ' ').title() for match in matches if len(match) > 2])
        
        # Remove duplicates and common words (and specific non-algorithm words)
        common_words = {"the", "and", "for", "with", "this", "that", "from", "using", "based", "a", "an", "of", "in", "it", "to", "is", "our", "we", "can", "paper", "novel", "new", "proposed", "model", "method", "approach", "framework", "system", "neural", "deep"}
        key_algorithms = list(set([alg for alg in key_algorithms if alg.lower() not in common_words and alg.lower() not in project_type]))[:10] # Limit and remove words matching project type
        
        if not key_algorithms:
            key_algorithms = ["Content-Based Analysis Algorithm"] # More descriptive fallback
        
        # Get template and enhance it
        template = self.project_templates.get(project_type, self.project_templates["optimization"])
        
        enhanced_plan = {
            "project_type": project_type,
            "project_name": clean_project_name(paper_name),
            "description": f"Enhanced AI project based on {paper_name} with content analysis. Detected project type: {project_type.replace('_', ' ')} (confidence score: {best_type_item[1]} matches).",
            "key_algorithms": key_algorithms,
            "main_libraries": template.get("main_libraries", ["torch", "numpy", "pandas"]), # Fallback to common libs
            "data_sources": ["content_derived", "paper_specific_datasets_to_be_identified"],
            "model_architecture": f"Content-analyzed architecture for {project_type.replace('_', ' ')} - requires detailed implementation from paper.",
            "training_requirements": {
                "gpu_required": True, # Most AI papers imply GPU
                "estimated_training_time": "requires in-depth analysis based on content and model scale",
                "memory_requirements": "16GB+ (initial estimate, content-based estimate)", # Increased general estimate
                "distributed_training": False # Default to False, can be updated manually
            },
            "implementation_priority": "high" if best_type_item[1] > 3 else "medium",
            "complexity_level": "advanced" if best_type_item[1] > 5 else "intermediate",
            "production_ready": False,
            "required_files": template["files"],
            "required_folders": template["folders"],
            "special_requirements": [
                f"Content analysis confidence: {best_type_item[1]} keyword matches for {project_type.replace('_', ' ')}.",
                "Enhanced fallback with automated content analysis.",
                "Manual verification of structure and further detail extraction from paper is highly recommended."
            ],
            "paper_specific_notes": f"Generated from enhanced template fallback using content analysis. Detected {len(key_algorithms)} potential key algorithms. Primary project type identified as '{project_type.replace('_', ' ')}'.",
            "evaluation_metrics": ["accuracy", "loss", "F1-score", "content_specific_metrics_to_be_identified"], # Added common metrics
            "deployment_considerations": ["GPU requirements", "memory optimization", "scalability", "latency for inference", "content-specific deployment needs"],
            "fallback_used": True,
            "enhancement_level": "content_analyzed"
        }
        
        print(f"📋 Enhanced Fallback Project Type: {project_type} (confidence: {best_type_item[1]})")
        print(f"📋 Detected Algorithms: {', '.join(key_algorithms) if key_algorithms else 'None'}")
        
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
            # Update self.api_key and self.current_model
            self.api_key = current_api_config["key"]
            self.current_model = current_api_config["models"][self.current_model_index]
            return True
        
        # Try next API
        api_names = list(self.apis.keys())
        current_api_index = api_names.index(self.current_api)
        
        # Iterate through available APIs starting from the next one
        for i in range(1, len(api_names)):
            next_api_index = (current_api_index + i) % len(api_names)
            next_api = api_names[next_api_index]
            
            # Check if API has a key and is not rate-limited
            if self.apis[next_api]["key"] and not self.is_rate_limited(next_api):
                self.current_api = next_api
                self.current_model_index = 0 # Reset model index for new API
                api_config, current_model = self.get_current_api_config()
                self.api_key = api_config["key"]
                self.current_model = current_model
                print(f"🔄 Switched to API: {self.current_api} with model: {self.apis[next_api]['models'][0]}")
                return True
        
        print("❌ No more available APIs to switch to (all exhausted or rate-limited)")
        return False
    
    def make_api_request(self, system_prompt, user_prompt, max_tokens=3000):
        """Make API request with current configuration"""
        api_config, current_model = self.get_current_api_config()
        
        if self.current_api == "gemini":
            return self.make_gemini_request(system_prompt, user_prompt, current_model, max_tokens)
        elif self.current_api == "huggingface":
            return self.make_huggingface_request(system_prompt, user_prompt, current_model, max_tokens)
        elif self.current_api == "cohere": # Cohere has a different structure
            return self.make_cohere_request(system_prompt, user_prompt, current_model, max_tokens)
        else: # OpenRouter, Groq are OpenAI-compatible
            return self.make_openai_compatible_request(system_prompt, user_prompt, current_model, max_tokens)
    
    def make_openai_compatible_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to OpenAI-compatible APIs (OpenRouter, Groq)"""
        api_config, _ = self.get_current_api_config() # Use actual config for current API
        
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
            "temperature": 0.1,
            "response_format": {"type": "json_object"} # Request JSON output if supported
        }
        
        response = requests.post(api_config["url"], headers=headers, json=payload, timeout=180)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def make_gemini_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to Google Gemini API using the new genai client (preferred) or REST API (fallback)"""
        try:
            from google import generativeai as genai
            
            # Initialize client (gets API key from environment, or from genai.configure)
            genai.configure(api_key=os.getenv("gemini_API")) # Ensure API key is configured
            
            # For Gemini, system instructions are often integrated into the first user turn or model config.
            # A common pattern is to set it as part of the prompt.
            model_instance = genai.GenerativeModel(model_name=model)
            
            # Combine system and user prompts for Gemini to guide its response
            # Gemini models often perform better when system instructions are part of the initial turn.
            prompt_parts = [
                {"text": system_prompt},
                {"text": user_prompt}
            ]
            
            response = model_instance.generate_content(
                prompt_parts,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1,
                    response_mime_type="application/json" # Request JSON output
                )
            )
            
            return response.text
            
        except ImportError:
            print("Warning: Google Generative AI library not found. Falling back to REST API for Gemini.")
            api_config, _ = self.get_current_api_config()
            
            url = api_config["url"].format(model=model) # Corrected URL format
            headers = {
                "Content-Type": "application/json"
            }
            
            # Combine system and user prompts for Gemini REST API
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": combined_prompt}]
                }],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.1,
                    "responseMimeType": "application/json" # Request JSON output
                }
            }
            
            response = requests.post(f"{url}?key={api_config['key']}", headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            # Gemini's REST API response structure for content
            if 'candidates' in result and result['candidates']:
                if 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
                    for part in result['candidates'][0]['content']['parts']:
                        if 'text' in part:
                            return part['text']
            raise ValueError("Unexpected Gemini REST API response format")
    
    def make_cohere_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to Cohere API (v2)"""
        api_config, _ = self.get_current_api_config()
        
        headers = {
            "Authorization": f"Bearer {api_config['key']}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Cohere V2 chat endpoint uses a list of messages. System prompt is typically role="system".
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": model,
            "chat_history": [], # You might want to add chat history here for conversational use
            "message": user_prompt, # Cohere's message for the current turn
            "preamble": system_prompt, # Cohere's way to provide system instruction
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "response_format": {"type": "json_object"} # Request JSON output
        }
        
        # Note: Cohere's 'message' field is for the *current* turn.
        # 'preamble' is typically for persistent system instructions.
        # If the system_prompt is meant for the *entire* context, 'preamble' is better.
        # If it's specific to this one request, keeping it in the 'message' or combining is fine.
        # Given your system prompt is detailed instructions for this specific task, preamble is good.

        response = requests.post(api_config["url"], headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        
        result = response.json()
        if 'text' in result:
            return result['text']
        elif 'message' in result and 'text' in result['message']:
            return result['message']['text']
        else:
            raise ValueError("Unexpected Cohere API response format")
    
    def make_huggingface_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to HuggingFace Inference API"""
        api_config, _ = self.get_current_api_config()
        
        # HuggingFace Inference API for chat completions might have different model paths
        # depending on if it's a specific model endpoint or a generic chat completions.
        # Your current URL format `https://api-inference.huggingface.co/models/{model}/v1/chat/completions`
        # suggests a specific model endpoint which may not fully support chat completions.
        # The standard HF inference API for LLMs often takes a simpler prompt.
        # For chat completion style, it usually expects messages array.
        
        # Let's assume the provided URL is for models that accept chat completions format.
        url = api_config["url"].format(model=model.replace("/", "--")) # Models with '/' in name might need conversion
        headers = {
            "Authorization": f"Bearer {api_config['key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_new_tokens": max_tokens, # HF often uses max_new_tokens
            "temperature": 0.1,
            "return_full_text": False # Only return generated text
            # Add other HF specific params like "do_sample", "top_p" if needed
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        
        result = response.json()
        # HuggingFace API responses can vary greatly by model.
        # This assumes a structure similar to OpenAI chat completions.
        if isinstance(result, list) and result:
            if 'generated_text' in result[0]: # Common for text generation tasks
                return result[0]['generated_text']
            elif 'choices' in result[0] and result[0]['choices']: # For chat completions
                return result[0]['choices'][0]['message']['content']
        elif 'choices' in result and result['choices']:
            return result['choices'][0]['message']['content']
        
        raise ValueError(f"Unexpected HuggingFace API response format for model {model}: {result}")
    
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
9. SPECIFIC FILES NEEDED with detailed descriptions, estimated lines of code, and key functions within each file.

RESPONSE FORMAT:
Respond with ONLY a JSON object that adheres strictly to the following schema:
{
    "project_type": "transformer|agent|computer_vision|nlp|reinforcement_learning|optimization|other",
    "project_name": "clean_short_name_max_3_words_e.g._alphazero_chess",
    "description": "Detailed project description based on paper content, summarizing its core idea and goals.",
    "key_algorithms": ["algorithm1_e.g._Monte_Carlo_Tree_Search", "algorithm2_e.g._ResNet"],
    "main_libraries": ["torch", "transformers", "numpy", "pandas", "specific_libs_like_scikit_learn"],
    "data_sources": ["huggingface_dataset_name", "custom_data_description", "specific_datasets_like_ImageNet"],
    "model_architecture": "Detailed description of the model architecture from paper, including layers, components, and flow.",
    "training_requirements": {
        "gpu_required": true,
        "estimated_training_time": "e.g._2_days_on_A100_GPU",
        "memory_requirements": "e.g._32GB_GPU_VRAM,_128GB_RAM",
        "distributed_training": false
    },
    "implementation_priority": "high|medium|low",
    "complexity_level": "beginner|intermediate|advanced",
    "production_ready": false,
    "required_files": [
        {
            "filename": "main_inference.py",
            "purpose": "Entry point for model inference and application logic.",
            "priority": "high",
            "dependencies": ["torch", "transformers"],
            "key_functions": ["load_model", "predict", "run_application"],
            "estimated_lines": 250,
            "complexity": "medium"
        },
        {
            "filename": "model_architecture.py",
            "purpose": "Defines the neural network architecture described in the paper.",
            "priority": "high",
            "dependencies": ["torch.nn"],
            "key_functions": ["__init__", "forward", "build_transformer_block"],
            "estimated_lines": 400,
            "complexity": "high"
        }
        // ... more files with detailed specifics
    ],
    "required_folders": ["data", "models", "logs", "checkpoints", "tests", "scripts", "config"],
    "special_requirements": ["custom_cuda_kernel_needed", "specific_hardware_required"],
    "paper_specific_notes": "Important implementation details, edge cases, or novel insights from the paper that must be considered.",
    "evaluation_metrics": ["accuracy", "f1_score", "rouge_l", "bleu_score"],
    "deployment_considerations": ["containerization_docker", "api_endpoints", "scalability_kubernetes", "inference_optimization_onnx"]
}

IMPORTANT: 
- Be very specific about `required_files` - include detailed `purpose`, `key_functions`, `estimated_lines`, and `complexity` for EACH file.
- Focus on the paper's specific algorithms and novel contributions.
- Include paper-specific implementation details.
- Estimate realistic file sizes and complexity levels for each component.
- List specific dependencies for each file based on its purpose.
- If the paper presents a novel dataset or data processing, reflect that in `data_sources` and related files.
- Ensure the JSON is valid and complete. Do NOT include any text before or after the JSON."""

        user_prompt = f"""Paper: {paper_name}

Content:
{paper_content}

Analyze this research paper and create a comprehensive, detailed project structure plan for implementation strictly in the specified JSON format."""

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
                    print(f"🎯 Switched to best available API: {self.current_api} with {self.current_model}")
                
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
                    # Robust JSON parsing: find the first { and last }
                    json_start = analysis_text.find('{')
                    json_end = analysis_text.rfind('}') + 1
                    
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_str = analysis_text[json_start:json_end]
                        plan = json.loads(json_str)
                    else:
                        raise ValueError("No valid JSON object found in response string.")
                    
                    # Basic validation of the plan structure
                    if not all(k in plan for k in ["project_type", "project_name", "required_files", "required_folders"]):
                        raise ValueError("Parsed JSON is missing essential keys.")
                    
                    print(f"📋 Project Type: {plan.get('project_type', 'unknown')}")
                    print(f"📋 Project Name: {plan.get('project_name', 'unknown')}")
                    print(f"📋 Complexity: {plan.get('complexity_level', 'unknown')}")
                    print(f"📋 Files to create: {len(plan.get('required_files', []))}")
                    
                    return plan
                    
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing or validating JSON response: {e}")
                    if attempt < max_retries:
                        print(f"Retrying in 30 seconds due to JSON error...")
                        time.sleep(30)
                        continue
                    
                    # If JSON parsing fails after retries, use enhanced fallback
                    print("🔄 Using enhanced fallback plan due to persistent JSON parsing errors.")
                    return self.create_enhanced_fallback_plan(paper_content, paper_name)
                    
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                print(f"API request error: {error_msg}")
                
                # Track failed API performance
                self.track_api_performance(self.current_api, success=False)
                
                # Enhanced error handling with intelligent rate limiting and API switching
                if "429" in error_msg or "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                    print(f"⚠️ Rate limit hit or quota exceeded on {self.current_api}")
                    self.set_rate_limit(self.current_api, 120)  # 2 minute backoff
                    
                    if self.switch_to_next_api():
                        print(f"Switched to next API. Retrying immediately...")
                        time.sleep(5) # Small pause before immediate retry on new API
                        continue # Re-enter loop to retry with new API
                    else:
                        print("🛑 All APIs exhausted or rate limited - falling back to enhanced plan.")
                        return self.create_enhanced_fallback_plan(paper_content, paper_name)
                
                elif "500" in error_msg or "502" in error_msg or "503" in error_msg or "busy" in error_msg.lower():
                    print(f"⚠️ Server error/busy on {self.current_api}. Trying next API if available.")
                    self.set_rate_limit(self.current_api, 60) # Short backoff for transient server issues
                    
                    if self.switch_to_next_api():
                        print(f"Switched to next API. Retrying immediately...")
                        time.sleep(10) # Longer pause for server errors
                        continue
                    else:
                        print("🛑 All APIs exhausted or unavailable - falling back to enhanced plan.")
                        return self.create_enhanced_fallback_plan(paper_content, paper_name)
                
                # For other request errors, or if switching didn't help after hitting limit/server error
                if attempt < max_retries:
                    print(f"⏳ Retrying in 30 seconds due to network/other API error...")
                    time.sleep(30)
                    continue
                else:
                    print("❌ Max retries reached for API requests. Falling back to enhanced plan.")
                    return self.create_enhanced_fallback_plan(paper_content, paper_name)
                    
            except Exception as e:
                print(f"Unexpected error during API request or processing: {e}")
                if attempt < max_retries:
                    print(f"⏳ Retrying in 30 seconds due to unexpected error...")
                    time.sleep(30)
                    continue
                else:
                    print("❌ Max retries reached for unexpected errors. Falling back to enhanced plan.")
                    return self.create_enhanced_fallback_plan(paper_content, paper_name)
        
        # This line should ideally be unreachable if max_retries handles all cases,
        # but as a final safeguard:
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
                print(f"   Successes: {stats['successes']}")
                print(f"   Failures: {stats['failures']}")
                print(f"   Success Rate: {success_rate:.1f}%")
                print(f"   Avg Response Time: {stats['avg_response_time']:.2f}s")
                print(f"   Status: {'Rate Limited' if self.is_rate_limited(api_name) else 'Available'}")
        
        # Show best performing API
        best_api = self.get_best_available_api()
        if best_api:
            print(f"\n🏆 Best Currently Available API (based on performance): {best_api}")
        
        print(f"{'='*60}")
    
    def create_fallback_plan(self, paper_name):
        """Create a basic fallback plan using templates when AI analysis fails"""
        # This is a simpler fallback than create_enhanced_fallback_plan,
        # mainly for cases where even content analysis isn't viable/desired.
        # Given your code structure, create_enhanced_fallback_plan is usually preferred.
        print("🔄 Creating basic fallback plan using template structure (less detailed than enhanced fallback).")
        
        # Try to guess project type from paper name (simpler heuristics)
        paper_lower = paper_name.lower()
        project_type = "optimization"  # Default fallback
        
        if any(word in paper_lower for word in ["transformer", "bert", "gpt"]):
            project_type = "transformer"
        elif any(word in paper_lower for word in ["agent", "rl", "reinforcement"]):
            project_type = "agent" # Group RL into agent for this basic fallback
        elif any(word in paper_lower for word in ["vision", "image", "cnn"]):
            project_type = "computer_vision"
        elif any(word in paper_lower for word in ["nlp", "language", "text"]):
            project_type = "nlp"
        
        # Get template structure
        template = self.project_templates.get(project_type, self.project_templates["optimization"])
        
        fallback_plan = {
            "project_type": project_type,
            "project_name": clean_project_name(paper_name),
            "description": f"AI project based on {paper_name} (Basic Template Fallback: Manual analysis needed).",
            "key_algorithms": ["template_based_placeholder"],
            "main_libraries": ["torch", "numpy", "pandas"],
            "data_sources": ["manual_data_definition_required"],
            "model_architecture": "Template-based architecture - requires manual specification from paper.",
            "training_requirements": {
                "gpu_required": True,
                "estimated_training_time": "unknown - requires manual analysis",
                "memory_requirements": "unknown - requires manual analysis",
                "distributed_training": False
            },
            "implementation_priority": "medium",
            "complexity_level": "intermediate",
            "production_ready": False,
            "required_files": template["files"],
            "required_folders": template["folders"],
            "special_requirements": ["Manual implementation required", "Thorough paper analysis needed"],
            "paper_specific_notes": "Basic template fallback used due to AI analysis failure - critical to manually review paper and populate details.",
            "evaluation_metrics": ["accuracy", "loss"],
            "deployment_considerations": ["basic_deployment_setup"],
            "fallback_used": True,
            "enhancement_level": "basic_template"
        }
        
        print(f"📋 Basic Fallback Project Type: {project_type}")
        print(f"📋 Basic Fallback Project Name: {fallback_plan['project_name']}")
        
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
        
        # Get all relevant paper directories (assuming each is a sub-directory in relevant_dir)
        paper_dirs = [d for d in self.relevant_dir.iterdir() if d.is_dir()]
        
        if not paper_dirs:
            print("No relevant papers found to process in the 'relevant' directory.")
            return
        
        print(f"Found {len(paper_dirs)} relevant papers to process.")
        print("Starting intelligent project planning...")
        
        successful_plans = 0
        total_processed = 0
        
        for paper_dir in paper_dirs:
            if self.process_stopped:
                print("🛑 Process stopped - aborting project planning.")
                break
            
            # Implement a global rate limit/cool-down between processing papers to be courteous to APIs
            if total_processed > 0:
                print(f"\nWaiting 30 seconds before processing next paper...")
                time.sleep(30)
            
            print(f"\n{'='*80}")
            print(f"Processing paper: {paper_dir.name}")
            print(f"{'='*80}")
            
            # Read paper content from chunks
            paper_content = self.read_paper_content(paper_dir)
            
            if not paper_content:
                print(f"❌ Could not read content from any chunk files in {paper_dir.name}. Skipping this paper.")
                total_processed += 1
                continue
            
            # Analyze paper and create plan
            project_plan = self.analyze_paper_and_create_plan(paper_content, paper_dir.name)
            
            if self.process_stopped: # Check again after plan generation, in case of interruption during that phase
                break
            
            if not project_plan:
                print(f"❌ Failed to generate a valid plan for {paper_dir.name} after all attempts and fallbacks.")
                total_processed += 1
                continue
            
            # Save project plan
            plan_file = self.save_project_plan(project_plan, paper_dir.name)
            
            if plan_file:
                successful_plans += 1
                print(f"✅ Successfully created plan for {paper_dir.name} and saved to {plan_file}.")
            else:
                print(f"❌ Failed to save plan for {paper_dir.name}.") # save_project_plan already prints error
            
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
        # Ensure chunk files are like 'chunk_0.txt', 'chunk_1.txt', etc. for proper sorting
        chunk_files = sorted(paper_dir.glob("chunk_*.txt"), key=lambda x: int(x.stem.split('_')[-1]))
        
        if not chunk_files:
            print(f"No chunk files found in {paper_dir}. Expected format: chunk_N.txt")
            return None
        
        content = ""
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_content = f.read()
                    # Remove chunk headers (e.g., lines starting with '====' or 'Chunk X:')
                    lines = chunk_content.split('\n')
                    content_only_lines = []
                    for line in lines:
                        if not (line.strip().startswith('====') or line.strip().startswith('Chunk ')):
                            content_only_lines.append(line)
                    content += '\n'.join(content_only_lines) + '\n\n' # Add newlines for separation
            except Exception as e:
                print(f"Error reading {chunk_file}: {e}")
                continue
        
        return content.strip() # Remove leading/trailing whitespace from combined content


def main():
    """Main execution function"""
    try:
        planner = IntelligentProjectPlanner()
        planner.process_relevant_papers()
        
    except ValueError as ve:
        print(f"Initialization error: {ve}")
        print("Please ensure your .env file has valid API keys for at least one service.")
    except Exception as e:
        print(f"An unexpected error occurred during project planning: {e}")


if __name__ == "__main__":
    main()
