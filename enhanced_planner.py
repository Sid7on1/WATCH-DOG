#!/usr/bin/env python3
"""
Enhanced Project Structure Planner for Advanced AI/ML Projects
Creates detailed implementation plans for RAG, NLP, Agents, and CI/CD projects
"""

import os
import requests
import json
import time
from pathlib import Path
from dotenv import load_dotenv

try:
    from pusher import GitHubRepositoryManager
except ImportError:
    print("Warning: GitHubRepositoryManager not found. Please ensure 'pusher.py' is available.")
    class GitHubRepositoryManager:
        def __init__(self, artifacts_dir):
            print(f"Dummy GitHubRepositoryManager initialized for {artifacts_dir}")

# Load environment variables
load_dotenv()

def env(*names):
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return None

class EnhancedProjectPlanner:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.relevant_dir = self.artifacts_dir / "relevant"
        self.structures_dir = self.artifacts_dir / "structures"
        
        # Enhanced multi-API configuration for planning
        self.apis = {
            "openrouter": {
                "key": env("OPEN_API", "OPENROUTER_API_KEY", "OPENROUTER_API_TOKEN"),
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "models": [
                    "deepseek/deepseek-r1-0528:free",           # Best reasoning for planning
                    "moonshotai/kimi-k2:free",                  # Excellent for planning
                    "google/gemini-2.0-flash-exp:free",         # Strong planning
                    "qwen/qwen3-coder:free"                     # Technical planning
                ]
            },
            "groq": {
                "key": env("groq_API", "GROQ_API"),
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "models": [
                    "llama-3.3-70b-versatile",                 # Best for planning
                    "qwen/qwen3-32b"                           # Technical planning
                ]
            },
            "cohere": {
                "key": env("cohere_API", "COHERE_API", "COHERE_API_KEY"),
                "url": "https://api.cohere.ai/v2/chat",
                "models": ["c4ai-aya-expanse-32b"]            # Best for planning
            }
        }
        
        # Validate API keys
        self.validate_api_keys()
        
        # Initialize GitHub Repository Manager
        self.github_manager = GitHubRepositoryManager(artifacts_dir=artifacts_dir)
        
        # Set initial API
        self.current_api = "openrouter"
        self.current_model_index = 0
        
        # Ensure directories exist
        self.structures_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced project templates for advanced AI/ML
        self.enhanced_project_templates = {
            "rag_system": {
                "files": [
                    {"name": "main_rag.py", "purpose": "Main RAG system implementation", "priority": "high", "complexity": "high"},
                    {"name": "vector_store.py", "purpose": "Vector database management", "priority": "high", "complexity": "medium"},
                    {"name": "retriever.py", "purpose": "Document retrieval engine", "priority": "high", "complexity": "high"},
                    {"name": "embeddings.py", "purpose": "Text embedding generation", "priority": "high", "complexity": "medium"},
                    {"name": "reranker.py", "purpose": "Result reranking and scoring", "priority": "medium", "complexity": "medium"},
                    {"name": "query_processor.py", "purpose": "Query understanding and processing", "priority": "medium", "complexity": "medium"},
                    {"name": "context_manager.py", "purpose": "Context window management", "priority": "medium", "complexity": "medium"},
                    {"name": "llm_interface.py", "purpose": "LLM integration and prompting", "priority": "high", "complexity": "medium"},
                    {"name": "evaluation.py", "purpose": "RAG system evaluation metrics", "priority": "medium", "complexity": "medium"},
                    {"name": "config.py", "purpose": "System configuration", "priority": "high", "complexity": "low"},
                    {"name": "utils.py", "purpose": "Utility functions", "priority": "low", "complexity": "low"},
                    {"name": "requirements.txt", "purpose": "Python dependencies", "priority": "high", "complexity": "low"},
                    {"name": "README.md", "purpose": "Project documentation", "priority": "medium", "complexity": "low"},
                    {"name": "docker-compose.yml", "purpose": "Container orchestration", "priority": "medium", "complexity": "low"}
                ],
                "folders": ["data", "models", "logs", "checkpoints", "tests", "configs", "docs", "scripts"]
            },
            "nlp_advanced": {
                "files": [
                    {"name": "main_nlp.py", "purpose": "Main NLP pipeline", "priority": "high", "complexity": "high"},
                    {"name": "transformer_model.py", "purpose": "Transformer architecture implementation", "priority": "high", "complexity": "high"},
                    {"name": "tokenizer.py", "purpose": "Advanced tokenization", "priority": "high", "complexity": "medium"},
                    {"name": "attention_layers.py", "purpose": "Custom attention mechanisms", "priority": "high", "complexity": "high"},
                    {"name": "fine_tuning.py", "purpose": "Model fine-tuning pipeline", "priority": "high", "complexity": "high"},
                    {"name": "prompt_engineering.py", "purpose": "Prompt optimization", "priority": "medium", "complexity": "medium"},
                    {"name": "text_processor.py", "purpose": "Text preprocessing and postprocessing", "priority": "medium", "complexity": "medium"},
                    {"name": "evaluation_metrics.py", "purpose": "NLP evaluation metrics", "priority": "medium", "complexity": "medium"},
                    {"name": "data_loader.py", "purpose": "Dataset loading and batching", "priority": "medium", "complexity": "medium"},
                    {"name": "inference.py", "purpose": "Model inference pipeline", "priority": "medium", "complexity": "medium"},
                    {"name": "config.py", "purpose": "Model configuration", "priority": "high", "complexity": "low"},
                    {"name": "requirements.txt", "purpose": "Python dependencies", "priority": "high", "complexity": "low"},
                    {"name": "README.md", "purpose": "Project documentation", "priority": "medium", "complexity": "low"}
                ],
                "folders": ["data", "models", "logs", "checkpoints", "tests", "configs", "embeddings", "results"]
            },
            "ai_agent": {
                "files": [
                    {"name": "main_agent.py", "purpose": "Main agent implementation", "priority": "high", "complexity": "high"},
                    {"name": "agent_framework.py", "purpose": "Agent framework and architecture", "priority": "high", "complexity": "high"},
                    {"name": "llm_agent.py", "purpose": "LLM-based agent implementation", "priority": "high", "complexity": "high"},
                    {"name": "tool_manager.py", "purpose": "Tool integration and management", "priority": "high", "complexity": "medium"},
                    {"name": "memory_system.py", "purpose": "Agent memory and context", "priority": "medium", "complexity": "medium"},
                    {"name": "planning_engine.py", "purpose": "Task planning and execution", "priority": "high", "complexity": "high"},
                    {"name": "communication.py", "purpose": "Multi-agent communication", "priority": "medium", "complexity": "medium"},
                    {"name": "reasoning_engine.py", "purpose": "Agent reasoning capabilities", "priority": "high", "complexity": "high"},
                    {"name": "environment.py", "purpose": "Agent environment interface", "priority": "medium", "complexity": "medium"},
                    {"name": "evaluation.py", "purpose": "Agent performance evaluation", "priority": "medium", "complexity": "medium"},
                    {"name": "config.py", "purpose": "Agent configuration", "priority": "high", "complexity": "low"},
                    {"name": "requirements.txt", "purpose": "Python dependencies", "priority": "high", "complexity": "low"},
                    {"name": "README.md", "purpose": "Project documentation", "priority": "medium", "complexity": "low"}
                ],
                "folders": ["agents", "tools", "memory", "logs", "tests", "configs", "environments", "results"]
            },
            "mlops_cicd": {
                "files": [
                    {"name": "main_pipeline.py", "purpose": "Main MLOps pipeline", "priority": "high", "complexity": "high"},
                    {"name": "model_trainer.py", "purpose": "Model training pipeline", "priority": "high", "complexity": "high"},
                    {"name": "model_deployer.py", "purpose": "Model deployment automation", "priority": "high", "complexity": "high"},
                    {"name": "model_monitor.py", "purpose": "Model monitoring and alerting", "priority": "high", "complexity": "medium"},
                    {"name": "experiment_tracker.py", "purpose": "Experiment tracking and versioning", "priority": "medium", "complexity": "medium"},
                    {"name": "data_validator.py", "purpose": "Data validation and quality checks", "priority": "medium", "complexity": "medium"},
                    {"name": "feature_store.py", "purpose": "Feature store implementation", "priority": "medium", "complexity": "medium"},
                    {"name": "model_registry.py", "purpose": "Model registry and versioning", "priority": "medium", "complexity": "medium"},
                    {"name": "ci_cd_pipeline.py", "purpose": "CI/CD automation", "priority": "high", "complexity": "high"},
                    {"name": "testing_framework.py", "purpose": "Automated testing for ML", "priority": "medium", "complexity": "medium"},
                    {"name": "config.py", "purpose": "Pipeline configuration", "priority": "high", "complexity": "low"},
                    {"name": "Dockerfile", "purpose": "Container configuration", "priority": "high", "complexity": "low"},
                    {"name": "kubernetes.yaml", "purpose": "Kubernetes deployment", "priority": "medium", "complexity": "medium"},
                    {"name": "requirements.txt", "purpose": "Python dependencies", "priority": "high", "complexity": "low"},
                    {"name": "README.md", "purpose": "Project documentation", "priority": "medium", "complexity": "low"}
                ],
                "folders": ["models", "data", "logs", "tests", "configs", "deployments", "monitoring", "experiments"]
            },
            "computer_vision": {
                "files": [
                    {"name": "main_cv.py", "purpose": "Main computer vision pipeline", "priority": "high", "complexity": "high"},
                    {"name": "cnn_model.py", "purpose": "CNN architecture implementation", "priority": "high", "complexity": "high"},
                    {"name": "vision_transformer.py", "purpose": "Vision transformer implementation", "priority": "high", "complexity": "high"},
                    {"name": "image_processor.py", "purpose": "Image preprocessing and augmentation", "priority": "medium", "complexity": "medium"},
                    {"name": "object_detector.py", "purpose": "Object detection implementation", "priority": "high", "complexity": "high"},
                    {"name": "segmentation.py", "purpose": "Image segmentation", "priority": "medium", "complexity": "high"},
                    {"name": "feature_extractor.py", "purpose": "Feature extraction layers", "priority": "medium", "complexity": "medium"},
                    {"name": "data_loader.py", "purpose": "Image data loading", "priority": "medium", "complexity": "medium"},
                    {"name": "evaluation.py", "purpose": "CV model evaluation", "priority": "medium", "complexity": "medium"},
                    {"name": "config.py", "purpose": "Model configuration", "priority": "high", "complexity": "low"},
                    {"name": "requirements.txt", "purpose": "Python dependencies", "priority": "high", "complexity": "low"},
                    {"name": "README.md", "purpose": "Project documentation", "priority": "medium", "complexity": "low"}
                ],
                "folders": ["data", "models", "logs", "checkpoints", "tests", "images", "results", "configs"]
            }
        }
        
        print(f"Enhanced Project Planner initialized with advanced AI/ML focus")
        print(f"Available APIs: {list(self.apis.keys())}")
        print(f"Valid API keys: {self.get_valid_api_count()}")
        print(f"Current API: {self.current_api}")
        print(f"Focus: RAG Systems, Advanced NLP, AI Agents, MLOps/CI-CD")
    
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
            raise ValueError("No valid API keys available to initialize EnhancedProjectPlanner.")
    
    def get_valid_api_count(self):
        """Get count of valid APIs"""
        return f"{len(self.valid_apis)}/{len(self.apis)}"
    
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
                api_config, current_model = self.get_current_api_config()
                print(f"🔄 Switched to API: {self.current_api} with model: {current_model}")
                return True
        
        print("❌ No more available APIs to switch to")
        return False
    
    def make_api_request(self, system_prompt, user_prompt, max_tokens=4000):
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
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(api_config["url"], headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def make_cohere_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to Cohere API"""
        try:
            from cohere import ClientV2
            
            api_config, _ = self.get_current_api_config()
            client = ClientV2(api_key=api_config['key'])
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = client.chat_stream(
                model=model,
                messages=messages,
                temperature=0.1
            )
            
            full_response = ""
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'message'):
                    if hasattr(chunk.delta.message, 'content'):
                        if hasattr(chunk.delta.message.content, 'text'):
                            full_response += chunk.delta.message.content.text
            
            return full_response
            
        except ImportError:
            # Fallback to REST API
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
            return result['message']['content'][0]['text']
    
    def analyze_paper_for_project_type(self, paper_content):
        """Analyze paper content to determine the best project type"""
        content_lower = paper_content.lower()
        
        # Enhanced scoring for different project types
        type_scores = {
            "rag_system": 0,
            "nlp_advanced": 0,
            "ai_agent": 0,
            "mlops_cicd": 0,
            "computer_vision": 0
        }
        
        # RAG System indicators
        rag_keywords = [
            "retrieval augmented generation", "rag", "vector database", "embedding retrieval",
            "semantic search", "knowledge retrieval", "document retrieval", "context retrieval",
            "llm retrieval", "hybrid search", "dense retrieval", "sparse retrieval"
        ]
        type_scores["rag_system"] = sum(3 if kw in content_lower else 0 for kw in rag_keywords)
        
        # Advanced NLP indicators
        nlp_keywords = [
            "transformer", "attention mechanism", "bert", "gpt", "llama", "language model",
            "text generation", "sentiment analysis", "named entity recognition", "question answering",
            "text classification", "machine translation", "summarization", "tokenization",
            "embedding", "fine-tuning", "prompt engineering", "instruction tuning"
        ]
        type_scores["nlp_advanced"] = sum(2 if kw in content_lower else 0 for kw in nlp_keywords)
        
        # AI Agent indicators
        agent_keywords = [
            "autonomous agent", "multi-agent", "agent-based", "intelligent agent",
            "reinforcement learning", "policy gradient", "actor-critic", "agent communication",
            "cooperative agents", "agent coordination", "llm agent", "tool-using agent",
            "planning agent", "reasoning agent", "agent framework"
        ]
        type_scores["ai_agent"] = sum(3 if kw in content_lower else 0 for kw in agent_keywords)
        
        # MLOps/CI-CD indicators
        mlops_keywords = [
            "mlops", "machine learning operations", "model deployment", "continuous integration",
            "continuous deployment", "ci/cd", "model monitoring", "model versioning",
            "experiment tracking", "automated testing", "pipeline automation", "devops",
            "kubernetes", "docker", "model serving", "feature store"
        ]
        type_scores["mlops_cicd"] = sum(3 if kw in content_lower else 0 for kw in mlops_keywords)
        
        # Computer Vision indicators
        cv_keywords = [
            "computer vision", "image processing", "object detection", "image segmentation",
            "face recognition", "optical character recognition", "image classification",
            "generative model", "style transfer", "video analysis", "3d vision"
        ]
        type_scores["computer_vision"] = sum(2 if kw in content_lower else 0 for kw in cv_keywords)
        
        # Select best matching type
        best_type_item = max(type_scores.items(), key=lambda x: x[1])
        project_type = best_type_item[0] if best_type_item[1] > 0 else "nlp_advanced"
        
        return project_type, type_scores
    
    def create_enhanced_project_plan(self, paper_content, paper_name, max_retries=2):
        """Create enhanced project plan using AI analysis"""
        
        # Determine project type
        project_type, type_scores = self.analyze_paper_for_project_type(paper_content)
        
        system_prompt = f"""You are an expert AI/ML engineer and project architect specializing in advanced AI systems.

MISSION: Create a comprehensive, production-ready project plan for implementing the research paper as a working system.

PROJECT TYPE DETECTED: {project_type.upper().replace('_', ' ')}

FOCUS AREAS:
- RAG Systems: Vector databases, retrieval engines, embedding systems
- Advanced NLP: Transformers, language models, text processing
- AI Agents: Autonomous agents, multi-agent systems, LLM agents
- MLOps/CI-CD: Model deployment, monitoring, automation
- Computer Vision: Image processing, object detection, neural networks

REQUIREMENTS:
1. Analyze the paper's core algorithms and methods
2. Identify key technical components that can be implemented
3. Create a structured project with clear file organization
4. Focus on production-ready, scalable implementations
5. Include proper testing, monitoring, and deployment considerations

RESPONSE FORMAT: Valid JSON with this structure:
{{
  "project_type": "{project_type}",
  "project_name": "descriptive_project_name",
  "description": "detailed project description",
  "key_algorithms": ["algorithm1", "algorithm2", ...],
  "main_libraries": ["library1", "library2", ...],
  "data_sources": ["source1", "source2", ...],
  "model_architecture": "architecture description",
  "training_requirements": {{
    "gpu_required": true/false,
    "estimated_training_time": "time estimate",
    "memory_requirements": "memory estimate",
    "distributed_training": true/false
  }},
  "implementation_priority": "high/medium/low",
  "complexity_level": "advanced/intermediate/basic",
  "production_ready": true/false,
  "required_files": [
    {{
      "filename": "file.py",
      "purpose": "file purpose",
      "priority": "high/medium/low",
      "complexity": "high/medium/low",
      "dependencies": ["dep1", "dep2"],
      "key_functions": ["func1", "func2"]
    }}
  ],
  "required_folders": ["folder1", "folder2"],
  "special_requirements": ["req1", "req2"],
  "evaluation_metrics": ["metric1", "metric2"],
  "deployment_considerations": ["consideration1", "consideration2"]
}}"""

        user_prompt = f"""Paper: {paper_name}

Content (first 2000 chars):
{paper_content[:2000]}

Create a comprehensive project plan for implementing this research paper as a production-ready {project_type.replace('_', ' ')} system. Focus on the core algorithms and methods that can be practically implemented."""

        for attempt in range(max_retries + 1):
            try:
                api_config, current_model = self.get_current_api_config()
                print(f"🧠 Creating enhanced plan using {self.current_api} with {current_model}...")
                
                response_text = self.make_api_request(system_prompt, user_prompt, 4000)
                
                # Parse JSON response
                try:
                    plan = json.loads(response_text)
                    
                    # Enhance with template if needed
                    if project_type in self.enhanced_project_templates:
                        template = self.enhanced_project_templates[project_type]
                        if not plan.get("required_files"):
                            plan["required_files"] = template["files"]
                        if not plan.get("required_folders"):
                            plan["required_folders"] = template["folders"]
                    
                    # Add metadata
                    plan["paper_name"] = paper_name
                    plan["type_scores"] = type_scores
                    plan["planning_model"] = f"{self.current_api}/{current_model}"
                    plan["enhanced_planning"] = True
                    
                    print(f"✅ Enhanced plan created for {project_type} project")
                    return plan
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing error: {e}")
                    if attempt < max_retries:
                        print("🔄 Retrying with different model...")
                        if self.switch_to_next_api():
                            time.sleep(10)
                            continue
                    return self.create_enhanced_fallback_plan(paper_content, paper_name, project_type)
                    
            except Exception as e:
                print(f"❌ Error creating plan: {e}")
                if attempt < max_retries:
                    if self.switch_to_next_api():
                        time.sleep(10)
                        continue
                return self.create_enhanced_fallback_plan(paper_content, paper_name, project_type)
        
        return self.create_enhanced_fallback_plan(paper_content, paper_name, project_type)
    
    def create_enhanced_fallback_plan(self, paper_content, paper_name, project_type):
        """Create enhanced fallback plan with content analysis"""
        print("🔄 Creating enhanced fallback plan with advanced analysis")
        
        # Get template
        template = self.enhanced_project_templates.get(project_type, self.enhanced_project_templates["nlp_advanced"])
        
        # Extract key algorithms from content
        import re
        key_algorithms = []
        algorithm_patterns = [
            r"(\b\w+(?:-\w+)*)\s+(?:algorithm|method|approach|technique|model|framework|network)\b",
            r"(\b\w+)\s+(?:learning|prediction|generation|segmentation|detection|retrieval)\b"
        ]
        
        content_lower = paper_content.lower()
        for pattern in algorithm_patterns:
            matches = re.findall(pattern, content_lower)
            key_algorithms.extend([match.replace('_', ' ').title() for match in matches if len(match) > 2])
        
        # Remove duplicates and common words
        common_words = {"the", "and", "for", "with", "this", "that", "from", "using", "based", "our", "we", "can"}
        key_algorithms = list(set([alg for alg in key_algorithms if alg.lower() not in common_words]))[:8]
        
        if not key_algorithms:
            key_algorithms = [f"{project_type.replace('_', ' ').title()} Implementation"]
        
        enhanced_plan = {
            "project_type": project_type,
            "project_name": f"enhanced_{paper_name.replace(' ', '_').replace('-', '_')}",
            "description": f"Enhanced {project_type.replace('_', ' ')} implementation based on {paper_name}. Advanced AI/ML project with production-ready architecture.",
            "key_algorithms": key_algorithms,
            "main_libraries": self.get_libraries_for_type(project_type),
            "data_sources": ["paper_derived", "domain_specific_datasets"],
            "model_architecture": f"Advanced {project_type.replace('_', ' ')} architecture with modern best practices",
            "training_requirements": {
                "gpu_required": True,
                "estimated_training_time": "2-8 hours depending on dataset size",
                "memory_requirements": "16GB+ GPU memory recommended",
                "distributed_training": project_type in ["nlp_advanced", "computer_vision"]
            },
            "implementation_priority": "high",
            "complexity_level": "advanced",
            "production_ready": True,
            "required_files": template["files"],
            "required_folders": template["folders"],
            "special_requirements": [
                f"Enhanced {project_type.replace('_', ' ')} implementation",
                "Production-ready architecture",
                "Advanced AI/ML techniques",
                "Scalable and maintainable code"
            ],
            "evaluation_metrics": self.get_metrics_for_type(project_type),
            "deployment_considerations": [
                "Docker containerization",
                "Kubernetes orchestration",
                "API endpoint design",
                "Monitoring and logging",
                "Scalability planning"
            ],
            "enhanced_fallback": True,
            "paper_name": paper_name
        }
        
        print(f"📋 Enhanced Fallback Project Type: {project_type}")
        print(f"📋 Detected Algorithms: {', '.join(key_algorithms) if key_algorithms else 'None'}")
        
        return enhanced_plan
    
    def get_libraries_for_type(self, project_type):
        """Get appropriate libraries for project type"""
        library_map = {
            "rag_system": ["transformers", "faiss-cpu", "langchain", "chromadb", "sentence-transformers", "torch", "numpy", "pandas"],
            "nlp_advanced": ["transformers", "torch", "tokenizers", "datasets", "accelerate", "peft", "numpy", "pandas"],
            "ai_agent": ["transformers", "langchain", "openai", "torch", "gymnasium", "stable-baselines3", "numpy", "pandas"],
            "mlops_cicd": ["mlflow", "kubeflow", "docker", "kubernetes", "prometheus", "grafana", "torch", "scikit-learn"],
            "computer_vision": ["torch", "torchvision", "opencv-python", "albumentations", "timm", "numpy", "pillow"]
        }
        return library_map.get(project_type, ["torch", "numpy", "pandas", "scikit-learn"])
    
    def get_metrics_for_type(self, project_type):
        """Get appropriate evaluation metrics for project type"""
        metrics_map = {
            "rag_system": ["retrieval_accuracy", "answer_relevance", "context_precision", "context_recall", "faithfulness"],
            "nlp_advanced": ["accuracy", "f1_score", "bleu", "rouge", "perplexity", "bertscore"],
            "ai_agent": ["success_rate", "reward", "episode_length", "convergence_time", "task_completion"],
            "mlops_cicd": ["model_accuracy", "deployment_time", "uptime", "latency", "throughput"],
            "computer_vision": ["accuracy", "precision", "recall", "iou", "map", "fps"]
        }
        return metrics_map.get(project_type, ["accuracy", "precision", "recall", "f1_score"])
    
    def plan_all_papers(self):
        """Create enhanced plans for all relevant papers"""
        if not self.relevant_dir.exists():
            print(f"❌ Relevant papers directory not found: {self.relevant_dir}")
            return
        
        paper_dirs = [d for d in self.relevant_dir.iterdir() if d.is_dir()]
        
        if not paper_dirs:
            print("❌ No relevant papers found")
            return
        
        print(f"🎯 Creating enhanced plans for {len(paper_dirs)} papers")
        
        successful_plans = 0
        
        for paper_dir in paper_dirs:
            paper_name = paper_dir.name
            
            print(f"\n{'='*60}")
            print(f"📋 Planning: {paper_name}")
            print(f"{'='*60}")
            
            # Read paper content
            paper_content = self.read_paper_content(paper_dir)
            
            if paper_content:
                # Create enhanced plan
                plan = self.create_enhanced_project_plan(paper_content, paper_name)
                
                if plan:
                    # Save plan
                    if self.save_enhanced_plan(paper_name, plan):
                        successful_plans += 1
                        print(f"✅ Enhanced plan created for {paper_name}")
                    else:
                        print(f"❌ Failed to save plan for {paper_name}")
                else:
                    print(f"❌ Failed to create plan for {paper_name}")
            else:
                print(f"❌ Could not read content for {paper_name}")
        
        print(f"\n{'='*80}")
        print("🎉 ENHANCED PLANNING COMPLETE")
        print(f"{'='*80}")
        print(f"📊 Total papers: {len(paper_dirs)}")
        print(f"✅ Successful plans: {successful_plans}")
        print(f"📈 Success rate: {(successful_plans/len(paper_dirs))*100:.1f}%")
        print(f"📁 Plans saved to: {self.structures_dir}")
        print(f"🎯 Focus: Advanced AI/ML implementation projects")
        print(f"{'='*80}")
    
    def read_paper_content(self, paper_dir):
        """Read and combine paper content from chunks"""
        try:
            chunk_files = sorted(paper_dir.glob("chunk_*.txt"))
            
            if not chunk_files:
                print(f"❌ No chunk files found in {paper_dir}")
                return None
            
            combined_content = ""
            for chunk_file in chunk_files:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Skip metadata lines
                    lines = content.split('\n')
                    content_start = 0
                    for i, line in enumerate(lines):
                        if line.startswith('='):
                            content_start = i + 1
                            break
                    combined_content += '\n'.join(lines[content_start:]) + '\n\n'
            
            return combined_content.strip()
            
        except Exception as e:
            print(f"❌ Error reading paper content: {e}")
            return None
    
    def save_enhanced_plan(self, paper_name, plan):
        """Save enhanced plan to structures directory"""
        try:
            plan_dir = self.structures_dir / paper_name
            plan_dir.mkdir(parents=True, exist_ok=True)
            
            plan_file = plan_dir / "enhanced_plan.json"
            
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Enhanced plan saved: {plan_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving enhanced plan: {e}")
            return False


def main():
    """Main execution function"""
    try:
        planner = EnhancedProjectPlanner()
        planner.plan_all_papers()
        
    except Exception as e:
        print(f"❌ Error initializing enhanced project planner: {e}")


if __name__ == "__main__":
    main()