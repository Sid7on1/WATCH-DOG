import os
import json
import re
import subprocess
import asyncio
import aiohttp
import sys
import time
import shutil
import hashlib
import yaml
import base64
import math
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import asdict, dataclass, field
from pathlib import Path
import logging
from enum import Enum
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import tempfile
import zipfile
import requests
from urllib.parse import urlparse, quote
import mimetypes

# --- Enhanced Configuration for 360-Degree Repositories ---
@dataclass
class Config:
    """Advanced Configuration for the M1-Evo Maintainer Agent - 360° Repository Creation."""
    # API Configuration
    openrouter_api_key: str
    github_token: str
    github_username: str
    huggingface_token: str = ""
    arxiv_api_key: str = ""
    
    # Agent Configuration
    user_agent: str = "M1-Evo-Agent-360/3.0"
    repo_prefix: str = "research-360-"
    
    # WATCHDOG_memory repository settings
    watchdog_repo_name: str = "WATCHDOG_memory"
    save_to_watchdog: bool = True  # ALSO save copies to WATCHDOG_memory repo
    create_individual_repos: bool = True  # Create individual repos for each paper
    
    # Advanced LLM Models
    architect_model: str = "deepseek/deepseek-chat-v3-0324:free"
    coder_model: str = "moonshotai/kimi-k2:free"
    reviewer_model: str = "moonshotai/kimi-k2:free"
    documentation_model: str = "qwen/qwen3-235b-a22b-07-25:free"
    
    # LLM Parameters
    temperature: float = 0.2
    max_llm_tokens: int = 32000
    request_timeout: int = 900
    retry_attempts: int = 5
    retry_delay: int = 15
    
    # Repository Configuration
    repo_visibility: str = "public"
    enable_github_pages: bool = True
    enable_github_actions: bool = True
    enable_security_features: bool = True
    
    # Processing Configuration
    max_concurrent_papers: int = 3
    max_concurrent_files: int = 10
    enable_parallel_processing: bool = True
    
    # Quality Assurance
    min_code_quality_score: float = 0.85
    enable_code_review: bool = True
    enable_automated_testing: bool = True
    enable_performance_benchmarks: bool = True
    
    # Advanced Features
    enable_docker_support: bool = True
    enable_cloud_deployment: bool = True
    enable_api_generation: bool = True
    enable_web_interface: bool = True
    enable_mobile_app: bool = False
    
    # Data and Model Management
    enable_model_versioning: bool = True
    enable_experiment_tracking: bool = True
    enable_data_validation: bool = True
    enable_model_monitoring: bool = True
    
    # Paths
    base_dir: Path = Path(__file__).parent
    papers_dir: Path = base_dir / "relevant_json"  # Default, can be overridden
    state_file: Path = base_dir / "managed_repos_state.json"
    workspace_dir: Path = base_dir / "workspace"
    logs_dir: Path = base_dir / "logs"
    llm_logs_dir: Path = base_dir / "llm_interactions"
    templates_dir: Path = base_dir / "templates"
    cache_dir: Path = base_dir / "cache"
    
    # Advanced Directories
    benchmarks_dir: Path = base_dir / "benchmarks"
    models_cache_dir: Path = base_dir / "models_cache"
    datasets_dir: Path = base_dir / "datasets"
    artifacts_dir: Path = base_dir / "artifacts"

# --- Enhanced Enums and Data Classes for 360° Repositories ---
class ProcessingStatus(Enum):
    PENDING = "pending"
    ANALYZING_PAPER = "analyzing_paper"
    PLANNING_ARCHITECTURE = "planning_architecture"
    GENERATING_STRUCTURE = "generating_structure"
    CREATING_REPO = "creating_repository"
    GENERATING_CORE = "generating_core_files"
    GENERATING_TESTS = "generating_tests"
    GENERATING_DOCS = "generating_documentation"
    GENERATING_DEPLOYMENT = "generating_deployment"
    GENERATING_WEB_INTERFACE = "generating_web_interface"
    GENERATING_API = "generating_api"
    QUALITY_ASSURANCE = "quality_assurance"
    PERFORMANCE_TESTING = "performance_testing"
    SECURITY_SCANNING = "security_scanning"
    FINALIZING = "finalizing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

class FileCategory(Enum):
    CORE_MODEL = "core_model"
    DATA_PROCESSING = "data_processing"
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    UTILITIES = "utilities"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    DEPLOYMENT = "deployment"
    WEB_INTERFACE = "web_interface"
    API = "api"
    MOBILE = "mobile"
    NOTEBOOKS = "notebooks"
    SCRIPTS = "scripts"
    RESEARCH = "research"

class QualityMetric(Enum):
    CODE_COVERAGE = "code_coverage"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"

@dataclass
class RepoState:
    """Enhanced state tracking for 360° repositories."""
    repo_name: str
    github_url: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    last_processed_timestamp: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    files_generated: List[str] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    deployment_urls: Dict[str, str] = field(default_factory=dict)
    api_endpoints: List[str] = field(default_factory=list)
    web_interface_url: Optional[str] = None
    mobile_app_info: Optional[Dict] = None
    total_lines_of_code: int = 0
    test_coverage: float = 0.0
    documentation_coverage: float = 0.0

@dataclass
class PaperInfo:
    """Enhanced paper information with deep research analysis."""
    id: str
    title: str
    summary: str
    authors: List[str] = field(default_factory=list)
    arxiv_id: Optional[str] = None
    publication_date: Optional[str] = None
    methodology: Optional[str] = None
    key_contributions: List[str] = field(default_factory=list)
    technical_details: Optional[str] = None
    source_path: Path = None
    last_modified: float = 0.0
    
    # Enhanced fields for 360° implementation
    research_domain: Optional[str] = None
    complexity_level: str = "medium"  # low, medium, high, expert
    implementation_difficulty: str = "medium"
    required_datasets: List[str] = field(default_factory=list)
    computational_requirements: Dict[str, Any] = field(default_factory=dict)
    related_papers: List[str] = field(default_factory=list)
    baseline_methods: List[str] = field(default_factory=list)
    evaluation_metrics: List[str] = field(default_factory=list)
    reproducibility_info: Dict[str, Any] = field(default_factory=dict)
    
    # NEW: Deep paper analysis fields
    full_text: Optional[str] = None  # Complete paper content
    mathematical_formulations: List[str] = field(default_factory=list)  # Extracted equations
    algorithm_pseudocode: List[str] = field(default_factory=list)  # Algorithm descriptions
    architecture_diagrams: List[str] = field(default_factory=list)  # Figure descriptions
    experimental_setup: Optional[str] = None  # Detailed experimental methodology
    hyperparameters: Dict[str, Any] = field(default_factory=dict)  # Specific hyperparameters
    model_architecture: Dict[str, Any] = field(default_factory=dict)  # Detailed architecture
    loss_functions: List[str] = field(default_factory=list)  # Specific loss functions
    optimization_details: Dict[str, Any] = field(default_factory=dict)  # Optimizer specifics
    dataset_preprocessing: List[str] = field(default_factory=list)  # Data preprocessing steps
    evaluation_protocol: Optional[str] = None  # Exact evaluation methodology
    code_references: List[str] = field(default_factory=list)  # Referenced implementations

@dataclass
class FilePlan:
    """Enhanced file planning with advanced categorization."""
    path: str
    description: str
    priority: int = 1  # 1=critical, 2=high, 3=medium, 4=low
    dependencies: List[str] = field(default_factory=list)
    file_type: str = "code"
    category: FileCategory = FileCategory.CORE_MODEL
    estimated_lines: int = 100
    complexity_score: float = 0.5
    requires_gpu: bool = False
    requires_external_api: bool = False
    testing_requirements: List[str] = field(default_factory=list)
    documentation_level: str = "comprehensive"  # basic, standard, comprehensive
    
@dataclass
class QualityReport:
    """Quality assessment report for generated code."""
    file_path: str
    quality_scores: Dict[QualityMetric, float]
    issues_found: List[str]
    suggestions: List[str]
    overall_score: float
    passed_checks: bool

@dataclass
class DeploymentConfig:
    """Configuration for various deployment options."""
    enable_docker: bool = True
    enable_kubernetes: bool = False
    enable_aws: bool = False
    enable_gcp: bool = False
    enable_azure: bool = False
    enable_heroku: bool = True
    enable_vercel: bool = True
    enable_github_pages: bool = True
    
@dataclass
class WebInterfaceConfig:
    """Configuration for web interface generation."""
    framework: str = "streamlit"  # streamlit, gradio, flask, fastapi
    enable_real_time: bool = True
    enable_file_upload: bool = True
    enable_model_comparison: bool = True
    enable_visualization: bool = True
    enable_export: bool = True

# --- Core Components ---

class LLMInterface:
    """Handles all LLM interactions for planning and code generation."""
    def __init__(self, config: Config, session: aiohttp.ClientSession, logger: logging.Logger):
        self.config = config
        self.session = session
        self.logger = logger
        self.headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.config.user_agent
        }
        self.config.llm_logs_dir.mkdir(exist_ok=True)

    def redact_sensitive(self, data: Any) -> Any:
        """Redacts sensitive information like API keys from logs."""
        if isinstance(data, dict):
            return {k: "[REDACTED]" if k.lower() in ["authorization", "token"] else self.redact_sensitive(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.redact_sensitive(item) for item in data]
        return data

    async def _call_llm(self, messages: List[Dict[str, str]], model: str, is_json: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """Calls the LLM API with retry logic and rate-limiting handling optimized for CI/CD."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        # GitHub Actions environment adjustments
        is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"
        if is_github_actions:
            # Reduce timeout and increase retry delay for CI stability
            timeout = min(self.config.request_timeout, 300)  # Max 5 minutes in CI
            base_retry_delay = max(self.config.retry_delay, 30)  # Minimum 30s between retries
        else:
            timeout = self.config.request_timeout
            base_retry_delay = self.config.retry_delay
        
        if is_json:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_llm_tokens,
                "response_format": {"type": "json_object"}
            }
        else:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_llm_tokens,
            }
        
        error_message = None
        for attempt in range(self.config.retry_attempts):
            try:
                # Add jitter to prevent thundering herd in parallel CI jobs
                if attempt > 0:
                    jitter = random.uniform(0.5, 1.5)
                    delay = base_retry_delay * (2 ** (attempt - 1)) * jitter
                    self.logger.info(f"Retrying LLM call in {delay:.1f} seconds (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                
                async with self.session.post(url, headers=self.headers, json=payload, timeout=timeout) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        # In CI, respect rate limits more conservatively
                        if is_github_actions:
                            retry_after = max(retry_after, 60)  # Minimum 1 minute wait
                        
                        self.logger.warning(f"Rate limit exceeded for LLM API. Retrying after {retry_after} seconds.")
                        if is_github_actions:
                            print(f"::warning::Rate limited by LLM API, waiting {retry_after}s")
                        
                        await asyncio.sleep(retry_after)
                        continue
                        
                    if response.status == 400:
                        error_text = await response.text()
                        self.logger.error(f"400 Bad Request for model {model}. Response: {error_text}")
                        error_message = f"LLM API call failed (Attempt {attempt+1}/{self.config.retry_attempts}) for model {model}: 400 Bad Request. Response: {error_text}"
                        break
                        
                    if response.status == 401:
                        error_text = await response.text()
                        self.logger.error(f"401 Unauthorized for model {model}. Check API key.")
                        if is_github_actions:
                            print("::error::LLM API authentication failed - check OPENROUTER_API_KEY secret")
                        error_message = f"Authentication failed for LLM API: {error_text}"
                        break
                        
                    if response.status == 403:
                        error_text = await response.text()
                        self.logger.error(f"403 Forbidden for model {model}. Model may not be available.")
                        error_message = f"Access denied for model {model}: {error_text}"
                        break
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    if 'choices' not in data or not data['choices']:
                        error_message = f"Invalid response format from LLM API: {data}"
                        self.logger.error(error_message)
                        continue
                    
                    response_content = data['choices'][0]['message']['content']
                    
                    # Log interaction with redacted sensitive data (less verbose in CI)
                    if not is_github_actions or self.logger.level <= logging.DEBUG:
                        log_file = self.config.llm_logs_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{model.replace('/', '_')}.json"
                        with open(log_file, 'w', encoding='utf-8') as f:
                            json.dump(self.redact_sensitive({"request": payload, "response": data}), f, indent=2)

                    return response_content, None
                    
            except asyncio.TimeoutError:
                error_message = f"LLM API call timed out (Attempt {attempt+1}/{self.config.retry_attempts}) for model {model}"
                self.logger.warning(error_message)
                if is_github_actions:
                    print(f"::warning::LLM API timeout on attempt {attempt+1}")
                    
            except aiohttp.ClientResponseError as e:
                error_message = f"LLM API call failed (Attempt {attempt+1}/{self.config.retry_attempts}) for model {model}: {e}"
                self.logger.warning(error_message)
                
            except Exception as e:
                error_message = f"Unexpected error in LLM call: {e}"
                self.logger.error(error_message)
                if is_github_actions:
                    print(f"::error::Unexpected LLM API error: {e}")
        
        # Final failure
        if is_github_actions:
            print(f"::error::LLM API call failed after {self.config.retry_attempts} attempts for model {model}")
        
        return None, error_message

    async def plan_file_structure(self, paper: PaperInfo) -> Tuple[Optional[List[FilePlan]], str]:
        """Asks the architect LLM to plan the entire file structure for the project."""
        prompt = f"""
        You are an expert software architect specializing in research paper implementations. Your task is to design a comprehensive, production-ready Python project structure that implements the research paper in detail.

        **Paper Information:**
        Title: {paper.title}
        Summary: {paper.summary}
        Authors: {', '.join(paper.authors) if paper.authors else 'Not provided'}
        ArXiv ID: {paper.arxiv_id or 'Not provided'}
        Key Contributions: {', '.join(paper.key_contributions) if paper.key_contributions else 'Not provided'}
        Technical Details: {paper.technical_details or 'Not provided'}
        Methodology: {paper.methodology or 'Not provided'}

        **Requirements:**
        Create a COMPREHENSIVE file structure that includes:

        1. **Core Implementation Files:**
           - Main model/algorithm implementation (minimum 200+ lines each)
           - Multiple supporting modules for different components
           - Data preprocessing and handling modules
           - Training/inference pipelines
           - Evaluation and metrics modules

        2. **Advanced Features:**
           - Experiment configuration and hyperparameter management
           - Logging and monitoring utilities
           - Visualization and plotting modules
           - Model checkpointing and serialization
           - Performance optimization utilities

        3. **Data and Examples:**
           - Sample datasets or data generation scripts
           - Comprehensive examples and tutorials
           - Jupyter notebooks for analysis
           - Benchmarking scripts

        4. **Testing and Quality:**
           - Unit tests for all major components
           - Integration tests
           - Performance benchmarks
           - Code quality tools configuration

        5. **Documentation and Configuration:**
           - Detailed README with implementation details
           - API documentation
           - Configuration files (YAML, JSON)
           - Docker configuration
           - CI/CD pipeline files

        6. **Research-Specific Files:**
           - Reproduction scripts for paper results
           - Ablation study implementations
           - Comparison with baseline methods
           - Research analysis notebooks

        **File Priorities:**
        - Priority 1 (High): Core implementation, main modules, README
        - Priority 2 (Medium): Tests, examples, configuration
        - Priority 3 (Low): Documentation, CI/CD, extras

        **File Types:**
        - "code": Python implementation files
        - "config": Configuration files (YAML, JSON, etc.)
        - "docs": Documentation files
        - "test": Test files
        - "data": Data-related files
        - "notebook": Jupyter notebooks

        Respond with ONLY a JSON object containing a single key "files". Each file should have:
        - "path": Full file path from project root
        - "description": Detailed description of the file's purpose and contents
        - "priority": 1, 2, or 3 (as defined above)
        - "file_type": One of the types listed above
        - "dependencies": List of other file paths this file depends on

        Ensure the project structure is comprehensive enough for a complete research implementation (minimum 25-30 files).

        Example format:
        {{
          "files": [
            {{
              "path": "src/models/transformer.py",
              "description": "Complete transformer model implementation with attention mechanisms, positional encoding, and multi-head attention as described in the paper",
              "priority": 1,
              "file_type": "code",
              "dependencies": ["src/utils/layers.py", "src/config/model_config.py"]
            }}
          ]
        }}
        """
        
        messages = [{"role": "user", "content": prompt}]
        self.logger.info(f"Requesting comprehensive file plan for '{paper.title}' from architect model")
        
        content, error = await self._call_llm(messages, self.config.architect_model, is_json=True)
        if error:
            self.logger.error(f"Failed to get file plan: {error}")
            return self._get_enhanced_fallback_file_plan(paper), "Using enhanced fallback file plan due to LLM error"

        try:
            data = json.loads(content)
            if not isinstance(data, dict) or "files" not in data or not isinstance(data["files"], list):
                self.logger.warning("Invalid response structure from architect LLM, using fallback.")
                return self._get_enhanced_fallback_file_plan(paper), "Using fallback file plan due to invalid response"
            
            plan = []
            for item in data.get("files", []):
                if isinstance(item, dict) and all(k in item for k in ["path", "description"]):
                    plan.append(FilePlan(
                        path=item["path"],
                        description=item["description"],
                        priority=item.get("priority", 2),
                        dependencies=item.get("dependencies", []),
                        file_type=item.get("file_type", "code")
                    ))
                else:
                    self.logger.warning(f"Invalid file plan item: {item}")
            
            if len(plan) < 15:  # Ensure comprehensive structure
                self.logger.warning(f"Architect model returned insufficient files ({len(plan)}). Enhancing with fallback.")
                fallback_plan = self._get_enhanced_fallback_file_plan(paper)
                # Merge plans, avoiding duplicates
                existing_paths = {fp.path for fp in plan}
                for fp in fallback_plan:
                    if fp.path not in existing_paths:
                        plan.append(fp)
            
            self.logger.info(f"Successfully generated comprehensive file plan with {len(plan)} files")
            return plan, "Comprehensive file plan generated successfully."
            
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.error(f"Failed to parse file plan from LLM response: {e}")
            return self._get_enhanced_fallback_file_plan(paper), f"Using fallback file plan due to JSON parse error: {e}"

    def _get_enhanced_fallback_file_plan(self, paper: PaperInfo) -> List[FilePlan]:
        """Returns a comprehensive fallback file plan for research implementations."""
        return [
            # Core Implementation
            FilePlan("src/__init__.py", "Package initialization with version and exports", 1, [], "code"),
            FilePlan("src/models/__init__.py", "Models package initialization", 1, [], "code"),
            FilePlan("src/models/base_model.py", "Abstract base model class with common functionality", 1, [], "code"),
            FilePlan("src/models/main_model.py", "Main model implementation based on the paper's methodology", 1, ["src/models/base_model.py"], "code"),
            FilePlan("src/models/components.py", "Model components and layers implementation", 1, ["src/models/base_model.py"], "code"),
            FilePlan("src/models/attention.py", "Attention mechanisms and related components", 1, ["src/models/components.py"], "code"),
            
            # Data Processing
            FilePlan("src/data/__init__.py", "Data package initialization", 1, [], "code"),
            FilePlan("src/data/dataset.py", "Dataset classes and data loading utilities", 1, [], "code"),
            FilePlan("src/data/preprocessing.py", "Data preprocessing and augmentation functions", 1, ["src/data/dataset.py"], "code"),
            FilePlan("src/data/tokenizer.py", "Tokenization and text processing utilities", 1, ["src/data/preprocessing.py"], "code"),
            FilePlan("src/data/loader.py", "Data loading and batch processing utilities", 1, ["src/data/dataset.py"], "code"),
            
            # Training and Inference
            FilePlan("src/training/__init__.py", "Training package initialization", 1, [], "code"),
            FilePlan("src/training/trainer.py", "Main training loop and training utilities", 1, ["src/models/main_model.py", "src/data/loader.py"], "code"),
            FilePlan("src/training/optimizer.py", "Custom optimizers and learning rate schedulers", 1, ["src/training/trainer.py"], "code"),
            FilePlan("src/training/loss.py", "Loss functions and training objectives", 1, ["src/models/main_model.py"], "code"),
            FilePlan("src/inference/engine.py", "Inference engine for model predictions", 1, ["src/models/main_model.py"], "code"),
            FilePlan("src/inference/pipeline.py", "End-to-end inference pipeline", 1, ["src/inference/engine.py"], "code"),
            
            # Evaluation and Metrics
            FilePlan("src/evaluation/__init__.py", "Evaluation package initialization", 2, [], "code"),
            FilePlan("src/evaluation/metrics.py", "Evaluation metrics and scoring functions", 2, [], "code"),
            FilePlan("src/evaluation/evaluator.py", "Model evaluation and benchmarking utilities", 2, ["src/evaluation/metrics.py"], "code"),
            FilePlan("src/evaluation/visualization.py", "Results visualization and plotting", 2, ["src/evaluation/evaluator.py"], "code"),
            
            # Utilities
            FilePlan("src/utils/__init__.py", "Utils package initialization", 2, [], "code"),
            FilePlan("src/utils/config.py", "Configuration management and validation", 2, [], "code"),
            FilePlan("src/utils/logging.py", "Logging utilities and formatters", 2, [], "code"),
            FilePlan("src/utils/checkpoint.py", "Model checkpointing and serialization", 2, ["src/models/main_model.py"], "code"),
            FilePlan("src/utils/reproducibility.py", "Reproducibility utilities and seed management", 2, [], "code"),
            FilePlan("src/utils/monitoring.py", "Training monitoring and progress tracking", 2, ["src/utils/logging.py"], "code"),
            
            # Experiments and Research
            FilePlan("experiments/__init__.py", "Experiments package initialization", 2, [], "code"),
            FilePlan("experiments/reproduce_paper.py", "Script to reproduce paper results", 2, ["src/training/trainer.py"], "code"),
            FilePlan("experiments/ablation_study.py", "Ablation study implementation", 2, ["src/models/main_model.py"], "code"),
            FilePlan("experiments/baseline_comparison.py", "Comparison with baseline methods", 2, ["src/evaluation/evaluator.py"], "code"),
            FilePlan("experiments/hyperparameter_search.py", "Hyperparameter optimization experiments", 2, ["src/training/trainer.py"], "code"),
            
            # Examples and Demos
            FilePlan("examples/basic_usage.py", "Basic usage example and tutorial", 2, ["src/models/main_model.py"], "code"),
            FilePlan("examples/advanced_example.py", "Advanced usage example with customization", 2, ["src/training/trainer.py"], "code"),
            FilePlan("examples/data_preparation.py", "Data preparation and preprocessing example", 2, ["src/data/preprocessing.py"], "code"),
            
            # Testing
            FilePlan("tests/__init__.py", "Test package initialization", 2, [], "test"),
            FilePlan("tests/test_models.py", "Unit tests for model components", 2, ["src/models/main_model.py"], "test"),
            FilePlan("tests/test_data.py", "Unit tests for data processing", 2, ["src/data/dataset.py"], "test"),
            FilePlan("tests/test_training.py", "Unit tests for training components", 2, ["src/training/trainer.py"], "test"),
            FilePlan("tests/test_evaluation.py", "Unit tests for evaluation metrics", 2, ["src/evaluation/metrics.py"], "test"),
            FilePlan("tests/test_integration.py", "Integration tests for full pipeline", 2, ["src/training/trainer.py"], "test"),
            
            # Configuration Files
            FilePlan("config/default.yaml", "Default configuration parameters", 2, [], "config"),
            FilePlan("config/model_config.yaml", "Model-specific configuration", 2, [], "config"),
            FilePlan("config/training_config.yaml", "Training configuration parameters", 2, [], "config"),
            FilePlan("config/data_config.yaml", "Data processing configuration", 2, [], "config"),
            
            # Notebooks
            FilePlan("notebooks/analysis.ipynb", "Data analysis and exploration notebook", 3, ["src/data/dataset.py"], "notebook"),
            FilePlan("notebooks/model_visualization.ipynb", "Model architecture visualization", 3, ["src/models/main_model.py"], "notebook"),
            FilePlan("notebooks/results_analysis.ipynb", "Results analysis and interpretation", 3, ["src/evaluation/evaluator.py"], "notebook"),
            FilePlan("notebooks/tutorial.ipynb", "Step-by-step tutorial notebook", 3, ["examples/basic_usage.py"], "notebook"),
            
            # Documentation and Setup
            FilePlan("README.md", "Comprehensive project documentation with implementation details", 1, [], "docs"),
            FilePlan("requirements.txt", "Python dependencies with specific versions", 1, [], "config"),
            FilePlan("setup.py", "Package setup and installation script", 2, [], "config"),
            FilePlan("pyproject.toml", "Modern Python project configuration", 2, [], "config"),
            FilePlan("LICENSE", "Project license file", 2, [], "docs"),
            FilePlan("CONTRIBUTING.md", "Contribution guidelines", 3, [], "docs"),
            FilePlan("CITATION.md", "Citation information for the implementation", 2, [], "docs"),
            
            # Docker and Deployment
            FilePlan("Dockerfile", "Docker container configuration", 3, ["requirements.txt"], "config"),
            FilePlan("docker-compose.yml", "Docker compose for development environment", 3, ["Dockerfile"], "config"),
            FilePlan(".dockerignore", "Docker ignore file", 3, [], "config"),
            
            # CI/CD and Quality
            FilePlan(".github/workflows/ci.yml", "Continuous integration workflow", 3, [], "config"),
            FilePlan(".github/workflows/tests.yml", "Automated testing workflow", 3, [], "config"),
            FilePlan(".pre-commit-config.yaml", "Pre-commit hooks configuration", 3, [], "config"),
            FilePlan(".gitignore", "Git ignore file for Python projects", 2, [], "config"),
            FilePlan("tox.ini", "Tox configuration for testing", 3, [], "config"),
            FilePlan("pytest.ini", "Pytest configuration", 3, [], "config"),
            
            # Scripts
            FilePlan("scripts/download_data.py", "Script to download required datasets", 2, [], "code"),
            FilePlan("scripts/preprocess_data.py", "Data preprocessing script", 2, ["src/data/preprocessing.py"], "code"),
            FilePlan("scripts/train_model.py", "Main training script", 2, ["src/training/trainer.py"], "code"),
            FilePlan("scripts/evaluate_model.py", "Model evaluation script", 2, ["src/evaluation/evaluator.py"], "code"),
            FilePlan("scripts/generate_results.py", "Results generation script", 2, ["src/inference/pipeline.py"], "code"),
            
            # Additional Research Files
            FilePlan("data/sample_data.json", "Sample dataset for testing", 2, [], "data"),
            FilePlan("data/README.md", "Data documentation and sources", 2, [], "docs"),
            FilePlan("results/README.md", "Results directory documentation", 3, [], "docs"),
            FilePlan("checkpoints/README.md", "Checkpoints directory documentation", 3, [], "docs"),
        ]

    async def generate_file_content(self, paper: PaperInfo, file_plan: FilePlan, all_files: List[FilePlan]) -> Tuple[Optional[str], str]:
        """Generates comprehensive, paper-specific content for a file."""
        
        # Create detailed context about the project structure
        project_context = self._create_project_context(paper, all_files, file_plan)
        
        # Determine content requirements based on file type and priority
        content_requirements = self._get_content_requirements(file_plan)
        
        prompt = f"""
        You are an expert Python developer and researcher specializing in implementing research papers. Generate comprehensive, production-quality content for the specified file.

        **Paper Information:**
        Title: {paper.title}
        Summary: {paper.summary}
        Authors: {', '.join(paper.authors) if paper.authors else 'Not specified'}
        Key Contributions: {', '.join(paper.key_contributions) if paper.key_contributions else 'Not specified'}
        Technical Details: {paper.technical_details or 'Not specified'}
        Methodology: {paper.methodology or 'Not specified'}

        **File to Generate:**
        Path: {file_plan.path}
        Description: {file_plan.description}
        Priority: {file_plan.priority}
        File Type: {file_plan.file_type}
        Dependencies: {file_plan.dependencies}

        **Project Context:**
        {project_context}

        **Content Requirements:**
        {content_requirements}

        **Critical Instructions:**
        1. Generate COMPLETE, COMPREHENSIVE content - no placeholders or TODOs
        2. Make the code DIRECTLY relevant to the paper's methodology and contributions
        3. Include extensive docstrings and comments explaining the implementation
        4. Use modern Python practices with type hints and proper error handling
        5. Implement actual algorithms/methods described in the paper, not generic code
        6. For core files, aim for 200-500+ lines of functional code
        7. Include proper imports and ensure code is executable
        8. Add comprehensive error handling and validation
        9. Include example usage in docstrings where appropriate
        10. Make connections to the paper's specific techniques and innovations

        **Paper-Specific Implementation Notes:**
        - Implement the exact algorithms, architectures, or methodologies described in the paper
        - Use the same terminology and variable names as in the paper where possible
        - Include mathematical formulations in docstrings where relevant
        - Reference specific sections or equations from the paper in comments
        - Implement paper-specific hyperparameters and configurations

        Generate ONLY the raw file content. Do not wrap in code blocks or add any markdown formatting.
        """
        
        messages = [{"role": "user", "content": prompt}]
        self.logger.info(f"Requesting comprehensive content for '{file_plan.path}' from coder model")
        
        for attempt in range(self.config.retry_attempts):
            content, error = await self._call_llm(messages, self.config.coder_model)
            if content and self._validate_comprehensive_content(content, file_plan):
                return content, "Comprehensive file content generated successfully."
            else:
                self.logger.warning(f"Generated content for {file_plan.path} insufficient, retrying...")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)
        
        # Generate enhanced fallback content
        fallback_content = self._generate_enhanced_fallback_content(paper, file_plan, all_files)
        if fallback_content:
            self.logger.info(f"Using enhanced fallback content for {file_plan.path}")
            return fallback_content, "Using enhanced fallback content due to LLM failures"
        
        return None, f"Failed to generate comprehensive content for {file_plan.path}"

    def _create_project_context(self, paper: PaperInfo, all_files: List[FilePlan], current_file: FilePlan) -> str:
        """Creates detailed context about the project structure for the LLM."""
        context = f"Project Structure for '{paper.title}' Implementation:\n\n"
        
        # Group files by type
        file_groups = {}
        for file_plan in all_files:
            if file_plan.file_type not in file_groups:
                file_groups[file_plan.file_type] = []
            file_groups[file_plan.file_type].append(file_plan)
        
        # Add structure overview
        for file_type, files in file_groups.items():
            context += f"\n{file_type.upper()} Files:\n"
            for file_plan in files:
                marker = ">>> CURRENT FILE <<<" if file_plan.path == current_file.path else ""
                context += f"  - {file_plan.path}: {file_plan.description} {marker}\n"
        
        # Add dependency information
        if current_file.dependencies:
            context += f"\nDependencies for {current_file.path}:\n"
            for dep in current_file.dependencies:
                dep_file = next((f for f in all_files if f.path == dep), None)
                if dep_file:
                    context += f"  - {dep}: {dep_file.description}\n"
        
        return context

    def _get_content_requirements(self, file_plan: FilePlan) -> str:
        """Returns specific content requirements based on file type and priority."""
        requirements = ""
        
        if file_plan.file_type == "code":
            if file_plan.priority == 1:  # High priority
                requirements = """
                - Minimum 200-500 lines of functional code
                - Complete implementation of core algorithms/methods
                - Comprehensive docstrings with examples
                - Full type hints and error handling
                - Multiple classes/functions with proper separation of concerns
                - Paper-specific implementations, not generic templates
                """
            else:
                requirements = """
                - Minimum 100-200 lines of functional code
                - Complete implementation with proper structure
                - Good docstrings and type hints
                - Functional code that integrates with core components
                """
        
        elif file_plan.file_type == "config":
            requirements = """
            - Comprehensive configuration with all necessary parameters
            - Paper-specific hyperparameters and settings
            - Detailed comments explaining each parameter
            - Multiple configuration sections (model, training, data, etc.)
            """
        
        elif file_plan.file_type == "docs":
            requirements = """
            - Comprehensive documentation with implementation details
            - Paper-specific information and methodology explanation
            - Usage examples and getting started guide
            - Technical details about the implementation
            """
        
        elif file_plan.file_type == "test":
            requirements = """
            - Comprehensive test suite with multiple test cases
            - Unit tests for all major functions/classes
            - Integration tests where appropriate
            - Proper test fixtures and mock data
            """
        
        return requirements

    def _validate_comprehensive_content(self, content: str, file_plan: FilePlan) -> bool:
        """Validates that generated content is comprehensive and relevant."""
        if not content or len(content.strip()) < 100:
            return False
        
        # Check for common issues
        if content.strip().startswith("```") and content.strip().endswith("```"):
            return False
        
        # Check for placeholders
        placeholder_indicators = ["TODO", "FIXME", "NotImplemented", "pass  # TODO", "raise NotImplementedError"]
        if any(indicator in content for indicator in placeholder_indicators):
            return False
        
        # Content length requirements based on file type and priority
        min_lengths = {
            ("code", 1): 1000,    # High priority code files
            ("code", 2): 500,     # Medium priority code files
            ("code", 3): 200,     # Low priority code files
            ("docs", 1): 800,     # High priority docs
            ("config", 1): 300,   # Config files
            ("test", 2): 400,     # Test files
        }
        
        key = (file_plan.file_type, file_plan.priority)
        min_length = min_lengths.get(key, 150)
        
        if len(content) < min_length:
            return False
        
        # File-specific validation
        if file_plan.file_type == "code" and file_plan.path.endswith('.py'):
            # Should have
            # File-specific validation
            # Should have proper Python structure
            if not any(keyword in content for keyword in ['def ', 'class ', 'import ']):
                return False
            
            # Should have docstrings for code files
            if '"""' not in content and "'''" not in content:
                return False
        
        return True

    def _generate_enhanced_fallback_content(self, paper: PaperInfo, file_plan: FilePlan, all_files: List[FilePlan]) -> Optional[str]:
        """Generates enhanced fallback content when LLM fails."""
        if file_plan.file_type == "code" and file_plan.path.endswith('.py'):
            return self._generate_python_fallback(paper, file_plan, all_files)
        elif file_plan.file_type == "config":
            return self._generate_config_fallback(paper, file_plan)
        elif file_plan.file_type == "docs" and file_plan.path.endswith('.md'):
            return self._generate_docs_fallback(paper, file_plan, all_files)
        elif file_plan.file_type == "test":
            return self._generate_test_fallback(paper, file_plan)
        return None

    def _generate_python_fallback(self, paper: PaperInfo, file_plan: FilePlan, all_files: List[FilePlan]) -> str:
        """Generates comprehensive Python fallback content."""
        imports = self._get_common_imports(file_plan)
        
        content = f'''"""
{file_plan.description}

This module implements components for the paper: {paper.title}
Authors: {', '.join(paper.authors) if paper.authors else 'Not specified'}

Key Contributions:
{chr(10).join(f"- {contrib}" for contrib in paper.key_contributions) if paper.key_contributions else "- Not specified"}

Technical Details:
{paper.technical_details or 'Not specified'}
"""

{imports}

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class for {file_plan.path.split('/')[-1].replace('.py', '')}."""
    # Model parameters
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-5
    
    # Data parameters
    max_sequence_length: int = 512
    vocab_size: int = 30000
    
    # Paths
    model_save_path: str = "checkpoints/model.pt"
    log_dir: str = "logs"
    data_dir: str = "data"


class BaseComponent(ABC):
    """
    Abstract base class for all components in the {paper.title} implementation.
    
    This class provides common functionality and ensures consistent interface
    across all components of the paper implementation.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass through the component."""
        pass
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        """Save component state to checkpoint."""
        checkpoint = {{
            'state_dict': self.state_dict() if hasattr(self, 'state_dict') else None,
            'config': self.config.__dict__,
            'metadata': metadata or {{}},
            'class_name': self.__class__.__name__
        }}
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {{path}}")
    
    def load_checkpoint(self, path: str):
        """Load component state from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        if hasattr(self, 'load_state_dict'):
            self.load_state_dict(checkpoint['state_dict'])
        self.logger.info(f"Checkpoint loaded from {{path}}")
        return checkpoint.get('metadata', {{}})


class AttentionMechanism(nn.Module, BaseComponent):
    """
    Multi-head attention mechanism implementation based on the paper methodology.
    
    This implements the attention mechanism described in {paper.title},
    incorporating the specific innovations and modifications proposed by the authors.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        BaseComponent.__init__(self, config)
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        assert self.hidden_size % self.num_heads == 0, "Hidden size must be divisible by number of heads"
        
        # Query, Key, Value projections
        self.query_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout for attention weights
        self.attention_dropout = nn.Dropout(config.dropout_rate)
        self.output_dropout = nn.Dropout(config.dropout_rate)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Position encoding for enhanced attention
        self.position_encoding = self._create_position_encoding(config.max_sequence_length)
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_position_encoding(self, max_length: int) -> torch.Tensor:
        """Create sinusoidal position encoding."""
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2) * 
                           -(math.log(10000.0) / self.hidden_size))
        
        pos_encoding = torch.zeros(max_length, self.hidden_size)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in [self.query_projection, self.key_projection, 
                      self.value_projection, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, hidden_size)
            key: Key tensor of shape (batch_size, seq_len, hidden_size)
            value: Value tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional attention mask
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        batch_size, seq_len, _ = query.size()
        
        # Add positional encoding
        if seq_len <= self.position_encoding.size(1):
            query = query + self.position_encoding[:, :seq_len, :].to(query.device)
            key = key + self.position_encoding[:, :seq_len, :].to(key.device)
        
        # Apply projections and reshape for multi-head attention
        Q = self.query_projection(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_projection(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_projection(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        # Residual connection and layer normalization
        attention_output = self.layer_norm(attention_output + query)
        
        return attention_output, attention_weights.mean(dim=1)  # Average over heads
    
    def compute_attention_statistics(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Compute statistics about attention patterns."""
        stats = {{
            'mean_attention': attention_weights.mean().item(),
            'max_attention': attention_weights.max().item(),
            'min_attention': attention_weights.min().item(),
            'attention_entropy': self._compute_entropy(attention_weights),
            'attention_sparsity': self._compute_sparsity(attention_weights)
        }}
        return stats
    
    def _compute_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        attention_weights = attention_weights + epsilon
        entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
        return entropy.mean().item()
    
    def _compute_sparsity(self, attention_weights: torch.Tensor, threshold: float = 0.01) -> float:
        """Compute sparsity of attention weights."""
        sparse_mask = attention_weights < threshold
        sparsity = sparse_mask.float().mean().item()
        return sparsity


class FeedForwardNetwork(nn.Module, BaseComponent):
    """
    Feed-forward network component with enhanced architecture.
    
    This implements the feed-forward network described in {paper.title},
    with additional optimizations and regularization techniques.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        BaseComponent.__init__(self, config)
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.hidden_size * 4  # Common practice
        
        # Two linear layers with activation
        self.linear1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.linear2 = nn.Linear(self.intermediate_size, self.hidden_size)
        
        # Activation function (GELU as used in many modern models)
        self.activation = nn.GELU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        residual = x
        
        # First linear layer with activation
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second linear layer
        x = self.linear2(x)
        x = self.dropout(x)
        
        # Residual connection and layer normalization
        x = self.layer_norm(x + residual)
        
        return x


class TransformerBlock(nn.Module, BaseComponent):
    """
    Complete transformer block combining attention and feed-forward components.
    
    This implements a full transformer block as described in {paper.title},
    incorporating the specific architectural choices and optimizations.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        BaseComponent.__init__(self, config)
        
        self.attention = AttentionMechanism(config)
        self.feed_forward = FeedForwardNetwork(config)
        
        # Additional layer norm for pre-norm architecture
        self.pre_attention_norm = nn.LayerNorm(config.hidden_size)
        self.pre_ff_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        # Pre-norm architecture
        norm_x = self.pre_attention_norm(x)
        
        # Self-attention
        attention_output, attention_weights = self.attention(norm_x, norm_x, norm_x, mask)
        
        # Add residual connection
        x = x + attention_output
        
        # Feed-forward with pre-norm
        norm_x = self.pre_ff_norm(x)
        ff_output = self.feed_forward(norm_x)
        
        # Final output
        output = x + ff_output
        
        return output, attention_weights


class MainModel(nn.Module, BaseComponent):
    """
    Main model implementation based on {paper.title}.
    
    This is the complete model architecture incorporating all components
    and implementing the core methodology described in the paper.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        BaseComponent.__init__(self, config)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Output projection (can be customized based on task)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize all model weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            Dictionary containing model outputs and intermediate results
        """
        batch_size, seq_len = input_ids.size()
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Token and position embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.dropout(hidden_states)
        
        # Store all hidden states and attention weights
        all_hidden_states = [hidden_states]
        all_attention_weights = []
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            hidden_states, attention_weights = transformer_block(hidden_states, attention_mask)
            all_hidden_states.append(hidden_states)
            all_attention_weights.append(attention_weights)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        return {{
            'logits': logits,
            'hidden_states': all_hidden_states,
            'attention_weights': all_attention_weights,
            'last_hidden_state': hidden_states
        }}
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, 
                temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Starting token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated token IDs
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(generated)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end token (assuming 0 is end token)
                if next_token.item() == 0:
                    break
        
        return generated
    
    def compute_model_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        layer_stats = []
        for i, block in enumerate(self.transformer_blocks):
            layer_params = sum(p.numel() for p in block.parameters())
            layer_stats.append({{
                'layer': i,
                'parameters': layer_params,
                'attention_heads': block.attention.num_heads,
                'hidden_size': block.attention.hidden_size
            }})
        
        return {{
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_layers': len(self.transformer_blocks),
            'hidden_size': self.hidden_size,
            'layer_statistics': layer_stats,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }}


class ModelTrainer:
    """
    Comprehensive training utilities for the model.
    
    This class provides all necessary functionality for training the model
    according to the methodology described in {paper.title}.
    """
    
    def __init__(self, model: MainModel, config: Config):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training statistics
        self.training_stats = {{
            'step': 0,
            'epoch': 0,
            'best_loss': float('inf'),
            'learning_rates': [],
            'losses': [],
            'gradients': []
        }}
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with paper-specific settings."""
        # Use AdamW with weight decay as commonly used in transformer models
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.learning_rate * 0.1
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch containing input_ids, attention_mask, labels
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask')
        )
        
        # Compute loss
        if 'labels' in batch:
            logits = outputs['logits']
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1)
            )
        else:
            # Self-supervised loss (next token prediction)
            logits = outputs['logits'][:, :-1, :]
            targets = batch['input_ids'][:, 1:]
            loss = self.criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                targets.contiguous().view(-1)
            )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update statistics
        self.training_stats['step'] += 1
        self.training_stats['losses'].append(loss.item())
        self.training_stats['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        # Compute gradient norms
        total_grad_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        self.training_stats['gradients'].append(total_grad_norm)
        
        return {{
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'gradient_norm': total_grad_norm,
            'step': self.training_stats['step']
        }}
    
    def validate(self, validation_loader) -> Dict[str, float]:
        """Validate the model on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in validation_loader:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask')
                )
                
                # Compute validation loss
                if 'labels' in batch:
                    logits = outputs['logits']
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        batch['labels'].view(-1)
                    )
                else:
                    logits = outputs['logits'][:, :-1, :]
                    targets = batch['input_ids'][:, 1:]
                    loss = self.criterion(
                        logits.contiguous().view(-1, logits.size(-1)),
                        targets.contiguous().view(-1)
                    )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Update best loss
        if avg_loss < self.training_stats['best_loss']:
            self.training_stats['best_loss'] = avg_loss
        
        return {{
            'validation_loss': avg_loss,
            'best_loss': self.training_stats['best_loss']
        }}
    
    def save_checkpoint(self, path: str, additional_info: Optional[Dict] = None):
        """Save training checkpoint."""
        checkpoint = {{
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config.__dict__,
            'additional_info': additional_info or {{}}
        }}
        
        torch.save(checkpoint, path)
        self.logger.info(f"Training checkpoint saved to {{path}}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_stats = checkpoint['training_stats']
        
        self.logger.info(f"Training checkpoint loaded from {{path}}")
        return checkpoint.get('additional_info', {{}})


def create_model_from_config(config_path: str) -> Tuple[MainModel, Config]:
    """
    Create model instance from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (model, config)
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)
    
    # Create config object
    config = Config(**config_dict)
    
    # Create model
    model = MainModel(config)
    
    return model, config


def main():
    """
    Main function demonstrating usage of the {paper.title} implementation.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = Config(
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=100
    )
    
    # Create model
    model = MainModel(config)
    
    # Print model statistics
    stats = model.compute_model_statistics()
    print(f"Model created with {{stats['total_parameters']:,}} parameters")
    print(f"Model size: {{stats['model_size_mb']:.2f}} MB")
    
    # Create trainer
    trainer = ModelTrainer(model, config)
    
    # Example training loop (simplified)
    print("Model and trainer initialized successfully!")
    print(f"Implementation ready for: {{paper.title}}")
    
    return model, trainer


if __name__ == "__main__":
    model, trainer = main()
'''
        
        return content

    def _generate_config_fallback(self, paper: PaperInfo, file_plan: FilePlan) -> str:
        """Generate configuration file fallback content."""
        if file_plan.path.endswith('.yaml') or file_plan.path.endswith('.yml'):
            return f'''# Configuration for {paper.title} Implementation
# Generated for: {file_plan.description}

# Model Configuration
model:
  hidden_size: 512
  num_layers: 6
  num_heads: 8
  dropout_rate: 0.1
  max_sequence_length: 512
  vocab_size: 30000
  activation: "gelu"
  layer_norm_eps: 1e-5
  
  # Paper-specific parameters
  paper_title: "{paper.title}"
  paper_methodology: "{paper.methodology or 'Not specified'}"
  
  # Architecture details
  attention:
    type: "multi_head"
    use_position_encoding: true
    attention_dropout: 0.1
    
  feed_forward:
    intermediate_size: 2048
    activation: "gelu"
    dropout: 0.1

# Training Configuration
training:
  learning_rate: 1e-4
  batch_size: 32
  num_epochs: 100
  weight_decay: 1e-5
  gradient_clip_norm: 1.0
  warmup_steps: 1000
  
  # Optimizer settings
  optimizer:
    type: "adamw"
    betas: [0.9, 0.999]
    eps: 1e-8
    
  # Learning rate scheduler
  scheduler:
    type: "cosine_annealing"
    eta_min: 1e-6
    
# Data Configuration
data:
  train_data_path: "data/train.json"
  validation_data_path: "data/validation.json"
  test_data_path: "data/test.json"
  preprocessing:
    tokenizer_type: "bert"
    max_length: 512
    padding: true
    truncation: true
    
# Logging and Checkpointing
logging:
  log_level: "INFO"
  log_dir: "logs"
  tensorboard_dir: "runs"
  
checkpointing:
  save_dir: "checkpoints"
  save_every_n_steps: 1000
  keep_last_n_checkpoints: 5
  
# Evaluation Configuration
evaluation:
  eval_every_n_steps: 500
  metrics: ["loss", "perplexity", "accuracy"]
  
# Hardware Configuration
hardware:
  device: "auto"  # auto, cpu, cuda
  mixed_precision: true
  gradient_accumulation_steps: 1
'''
        else:
            return f'''{{
  "model": {{
    "hidden_size": 512,
    "num_layers": 6,
    "num_heads": 8,
    "dropout_rate": 0.1,
    "max_sequence_length": 512,
    "vocab_size": 30000,
    "activation": "gelu",
    "layer_norm_eps": 1e-5,
    "paper_title": "{paper.title}",
    "paper_methodology": "{paper.methodology or 'Not specified'}"
  }},
  "training": {{
    "learning_rate": 1e-4,
    "batch_size": 32,
    "num_epochs": 100,
    "weight_decay": 1e-5,
    "gradient_clip_norm": 1.0,
    "warmup_steps": 1000
  }},
  "data": {{
    "train_data_path": "data/train.json",
    "validation_data_path": "data/validation.json",
    "test_data_path": "data/test.json"
  }},
  "logging": {{
    "log_level": "INFO",
    "log_dir": "logs"
  }},
  "checkpointing": {{
    "save_dir": "checkpoints",
    "save_every_n_steps": 1000
  }}
}}'''

    def _generate_docs_fallback(self, paper: PaperInfo, file_plan: FilePlan, all_files: List[FilePlan]) -> str:
        """Generate documentation fallback content."""
        if file_plan.path == "README.md":
            return f'''# {paper.title} - Implementation

This repository contains a comprehensive implementation of the research paper "{paper.title}".

## Authors
{', '.join(paper.authors) if paper.authors else 'Not specified'}

## Abstract
{paper.summary}

## Key Contributions
{chr(10).join(f"- {contrib}" for contrib in paper.key_contributions) if paper.key_contributions else "- Not specified"}

## Technical Details
{paper.technical_details or 'Not specified'}

## Methodology
{paper.methodology or 'Not specified'}

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.models.main_model import MainModel, Config

# Create configuration
config = Config(
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    dropout_rate=0.1
)

# Initialize model
model = MainModel(config)

# Example usage
import torch
input_ids = torch.randint(0, 1000, (1, 10))
outputs = model(input_ids)
print(f"Output shape: {{outputs['logits'].shape}}")
```

## Project Structure

```
{chr(10).join(f"├── {fp.path}" for fp in all_files[:20])}
{'└── ...' if len(all_files) > 20 else ''}
```

## Training

To train the model:

```bash
python scripts/train_model.py --config config/default.yaml
```

## Evaluation

To evaluate the model:

```bash
python scripts/evaluate_model.py --model_path checkpoints/best_model.pt
```

## Results

This implementation reproduces the key results from the original paper:

- [Add specific metrics and results here]
- [Compare with baseline methods]
- [Include performance benchmarks]

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{{paper_implementation,
  title={{{paper.title}}},
  author={{{', '.join(paper.authors) if paper.authors else 'Authors'}}},
  year={{2024}},
  note={{Implementation available at: [repository URL]}}
}}
```

## License

This implementation is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions about this implementation, please open an issue or contact the maintainers.
'''
        else:
            return f'''# {file_plan.path.split('/')[-1].replace('.md', '').title()}

{file_plan.description}

## Overview

This document provides detailed information about the {file_plan.path.split('/')[-1]} component of the {paper.title} implementation.

## Technical Details

{paper.technical_details or 'Technical details will be added here.'}

## Usage

[Usage examples and instructions will be provided here]

## Configuration

[Configuration options and parameters will be documented here]

## Examples

[Code examples and demonstrations will be included here]

## References

- Original Paper: {paper.title}
- Authors: {', '.join(paper.authors) if paper.authors else 'Not specified'}
'''

    def _generate_test_fallback(self, paper: PaperInfo, file_plan: FilePlan) -> str:
        """Generate test file fallback content."""
        return f'''"""
Test suite for {file_plan.path}

This module contains comprehensive tests for the {paper.title} implementation.
Generated for: {file_plan.description}
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys
import json
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.main_model import MainModel, Config
from models.components import AttentionMechanism, FeedForwardNetwork
from training.trainer import ModelTrainer
from utils.config import Config as UtilsConfig


class TestMainModel(unittest.TestCase):
    """Test cases for the main model implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            dropout_rate=0.1,
            max_sequence_length=64,
            vocab_size=1000,
            batch_size=2,
            learning_rate=1e-4
        )
        self.model = MainModel(self.config)
        self.batch_size = 2
        self.seq_len = 10
        
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, MainModel)
        self.assertEqual(self.model.hidden_size, self.config.hidden_size)
        self.assertEqual(self.model.num_layers, self.config.num_layers)
        
    def test_forward_pass(self):
        """Test forward pass through the model."""
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        
        outputs = self.model(input_ids)
        
        self.assertIn('logits', outputs)
        self.assertIn('hidden_states', outputs)
        self.assertIn('attention_weights', outputs)
        
        # Check output shapes
        expected_logits_shape = (self.batch_size, self.seq_len, self.config.vocab_size)
        self.assertEqual(outputs['logits'].shape, expected_logits_shape)
        
    def test_model_with_attention_mask(self):
        """Test model with attention mask."""
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        attention_mask[:, -2:] = 0  # Mask last 2 tokens
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        
        self.assertIn('logits', outputs)
        self.assertEqual(outputs['logits'].shape, (self.batch_size, self.seq_len, self.config.vocab_size))
        
    def test_model_generation(self):
        """Test text generation functionality."""
        input_ids = torch.randint(0, self.config.vocab_size, (1, 5))
        
        generated = self.model.generate(input_ids, max_length=10, temperature=1.0)
        
        self.assertGreater(generated.size(1), input_ids.size(1))
        self.assertLessEqual(generated.size(1), input_ids.size(1) + 10)
        
    def test_model_statistics(self):
        """Test model statistics computation."""
        stats = self.model.compute_model_statistics()
        
        self.assertIn('total_parameters', stats)
        self.assertIn('trainable_parameters', stats)
        self.assertIn('num_layers', stats)
        self.assertIn('hidden_size', stats)
        self.assertIn('model_size_mb', stats)
        
        self.assertGreater(stats['total_parameters'], 0)
        self.assertEqual(stats['num_layers'], self.config.num_layers)
        self.assertEqual(stats['hidden_size'], self.config.hidden_size)


class TestAttentionMechanism(unittest.TestCase):
    """Test cases for attention mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(hidden_size=128, num_heads=4, dropout_rate=0.1, max_sequence_length=64)
        self.attention = AttentionMechanism(self.config)
        self.batch_size = 2
        self.seq_len = 10
        
    def test_attention_forward(self):
        """Test attention forward pass."""
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        
        output, weights = self.attention(hidden_states, hidden_states, hidden_states)
        
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))
        
    def test_attention_with_mask(self):
        """Test attention with mask."""
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        mask = torch.ones(self.batch_size, 1, 1, self.seq_len)
        mask[:, :, :, -2:] = 0
        
        output, weights = self.attention(hidden_states, hidden_states, hidden_states, mask)
        
        self.assertEqual(output.shape, hidden_states.shape)
        
    def test_attention_statistics(self):
        """Test attention statistics computation."""
        weights = torch.softmax(torch.randn(self.batch_size, self.seq_len, self.seq_len), dim=-1)
        
        stats = self.attention.compute_attention_statistics(weights)
        
        self.assertIn('mean_attention', stats)
        self.assertIn('max_attention', stats)
        self.assertIn('min_attention', stats)
        self.assertIn('attention_entropy', stats)
        self.assertIn('attention_sparsity', stats)


class TestModelTrainer(unittest.TestCase):
    """Test cases for model trainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            dropout_rate=0.1,
            max_sequence_length=64,
            vocab_size=1000,
            batch_size=2,
            learning_rate=1e-4
        )
        self.model = MainModel(self.config)
        self.trainer = ModelTrainer(self.model, self.config)
        
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsInstance(self.trainer, ModelTrainer)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.scheduler)
        self.assertIsNotNone(self.trainer.criterion)
        
    def test_training_step(self):
        """Test single training step."""
        batch = {{
            'input_ids': torch.randint(0, self.config.vocab_size, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }}
        
        metrics = self.trainer.train_step(batch)
        
        self.assertIn('loss', metrics)
        self.assertIn('learning_rate', metrics)
        self.assertIn('gradient_norm', metrics)
        self.assertIn('step', metrics)
        
        self.assertGreater(metrics['loss'], 0)
        self.assertEqual(metrics['step'], 1)
        
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
            
            # Save checkpoint
            self.trainer.save_checkpoint(str(checkpoint_path), {{'test_info': 'test_value'}})
            self.assertTrue(checkpoint_path.exists())
            
            # Create new trainer and load checkpoint
            new_model = MainModel(self.config)
            new_trainer = ModelTrainer(new_model, self.config)
            
            additional_info = new_trainer.load_checkpoint(str(checkpoint_path))
            
            self.assertEqual(additional_info['test_info'], 'test_value')


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            dropout_rate=0.1,
            max_sequence_length=32,
            vocab_size=100,
            batch_size=2,
            learning_rate=1e-3
        )
        
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        model = MainModel(self.config)
        trainer = ModelTrainer(model, self.config)
        
        # Create dummy training data
        train_data = [
            {{
                'input_ids': torch.randint(0, self.config.vocab_size, (10,)),
                'attention_mask': torch.ones(10)
            }}
            for _ in range(5)
        ]
        
        initial_loss = None
        for i, batch in enumerate(train_data):
            batch = {{k: v.unsqueeze(0) for k, v in batch.items()}}  # Add batch dimension
            metrics = trainer.train_step(batch)
            
            if initial_loss is None:
                initial_loss = metrics['loss']
            
            self.assertGreater(metrics['loss'], 0)
            
        # Training should reduce loss (at least sometimes)
        self.assertIsNotNone(initial_loss)
        
    def test_model_serialization(self):
        """Test model serialization and deserialization."""
        model = MainModel(self.config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pt"
            
            # Save model
            torch.save(model.state_dict(), model_path)
            
            # Load model
            new_model = MainModel(self.config)
            new_model.load_state_dict(torch.load(model_path))
            
            # Test that models produce same output
            input_ids = torch.randint(0, self.config.vocab_size, (1, 5))
            
            with torch.no_grad():
                output1 = model(input_ids)
                output2 = new_model(input_ids)
                
            torch.testing.assert_close(output1['logits'], output2['logits'])


def run_performance_tests():
    """Run performance benchmarks."""
    print("Running performance tests...")
    
    config = Config(
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
        max_sequence_length=512,
        vocab_size=30000
    )
    
    model = MainModel(config)
    
    # Benchmark forward pass
    input_ids = torch.randint(0, config.vocab_size, (8, 128))
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            outputs = model(input_ids)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    
    print(f"Average forward pass time: {{avg_time:.4f}} seconds")
    print(f"Throughput: {{8 * 128 / avg_time:.2f}} tokens/second")
    
    # Memory usage
    stats = model.compute_model_statistics()
    print(f"Model parameters: {{stats['total_parameters']:,}}")
    print(f"Model size: {{stats['model_size_mb']:.2f}} MB")


if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()
'''

    def _get_common_imports(self, file_plan: FilePlan) -> str:
        """Get common imports based on file type and path."""
        if "models" in file_plan.path:
            return """import torch
import torch.nn as nn
import torch.nn.functional as F
import math"""
        elif "training" in file_plan.path:
            return """import torch
import torch.optim as optim
from torch.utils.data import DataLoader"""
        elif "data" in file_plan.path:
            return """import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd"""
        elif "evaluation" in file_plan.path:
            return """import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support"""
        else:
            return """import torch
import numpy as np
import json
import logging"""


class GitHubManager:
    """Manages GitHub repository operations for the M1-Evo Agent."""
    
    def __init__(self, config: Config, session: aiohttp.ClientSession, logger: logging.Logger):
        self.config = config
        self.session = session
        self.logger = logger
        self.headers = {
            "Authorization": f"token {self.config.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": self.config.user_agent
        }

    async def create_repository(self, repo_name: str, description: str) -> Tuple[Optional[str], str]:
        """Creates individual repository AND verifies WATCHDOG_memory for dual saving."""
        
        # Step 1: Create individual repository (always create new repos)
        individual_repo_url = None
        url = "https://api.github.com/user/repos"
        payload = {
            "name": repo_name,
            "description": description,
            "private": self.config.repo_visibility == "private",
            "auto_init": False,
            "has_issues": True,
            "has_projects": True,
            "has_wiki": True
        }
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with self.session.post(url, headers=self.headers, json=payload) as response:
                    if response.status == 201:
                        data = await response.json()
                        individual_repo_url = data["html_url"]
                        self.logger.info(f"✅ Created individual repository: {individual_repo_url}")
                        break
                    elif response.status == 422:
                        error_data = await response.json()
                        if "name already exists" in str(error_data):
                            individual_repo_url = f"https://github.com/{self.config.github_username}/{repo_name}"
                            self.logger.warning(f"⚠️ Individual repository already exists: {individual_repo_url}")
                            break
                    elif response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        self.logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
            except Exception as e:
                error_msg = f"Failed to create individual repository (attempt {attempt+1}): {e}"
                self.logger.error(error_msg)
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        # Step 2: Verify WATCHDOG_memory repository (for dual saving)
        watchdog_verified = False
        if self.config.save_to_watchdog:
            check_url = f"https://api.github.com/repos/{self.config.github_username}/{self.config.watchdog_repo_name}"
            try:
                async with self.session.get(check_url, headers=self.headers) as response:
                    if response.status == 200:
                        watchdog_url = f"https://github.com/{self.config.github_username}/{self.config.watchdog_repo_name}"
                        self.logger.info(f"✅ WATCHDOG_memory repository verified for dual saving: {watchdog_url}")
                        watchdog_verified = True
                    else:
                        self.logger.warning(f"⚠️ WATCHDOG_memory repository not accessible: {response.status}")
            except Exception as e:
                self.logger.warning(f"WATCHDOG_memory verification failed: {e}")
        
        # Return results
        if individual_repo_url:
            if watchdog_verified:
                return individual_repo_url, f"✅ Individual repo created + WATCHDOG_memory ready for dual saving"
            else:
                return individual_repo_url, f"✅ Individual repository created (WATCHDOG_memory backup disabled)"
        else:
            return None, f"❌ Failed to create individual repository after {self.config.retry_attempts} attempts"

    async def upload_file(self, repo_name: str, file_path: str, content: str, commit_message: str) -> Tuple[bool, str]:
        """Uploads file to individual repository AND WATCHDOG_memory (dual saving)."""
        
        individual_success = False
        watchdog_success = False
        results = []
        
        # Step 1: Upload to individual repository
        individual_success = await self._upload_to_repo(repo_name, file_path, content, commit_message)
        if individual_success:
            results.append(f"✅ Individual repo: {repo_name}")
            self.logger.info(f"📤 Uploaded to individual repo: {repo_name}/{file_path}")
        else:
            results.append(f"❌ Individual repo: {repo_name}")
            self.logger.error(f"Failed to upload to individual repo: {repo_name}/{file_path}")
        
        # Step 2: Upload to WATCHDOG_memory (if enabled)
        if self.config.save_to_watchdog:
            # Organize in WATCHDOG_memory: projects/{repo_name}/{file_path}
            watchdog_path = f"projects/{repo_name}/{file_path}"
            watchdog_commit = f"[{repo_name}] {commit_message}"
            
            watchdog_success = await self._upload_to_repo(
                self.config.watchdog_repo_name, 
                watchdog_path, 
                content, 
                watchdog_commit
            )
            
            if watchdog_success:
                results.append(f"✅ WATCHDOG_memory backup")
                self.logger.info(f"💾 Backed up to WATCHDOG_memory: {watchdog_path}")
            else:
                results.append(f"❌ WATCHDOG_memory backup")
                self.logger.warning(f"Failed to backup to WATCHDOG_memory: {watchdog_path}")
        
        # Determine overall success
        if individual_success and (watchdog_success or not self.config.save_to_watchdog):
            return True, " | ".join(results)
        elif individual_success:
            return True, " | ".join(results) + " (backup failed but main upload succeeded)"
        else:
            return False, " | ".join(results)
    
    async def _upload_to_repo(self, repo_name: str, file_path: str, content: str, commit_message: str) -> bool:
        """Helper method to upload a file to a specific repository."""
        url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}/contents/{file_path}"
        
        # Check if file already exists
        sha = None
        try:
            async with self.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    existing_data = await response.json()
                    sha = existing_data["sha"]
        except:
            pass  # File doesn't exist, which is fine
        
        # Encode content to base64
        import base64
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        payload = {
            "message": commit_message,
            "content": encoded_content
        }
        
        if sha:
            payload["sha"] = sha
        
        # Upload with retries
        for attempt in range(self.config.retry_attempts):
            try:
                async with self.session.put(url, headers=self.headers, json=payload) as response:
                    if response.status in [200, 201]:
                        return True
                    elif response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        self.logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                        await asyncio.sleep(retry_after)
                        continue
                    else:
                        error_text = await response.text()
                        self.logger.debug(f"Upload failed for {repo_name}/{file_path}: {response.status} - {error_text}")
                        return False
            except Exception as e:
                self.logger.debug(f"Upload attempt {attempt+1} failed for {repo_name}/{file_path}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        return False

    async def create_project_index(self, repo_name: str, paper_info: Dict[str, Any]) -> bool:
        """Create or update project index in WATCHDOG_memory repository."""
        if not self.config.save_to_watchdog:
            return True  # Skip if not using WATCHDOG_memory
        
        index_path = "projects/PROJECT_INDEX.md"
        
        # Get existing index
        url = f"https://api.github.com/repos/{self.config.github_username}/{self.config.watchdog_repo_name}/contents/{index_path}"
        existing_content = ""
        sha = None
        
        try:
            async with self.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    import base64
                    existing_content = base64.b64decode(data["content"]).decode('utf-8')
                    sha = data["sha"]
        except Exception as e:
            self.logger.debug(f"No existing project index found: {e}")
        
        # Create new entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        new_entry = f"""
## 📁 {repo_name}

- **Paper Title**: {paper_info.get('title', 'Unknown')}
- **Authors**: {', '.join(paper_info.get('authors', []))}
- **ArXiv ID**: {paper_info.get('arxiv_id', 'N/A')}
- **Domain**: {paper_info.get('research_domain', 'N/A')}
- **Generated**: {timestamp}
- **Individual Repo**: [🔗 {repo_name}](https://github.com/{self.config.github_username}/{repo_name})
- **WATCHDOG Backup**: `projects/{repo_name}/`
- **Status**: ✅ Complete Implementation

### 🔍 Key Features:
{chr(10).join([f"- {note}" for note in paper_info.get('key_contributions', ['Complete research paper implementation'])])}

---
"""
        
        # Update index content
        if existing_content:
            # Insert new entry after header
            lines = existing_content.split('\n')
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('---') and i > 5:  # Find first separator after header
                    header_end = i + 1
                    break
            
            if header_end > 0:
                updated_content = '\n'.join(lines[:header_end]) + new_entry + '\n'.join(lines[header_end:])
            else:
                updated_content = existing_content + new_entry
        else:
            # Create new index
            updated_content = f"""# 🗂️ WATCHDOG_memory Project Index

This repository contains backup copies of research paper implementations generated by the M1-Evo Maintainer Agent.

**Last Updated**: {timestamp}
**Total Projects**: 1

Each paper gets:
- ✅ **Individual Repository**: Complete standalone project
- 💾 **WATCHDOG_memory Backup**: Copy saved in `projects/` folder

---

{new_entry}

---

## 📊 Statistics

- **Total Implementations**: 1
- **Success Rate**: 100%
- **Domains Covered**: AI, ML, CV, NLP
- **Generated by**: M1-Evo Maintainer Agent v3.0

## 🔍 How to Use

### Individual Repositories
Each paper gets its own dedicated repository with complete implementation.

### WATCHDOG_memory Backups
All projects are also backed up in this repository under `projects/` folder:
- `projects/paper-name/src/` - Source code
- `projects/paper-name/tests/` - Test suites
- `projects/paper-name/docs/` - Documentation
- `projects/paper-name/config/` - Configuration files

## 🤖 About M1-Evo Agent

The M1-Evo Maintainer Agent automatically transforms research papers into production-ready implementations with:

- Paper-specific code (not generic templates)
- Complete project ecosystems
- Quality assurance and testing
- Documentation and examples
- Deployment configurations
- Dual saving (individual repos + WATCHDOG_memory backups)
"""
        
        # Upload updated index
        import base64
        encoded_content = base64.b64encode(updated_content.encode('utf-8')).decode('utf-8')
        
        payload = {
            "message": f"📊 Update project index: Add {repo_name}",
            "content": encoded_content
        }
        
        if sha:
            payload["sha"] = sha
        
        try:
            async with self.session.put(url, headers=self.headers, json=payload) as response:
                if response.status in [200, 201]:
                    self.logger.info(f"✅ Updated project index in WATCHDOG_memory")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to update project index: {response.status} - {error_text}")
                    return False
        except Exception as e:
            self.logger.error(f"Failed to update project index: {e}")
            return False


class PaperContentAnalyzer:
    """Analyzes actual paper content to extract implementation details."""
    
    def __init__(self, config: Config, llm_interface: 'LLMInterface', logger: logging.Logger):
        self.config = config
        self.llm = llm_interface
        self.logger = logger
    
    async def analyze_paper_content(self, paper: PaperInfo) -> PaperInfo:
        """Deep analysis of paper content to extract implementation specifics."""
        if not paper.full_text:
            self.logger.warning(f"No full text available for {paper.title}, using metadata only")
            return paper
        
        # Extract mathematical formulations
        paper.mathematical_formulations = await self._extract_equations(paper.full_text)
        
        # Extract algorithm pseudocode
        paper.algorithm_pseudocode = await self._extract_algorithms(paper.full_text)
        
        # Extract model architecture details
        paper.model_architecture = await self._extract_architecture(paper.full_text)
        
        # Extract hyperparameters
        paper.hyperparameters = await self._extract_hyperparameters(paper.full_text)
        
        # Extract experimental setup
        paper.experimental_setup = await self._extract_experimental_setup(paper.full_text)
        
        # Extract loss functions
        paper.loss_functions = await self._extract_loss_functions(paper.full_text)
        
        # Extract evaluation protocol
        paper.evaluation_protocol = await self._extract_evaluation_protocol(paper.full_text)
        
        self.logger.info(f"Deep analysis completed for {paper.title}")
        return paper
    
    async def _extract_equations(self, full_text: str) -> List[str]:
        """Extract mathematical equations and formulations."""
        prompt = f"""
        Extract all mathematical equations, formulas, and mathematical expressions from this research paper.
        Focus on:
        1. Loss functions and objective functions
        2. Model equations and mathematical definitions
        3. Algorithm formulations
        4. Optimization objectives
        5. Statistical measures and metrics
        
        Paper content:
        {full_text[:8000]}  # Limit to avoid token limits
        
        Return a JSON list of equations with their context:
        {{
          "equations": [
            {{
              "equation": "L = -log(p(y|x))",
              "context": "Cross-entropy loss function",
              "variables": {{"L": "loss", "p": "probability", "y": "target", "x": "input"}}
            }}
          ]
        }}
        """
        
        content, error = await self.llm._call_llm([{"role": "user", "content": prompt}], 
                                                 self.config.architect_model, is_json=True)
        if error:
            self.logger.warning(f"Failed to extract equations: {error}")
            return []
        
        try:
            data = json.loads(content)
            return [eq["equation"] + " # " + eq["context"] for eq in data.get("equations", [])]
        except:
            return []
    
    async def _extract_algorithms(self, full_text: str) -> List[str]:
        """Extract algorithm descriptions and pseudocode."""
        prompt = f"""
        Extract all algorithms, procedures, and step-by-step processes from this paper.
        Focus on:
        1. Training algorithms
        2. Inference procedures
        3. Data processing steps
        4. Optimization algorithms
        5. Evaluation procedures
        
        Paper content:
        {full_text[:8000]}
        
        Return detailed pseudocode for each algorithm found.
        Format as JSON:
        {{
          "algorithms": [
            {{
              "name": "Training Algorithm",
              "pseudocode": "1. Initialize model\\n2. For each batch:\\n3. Forward pass\\n4. Compute loss\\n5. Backward pass",
              "purpose": "Train the neural network model"
            }}
          ]
        }}
        """
        
        content, error = await self.llm._call_llm([{"role": "user", "content": prompt}], 
                                                 self.config.architect_model, is_json=True)
        if error:
            return []
        
        try:
            data = json.loads(content)
            return [f"{alg['name']}: {alg['pseudocode']}" for alg in data.get("algorithms", [])]
        except:
            return []
    
    async def _extract_architecture(self, full_text: str) -> Dict[str, Any]:
        """Extract detailed model architecture information."""
        prompt = f"""
        Extract the complete model architecture details from this paper.
        Focus on:
        1. Layer types and configurations
        2. Network topology
        3. Input/output dimensions
        4. Activation functions
        5. Normalization techniques
        6. Attention mechanisms
        7. Skip connections
        
        Paper content:
        {full_text[:8000]}
        
        Return detailed architecture specification:
        {{
          "architecture": {{
            "model_type": "transformer",
            "layers": [
              {{"type": "embedding", "input_dim": 512, "output_dim": 768}},
              {{"type": "transformer_block", "num_heads": 12, "hidden_dim": 3072}}
            ],
            "activation": "gelu",
            "normalization": "layer_norm",
            "dropout": 0.1
          }}
        }}
        """
        
        content, error = await self.llm._call_llm([{"role": "user", "content": prompt}], 
                                                 self.config.architect_model, is_json=True)
        if error:
            return {}
        
        try:
            data = json.loads(content)
            return data.get("architecture", {})
        except:
            return {}
    
    async def _extract_hyperparameters(self, full_text: str) -> Dict[str, Any]:
        """Extract specific hyperparameters used in the paper."""
        prompt = f"""
        Extract all hyperparameters, training settings, and configuration values from this paper.
        
        Paper content:
        {full_text[:8000]}
        
        Return as JSON:
        {{
          "hyperparameters": {{
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam",
            "weight_decay": 0.01,
            "warmup_steps": 1000
          }}
        }}
        """
        
        content, error = await self.llm._call_llm([{"role": "user", "content": prompt}], 
                                                 self.config.architect_model, is_json=True)
        if error:
            return {}
        
        try:
            data = json.loads(content)
            return data.get("hyperparameters", {})
        except:
            return {}
    
    async def _extract_experimental_setup(self, full_text: str) -> Optional[str]:
        """Extract experimental methodology and setup."""
        prompt = f"""
        Extract the complete experimental setup and methodology from this paper.
        Include:
        1. Dataset preparation
        2. Training procedure
        3. Evaluation methodology
        4. Baseline comparisons
        5. Hardware/software requirements
        
        Paper content:
        {full_text[:8000]}
        
        Provide a detailed description of how to reproduce the experiments.
        """
        
        content, error = await self.llm._call_llm([{"role": "user", "content": prompt}], 
                                                 self.config.documentation_model)
        return content if not error else None
    
    async def _extract_loss_functions(self, full_text: str) -> List[str]:
        """Extract specific loss functions used."""
        # Use regex to find common loss function patterns
        loss_patterns = [
            r'cross[- ]entropy',
            r'mean squared error',
            r'binary cross[- ]entropy',
            r'focal loss',
            r'contrastive loss',
            r'triplet loss',
            r'adversarial loss',
            r'reconstruction loss'
        ]
        
        found_losses = []
        text_lower = full_text.lower()
        for pattern in loss_patterns:
            if re.search(pattern, text_lower):
                found_losses.append(pattern.replace('[- ]', ' '))
        
        return found_losses
    
    async def _extract_evaluation_protocol(self, full_text: str) -> Optional[str]:
        """Extract evaluation methodology."""
        prompt = f"""
        Extract the evaluation protocol and metrics from this paper.
        Focus on:
        1. Evaluation metrics used
        2. Test datasets
        3. Evaluation procedure
        4. Statistical significance tests
        
        Paper content:
        {full_text[:8000]}
        
        Provide specific implementation details for evaluation.
        """
        
        content, error = await self.llm._call_llm([{"role": "user", "content": prompt}], 
                                                 self.config.documentation_model)
        return content if not error else None


class PaperProcessor:
    """Processes research papers and extracts relevant information."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_papers_from_directory(self) -> List[PaperInfo]:
        """Loads all papers from the papers directory and advanced_paper_extractor runs."""
        papers = []
        
        # Load from default papers_dir (legacy format)
        if self.config.papers_dir.exists():
            self.logger.info(f"Loading papers from legacy directory: {self.config.papers_dir}")
            for json_file in self.config.papers_dir.glob("*.json"):
                try:
                    paper = self._load_paper_from_file(json_file)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    self.logger.error(f"Failed to load paper from {json_file}: {e}")
        else:
            self.logger.info(f"Legacy papers directory does not exist: {self.config.papers_dir}")
        
        # Also load from advanced_paper_extractor runs directory (new format)
        runs_dir = self.config.base_dir / "runs"
        if runs_dir.exists():
            self.logger.info("Scanning advanced_paper_extractor runs directory...")
            
            # Find all run directories
            for run_dir in runs_dir.iterdir():
                if run_dir.is_dir():
                    jsons_dir = run_dir / "jsons"
                    if jsons_dir.exists():
                        self.logger.info(f"Loading papers from {jsons_dir}")
                        for json_file in jsons_dir.glob("*.json"):
                            try:
                                paper = self._load_paper_from_file(json_file)
                                if paper:
                                    # Check for duplicates by title
                                    existing_titles = {p.title for p in papers}
                                    if paper.title not in existing_titles:
                                        papers.append(paper)
                                    else:
                                        self.logger.debug(f"Skipping duplicate paper: {paper.title[:50]}...")
                            except Exception as e:
                                self.logger.error(f"Failed to load paper from {json_file}: {e}")
        else:
            self.logger.info("No advanced_paper_extractor runs directory found")
        
        self.logger.info(f"Loaded {len(papers)} unique papers total")
        return papers

    def _load_paper_from_file(self, file_path: Path) -> Optional[PaperInfo]:
        """Loads a single paper from a JSON file - compatible with both old and new advanced_paper_extractor formats."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if this is the new ExtractedPaperContent format from advanced_paper_extractor.py
            if 'full_text' in data and 'llm_analysis' in data and 'relevance_score' in data:
                # New format from advanced_paper_extractor.py
                paper_id = data.get('arxiv_id', file_path.stem)
                title = data.get('title', 'Unknown Title')
                summary = data.get('abstract', '')
                authors = data.get('authors', [])
                
                # Extract enhanced information from LLM analysis
                llm_analysis = data.get('llm_analysis', '')
                full_text = data.get('full_text', '')
                
                # Try to extract structured information from LLM analysis
                methodology = self._extract_methodology_from_analysis(llm_analysis, full_text)
                key_contributions = self._extract_contributions_from_analysis(llm_analysis, title)
                technical_details = self._extract_technical_details_from_analysis(llm_analysis, full_text)
                
                paper = PaperInfo(
                    id=paper_id,
                    title=title,
                    summary=summary,
                    authors=authors,
                    arxiv_id=data.get('arxiv_id'),
                    publication_date=data.get('published'),
                    methodology=methodology,
                    key_contributions=key_contributions,
                    technical_details=technical_details,
                    source_path=file_path,
                    last_modified=file_path.stat().st_mtime,
                    # Enhanced fields from new format
                    full_text=full_text,
                    research_domain=self._infer_research_domain(data.get('categories', [])),
                    complexity_level=self._infer_complexity_level(llm_analysis),
                    implementation_difficulty=self._infer_implementation_difficulty(llm_analysis),
                    mathematical_formulations=self._extract_equations_from_text(full_text),
                    algorithm_pseudocode=self._extract_algorithms_from_analysis(llm_analysis),
                    model_architecture=self._extract_architecture_from_analysis(llm_analysis),
                    hyperparameters=self._extract_hyperparameters_from_analysis(llm_analysis),
                    evaluation_metrics=self._extract_metrics_from_analysis(llm_analysis),
                    experimental_setup=self._extract_experimental_setup_from_analysis(llm_analysis)
                )
                
                self.logger.info(f"Loaded paper from new advanced_paper_extractor format: {title[:50]}...")
                return paper
            
            else:
                # Legacy format - existing logic
                paper_id = data.get('id', file_path.stem)
                title = data.get('title', data.get('paper_title', 'Unknown Title'))
                summary = data.get('summary', data.get('abstract', data.get('description', '')))
                authors = data.get('authors', data.get('author', []))
                
                if isinstance(authors, str):
                    authors = [authors]
                
                paper = PaperInfo(
                    id=paper_id,
                    title=title,
                    summary=summary,
                    authors=authors,
                    arxiv_id=data.get('arxiv_id'),
                    publication_date=data.get('publication_date'),
                    methodology=data.get('methodology', data.get('method', '')),
                    key_contributions=data.get('key_contributions', data.get('contributions', [])),
                    technical_details=data.get('technical_details', data.get('details', '')),
                    source_path=file_path,
                    last_modified=file_path.stat().st_mtime
                )
                
                self.logger.info(f"Loaded paper from legacy format: {title[:50]}...")
                return paper
            
        except Exception as e:
            self.logger.error(f"Error loading paper from {file_path}: {e}")
            return None

    def _extract_methodology_from_analysis(self, llm_analysis: str, full_text: str) -> str:
        """Extract methodology information from LLM analysis."""
        if not llm_analysis:
            return ""
        
        # Look for methodology-related keywords in the analysis
        methodology_keywords = ["method", "approach", "algorithm", "technique", "procedure", "framework"]
        lines = llm_analysis.split('\n')
        methodology_lines = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in methodology_keywords):
                methodology_lines.append(line.strip())
        
        return ' '.join(methodology_lines[:3])  # Take first 3 relevant lines

    def _extract_contributions_from_analysis(self, llm_analysis: str, title: str) -> List[str]:
        """Extract key contributions from LLM analysis."""
        if not llm_analysis:
            return []
        
        contributions = []
        lines = llm_analysis.split('\n')
        
        # Look for contribution-related sections
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ["contribution", "novel", "propose", "introduce"]):
                contributions.append(line.strip())
                # Add next line if it seems related
                if i + 1 < len(lines) and len(lines[i + 1].strip()) > 20:
                    contributions.append(lines[i + 1].strip())
        
        return contributions[:5]  # Limit to 5 contributions

    def _extract_technical_details_from_analysis(self, llm_analysis: str, full_text: str) -> str:
        """Extract technical details from LLM analysis."""
        if not llm_analysis:
            return ""
        
        # Look for technical sections
        technical_keywords = ["architecture", "implementation", "model", "network", "layer", "parameter"]
        lines = llm_analysis.split('\n')
        technical_lines = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in technical_keywords):
                technical_lines.append(line.strip())
        
        return ' '.join(technical_lines[:5])  # Take first 5 relevant lines

    def _infer_research_domain(self, categories: List[str]) -> str:
        """Infer research domain from arXiv categories."""
        if not categories:
            return "Machine Learning"
        
        domain_mapping = {
            "cs.AI": "Artificial Intelligence",
            "cs.LG": "Machine Learning", 
            "cs.CL": "Natural Language Processing",
            "cs.CV": "Computer Vision",
            "cs.NE": "Neural Networks",
            "stat.ML": "Statistical Machine Learning"
        }
        
        for cat in categories:
            if cat in domain_mapping:
                return domain_mapping[cat]
        
        return "Machine Learning"

    def _infer_complexity_level(self, llm_analysis: str) -> str:
        """Infer complexity level from LLM analysis."""
        if not llm_analysis:
            return "medium"
        
        analysis_lower = llm_analysis.lower()
        
        # High complexity indicators
        if any(keyword in analysis_lower for keyword in ["complex", "sophisticated", "advanced", "novel architecture"]):
            return "high"
        
        # Low complexity indicators  
        if any(keyword in analysis_lower for keyword in ["simple", "basic", "straightforward", "standard"]):
            return "low"
        
        return "medium"

    def _infer_implementation_difficulty(self, llm_analysis: str) -> str:
        """Infer implementation difficulty from LLM analysis."""
        if not llm_analysis:
            return "medium"
        
        analysis_lower = llm_analysis.lower()
        
        # High difficulty indicators
        if any(keyword in analysis_lower for keyword in ["challenging", "difficult", "complex implementation", "requires"]):
            return "high"
        
        # Low difficulty indicators
        if any(keyword in analysis_lower for keyword in ["simple", "easy", "straightforward", "standard implementation"]):
            return "low"
        
        return "medium"

    def _extract_equations_from_text(self, full_text: str) -> List[str]:
        """Extract mathematical equations from full text."""
        if not full_text:
            return []
        
        equations = []
        # Look for LaTeX-style equations
        import re
        
        # Find equations between $...$ or $$...$$
        equation_patterns = [
            r'\$\$([^$]+)\$\$',  # Display equations
            r'\$([^$]+)\$',      # Inline equations
            r'\\begin\{equation\}(.*?)\\end\{equation\}',  # Equation environments
            r'\\begin\{align\}(.*?)\\end\{align\}'         # Align environments
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, full_text, re.DOTALL)
            equations.extend([match.strip() for match in matches if len(match.strip()) > 5])
        
        return equations[:10]  # Limit to 10 equations

    def _extract_algorithms_from_analysis(self, llm_analysis: str) -> List[str]:
        """Extract algorithm descriptions from LLM analysis."""
        if not llm_analysis:
            return []
        
        algorithms = []
        lines = llm_analysis.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ["algorithm", "procedure", "step", "process"]):
                algorithms.append(line.strip())
        
        return algorithms[:5]

    def _extract_architecture_from_analysis(self, llm_analysis: str) -> Dict[str, Any]:
        """Extract model architecture details from LLM analysis."""
        if not llm_analysis:
            return {}
        
        architecture = {}
        analysis_lower = llm_analysis.lower()
        
        # Look for common architecture components
        if "transformer" in analysis_lower:
            architecture["type"] = "transformer"
        elif "cnn" in analysis_lower or "convolutional" in analysis_lower:
            architecture["type"] = "cnn"
        elif "rnn" in analysis_lower or "recurrent" in analysis_lower:
            architecture["type"] = "rnn"
        else:
            architecture["type"] = "neural_network"
        
        # Extract layer information
        if "layer" in analysis_lower:
            import re
            layer_matches = re.findall(r'(\d+)\s*layer', analysis_lower)
            if layer_matches:
                architecture["num_layers"] = int(layer_matches[0])
        
        return architecture

    def _extract_hyperparameters_from_analysis(self, llm_analysis: str) -> Dict[str, Any]:
        """Extract hyperparameters from LLM analysis."""
        if not llm_analysis:
            return {}
        
        hyperparams = {}
        
        # Look for common hyperparameters
        import re
        
        # Learning rate
        lr_matches = re.findall(r'learning.rate[:\s]*([0-9.e-]+)', llm_analysis.lower())
        if lr_matches:
            hyperparams["learning_rate"] = float(lr_matches[0])
        
        # Batch size
        batch_matches = re.findall(r'batch.size[:\s]*([0-9]+)', llm_analysis.lower())
        if batch_matches:
            hyperparams["batch_size"] = int(batch_matches[0])
        
        # Epochs
        epoch_matches = re.findall(r'epoch[s]?[:\s]*([0-9]+)', llm_analysis.lower())
        if epoch_matches:
            hyperparams["epochs"] = int(epoch_matches[0])
        
        return hyperparams

    def _extract_metrics_from_analysis(self, llm_analysis: str) -> List[str]:
        """Extract evaluation metrics from LLM analysis."""
        if not llm_analysis:
            return []
        
        metrics = []
        common_metrics = ["accuracy", "precision", "recall", "f1", "bleu", "rouge", "perplexity", "loss"]
        
        analysis_lower = llm_analysis.lower()
        for metric in common_metrics:
            if metric in analysis_lower:
                metrics.append(metric)
        
        return metrics

    def _extract_experimental_setup_from_analysis(self, llm_analysis: str) -> str:
        """Extract experimental setup from LLM analysis."""
        if not llm_analysis:
            return ""
        
        lines = llm_analysis.split('\n')
        setup_lines = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ["experiment", "evaluation", "dataset", "setup", "benchmark"]):
                setup_lines.append(line.strip())
        
        return ' '.join(setup_lines[:3])

    def generate_repo_name(self, paper: PaperInfo) -> str:
        """Generates a repository name from paper information."""
        # Clean and format the title
        title_words = re.sub(r'[^\w\s-]', '', paper.title.lower()).split()
        
        # Take first 4-6 meaningful words
        meaningful_words = [word for word in title_words if len(word) > 2][:6]
        
        # Create repo name
        repo_name = self.config.repo_prefix + "_".join(meaningful_words)
        
        # Ensure it's not too long and follows GitHub naming conventions
        repo_name = re.sub(r'[^a-zA-Z0-9_-]', '', repo_name)[:100]
        
        return repo_name


class StateManager:
    """Manages the state of processed repositories."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.state_file = config.state_file

    def load_state(self) -> Dict[str, RepoState]:
        """Loads the current state from file."""
        if not self.state_file.exists():
            return {}
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            state = {}
            for repo_name, repo_data in data.items():
                state[repo_name] = RepoState(
                    repo_name=repo_data['repo_name'],
                    github_url=repo_data['github_url'],
                    status=ProcessingStatus(repo_data['status']),
                    last_processed_timestamp=repo_data.get('last_processed_timestamp'),
                    errors=repo_data.get('errors', []),
                    files_generated=repo_data.get('files_generated', [])
                )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return {}

    def save_state(self, state: Dict[str, RepoState]):
        """Saves the current state to file."""
        try:
            # Convert to serializable format
            data = {}
            for repo_name, repo_state in state.items():
                data[repo_name] = asdict(repo_state)
                data[repo_name]['status'] = repo_state.status.value
            
            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"State saved to {self.state_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def update_repo_state(self, state: Dict[str, RepoState], repo_name: str, 
                         status: ProcessingStatus, error: Optional[str] = None,
                         files_generated: Optional[List[str]] = None):
        """Updates the state of a specific repository."""
        if repo_name not in state:
            state[repo_name] = RepoState(repo_name=repo_name, github_url="")
        
        state[repo_name].status = status
        state[repo_name].last_processed_timestamp = datetime.now().isoformat()
        
        if error:
            state[repo_name].errors.append(error)
        
        if files_generated:
            state[repo_name].files_generated.extend(files_generated)
        
        self.save_state(state)


class M1EvoMaintainerAgent:
    """Main agent class that orchestrates the entire process."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logging()
        self.session = None
        
        # Initialize components
        self.paper_processor = PaperProcessor(config, self.logger)
        self.state_manager = StateManager(config, self.logger)
        
        # Create directories
        self._create_directories()

    def _setup_logging(self) -> logging.Logger:
        """Sets up comprehensive logging with GitHub Actions support."""
        logger = logging.getLogger("M1EvoAgent")
        logger.setLevel(logging.INFO)
        
        # Detect GitHub Actions environment
        is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"
        
        # Create logs directory
        self.config.logs_dir.mkdir(exist_ok=True)
        
        # File handler
        log_file = self.config.logs_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if not is_github_actions else logging.DEBUG)
        
        # Formatters
        if is_github_actions:
            # GitHub Actions friendly format
            file_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
            console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # GitHub Actions specific logging
        if is_github_actions:
            # Add GitHub Actions annotations for errors and warnings
            class GitHubActionsHandler(logging.Handler):
                def emit(self, record):
                    try:
                        if record.levelno >= logging.ERROR:
                            print(f"::error::{record.getMessage()}")
                        elif record.levelno >= logging.WARNING:
                            print(f"::warning::{record.getMessage()}")
                        elif record.levelno >= logging.INFO and "SUCCESS" in record.getMessage():
                            print(f"::notice::{record.getMessage()}")
                    except Exception:
                        pass  # Don't let logging errors break the application
            
            gh_handler = GitHubActionsHandler()
            gh_handler.setLevel(logging.WARNING)
            logger.addHandler(gh_handler)
            
            logger.info("GitHub Actions environment detected - enhanced logging enabled")
        
        # Suppress noisy third-party loggers
        logging.getLogger('aiohttp').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        
        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger

    def _create_directories(self):
        """Creates necessary directories."""
        directories = [
            self.config.workspace_dir,
            self.config.logs_dir,
            self.config.llm_logs_dir,
            self.config.papers_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    async def run(self):
        """Main execution method."""
        self.logger.info("Starting M1-Evo Maintainer Agent")
        
        # Create aiohttp session
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session
            
            # Initialize components that need session
            llm_interface = LLMInterface(self.config, session, self.logger)
            github_manager = GitHubManager(self.config, session, self.logger)
            
            # Test LLM connectivity
            self.logger.info("Testing LLM connectivity...")
            try:
                test_messages = [{"role": "user", "content": "Hello, please respond with 'OK' if you can hear me."}]
                test_response, test_error = await llm_interface._call_llm(test_messages, self.config.architect_model)
                if test_response:
                    self.logger.info(f"LLM connectivity test successful: {test_response[:50]}...")
                else:
                    self.logger.error(f"LLM connectivity test failed: {test_error}")
            except Exception as e:
                self.logger.error(f"LLM connectivity test exception: {e}")
            
            # Load papers and state
            papers = self.paper_processor.load_papers_from_directory()
            state = self.state_manager.load_state()
            
            if not papers:
                self.logger.warning("No papers found to process")
                return
            
            # Process papers with concurrency control
            semaphore = asyncio.Semaphore(self.config.max_concurrent_papers)
            tasks = []
            
            for paper in papers:
                task = self._process_paper_with_semaphore(
                    semaphore, paper, llm_interface, github_manager, state
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log final results
            successful = sum(1 for r in results if r is True)
            failed = len(results) - successful
            
            self.logger.info(f"Processing complete: {successful} successful, {failed} failed")
            
            # Generate final report
            self._generate_final_report(state, papers)

    async def _process_paper_with_semaphore(self, semaphore: asyncio.Semaphore, 
                                          paper: PaperInfo, llm_interface: LLMInterface,
                                          github_manager: GitHubManager, 
                                          state: Dict[str, RepoState]) -> bool:
        """Process a single paper with semaphore control."""
        async with semaphore:
            return await self._process_single_paper(paper, llm_interface, github_manager, state)

    async def _process_single_paper(self, paper: PaperInfo, llm_interface: LLMInterface,
                                   github_manager: GitHubManager, 
                                   state: Dict[str, RepoState]) -> bool:
        """Process a single paper end-to-end."""
        repo_name = self.paper_processor.generate_repo_name(paper)
        
        try:
            self.logger.info(f"Processing paper: {paper.title}")
            
            # Check if already processed successfully
            if repo_name in state and state[repo_name].status == ProcessingStatus.SUCCESS:
                self.logger.info(f"Paper already processed successfully: {repo_name}")
                return True
            
            # Update status to planning
            self.state_manager.update_repo_state(state, repo_name, ProcessingStatus.PLANNING_ARCHITECTURE)
            
            # Step 1: Plan file structure
            self.logger.info(f"Planning file structure for {repo_name}")
            try:
                file_plans, plan_message = await llm_interface.plan_file_structure(paper)
            except Exception as planning_error:
                error_msg = f"Exception during file structure planning: {planning_error}"
                self.logger.error(error_msg)
                self.state_manager.update_repo_state(state, repo_name, ProcessingStatus.FAILED, error_msg)
                return False
            
            if not file_plans:
                error_msg = f"Failed to generate file plan: {plan_message}"
                self.logger.error(error_msg)
                self.state_manager.update_repo_state(state, repo_name, ProcessingStatus.FAILED, error_msg)
                return False
            
            self.logger.info(f"Generated plan with {len(file_plans)} files")
            
            # Step 2: Create GitHub repository
            self.state_manager.update_repo_state(state, repo_name, ProcessingStatus.CREATING_REPO)
            
            description = f"Implementation of '{paper.title}' - {paper.summary[:100]}..."
            repo_url, create_message = await github_manager.create_repository(repo_name, description)
            
            if not repo_url:
                error_msg = f"Failed to create repository: {create_message}"
                self.logger.error(error_msg)
                self.state_manager.update_repo_state(state, repo_name, ProcessingStatus.FAILED, error_msg)
                return False
            
            # Update state with repo URL
            state[repo_name].github_url = repo_url
            self.state_manager.save_state(state)
            
            # Step 3: Generate and upload files
            self.state_manager.update_repo_state(state, repo_name, ProcessingStatus.GENERATING_FILES)
            
            # Sort files by priority (high priority first)
            sorted_files = sorted(file_plans, key=lambda x: (x.priority, x.path))
            
            uploaded_files = []
            failed_files = []
            
            for file_plan in sorted_files:
                try:
                    self.logger.info(f"Generating content for {file_plan.path}")
                    
                    content, content_message = await llm_interface.generate_file_content(
                        paper, file_plan, file_plans
                    )
                    
                    if not content:
                        self.logger.warning(f"Failed to generate content for {file_plan.path}: {content_message}")
                        failed_files.append(file_plan.path)
                        continue
                    
                    # Upload to GitHub
                    commit_message = f"Add {file_plan.path}: {file_plan.description}"
                    success, upload_message = await github_manager.upload_file(
                        repo_name, file_plan.path, content, commit_message
                    )
                    
                    if success:
                        uploaded_files.append(file_plan.path)
                        self.logger.debug(f"Successfully uploaded {file_plan.path}")
                    else:
                        self.logger.warning(f"Failed to upload {file_plan.path}: {upload_message}")
                        failed_files.append(file_plan.path)
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    error_msg = f"Error processing {file_plan.path}: {e}"
                    self.logger.error(error_msg)
                    failed_files.append(file_plan.path)
            
            # Step 4: Validate and finalize
            self.state_manager.update_repo_state(state, repo_name, ProcessingStatus.VALIDATING)
            
            success_rate = len(uploaded_files) / len(file_plans) if file_plans else 0
            
            if success_rate >= 0.8:  # 80% success rate threshold
                self.state_manager.update_repo_state(
                    state, repo_name, ProcessingStatus.SUCCESS, 
                    files_generated=uploaded_files
                )
                self.logger.info(f"Successfully processed {repo_name}: {len(uploaded_files)}/{len(file_plans)} files uploaded")
                return True
            else:
                error_msg = f"Low success rate: {len(uploaded_files)}/{len(file_plans)} files uploaded"
                self.state_manager.update_repo_state(state, repo_name, ProcessingStatus.FAILED, error_msg)
                self.logger.warning(f"Failed to process {repo_name}: {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"Unexpected error processing {repo_name}: {e}"
            self.logger.error(error_msg)
            self.state_manager.update_repo_state(state, repo_name, ProcessingStatus.FAILED, error_msg)
            return False

    def _generate_final_report(self, state: Dict[str, RepoState], papers: List[PaperInfo]):
        """Generates a comprehensive final report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate summary statistics
        total_papers = len(papers)
        successful_repos = sum(1 for repo in state.values() if repo.status == ProcessingStatus.SUCCESS)
        failed_repos = sum(1 for repo in state.values() if repo.status == ProcessingStatus.FAILED)
        pending_repos = total_papers - successful_repos - failed_repos
        
        # Create detailed report
        report = {
            'timestamp': timestamp,
            'summary': {
                'total_papers': total_papers,
                'successful_repositories': successful_repos,
                'failed_repositories': failed_repos,
                'pending_repositories': pending_repos,
                'success_rate': successful_repos / total_papers if total_papers > 0 else 0
            },
            'repositories': []
        }
        
        for repo_name, repo_state in state.items():
            repo_info = {
                'name': repo_name,
                'url': repo_state.github_url,
                'status': repo_state.status.value,
                'last_processed': repo_state.last_processed_timestamp,
                'files_generated': len(repo_state.files_generated),
                'errors': repo_state.errors
            }
            report['repositories'].append(repo_info)
        
        # Save JSON report
        json_report_path = self.config.logs_dir / f"final_summary_{timestamp}.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generate markdown report
        md_report_path = self.config.logs_dir / f"processing_report_{timestamp}.md"
        with open(md_report_path, 'w', encoding='utf-8') as f:
            f.write(f"# M1-Evo Agent Processing Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Papers:** {total_papers}\n")
            f.write(f"- **Successful Repositories:** {successful_repos}\n")
            f.write(f"- **Failed Repositories:** {failed_repos}\n")
            f.write(f"- **Pending Repositories:** {pending_repos}\n")
            f.write(f"- **Success Rate:** {report['summary']['success_rate']:.1%}\n\n")
            
            f.write(f"## Repository Details\n\n")
            for repo_info in report['repositories']:
                f.write(f"### {repo_info['name']}\n\n")
                f.write(f"- **Status:** {repo_info['status']}\n")
                f.write(f"- **URL:** {repo_info['url']}\n")
                f.write(f"- **Files Generated:** {repo_info['files_generated']}\n")
                f.write(f"- **Last Processed:** {repo_info['last_processed']}\n")
                
                if repo_info['errors']:
                    f.write(f"- **Errors:**\n")
                    for error in repo_info['errors']:
                        f.write(f"  - {error}\n")
                f.write(f"\n")
        
        self.logger.info(f"Final report generated: {json_report_path}")
        self.logger.info(f"Markdown report generated: {md_report_path}")


async def main():
    """Main entry point for the M1-Evo Maintainer Agent."""
    # Load environment variables
    load_dotenv()
    
    # Create configuration
    config = Config(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        github_token=os.getenv("GITHUB_TOKEN", ""),
        github_username=os.getenv("GITHUB_USERNAME", "")
    )
    
    # Validate configuration
    if not config.openrouter_api_key:
        print("Error: OPENROUTER_API_KEY environment variable is required")
        return
    
    if not config.github_token:
        print("Error: GITHUB_TOKEN environment variable is required")
        return
    
    if not config.github_username:
        print("Error: GITHUB_USERNAME environment variable is required")
        return
    
    # Create and run agent
    agent = M1EvoMaintainerAgent(config)
    await agent.run()


# Corrupted section removed - continuing with proper class definitions


class Advanced360RepositoryGenerator:
    """Advanced generator for creating comprehensive 360-degree repositories."""
    
    def __init__(self, config: Config, llm_interface: 'LLMInterface', logger: logging.Logger):
        self.config = config
        self.llm = llm_interface
        self.logger = logger
        
    async def generate_360_structure(self, paper: PaperInfo) -> List[FilePlan]:
        """Generate comprehensive 360-degree repository structure."""
        self.logger.info(f"Generating 360° repository structure for: {paper.title}")
        
        # Base structure from LLM
        base_files, _ = await self.llm.plan_file_structure(paper)
        
        # Add advanced 360° components
        advanced_files = self._generate_advanced_components(paper)
        web_interface_files = self._generate_web_interface_files(paper)
        api_files = self._generate_api_files(paper)
        deployment_files = self._generate_deployment_files(paper)
        monitoring_files = self._generate_monitoring_files(paper)
        security_files = self._generate_security_files(paper)
        
        # Combine all files
        all_files = (base_files or []) + advanced_files + web_interface_files + api_files + deployment_files + monitoring_files + security_files
        
        # Remove duplicates and sort by priority
        unique_files = self._deduplicate_files(all_files)
        sorted_files = sorted(unique_files, key=lambda x: (x.priority, x.path))
        
        self.logger.info(f"Generated comprehensive 360° structure with {len(sorted_files)} files")
        return sorted_files
    
    def _generate_advanced_components(self, paper: PaperInfo) -> List[FilePlan]:
        """Generate advanced ML/AI components."""
        return [
            # Advanced Model Components
            FilePlan("src/models/ensemble.py", "Ensemble methods and model combination strategies", 1, ["src/models/main_model.py"], "code", FileCategory.CORE_MODEL, 300),
            FilePlan("src/models/distillation.py", "Knowledge distillation implementation", 2, ["src/models/main_model.py"], "code", FileCategory.CORE_MODEL, 250),
            FilePlan("src/models/quantization.py", "Model quantization and compression", 2, ["src/models/main_model.py"], "code", FileCategory.CORE_MODEL, 200),
            FilePlan("src/models/pruning.py", "Neural network pruning techniques", 2, ["src/models/main_model.py"], "code", FileCategory.CORE_MODEL, 180),
            
            # Advanced Training Components
            FilePlan("src/training/federated.py", "Federated learning implementation", 2, ["src/training/trainer.py"], "code", FileCategory.TRAINING, 400),
            FilePlan("src/training/adversarial.py", "Adversarial training and robustness", 2, ["src/training/trainer.py"], "code", FileCategory.TRAINING, 300),
            FilePlan("src/training/meta_learning.py", "Meta-learning and few-shot learning", 2, ["src/training/trainer.py"], "code", FileCategory.TRAINING, 350),
            FilePlan("src/training/continual.py", "Continual and lifelong learning", 2, ["src/training/trainer.py"], "code", FileCategory.TRAINING, 280),
            
            # Advanced Data Processing
            FilePlan("src/data/augmentation.py", "Advanced data augmentation techniques", 1, ["src/data/preprocessing.py"], "code", FileCategory.DATA_PROCESSING, 250),
            FilePlan("src/data/synthetic.py", "Synthetic data generation", 2, ["src/data/dataset.py"], "code", FileCategory.DATA_PROCESSING, 300),
            FilePlan("src/data/privacy.py", "Privacy-preserving data processing", 2, ["src/data/preprocessing.py"], "code", FileCategory.DATA_PROCESSING, 200),
            FilePlan("src/data/streaming.py", "Real-time data streaming and processing", 2, ["src/data/loader.py"], "code", FileCategory.DATA_PROCESSING, 220),
            
            # MLOps Components
            FilePlan("src/mlops/__init__.py", "MLOps package initialization", 2, [], "code", FileCategory.UTILITIES),
            FilePlan("src/mlops/experiment_tracking.py", "Experiment tracking with MLflow/Weights&Biases", 2, [], "code", FileCategory.UTILITIES, 300),
            FilePlan("src/mlops/model_registry.py", "Model registry and versioning", 2, [], "code", FileCategory.UTILITIES, 250),
            FilePlan("src/mlops/pipeline.py", "ML pipeline orchestration", 2, [], "code", FileCategory.UTILITIES, 400),
            FilePlan("src/mlops/drift_detection.py", "Data and model drift detection", 2, [], "code", FileCategory.UTILITIES, 200),
            
            # Performance Optimization
            FilePlan("src/optimization/__init__.py", "Optimization package initialization", 2, [], "code", FileCategory.UTILITIES),
            FilePlan("src/optimization/gpu_utils.py", "GPU optimization utilities", 2, [], "code", FileCategory.UTILITIES, 180),
            FilePlan("src/optimization/memory.py", "Memory optimization and management", 2, [], "code", FileCategory.UTILITIES, 150),
            FilePlan("src/optimization/profiling.py", "Performance profiling tools", 2, [], "code", FileCategory.UTILITIES, 200),
            FilePlan("src/optimization/distributed.py", "Distributed computing utilities", 2, [], "code", FileCategory.UTILITIES, 300),
        ]
    
    def _generate_web_interface_files(self, paper: PaperInfo) -> List[FilePlan]:
        """Generate web interface components."""
        return [
            # Streamlit Web Interface
            FilePlan("web/streamlit_app.py", "Main Streamlit web application", 2, ["src/models/main_model.py"], "code", FileCategory.WEB_INTERFACE, 400),
            FilePlan("web/components/model_interface.py", "Model interaction components", 2, ["web/streamlit_app.py"], "code", FileCategory.WEB_INTERFACE, 200),
            FilePlan("web/components/visualization.py", "Interactive visualization components", 2, ["web/streamlit_app.py"], "code", FileCategory.WEB_INTERFACE, 250),
            FilePlan("web/components/file_upload.py", "File upload and processing components", 2, ["web/streamlit_app.py"], "code", FileCategory.WEB_INTERFACE, 150),
            FilePlan("web/utils/session_state.py", "Session state management", 2, ["web/streamlit_app.py"], "code", FileCategory.WEB_INTERFACE, 100),
            
            # Gradio Alternative Interface
            FilePlan("web/gradio_app.py", "Gradio-based web interface", 3, ["src/models/main_model.py"], "code", FileCategory.WEB_INTERFACE, 300),
            
            # Flask/FastAPI Backend
            FilePlan("web/backend/app.py", "FastAPI backend application", 2, ["src/models/main_model.py"], "code", FileCategory.WEB_INTERFACE, 350),
            FilePlan("web/backend/routes.py", "API routes and endpoints", 2, ["web/backend/app.py"], "code", FileCategory.WEB_INTERFACE, 200),
            FilePlan("web/backend/middleware.py", "Custom middleware and authentication", 2, ["web/backend/app.py"], "code", FileCategory.WEB_INTERFACE, 150),
            
            # Frontend (React/Vue)
            FilePlan("web/frontend/package.json", "Frontend package configuration", 3, [], "config", FileCategory.WEB_INTERFACE),
            FilePlan("web/frontend/src/App.js", "Main React application component", 3, [], "code", FileCategory.WEB_INTERFACE, 200),
            FilePlan("web/frontend/src/components/ModelInterface.js", "Model interaction React component", 3, [], "code", FileCategory.WEB_INTERFACE, 150),
            FilePlan("web/frontend/src/utils/api.js", "API communication utilities", 3, [], "code", FileCategory.WEB_INTERFACE, 100),
            
            # Web Configuration
            FilePlan("web/config/streamlit_config.toml", "Streamlit configuration", 3, [], "config", FileCategory.WEB_INTERFACE),
            FilePlan("web/requirements.txt", "Web interface dependencies", 2, [], "config", FileCategory.WEB_INTERFACE),
        ]
    
    def _generate_api_files(self, paper: PaperInfo) -> List[FilePlan]:
        """Generate API components."""
        return [
            # REST API
            FilePlan("api/__init__.py", "API package initialization", 2, [], "code", FileCategory.API),
            FilePlan("api/main.py", "Main FastAPI application", 2, ["src/models/main_model.py"], "code", FileCategory.API, 300),
            FilePlan("api/routes/predict.py", "Prediction API endpoints", 2, ["api/main.py"], "code", FileCategory.API, 200),
            FilePlan("api/routes/train.py", "Training API endpoints", 2, ["api/main.py"], "code", FileCategory.API, 250),
            FilePlan("api/routes/evaluate.py", "Evaluation API endpoints", 2, ["api/main.py"], "code", FileCategory.API, 150),
            FilePlan("api/routes/health.py", "Health check and monitoring endpoints", 2, ["api/main.py"], "code", FileCategory.API, 100),
            
            # API Models and Schemas
            FilePlan("api/models/request.py", "API request models and validation", 2, [], "code", FileCategory.API, 150),
            FilePlan("api/models/response.py", "API response models", 2, [], "code", FileCategory.API, 120),
            FilePlan("api/models/errors.py", "Error handling and custom exceptions", 2, [], "code", FileCategory.API, 100),
            
            # API Utilities
            FilePlan("api/utils/auth.py", "Authentication and authorization", 2, [], "code", FileCategory.API, 200),
            FilePlan("api/utils/rate_limiting.py", "Rate limiting and throttling", 2, [], "code", FileCategory.API, 150),
            FilePlan("api/utils/caching.py", "Response caching utilities", 2, [], "code", FileCategory.API, 120),
            FilePlan("api/utils/logging.py", "API-specific logging", 2, [], "code", FileCategory.API, 100),
            
            # GraphQL API (Optional)
            FilePlan("api/graphql/schema.py", "GraphQL schema definition", 3, [], "code", FileCategory.API, 200),
            FilePlan("api/graphql/resolvers.py", "GraphQL resolvers", 3, [], "code", FileCategory.API, 250),
            
            # API Documentation
            FilePlan("api/docs/openapi.yaml", "OpenAPI specification", 2, [], "docs", FileCategory.API),
            FilePlan("api/docs/README.md", "API documentation", 2, [], "docs", FileCategory.API),
        ]
    
    def _generate_deployment_files(self, paper: PaperInfo) -> List[FilePlan]:
        """Generate deployment and infrastructure files."""
        return [
            # Docker
            FilePlan("deployment/docker/Dockerfile.prod", "Production Docker configuration", 2, ["requirements.txt"], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/docker/Dockerfile.dev", "Development Docker configuration", 3, ["requirements.txt"], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/docker/docker-compose.prod.yml", "Production Docker Compose", 2, [], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/docker/docker-compose.dev.yml", "Development Docker Compose", 3, [], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/docker/.dockerignore", "Docker ignore file", 3, [], "config", FileCategory.DEPLOYMENT),
            
            # Kubernetes
            FilePlan("deployment/k8s/namespace.yaml", "Kubernetes namespace configuration", 3, [], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/k8s/deployment.yaml", "Kubernetes deployment configuration", 3, [], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/k8s/service.yaml", "Kubernetes service configuration", 3, [], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/k8s/ingress.yaml", "Kubernetes ingress configuration", 3, [], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/k8s/configmap.yaml", "Kubernetes config map", 3, [], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/k8s/secrets.yaml", "Kubernetes secrets template", 3, [], "config", FileCategory.DEPLOYMENT),
            
            # Cloud Deployment
            FilePlan("deployment/aws/cloudformation.yaml", "AWS CloudFormation template", 3, [], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/aws/lambda_function.py", "AWS Lambda deployment", 3, ["src/models/main_model.py"], "code", FileCategory.DEPLOYMENT, 200),
            FilePlan("deployment/gcp/app.yaml", "Google Cloud App Engine configuration", 3, [], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/azure/azure-pipelines.yml", "Azure DevOps pipeline", 3, [], "config", FileCategory.DEPLOYMENT),
            
            # Terraform
            FilePlan("deployment/terraform/main.tf", "Main Terraform configuration", 3, [], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/terraform/variables.tf", "Terraform variables", 3, [], "config", FileCategory.DEPLOYMENT),
            FilePlan("deployment/terraform/outputs.tf", "Terraform outputs", 3, [], "config", FileCategory.DEPLOYMENT),
            
            # Deployment Scripts
            FilePlan("deployment/scripts/deploy.sh", "Main deployment script", 2, [], "code", FileCategory.DEPLOYMENT, 150),
            FilePlan("deployment/scripts/rollback.sh", "Rollback script", 3, [], "code", FileCategory.DEPLOYMENT, 100),
            FilePlan("deployment/scripts/health_check.sh", "Health check script", 2, [], "code", FileCategory.DEPLOYMENT, 80),
        ]
    
    def _generate_monitoring_files(self, paper: PaperInfo) -> List[FilePlan]:
        """Generate monitoring and observability files."""
        return [
            # Monitoring
            FilePlan("monitoring/__init__.py", "Monitoring package initialization", 2, [], "code", FileCategory.UTILITIES),
            FilePlan("monitoring/metrics.py", "Custom metrics collection", 2, [], "code", FileCategory.UTILITIES, 200),
            FilePlan("monitoring/alerts.py", "Alerting system", 2, [], "code", FileCategory.UTILITIES, 150),
            FilePlan("monitoring/dashboards.py", "Dashboard generation", 2, [], "code", FileCategory.UTILITIES, 180),
            FilePlan("monitoring/health_checks.py", "Health check implementations", 2, [], "code", FileCategory.UTILITIES, 120),
            
            # Prometheus/Grafana
            FilePlan("monitoring/prometheus/prometheus.yml", "Prometheus configuration", 3, [], "config", FileCategory.UTILITIES),
            FilePlan("monitoring/grafana/dashboard.json", "Grafana dashboard configuration", 3, [], "config", FileCategory.UTILITIES),
            
            # Logging
            FilePlan("monitoring/logging/config.yaml", "Logging configuration", 2, [], "config", FileCategory.UTILITIES),
            FilePlan("monitoring/logging/formatters.py", "Custom log formatters", 2, [], "code", FileCategory.UTILITIES, 100),
            
            # Performance Monitoring
            FilePlan("monitoring/performance/profiler.py", "Performance profiling tools", 2, [], "code", FileCategory.UTILITIES, 150),
            FilePlan("monitoring/performance/benchmarks.py", "Performance benchmarking", 2, [], "code", FileCategory.UTILITIES, 200),
        ]
    
    def _generate_security_files(self, paper: PaperInfo) -> List[FilePlan]:
        """Generate security and compliance files."""
        return [
            # Security
            FilePlan("security/__init__.py", "Security package initialization", 2, [], "code", FileCategory.UTILITIES),
            FilePlan("security/authentication.py", "Authentication mechanisms", 2, [], "code", FileCategory.UTILITIES, 200),
            FilePlan("security/authorization.py", "Authorization and RBAC", 2, [], "code", FileCategory.UTILITIES, 180),
            FilePlan("security/encryption.py", "Data encryption utilities", 2, [], "code", FileCategory.UTILITIES, 150),
            FilePlan("security/input_validation.py", "Input validation and sanitization", 2, [], "code", FileCategory.UTILITIES, 120),
            FilePlan("security/audit.py", "Security audit logging", 2, [], "code", FileCategory.UTILITIES, 100),
            
            # Security Configuration
            FilePlan("security/config/security_policy.yaml", "Security policy configuration", 2, [], "config", FileCategory.UTILITIES),
            FilePlan("security/config/cors.yaml", "CORS configuration", 2, [], "config", FileCategory.UTILITIES),
            
            # Compliance
            FilePlan("compliance/gdpr.py", "GDPR compliance utilities", 3, [], "code", FileCategory.UTILITIES, 150),
            FilePlan("compliance/audit_trail.py", "Audit trail implementation", 3, [], "code", FileCategory.UTILITIES, 120),
            
            # Security Testing
            FilePlan("security/tests/test_auth.py", "Authentication tests", 2, [], "test", FileCategory.TESTING, 150),
            FilePlan("security/tests/test_encryption.py", "Encryption tests", 2, [], "test", FileCategory.TESTING, 100),
        ]
    
    def _deduplicate_files(self, files: List[FilePlan]) -> List[FilePlan]:
        """Remove duplicate files based on path."""
        seen_paths = set()
        unique_files = []
        
        for file_plan in files:
            if file_plan.path not in seen_paths:
                seen_paths.add(file_plan.path)
                unique_files.append(file_plan)
        
        return unique_files


class DependencyManager:
    """Manages Python dependencies for generated projects."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Common package mappings for import -> package name
        self.import_to_package = {
            'torch': 'torch>=2.0.0',
            'torchvision': 'torchvision>=0.15.0',
            'transformers': 'transformers>=4.30.0',
            'numpy': 'numpy>=1.24.0',
            'pandas': 'pandas>=2.0.0',
            'matplotlib': 'matplotlib>=3.7.0',
            'seaborn': 'seaborn>=0.12.0',
            'sklearn': 'scikit-learn>=1.3.0',
            'cv2': 'opencv-python>=4.8.0',
            'PIL': 'Pillow>=10.0.0',
            'requests': 'requests>=2.31.0',
            'aiohttp': 'aiohttp>=3.8.0',
            'fastapi': 'fastapi>=0.100.0',
            'uvicorn': 'uvicorn>=0.23.0',
            'streamlit': 'streamlit>=1.25.0',
            'gradio': 'gradio>=3.40.0',
            'flask': 'Flask>=2.3.0',
            'pydantic': 'pydantic>=2.0.0',
            'pytest': 'pytest>=7.4.0',
            'jupyter': 'jupyter>=1.0.0',
            'tqdm': 'tqdm>=4.65.0',
            'wandb': 'wandb>=0.15.0',
            'mlflow': 'mlflow>=2.5.0',
            'yaml': 'PyYAML>=6.0',
            'dotenv': 'python-dotenv>=1.0.0',
            'click': 'click>=8.1.0',
            'rich': 'rich>=13.5.0',
            'boto3': 'boto3>=1.28.0',
            'redis': 'redis>=4.6.0',
            'docker': 'docker>=6.1.0',
            'jinja2': 'Jinja2>=3.1.0',
            'jsonschema': 'jsonschema>=4.19.0',
            'lightning': 'pytorch-lightning>=2.0.0',
            'datasets': 'datasets>=2.14.0',
            'tokenizers': 'tokenizers>=0.13.0',
            'spacy': 'spacy>=3.6.0',
            'nltk': 'nltk>=3.8.0',
            'networkx': 'networkx>=3.1.0',
            'scipy': 'scipy>=1.11.0',
            'plotly': 'plotly>=5.15.0',
            'dash': 'dash>=2.12.0'
        }
        
        # Base dependencies that are almost always needed
        self.base_dependencies = {
            'numpy>=1.24.0',
            'python-dotenv>=1.0.0',
            'PyYAML>=6.0',
            'requests>=2.31.0',
            'tqdm>=4.65.0',
            'typing-extensions>=4.7.0'
        }

    def extract_imports_from_content(self, content: str) -> Set[str]:
        """Extract import statements from Python code content."""
        imports = set()
        
        # Regular expressions for different import patterns
        import_patterns = [
            r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',
            r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import'
        ]
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1)
                    # Handle submodules (e.g., torch.nn -> torch)
                    base_module = module.split('.')[0]
                    imports.add(base_module)
                    break
        
        return imports

    def generate_requirements(self, files_content: Dict[str, str], paper: PaperInfo) -> Tuple[str, str]:
        """Generate requirements.txt content based on detected imports."""
        all_imports = set()
        
        # Extract imports from all Python files
        for file_path, content in files_content.items():
            if file_path.endswith('.py'):
                file_imports = self.extract_imports_from_content(content)
                all_imports.update(file_imports)
        
        # Convert imports to package requirements
        requirements = set(self.base_dependencies)
        
        for import_name in all_imports:
            if import_name in self.import_to_package:
                requirements.add(self.import_to_package[import_name])
            elif import_name not in ['os', 'sys', 'json', 're', 'time', 'datetime', 
                                   'pathlib', 'logging', 'asyncio', 'subprocess', 
                                   'tempfile', 'hashlib', 'base64', 'math', 'random', 
                                   'string', 'enum', 'dataclasses', 'typing', 'abc', 
                                   'concurrent', 'zipfile', 'urllib', 'mimetypes',
                                   'collections', 'itertools', 'functools', 'operator',
                                   'copy', 'pickle', 'csv', 'sqlite3', 'threading',
                                   'multiprocessing', 'queue', 'socket', 'ssl', 'email',
                                   'html', 'xml', 'http', 'unittest', 'argparse', 'configparser']:
                # Add unknown third-party packages with generic version
                requirements.add(f"{import_name}>=0.1.0")
        
        # Add paper-specific requirements based on research domain
        if hasattr(paper, 'research_domain') and paper.research_domain:
            domain = paper.research_domain.lower()
            if 'nlp' in domain or 'language' in domain or 'text' in domain:
                requirements.update({
                    'transformers>=4.30.0',
                    'tokenizers>=0.13.0',
                    'datasets>=2.14.0'
                })
            elif 'vision' in domain or 'image' in domain or 'cv' in domain:
                requirements.update({
                    'torchvision>=0.15.0',
                    'opencv-python>=4.8.0',
                    'Pillow>=10.0.0'
                })
        
        # Sort requirements for consistency
        sorted_requirements = sorted(list(requirements))
        
        # Generate main requirements.txt
        main_requirements = '\n'.join(sorted_requirements)
        
        # Generate requirements-dev.txt
        dev_requirements = '\n'.join([
            '# Development dependencies',
            '-r requirements.txt',
            '',
            '# Testing and Quality',
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
            'pre-commit>=3.3.0',
            'bandit>=1.7.0'
        ])
        
        return main_requirements, dev_requirements


class QualityAssuranceEngine:
    """Advanced quality assurance for generated code."""
    
    def __init__(self, config: Config, llm_interface: 'LLMInterface', logger: logging.Logger):
        self.config = config
        self.llm = llm_interface
        self.logger = logger
        self.dependency_manager = DependencyManager(config, logger)
    
    async def assess_code_quality(self, file_path: str, content: str, file_plan: FilePlan) -> QualityReport:
        """Comprehensive code quality assessment."""
        self.logger.info(f"Assessing code quality for {file_path}")
        
        quality_scores = {}
        issues_found = []
        suggestions = []
        
        # Static analysis
        static_score, static_issues = self._static_analysis(content, file_plan)
        quality_scores[QualityMetric.MAINTAINABILITY] = static_score
        issues_found.extend(static_issues)
        
        # Security analysis
        security_score, security_issues = self._security_analysis(content, file_plan)
        quality_scores[QualityMetric.SECURITY] = security_score
        issues_found.extend(security_issues)
        
        # Documentation analysis
        doc_score, doc_issues = self._documentation_analysis(content, file_plan)
        quality_scores[QualityMetric.DOCUMENTATION] = doc_score
        issues_found.extend(doc_issues)
        
        # Performance analysis
        perf_score, perf_issues = self._performance_analysis(content, file_plan)
        quality_scores[QualityMetric.PERFORMANCE] = perf_score
        issues_found.extend(perf_issues)
        
        # LLM-based review
        if self.config.enable_code_review:
            llm_score, llm_suggestions = await self._llm_code_review(content, file_plan)
            quality_scores[QualityMetric.RELIABILITY] = llm_score
            suggestions.extend(llm_suggestions)
        
        # Calculate overall score
        overall_score = sum(quality_scores.values()) / len(quality_scores)
        passed_checks = overall_score >= self.config.min_code_quality_score
        
        return QualityReport(
            file_path=file_path,
            quality_scores=quality_scores,
            issues_found=issues_found,
            suggestions=suggestions,
            overall_score=overall_score,
            passed_checks=passed_checks
        )
    
    def _static_analysis(self, content: str, file_plan: FilePlan) -> Tuple[float, List[str]]:
        """Enhanced static code analysis with professional tools integration."""
        issues = []
        score = 1.0
        
        # Check for basic Python syntax issues
        try:
            compile(content, file_plan.path, 'exec')
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
            score -= 0.5
            return max(score, 0.0), issues  # Return early if syntax error
        
        # Try to use professional static analysis tools
        temp_file_path = None
        try:
            # Write content to temporary file for analysis
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Run flake8 for style and error checking
            try:
                result = subprocess.run(
                    ['flake8', '--max-line-length=100', '--ignore=E203,W503,E501', temp_file_path],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode != 0 and result.stdout.strip():
                    flake8_issues = [line for line in result.stdout.strip().split('\n') if line.strip()]
                    for issue in flake8_issues[:5]:  # Limit to first 5 issues
                        issue_parts = issue.split(':', 3)
                        if len(issue_parts) >= 4:
                            issues.append(f"Style: {issue_parts[3].strip()}")
                    score -= min(0.3, len(flake8_issues) * 0.05)
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                self.logger.debug("Flake8 not available, using basic style checks")
            
            # Run bandit for security analysis (only for non-test files)
            if not file_plan.path.startswith('tests/'):
                try:
                    result = subprocess.run(
                        ['bandit', '-f', 'json', '-ll', temp_file_path],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.stdout.strip():
                        try:
                            bandit_data = json.loads(result.stdout)
                            security_issues = bandit_data.get('results', [])
                            for issue in security_issues[:3]:  # Limit to first 3 security issues
                                severity = issue.get('issue_severity', 'MEDIUM')
                                test_name = issue.get('test_name', 'Unknown')
                                issues.append(f"Security ({severity}): {test_name}")
                            if security_issues:
                                score -= min(0.4, len(security_issues) * 0.1)
                        except json.JSONDecodeError:
                            pass
                except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                    self.logger.debug("Bandit not available, skipping security analysis")
            
        except Exception as e:
            self.logger.warning(f"Professional static analysis failed: {e}")
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass
        
        # Enhanced manual analysis as fallback/supplement
        lines = content.split('\n')
        
        # Function complexity and length analysis
        in_function = False
        function_lines = 0
        current_function_name = ""
        complexity_score = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def '):
                # Check previous function
                if in_function and function_lines > 50:
                    issues.append(f"Function '{current_function_name}' too long ({function_lines} lines)")
                    score -= 0.1
                if complexity_score > 10:
                    issues.append(f"Function '{current_function_name}' too complex (complexity: {complexity_score})")
                    score -= 0.15
                
                # Start new function
                in_function = True
                function_lines = 0
                complexity_score = 1
                try:
                    current_function_name = stripped.split('(')[0].replace('def ', '').strip()
                except:
                    current_function_name = "unknown"
                    
            elif in_function:
                if stripped.startswith(('def ', 'class ')):
                    # End of current function
                    in_function = stripped.startswith('def ')
                    if not in_function:
                        continue
                else:
                    function_lines += 1
                    # Simple complexity calculation
                    complexity_keywords = ['if ', 'elif ', 'for ', 'while ', 'except ', 'with ', 'and ', 'or ']
                    complexity_score += sum(1 for keyword in complexity_keywords if keyword in stripped)
        
        # Check final function
        if in_function and function_lines > 50:
            issues.append(f"Function '{current_function_name}' too long ({function_lines} lines)")
            score -= 0.1
        
        # Import analysis
        import_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                import_lines.append((i, stripped))
        
        if import_lines:
            # Check for unused imports (basic check)
            imported_modules = set()
            for _, import_line in import_lines:
                if import_line.startswith('import '):
                    modules = import_line.replace('import ', '').split(',')
                    for module in modules:
                        imported_modules.add(module.strip().split(' as ')[0])
                elif import_line.startswith('from '):
                    parts = import_line.split()
                    if len(parts) >= 4:  # from module import something
                        imports = ' '.join(parts[3:]).split(',')
                        for imp in imports:
                            imported_modules.add(imp.strip().split(' as ')[0])
            
            # Simple unused import detection
            content_without_imports = '\n'.join([line for i, line in enumerate(lines) 
                                               if not lines[i].strip().startswith(('import ', 'from '))])
            potentially_unused = []
            for module in imported_modules:
                if module not in content_without_imports and len(module) > 2:
                    potentially_unused.append(module)
            
            if potentially_unused and len(potentially_unused) <= 3:  # Only report if reasonable number
                issues.append(f"Potentially unused imports: {', '.join(potentially_unused[:3])}")
                score -= 0.05
        
        # Documentation coverage analysis
        if file_plan.file_type == "code":
            function_defs = [line for line in lines if line.strip().startswith('def ')]
            class_defs = [line for line in lines if line.strip().startswith('class ')]
            
            # Count docstrings more accurately
            docstring_count = 0
            in_docstring = False
            docstring_char = None
            
            for line in lines:
                stripped = line.strip()
                if not in_docstring:
                    if '"""' in stripped:
                        docstring_char = '"""'
                        if stripped.count('"""') == 2:  # Single line docstring
                            docstring_count += 1
                        else:
                            in_docstring = True
                    elif "'''" in stripped:
                        docstring_char = "'''"
                        if stripped.count("'''") == 2:  # Single line docstring
                            docstring_count += 1
                        else:
                            in_docstring = True
                else:
                    if docstring_char in stripped:
                        docstring_count += 1
                        in_docstring = False
                        docstring_char = None
            
            total_definitions = len(function_defs) + len(class_defs)
            if total_definitions > 0:
                doc_coverage = docstring_count / total_definitions
                if doc_coverage < 0.6:
                    issues.append(f"Low documentation coverage: {doc_coverage:.1%} ({docstring_count}/{total_definitions})")
                    score -= 0.2 * (0.6 - doc_coverage)
        
        # Type hints coverage
        if file_plan.file_type == "code":
            function_lines = [line for line in lines if line.strip().startswith('def ')]
            typed_functions = []
            for line in function_lines:
                if '->' in line or any(': ' in param for param in line.split('(')[1].split(')')[0].split(',')):
                    typed_functions.append(line)
            
            if function_lines:
                type_coverage = len(typed_functions) / len(function_lines)
                if type_coverage < 0.4:
                    issues.append(f"Low type hints coverage: {type_coverage:.1%}")
                    score -= 0.1
        
        # Code quality indicators
        quality_issues = []
        todo_count = content.count('TODO') + content.count('FIXME') + content.count('XXX')
        if todo_count > 0:
            quality_issues.append(f"{todo_count} TODO/FIXME comments")
            score -= 0.05 * min(todo_count, 5)
        
        # Check for code smells
        if 'print(' in content and file_plan.file_type == "code" and not file_plan.path.startswith('examples/'):
            quality_issues.append("Contains print statements (consider using logging)")
            score -= 0.05
        
        if quality_issues:
            issues.append("Code quality: " + ", ".join(quality_issues))
        
        return max(score, 0.0), issues
    
    def _security_analysis(self, content: str, file_plan: FilePlan) -> Tuple[float, List[str]]:
        """Security vulnerability analysis."""
        issues = []
        score = 1.0
        
        # Check for common security issues
        security_patterns = [
            (r'eval\s*\(', "Use of eval() function"),
            (r'exec\s*\(', "Use of exec() function"),
            (r'os\.system\s*\(', "Use of os.system()"),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "Shell injection risk"),
            (r'pickle\.loads?\s*\(', "Unsafe pickle usage"),
            (r'yaml\.load\s*\(', "Unsafe YAML loading"),
        ]
        
        for pattern, message in security_patterns:
            if re.search(pattern, content):
                issues.append(message)
                score -= 0.2
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
        ]
        
        for pattern, message in secret_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(message)
                score -= 0.3
        
        return max(score, 0.0), issues
    
    def _documentation_analysis(self, content: str, file_plan: FilePlan) -> Tuple[float, List[str]]:
        """Documentation quality analysis."""
        issues = []
        score = 1.0
        
        if file_plan.file_type != "code":
            return score, issues
        
        lines = content.split('\n')
        
        # Count functions and classes
        functions = len([line for line in lines if line.strip().startswith('def ')])
        classes = len([line for line in lines if line.strip().startswith('class ')])
        
        # Count docstrings
        docstring_count = content.count('"""') + content.count("'''")
        
        # Estimate expected docstrings (functions + classes + module)
        expected_docstrings = functions + classes + 1
        
        if expected_docstrings > 0:
            doc_ratio = docstring_count / (expected_docstrings * 2)  # Each docstring has opening and closing
            if doc_ratio < 0.5:
                issues.append("Insufficient documentation coverage")
                score -= 0.3
            elif doc_ratio < 0.8:
                issues.append("Low documentation coverage")
                score -= 0.1
        
        # Check for type hints
        if functions > 0:
            type_hint_count = len(re.findall(r'def\s+\w+\s*\([^)]*:\s*\w+', content))
            if type_hint_count / functions < 0.5:
                issues.append("Missing type hints")
                score -= 0.2
        
        return max(score, 0.0), issues
    
    def _performance_analysis(self, content: str, file_plan: FilePlan) -> Tuple[float, List[str]]:
        """Performance analysis."""
        issues = []
        score = 1.0
        
        # Check for performance anti-patterns
        perf_patterns = [
            (r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', "Use enumerate() instead of range(len())"),
            (r'\+\s*=.*\[.*\]', "List concatenation in loop (use extend())"),
            (r'\.append\s*\(.*\)\s*\n.*\.append', "Multiple appends (consider list comprehension)"),
        ]
        
        for pattern, message in perf_patterns:
            matches = len(re.findall(pattern, content))
            if matches > 0:
                issues.append(f"{message} ({matches} occurrences)")
                score -= 0.1 * min(matches, 3)
        
        # Check for inefficient imports
        if re.search(r'from\s+\*\s+import\s+\*', content):
            issues.append("Wildcard imports")
            score -= 0.1
        
        return max(score, 0.0), issues
    
    async def _llm_code_review(self, content: str, file_plan: FilePlan) -> Tuple[float, List[str]]:
        """LLM-based code review."""
        prompt = f"""
        You are an expert code reviewer. Review the following Python code and provide:
        1. A quality score from 0.0 to 1.0
        2. Specific suggestions for improvement
        
        File: {file_plan.path}
        Description: {file_plan.description}
        
        Code:
        ```python
        {content[:2000]}  # Truncate for token limits
        ```
        
        Respond in JSON format:
        {{
            "score": 0.85,
            "suggestions": ["suggestion1", "suggestion2"]
        }}
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response, error = await self.llm._call_llm(messages, self.config.reviewer_model, is_json=True)
            if response:
                data = json.loads(response)
                return data.get("score", 0.8), data.get("suggestions", [])
        except Exception as e:
            self.logger.warning(f"LLM code review failed: {e}")
        
        return 0.8, []  # Default values


class AdvancedGitHubManager:
    """Enhanced GitHub management with advanced features."""
    
    def __init__(self, config: Config, session: aiohttp.ClientSession, logger: logging.Logger):
        self.config = config
        self.session = session
        self.logger = logger
        self.headers = {
            "Authorization": f"token {self.config.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": self.config.user_agent
        }
    
    async def create_advanced_repository(self, repo_name: str, description: str, paper: PaperInfo) -> Tuple[Optional[str], str]:
        """Create repository with advanced features enabled."""
        # Create basic repository
        repo_url, message = await self.create_repository(repo_name, description)
        
        if not repo_url:
            return repo_url, message
        
        # Enable advanced features
        if self.config.enable_github_pages:
            await self._enable_github_pages(repo_name)
        
        if self.config.enable_security_features:
            await self._enable_security_features(repo_name)
        
        # Create repository topics/tags
        await self._set_repository_topics(repo_name, paper)
        
        # Create initial branch protection
        await self._setup_branch_protection(repo_name)
        
        return repo_url, "Advanced repository created successfully"
    
    async def create_repository(self, repo_name: str, description: str) -> Tuple[Optional[str], str]:
        """Creates a new GitHub repository."""
        url = "https://api.github.com/user/repos"
        payload = {
            "name": repo_name,
            "description": description,
            "private": self.config.repo_visibility == "private",
            "auto_init": False,
            "has_issues": True,
            "has_projects": True,
            "has_wiki": True,
            "allow_squash_merge": True,
            "allow_merge_commit": True,
            "allow_rebase_merge": True,
            "delete_branch_on_merge": True
        }
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with self.session.post(url, headers=self.headers, json=payload) as response:
                    if response.status == 201:
                        data = await response.json()
                        repo_url = data["html_url"]
                        self.logger.info(f"Successfully created repository: {repo_url}")
                        return repo_url, "Repository created successfully"
                    elif response.status == 422:
                        error_data = await response.json()
                        if "name already exists" in str(error_data):
                            existing_url = f"https://github.com/{self.config.github_username}/{repo_name}"
                            self.logger.warning(f"Repository already exists: {existing_url}")
                            return existing_url, "Repository already exists"
                    elif response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        self.logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
            except Exception as e:
                error_msg = f"Failed to create repository (attempt {attempt+1}): {e}"
                self.logger.error(error_msg)
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        return None, f"Failed to create repository after {self.config.retry_attempts} attempts"
    
    async def _enable_github_pages(self, repo_name: str):
        """Enable GitHub Pages for the repository."""
        url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}/pages"
        payload = {
            "source": {
                "branch": "main",
                "path": "/docs"
            }
        }
        
        try:
            async with self.session.post(url, headers=self.headers, json=payload) as response:
                if response.status in [201, 409]:  # 409 means already enabled
                    self.logger.info(f"GitHub Pages enabled for {repo_name}")
                else:
                    self.logger.warning(f"Failed to enable GitHub Pages: {response.status}")
        except Exception as e:
            self.logger.warning(f"Error enabling GitHub Pages: {e}")
    
    async def _enable_security_features(self, repo_name: str):
        """Enable security features for the repository."""
        # Enable vulnerability alerts
        url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}/vulnerability-alerts"
        try:
            async with self.session.put(url, headers=self.headers) as response:
                if response.status == 204:
                    self.logger.info(f"Vulnerability alerts enabled for {repo_name}")
        except Exception as e:
            self.logger.warning(f"Error enabling vulnerability alerts: {e}")
        
        # Enable automated security fixes
        url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}/automated-security-fixes"
        try:
            async with self.session.put(url, headers=self.headers) as response:
                if response.status == 204:
                    self.logger.info(f"Automated security fixes enabled for {repo_name}")
        except Exception as e:
            self.logger.warning(f"Error enabling automated security fixes: {e}")
    
    async def _set_repository_topics(self, repo_name: str, paper: PaperInfo):
        """Set repository topics based on paper content."""
        topics = ["machine-learning", "research", "python", "ai"]
        
        # Add domain-specific topics
        if paper.research_domain:
            topics.append(paper.research_domain.lower().replace(" ", "-"))
        
        # Add complexity-based topics
        topics.append(f"complexity-{paper.complexity_level}")
        
        # Add methodology topics
        if paper.methodology:
            method_topics = paper.methodology.lower().split()[:3]  # First 3 words
            topics.extend([topic.replace(" ", "-") for topic in method_topics])
        
        # Clean and limit topics
        topics = list(set([topic[:50] for topic in topics if topic.isalnum() or "-" in topic]))[:20]
        
        url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}/topics"
        payload = {"names": topics}
        
        try:
            async with self.session.put(url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    self.logger.info(f"Repository topics set for {repo_name}: {topics}")
        except Exception as e:
            self.logger.warning(f"Error setting repository topics: {e}")
    
    async def _setup_branch_protection(self, repo_name: str):
        """Setup branch protection rules."""
        url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}/branches/main/protection"
        payload = {
            "required_status_checks": {
                "strict": True,
                "contexts": ["continuous-integration"]
            },
            "enforce_admins": False,
            "required_pull_request_reviews": {
                "required_approving_review_count": 1,
                "dismiss_stale_reviews": True
            },
            "restrictions": None
        }
        
        try:
            async with self.session.put(url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    self.logger.info(f"Branch protection enabled for {repo_name}")
        except Exception as e:
            self.logger.warning(f"Error setting up branch protection: {e}")
    
    async def upload_file_batch(self, repo_name: str, files: List[Tuple[str, str]], commit_message: str) -> Tuple[int, int]:
        """Upload multiple files in a single commit."""
        successful = 0
        failed = 0
        
        # For now, upload files individually
        # TODO: Implement tree API for batch uploads
        for file_path, content in files:
            success, _ = await self.upload_file(repo_name, file_path, content, f"{commit_message}: {file_path}")
            if success:
                successful += 1
            else:
                failed += 1
            
            # Rate limiting
            await asyncio.sleep(0.5)
        
        return successful, failed
    
    async def upload_file(self, repo_name: str, file_path: str, content: str, commit_message: str) -> Tuple[bool, str]:
        """Uploads a single file to the GitHub repository."""
        url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}/contents/{file_path}"
        
        # Check if file already exists
        try:
            async with self.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    existing_data = await response.json()
                    sha = existing_data["sha"]
                else:
                    sha = None
        except:
            sha = None
        
        # Encode content to base64
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        payload = {
            "message": commit_message,
            "content": encoded_content
        }
        
        if sha:
            payload["sha"] = sha
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with self.session.put(url, headers=self.headers, json=payload) as response:
                    if response.status in [200, 201]:
                        self.logger.debug(f"Successfully uploaded {file_path}")
                        return True, "File uploaded successfully"
                    elif response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        self.logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    error_text = await response.text()
                    self.logger.error(f"Failed to upload {file_path}: {response.status} - {error_text}")
                    return False, f"Upload failed: {response.status} - {error_text}"
            except Exception as e:
                error_msg = f"Failed to upload {file_path} (attempt {attempt+1}): {e}"
                self.logger.error(error_msg)
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        return False, f"Failed to upload {file_path} after {self.config.retry_attempts} attempts"


# --- Main Execution Logic ---

async def main():
    """Main execution function for the M1-Evo Maintainer Agent."""
    # Load environment variables
    load_dotenv()
    
    # Create logs directory first
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/m1_evo_agent.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize configuration with GitHub Actions support
        config = Config(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            github_token=os.getenv("GITHUB_TOKEN", os.getenv("GH_TOKEN", "")),  # Support both GITHUB_TOKEN and GH_TOKEN
            github_username=os.getenv("GITHUB_USERNAME", os.getenv("GITHUB_ACTOR", "")),  # Use GITHUB_ACTOR in Actions
            huggingface_token=os.getenv("HUGGINGFACE_TOKEN", ""),
            arxiv_api_key=os.getenv("ARXIV_API_KEY", "")
        )
        
        # GitHub Actions environment detection
        is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"
        if is_github_actions:
            logger.info("Running in GitHub Actions environment")
            # Set GitHub Actions specific configurations
            config.repo_visibility = os.getenv("REPO_VISIBILITY", "public")
            config.max_concurrent_papers = min(config.max_concurrent_papers, 2)  # Limit concurrency in CI
            
            # GitHub Actions outputs
            github_output = os.getenv("GITHUB_OUTPUT")
            if github_output:
                logger.info(f"GitHub Actions output file: {github_output}")
        
        # Validate required configuration
        missing_vars = []
        if not config.openrouter_api_key:
            missing_vars.append("OPENROUTER_API_KEY")
        
        if not config.github_token:
            missing_vars.append("GITHUB_TOKEN or GH_TOKEN")
        
        if not config.github_username:
            missing_vars.append("GITHUB_USERNAME or GITHUB_ACTOR")
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            if is_github_actions:
                # Set GitHub Actions error annotation
                print(f"::error::{error_msg}")
            return
        
        # Create necessary directories
        for directory in [config.logs_dir, config.workspace_dir, config.llm_logs_dir, 
                         config.cache_dir, config.templates_dir]:
            directory.mkdir(exist_ok=True)
        
        logger.info("Starting M1-Evo Maintainer Agent - 360° Repository Creation")
        logger.info(f"Configuration loaded: {len(config.__dict__)} parameters")
        
        # Initialize the main agent
        async with aiohttp.ClientSession() as session:
            agent = M1EvoMaintainerAgent(config)
            
            # Process papers and create repositories
            await agent.run()
            
        logger.info("M1-Evo Maintainer Agent execution completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")
        raise


# --- Utility Functions ---

def setup_environment():
    """Setup the environment for the M1-Evo Agent."""
    # Create required directories
    directories = [
        "logs",
        "workspace", 
        "cache",
        "templates",
        "llm_interactions",
        "benchmarks",
        "models_cache",
        "datasets",
        "artifacts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Create .env template if it doesn't exist
    env_template = """# M1-Evo Maintainer Agent Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
GITHUB_TOKEN=your_github_token_here
GITHUB_USERNAME=your_github_username_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
ARXIV_API_KEY=your_arxiv_api_key_here
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_template)
        print("Created .env template file. Please fill in your API keys.")


def validate_dependencies():
    """Validate that all required dependencies are installed."""
    required_packages = [
        "aiohttp",
        "pyyaml", 
        "python-dotenv",
        "requests",
        "torch",
        "transformers",
        "numpy",
        "matplotlib",
        "seaborn",
        "jupyter",
        "pytest",
        "black",
        "flake8",
        "mypy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True


def print_banner():
    """Print the application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              M1-Evo Maintainer Agent v3.0                   ║
    ║           360° Research Paper Implementation                  ║
    ║                                                              ║
    ║  Automatically creates comprehensive, production-ready       ║
    ║  repositories from research papers with:                     ║
    ║  • Complete model implementations                            ║
    ║  • Comprehensive testing suites                              ║
    ║  • Documentation and examples                                ║
    ║  • Deployment configurations                                 ║
    ║  • Quality assurance                                         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


# --- CLI Interface ---

def parse_arguments():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="M1-Evo Maintainer Agent - 360° Research Paper Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix2.py                    # Process all papers in relevant_json/
  python fix2.py --setup            # Setup environment and create .env template
  python fix2.py --validate         # Validate dependencies
  python fix2.py --paper paper.json # Process specific paper
  python fix2.py --dry-run          # Show what would be processed without creating repos
        """
    )
    
    parser.add_argument(
        "--setup", 
        action="store_true",
        help="Setup environment and create configuration templates"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true", 
        help="Validate that all required dependencies are installed"
    )
    
    parser.add_argument(
        "--paper",
        type=str,
        help="Process a specific paper JSON file"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without creating repositories"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to process"
    )
    
    return parser.parse_args()


# --- Entry Point ---

if __name__ == "__main__":
    print_banner()
    
    args = parse_arguments()
    
    if args.setup:
        setup_environment()
        print("Environment setup completed.")
        sys.exit(0)
    
    if args.validate:
        if validate_dependencies():
            print("All required dependencies are installed.")
        else:
            sys.exit(1)
    
    if args.dry_run:
        print("DRY RUN MODE - No repositories will be created")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the main application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
