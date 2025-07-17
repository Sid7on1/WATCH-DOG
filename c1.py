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
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict, dataclass, field
from pathlib import Path
import logging
from enum import Enum
from dotenv import load_dotenv

# --- Configuration ---
@dataclass
class Config:
    """Central Configuration for the M1-Evo Maintainer Agent."""
    openrouter_api_key: str
    github_token: str
    github_username: str
    user_agent: str = "M1-Evo-Agent/2.0"
    repo_prefix: str = "paper-impl-"
    architect_model: str = "meta-llama/llama-3.1-405b-instruct"
    coder_model: str = "deepseek/deepseek-coder-v2"
    temperature: float = 0.3
    max_llm_tokens: int = 16000  # Increased for longer content
    request_timeout: int = 600  # Increased timeout
    retry_attempts: int = 3
    retry_delay: int = 10
    repo_visibility: str = "public"
    max_concurrent_papers: int = 2  # Reduced for better quality
    base_dir: Path = Path(__file__).parent
    papers_dir: Path = base_dir / "relevant_json"
    state_file: Path = base_dir / "managed_repos_state.json"
    workspace_dir: Path = base_dir / "workspace"
    logs_dir: Path = base_dir / "logs"
    llm_logs_dir: Path = base_dir / "llm_interactions"

# --- Enums and Data Classes ---
class ProcessingStatus(Enum):
    PENDING = "pending"
    PLANNING = "planning_structure"
    CREATING_REPO = "creating_repository"
    GENERATING_FILES = "generating_files"
    VALIDATING = "validating_repo"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class RepoState:
    """Represents the state of a single managed repository."""
    repo_name: str
    github_url: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    last_processed_timestamp: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    files_generated: List[str] = field(default_factory=list)

@dataclass
class PaperInfo:
    """Information about a research paper to be processed."""
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

@dataclass
class FilePlan:
    """A planned file to be generated."""
    path: str
    description: str
    priority: int = 1  # 1=high, 2=medium, 3=low
    dependencies: List[str] = field(default_factory=list)
    file_type: str = "code"  # code, config, docs, test

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
        """Calls the LLM API with retry logic and rate-limiting handling."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        
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
                async with self.session.post(url, headers=self.headers, json=payload, timeout=self.config.request_timeout) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        self.logger.warning(f"Rate limit exceeded for LLM API. Retrying after {retry_after} seconds.")
                        await asyncio.sleep(retry_after)
                        continue
                    if response.status == 400:
                        error_text = await response.text()
                        self.logger.error(f"400 Bad Request for model {model}. Response: {error_text}")
                        error_message = f"LLM API call failed (Attempt {attempt+1}/{self.config.retry_attempts}) for model {model}: 400 Bad Request. Response: {error_text}"
                        break
                    response.raise_for_status()
                    data = await response.json()
                    response_content = data['choices'][0]['message']['content']
                    
                    # Log interaction with redacted sensitive data
                    log_file = self.config.llm_logs_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{model.replace('/', '_')}.json"
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump(self.redact_sensitive({"request": payload, "response": data}), f, indent=2)

                    return response_content, None
            except aiohttp.ClientResponseError as e:
                error_message = f"LLM API call failed (Attempt {attempt+1}/{self.config.retry_attempts}) for model {model}: {e}"
                self.logger.warning(error_message)
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            except Exception as e:
                error_message = f"Unexpected error in LLM call: {e}"
                self.logger.error(error_message)
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
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
        """Creates a new GitHub repository."""
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
        import base64
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


class PaperProcessor:
    """Processes research papers and extracts relevant information."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_papers_from_directory(self) -> List[PaperInfo]:
        """Loads all papers from the papers directory."""
        papers = []
        
        if not self.config.papers_dir.exists():
            self.logger.warning(f"Papers directory does not exist: {self.config.papers_dir}")
            return papers
        
        for json_file in self.config.papers_dir.glob("*.json"):
            try:
                paper = self._load_paper_from_file(json_file)
                if paper:
                    papers.append(paper)
            except Exception as e:
                self.logger.error(f"Failed to load paper from {json_file}: {e}")
        
        self.logger.info(f"Loaded {len(papers)} papers from {self.config.papers_dir}")
        return papers

    def _load_paper_from_file(self, file_path: Path) -> Optional[PaperInfo]:
        """Loads a single paper from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract paper information with flexible field mapping
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
            
            return paper
            
        except Exception as e:
            self.logger.error(f"Error loading paper from {file_path}: {e}")
            return None

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
        """Sets up comprehensive logging."""
        logger = logging.getLogger("M1EvoAgent")
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        self.config.logs_dir.mkdir(exist_ok=True)
        
        # File handler
        log_file = self.config.logs_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
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
            self.state_manager.update_repo_state(state, repo_name, ProcessingStatus.PLANNING)
            
            # Step 1: Plan file structure
            self.logger.info(f"Planning file structure for {repo_name}")
            file_plans, plan_message = await llm_interface.plan_file_structure(paper)
            
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


if __name__ == "__main__":
    asyncio.run(main())
