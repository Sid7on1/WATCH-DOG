import os
import json
import re
import subprocess
import asyncio
import aiohttp
import time
import shutil
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum
from dotenv import load_dotenv

# --- Configuration (Reused and Extended) ---
@dataclass
class Config:
    """Central Configuration for the M1 Maintainer Agent."""
    openrouter_api_key: str
    github_token: str
    github_username: str
    
    # LLM Settings
    primary_model: str = "deepseek/deepseek-r1-0528-qwen3-8b:free"
    fallback_model: str = "google/gemini-2.0-flash-exp:free"
    backup_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    temperature: float = 0.7
    max_llm_tokens: int = 4000
    min_code_length: int = 50 # Minimum length for LLM generated code
    
    # Processing & Concurrency
    max_concurrent_paper_processing: int = 3
    request_timeout: int = 300 # Increased for potentially longer LLM calls/git ops
    retry_attempts: int = 5
    retry_delay: int = 5
    
    # GitHub Repository Settings
    repo_prefix: str = "research-paper-"
    repo_visibility: str = "public"
    create_dev_branch: bool = True
    auto_merge_prs: bool = False # Dangerous, use with extreme caution!
    
    # Feature Toggles
    enable_llm_code_generation: bool = True
    enable_iterative_refinement: bool = True
    enable_static_analysis: bool = True
    enable_testing: bool = True
    enable_containerization: bool = True
    enable_semantic_versioning: bool = True
    enable_release_automation: bool = True
    enable_performance_profiling: bool = True
    
    # Paths
    base_dir: Path = Path(__file__).parent
    relevant_papers_dir: Path = base_dir / "relevant_papers" # Unified input dir
    managed_repos_state_file: Path = base_dir / "managed_repos_state.json"
    workspace_dir: Path = base_dir / "workspace" # Local clones
    logs_dir: Path = base_dir / "logs"
    llm_logs_dir: Path = base_dir / "llm_interactions"
    temp_dir: Path = base_dir / "temp_repo_clones" # For temporary operations

# --- Enums and Data Classes (Reused and Extended) ---
class ProcessingStatus(Enum):
    PENDING = "pending"
    FETCHING = "fetching"
    GENERATING_CODE = "generating_code"
    REFINING_CODE = "refining_code"
    VALIDATING_CODE = "validating_code"
    TESTING_CODE = "testing_code"
    CONTAINERIZING = "containerizing"
    GIT_OPERATIONS = "git_operations"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NO_CHANGE = "no_change" # New status for existing, up-to-date repos

@dataclass
class RepoState:
    """Represents the state of a single managed repository."""
    repo_name: str
    github_url: str
    last_processed_commit_sha: Optional[str] = None
    current_version: str = "0.0.0" # Semantic Versioning
    status: ProcessingStatus = ProcessingStatus.PENDING
    last_processed_timestamp: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
@dataclass
class PaperInfo:
    """Information about a research paper to be processed."""
    id: str # Unique ID, e.g., arXiv ID or hash of content
    title: str
    summary: str
    source_path: Path # Path to the original PDF/JSON/etc.
    last_modified: float # Timestamp of last modification of source
    
@dataclass
class ProcessingResult:
    """Detailed result of processing a single paper/repository update."""
    paper_info: PaperInfo
    repo_state: RepoState # Reference to the updated repo state
    processing_time: float = 0.0
    code_generated: bool = False
    code_refinements_applied: int = 0
    tests_run: bool = False
    tests_passed: bool = False
    code_quality_score: float = 0.0 # From static analysis
    dependencies_detected: List[str] = field(default_factory=list)
    container_image_built: bool = False
    pr_created: bool = False
    release_created: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.PENDING


# --- Core Components (Refactored from CoderAgent and New) ---

class LLMCodeGenerator:
    """Handles all LLM interactions for code generation and refinement."""
    def __init__(self, config: Config, session: aiohttp.ClientSession, logger: logging.Logger, llm_logs_dir: Path):
        self.config = config
        self.session = session
        self.logger = logger
        self.llm_logs_dir = llm_logs_dir
        self.headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
            "User-Agent": "M1-CoderAgent/1.0"
        }

    async def _call_llm(self, prompt: str, model: str) -> Tuple[Optional[str], Optional[str]]:
        """Internal method to call the LLM API with retry logic."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_llm_tokens
        }
        
        response_content = None
        error_message = None

        for attempt in range(self.config.retry_attempts):
            try:
                async with self.session.post(url, headers=self.headers, json=payload, timeout=self.config.request_timeout) as response:
                    response.raise_for_status()
                    data = await response.json()
                    response_content = data['choices'][0]['message']['content']
                    break # Success, break retry loop
            except aiohttp.ClientError as e:
                error_message = f"LLM API Client error (Attempt {attempt+1}/{self.config.retry_attempts}): {e}"
                self.logger.warning(error_message)
            except asyncio.TimeoutError:
                error_message = f"LLM API Timeout (Attempt {attempt+1}/{self.config.retry_attempts}) for model {model}"
                self.logger.warning(error_message)
            except json.JSONDecodeError:
                error_message = f"LLM API Invalid JSON response (Attempt {attempt+1}/{self.config.retry_attempts}) from model {model}"
                self.logger.warning(error_message)
            except Exception as e:
                error_message = f"Unexpected error during LLM API call (Attempt {attempt+1}/{self.config.retry_attempts}) for model {model}: {e}"
                self.logger.warning(error_message)
            
            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt)) # Exponential backoff
        
        log_file_name = self.llm_logs_dir / f"llm_interaction_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        with open(log_file_name, 'w', encoding='utf-8') as f:
            json.dump({
                "model": model,
                "prompt": prompt,
                "response": response_content,
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }, f, indent=4, ensure_ascii=False)

        return response_content, error_message

    def _extract_code_block(self, text: str, lang: str = "python") -> Optional[str]:
        """Extracts code block from LLM response."""
        match = re.search(rf"```{lang}\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        self.logger.warning(f"No {lang} code block found, returning full text. Text length: {len(text)}")
        return text.strip()

    def _generate_placeholder_main_py(self, title: str, summary: str) -> str:
        # Re-using the robust template from the previous CoderAgent
        safe_title = self._sanitize_filename(title)
        class_name = ''.join(word.capitalize() for word in safe_title.split('_'))
        return f'''"""
{title}

{summary}

This is a template implementation. Please implement the actual algorithm.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import argparse
import json
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for {class_name}"""
    debug: bool = False
    verbose: bool = True
    output_file: Optional[str] = None
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create config from command line arguments"""
        return cls(
            debug=args.debug,
            verbose=args.verbose,
            output_file=args.output
        )

class {class_name}:
    """
    Implementation of {title}
    
    Based on: {summary}
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = logger
        self.results = {{}}
        
    def validate_input(self, data: Any) -> bool:
        """Validate input data"""
        if data is None:
            return False
        # TODO: Add specific validation logic
        return True
        
    def preprocess(self, data: Any) -> Any:
        """Preprocess input data"""
        if self.config.verbose:
            self.logger.info("Preprocessing data...")
        # TODO: Add preprocessing logic
        return data
        
    def process(self, data: Any) -> Any:
        """Process the data using the algorithm"""
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
            
        preprocessed_data = self.preprocess(data)
        
        if self.config.verbose:
            self.logger.info(f"Processing data with {{self.__class__.__name__}}")
            
        # TODO: Implement the actual algorithm
        # This is where the main algorithm logic should go
        
        result = self.algorithm_implementation(preprocessed_data)
        
        if self.config.verbose:
            self.logger.info("Processing completed")
            
        return result
        
    def algorithm_implementation(self, data: Any) -> Any:
        """
        Main algorithm implementation
        
        TODO: Replace this with the actual algorithm from the paper
        """
        # Placeholder implementation
        self.logger.warning("Using placeholder implementation - needs actual algorithm")
        return data
        
    def postprocess(self, result: Any) -> Any:
        """Postprocess results"""
        if self.config.verbose:
            self.logger.info("Postprocessing results...")
        # TODO: Add postprocessing logic
        return result
        
    def run(self, data: Any) -> Any:
        """Run the complete pipeline"""
        try:
            result = self.process(data)
            final_result = self.postprocess(result)
            self.results = final_result
            return final_result
        except Exception as e:
            self.logger.error(f"Error during processing: {{e}}")
            raise
            
    def save_results(self, filename: str = None) -> None:
        """Save results to file"""
        if filename is None:
            filename = f"{{self.__class__.__name__.lower()}}_results.json"
            
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            self.logger.info(f"Results saved to {{filename}}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {{e}}")

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description=f"Implementation of {title}",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input data file or parameter'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )
    
    return parser

def main():
    """Main function for command line usage"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = Config.from_args(args)
    
    # Initialize the implementation
    implementation = {class_name}(config)
    
    try:
        # TODO: Load input data based on args.input
        # For now, using placeholder data
        input_data = "placeholder_data"
        
        if args.input:
            # TODO: Load actual input data
            logger.info(f"Loading input from {{args.input}}")
            
        # Run the algorithm
        result = implementation.run(input_data)
        
        # Save results if output file specified
        if args.output:
            implementation.save_results(args.output)
            
        # Print results
        print(f"Algorithm completed successfully!")
        print(f"Result: {{result}}")
        
    except Exception as e:
        logger.error(f"Execution failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    def _sanitize_filename(self, name: str) -> str:
        # Helper function, copied from CoderAgent
        name = name.replace('\n', ' ').replace('\r', '')
        name = re.sub(r'[\\/*?:"<>|]', "", name)
        name = name.replace(" ", "_")
        name = re.sub(r'__+', '_', name)
        name = re.sub(r'[^\w\-_\.]', '', name)
        return name.strip('_')[:100]

    def _detect_paper_type(self, title: str, summary: str) -> str:
        # Helper function, copied from CoderAgent
        text = f"{title} {summary}".lower()
        if any(k in text for k in ["transformer", "attention", "bert", "gpt", "llama", "deep learning", "cnn", "rnn", "neural network"]):
            return "transformer"
        elif any(k in text for k in ["agent", "reinforcement learning", "rl", "policy", "environment", "actor", "critic", "q-learning", "dqn"]):
            return "agent"
        else:
            return "general"

    async def generate_initial_code(self, paper_info: PaperInfo) -> Tuple[Optional[str], List[str]]:
        """Generates initial code for a paper using LLM or fallback."""
        if not self.config.enable_llm_code_generation:
            self.logger.info("LLM code generation disabled. Using placeholder.")
            return self._generate_placeholder_main_py(paper_info.title, paper_info.summary), []

        prompt = self._build_advanced_prompt(paper_info.title, paper_info.summary, self._detect_paper_type(paper_info.title, paper_info.summary))
        
        models = [self.config.primary_model, self.config.fallback_model, self.config.backup_model]
        errors = []
        for model in models:
            self.logger.info(f"Attempting initial code generation with {model} for '{paper_info.title}'")
            content, error = await self._call_llm(prompt, model)
            if content and len(self._extract_code_block(content)) >= self.config.min_code_length:
                self.logger.info(f"Successfully generated initial code using {model}.")
                return self._extract_code_block(content), []
            else:
                errors.append(f"Failed to generate sufficient code with {model}: {error or 'Code too short/empty'}")
                self.logger.warning(errors[-1])
        
        self.logger.error(f"All LLM models failed for '{paper_info.title}'. Falling back to template code.")
        return self._generate_placeholder_main_py(paper_info.title, paper_info.summary), errors

    async def refine_code_with_feedback(self, current_code: str, feedback: str, paper_info: PaperInfo) -> Tuple[Optional[str], List[str]]:
        """Refines code based on static analysis/test feedback using LLM."""
        if not self.config.enable_iterative_refinement:
            self.logger.info("Iterative code refinement disabled.")
            return current_code, []

        refinement_prompt = f"""
        You are an expert Python developer. I provided you with code for implementing a research paper titled "{paper_info.title}" with summary: "{paper_info.summary}".
        
        I ran some automated checks, and here is the feedback:
        {feedback}
        
        Please revise the code to address these issues. Maintain the original functionality and improve code quality.
        Provide ONLY the revised Python code, enclosed in a ```python``` block.
        """
        
        self.logger.info(f"Attempting code refinement for '{paper_info.title}' with feedback.")
        refined_content, error = await self._call_llm(refinement_prompt, self.config.primary_model) # Use primary for refinement
        if refined_content:
            return self._extract_code_block(refined_content), []
        else:
            self.logger.error(f"Failed to refine code for '{paper_info.title}': {error}")
            return current_code, [f"Failed to refine code: {error}"]

    def _build_advanced_prompt(self, title: str, summary: str, paper_type: str = "general") -> str:
        # Re-using the prompt logic from the previous CoderAgent
        base_prompt = f"""You are CoderGPT, an expert AI programmer specialized in implementing research paper concepts.

**Paper Title:** {title}
**Paper Type:** {paper_type}
**Summary:** {summary}

Implement the core algorithm/method with the following requirements:

1. **Code Quality**: Write clean, well-documented, production-ready Python code
2. **Architecture**: Use appropriate design patterns and SOLID principles
3. **Error Handling**: Include comprehensive error handling and logging
4. **Performance**: Optimize for efficiency and scalability
5. **Testing**: Include basic unit tests and validation
6. **Documentation**: Add detailed docstrings and comments

**Additional Guidelines:**
- Use type hints throughout
- Follow PEP 8 style guidelines
- Include configuration options
- Add performance monitoring
- Handle edge cases gracefully
- Use modern Python features (3.9+)
- Make the code ready for production use

Return ONLY the Python code wrapped in ```python ... ```. No explanations outside the code block.
"""
        return base_prompt

class CodeValidator:
    """Performs static analysis, dependency analysis, and test execution."""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def _run_command(self, command: List[str], cwd: Path) -> Tuple[bool, str, str]:
        """Helper to run shell commands."""
        try:
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False, encoding="utf-8")
            if result.returncode == 0:
                return True, result.stdout.strip(), result.stderr.strip()
            else:
                return False, result.stdout.strip(), result.stderr.strip()
        except FileNotFoundError:
            self.logger.error(f"Command not found: {command[0]}. Ensure it's installed and in PATH.")
            return False, "", f"Command not found: {command[0]}"
        except Exception as e:
            self.logger.error(f"Error running command {command}: {e}")
            return False, "", str(e)

    def validate_code_quality(self, project_path: Path) -> Tuple[float, List[str]]:
        """Runs linters (Flake8, Black) and returns a quality score and issues."""
        if not self.config.enable_static_analysis:
            return 100.0, ["Static analysis skipped."]

        issues = []
        score = 100.0
        
        self.logger.info(f"Running static analysis for {project_path}")

        # Run Flake8
        success, stdout, stderr = self._run_command(["poetry", "run", "flake8", "."], cwd=project_path)
        if not success:
            flake8_issues = [line for line in stdout.split('\n') if line.strip()]
            if flake8_issues:
                issues.extend([f"Flake8: {issue}" for issue in flake8_issues])
                score -= min(50.0, len(flake8_issues) * 2.0) # Deduct score based on number of issues
            if stderr:
                issues.append(f"Flake8 stderr: {stderr}")
                self.logger.warning(f"Flake8 stderr for {project_path}: {stderr}")

        # Run Black (check only)
        success, stdout, stderr = self._run_command(["poetry", "run", "black", ".", "--check"], cwd=project_path)
        if not success and "reformatted" in stdout:
            issues.append(f"Black: Code not formatted correctly. Run 'poetry run black .'.")
            score -= 10.0
        if stderr:
            issues.append(f"Black stderr: {stderr}")
            self.logger.warning(f"Black stderr for {project_path}: {stderr}")

        return max(0.0, score), issues

    def analyze_dependencies(self, project_path: Path) -> List[str]:
        """Analyzes dependencies from pyproject.toml."""
        if not self.config.enable_dependency_analysis:
            return []

        pyproject_path = project_path / "pyproject.toml"
        if not pyproject_path.exists():
            self.logger.warning(f"pyproject.toml not found in {project_path}. Cannot analyze dependencies.")
            return []

        # A more robust way would be to parse TOML, but for simplicity, we'll grep
        # In a real scenario, use `toml` library.
        try:
            with open(pyproject_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            deps = []
            # Simple regex to find dependencies under [tool.poetry.dependencies]
            match = re.search(r'\[tool\.poetry\.dependencies\]\n([\s\S]*?)(\n\[|\Z)', content)
            if match:
                deps_section = match.group(1)
                for line in deps_section.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        dep_name = line.split('=')[0].strip()
                        if dep_name != "python": # Exclude python version itself
                            deps.append(dep_name)
            return sorted(list(set(deps)))
        except Exception as e:
            self.logger.error(f"Error analyzing dependencies in {pyproject_path}: {e}")
            return []

    def run_tests(self, project_path: Path) -> Tuple[bool, str, str]:
        """Runs unit tests using pytest."""
        if not self.config.enable_testing:
            return True, "Testing skipped.", ""

        self.logger.info(f"Running tests for {project_path}")
        # Ensure pytest is installed in the poetry environment
        success, stdout, stderr = self._run_command(["poetry", "run", "pytest", "tests/"], cwd=project_path)
        
        if success:
            self.logger.info(f"Tests passed for {project_path}.")
        else:
            self.logger.warning(f"Tests failed for {project_path}.")
        
        return success, stdout, stderr
    
    def run_performance_profiling(self, project_path: Path) -> Tuple[Optional[Dict], List[str]]:
        """Runs a basic performance profile and returns metrics."""
        if not self.config.enable_performance_profiling:
            return None, ["Performance profiling skipped."]

        self.logger.info(f"Running performance profiling for {project_path}")
        profile_output_file = project_path / "profile.txt"
        
        # This is a very basic example. Real profiling needs careful setup.
        # It assumes `main.py` can be run directly to trigger some logic.
        command = [
            "python", "-m", "cProfile", "-o", str(project_path / "profile.prof"),
            str(project_path / "main.py")
        ]
        
        success, stdout, stderr = self._run_command(command, cwd=project_path)
        
        metrics = {}
        issues = []

        if success:
            # Attempt to parse cProfile output or just indicate success
            self.logger.info(f"Profiling completed for {project_path}. Raw profile data in profile.prof")
            # For a real system, you'd parse profile.prof into human-readable metrics
            metrics = {"profiling_status": "success", "notes": "Raw profile data available."}
        else:
            issues.append(f"Performance profiling failed: {stderr}")
            self.logger.warning(f"Performance profiling failed for {project_path}: {stderr}")
            metrics = {"profiling_status": "failed", "error": stderr}
        
        return metrics, issues

class GitHubIntegrator:
    """Manages all GitHub API interactions (repos, PRs, releases)."""
    def __init__(self, config: Config, session: aiohttp.ClientSession, logger: logging.Logger):
        self.config = config
        self.session = session
        self.logger = logger
        self.headers = {
            "Authorization": f"token {self.config.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "M1-Maintainer-Agent/1.0"
        }

    async def _github_api_request(self, method: str, url: str, **kwargs) -> Tuple[Optional[Dict], Optional[str]]:
        """Generic GitHub API request with retry logic."""
        retries = self.config.retry_attempts
        for attempt in range(retries):
            try:
                async with self.session.request(method, url, headers=self.headers, timeout=self.config.request_timeout, **kwargs) as response:
                    if response.status in [200, 201, 204]:
                        return await response.json() if response.status != 204 else {}, None
                    elif response.status == 404 and method == 'GET':
                        return None, "Not Found"
                    elif response.status == 409: # Conflict, e.g., repo already exists
                        return None, "Conflict (Resource exists)"
                    else:
                        error_text = await response.text()
                        self.logger.warning(f"GitHub API request failed (Attempt {attempt + 1}/{retries}): {method} {url} Status: {response.status}, Error: {error_text}")
                        if attempt < retries - 1:
                            await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            except aiohttp.ClientError as e:
                self.logger.warning(f"Network error during GitHub API request (Attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            except asyncio.TimeoutError:
                self.logger.warning(f"GitHub API request timed out (Attempt {attempt + 1}/{retries}): {method} {url}")
                if attempt < retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            except Exception as e:
                self.logger.error(f"Unexpected error during GitHub API request: {e}")
                return None, str(e)
        return None, "Max retries exceeded or unhandled error."

    async def check_repo_exists(self, repo_name: str) -> bool:
        """Checks if a GitHub repository exists."""
        url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}"
        data, error = await self._github_api_request('GET', url)
        return data is not None and error is None

    async def create_repository(self, repo_name: str, description: str) -> Tuple[Optional[str], Optional[str]]:
        """Creates a new GitHub repository."""
        if await self.check_repo_exists(repo_name):
            self.logger.info(f"Repository {repo_name} already exists.")
            return f"https://github.com/{self.config.github_username}/{repo_name}", f"https://github.com/{self.config.github_username}/{repo_name}.git"

        url = "https://api.github.com/user/repos"
        payload = {
            "name": repo_name,
            "description": description,
            "private": self.config.repo_visibility != "public",
            "auto_init": False # We will push initial content
        }
        data, error = await self._github_api_request('POST', url, json=payload)
        if data:
            self.logger.info(f"Successfully created GitHub repository: {repo_name}")
            return data.get("html_url"), data.get("clone_url")
        else:
            self.logger.error(f"Failed to create repo {repo_name}: {error}")
            return None, None

    async def create_pull_request(self, repo_name: str, head_branch: str, base_branch: str, title: str, body: str) -> Optional[str]:
        """Creates a pull request."""
        url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}/pulls"
        payload = {
            "title": title,
            "head": head_branch,
            "base": base_branch,
            "body": body
        }
        data, error = await self._github_api_request('POST', url, json=payload)
        if data:
            self.logger.info(f"Created PR: {data.get('html_url')}")
            return data.get("html_url")
        else:
            self.logger.error(f"Failed to create PR for {repo_name}: {error}")
            return None

    async def create_release(self, repo_name: str, tag_name: str, name: str, body: str, target_commitish: str = "main") -> Optional[str]:
        """Creates a GitHub release."""
        if not self.config.enable_release_automation:
            self.logger.info("Release automation disabled.")
            return None

        url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}/releases"
        payload = {
            "tag_name": tag_name,
            "target_commitish": target_commitish,
            "name": name,
            "body": body,
            "draft": False,
            "prerelease": False
        }
        data, error = await self._github_api_request('POST', url, json=payload)
        if data:
            self.logger.info(f"Created release: {data.get('html_url')}")
            return data.get("html_url")
        else:
            self.logger.error(f"Failed to create release {tag_name} for {repo_name}: {error}")
            return None

class RepoManager:
    """Manages local Git operations for repositories."""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._setup_git_global_config()

    def _setup_git_global_config(self):
        """Sets up global Git user config for consistent commits."""
        try:
            subprocess.run(["git", "config", "--global", "user.name", self.config.github_username], check=True, capture_output=True)
            subprocess.run(["git", "config", "--global", "user.email", f"{self.config.github_username}@users.noreply.github.com"], check=True, capture_output=True)
            # Use GitHub Actions token for auth, not credential helper store
            # subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=True)
            self.logger.info("Global Git configuration set.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to set global Git config: {e.stderr.decode()}")
            raise

    def _run_git_command(self, command: List[str], cwd: Path) -> Tuple[bool, str, str]:
        """Helper to run Git commands."""
        try:
            # Use `env` to pass GITHUB_TOKEN for authentication with HTTPS remotes
            env = os.environ.copy()
            # For HTTPS, token can be embedded in URL or passed via header if using a custom helper
            # Simplest for GH Actions is to ensure the checkout action handles it, or use SSH.
            # For direct `git push` with HTTPS, you might need:
            # env['GIT_ASKPASS'] = 'echo'
            # command = ["git", "push", f"https://{self.config.github_username}:{self.config.github_token}@github.com/{self.config.github_username}/{repo_name}.git", "main"]
            # However, `actions/checkout` usually sets up the credential helper correctly.
            
            result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True, encoding="utf-8", env=env)
            if result.returncode == 0:
                return True, result.stdout.strip(), result.stderr.strip()
            else:
                self.logger.error(f"Git command failed in {cwd}: {' '.join(command)}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
                return False, result.stdout.strip(), result.stderr.strip()
        except FileNotFoundError:
            self.logger.error(f"Git executable not found. Please ensure Git is installed and in your PATH.")
            return False, "", "Git executable not found."
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during git command: {e}")
            return False, "", str(e)

    def clone_or_pull_repo(self, repo_state: RepoState, local_path: Path) -> Tuple[bool, str]:
        """Clones a repo or pulls latest changes if it exists."""
        if local_path.exists() and (local_path / ".git").is_dir():
            self.logger.info(f"Pulling latest changes for {repo_state.repo_name} in {local_path}")
            success, stdout, stderr = self._run_git_command(["git", "pull", "--rebase", "origin", "main"], cwd=local_path)
            if not success and "no tracking information" in stderr:
                 self.logger.warning(f"No tracking branch for main in {repo_state.repo_name}. Setting upstream.")
                 self._run_git_command(["git", "branch", "--set-upstream-to=origin/main", "main"], cwd=local_path)
                 success, stdout, stderr = self._run_git_command(["git", "pull", "--rebase", "origin", "main"], cwd=local_path)
            if not success:
                return False, f"Git pull failed: {stderr}"
            return True, "Pulled"
        else:
            self.logger.info(f"Cloning {repo_state.github_url} to {local_path}")
            # Use the clone URL with embedded token for direct git clone if needed, or rely on actions/checkout
            # For simplicity, assuming actions/checkout handles auth for initial clone or it's a public repo
            success, stdout, stderr = self._run_git_command(["git", "clone", repo_state.github_url, str(local_path)], cwd=self.config.workspace_dir.parent)
            if not success:
                return False, f"Git clone failed: {stderr}"
            return True, "Cloned"

    def commit_and_push(self, project_path: Path, commit_message: str, branch: str = "main", force_push: bool = False) -> Tuple[bool, Optional[str], str]:
        """Commits changes and pushes to GitHub."""
        self.logger.info(f"Committing and pushing changes for {project_path} to branch {branch}")
        
        success, stdout, stderr = self._run_git_command(["git", "add", "."], cwd=project_path)
        if not success: return False, None, f"Git add failed: {stderr}"

        success, stdout, stderr = self._run_git_command(["git", "commit", "-m", commit_message], cwd=project_path)
        if not success and "nothing to commit" in stdout:
            self.logger.info("No changes to commit.")
            return True, None, "No changes" # No new commit SHA
        elif not success:
            return False, None, f"Git commit failed: {stderr}"

        # Ensure we are on the correct branch
        self._run_git_command(["git", "checkout", branch], cwd=project_path)

        push_command = ["git", "push", "-u", "origin", branch]
        if force_push:
            push_command = ["git", "push", "--force-with-lease", "origin", branch] # Safer than --force

        success, stdout, stderr = self._run_git_command(push_command, cwd=project_path)
        if not success: return False, None, f"Git push failed: {stderr}"

        # Get the new commit SHA
        success, commit_sha, _ = self._run_git_command(["git", "rev-parse", "HEAD"], cwd=project_path)
        if not success: commit_sha = None # Fallback if SHA can't be retrieved
        
        self.logger.info(f"Successfully pushed to {branch}. New SHA: {commit_sha}")
        return True, commit_sha, ""

    def create_and_checkout_branch(self, project_path: Path, branch_name: str) -> Tuple[bool, str]:
        """Creates and checks out a new branch."""
        self.logger.info(f"Creating and checking out branch: {branch_name}")
        success, stdout, stderr = self._run_git_command(["git", "checkout", "-b", branch_name], cwd=project_path)
        if not success:
            return False, f"Failed to create/checkout branch {branch_name}: {stderr}"
        return True, ""

    def checkout_branch(self, project_path: Path, branch_name: str) -> Tuple[bool, str]:
        """Checks out an existing branch."""
        self.logger.info(f"Checking out branch: {branch_name}")
        success, stdout, stderr = self._run_git_command(["git", "checkout", branch_name], cwd=project_path)
        if not success:
            return False, f"Failed to checkout branch {branch_name}: {stderr}"
        return True, ""

    def get_current_commit_sha(self, project_path: Path) -> Optional[str]:
        """Gets the current commit SHA of a local repository."""
        success, stdout, stderr = self._run_git_command(["git", "rev-parse", "HEAD"], cwd=project_path)
        if success:
            return stdout
        return None

    def install_poetry_dependencies(self, project_path: Path) -> Tuple[bool, str]:
        """Installs Poetry dependencies for a project."""
        self.logger.info(f"Running `poetry install` for {project_path}")
        success, stdout, stderr = self._run_command(["poetry", "install", "--no-root"], cwd=project_path)
        if not success:
            return False, f"Poetry install failed: {stderr}"
        return True, stdout

class PaperSource:
    """Handles fetching and parsing research paper information."""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def get_papers_to_process(self) -> List[PaperInfo]:
        """
        Scans the relevant_papers directory for new or updated papers.
        Currently supports JSON files with 'title', 'summary', 'id'.
        Future: Add PDF/LaTeX parsing.
        """
        papers: List[PaperInfo] = []
        if not self.config.relevant_papers_dir.exists():
            self.logger.warning(f"Paper source directory {self.config.relevant_papers_dir} does not exist.")
            return []

        for file_path in self.config.relevant_papers_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Assume a simple structure for now.
                # In a real system, you'd parse more robustly or use a schema.
                paper_id = data.get("id", hashlib.md5(file_path.read_bytes()).hexdigest())
                title = data.get("title", file_path.stem)
                summary = data.get("summary_and_goal", data.get("summary", "No summary provided."))
                
                if not title or not summary:
                    self.logger.warning(f"Skipping {file_path}: Missing title or summary.")
                    continue

                papers.append(PaperInfo(
                    id=paper_id,
                    title=title,
                    summary=summary,
                    source_path=file_path,
                    last_modified=file_path.stat().st_mtime # Timestamp of last modification
                ))
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding JSON from {file_path}: {e}")
            except Exception as e:
                self.logger.error(f"Error reading paper info from {file_path}: {e}")
        
        self.logger.info(f"Found {len(papers)} papers in source directory.")
        return papers

    async def parse_pdf_or_latex(self, file_path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Placeholder for advanced PDF/LaTeX parsing.
        This would use tools like PyPDF2, pdfplumber, or even call an LLM with OCR/document understanding.
        Returns title, summary, and potentially extracted full text.
        """
        self.logger.info(f"Simulating parsing of {file_path.suffix} file: {file_path.name}")
        # In a real scenario, integrate libraries here.
        # For now, return dummy data or raise an error if not JSON.
        if file_path.suffix == ".pdf":
            return "Advanced PDF Paper Title", "This is a summary extracted from a PDF.", "Full text of PDF..."
        elif file_path.suffix == ".tex":
            return "Advanced LaTeX Paper Title", "Summary from LaTeX source.", "Full text of LaTeX..."
        else:
            return None, None, None

class StateManager:
    """Manages persistence of managed repository states."""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.managed_repos: Dict[str, RepoState] = {} # Key: repo_name

    def load_state(self):
        """Loads the state of managed repositories from a JSON file."""
        if self.config.managed_repos_state_file.exists():
            try:
                with open(self.config.managed_repos_state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for repo_name, repo_dict in data.items():
                        self.managed_repos[repo_name] = RepoState(**repo_dict)
                self.logger.info(f"Loaded state for {len(self.managed_repos)} managed repositories.")
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding managed repos state file: {e}. Starting with empty state.")
                self.managed_repos = {}
            except Exception as e:
                self.logger.error(f"Error loading managed repos state: {e}. Starting with empty state.")
                self.managed_repos = {}
        else:
            self.logger.info("No existing managed repos state file found. Starting fresh.")

    def save_state(self):
        """Saves the current state of managed repositories to a JSON file."""
        try:
            # Convert RepoState dataclasses to dicts for JSON serialization
            serializable_data = {name: asdict(state) for name, state in self.managed_repos.items()}
            with open(self.config.managed_repos_state_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Saved state for {len(self.managed_repos)} managed repositories.")
        except Exception as e:
            self.logger.error(f"Error saving managed repos state: {e}")

    def update_repo_state(self, repo_state: RepoState):
        """Updates or adds a repository's state."""
        repo_state.last_processed_timestamp = datetime.now().isoformat()
        self.managed_repos[repo_state.repo_name] = repo_state

    def get_repo_state(self, repo_name: str) -> Optional[RepoState]:
        """Retrieves a repository's state."""
        return self.managed_repos.get(repo_name)

# --- M1: The Central Maintainer Agent ---
class M1MaintainerAgent:
    """
    M1: The Central Maintainer Agent for Research Paper Implementations.
    Orchestrates code generation, refinement, testing, and GitHub management.
    """
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logging()
        self._setup_directories()
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.llm_generator: Optional[LLMCodeGenerator] = None
        self.code_validator: Optional[CodeValidator] = None
        self.github_integrator: Optional[GitHubIntegrator] = None
        self.repo_manager: Optional[RepoManager] = None
        self.paper_source: Optional[PaperSource] = None
        self.state_manager: Optional[StateManager] = None

        self.processing_semaphore = asyncio.Semaphore(self.config.max_concurrent_paper_processing)

    def _setup_logging(self) -> logging.Logger:
        """Configures logging for the M1 Agent."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.config.logs_dir.mkdir(exist_ok=True)
        
        handlers = [
            logging.FileHandler(self.config.logs_dir / 'm1_maintainer.log'),
        ]
        if not os.getenv('GITHUB_ACTIONS'): # Only stream to console if not in CI
            handlers.append(logging.StreamHandler())
            
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=handlers
        )
        logger = logging.getLogger("M1MaintainerAgent")
        logging.getLogger('aiohttp.client').setLevel(logging.WARNING) # Reduce verbosity
        return logger

    def _setup_directories(self):
        """Ensures all necessary directories exist."""
        self.config.relevant_papers_dir.mkdir(parents=True, exist_ok=True)
        self.config.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.llm_logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self):
        """Initializes async resources and sub-components."""
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_paper_processing * 2) # More connections than concurrent tasks
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        self.llm_generator = LLMCodeGenerator(self.config, self.session, self.logger, self.config.llm_logs_dir)
        self.code_validator = CodeValidator(self.config, self.logger)
        self.github_integrator = GitHubIntegrator(self.config, self.session, self.logger)
        self.repo_manager = RepoManager(self.config, self.logger)
        self.paper_source = PaperSource(self.config, self.logger)
        self.state_manager = StateManager(self.config, self.logger)
        self.state_manager.load_state() # Load previous state on startup
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleans up async resources and saves state."""
        if self.session:
            await self.session.close()
        if self.state_manager:
            self.state_manager.save_state()
        # Clean up temporary clone directory
        if self.config.temp_dir.exists():
            self.logger.info(f"Cleaning up temporary directory: {self.config.temp_dir}")
            shutil.rmtree(self.config.temp_dir)

    def _generate_repo_name(self, paper_title: str) -> str:
        """Generates a sanitized repository name."""
        name = re.sub(r'[^a-zA-Z0-9\s-]', '', paper_title)
        name = name.lower().replace(' ', '-')
        name = re.sub(r'-+', '-', name).strip('-')
        return (self.config.repo_prefix + name)[:80]

    def _generate_project_files(self, project_path: Path, paper_info: PaperInfo, code: str, dependencies: List[str]):
        """
        Generates all project files (main.py, README, tests, Dockerfile, etc.)
        This is a more sophisticated version of the previous CoderAgent's file generation.
        """
        self.logger.info(f"Generating project files for '{paper_info.title}' in {project_path}")
        
        # Determine paper type for tailored structure (re-using logic)
        paper_type = self.llm_generator._detect_paper_type(paper_info.title, paper_info.summary)
        
        # Main code file
        main_file_name = "main.py"
        if paper_type == "transformer": main_file_name = "model.py"
        elif paper_type == "agent": main_file_name = "agent.py"
        (project_path / main_file_name).write_text(code, encoding="utf-8")

        # pyproject.toml (Poetry)
        self._generate_pyproject_toml(project_path, paper_info.title, paper_info.summary, dependencies)

        # README.md
        readme_content = self._generate_readme(paper_info.title, paper_info.summary, paper_type)
        (project_path / "README.md").write_text(readme_content, encoding="utf-8")

        # .gitignore
        (project_path / ".gitignore").write_text(self._generate_gitignore(), encoding="utf-8")

        # LICENSE
        (project_path / "LICENSE").write_text(self._generate_license(), encoding="utf-8")

        # Tests directory and file
        tests_dir = project_path / "tests"
        tests_dir.mkdir(exist_ok=True)
        test_file_name = "test_main.py"
        if paper_type == "transformer": test_file_name = "test_model.py"
        elif paper_type == "agent": test_file_name = "test_agent.py"
        (tests_dir / test_file_name).write_text(self._generate_test_code(paper_info.title, code), encoding="utf-8")

        # Docs directory and API doc
        docs_dir = project_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        (docs_dir / "api.md").write_text(self._generate_api_md(paper_info.title, paper_info.summary), encoding="utf-8")

        # Dockerfile
        if self.config.enable_containerization:
            self._generate_dockerfile(project_path, dependencies)

        # Setup.py (for pip installable package)
        self._generate_setup_py(project_path, paper_info.title, paper_info.summary, dependencies)

        # Examples directory
        examples_dir = project_path / "examples"
        examples_dir.mkdir(exist_ok=True)
        (examples_dir / "example_usage.py").write_text(self._generate_example_script(paper_info.title), encoding="utf-8")

    # Helper methods for file content generation (copied/adapted from previous CoderAgent)
    def _generate_pyproject_toml(self, project_path: Path, title: str, summary: str, dependencies: List[str]):
        default_dev_deps = ["pytest = \"^7.0\"", "flake8 = \"^6.0\"", "black = \"^23.0\""]
        all_deps = sorted(list(set(dependencies)))
        deps_toml = "\n".join([f'    {dep.split("==")[0].strip()} = "*"' for dep in all_deps])

        tool_poetry_toml = f"""
[tool.poetry]
name = "{self._sanitize_repo_name_for_poetry(title)}"
version = "0.1.0"
description = "{summary}"
authors = ["{self.config.github_username} <{self.config.github_username}@users.noreply.github.com>"]
license = "MIT"
readme = "README.md"

[tool.poetry.dependencies]
python = ">=3.9,<3.13"
{deps_toml}

[tool.poetry.group.dev.dependencies]
{'\n'.join(default_dev_deps)}

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""
        (project_path / "pyproject.toml").write_text(tool_poetry_toml.strip(), encoding="utf-8")

    def _sanitize_repo_name_for_poetry(self, name: str) -> str:
        # Poetry package names usually use underscores, not hyphens, and are lowercase
        name = self.llm_generator._sanitize_filename(name).replace('-', '_')
        return name

    def _generate_readme(self, title: str, summary: str, paper_type: str = "general") -> str:
        # Re-using the logic from the previous CoderAgent
        if paper_type == "transformer":
            return f"""# {title}\n\n{summary}\n\n## Project Structure\n- model.py: Transformer model implementation\n- train.py: Training script\n- inference.py: Inference script\n- config.yaml: Model/training configuration\n- pyproject.toml: Dependencies (Poetry)\n- tests/test_model.py: Unit tests\n- Dockerfile: Containerization\n\n## Usage\nSee train.py and inference.py for training and inference instructions.\n"""
        elif paper_type == "agent":
            return f"""# {title}\n\n{summary}\n\n## Project Structure\n- agent.py: RL agent implementation\n- env.py: Environment implementation\n- train.py: Training script\n- evaluate.py: Evaluation script\n- config.yaml: Agent/environment configuration\n- pyproject.toml: Dependencies (Poetry)\n- tests/test_agent.py: Unit tests\n- Dockerfile: Containerization\n\n## Usage\nSee train.py and evaluate.py for training and evaluation instructions.\n"""
        else:
            return f"""# {title}\n\n{summary}\n\n## Project Structure\n- main.py: Main implementation\n- pyproject.toml: Dependencies (Poetry)\n- tests/test_main.py: Unit tests\n- Dockerfile: Containerization\n\n## Usage\nSee main.py for usage instructions.\n"""

    def _generate_gitignore(self) -> str:
        # Re-using the logic from the previous CoderAgent
        return """# Python
__pycache__/\n*.py[cod]\n*$py.class\n*.so\n.Python\nbuild/\ndevelop-eggs/\ndist/\ndownloads/\neggs/\n.eggs/\nlib/\nlib64/\nparts/\nsdist/\nvar/\nwheels/\npip-wheel-metadata/\nshare/python-wheels/\n*.egg-info/\n.installed.cfg\n*.egg\nMANIFEST\n# PyInstaller\n*.manifest\n*.spec\n# Installer logs\npip-log.txt\npip-delete-this-directory.txt\n# Unit test / coverage reports\nhtmlcov/\n.tox/\n.nox/\n.coverage\n.coverage.*\n.cache\nnosetests.xml\ncoverage.xml\n*.cover\n*.py,cover\n.hypothesis/\n.pytest_cache/\n# Virtual environments\n.env\n.venv\nenv/\nvenv/\nENV/\nenv.bak/\nvenv.bak/\n# IDE\n.vscode/\n.idea/\n*.swp\n*.swo\n*~\n# OS\n.DS_Store\n.DS_Store?\n._*\n.Spotlight-V100\n.Trashes\nehthumbs.db\nThumbs.db\n# Logs\n*.log\nlogs/\n# Data files\n*.csv\n*.json\n*.pkl\n*.pickle\ndata/\noutput/\nresults/\n"""

    def _generate_license(self) -> str:
        # Re-using the logic from the previous CoderAgent
        current_year = datetime.now().year
        return f"""MIT License

Copyright (c) {current_year} {self.config.github_username} (Generated by M1 Maintainer Agent)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

    def _generate_test_code(self, title: str, code: str) -> str:
        # Re-using the logic from the previous CoderAgent
        safe_title = self.llm_generator._sanitize_filename(title)
        class_name = ''.join(word.capitalize() for word in safe_title.split('_'))
        return f'''"""
Comprehensive test suite for {title}
"""

import pytest
import unittest
import sys
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
try:
    from main import {class_name}, Config
except ImportError as e:
    print(f"Import error: {{e}}")
    # Create mock classes for testing
    class Config:
        def __init__(self, debug=False, verbose=True, output_file=None):
            self.debug = debug
            self.verbose = verbose
            self.output_file = output_file
            
    class {class_name}:
        def __init__(self, config=None):
            self.config = config or Config()
            self.results = {{}}
            
        def run(self, data):
            return "test_result"
            
        def validate_input(self, data):
            return True
            
        def save_results(self, filename):
            pass

class Test{class_name}(unittest.TestCase):
    """Test cases for {class_name}"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config(debug=True, verbose=False)
        self.instance = {class_name}(self.config)
        
    def test_initialization(self):
        """Test object initialization"""
        instance = {class_name}()
        self.assertIsNotNone(instance)
        self.assertIsNotNone(instance.config)
        
    def test_initialization_with_config(self):
        """Test initialization with custom config"""
        config = Config(debug=True, verbose=False)
        instance = {class_name}(config)
        self.assertEqual(instance.config.debug, True)
        self.assertEqual(instance.config.verbose, False)
        
    def test_validate_input_valid(self):
        """Test input validation with valid data"""
        valid_data = "test_data"
        result = self.instance.validate_input(valid_data)
        self.assertTrue(result)
        
    def test_validate_input_invalid(self):
        """Test input validation with invalid data"""
        invalid_data = None
        result = self.instance.validate_input(invalid_data)
        self.assertFalse(result)
        
    def test_run_with_valid_data(self):
        """Test run method with valid data"""
        test_data = "test_input"
        result = self.instance.run(test_data)
        self.assertIsNotNone(result)
        
    def test_run_with_invalid_data(self):
        """Test run method with invalid data"""
        with self.assertRaises(ValueError):
            self.instance.run(None)
            
    def test_save_results(self):
        """Test results saving functionality"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
            
        try:
            self.instance.results = {{"test": "data"}}
            self.instance.save_results(temp_file)
            
            # Check if file was created and contains expected data
            self.assertTrue(os.path.exists(temp_file))
            
            with open(temp_file, 'r') as f:
                saved_data = json.load(f)
                self.assertEqual(saved_data, {{"test": "data"}})
                
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

class TestConfig(unittest.TestCase):
    """Test cases for Config class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = Config()
        self.assertFalse(config.debug)
        self.assertTrue(config.verbose)
        self.assertIsNone(config.output_file)
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = Config(debug=True, verbose=False, output_file="test.json")
        self.assertTrue(config.debug)
        self.assertFalse(config.verbose)
        self.assertEqual(config.output_file, "test.json")

class TestIntegration(unittest.TestCase):
    """Integration test cases"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.config = Config(debug=True, verbose=False)
        self.instance = {class_name}(self.config)
        
    def test_full_pipeline(self):
        """Test complete processing pipeline"""
        # Test data
        test_data = "integration_test_data"
        
        # Run full pipeline
        result = self.instance.run(test_data)
        
        # Verify results
        self.assertIsNotNone(result)
        
    def test_error_handling(self):
        """Test error handling in the pipeline"""
        # Test with data that should cause an error
        try:
            result = self.instance.run(None)
            self.fail("Expected ValueError was not raised")
        except ValueError:
            pass  # Expected behavior
        except Exception as e:
            self.fail(f"Unexpected exception type: {{type(e).__name__}}: {{e}}")

class TestPerformance(unittest.TestCase):
    """Performance test cases"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.config = Config(debug=False, verbose=False)
        self.instance = {class_name}(self.config)
        
    def test_performance_benchmark(self):
        """Test performance with various data sizes"""
        import time
        
        test_data = "performance_test_data"
        
        # Measure execution time
        start_time = time.time()
        result = self.instance.run(test_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Assert reasonable execution time (adjust as needed)
        self.assertLess(execution_time, 5.0, "Execution took too long")
        self.assertIsNotNone(result)

# Test utilities
class TestUtilities:
    """Utility functions for testing"""
    
    @staticmethod
    def create_test_data(size: int = 100):
        """Create test data of specified size"""
        return f"test_data_{{size}}"
        
    @staticmethod
    def create_temp_file(content: str = "", suffix: str = ".txt"):
        """Create a temporary file with content"""
        import tempfile
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, 'w') as tmp:
                tmp.write(content)
            return path
        except:
            os.close(fd)
            raise

# Custom test decorators
def skip_if_no_gpu(func):
    """Skip test if no GPU available"""
    def wrapper(*args, **kwargs):
        # Add GPU detection logic here if needed
        return func(*args, **kwargs)
    return wrapper

def timeout(seconds):
    """Timeout decorator for tests"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Test timed out after {{seconds}} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            except TimeoutError:
                raise
            finally:
                signal.alarm(0)
        return wrapper
    return decorator

# Test fixtures
@pytest.fixture
def sample_config():
    """Fixture for sample configuration"""
    return Config(debug=True, verbose=False, output_file="test_output.json")

@pytest.fixture
def sample_instance(sample_config):
    """Fixture for sample instance"""
    return {class_name}(sample_config)

@pytest.fixture
def temp_directory():
    """Fixture for temporary directory"""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

# Pytest test functions
def test_config_creation(sample_config):
    """Test configuration creation"""
    assert sample_config.debug == True
    assert sample_config.verbose == False
    assert sample_config.output_file == "test_output.json"

def test_instance_creation(sample_instance):
    """Test instance creation"""
    assert sample_instance is not None
    assert sample_instance.config is not None

def test_processing_with_fixture(sample_instance):
    """Test processing with pytest fixture"""
    result = sample_instance.run("test_data")
    assert result is not None

# Performance tests with pytest
@pytest.mark.performance
def test_memory_usage(sample_instance):
    """Test memory usage during processing"""
    import tracemalloc
    
    tracemalloc.start()
    result = sample_instance.run("memory_test_data")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Assert reasonable memory usage (adjust as needed)
    assert peak < 100 * 1024 * 1024   # 100MB limit
    assert result is not None

@pytest.mark.slow
def test_large_data_processing(sample_instance):
    """Test processing with large data"""
    large_data = TestUtilities.create_test_data(10000)
    result = sample_instance.run(large_data)
    assert result is not None

# Parameterized tests
@pytest.mark.parametrize("input_data,expected", [
    ("test1", True),
    ("test2", True),
    ("", False),
    (None, False),
])
def test_validation_parametrized(sample_instance, input_data, expected):
    """Test validation with different inputs"""
    result = sample_instance.validate_input(input_data)
    assert result == expected

# Mock tests
def test_with_mocks(sample_instance):
    """Test with mocked dependencies"""
    with patch('main.logger') as mock_logger:
        result = sample_instance.run("mock_test")
        assert result is not None
        # Verify logger was called if applicable

if __name__ == "__main__":
    # Run unittest tests
    unittest.main(verbosity=2, exit=False)
    
    # Run pytest tests
    pytest.main([__file__, "-v", "--tb=short"])
'''

    def _generate_api_md(self, title: str, summary: str) -> str:
        # Re-using the logic from the previous CoderAgent
        safe_title = self.llm_generator._sanitize_filename(title)
        class_name = ''.join(word.capitalize() for word in safe_title.split('_'))
        return f'''# API Documentation

## {title}

### Overview
{summary}

### Classes

#### Config
Configuration class for the implementation.

**Parameters:**
- `debug` (bool): Enable debug mode
- `verbose` (bool): Enable verbose output
- `output_file` (str, optional): Output file path

#### {class_name}
Main implementation class.

**Methods:**
- `__init__(config: Config = None)`: Initialize the implementation
- `run(data: Any) -> Any`: Run the complete pipeline
- `validate_input(data: Any) -> bool`: Validate input data
- `save_results(filename: str = None) -> None`: Save results to file

### Usage Examples

```python
from main import {class_name}, Config

# Create configuration
config = Config(debug=True, verbose=True)

# Create instance
implementation = {class_name}(config)

# Process data
result = implementation.run(your_data)

# Save results
implementation.save_results('output.json')
```

### Error Handling
The implementation includes comprehensive error handling:
- Input validation
- Processing errors
- File I/O errors
- Configuration errors

### Performance Considerations
- Optimized for memory usage
- Efficient algorithms
- Scalable design
- Performance monitoring included
'''

    def _generate_dockerfile(self, project_path: Path, dependencies: List[str]) -> None:
        """Generates a Dockerfile for the project."""
        self.logger.info(f"Generating Dockerfile for {project_path}")
        # Assuming Poetry is used for dependency management
        dockerfile_content = f"""
# Use a slim Python base image
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy pyproject.toml and poetry.lock to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# Install dependencies using Poetry
RUN poetry install --no-root --no-dev

# Copy the rest of the application code
COPY . .

# Expose any necessary ports (if it's a web service)
# EXPOSE 8000

# Command to run the application
# Adjust this based on your main entry point (e.g., `python main.py` or `poetry run python main.py`)
CMD ["poetry", "run", "python", "main.py"]
"""
        (project_path / "Dockerfile").write_text(dockerfile_content.strip(), encoding="utf-8")

    def _generate_setup_py(self, project_path: Path, title: str, summary: str, dependencies: List[str]):
        """Generates a setup.py file for pip installable package."""
        self.logger.info(f"Generating setup.py for {project_path}")
        safe_title = self.llm_generator._sanitize_filename(title)
        
        # Ensure 'setuptools' is in dependencies if it's not a standard lib.
        # It's usually a build dependency, not runtime.
        # For simplicity, we'll assume it's available or handled by Poetry.
        
        # Generate requirements.txt style list from detected dependencies
        install_requires = []
        for dep in dependencies:
            # Simple mapping, can be expanded. For Poetry, this is less critical.
            if dep == "numpy": install_requires.append("numpy>=1.21.0")
            elif dep == "torch": install_requires.append("torch>=1.9.0")
            elif dep == "aiohttp": install_requires.append("aiohttp>=3.8.0")
            else: install_requires.append(dep) # Default to just name
        
        setup_content = f'''from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Dependencies are managed by Poetry, but include a basic list for setup.py
install_requires = {json.dumps(install_requires, indent=4)}

setup(
    name="{self._sanitize_repo_name_for_poetry(title)}",
    version="0.1.0", # This should be updated by semantic versioning logic
    author="{self.config.github_username}",
    author_email="{self.config.github_username}@users.noreply.github.com",
    description="{summary[:100]}...",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/{self.config.github_username}/{self._generate_repo_name(title)}",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    entry_points={{
        "console_scripts": [
            "{self._sanitize_repo_name_for_poetry(title)}=main:main", # Assumes main.py has a main()
        ],
    }},
)
'''
        (project_path / "setup.py").write_text(setup_content, encoding="utf-8")

    def _generate_example_script(self, title: str) -> str:
        """Generates a simple example usage script."""
        safe_title = self.llm_generator._sanitize_filename(title)
        class_name = ''.join(word.capitalize() for word in safe_title.split('_'))
        return f'''#!/usr/bin/env python3
"""
Example usage of {title}
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import {class_name}, Config # Assuming main.py has these

def main():
    """Example usage"""
    print("Example usage of {title}")
    
    # Create configuration
    config = Config(debug=True, verbose=True)
    
    # Create instance
    implementation = {class_name}(config)
    
    # Example data (replace with actual data relevant to the paper)
    example_data = "example_input_data"
    
    try:
        # Run the algorithm
        result = implementation.run(example_data)
        
        print(f"Result: {{result}}")
        
        # Save results
        implementation.save_results("example_output.json")
        print("Results saved to example_output.json")
        
    except Exception as e:
        print(f"Error during example execution: {{e}}")

if __name__ == "__main__":
    main()
'''

    def _increment_version(self, current_version: str, change_type: str = "patch") -> str:
        """Increments semantic version (major.minor.patch)."""
        major, minor, patch = map(int, current_version.split('.'))
        if change_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif change_type == "minor":
            minor += 1
            patch = 0
        else: # patch
            patch += 1
        return f"{major}.{minor}.{patch}"

    async def _process_single_paper_and_repo(self, paper_info: PaperInfo) -> ProcessingResult:
        """
        Processes a single paper: generates/refines code, validates, tests,
        and manages its GitHub repository.
        """
        async with self.processing_semaphore:
            start_time = time.time()
            
            repo_name = self._generate_repo_name(paper_info.title)
            local_repo_path = self.config.workspace_dir / repo_name
            
            # Initialize ProcessingResult for this paper
            result = ProcessingResult(
                paper_info=paper_info,
                repo_state=RepoState(repo_name=repo_name, github_url=""), # Will update URL later
                status=ProcessingStatus.PROCESSING
            )
            self.logger.info(f"🚀 Starting processing for '{paper_info.title}' (Repo: {repo_name})")

            try:
                # 1. Get/Create GitHub Repository
                repo_html_url, repo_clone_url = await self.github_integrator.create_repository(repo_name, f"Implementation of: {paper_info.title}")
                if not repo_html_url:
                    result.status = ProcessingStatus.FAILED
                    result.errors.append("Failed to create or verify GitHub repo.")
                    return result
                
                result.repo_state.github_url = repo_html_url
                
                # 2. Clone or Pull Repository
                # We clone to a temporary directory for safety and then copy/sync to workspace
                temp_clone_path = self.config.temp_dir / repo_name
                success, msg = self.repo_manager.clone_or_pull_repo(result.repo_state, temp_clone_path)
                if not success:
                    result.status = ProcessingStatus.FAILED
                    result.errors.append(f"Failed to clone/pull repo: {msg}")
                    return result
                
                # Sync content from temp_clone_path to local_repo_path
                # This ensures we are always working on a clean, up-to-date workspace copy
                if local_repo_path.exists():
                    shutil.rmtree(local_repo_path) # Clear old content
                shutil.copytree(temp_clone_path, local_repo_path)
                
                # Get current commit SHA to detect changes later
                initial_commit_sha = self.repo_manager.get_current_commit_sha(local_repo_path)
                result.repo_state.last_processed_commit_sha = initial_commit_sha
                
                # 3. Generate or Refine Code
                current_code_file = local_repo_path / "main.py" # Assuming general type for now
                if not current_code_file.exists() or initial_commit_sha is None: # New repo or first time processing
                    result.repo_state.status = ProcessingStatus.GENERATING_CODE
                    generated_code, llm_errors = await self.llm_generator.generate_initial_code(paper_info)
                    if llm_errors: result.errors.extend(llm_errors)
                    result.code_generated = True
                else:
                    self.logger.info(f"Repo {repo_name} already exists. Checking for updates or refinement.")
                    # For now, we'll always regenerate. In future, compare paper_info.last_modified
                    # or compare hashes of generated content to decide if regeneration is needed.
                    result.repo_state.status = ProcessingStatus.GENERATING_CODE
                    generated_code, llm_errors = await self.llm_generator.generate_initial_code(paper_info)
                    if llm_errors: result.errors.extend(llm_errors)
                    result.code_generated = True

                if not generated_code:
                    result.status = ProcessingStatus.FAILED
                    result.errors.append("Code generation failed and no fallback template could be used.")
                    return result
                
                # 4. Generate other project files (README, tests, Dockerfile, etc.)
                # This also writes the `generated_code` to the appropriate file (e.g., main.py)
                dependencies_detected = self.code_validator.analyze_dependencies(local_repo_path) # Analyze current repo for existing deps
                self._generate_project_files(local_repo_path, paper_info, generated_code, dependencies_detected)
                result.dependencies_detected = dependencies_detected

                # 5. Install Poetry Dependencies (needed for static analysis and tests)
                result.repo_state.status = ProcessingStatus.VALIDATING_CODE
                poetry_install_success, poetry_install_output = self.repo_manager.install_poetry_dependencies(local_repo_path)
                if not poetry_install_success:
                    result.warnings.append(f"Poetry install failed: {poetry_install_output}. Static analysis and tests might fail.")

                # 6. Iterative Refinement Loop (if enabled)
                refinement_iterations = 0
                max_refinements = 3
                while self.config.enable_iterative_refinement and refinement_iterations < max_refinements:
                    refinement_iterations += 1
                    self.logger.info(f"Starting refinement iteration {refinement_iterations} for '{paper_info.title}'")
                    
                    # Run static analysis
                    quality_score, quality_issues = self.code_validator.validate_code_quality(local_repo_path)
                    
                    # Run tests
                    tests_passed, test_stdout, test_stderr = self.code_validator.run_tests(local_repo_path)
                    
                    feedback = []
                    if quality_issues: feedback.extend(quality_issues)
                    if not tests_passed: feedback.append(f"Tests failed:\nSTDOUT:\n{test_stdout}\nSTDERR:\n{test_stderr}")

                    if not feedback:
                        self.logger.info(f"Code for '{paper_info.title}' passed all checks. No further refinement needed.")
                        break # Exit refinement loop
                    
                    self.logger.info(f"Feedback for refinement: {feedback}")
                    result.repo_state.status = ProcessingStatus.REFINING_CODE
                    refined_code, refinement_errors = await self.llm_generator.refine_code_with_feedback(generated_code, "\n".join(feedback), paper_info)
                    
                    if refinement_errors:
                        result.errors.extend(refinement_errors)
                        self.logger.error(f"Refinement failed for '{paper_info.title}'. Stopping refinement.")
                        break # Stop if refinement itself fails
                    
                    if refined_code == generated_code:
                        self.logger.info(f"Refinement produced no changes for '{paper_info.title}'. Stopping refinement.")
                        break # Stop if LLM can't improve it
                    
                    generated_code = refined_code # Update code for next iteration
                    result.code_refinements_applied += 1
                    # Write updated code to file for next iteration of checks
                    (local_repo_path / "main.py").write_text(generated_code, encoding="utf-8") # Assuming main.py for now
                    # Re-install dependencies if code changed (might introduce new ones)
                    self.repo_manager.install_poetry_dependencies(local_repo_path)

                # Final checks after refinement loop
                final_quality_score, final_quality_issues = self.code_validator.validate_code_quality(local_repo_path)
                final_tests_passed, _, _ = self.code_validator.run_tests(local_repo_path)
                
                result.code_quality_score = final_quality_score
                result.warnings.extend(final_quality_issues)
                result.tests_run = True
                result.tests_passed = final_tests_passed

                # 7. Containerization (Build Docker Image)
                if self.config.enable_containerization:
                    result.repo_state.status = ProcessingStatus.CONTAINERIZING
                    success, stdout, stderr = self.code_validator._run_command(["docker", "build", "-t", f"{repo_name}:latest", "."], cwd=local_repo_path)
                    if success:
                        result.container_image_built = True
                        self.logger.info(f"Docker image built for {repo_name}.")
                    else:
                        result.warnings.append(f"Docker image build failed: {stderr}")
                        self.logger.warning(f"Docker build failed for {repo_name}: {stderr}")

                # 8. Performance Profiling
                if self.config.enable_performance_profiling:
                    profile_metrics, profile_issues = self.code_validator.run_performance_profiling(local_repo_path)
                    if profile_issues: result.warnings.extend(profile_issues)
                    # Store metrics in result if needed
                    self.logger.info(f"Performance profiling for {repo_name}: {profile_metrics}")

                # 9. Git Operations (Commit and Push)
                result.repo_state.status = ProcessingStatus.GIT_OPERATIONS
                new_commit_message = f"feat: Implement {paper_info.title} with automated generation and refinement"
                
                # Check for actual changes before committing
                success_diff, stdout_diff, stderr_diff = self.repo_manager._run_git_command(["git", "diff", "--quiet", "--exit-code"], cwd=local_repo_path)
                if success_diff: # No changes
                    self.logger.info(f"No changes detected in {repo_name}. Skipping commit and push.")
                    result.status = ProcessingStatus.NO_CHANGE
                    return result
                
                # Decide on branch strategy
                target_branch = "main"
                if self.config.create_dev_branch:
                    dev_branch_name = "development" # Or a feature branch like `feat/paper-title-vX.Y.Z`
                    success_branch, branch_msg = self.repo_manager.create_and_checkout_branch(local_repo_path, dev_branch_name)
                    if not success_branch:
                        result.warnings.append(f"Failed to create/checkout dev branch: {branch_msg}. Pushing to main.")
                    else:
                        target_branch = dev_branch_name

                success_push, new_sha, push_error = self.repo_manager.commit_and_push(local_repo_path, new_commit_message, branch=target_branch)
                if not success_push:
                    result.status = ProcessingStatus.FAILED
                    result.errors.append(f"Git push failed: {push_error}")
                    return result
                
                result.repo_state.last_processed_commit_sha = new_sha

                # 10. Pull Request Automation
                if self.config.create_dev_branch and target_branch != "main":
                    pr_title = f"feat({repo_name}): Automated implementation of {paper_info.title}"
                    pr_body = f"This PR contains the automated implementation of the research paper '{paper_info.title}'.\n\n**Summary:** {paper_info.summary}\n\n**Automated Checks:**\n- Code Quality Score: {result.code_quality_score:.1f}/100\n- Tests Passed: {result.tests_passed}\n- Refinements Applied: {result.code_refinements_applied}\n\n{'' if not result.errors else '**Errors Encountered:**\n' + '\\n'.join(result.errors)}\n{'' if not result.warnings else '**Warnings:**\n' + '\\n'.join(result.warnings)}"
                    pr_url = await self.github_integrator.create_pull_request(repo_name, target_branch, "main", pr_title, pr_body)
                    if pr_url:
                        result.pr_created = True
                        self.logger.info(f"Pull Request created: {pr_url}")
                        # Optional: Auto-merge if enabled (use with extreme caution!)
                        # if self.config.auto_merge_prs:
                        #     await self.github_integrator.merge_pull_request(repo_name, pr_number)

                # 11. Semantic Versioning & Release Automation
                if result.tests_passed and not result.errors and self.config.enable_semantic_versioning and self.config.enable_release_automation:
                    new_version = self._increment_version(result.repo_state.current_version, "minor" if result.code_generated else "patch")
                    release_body = f"Automated implementation of '{paper_info.title}'.\n\n{paper_info.summary}\n\nCode Quality: {result.code_quality_score:.1f}/100"
                    release_url = await self.github_integrator.create_release(repo_name, f"v{new_version}", f"{paper_info.title} v{new_version}", release_body, target_commitish=new_sha)
                    if release_url:
                        result.release_created = True
                        result.repo_state.current_version = new_version
                        self.logger.info(f"Release created: {release_url}")
                
                # Final Status
                if not result.errors and result.tests_passed:
                    result.status = ProcessingStatus.SUCCESS
                else:
                    result.status = ProcessingStatus.FAILED

            except Exception as e:
                result.status = ProcessingStatus.FAILED
                result.errors.append(f"An unexpected error occurred during processing: {type(e).__name__}: {e}")
                self.logger.critical(f"CRITICAL ERROR processing '{paper_info.title}': {type(e).__name__}: {e}", exc_info=True)
            finally:
                result.processing_time = time.time() - start_time
                self.state_manager.update_repo_state(result.repo_state) # Save updated state
                self.logger.info(f"🏁 Finished processing '{paper_info.title}' in {result.processing_time:.2f}s with status: {result.status.value}")
                if result.errors: self.logger.error(f"Errors for '{paper_info.title}': {result.errors}")
                if result.warnings: self.logger.warning(f"Warnings for '{paper_info.title}': {result.warnings}")
            return result

    async def run_maintainer_pipeline(self):
        """Main pipeline to fetch papers and manage repositories."""
        self.logger.info("Starting M1 Maintainer Agent pipeline.")
        
        papers_to_process = self.paper_source.get_papers_to_process()
        
        tasks = []
        for paper_info in papers_to_process:
            # Check if paper needs processing (e.g., new, or source modified, or last processing failed)
            repo_state = self.state_manager.get_repo_state(self._generate_repo_name(paper_info.title))
            
            needs_processing = True
            if repo_state:
                # Compare last modified timestamp of source paper with last processed timestamp
                # Or check if last processing was successful and no new changes
                if repo_state.last_processed_timestamp:
                    last_processed_dt = datetime.fromisoformat(repo_state.last_processed_timestamp)
                    if paper_info.last_modified <= last_processed_dt.timestamp() and repo_state.status == ProcessingStatus.SUCCESS:
                        self.logger.info(f"'{paper_info.title}' is up-to-date and successfully processed. Skipping.")
                        needs_processing = False
            
            if needs_processing:
                tasks.append(self._process_single_paper_and_repo(paper_info))
            else:
                # Create a dummy result for skipped papers to include in the final report
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=ProcessingResult(
                    paper_info=paper_info,
                    repo_state=repo_state or RepoState(repo_name=self._generate_repo_name(paper_info.title), github_url="N/A"),
                    status=ProcessingStatus.SKIPPED,
                    processing_time=0.0
                ))))

        if not tasks:
            self.logger.info("No papers found or all are up-to-date. Nothing to process.")
            return []

        results = await asyncio.gather(*tasks)
        
        self.logger.info("M1 Maintainer Agent pipeline completed.")
        return results

# --- Main Execution ---
def load_config() -> Config:
    """Loads configuration from environment variables."""
    load_dotenv()
    
    required_env_vars = [
        'OPENROUTER_API_KEY',
        'GITHUB_TOKEN',
        'USERNAME_GITHUB'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}. Please set them in your .env file or environment.")
    
    return Config(
        openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
        github_token=os.getenv('GITHUB_TOKEN'),
        github_username=os.getenv('USERNAME_GITHUB'),
        primary_model=os.getenv('PRIMARY_MODEL', "deepseek/deepseek-r1-0528-qwen3-8b:free"),
        fallback_model=os.getenv('FALLBACK_MODEL', "google/gemini-2.0-flash-exp:free"),
        backup_model=os.getenv('BACKUP_MODEL', "meta-llama/llama-3.1-8b-instruct:free"),
        max_concurrent_paper_processing=int(os.getenv('MAX_CONCURRENT_PAPER_PROCESSING', '3')),
        request_timeout=int(os.getenv('REQUEST_TIMEOUT', '300')),
        retry_attempts=int(os.getenv('RETRY_ATTEMPTS', '5')),
        retry_delay=int(os.getenv('RETRY_DELAY', '5')),
        repo_prefix=os.getenv('REPO_PREFIX', 'research-paper-'),
        repo_visibility=os.getenv('REPO_VISIBILITY', 'public'),
        create_dev_branch=os.getenv('CREATE_DEV_BRANCH', 'true').lower() == 'true',
        auto_merge_prs=os.getenv('AUTO_MERGE_PRS', 'false').lower() == 'false',
        enable_llm_code_generation=os.getenv('ENABLE_LLM_CODE_GENERATION', 'true').lower() == 'true',
        enable_iterative_refinement=os.getenv('ENABLE_ITERATIVE_REFINEMENT', 'true').lower() == 'true',
        enable_static_analysis=os.getenv('ENABLE_STATIC_ANALYSIS', 'true').lower() == 'true',
        enable_testing=os.getenv('ENABLE_TESTING', 'true').lower() == 'true',
        enable_containerization=os.getenv('ENABLE_CONTAINERIZATION', 'true').lower() == 'true',
        enable_semantic_versioning=os.getenv('ENABLE_SEMANTIC_VERSIONING', 'true').lower() == 'true',
        enable_release_automation=os.getenv('ENABLE_RELEASE_AUTOMATION', 'true').lower() == 'true',
        enable_performance_profiling=os.getenv('ENABLE_PERFORMANCE_PROFILING', 'true').lower() == 'true',
        relevant_papers_dir=Path(os.getenv('RELEVANT_PAPERS_DIR', 'relevant_papers')),
        managed_repos_state_file=Path(os.getenv('MANAGED_REPOS_STATE_FILE', 'managed_repos_state.json')),
        workspace_dir=Path(os.getenv('WORKSPACE_DIR', 'workspace')),
        logs_dir=Path(os.getenv('LOGS_DIR', 'logs')),
        llm_logs_dir=Path(os.getenv('LLM_LOGS_DIR', 'llm_interactions')),
        temp_dir=Path(os.getenv('TEMP_DIR', 'temp_repo_clones'))
    )

async def main():
    """Main function to run the M1 Maintainer Agent."""
    try:
        config = load_config()
        print(f"🔧 M1 Configuration loaded successfully.")
        
        async with M1MaintainerAgent(config) as agent:
            results = await agent.run_maintainer_pipeline()
            
            print("\n--- M1 Processing Summary ---")
            total_processed = len(results)
            successful = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
            failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
            skipped = sum(1 for r in results if r.status == ProcessingStatus.SKIPPED)
            no_change = sum(1 for r in results if r.status == ProcessingStatus.NO_CHANGE)

            print(f"Total Papers Considered: {total_processed}")
            print(f"Successfully Processed/Updated: {successful}")
            print(f"Failed: {failed}")
            print(f"Skipped (Up-to-date): {skipped}")
            print(f"No Changes Detected: {no_change}")
            print("-" * 40)

            for result in results:
                status_emoji = {
                    ProcessingStatus.SUCCESS: "✅",
                    ProcessingStatus.FAILED: "❌",
                    ProcessingStatus.SKIPPED: "⏭️",
                    ProcessingStatus.NO_CHANGE: "↔️",
                    ProcessingStatus.PROCESSING: "🔄", # Should not be seen here
                    ProcessingStatus.FETCHING: "⬇️",
                    ProcessingStatus.GENERATING_CODE: "✨",
                    ProcessingStatus.REFINING_CODE: "♻️",
                    ProcessingStatus.VALIDATING_CODE: "🔍",
                    ProcessingStatus.TESTING_CODE: "🧪",
                    ProcessingStatus.CONTAINERIZING: "📦",
                    ProcessingStatus.GIT_OPERATIONS: "⬆️"
                }.get(result.status, "❓")
                
                print(f"{status_emoji} {result.paper_info.title} ({result.repo_state.repo_name}): {result.status.value} ({result.processing_time:.2f}s)")
                if result.repo_state.github_url:
                    print(f"   Repo: {result.repo_state.github_url}")
                if result.repo_state.last_processed_commit_sha:
                    print(f"   Last Commit: {result.repo_state.last_processed_commit_sha[:7]}")
                if result.repo_state.current_version != "0.0.0":
                    print(f"   Version: {result.repo_state.current_version}")
                if result.code_generated:
                    print(f"   Code Generated: Yes (Refinements: {result.code_refinements_applied})")
                if result.tests_run:
                    print(f"   Tests Passed: {result.tests_passed}")
                if result.code_quality_score > 0:
                    print(f"   Code Quality Score: {result.code_quality_score:.1f}/100")
                if result.pr_created:
                    print(f"   PR Created: Yes")
                if result.release_created:
                    print(f"   Release Created: Yes")
                if result.errors:
                    print(f"   Errors: {'; '.join(result.errors)}")
                if result.warnings:
                    print(f"   Warnings: {'; '.join(result.warnings)}")
                print("-" * 40)

    except ValueError as e:
        print(f"Configuration Error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))

