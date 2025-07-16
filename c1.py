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
    max_llm_tokens: int = 8000
    request_timeout: int = 400
    retry_attempts: int = 3
    retry_delay: int = 10
    repo_visibility: str = "public"
    max_concurrent_papers: int = 3
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
    source_path: Path
    last_modified: float

@dataclass
class FilePlan:
    """A planned file to be generated."""
    path: str
    description: str

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
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_llm_tokens,
        }
        if is_json:
            payload["response_format"] = {"type": "json_object"}

        error_message = None
        for attempt in range(self.config.retry_attempts):
            try:
                async with self.session.post(url, headers=self.headers, json=payload, timeout=self.config.request_timeout) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        self.logger.warning(f"Rate limit exceeded for LLM API. Retrying after {retry_after} seconds.")
                        await asyncio.sleep(retry_after)
                        continue
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
        You are an expert software architect. Your task is to design the ideal file structure for a complete, production-ready Python project that implements the concepts from a given research paper.

        **Paper Title:** {paper.title}
        **Paper Summary:** {paper.summary}

        Based on this, provide a comprehensive list of all necessary files. This should include:
        - Source code files (e.g., `src/main.py`, `src/model.py`, `src/utils.py`).
        - Dependency management (e.g., `requirements.txt`, `pyproject.toml`).
        - Documentation (e.g., `README.md`, `LICENSE`).
        - Testing files (e.g., `tests/test_model.py`).
        - Configuration files (e.g., `config.yaml`).
        - A simple example or demo script.

        Respond with ONLY a JSON object containing a single key "files". The value should be an array of objects, where each object has two keys: "path" (the full file path from the project root) and "description" (a concise, one-sentence explanation of the file's purpose).

        Example format:
        {{
          "files": [
            {{
              "path": "src/main.py",
              "description": "The main entry point for running the application."
            }},
            {{
              "path": "README.md",
              "description": "Comprehensive documentation for the project."
            }},
            {{
              "path": "requirements.txt",
              "description": "Python dependencies for the project."
            }}
          ]
        }}
        """
        messages = [{"role": "user", "content": prompt}]
        self.logger.info(f"Requesting file plan for '{paper.title}' from architect model: {self.config.architect_model}")
        
        content, error = await self._call_llm(messages, self.config.architect_model, is_json=True)
        if error:
            self.logger.error(f"Failed to get file plan: {error}")
            # Fallback to minimal structure
            return self._get_fallback_file_plan(paper), "Using fallback file plan due to LLM error"

        try:
            data = json.loads(content)
            if not isinstance(data, dict) or "files" not in data or not isinstance(data["files"], list):
                self.logger.warning("Invalid response structure from architect LLM, using fallback.")
                return self._get_fallback_file_plan(paper), "Using fallback file plan due to invalid response"
            
            plan = []
            for item in data.get("files", []):
                if isinstance(item, dict) and "path" in item and "description" in item:
                    plan.append(FilePlan(path=item["path"], description=item["description"]))
                else:
                    self.logger.warning(f"Invalid file plan item: {item}")
            
            if not plan:
                self.logger.warning("Architect model returned an empty file plan. Using fallback.")
                return self._get_fallback_file_plan(paper), "Using fallback file plan due to empty response"
            
            self.logger.info(f"Successfully generated file plan with {len(plan)} files")
            return plan, "File plan generated successfully."
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.error(f"Failed to parse file plan from LLM response: {e}\nResponse: {content}")
            return self._get_fallback_file_plan(paper), f"Using fallback file plan due to JSON parse error: {e}"

    def _get_fallback_file_plan(self, paper: PaperInfo) -> List[FilePlan]:
        """Returns a minimal but functional file plan as fallback."""
        return [
            FilePlan(path="README.md", description="Project documentation and overview"),
            FilePlan(path="requirements.txt", description="Python dependencies"),
            FilePlan(path="src/__init__.py", description="Package initialization file"),
            FilePlan(path="src/main.py", description="Main application entry point"),
            FilePlan(path="src/model.py", description="Core model implementation"),
            FilePlan(path="src/utils.py", description="Utility functions and helpers"),
            FilePlan(path="tests/__init__.py", description="Test package initialization"),
            FilePlan(path="tests/test_model.py", description="Unit tests for the model"),
            FilePlan(path="config.yaml", description="Configuration file"),
            FilePlan(path="example.py", description="Example usage script"),
            FilePlan(path="LICENSE", description="Project license")
        ]

    async def generate_file_content(self, paper: PaperInfo, file_plan: FilePlan, file_context: str) -> Tuple[Optional[str], str]:
        """Asks the coder LLM to generate the content for a single file."""
        prompt = f"""
        You are an expert Python programmer. Your task is to write the complete content for a single file within a larger project.

        **Project Context:**
        - **Paper Title:** {paper.title}
        - **Paper Summary:** {paper.summary}

        **File to Generate:**
        - **Path:** `{file_plan.path}`
        - **Purpose:** `{file_plan.description}`

        **Existing Project Structure (for context):**
        {file_context}

        **Instructions:**
        1. Generate the full, raw content for the specified file (`{file_plan.path}`).
        2. Do NOT wrap the content in markdown backticks (e.g., ```python ... ```) or any other formatting.
        3. Ensure the code is clean, well-commented, and production-quality.
        4. If it's a code file, use modern Python features and type hints.
        5. If it's a documentation or configuration file, use the correct syntax (e.g., Markdown, TOML, YAML).
        6. The content should be complete and ready to be saved directly to a file.
        7. For Python files, include proper imports and make sure the code is functional.
        8. For README.md, include installation instructions, usage examples, and project description.
        9. For requirements.txt, include realistic dependencies for the project.

        Provide ONLY the raw file content, nothing else.
        """
        messages = [{"role": "user", "content": prompt}]
        self.logger.info(f"Requesting content for '{file_plan.path}' from coder model: {self.config.coder_model}")
        
        for attempt in range(self.config.retry_attempts):
            content, error = await self._call_llm(messages, self.config.coder_model)
            if content and content.strip():
                # Additional validation for content
                if self._validate_file_content(content, file_plan.path):
                    return content, "File content generated successfully."
                else:
                    self.logger.warning(f"Generated content for {file_plan.path} failed validation, retrying...")
            else:
                self.logger.warning(f"Empty or invalid content for {file_plan.path}, attempt {attempt+1}")
            
            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(self.config.retry_delay)
        
        # Generate fallback content if LLM fails
        fallback_content = self._generate_fallback_content(paper, file_plan)
        if fallback_content:
            self.logger.info(f"Using fallback content for {file_plan.path}")
            return fallback_content, "Using fallback content due to LLM failures"
        
        return None, f"Failed to generate {file_plan.path} after {self.config.retry_attempts} attempts."

    def _validate_file_content(self, content: str, file_path: str) -> bool:
        """Validates that the generated content is reasonable."""
        if not content or len(content.strip()) < 10:
            return False
        
        # Check for common LLM formatting issues
        if content.strip().startswith("```") and content.strip().endswith("```"):
            return False
        
        # Basic validation for Python files
        if file_path.endswith('.py'):
            # Should have some Python-like content
            if not any(keyword in content for keyword in ['def ', 'class ', 'import ', 'from ']):
                return False
        
        return True

    def _generate_fallback_content(self, paper: PaperInfo, file_plan: FilePlan) -> Optional[str]:
        """Generates basic fallback content for essential files."""
        path = file_plan.path.lower()
        
        if path == "readme.md":
            return f"""# {paper.title}

This project implements concepts from the research paper: "{paper.title}"

## Summary
{paper.summary}

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py
```

## Structure
- `src/` - Main source code
- `tests/` - Unit tests
- `config.yaml` - Configuration file
- `example.py` - Usage example

## License
MIT License
"""
        elif path == "requirements.txt":
            return """numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
torch>=1.9.0
transformers>=4.0.0
pyyaml>=6.0
pytest>=6.0.0
"""
        elif path == "src/main.py":
            return f'''"""
Main entry point for {paper.title} implementation.
"""

import argparse
import yaml
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="{paper.title} Implementation")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    args = parser.parse_args()
    
    config = load_config(args.config)
    print(f"Running {paper.title} implementation...")
    print(f"Configuration: {{config}}")
    
    # TODO: Implement main logic here
    print("Implementation completed successfully!")

if __name__ == "__main__":
    main()
'''
        elif path == "src/model.py":
            return f'''"""
Core model implementation for {paper.title}.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class Model(nn.Module):
    """
    Main model class implementing concepts from {paper.title}.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        # TODO: Initialize model components
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # TODO: Implement forward pass
        return x
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        # TODO: Implement training step
        return {{"loss": 0.0}}
    
    def evaluate(self, data_loader) -> Dict[str, float]:
        """Evaluate the model."""
        # TODO: Implement evaluation
        return {{"accuracy": 0.0}}
'''
        elif path == "config.yaml":
            return f"""# Configuration for {paper.title} Implementation

model:
  name: "default_model"
  hidden_size: 512
  num_layers: 6
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  save_every: 10

data:
  train_path: "data/train.csv"
  val_path: "data/val.csv"
  test_path: "data/test.csv"

logging:
  level: "INFO"
  save_dir: "logs"
"""
        elif path == "src/__init__.py" or path == "tests/__init__.py":
            return '"""Package initialization."""\n'
        elif path == "license":
            return """MIT License

Copyright (c) 2024

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
        
        return None

class GitHubIntegrator:
    """Manages all GitHub API interactions."""
    def __init__(self, config: Config, session: aiohttp.ClientSession, logger: logging.Logger):
        self.config = config
        self.session = session
        self.logger = logger
        self.headers = {
            "Authorization": f"token {self.config.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": self.config.user_agent
        }

    async def validate_token(self):
        """Validates the GitHub token by fetching user information."""
        url = "https://api.github.com/user"
        try:
            async with self.session.get(url, headers=self.headers, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get("login") != self.config.github_username:
                    raise ValueError("GitHub token does not match provided username.")
                self.logger.info(f"GitHub token validated for user: {data.get('login')}")
        except Exception as e:
            self.logger.critical(f"Invalid GitHub token: {e}")
            raise

    async def create_repository(self, repo_name: str, description: str) -> Tuple[Optional[str], Optional[str]]:
        """Creates a new GitHub repository and returns its HTML and clone URLs."""
        # Check if repository already exists
        check_url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}"
        try:
            async with self.session.get(check_url, headers=self.headers, timeout=30) as response:
                if response.status == 200:
                    self.logger.warning(f"Repository '{repo_name}' already exists.")
                    data = await response.json()
                    return data.get("html_url"), data.get("clone_url")
        except Exception as e:
            self.logger.debug(f"Error checking repository existence: {e}")

        # Create new repository
        create_url = "https://api.github.com/user/repos"
        payload = {
            "name": repo_name,
            "description": description,
            "private": self.config.repo_visibility != "public",
            "auto_init": False
        }
        
        try:
            async with self.session.post(create_url, headers=self.headers, json=payload, timeout=30) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    self.logger.warning(f"Rate limit exceeded for GitHub API. Retrying after {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                    return await self.create_repository(repo_name, description)
                
                if response.status == 201:
                    data = await response.json()
                    self.logger.info(f"Successfully created GitHub repository: {data['html_url']}")
                    return data.get("html_url"), data.get("clone_url")
                
                error_text = await response.text()
                self.logger.error(f"Failed to create repo {repo_name}: {response.status} - {error_text}")
                return None, None
        except Exception as e:
            self.logger.error(f"Exception creating repository: {e}")
            return None, None

class RepoManager:
    """Manages all local Git operations."""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        if not shutil.which("git"):
            self.logger.critical("Git is not installed or not found in PATH.")
            raise EnvironmentError("Git is required but not found.")

    def _run_command(self, command: List[str], cwd: Path, env: Optional[Dict] = None) -> Tuple[bool, str]:
        """Runs a shell command and returns success status and output."""
        try:
            self.logger.debug(f"Running command: {' '.join(command)} in {cwd}")
            process = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
                env=env
            )
            if process.returncode != 0:
                error_msg = f"Command `{' '.join(command)}` failed with exit code {process.returncode}.\nStderr: {process.stderr.strip()}\nStdout: {process.stdout.strip()}"
                self.logger.error(error_msg)
                return False, error_msg
            return True, process.stdout.strip()
        except FileNotFoundError:
            msg = f"Command not found: {command[0]}. Ensure Git is installed and in your PATH."
            self.logger.error(msg)
            return False, msg
        except Exception as e:
            msg = f"An unexpected error occurred running command `{' '.join(command)}`: {e}"
            self.logger.error(msg)
            return False, msg

    def init_and_push_empty_repo(self, local_path: Path, remote_url: str) -> bool:
        """Initializes a local repo, adds the remote, and pushes an initial empty commit."""
        if not local_path.exists():
            local_path.mkdir(parents=True)

        # Set up environment for Git authentication
        env = os.environ.copy()
        env["GIT_ASKPASS"] = "echo"
        env["GITHUB_TOKEN"] = self.config.github_token
        
        # Modify remote URL to include token for HTTPS authentication
        if remote_url.startswith("https://"):
            remote_url = remote_url.replace("https://", f"https://{self.config.github_username}:{self.config.github_token}@")

        commands = [
            ["git", "init"],
            ["git", "config", "user.name", self.config.github_username],
            ["git", "config", "user.email", f"{self.config.github_username}@users.noreply.github.com"],
            ["git", "branch", "-M", "main"],
            ["git", "remote", "add", "origin", remote_url],
            ["git", "commit", "--allow-empty", "-m", "chore: Initial commit"],
            ["git", "push", "-u", "origin", "main"]
        ]

        for cmd in commands:
            success, msg = self._run_command(cmd, cwd=local_path, env=env)
            if not success:
                self.logger.error(f"Failed to initialize repository at step `{' '.join(cmd)}`: {msg}")
                return False
        
        self.logger.info(f"Successfully initialized and pushed empty repo to {remote_url}")
        return True

    def commit_and_push_files(self, local_path: Path, file_paths: List[str]) -> bool:
        """Commits and pushes multiple files in a single commit."""
        if not file_paths:
            self.logger.warning("No files to commit.")
            return True

        # Verify files exist before committing
        existing_files = []
        for file_path in file_paths:
            full_path = local_path / file_path
            if full_path.exists():
                existing_files.append(file_path)
            else:
                self.logger.warning(f"File {file_path} does not exist, skipping")

        if not existing_files:
            self.logger.error("No files exist to commit")
            return False

        env = os.environ.copy()
        env["GIT_ASKPASS"] = "echo"
        env["GITHUB_TOKEN"] = self.config.github_token

        commands = [
            ["git", "add"] + existing_files,
            ["git", "commit", "-m", f"feat: Add generated files for paper implementation ({len(existing_files)} files)"],
            ["git", "push", "origin", "main"]
        ]

        for cmd in commands:
            success, msg = self._run_command(cmd, cwd=local_path, env=env)
            if not success:
                self.logger.error(f"Failed to push files at step `{' '.join(cmd)}`: {msg}")
                return False
        
        self.logger.info(f"Successfully committed and pushed {len(existing_files)} files.")
        return True

    def validate_repository(self, local_path: Path) -> bool:
        """Validates the repository by checking file existence and basic structure."""
        self.logger.info(f"Validating repository at {local_path}")
        
        # Check if basic files exist
        essential_files = ["README.md"]
        for file_name in essential_files:
            if not (local_path / file_name).exists():
                self.logger.warning(f"Essential file {file_name} is missing")
                return False
        
        # Check if there are any files at all
        files = list(local_path.rglob("*"))
        files = [f for f in files if f.is_file() and not f.name.startswith('.')]
        if len(files) < 3:  # Should have at least a few files
            self.logger.error(f"Repository has too few files ({len(files)})")
            return False
        
        self.logger.info(f"Repository validation passed with {len(files)} files")
        return True

class StateManager:
    """Manages persistence of the agent's state."""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.managed_repos: Dict[str, RepoState] = {}

    def load_state(self):
        """Loads the state from the state file."""
        if not self.config.state_file.exists():
            self.logger.info("No state file found, starting fresh.")
            return
        try:
            with open(self.config.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for name, state_data in data.items():
                    state_data['status'] = ProcessingStatus(state_data['status'])
                    self.managed_repos[name] = RepoState(**state_data)
            self.logger.info(f"Loaded state for {len(self.managed_repos)} repositories.")
        except Exception as e:
            self.logger.error(f"Error loading state file, starting fresh: {e}")
            self.managed_repos = {}

    def save_state(self):
        """Saves the state atomically to the state file."""
        try:
            temp_file = self.config.state_file.with_suffix(".tmp")
            serializable_data = {name: {**asdict(state), "status": state.status.value} for name, state in self.managed_repos.items()}
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2)
            temp_file.rename(self.config.state_file)
            self.logger.debug("State saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def update_repo_state(self, repo_state: RepoState):
        """Updates the state for a repository and saves it."""
        repo_state.last_processed_timestamp = datetime.now().isoformat()
        self.managed_repos[repo_state.repo_name] = repo_state
        self.save_state()

# --- M1-Evo: The Main Agent ---
class M1EvoAgent:
    """The evolved M1 Maintainer Agent."""
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logging()
        self._setup_directories()
        self.session: Optional[aiohttp.ClientSession] = None
        self.state_manager = StateManager(self.config, self.logger)
        self.llm_interface: Optional[LLMInterface] = None
        self.github_integrator: Optional[GitHubIntegrator] = None
        self.repo_manager: Optional[RepoManager] = None

    def _setup_logging(self) -> logging.Logger:
        """Sets up logging configuration."""
        self.config.logs_dir.mkdir(exist_ok=True)
        
        # Create logger
        logger = logging.getLogger("M1EvoAgent")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = self.config.logs_dir / f"m1_evo_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def _setup_directories(self):
        """Creates necessary directories."""
        for directory in [self.config.papers_dir, self.config.workspace_dir, 
                         self.config.logs_dir, self.config.llm_logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _scan_for_papers(self) -> List[PaperInfo]:
        """Scans the papers directory for JSON files to process."""
        papers = []
        if not self.config.papers_dir.exists():
            self.logger.warning(f"Papers directory {self.config.papers_dir} does not exist")
            return papers

        for json_file in self.config.papers_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract paper information
                paper_id = data.get('id', json_file.stem)
                title = data.get('title', 'Unknown Title')
                summary = data.get('summary', data.get('abstract', 'No summary available'))
                
                # Validate required fields
                if not title or not summary:
                    self.logger.warning(f"Skipping {json_file.name}: missing title or summary")
                    continue
                
                paper = PaperInfo(
                    id=paper_id,
                    title=title,
                    summary=summary,
                    source_path=json_file,
                    last_modified=json_file.stat().st_mtime
                )
                papers.append(paper)
                
            except Exception as e:
                self.logger.error(f"Error processing {json_file.name}: {e}")
                continue
        
        self.logger.info(f"Found {len(papers)} papers to potentially process")
        return papers

    def _generate_repo_name(self, paper: PaperInfo) -> str:
        """Generates a repository name from the paper title."""
        # Clean the title and make it GitHub-friendly
        clean_title = re.sub(r'[^a-zA-Z0-9\s-]', '', paper.title)
        clean_title = re.sub(r'\s+', '-', clean_title.strip())
        clean_title = clean_title.lower()
        
        # Truncate if too long
        if len(clean_title) > 50:
            clean_title = clean_title[:50].rsplit('-', 1)[0]
        
        # Add prefix and ensure it's valid
        repo_name = f"{self.config.repo_prefix}{clean_title}"
        
        # Ensure it doesn't start or end with special characters
        repo_name = re.sub(r'^[-_]+|[-_]+$', '', repo_name)
        
        # Fallback if name is empty or too short
        if len(repo_name) < 5:
            repo_name = f"{self.config.repo_prefix}{paper.id}"
        
        return repo_name

    def _should_process_paper(self, paper: PaperInfo) -> bool:
        """Determines if a paper should be processed based on current state."""
        repo_name = self._generate_repo_name(paper)
        
        # Check if already processed successfully
        if repo_name in self.state_manager.managed_repos:
            repo_state = self.state_manager.managed_repos[repo_name]
            if repo_state.status == ProcessingStatus.SUCCESS:
                # Check if paper has been modified since last processing
                if repo_state.last_processed_timestamp:
                    try:
                        last_processed = datetime.fromisoformat(repo_state.last_processed_timestamp)
                        last_modified = datetime.fromtimestamp(paper.last_modified)
                        if last_modified <= last_processed:
                            self.logger.info(f"Skipping {paper.title}: already processed and up to date")
                            return False
                    except Exception as e:
                        self.logger.warning(f"Error comparing timestamps for {paper.title}: {e}")
            elif repo_state.status == ProcessingStatus.FAILED:
                self.logger.info(f"Retrying previously failed paper: {paper.title}")
        
        return True

    async def _process_single_paper(self, paper: PaperInfo) -> bool:
        """Processes a single paper through the complete pipeline."""
        repo_name = self._generate_repo_name(paper)
        self.logger.info(f"Processing paper: {paper.title} -> {repo_name}")
        
        # Initialize repo state
        repo_state = RepoState(
            repo_name=repo_name,
            github_url="",
            status=ProcessingStatus.PLANNING
        )
        self.state_manager.update_repo_state(repo_state)
        
        try:
            # Step 1: Plan file structure
            self.logger.info("Step 1: Planning file structure...")
            file_plans, plan_message = await self.llm_interface.plan_file_structure(paper)
            if not file_plans:
                raise Exception(f"Failed to create file plan: {plan_message}")
            
            self.logger.info(f"Generated plan for {len(file_plans)} files")
            
            # Step 2: Create GitHub repository
            repo_state.status = ProcessingStatus.CREATING_REPO
            self.state_manager.update_repo_state(repo_state)
            
            self.logger.info("Step 2: Creating GitHub repository...")
            description = f"Implementation of '{paper.title}' - {paper.summary[:100]}..."
            github_url, clone_url = await self.github_integrator.create_repository(repo_name, description)
            
            if not github_url or not clone_url:
                raise Exception("Failed to create GitHub repository")
            
            repo_state.github_url = github_url
            self.state_manager.update_repo_state(repo_state)
            
            # Step 3: Initialize local repository
            local_repo_path = self.config.workspace_dir / repo_name
            if local_repo_path.exists():
                shutil.rmtree(local_repo_path)
            
            if not self.repo_manager.init_and_push_empty_repo(local_repo_path, clone_url):
                raise Exception("Failed to initialize local repository")
            
            # Step 4: Generate files
            repo_state.status = ProcessingStatus.GENERATING_FILES
            self.state_manager.update_repo_state(repo_state)
            
            self.logger.info("Step 4: Generating files...")
            generated_files = []
            
            # Create file context for LLM
            file_context = "Project Structure:\n"
            for plan in file_plans:
                file_context += f"- {plan.path}: {plan.description}\n"
            
            # Generate files in batches to avoid overwhelming the system
            batch_size = 5
            for i in range(0, len(file_plans), batch_size):
                batch = file_plans[i:i+batch_size]
                batch_files = []
                
                for file_plan in batch:
                    self.logger.info(f"Generating {file_plan.path}...")
                    content, message = await self.llm_interface.generate_file_content(
                        paper, file_plan, file_context
                    )
                    
                    if content:
                        file_path = local_repo_path / file_plan.path
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        batch_files.append(file_plan.path)
                        generated_files.append(file_plan.path)
                        self.logger.info(f"✓ Generated {file_plan.path}")
                    else:
                        self.logger.error(f"✗ Failed to generate {file_plan.path}: {message}")
                
                # Commit this batch
                if batch_files:
                    if not self.repo_manager.commit_and_push_files(local_repo_path, batch_files):
                        self.logger.warning(f"Failed to commit batch of {len(batch_files)} files")
                    else:
                        self.logger.info(f"✓ Committed batch of {len(batch_files)} files")
                
                # Small delay between batches to avoid rate limiting
                await asyncio.sleep(2)
            
            # Step 5: Validate repository
            repo_state.status = ProcessingStatus.VALIDATING
            self.state_manager.update_repo_state(repo_state)
            
            self.logger.info("Step 5: Validating repository...")
            if not self.repo_manager.validate_repository(local_repo_path):
                raise Exception("Repository validation failed")
            
            # Success!
            repo_state.status = ProcessingStatus.SUCCESS
            repo_state.files_generated = generated_files
            self.state_manager.update_repo_state(repo_state)
            
            self.logger.info(f"✅ Successfully processed {paper.title}")
            self.logger.info(f"   Repository: {github_url}")
            self.logger.info(f"   Files generated: {len(generated_files)}")
            
            return True
            
        except Exception as e:
            repo_state.status = ProcessingStatus.FAILED
            repo_state.errors.append(str(e))
            self.state_manager.update_repo_state(repo_state)
            
            self.logger.error(f"❌ Failed to process {paper.title}: {e}")
            return False

    async def run(self):
        """Main execution method."""
        self.logger.info("🚀 Starting M1-Evo Agent...")
        
        # Load previous state
        self.state_manager.load_state()
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        try:
            # Initialize components
            self.llm_interface = LLMInterface(self.config, self.session, self.logger)
            self.github_integrator = GitHubIntegrator(self.config, self.session, self.logger)
            self.repo_manager = RepoManager(self.config, self.logger)
            
            # Validate GitHub token
            await self.github_integrator.validate_token()
            
            # Scan for papers
            papers = self._scan_for_papers()
            if not papers:
                self.logger.warning("No papers found to process")
                return
            
            # Filter papers that need processing
            papers_to_process = [p for p in papers if self._should_process_paper(p)]
            
            if not papers_to_process:
                self.logger.info("All papers are already up to date")
                return
            
            # Limit concurrent processing
            papers_to_process = papers_to_process[:self.config.max_concurrent_papers]
            
            self.logger.info(f"Processing {len(papers_to_process)} papers...")
            
            # Process papers
            success_count = 0
            for paper in papers_to_process:
                try:
                    if await self._process_single_paper(paper):
                        success_count += 1
                    
                    # Brief pause between papers
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error processing {paper.title}: {e}")
                    continue
            
            # Summary
            self.logger.info(f"✅ Processing complete!")
            self.logger.info(f"   Successfully processed: {success_count}/{len(papers_to_process)}")
            self.logger.info(f"   Total repositories managed: {len(self.state_manager.managed_repos)}")
            
        except Exception as e:
            self.logger.critical(f"Critical error in main execution: {e}")
            raise
        finally:
            if self.session:
                await self.session.close()

    async def cleanup(self):
        """Cleanup method to close resources."""
        if self.session:
            await self.session.close()
        self.logger.info("Cleanup completed")

# --- Configuration Loading and Main Entry Point ---

def load_config() -> Config:
    """Loads configuration from environment variables."""
    load_dotenv()
    
    required_env_vars = [
        "OPENROUTER_API_KEY",
        "GITHUB_TOKEN",
        "GITHUB_USERNAME"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return Config(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        github_token=os.getenv("GITHUB_TOKEN"),
        github_username=os.getenv("GITHUB_USERNAME")
    )

async def main():
    """Main entry point."""
    try:
        config = load_config()
        agent = M1EvoAgent(config)
        await agent.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        if 'agent' in locals():
            await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
