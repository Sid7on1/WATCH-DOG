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
    temperature: float = 0.5
    max_llm_tokens: int = 8000
    request_timeout: int = 400
    retry_attempts: int = 3
    retry_delay: int = 10
    repo_visibility: str = "public"
    max_concurrent_papers: int = 3
    base_dir: Path = Path(__file__).parent
    papers_dir: Path = base_dir / "papers_to_implement"
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
        
        return None, error_message

    async def plan_file_structure(self, paper: PaperInfo) -> Tuple[Optional[List[FilePlan]], str]:
        """Asks the architect LLM to plan the entire file structure for the project."""
        prompt = f"""
        You are an expert software architect. Your task is to design the ideal file structure for a complete, production-ready Python project that implements the concepts from a given research paper.

        **Paper Title:** {paper.title}
        **Paper Summary:** {paper.summary}

        Based on this, provide a comprehensive list of all necessary files. This should include:
        - Source code files (e.g., `src/main.py`, `src/model.py`, `src/utils.py`).
        - Dependency management (e.g., `pyproject.toml`).
        - Documentation (e.g., `README.md`, `LICENSE`).
        - Testing files (e.g., `tests/test_model.py`).
        - Containerization (e.g., `Dockerfile`).
        - Configuration files (e.g., `config.yaml`).

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
            }}
          ]
        }}
        """
        messages = [{"role": "user", "content": prompt}]
        self.logger.info(f"Requesting file plan for '{paper.title}' from architect model: {self.config.architect_model}")
        
        content, error = await self._call_llm(messages, self.config.architect_model, is_json=True)
        if error:
            return None, error

        try:
            data = json.loads(content)
            if not isinstance(data, dict) or "files" not in data or not isinstance(data["files"], list):
                return None, "Invalid response structure from architect LLM."
            plan = [FilePlan(**item) for item in data.get("files", [])]
            if not plan:
                self.logger.warning("Architect model returned an empty file plan. Using default minimal structure.")
                plan = [FilePlan(path="README.md", description="Basic project documentation.")]
            return plan, "File plan generated successfully."
        except (json.JSONDecodeError, TypeError) as e:
            self.logger.error(f"Failed to parse file plan from LLM response: {e}\nResponse: {content}")
            return None, f"Could not parse JSON response from architect: {e}"

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
        1.  Generate the full, raw content for the specified file (`{file_plan.path}`).
        2.  Do NOT wrap the content in markdown backticks (e.g., ```python ... ```) or any other formatting.
        3.  Ensure the code is clean, well-commented, and production-quality.
        4.  If it's a code file, use modern Python features and type hints.
        5.  If it's a documentation or configuration file, use the correct syntax (e.g., Markdown, TOML, YAML).
        6.  The content should be complete and ready to be saved directly to a file.

        Provide ONLY the raw file content.
        """
        messages = [{"role": "user", "content": prompt}]
        self.logger.info(f"Requesting content for '{file_plan.path}' from coder model: {self.config.coder_model}")
        
        for attempt in range(self.config.retry_attempts):
            content, error = await self._call_llm(messages, self.config.coder_model)
            if content:
                return content, "File content generated successfully."
            self.logger.warning(f"Retry {attempt+1}/{self.config.retry_attempts} for {file_plan.path}: {error}")
            await asyncio.sleep(self.config.retry_delay)
        return None, f"Failed to generate {file_plan.path} after {self.config.retry_attempts} attempts."

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
            async with self.session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get("login") != self.config.github_username:
                    raise ValueError("GitHub token does not match provided username.")
        except Exception as e:
            self.logger.critical(f"Invalid GitHub token: {e}")
            raise

    async def create_repository(self, repo_name: str, description: str) -> Tuple[Optional[str], Optional[str]]:
        """Creates a new GitHub repository and returns its HTML and clone URLs."""
        check_url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}"
        async with self.session.get(check_url, headers=self.headers) as response:
            if response.status == 200:
                self.logger.warning(f"Repository '{repo_name}' already exists.")
                data = await response.json()
                return data.get("html_url"), data.get("clone_url")

        create_url = "https://api.github.com/user/repos"
        payload = {
            "name": repo_name,
            "description": description,
            "private": self.config.repo_visibility != "public",
            "auto_init": False
        }
        async with self.session.post(create_url, headers=self.headers, json=payload) as response:
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
                error_msg = f"Command `{' '.join(command)}` failed with exit code {process.returncode}.\nStderr: {process.stderr.strip()}"
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

        env = os.environ.copy()
        env["GIT_ASKPASS"] = "echo"  # GitHub Actions handles token via environment
        env["GITHUB_TOKEN"] = self.config.github_token

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
        commands = [
            ["git", "add"] + file_paths,
            ["git", "commit", "-m", f"feat: Add generated files for paper implementation"],
            ["git", "push", "origin", "main"]
        ]
        
        env = os.environ.copy()
        env["GIT_ASKPASS"] = "echo"
        env["GITHUB_TOKEN"] = self.config.github_token

        for cmd in commands:
            success, msg = self._run_command(cmd, cwd=local_path, env=env)
            if not success:
                self.logger.error(f"Failed to push files at step `{' '.join(cmd)}`: {msg}")
                return False
        
        self.logger.info(f"Successfully committed and pushed {len(file_paths)} files.")
        return True

    def validate_repository(self, local_path: Path) -> bool:
        """Validates the repository by running basic linting and tests."""
        self.logger.info(f"Validating repository at {local_path}")
        commands = [
            ["flake8", "."] if (local_path / "flake8").exists() else None,
            ["pytest", "tests"] if (local_path / "tests").exists() else None
        ]
        for cmd in [c for c in commands if c]:
            success, msg = self._run_command(cmd, cwd=local_path)
            if not success:
                self.logger.error(f"Validation failed for {cmd[0]}: {msg}")
                return False
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
        self.llm: Optional[LLMInterface] = None
        self.github: Optional[GitHubIntegrator] = None
        self.repo_manager: Optional[RepoManager] = None
        self.state_manager: Optional[StateManager] = None

    def _setup_logging(self) -> logging.Logger:
        """Sets up logging with configurable verbosity."""
        self.config.logs_dir.mkdir(exist_ok=True)
        log_file = self.config.logs_dir / 'm1_evo_agent.log'
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        
        logger = logging.getLogger("M1EvoAgent")
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        if not logger.handlers:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(ch)
            
        return logger

    def _setup_directories(self):
        """Sets up required directories with permission checks."""
        for directory in [self.config.papers_dir, self.config.workspace_dir, self.config.logs_dir, self.config.llm_logs_dir]:
            try:
                directory.mkdir(exist_ok=True)
                if not os.access(directory, os.W_OK):
                    raise PermissionError(f"No write permissions for {directory}")
            except PermissionError as e:
                self.logger.critical(f"Permission error for directory {directory}: {e}")
                raise

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        self.llm = LLMInterface(self.config, self.session, self.logger)
        self.github = GitHubIntegrator(self.config, self.session, self.logger)
        self.repo_manager = RepoManager(self.config, self.logger)
        self.state_manager = StateManager(self.config, self.logger)
        await self.github.validate_token()
        self.state_manager.load_state()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.state_manager:
            self.state_manager.save_state()
        self.logger.info("M1-Evo Agent shutdown complete.")

    def get_papers_to_process(self) -> List[PaperInfo]:
        """Scans the papers directory for new or updated JSON files."""
        papers = []
        for file_path in self.config.papers_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    self.logger.error(f"Invalid JSON structure in {file_path}: Expected a dictionary.")
                    continue
                paper_id = data.get("id", hashlib.md5(file_path.read_bytes()).hexdigest())
                title = data.get("title")
                if not title:
                    self.logger.error(f"Missing 'title' in {file_path}.")
                    continue
                papers.append(PaperInfo(
                    id=paper_id,
                    title=title,
                    summary=data.get("summary", "No summary provided."),
                    source_path=file_path,
                    last_modified=file_path.stat().st_mtime
                ))
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in {file_path}: {e}")
            except Exception as e:
                self.logger.error(f"Could not process paper file {file_path}: {e}")
        return papers

    def _generate_repo_name(self, paper_title: str, paper_id: str) -> str:
        """Generates a sanitized repository name."""
        name = re.sub(r'[^a-zA-Z0-9\s-]', '', paper_title).lower()
        name = re.sub(r'\s+', '-', name).strip('-')
        max_len = 100 - len(self.config.repo_prefix) - len(paper_id) - 1
        name = name[:max_len] + f"-{paper_id[:8]}"
        return self.config.repo_prefix + name

    def sanitize_path(self, path: str, base_path: Path) -> Path:
        """Sanitizes file paths to prevent directory traversal."""
        normalized = (base_path / path).resolve()
        if not normalized.is_relative_to(base_path.resolve()):
            raise ValueError(f"Invalid file path: {path} attempts to escape repository root.")
        return normalized

    async def process_paper(self, paper: PaperInfo):
        """Main processing pipeline for a single paper."""
        repo_name = self._generate_repo_name(paper.title, paper.id)
        local_repo_path = self.config.workspace_dir / repo_name
        repo_state = self.state_manager.managed_repos.get(repo_name, RepoState(repo_name=repo_name, github_url=""))
        
        self.logger.info(f"--- Starting processing for paper: '{paper.title}' ---")
        
        try:
            if repo_state.status == ProcessingStatus.SUCCESS and local_repo_path.exists():
                self.logger.info("Repository already successfully processed. Skipping.")
                return

            if local_repo_path.exists():
                shutil.rmtree(local_repo_path)

            repo_state.status = ProcessingStatus.CREATING_REPO
            self.state_manager.update_repo_state(repo_state)
            html_url, clone_url = await self.github.create_repository(repo_name, f"AI-generated implementation of: {paper.title}")
            if not clone_url:
                raise Exception("Failed to create GitHub repository.")
            repo_state.github_url = html_url

            if not self.repo_manager.init_and_push_empty_repo(local_repo_path, clone_url):
                raise Exception("Failed to initialize local repository and push initial commit.")

            repo_state.status = ProcessingStatus.PLANNING
            self.state_manager.update_repo_state(repo_state)
            file_plan, msg = await self.llm.plan_file_structure(paper)
            if not file_plan:
                raise Exception(f"Failed to get file plan from LLM: {msg}")
            self.logger.info(f"Architect planned {len(file_plan)} files for the repository.")

            repo_state.status = ProcessingStatus.GENERATING_FILES
            self.state_manager.update_repo_state(repo_state)
            generated_files_context = "The following files have been created so far:\n"
            files_to_commit = []
            
            for i, file_to_gen in enumerate(file_plan):
                self.logger.info(f"[{i+1}/{len(file_plan)}] Generating file: {file_to_gen.path}")
                content, msg = await self.llm.generate_file_content(paper, file_to_gen, generated_files_context)
                if not content:
                    self.logger.error(f"Failed to generate content for {file_to_gen.path}: {msg}. Skipping file.")
                    continue

                full_path = self.sanitize_path(file_to_gen.path, local_repo_path)
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                files_to_commit.append(str(file_to_gen.path))
                repo_state.files_generated.append(file_to_gen.path)
                generated_files_context += f"- {file_to_gen.path}\n"
                await asyncio.sleep(1)

            if files_to_commit:
                if not self.repo_manager.commit_and_push_files(local_repo_path, files_to_commit):
                    raise Exception("Failed to commit and push generated files.")

            repo_state.status = ProcessingStatus.VALIDATING
            self.state_manager.update_repo_state(repo_state)
            if not self.repo_manager.validate_repository(local_repo_path):
                raise Exception("Repository validation failed.")

            self.logger.info(f"Successfully generated, pushed, and validated {len(repo_state.files_generated)} files.")
            repo_state.status = ProcessingStatus.SUCCESS

        except Exception as e:
            self.logger.critical(f"CRITICAL ERROR processing '{paper.title}': {e}", exc_info=True)
            repo_state.status = ProcessingStatus.FAILED
            repo_state.errors.append(str(e))
        finally:
            self.state_manager.update_repo_state(repo_state)
            self.logger.info(f"--- Finished processing for '{paper.title}' with status: {repo_state.status.value} ---")

    async def run(self):
        """Main entry point to run the agent's lifecycle."""
        self.logger.info("M1-Evo Agent is now running. Scraping for papers...")
        papers = self.get_papers_to_process()
        if not papers:
            self.logger.info("No papers found in the input directory. Shutting down.")
            return

        self.logger.info(f"Found {len(papers)} papers to process.")
        success_count = 0
        tasks = [self.process_paper(paper) for paper in papers]
        for i in range(0, len(tasks), self.config.max_concurrent_papers):
            await asyncio.gather(*tasks[i:i + self.config.max_concurrent_papers])
        
        for paper in papers:
            repo_name = self._generate_repo_name(paper.title, paper.id)
            if self.state_manager.managed_repos.get(repo_name, RepoState(repo_name=repo_name, github_url="")).status == ProcessingStatus.SUCCESS:
                success_count += 1
        self.logger.info(f"Completed processing: {success_count}/{len(papers)} papers successful.")

# --- Main Execution ---
async def main():
    try:
        load_dotenv()
        config = Config(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            github_token=os.getenv("GITHUB_TOKEN"),
            github_username=os.getenv("USERNAME_GITHUB")
        )
        if not all([config.openrouter_api_key, config.github_token, config.github_username]):
            raise ValueError("Missing one or more required environment variables: OPENROUTER_API_KEY, GITHUB_TOKEN, USERNAME_GITHUB")
            
        async with M1EvoAgent(config) as agent:
            await agent.run()
        return 0
    except Exception as e:
        logging.basicConfig()
        logging.getLogger().critical(f"A critical error occurred at the top level: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.exit(asyncio.run(main()))
