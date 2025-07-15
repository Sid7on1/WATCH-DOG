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

    # LLM Settings
    architect_model: str = "meta-llama/llama-3.1-405b-instruct"
    coder_model: str = "deepseek/deepseek-coder-v2"
    temperature: float = 0.5
    max_llm_tokens: int = 8000

    # Processing & Concurrency
    request_timeout: int = 400
    retry_attempts: int = 3
    retry_delay: int = 10

    # GitHub Repository Settings
    repo_prefix: str = "paper-impl-"
    repo_visibility: str = "public"

    # Paths
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
            "User-Agent": "M1-Evo-Agent/2.0"
        }
        self.config.llm_logs_dir.mkdir(exist_ok=True)

    async def _call_llm(self, messages: List[Dict[str, str]], model: str, is_json: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """Internal method to call the LLM API with retry logic."""
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
                    response.raise_for_status()
                    data = await response.json()
                    response_content = data['choices'][0]['message']['content']
                    
                    # Log interaction
                    log_file = self.config.llm_logs_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{model.replace('/', '_')}.json"
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump({"request": payload, "response": data}, f, indent=2)

                    return response_content, None
            except Exception as e:
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
        self.logger.info(f"Requesting file plan from architect model: {self.config.architect_model}")
        
        content, error = await self._call_llm(messages, self.config.architect_model, is_json=True)
        if error:
            return None, error

        try:
            data = json.loads(content)
            plan = [FilePlan(**item) for item in data.get("files", [])]
            if not plan:
                return None, "Architect model returned an empty or invalid file plan."
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
        
        content, error = await self._call_llm(messages, self.config.coder_model)
        if error:
            return None, error
        
        return content, "File content generated successfully."

class GitHubIntegrator:
    """Manages all GitHub API interactions."""
    def __init__(self, config: Config, session: aiohttp.ClientSession, logger: logging.Logger):
        self.config = config
        self.session = session
        self.logger = logger
        self.headers = {
            "Authorization": f"token {self.config.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

    async def create_repository(self, repo_name: str, description: str) -> Tuple[Optional[str], Optional[str]]:
        """Creates a new GitHub repository and returns its HTML and clone URLs."""
        # Check if repo exists first
        check_url = f"https://api.github.com/repos/{self.config.github_username}/{repo_name}"
        async with self.session.get(check_url, headers=self.headers) as response:
            if response.status == 200:
                self.logger.warning(f"Repository '{repo_name}' already exists.")
                data = await response.json()
                return data.get("html_url"), data.get("clone_url")

        # Create new repo if it doesn't exist
        create_url = "https://api.github.com/user/repos"
        payload = {
            "name": repo_name,
            "description": description,
            "private": self.config.repo_visibility != "public",
            "auto_init": False # We will initialize it locally
        }
        async with self.session.post(create_url, headers=self.headers, json=payload) as response:
            if response.status == 201:
                data = await response.json()
                self.logger.info(f"Successfully created GitHub repository: {data['html_url']}")
                return data.get("html_url"), data.get("clone_url")
            else:
                error_text = await response.text()
                self.logger.error(f"Failed to create repo {repo_name}: {response.status} - {error_text}")
                return None, None

class RepoManager:
    """Manages all local Git operations."""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def _run_command(self, command: List[str], cwd: Path, env: Optional[Dict] = None) -> Tuple[bool, str]:
        """Helper to run shell commands."""
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

        # Use a token-authenticated URL for all operations
        auth_url = remote_url.replace("https://", f"https://{self.config.github_username}:{self.config.github_token}@")
        
        commands = [
            ["git", "init"],
            ["git", "config", "user.name", self.config.github_username],
            ["git", "config", "user.email", f"{self.config.github_username}@users.noreply.github.com"],
            ["git", "branch", "-M", "main"],
            ["git", "remote", "add", "origin", auth_url],
            ["git", "commit", "--allow-empty", "-m", "chore: Initial commit"],
            ["git", "push", "-u", "origin", "main"]
        ]

        for cmd in commands:
            success, msg = self._run_command(cmd, cwd=local_path)
            if not success:
                self.logger.error(f"Failed to initialize repository at step `{' '.join(cmd)}`: {msg}")
                return False
        
        self.logger.info(f"Successfully initialized and pushed empty repo to {remote_url}")
        return True

    def add_commit_and_push_file(self, local_path: Path, file_path: str) -> bool:
        """Adds a single file, commits, and pushes to the main branch."""
        full_file_path = local_path / file_path
        if not full_file_path.exists():
            self.logger.error(f"Cannot commit non-existent file: {full_file_path}")
            return False
            
        commands = [
            ["git", "add", str(full_file_path)],
            ["git", "commit", "-m", f"feat: Create {file_path}"],
            ["git", "push", "origin", "main"]
        ]
        
        for cmd in commands:
            success, msg = self._run_command(cmd, cwd=local_path)
            if not success:
                self.logger.error(f"Failed to push file '{file_path}' at step `{' '.join(cmd)}`: {msg}")
                return False
        
        self.logger.info(f"Successfully committed and pushed '{file_path}'")
        return True

class StateManager:
    """Manages persistence of the agent's state."""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.managed_repos: Dict[str, RepoState] = {}

    def load_state(self):
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
        try:
            serializable_data = {}
            for name, state in self.managed_repos.items():
                state_dict = asdict(state)
                state_dict['status'] = state_dict['status'].value
                serializable_data[name] = state_dict
            
            with open(self.config.state_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def update_repo_state(self, repo_state: RepoState):
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
        self.config.logs_dir.mkdir(exist_ok=True)
        log_file = self.config.logs_dir / 'm1_evo_agent.log'
        
        logger = logging.getLogger("M1EvoAgent")
        logger.setLevel(logging.INFO)
        
        # Prevent adding handlers multiple times
        if not logger.handlers:
            # File handler
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(ch)
            
        return logger

    def _setup_directories(self):
        self.config.papers_dir.mkdir(exist_ok=True)
        self.config.workspace_dir.mkdir(exist_ok=True)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        self.llm = LLMInterface(self.config, self.session, self.logger)
        self.github = GitHubIntegrator(self.config, self.session, self.logger)
        self.repo_manager = RepoManager(self.config, self.logger)
        self.state_manager = StateManager(self.config, self.logger)
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
                paper_id = data.get("id", hashlib.md5(file_path.read_bytes()).hexdigest())
                papers.append(PaperInfo(
                    id=paper_id,
                    title=data.get("title", file_path.stem),
                    summary=data.get("summary", "No summary provided."),
                    source_path=file_path,
                    last_modified=file_path.stat().st_mtime
                ))
            except Exception as e:
                self.logger.error(f"Could not process paper file {file_path}: {e}")
        return papers

    def _generate_repo_name(self, paper_title: str) -> str:
        """Generates a sanitized repository name."""
        name = re.sub(r'[^a-zA-Z0-9\s-]', '', paper_title).lower()
        name = re.sub(r'\s+', '-', name).strip('-')
        return (self.config.repo_prefix + name)[:100]

    async def process_paper(self, paper: PaperInfo):
        """Main processing pipeline for a single paper."""
        repo_name = self._generate_repo_name(paper.title)
        local_repo_path = self.config.workspace_dir / repo_name
        
        repo_state = self.state_manager.managed_repos.get(repo_name, RepoState(repo_name=repo_name, github_url=""))
        
        self.logger.info(f"--- Starting processing for paper: '{paper.title}' ---")
        
        try:
            # Step 1: Create GitHub repository
            repo_state.status = ProcessingStatus.CREATING_REPO
            self.state_manager.update_repo_state(repo_state)
            html_url, clone_url = await self.github.create_repository(repo_name, f"AI-generated implementation of: {paper.title}")
            if not clone_url:
                raise Exception("Failed to create GitHub repository.")
            repo_state.github_url = html_url

            # Step 2: Initialize repo locally and push first commit
            if local_repo_path.exists():
                shutil.rmtree(local_repo_path) # Start fresh
            if not self.repo_manager.init_and_push_empty_repo(local_repo_path, clone_url):
                raise Exception("Failed to initialize local repository and push initial commit.")

            # Step 3: Plan the file structure
            repo_state.status = ProcessingStatus.PLANNING
            self.state_manager.update_repo_state(repo_state)
            file_plan, msg = await self.llm.plan_file_structure(paper)
            if not file_plan:
                raise Exception(f"Failed to get file plan from LLM: {msg}")
            self.logger.info(f"Architect planned {len(file_plan)} files for the repository.")

            # Step 4: Generate and push each file incrementally
            repo_state.status = ProcessingStatus.GENERATING_FILES
            self.state_manager.update_repo_state(repo_state)
            
            generated_files_context = "The following files have been created so far:\n"
            for i, file_to_gen in enumerate(file_plan):
                self.logger.info(f"[{i+1}/{len(file_plan)}] Generating file: {file_to_gen.path}")
                content, msg = await self.llm.generate_file_content(paper, file_to_gen, generated_files_context)
                if not content:
                    self.logger.error(f"Failed to generate content for {file_to_gen.path}: {msg}. Skipping file.")
                    continue

                # Save file locally
                full_path = local_repo_path / file_to_gen.path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                # Commit and push the single file
                if not self.repo_manager.add_commit_and_push_file(local_repo_path, file_to_gen.path):
                    self.logger.warning(f"Failed to push {file_to_gen.path}. The repository may be incomplete.")
                    # Decide whether to continue or fail hard
                    continue
                
                repo_state.files_generated.append(file_to_gen.path)
                self.state_manager.update_repo_state(repo_state)
                generated_files_context += f"- {file_to_gen.path}\n"
                await asyncio.sleep(1) # Small delay to avoid rate limiting

            self.logger.info(f"Successfully generated and pushed {len(repo_state.files_generated)} files.")
            repo_state.status = ProcessingStatus.SUCCESS

        except Exception as e:
            self.logger.critical(f"CRITICAL ERROR processing '{paper.title}': {e}", exc_info=True)
            repo_state.status = ProcessingStatus.FAILED
            repo_state.errors.append(str(e))
        finally:
            self.state_manager.update_repo_state(repo_state)
            self.logger.info(f"--- Finished processing for '{paper.title}' with status: {repo_state.status.value} ---")
            
    async def run(self):
        """The main entry point to run the agent's lifecycle."""
        self.logger.info("M1-Evo Agent is now running. Scraping for papers...")
        papers = self.get_papers_to_process()

        if not papers:
            self.logger.info("No papers found in the input directory. Shutting down.")
            return

        self.logger.info(f"Found {len(papers)} papers to process.")
        
        for paper in papers:
            repo_name = self._generate_repo_name(paper.title)
            state = self.state_manager.managed_repos.get(repo_name)
            
            # Simple check to avoid reprocessing successful repos. More complex logic can be added.
            if state and state.status == ProcessingStatus.SUCCESS:
                self.logger.info(f"Skipping '{paper.title}' as it has already been processed successfully.")
                continue

            await self.process_paper(paper)
            self.logger.info("="*50)

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

    except Exception as e:
        logging.basicConfig()
        logging.getLogger().critical(f"A critical error occurred at the top level: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    # To run this, you would need a `papers_to_implement` directory with JSON files,
    # and a .env file with your API keys.
    # Example paper.json:
    # {
    #   "id": "2305.12978",
    #   "title": "Sparks of Artificial General Intelligence: Early experiments with GPT-4",
    #   "summary": "A study on the surprising emergent abilities of large language models like GPT-4, suggesting early signs of AGI-like behavior in areas like reasoning, coding, and theory of mind."
    # }
    
    # Setup asyncio event loop policy for Windows if needed
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    sys.exit(asyncio.run(main()))
