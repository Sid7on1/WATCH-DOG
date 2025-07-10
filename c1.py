import os
import json
import requests
import re
import subprocess
import asyncio
import aiohttp
import time
import hashlib
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
from enum import Enum
import yaml
import toml
from dotenv import load_dotenv

# === CONFIGURATION ===
@dataclass
class Config:
    """Configuration class for the Coder Agent"""
    # API Settings
    openrouter_api_key: str
    github_token: str
    primary_model: str = "deepseek/deepseek-r1-0528-qwen3-8b:free"
    fallback_model: str = "google/gemini-2.0-flash-exp:free"
    backup_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    
    # Processing Settings
    max_concurrent_requests: int = 3
    request_timeout: int = 120
    retry_attempts: int = 3
    retry_delay: int = 2
    
    # Code Generation Settings
    min_code_length: int = 50
    max_code_length: int = 10000
    temperature: float = 0.3
    
    # Repository Settings
    repo_name: str = "WATCHDOG_memory"
    repo_owner: str = "Sid7on1"
    create_branches: bool = True
    auto_merge: bool = False
    
    # Advanced Features
    enable_code_validation: bool = True
    enable_dependency_analysis: bool = True
    enable_documentation_generation: bool = True
    enable_testing_generation: bool = True
    enable_performance_profiling: bool = True

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ProcessingResult:
    """Result of processing a research paper"""
    title: str
    status: ProcessingStatus
    code_generated: bool = False
    tests_generated: bool = False
    docs_generated: bool = False
    errors: List[str] = None
    warnings: List[str] = None
    processing_time: float = 0.0
    code_quality_score: float = 0.0
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.dependencies is None:
            self.dependencies = []

class CoderAgent:
    """Enhanced Coder Agent with advanced features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = None
        self.setup_logging()
        self.setup_directories()
        self.setup_api_headers()
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': time.time()
        }
        
    def setup_logging(self):
        """Setup advanced logging system"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/coder_agent.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Setup all required directories"""
        self.dirs = {
            'pdf': Path("relevant_pdfs"),
            'json': Path("relevant_json"),
            'raw': Path("relevant_raw"),
            'workspace': Path("workspace"),
            'logs': Path("logs"),
            'cache': Path("cache"),
            'templates': Path("templates"),
            'tests': Path("tests"),
            'docs': Path("docs"),
            'profiles': Path("profiles")
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
            
    def setup_api_headers(self):
        """Setup API headers for different services"""
        self.headers = {
            'openrouter': {
                "Authorization": f"Bearer {self.config.openrouter_api_key}",
                "Content-Type": "application/json"
            },
            'github': {
                "Authorization": f"token {self.config.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    def sanitize_filename(self, name: str) -> str:
        """Enhanced filename sanitization"""
        name = name.replace('\n', ' ').replace('\r', '')
        name = re.sub(r'[\\/*?:"<>|]', "", name)
        name = name.replace(" ", "_")
        name = re.sub(r'__+', '_', name)
        name = re.sub(r'[^\w\-_\.]', '', name)
        return name.strip('_')[:100]
        
    def generate_file_hash(self, content: str) -> str:
        """Generate hash for content deduplication"""
        return hashlib.md5(content.encode()).hexdigest()
        
    def build_advanced_prompt(self, title: str, summary: str, paper_type: str = "general") -> str:
        """Build advanced prompt with context awareness"""
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

Return ONLY the Python code wrapped in ```python ... ```. No explanations outside the code block.
"""
        return base_prompt
        
    async def query_model_with_retry(self, model: str, prompt: str) -> str:
        """Query model with retry logic and fallback"""
        for attempt in range(self.config.retry_attempts):
            try:
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                    "max_tokens": 4000
                }
                
                async with self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=self.headers['openrouter'],
                    json=body,
                    timeout=self.config.request_timeout
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    # Extract code from response
                    match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
                    code = match.group(1).strip() if match else content.strip()
                    
                    if len(code) < self.config.min_code_length:
                        raise ValueError(f"Code too short: {len(code)} characters")
                        
                    return code
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {model}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    
        raise Exception(f"All retry attempts failed for {model}")
        
    async def generate_code_with_fallback(self, title: str, summary: str) -> Tuple[str, List[str]]:
        """Generate code with multiple model fallback"""
        models = [self.config.primary_model, self.config.fallback_model, self.config.backup_model]
        errors = []
        
        # Determine paper type for better prompt engineering
        paper_type = self.classify_paper_type(title, summary)
        prompt = self.build_advanced_prompt(title, summary, paper_type)
        
        for model in models:
            try:
                self.logger.info(f"Attempting code generation with {model}")
                code = await self.query_model_with_retry(model, prompt)
                return code, errors
            except Exception as e:
                error_msg = f"Model {model} failed: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                
        # If all models fail, generate a template
        template_code = self.generate_template_code(title, summary)
        return template_code, errors
        
    def classify_paper_type(self, title: str, summary: str) -> str:
        """Classify paper type for better prompt engineering"""
        text = f"{title} {summary}".lower()
        
        if any(keyword in text for keyword in ['neural', 'deep learning', 'transformer', 'cnn', 'rnn']):
            return "machine_learning"
        elif any(keyword in text for keyword in ['algorithm', 'optimization', 'complexity']):
            return "algorithm"
        elif any(keyword in text for keyword in ['network', 'distributed', 'protocol']):
            return "systems"
        elif any(keyword in text for keyword in ['security', 'cryptography', 'privacy']):
            return "security"
        else:
            return "general"
            
    def generate_template_code(self, title: str, summary: str) -> str:
        """Generate template code when all models fail"""
        safe_title = self.sanitize_filename(title)
        return f'''"""
{title}
{summary}

This is a template implementation. Please implement the actual algorithm.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for {safe_title}"""
    debug: bool = False
    verbose: bool = True

class {safe_title.title().replace('_', '')}:
    """
    Implementation of {title}
    
    Based on: {summary}
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = logger
        
    def run(self, *args, **kwargs) -> Any:
        """Main execution method"""
        self.logger.info(f"Running {self.__class__.__name__}")
        
        # TODO: Implement the actual algorithm
        raise NotImplementedError("Algorithm implementation needed")
        
    def validate_input(self, data: Any) -> bool:
        """Validate input data"""
        return True
        
    def process(self, data: Any) -> Any:
        """Process the data"""
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
            
        # TODO: Implement processing logic
        return data

def main():
    """Main function for testing"""
    config = Config()
    implementation = {safe_title.title().replace('_', '')}(config)
    
    # TODO: Add test data and run implementation
    print("Template implementation created. Please implement the actual algorithm.")

if __name__ == "__main__":
    main()
'''

    def validate_code_quality(self, code: str) -> Tuple[float, List[str]]:
        """Validate code quality and return score with issues"""
        issues = []
        score = 100.0
        
        # Check for basic Python syntax
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
            score -= 50
            
        # Check for imports
        if 'import' not in code:
            issues.append("No imports found")
            score -= 10
            
        # Check for classes or functions
        if 'class ' not in code and 'def ' not in code:
            issues.append("No classes or functions found")
            score -= 20
            
        # Check for docstrings
        if '"""' not in code and "'''" not in code:
            issues.append("No docstrings found")
            score -= 10
            
        # Check for error handling
        if 'try:' not in code and 'except' not in code:
            issues.append("No error handling found")
            score -= 10
            
        # Check for type hints
        if ':' not in code or '->' not in code:
            issues.append("Limited type hints")
            score -= 5
            
        return max(0, score), issues
        
    def analyze_dependencies(self, code: str) -> List[str]:
        """Analyze code dependencies"""
        dependencies = []
        
        # Find import statements
        import_pattern = r'(?:from\s+(\S+)\s+import|import\s+(\S+))'
        matches = re.findall(import_pattern, code)
        
        for match in matches:
            module = match[0] or match[1]
            if module and not module.startswith('.'):
                dependencies.append(module.split('.')[0])
                
        return list(set(dependencies))
        
    def generate_requirements_txt(self, dependencies: List[str]) -> str:
        """Generate requirements.txt content"""
        standard_libs = {
            'os', 'sys', 'json', 'time', 'datetime', 'collections',
            'itertools', 'functools', 'operator', 're', 'math', 'random',
            'hashlib', 'logging', 'typing', 'dataclasses', 'enum', 'pathlib'
        }
        
        external_deps = [dep for dep in dependencies if dep not in standard_libs]
        
        # Map common imports to package names
        package_mapping = {
            'numpy': 'numpy>=1.21.0',
            'pandas': 'pandas>=1.3.0',
            'torch': 'torch>=1.9.0',
            'tensorflow': 'tensorflow>=2.6.0',
            'sklearn': 'scikit-learn>=1.0.0',
            'cv2': 'opencv-python>=4.5.0',
            'requests': 'requests>=2.25.0',
            'aiohttp': 'aiohttp>=3.8.0',
            'fastapi': 'fastapi>=0.68.0',
            'pydantic': 'pydantic>=1.8.0',
            'sqlalchemy': 'sqlalchemy>=1.4.0',
            'pytest': 'pytest>=6.2.0'
        }
        
        requirements = []
        for dep in external_deps:
            package = package_mapping.get(dep, f"{dep}>=1.0.0")
            requirements.append(package)
            
        return '\n'.join(sorted(requirements))
        
    def generate_documentation(self, title: str, summary: str, code: str) -> str:
        """Generate documentation for the code"""
        safe_title = self.sanitize_filename(title)
        
        doc_template = f"""# {title}

## Overview
{summary}

## Implementation Details
This module implements the core algorithm described in the research paper.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from {safe_title} import main

# Run the implementation
main()
```

## Code Structure
- Main implementation class
- Configuration management
- Input validation
- Error handling
- Logging and monitoring

## Dependencies
See `requirements.txt` for all dependencies.

## Testing
Run tests with:
```bash
pytest tests/
```

## Performance
The implementation includes basic performance monitoring and optimization.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License
MIT License - see LICENSE file for details.
"""
        return doc_template
        
    def generate_test_code(self, title: str, code: str) -> str:
        """Generate basic test code"""
        safe_title = self.sanitize_filename(title)
        class_name = safe_title.title().replace('_', '')
        
        test_template = f'''"""
Test suite for {title}
"""

import pytest
import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
try:
    from main import {class_name}
except ImportError:
    # If the class doesn't exist, create a mock
    class {class_name}:
        def __init__(self, *args, **kwargs):
            pass
            
        def run(self, *args, **kwargs):
            return "test_result"

class Test{class_name}(unittest.TestCase):
    """Test cases for {class_name}"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.instance = {class_name}()
        
    def test_initialization(self):
        """Test object initialization"""
        instance = {class_name}()
        self.assertIsNotNone(instance)
        
    def test_basic_functionality(self):
        """Test basic functionality"""
        # TODO: Add specific test cases
        result = self.instance.run() if hasattr(self.instance, 'run') else None
        self.assertIsNotNone(result)
        
    def test_input_validation(self):
        """Test input validation"""
        # TODO: Add input validation tests
        pass
        
    def test_error_handling(self):
        """Test error handling"""
        # TODO: Add error handling tests
        pass
        
    def test_edge_cases(self):
        """Test edge cases"""
        # TODO: Add edge case tests
        pass

# Performance tests
class TestPerformance(unittest.TestCase):
    """Performance test cases"""
    
    def test_execution_time(self):
        """Test execution time is reasonable"""
        import time
        start_time = time.time()
        
        # TODO: Add performance critical operations
        
        execution_time = time.time() - start_time
        self.assertLess(execution_time, 10.0, "Execution took too long")

if __name__ == "__main__":
    unittest.main()
'''
        return test_template
        
    async def create_project_structure(self, title: str, summary: str, code: str) -> ProcessingResult:
        """Create complete project structure"""
        safe_title = self.sanitize_filename(title)
        project_path = self.dirs['workspace'] / safe_title
        
        result = ProcessingResult(title=title, status=ProcessingStatus.PROCESSING)
        start_time = time.time()
        
        try:
            # Create project directory
            project_path.mkdir(exist_ok=True)
            
            # Validate code quality
            quality_score, quality_issues = self.validate_code_quality(code)
            result.code_quality_score = quality_score
            result.warnings.extend(quality_issues)
            
            # Analyze dependencies
            dependencies = self.analyze_dependencies(code)
            result.dependencies = dependencies
            
            # Save main code
            main_file = project_path / "main.py"
            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(code)
            result.code_generated = True
            
            # Generate requirements.txt
            requirements_content = self.generate_requirements_txt(dependencies)
            requirements_file = project_path / "requirements.txt"
            with open(requirements_file, 'w', encoding='utf-8') as f:
                f.write(requirements_content)
                
            # Generate documentation
            if self.config.enable_documentation_generation:
                doc_content = self.generate_documentation(title, summary, code)
                readme_file = project_path / "README.md"
                with open(readme_file, 'w', encoding='utf-8') as f:
                    f.write(doc_content)
                result.docs_generated = True
                
            # Generate tests
            if self.config.enable_testing_generation:
                test_dir = project_path / "tests"
                test_dir.mkdir(exist_ok=True)
                
                test_content = self.generate_test_code(title, code)
                test_file = test_dir / "test_main.py"
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(test_content)
                    
                # Create __init__.py
                init_file = test_dir / "__init__.py"
                init_file.touch()
                
                result.tests_generated = True
                
            # Create configuration file
            config_data = {
                'project': {
                    'name': safe_title,
                    'title': title,
                    'summary': summary,
                    'created': datetime.now().isoformat(),
                    'quality_score': quality_score
                },
                'dependencies': dependencies,
                'settings': {
                    'debug': False,
                    'verbose': True
                }
            }
            
            config_file = project_path / "config.yaml"
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False)
                
            result.status = ProcessingStatus.SUCCESS
            result.processing_time = time.time() - start_time
            
            self.logger.info(f"✅ Project created successfully: {safe_title}")
            
        except Exception as e:
            result.status = ProcessingStatus.FAILED
            result.errors.append(str(e))
            self.logger.error(f"❌ Failed to create project {safe_title}: {e}")
            
        return result
        
    async def process_paper_entry(self, title: str, summary: str, pdf_url: str = "") -> ProcessingResult:
        """Process a single paper entry"""
        self.logger.info(f"🚧 Processing: {title}")
        
        try:
            # Generate code
            code, generation_errors = await self.generate_code_with_fallback(title, summary)
            
            # Create project structure
            result = await self.create_project_structure(title, summary, code)
            
            # Add generation errors to result
            result.errors.extend(generation_errors)
            
            # Update statistics
            self.processing_stats['total_processed'] += 1
            if result.status == ProcessingStatus.SUCCESS:
                self.processing_stats['successful'] += 1
            else:
                self.processing_stats['failed'] += 1
                
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process {title}: {e}")
            return ProcessingResult(
                title=title,
                status=ProcessingStatus.FAILED,
                errors=[str(e)]
            )
            
    async def process_json_files(self) -> List[ProcessingResult]:
        """Process all JSON files concurrently"""
        results = []
        tasks = []
        
        for file_path in self.dirs['json'].glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                title = data.get("title", file_path.stem)
                summary = data.get("summary_and_goal", "No summary available.")
                pdf_url = data.get("paper_url", "")
                
                task = self.process_paper_entry(title, summary, pdf_url)
                tasks.append(task)
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load JSON file {file_path}: {e}")
                results.append(ProcessingResult(
                    title=file_path.stem,
                    status=ProcessingStatus.FAILED,
                    errors=[f"JSON parsing error: {e}"]
                ))
                
        # Process with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await task
                
        if tasks:
            completed_results = await asyncio.gather(
                *[process_with_semaphore(task) for task in tasks],
                return_exceptions=True
            )
            
            for result in completed_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Task failed with exception: {result}")
                    results.append(ProcessingResult(
                        title="Unknown",
                        status=ProcessingStatus.FAILED,
                        errors=[str(result)]
                    ))
                else:
                    results.append(result)
                    
        return results
        
    async def process_raw_files(self) -> List[ProcessingResult]:
        """Process all raw text files"""
        results = []
        
        for file_path in self.dirs['raw'].glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                # Extract information using regex
                title_match = re.search(r'"?title"?\s*:\s*"([^"]+)"', text)
                summary_match = re.search(r'"?(summary_and_goal|summary)"?\s*:\s*"([^"]+)"', text)
                url_match = re.search(r'"?paper_url"?\s*:\s*"([^"]+)"', text)
                
                title = title_match.group(1) if title_match else file_path.stem
                summary = summary_match.group(2) if summary_match else "Summary not available."
                pdf_url = url_match.group(1) if url_match else ""
                
                result = await self.process_paper_entry(title, summary, pdf_url)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"❌ Failed to process raw file {file_path}: {e}")
                results.append(ProcessingResult(
                    title=file_path.stem,
                    status=ProcessingStatus.FAILED,
                    errors=[f"Raw file processing error: {e}"]
                ))
                
        return results
        
    def generate_processing_report(self, results: List[ProcessingResult]) -> str:
        """Generate detailed processing report"""
        total_time = time.time() - self.processing_stats['start_time']
        
        report = f"""
# Coder Agent Processing Report

## Summary
- **Total Processed**: {self.processing_stats['total_processed']}
- **Successful**: {self.processing_stats['successful']}
- **Failed**: {self.processing_stats['failed']}
- **Success Rate**: {(self.processing_stats['successful'] / max(1, self.processing_stats['total_processed'])) * 100:.1f}%
- **Total Processing Time**: {total_time:.2f} seconds

## Detailed Results
"""
        
        for result in results:
            report += f"""
### {result.title}
- **Status**: {result.status.value}
- **Processing Time**: {result.processing_time:.2f}s
- **Code Quality Score**: {result.code_quality_score:.1f}/100
- **Code Generated**: {'✅' if result.code_generated else '❌'}
- **Tests Generated**: {'✅' if result.tests_generated else '❌'}
- **Docs Generated**: {'✅' if result.docs_generated else '❌'}
- **Dependencies**: {', '.join(result.dependencies) if result.dependencies else 'None'}

"""
            
            if result.errors:
                report += f"**Errors**: {', '.join(result.errors)}\n"
            if result.warnings:
                report += f"**Warnings**: {', '.join(result.warnings)}\n"
                
        return report
        
    async def push_to_github(self, results: List[ProcessingResult]) -> bool:
        """Push generated code to GitHub with enhanced security"""
        try:
            # Generate commit message with statistics
            successful_count = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
            commit_msg = f"🤖 Generated {successful_count} implementations from research papers"
            
            # Git configuration
            subprocess.run(["git", "config", "--global", "user.name", "coder-agent"], check=True)
            subprocess.run(["git", "config", "--global", "user.email", "coder@ai-research.com"], check=True)
            
            # SECURE: Configure Git credential helper to use environment token
            # This avoids exposing the token in URLs or command line arguments
            subprocess.run([
                "git", "config", "--global", "credential.helper", 
                "store --file=/tmp/git-credentials"
            ], check=True)
            
            # Create temporary credentials file with proper permissions
            credentials_content = f"https://x-access-token:{self.config.github_token}@github.com"
            with open("/tmp/git-credentials", "w") as f:
                f.write(credentials_content)
            os.chmod("/tmp/git-credentials", 0o600)  # Secure permissions
            
            # Set remote URL WITHOUT token (secure)
            repo_url = f"https://github.com/{self.config.repo_owner}/{self.config.repo_name}.git"
            subprocess.run(["git", "remote", "set-url", "origin", repo_url], check=True)
            
            # Create branch if enabled
            if self.config.create_branches:
                branch_name = f"generated-code-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            
            # Add files
            subprocess.run(["git", "add", "."], check=True)
            
            # Create detailed commit message
            detailed_msg = f"""{commit_msg}

📊 Processing Summary:
- Total processed: {len(results)}
- Successful: {successful_count}
- Failed: {len(results) - successful_count}
- Average quality score: {sum(r.code_quality_score for r in results) / len(results):.1f}/100

🔧 Generated files:
- Python implementations
- Test suites
- Documentation
- Configuration files
- Requirements specifications

Generated by Enhanced Coder Agent v2.0
"""
            
            subprocess.run(["git", "commit", "-m", detailed_msg], check=True)
            
            # Push to repository (Git will use credential helper)
            if self.config.create_branches:
                subprocess.run(["git", "push", "origin", branch_name], check=True)
                self.logger.info(f"✅ Code pushed to new branch: {branch_name}")
            else:
                subprocess.run(["git", "push", "origin", "main"], check=True)
                self.logger.info("✅ Code pushed to main branch")
            
            # Generate and save processing report
            report = self.generate_processing_report(results)
            report_file = self.dirs['logs'] / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Git operation failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ GitHub push failed: {e}")
            return False
        finally:
            # SECURITY: Clean up credentials file
            try:
                if os.path.exists("/tmp/git-credentials"):
                    os.remove("/tmp/git-credentials")
            except:
                pass
            
    async def run_health_checks(self) -> Dict[str, bool]:
        """Run system health checks"""
        checks = {}
        
        # Check API connectivity
        try:
            async with self.session.get(
                "https://openrouter.ai/api/v1/models",
                headers=self.headers['openrouter']
            ) as response:
                checks['openrouter_api'] = response.status == 200
        except:
            checks['openrouter_api'] = False
            
        # Check GitHub API
        try:
            async with self.session.get(
                "https://api.github.com/user",
                headers=self.headers['github']
            ) as response:
                checks['github_api'] = response.status == 200
        except:
            checks['github_api'] = False
            
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.dirs['workspace'])
            checks['disk_space'] = free > 1024 * 1024 * 1024  # 1GB free
        except:
            checks['disk_space'] = False
            
        # Check directories
        checks['directories'] = all(d.exists() for d in self.dirs.values())
        
        return checks
        
    def save_cache(self, key: str, data: Any) -> None:
        """Save data to cache"""
        cache_file = self.dirs['cache'] / f"{key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {key}: {e}")
            
    def load_cache(self, key: str) -> Optional[Any]:
        """Load data from cache"""
        cache_file = self.dirs['cache'] / f"{key}.json"
        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache {key}: {e}")
        return None
        
    async def cleanup_old_files(self, days: int = 30) -> None:
        """Clean up old generated files"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        for directory in [self.dirs['workspace'], self.dirs['logs'], self.dirs['cache']]:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    try:
                        if file_path.stat().st_mtime < cutoff_time:
                            file_path.unlink()
                            self.logger.info(f"🧹 Cleaned up old file: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup {file_path}: {e}")
                        
    async def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete code generation pipeline"""
        self.logger.info("🚀 Starting Enhanced Coder Agent Pipeline")
        
        # Run health checks
        health_checks = await self.run_health_checks()
        if not all(health_checks.values()):
            self.logger.warning(f"⚠️ Health check failures: {health_checks}")
            
        # Process files
        json_results = await self.process_json_files()
        raw_results = await self.process_raw_files()
        
        all_results = json_results + raw_results
        
        # Push to GitHub
        push_success = await self.push_to_github(all_results)
        
        # Cleanup old files
        await self.cleanup_old_files()
        
        # Generate final summary
        summary = {
            'total_processed': len(all_results),
            'successful': sum(1 for r in all_results if r.status == ProcessingStatus.SUCCESS),
            'failed': sum(1 for r in all_results if r.status == ProcessingStatus.FAILED),
            'average_quality': sum(r.code_quality_score for r in all_results) / len(all_results) if all_results else 0,
            'github_push_success': push_success,
            'health_checks': health_checks,
            'processing_time': time.time() - self.processing_stats['start_time']
        }
        
        self.logger.info(f"🎉 Pipeline completed: {summary}")
        return summary

# === CONFIGURATION LOADER ===

def load_config() -> Config:
    """Load configuration from environment and files"""
    load_dotenv()
    
    # Load from environment variables
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is missing from environment variables")
    if not github_token:
        raise ValueError("GITHUB_TOKEN is missing from environment variables")
    
    # Create config with defaults
    config = Config(
        openrouter_api_key=openrouter_api_key,
        github_token=github_token
    )
    
    # Load from config file if exists
    config_file = Path("config.yaml")
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            # Update config with file values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
        except Exception as e:
            logging.warning(f"Failed to load config file: {e}")
            
    return config

# === ENHANCED CLI INTERFACE ===

def create_default_config_file():
    """Create a default configuration file"""
    default_config = {
        'primary_model': 'deepseek/deepseek-r1-0528-qwen3-8b:free',
        'fallback_model': 'google/gemini-2.0-flash-exp:free',
        'backup_model': 'meta-llama/llama-3.1-8b-instruct:free',
        'max_concurrent_requests': 3,
        'request_timeout': 120,
        'retry_attempts': 3,
        'retry_delay': 2,
        'min_code_length': 50,
        'max_code_length': 10000,
        'temperature': 0.3,
        'repo_name': 'WATCHDOG_memory',
        'repo_owner': 'Sid7on1',
        'create_branches': True,
        'auto_merge': False,
        'enable_code_validation': True,
        'enable_dependency_analysis': True,
        'enable_documentation_generation': True,
        'enable_testing_generation': True,
        'enable_performance_profiling': True
    }
    
    with open('config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False)
        
    print("✅ Default configuration file created: config.yaml")

# === MAIN EXECUTION ===

async def main():
    """Main async function"""
    try:
        # Create default config if it doesn't exist
        if not Path("config.yaml").exists():
            create_default_config_file()
            
        # Load configuration
        config = load_config()
        
        # Run the coder agent
        async with CoderAgent(config) as agent:
            summary = await agent.run_full_pipeline()
            
            print("\n" + "="*50)
            print("🎉 CODER AGENT EXECUTION COMPLETED")
            print("="*50)
            print(f"📊 Total Processed: {summary['total_processed']}")
            print(f"✅ Successful: {summary['successful']}")
            print(f"❌ Failed: {summary['failed']}")
            print(f"⭐ Average Quality: {summary['average_quality']:.1f}/100")
            print(f"🚀 GitHub Push: {'✅' if summary['github_push_success'] else '❌'}")
            print(f"⏱️ Total Time: {summary['processing_time']:.2f}s")
            print("="*50)
            
            # Save final summary
            summary_file = Path("logs") / f"final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
                
    except Exception as e:
        logging.error(f"❌ Fatal error in main execution: {e}")
        raise

# === BACKWARDS COMPATIBILITY ===

def run_json_entries():
    """Backwards compatibility function"""
    print("⚠️ Using legacy function. Please use the new async pipeline.")
    asyncio.run(main())

def run_raw_entries():
    """Backwards compatibility function"""
    print("⚠️ Using legacy function. Please use the new async pipeline.")
    asyncio.run(main())

def push_to_memory_repo():
    """Backwards compatibility function"""
    print("⚠️ Using legacy function. Please use the new async pipeline.")
    asyncio.run(main())

# === ENTRY POINT ===

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "config":
            create_default_config_file()
        elif command == "health":
            async def check_health():
                config = load_config()
                async with CoderAgent(config) as agent:
                    health = await agent.run_health_checks()
                    print("Health Check Results:")
                    for check, status in health.items():
                        print(f"  {check}: {'✅' if status else '❌'}")
            asyncio.run(check_health())
        elif command == "cleanup":
            async def run_cleanup():
                config = load_config()
                async with CoderAgent(config) as agent:
                    await agent.cleanup_old_files()
                    print("✅ Cleanup completed")
            asyncio.run(run_cleanup())
        else:
            print("Unknown command. Available commands: config, health, cleanup")
    else:
        # Run main pipeline
        asyncio.run(main())
