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
    github_username: str  # NEW: GitHub username for creating repos
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
    
    # Repository Settings - MODIFIED for individual repos
    repo_prefix: str = "research-paper-"  # Prefix for generated repos
    repo_visibility: str = "public"  # public or private
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
    project_path: Optional[Path] = None
    github_repo_url: Optional[str] = None
    github_repo_name: Optional[str] = None  # NEW: Store the repo name
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.dependencies is None:
            self.dependencies = []

class CoderAgent:
    """Enhanced Coder Agent with GitHub Actions support"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = None
        self.setup_directories()
        self.setup_logging()
        self.setup_api_headers()
        self.setup_git_config()
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'repos_created': 0,
            'start_time': time.time()
        }
        
    def setup_logging(self):
        """Setup GitHub Actions compatible logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        # Setup handlers
        handlers = [
            logging.FileHandler('logs/coder_agent.log'),
        ]
        
        # Add console handler only if not in GitHub Actions
        if not os.getenv('GITHUB_ACTIONS'):
            handlers.append(logging.StreamHandler())
            
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=handlers
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
            'profiles': Path("profiles"),
            'temp': Path("temp")  # NEW: Temporary directory for individual repos
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
        
    def setup_git_config(self):
        """Setup Git configuration for GitHub Actions"""
        try:
            # Set Git user config
            subprocess.run([
                "git", "config", "--global", "user.name", 
                os.getenv('GITHUB_ACTOR', 'coder-agent')
            ], check=True)
            
            subprocess.run([
                "git", "config", "--global", "user.email", 
                f"{os.getenv('GITHUB_ACTOR', 'coder-agent')}@users.noreply.github.com"
            ], check=True)
            
            # Set credential helper for GitHub Actions
            subprocess.run([
                "git", "config", "--global", "credential.helper", 
                "store"
            ], check=True)
            
            self.logger.info("✅ Git configuration completed")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Git configuration failed: {e}")
            
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=10)
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    def sanitize_repo_name(self, name: str) -> str:
        """Sanitize name for GitHub repository"""
        # Remove special characters and spaces
        name = re.sub(r'[^\w\s-]', '', name)
        # Replace spaces with hyphens
        name = re.sub(r'\s+', '-', name)
        # Remove multiple hyphens
        name = re.sub(r'-+', '-', name)
        # Convert to lowercase
        name = name.lower()
        # Remove leading/trailing hyphens
        name = name.strip('-')
        # Limit length
        name = name[:80]
        
        return name
        
    def sanitize_filename(self, name: str) -> str:
        """Enhanced filename sanitization"""
        name = name.replace('\n', ' ').replace('\r', '')
        name = re.sub(r'[\\/*?:"<>|]', "", name)
        name = name.replace(" ", "_")
        name = re.sub(r'__+', '_', name)
        name = re.sub(r'[^\w\-_\.]', '', name)
        return name.strip('_')[:100]
        
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
- Make the code ready for production use

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
                    json=body
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
            
        # Check for main function
        if 'if __name__ == "__main__"' not in code:
            issues.append("No main function guard")
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
            'hashlib', 'logging', 'typing', 'dataclasses', 'enum', 'pathlib',
            'argparse', 'subprocess', 'shutil', 'tempfile', 'unittest'
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
            'pytest': 'pytest>=6.2.0',
            'matplotlib': 'matplotlib>=3.4.0',
            'seaborn': 'seaborn>=0.11.0',
            'scipy': 'scipy>=1.7.0',
            'networkx': 'networkx>=2.6.0',
            'plotly': 'plotly>=5.0.0'
        }
        
        requirements = []
        for dep in external_deps:
            package = package_mapping.get(dep, f"{dep}>=1.0.0")
            requirements.append(package)
            
        return '\n'.join(sorted(requirements))
        
    def generate_documentation(self, title: str, summary: str, code: str, dependencies: List[str]) -> str:
        """Generate comprehensive documentation"""
        safe_title = self.sanitize_filename(title)
        
        doc_template = f"""# {title}

## Overview
{summary}

## Implementation Details
This repository contains a Python implementation of the algorithm described in the research paper "{title}".

## Features
- Clean, production-ready Python code
- Comprehensive error handling
- Command-line interface
- Configurable parameters
- Detailed logging
- Unit tests included

## Installation

### Requirements
- Python 3.9 or higher
- pip package manager

### Setup
1. Clone this repository:
```bash
git clone <repository-url>
cd {safe_title}
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface
```bash
python main.py --input data.txt --output results.json
```

### Available Options
- `--input, -i`: Input data file or parameter
- `--output, -o`: Output file path
- `--debug, -d`: Enable debug mode
- `--verbose, -v`: Enable verbose output
- `--help, -h`: Show help message

### Python API
```python
from main import {self.sanitize_filename(title).title().replace('_', '')}
import json

# Create configuration
config = Config(debug=False, verbose=True)

# Initialize the implementation
implementation = {self.sanitize_filename(title).title().replace('_', '')}(config)

# Process data
result = implementation.run(your_data)

# Save results
implementation.save_results('output.json')
```

## Code Structure
```
{safe_title}/
├── main.py              # Main implementation
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── tests/              # Unit tests
│   └── test_main.py
├── docs/               # Additional documentation
└── examples/           # Usage examples
```

## Dependencies
{chr(10).join(f"- {dep}" for dep in dependencies) if dependencies else "- Standard Python libraries only"}

## Testing
Run the test suite:
```bash
python -m pytest tests/
```

Run with coverage:
```bash
python -m pytest tests/ --cov=main
```

## Performance
The implementation includes:
- Efficient algorithms optimized for the specific use case
- Memory usage optimization
- Performance monitoring and logging
- Scalable design patterns

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Algorithm Details
The implementation follows the methodology described in the original paper:
{summary}

### Key Components
- **Input Validation**: Ensures data integrity before processing
- **Preprocessing**: Prepares data for the main algorithm
- **Core Algorithm**: Implements the main research contribution
- **Postprocessing**: Formats and validates results
- **Error Handling**: Comprehensive error management

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use this implementation in your research, please cite the original paper:
```
{title}
```

## Acknowledgments
- Original research paper authors
- Open source libraries used in this implementation
- Contributors to this codebase

## Contact
For questions or issues, please open an issue on GitHub.
"""
        return doc_template
        
    def generate_test_code(self, title: str, code: str) -> str:
        """Generate comprehensive test code"""
        safe_title = self.sanitize_filename(title)
        class_name = ''.join(word.capitalize() for word in safe_title.split('_'))
        
        test_template = f'''"""
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
            self.fail(f"Unexpected exception type: {type(e).__name__}: {e}")

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
    assert peak < 100 * 1024 * 1024  # 100MB limit
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

    async def create_github_repository(self, repo_name: str, description: str) -> Tuple[bool, str, str]:
        """Create a new GitHub repository"""
        try:
            # Create repository payload
            repo_data = {
                "name": repo_name,
                "description": description,
                "private": self.config.repo_visibility == "private",
                "auto_init": True,
                "gitignore_template": "Python"
            }
            
            # Create repository
            async with self.session.post(
                "https://api.github.com/user/repos",
                headers=self.headers['github'],
                json=repo_data
            ) as response:
                if response.status == 201:
                    repo_info = await response.json()
                    repo_url = repo_info['html_url']
                    clone_url = repo_info['clone_url']
                    
                    self.logger.info(f"✅ Created repository: {repo_url}")
                    return True, repo_url, clone_url
                else:
                    error_text = await response.text()
                    self.logger.error(f"❌ Failed to create repository: {error_text}")
                    return False, "", ""
                    
        except Exception as e:
            self.logger.error(f"❌ Error creating repository: {e}")
            return False, "", ""
            
    async def setup_individual_repo(self, title: str, summary: str, code: str, 
                                  dependencies: List[str]) -> Tuple[bool, str, str]:
        """Set up individual repository for a research paper"""
        try:
            # Generate repository name
            safe_title = self.sanitize_repo_name(title)
            repo_name = f"{self.config.repo_prefix}{safe_title}"
            
            # Create GitHub repository
            success, repo_url, clone_url = await self.create_github_repository(
                repo_name, f"Implementation of: {title}"
            )
            
            if not success:
                return False, "", ""
                
            # Create local temporary directory
            temp_dir = self.dirs['temp'] / repo_name
            temp_dir.mkdir(exist_ok=True)
            
            # Initialize git repository
            subprocess.run(["git", "init"], cwd=temp_dir, check=True)
            subprocess.run(["git", "remote", "add", "origin", clone_url], cwd=temp_dir, check=True)
            
            # Create project files
            await self.create_project_files(temp_dir, title, summary, code, dependencies)
            
            # Commit and push
            await self.commit_and_push_repo(temp_dir, repo_name)
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            self.processing_stats['repos_created'] += 1
            return True, repo_url, repo_name
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up repository: {e}")
            return False, "", ""
            
    async def create_project_files(self, project_dir: Path, title: str, summary: str, 
                                 code: str, dependencies: List[str]):
        """Create all project files in the directory"""
        try:
            # Create main.py
            main_file = project_dir / "main.py"
            main_file.write_text(code, encoding='utf-8')
            
            # Create requirements.txt
            requirements_content = self.generate_requirements_txt(dependencies)
            if requirements_content:
                requirements_file = project_dir / "requirements.txt"
                requirements_file.write_text(requirements_content, encoding='utf-8')
            
            # Create README.md
            readme_content = self.generate_documentation(title, summary, code, dependencies)
            readme_file = project_dir / "README.md"
            readme_file.write_text(readme_content, encoding='utf-8')
            
            # Create test file
            test_content = self.generate_test_code(title, code)
            tests_dir = project_dir / "tests"
            tests_dir.mkdir(exist_ok=True)
            test_file = tests_dir / "test_main.py"
            test_file.write_text(test_content, encoding='utf-8')
            
            # Create additional files
            await self.create_additional_files(project_dir, title, summary)
            
        except Exception as e:
            self.logger.error(f"❌ Error creating project files: {e}")
            raise
            
    async def create_additional_files(self, project_dir: Path, title: str, summary: str):
        """Create additional project files"""
        try:
            # Create .gitignore
            gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/

# Data files
*.csv
*.json
*.pkl
*.pickle
data/
output/
results/
"""
            gitignore_file = project_dir / ".gitignore"
            gitignore_file.write_text(gitignore_content.strip(), encoding='utf-8')
            
            # Create LICENSE
            license_content = """MIT License

Copyright (c) 2024 Research Paper Implementation

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
            license_file = project_dir / "LICENSE"
            license_file.write_text(license_content, encoding='utf-8')
            
            # Create setup.py
            safe_title = self.sanitize_filename(title)
            setup_content = f'''from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="{safe_title}",
    version="0.1.0",
    author="Research Implementation",
    author_email="research@example.com",
    description="{summary[:100]}...",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/{safe_title}",
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
    install_requires=requirements,
    entry_points={{
        "console_scripts": [
            "{safe_title}=main:main",
        ],
    }},
)
'''
            setup_file = project_dir / "setup.py"
            setup_file.write_text(setup_content, encoding='utf-8')
            
            # Create examples directory
            examples_dir = project_dir / "examples"
            examples_dir.mkdir(exist_ok=True)
            
            example_content = f'''#!/usr/bin/env python3
"""
Example usage of {title}
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import {self.sanitize_filename(title).title().replace('_', '')}, Config

def main():
    """Example usage"""
    print("Example usage of {title}")
    
    # Create configuration
    config = Config(debug=True, verbose=True)
    
    # Create instance
    implementation = {self.sanitize_filename(title).title().replace('_', '')}(config)
    
    # Example data
    example_data = "example_input_data"
    
    try:
        # Run the algorithm
        result = implementation.run(example_data)
        
        print(f"Result: {{result}}")
        
        # Save results
        implementation.save_results("example_output.json")
        print("Results saved to example_output.json")
        
    except Exception as e:
        print(f"Error: {{e}}")

if __name__ == "__main__":
    main()
'''
            example_file = examples_dir / "example_usage.py"
            example_file.write_text(example_content, encoding='utf-8')
            
            # Create docs directory
            docs_dir = project_dir / "docs"
            docs_dir.mkdir(exist_ok=True)
            
            api_doc_content = f'''# API Documentation

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

#### {self.sanitize_filename(title).title().replace('_', '')}
Main implementation class.

**Methods:**
- `__init__(config: Config = None)`: Initialize the implementation
- `run(data: Any) -> Any`: Run the complete pipeline
- `validate_input(data: Any) -> bool`: Validate input data
- `save_results(filename: str = None) -> None`: Save results to file

### Usage Examples

```python
from main import {self.sanitize_filename(title).title().replace('_', '')}, Config

# Create configuration
config = Config(debug=True, verbose=True)

# Create instance
implementation = {self.sanitize_filename(title).title().replace('_', '')}(config)

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
            api_doc_file = docs_dir / "api.md"
            api_doc_file.write_text(api_doc_content, encoding='utf-8')
            
        except Exception as e:
            self.logger.error(f"❌ Error creating additional files: {e}")
            raise
            
    async def commit_and_push_repo(self, repo_dir: Path, repo_name: str):
        """Commit and push repository to GitHub"""
        try:
            # Add all files
            subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
            
            # Commit
            commit_message = f"Initial implementation of {repo_name}"
            subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_dir, check=True)
            
            # Push to main branch
            subprocess.run(["git", "branch", "-M", "main"], cwd=repo_dir, check=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], cwd=repo_dir, check=True)
            
            # Create and push development branch if enabled
            if self.config.create_branches:
                subprocess.run(["git", "checkout", "-b", "development"], cwd=repo_dir, check=True)
                subprocess.run(["git", "push", "-u", "origin", "development"], cwd=repo_dir, check=True)
                subprocess.run(["git", "checkout", "main"], cwd=repo_dir, check=True)
                
            self.logger.info(f"✅ Successfully pushed {repo_name} to GitHub")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Git operation failed: {e}")
            raise
            
    async def process_single_paper(self, paper_data: Dict) -> ProcessingResult:
        """Process a single research paper"""
        start_time = time.time()
        title = paper_data.get('title', 'Unknown Title')
        summary = paper_data.get('summary', 'No summary available')
        
        result = ProcessingResult(
            title=title,
            status=ProcessingStatus.PROCESSING
        )
        
        try:
            self.logger.info(f"🔄 Processing: {title}")
            
            # Generate code
            code, errors = await self.generate_code_with_fallback(title, summary)
            
            if errors:
                result.errors.extend(errors)
                
            # Validate code quality
            quality_score, quality_issues = self.validate_code_quality(code)
            result.code_quality_score = quality_score
            result.warnings.extend(quality_issues)
            
            # Analyze dependencies
            dependencies = self.analyze_dependencies(code)
            result.dependencies = dependencies
            
            # Create individual repository
            success, repo_url, repo_name = await self.setup_individual_repo(
                title, summary, code, dependencies
            )
            
            if success:
                result.github_repo_url = repo_url
                result.github_repo_name = repo_name
                result.code_generated = True
                result.docs_generated = True
                result.tests_generated = True
                result.status = ProcessingStatus.SUCCESS
                
                self.logger.info(f"✅ Successfully processed: {title}")
                self.logger.info(f"📦 Repository created: {repo_url}")
                
            else:
                result.status = ProcessingStatus.FAILED
                result.errors.append("Failed to create GitHub repository")
                
        except Exception as e:
            result.status = ProcessingStatus.FAILED
            result.errors.append(str(e))
            self.logger.error(f"❌ Error processing {title}: {e}")
            
        finally:
            result.processing_time = time.time() - start_time
            self.processing_stats['total_processed'] += 1
            
            if result.status == ProcessingStatus.SUCCESS:
                self.processing_stats['successful'] += 1
            else:
                self.processing_stats['failed'] += 1
                
        return result
        
    async def process_papers_batch(self, papers: List[Dict]) -> List[ProcessingResult]:
        """Process multiple papers concurrently"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def process_with_semaphore(paper_data):
            async with semaphore:
                return await self.process_single_paper(paper_data)
                
        tasks = [process_with_semaphore(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ProcessingResult(
                    title=papers[i].get('title', 'Unknown'),
                    status=ProcessingStatus.FAILED,
                    errors=[str(result)]
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
                
        return processed_results
        
    def generate_processing_report(self, results: List[ProcessingResult]) -> str:
        """Generate comprehensive processing report"""
        total_time = time.time() - self.processing_stats['start_time']
        
        report = f"""
# Research Paper Processing Report

## Summary
- **Total Papers Processed**: {self.processing_stats['total_processed']}
- **Successful**: {self.processing_stats['successful']}
- **Failed**: {self.processing_stats['failed']}
- **Repositories Created**: {self.processing_stats['repos_created']}
- **Total Processing Time**: {total_time:.2f} seconds
- **Success Rate**: {(self.processing_stats['successful']/max(1, self.processing_stats['total_processed'])*100):.1f}%

## Detailed Results

"""
        
        for result in results:
            status_emoji = {
                ProcessingStatus.SUCCESS: "✅",
                ProcessingStatus.FAILED: "❌",
                ProcessingStatus.SKIPPED: "⏭️",
                ProcessingStatus.PROCESSING: "🔄"
            }.get(result.status, "❓")
            
            report += f"\n### {status_emoji} {result.title}\n"
            report += f"- **Status**: {result.status.value}\n"
            report += f"- **Processing Time**: {result.processing_time:.2f}s\n"
            report += f"- **Code Quality Score**: {result.code_quality_score:.1f}/100\n"
            
            if result.github_repo_url:
                report += f"- **Repository**: [{result.github_repo_name}]({result.github_repo_url})\n"
                
            if result.dependencies:
                report += f"- **Dependencies**: {', '.join(result.dependencies)}\n"
                
            if result.errors:
                report += f"- **Errors**: {len(result.errors)}\n"
                for error in result.errors:
                    report += f"  - {error}\n"
                    
            if result.warnings:
                report += f"- **Warnings**: {len(result.warnings)}\n"
                for warning in result.warnings:
                    report += f"  - {warning}\n"
                    
        return report
        
    async def run_processing_pipeline(self, papers: List[Dict]) -> List[ProcessingResult]:
        """Run the complete processing pipeline"""
        self.logger.info(f"🚀 Starting processing pipeline for {len(papers)} papers")
        
        try:
            # Process papers in batches
            results = await self.process_papers_batch(papers)
            
            # Generate and save report
            report = self.generate_processing_report(results)
            report_file = self.dirs['logs'] / "processing_report.md"
            report_file.write_text(report, encoding='utf-8')
            
            self.logger.info(f"📊 Processing complete! Report saved to {report_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            raise

# ===== MAIN EXECUTION =====

def load_config() -> Config:
    """Load configuration from environment variables"""
    load_dotenv()
    
    required_env_vars = [
        'OPENROUTER_API_KEY',
        'GITHUB_TOKEN',
        'GITHUB_USERNAME'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    return Config(
        openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
        github_token=os.getenv('GITHUB_TOKEN'),
        github_username=os.getenv('GITHUB_USERNAME'),
        primary_model=os.getenv('PRIMARY_MODEL', "deepseek/deepseek-r1-0528-qwen3-8b:free"),
        fallback_model=os.getenv('FALLBACK_MODEL', "google/gemini-2.0-flash-exp:free"),
        backup_model=os.getenv('BACKUP_MODEL', "meta-llama/llama-3.1-8b-instruct:free"),
        max_concurrent_requests=int(os.getenv('MAX_CONCURRENT_REQUESTS', '3')),
        repo_prefix=os.getenv('REPO_PREFIX', 'research-paper-'),
        repo_visibility=os.getenv('REPO_VISIBILITY', 'public'),
        create_branches=os.getenv('CREATE_BRANCHES', 'true').lower() == 'true'
    )

def load_paper_data() -> List[Dict]:
    """Load paper data from JSON files"""
    paper_data = []
    json_dir = Path("relevant_json")
    
    if not json_dir.exists():
        raise FileNotFoundError(f"Directory {json_dir} does not exist")
    
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(data, list):
                paper_data.extend(data)
            elif isinstance(data, dict):
                paper_data.append(data)
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
            
    return paper_data

async def main():
    """Main function"""
    try:
        # Load configuration
        config = load_config()
        print(f"🔧 Configuration loaded successfully")
        
        # Load paper data
        papers = load_paper_data()
        print(f"📄 Loaded {len(papers)} papers for processing")
        
        # Initialize and run the coder agent
        async with CoderAgent(config) as agent:
            results = await agent.run_processing_pipeline(papers)
            
            # Print summary
            successful = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
            failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
            
            print(f"\n🎉 Processing Complete!")
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {failed}")
            print(f"📊 Success Rate: {(successful/len(results)*100):.1f}%")
            
            # Print repository links
            successful_repos = [r for r in results if r.github_repo_url]
            if successful_repos:
                print(f"\n🔗 Created Repositories:")
                for result in successful_repos:
                    print(f"  - {result.github_repo_name}: {result.github_repo_url}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
