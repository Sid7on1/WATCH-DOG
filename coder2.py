#!/usr/bin/env python3
"""
Coding Agent 2 - Multi-Agent Development System
Implements assigned files using Qwen Coder model
"""

import os
import json
import sys
import time
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def env(*names):
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return None

class CodingAgent:
    def __init__(self, agent_id="coder2"):
        self.agent_id = agent_id
        
        # Multi-API configuration - CODING OPTIMIZED MODELS (Different priority for coder2)
        self.apis = {
            "openrouter": {
                "key": env("OPEN_API", "OPENROUTER_API_KEY", "OPENROUTER_API_TOKEN"),
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "models": [
                    # CODING OPTIMIZED (different order for load balancing)
                    "deepseek/deepseek-r1-0528:free",           # Strong coding + reasoning
                    "qwen/qwen3-coder:free",                    # Best for coding
                    "moonshotai/kimi-k2:free",                  # Good for complex code
                    "google/gemini-2.0-flash-exp:free",         # Fast + reliable
                    "mistralai/mistral-nemo:free"               # High rate limits
                ]
            },
            "groq": {
                "key": env("groq_API", "GROQ_API"),
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "models": ["llama-3.1-8b-instant", "gemma2-9b-it", "llama3-8b-8192"]
            },
            "gemini": {
                "key": env("gemini_API", "GEMINI_API_KEY"),
                "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                "models": ["gemini-1.5-flash", "gemini-1.5-flash-8b"]
            },
            "cohere": {
                "key": env("cohere_API", "COHERE_API", "COHERE_API_KEY"),
                "url": "https://api.cohere.ai/v1/chat",
                "models": ["command-r", "command-light"]
            }
        }
        
        # Current API selection (start with Groq for speed)
        self.current_api = "groq"
        self.current_model_index = 0
        
        # Task management
        self.current_task = None
        self.project_dir = None
        self.comm_dir = None
        
        api_config, current_model = self.get_current_api_config()
        print(f"🤖 {self.agent_id} initialized with multi-API support")
        print(f"🔑 Current API: {self.current_api}")
        print(f"🧠 Current Model: {current_model}")
        print(f"🌐 Available APIs: {list(self.apis.keys())}")
    
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
            print(f"🔄 {self.agent_id} switched to next model in {self.current_api}: {current_api_config['models'][self.current_model_index]}")
            return True
        
        # Try next API
        api_names = list(self.apis.keys())
        current_api_index = api_names.index(self.current_api)
        
        for i in range(1, len(api_names)):
            next_api_index = (current_api_index + i) % len(api_names)
            next_api = api_names[next_api_index]
            
            if self.apis[next_api]["key"]:  # Check if API key exists
                self.current_api = next_api
                self.current_model_index = 0
                print(f"🔄 {self.agent_id} switched to API: {self.current_api} with model: {self.apis[next_api]['models'][0]}")
                return True
        
        print(f"❌ {self.agent_id} no more APIs available to switch to")
        return False
    
    def make_api_request(self, system_prompt, user_prompt, max_tokens=8000):
        """Make API request with current configuration"""
        api_config, current_model = self.get_current_api_config()
        
        if self.current_api == "gemini":
            return self.make_gemini_request(system_prompt, user_prompt, current_model, max_tokens)
        elif self.current_api == "cohere":
            return self.make_cohere_request(system_prompt, user_prompt, current_model, max_tokens)
        else:
            return self.make_openai_compatible_request(system_prompt, user_prompt, current_model, max_tokens)
    
    def make_openai_compatible_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to OpenAI-compatible APIs (OpenRouter, Groq)"""
        api_config, _ = self.get_current_api_config()
        
        # Use official Groq client if available
        if self.current_api == "groq":
            try:
                from groq import Groq
                
                client = Groq(api_key=api_config['key'])
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                
                return completion.choices[0].message.content
                
            except ImportError:
                pass  # Fall back to REST API
        
        # Standard REST API for OpenRouter and fallback for Groq
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
        
        response = requests.post(api_config["url"], headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def make_gemini_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to Google Gemini API using the new genai client"""
        try:
            from google import genai
            
            # Initialize client (gets API key from environment)
            client = genai.Client()
            
            # Combine system and user prompts for Gemini
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = client.models.generate_content(
                model=model,
                contents=combined_prompt
            )
            
            return response.text
            
        except ImportError:
            # Fallback to REST API if genai library not available
            api_config, _ = self.get_current_api_config()
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            headers = {
                "Content-Type": "application/json"
            }
            
            # Combine system and user prompts for Gemini
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": combined_prompt}]
                }],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.1
                }
            }
            
            response = requests.post(f"{url}?key={api_config['key']}", headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
    
    def make_cohere_request(self, system_prompt, user_prompt, model, max_tokens):
        """Make request to Cohere API using the official client"""
        try:
            import cohere
            
            # Initialize Cohere client
            api_config, _ = self.get_current_api_config()
            co = cohere.Client(api_config['key'])
            
            # Combine system and user prompts for Cohere
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = co.chat(
                model=model,
                message=combined_prompt,
                max_tokens=max_tokens,
                temperature=0.1
            )
            
            return response.text
            
        except ImportError:
            # Fallback to REST API if cohere library not available
            api_config, _ = self.get_current_api_config()
            
            headers = {
                "Authorization": f"Bearer {api_config['key']}",
                "Content-Type": "application/json"
            }
            
            # Combine system and user prompts for Cohere
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            payload = {
                "model": model,
                "message": combined_prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1
            }
            
            response = requests.post(api_config["url"], headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result['text']
    
    def load_task(self, task_file_path):
        """Load task assignment from manager"""
        try:
            with open(task_file_path, 'r', encoding='utf-8') as f:
                task = json.load(f)
            
            self.current_task = task
            self.project_dir = Path(task['project_dir'])
            self.comm_dir = Path(task['communication_dir'])
            
            print(f"📋 {self.agent_id} loaded task: {task['task_id']}")
            print(f"📁 Project: {task['project_info']['project_name']}")
            print(f"📝 Files to implement: {len(task['files'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading task: {e}")
            return False
    
    def communicate_with_agent(self, target_agent, message, msg_type="info", details=None):
        """Send message to another agent"""
        try:
            comm_message = {
                "from": self.agent_id,
                "to": target_agent,
                "type": msg_type,
                "message": message,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            }
            
            comm_file = self.comm_dir / f"{self.agent_id}_comm_{target_agent}.json"
            with open(comm_file, 'w', encoding='utf-8') as f:
                json.dump(comm_message, f, indent=2)
            
            print(f"💬 {self.agent_id} -> {target_agent}: {message}")
            
        except Exception as e:
            print(f"❌ Communication error: {e}")
    
    def broadcast_message(self, message, msg_type="info", details=None):
        """Broadcast message to all other agents"""
        other_agents = ["coder1", "coder2", "coder3", "coder4"]
        other_agents.remove(self.agent_id)
        
        for agent in other_agents:
            self.communicate_with_agent(agent, message, msg_type, details)
    
    def report_critical_error(self, error_message, error_details=None):
        """Report critical error to manager"""
        try:
            error_report = {
                "from": self.agent_id,
                "to": "manager",
                "type": "critical_error",
                "message": error_message,
                "details": error_details or {},
                "timestamp": datetime.now().isoformat()
            }
            
            error_file = self.comm_dir / f"{self.agent_id}_critical_error.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_report, f, indent=2)
            
            print(f"🚨 {self.agent_id} reported critical error: {error_message}")
            
        except Exception as e:
            print(f"❌ Error reporting critical error: {e}")
    
    def check_for_conflicts(self, file_info):
        """Check if other agents are working on related components"""
        filename = file_info.get('filename', file_info.get('name', 'unknown_file'))
        purpose = file_info['purpose']
        
        # Check for potential conflicts based on file purpose
        conflict_keywords = {
            'model': ['model', 'network', 'architecture'],
            'training': ['train', 'optimizer', 'loss'],
            'data': ['data', 'loader', 'dataset', 'preprocess'],
            'utils': ['util', 'helper', 'common']
        }
        
        for category, keywords in conflict_keywords.items():
            if any(keyword in purpose.lower() for keyword in keywords):
                message = f"Working on {filename} ({category} component). Please coordinate if you're working on related files."
                self.broadcast_message(message, "coordination", {
                    "component": category,
                    "filename": filename,
                    "purpose": purpose
                })
                break
    
    def generate_code(self, file_info, project_info, max_retries=2):
        """Generate code for a specific file using Qwen Coder"""
        filename = file_info.get('filename', file_info.get('name', 'unknown_file'))
        purpose = file_info['purpose']
        dependencies = file_info.get('dependencies', [])
        key_functions = file_info.get('key_functions', [])
        
        # Get paper content from task
        paper_content = self.current_task.get('paper_content', 'No paper content available')
        paper_summary = paper_content[:1000] + "..." if len(paper_content) > 1000 else paper_content
        
        system_prompt = f"""You are a SENIOR Python developer creating PRODUCTION-GRADE code for an enterprise XR eye tracking system.

PROJECT CONTEXT:
- Project: {project_info['project_name']}
- Type: {project_info['project_type']}
- Description: {project_info['description']}
- Key Algorithms: {', '.join(project_info.get('key_algorithms', []))}
- Main Libraries: {', '.join(project_info.get('main_libraries', []))}

RESEARCH PAPER CONTEXT:
{paper_summary}

CRITICAL TASK: Generate COMPREHENSIVE, PRODUCTION-READY Python code for: {filename}

TARGET: 400-800+ LINES OF ROBUST, ENTERPRISE-GRADE CODE

FILE SPECIFICATIONS:
- Purpose: {purpose}
- Required Dependencies: {', '.join(dependencies)}
- Key Functions to Implement: {', '.join(key_functions)}
- Priority: {file_info.get('priority', 'medium')}
- Complexity: {file_info.get('complexity', 'medium')}

MANDATORY REQUIREMENTS:
1. COMPREHENSIVE IMPLEMENTATION - Not just stubs or basic functions
2. FULL ERROR HANDLING - Try-catch blocks, validation, edge cases
3. EXTENSIVE LOGGING - Debug, info, warning, error levels
4. COMPLETE DOCSTRINGS - Class, method, parameter documentation
5. TYPE HINTS - Full typing annotations throughout
6. CONFIGURATION SUPPORT - Settings, parameters, customization
7. UNIT TEST COMPATIBILITY - Testable, mockable code structure
8. PERFORMANCE OPTIMIZATION - Efficient algorithms, memory management
9. THREAD SAFETY - Proper locking, concurrent access handling
10. INTEGRATION READY - Clean interfaces, dependency injection

IMPLEMENTATION DEPTH REQUIRED:
- Multiple classes with inheritance/composition
- Complex algorithms with step-by-step implementation
- Comprehensive parameter validation
- Detailed error messages and recovery
- Configuration management
- Performance monitoring
- Resource cleanup
- Event handling
- State management
- Data persistence

RESEARCH PAPER INTEGRATION:
- Implement EXACT algorithms from the paper (velocity-threshold, Flow Theory)
- Use paper's mathematical formulas and equations
- Follow paper's methodology precisely
- Include paper-specific constants and thresholds
- Implement all metrics mentioned in the paper

CODE STRUCTURE EXPECTATIONS:
- Main class with 10+ methods
- Helper classes and utilities
- Constants and configuration
- Exception classes
- Data structures/models
- Validation functions
- Utility methods
- Integration interfaces

QUALITY STANDARDS:
- Enterprise-grade error handling
- Professional logging throughout
- Comprehensive input validation
- Resource management (context managers)
- Clean code principles
- SOLID design patterns
- Performance considerations
- Security best practices

Generate SUBSTANTIAL, COMPLETE, PRODUCTION-READY code - NOT minimal examples!

IMPORTANT: Use the research paper content above to understand the specific algorithms, methods, and implementation details needed for this project. The code should reflect the paper's methodology and findings.

FILE SPECIFICATIONS:
- Purpose: {purpose}
- Required Dependencies: {', '.join(dependencies)}
- Key Functions to Implement: {', '.join(key_functions)}
- Priority: {file_info.get('priority', 'medium')}
- Estimated Complexity: {file_info.get('complexity', 'medium')}

CODE REQUIREMENTS:
1. Write complete, runnable Python code
2. Include all necessary imports
3. Implement all key functions specified
4. Add comprehensive docstrings and comments
5. Follow Python best practices (PEP 8)
6. Include proper error handling
7. Make code modular and maintainable
8. Add logging where appropriate
9. Include type hints where possible
10. Consider integration with other project components

IMPORTANT: 
- This is part of a larger project with multiple files
- Other agents are working on related components
- Ensure your code can integrate well with the overall system
- Use the specified dependencies and libraries

Respond with ONLY the complete Python code without markdown formatting."""

        user_prompt = f"""Generate production-ready Python code for {filename}.

The file should serve this purpose: {purpose}

Make sure to implement these key functions: {', '.join(key_functions)}

Create complete, functional code that fits into the {project_info['project_type']} project architecture."""

        for attempt in range(max_retries + 1):
            try:
                api_config, current_model = self.get_current_api_config()
                print(f"🔧 {self.agent_id} generating code for {filename} using {self.current_api} with {current_model}..." + 
                      (f" (Attempt {attempt + 1})" if attempt > 0 else ""))
                
                code_content = self.make_api_request(system_prompt, user_prompt, 8000)
                print(f"📝 {self.agent_id} extracted code content ({len(code_content)} characters)")
                
                # Clean up the response (remove markdown if present)
                if "```python" in code_content:
                    code_content = code_content.split("```python")[1].split("```")[0]
                elif "```" in code_content:
                    code_content = code_content.split("```")[1].split("```")[0]
                
                return code_content.strip()
                
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                print(f"❌ API error for {filename}: {error_msg}")
                
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    print(f"⚠️ Rate limit hit on {self.current_api}")
                    if self.switch_to_next_api() and attempt < max_retries:
                        time.sleep(30)
                        continue
                    else:
                        self.report_critical_error(f"Rate limit exceeded while generating {filename}")
                        return None
                
                elif "503" in error_msg or "busy" in error_msg.lower():
                    print(f"⚠️ Model busy on {self.current_api}")
                    if self.switch_to_next_api() and attempt < max_retries:
                        time.sleep(30)
                        continue
                    else:
                        self.report_critical_error(f"Model unavailable for {filename}")
                        return None
                
                if attempt < max_retries:
                    print(f"⏳ Retrying in 30 seconds...")
                    time.sleep(30)
                    continue
                else:
                    self.report_critical_error(f"Failed to generate code for {filename} after {max_retries + 1} attempts")
                    return None
                    
            except Exception as e:
                print(f"❌ Unexpected error generating {filename}: {e}")
                if self.switch_to_next_api() and attempt < max_retries:
                    time.sleep(30)
                    continue
                else:
                    self.report_critical_error(f"Unexpected error generating {filename}: {str(e)}")
                    return None
        
        return None
    
    def save_generated_file(self, filename, code_content):
        """Save generated code to project directory"""
        try:
            file_path = self.project_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code_content)
            
            print(f"💾 {self.agent_id} saved: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")
            self.report_critical_error(f"Failed to save {filename}: {str(e)}")
            return False
    
    def execute_task(self):
        """Execute the assigned task"""
        if not self.current_task:
            print(f"❌ {self.agent_id} has no task assigned")
            return False
        
        project_info = self.current_task['project_info']
        files_to_implement = self.current_task['files']
        
        print(f"🚀 {self.agent_id} starting task execution")
        
        # Announce start of work
        file_names = [f.get('filename', f.get('name', 'unknown_file')) for f in files_to_implement]
        self.broadcast_message(f"Starting work on: {', '.join(file_names)}", "status")
        
        success_count = 0
        
        for file_info in files_to_implement:
            filename = file_info.get('filename', file_info.get('name', 'unknown_file'))
            
            # Check for potential conflicts
            self.check_for_conflicts(file_info)
            
            # Generate code
            code_content = self.generate_code(file_info, project_info)
            
            if code_content:
                # Save the file
                if self.save_generated_file(filename, code_content):
                    success_count += 1
                    
                    # Announce completion
                    self.broadcast_message(f"Completed: {filename}", "completion", {
                        "filename": filename,
                        "purpose": file_info['purpose']
                    })
                else:
                    print(f"❌ Failed to save {filename}")
            else:
                print(f"❌ Failed to generate code for {filename}")
            
            # Small delay between files
            time.sleep(2)
        
        # Final status report
        total_files = len(files_to_implement)
        if success_count == total_files:
            print(f"✅ {self.agent_id} completed all {total_files} files successfully")
            self.broadcast_message(f"Task completed successfully: {success_count}/{total_files} files", "completion")
            return True
        else:
            print(f"⚠️  {self.agent_id} completed {success_count}/{total_files} files")
            self.broadcast_message(f"Task partially completed: {success_count}/{total_files} files", "warning")
            return False


def main():
    """Main execution function"""
    if len(sys.argv) != 2:
        print("Usage: python coder2.py <task_file_path>")
        sys.exit(1)
    
    task_file_path = sys.argv[1]
    
    try:
        agent = CodingAgent("coder2")
        
        if agent.load_task(task_file_path):
            success = agent.execute_task()
            sys.exit(0 if success else 1)
        else:
            print("❌ Failed to load task")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Agent error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
