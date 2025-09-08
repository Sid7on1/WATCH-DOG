#!/usr/bin/env python3
"""
Enhanced Coding Agent 4 - MLOps/CI-CD & Computer Vision Specialist
Specializes in: MLOps/CI-CD, deployment, monitoring, computer vision, low-priority tasks
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

class EnhancedCodingAgent:
    def __init__(self, agent_id="enhanced_coder4"):
        self.agent_id = agent_id
        self.specialization = ["mlops_cicd", "deployment", "monitoring", "computer_vision", "low_priority"]
        
        # Enhanced multi-API configuration optimized for MLOps/CV coding
        self.apis = {
            "openrouter": {
                "key": env("OPEN_API", "OPENROUTER_API_KEY", "OPENROUTER_API_TOKEN"),
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "models": [
                    "anthropic/claude-3.5-sonnet",             # Excellent for system architecture
                    "qwen/qwen3-coder:free",                    # Good for DevOps coding
                    "deepseek/deepseek-r1-0528:free",           # Strong reasoning for complex systems
                    "google/gemini-2.0-flash-exp:free"         # Fast and reliable
                ]
            },
            "groq": {
                "key": env("groq_API", "GROQ_API"),
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "models": [
                    "llama-3.3-70b-versatile",                 # Great for system design
                    "llama-3.1-70b-versatile",                 # Reliable for MLOps
                    "mixtral-8x7b-32768"                       # Good for deployment
                ]
            },
            "cohere": {
                "key": env("cohere_API", "COHERE_API", "COHERE_API_KEY"),
                "url": "https://api.cohere.ai/v2/chat",
                "models": ["command-r-plus", "command-r"]
            }
        }
        
        # Current API selection
        self.current_api = "openrouter"
        self.current_model_index = 0
        
        # Task management
        self.current_task = None
        self.project_dir = None
        self.comm_dir = None
        
        print(f"🤖 {self.agent_id} initialized - MLOps/CI-CD & Computer Vision Specialist")
        print(f"🎯 Specializations: {', '.join(self.specialization)}")
        print(f"🔑 Current API: {self.current_api}")
        print(f"🧠 Available models: {len(sum([config['models'] for config in self.apis.values()], []))}")
    
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
            print(f"🔄 {self.agent_id} switched to next model: {current_api_config['models'][self.current_model_index]}")
            return True
        
        api_names = list(self.apis.keys())
        current_api_index = api_names.index(self.current_api)
        
        for i in range(1, len(api_names)):
            next_api_index = (current_api_index + i) % len(api_names)
            next_api = api_names[next_api_index]
            
            if self.apis[next_api]["key"]:
                self.current_api = next_api
                self.current_model_index = 0
                print(f"🔄 {self.agent_id} switched to API: {self.current_api}")
                return True
        
        print(f"❌ {self.agent_id} no more APIs available")
        return False
    
    def make_api_request(self, system_prompt, user_prompt, max_tokens=10000):
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
            "temperature": 0.1
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
    
    def load_enhanced_task(self, task_file_path):
        """Load enhanced task assignment from manager"""
        try:
            with open(task_file_path, 'r', encoding='utf-8') as f:
                task = json.load(f)
            
            self.current_task = task
            self.project_dir = Path(task['project_dir'])
            self.comm_dir = Path(task['communication_dir'])
            
            print(f"📋 {self.agent_id} loaded enhanced task: {task['task_id']}")
            print(f"📁 Project: {task['project_info']['project_name']}")
            print(f"🎯 Project Type: {task['project_info']['project_type']}")
            print(f"📝 Files to implement: {len(task['files'])}")
            print(f"🔧 Complexity Level: {task['project_info'].get('complexity_level', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading enhanced task: {e}")
            return False
    
    def generate_enhanced_code(self, file_info, project_info, max_retries=2):
        """Generate enhanced code specialized for MLOps/CI-CD and Computer Vision"""
        filename = file_info.get('filename', file_info.get('name', 'unknown_file'))
        purpose = file_info['purpose']
        dependencies = file_info.get('dependencies', [])
        key_functions = file_info.get('key_functions', [])
        
        # Get paper content from task
        paper_content = self.current_task.get('paper_content', 'No paper content available')
        paper_summary = paper_content[:2000] + "..." if len(paper_content) > 2000 else paper_content
        
        system_prompt = f"""You are a SENIOR AI/ML Engineer specializing in MLOPS/CI-CD, DEPLOYMENT, MONITORING, and COMPUTER VISION.

AGENT SPECIALIZATION: {', '.join(self.specialization)}

PROJECT CONTEXT:
- Project: {project_info['project_name']}
- Type: {project_info['project_type']}
- Description: {project_info['description']}
- Complexity: {project_info.get('complexity_level', 'advanced')}
- Key Algorithms: {', '.join(project_info.get('key_algorithms', []))}
- Main Libraries: {', '.join(project_info.get('main_libraries', []))}

RESEARCH PAPER CONTEXT:
{paper_summary}

CRITICAL TASK: Generate PRODUCTION-READY Python code for: {filename}

TARGET: 600-1200+ LINES OF ENTERPRISE-GRADE CODE

SPECIALIZATION FOCUS:
🎯 MLOPS/CI-CD: Model deployment, monitoring, automation pipelines
🎯 DEPLOYMENT: Container orchestration, API serving, scaling
🎯 MONITORING: Performance metrics, alerting, observability
🎯 COMPUTER VISION: Image processing, object detection, neural networks
🎯 DEVOPS: Infrastructure as code, testing, quality assurance

FILE SPECIFICATIONS:
- Purpose: {purpose}
- Dependencies: {', '.join(dependencies)}
- Key Functions: {', '.join(key_functions)}
- Priority: {file_info.get('priority', 'low')}
- Complexity: {file_info.get('complexity', 'medium')}

MANDATORY REQUIREMENTS:
1. PRODUCTION DEPLOYMENT - Enterprise-grade deployment systems
2. MONITORING & OBSERVABILITY - Comprehensive metrics and logging
3. SCALABILITY - Handle high-throughput production workloads
4. COMPREHENSIVE ERROR HANDLING - Robust failure recovery
5. EXTENSIVE LOGGING - Debug, info, warning, error levels
6. COMPLETE DOCSTRINGS - Detailed deployment documentation
7. TYPE HINTS - Full typing annotations
8. CONFIGURATION SUPPORT - Flexible deployment parameters
9. PERFORMANCE OPTIMIZATION - Low-latency serving
10. SECURITY INTEGRATION - Authentication, authorization, encryption

IMPLEMENTATION DEPTH REQUIRED:
- Container orchestration (Docker, Kubernetes)
- Model serving and API endpoints
- Monitoring and alerting systems
- CI/CD pipeline automation
- Performance optimization
- Load balancing and scaling
- Data validation and quality checks
- Model versioning and registry
- A/B testing frameworks
- Computer vision pipelines

MLOPS-SPECIFIC REQUIREMENTS:
- Model lifecycle management
- Experiment tracking and versioning
- Automated testing and validation
- Feature store integration
- Data drift detection
- Model performance monitoring
- Rollback and canary deployments
- Infrastructure as code
- Security and compliance
- Cost optimization

COMPUTER VISION REQUIREMENTS:
- Image preprocessing pipelines
- Real-time inference optimization
- Batch processing capabilities
- Model compression and quantization
- GPU acceleration support
- Multi-modal input handling
- Result visualization
- Performance benchmarking
- Edge deployment support
- Quality assurance testing

CODE STRUCTURE EXPECTATIONS:
- Main deployment/CV class with 15+ methods
- Container and orchestration configs
- Monitoring and metrics collection
- API serving and endpoints
- Testing and validation frameworks
- Configuration management
- Performance optimization
- Security implementations
- Documentation and examples
- Integration interfaces

QUALITY STANDARDS:
- Enterprise-grade error handling
- Professional logging throughout
- Comprehensive input validation
- Resource management (CPU, GPU, memory)
- Clean code principles (SOLID, DRY)
- Performance benchmarking
- Security best practices
- Scalability considerations

Generate SUBSTANTIAL, COMPLETE, PRODUCTION-READY code optimized for MLOps/CI-CD and computer vision!

Respond with ONLY the complete Python code without markdown formatting."""

        user_prompt = f"""Generate production-ready Python code for {filename}.

This file should serve the purpose: {purpose}

Key functions to implement: {', '.join(key_functions)}

Focus on MLOps/CI-CD, deployment, monitoring, and computer vision. Create enterprise-grade code that can handle production workloads with advanced deployment and monitoring capabilities."""

        for attempt in range(max_retries + 1):
            try:
                api_config, current_model = self.get_current_api_config()
                print(f"🔧 {self.agent_id} generating MLOps/CV code for {filename} using {self.current_api}/{current_model}..." + 
                      (f" (Attempt {attempt + 1})" if attempt > 0 else ""))
                
                code_content = self.make_api_request(system_prompt, user_prompt, 10000)
                print(f"📝 {self.agent_id} generated specialized code ({len(code_content)} characters)")
                
                # Clean up the response
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
                        time.sleep(45)
                        continue
                    else:
                        return None
                
                if attempt < max_retries:
                    print(f"⏳ Retrying in 45 seconds...")
                    time.sleep(45)
                    continue
                else:
                    return None
                    
            except Exception as e:
                print(f"❌ Unexpected error generating {filename}: {e}")
                if self.switch_to_next_api() and attempt < max_retries:
                    time.sleep(45)
                    continue
                else:
                    return None
        
        return None
    
    def save_generated_file(self, filename, code_content):
        """Save generated code to project directory"""
        try:
            file_path = self.project_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code_content)
            
            print(f"💾 {self.agent_id} saved MLOps/CV specialized file: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")
            return False
    
    def execute_enhanced_task(self):
        """Execute the enhanced task with MLOps/CV specialization"""
        if not self.current_task:
            print(f"❌ {self.agent_id} has no enhanced task assigned")
            return False
        
        project_info = self.current_task['project_info']
        files_to_implement = self.current_task['files']
        
        print(f"🚀 {self.agent_id} starting enhanced MLOps/CV task execution")
        print(f"🎯 Specializing in: {', '.join(self.specialization)}")
        
        success_count = 0
        
        for file_info in files_to_implement:
            filename = file_info.get('filename', file_info.get('name', 'unknown_file'))
            
            print(f"\n🔧 Generating MLOps/CV specialized code for: {filename}")
            print(f"   Purpose: {file_info['purpose']}")
            print(f"   Priority: {file_info.get('priority', 'low')}")
            print(f"   Complexity: {file_info.get('complexity', 'medium')}")
            
            # Generate specialized code
            code_content = self.generate_enhanced_code(file_info, project_info)
            
            if code_content:
                if self.save_generated_file(filename, code_content):
                    success_count += 1
                    print(f"✅ Successfully implemented {filename}")
                else:
                    print(f"❌ Failed to save {filename}")
            else:
                print(f"❌ Failed to generate code for {filename}")
            
            # Delay between files for complex implementations
            time.sleep(3)
        
        # Final status report
        total_files = len(files_to_implement)
        if success_count == total_files:
            print(f"✅ {self.agent_id} completed all {total_files} MLOps/CV files successfully")
            return True
        else:
            print(f"⚠️  {self.agent_id} completed {success_count}/{total_files} files")
            return False


def main():
    """Main execution function"""
    if len(sys.argv) != 2:
        print("Usage: python enhanced_coder4.py <task_file_path>")
        sys.exit(1)
    
    task_file_path = sys.argv[1]
    
    try:
        agent = EnhancedCodingAgent("enhanced_coder4")
        
        if agent.load_enhanced_task(task_file_path):
            success = agent.execute_enhanced_task()
            sys.exit(0 if success else 1)
        else:
            print("❌ Failed to load enhanced task")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Enhanced agent error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()