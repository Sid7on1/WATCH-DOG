#!/usr/bin/env python3
"""
Enhanced Multi-Agent Project Manager
Coordinates 4 coding agents to implement advanced AI/ML projects
Focus: RAG, NLP, Agents, MLOps/CI-CD
"""

import os
import json
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import subprocess
import sys
from pusher import GitHubRepositoryManager

# Load environment variables
load_dotenv()

class EnhancedProjectManager:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.structures_dir = self.artifacts_dir / "structures"
        self.projects_dir = self.artifacts_dir / "projects"
        
        # Enhanced agent management with specialization for advanced AI/ML
        self.agents = {
            "coder1": {
                "status": "idle", 
                "current_task": None, 
                "process": None,
                "performance": {"tasks_completed": 0, "tasks_failed": 0, "avg_completion_time": 0, "total_time": 0},
                "health": {"last_heartbeat": None, "consecutive_failures": 0, "is_healthy": True},
                "specialization": ["rag_systems", "vector_databases", "retrieval_engines", "high_priority"]
            },
            "coder2": {
                "status": "idle", 
                "current_task": None, 
                "process": None,
                "performance": {"tasks_completed": 0, "tasks_failed": 0, "avg_completion_time": 0, "total_time": 0},
                "health": {"last_heartbeat": None, "consecutive_failures": 0, "is_healthy": True},
                "specialization": ["nlp_advanced", "transformers", "language_models", "medium_priority"]
            },
            "coder3": {
                "status": "idle", 
                "current_task": None, 
                "process": None,
                "performance": {"tasks_completed": 0, "tasks_failed": 0, "avg_completion_time": 0, "total_time": 0},
                "health": {"last_heartbeat": None, "consecutive_failures": 0, "is_healthy": True},
                "specialization": ["ai_agents", "multi_agent_systems", "reinforcement_learning", "medium_priority"]
            },
            "coder4": {
                "status": "idle", 
                "current_task": None, 
                "process": None,
                "performance": {"tasks_completed": 0, "tasks_failed": 0, "avg_completion_time": 0, "total_time": 0},
                "health": {"last_heartbeat": None, "consecutive_failures": 0, "is_healthy": True},
                "specialization": ["mlops_cicd", "deployment", "monitoring", "computer_vision", "low_priority"]
            }
        }
        
        # Enhanced task management
        self.max_retries = 3
        self.retry_delay = 45  # Longer delay for complex AI/ML tasks
        self.task_timeout = 900  # 15 minutes per task for complex implementations
        
        # Communication system
        self.message_queue = queue.Queue()
        self.agent_communications = {}
        self.critical_errors = []
        
        # Task management
        self.current_project = None
        self.project_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Initialize GitHub Repository Manager
        self.github_manager = GitHubRepositoryManager(artifacts_dir=artifacts_dir)
        
        # Ensure directories exist
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 Enhanced Project Manager initialized for Advanced AI/ML")
        print(f"📁 Structures directory: {self.structures_dir}")
        print(f"📁 Projects directory: {self.projects_dir}")
        print(f"👥 Managing 4 specialized AI/ML coding agents:")
        for agent_id, agent_info in self.agents.items():
            print(f"   🤖 {agent_id}: {', '.join(agent_info['specialization'])}")
        print(f"🔧 Features: Advanced AI/ML focus, Performance tracking, Health monitoring")
        print(f"📚 GitHub integration ready with {len(self.github_manager.seen_titles)} seen titles")
    
    def load_enhanced_project_plan(self, paper_name):
        """Load enhanced project plan from structures directory"""
        plan_file = self.structures_dir / paper_name / "enhanced_plan.json"
        
        if not plan_file.exists():
            # Fallback to regular plan
            plan_file = self.structures_dir / paper_name / "plan.json"
            if not plan_file.exists():
                print(f"❌ No project plan found for {paper_name}")
                return None
        
        try:
            with open(plan_file, 'r', encoding='utf-8') as f:
                plan = json.load(f)
            
            print(f"📋 Loaded enhanced project plan: {plan.get('project_name', paper_name)}")
            print(f"📋 Project type: {plan.get('project_type', 'unknown')}")
            print(f"📋 Complexity level: {plan.get('complexity_level', 'unknown')}")
            print(f"📋 Files to implement: {len(plan.get('required_files', []))}")
            
            return plan
            
        except Exception as e:
            print(f"❌ Error loading enhanced project plan: {e}")
            return None
    
    def create_enhanced_project_structure(self, plan, paper_name):
        """Create enhanced project directory structure"""
        project_name = plan.get('project_name', f'enhanced_project_from_{paper_name}')
        project_dir = self.projects_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create communication directory for agents
        comm_dir = project_dir / ".agent_comm"
        comm_dir.mkdir(exist_ok=True)
        
        # Create project-specific directories based on type
        project_type = plan.get('project_type', 'nlp_advanced')
        required_folders = plan.get('required_folders', [])
        
        for folder in required_folders:
            folder_path = project_dir / folder
            folder_path.mkdir(exist_ok=True)
        
        # Create additional directories for advanced AI/ML projects
        additional_dirs = {
            "rag_system": ["vector_stores", "embeddings", "retrievers", "evaluations"],
            "nlp_advanced": ["models", "tokenizers", "datasets", "fine_tuning"],
            "ai_agent": ["agents", "environments", "policies", "tools"],
            "mlops_cicd": ["pipelines", "monitoring", "deployments", "experiments"],
            "computer_vision": ["models", "datasets", "augmentations", "visualizations"]
        }
        
        if project_type in additional_dirs:
            for dir_name in additional_dirs[project_type]:
                dir_path = project_dir / dir_name
                dir_path.mkdir(exist_ok=True)
        
        print(f"🏗️  Created enhanced project structure: {project_dir}")
        print(f"📁 Project type: {project_type}")
        print(f"📁 Created {len(required_folders) + len(additional_dirs.get(project_type, []))} specialized directories")
        
        return project_dir
    
    def get_paper_content_for_agent(self, agent_id, task):
        """Get relevant paper content for this agent's task with enhanced context"""
        try:
            paper_name = self.current_project
            relevant_paper_dir = self.artifacts_dir / "relevant" / paper_name
            
            if not relevant_paper_dir.exists():
                print(f"⚠️ No paper content found for {paper_name}")
                return "No paper content available"
            
            # Check for enhanced metadata
            metadata_file = relevant_paper_dir / "enhanced_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"📊 Using enhanced metadata for {agent_id}")
            
            # Read all chunks
            chunk_files = sorted(relevant_paper_dir.glob("chunk_*.txt"))
            
            if not chunk_files:
                print(f"⚠️ No chunk files found for {paper_name}")
                return "No paper content available"
            
            # Enhanced content distribution based on agent specialization
            agent_specialization = self.agents[agent_id]["specialization"]
            
            # Intelligent chunk selection based on specialization
            selected_chunks = []
            
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_content = f.read().lower()
                    
                    # Check relevance to agent's specialization
                    relevance_score = 0
                    for spec in agent_specialization:
                        if spec.replace('_', ' ') in chunk_content:
                            relevance_score += 2
                        # Check for related terms
                        related_terms = self.get_related_terms(spec)
                        for term in related_terms:
                            if term in chunk_content:
                                relevance_score += 1
                    
                    if relevance_score > 0 or len(selected_chunks) < 2:  # Ensure minimum content
                        selected_chunks.append((chunk_file, relevance_score))
                        
                except Exception as e:
                    print(f"⚠️ Error reading chunk {chunk_file}: {e}")
            
            # Sort by relevance and select top chunks
            selected_chunks.sort(key=lambda x: x[1], reverse=True)
            selected_chunks = selected_chunks[:3]  # Top 3 most relevant chunks
            
            # Combine selected chunks
            combined_content = ""
            for chunk_file, score in selected_chunks:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_content = f.read()
                        combined_content += f"\n--- {chunk_file.name} (Relevance: {score}) ---\n{chunk_content}\n"
                except Exception as e:
                    print(f"⚠️ Error reading chunk {chunk_file}: {e}")
            
            print(f"📄 {agent_id} receives {len(selected_chunks)} specialized chunks ({len(combined_content)} chars)")
            return combined_content if combined_content else "No paper content available"
            
        except Exception as e:
            print(f"❌ Error getting paper content for {agent_id}: {e}")
            return "Error loading paper content"
    
    def get_related_terms(self, specialization):
        """Get related terms for agent specialization"""
        term_map = {
            "rag_systems": ["retrieval", "vector", "embedding", "search", "knowledge"],
            "vector_databases": ["faiss", "chroma", "pinecone", "weaviate", "similarity"],
            "retrieval_engines": ["search", "ranking", "relevance", "indexing", "query"],
            "nlp_advanced": ["transformer", "attention", "language", "text", "token"],
            "transformers": ["bert", "gpt", "attention", "encoder", "decoder"],
            "language_models": ["llm", "generation", "completion", "prompt", "fine-tuning"],
            "ai_agents": ["agent", "autonomous", "planning", "reasoning", "action"],
            "multi_agent_systems": ["coordination", "communication", "cooperation", "distributed"],
            "reinforcement_learning": ["reward", "policy", "q-learning", "actor-critic", "environment"],
            "mlops_cicd": ["deployment", "monitoring", "pipeline", "automation", "devops"],
            "deployment": ["serving", "api", "container", "kubernetes", "scaling"],
            "monitoring": ["metrics", "logging", "alerting", "performance", "health"],
            "computer_vision": ["image", "vision", "cnn", "detection", "segmentation"]
        }
        return term_map.get(specialization, [])
    
    def divide_enhanced_tasks(self, plan):
        """Divide project files into tasks optimized for advanced AI/ML projects"""
        required_files = plan.get('required_files', [])
        
        if not required_files:
            print("❌ No files to implement")
            return []
        
        # Enhanced priority and complexity-based sorting
        def get_priority_score(file_info):
            priority_scores = {"high": 3, "medium": 2, "low": 1}
            complexity_scores = {"high": 3, "medium": 2, "low": 1}
            
            priority = priority_scores.get(file_info.get('priority', 'medium'), 2)
            complexity = complexity_scores.get(file_info.get('complexity', 'medium'), 2)
            
            return priority * 10 + complexity  # Priority is more important
        
        sorted_files = sorted(required_files, key=get_priority_score, reverse=True)
        
        # Enhanced task grouping for AI/ML projects
        tasks = []
        current_task = []
        current_complexity = 0
        max_task_complexity = 6  # Higher for advanced projects
        
        for file_info in sorted_files:
            file_complexity = {"low": 1, "medium": 3, "high": 5}.get(file_info.get('complexity', 'medium'), 3)
            
            # Group related files together
            if self.should_group_files(current_task, file_info) and current_complexity + file_complexity <= max_task_complexity:
                current_task.append(file_info)
                current_complexity += file_complexity
            else:
                if current_task:
                    tasks.append(current_task)
                current_task = [file_info]
                current_complexity = file_complexity
        
        # Add the last task
        if current_task:
            tasks.append(current_task)
        
        print(f"📋 Divided work into {len(tasks)} enhanced tasks:")
        for i, task in enumerate(tasks, 1):
            files = [f.get('filename', f.get('name', 'unknown_file')) for f in task]
            priorities = [f.get('priority', 'medium') for f in task]
            print(f"  Task {i}: {', '.join(files)} (priorities: {', '.join(priorities)})")
        
        return tasks
    
    def should_group_files(self, current_task, new_file):
        """Determine if files should be grouped together based on AI/ML project logic"""
        if not current_task:
            return True
        
        # Get file purposes
        current_purposes = [f.get('purpose', '').lower() for f in current_task]
        new_purpose = new_file.get('purpose', '').lower()
        
        # Group related AI/ML components
        related_groups = [
            ["model", "architecture", "network"],
            ["training", "optimizer", "loss"],
            ["data", "loader", "preprocessing"],
            ["evaluation", "metrics", "testing"],
            ["retrieval", "vector", "embedding"],
            ["agent", "planning", "reasoning"],
            ["deployment", "serving", "monitoring"]
        ]
        
        for group in related_groups:
            current_in_group = any(any(term in purpose for term in group) for purpose in current_purposes)
            new_in_group = any(term in new_purpose for term in group)
            
            if current_in_group and new_in_group:
                return True
        
        return False
    
    def get_best_agent_for_enhanced_task(self, task):
        """Intelligently select the best agent for advanced AI/ML tasks"""
        # Analyze task characteristics
        task_purposes = [f.get('purpose', '').lower() for f in task]
        task_filenames = [f.get('filename', f.get('name', '')).lower() for f in task]
        task_priorities = [f.get('priority', 'medium') for f in task]
        
        # Score agents based on specialization match
        agent_scores = {}
        
        for agent_id, agent_info in self.agents.items():
            if agent_info["status"] != "idle" or not agent_info["health"]["is_healthy"]:
                continue
            
            score = 0
            specializations = agent_info["specialization"]
            
            # Enhanced specialization matching for AI/ML
            for spec in specializations:
                spec_terms = [spec.replace('_', ' ')] + self.get_related_terms(spec)
                
                for purpose in task_purposes:
                    for term in spec_terms:
                        if term in purpose:
                            score += 3
                
                for filename in task_filenames:
                    for term in spec_terms:
                        if term in filename:
                            score += 2
            
            # Priority matching
            if "high_priority" in specializations and "high" in task_priorities:
                score += 5
            elif "medium_priority" in specializations and "medium" in task_priorities:
                score += 3
            elif "low_priority" in specializations and "low" in task_priorities:
                score += 1
            
            # Performance bonus
            perf = agent_info["performance"]
            total_tasks = perf["tasks_completed"] + perf["tasks_failed"]
            if total_tasks > 0:
                success_rate = perf["tasks_completed"] / total_tasks
                score += success_rate * 3
            else:
                score += 2  # New agents get neutral score
            
            # Health penalty
            if agent_info["health"]["consecutive_failures"] > 0:
                score -= agent_info["health"]["consecutive_failures"] * 1.0
            
            agent_scores[agent_id] = score
        
        if not agent_scores:
            return None
        
        # Return best scoring agent
        best_agent = max(agent_scores.items(), key=lambda x: x[1])
        print(f"🎯 Selected {best_agent[0]} for enhanced task (score: {best_agent[1]:.1f})")
        print(f"   Specializations: {', '.join(self.agents[best_agent[0]]['specialization'])}")
        return best_agent[0]
    
    def assign_enhanced_task_to_agent(self, agent_id, task, project_dir, plan):
        """Assign an enhanced task to a specific agent"""
        if self.agents[agent_id]["status"] != "idle":
            print(f"⚠️  Agent {agent_id} is busy, cannot assign task")
            return False
        
        # Create enhanced task file for agent
        task_file = project_dir / ".agent_comm" / f"{agent_id}_enhanced_task.json"
        
        # Get specialized paper content for this agent
        paper_content = self.get_paper_content_for_agent(agent_id, task)
        
        # Enhanced task data with AI/ML specific information
        task_data = {
            "agent_id": agent_id,
            "task_id": f"enhanced_task_{len(self.completed_tasks) + len([a for a in self.agents.values() if a['status'] == 'working'])}",
            "files": task,
            "project_info": {
                "project_name": plan.get('project_name'),
                "project_type": plan.get('project_type'),
                "description": plan.get('description'),
                "key_algorithms": plan.get('key_algorithms', []),
                "main_libraries": plan.get('main_libraries', []),
                "complexity_level": plan.get('complexity_level', 'advanced'),
                "implementation_priority": plan.get('implementation_priority', 'high')
            },
            "paper_content": paper_content,
            "project_dir": str(project_dir),
            "communication_dir": str(project_dir / ".agent_comm"),
            "assigned_at": datetime.now().isoformat(),
            "status": "assigned",
            "agent_specialization": self.agents[agent_id]["specialization"],
            "enhanced_task": True
        }
        
        try:
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_data, f, indent=2)
            
            # Start agent process with enhanced script
            agent_script = Path(__file__).parent / f"enhanced_{agent_id}.py"
            if not agent_script.exists():
                # Fallback to regular agent script
                agent_script = Path(__file__).parent / f"{agent_id}.py"
            
            if agent_script.exists():
                process = subprocess.Popen([
                    sys.executable, str(agent_script), str(task_file)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                self.agents[agent_id]["status"] = "working"
                self.agents[agent_id]["current_task"] = task_data
                self.agents[agent_id]["process"] = process
                
                files = [f.get('filename', f.get('name', 'unknown_file')) for f in task]
                print(f"👤 Assigned enhanced task to {agent_id}: {', '.join(files)}")
                return True
            else:
                print(f"❌ Agent script not found: {agent_script}")
                return False
                
        except Exception as e:
            print(f"❌ Error assigning enhanced task to {agent_id}: {e}")
            return False
    
    def implement_enhanced_project(self, paper_name):
        """Main method to implement an enhanced AI/ML project using multiple agents"""
        print(f"\n{'='*80}")
        print(f"🚀 STARTING ENHANCED PROJECT IMPLEMENTATION: {paper_name}")
        print(f"🎯 Focus: Advanced AI/ML Systems (RAG, NLP, Agents, MLOps)")
        print(f"{'='*80}")
        
        # Load enhanced project plan
        plan = self.load_enhanced_project_plan(paper_name)
        if not plan:
            return False
        
        # Create enhanced project structure
        project_dir = self.create_enhanced_project_structure(plan, paper_name)
        
        # Divide work into enhanced tasks
        tasks = self.divide_enhanced_tasks(plan)
        if not tasks:
            return False
        
        self.project_tasks = tasks
        self.current_project = paper_name
        
        print(f"\n🎯 Enhanced Implementation Strategy:")
        print(f"   📋 Project Type: {plan.get('project_type', 'unknown')}")
        print(f"   📊 Complexity Level: {plan.get('complexity_level', 'unknown')}")
        print(f"   🔧 Key Algorithms: {', '.join(plan.get('key_algorithms', [])[:3])}")
        print(f"   📚 Main Libraries: {', '.join(plan.get('main_libraries', [])[:3])}")
        print(f"   📝 Total Tasks: {len(tasks)}")
        
        # Intelligently assign enhanced tasks to best available agents
        for i, task in enumerate(tasks):
            print(f"\n📋 Assigning Enhanced Task {i+1}/{len(tasks)}:")
            
            # Find best agent for this enhanced task
            best_agent = self.get_best_agent_for_enhanced_task(task)
            
            if best_agent:
                success = self.assign_enhanced_task_to_agent(best_agent, task, project_dir, plan)
                if success:
                    print(f"✅ Enhanced task {i+1} assigned to {best_agent}")
                else:
                    print(f"❌ Failed to assign enhanced task {i+1}")
                    self.failed_tasks.append({"files": task, "retry_count": 0})
            else:
                print(f"❌ No available agents for enhanced task {i+1}")
                self.failed_tasks.append({"files": task, "retry_count": 0})
            
            # Wait between task assignments
            time.sleep(5)
        
        # Monitor enhanced project progress
        return self.monitor_enhanced_project_progress(project_dir, plan)
    
    def monitor_enhanced_project_progress(self, project_dir, plan):
        """Monitor enhanced project progress with advanced AI/ML metrics"""
        print(f"\n🔍 Monitoring Enhanced Project Progress...")
        
        total_tasks = len(self.project_tasks)
        monitoring_interval = 30  # Check every 30 seconds
        max_monitoring_time = 3600  # 1 hour maximum
        start_time = time.time()
        
        while time.time() - start_time < max_monitoring_time:
            # Check agent status
            self.check_agent_status()
            
            # Check agent health
            self.check_agent_health()
            
            # Check for agent communications
            self.check_agent_communications(project_dir)
            
            # Monitor progress
            completed = len(self.completed_tasks)
            failed = len(self.failed_tasks)
            working = len([a for a in self.agents.values() if a["status"] == "working"])
            
            print(f"\n📊 Enhanced Project Progress:")
            print(f"   📋 Total tasks: {total_tasks}")
            print(f"   ✅ Completed: {completed}")
            print(f"   ❌ Failed: {failed}")
            print(f"   🔄 In progress: {working}")
            print(f"   📈 Progress: {(completed/total_tasks)*100:.1f}%")
            
            # Check for completion
            if completed + failed == total_tasks:
                if failed == 0:
                    print("🎉 ENHANCED PROJECT COMPLETED SUCCESSFULLY!")
                    self.print_enhanced_performance_summary()
                    return True
                else:
                    print(f"⚠️  ENHANCED PROJECT COMPLETED WITH {failed} FAILED TASKS")
                    # Attempt to retry failed tasks
                    self.retry_failed_tasks(project_dir, plan)
                    self.print_enhanced_performance_summary()
                    return failed == 0
            
            # Wait before next check
            time.sleep(monitoring_interval)
        
        print("⏰ Enhanced project monitoring timeout reached")
        self.print_enhanced_performance_summary()
        return False
    
    def print_enhanced_performance_summary(self):
        """Print enhanced agent performance summary with AI/ML metrics"""
        print(f"\n{'='*80}")
        print("🤖 ENHANCED AGENT PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        for agent_id, agent_info in self.agents.items():
            perf = agent_info["performance"]
            health = agent_info["health"]
            specializations = agent_info["specialization"]
            
            total_tasks = perf["tasks_completed"] + perf["tasks_failed"]
            success_rate = (perf["tasks_completed"] / total_tasks * 100) if total_tasks > 0 else 0
            
            print(f"\n🤖 {agent_id.upper()}:")
            print(f"   🎯 Specializations: {', '.join(specializations)}")
            print(f"   ✅ Tasks Completed: {perf['tasks_completed']}")
            print(f"   ❌ Tasks Failed: {perf['tasks_failed']}")
            print(f"   📈 Success Rate: {success_rate:.1f}%")
            print(f"   ⏱️  Avg Completion Time: {perf['avg_completion_time']:.1f}s")
            print(f"   💚 Health Status: {'✅ Healthy' if health['is_healthy'] else '❌ Unhealthy'}")
            print(f"   🔄 Consecutive Failures: {health['consecutive_failures']}")
        
        print(f"{'='*80}")
        print(f"🎯 Enhanced AI/ML Project Implementation Complete")
        print(f"{'='*80}")
    
    # Inherit other methods from the original manager
    def check_agent_status(self):
        """Check status of all agents (inherited from original)"""
        for agent_id, agent_info in self.agents.items():
            if agent_info["status"] == "working" and agent_info["process"]:
                if agent_info["process"].poll() is not None:
                    stdout, stderr = agent_info["process"].communicate()
                    
                    if agent_info["process"].returncode == 0:
                        print(f"✅ {agent_id} completed enhanced task successfully")
                        if stdout.strip():
                            print(f"   Output: {stdout.strip()}")
                        self.completed_tasks.append(agent_info["current_task"])
                        self.update_agent_performance(agent_id, success=True)
                        agent_info["status"] = "idle"
                    else:
                        print(f"❌ {agent_id} failed enhanced task (return code: {agent_info['process'].returncode})")
                        if stderr.strip():
                            print(f"   STDERR: {stderr.strip()}")
                        self.failed_tasks.append(agent_info["current_task"])
                        self.update_agent_performance(agent_id, success=False)
                        agent_info["status"] = "idle"
                    
                    agent_info["current_task"] = None
                    agent_info["process"] = None
    
    def update_agent_performance(self, agent_id, success=True, completion_time=0):
        """Update agent performance metrics"""
        perf = self.agents[agent_id]["performance"]
        health = self.agents[agent_id]["health"]
        
        if success:
            perf["tasks_completed"] += 1
            health["consecutive_failures"] = 0
            health["is_healthy"] = True
        else:
            perf["tasks_failed"] += 1
            health["consecutive_failures"] += 1
            
            if health["consecutive_failures"] >= 3:
                health["is_healthy"] = False
                print(f"⚠️ Agent {agent_id} marked as unhealthy after {health['consecutive_failures']} failures")
        
        if completion_time > 0:
            perf["total_time"] += completion_time
            total_tasks = perf["tasks_completed"] + perf["tasks_failed"]
            if total_tasks > 0:
                perf["avg_completion_time"] = perf["total_time"] / total_tasks
        
        health["last_heartbeat"] = datetime.now().isoformat()
    
    def check_agent_health(self):
        """Check and update agent health status"""
        current_time = datetime.now()
        
        for agent_id, agent_info in self.agents.items():
            health = agent_info["health"]
            
            if (agent_info["status"] == "working" and 
                agent_info["current_task"] and 
                agent_info["current_task"].get("assigned_at")):
                
                assigned_time = datetime.fromisoformat(agent_info["current_task"]["assigned_at"])
                elapsed = (current_time - assigned_time).total_seconds()
                
                if elapsed > self.task_timeout:
                    print(f"⏰ Agent {agent_id} timeout detected ({elapsed:.0f}s > {self.task_timeout}s)")
                    self.handle_agent_timeout(agent_id)
    
    def handle_agent_timeout(self, agent_id):
        """Handle agent timeout"""
        agent_info = self.agents[agent_id]
        
        if agent_info["process"]:
            agent_info["process"].terminate()
            print(f"🛑 Terminated {agent_id} due to timeout")
        
        if agent_info["current_task"]:
            task = agent_info["current_task"]
            task["retry_count"] = task.get("retry_count", 0) + 1
            
            if task["retry_count"] < self.max_retries:
                print(f"🔄 Scheduling retry {task['retry_count']}/{self.max_retries} for task")
                self.failed_tasks.append(task)
            else:
                print(f"❌ Task exceeded max retries ({self.max_retries})")
                self.failed_tasks.append(task)
        
        self.update_agent_performance(agent_id, success=False)
        agent_info["status"] = "idle"
        agent_info["current_task"] = None
        agent_info["process"] = None
    
    def retry_failed_tasks(self, project_dir, plan):
        """Retry failed tasks"""
        retry_tasks = []
        permanent_failures = []
        
        for task in self.failed_tasks:
            retry_count = task.get("retry_count", 0)
            if retry_count < self.max_retries:
                retry_tasks.append(task)
            else:
                permanent_failures.append(task)
        
        if retry_tasks:
            print(f"🔄 Retrying {len(retry_tasks)} failed enhanced tasks...")
            
            for task in retry_tasks:
                print(f"⏳ Waiting {self.retry_delay}s before retry...")
                time.sleep(self.retry_delay)
                
                task_files = task.get("files", [])
                best_agent = self.get_best_agent_for_enhanced_task(task_files)
                
                if best_agent:
                    self.failed_tasks.remove(task)
                    success = self.assign_enhanced_task_to_agent(best_agent, task_files, project_dir, plan)
                    
                    if success:
                        print(f"✅ Retry assigned to {best_agent}")
                    else:
                        print(f"❌ Retry assignment failed")
                        task["retry_count"] = task.get("retry_count", 0) + 1
                        self.failed_tasks.append(task)
                else:
                    print(f"❌ No healthy agents available for retry")
                    break
        
        self.failed_tasks = permanent_failures
        
        if permanent_failures:
            print(f"❌ {len(permanent_failures)} tasks permanently failed after {self.max_retries} retries")
    
    def check_agent_communications(self, project_dir):
        """Check for inter-agent communications"""
        comm_dir = project_dir / ".agent_comm"
        
        for comm_file in comm_dir.glob("*_comm_*.json"):
            try:
                with open(comm_file, 'r', encoding='utf-8') as f:
                    message = json.load(f)
                
                print(f"💬 Agent Communication: {message.get('from')} -> {message.get('to')}")
                print(f"   Message: {message.get('message')}")
                
                self.handle_agent_communication(message, project_dir)
                comm_file.unlink()
                
            except Exception as e:
                print(f"❌ Error processing communication: {e}")
    
    def handle_agent_communication(self, message, project_dir):
        """Handle communication between agents"""
        msg_type = message.get('type', 'info')
        
        if msg_type == 'conflict':
            print(f"⚠️  CONFLICT DETECTED: {message.get('message')}")
        elif msg_type == 'question':
            print(f"❓ QUESTION: {message.get('message')}")
        elif msg_type == 'critical_error':
            print(f"🚨 CRITICAL ERROR: {message.get('message')}")
            self.critical_errors.append(message)


def main():
    """Main execution function"""
    if len(sys.argv) != 2:
        print("Usage: python enhanced_manager.py <paper_name>")
        sys.exit(1)
    
    paper_name = sys.argv[1]
    
    try:
        manager = EnhancedProjectManager()
        success = manager.implement_enhanced_project(paper_name)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Enhanced manager error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()