#!/usr/bin/env python3
"""
Multi-Agent Project Manager
Coordinates 4 coding agents to implement projects from structure plans
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

# Load environment variables
load_dotenv()

class ProjectManager:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.structures_dir = self.artifacts_dir / "structures"
        self.projects_dir = self.artifacts_dir / "projects"
        
        # Enhanced agent management with performance tracking
        self.agents = {
            "coder1": {
                "status": "idle", 
                "current_task": None, 
                "process": None,
                "performance": {"tasks_completed": 0, "tasks_failed": 0, "avg_completion_time": 0, "total_time": 0},
                "health": {"last_heartbeat": None, "consecutive_failures": 0, "is_healthy": True},
                "specialization": ["high_priority", "core_logic"]
            },
            "coder2": {
                "status": "idle", 
                "current_task": None, 
                "process": None,
                "performance": {"tasks_completed": 0, "tasks_failed": 0, "avg_completion_time": 0, "total_time": 0},
                "health": {"last_heartbeat": None, "consecutive_failures": 0, "is_healthy": True},
                "specialization": ["medium_priority", "utilities"]
            },
            "coder3": {
                "status": "idle", 
                "current_task": None, 
                "process": None,
                "performance": {"tasks_completed": 0, "tasks_failed": 0, "avg_completion_time": 0, "total_time": 0},
                "health": {"last_heartbeat": None, "consecutive_failures": 0, "is_healthy": True},
                "specialization": ["data_processing", "analysis"]
            },
            "coder4": {
                "status": "idle", 
                "current_task": None, 
                "process": None,
                "performance": {"tasks_completed": 0, "tasks_failed": 0, "avg_completion_time": 0, "total_time": 0},
                "health": {"last_heartbeat": None, "consecutive_failures": 0, "is_healthy": True},
                "specialization": ["low_priority", "documentation"]
            }
        }
        
        # Enhanced task management with retry logic
        self.max_retries = 3
        self.retry_delay = 30  # seconds
        self.task_timeout = 600  # 10 minutes per task
        
        # Communication system
        self.message_queue = queue.Queue()
        self.agent_communications = {}
        self.critical_errors = []
        
        # Task management
        self.current_project = None
        self.project_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Ensure directories exist
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 Enhanced Project Manager initialized")
        print(f"📁 Structures directory: {self.structures_dir}")
        print(f"📁 Projects directory: {self.projects_dir}")
        print(f"👥 Managing 4 specialized coding agents: {list(self.agents.keys())}")
        print(f"🔧 Features: Performance tracking, Health monitoring, Intelligent assignment, Auto-retry")
    
    def load_project_plan(self, paper_name):
        """Load project plan from structures directory"""
        plan_file = self.structures_dir / paper_name / "plan.json"
        
        if not plan_file.exists():
            print(f"❌ No project plan found for {paper_name}")
            return None
        
        try:
            with open(plan_file, 'r', encoding='utf-8') as f:
                plan = json.load(f)
            
            print(f"📋 Loaded project plan: {plan.get('project_name', paper_name)}")
            print(f"📋 Project type: {plan.get('project_type', 'unknown')}")
            print(f"📋 Files to implement: {len(plan.get('required_files', []))}")
            
            return plan
            
        except Exception as e:
            print(f"❌ Error loading project plan: {e}")
            return None
    
    def create_project_structure(self, plan, paper_name):
        """Create project directory structure"""
        project_name = plan.get('project_name', f'project_from_{paper_name}')
        project_dir = self.projects_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Only create communication directory for agents (actually needed)
        comm_dir = project_dir / ".agent_comm"
        comm_dir.mkdir(exist_ok=True)
        
        print(f"🏗️  Created project structure: {project_dir}")
        print(f"📁 Created communication directory: .agent_comm")
        print(f"📝 Note: Other directories will be created only if referenced in generated code")
        return project_dir
    
    def get_paper_content_for_agent(self, agent_id, task):
        """Get relevant paper content for this agent's task"""
        try:
            # Find the paper directory in relevant folder
            paper_name = self.current_project
            relevant_paper_dir = self.artifacts_dir / "relevant" / paper_name
            
            if not relevant_paper_dir.exists():
                print(f"⚠️ No paper content found for {paper_name}")
                return "No paper content available"
            
            # Read all chunks
            chunk_files = sorted(relevant_paper_dir.glob("chunk_*.txt"))
            
            if not chunk_files:
                print(f"⚠️ No chunk files found for {paper_name}")
                return "No paper content available"
            
            # If only one chunk, all agents get the same content
            if len(chunk_files) == 1:
                with open(chunk_files[0], 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"📄 Single chunk: All agents receive same paper content ({len(content)} chars)")
                return content
            
            # Multiple chunks: distribute based on agent specialization and task
            agent_index = int(agent_id[-1]) - 1  # coder1=0, coder2=1, etc.
            
            # Intelligent chunk distribution based on task content
            task_keywords = []
            for file_info in task:
                purpose = file_info.get('purpose', '').lower()
                filename = file_info.get('filename', file_info.get('name', '')).lower()
                task_keywords.extend([purpose, filename])
            
            # Assign chunks based on content relevance
            selected_chunks = []
            
            # Primary chunk assignment (round-robin as base)
            primary_chunk_index = agent_index % len(chunk_files)
            selected_chunks.append(chunk_files[primary_chunk_index])
            
            # Add relevant chunks based on task keywords
            for i, chunk_file in enumerate(chunk_files):
                if i == primary_chunk_index:
                    continue  # Already added
                
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_content = f.read().lower()
                    
                    # Check if chunk contains task-relevant keywords
                    relevance_score = sum(1 for keyword in task_keywords if keyword in chunk_content)
                    
                    if relevance_score > 0:
                        selected_chunks.append(chunk_file)
                        
                except Exception as e:
                    print(f"⚠️ Error reading chunk {chunk_file}: {e}")
            
            # Combine selected chunks
            combined_content = ""
            for chunk_file in selected_chunks:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_content = f.read()
                        combined_content += f"\n--- {chunk_file.name} ---\n{chunk_content}\n"
                except Exception as e:
                    print(f"⚠️ Error reading chunk {chunk_file}: {e}")
            
            print(f"📄 {agent_id} receives {len(selected_chunks)} chunks ({len(combined_content)} chars)")
            return combined_content if combined_content else "No paper content available"
            
        except Exception as e:
            print(f"❌ Error getting paper content for {agent_id}: {e}")
            return "Error loading paper content"
    
    def divide_tasks(self, plan):
        """Divide project files into tasks for agents"""
        required_files = plan.get('required_files', [])
        
        if not required_files:
            print("❌ No files to implement")
            return []
        
        # Sort files by priority (high -> medium -> low)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_files = sorted(required_files, 
                            key=lambda x: priority_order.get(x.get('priority', 'medium'), 1))
        
        # Group files into tasks (1-3 files per task depending on complexity)
        tasks = []
        current_task = []
        current_complexity = 0
        
        for file_info in sorted_files:
            file_complexity = {"low": 1, "medium": 2, "high": 3}.get(file_info.get('complexity', 'medium'), 2)
            
            # If adding this file would make task too complex, start new task
            if current_complexity + file_complexity > 4 and current_task:
                tasks.append(current_task)
                current_task = [file_info]
                current_complexity = file_complexity
            else:
                current_task.append(file_info)
                current_complexity += file_complexity
        
        # Add the last task
        if current_task:
            tasks.append(current_task)
        
        print(f"📋 Divided work into {len(tasks)} tasks:")
        for i, task in enumerate(tasks, 1):
            files = [f.get('filename', f.get('name', 'unknown_file')) for f in task]
            print(f"  Task {i}: {', '.join(files)}")
        
        return tasks
    
    def get_best_agent_for_task(self, task):
        """Intelligently select the best agent for a task based on specialization and performance"""
        # Analyze task characteristics
        task_priorities = [f.get('priority', 'medium') for f in task]
        task_types = [f.get('purpose', '').lower() for f in task]
        
        # Score agents based on specialization match and performance
        agent_scores = {}
        
        for agent_id, agent_info in self.agents.items():
            if agent_info["status"] != "idle" or not agent_info["health"]["is_healthy"]:
                continue
            
            score = 0
            specializations = agent_info["specialization"]
            
            # Specialization matching
            if "high" in task_priorities and "high_priority" in specializations:
                score += 3
            elif "medium" in task_priorities and "medium_priority" in specializations:
                score += 2
            elif "low" in task_priorities and "low_priority" in specializations:
                score += 1
            
            # Task type matching
            for task_type in task_types:
                if "core" in task_type or "main" in task_type:
                    if "core_logic" in specializations:
                        score += 2
                elif "util" in task_type or "helper" in task_type:
                    if "utilities" in specializations:
                        score += 2
                elif "data" in task_type or "process" in task_type:
                    if "data_processing" in specializations:
                        score += 2
                elif "doc" in task_type or "readme" in task_type:
                    if "documentation" in specializations:
                        score += 2
            
            # Performance bonus (success rate)
            perf = agent_info["performance"]
            total_tasks = perf["tasks_completed"] + perf["tasks_failed"]
            if total_tasks > 0:
                success_rate = perf["tasks_completed"] / total_tasks
                score += success_rate * 2
            else:
                score += 1  # New agents get neutral score
            
            # Health penalty
            if agent_info["health"]["consecutive_failures"] > 0:
                score -= agent_info["health"]["consecutive_failures"] * 0.5
            
            agent_scores[agent_id] = score
        
        if not agent_scores:
            return None
        
        # Return best scoring agent
        best_agent = max(agent_scores.items(), key=lambda x: x[1])
        print(f"🎯 Selected {best_agent[0]} for task (score: {best_agent[1]:.1f})")
        return best_agent[0]
    
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
            
            # Mark as unhealthy after 3 consecutive failures
            if health["consecutive_failures"] >= 3:
                health["is_healthy"] = False
                print(f"⚠️ Agent {agent_id} marked as unhealthy after {health['consecutive_failures']} failures")
        
        # Update timing
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
            
            # Check for timeout (agent taking too long)
            if (agent_info["status"] == "working" and 
                agent_info["current_task"] and 
                agent_info["current_task"].get("assigned_at")):
                
                assigned_time = datetime.fromisoformat(agent_info["current_task"]["assigned_at"])
                elapsed = (current_time - assigned_time).total_seconds()
                
                if elapsed > self.task_timeout:
                    print(f"⏰ Agent {agent_id} timeout detected ({elapsed:.0f}s > {self.task_timeout}s)")
                    self.handle_agent_timeout(agent_id)
    
    def handle_agent_timeout(self, agent_id):
        """Handle agent timeout by terminating and marking for retry"""
        agent_info = self.agents[agent_id]
        
        if agent_info["process"]:
            agent_info["process"].terminate()
            print(f"🛑 Terminated {agent_id} due to timeout")
        
        # Mark task for retry
        if agent_info["current_task"]:
            task = agent_info["current_task"]
            task["retry_count"] = task.get("retry_count", 0) + 1
            
            if task["retry_count"] < self.max_retries:
                print(f"🔄 Scheduling retry {task['retry_count']}/{self.max_retries} for task")
                self.failed_tasks.append(task)  # Will be retried later
            else:
                print(f"❌ Task exceeded max retries ({self.max_retries})")
                self.failed_tasks.append(task)
        
        # Update agent status
        self.update_agent_performance(agent_id, success=False)
        agent_info["status"] = "idle"
        agent_info["current_task"] = None
        agent_info["process"] = None
    
    def retry_failed_tasks(self, project_dir, plan):
        """Retry failed tasks that haven't exceeded max retries"""
        retry_tasks = []
        permanent_failures = []
        
        for task in self.failed_tasks:
            retry_count = task.get("retry_count", 0)
            if retry_count < self.max_retries:
                retry_tasks.append(task)
            else:
                permanent_failures.append(task)
        
        if retry_tasks:
            print(f"🔄 Retrying {len(retry_tasks)} failed tasks...")
            
            for task in retry_tasks:
                # Wait for retry delay
                print(f"⏳ Waiting {self.retry_delay}s before retry...")
                time.sleep(self.retry_delay)
                
                # Find best available agent
                task_files = task.get("files", [])
                best_agent = self.get_best_agent_for_task(task_files)
                
                if best_agent:
                    # Remove from failed tasks and retry
                    self.failed_tasks.remove(task)
                    success = self.assign_task_to_agent(best_agent, task_files, project_dir, plan)
                    
                    if success:
                        print(f"✅ Retry assigned to {best_agent}")
                    else:
                        print(f"❌ Retry assignment failed")
                        task["retry_count"] = task.get("retry_count", 0) + 1
                        self.failed_tasks.append(task)
                else:
                    print(f"❌ No healthy agents available for retry")
                    break
        
        # Update failed tasks list
        self.failed_tasks = permanent_failures
        
        if permanent_failures:
            print(f"❌ {len(permanent_failures)} tasks permanently failed after {self.max_retries} retries")
    
    def print_agent_performance_summary(self):
        """Print detailed agent performance summary"""
        print(f"\n{'='*60}")
        print("AGENT PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        for agent_id, agent_info in self.agents.items():
            perf = agent_info["performance"]
            health = agent_info["health"]
            
            total_tasks = perf["tasks_completed"] + perf["tasks_failed"]
            success_rate = (perf["tasks_completed"] / total_tasks * 100) if total_tasks > 0 else 0
            
            print(f"\n🤖 {agent_id.upper()}:")
            print(f"   Specialization: {', '.join(agent_info['specialization'])}")
            print(f"   Tasks Completed: {perf['tasks_completed']}")
            print(f"   Tasks Failed: {perf['tasks_failed']}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Avg Completion Time: {perf['avg_completion_time']:.1f}s")
            print(f"   Health Status: {'✅ Healthy' if health['is_healthy'] else '❌ Unhealthy'}")
            print(f"   Consecutive Failures: {health['consecutive_failures']}")
        
        print(f"{'='*60}")
    
    def assign_task_to_agent(self, agent_id, task, project_dir, plan):
        """Assign a task to a specific agent"""
        if self.agents[agent_id]["status"] != "idle":
            print(f"⚠️  Agent {agent_id} is busy, cannot assign task")
            return False
        
        # Create task file for agent
        task_file = project_dir / ".agent_comm" / f"{agent_id}_task.json"
        
        # Get paper content for this agent
        paper_content = self.get_paper_content_for_agent(agent_id, task)
        
        task_data = {
            "agent_id": agent_id,
            "task_id": f"task_{len(self.completed_tasks) + len([a for a in self.agents.values() if a['status'] == 'working'])}",
            "files": task,
            "project_info": {
                "project_name": plan.get('project_name'),
                "project_type": plan.get('project_type'),
                "description": plan.get('description'),
                "key_algorithms": plan.get('key_algorithms', []),
                "main_libraries": plan.get('main_libraries', [])
            },
            "paper_content": paper_content,  # ADD PAPER CONTENT
            "project_dir": str(project_dir),
            "communication_dir": str(project_dir / ".agent_comm"),
            "assigned_at": datetime.now().isoformat(),
            "status": "assigned"
        }
        
        try:
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_data, f, indent=2)
            
            # Start agent process
            agent_script = Path(__file__).parent / f"{agent_id}.py"
            if agent_script.exists():
                process = subprocess.Popen([
                    sys.executable, str(agent_script), str(task_file)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                self.agents[agent_id]["status"] = "working"
                self.agents[agent_id]["current_task"] = task_data
                self.agents[agent_id]["process"] = process
                
                files = [f.get('filename', f.get('name', 'unknown_file')) for f in task]
                print(f"👤 Assigned to {agent_id}: {', '.join(files)}")
                return True
            else:
                print(f"❌ Agent script not found: {agent_script}")
                return False
                
        except Exception as e:
            print(f"❌ Error assigning task to {agent_id}: {e}")
            return False
    
    def check_agent_status(self):
        """Check status of all agents"""
        for agent_id, agent_info in self.agents.items():
            if agent_info["status"] == "working" and agent_info["process"]:
                # Check if process is still running
                if agent_info["process"].poll() is not None:
                    # Process finished
                    stdout, stderr = agent_info["process"].communicate()
                    
                    if agent_info["process"].returncode == 0:
                        print(f"✅ {agent_id} completed task successfully")
                        if stdout.strip():
                            print(f"   Output: {stdout.strip()}")
                        self.completed_tasks.append(agent_info["current_task"])
                        
                        # Update performance metrics
                        self.update_agent_performance(agent_id, success=True)
                        agent_info["status"] = "idle"
                    else:
                        print(f"❌ {agent_id} failed task with error (return code: {agent_info['process'].returncode}):")
                        if stderr.strip():
                            print(f"   STDERR: {stderr.strip()}")
                        if stdout.strip():
                            print(f"   STDOUT: {stdout.strip()}")
                        
                        # Check for specific error types
                        error_output = stderr + stdout
                        if "OPEN_API" in error_output:
                            print(f"   🔑 API Key issue detected for {agent_id}")
                        elif "rate limit" in error_output.lower():
                            print(f"   ⏳ Rate limit issue detected for {agent_id}")
                        elif "connection" in error_output.lower():
                            print(f"   🌐 Connection issue detected for {agent_id}")
                        
                        self.failed_tasks.append(agent_info["current_task"])
                        
                        # Update performance metrics for failure
                        self.update_agent_performance(agent_id, success=False)
                        agent_info["status"] = "idle"
                    
                    agent_info["current_task"] = None
                    agent_info["process"] = None
    
    def check_agent_communications(self, project_dir):
        """Check for inter-agent communications"""
        comm_dir = project_dir / ".agent_comm"
        
        # Check for communication files
        for comm_file in comm_dir.glob("*_comm_*.json"):
            try:
                with open(comm_file, 'r', encoding='utf-8') as f:
                    message = json.load(f)
                
                print(f"💬 Agent Communication: {message.get('from')} -> {message.get('to')}")
                print(f"   Message: {message.get('message')}")
                
                # Handle the communication
                self.handle_agent_communication(message, project_dir)
                
                # Remove processed communication file
                comm_file.unlink()
                
            except Exception as e:
                print(f"❌ Error processing communication: {e}")
    
    def handle_agent_communication(self, message, project_dir):
        """Handle communication between agents"""
        msg_type = message.get('type', 'info')
        
        if msg_type == 'conflict':
            print(f"⚠️  CONFLICT DETECTED: {message.get('message')}")
            # Create resolution file
            resolution_file = project_dir / ".agent_comm" / "manager_resolution.json"
            resolution = {
                "type": "conflict_resolution",
                "original_message": message,
                "resolution": f"Agent {message.get('from')} should handle {message.get('details', {}).get('component', 'unknown')}",
                "timestamp": datetime.now().isoformat()
            }
            
            with open(resolution_file, 'w', encoding='utf-8') as f:
                json.dump(resolution, f, indent=2)
                
        elif msg_type == 'question':
            print(f"❓ QUESTION: {message.get('message')}")
            # Provide guidance
            guidance_file = project_dir / ".agent_comm" / f"guidance_{message.get('from')}.json"
            guidance = {
                "type": "guidance",
                "question": message.get('message'),
                "answer": "Proceed with your implementation. Check project plan for details.",
                "timestamp": datetime.now().isoformat()
            }
            
            with open(guidance_file, 'w', encoding='utf-8') as f:
                json.dump(guidance, f, indent=2)
                
        elif msg_type == 'critical_error':
            print(f"🚨 CRITICAL ERROR: {message.get('message')}")
            self.critical_errors.append(message)
            # Stop all agents and reassess
            self.handle_critical_error(message, project_dir)
    
    def handle_critical_error(self, error_message, project_dir):
        """Handle critical errors reported by agents"""
        print(f"🚨 Handling critical error from {error_message.get('from')}")
        
        # Stop all agents
        for agent_id, agent_info in self.agents.items():
            if agent_info["status"] == "working" and agent_info["process"]:
                agent_info["process"].terminate()
                agent_info["status"] = "idle"
                agent_info["current_task"] = None
                agent_info["process"] = None
        
        # Create error resolution plan
        error_resolution = {
            "error": error_message,
            "action": "Project paused for manual review",
            "recommendation": "Check project plan and agent implementations",
            "timestamp": datetime.now().isoformat()
        }
        
        error_file = project_dir / ".agent_comm" / "critical_error.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_resolution, f, indent=2)
        
        print("🛑 All agents stopped due to critical error")
    
    def monitor_project_progress(self, project_dir):
        """Monitor overall project progress"""
        total_tasks = len(self.project_tasks)
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        working = len([a for a in self.agents.values() if a["status"] == "working"])
        
        print(f"\n📊 PROJECT PROGRESS:")
        print(f"   Total tasks: {total_tasks}")
        print(f"   Completed: {completed}")
        print(f"   Failed: {failed}")
        print(f"   In progress: {working}")
        print(f"   Progress: {(completed/total_tasks)*100:.1f}%")
        
        # Check for completion
        if completed + failed == total_tasks:
            if failed == 0:
                print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
            else:
                print(f"⚠️  PROJECT COMPLETED WITH {failed} FAILED TASKS")
            return True
        
        return False
    
    def implement_project(self, paper_name):
        """Main method to implement a project using multiple agents"""
        print(f"\n{'='*80}")
        print(f"🚀 STARTING PROJECT IMPLEMENTATION: {paper_name}")
        print(f"{'='*80}")
        
        # Load project plan
        plan = self.load_project_plan(paper_name)
        if not plan:
            return False
        
        # Create project structure
        project_dir = self.create_project_structure(plan, paper_name)
        
        # Divide work into tasks
        tasks = self.divide_tasks(plan)
        if not tasks:
            return False
        
        self.project_tasks = tasks
        self.current_project = paper_name
        
        # Intelligently assign tasks to best available agents
        for i, task in enumerate(tasks):
            # Use intelligent agent selection instead of round-robin
            best_agent = self.get_best_agent_for_task(task)
            
            if not best_agent:
                # Wait for any agent to become available
                print("⏳ Waiting for agents to become available...")
                while not best_agent:
                    time.sleep(2)
                    self.check_agent_status()
                    self.check_agent_health()  # Check for timeouts
                    best_agent = self.get_best_agent_for_task(task)
            
            success = self.assign_task_to_agent(best_agent, task, project_dir, plan)
            if not success:
                print(f"❌ Failed to assign task {i+1}, will retry later")
                # Add to failed tasks for retry
                task_data = {
                    "files": task,
                    "retry_count": 0,
                    "assigned_at": datetime.now().isoformat()
                }
                self.failed_tasks.append(task_data)
            
            time.sleep(1)  # Small delay between assignments
        
        # Monitor progress with enhanced features
        print(f"\n🔄 MONITORING PROJECT PROGRESS...")
        retry_attempted = False
        
        while True:
            self.check_agent_status()
            self.check_agent_health()  # Check for timeouts
            self.check_agent_communications(project_dir)
            
            # Check if we should retry failed tasks
            if (not retry_attempted and 
                len(self.failed_tasks) > 0 and 
                len([a for a in self.agents.values() if a["status"] == "working"]) == 0):
                
                print("🔄 Attempting to retry failed tasks...")
                self.retry_failed_tasks(project_dir, plan)
                retry_attempted = True
            
            if self.monitor_project_progress(project_dir):
                break
            
            # Check for critical errors
            if self.critical_errors:
                print("🚨 Critical errors detected, stopping project")
                break
            
            time.sleep(5)  # Check every 5 seconds
        
        # Final summary with performance analytics
        self.generate_project_summary(project_dir, plan)
        self.print_agent_performance_summary()
        return True
    
    def generate_project_summary(self, project_dir, plan):
        """Generate final project summary"""
        summary = {
            "project_name": plan.get('project_name'),
            "project_type": plan.get('project_type'),
            "completion_time": datetime.now().isoformat(),
            "total_tasks": len(self.project_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "critical_errors": len(self.critical_errors),
            "agents_used": list(self.agents.keys()),
            "project_directory": str(project_dir)
        }
        
        summary_file = project_dir / "project_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Project summary saved: {summary_file}")
    
    def process_all_projects(self):
        """Process all projects in structures directory"""
        if not self.structures_dir.exists():
            print(f"❌ Structures directory not found: {self.structures_dir}")
            return
        
        # Get all project plans
        project_dirs = [d for d in self.structures_dir.iterdir() if d.is_dir()]
        
        if not project_dirs:
            print("❌ No project plans found")
            return
        
        print(f"Found {len(project_dirs)} projects to implement")
        
        for project_dir in project_dirs:
            paper_name = project_dir.name
            
            # Reset state for new project
            self.project_tasks = []
            self.completed_tasks = []
            self.failed_tasks = []
            self.critical_errors = []
            
            success = self.implement_project(paper_name)
            
            if success:
                print(f"✅ Successfully processed {paper_name}")
            else:
                print(f"❌ Failed to process {paper_name}")
            
            # Wait between projects
            time.sleep(10)


def main():
    """Main execution function"""
    try:
        manager = ProjectManager()
        manager.process_all_projects()
        
    except KeyboardInterrupt:
        print("\n🛑 Manager stopped by user")
    except Exception as e:
        print(f"❌ Manager error: {e}")


if __name__ == "__main__":
    main()