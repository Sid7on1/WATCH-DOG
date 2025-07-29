#!/usr/bin/env python3
"""
GitHub Repository Manager for WATCHDOG_memory
Manages persistent storage of artifacts, seen titles, and workflow data
"""

import os
import json
import time
import base64
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import shutil
import tempfile

# Load environment variables
load_dotenv()

class GitHubRepositoryManager:
    def __init__(self, repo_name="WATCHDOG_memory", artifacts_dir="artifacts"):
        self.repo_name = repo_name
        self.artifacts_dir = Path(artifacts_dir)
        self.github_token = os.getenv("GITHUB_PAT") or os.getenv("GITHUB_API")
        
        # GitHub API configuration
        self.api_base = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        }
        
        self.github_username = self.get_github_username()
        
        # Repository URLs
        self.repo_url = f"{self.api_base}/repos/{self.github_username}/{self.repo_name}"
        self.contents_url = f"{self.repo_url}/contents"
        
        # Seen titles management
        self.seen_titles_file = "seen_titles.json"
        self.seen_titles = set()
        
        # Rate limiting
        self.rate_limit_delay = 1  # seconds between API calls
        self.last_api_call = 0
        
        # Activity logging
        self.activity_log = []
        self.workflow_start_time = None
        
        print(f"🔗 GitHub Repository Manager initialized")
        print(f"📁 Repository: {self.github_username}/{self.repo_name}")
        print(f"🔑 Token configured: {'✅' if self.github_token else '❌'}")
        print(f"📂 Local artifacts: {self.artifacts_dir}")
        
        # Initialize repository and load seen titles
        self.initialize_repository()
        self.load_seen_titles()
    
    def get_github_username(self):
        """Get GitHub username from API"""
        try:
            api_base = "https://api.github.com"
            response = requests.get(f"{api_base}/user", headers=self.headers)
            if response.status_code == 200:
                username = response.json()["login"]
                print(f"👤 GitHub user: {username}")
                return username
            else:
                print(f"❌ Failed to get GitHub username: {response.status_code}")
                return "unknown"
        except Exception as e:
            print(f"❌ Error getting GitHub username: {e}")
            return "unknown"
    
    def rate_limit_wait(self):
        """Implement rate limiting for GitHub API"""
        current_time = time.time()
        time_since_last = current_time - self.last_api_call
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
    
    def initialize_repository(self):
        """Initialize or verify the WATCHDOG_memory repository exists"""
        try:
            self.rate_limit_wait()
            response = requests.get(self.repo_url, headers=self.headers)
            
            if response.status_code == 200:
                print(f"✅ Repository {self.repo_name} exists")
                return True
            elif response.status_code == 404:
                print(f"📁 Repository {self.repo_name} not found, creating...")
                return self.create_repository()
            else:
                print(f"❌ Error checking repository: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error initializing repository: {e}")
            return False
    
    def create_repository(self):
        """Create the WATCHDOG_memory repository"""
        try:
            repo_data = {
                "name": self.repo_name,
                "description": "WATCHDOG AI Research Paper Processing Memory - Persistent storage for artifacts, seen titles, and workflow data",
                "private": False,
                "auto_init": True,
                "gitignore_template": "Python"
            }
            
            self.rate_limit_wait()
            response = requests.post(f"{self.api_base}/user/repos", 
                                   headers=self.headers, 
                                   json=repo_data)
            
            if response.status_code == 201:
                print(f"✅ Created repository {self.repo_name}")
                time.sleep(5)  # Wait for repository to be fully initialized
                return True
            else:
                print(f"❌ Failed to create repository: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating repository: {e}")
            return False
    
    def load_seen_titles(self):
        """Load seen titles from GitHub repository"""
        try:
            print("📥 Loading seen titles from GitHub...")
            
            self.rate_limit_wait()
            response = requests.get(f"{self.contents_url}/{self.seen_titles_file}", 
                                  headers=self.headers)
            
            if response.status_code == 200:
                # File exists, decode and load
                file_data = response.json()
                content = base64.b64decode(file_data["content"]).decode('utf-8')
                titles_data = json.loads(content)
                
                self.seen_titles = set(titles_data.get("titles", []))
                print(f"✅ Loaded {len(self.seen_titles)} seen titles from GitHub")
                
                # Store SHA for updates
                self.seen_titles_sha = file_data["sha"]
                
            elif response.status_code == 404:
                # File doesn't exist, create empty set
                print("📝 No seen titles file found, starting fresh")
                self.seen_titles = set()
                self.seen_titles_sha = None
                
                # Create initial file
                self.save_seen_titles()
                
            else:
                print(f"❌ Error loading seen titles: {response.status_code}")
                self.seen_titles = set()
                self.seen_titles_sha = None
                
        except Exception as e:
            print(f"❌ Error loading seen titles: {e}")
            self.seen_titles = set()
            self.seen_titles_sha = None
    
    def save_seen_titles(self):
        """Save seen titles to GitHub repository"""
        try:
            print(f"💾 Saving {len(self.seen_titles)} seen titles to GitHub...")
            
            # Prepare data
            titles_data = {
                "titles": sorted(list(self.seen_titles)),
                "last_updated": datetime.now().isoformat(),
                "total_count": len(self.seen_titles)
            }
            
            content = json.dumps(titles_data, indent=2, ensure_ascii=False)
            encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
            
            # Prepare request data
            request_data = {
                "message": f"Update seen titles - {len(self.seen_titles)} total titles",
                "content": encoded_content
            }
            
            # Add SHA if file exists (for updates)
            if hasattr(self, 'seen_titles_sha') and self.seen_titles_sha:
                request_data["sha"] = self.seen_titles_sha
            
            self.rate_limit_wait()
            response = requests.put(f"{self.contents_url}/{self.seen_titles_file}",
                                  headers=self.headers,
                                  json=request_data)
            
            if response.status_code in [200, 201]:
                # Update SHA for next update
                self.seen_titles_sha = response.json()["content"]["sha"]
                print(f"✅ Saved seen titles to GitHub")
                return True
            else:
                print(f"❌ Failed to save seen titles: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving seen titles: {e}")
            return False
    
    def add_seen_titles(self, new_titles):
        """Add new titles to seen titles set"""
        if isinstance(new_titles, str):
            new_titles = [new_titles]
        
        initial_count = len(self.seen_titles)
        self.seen_titles.update(new_titles)
        added_count = len(self.seen_titles) - initial_count
        
        if added_count > 0:
            print(f"📝 Added {added_count} new titles to seen list")
            return True
        else:
            print("📝 No new titles to add")
            return False
    
    def is_title_seen(self, title):
        """Check if a title has been seen before"""
        return title in self.seen_titles
    
    def get_file_from_repo(self, file_path):
        """Download a file from the GitHub repository"""
        try:
            self.rate_limit_wait()
            response = requests.get(f"{self.contents_url}/{file_path}", 
                                  headers=self.headers)
            
            if response.status_code == 200:
                file_data = response.json()
                content = base64.b64decode(file_data["content"])
                return content, file_data["sha"]
            elif response.status_code == 404:
                return None, None
            else:
                print(f"❌ Error getting file {file_path}: {response.status_code}")
                return None, None
                
        except Exception as e:
            print(f"❌ Error getting file {file_path}: {e}")
            return None, None
    
    def upload_file_to_repo(self, local_file_path, repo_file_path, commit_message=None):
        """Upload a file to the GitHub repository"""
        try:
            # Read local file
            with open(local_file_path, 'rb') as f:
                content = f.read()
            
            encoded_content = base64.b64encode(content).decode('utf-8')
            
            # Check if file exists to get SHA
            existing_content, existing_sha = self.get_file_from_repo(repo_file_path)
            
            # Prepare request data
            if not commit_message:
                commit_message = f"Update {repo_file_path}"
            
            request_data = {
                "message": commit_message,
                "content": encoded_content
            }
            
            if existing_sha:
                request_data["sha"] = existing_sha
            
            self.rate_limit_wait()
            response = requests.put(f"{self.contents_url}/{repo_file_path}",
                                  headers=self.headers,
                                  json=request_data)
            
            if response.status_code in [200, 201]:
                print(f"✅ Uploaded {local_file_path} -> {repo_file_path}")
                return True
            else:
                print(f"❌ Failed to upload {repo_file_path}: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error uploading {local_file_path}: {e}")
            return False
    
    def download_file_from_repo(self, repo_file_path, local_file_path):
        """Download a file from repository to local path"""
        try:
            content, sha = self.get_file_from_repo(repo_file_path)
            
            if content is not None:
                # Ensure local directory exists
                local_path = Path(local_file_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write file
                with open(local_path, 'wb') as f:
                    f.write(content)
                
                print(f"✅ Downloaded {repo_file_path} -> {local_file_path}")
                return True
            else:
                print(f"❌ File {repo_file_path} not found in repository")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading {repo_file_path}: {e}")
            return False
    
    def sync_artifacts_to_repo(self):
        """Sync all local artifacts to GitHub repository"""
        try:
            print(f"🔄 Syncing artifacts to GitHub repository...")
            
            if not self.artifacts_dir.exists():
                print(f"❌ Artifacts directory not found: {self.artifacts_dir}")
                return False
            
            uploaded_count = 0
            failed_count = 0
            
            # Walk through all files in artifacts directory
            for file_path in self.artifacts_dir.rglob("*"):
                if file_path.is_file():
                    # Calculate relative path for repository
                    relative_path = file_path.relative_to(self.artifacts_dir)
                    repo_path = f"artifacts/{relative_path.as_posix()}"
                    
                    # Upload file
                    if self.upload_file_to_repo(file_path, repo_path):
                        uploaded_count += 1
                    else:
                        failed_count += 1
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
            
            print(f"📊 Sync complete: {uploaded_count} uploaded, {failed_count} failed")
            return failed_count == 0
            
        except Exception as e:
            print(f"❌ Error syncing artifacts: {e}")
            return False
    
    def sync_artifacts_from_repo(self):
        """Sync artifacts from GitHub repository to local"""
        try:
            print(f"📥 Syncing artifacts from GitHub repository...")
            
            # Get repository tree
            self.rate_limit_wait()
            response = requests.get(f"{self.repo_url}/git/trees/main?recursive=1",
                                  headers=self.headers)
            
            if response.status_code != 200:
                print(f"❌ Failed to get repository tree: {response.status_code}")
                return False
            
            tree_data = response.json()
            downloaded_count = 0
            failed_count = 0
            
            # Download all files in artifacts directory
            for item in tree_data.get("tree", []):
                if item["type"] == "blob" and item["path"].startswith("artifacts/"):
                    repo_path = item["path"]
                    local_path = self.artifacts_dir / repo_path[10:]  # Remove "artifacts/" prefix
                    
                    if self.download_file_from_repo(repo_path, local_path):
                        downloaded_count += 1
                    else:
                        failed_count += 1
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
            
            print(f"📊 Download complete: {downloaded_count} downloaded, {failed_count} failed")
            return failed_count == 0
            
        except Exception as e:
            print(f"❌ Error syncing from repository: {e}")
            return False
    
    def log_activity(self, activity_type, description, status="success", details=None):
        """Log workflow activities"""
        activity = {
            "timestamp": datetime.now().isoformat(),
            "type": activity_type,
            "description": description,
            "status": status,
            "details": details or {}
        }
        self.activity_log.append(activity)
        
        # Print activity with appropriate emoji
        emoji = "✅" if status == "success" else "❌" if status == "error" else "⚠️"
        print(f"{emoji} {activity_type.upper()}: {description}")

    def workflow_start(self):
        """Called at the start of workflow - loads seen titles and syncs artifacts"""
        self.workflow_start_time = datetime.now()
        
        print(f"\n{'='*80}")
        print("🚀 WORKFLOW START - GITHUB SYNC")
        print(f"{'='*80}")
        
        self.log_activity("workflow", "Workflow started", "success")
        
        # Load seen titles (already done in __init__, but refresh)
        self.load_seen_titles()
        self.log_activity("sync", f"Loaded {len(self.seen_titles)} seen titles from GitHub", "success")
        
        # Sync artifacts from repository
        if self.sync_artifacts_from_repo():
            self.log_activity("sync", "Artifacts synced from GitHub repository", "success")
        else:
            self.log_activity("sync", "Failed to sync artifacts from GitHub", "error")
        
        print(f"✅ Workflow initialized with {len(self.seen_titles)} seen titles")
        return True
    
    def workflow_end(self, new_titles=None):
        """Called at the end of workflow - saves seen titles, syncs artifacts, creates project repositories, logs activities, and cleans up"""
        print(f"\n{'='*80}")
        print("🏁 WORKFLOW END - GITHUB SYNC")
        print(f"{'='*80}")
        
        # Add new titles if provided
        if new_titles:
            added = self.add_seen_titles(new_titles)
            if added:
                self.log_activity("titles", f"Added {len(new_titles)} new titles to seen list", "success")
        
        # Save seen titles
        if self.save_seen_titles():
            self.log_activity("sync", f"Saved {len(self.seen_titles)} seen titles to GitHub", "success")
        else:
            self.log_activity("sync", "Failed to save seen titles to GitHub", "error")
        
        # Sync artifacts to main repository
        if self.sync_artifacts_to_repo():
            self.log_activity("sync", "Artifacts synced to GitHub repository", "success")
        else:
            self.log_activity("sync", "Failed to sync artifacts to GitHub", "error")
        
        # Process all projects and create individual repositories
        if self.process_all_projects():
            self.log_activity("projects", "All projects processed and pushed to GitHub", "success")
        else:
            self.log_activity("projects", "Some projects failed to process", "warning")
        
        # Upload activity log to GitHub
        self.upload_activity_log()
        
        # Clean up all local artifacts
        self.cleanup_all_artifacts()
        
        print(f"✅ Workflow completed with {len(self.seen_titles)} total seen titles")
        return True
    
    def get_repository_stats(self):
        """Get repository statistics"""
        try:
            self.rate_limit_wait()
            response = requests.get(self.repo_url, headers=self.headers)
            
            if response.status_code == 200:
                repo_data = response.json()
                stats = {
                    "name": repo_data["name"],
                    "size": repo_data["size"],
                    "created_at": repo_data["created_at"],
                    "updated_at": repo_data["updated_at"],
                    "default_branch": repo_data["default_branch"],
                    "seen_titles_count": len(self.seen_titles)
                }
                return stats
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error getting repository stats: {e}")
            return None
    
    def repository_already_exists(self, repo_name):
        """Check if repository already exists"""
        try:
            self.rate_limit_wait()
            response = requests.get(f"{self.api_base}/repos/{self.github_username}/{repo_name}", 
                                  headers=self.headers)
            return response.status_code == 200
        except Exception as e:
            print(f"⚠️ Error checking if repository exists: {e}")
            return False
    
    def create_project_repository(self, project_name, project_path):
        """Create a new GitHub repository for a specific project"""
        try:
            # Clean project name for repository (remove special characters)
            repo_name = project_name.lower().replace(' ', '_').replace('-', '_')
            repo_name = ''.join(c for c in repo_name if c.isalnum() or c == '_')
            
            # Check if repository already exists
            if self.repository_already_exists(repo_name):
                print(f"⏭️ Repository {self.github_username}/{repo_name} already exists, skipping creation...")
                return None  # Skip this project to avoid conflicts
            
            print(f"🚀 Creating repository for project: {project_name}")
            
            # Repository data
            repo_data = {
                "name": repo_name,
                "description": f"AI-Generated Project: {project_name} - Created by WATCHDOG Multi-Agent System",
                "private": False,
                "auto_init": True,
                "gitignore_template": "Python"
            }
            
            self.rate_limit_wait()
            response = requests.post(f"{self.api_base}/user/repos", 
                                   headers=self.headers, 
                                   json=repo_data)
            
            if response.status_code == 201:
                print(f"✅ Created repository: {self.github_username}/{repo_name}")
                time.sleep(3)  # Wait for repository to be fully initialized
                return repo_name
            elif response.status_code == 422:
                # Repository might already exist (race condition)
                print(f"⚠️ Repository {repo_name} already exists (race condition), skipping...")
                return None
            else:
                print(f"❌ Failed to create repository {repo_name}: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating repository for {project_name}: {e}")
            return None
    
    def upload_project_to_repository(self, project_path, repo_name):
        """Upload all project files to the specific repository"""
        try:
            print(f"📤 Uploading project files to {repo_name}...")
            
            project_dir = Path(project_path)
            if not project_dir.exists():
                print(f"❌ Project directory not found: {project_path}")
                return False
            
            uploaded_count = 0
            failed_count = 0
            
            # Upload all files in the project directory
            for file_path in project_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    # Calculate relative path within the project
                    relative_path = file_path.relative_to(project_dir)
                    repo_file_path = relative_path.as_posix()
                    
                    # Upload file to the project repository
                    if self.upload_file_to_project_repo(file_path, repo_file_path, repo_name):
                        uploaded_count += 1
                    else:
                        failed_count += 1
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.8)
            
            print(f"📊 Upload complete for {repo_name}: {uploaded_count} uploaded, {failed_count} failed")
            return failed_count == 0
            
        except Exception as e:
            print(f"❌ Error uploading project to {repo_name}: {e}")
            return False
    
    def upload_file_to_project_repo(self, local_file_path, repo_file_path, repo_name):
        """Upload a file to a specific project repository"""
        try:
            # Read local file
            with open(local_file_path, 'rb') as f:
                content = f.read()
            
            encoded_content = base64.b64encode(content).decode('utf-8')
            
            # Check if file exists to get SHA
            project_contents_url = f"{self.api_base}/repos/{self.github_username}/{repo_name}/contents"
            existing_content, existing_sha = self.get_file_from_project_repo(repo_file_path, repo_name)
            
            # Prepare request data
            commit_message = f"Add {repo_file_path}"
            if existing_sha:
                commit_message = f"Update {repo_file_path}"
            
            request_data = {
                "message": commit_message,
                "content": encoded_content
            }
            
            if existing_sha:
                request_data["sha"] = existing_sha
            
            self.rate_limit_wait()
            response = requests.put(f"{project_contents_url}/{repo_file_path}",
                                  headers=self.headers,
                                  json=request_data)
            
            if response.status_code in [200, 201]:
                print(f"  ✅ {repo_file_path}")
                return True
            elif response.status_code == 409:
                print(f"  ⚠️ File {repo_file_path} already exists, attempting to update...")
                # Try to get the current file SHA and retry
                time.sleep(2)  # Brief delay
                existing_content, existing_sha = self.get_file_from_project_repo(repo_file_path, repo_name)
                if existing_sha:
                    request_data["sha"] = existing_sha
                    retry_response = requests.put(f"{project_contents_url}/{repo_file_path}",
                                                headers=self.headers,
                                                json=request_data)
                    if retry_response.status_code in [200, 201]:
                        print(f"  ✅ {repo_file_path} (updated)")
                        return True
                    else:
                        print(f"  ❌ Retry failed for {repo_file_path}: {retry_response.status_code}")
                        return False
                else:
                    print(f"  ❌ Could not get SHA for existing {repo_file_path}")
                    return False
            else:
                print(f"  ❌ Failed to upload {repo_file_path}: {response.status_code}")
                if response.text:
                    print(f"      Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error uploading {local_file_path}: {e}")
            return False
    
    def get_file_from_project_repo(self, file_path, repo_name):
        """Get a file from a specific project repository"""
        try:
            project_contents_url = f"{self.api_base}/repos/{self.github_username}/{repo_name}/contents"
            
            self.rate_limit_wait()
            response = requests.get(f"{project_contents_url}/{file_path}", 
                                  headers=self.headers)
            
            if response.status_code == 200:
                file_data = response.json()
                content = base64.b64decode(file_data["content"])
                return content, file_data["sha"]
            elif response.status_code == 404:
                return None, None
            else:
                return None, None
                
        except Exception as e:
            return None, None
    
    def cleanup_local_project(self, project_path, project_name):
        """Delete local project directory after successful push to GitHub"""
        try:
            print(f"🧹 Cleaning up local project directory: {project_name}")
            
            if project_path.exists() and project_path.is_dir():
                shutil.rmtree(project_path)
                print(f"✅ Deleted local project directory: {project_path}")
                return True
            else:
                print(f"⚠️ Project directory not found or not a directory: {project_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error cleaning up project directory {project_path}: {e}")
            return False
    
    def upload_activity_log(self):
        """Upload workflow activity log to GitHub repository"""
        try:
            if not self.activity_log:
                print("📝 No activities to log")
                return True
            
            print(f"📝 Uploading activity log with {len(self.activity_log)} activities...")
            
            # Calculate workflow duration
            workflow_duration = None
            if self.workflow_start_time:
                duration_seconds = (datetime.now() - self.workflow_start_time).total_seconds()
                workflow_duration = f"{duration_seconds:.2f} seconds"
            
            # Create comprehensive log data
            log_data = {
                "workflow_id": f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "start_time": self.workflow_start_time.isoformat() if self.workflow_start_time else None,
                "end_time": datetime.now().isoformat(),
                "duration": workflow_duration,
                "total_activities": len(self.activity_log),
                "activities": self.activity_log,
                "summary": {
                    "success_count": len([a for a in self.activity_log if a["status"] == "success"]),
                    "error_count": len([a for a in self.activity_log if a["status"] == "error"]),
                    "warning_count": len([a for a in self.activity_log if a["status"] == "warning"])
                }
            }
            
            # Convert to JSON
            log_content = json.dumps(log_data, indent=2, ensure_ascii=False)
            encoded_content = base64.b64encode(log_content.encode('utf-8')).decode('utf-8')
            
            # Create log file path with timestamp
            log_filename = f"logs/workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Upload log file
            request_data = {
                "message": f"Add workflow activity log - {len(self.activity_log)} activities",
                "content": encoded_content
            }
            
            self.rate_limit_wait()
            response = requests.put(f"{self.contents_url}/{log_filename}",
                                  headers=self.headers,
                                  json=request_data)
            
            if response.status_code in [200, 201]:
                print(f"✅ Activity log uploaded: {log_filename}")
                self.log_activity("logging", f"Activity log uploaded with {len(self.activity_log)} activities", "success")
                return True
            else:
                print(f"❌ Failed to upload activity log: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error uploading activity log: {e}")
            return False
    
    def cleanup_all_artifacts(self):
        """Clean up all local artifacts except seen titles after successful workflow completion"""
        try:
            print(f"\n{'='*80}")
            print("🧹 CLEANING UP LOCAL ARTIFACTS (PRESERVING SEEN TITLES)")
            print(f"{'='*80}")
            
            if not self.artifacts_dir.exists():
                print(f"📁 Artifacts directory doesn't exist: {self.artifacts_dir}")
                return True
            
            # Preserve seen-pdfs.txt file
            seen_pdfs_file = self.artifacts_dir / "seen-pdfs.txt"
            seen_pdfs_backup = None
            
            if seen_pdfs_file.exists():
                # Create backup of seen titles
                seen_pdfs_backup = seen_pdfs_file.read_text(encoding='utf-8')
                print(f"💾 Backing up seen titles: {len(seen_pdfs_backup.splitlines())} titles")
            
            # Count files before cleanup
            total_files = 0
            total_dirs = 0
            
            for item in self.artifacts_dir.rglob("*"):
                if item.is_file():
                    total_files += 1
                elif item.is_dir():
                    total_dirs += 1
            
            print(f"📊 Found {total_files} files and {total_dirs} directories to clean up")
            
            # Remove entire artifacts directory
            shutil.rmtree(self.artifacts_dir)
            print(f"✅ Deleted entire artifacts directory: {self.artifacts_dir}")
            
            # Recreate empty artifacts directory for next run
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created fresh artifacts directory for next workflow")
            
            # Restore seen titles if they existed
            if seen_pdfs_backup:
                seen_pdfs_file.write_text(seen_pdfs_backup, encoding='utf-8')
                titles_count = len(seen_pdfs_backup.splitlines())
                print(f"✅ Restored seen titles: {titles_count} titles preserved")
                self.log_activity("cleanup", f"Preserved {titles_count} seen titles during cleanup", "success")
            
            self.log_activity("cleanup", f"Cleaned up {total_files} files and {total_dirs} directories (preserved seen titles)", "success", {
                "files_deleted": total_files,
                "directories_deleted": total_dirs,
                "seen_titles_preserved": len(seen_pdfs_backup.splitlines()) if seen_pdfs_backup else 0,
                "artifacts_path": str(self.artifacts_dir)
            })
            
            print(f"🎉 Cleanup completed successfully! Seen titles preserved for next run.")
            print(f"{'='*80}")
            return True
            
        except Exception as e:
            print(f"❌ Error cleaning up artifacts: {e}")
            self.log_activity("cleanup", f"Failed to clean up artifacts: {str(e)}", "error")
            return False
    
    def process_all_projects(self):
        """Process all projects in the artifacts/projects directory"""
        try:
            projects_dir = self.artifacts_dir / "projects"
            
            if not projects_dir.exists():
                print(f"📁 No projects directory found at {projects_dir}")
                return True
            
            print(f"\n{'='*80}")
            print("🚀 PROCESSING PROJECT REPOSITORIES")
            print(f"{'='*80}")
            
            processed_count = 0
            failed_count = 0
            
            # Process each project directory
            for project_path in projects_dir.iterdir():
                if project_path.is_dir() and not project_path.name.startswith('.'):
                    project_name = project_path.name
                    print(f"\n📦 Processing project: {project_name}")
                    
                    # Create repository for this project
                    repo_name = self.create_project_repository(project_name, project_path)
                    
                    if repo_name:
                        # Upload project files to the repository
                        if self.upload_project_to_repository(project_path, repo_name):
                            processed_count += 1
                            print(f"✅ Successfully processed {project_name} -> {repo_name}")
                            
                            # Clean up local project directory after successful push
                            self.cleanup_local_project(project_path, project_name)
                        else:
                            failed_count += 1
                            print(f"❌ Failed to upload files for {project_name}")
                    else:
                        failed_count += 1
                        print(f"❌ Failed to create repository for {project_name}")
                    
                    # Longer delay between projects to avoid rate limiting
                    time.sleep(2)
            
            print(f"\n📊 Project processing complete:")
            print(f"  ✅ Successfully processed: {processed_count}")
            print(f"  ❌ Failed: {failed_count}")
            print(f"{'='*80}")
            
            return failed_count == 0
            
        except Exception as e:
            print(f"❌ Error processing projects: {e}")
            return False
    
    def print_status(self):
        """Print current status and statistics"""
        print(f"\n{'='*60}")
        print("GITHUB REPOSITORY STATUS")
        print(f"{'='*60}")
        
        stats = self.get_repository_stats()
        if stats:
            print(f"📁 Repository: {self.github_username}/{self.repo_name}")
            print(f"📊 Size: {stats['size']} KB")
            print(f"📅 Created: {stats['created_at']}")
            print(f"🔄 Updated: {stats['updated_at']}")
            print(f"🌿 Branch: {stats['default_branch']}")
        
        print(f"📝 Seen Titles: {len(self.seen_titles)}")
        print(f"📂 Local Artifacts: {self.artifacts_dir}")
        print(f"🔑 GitHub Token: {'✅ Configured' if self.github_token else '❌ Missing'}")
        
        # Check for projects
        projects_dir = self.artifacts_dir / "projects"
        if projects_dir.exists():
            project_count = len([p for p in projects_dir.iterdir() if p.is_dir() and not p.name.startswith('.')])
            print(f"📦 Projects found: {project_count}")
        else:
            print(f"📦 Projects found: 0")
        
        print(f"{'='*60}")


def main():
    """Main execution function for testing"""
    try:
        # Initialize GitHub manager
        github_manager = GitHubRepositoryManager()
        
        # Print status
        github_manager.print_status()
        
        # Test workflow
        print("\n🧪 Testing workflow...")
        
        # Simulate workflow start
        github_manager.workflow_start()
        
        # Simulate adding new titles
        test_titles = [
            "Test Paper 1: Advanced AI Methods",
            "Test Paper 2: Machine Learning Optimization"
        ]
        
        # Simulate workflow end
        github_manager.workflow_end(test_titles)
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in main: {e}")


if __name__ == "__main__":
    main()
