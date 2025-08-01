import os
import requests
import sys
from typing import Optional, Dict, Any

class GitHubArtifactsCleaner:
    def __init__(self, token: str, owner: str, repo: str):
        """
        Initialize the GitHub API client
        
        Args:
            token: GitHub Personal Access Token
            owner: Repository owner (username or organization)
            repo: Repository name
        """
        self.token = token
        self.owner = owner
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Artifacts-Cleaner/1.0"
        }
    
    def get_directory_contents(self, path: str = "artifacts") -> Optional[list]:
        """
        Get contents of a directory in the repository
        
        Args:
            path: Directory path to check
            
        Returns:
            List of directory contents or None if directory doesn't exist
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{path}"
        
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                print(f"Directory '{path}' not found in repository")
                return None
            else:
                print(f"Error fetching directory contents: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    def delete_file(self, file_path: str, sha: str) -> bool:
        """
        Delete a single file from the repository
        
        Args:
            file_path: Path to the file
            sha: SHA hash of the file
            
        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{file_path}"
        
        data = {
            "message": f"Delete {file_path}",
            "sha": sha
        }
        
        try:
            response = requests.delete(url, headers=self.headers, json=data)
            
            if response.status_code == 200:
                print(f"✓ Deleted: {file_path}")
                return True
            else:
                print(f"✗ Failed to delete {file_path}: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {file_path}: {e}")
            return False
    
    def delete_directory_recursive(self, path: str = "artifacts") -> bool:
        """
        Recursively delete all files in a directory
        
        Args:
            path: Directory path to delete
            
        Returns:
            True if all files were deleted successfully
        """
        contents = self.get_directory_contents(path)
        
        if contents is None:
            return False
        
        if not contents:
            print(f"Directory '{path}' is already empty")
            return True
        
        success = True
        
        for item in contents:
            item_path = item['path']
            
            if item['type'] == 'file':
                # Delete file
                if not self.delete_file(item_path, item['sha']):
                    success = False
            elif item['type'] == 'dir':
                # Recursively delete subdirectory
                if not self.delete_directory_recursive(item_path):
                    success = False
        
        return success
    
    def verify_repository_access(self) -> bool:
        """
        Verify that we can access the repository
        
        Returns:
            True if repository is accessible
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}"
        
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                repo_data = response.json()
                print(f"✓ Repository access verified: {repo_data['full_name']}")
                return True
            elif response.status_code == 404:
                print("✗ Repository not found or no access")
                return False
            else:
                print(f"✗ Error accessing repository: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return False

def main():
    """
    Main function to execute the artifacts directory deletion
    """
    # Get environment variables
    github_token = os.getenv('API_GITHUB')
    
    if not github_token:
        print("Error: API_GITHUB environment variable not found")
        print("Make sure your GitHub PAT is stored in the environment variable API_GITHUB")
        sys.exit(1)
    
    # Repository configuration
    OWNER = "your-username"  # Replace with your GitHub username/organization
    REPO = "WATCHDOG_memory"
    
    print(f"Starting cleanup of artifacts directory in {OWNER}/{REPO}")
    print("=" * 50)
    
    # Initialize the cleaner
    cleaner = GitHubArtifactsCleaner(github_token, OWNER, REPO)
    
    # Verify repository access
    if not cleaner.verify_repository_access():
        sys.exit(1)
    
    # Delete artifacts directory
    print("\nStarting deletion of artifacts directory...")
    
    if cleaner.delete_directory_recursive("artifacts"):
        print("\n✓ Successfully deleted all files in artifacts directory")
    else:
        print("\n✗ Some files could not be deleted")
        sys.exit(1)
    
    print("\nCleanup completed!")

if __name__ == "__main__":
    main()
