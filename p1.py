import os
import json
import requests
import subprocess
import time
from dotenv import load_dotenv

# === ENV SETUP ===
load_dotenv()
GITHUB_USERNAME = "Sid7on1"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # not GITHUB_TOKEN
WORKSPACE_DIR = "workspace"
RELEVANT_JSON_DIR = "relevant_json"
PUSH_LOG_FILE = "pushed_repos.txt"
CREATED_REPOS_FILE = "created_repos.txt"

if not GITHUB_USERNAME or not GITHUB_TOKEN:
    raise ValueError("Missing GITHUB_USERNAME or GITHUB_TOKEN in .env")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

# === UTILS ===

def truncate_description(summary):
    """Ensure GitHub repo description is <= 350 characters"""
    return (summary[:347] + "...") if len(summary) > 350 else summary

def create_github_repo(repo_name, description):
    """Create a GitHub repo using the GitHub API."""
    url = "https://api.github.com/user/repos"
    payload = {
        "name": repo_name,
        "description": truncate_description(description),
        "private": False,
        "auto_init": False
    }

    response = requests.post(url, headers=HEADERS, json=payload)
    if response.status_code == 201:
        print(f"[✅ Created Repo] {repo_name}")
        return True
    else:
        print(f"[❌ Repo Create Failed] {repo_name}: {response.text}")
        return False

def git_push_project(project_path, repo_name):
    """Initialize Git repo (if needed), commit, and push code to GitHub."""
    os.chdir(project_path)

    # Only init git if not already a git repo
    if not os.path.exists(os.path.join(project_path, ".git")):
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "branch", "-M", "main"], check=True)
        subprocess.run([
            "git", "remote", "add", "origin",
            f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{repo_name}.git"
        ], check=True)
    else:
        # Ensure remote origin is correctly set
        subprocess.run([
            "git", "remote", "set-url", "origin",
            f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{repo_name}.git"
        ], check=True)

    # Stage, commit, and push
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit from Coder Agent", "--allow-empty"], check=True)
    subprocess.run(["git", "push", "-u", "origin", "main"], check=True)

    print(f"[🚀 Pushed] {repo_name}")
    os.chdir("../../")  # back to root

def load_pushed_repo_cache():
    """Returns a set of already-pushed repo names."""
    if not os.path.exists(PUSH_LOG_FILE):
        return set()
    with open(PUSH_LOG_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def update_push_log(repo_name):
    """Appends a pushed repo name to the log."""
    with open(PUSH_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(repo_name + "\n")

def load_created_repos():
    if not os.path.exists(CREATED_REPOS_FILE):
        return set()
    with open(CREATED_REPOS_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def add_to_created_repos(repo_name):
    with open(CREATED_REPOS_FILE, "a", encoding="utf-8") as f:
        f.write(repo_name + "\n")

def main_py_valid(project_path):
    """Checks if main.py exists and is meaningfully non-empty."""
    main_py = os.path.join(project_path, "main.py")
    
    if not os.path.exists(main_py):
        print(f"[❌ main.py MISSING] → {main_py}")
        return False

    size = os.path.getsize(main_py)
    if size < 10:
        print(f"[⚠️ main.py Too Small: {size} bytes] → {main_py}")
        return False

    # Optional: Read content and check for useful code
    with open(main_py, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content or len(content) < 10:
            print(f"[⚠️ main.py Seems Empty or Only Comments] → {main_py}")
            return False
        if all(line.strip().startswith("#") for line in content.splitlines()):
            print(f"[⚠️ main.py Only Comments] → {main_py}")
            return False

    return True
def load_projects():
    """Load projects with summary from JSON or fallback to .txt raw file."""
    projects = []

    for project_name in os.listdir(WORKSPACE_DIR):
        project_path = os.path.join(WORKSPACE_DIR, project_name)
        main_py = os.path.join(project_path, "main.py")
        json_file = os.path.join(RELEVANT_JSON_DIR, f"{project_name}.json")
        raw_txt = os.path.join("relevant_raw", f"{project_name}.txt")

        if not os.path.isdir(project_path):
            continue

        if not main_py_valid(project_path):
            print(f"[⚠️ Skipped] No valid main.py in {project_name}")
            continue

        # Try loading JSON
        title = project_name
        summary = "Code generated from research paper. Summary unavailable."

        if os.path.exists(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    title = data.get("title", title)
                    summary = data.get("summary_and_goal", summary)
            except Exception as e:
                print(f"[⚠️ JSON Read Failed] {json_file}: {e}")
                # Try raw txt fallback if JSON fails
                if os.path.exists(raw_txt):
                    try:
                        with open(raw_txt, "r", encoding="utf-8") as f:
                            raw_content = f.read()
                            summary = raw_content.strip()[:1000]  # limit length
                    except Exception as e:
                        print(f"[⚠️ RAW TXT Read Failed] {raw_txt}: {e}")
        elif os.path.exists(raw_txt):
            try:
                with open(raw_txt, "r", encoding="utf-8") as f:
                    raw_content = f.read()
                    summary = raw_content.strip()[:1000]  # limit length
            except Exception as e:
                print(f"[⚠️ RAW TXT Read Failed] {raw_txt}: {e}")
        else:
            print(f"[⚠️ No Metadata] JSON and TXT missing for {project_name}, using defaults.")

        projects.append((project_name, project_path, title, summary))

    return projects

def clean_summary(summary):
    """Clean summary for GitHub description: strip, replace newlines with spaces."""
    return summary.strip().replace('\n', ' ')

# === MAIN ===
def run():
    pushed_repos = load_pushed_repo_cache()
    created_repos = load_created_repos()
    projects = load_projects()

    for project_name, project_path, title, summary in projects:
        if project_name in pushed_repos:
            print(f"[⏭️ Already Pushed] {project_name} — skipping.")
            continue

        repo_name = project_name
        if project_name in created_repos:
            print(f"[⚠️ Skipped] Repo already created: {project_name}")
            continue

        description = clean_summary(summary)
        # --- Robust repo creation with error handling ---
        url = "https://api.github.com/user/repos"
        payload = {
            "name": repo_name,
            "description": truncate_description(description),
            "private": False,
            "auto_init": False
        }
        try:
            response = requests.post(url, headers=HEADERS, json=payload)
            response.raise_for_status()
            print(f"[✅ Created Repo] {project_name}")
            add_to_created_repos(project_name)
        except requests.exceptions.HTTPError as e:
            try:
                error_json = response.json()
            except Exception:
                error_json = {}
            if response.status_code == 422:
                errors = error_json.get("errors", [])
                name_error = any("name" in err.get("field", "") and "already exists" in err.get("message", "") for err in errors)
                if name_error:
                    print(f"[⚠️ Already Exists] {project_name} — marking as created.")
                    add_to_created_repos(project_name)
                    continue  # skip further processing
            print(f"[❌ Repo Create Failed] {project_name}: {error_json}")
            continue
        # --- End repo creation ---
        git_push_project(project_path, repo_name)
        update_push_log(project_name)
        print("[⏳ Delay] Waiting 25 seconds before next push...")
        time.sleep(25)

if __name__ == "__main__":
    run()
