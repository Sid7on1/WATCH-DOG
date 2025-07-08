import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import time
import json
import re
import base64

# --- ENV SETUP ---
# Load environment variables from a .env file for secure key management.
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# --- GITHUB CONFIG ---
GITHUB_USER = "Sid7on1"
GITHUB_REPO = "WATCHDOG_memory"
GITHUB_FILE_PATH = "seen_titles.txt"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
GITHUB_BASE_URL = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents"
GITHUB_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Date-based folder for current run
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
GITHUB_DATE_FOLDER = f"runs/{CURRENT_DATE}"

# --- CONFIGURATION ---
# Primary and fallback models for analysis.
MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"
FALLBACK_MODEL = "google/gemini-2.0-flash-exp:free"

# arXiv categories and keywords to search for.
CATEGORIES = ["cs.AI", "cs.LG", "stat.ML"]
KEYWORDS = ["agentic", "transformer architecture", "vision transformer", "Agents"]

# Script behavior settings.
TARGET_PAPERS_PER_DOMAIN = 2  # Number of unseen papers to fetch per domain
MAX_SEARCH_ATTEMPTS = 50  # Maximum papers to check per domain to find unseen ones
DAYS_LIMIT = 30
DELAY_BETWEEN_REQUESTS = 30

# Directory and file names for storing output.
PDF_DIR = "relevant_pdfs"
JSON_DIR = "relevant_json"
RAW_DIR = "relevant_raw"  # For saving raw model output on JSON parse failure
TITLE_CACHE_FILE = "seen_titles.txt"

# --- DIRECTORY AND CACHE SETUP ---
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# Global variables for tracking updates
seen_titles = set()
was_updated = False
files_to_push = []  # Track files that need to be pushed to GitHub

# --- GITHUB FUNCTIONS ---

def fetch_seen_titles_from_github():
    """Fetches the seen_titles.txt file from GitHub repository."""
    try:
        print("[⬇️ GITHUB] Fetching seen_titles.txt from repository...")
        response = requests.get(GITHUB_API_URL, headers=GITHUB_HEADERS)
        response.raise_for_status()
        
        content = response.json()["content"]
        decoded_content = base64.b64decode(content).decode("utf-8")
        
        with open(TITLE_CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(decoded_content)
        
        print("[✅ GITHUB SYNCED] Local seen_titles.txt updated from GitHub")
        
    except requests.exceptions.RequestException as e:
        print(f"[⚠️ GITHUB ERROR] Failed to fetch seen_titles.txt: {e}")
        print("[ℹ️ GITHUB] Continuing with local cache if available...")
    except Exception as e:
        print(f"[⚠️ GITHUB ERROR] Unexpected error: {e}")
        print("[ℹ️ GITHUB] Continuing with local cache if available...")

def push_file_to_github(local_file_path, github_file_path):
    """Pushes a single file to the GitHub repository."""
    try:
        # Read the local file
        with open(local_file_path, "rb") as f:
            file_content = f.read()
        
        # Encode content for GitHub API
        encoded_content = base64.b64encode(file_content).decode("utf-8")
        
        # Check if file already exists (to get SHA for updates)
        file_url = f"{GITHUB_BASE_URL}/{github_file_path}"
        
        # Prepare the data for GitHub API
        commit_data = {
            "message": f"Add {github_file_path} from WATCHDOG agent run on {CURRENT_DATE}",
            "content": encoded_content
        }
        
        # Check if file exists to get SHA
        try:
            response = requests.get(file_url, headers=GITHUB_HEADERS)
            if response.status_code == 200:
                # File exists, need SHA for update
                commit_data["sha"] = response.json()["sha"]
                print(f"[🔄 GITHUB] Updating existing file: {github_file_path}")
            else:
                print(f"[📁 GITHUB] Creating new file: {github_file_path}")
        except Exception as e:
            print(f"[ℹ️ GITHUB] Creating new file (couldn't check existence): {github_file_path}")
        
        # Push the file
        put_response = requests.put(file_url, headers=GITHUB_HEADERS, json=commit_data)
        put_response.raise_for_status()
        
        print(f"[✅ GITHUB] Successfully pushed: {github_file_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"[❌ GITHUB ERROR] Failed to push {github_file_path}: {e}")
        return False
    except Exception as e:
        print(f"[❌ GITHUB ERROR] Unexpected error pushing {github_file_path}: {e}")
        return False

def push_all_files_to_github():
    """Pushes all created files to the GitHub repository under date-based folder."""
    if not files_to_push:
        print("[ℹ️ GITHUB] No files to push to repository")
        return
    
    print(f"[📤 GITHUB] Pushing {len(files_to_push)} files to repository under folder: {GITHUB_DATE_FOLDER}")
    
    # Create a README file for the date folder
    readme_content = f"""# WATCHDOG Agent Run - {CURRENT_DATE}

This folder contains the results from the WATCHDOG agent run on {CURRENT_DATE}.

## Contents:
- **PDFs/**: Downloaded research papers
- **JSON/**: Processed metadata and analysis results
- **Raw/**: Raw model outputs (for debugging)

## Run Statistics:
- Target papers per domain: {TARGET_PAPERS_PER_DOMAIN}
- Max search attempts: {MAX_SEARCH_ATTEMPTS}
- Days limit: {DAYS_LIMIT}
- Categories monitored: {', '.join(CATEGORIES)}
- Keywords monitored: {', '.join(KEYWORDS)}

Generated by WATCHDOG Agent on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save README locally and add to push queue
    readme_path = "README_temp.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Add README to the beginning of the push queue
    files_to_push.insert(0, (readme_path, f"{GITHUB_DATE_FOLDER}/README.md"))
    
    success_count = 0
    for local_path, github_path in files_to_push:
        if os.path.exists(local_path):
            # Ensure the github_path is under the date folder
            if not github_path.startswith(GITHUB_DATE_FOLDER):
                # Extract filename and put it under appropriate subfolder
                filename = os.path.basename(github_path)
                if filename.endswith('.pdf'):
                    github_path = f"{GITHUB_DATE_FOLDER}/PDFs/{filename}"
                elif filename.endswith('.json'):
                    github_path = f"{GITHUB_DATE_FOLDER}/JSON/{filename}"
                elif filename.endswith('.txt') and 'raw' in local_path.lower():
                    github_path = f"{GITHUB_DATE_FOLDER}/Raw/{filename}"
                else:
                    github_path = f"{GITHUB_DATE_FOLDER}/{filename}"
            
            if push_file_to_github(local_path, github_path):
                success_count += 1
        else:
            print(f"[⚠️ GITHUB] Local file not found: {local_path}")
    
    print(f"[📊 GITHUB] Successfully pushed {success_count}/{len(files_to_push)} files")
    
    # Clean up temporary README
    if os.path.exists(readme_path):
        os.remove(readme_path)
    
    # Clear the list after pushing
    files_to_push.clear()

def push_seen_titles_to_github():
    """Pushes the updated seen_titles.txt file to GitHub repository."""
    try:
        print("[⬆️ GITHUB] Uploading updated seen_titles.txt to GitHub...")
        
        # Read the local file content
        with open(TITLE_CACHE_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Encode content for GitHub API
        encoded_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        
        # Get current file SHA (required for updates)
        response = requests.get(GITHUB_API_URL, headers=GITHUB_HEADERS)
        response.raise_for_status()
        current_sha = response.json()["sha"]
        
        # Prepare update data
        update_data = {
            "message": f"Update seen_titles.txt from WATCHDOG agent run on {CURRENT_DATE}",
            "content": encoded_content,
            "sha": current_sha
        }
        
        # Push the update
        put_response = requests.put(GITHUB_API_URL, headers=GITHUB_HEADERS, json=update_data)
        put_response.raise_for_status()
        
        print("[✅ GITHUB SYNCED] seen_titles.txt successfully pushed to GitHub")
        
    except requests.exceptions.RequestException as e:
        print(f"[⚠️ GITHUB ERROR] Failed to push seen_titles.txt: {e}")
    except Exception as e:
        print(f"[⚠️ GITHUB ERROR] Unexpected error during push: {e}")

def track_file_for_github_push(local_path, github_path=None):
    """Adds a file to the list of files to be pushed to GitHub under the date folder."""
    global files_to_push
    
    if github_path is None:
        # Auto-generate github path based on file type and name
        filename = os.path.basename(local_path)
        if filename.endswith('.pdf'):
            github_path = f"{GITHUB_DATE_FOLDER}/PDFs/{filename}"
        elif filename.endswith('.json'):
            github_path = f"{GITHUB_DATE_FOLDER}/JSON/{filename}"
        elif filename.endswith('.txt') and 'raw' in local_path.lower():
            github_path = f"{GITHUB_DATE_FOLDER}/Raw/{filename}"
        else:
            github_path = f"{GITHUB_DATE_FOLDER}/{filename}"
    
    files_to_push.append((local_path, github_path))
    print(f"[📝 GITHUB QUEUE] Added to push queue: {github_path}")

def load_seen_titles():
    """Loads previously processed paper titles from the local cache file to avoid duplicates."""
    global seen_titles
    
    if not os.path.exists(TITLE_CACHE_FILE):
        seen_titles = set()
        return
    
    try:
        with open(TITLE_CACHE_FILE, "r", encoding="utf-8") as f:
            seen_titles = {line.strip() for line in f if line.strip()}
        print(f"[📚 CACHE] Loaded {len(seen_titles)} previously seen titles")
    except IOError as e:
        print(f"[⚠️ CACHE WARNING] Could not read cache file {TITLE_CACHE_FILE}: {e}")
        seen_titles = set()

# --- HELPER FUNCTIONS ---

def sanitize_filename(name):
    """Removes or replaces characters from a string to make it a valid filename."""
    # Remove newlines and carriage returns
    name = name.replace('\n', ' ').replace('\r', '')
    
    # Handle special transformations first
    # Replace colon with "Using" if it appears to be a title separator
    if ':' in name:
        parts = name.split(':', 1)
        if len(parts) == 2:
            name = f"{parts[0]} Using {parts[1].strip()}"
    
    # Remove other invalid filename characters
    name = re.sub(r'[\\/*?"<>|]', "", name)
    
    # Replace hyphens with spaces for better word separation
    name = name.replace('-', ' ')
    
    # Replace "sim-to-real" pattern specifically 
    name = re.sub(r'\bsim[\s\-]*to[\s\-]*real\b', 'Sim to Real', name, flags=re.IGNORECASE)
    
    # Convert to title case for better readability
    name = ' '.join(word.capitalize() for word in name.split())
    
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    
    # Remove multiple consecutive underscores
    name = re.sub(r'__+', '_', name)
    
    # Remove leading/trailing underscores and limit length
    return name.strip('_')[:100]

def try_parse_json(raw_text):
    """Attempts to parse a JSON object from a string that might contain extra text."""
    match = re.search(r"{\s*\"title\":.*}", raw_text, re.DOTALL)
    if not match:
        return None
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[⚠️ JSON PARSE ERROR] Could not parse JSON: {e}")
        return None

def fetch_arxiv_with_start(query, start_index=0):
    """Fetches paper data from the arXiv API for a given query with pagination support."""
    base_url = "http://export.arxiv.org/api/query?"
    paginated_query = f"{query}&start={start_index}"
    full_url = base_url + paginated_query
    print(f"[📡 Fetching from arXiv] {full_url}")
    try:
        with urllib.request.urlopen(full_url) as response:
            return response.read()
    except urllib.error.URLError as e:
        print(f"[❌ ARXIV ERROR] Failed to fetch from arXiv: {e}")
        return None

def ask_openrouter_for_relevance(text):
    """
    Sends a paper's abstract to OpenRouter for relevance analysis.
    Switches to a fallback model if the primary is rate-limited.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    models_to_try = [MODEL, FALLBACK_MODEL]

    for model in models_to_try:
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are DOG, an AI research assistant. Your task is to evaluate arXiv abstracts.\n"
                        "Focus on AI agent design or advanced Transformer architectures, including SLAM systems.\n\n"
                        "CRITERIA:\n"
                        "- Transformer variant, attention mechanism, routing, or efficiency improvement.\n"
                        "- Agentic systems: memory, modularity, reasoning, vision-language.\n"
                        "- Feasible implementation (e.g., PyTorch).\n"
                        "- Usefulness to autonomous agent projects.\n\n"
                        "IF RELEVANT, return JSON (no markdown):\n"
                        "{\n"
                        " \"title\": \"Exact title\",\n"
                        " \"summary_and_goal\": \"Short summary + use\",\n"
                        " \"label\": \"cs_AI\",\n"
                        " \"paper_url\": \"https://arxiv.org/pdf/xxxx.xxxxx.pdf\",\n"
                        " \"relevance_score\": 0.0 to 1.0\n"
                        "}\n"
                        "IF IRRELEVANT: Return only SKIP"
                    )
                },
                {"role": "user", "content": f"Abstract:\n{text}"}
            ],
            "provider": {
                "order": ["chutes", "targon"],
                "allow_fallbacks": True
            }
        }

        try:
            print(f"[🔍 DEBUG] Calling OpenRouter model: {model}")
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            print(f"[✅ STATUS] {response.status_code}")

            if response.status_code == 429:
                print(f"[🚫 RATE LIMITED] Model {model} is limited. Switching to fallback...")
                continue

            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            return content

        except requests.exceptions.RequestException as e:
            print(f"[❌ ERROR] Request failed for {model}: {e}")
            continue

    return None

def save_pdf(title, pdf_url):
    """Downloads and saves a PDF from a URL."""
    safe_name = sanitize_filename(title)
    pdf_path = os.path.join(PDF_DIR, f"{safe_name}.pdf")
    try:
        with requests.get(pdf_url, stream=True) as r:
            r.raise_for_status()
            with open(pdf_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"[📄 PDF Saved] {pdf_path}")
        
        # Track file for GitHub push
        track_file_for_github_push(pdf_path)
        
        return pdf_path
    except requests.exceptions.RequestException as e:
        print(f"[❌ PDF Download Failed] {e}")
        return None

def save_json(info):
    """Saves the processed information as a JSON file."""
    safe_name = sanitize_filename(info["title"])
    json_path = os.path.join(JSON_DIR, f"{safe_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"[📁 JSON Saved] {json_path}")
    
    # Track file for GitHub push
    track_file_for_github_push(json_path)

def add_title_to_cache(title):
    """Appends a new title to the cache file and the in-memory set."""
    global was_updated
    
    title = title.strip()
    if title not in seen_titles:
        seen_titles.add(title)
        with open(TITLE_CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(title + "\n")
        was_updated = True
        print(f"[📝 CACHE] Added new title to cache: {title}")

def process_entry(entry, label):
    """Processes a single paper entry from the arXiv XML feed."""
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        title = entry.find("atom:title", ns).text.strip()
        title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
        summary = entry.find("atom:summary", ns).text.strip()
        published_str = entry.find("atom:published", ns).text
        pub_date = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ")
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        if not pdf_url:
            return False
    except AttributeError as e:
        print(f"[⚠️ XML PARSE WARNING] Could not parse entry, skipping: {e}")
        return False

    if title in seen_titles:
        print(f"[⏭️ Already Seen] Skipping: {title}")
        return False
    
    if datetime.now() - pub_date > timedelta(days=DAYS_LIMIT):
        print(f"[⏭️ Too Old] Skipping paper older than {DAYS_LIMIT} days: {title}")
        return False

    print(f"\n✅ Processing: {title}")
    save_pdf(title, pdf_url)
    # Cache title immediately after download to prevent re-processing.
    add_title_to_cache(title)

    try:
        response_text = ask_openrouter_for_relevance(summary)

        if response_text is None:
            print("[⏭️ Skipping analysis] Could not get response from OpenRouter.")
            return True  # Still counts as processed since we downloaded it
        if response_text.strip().upper() == "SKIP":
            print("[⏭️ Skipped by DOG] Model determined paper is not relevant.")
            return True  # Still counts as processed

        json_data = try_parse_json(response_text)
        if json_data:
            # Override model's title with the true XML title to ensure accuracy.
            json_data["title"] = title
            json_data["label"] = label
            json_data["paper_url"] = pdf_url
            save_json(json_data)
        else:
            print("⚠️ Model flagged paper as relevant, but failed to return proper JSON.")
            # Save raw response for manual review
            fallback_name = sanitize_filename(title)
            fallback_path = os.path.join(RAW_DIR, f"{fallback_name}.txt")
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write(response_text)
            print(f"[📄 RAW TEXT Saved to] {fallback_path}")
            
            # Track raw file for GitHub push
            track_file_for_github_push(fallback_path)

    except Exception as e:
        print(f"[⚠️ UNHANDLED ERROR] An error occurred while processing '{title}': {e}")
    finally:
        print(f"--- Waiting {DELAY_BETWEEN_REQUESTS} seconds ---")
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    return True  # Successfully processed

def fetch_unseen_papers_from_domain(query_info):
    """
    Fetches a specific number of unseen papers from a domain.
    Keeps searching until TARGET_PAPERS_PER_DOMAIN unseen papers are found.
    """
    label = query_info["label"]
    base_query = query_info["query"]
    
    print(f"\n{'='*50}")
    print(f"🎯 TARGET: {TARGET_PAPERS_PER_DOMAIN} unseen papers from domain: {label}")
    print(f"{'='*50}")
    
    papers_processed = 0
    start_index = 0
    papers_checked = 0
    
    while papers_processed < TARGET_PAPERS_PER_DOMAIN and papers_checked < MAX_SEARCH_ATTEMPTS:
        print(f"\n[🔍 SEARCH] Batch {start_index//10 + 1} for domain '{label}' (Papers processed: {papers_processed}/{TARGET_PAPERS_PER_DOMAIN})")
        
        # Fetch papers with pagination
        query_with_pagination = f"{base_query}&start={start_index}"
        xml_data = fetch_arxiv_with_start(base_query, start_index)
        
        if not xml_data:
            print(f"[❌ FETCH FAILED] Could not fetch papers for domain '{label}' at start_index {start_index}")
            break
        
        root = ET.fromstring(xml_data)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        
        if not entries:
            print(f"[📄 NO MORE PAPERS] No more papers found for domain '{label}'")
            break
        
        batch_processed = 0
        for entry in entries:
            papers_checked += 1
            
            if papers_checked > MAX_SEARCH_ATTEMPTS:
                print(f"[🛑 SEARCH LIMIT] Reached maximum search attempts ({MAX_SEARCH_ATTEMPTS}) for domain '{label}'")
                break
            
            # Process entry and check if it was actually processed (not skipped)
            if process_entry(entry, label):
                papers_processed += 1
                batch_processed += 1
                print(f"[✅ PROGRESS] Domain '{label}': {papers_processed}/{TARGET_PAPERS_PER_DOMAIN} papers processed")
                
                if papers_processed >= TARGET_PAPERS_PER_DOMAIN:
                    print(f"[🎉 DOMAIN COMPLETE] Found {TARGET_PAPERS_PER_DOMAIN} unseen papers for domain '{label}'")
                    break
        
        if batch_processed == 0:
            print(f"[⚠️ NO NEW PAPERS] No unseen papers found in this batch for domain '{label}'")
        
        # Move to next batch
        start_index += len(entries)
        
        # Small delay between batches
        if papers_processed < TARGET_PAPERS_PER_DOMAIN:
            print(f"[⏱️ BATCH DELAY] Waiting before next batch...")
            time.sleep(5)
    
    if papers_processed < TARGET_PAPERS_PER_DOMAIN:
        print(f"[⚠️ INCOMPLETE] Only found {papers_processed}/{TARGET_PAPERS_PER_DOMAIN} unseen papers for domain '{label}'")
    
    return papers_processed

def run():
    """Main function to run the arXiv fetching and processing loop."""
    global was_updated
    
    # Initialize GitHub sync and load cache
    fetch_seen_titles_from_github()
    load_seen_titles()
    
    # Build query list
    queries = []
    for cat in CATEGORIES:
        query_str = f"search_query=cat:{urllib.parse.quote(cat)}&sortBy=submittedDate&sortOrder=descending&max_results=10"
        queries.append({"query": query_str, "label": cat.replace(".", "_")})

    for keyword in KEYWORDS:
        query_str = f"search_query=all:{urllib.parse.quote(keyword)}&sortBy=submittedDate&sortOrder=descending&max_results=10"
        queries.append({"query": query_str, "label": keyword.replace(" ", "_")})

    # Process each domain to get TARGET_PAPERS_PER_DOMAIN unseen papers
    total_papers_processed = 0
    for query_info in queries:
        papers_from_domain = fetch_unseen_papers_from_domain(query_info)
        total_papers_processed += papers_from_domain
        
        print(f"\n[📊 DOMAIN SUMMARY] '{query_info['label']}': {papers_from_domain} papers processed")
    
    print(f"\n[📊 FINAL SUMMARY] Total papers processed across all domains: {total_papers_processed}")
    
    # Push all files to GitHub under the date folder
    if files_to_push:
        push_all_files_to_github()
    
    # Sync back to GitHub if there were updates
    if was_updated:
        push_seen_titles_to_github()
        print(f"[✅ SYNC COMPLETE] Updated GitHub with {len(seen_titles)} total cached titles")
    else:
        print("[ℹ️ GITHUB] No new titles added. Skipping GitHub push.")

if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set. Please create a .env file.")
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN environment variable not set. Please add it to your .env file.")
    
    print("🐕 WATCHDOG Agent Starting...")
    print(f"📊 Monitoring {len(CATEGORIES)} categories and {len(KEYWORDS)} keywords")
    print(f"🎯 Target: {TARGET_PAPERS_PER_DOMAIN} unseen papers per domain")
    print(f"🔍 Max search attempts per domain: {MAX_SEARCH_ATTEMPTS}")
    print(f"🔄 Syncing with GitHub repo: {GITHUB_USER}/{GITHUB_REPO}")
    print(f"📁 Files will be organized under: {GITHUB_DATE_FOLDER}")
    
    run()
    
    print("🐕 WATCHDOG Agent completed successfully!")
