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

# --- ENV SETUP ---
# Load environment variables from a .env file for secure key management.
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- CONFIGURATION ---
# Primary and fallback models for analysis.
MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"
FALLBACK_MODEL = "google/gemini-2.0-flash-exp:free"

# arXiv categories and keywords to search for.
CATEGORIES = ["cs.AI", "cs.LG", "stat.ML"]
KEYWORDS = ["agentic", "transformer architecture", "vision transformer", "slam"]

# Script behavior settings.
MAX_RESULTS_PER_QUERY = 1
DAYS_LIMIT = 7
DELAY_BETWEEN_REQUESTS = 25

# Directory and file names for storing output.
PDF_DIR = "relevant_pdfs"
JSON_DIR = "relevant_json"
RAW_DIR = "relevant_raw" # For saving raw model output on JSON parse failure
TITLE_CACHE_FILE = "seen_titles.txt"

# --- DIRECTORY AND CACHE SETUP ---
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

def load_seen_titles():
    """Loads previously processed paper titles from a cache file to avoid duplicates."""
    if not os.path.exists(TITLE_CACHE_FILE):
        return set()
    try:
        with open(TITLE_CACHE_FILE, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except IOError as e:
        print(f"[⚠️ CACHE WARNING] Could not read cache file {TITLE_CACHE_FILE}: {e}")
        return set()

seen_titles = load_seen_titles()

# --- HELPER FUNCTIONS ---

def sanitize_filename(name):
    """Removes or replaces characters from a string to make it a valid filename."""
    sanitized = name.replace('\n', ' ').replace('\r', '')
    sanitized = re.sub(r'[\\/*?:"<>|]', "", sanitized)
    sanitized = sanitized.replace(" ", "_")
    sanitized = re.sub(r'__+', '_', sanitized)
    return sanitized.strip('_')[:100]

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

def fetch_arxiv(query):
    """Fetches paper data from the arXiv API for a given query."""
    base_url = "http://export.arxiv.org/api/query?"
    full_url = base_url + query
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
        # **FIX**: Re-added the provider object to ensure requests are routed correctly.
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
                        "  \"title\": \"Exact title\",\n"
                        "  \"summary_and_goal\": \"Short summary + use\",\n"
                        "  \"label\": \"cs_AI\",\n"
                        "  \"paper_url\": \"https://arxiv.org/pdf/xxxx.xxxxx.pdf\",\n"
                        "  \"relevance_score\": 0.0 to 1.0\n"
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

def add_title_to_cache(title):
    """Appends a new title to the cache file and the in-memory set."""
    title = title.strip()
    with open(TITLE_CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(title + "\n")
    seen_titles.add(title)

def process_entry(entry, label):
    """Processes a single paper entry from the arXiv XML feed."""
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        title = entry.find("atom:title", ns).text.strip()
        title = re.sub(r'\s+', ' ', title) # Normalize whitespace
        summary = entry.find("atom:summary", ns).text.strip()
        published_str = entry.find("atom:published", ns).text
        pub_date = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ")
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        if not pdf_url:
            return
    except AttributeError as e:
        print(f"[⚠️ XML PARSE WARNING] Could not parse entry, skipping: {e}")
        return

    if title in seen_titles:
        # No need to print here, it's just a normal skip.
        return
    if datetime.now() - pub_date > timedelta(days=DAYS_LIMIT):
        return

    print(f"\n✅ Processing: {title}")
    save_pdf(title, pdf_url)
    # **FIX**: Cache title immediately after download to prevent re-processing.
    add_title_to_cache(title)

    try:
        response_text = ask_openrouter_for_relevance(summary)

        if response_text is None:
            print("[⏭️ Skipping analysis] Could not get response from OpenRouter.")
            return
        if response_text.strip().upper() == "SKIP":
            print("[⏭️ Skipped by DOG] Model determined paper is not relevant.")
            return

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

    except Exception as e:
        print(f"[⚠️ UNHANDLED ERROR] An error occurred while processing '{title}': {e}")
    finally:
        print(f"--- Waiting {DELAY_BETWEEN_REQUESTS} seconds ---")
        time.sleep(DELAY_BETWEEN_REQUESTS)

def run():
    """Main function to run the arXiv fetching and processing loop."""
    queries = []
    for cat in CATEGORIES:
        query_str = f"search_query=cat:{urllib.parse.quote(cat)}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_RESULTS_PER_QUERY}"
        queries.append({"query": query_str, "label": cat.replace(".", "_")})

    for keyword in KEYWORDS:
        query_str = f"search_query=all:{urllib.parse.quote(keyword)}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_RESULTS_PER_QUERY}"
        queries.append({"query": query_str, "label": keyword.replace(" ", "_")})

    for q in queries:
        print(f"\n{'='*20}\n📂 Running Query for Label: {q['label']}\n{'='*20}")
        xml_data = fetch_arxiv(q["query"])
        if not xml_data:
            continue

        root = ET.fromstring(xml_data)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        if not entries:
            print("...No new papers found for this query.")
            continue

        for entry in entries:
            process_entry(entry, q["label"])

if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set. Please create a .env file.")
    run()

