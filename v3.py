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
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_PAT")

# --- GITHUB CONFIG ---
GITHUB_USERNAME = "Sid7on1"                 # 👈 Change if needed
REPO_NAME = "WATCHDOG_memory"              # 👈 Change if needed
SEEN_FILE_PATH = "seen_titles.txt"
LOCAL_SEEN_FILE = "seen_titles.txt"

# --- CONFIGURATION ---
MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"
FALLBACK_MODEL = "google/gemini-2.0-flash-exp:free"
CATEGORIES = ["cs.AI", "cs.LG", "stat.ML"]
KEYWORDS = ["agentic", "transformer architecture", "vision transformer", "slam"]
MAX_RESULTS_PER_QUERY = 1
DAYS_LIMIT = 7
DELAY_BETWEEN_REQUESTS = 25

PDF_DIR = "relevant_pdfs"
JSON_DIR = "relevant_json"
RAW_DIR = "relevant_raw"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# ========================== GitHub Seen Titles ==========================

def get_remote_seen_titles():
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents/{SEEN_FILE_PATH}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        content = resp.json()
        decoded = base64.b64decode(content["content"]).decode("utf-8").splitlines()
        print(f"[⬇️ Pulled {len(decoded)} titles from GitHub]")
        return set(line.strip() for line in decoded if line.strip())
    else:
        print(f"[⚠️ GitHub GET Failed] Status {resp.status_code}")
        return set()

def load_seen_titles():
    remote_titles = get_remote_seen_titles()
    local_titles = set()

    if os.path.exists(LOCAL_SEEN_FILE):
        with open(LOCAL_SEEN_FILE, "r", encoding="utf-8") as f:
            local_titles = {line.strip() for line in f if line.strip()}

    all_titles = remote_titles.union(local_titles)
    with open(LOCAL_SEEN_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(all_titles)) + "\n")

    print(f"[📖 Titles Loaded] Total: {len(all_titles)}")
    return all_titles

def push_seen_titles_to_github():
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents/{SEEN_FILE_PATH}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

    with open(LOCAL_SEEN_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    b64_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    resp = requests.get(url, headers=headers)
    sha = resp.json().get("sha") if resp.status_code == 200 else None

    data = {
        "message": "Update seen_titles.txt",
        "content": b64_content,
        "branch": "main"
    }
    if sha:
        data["sha"] = sha

    resp = requests.put(url, headers=headers, json=data)
    if resp.status_code in [200, 201]:
        print("[✅ GitHub PUSH Success] seen_titles.txt updated")
    else:
        print(f"[❌ GitHub PUSH Failed] Status {resp.status_code} — {resp.text}")

# ========================== UTILS ==========================

def sanitize_filename(name):
    name = re.sub(r'[\\/*?:"<>|]', "", name.replace('\n', ' '))
    name = re.sub(r'\s+', '_', name)
    return name.strip('_')[:100]

def try_parse_json(raw_text):
    match = re.search(r"{\s*\"title\":.*}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        print(f"[⚠️ JSON PARSE ERROR] {e}")
        return None

def fetch_arxiv(query):
    full_url = f"http://export.arxiv.org/api/query?{query}"
    print(f"[📡 Fetching] {full_url}")
    try:
        with urllib.request.urlopen(full_url) as response:
            return response.read()
    except Exception as e:
        print(f"[❌ FETCH ERROR] {e}")
        return None

def ask_openrouter_for_relevance(text):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    models = [MODEL, FALLBACK_MODEL]
    for model in models:
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are DOG, an AI research assistant. Evaluate arXiv abstracts.\n"
                        "Return relevance JSON or SKIP if irrelevant.\n"
                    )
                },
                {"role": "user", "content": f"Abstract:\n{text}"}
            ],
            "provider": {"order": ["chutes", "targon"]}
        }

        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            if resp.status_code == 429:
                print("[⏳ Rate limited, retrying]")
                time.sleep(10)
                continue
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return content
        except Exception as e:
            print(f"[⚠️ Model Error] {e}")
            continue
    return None

def save_pdf(title, pdf_url):
    safe = sanitize_filename(title)
    pdf_path = os.path.join(PDF_DIR, f"{safe}.pdf")
    try:
        r = requests.get(pdf_url)
        with open(pdf_path, "wb") as f:
            f.write(r.content)
        print(f"[📄 PDF Saved] {pdf_path}")
    except Exception as e:
        print(f"[❌ PDF Download Error] {e}")

def save_json(info):
    safe = sanitize_filename(info["title"])
    path = os.path.join(JSON_DIR, f"{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"[📁 JSON Saved] {path}")

def add_title_to_cache(title):
    title = title.strip()
    with open(LOCAL_SEEN_FILE, "a", encoding="utf-8") as f:
        f.write(title + "\n")
    seen_titles.add(title)

# ========================== MAIN PROCESS ==========================

def process_entry(entry, label):
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()
        published_str = entry.find("atom:published", ns).text
        pub_date = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ")
        pdf_url = next((l.get("href") for l in entry.findall("atom:link", ns) if l.get("title") == "pdf"), None)
        if not pdf_url or title in seen_titles or (datetime.now() - pub_date).days > DAYS_LIMIT:
            return
    except Exception as e:
        print(f"[⚠️ Entry Parse Error] {e}")
        return

    print(f"\n✅ Processing: {title}")
    save_pdf(title, pdf_url)
    add_title_to_cache(title)

    response = ask_openrouter_for_relevance(summary)
    if response is None or response.strip().upper() == "SKIP":
        print("[⏭️ Skipped]")
        return

    data = try_parse_json(response)
    if data:
        data["title"] = title
        data["label"] = label
        data["paper_url"] = pdf_url
        save_json(data)
    else:
        raw_path = os.path.join(RAW_DIR, f"{sanitize_filename(title)}.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"[📄 RAW Saved] {raw_path}")

    print(f"--- Waiting {DELAY_BETWEEN_REQUESTS}s ---")
    time.sleep(DELAY_BETWEEN_REQUESTS)

def run():
    queries = [
        {"query": f"search_query=cat:{urllib.parse.quote(cat)}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_RESULTS_PER_QUERY}", "label": cat.replace(".", "_")}
        for cat in CATEGORIES
    ] + [
        {"query": f"search_query=all:{urllib.parse.quote(keyword)}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_RESULTS_PER_QUERY}", "label": keyword.replace(" ", "_")}
        for keyword in KEYWORDS
    ]

    for q in queries:
        print(f"\n=== 🔍 Query: {q['label']} ===")
        xml_data = fetch_arxiv(q["query"])
        if not xml_data:
            continue

        root = ET.fromstring(xml_data)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        for entry in entries:
            process_entry(entry, q["label"])

# ========================== ENTRYPOINT ==========================

if __name__ == "__main__":
    if not OPENROUTER_API_KEY or not GITHUB_TOKEN:
        raise EnvironmentError("OPENROUTER_API_KEY or GITHUB_PAT not set in .env")

    seen_titles = load_seen_titles()
    run()
    push_seen_titles_to_github()
