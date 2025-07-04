import os
import json
import requests
import re
import subprocess
from dotenv import load_dotenv

# === ENV SETUP ===
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing from the .env file.")

MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"
FALLBACK_MODEL = "google/gemini-2.0-flash-exp:free"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# === DIRECTORIES ===
PDF_DIR = "relevant_pdfs"
JSON_DIR = "relevant_json"
RAW_DIR = "relevant_raw"
WORKSPACE_DIR = "workspace"
LOG_DIR = "logs"

# === Ensure folders exist ===
for d in [PDF_DIR, JSON_DIR, RAW_DIR, WORKSPACE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# === UTILITIES ===

def sanitize_filename(name):
    name = name.replace('\n', ' ').replace('\r', '')
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_")
    name = re.sub(r'__+', '_', name)
    return name.strip('_')[:100]

def build_prompt(title, summary):
    return f"""You are CoderGPT, an expert AI programmer that reads research paper summaries and implements the core ideas in clean, self-contained Python code.

**Paper Title:** {title}

**Summary of Paper:** {summary}

Based on the title and summary, write a Python script that implements the core method or concept described. The code should be clear, well-commented, and runnable.

**Instructions:**
1. Focus on the main algorithm or architecture.
2. Use standard libraries like numpy or torch if necessary.
3. Return ONLY the Python code, wrapped in ```python ... ```. Do not add any explanation before or after the code block.
"""

def query_openrouter_model(model, prompt):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    response = requests.post(ENDPOINT, headers=HEADERS, json=body, timeout=60)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()

def generate_code(title, summary):
    prompt = build_prompt(title, summary)
    try:
        code = query_openrouter_model(MODEL, prompt)
        if not code.strip() or len(code.strip()) < 20:
            raise ValueError("Model returned empty or invalid code.")
        return code
    except Exception as e:
        print(f"[⚠️ Primary Model Failed] {e}")
        try:
            print("[🔁 Retrying with fallback model...]")
            code = query_openrouter_model(FALLBACK_MODEL, prompt)
            if not code.strip() or len(code.strip()) < 20:
                raise ValueError("Fallback model returned empty code.")
            return code
        except Exception as e2:
            print(f"[❌ Fallback Model Failed] {e2}")
            return f"# ERROR: Both models failed.\n# Primary: {e}\n# Fallback: {e2}"

def save_code(project_name, code):
    if not code.strip() or len(code.strip()) < 20:
        print(f"[❌ Skipping Save] Code too short for {project_name}")
        return False
    path = os.path.join(WORKSPACE_DIR, project_name)
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, "main.py")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[✅ Code Saved] → {filepath}")
    return True

def log_result(project_name, success, details=""):
    log_path = os.path.join(LOG_DIR, f"{project_name}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Project: {project_name}\nSuccess: {success}\n\nDetails:\n{details}")
    print(f"[📄 Log Written] → {log_path}")

def find_pdf(safe_title):
    expected = f"{safe_title}.pdf"
    for f in os.listdir(PDF_DIR):
        if f.lower() == expected.lower():
            return os.path.join(PDF_DIR, f)
    return None

def download_pdf(pdf_url, safe_title):
    pdf_path = os.path.join(PDF_DIR, f"{safe_title}.pdf")
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        print(f"[⬇️ PDF Downloaded] → {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"[❌ PDF Download Failed] {e}")
        return None

# === MAIN LOGIC ===

def process_entry(title, summary, pdf_url):
    safe_title = sanitize_filename(title)
    pdf_path = find_pdf(safe_title)
    if not pdf_path and pdf_url:
        print(f"[📎 Missing PDF] → Attempting download for: {title}")
        pdf_path = download_pdf(pdf_url, safe_title)
        if not pdf_path:
            return

    print(f"\n🚧 Generating Code for: {title}")
    code = generate_code(title, summary)

    if code.startswith("# ERROR"):
        log_result(safe_title, False, code)
    else:
        if save_code(safe_title, code):
            log_result(safe_title, True, "Code generated successfully.")

def run_json_entries():
    for file in os.listdir(JSON_DIR):
        if not file.endswith(".json"): continue
        try:
            with open(os.path.join(JSON_DIR, file), "r", encoding="utf-8") as f:
                data = json.load(f)
            title = data.get("title", file.replace(".json", ""))
            summary = data.get("summary_and_goal", "No summary.")
            pdf_url = data.get("paper_url", "")
            process_entry(title, summary, pdf_url)
        except Exception as e:
            print(f"[❌ JSON Error] {file}: {e}")

def run_raw_entries():
    for file in os.listdir(RAW_DIR):
        if not file.endswith(".txt"): continue
        try:
            with open(os.path.join(RAW_DIR, file), "r", encoding="utf-8") as f:
                text = f.read()
            title = re.search(r'"?title"?\s*:\s*"([^"]+)"', text)
            summary = re.search(r'"?(summary_and_goal|summary)"?\s*:\s*"([^"]+)"', text)
            url = re.search(r'"?paper_url"?\s*:\s*"([^"]+)"', text)
            title = title.group(1) if title else file.replace(".txt", "")
            summary = summary.group(2) if summary else "Summary not available."
            pdf_url = url.group(1) if url else ""
            process_entry(title, summary, pdf_url)
        except Exception as e:
            print(f"[❌ RAW Parse Error] {file}: {e}")

def push_to_memory_repo():
    print("\n📤 Pushing code to WATCHDOG_memory...")
    subprocess.run(["git", "config", "--global", "user.name", "coder-agent"])
    subprocess.run(["git", "config", "--global", "user.email", "coder@openai.com"])
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "🤖 Generated code from latest research"])
    subprocess.run(["git", "remote", "set-url", "origin",
        f"https://x-access-token:{os.getenv('GITHUB_TOKEN')}@github.com/Sid7on1/WATCHDOG_memory.git"
    ])
    subprocess.run(["git", "push", "origin", "main"])
    print("✅ Code pushed to WATCHDOG_memory.")

# === RUN ===
if __name__ == "__main__":
    run_json_entries()
    run_raw_entries()
    push_to_memory_repo()
