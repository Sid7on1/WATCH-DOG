import os
import json
import base64
from dotenv import load_dotenv
import requests
import re

# === ENV SETUP ===
# Load environment variables from a .env file for secure key management.
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing from the .env file. Please ensure it is set.")

MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"
FALLBACK_MODEL = "google/gemini-2.0-flash-exp:free"

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# === DIRECTORIES ===
JSON_DIR = "relevant_json"
PDF_DIR = "relevant_pdfs"
WORKSPACE_DIR = "workspace"
LOG_DIR = "logs"

# Ensure all necessary directories exist.
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === UTILITIES ===

def sanitize_filename(name):
    """Removes or replaces characters from a string to make it a valid filename."""
    sanitized = name.replace('\n', ' ').replace('\r', '')
    sanitized = re.sub(r'[\\/*?:"<>|]', "", sanitized)
    sanitized = sanitized.replace(" ", "_")
    sanitized = re.sub(r'__+', '_', sanitized)
    return sanitized.strip('_')[:100]

def load_relevant_jsons():
    """Generator function to load each JSON file from the relevant_json directory."""
    if not os.path.exists(JSON_DIR):
        print(f"[⚠️ WARNING] JSON directory not found: {JSON_DIR}")
        return

    for filename in os.listdir(JSON_DIR):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(JSON_DIR, filename), "r", encoding="utf-8") as f:
                    yield json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[❌ ERROR] Failed to read or parse {filename}: {e}")

def build_prompt(title, summary):
    """Formats the prompt to send to the model."""
    return f"""You are CoderGPT, an expert AI programmer that reads research paper summaries and implements the core ideas in clean, self-contained Python code.

**Paper Title:** {title}

**Summary of Paper:** {summary}

Based on the title and summary, write a Python script that implements the core method or concept described. The code should be clear, well-commented, and runnable.

**Instructions:**
1.  Focus on the main algorithm or architecture.
2.  Use standard libraries like numpy or torch if necessary.
3.  Return ONLY the Python code, wrapped in ```python ... ```. Do not add any explanation before or after the code block.
"""

def query_openrouter_model(model, prompt):
    """Queries OpenRouter with the provided prompt using the specified model."""
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    response = requests.post(OPENROUTER_ENDPOINT, headers=HEADERS, json=body, timeout=60)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()

def generate_code(title, summary):
    """Attempts to generate code from the model, with a fallback if the first one fails."""
    prompt = build_prompt(title, summary)
    try:
        return query_openrouter_model(MODEL, prompt)
    except Exception as e:
        print(f"[⚠️ Primary Model Failed] {e}")
        try:
            print("[🔁 Retrying with fallback model...]")
            return query_openrouter_model(FALLBACK_MODEL, prompt)
        except Exception as e2:
            print(f"[❌ Fallback Model Failed] {e2}")
            return f"# ERROR: Both primary and fallback models failed.\n# Primary: {e}\n# Fallback: {e2}"

def save_code(project_name, code):
    """Saves the generated code to a Python file in a dedicated project folder."""
    path = os.path.join(WORKSPACE_DIR, project_name)
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, "main.py")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[✅ Code Saved] → {filepath}")
    return filepath

def log_result(project_name, success, details=""):
    """Logs the outcome of the code generation process."""
    log_path = os.path.join(LOG_DIR, f"{project_name}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Project: {project_name}\nSuccess: {success}\n\nDetails:\n{details}")
    print(f"[📄 Log Written] → {log_path}")

# === MAIN LOGIC ===
def run():
    """Main loop to process each paper, find its PDF, and generate corresponding code."""
    for entry in load_relevant_jsons():
        title = entry.get("title", "untitled_paper")
        summary = entry.get("summary_and_goal", "No summary available.")
        safe_title = sanitize_filename(title)

        pdf_filename = f"{sanitize_filename(title)}.pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_filename)

        if not os.path.exists(pdf_path):
            print(f"[❌ Missing PDF] Could not find '{pdf_filename}' in '{PDF_DIR}'")
            continue

        print(f"\n🚧 Generating Code for: {title}")
        
        code = generate_code(title, summary)

        if isinstance(code, str) and code.strip().startswith("# ERROR"):
            log_result(safe_title, False, code)
        else:
            save_code(safe_title, code)
            log_result(safe_title, True, "Code generated successfully.")

if __name__ == "__main__":
    run()