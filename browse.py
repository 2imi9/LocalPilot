"""
Web research agent for autoresearch. Uses local MolmoWeb-4B model for
visual web browsing, with lightweight API fallback for quick searches.

No API tokens needed. Runs entirely on local hardware.

Usage:
    VISUAL BROWSING (local MolmoWeb model):
    uv run browse.py browse "go to arxiv.org and search for transformer optimization"
    uv run browse.py browse "find the abstract of paper 2502.18845 on arxiv"

    QUICK SEARCH (API fallback, no GPU needed):
    uv run browse.py search "query"          # Search arxiv + Semantic Scholar
    uv run browse.py arxiv "query"           # Search arxiv only
    uv run browse.py fetch "url"             # Fetch and extract text from URL
    uv run browse.py ideas "topic"           # Search and list actionable ideas
"""

import sys
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote_plus

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
from bs4 import BeautifulSoup
from jinja2 import Template

HEADERS = {"User-Agent": "autoresearch/0.1 (autonomous ML research agent)"}
TIMEOUT = 15
MAX_BROWSE_STEPS = 15
SCREENSHOT_DIR = Path("screenshots")
VIEWPORT = {"width": 1280, "height": 800}

# Local model path
LOCAL_MODEL_DIR = Path(__file__).parent / "models" / "MolmoWeb-4B"
MODEL_ID = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else "allenai/MolmoWeb-4B"

# Official MolmoWeb prompt template (from model card)
MOLMOWEB_THINK_TEMPLATE = Template(
"""
# GOAL
{{ task_description }}

# PREVIOUS STEPS
{% for action in past_actions -%}
## Step {{ action['index'] }}
THOUGHT: {{ action['thought'] }}
ACTION: {{ action['action'] }}
{% endfor %}
# CURRENTLY ACTIVE PAGE
Page {{ page_index }}: {{ page_title }} | {{ page_url }}

# NEXT STEP

"""
)


# ===========================================================================
# MolmoWeb Visual Browser Agent (local model, no API tokens)
# ===========================================================================

class MolmoWebAgent:
    """Local MolmoWeb-4B agent for visual web browsing.

    Loads the model on GPU, takes screenshots, predicts actions.
    Designed to run BETWEEN training experiments when GPU is free.
    Requires transformers==4.57.6 (model is incompatible with transformers>=5).
    """

    def __init__(self, model_name: str = MODEL_ID):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.browser = None
        self.page = None
        self.action_history = []  # list of {index, thought, action}

    def load_model(self):
        """Load MolmoWeb model onto GPU."""
        if self.model is not None:
            return
        print(f"[loading {self.model_name}...]", file=sys.stderr)
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        self.model.eval()
        print(f"[model loaded on {next(self.model.parameters()).device}]", file=sys.stderr)

    def unload_model(self):
        """Free GPU memory for training."""
        if self.model is None:
            return
        import torch
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("[model unloaded, GPU freed]", file=sys.stderr)

    def start_browser(self):
        """Launch headless browser via Playwright."""
        if self.page is not None:
            return
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=True)
        self.context = self.browser.new_context(
            viewport=VIEWPORT,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        self.page = self.context.new_page()
        SCREENSHOT_DIR.mkdir(exist_ok=True)

    def stop_browser(self):
        """Close browser."""
        if self.browser:
            self.browser.close()
            self._pw.stop()
            self.browser = None
            self.page = None

    def take_screenshot(self, step: int = 0) -> "Image":
        """Capture current page as PIL Image."""
        from PIL import Image
        import io
        path = SCREENSHOT_DIR / f"step_{step:03d}.png"
        self.page.screenshot(path=str(path))
        return Image.open(path).convert("RGB")

    def predict_action(self, task: str, screenshot: "Image", step: int = 0) -> dict:
        """Run MolmoWeb inference: screenshot + task → thought + action."""
        import torch

        # Build prompt using official MolmoWeb template
        user_message = MOLMOWEB_THINK_TEMPLATE.render(
            task_description=task,
            past_actions=self.action_history[-10:],
            page_index=0,
            page_title=self.page.title() or "",
            page_url=self.page.url,
        )
        prompt = f"molmo_web_think: {user_message}"

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": screenshot},
            ]
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        n_input_tokens = inputs["input_ids"].size(1)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model.generate(**inputs, max_new_tokens=200)

        generated_tokens = outputs[0, n_input_tokens:]
        result = self.processor.decode(generated_tokens, skip_special_tokens=True).strip()
        print(f"  [RAW]: {result[:400]!r}", file=sys.stderr)
        return self._parse_response(result, step)

    def _parse_response(self, text: str, step: int = 0) -> dict:
        """Parse MolmoWeb output (official format):
        {"thought": "...", "action": {"name": "goto", "url": "..."}}
        """
        import json
        thought = ""
        action = None

        # Try parsing whole text as JSON first
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict) and "action" in obj:
                thought = obj.get("thought", "")
                action = obj["action"]
                return {"thought": thought, "action": action, "raw": text}
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to find a JSON object within the text
        json_match = re.search(r'\{.*\}', text, re.S)
        if json_match:
            try:
                obj = json.loads(json_match.group(0))
                if isinstance(obj, dict) and "action" in obj:
                    thought = obj.get("thought", "")
                    action = obj["action"]
                    return {"thought": thought, "action": action, "raw": text}
            except (json.JSONDecodeError, ValueError):
                pass

        # The model completed a partial JSON: extract "name" from '...{"name": "goto", "url": "..."}'
        # Try to parse the action sub-object directly
        action_match = re.search(r'"action"\s*:\s*(\{[^}]+\})', text, re.S)
        if action_match:
            try:
                action = json.loads(action_match.group(1))
                return {"thought": "", "action": action, "raw": text}
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: extract name + key=value pairs from the partial action string
        name_match = re.search(r'"name"\s*:\s*"(\w+)"', text)
        if name_match:
            action = {"name": name_match.group(1)}
            # Extract other string fields
            for key in ("url", "text", "key", "msg", "nav_type"):
                m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
                if m:
                    action[key] = m.group(1)
            for key in ("x", "y", "delta_x", "delta_y"):
                m = re.search(rf'"{key}"\s*:\s*([\d.]+)', text)
                if m:
                    action[key] = float(m.group(1))
            return {"thought": "", "action": action, "raw": text}

        return {"thought": thought, "action": None, "raw": text}

    def execute_action(self, action: dict | None) -> bool:
        """Execute a predicted browser action dict. Returns False if task is done."""
        if not isinstance(action, dict):
            print(f"  [unparseable action: {action}]")
            return True

        action_type = action.get("name", "")

        try:
            if action_type == "goto":
                url = action.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                self.page.goto(url, wait_until="domcontentloaded", timeout=15000)

            elif action_type in ("mouse_click", "click", "dblclick"):
                px = float(action.get("x", 0)) / 100 * VIEWPORT["width"]
                py = float(action.get("y", 0)) / 100 * VIEWPORT["height"]
                click_type = "double" if action_type == "dblclick" else "single"
                if click_type == "double":
                    self.page.mouse.dblclick(px, py)
                else:
                    button = action.get("button", "left")
                    self.page.mouse.click(px, py, button=button)

            elif action_type in ("keyboard_type", "type"):
                self.page.keyboard.type(action.get("text", ""))

            elif action_type in ("keyboard_press", "keypress"):
                self.page.keyboard.press(action.get("key", ""))

            elif action_type == "scroll":
                dx = float(action.get("delta_x", 0)) / 100 * VIEWPORT["width"]
                dy = float(action.get("delta_y", 0)) / 100 * VIEWPORT["height"]
                self.page.mouse.wheel(dx, dy)

            elif action_type == "scroll_at":
                px = float(action.get("x", 0)) / 100 * VIEWPORT["width"]
                py = float(action.get("y", 0)) / 100 * VIEWPORT["height"]
                self.page.mouse.move(px, py)
                dx = float(action.get("delta_x", 0)) / 100 * VIEWPORT["width"]
                dy = float(action.get("delta_y", 0)) / 100 * VIEWPORT["height"]
                self.page.mouse.wheel(dx, dy)

            elif action_type == "hover_at":
                px = float(action.get("x", 0)) / 100 * VIEWPORT["width"]
                py = float(action.get("y", 0)) / 100 * VIEWPORT["height"]
                self.page.mouse.move(px, py)

            elif action_type == "browser_nav":
                nav = action.get("nav_type", "go_back")
                if nav == "go_back":
                    self.page.go_back()
                elif nav == "go_forward":
                    self.page.go_forward()
                elif nav == "new_tab":
                    self.page = self.context.new_page()

            elif action_type == "noop":
                time.sleep(1)

            elif action_type == "send_msg_to_user":
                msg = action.get("msg", "")
                print(f"  [AGENT]: {msg}")
                if "[EXIT]" in msg.upper():
                    return False  # Task complete

            elif action_type == "report_infeasible":
                print(f"  [INFEASIBLE]: {action.get('infeasibility_reason', '')}")
                return False

            else:
                print(f"  [unknown action: {action_type}]")

            time.sleep(0.5)

        except Exception as e:
            print(f"  [action error: {e}]", file=sys.stderr)

        return True

    def browse(self, task: str, max_steps: int = MAX_BROWSE_STEPS) -> str:
        """Execute a full browsing task. Returns collected findings."""
        self.load_model()
        self.start_browser()
        self.action_history = []
        findings = []

        # Start on a neutral page so the model sees something real
        try:
            self.page.goto("about:blank", wait_until="domcontentloaded", timeout=5000)
        except Exception:
            pass

        print(f"=== Browsing: {task} ===")

        for step in range(max_steps):
            screenshot = self.take_screenshot(step)
            result = self.predict_action(task, screenshot, step)

            thought = result["thought"]
            action = result["action"]

            print(f"\n  Step {step + 1}:")
            if thought:
                print(f"  Thought: {thought[:200]}")
            print(f"  Action:  {action}")

            import json as _json
            action_str = _json.dumps(action) if isinstance(action, dict) else str(action)
            self.action_history.append({"index": step + 1, "thought": thought, "action": action_str})

            if not self.execute_action(action):
                # Agent signaled task complete
                break

            # Capture any visible text for research findings
            if step > 0 and step % 3 == 0:
                try:
                    page_text = self.page.inner_text("body")[:2000]
                    findings.append(page_text)
                except Exception:
                    pass

        # Final screenshot and text capture
        self.take_screenshot(step + 1)
        try:
            final_text = self.page.inner_text("body")[:3000]
            findings.append(final_text)
        except Exception:
            pass

        self.stop_browser()
        self.unload_model()

        return "\n---\n".join(findings)


# ===========================================================================
# Lightweight API Search (no GPU, fallback for quick queries)
# ===========================================================================

def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    search = f"all:{query} AND (cat:cs.LG OR cat:cs.CL OR cat:cs.AI OR cat:stat.ML)"
    url = f"http://export.arxiv.org/api/query?search_query={quote_plus(search)}&start=0&max_results={max_results}&sortBy=relevance"
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        link = entry.find("atom:id", ns).text.strip()
        published = entry.find("atom:published", ns).text.strip()[:10]
        results.append({"title": title, "abstract": abstract[:500], "url": link, "date": published})
    return results


def scholar_search(query: str, max_results: int = 5) -> list[dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": max_results, "fields": "title,abstract,url,year,citationCount"}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    if resp.status_code == 429:
        time.sleep(3)
        resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    if resp.status_code == 429:
        print("[scholar: rate limited, skipping]", file=sys.stderr)
        return []
    resp.raise_for_status()
    data = resp.json()
    results = []
    for paper in data.get("data", []):
        abstract = (paper.get("abstract") or "")[:500]
        results.append({
            "title": paper.get("title", ""),
            "abstract": abstract,
            "url": paper.get("url", ""),
            "year": paper.get("year"),
            "citations": paper.get("citationCount", 0),
        })
    return results


def fetch_page_text(url: str, max_chars: int = 3000) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars]


def combined_search(query: str, max_results: int = 5) -> list[dict]:
    results = []
    try:
        results.extend(arxiv_search(query, max_results))
    except Exception as e:
        print(f"[arxiv error: {e}]", file=sys.stderr)
    try:
        results.extend(scholar_search(query, max_results))
    except Exception as e:
        print(f"[scholar error: {e}]", file=sys.stderr)
    seen = set()
    unique = []
    for r in results:
        key = r["title"].lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def extract_ideas(query: str) -> str:
    papers = combined_search(query, max_results=8)
    if not papers:
        return "No papers found."
    lines = [f"=== Research ideas for: {query} ===\n"]
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. [{p.get('date') or p.get('year', '?')}] {p['title']}")
        if p["abstract"]:
            lines.append(f"   {p['abstract'][:300]}")
        lines.append(f"   {p['url']}")
        lines.append("")
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================

def print_papers(papers: list[dict]):
    for i, p in enumerate(papers, 1):
        date = p.get("date") or str(p.get("year", "?"))
        cites = f" [{p['citations']} cites]" if "citations" in p else ""
        print(f"{i}. [{date}]{cites} {p['title']}")
        if p["abstract"]:
            print(f"   {p['abstract'][:200]}")
        print(f"   {p['url']}")
        print()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    arg = " ".join(sys.argv[2:])

    if cmd == "browse":
        # Full visual browsing with local MolmoWeb model
        agent = MolmoWebAgent()
        findings = agent.browse(arg)
        print("\n=== Findings ===")
        print(findings[:3000])
    elif cmd == "search":
        papers = combined_search(arg)
        print_papers(papers)
    elif cmd == "arxiv":
        papers = arxiv_search(arg)
        print_papers(papers)
    elif cmd == "scholar":
        papers = scholar_search(arg)
        print_papers(papers)
    elif cmd == "fetch":
        text = fetch_page_text(arg)
        print(text)
    elif cmd == "ideas":
        print(extract_ideas(arg))
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
