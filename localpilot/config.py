"""
LocalPilot — Hardware-Aware Model Configuration
================================================
Auto-detects GPU VRAM and recommends appropriate models for:
  - Web Agent  : MolmoWeb (visual arXiv browsing)
  - Code Agent : Devstral / Qwen-Coder (experiment script generation)

Override any setting via environment variables or localpilot.yaml (project root).

Usage:
    from localpilot.config import LocalPilotConfig
    cfg = LocalPilotConfig()
    cfg.print_summary()

    # CLI:
    python -m localpilot.config --show
    python -m localpilot.config --models
    python -m localpilot.config --download-code-agent
"""

from __future__ import annotations
import os, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Project root: localpilot/config.py → localpilot/ → project root
ROOT = Path(__file__).resolve().parent.parent

# ── Model catalog ────────────────────────────────────────────────────────────

WEB_AGENT_MODELS = {
    "MolmoWeb-4B": {
        "hf_id": "allenai/MolmoWeb-4B",
        "vram_gb": 8,
        "description": "4B visual web agent — fits most GPUs (≥8 GB VRAM)",
        "min_vram_gb": 8,
    },
    "MolmoWeb-8B": {
        "hf_id": "allenai/MolmoWeb-8B",
        "vram_gb": 18,
        "description": "8B visual web agent — state-of-the-art, Qwen3+SigLIP2 (≥18 GB VRAM)",
        "min_vram_gb": 18,
        "arxiv": "2601.10611",
    },
}

CODE_AGENT_MODELS = {
    # ── Devstral variants (best for SWE tasks) ──────────────────────────────
    "Devstral-24B-Q8": {
        "hf_repo": "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
        "filename": "Devstral-Small-2-24B-Instruct-2512-Q8_0.gguf",
        "quant": "Q8_0",
        "vram_gb": 25.1,
        "min_vram_gb": 25,
        "description": "Devstral 24B Q8 — maximum quality, needs 25 GB VRAM",
        "swe_bench": 68.0,
    },
    "Devstral-24B-Q6": {
        "hf_repo": "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
        "filename": "Devstral-Small-2-24B-Instruct-2512-Q6_K.gguf",
        "quant": "Q6_K",
        "vram_gb": 19.3,
        "min_vram_gb": 20,
        "description": "Devstral 24B Q6_K — high quality, needs 20 GB VRAM",
        "swe_bench": 67.5,
    },
    "Devstral-24B-Q4": {
        "hf_repo": "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
        "filename": "Devstral-Small-2-24B-Instruct-2512-Q4_K_S.gguf",
        "quant": "Q4_K_S",
        "vram_gb": 13.5,
        "min_vram_gb": 14,
        "description": "Devstral 24B Q4_K_S — good quality, needs 14 GB VRAM",
        "swe_bench": 66.0,
    },
    # ── Qwen-Coder (smaller, great for tight VRAM) ──────────────────────────
    "Qwen-Coder-14B-Q6": {
        "hf_repo": "unsloth/Qwen2.5-Coder-14B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-14B-Instruct-Q6_K.gguf",
        "quant": "Q6_K",
        "vram_gb": 11.5,
        "min_vram_gb": 12,
        "description": "Qwen2.5-Coder 14B Q6 — solid coder, needs 12 GB VRAM",
        "swe_bench": 37.0,
    },
    "Qwen-Coder-14B-Q4": {
        "hf_repo": "unsloth/Qwen2.5-Coder-14B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "vram_gb": 8.7,
        "min_vram_gb": 9,
        "description": "Qwen2.5-Coder 14B Q4 — compact, needs 9 GB VRAM",
        "swe_bench": 36.0,
    },
    "Qwen-Coder-7B-Q4": {
        "hf_repo": "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "vram_gb": 4.4,
        "min_vram_gb": 5,
        "description": "Qwen2.5-Coder 7B Q4 — lightweight, needs 5 GB VRAM",
        "swe_bench": 33.0,
    },
    # ── CPU fallback ────────────────────────────────────────────────────────
    "Qwen-Coder-7B-CPU": {
        "hf_repo": "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "vram_gb": 0,
        "min_vram_gb": 0,
        "description": "Qwen2.5-Coder 7B Q4 — CPU only (slow but works anywhere)",
        "swe_bench": 33.0,
    },
}

# ── Hardware detection ────────────────────────────────────────────────────────

def detect_vram_gb() -> float:
    """Return total VRAM in GB for the primary GPU, or 0 if no GPU."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()
        return round(int(out[0]) / 1024, 1)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    except Exception:
        pass
    return 0.0


def detect_gpu_name() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()
        return out[0].strip()
    except Exception:
        return "CPU / Unknown"


def recommend_web_agent(vram_gb: float) -> Optional[str]:
    """Return recommended web agent model key based on available VRAM."""
    if vram_gb >= 18:  return "MolmoWeb-8B"
    if vram_gb >= 8:   return "MolmoWeb-4B"
    return None


def recommend_code_agent(vram_gb: float) -> str:
    """Return recommended code agent model key based on available VRAM.
    Assumes web agent is NOT loaded simultaneously (sequential workflow)."""
    if vram_gb >= 25:  return "Devstral-24B-Q8"
    if vram_gb >= 20:  return "Devstral-24B-Q6"
    if vram_gb >= 14:  return "Devstral-24B-Q4"
    if vram_gb >= 12:  return "Qwen-Coder-14B-Q6"
    if vram_gb >= 9:   return "Qwen-Coder-14B-Q4"
    if vram_gb >= 5:   return "Qwen-Coder-7B-Q4"
    return "Qwen-Coder-7B-CPU"


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class LocalPilotConfig:
    # Detected hardware
    gpu_name: str = field(default_factory=detect_gpu_name)
    vram_gb: float = field(default_factory=detect_vram_gb)

    # Selected models (auto-filled from hardware, overridable)
    web_agent_key: Optional[str] = None
    code_agent_key: Optional[str] = None

    # Paths — anchored to project ROOT
    models_dir: Path = field(default_factory=lambda: ROOT / "models")
    llama_server: Path = field(default_factory=lambda: ROOT.parent / "llama.cpp" / "llama-server.exe")

    # Server settings
    llama_port: int = 8080
    llama_ctx_size: int = 8192
    llama_threads: int = 8
    llama_gpu_layers: int = 99  # 0 = CPU only

    def __post_init__(self):
        # Load yaml overrides from project root
        cfg_file = ROOT / "localpilot.yaml"
        if cfg_file.exists():
            self._load_yaml(cfg_file)

        # Load env overrides
        if os.getenv("LOCALPILOT_WEB_AGENT"):
            self.web_agent_key = os.getenv("LOCALPILOT_WEB_AGENT")
        if os.getenv("LOCALPILOT_CODE_AGENT"):
            self.code_agent_key = os.getenv("LOCALPILOT_CODE_AGENT")

        # Auto-recommend if not set
        if self.web_agent_key is None:
            self.web_agent_key = recommend_web_agent(self.vram_gb)
        if self.code_agent_key is None:
            self.code_agent_key = recommend_code_agent(self.vram_gb)

    def _load_yaml(self, path: Path):
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if "web_agent" in data:      self.web_agent_key = data["web_agent"]
            if "code_agent" in data:     self.code_agent_key = data["code_agent"]
            if "llama_port" in data:     self.llama_port = data["llama_port"]
            if "llama_ctx_size" in data: self.llama_ctx_size = data["llama_ctx_size"]
            if "gpu_layers" in data:     self.llama_gpu_layers = data["gpu_layers"]
            if "models_dir" in data:     self.models_dir = Path(data["models_dir"])
        except ImportError:
            pass  # yaml not installed — skip

    @property
    def web_agent(self) -> Optional[dict]:
        return WEB_AGENT_MODELS.get(self.web_agent_key)

    @property
    def code_agent(self) -> Optional[dict]:
        return CODE_AGENT_MODELS.get(self.code_agent_key)

    @property
    def code_agent_gguf_path(self) -> Optional[Path]:
        if self.code_agent is None:
            return None
        return self.models_dir / "code_agent" / self.code_agent["filename"]

    @property
    def web_agent_path(self) -> Path:
        if self.web_agent_key:
            return self.models_dir / self.web_agent_key.replace("/", "_")
        return self.models_dir / "MolmoWeb-4B"

    def is_web_agent_downloaded(self) -> bool:
        return self.web_agent_path.exists() and any(self.web_agent_path.iterdir())

    def is_code_agent_downloaded(self) -> bool:
        p = self.code_agent_gguf_path
        return p is not None and p.exists()

    def print_summary(self):
        print("\n" + "="*60)
        print("  LocalPilot — Hardware Configuration")
        print("="*60)
        print(f"  GPU  : {self.gpu_name}")
        print(f"  VRAM : {self.vram_gb} GB")
        print()

        wa = self.web_agent
        print(f"  Web Agent  : {self.web_agent_key or 'None (VRAM < 8 GB)'}")
        if wa:
            print(f"               {wa['description']}")
            status = "[OK] Downloaded" if self.is_web_agent_downloaded() else "[--] Not downloaded"
            print(f"               {status}")
        print()

        ca = self.code_agent
        print(f"  Code Agent : {self.code_agent_key or 'None'}")
        if ca:
            print(f"               {ca['description']}")
            print(f"               SWE-bench: {ca['swe_bench']}%")
            status = "[OK] Downloaded" if self.is_code_agent_downloaded() else "[--] Not downloaded"
            print(f"               {status}")
        print("="*60 + "\n")

    def download_code_agent(self):
        """Download the GGUF file for the selected code agent."""
        if self.code_agent is None:
            print("No code agent selected.")
            return
        ca = self.code_agent
        dest = self.code_agent_gguf_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"Already downloaded: {dest}")
            return
        print(f"Downloading {self.code_agent_key} ({ca['vram_gb']} GB) ...")
        print(f"  From: {ca['hf_repo']}")
        print(f"  File: {ca['filename']}")
        print(f"  To  : {dest}")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=ca["hf_repo"],
            filename=ca["filename"],
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
        )
        print(f"Done: {dest}")

    def download_web_agent(self):
        """Download the MolmoWeb model from HuggingFace."""
        wa = self.web_agent
        if wa is None:
            print("No web agent selected (VRAM < 8 GB).")
            return
        dest = self.web_agent_path
        if dest.exists() and any(dest.iterdir()):
            print(f"Already downloaded: {dest}")
            return
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {self.web_agent_key} ...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=wa["hf_id"], local_dir=str(dest))
        print(f"Done: {dest}")

    def show_available_models(self):
        """Print all available models with VRAM requirements."""
        print("\n  Available Web Agent Models:")
        print(f"  {'Key':<20} {'VRAM':>8}  Description")
        print(f"  {'-'*20} {'-'*8}  {'-'*40}")
        for k, v in WEB_AGENT_MODELS.items():
            tag = " <-- recommended" if k == self.web_agent_key else ""
            print(f"  {k:<20} {v['vram_gb']:>6} GB  {v['description']}{tag}")

        print("\n  Available Code Agent Models:")
        print(f"  {'Key':<22} {'VRAM':>8}  {'SWE-bench':>10}  Description")
        print(f"  {'-'*22} {'-'*8}  {'-'*10}  {'-'*40}")
        for k, v in CODE_AGENT_MODELS.items():
            tag = " <-- recommended" if k == self.code_agent_key else ""
            fits = "[Y]" if v["min_vram_gb"] <= self.vram_gb else "[ ]"
            print(f"  {fits} {k:<20} {v['vram_gb']:>6} GB  {v['swe_bench']:>9}%  {v['description']}{tag}")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LocalPilot hardware configuration")
    parser.add_argument("--show", action="store_true", help="Show current config summary")
    parser.add_argument("--models", action="store_true", help="List all available models")
    parser.add_argument("--download-code-agent", action="store_true", help="Download recommended code agent")
    parser.add_argument("--download-web-agent", action="store_true", help="Download recommended web agent")
    parser.add_argument("--web-agent", type=str, help="Override web agent key")
    parser.add_argument("--code-agent", type=str, help="Override code agent key")
    args = parser.parse_args()

    cfg = LocalPilotConfig()
    if args.web_agent:  cfg.web_agent_key = args.web_agent
    if args.code_agent: cfg.code_agent_key = args.code_agent

    if args.show or not any(vars(args).values()):
        cfg.print_summary()
    if args.models:
        cfg.show_available_models()
    if args.download_web_agent:
        cfg.download_web_agent()
    if args.download_code_agent:
        cfg.download_code_agent()
