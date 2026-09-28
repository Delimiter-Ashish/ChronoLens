import os
from pathlib import Path

__version__ = "0.1.0"

_project_root = Path(__file__).resolve().parents[2]
_hf_home = _project_root / ".cache" / "huggingface"
_hf_home.mkdir(parents=True, exist_ok=True)
(_hf_home / "hub").mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(_hf_home)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_hf_home / "hub")
os.environ["HF_TOKEN_PATH"] = str(_hf_home / "token")
os.environ.pop("TRANSFORMERS_CACHE", None)
