import os
import sys

# Resolve and append repo root to sys.path if not already present
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

class DirectoryHelper:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.paths = {
            "models": os.path.join(base_dir, "models"),
            "data": os.path.join(base_dir, "data"),
            "figures": os.path.join(base_dir, "figures"),
            "logs": os.path.join(base_dir, "logs"),
            "embeddings": os.path.join(base_dir, "embeddings"),
        }
        self._ensure_directories_exist()
    
    application_root_dir = os.path.dirname(os.path.abspath(__file__))
    application_root_dir = os.path.abspath(os.path.join(application_root_dir, os.pardir))

    def get_path(self, key):
        return self.paths.get(key)

    def _ensure_directories_exist(self):
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
    
    def ensure_file_exists(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
