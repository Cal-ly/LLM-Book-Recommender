import os
import sys

# Resolve and append repo root to sys.path if not already present
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import datetime
import logging
from app.classes.directory_helper import DirectoryHelper

class LogHelper:
    def __init__(self, base_dir, file_name):
        self.base_dir = base_dir
        self.file_name = file_name
        self.logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.log_file = os.path.join(self.logs_dir, f"{file_name}.{datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S')}.log")
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(filename=self.log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def get_logger(self):
        return self.logger
