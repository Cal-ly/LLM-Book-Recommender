import os
import sys

# Resolve and append repo root to sys.path if not already present
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from app.classes.log_helper import LogHelper
from app.classes.directory_helper import DirectoryHelper

class MainHelper:
    """
    MainHelper class to manage directory paths and logging.
    This class initializes the directory structure and logging system for the application.
    It provides methods to get paths for various directories and ensure that files exist.
    The constructor takes a base directory and a log file name as parameters.
    The base directory is the root directory of the application and is used to create paths for models, figures, logs, and embeddings.
    """
    def __init__(self, base_dir, log_file_name):
        self.base_dir = base_dir
        self.directory_helper = DirectoryHelper(base_dir)
        self.log_helper = LogHelper(base_dir, log_file_name)
        self.logger = self.log_helper.get_logger()
        self.logger.info("MainHelper initialized with base directory: %s", base_dir)

    def get_path(self, key):
        return self.directory_helper.get_path(key)

    def ensure_file_exists(self, file_path):
        self.directory_helper.ensure_file_exists(file_path)