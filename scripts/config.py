
import os

# Find directories of main python files and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR: str = os.path.dirname(SCRIPT_DIR)

# Define directories of data access files
CACHE_DIR = os.path.join(ROOT_DIR, "cache_files")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Create them if they don't exist
for path in [CACHE_DIR, MODEL_DIR, DATA_DIR, LOG_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)
