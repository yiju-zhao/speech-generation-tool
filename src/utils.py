"""
Utility functions for the Digital Presenter project.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables
def load_config():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / "config" / ".env"
    load_dotenv(env_path)

    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "minimax_api_key": os.getenv("MINIMAX_API_KEY"),
    }


def ensure_directory(directory):
    """Ensure a directory exists, create it if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_project_paths():
    """Get standard project paths."""
    base_dir = Path(__file__).parent.parent
    return {
        "base_dir": base_dir,
        "input_dir": base_dir / "data" / "input",
        "output_dir": base_dir / "data" / "output",
        "config_dir": base_dir / "config",
        "notebooks_dir": base_dir / "notebooks",
    }


def save_json(data, filename, directory=None):
    """Save data to a JSON file."""
    if directory:
        ensure_directory(directory)
        filepath = Path(directory) / filename
    else:
        filepath = Path(filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath


def load_json(filepath):
    """Load data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
