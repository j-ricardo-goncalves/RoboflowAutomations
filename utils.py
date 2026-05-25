import json
import operator
import os
from functools import reduce
from glob import glob
from typing import List, Optional, TypeVar

V = TypeVar("V")


# ── kept from original ────────────────────────────────────────────────────────

def get_directory_content(directory_path: str, extension: Optional[str] = None) -> List[str]:
    wildcard = "*" if extension is None else f"*.{extension}"
    return glob(os.path.join(directory_path, wildcard))


def flatten_lists(lists: List[List[V]]) -> List[V]:
    return reduce(operator.add, lists, [])


def safe_create_path_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def dump_to_json(target_path: str, content: dict) -> None:
    safe_create_path_parent(target_path)
    with open(target_path, "w") as f:
        json.dump(content, f, indent=4)


# ── new helpers ───────────────────────────────────────────────────────────────

def get_image_paths(directory: str, supported_formats: List[str]) -> List[str]:
    """Return all image paths in a directory matching supported extensions."""
    return flatten_lists([
        get_directory_content(directory, ext)
        for ext in supported_formats
    ])


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
