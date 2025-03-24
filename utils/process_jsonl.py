"""Process JSONL files."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def read_jsonl(file_path: Union[str, Path]) -> List[dict[str, Any]]:
    """
    Read a JSONL file and return a list of dictionaries.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries, where each dictionary represents a JSON object from each line
    """
    file_path = Path(file_path)
    data = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Write a list of dictionaries to a JSONL file.

    Args:
        data: List of dictionaries to write
        file_path: Path where the JSONL file should be saved
    """
    file_path = Path(file_path)

    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + "\n")
