import json
from pathlib import Path
from typing import List

def load_prompts(images_dir: Path) -> List[str]:
    meta = images_dir.parent / "prompts.json"
    if meta.exists():
        with meta.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return ["a medieval castle on a floating island"]


