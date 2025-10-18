import json
from pathlib import Path
from typing import List, Dict, Any
import torch
from .materials import MATERIALS, AIR

def grid_to_actions(W_logits: torch.Tensor) -> List[Dict[str, Any]]:
    with torch.no_grad():
        W_hard = W_logits.argmax(dim=-1)
        X, Y, Z = W_hard.shape
        actions: List[Dict[str, Any]] = []
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    m = int(W_hard[x, y, z].item())
                    if m == AIR:
                        continue
                    actions.append({
                        "type": "place_block",
                        "params": {"position": [x, y, z], "blockType": MATERIALS[m]},
                    })
    return actions

def write_map_json(path: Path, actions: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    data = {"sequence": 1, "actions": actions, "metadata": metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


