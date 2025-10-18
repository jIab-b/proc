import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
from .materials import MATERIALS, AIR
try:
    # Shared DSL parser if available
    from dsl.python import parse_dsl as _parse_dsl_shared  # type: ignore
except Exception:
    _parse_dsl_shared = None

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


def actions_to_logits(actions: List[Dict[str, Any]], grid_xyz: Tuple[int, int, int], num_mats: int, device: str, logit_solid: float = 6.0) -> torch.Tensor:
    X, Y, Z = grid_xyz
    W = torch.zeros((X, Y, Z, num_mats), device=device, dtype=torch.float32)
    for a in actions:
        if a.get("type") != "place_block":
            continue
        p = a.get("params", {}).get("position")
        bt = a.get("params", {}).get("blockType")
        if not isinstance(p, (list, tuple)) or len(p) != 3 or not isinstance(bt, str):
            continue
        x, y, z = int(p[0]), int(p[1]), int(p[2])
        if 0 <= x < X and 0 <= y < Y and 0 <= z < Z:
            try:
                m = MATERIALS.index(bt)
            except ValueError:
                continue
            W[x, y, z, m] = logit_solid
    return W


def parse_dsl_text(text: str) -> Optional[List[Dict[str, Any]]]:
    if _parse_dsl_shared is None:
        return None
    try:
        return _parse_dsl_shared(text)  # type: ignore[return-value]
    except Exception:
        return None

