"""Probabilistic voxel DSL utilities.

This module keeps the project-specific representation for Minecraft-style voxel
worlds.  It reads the existing ``map.json`` format, initialises a tensor of
logits over block materials, exposes helpers for computing differentiable
(soft) densities/colours, and exports updated maps after optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Palette definitions --------------------------------------------------------
# ---------------------------------------------------------------------------

_BLOCK_TYPES: Sequence[str] = (
    "Air",
    "Grass",
    "Dirt",
    "Stone",
    "Plank",
    "Snow",
    "Sand",
    "Water",
)

# Colours copied from the Svelte frontend palette (src/chunks.ts).  The
# entries are RGB values in the range [0, 1].
_BLOCK_COLOURS: Dict[str, Tuple[float, float, float]] = {
    "Air": (0.0, 0.0, 0.0),
    "Grass": (0.34, 0.68, 0.36),
    "Dirt": (0.42, 0.32, 0.20),
    "Stone": (0.58, 0.60, 0.64),
    "Plank": (0.78, 0.68, 0.50),
    "Snow": (0.92, 0.94, 0.96),
    "Sand": (0.88, 0.82, 0.60),
    "Water": (0.22, 0.40, 0.66),
}

# Density multipliers for each block.  These can be tuned per project but
# default to 0 for air and 100 for opaque blocks so the renderer approximates a
# hard surface when the probability mass concentrates on one material.
_BLOCK_DENSITIES: Dict[str, float] = {
    "Air": 0.0,
    "Grass": 100.0,
    "Dirt": 100.0,
    "Stone": 120.0,
    "Plank": 80.0,
    "Snow": 40.0,
    "Sand": 70.0,
    "Water": 15.0,
}

_BLOCK_NAME_TO_INDEX = {name: idx for idx, name in enumerate(_BLOCK_TYPES)}
_INDEX_TO_BLOCK_NAME = {idx: name for name, idx in _BLOCK_NAME_TO_INDEX.items()}


@dataclass(frozen=True)
class BlockPalette:
    names: Sequence[str]
    colours: torch.Tensor  # (M, 3)
    densities: torch.Tensor  # (M,)
    air_index: int = 0

    @classmethod
    def default(cls, device: torch.device | None = None) -> "BlockPalette":
        colours = torch.tensor(
            [_BLOCK_COLOURS[name] for name in _BLOCK_TYPES], dtype=torch.float32, device=device
        )
        densities = torch.tensor(
            [_BLOCK_DENSITIES[name] for name in _BLOCK_TYPES], dtype=torch.float32, device=device
        )
        return cls(names=_BLOCK_TYPES, colours=colours, densities=densities, air_index=_BLOCK_NAME_TO_INDEX["Air"])

    def name_to_index(self, name: str) -> int:
        if name not in _BLOCK_NAME_TO_INDEX:
            raise KeyError(f"Unknown block type '{name}'. Update palette if new types are required.")
        return _BLOCK_NAME_TO_INDEX[name]

    def index_to_name(self, index: int) -> str:
        return _INDEX_TO_BLOCK_NAME[index]

    @property
    def num_materials(self) -> int:
        return len(self.names)


# ---------------------------------------------------------------------------
# Probabilistic voxel grid ---------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass
class WorldMetadata:
    dimensions: Tuple[int, int, int]
    world_scale: float
    capture_id: Optional[str]


class ProbabilisticVoxelGrid:
    """Torch wrapper around the project DSL parameters.

    ``logits`` carries the optimisation variables with shape ``(X, Y, Z, M)``
    where ``M`` is the number of materials.  The grid stores references to the
    original ``map.json`` so we can round-trip edits back into that format.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        palette: BlockPalette,
        metadata: WorldMetadata,
        base_temperature: float = 1.0,
    ) -> None:
        if logits.ndim != 4:
            raise ValueError("logits must have shape (X, Y, Z, M)")
        if logits.shape[-1] != palette.num_materials:
            raise ValueError("logits last dimension must match palette materials")
        self.logits = logits
        self.palette = palette
        self.metadata = metadata
        self.base_temperature = base_temperature

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_map(
        cls,
        map_path: Path | str,
        palette: BlockPalette | None = None,
        device: torch.device | None = None,
        base_logit: float = -4.0,
        confident_logit: float = 4.0,
    ) -> "ProbabilisticVoxelGrid":
        """Construct logits from an existing ``map.json``.

        Parameters
        ----------
        map_path:
            Path to the map JSON.
        base_logit:
            Baseline logit value applied to every material.
        confident_logit:
            Additional logit boost applied to the material present in the map.
        """

        palette = palette or BlockPalette.default(device=device)
        map_path = Path(map_path)
        with map_path.open("r", encoding="utf-8") as fh:
            map_data = json.load(fh)

        dims_dict = map_data.get("worldConfig", {}).get("dimensions", {})
        dims = (
            int(dims_dict.get("x", 32)),
            int(dims_dict.get("y", 32)),
            int(dims_dict.get("z", 32)),
        )
        world_scale = float(map_data.get("worldScale", 1.0))
        capture_id = map_data.get("captureId")
        metadata = WorldMetadata(dimensions=dims, world_scale=world_scale, capture_id=capture_id)

        logits = torch.full(
            (*dims, palette.num_materials),
            fill_value=base_logit,
            dtype=torch.float32,
            device=device,
        )

        for block in map_data.get("blocks", []):
            position = block.get("position")
            if not position or len(position) != 3:
                continue
            x, y, z = (int(position[0]), int(position[1]), int(position[2]))
            if not (0 <= x < dims[0] and 0 <= y < dims[1] and 0 <= z < dims[2]):
                continue
            block_name = block.get("blockType", "Air")
            try:
                material_index = palette.name_to_index(block_name)
            except KeyError:
                # Skip unknown blocks for now; user can extend palette as needed.
                continue

            logits[x, y, z, :] = base_logit
            logits[x, y, z, material_index] = confident_logit

        return cls(logits=logits, palette=palette, metadata=metadata)

    # ------------------------------------------------------------------
    # Probability / material helpers
    # ------------------------------------------------------------------
    def material_probabilities(self, temperature: Optional[float] = None) -> torch.Tensor:
        temp = temperature or self.base_temperature
        return torch.softmax(self.logits / temp, dim=-1)

    def densities_and_colours(
        self, temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.material_probabilities(temperature=temperature)
        densities = torch.tensordot(probs, self.palette.densities, dims=([-1], [0]))
        colours = torch.tensordot(probs, self.palette.colours, dims=([-1], [0]))
        return densities, colours

    # ------------------------------------------------------------------
    # Map export helpers
    # ------------------------------------------------------------------
    def collapse_to_blocks(
        self,
        min_probability: float = 0.55,
        temperature: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        probs = self.material_probabilities(temperature=temperature)
        solid_probs = probs.clone()
        solid_probs[..., self.palette.air_index] = 0.0
        max_vals, max_indices = torch.max(solid_probs, dim=-1)

        mask = max_vals >= min_probability
        coords = torch.nonzero(mask, as_tuple=False)

        blocks: List[Dict[str, object]] = []
        for coord in coords.cpu().tolist():
            x, y, z = coord
            material_index = int(max_indices[x, y, z].item())
            block_name = self.palette.index_to_name(material_index)
            if block_name == "Air":
                continue
            prob = float(max_vals[x, y, z].item())
            blocks.append({"position": [x, y, z], "blockType": block_name, "probability": prob})
        return blocks

    def update_map_json(
        self,
        input_map_path: Path | str,
        output_map_path: Path | str,
        min_probability: float = 0.55,
        temperature: Optional[float] = None,
    ) -> Dict[str, object]:
        """Write an updated ``map.json`` with current block predictions."""
        input_map_path = Path(input_map_path)
        with input_map_path.open("r", encoding="utf-8") as fh:
            original = json.load(fh)

        new_blocks = self.collapse_to_blocks(min_probability=min_probability, temperature=temperature)
        original["blocks"] = [
            {"position": b["position"], "blockType": b["blockType"]}
            for b in new_blocks
        ]
        original["lastUpdated"] = datetime.utcnow().isoformat() + "Z"

        output_map_path = Path(output_map_path)
        output_map_path.parent.mkdir(parents=True, exist_ok=True)
        with output_map_path.open("w", encoding="utf-8") as fh:
            json.dump(original, fh, indent=2)
        return original


__all__ = [
    "BlockPalette",
    "ProbabilisticVoxelGrid",
    "WorldMetadata",
]
