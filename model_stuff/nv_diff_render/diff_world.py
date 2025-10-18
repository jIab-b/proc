"""
Differentiable block world with place_block interface.

Analogous to ChunkManager but with differentiable material parameters.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any

from .materials import MATERIALS, material_name_to_index
from .renderer import DifferentiableBlockRenderer


class DifferentiableBlockWorld(nn.Module):
    """
    Manages block placements with differentiable material parameters.

    Each block has:
    - Fixed discrete position (x, y, z)
    - Differentiable material logits (N parameters, one per material)

    This allows gradient-based optimization of which material to place
    while keeping positions fixed.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (64, 48, 64),
        world_scale: float = 2.0,
        materials: List[str] = MATERIALS,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize empty world.

        Args:
            grid_size: (sx, sy, sz) world dimensions
            world_scale: Scale multiplier
            materials: List of material names
            device: Torch device
        """
        super().__init__()

        self.grid_size = grid_size
        self.world_scale = world_scale
        self.materials = materials
        self.M = len(materials)
        self.device = device

        # Storage for blocks: list of (position, parameter)
        self.blocks: List[Tuple[Tuple[int, int, int], nn.Parameter]] = []

        # Renderer
        self.renderer = DifferentiableBlockRenderer(
            grid_size=grid_size,
            world_scale=world_scale,
            device=device
        )

    def to(self, device: torch.device) -> 'DifferentiableBlockWorld':
        """Move world to device."""
        super().to(device)
        self.device = device
        self.renderer = self.renderer.to(device)

        # Move all block parameters
        for i, (pos, param) in enumerate(self.blocks):
            self.blocks[i] = (pos, nn.Parameter(param.data.to(device)))
            # Re-register parameter
            self.register_parameter(f'block_{i}', self.blocks[i][1])

        return self

    def place_block(
        self,
        position: Tuple[int, int, int],
        material: Optional[int | str] = None,
        logits: Optional[torch.Tensor] = None,
        bias_strength: float = 10.0
    ) -> nn.Parameter:
        """
        Place a block at discrete position with differentiable material.

        Args:
            position: (x, y, z) integer grid coordinates
            material: If provided (int or str), initialize to favor this material
            logits: If provided, use as initial logit values (shape: (M,))
            bias_strength: How strongly to bias toward initial material

        Returns:
            Parameter tensor for this block's material logits
        """
        # Validate position
        sx, sy, sz = self.grid_size
        x, y, z = position
        if not (0 <= x < sx and 0 <= y < sy and 0 <= z < sz):
            raise ValueError(f"Position {position} out of bounds for grid {self.grid_size}")

        # Check if block already exists at this position
        for i, (pos, _) in enumerate(self.blocks):
            if pos == position:
                # Update existing block
                if logits is not None:
                    self.blocks[i][1].data.copy_(logits)
                elif material is not None:
                    mat_idx = material if isinstance(material, int) else material_name_to_index(material)
                    self.blocks[i][1].data.zero_()
                    self.blocks[i][1].data[mat_idx] = bias_strength
                return self.blocks[i][1]

        # Create new block
        if logits is not None:
            block_logits = logits.clone().to(self.device)
        elif material is not None:
            # Strong bias toward specified material
            mat_idx = material if isinstance(material, int) else material_name_to_index(material)
            block_logits = torch.zeros(self.M, device=self.device, dtype=torch.float32)
            block_logits[mat_idx] = bias_strength
        else:
            # Uniform initialization
            block_logits = torch.randn(self.M, device=self.device, dtype=torch.float32) * 0.1

        param = nn.Parameter(block_logits)
        self.blocks.append((position, param))

        # Register parameter with module
        param_name = f'block_{len(self.blocks)-1}'
        self.register_parameter(param_name, param)

        return param

    def remove_block(self, position: Tuple[int, int, int]) -> bool:
        """
        Remove block at position.

        Args:
            position: (x, y, z) to remove

        Returns:
            True if block was removed, False if no block at position
        """
        for i, (pos, param) in enumerate(self.blocks):
            if pos == position:
                # Unregister parameter
                delattr(self, f'block_{i}')
                self.blocks.pop(i)

                # Re-register remaining blocks with updated indices
                for j in range(i, len(self.blocks)):
                    old_name = f'block_{j+1}'
                    new_name = f'block_{j}'
                    if hasattr(self, old_name):
                        delattr(self, old_name)
                    self.register_parameter(new_name, self.blocks[j][1])

                return True

        return False

    def get_block(self, position: Tuple[int, int, int]) -> Optional[nn.Parameter]:
        """Get material logits for block at position, or None."""
        for pos, param in self.blocks:
            if pos == position:
                return param
        return None

    def clear(self):
        """Remove all blocks."""
        for i in range(len(self.blocks)):
            if hasattr(self, f'block_{i}'):
                delattr(self, f'block_{i}')
        self.blocks.clear()

    def get_positions(self) -> List[Tuple[int, int, int]]:
        """Get list of all block positions."""
        return [pos for pos, _ in self.blocks]

    def get_material_logits(self) -> torch.Tensor:
        """
        Get material logits for all blocks.

        Returns:
            (N, M) tensor where N is number of blocks
        """
        if len(self.blocks) == 0:
            return torch.zeros((0, self.M), device=self.device)

        return torch.stack([param for _, param in self.blocks], dim=0)

    def render(
        self,
        camera_view: torch.Tensor,
        camera_proj: torch.Tensor,
        img_h: int,
        img_w: int,
        temperature: float = 1.0,
        hard_materials: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Render current world state.

        Args:
            camera_view: (4, 4) view matrix
            camera_proj: (4, 4) projection matrix
            img_h: Image height
            img_w: Image width
            temperature: Material softmax temperature
            hard_materials: Use hard one-hot assignment
            **kwargs: Additional args for renderer

        Returns:
            (1, 4, H, W) RGBA image
        """
        positions = self.get_positions()
        material_logits = self.get_material_logits()

        # Create neighbor check function
        pos_set = set(positions)
        def neighbor_check(pos):
            return pos in pos_set

        return self.renderer.render(
            positions,
            material_logits,
            camera_view,
            camera_proj,
            img_h,
            img_w,
            neighbor_check=neighbor_check,
            temperature=temperature,
            hard_materials=hard_materials,
            **kwargs
        )

    def state_dict_blocks(self) -> Dict[str, Any]:
        """
        Get state dict for block data (positions + logits).

        Returns:
            Dict with 'positions' and 'logits' keys
        """
        positions = [pos for pos, _ in self.blocks]
        logits = torch.stack([param.data for _, param in self.blocks], dim=0) if self.blocks else torch.empty(0, self.M)

        return {
            'positions': positions,
            'logits': logits,
            'grid_size': self.grid_size,
            'world_scale': self.world_scale
        }

    def load_state_dict_blocks(self, state: Dict[str, Any]):
        """
        Load block data from state dict.

        Args:
            state: Dict from state_dict_blocks()
        """
        self.clear()

        positions = state['positions']
        logits = state['logits']

        for pos, logit in zip(positions, logits):
            self.place_block(pos, logits=logit)

    def __len__(self) -> int:
        """Return number of blocks."""
        return len(self.blocks)

    def __repr__(self) -> str:
        return f"DifferentiableBlockWorld(grid_size={self.grid_size}, blocks={len(self.blocks)}, materials={self.M})"
