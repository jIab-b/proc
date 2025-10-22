"""Public entry points for the voxel SDS training stack."""

from .dataset import MultiViewDataset, ViewRecord
from .cameras import CameraSampler, CameraSample
from .model import VoxelScene
from .sds import score_distillation_loss
from .losses import photometric_loss, regularisation_losses
from .renderer_nvd import VoxelRenderer
from .train import TrainConfig, train
from .export import export_trained_map

__all__ = [
    "MultiViewDataset",
    "ViewRecord",
    "CameraSampler",
    "CameraSample",
    "VoxelScene",
    "VoxelRenderer",
    "TrainConfig",
    "train",
    "score_distillation_loss",
    "photometric_loss",
    "regularisation_losses",
    "export_trained_map",
]
