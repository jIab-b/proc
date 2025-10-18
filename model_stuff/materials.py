import torch

MATERIALS = [
    "Air",
    "Grass",
    "Dirt",
    "Stone",
    "Plank",
    "Snow",
    "Sand",
    "Water",
]
AIR = 0
sigma_m = torch.tensor([
    0.0,
    20.0,
    22.0,
    30.0,
    15.0,
    10.0,
    18.0,
    2.0,
], dtype=torch.float32)
c_m = torch.tensor([
    [0.0, 0.0, 0.0],
    [0.45, 0.58, 0.30],
    [0.40, 0.30, 0.18],
    [0.56, 0.58, 0.62],
    [0.74, 0.63, 0.45],
    [0.88, 0.90, 0.93],
    [0.87, 0.80, 0.58],
    [0.20, 0.38, 0.64],
], dtype=torch.float32)


