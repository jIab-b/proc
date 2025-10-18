# nvdiffrast Block Renderer Specification

**Goal:** Create a high-accuracy, differentiable block renderer using nvdiffrast that matches the WebGPU renderer output pixel-for-pixel while allowing gradient flow through block placement parameters.

## 1. WebGPU Renderer Reference Implementation

### 1.1 Mesh Generation Pipeline

**Source:** `src/chunks.ts:234-289` - `buildChunkMesh()`

```typescript
// For each block at grid position (x, y, z):
for (let y = 0; y < sy; y++) {
  for (let z = 0; z < sz; z++) {
    for (let x = 0; x < sx; x++) {
      const block = chunk.getBlock(x, y, z)
      if (block === BlockType.Air) continue

      // For each of 6 face directions:
      for (let f = 0; f < 6; f++) {
        const face = faceDefs[f]
        const [nx, ny, nz] = [x + face.offset[0], y + face.offset[1], z + face.offset[2]]

        // Culling: only render if neighbor is Air
        if (chunk.getBlock(nx, ny, nz) !== BlockType.Air) continue

        // Generate 6 vertices (2 triangles) for this face
        const color = palette[face.colorSlot]  // top/bottom/side
        const textureLayer = textureConfig ? textureConfig[faceKey] : -1

        for (let i = 0; i < faceIndices.length; i++) {  // [0,1,2,0,2,3]
          const idx = faceIndices[i]
          const corner = face.corners[idx]
          const uv = faceUVs[f][idx]

          vertices.push(
            (baseX + corner[0]) * worldScale,  // position.x
            (baseY + corner[1]) * worldScale,  // position.y
            (baseZ + corner[2]) * worldScale,  // position.z
            face.normal[0],                     // normal.x
            face.normal[1],                     // normal.y
            face.normal[2],                     // normal.z
            color[0],                           // color.r
            color[1],                           // color.g
            color[2],                           // color.b
            uv[0],                              // uv.u
            uv[1],                              // uv.v
            textureLayer                        // textureIndex
          )
        }
      }
    }
  }
}
```

**Key Constants:**

```typescript
// chunks.ts:237-238
const offsetX = -sx / 2  // Center chunk at world X=0
const offsetZ = -sz / 2  // Center chunk at world Z=0
const offsetY = 0        // Chunk bottom at Y=0

// Vertex stride: 12 floats per vertex (chunks.ts:297)
// [pos.xyz, normal.xyz, color.rgb, uv.xy, texIndex]
```

### 1.2 Face Definitions

**Source:** `src/chunks.ts:92-159`

```typescript
const faceDefs = {
  [FaceIndex.PX]: {  // East (+X)
    normal: [1, 0, 0],
    offset: [1, 0, 0],
    corners: [[1,0,0], [1,1,0], [1,1,1], [1,0,1]],
    colorSlot: 'side'
  },
  [FaceIndex.NX]: {  // West (-X)
    normal: [-1, 0, 0],
    offset: [-1, 0, 0],
    corners: [[0,0,0], [0,0,1], [0,1,1], [0,1,0]],
    colorSlot: 'side'
  },
  [FaceIndex.PY]: {  // Top (+Y)
    normal: [0, 1, 0],
    offset: [0, 1, 0],
    corners: [[0,1,0], [0,1,1], [1,1,1], [1,1,0]],
    colorSlot: 'top'
  },
  [FaceIndex.NY]: {  // Bottom (-Y)
    normal: [0, -1, 0],
    offset: [0, -1, 0],
    corners: [[0,0,0], [1,0,0], [1,0,1], [0,0,1]],
    colorSlot: 'bottom'
  },
  [FaceIndex.PZ]: {  // South (+Z)
    normal: [0, 0, 1],
    offset: [0, 0, 1],
    corners: [[0,0,1], [1,0,1], [1,1,1], [0,1,1]],
    colorSlot: 'side'
  },
  [FaceIndex.NZ]: {  // North (-Z)
    normal: [0, 0, -1],
    offset: [0, 0, -1],
    corners: [[0,0,0], [0,1,0], [1,1,0], [1,0,0]],
    colorSlot: 'side'
  }
}

// Triangle indices for quad (2 triangles)
const faceIndices = [0, 1, 2, 0, 2, 3]
```

### 1.3 UV Mapping

**Source:** `src/chunks.ts:47-84`

```typescript
const faceUVs = {
  [FaceIndex.PX]: [[0,1], [0,0], [1,0], [1,1]],  // East
  [FaceIndex.NX]: [[1,1], [0,1], [0,0], [1,0]],  // West
  [FaceIndex.PY]: [[0,1], [0,0], [1,0], [1,1]],  // Top
  [FaceIndex.NY]: [[0,0], [1,0], [1,1], [0,1]],  // Bottom
  [FaceIndex.PZ]: [[0,1], [1,1], [1,0], [0,0]],  // South
  [FaceIndex.NZ]: [[1,1], [1,0], [0,0], [0,1]]   // North
}
```

### 1.4 Color Palette

**Source:** `src/chunks.ts:163-200`

```typescript
const blockPalette = {
  [BlockType.Air]: undefined,
  [BlockType.Grass]: {
    top: [0.34, 0.68, 0.36],
    bottom: [0.40, 0.30, 0.16],
    side: [0.45, 0.58, 0.30]
  },
  [BlockType.Dirt]: {
    top: [0.42, 0.32, 0.20],
    bottom: [0.38, 0.26, 0.16],
    side: [0.40, 0.30, 0.18]
  },
  [BlockType.Stone]: {
    top: [0.58, 0.60, 0.64],
    bottom: [0.55, 0.57, 0.60],
    side: [0.56, 0.58, 0.62]
  },
  [BlockType.Plank]: {
    top: [0.78, 0.68, 0.50],
    bottom: [0.72, 0.60, 0.42],
    side: [0.74, 0.63, 0.45]
  },
  [BlockType.Snow]: {
    top: [0.92, 0.94, 0.96],
    bottom: [0.90, 0.92, 0.94],
    side: [0.88, 0.90, 0.93]
  },
  [BlockType.Sand]: {
    top: [0.88, 0.82, 0.60],
    bottom: [0.86, 0.78, 0.56],
    side: [0.87, 0.80, 0.58]
  },
  [BlockType.Water]: {
    top: [0.22, 0.40, 0.66],
    bottom: [0.20, 0.34, 0.60],
    side: [0.20, 0.38, 0.64]
  }
}
```

### 1.5 Lighting Shader

**Source:** `src/pipelines/render/terrain.wgsl:34-54`

```wgsl
@fragment
fn fs_main(in_: VSOut) -> @location(0) vec4<f32> {
  // Constants
  let sunDir = normalize(vec3<f32>(-0.4, -0.85, -0.5));
  let sunColor = vec3<f32>(1.0, 0.97, 0.9);
  let skyColor = vec3<f32>(0.53, 0.81, 0.92);
  let horizonColor = vec3<f32>(0.18, 0.16, 0.12);

  // Lighting calculation
  let n = normalize(in_.normal);
  let sunDiffuse = max(dot(n, -sunDir), 0.0);
  let skyFactor = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
  let ambient = (horizonColor * (1.0 - skyFactor) + skyColor * skyFactor) * 0.55;
  let diffuse = sunDiffuse * sunColor;

  // Base surface color (from vertex attribute)
  var surfaceColor = in_.color;
  var surfaceAlpha = 1.0;

  // Optional texture sampling
  if (in_.textureIndex >= 0.0) {
    let layer = i32(in_.textureIndex + 0.5);
    let sampleColor = textureSample(uTileTextures, uAtlasSampler, in_.uv, layer);
    surfaceAlpha = sampleColor.a;
    surfaceColor = in_.color * (1.0 - surfaceAlpha) + sampleColor.rgb * surfaceAlpha;
  }

  // Final shading
  let litColor = surfaceColor * (ambient + diffuse);
  return vec4<f32>(litColor, surfaceAlpha);
}
```

**Clear color (background):** `{r: 0.53, g: 0.81, b: 0.92, a: 1}` (sky blue)

### 1.6 Camera System

**Source:** `src/camera.ts:4-54`

```typescript
// Projection matrix (OpenGL convention: Z in [-1, 1])
function createPerspective(fovYRad, aspect, near, far) {
  const f = 1.0 / Math.tan(fovYRad / 2)
  const nf = 1 / (near - far)
  const out = new Float32Array(16)  // Column-major
  out[0] = f / aspect  // [0,0]
  out[5] = f           // [1,1]
  out[10] = (far + near) * nf      // [2,2]
  out[11] = -1                      // [2,3]
  out[14] = (2 * far * near) * nf  // [3,2]
  return out
}

// View matrix (world <- camera)
function lookAt(eye, center, up) {
  // Returns column-major 4x4 matrix
  // [right.x, up.x, -forward.x, 0]
  // [right.y, up.y, -forward.y, 0]
  // [right.z, up.z, -forward.z, 0]
  // [-dot(right,eye), -dot(up,eye), dot(forward,eye), 1]
}
```

**Default camera settings:**
- FOV: 60° (1.047 radians)
- Near: 0.1
- Far: 500.0
- World scale: 2.0

---

## 2. nvdiffrast Renderer Architecture

### 2.1 Module Structure

```
model_stuff/nv_diff_render/
├── __init__.py              # Exports main API
├── SPEC.md                  # This file
├── mesh_builder.py          # Block mesh generation
├── materials.py             # Material definitions and palette
├── shading.py               # Fragment shader implementation
├── renderer.py              # Main DifferentiableBlockRenderer class
├── diff_world.py            # DifferentiableBlockWorld with place_block()
├── test_render.py           # Test renderer matching test_render_dataset.py
└── utils.py                 # Coordinate transforms, matrix utilities
```

### 2.2 Core Data Flow

```
Block Placements                 Camera Matrices
   ↓                                  ↓
[positions, material_logits]     [view, proj]
   ↓                                  ↓
mesh_builder.build_mesh()            │
   ↓                                  │
[vertices, faces, attributes] ←──────┘
   ↓
nvdiffrast.rasterize()
   ↓
[rasterized, interpolated_attributes]
   ↓
shading.fragment_shade()
   ↓
[RGBA image (H,W,4)]
   ↓
composite_over_sky()
   ↓
Final Image (1,4,H,W)
```

### 2.3 API Design

#### 2.3.1 Mesh Builder

```python
def build_block_mesh(
    positions: List[Tuple[int, int, int]],
    material_logits: torch.Tensor,  # (N_blocks, M)
    grid_size: Tuple[int, int, int],
    world_scale: float = 2.0,
    neighbor_check: Optional[Callable] = None,
    temperature: float = 1.0,
    hard_assignment: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Build mesh from block placements with differentiable materials.

    Args:
        positions: List of (x, y, z) integer grid coordinates
        material_logits: (N, M) logits for M materials per block
        grid_size: (X, Y, Z) world dimensions for bounds checking
        world_scale: Multiplier for vertex positions (default: 2.0)
        neighbor_check: Optional function(pos) -> is_solid for culling
        temperature: Gumbel-Softmax temperature (default: 1.0)
        hard_assignment: Use hard one-hot if True (default: False)

    Returns:
        vertices: (V, 3) float32 world-space positions
        faces: (F, 3) int32 triangle indices
        attributes: {
            'normals': (V, 3) face normals,
            'colors': (V, 3) RGB from palette (material-weighted),
            'uvs': (V, 2) texture coordinates,
            'material_weights': (V, M) per-vertex material probabilities
        }
    """
```

#### 2.3.2 Fragment Shader

```python
class TerrainShader:
    """Matches terrain.wgsl fragment shader exactly."""

    def __init__(self,
                 material_palette: torch.Tensor,  # (M, 3, 3) [material, face_type, RGB]
                 texture_atlas: Optional[torch.Tensor] = None):  # (M, 6, 4, H, W)
        self.SUN_DIR = torch.tensor([-0.4, -0.85, -0.5])
        self.SUN_COLOR = torch.tensor([1.0, 0.97, 0.9])
        self.SKY_COLOR = torch.tensor([0.53, 0.81, 0.92])
        self.HORIZON_COLOR = torch.tensor([0.18, 0.16, 0.12])
        self.AMBIENT_SCALE = 0.55

    def shade(self,
              normals: torch.Tensor,  # (H, W, 3)
              colors: torch.Tensor,   # (H, W, 3)
              uvs: torch.Tensor,      # (H, W, 2)
              material_weights: torch.Tensor,  # (H, W, M)
              mask: torch.Tensor      # (H, W, 1) pixel validity
             ) -> torch.Tensor:       # (H, W, 3) RGB
        """
        Apply lighting matching WebGPU terrain.wgsl exactly.

        Returns: (H, W, 3) lit RGB values in [0, 1]
        """
```

#### 2.3.3 Main Renderer

```python
class DifferentiableBlockRenderer:
    """High-accuracy nvdiffrast renderer matching WebGPU output."""

    def __init__(self,
                 grid_size: Tuple[int, int, int] = (64, 48, 64),
                 world_scale: float = 2.0,
                 materials: List[str] = MATERIALS,
                 texture_atlas: Optional[torch.Tensor] = None):
        self.grid_size = grid_size
        self.world_scale = world_scale
        self.shader = TerrainShader(get_material_palette(), texture_atlas)

    def render(self,
               positions: List[Tuple[int, int, int]],
               material_logits: torch.Tensor,  # (N, M)
               camera_view: torch.Tensor,      # (4, 4)
               camera_proj: torch.Tensor,      # (4, 4)
               img_h: int,
               img_w: int,
               temperature: float = 1.0,
               hard_materials: bool = False
              ) -> torch.Tensor:  # (1, 4, H, W)
        """
        Render blocks from camera view.

        Returns: (1, 4, H, W) RGBA image with sky composite
        """
```

#### 2.3.4 Differentiable World

```python
class DifferentiableBlockWorld(nn.Module):
    """
    Manages block placements with differentiable material parameters.
    Analogous to ChunkManager but with gradient support.
    """

    def __init__(self,
                 grid_size: Tuple[int, int, int] = (64, 48, 64),
                 materials: List[str] = MATERIALS):
        super().__init__()
        self.blocks: List[Tuple[Tuple[int,int,int], nn.Parameter]] = []

    def place_block(self,
                    position: Tuple[int, int, int],
                    material: Optional[int] = None,
                    logits: Optional[torch.Tensor] = None
                   ) -> nn.Parameter:
        """
        Place a block at discrete position with differentiable material.

        Args:
            position: (x, y, z) integer coordinates
            material: If provided, initialize to strongly favor this material
            logits: If provided, use as initial logit values

        Returns:
            Parameter tensor for this block's material logits
        """

    def remove_block(self, position: Tuple[int, int, int]):
        """Remove block at position."""

    def render(self, camera_view, camera_proj, img_h, img_w, **kwargs):
        """Render current world state."""
```

---

## 3. Implementation Requirements

### 3.1 Coordinate System Match

**Critical:** Must exactly match WebGPU coordinate transformations.

```python
# Block grid to world coordinates
def block_to_world(x: int, y: int, z: int,
                   grid_size: Tuple[int,int,int],
                   world_scale: float) -> Tuple[float, float, float]:
    sx, sy, sz = grid_size
    offset_x = -sx / 2.0
    offset_z = -sz / 2.0
    offset_y = 0.0

    return (
        (x + offset_x) * world_scale,
        (y + offset_y) * world_scale,
        (z + offset_z) * world_scale
    )

# Face corner to world vertex
def get_face_vertex(block_pos: Tuple[int,int,int],
                   face_index: int,
                   corner_index: int,
                   grid_size: Tuple[int,int,int],
                   world_scale: float) -> Tuple[float, float, float]:
    x, y, z = block_pos
    corner = FACE_DEFS[face_index]['corners'][corner_index]

    # Transform to world
    base_x, base_y, base_z = block_to_world(x, y, z, grid_size, world_scale)

    return (
        base_x + corner[0] * world_scale,
        base_y + corner[1] * world_scale,
        base_z + corner[2] * world_scale
    )
```

### 3.2 Camera Matrix Handling

**WebGPU uses column-major matrices, PyTorch uses row-major.**

```python
def load_camera_from_metadata(metadata: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load camera matrices from dataset metadata.

    Args:
        metadata: Dict with 'viewMatrix' and 'projectionMatrix' as flat lists

    Returns:
        view: (4, 4) row-major tensor
        proj: (4, 4) row-major tensor
    """
    # Metadata stores column-major (WebGPU convention)
    view_col = np.array(metadata['viewMatrix']).reshape(4, 4)
    proj_col = np.array(metadata['projectionMatrix']).reshape(4, 4)

    # Transpose to row-major for PyTorch
    view = torch.from_numpy(view_col.T).float()
    proj = torch.from_numpy(proj_col.T).float()

    return view, proj
```

### 3.3 Material Palette Format

```python
# Shape: (M, 3, 3) where:
#   M = number of materials (8: Air, Grass, Dirt, Stone, Plank, Snow, Sand, Water)
#   First dim = material index
#   Second dim = face type (0=top, 1=bottom, 2=side)
#   Third dim = RGB

MATERIAL_PALETTE = torch.tensor([
    # Air (not rendered)
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    # Grass
    [[0.34, 0.68, 0.36], [0.40, 0.30, 0.16], [0.45, 0.58, 0.30]],
    # Dirt
    [[0.42, 0.32, 0.20], [0.38, 0.26, 0.16], [0.40, 0.30, 0.18]],
    # Stone
    [[0.58, 0.60, 0.64], [0.55, 0.57, 0.60], [0.56, 0.58, 0.62]],
    # Plank
    [[0.78, 0.68, 0.50], [0.72, 0.60, 0.42], [0.74, 0.63, 0.45]],
    # Snow
    [[0.92, 0.94, 0.96], [0.90, 0.92, 0.94], [0.88, 0.90, 0.93]],
    # Sand
    [[0.88, 0.82, 0.60], [0.86, 0.78, 0.56], [0.87, 0.80, 0.58]],
    # Water
    [[0.22, 0.40, 0.66], [0.20, 0.34, 0.60], [0.20, 0.38, 0.64]],
])
```

### 3.4 Face Culling Logic

```python
def should_render_face(block_pos: Tuple[int,int,int],
                      face_index: int,
                      neighbor_check: Callable[[Tuple[int,int,int]], bool]
                     ) -> bool:
    """
    Determine if a block face should be rendered.

    Args:
        block_pos: (x, y, z) of current block
        face_index: 0-5 (PX, NX, PY, NY, PZ, NZ)
        neighbor_check: Function that returns True if position is solid

    Returns:
        True if face should be rendered (neighbor is Air or out of bounds)
    """
    offset = FACE_DEFS[face_index]['offset']
    neighbor_pos = (
        block_pos[0] + offset[0],
        block_pos[1] + offset[1],
        block_pos[2] + offset[2]
    )

    # Render if neighbor is Air (or out of bounds)
    return not neighbor_check(neighbor_pos)
```

### 3.5 Differentiable Material Assignment

**Two modes:**

1. **Training mode (soft):** Gumbel-Softmax for gradient flow
2. **Inference mode (hard):** Argmax with straight-through estimator

```python
def assign_materials(logits: torch.Tensor,
                    temperature: float = 1.0,
                    hard: bool = False) -> torch.Tensor:
    """
    Convert material logits to probabilities or one-hot.

    Args:
        logits: (N, M) material logits per block
        temperature: Softmax temperature
        hard: If True, return hard one-hot (with straight-through grad)

    Returns:
        (N, M) material probabilities or one-hot
    """
    if hard:
        return F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
    else:
        return F.softmax(logits / temperature, dim=-1)
```

### 3.6 Per-Vertex Material Weighting

When a vertex is shared by multiple blocks (corner/edge cases), blend materials:

```python
def compute_vertex_colors(face_vertices: List[VertexInfo],
                         material_probs: torch.Tensor  # (N_blocks, M)
                        ) -> torch.Tensor:  # (V, 3)
    """
    Compute per-vertex colors from material probabilities.

    For each vertex:
        - If belongs to 1 block: color = sum(prob[m] * palette[m][face_type])
        - If shared by multiple blocks: average the above

    Returns: (V, 3) RGB colors
    """
```

---

## 4. Testing & Validation

### 4.1 Test Renderer Script

**File:** `model_stuff/nv_diff_render/test_render.py`

Must support same interface as `model_stuff/renderer/test_render_dataset.py`:

```python
# Usage:
# python -m model_stuff.nv_diff_render.test_render --dataset_id 1 --view_index 0

def test_render_dataset(dataset_id: int, view_index: int):
    """
    Load dataset, render with nvdiffrast, compare to ground truth.

    Steps:
    1. Load metadata from datasets/{dataset_id}/metadata.json
    2. Load map.json (block placements)
    3. Extract camera matrices for view_index
    4. Render with nvdiffrast
    5. Save output to out_local/nvdiff_test/dataset_{id}_view_{index}.png
    6. Compute MSE vs ground truth RGB image
    7. Print comparison metrics
    """
```

### 4.2 Validation Metrics

```python
def validate_render_accuracy(nvdiff_img: np.ndarray,
                            webgpu_img: np.ndarray) -> Dict[str, float]:
    """
    Compare nvdiffrast output to WebGPU ground truth.

    Returns:
        {
            'mse': Mean squared error (should be < 0.001 for exact match),
            'psnr': Peak signal-to-noise ratio (should be > 40 dB),
            'ssim': Structural similarity (should be > 0.99),
            'max_error': Maximum per-pixel difference,
            'geometry_match': Fraction of pixels with same binary mask
        }
    """
```

### 4.3 Unit Tests

**Required test cases:**

1. **Single block render** - Stone cube at origin
2. **Multi-block colors** - Grass, Dirt, Stone side-by-side
3. **Face culling** - Verify hidden faces not rendered
4. **Coordinate transform** - Check vertex positions match WebGPU
5. **Lighting match** - Verify diffuse/ambient values
6. **Camera transforms** - Test multiple viewpoints
7. **Material gradient flow** - Verify backprop through material logits

---

## 5. Performance Considerations

### 5.1 Memory Optimization

- **Vertex deduplication:** Share vertices between adjacent faces
- **Batch processing:** Render multiple views in parallel
- **Gradient checkpointing:** For large worlds, checkpoint mesh building

### 5.2 Speed Targets

- **Mesh building:** < 10ms for 1000 blocks
- **Rasterization:** < 50ms for 512x512 image
- **Total render:** < 100ms end-to-end

### 5.3 Differentiability

**Gradients flow through:**
- ✅ Material logits → material probabilities → vertex colors
- ✅ Material probabilities → texture blending weights
- ❌ Block positions (discrete, not differentiable in this design)

**Future extension:** Continuous position parameters with spatial transformer networks.

---

## 6. Integration with Training Pipeline

### 6.1 SDS Training Modifications

**Current:** `train_sds.py` optimizes continuous voxel logits `W[X,Y,Z,M]`

**Proposed:** Optimize discrete block placements with material logits

```python
# Old approach
W_logits = nn.Parameter(torch.randn(X, Y, Z, M) * 0.1)

# New approach
world = DifferentiableBlockWorld(grid_size=(X, Y, Z))
for pos in initial_positions:
    world.place_block(pos)  # Creates nn.Parameter for each block

# Training loop
for step in range(STEPS):
    img = world.render(view, proj, img_h, img_w)
    loss = sds_loss(img, text_prompt)
    loss.backward()
    optimizer.step()  # Updates material logits
```

### 6.2 Block Initialization Strategies

1. **From DSL:** Parse existing map.json, place blocks at positions
2. **From voxel grid:** Convert soft W to hard placements via thresholding
3. **Random:** Sample positions, initialize materials uniformly
4. **Sparse seeding:** Start with few blocks, add more during training

---

## 7. File I/O Formats

### 7.1 Input: Dataset Metadata

**File:** `datasets/{id}/metadata.json`

```json
{
  "formatVersion": "1.0",
  "exportedAt": "2024-01-15T10:30:00Z",
  "imageSize": {"width": 512, "height": 512},
  "viewCount": 5,
  "captureId": "capture_abc123",
  "views": [
    {
      "id": "view_001",
      "index": 0,
      "position": [10.5, 20.0, 15.3],
      "forward": [0.707, -0.5, 0.5],
      "up": [0.0, 1.0, 0.0],
      "right": [0.707, 0.0, -0.707],
      "intrinsics": {
        "fovYDegrees": 60.0,
        "aspect": 1.0,
        "near": 0.1,
        "far": 500.0
      },
      "viewMatrix": [ /* 16 floats, column-major */ ],
      "projectionMatrix": [ /* 16 floats, column-major */ ],
      "viewProjectionMatrix": [ /* 16 floats, column-major */ ],
      "rgbBase64": "..."
    }
  ]
}
```

### 7.2 Input: Map File

**File:** `maps/{id}/map.json`

```json
{
  "sequence": 1,
  "worldScale": 2.0,
  "blocks": [
    {"position": [10, 5, 10], "blockType": "Stone"},
    {"position": [11, 5, 10], "blockType": "Grass"}
  ]
}
```

### 7.3 Output: Rendered Images

**File:** `out_local/nvdiff_test/dataset_{id}_view_{index}.png`

- Format: PNG, RGB or RGBA
- Size: Match metadata imageSize
- Color space: sRGB

---

## 8. Development Roadmap

### Phase 1: Core Implementation (Week 1)
- [ ] `materials.py` - Define constants, palette
- [ ] `utils.py` - Coordinate transforms, matrix utilities
- [ ] `mesh_builder.py` - Basic mesh generation (no gradients)
- [ ] `shading.py` - Fragment shader implementation

### Phase 2: Differentiability (Week 2)
- [ ] `diff_world.py` - DifferentiableBlockWorld class
- [ ] Gumbel-Softmax material assignment
- [ ] Gradient flow tests

### Phase 3: Validation (Week 3)
- [ ] `test_render.py` - Test script
- [ ] Pixel-perfect validation vs WebGPU
- [ ] Performance benchmarks

### Phase 4: Training Integration (Week 4)
- [ ] Modify `train_sds.py` to use nvdiffrast renderer
- [ ] Block initialization from DSL
- [ ] End-to-end training test

---

## 9. Open Questions & Decisions

### 9.1 Texture Atlas Format

**Question:** How to handle custom block textures from frontend?

**Options:**
1. Load texture atlas from disk matching WebGPU texture array
2. Generate procedural textures in PyTorch
3. Hybrid: base textures + per-instance color modulation

**Decision:** Start with option 1 (load from disk), add option 3 later.

### 9.2 Vertex Sharing

**Question:** Should vertices be shared between adjacent faces?

**Pros:** Less memory, matches typical mesh structure
**Cons:** Complicates per-vertex material weighting, harder to implement

**Decision:** Start without sharing (duplicate vertices), optimize later if needed.

### 9.3 Backface Culling

**Question:** Apply backface culling in nvdiffrast?

**WebGPU:** Uses `cullMode: 'back'` (cull back-facing triangles)

**Decision:** Enable backface culling to match WebGPU exactly.

### 9.4 Antialiasing

**Question:** Use MSAA or other antialiasing?

**WebGPU:** No explicit MSAA in current implementation

**Decision:** No antialiasing initially, add if needed for better comparisons.

---

## 10. References

### WebGPU Source Files
- `src/chunks.ts` - Mesh generation
- `src/pipelines/render/terrain.wgsl` - Fragment shader
- `src/camera.ts` - Camera matrices
- `src/webgpuEngine.ts` - Rendering pipeline
- `src/blockUtils.ts` - Block type definitions

### nvdiffrast Documentation
- [nvdiffrast GitHub](https://github.com/NVlabs/nvdiffrast)
- API: `dr.rasterize()`, `dr.interpolate()`, `dr.texture()`

### Related Code
- `model_stuff/renderer/core.py` - Current volumetric renderer
- `model_stuff/train_sds.py` - SDS training loop
- `model_stuff/materials.py` - Material definitions

---

## Appendix A: Face Index Mapping

```python
# Enum matching chunks.ts:22-29
class FaceIndex:
    PX = 0  # East  (+X)
    NX = 1  # West  (-X)
    PY = 2  # Top   (+Y)
    NY = 3  # Bottom (-Y)
    PZ = 4  # South (+Z)
    NZ = 5  # North (-Z)

# Face key strings for texture lookup
FACE_KEYS = ['east', 'west', 'top', 'bottom', 'south', 'north']

# Mapping to palette slot
FACE_PALETTE_SLOT = [
    2,  # PX/east   -> side
    2,  # NX/west   -> side
    0,  # PY/top    -> top
    1,  # NY/bottom -> bottom
    2,  # PZ/south  -> side
    2   # NZ/north  -> side
]
```

## Appendix B: Complete Face Definition Data

```python
FACE_DEFS = [
    {  # FaceIndex.PX (East, +X)
        'normal': (1, 0, 0),
        'offset': (1, 0, 0),
        'corners': [(1,0,0), (1,1,0), (1,1,1), (1,0,1)],
        'uvs': [(0,1), (0,0), (1,0), (1,1)],
        'palette_slot': 2  # side
    },
    {  # FaceIndex.NX (West, -X)
        'normal': (-1, 0, 0),
        'offset': (-1, 0, 0),
        'corners': [(0,0,0), (0,0,1), (0,1,1), (0,1,0)],
        'uvs': [(1,1), (0,1), (0,0), (1,0)],
        'palette_slot': 2  # side
    },
    {  # FaceIndex.PY (Top, +Y)
        'normal': (0, 1, 0),
        'offset': (0, 1, 0),
        'corners': [(0,1,0), (0,1,1), (1,1,1), (1,1,0)],
        'uvs': [(0,1), (0,0), (1,0), (1,1)],
        'palette_slot': 0  # top
    },
    {  # FaceIndex.NY (Bottom, -Y)
        'normal': (0, -1, 0),
        'offset': (0, -1, 0),
        'corners': [(0,0,0), (1,0,0), (1,0,1), (0,0,1)],
        'uvs': [(0,0), (1,0), (1,1), (0,1)],
        'palette_slot': 1  # bottom
    },
    {  # FaceIndex.PZ (South, +Z)
        'normal': (0, 0, 1),
        'offset': (0, 0, 1),
        'corners': [(0,0,1), (1,0,1), (1,1,1), (0,1,1)],
        'uvs': [(0,1), (1,1), (1,0), (0,0)],
        'palette_slot': 2  # side
    },
    {  # FaceIndex.NZ (North, -Z)
        'normal': (0, 0, -1),
        'offset': (0, 0, -1),
        'corners': [(0,0,0), (0,1,0), (1,1,0), (1,0,0)],
        'uvs': [(1,1), (1,0), (0,0), (0,1)],
        'palette_slot': 2  # side
    }
]

# Triangle indices for quad (winding order for backface culling)
FACE_INDICES = [0, 1, 2, 0, 2, 3]
```
