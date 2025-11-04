// Plane shader with dots for highlight visualization
struct Camera {
  viewProj : mat4x4<f32>,
  position : vec3<f32>,
  _pad : f32
}
@group(0) @binding(0) var<uniform> uCamera : Camera;

struct PlaneUniforms {
  center: vec3<f32>,
  _pad1: f32,
  sizeX: f32,
  sizeZ: f32,
  color: vec3<f32>,
  _pad2: f32,
}
@group(0) @binding(1) var<uniform> uPlane : PlaneUniforms;

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
}

struct VSOut {
  @builtin(position) position : vec4<f32>,
  @location(0) worldPos : vec3<f32>,
  @location(1) uv : vec2<f32>,
}

@vertex
fn vs_main(in_ : VSIn) -> VSOut {
  var out : VSOut;

  // Transform local position to world position based on plane parameters
  let worldPos = vec3<f32>(
    uPlane.center.x + in_.position.x * uPlane.sizeX,
    uPlane.center.y,
    uPlane.center.z + in_.position.z * uPlane.sizeZ
  );

  out.position = uCamera.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldPos = worldPos;
  out.uv = in_.uv;
  return out;
}

@fragment
fn fs_main(in_ : VSOut) -> @location(0) vec4<f32> {
  // Calculate grid position relative to plane center
  let relX = in_.worldPos.x - uPlane.center.x;
  let relZ = in_.worldPos.z - uPlane.center.z;

  // Dot spacing in world units
  let dotSpacing = 2.0;

  // Find nearest grid point
  let gridX = round(relX / dotSpacing) * dotSpacing;
  let gridZ = round(relZ / dotSpacing) * dotSpacing;

  // Distance to nearest grid point
  let distX = abs(relX - gridX);
  let distZ = abs(relZ - gridZ);
  let dist = sqrt(distX * distX + distZ * distZ);

  // Dot radius
  let dotRadius = 0.15;

  // Check if we're inside a dot
  if (dist < dotRadius) {
    return vec4<f32>(uPlane.color, 0.9);
  }

  // Draw outline for the plane boundaries
  let edgeThickness = 0.08;
  let normX = abs(relX) / uPlane.sizeX;
  let normZ = abs(relZ) / uPlane.sizeZ;

  if (normX > 1.0 - edgeThickness || normZ > 1.0 - edgeThickness) {
    return vec4<f32>(uPlane.color, 0.8);
  }

  // Slightly visible fill
  return vec4<f32>(uPlane.color, 0.1);
}
