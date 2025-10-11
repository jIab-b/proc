struct Params {
  dimsCells : vec3<u32>,
  pad0 : u32,
  origin : vec3<f32>,
  voxelSize : f32,
  isoLevel : f32,
  noiseScale : f32,
  pad1 : vec2<f32>
}

@group(0) @binding(0) var<uniform> uParams : Params;
@group(0) @binding(1) var<storage, read> caseMask : array<u32>;
@group(0) @binding(2) var<storage, read> scanOffsets : array<u32>;
@group(0) @binding(3) var<storage, read_write> positions : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> normals : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> indices : array<u32>;
@group(0) @binding(6) var<storage, read_write> indirect : array<u32>;
@group(0) @binding(7) var<storage, read> totalOut : array<u32>;

@compute @workgroup_size(8,8,4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= uParams.dimsCells)) { return; }
  let idx = gid.x + uParams.dimsCells.x * (gid.y + uParams.dimsCells.y * gid.z);
  if (caseMask[idx] == 0u) { return; }
  let baseVertex = scanOffsets[idx] * 3u;
  positions[baseVertex + 0u] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  positions[baseVertex + 1u] = vec4<f32>(1.0, 0.0, 0.0, 1.0);
  positions[baseVertex + 2u] = vec4<f32>(0.0, 1.0, 0.0, 1.0);
  normals[baseVertex + 0u] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
  normals[baseVertex + 1u] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
  normals[baseVertex + 2u] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
  if (idx == 0u) {
    indirect[0] = totalOut[0] * 3u;
    indirect[1] = 1u;
    indirect[2] = 0u;
    indirect[3] = 0u;
  }
}
