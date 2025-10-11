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
@group(0) @binding(1) var<storage, read> triTable : array<i32>;
@group(0) @binding(2) var<storage, read_write> caseMask : array<u32>;
@group(0) @binding(3) var<storage, read_write> triCount : array<u32>;

fn sampleField(p: vec3<f32>) -> f32 {
  return fbm(p * uParams.noiseScale) - 0.5;
}

@compute @workgroup_size(8,8,4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= uParams.dimsCells)) { return; }
  let idx = gid.x + uParams.dimsCells.x * (gid.y + uParams.dimsCells.y * gid.z);
  let base = uParams.origin + vec3<f32>(gid) * uParams.voxelSize;
  let s = uParams.voxelSize;
  let p000 = base + vec3<f32>(0.0,0.0,0.0) * s;
  let p100 = base + vec3<f32>(1.0,0.0,0.0) * s;
  let p010 = base + vec3<f32>(0.0,1.0,0.0) * s;
  let p110 = base + vec3<f32>(1.0,1.0,0.0) * s;
  let p001 = base + vec3<f32>(0.0,0.0,1.0) * s;
  let p101 = base + vec3<f32>(1.0,0.0,1.0) * s;
  let p011 = base + vec3<f32>(0.0,1.0,1.0) * s;
  let p111 = base + vec3<f32>(1.0,1.0,1.0) * s;
  let d0 = sampleField(p000);
  let d1 = sampleField(p100);
  let d2 = sampleField(p010);
  let d3 = sampleField(p110);
  let d4 = sampleField(p001);
  let d5 = sampleField(p101);
  let d6 = sampleField(p011);
  let d7 = sampleField(p111);
  var cubeIndex : u32 = 0u;
  if (d0 < uParams.isoLevel) { cubeIndex = cubeIndex | 1u; }
  if (d1 < uParams.isoLevel) { cubeIndex = cubeIndex | 2u; }
  if (d2 < uParams.isoLevel) { cubeIndex = cubeIndex | 4u; }
  if (d3 < uParams.isoLevel) { cubeIndex = cubeIndex | 8u; }
  if (d4 < uParams.isoLevel) { cubeIndex = cubeIndex | 16u; }
  if (d5 < uParams.isoLevel) { cubeIndex = cubeIndex | 32u; }
  if (d6 < uParams.isoLevel) { cubeIndex = cubeIndex | 64u; }
  if (d7 < uParams.isoLevel) { cubeIndex = cubeIndex | 128u; }
  caseMask[idx] = cubeIndex;
  var tcount : u32 = 0u;
  let row = i32(cubeIndex) * 16;
  for (var i = 0; i < 16; i = i + 3) {
    if (triTable[row + i] < 0) { break; }
    tcount = tcount + 1u;
  }
  triCount[idx] = tcount;
}

