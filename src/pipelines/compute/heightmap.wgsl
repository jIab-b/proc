struct Params {
  dims : vec2<u32>,
  pad0 : vec2<u32>,
  originSpacing : vec4<f32>,
  heightNoise : vec4<f32>
}

@group(0) @binding(0) var<uniform> uParams : Params;
@group(0) @binding(1) var<storage, read_write> positions : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> indirect : array<u32>;

fn ridge(v: f32) -> f32 {
  let r = 1.0 - abs(v);
  return r * r;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= uParams.dims.x || gid.y >= uParams.dims.y) {
    return;
  }

  let idx = gid.y * uParams.dims.x + gid.x;
  let spacing = uParams.originSpacing.zw;
  let offset = uParams.originSpacing.xy;
  let dimsF = vec2<f32>(uParams.dims);
  let center = (vec2<f32>(vec2<u32>(gid.xy)) / max(dimsF - vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 1.0))) - 0.5;
  let posXZ = (vec2<f32>(vec2<u32>(gid.xy)) * spacing) + offset;

  let time = uParams.heightNoise.w;
  let basePos = vec3<f32>(posXZ.x * uParams.heightNoise.y, posXZ.y * uParams.heightNoise.y, time);
  var height = fbm(basePos) * uParams.heightNoise.x;

  let ridged = ridge(fbm(basePos * 0.6) * 2.0 - 1.0) * uParams.heightNoise.z;
  height = height + ridged;

  let mask = max(0.0, 1.0 - length(center) * 1.6);
  height = height + mask * uParams.heightNoise.x * 0.6;

  positions[idx] = vec4<f32>(posXZ.x, height, posXZ.y, 1.0);

  if (gid.x == 0u && gid.y == 0u) {
    indirect[0] = uParams.dims.x * uParams.dims.y;
    indirect[1] = 1u;
    indirect[2] = 0u;
    indirect[3] = 0u;
  }
}

