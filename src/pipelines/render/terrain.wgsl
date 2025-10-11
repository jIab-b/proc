struct Camera { viewProj : mat4x4<f32> }
@group(0) @binding(0) var<uniform> uCamera : Camera;
struct VSIn {
  @location(0) position : vec4<f32>
}

struct VSOut {
  @builtin(position) position : vec4<f32>
}

@vertex
fn vs_main(in_ : VSIn) -> VSOut {
  var out : VSOut;
  let pos = in_.position.xyz;
  out.position = uCamera.viewProj * vec4<f32>(pos, 1.0);
  return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
