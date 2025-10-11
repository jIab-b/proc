struct Camera { viewProj : mat4x4<f32> }
@group(0) @binding(0) var<uniform> uCamera : Camera;

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) color : vec3<f32>
}

struct VSOut {
  @builtin(position) position : vec4<f32>,
  @location(0) normal : vec3<f32>,
  @location(1) color : vec3<f32>
}

@vertex
fn vs_main(in_ : VSIn) -> VSOut {
  var out : VSOut;
  out.position = uCamera.viewProj * vec4<f32>(in_.position, 1.0);
  out.normal = normalize(in_.normal);
  out.color = in_.color;
  return out;
}

@fragment
fn fs_main(in_ : VSOut) -> @location(0) vec4<f32> {
  let lightDir = normalize(vec3<f32>(-0.4, -0.85, -0.5));
  let n = normalize(in_.normal);
  let lambert = max(dot(n, -lightDir), 0.0);
  let ambient = 0.25;
  let diffuse = lambert * 0.75;
  let litColor = in_.color * (ambient + diffuse);
  return vec4<f32>(litColor, 1.0);
}
