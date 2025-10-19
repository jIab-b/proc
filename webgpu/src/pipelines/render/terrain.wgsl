struct Camera { viewProj : mat4x4<f32> }
@group(0) @binding(0) var<uniform> uCamera : Camera;
@group(0) @binding(1) var uAtlasSampler : sampler;
@group(0) @binding(2) var uTileTextures : texture_2d_array<f32>;

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) color : vec3<f32>,
  @location(3) uv : vec2<f32>,
  @location(4) textureIndex : f32
}

struct VSOut {
  @builtin(position) position : vec4<f32>,
  @location(0) normal : vec3<f32>,
  @location(1) color : vec3<f32>,
  @location(2) uv : vec2<f32>,
  @location(3) textureIndex : f32
}

@vertex
fn vs_main(in_ : VSIn) -> VSOut {
  var out : VSOut;
  out.position = uCamera.viewProj * vec4<f32>(in_.position, 1.0);
  out.normal = normalize(in_.normal);
  out.color = in_.color;
  out.uv = in_.uv;
  out.textureIndex = in_.textureIndex;
  return out;
}

@fragment
fn fs_main(in_ : VSOut) -> @location(0) vec4<f32> {
  let sunDir = normalize(vec3<f32>(-0.4, -0.85, -0.5));
  let sunColor = vec3<f32>(1.0, 0.97, 0.9);
  let n = normalize(in_.normal);
  let sunDiffuse = max(dot(n, -sunDir), 0.0);
  let skyFactor = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
  let skyColor = vec3<f32>(0.53, 0.81, 0.92);
  let horizonColor = vec3<f32>(0.18, 0.16, 0.12);
  let ambient = (horizonColor * (1.0 - skyFactor) + skyColor * skyFactor) * 0.55;
  let diffuse = sunDiffuse * sunColor;
  var surfaceColor = in_.color;
  var surfaceAlpha = 1.0;
  if (in_.textureIndex >= 0.0) {
    var layer = i32(in_.textureIndex + 0.5);
    layer = clamp(layer, 0, i32(textureNumLayers(uTileTextures)) - 1);
    let sampleColor = textureSample(uTileTextures, uAtlasSampler, in_.uv, layer);
    surfaceAlpha = sampleColor.a;
    surfaceColor = in_.color * (1.0 - surfaceAlpha) + sampleColor.rgb * surfaceAlpha;
  }
  let litColor = surfaceColor * (ambient + diffuse);
  return vec4<f32>(litColor, surfaceAlpha);
}
