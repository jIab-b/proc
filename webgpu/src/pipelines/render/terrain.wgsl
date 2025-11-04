struct Camera {
  viewProj : mat4x4<f32>,
  position : vec3<f32>,
  _pad : f32
}
@group(0) @binding(0) var<uniform> uCamera : Camera;
@group(0) @binding(1) var uAtlasSampler : sampler;
@group(0) @binding(2) var uTileTextures : texture_2d_array<f32>;
@group(0) @binding(3) var<uniform> uCamParams : vec2<f32>; // near, far

// PBR Material buffer (256 block types max)
// Note: WGSL vec3 aligns to 16 bytes in structs
struct MaterialParams {
  albedo : vec3<f32>,
  _pad0 : f32,              // vec3 padding to 16 bytes
  roughness : f32,
  metallic : f32,
  emissiveStrength : f32,
  ao : f32,
  _pad1 : vec4<f32>,       // padding to next 16-byte boundary
  emissive : vec3<f32>,
  _pad2 : f32              // vec3 padding to 16 bytes
}
@group(0) @binding(4) var<storage, read> uMaterials : array<MaterialParams>;

// Lighting parameters
struct LightingParams {
  sunDirection : vec3<f32>,
  sunIntensity : f32,
  sunColor : vec3<f32>,
  skyIntensity : f32,
  skyZenith : vec3<f32>,
  _pad1 : f32,
  skyHorizon : vec3<f32>,
  _pad2 : f32,
  skyGround : vec3<f32>,
  _pad3 : f32,
  ambientColor : vec3<f32>,
  ambientIntensity : f32
}
@group(0) @binding(5) var<uniform> uLighting : LightingParams;

// Point lights (max 16)
struct PointLight {
  position : vec3<f32>,
  intensity : f32,
  color : vec3<f32>,
  radius : f32
}
struct PointLightArray {
  count : u32,
  _pad : vec3<f32>,
  lights : array<PointLight, 16>
}
@group(0) @binding(6) var<storage, read> uPointLights : PointLightArray;

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) color : vec3<f32>,
  @location(3) uv : vec2<f32>,
  @location(4) textureIndex : f32,
  @location(5) blockType : f32
}

struct VSOut {
  @builtin(position) position : vec4<f32>,
  @location(0) normal : vec3<f32>,
  @location(1) color : vec3<f32>,
  @location(2) uv : vec2<f32>,
  @location(3) textureIndex : f32,
  @location(4) worldPos : vec3<f32>,
  @location(5) blockType : f32
}

struct FSOut {
  @location(0) color : vec4<f32>,
  @location(1) normalOut : vec4<f32>,
  @location(2) depthOut : vec4<f32>
}

@vertex
fn vs_main(in_ : VSIn) -> VSOut {
  var out : VSOut;
  out.position = uCamera.viewProj * vec4<f32>(in_.position, 1.0);
  out.normal = normalize(in_.normal);
  out.color = in_.color;
  out.uv = in_.uv;
  out.textureIndex = in_.textureIndex;
  out.worldPos = in_.position;
  out.blockType = in_.blockType;
  return out;
}

// PBR helper functions
const PI = 3.14159265359;

fn fresnelSchlick(cosTheta : f32, F0 : vec3<f32>) -> vec3<f32> {
  return F0 + (vec3<f32>(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn distributionGGX(N : vec3<f32>, H : vec3<f32>, roughness : f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let NdotH = max(dot(N, H), 0.0);
  let NdotH2 = NdotH * NdotH;
  let num = a2;
  var denom = (NdotH2 * (a2 - 1.0) + 1.0);
  denom = PI * denom * denom;
  return num / denom;
}

fn geometrySchlickGGX(NdotV : f32, roughness : f32) -> f32 {
  let r = (roughness + 1.0);
  let k = (r * r) / 8.0;
  let num = NdotV;
  let denom = NdotV * (1.0 - k) + k;
  return num / denom;
}

fn geometrySmith(N : vec3<f32>, V : vec3<f32>, L : vec3<f32>, roughness : f32) -> f32 {
  let NdotV = max(dot(N, V), 0.0);
  let NdotL = max(dot(N, L), 0.0);
  let ggx2 = geometrySchlickGGX(NdotV, roughness);
  let ggx1 = geometrySchlickGGX(NdotL, roughness);
  return ggx1 * ggx2;
}

@fragment
fn fs_main(in_ : VSOut) -> FSOut {
  let n = normalize(in_.normal);
  let v = normalize(uCamera.position - in_.worldPos);

  // Get base surface color from vertex color or texture
  var surfaceColor = in_.color;
  var surfaceAlpha = 1.0;
  if (in_.textureIndex >= 0.0) {
    var layer = i32(in_.textureIndex + 0.5);
    layer = clamp(layer, 0, i32(textureNumLayers(uTileTextures)) - 1);
    let sampleColor = textureSample(uTileTextures, uAtlasSampler, in_.uv, layer);
    surfaceAlpha = sampleColor.a;
    surfaceColor = in_.color * (1.0 - surfaceAlpha) + sampleColor.rgb * surfaceAlpha;
  }

  // Get material properties (default to simple diffuse if not set)
  let blockIdx = i32(in_.blockType);
  var albedo = surfaceColor;
  var roughness = 0.8;
  var metallic = 0.0;
  var ao = 1.0;
  var emissive = vec3<f32>(0.0);
  var emissiveStrength = 0.0;

  if (blockIdx >= 0 && blockIdx < 256) {
    let mat = uMaterials[blockIdx];
    if (length(mat.albedo) > 0.01) {
      albedo = mat.albedo * surfaceColor;
    }
    roughness = mat.roughness;
    metallic = mat.metallic;
    ao = mat.ao;
    emissive = mat.emissive;
    emissiveStrength = mat.emissiveStrength;
  }

  // PBR calculation
  let F0 = mix(vec3<f32>(0.04), albedo, metallic);
  var Lo = vec3<f32>(0.0);

  // Sun light (directional)
  let sunDir = normalize(uLighting.sunDirection);
  let L_sun = -sunDir;
  let H_sun = normalize(v + L_sun);
  let NdotL_sun = max(dot(n, L_sun), 0.0);

  if (NdotL_sun > 0.0) {
    let radiance = uLighting.sunColor * uLighting.sunIntensity;
    let NDF = distributionGGX(n, H_sun, roughness);
    let G = geometrySmith(n, v, L_sun, roughness);
    let F = fresnelSchlick(max(dot(H_sun, v), 0.0), F0);

    let numerator = NDF * G * F;
    let denominator = 4.0 * max(dot(n, v), 0.0) * NdotL_sun + 0.0001;
    let specular = numerator / denominator;

    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    Lo += (kD * albedo / PI + specular) * radiance * NdotL_sun;
  }

  // Point lights
  for (var i = 0u; i < uPointLights.count && i < 16u; i++) {
    let light = uPointLights.lights[i];
    let lightVec = light.position - in_.worldPos;
    let distance = length(lightVec);
    if (distance < light.radius) {
      let L_point = normalize(lightVec);
      let H_point = normalize(v + L_point);
      let NdotL_point = max(dot(n, L_point), 0.0);

      if (NdotL_point > 0.0) {
        let attenuation = pow(clamp(1.0 - (distance / light.radius), 0.0, 1.0), 2.0);
        let radiance = light.color * light.intensity * attenuation;

        let NDF = distributionGGX(n, H_point, roughness);
        let G = geometrySmith(n, v, L_point, roughness);
        let F = fresnelSchlick(max(dot(H_point, v), 0.0), F0);

        let numerator = NDF * G * F;
        let denominator = 4.0 * max(dot(n, v), 0.0) * NdotL_point + 0.0001;
        let specular = numerator / denominator;

        let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL_point;
      }
    }
  }

  // Sky ambient
  let skyFactor = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
  let skyAmbient = mix(uLighting.skyGround, mix(uLighting.skyHorizon, uLighting.skyZenith, skyFactor), skyFactor) * uLighting.skyIntensity;
  let ambient = (uLighting.ambientColor * uLighting.ambientIntensity + skyAmbient) * albedo * ao;

  var finalColor = ambient + Lo;

  // Add emissive
  if (emissiveStrength > 0.0) {
    finalColor += emissive * emissiveStrength;
  }

  // Output
  var outv : FSOut;
  outv.color = vec4<f32>(finalColor, surfaceAlpha);
  outv.normalOut = vec4<f32>(n * 0.5 + 0.5, 1.0);

  let near = uCamParams.x;
  let far = uCamParams.y;
  let z = clamp(in_.position.z, 0.0, 1.0);
  let z_ndc = z * 2.0 - 1.0;
  let linear = (2.0 * near * far) / (far + near - z_ndc * (far - near));
  let depth01 = clamp(linear / far, 0.0, 1.0);
  outv.depthOut = vec4<f32>(depth01, depth01, depth01, 1.0);

  return outv;
}
