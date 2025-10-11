fn hash3(p: vec3<f32>) -> f32 {
  let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
  return fract(sin(h) * 43758.5453123);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 { return a + (b - a) * t; }

fn noise3(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let n000 = hash3(i + vec3<f32>(0.0,0.0,0.0));
  let n100 = hash3(i + vec3<f32>(1.0,0.0,0.0));
  let n010 = hash3(i + vec3<f32>(0.0,1.0,0.0));
  let n110 = hash3(i + vec3<f32>(1.0,1.0,0.0));
  let n001 = hash3(i + vec3<f32>(0.0,0.0,1.0));
  let n101 = hash3(i + vec3<f32>(1.0,0.0,1.0));
  let n011 = hash3(i + vec3<f32>(0.0,1.0,1.0));
  let n111 = hash3(i + vec3<f32>(1.0,1.0,1.0));
  let nx00 = lerp(n000, n100, u.x);
  let nx10 = lerp(n010, n110, u.x);
  let nx01 = lerp(n001, n101, u.x);
  let nx11 = lerp(n011, n111, u.x);
  let nxy0 = lerp(nx00, nx10, u.y);
  let nxy1 = lerp(nx01, nx11, u.y);
  return lerp(nxy0, nxy1, u.z);
}

fn fbm(p: vec3<f32>) -> f32 {
  var a = 0.5;
  var f = 1.0;
  var s = 0.0;
  for (var i = 0; i < 5; i = i + 1) {
    s = s + a * noise3(p * f);
    f = f * 2.0;
    a = a * 0.5;
  }
  return s;
}

