@group(0) @binding(0) var<storage, read> triCount : array<u32>;
@group(0) @binding(1) var<storage, read_write> scanOut : array<u32>;
@group(0) @binding(2) var<storage, read_write> totalOut : array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x != 0u) { return; }
  var sum : u32 = 0u;
  for (var i: u32 = 0u; i < arrayLength(&triCount); i = i + 1u) {
    scanOut[i] = sum;
    sum = sum + triCount[i];
  }
  totalOut[0] = sum;
}
 

