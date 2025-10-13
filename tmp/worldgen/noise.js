"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.valueNoise2D = valueNoise2D;
exports.fbmNoise2D = fbmNoise2D;
function valueNoise2D(x, z, seed) {
    const xi = Math.floor(x);
    const zi = Math.floor(z);
    const xf = x - xi;
    const zf = z - zi;
    const h00 = hash2D(xi, zi, seed);
    const h10 = hash2D(xi + 1, zi, seed);
    const h01 = hash2D(xi, zi + 1, seed);
    const h11 = hash2D(xi + 1, zi + 1, seed);
    const u = smoothstep(xf);
    const v = smoothstep(zf);
    const x1 = lerp(h00, h10, u);
    const x2 = lerp(h01, h11, u);
    return lerp(x1, x2, v);
}
function fbmNoise2D(x, z, seed, octaves = 4, lacunarity = 2, gain = 0.5) {
    let freq = 1;
    let amp = 1;
    let sum = 0;
    let max = 0;
    for (let i = 0; i < octaves; i++) {
        sum += valueNoise2D(x * freq, z * freq, seed + i * 131) * amp;
        max += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    return max > 0 ? sum / max : 0;
}
function lerp(a, b, t) {
    return a * (1 - t) + b * t;
}
function smoothstep(t) {
    return t * t * (3 - 2 * t);
}
function hash2D(x, z, seed) {
    let h = seed >>> 0;
    h ^= Math.imul(0x27d4eb2d, x);
    h = (h ^ (h >>> 15)) >>> 0;
    h ^= Math.imul(0x165667b1, z);
    h = (h ^ (h >>> 13)) >>> 0;
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
