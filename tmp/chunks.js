"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ChunkManager = exports.BlockType = void 0;
exports.buildChunkMesh = buildChunkMesh;
exports.setBlockTextureIndices = setBlockTextureIndices;
var BlockType;
(function (BlockType) {
    BlockType[BlockType["Air"] = 0] = "Air";
    BlockType[BlockType["Grass"] = 1] = "Grass";
    BlockType[BlockType["Dirt"] = 2] = "Dirt";
    BlockType[BlockType["Stone"] = 3] = "Stone";
    BlockType[BlockType["Plank"] = 4] = "Plank";
    BlockType[BlockType["Snow"] = 5] = "Snow";
    BlockType[BlockType["Sand"] = 6] = "Sand";
    BlockType[BlockType["Water"] = 7] = "Water";
})(BlockType || (exports.BlockType = BlockType = {}));
const blockTextureLayers = {};
const faceIndexToKey = {
    [0 /* FaceIndex.PX */]: 'east',
    [1 /* FaceIndex.NX */]: 'west',
    [2 /* FaceIndex.PY */]: 'top',
    [3 /* FaceIndex.NY */]: 'bottom',
    [4 /* FaceIndex.PZ */]: 'south',
    [5 /* FaceIndex.NZ */]: 'north'
};
const faceUVs = {
    [0 /* FaceIndex.PX */]: [
        [0, 1],
        [0, 0],
        [1, 0],
        [1, 1]
    ],
    [1 /* FaceIndex.NX */]: [
        [1, 1],
        [0, 1],
        [0, 0],
        [1, 0]
    ],
    [2 /* FaceIndex.PY */]: [
        [0, 1],
        [0, 0],
        [1, 0],
        [1, 1]
    ],
    [3 /* FaceIndex.NY */]: [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ],
    [4 /* FaceIndex.PZ */]: [
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0]
    ],
    [5 /* FaceIndex.NZ */]: [
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 1]
    ]
};
const faceDefs = {
    [0 /* FaceIndex.PX */]: {
        normal: [1, 0, 0],
        offset: [1, 0, 0],
        corners: [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1]
        ],
        colorSlot: 'side'
    },
    [1 /* FaceIndex.NX */]: {
        normal: [-1, 0, 0],
        offset: [-1, 0, 0],
        corners: [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0]
        ],
        colorSlot: 'side'
    },
    [2 /* FaceIndex.PY */]: {
        normal: [0, 1, 0],
        offset: [0, 1, 0],
        corners: [
            [0, 1, 0],
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0]
        ],
        colorSlot: 'top'
    },
    [3 /* FaceIndex.NY */]: {
        normal: [0, -1, 0],
        offset: [0, -1, 0],
        corners: [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 1]
        ],
        colorSlot: 'bottom'
    },
    [4 /* FaceIndex.PZ */]: {
        normal: [0, 0, 1],
        offset: [0, 0, 1],
        corners: [
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ],
        colorSlot: 'side'
    },
    [5 /* FaceIndex.NZ */]: {
        normal: [0, 0, -1],
        offset: [0, 0, -1],
        corners: [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0]
        ],
        colorSlot: 'side'
    }
};
const faceIndices = [0, 1, 2, 0, 2, 3];
const blockPalette = {
    [BlockType.Air]: undefined,
    [BlockType.Grass]: {
        top: [0.34, 0.68, 0.36],
        bottom: [0.40, 0.30, 0.16],
        side: [0.45, 0.58, 0.30]
    },
    [BlockType.Dirt]: {
        top: [0.42, 0.32, 0.20],
        bottom: [0.38, 0.26, 0.16],
        side: [0.40, 0.30, 0.18]
    },
    [BlockType.Stone]: {
        top: [0.58, 0.60, 0.64],
        bottom: [0.55, 0.57, 0.60],
        side: [0.56, 0.58, 0.62]
    },
    [BlockType.Plank]: {
        top: [0.78, 0.68, 0.50],
        bottom: [0.72, 0.60, 0.42],
        side: [0.74, 0.63, 0.45]
    },
    [BlockType.Snow]: {
        top: [0.92, 0.94, 0.96],
        bottom: [0.90, 0.92, 0.94],
        side: [0.88, 0.90, 0.93]
    },
    [BlockType.Sand]: {
        top: [0.88, 0.82, 0.60],
        bottom: [0.86, 0.78, 0.56],
        side: [0.87, 0.80, 0.58]
    },
    [BlockType.Water]: {
        top: [0.22, 0.40, 0.66],
        bottom: [0.20, 0.34, 0.60],
        side: [0.20, 0.38, 0.64]
    }
};
class ChunkManager {
    constructor(size = { x: 32, y: 32, z: 32 }) {
        this.size = size;
        this.blocks = new Uint8Array(size.x * size.y * size.z);
    }
    getBlock(x, y, z) {
        if (!this.inBounds(x, y, z))
            return BlockType.Air;
        return this.blocks[this.index(x, y, z)];
    }
    setBlock(x, y, z, type) {
        if (!this.inBounds(x, y, z))
            return;
        this.blocks[this.index(x, y, z)] = type;
    }
    snapshotBlocks() {
        return new Uint8Array(this.blocks);
    }
    generateDefaultTerrain(seed = 1) {
        const { x: sx, y: sy, z: sz } = this.size;
        for (let x = 0; x < sx; x++) {
            for (let z = 0; z < sz; z++) {
                const height = Math.min(sy - 1, Math.floor(6 + this.sampleElevation(x, z, seed) * (sy - 10)));
                for (let y = 0; y < sy; y++) {
                    if (y > height) {
                        this.setBlock(x, y, z, BlockType.Air);
                        continue;
                    }
                    if (y === height) {
                        this.setBlock(x, y, z, BlockType.Grass);
                    }
                    else if (y >= height - 2) {
                        this.setBlock(x, y, z, BlockType.Dirt);
                    }
                    else {
                        this.setBlock(x, y, z, BlockType.Stone);
                    }
                }
            }
        }
    }
    sampleElevation(x, z, seed) {
        const scale = 0.08;
        const p1 = this.valueNoise(x * scale, z * scale, seed);
        const p2 = this.valueNoise(x * scale * 2, z * scale * 2, seed + 73) * 0.5;
        const p3 = this.valueNoise(x * scale * 4, z * scale * 4, seed + 139) * 0.25;
        return (p1 + p2 + p3) / (1 + 0.5 + 0.25);
    }
    valueNoise(x, z, seed) {
        const xi = Math.floor(x);
        const zi = Math.floor(z);
        const xf = x - xi;
        const zf = z - zi;
        const h00 = this.hash(xi, zi, seed);
        const h10 = this.hash(xi + 1, zi, seed);
        const h01 = this.hash(xi, zi + 1, seed);
        const h11 = this.hash(xi + 1, zi + 1, seed);
        const u = smoothstep(xf);
        const v = smoothstep(zf);
        const x1 = mix(h00, h10, u);
        const x2 = mix(h01, h11, u);
        return mix(x1, x2, v);
    }
    hash(x, z, seed) {
        const h = Math.sin(x * 157.31 + z * 311.7 + seed * 93.13) * 43758.5453;
        return h - Math.floor(h);
    }
    inBounds(x, y, z) {
        return x >= 0 && y >= 0 && z >= 0 && x < this.size.x && y < this.size.y && z < this.size.z;
    }
    index(x, y, z) {
        return x + this.size.x * (z + this.size.z * y);
    }
}
exports.ChunkManager = ChunkManager;
function buildChunkMesh(chunk, worldScale = 1) {
    const vertices = [];
    const { x: sx, y: sy, z: sz } = chunk.size;
    const offsetX = -sx / 2;
    const offsetZ = -sz / 2;
    for (let y = 0; y < sy; y++) {
        for (let z = 0; z < sz; z++) {
            for (let x = 0; x < sx; x++) {
                const block = chunk.getBlock(x, y, z);
                if (block === BlockType.Air)
                    continue;
                const palette = blockPalette[block];
                const textureConfig = blockTextureLayers[block];
                for (let f = 0; f < 6; f++) {
                    const face = faceDefs[f];
                    const nx = x + face.offset[0];
                    const ny = y + face.offset[1];
                    const nz = z + face.offset[2];
                    if (chunk.getBlock(nx, ny, nz) !== BlockType.Air)
                        continue;
                    const color = palette[face.colorSlot];
                    const faceKey = faceIndexToKey[f];
                    const textureLayer = textureConfig ? textureConfig[faceKey] ?? -1 : -1;
                    const uvs = faceUVs[f];
                    const baseX = x + offsetX;
                    const baseY = y;
                    const baseZ = z + offsetZ;
                    for (let i = 0; i < faceIndices.length; i++) {
                        const idx = faceIndices[i];
                        const corner = face.corners[idx];
                        const uv = uvs[idx];
                        vertices.push((baseX + corner[0]) * worldScale, (baseY + corner[1]) * worldScale, (baseZ + corner[2]) * worldScale, face.normal[0], face.normal[1], face.normal[2], color[0], color[1], color[2], uv[0], uv[1], textureLayer);
                    }
                }
            }
        }
    }
    const vertexData = new Float32Array(vertices);
    const vertexCount = vertexData.length / 12;
    return { vertexData, vertexCount };
}
function setBlockTextureIndices(block, config) {
    if (!config) {
        delete blockTextureLayers[block];
        return;
    }
    blockTextureLayers[block] = { ...config };
}
function mix(a, b, t) {
    return a * (1 - t) + b * t;
}
function smoothstep(t) {
    return t * t * (3 - 2 * t);
}
