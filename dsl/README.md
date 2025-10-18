Shared DSL
===========

Purpose
- Provide a single, canonical description of high‑level world edits that both
  the WebGPU renderer (TypeScript) and the differentiable PyTorch renderer
  (Python) consume.

Commands
- place_block({ position: [x,y,z], blockType: 'Stone', customBlockId? })
- remove_block({ position: [x,y,z] })

Types live in:
- TypeScript: dsl/ts/index.ts
- Python: dsl/python/__init__.py

Parsing
- Both TS and Python expose a parse function that turns free‑form text into a
  list of canonical DSL actions. Minimal token and JSON‑ish forms are accepted.

Usage
- WebGPU: src/webgpuEngine.ts imports parseDSL from dsl/ts and converts the
  resulting actions into internal WorldAction with BlockType enum mapping.
- Python: import dsl.python as the reference; apply_actions(world, actions) to
  mutate a WorldLike grid.

Notes
- BlockTypeName values must match names used in src/chunks.ts and
  model_stuff/materials.py.
- Keep rendering parameters (sun_dir, ambient, sky_color) in renderer configs;
  DSL describes scene content, not shading.

