Shared DSL
===========

Purpose
- Provide a single, canonical description of high‑level world edits that both
  the WebGPU renderer (TypeScript) and the differentiable PyTorch renderer
  (Python) consume.

Commands
- place_block({ position: [x,y,z], blockType: 'Stone', customBlockId? })
- remove_block({ position: [x,y,z] })

Canonical source
- Python owns the DSL (pydantic models) in dsl/python/__init__.py
- JSON Schema is served by backend at GET /api/dsl-schema
- Frontend obtains canonical actions by POST /api/parse-dsl { text }
  and receives { dslVersion, actions }

TypeScript usage
- Do not maintain a separate TS DSL. Frontend always calls /api/parse-dsl to
  obtain canonical actions and then applies them.

Parsing
- Python exposes a parse function that turns free‑form text into canonical
  actions. The frontend calls /api/parse-dsl to obtain these actions.

Usage
- WebGPU: src/webgpuEngine.ts posts free‑form text to /api/parse-dsl for
  canonical actions, then converts them into internal WorldAction with BlockType
  enum mapping.
- Python: import dsl.python as the reference; apply_actions(world, actions) to
  mutate a WorldLike grid.

Notes
- BlockTypeName values must match names used in src/chunks.ts and
  model_stuff/materials.py.
- Keep rendering parameters (sun_dir, ambient, sky_color) in renderer configs;
  DSL describes scene content, not shading.
