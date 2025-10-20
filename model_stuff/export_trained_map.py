"""
Export trained map to maps/ directory with proper naming.

Copies the optimized map from training output to maps/ directory
with name based on the source map ID.

Usage:
    python -m model_stuff.export_trained_map --training_dir out_local/sds_training --source_map_id 1
"""

import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def find_next_trained_map_id(base_dir: Path, source_map_id: int) -> int:
    """
    Find the next available trained_map_{source_map_id}_{x} ID.

    Args:
        base_dir: maps/ directory
        source_map_id: Source map ID

    Returns:
        Next available ID number
    """
    pattern = f"trained_map_{source_map_id}_"
    max_id = 0

    for dir_path in base_dir.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith(pattern):
            try:
                # Extract the number after the pattern
                id_str = dir_path.name[len(pattern):]
                id_num = int(id_str)
                max_id = max(max_id, id_num)
            except ValueError:
                continue

    return max_id + 1


def export_trained_map(
    training_dir: str,
    source_map_id: int,
    output_name: str = None,
    maps_base_dir: str = "maps"
) -> Path:
    """
    Export trained map to maps/ directory.

    Args:
        training_dir: Directory containing training output
        source_map_id: Source map ID (e.g., 1 for maps/1/map.json)
        output_name: Optional custom output name
        maps_base_dir: Base maps directory

    Returns:
        Path to exported map directory
    """
    training_path = Path(training_dir)
    maps_path = Path(maps_base_dir)

    # Find the optimized map
    optimized_map = training_path / "map_optimized.json"

    if not optimized_map.exists():
        raise FileNotFoundError(f"Optimized map not found: {optimized_map}")

    # Load the map to get metadata
    with open(optimized_map) as f:
        map_data = json.load(f)

    num_blocks = len(map_data.get('blocks', []))
    metadata = map_data.get('metadata', {})

    print(f"Loading trained map from {optimized_map}")
    print(f"  Blocks: {num_blocks}")
    if 'prompt' in metadata:
        print(f"  Prompt: '{metadata['prompt']}'")
    if 'steps' in metadata:
        print(f"  Training steps: {metadata['steps']}")

    # Determine output directory name
    if output_name:
        output_dir_name = output_name
    else:
        # Auto-generate: trained_map_{source_id}_{x}
        next_id = find_next_trained_map_id(maps_path, source_map_id)
        output_dir_name = f"trained_map_{source_map_id}_{next_id}"

    output_dir = maps_path / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy map.json
    output_map_path = output_dir / "map.json"
    shutil.copy2(optimized_map, output_map_path)

    print(f"\nExported to {output_dir}/")
    print(f"  Map: map.json ({num_blocks} blocks)")

    # Create metadata file with training info
    export_metadata = {
        "exportedAt": datetime.now().isoformat(),
        "sourceMap": f"maps/{source_map_id}/map.json",
        "trainingDir": str(training_path),
        "prompt": metadata.get('prompt', 'unknown'),
        "trainingSteps": metadata.get('steps', 'unknown'),
        "cfgScale": metadata.get('cfg_scale', 'unknown'),
        "numBlocks": num_blocks
    }

    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(export_metadata, f, indent=2)

    print(f"  Metadata: training_metadata.json")

    # Copy training logs if available
    log_file = training_path / "train.jsonl"
    if log_file.exists():
        output_log = output_dir / "train.jsonl"
        shutil.copy2(log_file, output_log)
        print(f"  Logs: train.jsonl")

    # Copy sample images if available
    images_dir = training_path / "images"
    if images_dir.exists() and images_dir.is_dir():
        output_images_dir = output_dir / "images"
        output_images_dir.mkdir(exist_ok=True)

        # Copy first, last, and a few intermediate images
        image_files = sorted(images_dir.glob("step_*.png"))

        if image_files:
            # First image
            shutil.copy2(image_files[0], output_images_dir / image_files[0].name)

            # Last image
            shutil.copy2(image_files[-1], output_images_dir / image_files[-1].name)

            # A few intermediate (sample every ~20%)
            num_samples = min(3, len(image_files) - 2)
            if num_samples > 0:
                step_size = len(image_files) // (num_samples + 1)
                for i in range(1, num_samples + 1):
                    idx = i * step_size
                    if idx < len(image_files):
                        shutil.copy2(image_files[idx], output_images_dir / image_files[idx].name)

            print(f"  Images: {len(list(output_images_dir.glob('*.png')))} sample images")

    # Update maps registry
    update_registry(maps_path, output_dir_name, export_metadata)

    print(f"\n✅ Export complete: {output_dir_name}")
    print(f"\nTo view in WebGPU editor:")
    print(f"  Load map: {output_dir_name}")

    return output_dir


def update_registry(maps_base_dir: Path, map_name: str, metadata: dict):
    """
    Update maps/registry.json with new trained map.

    Args:
        maps_base_dir: maps/ directory
        map_name: Name of the map directory
        metadata: Export metadata
    """
    registry_path = maps_base_dir / "registry.json"

    # Load existing registry or create new
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"maps": []}

    # Check if entry already exists
    existing_idx = None
    for i, entry in enumerate(registry.get('maps', [])):
        if entry.get('name') == map_name:
            existing_idx = i
            break

    # Create entry
    entry = {
        "name": map_name,
        "type": "trained",
        "sourceMap": metadata.get('sourceMap', 'unknown'),
        "prompt": metadata.get('prompt', 'unknown'),
        "exportedAt": metadata.get('exportedAt'),
        "numBlocks": metadata.get('numBlocks', 0)
    }

    # Add or update
    if existing_idx is not None:
        registry['maps'][existing_idx] = entry
    else:
        if 'maps' not in registry:
            registry['maps'] = []
        registry['maps'].append(entry)

    # Save registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)

    print(f"  Updated registry: {registry_path}")


def main():
    parser = argparse.ArgumentParser(description="Export trained map to maps/ directory")

    parser.add_argument("--training_dir", type=str, default="out_local/sds_training",
                        help="Training output directory")
    parser.add_argument("--source_map_id", type=int, required=True,
                        help="Source map ID (e.g., 1 for maps/1/map.json)")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Custom output directory name (optional)")
    parser.add_argument("--maps_base_dir", type=str, default="maps",
                        help="Base maps directory")

    args = parser.parse_args()

    try:
        export_trained_map(
            training_dir=args.training_dir,
            source_map_id=args.source_map_id,
            output_name=args.output_name,
            maps_base_dir=args.maps_base_dir
        )
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
