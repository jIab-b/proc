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
    training_path = Path(training_dir)
    maps_path = Path(maps_base_dir)

    optimized_map = training_path / "map_optimized.json"

    if not optimized_map.exists():
        raise FileNotFoundError(f"Optimized map not found: {optimized_map}")

    with open(optimized_map) as f:
        map_data = json.load(f)

    num_blocks = len(map_data.get('blocks', []))

    if output_name:
        output_dir_name = output_name
    else:
        next_id = find_next_trained_map_id(maps_path, source_map_id)
        output_dir_name = f"trained_map_{source_map_id}_{next_id}"

    output_dir = maps_path / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_map_path = output_dir / "map.json"
    shutil.copy2(optimized_map, output_map_path)

    print(f"✅ Exported to {output_dir_name}/map.json ({num_blocks} blocks)")

    return output_dir


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
