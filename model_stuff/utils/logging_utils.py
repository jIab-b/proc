"""Logging helpers shared across training scripts."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def configure_logging(run_dir: Path, verbose: bool = True) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    
    root_logger.info("Logging initialised. Writing to %s", log_path)
    return log_path


def create_run_directory(base_dir: Path, run_name: Optional[str] = None) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or "train"
    run_dir = base_dir / "logs" / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
