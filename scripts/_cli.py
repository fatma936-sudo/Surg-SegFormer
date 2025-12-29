import argparse, yaml
from pathlib import Path

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--config", required=True, help="Path to YAML config.")
