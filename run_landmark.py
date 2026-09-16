"""Run the repository's landmark stage with a task-local config.

``train.py`` hardcodes ``teethland/config/config.yaml``. Instead of modifying
the repository entry point, this runner temporarily substitutes that file with
the requested task-local config, runs the unchanged ``train.main()``, and then
restores the original ``config.yaml`` byte-for-byte (even on failure).

Usage
-----
python landmark_extension/run_landmark.py --config landmark_extension/configs/landmark_upper.yaml
python landmark_extension/run_landmark.py --config landmark_extension/configs/landmark_lower.yaml --devices 1
python landmark_extension/run_landmark.py --config landmark_extension/configs/landmark_upper.yaml --checkpoint <ckpt>

IMPORTANT: do not start this until the environment prerequisites in
landmark_extension/README.md are resolved and training has been authorised.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train import main as train_main  # noqa: E402  (must follow sys.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path,
                        help='Task-local YAML config to run.')
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument(
        '--stage', default='landmarks',
        help='Stage argument forwarded to train.py (default: landmarks).',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        raise SystemExit(f'config not found: {config_path}')

    repo_config = REPO_ROOT / 'teethland' / 'config' / 'config.yaml'
    backup = repo_config.with_suffix('.yaml.codex_backup')
    original = repo_config.read_bytes()

    try:
        backup.write_bytes(original)
        repo_config.write_bytes(config_path.read_bytes())
        print(f'[run_landmark] using {config_path}')
        train_main(args.stage, args.devices, args.checkpoint)
    finally:
        repo_config.write_bytes(original)
        if backup.exists() and backup.read_bytes() == original:
            backup.unlink()
        print('[run_landmark] restored teethland/config/config.yaml')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
