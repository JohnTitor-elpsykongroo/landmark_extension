"""Score landmark predictions with the official 3DTeethLand evaluation code.

The challenge metric implementation lives in ``evaluation/3dteethland.py`` and
expects

  * ``--predictions_file`` : CSV with columns
        key, coord_x, coord_y, coord_z, class, score
  * ``--goldstandard_file`` : a pickle of
        {classname: {meshname: [coord, ...]}}  (see evaluation/prepare_gt.py)

This script builds the gold standard from the local Teeth3DS+ landmark JSONs
and then calls the repository's own ``score``/``reformat_scores`` functions, so
the numbers are directly comparable to the challenge leaderboard.

Usage
-----
python landmark_extension/eval/score_predictions.py make-gold --jaw upper --fold landmark_extension/prepare/landmark_upper_fold_0.txt
python landmark_extension/eval/score_predictions.py score --predictions landmark_extension/preds/upper/predictions.csv --goldstand landmark_extension/eval/gold_upper_landmark_upper_fold_0.pkl --out landmark_extension/eval/scores_upper_fold_0.json

The gold standard pickle stores coordinates in the original scan frame, which is
what ``FullNet.save_landmarks`` writes as well, so no extra alignment is needed.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path(r'D:\WorkSpace\Dental\data\Teeth3DS')
CLASSES = ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'Cusp']


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)

    gold = sub.add_parser('make-gold', help='build the gold standard pickle')
    gold.add_argument('--jaw', default='upper', choices=['upper', 'lower'])
    gold.add_argument(
        '--fold', type=Path, default=None,
        help='Fold file listing validation stems (default: all landmark cases).',
    )
    gold.add_argument('--split', default='train', choices=['train', 'test'])
    gold.add_argument('--out', type=Path, default=None)

    score = sub.add_parser('score', help='score a predictions CSV')
    score.add_argument('--predictions', type=Path, required=True)
    score.add_argument('--goldstand', type=Path, required=True)
    score.add_argument('--out', type=Path, required=True)
    return p.parse_args()


def stem_of_fold_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ''
    # fold files are written as "<case>_<jaw>.obj" by
    # prepare_landmark_training_data.py
    return line[:-4] if line.lower().endswith('.obj') else line


def make_gold(args: argparse.Namespace) -> int:
    if args.out is None:
        suffix = args.fold.stem if args.fold else 'all'
        args.out = Path(__file__).resolve().parent / f'gold_{args.jaw}_{suffix}.pkl'

    if args.fold is not None:
        wanted = {
            stem_of_fold_line(line)
            for line in args.fold.read_text().splitlines()
            if stem_of_fold_line(line)
        }
        print(f'fold {args.fold.name}: {len(wanted)} stems')
    else:
        wanted = None

    root = DATA_ROOT / f'3DTeethLand_landmarks_{args.split}' / args.jaw
    out = {label: {} for label in CLASSES}
    num_cases = 0

    for kpt_file in sorted(root.rglob('*.json')):
        stem = kpt_file.name.split('__')[0]
        if wanted is not None and stem not in wanted:
            continue
        with open(kpt_file, 'r') as f:
            kpt_dict = json.load(f)
        for label in CLASSES:
            coords = [
                obj['coord'][:3]
                for obj in kpt_dict['objects']
                if obj['class'] == label
            ]
            out[label][stem] = [[float(c) for c in c[:3]] for c in coords]
        num_cases += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)

    counts = {label: sum(len(v) for v in out[label].values()) for label in CLASSES}
    print(f'gold standard: {num_cases} cases -> {args.out}')
    print('landmarks per class:', counts)
    return 0


def score(args: argparse.Namespace) -> int:
    import pandas as pd

    sys.path.insert(0, str(REPO_ROOT / 'evaluation'))
    from importlib import import_module
    evaluation = import_module('3dteethland')

    predictions = pd.read_csv(args.predictions)
    pred_all_map = {label: {} for label in CLASSES}
    for _, row in predictions.iterrows():
        class_name = row['class']
        if class_name not in pred_all_map:
            continue
        key = row['key']
        coord = [row['coord_x'], row['coord_y'], row['coord_z']]
        prob = row['score']
        if key not in pred_all_map[class_name]:
            pred_all_map[class_name][key] = [[coord, prob]]
        else:
            pred_all_map[class_name][key].append([coord, prob])

    with open(args.goldstand, 'rb') as f:
        gold = pickle.load(f)

    scores = evaluation.reformat_scores(evaluation.score(gold, pred_all_map))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'submission_status': 'SCORED', **scores}, f, indent=2)
    print(json.dumps({'submission_status': 'SCORED', **scores}, indent=2))
    print('written:', args.out)
    return 0


def main() -> int:
    args = parse_args()
    if args.command == 'make-gold':
        return make_gold(args)
    return score(args)


if __name__ == '__main__':
    raise SystemExit(main())
