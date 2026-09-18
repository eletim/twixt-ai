"""Streaming Policy/Value comparisons on identical, game-held-out positions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from twixt_ai.game import GameState, legal_peg_placements
from twixt_ai.models import load_policy_value_checkpoint
from twixt_ai.models.versioned_encoding import encode_position_for_version, move_to_action_index_for_version


def phase(ply: int) -> str:
    return 'early' if ply < 16 else 'middle' if ply < 40 else 'late'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def examples(root: Path, splits=('train', 'validation')):
    manifest = json.loads((root / 'manifest.json').read_text())
    for split in splits:
        seen = 0
        for shard in manifest['splits'][split]['shards']:
            path = root / shard['path']
            if not path.resolve().is_relative_to(root.resolve()) or sha(path) != shard['sha256']:
                raise ValueError(f'invalid shard: {path}')
            count = 0
            with path.open() as stream:
                for line in stream:
                    count += 1
                    yield split, json.loads(line)
            if count != shard['examples']:
                raise ValueError(f'shard count mismatch: {path}')
            seen += count
        if seen != manifest['splits'][split]['examples']:
            raise ValueError(f'{split} count mismatch')


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    return float(-(p[p > 0] * np.log(p[p > 0])).sum())


def concentration(p):
    ordered = np.sort(p)[::-1]
    return {'entropy': entropy(p), 'top1_probability': float(ordered[0]),
            'top3_cumulative_probability': float(ordered[:3].sum()),
            'top5_cumulative_probability': float(ordered[:5].sum())}


class Means:
    def __init__(self):
        self.count = 0
        self.total = defaultdict(float)

    def add(self, values):
        self.count += 1
        for key, value in values.items():
            self.total[key] += float(value)

    def result(self):
        return {'positions': self.count, **{k: v / self.count for k, v in self.total.items()}} if self.count else {'positions': 0}


def diagnose_targets(root: Path):
    groups = defaultdict(Means)
    games = {'train': set(), 'validation': set()}
    for split, example in examples(root):
        source = example['source']
        games[split].add(source['game_id'])
        moves = source['decision']['metadata']['root_moves']
        prior = np.array([m['prior'] for m in moves], dtype=np.float64)
        visit = np.array([m['visits'] for m in moves], dtype=np.float64)
        visit /= visit.sum()
        if not np.isclose(prior.sum(), 1):
            raise ValueError('prior not normalized')
        # Canonical argmax has the existing row-major tie break. Also report
        # tie-accepting agreement because 64 simulations create visit ties.
        p1, v1 = int(prior.argmax()), int(visit.argmax())
        selected = next(i for i, m in enumerate(moves)
                        if (m['x'], m['y']) == (example['action']['x'], example['action']['y']))
        tv = float(np.abs(prior - visit).sum() / 2)
        values = {**concentration(visit), 'legal_move_count': len(moves),
                  'normalized_entropy_legal': entropy(visit) / np.log(len(moves)) if len(moves) > 1 else 0,
                  'support': int((visit > 0).sum()),
                  'prior_entropy': entropy(prior), 'prior_top1_visit_top1_agreement': p1 == v1,
                  'prior_top1_in_visit_max_ties': visit[p1] == visit.max(),
                  'prior_top1_selected_move_agreement': p1 == selected,
                  'mcts_top1_changed_fraction': p1 != v1,
                  'prior_visit_total_variation': tv,
                  'mcts_tv_over_0_1_fraction': tv > .1,
                  'visit_minus_prior_entropy': entropy(visit) - entropy(prior)}
        for group in ('overall', split, phase(source['ply']), f'{split}/{phase(source["ply"])}',
                      f"side/{example['position']['side_to_move']}"):
            groups[group].add(values)
    if games['train'] & games['validation']:
        raise ValueError('game split leakage')
    return {'manifest_sha256': sha(root / 'manifest.json'),
            'phase_definition': 'early ply 0-15; middle 16-39; late >=40',
            'definitions': {'entropy': 'nats, full legal support including zero visits',
                            'mcts_correction': 'argmax changed and total variation >0.1 reported separately'},
            'game_split_audit': {k: len(v) for k, v in games.items()},
            'groups': {k: v.result() for k, v in groups.items()}}


def diagnose_models(root: Path, checkpoints: dict[str, Path], *, batch_size=512, device='cuda'):
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this model comparison')
    loaded = {name: load_policy_value_checkpoint(path, map_location=device).model.eval()
              for name, path in checkpoints.items()}
    versions = {model.config.encoding_version for model in loaded.values()}
    if len(versions) != 1:
        raise ValueError('comparisons must share encoding')
    encoding = versions.pop()
    policy = {name: defaultdict(Means) for name in loaded}
    value = {name: defaultdict(Means) for name in loaded}
    decisive = {name: defaultdict(Means) for name in loaded}
    comparisons = {name: defaultdict(Means) for name in loaded if name != 'bootstrap'}
    prediction_bins = {name: defaultdict(Means) for name in loaded}
    rows = []
    def consume():
        if not rows:
            return
        inputs = torch.stack([r[0] for r in rows]).to(device)
        masks = np.stack([r[1] for r in rows])
        targets = np.stack([r[2] for r in rows])
        outcomes = np.array([r[3] for r in rows])
        predictions = {}
        values = {}
        with torch.inference_mode():
            for name, model in loaded.items():
                logits, estimates = model(inputs)
                logits = logits.masked_fill(~torch.as_tensor(masks, device=device), -torch.inf)
                predictions[name] = torch.softmax(logits, 1).cpu().numpy().astype(np.float64)
                values[name] = estimates.cpu().numpy().astype(np.float64)
        for i, row in enumerate(rows):
            groups = ('overall', phase(row[4]), f'side/{row[5]}')
            target = targets[i]
            legal_indices = np.flatnonzero(masks[i])
            ttop = legal_indices[np.argsort(-target[legal_indices], kind='stable')[:3]]
            t1 = int(target.argmax())
            for name, probs in predictions.items():
                p = probs[i]
                ptop = legal_indices[np.argsort(-p[legal_indices], kind='stable')[:3]]
                p1 = int(p.argmax())
                ce = float(-(target[target > 0] * np.log(np.maximum(p[target > 0], 1e-30))).sum())
                metrics = {**concentration(p), 'cross_entropy': ce,
                           'top1_visit_agreement': p1 == t1,
                           'top1_in_visit_max_ties': target[p1] == target.max(),
                           'top3_visit_set_overlap': len(set(ptop) & set(ttop)) / min(3,len(legal_indices)),
                           'visit_top1_in_policy_top3': t1 in ptop,
                           'policy_top3_visit_mass': float(target[ptop].sum())}
                estimate = float(values[name][i]); outcome = float(outcomes[i])
                vm = {'mse': (estimate-outcome)**2, 'mae': abs(estimate-outcome),
                      'prediction': estimate, 'target': outcome, 'calibration_bias': estimate-outcome}
                for group in groups:
                    policy[name][group].add(metrics)
                    value[name][group].add(vm)
                    if outcome != 0:
                        decisive[name][group].add({'sign_error': (estimate >= 0) != (outcome > 0)})
                confidence = min(4, int(abs(estimate) * 5))
                value[name][f'confidence-{confidence}'].add(vm)
                if outcome != 0:
                    decisive[name][f'confidence-{confidence}'].add({'sign_error': (estimate >= 0) != (outcome > 0)})
                index = min(9, max(0, int((estimate+1)*5)))
                prediction_bins[name][str(index)].add(vm)
                if name != 'bootstrap' and 'bootstrap' in predictions:
                    b = predictions['bootstrap'][i]
                    cm = {'bootstrap_top1_agreement': p1 == int(b.argmax()),
                          'bootstrap_total_variation': float(np.abs(p-b).sum()/2),
                          'kl_bootstrap_to_model': float((b[b > 0]*np.log(b[b > 0]/np.maximum(p[b > 0],1e-30))).sum()),
                          'entropy_delta_vs_bootstrap': entropy(p)-entropy(b)}
                    for group in groups:
                        comparisons[name][group].add(cm)
        rows.clear()
    for _, example in examples(root, ('validation',)):
        state = GameState.from_dict(example['position'])
        moves = legal_peg_placements(state)
        mask = np.zeros(100, dtype=bool)
        for move in moves:
            mask[move_to_action_index_for_version(move, encoding, board_width=10, board_height=10)] = True
        target = np.zeros(100, dtype=np.float64)
        from twixt_ai.game import Coordinate, PegPlacement
        for entry in example['policy']:
            move = PegPlacement(state.side_to_move, Coordinate(**entry['coordinate']))
            target[move_to_action_index_for_version(move, encoding, board_width=10, board_height=10)] = entry['probability']
        if not np.isclose(target.sum(), 1) or np.any(target[~mask] != 0):
            raise ValueError('invalid visit target')
        rows.append((encode_position_for_version(state, encoding), mask, target, example['outcome'], example['source']['ply'], state.side_to_move.value))
        if len(rows) == batch_size:
            consume()
    consume()
    policy_report = {'manifest_sha256': sha(root/'manifest.json'), 'split': 'validation',
                     'checkpoints': {k: {'path': str(p), 'sha256': sha(p)} for k,p in checkpoints.items()},
                     'phase_definition': 'early 0-15; middle 16-39; late >=40',
                     'metrics': {name: {k: v.result() for k,v in groups.items()} for name,groups in policy.items()},
                     'vs_bootstrap': {name: {k: v.result() for k,v in groups.items()} for name,groups in comparisons.items()},
                     'definitions': {'top1': 'canonical row-major argmax; tied maximum acceptance also reported',
                                     'top3_visit_set_overlap': 'intersection size / min(3, legal count)',
                                     'policy_top3_visit_mass': 'visit target probability covered by NN top3',
                                     'CE': 'legal-masked probabilities; training CE remains unmasked'}}
    value_report = {'manifest_sha256': sha(root/'manifest.json'), 'split': 'validation',
                    'checkpoints': policy_report['checkpoints'], 'perspective': 'side-to-move',
                    'metrics': {}, 'definitions': {'calibration_error': 'weighted absolute predicted-bin mean minus outcome-bin mean (10 bins)',
                                                  'sign_error': 'negative prediction for +1 or nonnegative for -1; draw excluded',
                                                  'confidence_bins': '[0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]'}}
    for name, groups in value.items():
        bins = {k: v.result() for k,v in prediction_bins[name].items()}
        n = groups['overall'].count
        value_report['metrics'][name] = {'groups': {k: v.result() for k,v in groups.items()},
                                        'decisive': {k:v.result() for k,v in decisive[name].items()},
                                        'calibration_bins': bins,
                                        'calibration_error': sum(b['positions']*abs(b['calibration_bias']) for b in bins.values())/n if n else None}
    return policy_report, value_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', default=[], help='NAME=PATH')
    parser.add_argument('--targets', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def write(name, data):
        (args.output_dir/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    torch.set_num_threads(4)
    if args.targets:
        write('dataset-diagnostics.json', diagnose_targets(args.dataset))
    if args.checkpoint:
        p,v=diagnose_models(args.dataset,{s.split('=',1)[0]:Path(s.split('=',1)[1]) for s in args.checkpoint})
        write('policy-diagnostics.json',p); write('value-diagnostics.json',v)

if __name__ == '__main__':
    main()
