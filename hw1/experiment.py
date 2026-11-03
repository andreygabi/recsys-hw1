"""Validation search, frozen configuration, then one final test comparison."""

import gc
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from evaluation import evaluate, evaluation_users
from models import CustomTargetALS, EASE, FM, IALS, ItemKNN


MODEL_CLASSES = {'knn': ItemKNN, 'ials': IALS, 'ease': EASE, 'fm': FM, 'custom_als': CustomTargetALS}


def grid(space):
    return [dict(zip(space, values)) for values in product(*space.values())]


def make_model(name, params, config, use_metadata=True):
    options = dict(params)
    if name in ('ials', 'custom_als'):
        options.update(iterations=config['als_iterations'], seed=config['seed'])
    if name == 'fm':
        options.update(epochs=config['fm_epochs'], learning_rate=config['fm_learning_rate'],
                       negative_ratio=config['fm_negative_ratio'], batch_size=config['fm_batch_size'],
                       seed=config['seed'], use_metadata=use_metadata)
    return MODEL_CLASSES[name](**options)


def fit(model, frame, bundle, cutoff):
    started = perf_counter()
    model.fit(frame, bundle['shape'], codes=bundle['codes'], cutoff=cutoff)
    return perf_counter() - started


def run_validation(bundle, config, output_dir='.cache/results'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train, validation = (bundle['parts'][part] for part in ('train', 'validation'))
    users = evaluation_users(train, validation, bundle['shape'])
    records, selected = [], {}
    for name in MODEL_CLASSES:
        trials = grid(config[name])
        if name == 'custom_als':
            trials = [{**selected['ials'], **target_params} for target_params in trials]
        best_score = -np.inf
        for params in trials:
            model = make_model(name, params, config)
            seconds = fit(model, train, bundle, bundle['cuts'][0])
            metrics, _, _ = evaluate(model, users, train, validation, bundle['shape'])
            record = {'model': name, 'params': params, **metrics, 'fit_seconds': seconds}
            records.append(record)
            print(f'{name:10s} {params} NDCG={metrics["NDCG@10"]:.6f} '
                  f'Recall={metrics["Recall@10"]:.6f} fit={seconds:.1f}s', flush=True)
            if metrics[config['primary_metric']] > best_score:
                best_score = metrics[config['primary_metric']]
                selected[name] = params
            (output / 'validation.json').write_text(json.dumps(records, indent=2))
            del model
            gc.collect()
    model = make_model('fm', selected['fm'], config, use_metadata=False)
    seconds = fit(model, train, bundle, bundle['cuts'][0])
    ablation, _, _ = evaluate(model, users, train, validation, bundle['shape'])
    ablation.update(model='fm_ids_only', params=selected['fm'], fit_seconds=seconds)
    print(f'FM IDs-only ablation: {ablation}', flush=True)
    (output / 'selected_config.json').write_text(json.dumps(selected, indent=2))
    return pd.DataFrame(records), selected, ablation, users


def run_test(bundle, config, selected, output_dir='.cache/results'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Save the choice before any test score is computed.
    (output / 'selected_config.json').write_text(json.dumps(selected, indent=2))
    history = pd.concat([bundle['parts']['train'], bundle['parts']['validation']], ignore_index=True)
    test = bundle['parts']['test']
    users = evaluation_users(history, test, bundle['shape'])
    records, recommendations, per_user, losses = [], {}, {}, {}
    for name, params in selected.items():
        model = make_model(name, params, config)
        seconds = fit(model, history, bundle, bundle['cuts'][1])
        metrics, top, ndcg = evaluate(model, users, history, test, bundle['shape'])
        recommendations[name], per_user[name] = top, ndcg
        if hasattr(model, 'losses'):
            losses[name] = model.losses
        records.append({'model': name, **metrics, 'fit_seconds': seconds, 'params': json.dumps(params)})
        print(f'Test {name:10s}: {metrics}; fit={seconds:.1f}s', flush=True)
        del model
        gc.collect()
    results = pd.DataFrame(records)
    results.to_csv(output / 'test_metrics.csv', index=False)
    return results, recommendations, per_user, losses, users


def paired_bootstrap(first, second, seed=42, repeats=2000):
    differences = np.asarray(first) - np.asarray(second)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(differences, len(differences), replace=True).mean() for _ in range(repeats)])
    return float(differences.mean()), np.quantile(means, [0.025, 0.975])
