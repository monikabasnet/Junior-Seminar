"""Normal-only Isolation Forest, train-file selection/calibration and explicit test audit."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from utils import (ROOT, MODEL_PATH, NUMERIC, CATEGORICAL, FEATURES, clean, labels,
                   predict, evaluate, signatures, data_quality, attack_breakdown, predict_supervised)


def make_pipeline(transform):
    steps = [('impute', SimpleImputer(strategy='median'))]
    if transform == 'log1p':
        steps.append(('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')))
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline(steps), NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10,
                                      sparse_output=False), CATEGORICAL)])
    return Pipeline([('preprocessing', preprocessing), ('model', IsolationForest(
        n_estimators=250, max_samples=512, contamination='auto', random_state=42, n_jobs=-1))])


def make_supervised_pipeline():
    """Fixed, untuned supervised reference with its own fitting-only preprocessing."""
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scale', StandardScaler())])
    preprocessing = ColumnTransformer([
        ('numeric', numeric, NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                     min_frequency=10, sparse_output=False), CATEGORICAL)])
    return Pipeline([('preprocessing', preprocessing),
        ('model', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default=str(ROOT / 'data/UNSW_NB15_training-set.csv'))
    parser.add_argument('--test', default=str(ROOT / 'data/UNSW_NB15_testing-set.csv'))
    parser.add_argument('--max-normal', type=int, default=50000)
    parser.add_argument('--false-alarm-budget', type=float, default=.05)
    args = parser.parse_args()
    if args.max_normal < 100 or not 0 < args.false_alarm_budget < .5:
        parser.error('Use max-normal >= 100 and 0 < false-alarm-budget < .5.')
    start = time.perf_counter()
    try:
        training = pd.read_csv(args.train)
        x = clean(training)
        y = labels(training)
        hashes = signatures(x)
        # Keep feature-equivalent observations in a single partition. Exclude ambiguous
        # groups (identical inputs with conflicting labels) from development only.
        group_labels = pd.DataFrame({'group': hashes, 'label': y}).groupby('group').label.nunique()
        ambiguous = set(group_labels[group_labels > 1].index)
        keep = ~hashes.isin(ambiguous) & ~hashes.duplicated()
        unique = training.loc[keep].reset_index(drop=True)
        fit, remainder = train_test_split(unique, test_size=.4, stratify=labels(unique), random_state=42)
        selection, calibration = train_test_split(remainder, test_size=.5,
                                                  stratify=labels(remainder), random_state=42)
        normal = clean(fit.loc[labels(fit) == 0])
        normal = normal.sample(min(len(normal), args.max_normal), random_state=42)
        if len(normal) < 100 or normal[NUMERIC].isna().all().any():
            raise ValueError('Need >=100 normal fitting flows and an observed value for each numeric feature.')
        normal_calibration = clean(calibration.loc[labels(calibration) == 0])
        if len(normal_calibration) < 20:
            raise ValueError('Not enough normal calibration records.')
        candidates = []
        pipelines = {}
        # This two-candidate comparison is fixed before test evaluation. Validation
        # attack labels guide transform selection; forest fitting is still normal-only.
        for transform in ['raw', 'log1p']:
            pipeline = make_pipeline(transform)
            pipeline.fit(normal)
            cutoff = float(np.quantile(-pipeline.score_samples(normal), 1-args.false_alarm_budget))
            temp = {'pipeline': pipeline, 'schema_version': 2, 'threshold': cutoff}
            metrics = evaluate(selection, predict(selection, temp))
            candidates.append({'transform': transform, 'provisional_threshold': cutoff, **metrics})
            pipelines[transform] = pipeline
            print(f"Selection {transform}: F1={metrics['f1']:.3f}, recall={metrics['recall']:.3f}, FPR={metrics['false_positive_rate']:.3f}", flush=True)
        # Prefer highest selection F1 within a predeclared 10% validation FPR ceiling.
        eligible = [c for c in candidates if c['false_positive_rate'] <= .10]
        chosen = max(eligible, key=lambda c: c['f1']) if eligible else min(candidates, key=lambda c: c['false_positive_rate'])
        pipeline = pipelines[chosen['transform']]
        calibration_scores = -pipeline.score_samples(normal_calibration)
        threshold = float(np.quantile(calibration_scores, 1-args.false_alarm_budget, method='higher'))
        tail = calibration_scores[calibration_scores > threshold] - threshold
        medium, high = (np.quantile(tail, [.5, .9]).tolist() if len(tail) >= 10 else [.025, .075])
        reference = normal[NUMERIC].agg(['mean','std','median']).T
        reference['p05'] = normal[NUMERIC].quantile(.05)
        reference['p95'] = normal[NUMERIC].quantile(.95)
        reference['log_mean'] = np.log1p(normal[NUMERIC]).mean()
        reference['log_std'] = np.log1p(normal[NUMERIC]).std()
        groups = [set(signatures(clean(part))) for part in [fit, selection, calibration]]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        bundle = dict(schema_version=2, pipeline=pipeline, features=FEATURES,
            reference=reference.to_dict(), dataset='UNSW-NB15 · prepared split',
            trained_at=datetime.now(timezone.utc).isoformat(), seed=42,
            normal_training_rows=len(normal), training_rows=len(training),
            selected_transform=chosen['transform'], threshold=threshold,
            severity_thresholds=[float(medium), float(high)],
            false_alarm_budget=args.false_alarm_budget, sklearn_version=sklearn.__version__,
            candidate_selection=candidates, validation_fpr_ceiling=.10,
            splits={'fit_pool':len(fit), 'selection':len(selection), 'calibration':len(calibration),
                    'normal_calibration':len(normal_calibration), 'feature_group_overlap':0,
                    'ambiguous_groups_excluded':len(ambiguous),
                    'ambiguous_rows_excluded':int(hashes.isin(ambiguous).sum()),
                    'training_rows_after_group_deduplication':len(unique)},
            training_quality=data_quality(training),
            categories={c: sorted(normal[c].unique()) for c in CATEGORICAL})
        calibrated_result = predict(calibration, bundle)
        bundle['calibration_metrics'] = evaluate(calibration, calibrated_result)
        # Supervised reference uses both labels in the SAME fitting pool, and no
        # selection/calibration/test rows for fitting. Apply the same normal-calibration
        # false-alarm-budget policy to both models for a clearer comparison.
        supervised = make_supervised_pipeline()
        supervised.fit(clean(fit), labels(fit))
        supervised_threshold = float(np.quantile(supervised.predict_proba(normal_calibration)[:,1],
            1-args.false_alarm_budget, method='higher'))
        bundle['supervised_baseline'] = {'model': 'Logistic Regression',
            'fitting_rows': len(fit), 'training_labels_used': 'normal and attack',
            'threshold': supervised_threshold, 'threshold_policy': 'independent normal-calibration quantile',
            'false_alarm_budget': args.false_alarm_budget, 'C': 1.0, 'max_iter': 1000,
            'preprocessing': 'training-only medians, log1p, StandardScaler, one-hot categories',
            'iterations': int(supervised.named_steps['model'].n_iter_.max()),
            'note': 'Fixed supervised comparison, not a fair isolation of algorithm effect: it uses attack training labels.'}
        # Test labels are accessed only after both models and thresholds are finalized.
        testing = pd.read_csv(args.test)
        test_features = clean(testing)
        labels(testing)
        result = predict(testing, bundle)
        bundle['metrics'] = evaluate(testing, result)
        bundle['attack_breakdown'] = attack_breakdown(testing, result)
        # A trivial comparator reveals how much class imbalance alone can achieve.
        majority = int(labels(fit).mode().iloc[0])
        trivial = pd.DataFrame({'prediction': ['Anomalous' if majority else 'Normal'] * len(testing),
                                'raw_anomaly_score': np.full(len(testing), majority)})
        bundle['majority_baseline_metrics'] = evaluate(testing, trivial)
        bundle['majority_baseline_class'] = majority
        bundle['supervised_baseline']['calibration_metrics'] = evaluate(calibration, predict_supervised(calibration, supervised, supervised_threshold))
        bundle['supervised_baseline']['metrics'] = evaluate(testing, predict_supervised(testing, supervised, supervised_threshold))
        bundle['test_rows'] = len(testing)
        bundle['test_quality'] = data_quality(testing)
        test_hashes = signatures(test_features)
        overlap = test_hashes.isin(set(hashes))
        strict_mask = ~overlap & ~test_hashes.duplicated()
        strict = testing.loc[strict_mask]
        bundle['overlap_audit'] = {'test_rows_matching_training_features':int(overlap.sum()),
            'strict_test_rows':len(strict), 'test_feature_equivalent_rows':int(test_hashes.duplicated().sum()),
            'note':'Feature equivalence across the 26 selected inputs; not proof of identical original network connections.'}
        bundle['strict_metrics'] = evaluate(strict, predict(strict,bundle)) if len(strict) else None
        bundle['supervised_baseline']['strict_metrics'] = evaluate(strict, predict_supervised(strict, supervised, supervised_threshold)) if len(strict) else None
        bundle['sha256'] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                            for name,path in [('train',args.train),('test',args.test)]}
        bundle['training_seconds'] = round(time.perf_counter()-start, 2)
        # Original user results are retained, including the original model artifact.
        MODEL_PATH.parent.mkdir(exist_ok=True)
        legacy = MODEL_PATH.with_name('original_isolation_forest.joblib')
        if MODEL_PATH.exists() and not legacy.exists():
            old = joblib.load(MODEL_PATH)
            if old.get('schema_version',1) == 1:
                shutil.copy2(MODEL_PATH, legacy)
        if legacy.exists():
            bundle['original_baseline_metrics'] = joblib.load(legacy)['metrics']
        output = {k:v for k,v in bundle.items() if k not in ['pipeline','reference','categories']}
        temporary = MODEL_PATH.with_suffix('.tmp')
        joblib.dump({'pipeline':supervised, 'threshold':supervised_threshold}, MODEL_PATH.parent/'logistic_regression.joblib')
        joblib.dump(bundle, temporary)
        temporary.replace(MODEL_PATH)
        (MODEL_PATH.parent/'evaluation.json').write_text(json.dumps(output,indent=2,allow_nan=False))
        print(json.dumps({'selected_transform':bundle['selected_transform'],
              'official_test':bundle['metrics'], 'strict_test':bundle['strict_metrics'],
              'overlap':bundle['overlap_audit'], 'seconds':bundle['training_seconds']},indent=2))
        print(f'Saved {MODEL_PATH}. Refresh the dashboard.')
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        parser.exit(1,f'{error}\nSee README.md for setup instructions.\n')


if __name__ == '__main__':
    main()
