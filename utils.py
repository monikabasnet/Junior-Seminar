"""Shared data contract, leakage checks, inference and honest evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, roc_auc_score, average_precision_score)

ROOT = Path(__file__).resolve().parent
NUMERIC = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
           'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
           'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'ct_srv_src', 'ct_dst_ltm']
CATEGORICAL = ['proto', 'service', 'state']
FEATURES = NUMERIC + CATEGORICAL
MODEL_PATH = ROOT / 'models/isolation_forest.joblib'
COLORS = {'Normal': '#34d399', 'Anomalous': '#fb7185', 'High': '#fb7185',
          'Medium': '#fbbf24', 'Low': '#38bdf8', 'None': '#64748b'}


FEATURE_DESCRIPTIONS = {
    'dur': 'Connection duration (seconds)',
    'spkts': 'Source-to-destination packet count',
    'dpkts': 'Destination-to-source packet count',
    'sbytes': 'Source-to-destination bytes',
    'dbytes': 'Destination-to-source bytes',
    'rate': 'Flow packet-rate feature',
    'sttl': 'Source-to-destination time-to-live value',
    'dttl': 'Destination-to-source time-to-live value',
    'sload': 'Source traffic-load feature',
    'dload': 'Destination traffic-load feature',
    'sloss': 'Source packets retransmitted or dropped',
    'dloss': 'Destination packets retransmitted or dropped',
    'sinpkt': 'Source inter-packet interval',
    'dinpkt': 'Destination inter-packet interval',
    'sjit': 'Source packet jitter',
    'djit': 'Destination packet jitter',
    'tcprtt': 'TCP round-trip-time feature',
    'synack': 'Time between TCP SYN and SYN-ACK',
    'ackdat': 'Time between TCP SYN-ACK and ACK',
    'smean': 'Mean source packet size',
    'dmean': 'Mean destination packet size',
    'ct_srv_src': 'Service/source connection-count feature',
    'ct_dst_ltm': 'Destination connection-count feature',
    'proto': 'Transaction protocol category',
    'service': 'Application-service category',
    'state': 'Connection-state category',
}


def metric_calculations(metrics):
    """Expose the arithmetic behind sklearn metrics without rounding inputs."""
    tn, fp = metrics['confusion_matrix'][0]
    fn, tp = metrics['confusion_matrix'][1]
    n = tn + fp + fn + tp
    definitions = [
        ('Accuracy', '(TP + TN) / N', f'({tp:,} + {tn:,}) / {n:,}', tp+tn, n),
        ('Precision', 'TP / (TP + FP)', f'{tp:,} / ({tp:,} + {fp:,})', tp, tp+fp),
        ('Recall', 'TP / (TP + FN)', f'{tp:,} / ({tp:,} + {fn:,})', tp, tp+fn),
        ('F1-score', '2TP / (2TP + FP + FN)', f'{2*tp:,} / ({2*tp:,} + {fp:,} + {fn:,})', 2*tp, 2*tp+fp+fn),
        ('False-positive rate', 'FP / (FP + TN)', f'{fp:,} / ({fp:,} + {tn:,})', fp, fp+tn),
    ]
    return pd.DataFrame([{'metric': name, 'formula': formula, 'substitution': substitution,
                          'value': float(numerator/denominator) if denominator else None}
                         for name, formula, substitution, numerator, denominator in definitions])


def evaluation_errors(result):
    """Return actual mistakes only; ground truth is required, never guessed."""
    actual = labels(result).to_numpy()
    flagged = (result.prediction == 'Anomalous').to_numpy()
    mask = (actual == 0) & flagged | (actual == 1) & ~flagged
    errors = result.loc[mask].copy()
    errors['error_type'] = np.where(actual[mask] == 0, 'False positive', 'False negative')
    return errors


def predict_supervised(frame, pipeline, threshold=.5):
    """Comparison model only. Probabilities are used for ranking metrics."""
    features = clean(frame)
    result = frame.copy().reset_index(drop=True)
    probability = pipeline.predict_proba(features)[:, 1]
    result['prediction'] = np.where(probability > threshold, 'Anomalous', 'Normal')
    result['raw_anomaly_score'] = probability
    return result


def clean(frame):
    if not frame.columns.is_unique:
        raise ValueError('Duplicate column names are not allowed.')
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('The CSV contains no records.')
    result = frame[FEATURES].copy()
    for column in NUMERIC:
        original = result[column].copy()
        if not pd.api.types.is_numeric_dtype(original):
            original = original.mask(original.astype('string').str.fullmatch(r'\s*', na=False))
        converted = pd.to_numeric(original, errors='coerce')
        if (original.notna() & converted.isna()).any():
            raise ValueError(f'{column}: nonnumeric values found. Correct the CSV.')
        converted = converted.replace([np.inf, -np.inf], np.nan)
        if (converted < 0).any():
            raise ValueError(f'{column}: negative values are invalid for these nonnegative flow features.')
        result[column] = converted.astype(float)
    for column in CATEGORICAL:
        result[column] = (result[column].astype('string').str.strip().str.lower()
                          .replace('', 'unknown').fillna('unknown').astype(str))
    return result


def labels(frame):
    if 'label' not in frame:
        raise ValueError('Evaluation requires label: 0 = normal, 1 = attack.')
    result = pd.to_numeric(frame['label'], errors='coerce')
    if not result.isin([0, 1]).all():
        raise ValueError('Every label must be 0 or 1.')
    return result.astype(int)


def signatures(features):
    """Group identical model inputs, regardless of ID or label (not packet identity)."""
    return pd.util.hash_pandas_object(features[FEATURES], index=False)


def data_quality(frame):
    features = clean(frame)
    detail = pd.DataFrame({'feature': FEATURES,
        'missing_values': features.isna().sum().values,
        'unique_values': features.nunique().values})
    return {'rows': len(frame), 'columns': len(frame.columns),
            'model_features': len(FEATURES), 'missing_numeric_cells': int(features[NUMERIC].isna().sum().sum()),
            'feature_equivalent_rows': int(signatures(features).duplicated().sum()),
            'constant_features': detail.loc[detail.unique_values <= 1, 'feature'].tolist(),
            'detail': detail.to_dict('records')}


def predict(frame, bundle):
    features = clean(frame)
    if bundle.get('schema_version', 1) >= 2:
        raw = -bundle['pipeline'].score_samples(features)
        scores = raw - bundle['threshold']
    else:
        scores = -bundle['pipeline'].decision_function(features)
        raw = scores
    result = frame.copy().reset_index(drop=True)
    result['record_identifier'] = result['id'] if 'id' in result else np.arange(1, len(result) + 1)
    result['prediction'] = np.where(scores > 0, 'Anomalous', 'Normal')
    result['anomaly_score'] = scores
    result['raw_anomaly_score'] = raw
    medium, high = bundle.get('severity_thresholds', [.05, .10])
    result['alert_severity'] = np.select(
        [scores >= high, scores >= medium, scores > 0], ['High', 'Medium', 'Low'], default='None')
    return result


def evaluate(frame, result):
    actual = labels(frame).to_numpy()
    predicted = (result['prediction'] == 'Anomalous').astype(int).to_numpy()
    if len(actual) != len(predicted):
        raise ValueError('Evaluation row count mismatch.')
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    both = len(np.unique(actual)) == 2
    return dict(records=len(actual), accuracy=float(accuracy_score(actual, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(actual, predicted)),
        precision=float(precision_score(actual, predicted, zero_division=0)),
        recall=float(recall_score(actual, predicted, zero_division=0)),
        f1=float(f1_score(actual, predicted, zero_division=0)),
        false_positive_rate=float(fp / (fp + tn)) if fp + tn else None,
        roc_auc=float(roc_auc_score(actual, result.raw_anomaly_score)) if both else None,
        average_precision=float(average_precision_score(actual, result.raw_anomaly_score)) if both else None,
        attack_prevalence=float(actual.mean()),
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]])


def attack_breakdown(frame, result):
    if 'attack_cat' not in frame:
        return []
    report = pd.DataFrame({'family': frame.attack_cat.astype(str).str.strip().to_numpy(),
        'actual': labels(frame).to_numpy(), 'detected': (result.prediction == 'Anomalous').to_numpy()})
    report = report[report.actual == 1].groupby('family').detected.agg(['sum', 'count']).reset_index()
    report.columns = ['attack_family', 'detected', 'total']
    report['recall'] = report.detected / report.total
    return report.to_dict('records')


def investigate(row, bundle):
    """Compare log1p values with a normal-only reference; this is not SHAP."""
    reference = pd.DataFrame(bundle['reference'])
    values = pd.to_numeric(row.reindex(NUMERIC), errors='coerce')
    if 'log_mean' in reference:
        deviations = (np.log1p(values) - reference.log_mean) / reference.log_std.replace(0, np.nan)
    else:
        deviations = (values - reference['mean']) / reference['std'].replace(0, np.nan)
    deviations = deviations.replace([np.inf, -np.inf], np.nan).dropna()
    top = deviations.abs().nlargest(6).index
    return pd.DataFrame({'feature': top, 'value': values[top].values,
        'normal_mean': reference.loc[top, 'mean'].values,
        'normal_median': reference.loc[top, 'median'].values if 'median' in reference else reference.loc[top, 'mean'].values,
        'deviation_std': deviations[top].values})


def summary(result, bundle, metrics=None, source='Analyzed CSV'):
    count = int((result.prediction == 'Anomalous').sum())
    lines = ['Network Intrusion Detection Using AI', f"Dataset: {bundle['dataset']}",
        f'Source: {source}', f"Model: Isolation Forest / {bundle.get('selected_transform', 'original')}",
        f"Trained: {bundle['trained_at']}", f'Records: {len(result)}', f'Anomalies: {count}',
        f'Anomaly rate: {count / len(result):.2%}',
        'Score: positive distance above calibrated threshold; not a probability.',
        'Anomalies are not confirmed attacks. Severity is demonstration priority only.']
    if metrics:
        lines += [f'{name}: {value:.4f}' for name, value in metrics.items() if isinstance(value, float)]
        lines += [f"Confusion matrix [[TN,FP],[FN,TP]]: {metrics['confusion_matrix']}"]
    else:
        lines += ['No evaluation labels available for this analysis.']
    return '\n'.join(lines) + '\n'
