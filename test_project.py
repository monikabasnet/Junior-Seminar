"""Run: .venv/bin/python -m unittest -v test_project.py (real local artifacts required)."""
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils import ROOT, MODEL_PATH, NUMERIC, clean, predict, evaluate, investigate, summary, metric_calculations, evaluation_errors, predict_supervised, FEATURE_DESCRIPTIONS, FEATURES


class PipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(MODEL_PATH)
        cls.flows = pd.read_csv(ROOT/'data/UNSW_NB15_testing-set.csv',nrows=250)

    def test_labels_ids_and_extra_columns_do_not_change_scores(self):
        original = predict(self.flows,self.bundle)
        altered = self.flows.copy()
        altered['id'] = -999
        altered['label'] = 1-altered.label
        altered['attack_cat'] = 'UNTRUSTED'
        altered['irrelevant'] = 99999
        np.testing.assert_array_equal(original.anomaly_score,predict(altered,self.bundle).anomaly_score)

    def test_missing_unseen_and_outlier_inputs_do_not_refit(self):
        original = predict(self.flows,self.bundle)
        imputer = self.bundle['pipeline'].named_steps['preprocessing'].named_transformers_['numeric'].named_steps['impute']
        before = imputer.statistics_.copy()
        unusual = self.flows.copy()
        unusual['proto'] = 'unseen_protocol'
        unusual.loc[0,NUMERIC] = np.nan
        unusual.loc[1,'dur'] = np.inf
        unusual.loc[2,'sbytes'] = 1e15
        scores = predict(unusual,self.bundle)
        self.assertTrue(np.isfinite(scores.anomaly_score).all())
        np.testing.assert_array_equal(before,imputer.statistics_)
        np.testing.assert_array_equal(original.anomaly_score,predict(self.flows,self.bundle).anomaly_score)

    def test_schema_rejects_invalid_records(self):
        for column,value in [('dur',-1),('spkts','wrong')]:
            invalid = self.flows.astype({column:'object'}).copy()
            invalid.loc[0,column] = value
            with self.assertRaises(ValueError):
                clean(invalid)
        with self.assertRaises(ValueError):
            clean(self.flows.drop(columns='dur'))
        with self.assertRaises(ValueError):
            clean(self.flows.iloc[:0])

    def test_roundtrip_metrics_and_report(self):
        original = predict(self.flows,self.bundle)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trusted.joblib'
            joblib.dump(self.bundle,path)
            restored = predict(self.flows,joblib.load(path))
        np.testing.assert_array_equal(original.anomaly_score,restored.anomaly_score)
        metrics = evaluate(self.flows,original)
        self.assertEqual(sum(map(sum,metrics['confusion_matrix'])),len(self.flows))
        self.assertTrue((original.loc[original.prediction == 'Anomalous','anomaly_score'] > 0).all())
        self.assertIn('Confusion matrix',summary(original,self.bundle,metrics))
        self.assertGreater(len(investigate(original.iloc[0],self.bundle)),0)

    def test_full_test_evaluation_matches_saved_run(self):
        frame = pd.read_csv(ROOT/'data/UNSW_NB15_testing-set.csv')
        measured = evaluate(frame,predict(frame,self.bundle))
        self.assertEqual(measured,self.bundle['metrics'])
        self.assertEqual(self.bundle['splits']['feature_group_overlap'],0)
        self.assertLess(self.bundle['strict_metrics']['records'],len(frame))


class TechnicalEvidenceChecks(unittest.TestCase):
    def test_metric_arithmetic_from_known_confusion_matrix(self):
        # Independent, hand-computable counts: TN=3, FP=1, FN=2, TP=4.
        table = metric_calculations({'confusion_matrix': [[3,1],[2,4]]}).set_index('metric')
        expected = {'Accuracy': .7, 'Precision': .8, 'Recall': 4/6,
                    'F1-score': 8/11, 'False-positive rate': .25}
        for name, value in expected.items():
            self.assertAlmostEqual(table.loc[name, 'value'], value)
        empty = metric_calculations({'confusion_matrix': [[0,0],[0,0]]})
        self.assertTrue(empty.value.isna().all())
        self.assertEqual(set(FEATURE_DESCRIPTIONS), set(FEATURES))

    def test_error_explorer_distinguishes_false_alarms_and_misses(self):
        sample = pd.DataFrame({'label': [0,0,1,1],
            'prediction': ['Normal','Anomalous','Normal','Anomalous']})
        errors = evaluation_errors(sample)
        self.assertEqual(errors.index.tolist(), [1,2])
        self.assertEqual(errors.error_type.tolist(), ['False positive','False negative'])

    def test_supervised_artifact_reproduces_report_and_excludes_labels(self):
        bundle = joblib.load(MODEL_PATH)
        saved = joblib.load(ROOT/'models/logistic_regression.joblib')
        frame = pd.read_csv(ROOT/'data/UNSW_NB15_testing-set.csv')
        result = predict_supervised(frame, saved['pipeline'], saved['threshold'])
        self.assertEqual(evaluate(frame,result), bundle['supervised_baseline']['metrics'])
        changed = frame.head(50).copy()
        changed['label'] = 1-changed.label
        changed['attack_cat'] = 'changed'
        measured = predict_supervised(changed, saved['pipeline'], saved['threshold'])
        np.testing.assert_array_equal(result.raw_anomaly_score.head(50), measured.raw_anomaly_score)
        # A fitted scaler records only the fitting pool, not train+test rows.
        scaler = saved['pipeline'].named_steps['preprocessing'].named_transformers_['numeric'].named_steps['scale']
        self.assertEqual(int(scaler.n_samples_seen_), bundle['splits']['fit_pool'])


class DashboardChecks(unittest.TestCase):
    def test_all_pages_and_replay_controls(self):
        app = AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertFalse(app.exception)
        pages = list(app.sidebar.radio[0].options)
        for page in pages:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception,msg=f'{page}: {list(app.exception)}')
        app.sidebar.radio[0].set_value('Model Performance').run()
        next(r for r in app.radio if r.label == 'Evaluation population').set_value('Unique test inputs absent from training').run()
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('Live Traffic Simulation').run()
        app.button[0].click().run()
        self.assertGreater(app.session_state['cursor'],0)
        app.button[1].click().run()
        paused = app.session_state['cursor']
        app.run()
        self.assertEqual(app.session_state['cursor'],paused)
        app.button[2].click().run()
        self.assertEqual(app.session_state['cursor'],0)
        self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
