"""Train-only calibration and validation-selected fixed decision thresholds."""
import io
import unittest
import joblib
import numpy as np
import pandas as pd
from superagent.config import TaskConfig
from superagent.planning.search_space import build_model, metric_candidates, allocate_candidates


class MetricDecisionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'x': np.linspace(-3, 3, 60)})
        self.labels = np.array(['negative']*30+['positive']*30)
        self.profile = {'data_type': 'tabular', 'features': ['x'], 'numeric_columns': ['x'],
                        'categorical_columns': [], 'class_counts': {'negative': 30, 'positive': 30}}
        self.config = TaskConfig(train_csv='unused', target_column='label')

    def test_threshold_changes_decisions_without_changing_probabilities(self):
        seed = {'family': 'logistic', 'C': 1.}
        plain = build_model(self.profile, seed, self.config).fit(self.frame, self.labels)
        tuned = build_model(self.profile, dict(seed, decision_threshold=.65), self.config).fit(self.frame, self.labels)
        np.testing.assert_allclose(plain.predict_proba(self.frame), tuned.predict_proba(self.frame))
        expected = np.where(plain.predict_proba(self.frame)[:, 1] >= .65, 'positive', 'negative')
        np.testing.assert_array_equal(tuned.predict(self.frame), expected)
        buffer = io.BytesIO()
        joblib.dump(tuned, buffer)
        buffer.seek(0)
        np.testing.assert_array_equal(joblib.load(buffer).predict(self.frame), expected)

    def test_calibration_fits_complete_pipeline_in_each_training_fold(self):
        model = build_model(self.profile, {'family': 'logistic', 'C': 1., 'calibration': 'sigmoid'}, self.config)
        model.fit(self.frame, self.labels)
        calibrated = model.named_steps['classifier'].model_
        self.assertEqual(len(calibrated.calibrated_classifiers_), 3)
        # Fold-specific scaler means demonstrate independent fitted preprocessors.
        means = []
        for fold in calibrated.calibrated_classifiers_:
            pipeline = fold.estimator
            self.assertIn('preprocess', pipeline.named_steps)
            scaler = pipeline.named_steps['preprocess'].named_steps['columns'].named_transformers_['numeric'].named_steps['scale']
            means.append(float(scaler.mean_[0]))
        self.assertGreater(len(set(means)), 1)
        np.testing.assert_allclose(model.predict_proba(self.frame).sum(axis=1), 1)

    def test_metric_variants_leave_accuracy_candidates_unchanged(self):
        history = [{'candidate': {'family': 'logistic', 'C': 1.}, 'status': 'success', 'score': .7}]
        self.assertEqual(metric_candidates(self.profile, history, 'accuracy'), [])
        self.assertEqual([c['decision_threshold'] for c in metric_candidates(self.profile, history, 'f1_macro')], [.35, .65])
        self.assertEqual(metric_candidates(self.profile, history, 'log_loss')[0]['calibration'], 'sigmoid')

    def test_disabling_calibration_cannot_repeat_an_existing_seed(self):
        from superagent.planning.proposals import apply_proposal
        from superagent.planning.memory import trial_key
        seed = {'family': 'logistic', 'C': 1.}
        profile = dict(self.profile, metric='log_loss')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            apply_proposal(dict(seed, calibration='sigmoid'), {'calibration': None},
                           profile, {trial_key(seed): seed})

    def test_coverage_and_observed_cost_control_allocation(self):
        history = [{'candidate': {'family': 'logistic', 'C': 1.}, 'status': 'success',
                    'score': .8, 'elapsed_seconds': 100}]
        profile = dict(self.profile, remaining_search_seconds=30, trial_timeout_seconds=30)
        candidates = [{'family': 'logistic', 'C': .1}, {'family': 'extra_trees', 'n_estimators': 10}]
        self.assertEqual(allocate_candidates(candidates, profile, history), [candidates[1]])
        profile['remaining_search_seconds'] = profile['trial_timeout_seconds'] = 300
        self.assertEqual(allocate_candidates(candidates, profile, history)[0]['family'], 'extra_trees')


if __name__ == '__main__':
    unittest.main()
