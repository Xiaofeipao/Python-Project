"""Cases for the word-level overlap block added on 2026-11-25."""
import tempfile
import unittest
from pathlib import Path
import joblib
import pandas as pd
from superagent.planning.proposals import apply_proposal
from superagent.planning.search_space import build_model
from superagent.config import TaskConfig


class PairwiseOverlapTests(unittest.TestCase):
    def setUp(self):
        rows = [('iphone 15 price 999', 'iphone 15 price 899', 1),
                ('budget hotel deal', 'luxury resort offer', 0),
                ('election results today', 'election results yesterday', 1),
                ('stock market falls', 'stock market rises sharply', 0)] * 20
        self.frame = pd.DataFrame(rows, columns=['left', 'right', 'label'])
        self.profile = {'data_type': 'text', 'text_columns': ['left', 'right']}
        self.candidate = {'family': 'pairwise_overlap', 'C': 1., 'fit_intercept': True,
                          'vectorizer_options': {'ngram_range': [3, 6]}}

    def test_fit_predict_and_round_trip(self):
        model = build_model(self.profile, self.candidate,
                            TaskConfig(train_csv='unused', target_column='label'))
        model.fit(self.frame[['left', 'right']], self.frame['label'])
        predicted = model.predict(self.frame[['left', 'right']])
        self.assertGreater((predicted == self.frame['label']).mean(), 0.75)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            joblib.dump(model, path)
            reloaded = joblib.load(path)
        self.assertListEqual(predicted.tolist(), reloaded.predict(self.frame[['left', 'right']]).tolist())

    def test_overlap_block_is_added_to_the_vectorizer(self):
        features = build_model(self.profile, self.candidate,
                               TaskConfig(train_csv='unused', target_column='label')
                               ).named_steps['preprocess'].fit(self.frame[['left', 'right']])
        plain = features.transformer_list[0][1].transform(self.frame[['left', 'right']]).shape[1]
        full = features.transform(self.frame[['left', 'right']]).shape[1]
        self.assertEqual(full, plain + 7)

    def test_proposal_rules(self):
        self.assertEqual(apply_proposal(self.candidate, {'C': 0.3}, self.profile)['C'], 0.3)
        self.assertFalse(apply_proposal(self.candidate, {'fit_intercept': False}, self.profile)['fit_intercept'])
        with self.assertRaises(ValueError):
            apply_proposal(self.candidate, {'n_estimators': 10}, self.profile)
        with self.assertRaises(ValueError):
            apply_proposal({'family': 'pairwise_overlap', 'C': 1.}, {'fit_intercept': False},
                           {'data_type': 'tabular'})
        self.assertEqual(apply_proposal({'family': 'minilm_logistic', 'C': 1.}, {'C': 0.1}, self.profile)['C'], 0.1)


if __name__ == '__main__':
    unittest.main()
