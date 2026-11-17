import unittest
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from superagent.proposals import apply_proposal
from superagent.search import build_model
from superagent.config import TaskConfig


class ProposalTests(unittest.TestCase):
    def test_new_parameters_are_bounded_and_cannot_change_execution(self):
        base={'family':'linear_svm','C':1.,'analyzer':'char','representation':'separate'}
        profile={'data_type':'text','text_columns':['a','b']}
        proposed=apply_proposal(base,{'C':.37,'analyzer':'combined'},profile)
        self.assertEqual(proposed['C'],.37)
        self.assertEqual(base['C'],1.)
        for bad in ({'family':'catboost'},{'code':'print(1)'},{'C':float('nan')},{'C':True},{'C':10000}):
            with self.assertRaises(ValueError):apply_proposal(base,bad,profile)
        with self.assertRaises(ValueError):
            apply_proposal(base,{'representation':'difference'},{'text_columns':['a','b','c']})

    def test_blend_probabilities_and_serialization(self):
        X=pd.DataFrame({'a':['good story','bad story']*12,'b':['bad story','good story']*12})
        profile={'data_type':'text','text_columns':['a','b']}
        model=build_model(profile,{'family':'text_blend','linear_weight':.25,'C':.1},
                          TaskConfig(train_csv='unused',target_column='y'))
        model.fit(X,['yes','no']*12)
        expected=model.predict_proba(X)
        np.testing.assert_allclose(expected.sum(axis=1),1)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'m.joblib';joblib.dump(model,p)
            np.testing.assert_allclose(joblib.load(p).predict_proba(X),expected)
