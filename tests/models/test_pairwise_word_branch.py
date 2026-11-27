"""Regression cases for the imported pairwise branch; not executed during integration."""
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from superagent.preprocessing.transformers import MultiTextFeatures, preprocessor
from superagent.planning.proposals import apply_proposal
from superagent.planning.search_space import next_candidate, build_model
from superagent.config import TaskConfig


class PairwiseWordTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame({'left':['new phone launch','quarterly earnings report']*5,
                                 'right':['latest phone release','company earnings results']*5})
        self.profile={'data_type':'text','text_columns':['left','right']}
        self.base={'family':'logistic','C':1.,'analyzer':'char_wb','representation':'difference',
                   'fit_intercept':False,'vectorizer_options':{'ngram_range':[3,6]}}

    def test_shared_union_preserves_signed_features_and_vocabulary(self):
        plain=preprocessor(self.profile,self.base,'.').fit(self.frame)
        candidate=dict(self.base,pairwise_word_features=True)
        features=preprocessor(self.profile,candidate,'.').fit(self.frame)
        forward=features.transform(self.frame)
        swapped=self.frame.rename(columns={'left':'right','right':'left'})
        residual=(forward+features.transform(swapped)).tocsr()
        self.assertTrue(not residual.nnz or np.max(np.abs(residual.data))<1e-6)
        self.assertGreater(forward.shape[1],plain.transform(self.frame).shape[1])
        vocabulary=features.vectorizer_.transformer_list[1][1].vocabulary_.copy()
        features.transform(pd.DataFrame({'left':['validationonlytoken'],'right':['anothernewtoken']}))
        self.assertEqual(vocabulary,features.vectorizer_.transformer_list[1][1].vocabulary_)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.joblib';joblib.dump(features,path)
            residual=(forward-joblib.load(path).transform(self.frame)).tocsr()
            self.assertTrue(not residual.nnz or np.max(np.abs(residual.data))<1e-7)

    def test_candidate_construction_and_proposal_guards(self):
        model=build_model(self.profile,self.base,TaskConfig(train_csv='unused',target_column='label'))
        self.assertFalse(model.named_steps['classifier'].fit_intercept)
        augmented=apply_proposal(self.base,{'pairwise_word_features':True},self.profile)
        self.assertTrue(augmented['pairwise_word_features'])
        for overrides in ({'pairwise_word_features':1},{'fit_intercept':0},
                          {'pairwise_word_features':True,'C':2.}):
            with self.assertRaises(ValueError):apply_proposal(self.base,overrides,self.profile)
        for columns in (['only'],['one','two','three']):
            with self.assertRaises(ValueError):
                preprocessor({'data_type':'text','text_columns':columns},augmented,'.')
        with self.assertRaises(ValueError):
            apply_proposal({'family':'logistic','C':1.},{'fit_intercept':False},{'data_type':'tabular'})

    def test_discovery_uses_input_shape_not_column_identity(self):
        history=[{'candidate':{'family':'dummy'},'status':'success','score':.5}]
        first,_=next_candidate(self.profile,history,'accuracy')
        self.assertEqual(first['family'],'pairwise_overlap')
        history.append({'candidate':first,'status':'success','score':.5})
        plain,_=next_candidate(self.profile,history,'accuracy')
        self.assertEqual(plain,self.base)
        history.append({'candidate':plain,'status':'success','score':.5})
        second,_=next_candidate(self.profile,history,'accuracy')
        self.assertTrue(second['pairwise_word_features'])
        multi={'data_type':'text','text_columns':['one','two','three']}
        other,_=next_candidate(multi,history[:1],'accuracy')
        self.assertNotEqual(other.get('representation'),'difference')
