import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import sparse
from superagent.preprocessing.feature_selection import FeatureFilter
from superagent.evaluation.nested_cv import nested_diagnostic
from superagent.config import TaskConfig
from superagent.runtime.orchestrator import train_agent, run_worker
from superagent.runtime.checkpoint import run_lock


class ExtensionTests(unittest.TestCase):
    def test_restore_does_not_replay_model_request(self):
        from superagent.planning.llm import AutonomousPlanner
        planner=object.__new__(AutonomousPlanner)
        planner.output='unused';planner._resume_skip_api_once=True
        history=[{'candidate':{'family':'dummy'},'status':'success','score':.5}]
        with patch('superagent.planning.llm.write_json'),patch('superagent.planning.llm.event'):
            candidate,_=planner.choose({'data_type':'tabular'},history,'accuracy',0)
        self.assertIsNotNone(candidate)
        self.assertFalse(planner._resume_skip_api_once)

    def test_configurable_text_and_overlap(self):
        from superagent.preprocessing.transformers import checked_vectorizer_options
        from superagent.planning.search_space import build_model
        from superagent.evaluation.split_overlap import split_risk
        frame=pd.DataFrame({'a':['good story','bad headline']*10,'b':['bad headline','good story']*10})
        profile={'data_type':'text','features':['a','b'],'text_columns':['a','b']}
        candidate={'family':'logistic','C':1.,'analyzer':'word','representation':'separate',
                   'vectorizer_options':{'ngram_range':[1,1],'use_idf':False,'min_df':1,'max_features':1000}}
        model=build_model(profile,candidate,TaskConfig(train_csv='unused',target_column='label'))
        model.fit(frame,['yes','no']*10)
        vocabulary=model.named_steps['preprocess'].vectorizer_.vocabulary_.copy()
        model.predict(pd.DataFrame({'a':['validation_only_token'],'b':['another_only_token']}))
        self.assertEqual(vocabulary,model.named_steps['preprocess'].vectorizer_.vocabulary_)
        self.assertEqual(split_risk(frame,frame.iloc[:2],profile)['exact_feature_rows_in_both_splits'],2)
        with self.assertRaises(ValueError):checked_vectorizer_options({'ngram_range':[1,99]})

    def test_filters_fit_only_training_and_memory_bound(self):
        x=np.array([[0,1],[1,1],[2,1],[3,1]],dtype=float)
        f=FeatureFilter('variance').fit(x,[0,0,1,1])
        self.assertEqual(f.transform([[5,999]]).shape,(1,1))
        FeatureFilter('mutual_info').fit(np.tile(x,(8,1)),[0,0,1,1]*8)
        with self.assertRaises(ValueError):
            FeatureFilter('mutual_info').fit(sparse.csr_matrix((50000,1000)),np.zeros(50000))

    def test_nested_selection_is_training_only(self):
        rng=np.random.default_rng(42)
        frame=pd.DataFrame({'x':rng.normal(size=120),'z':rng.normal(size=120)})
        frame['label']=(frame.x>0).astype(str)
        profile={'data_type':'tabular','features':['x','z'],'numeric_columns':['x','z'],'categorical_columns':[]}
        cfg=TaskConfig(train_csv='unused',target_column='label',nested_cv_folds=2)
        result=nested_diagnostic(frame,profile,{'family':'logistic','C':1.},cfg)
        self.assertEqual(len(result['outer_scores']),2)
        self.assertEqual(result['status'],'completed')

    def test_interruption_resume_and_data_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rng=np.random.default_rng(9)
            for name in ['train','val']:
                x=rng.normal(size=50)
                pd.DataFrame({'x':x,'label':(x>0).astype(str)}).to_csv(root/f'{name}.csv',index=False)
            cfg=TaskConfig(train_csv=str(root/'train.csv'),validation_csv=str(root/'val.csv'),
                           target_column='label',max_trials=3,time_budget_seconds=300,enable_torch=False)
            calls=[0]
            def worker(request,*args):
                calls[0]+=1
                if calls[0]==2:raise KeyboardInterrupt()
                return run_worker(request,*args)
            def plan(profile,history,*args):
                return ({'family':'dummy'} if not history else {'family':'logistic','C':1/len(history)}),'test fixture'
            with patch('superagent.runtime.orchestrator.run_worker',side_effect=worker),patch('superagent.planning.llm.AutonomousPlanner.choose',side_effect=plan):
                with self.assertRaises(KeyboardInterrupt):train_agent(cfg,root/'first')
            self.assertEqual(cfg.stability_repeats, 1)
            # A real CLI restart reconstructs the requested config from disk.
            cfg = TaskConfig(**cfg.to_dict())
            with patch('superagent.planning.llm.AutonomousPlanner.choose',side_effect=plan):
                result=train_agent(cfg,root/'resumed',root/'first')
            self.assertEqual(result['status'],'completed')
            history=[json.loads(l) for l in (root/'resumed/trials.jsonl').read_text().splitlines()]
            self.assertEqual([h['trial'] for h in history],[1,2,3])
            self.assertEqual(history[1]['status'],'interrupted')
            old_usage=json.loads((root/'first/model_usage.json').read_text())
            new_usage=json.loads((root/'resumed/model_usage.json').read_text())
            self.assertEqual(old_usage['ledger_path'],new_usage['ledger_path'])
            from superagent.runtime.checkpoint import load_resume
            import os
            with patch.dict(os.environ,{'SUPERAGENT_USAGE_ROOT':str(root/'different_usage_root')}):
                with self.assertRaisesRegex(ValueError,'original shared usage root'):load_resume(root/'first',cfg)
            data=pd.read_csv(root/'train.csv');data.loc[0,'x']+=.01;data.to_csv(root/'train.csv',index=False)
            with self.assertRaisesRegex(ValueError,'data or validation'):
                train_agent(cfg,root/'changed',root/'first')
            with run_lock(root/'active'):
                with self.assertRaises(RuntimeError):
                    with run_lock(root/'active'):pass
