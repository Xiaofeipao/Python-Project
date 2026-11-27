"""Training-only nested diagnostics; official validation/test remain untouched."""
from dataclasses import replace
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from superagent.evaluation.metrics import score_model, utility


def nested_diagnostic(train, profile, candidate, config):
    from superagent.models.factory import build_model
    from superagent.preprocessing.datasets import labels
    if 'C' not in candidate or candidate['family'] not in {'logistic','linear_svm'}:
        return {'status':'skipped', 'reason':'No bounded regularization selection for this family'}
    y=labels(train,config.target_column)
    if len(train)>config.nested_cv_max_rows:
        idx,_=train_test_split(np.arange(len(train)),train_size=config.nested_cv_max_rows,
                               stratify=y,random_state=config.seed)
        train=train.iloc[idx].reset_index(drop=True);y=labels(train,config.target_column)
    if min(y.value_counts()) < 6*config.nested_cv_folds:
        return {'status':'skipped','reason':'Insufficient per-class rows for nested stratification'}
    outer=StratifiedKFold(config.nested_cv_folds,shuffle=True,random_state=config.seed)
    alternate=max(.001,candidate['C']/3)
    if alternate==candidate['C']:
        alternate=min(100.,candidate['C']*3)
    pool=[dict(candidate,C=alternate),dict(candidate)]
    scores=[];choices=[]
    for outer_train,outer_holdout in outer.split(train,y):
        inner=StratifiedKFold(2,shuffle=True,random_state=config.seed)
        fitting=train.iloc[outer_train];inner_y=labels(fitting,config.target_column)
        means=[]
        for option in pool:
            values=[]
            for a,b in inner.split(fitting,inner_y):
                model=build_model(profile,option,config)
                model.fit(fitting.iloc[a],inner_y.iloc[a])
                values.append(score_model(model,fitting.iloc[b],inner_y.iloc[b],config.metric,config.positive_label))
            means.append(np.mean(values))
        selected=max(range(len(pool)),key=lambda i:utility(means[i],config.metric))
        model=build_model(profile,pool[selected],config)
        model.fit(fitting,inner_y)
        scores.append(score_model(model,train.iloc[outer_holdout],y.iloc[outer_holdout],config.metric,config.positive_label))
        choices.append(pool[selected])
    return {'status':'completed','scope':'training-only subsample nested CV; diagnostic, not official validation score',
            'rows':len(train),'outer_scores':scores,'mean':float(np.mean(scores)),
            'std':float(np.std(scores)),'inner_selected_candidates':choices}
