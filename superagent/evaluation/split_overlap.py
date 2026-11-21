"""Descriptive train/validation overlap, never a claim of proven leakage."""
import pandas as pd


def split_risk(train, validation, profile):
    columns=profile['features']
    def hashes(frame):
        return pd.util.hash_pandas_object(frame[columns].astype('string').fillna(''),index=False)
    left=set(hashes(train))
    right=hashes(validation)
    report={'exact_feature_rows_in_both_splits':int(right.isin(left).sum()),
            'validation_rows':len(validation),
            'interpretation':'Descriptive overlap only; repeated entities can be valid task structure. No inferred timestamps or proof of leakage.'}
    text=profile.get('text_columns') or []
    if text:
        def texts(frame):
            return set(pd.concat([frame[c].fillna('').astype(str).str.lower().str.replace(r'\s+',' ',regex=True).str.strip() for c in text]))-{''}
        a,b=texts(train),texts(validation)
        report['shared_text_values']=len(a&b)
        report['unique_validation_text_values']=len(b)
    return report
