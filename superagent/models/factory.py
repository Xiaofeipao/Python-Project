"""Build executable estimators from validated candidate settings."""

import importlib.util

from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

from superagent.preprocessing.pipelines import hybrid_preprocessor, preprocessor, tabular_preprocessor


def build_model(profile, candidate, config):
    model = _build_model(profile, candidate, config)
    if candidate.get('feature_filter', 'none') != 'none':
        if candidate['family'] not in {'logistic','linear_svm','sgd','extra_trees','random_forest','nb','rbf_svm'}:
            raise ValueError('Feature filters require a numeric feature pipeline')
        from superagent.preprocessing.feature_selection import FeatureFilter
        model.steps.insert(-1, ('feature_filter', FeatureFilter(candidate['feature_filter'],
                           candidate.get('percentile',50),config.seed)))
    if candidate.get('decision_threshold') is not None or candidate.get('calibration'):
        from superagent.models.decision import DecisionClassifier
        model = Pipeline([('preprocess', 'passthrough'), ('classifier', DecisionClassifier(
            model, threshold=candidate.get('decision_threshold'), calibration=candidate.get('calibration'),
            positive_label=config.positive_label))])
    return model


def _build_model(profile, candidate, config):
    family = candidate["family"]
    if family == 'frozen_image_logistic':
        from superagent.models.neural import FrozenImageFeatures
        if profile['data_type'] != 'image':
            raise ValueError('Frozen image features require image input')
        return Pipeline([('preprocess', FrozenImageFeatures(profile['image_column'], config.image_root,
                    size=candidate.get('size', 96), device=config.device, threads=config.threads,
                    batch_size=candidate.get('batch_size', 32), random_state=config.seed,
                    pretrained=candidate.get('pretrained', True))),
                         ('classifier', LogisticRegression(C=candidate.get('C', 1.), max_iter=1200,
                                                          random_state=config.seed))])
    if family == 'hybrid_logistic':
        return Pipeline([('preprocess', hybrid_preprocessor(profile, candidate)),
                         ('classifier', LogisticRegression(C=candidate.get('C', 1.),
                                                           class_weight=candidate.get('class_weight'),
                                                           fit_intercept=candidate.get('fit_intercept', True),
                                                           solver='lbfgs', max_iter=1200,
                                                           random_state=config.seed))])
    if family == 'hybrid_sgd':
        return Pipeline([('preprocess', hybrid_preprocessor(profile, candidate)),
                         ('classifier', SGDClassifier(loss='log_loss', alpha=candidate.get('alpha', 1e-4),
                                                      max_iter=1500, tol=1e-4,
                                                      random_state=config.seed))])
    if family == 'hybrid_xgboost':
        from superagent.models.tabular import EncodedXGBoost
        return Pipeline([('preprocess', hybrid_preprocessor(profile, candidate)),
                         ('classifier', EncodedXGBoost(
                             iterations=candidate.get('iterations', 800),
                             depth=candidate.get('depth', 6),
                             learning_rate=candidate.get('learning_rate', .05),
                             reg_lambda=candidate.get('reg_lambda', 10),
                             threads=config.threads, balanced=candidate.get('balanced', False),
                             device=candidate.get('device', 'auto'),
                             random_state=config.seed))])
    if family == 'image_meta_blend':
        from superagent.models.tabular import ProbabilityBlend
        image = {'family': 'torch_pretrained', 'architecture': candidate.get('architecture', 'pretrained_resnet18'),
                 'size': candidate.get('size', 96), 'epochs': candidate.get('epochs', 18),
                 'learning_rate': candidate.get('learning_rate', 3e-4), 'weight_decay': 1e-4,
                 'dropout': .2, 'early_stopping': True, 'cosine_schedule': True,
                 'augment': True, 'imagenet_normalize': True, 'tta': True,
                 'pretrained': candidate.get('pretrained', True)}
        meta = dict(profile, data_type='tabular', image_column=None, text_columns=None,
                    features=profile['tabular_columns'],
                    numeric_columns=profile['numeric_columns'],
                    categorical_columns=profile['categorical_columns'],
                    datetime_columns=profile.get('datetime_columns', []))
        if importlib.util.find_spec('catboost') is not None:
            meta_candidate = {'family': 'catboost', 'iterations': candidate.get('meta_iterations', 900),
                              'depth': 6, 'learning_rate': .05, 'l2_leaf_reg': 5}
        else:
            meta_candidate = {'family': 'logistic', 'C': 1.}
        members = [build_model(profile, image, config), build_model(meta, meta_candidate, config)]
        weight = candidate.get('image_weight', .7)
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', ProbabilityBlend(members, weights=[weight, 1 - weight]))])
    if family == 'embedding_logistic':
        from superagent.models.text.encoders import TextEncoderFeatures
        from sklearn.preprocessing import StandardScaler
        if profile['data_type'] != 'text':
            raise ValueError('Frozen transformer probes require text input')
        columns = profile['text_columns']
        return Pipeline([('preprocess', TextEncoderFeatures(
                             columns, text=True, encoder=candidate.get('encoder', 'mpnet'),
                             device=config.device, batch_size=candidate.get('batch_size', 64),
                             max_length=candidate.get('max_length', 256))),
                         ('scale', StandardScaler()),
                         ('classifier', LogisticRegression(C=candidate.get('C', 1.),
                                                           class_weight=candidate.get('class_weight'),
                                                           max_iter=2500, random_state=config.seed))])
    if family == 'embedding_mlp':
        from superagent.models.text.encoders import TextEncoderFeatures
        from superagent.models.neural import TorchClassifier
        from sklearn.preprocessing import StandardScaler
        if profile['data_type'] != 'text':
            raise ValueError('Frozen transformer probes require text input')
        return Pipeline([('preprocess', TextEncoderFeatures(
                             profile['text_columns'], text=True,
                             encoder=candidate.get('encoder', 'mpnet'), device=config.device,
                             batch_size=candidate.get('batch_size', 64),
                             max_length=candidate.get('max_length', 256))),
                         ('scale', StandardScaler()),
                         ('classifier', TorchClassifier(
                             architecture='mlp', hidden=candidate.get('hidden', 256),
                             epochs=candidate.get('epochs', 30),
                             learning_rate=candidate.get('learning_rate', .001),
                             weight_decay=candidate.get('weight_decay', .01),
                             batch_size=config.neural_batch_size,
                             dropout=candidate.get('dropout', .2), device=config.device,
                             threads=config.threads, random_state=config.seed,
                             mixed_precision=config.mixed_precision,
                             gpu_cache_fraction=config.gpu_cache_fraction))])
    if family == 'transformer_finetune':
        from superagent.models.text.finetuning import TransformerFinetuneClassifier
        if profile['data_type'] != 'text':
            raise ValueError('Transformer fine-tuning requires text input')
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', TransformerFinetuneClassifier(
                             profile['text_columns'], encoder=candidate.get('encoder', 'mpnet'),
                             epochs=candidate.get('epochs', 3),
                             learning_rate=candidate.get('learning_rate', 2e-5),
                             weight_decay=candidate.get('weight_decay', .01),
                             batch_size=candidate.get('batch_size', 16),
                             max_length=candidate.get('max_length', 256),
                             freeze_layers=candidate.get('freeze_layers', 0),
                             device=config.device, threads=config.threads,
                             random_state=config.seed))])
    if family == 'text_xgboost':
        from superagent.models.tabular import EncodedXGBoost
        if profile['data_type'] != 'text':
            raise ValueError('Text XGBoost requires text input')
        return Pipeline([('preprocess', preprocessor(profile, candidate, config.image_root)),
                         ('classifier', EncodedXGBoost(
                             iterations=candidate.get('iterations', 1000),
                             depth=candidate.get('depth', 4),
                             learning_rate=candidate.get('learning_rate', .03),
                             reg_lambda=candidate.get('reg_lambda', 10),
                             threads=config.threads, balanced=candidate.get('balanced', False),
                             device=config.device, random_state=config.seed))])
    if family in {'text_fasttext', 'text_cnn', 'text_rnn'}:
        from superagent.models.text.networks import TextNeuralClassifier
        if profile['data_type'] != 'text':
            raise ValueError('Trainable text networks require text input')
        architecture = {'text_fasttext': 'fasttext', 'text_cnn': 'textcnn',
                        'text_rnn': 'textrnn'}[family]
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', TextNeuralClassifier(
                             profile['text_columns'], architecture=architecture,
                             embedding_dim=candidate.get('embedding_dim', 128),
                             hidden=candidate.get('hidden', 128),
                             epochs=candidate.get('epochs', 25),
                             learning_rate=candidate.get('learning_rate', .001),
                             weight_decay=candidate.get('weight_decay', .01),
                             batch_size=candidate.get('batch_size', config.neural_batch_size),
                             max_length=candidate.get('max_length', 256),
                             ngram_max=candidate.get('ngram_max', 2),
                             dropout=candidate.get('dropout', .2), device=config.device,
                             threads=config.threads, random_state=config.seed,
                             max_vocab=candidate.get('max_vocab', 30000)))])
    if family == 'minilm_logistic':
        from superagent.models.text.encoders import MiniLMFeatures
        from sklearn.preprocessing import StandardScaler
        columns = profile['text_columns'] if profile['data_type'] == 'text' else profile['features']
        return Pipeline([('preprocess', MiniLMFeatures(columns, text=profile['data_type'] == 'text',
                                                      device=config.device, batch_size=config.neural_batch_size)),
                         ('scale', StandardScaler()),
                         ('classifier', LogisticRegression(C=candidate['C'], max_iter=2000,
                                                           random_state=config.seed))])
    if family == 'text_blend':
        from superagent.models.tabular import ProbabilityBlend
        weight = candidate.get('linear_weight', .5)
        members = [build_model(profile, {'family': 'logistic', 'C': 1., 'analyzer': 'char',
                    'representation': 'separate', 'class_weight': 'balanced'}, config),
                   build_model(profile, {'family': 'linear_svm', 'C': candidate.get('C', .1),
                    'analyzer': 'combined', 'representation': 'separate'}, config)]
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', ProbabilityBlend(members, weights=[weight, 1-weight]))])
    if family == 'text_blend_diff':
        from superagent.models.tabular import ProbabilityBlend
        weight = candidate.get('linear_weight', .5)
        members = [build_model(profile, {'family': 'logistic', 'C': 1., 'analyzer': 'char_wb',
                    'representation': 'difference', 'class_weight': 'balanced'}, config),
                   build_model(profile, {'family': 'linear_svm', 'C': candidate.get('C', .1),
                    'analyzer': 'word', 'representation': 'difference'}, config)]
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', ProbabilityBlend(members, weights=[weight, 1-weight]))])
    if family == 'pairwise_overlap':
        from superagent.models.text.pairwise import overlap_classifier
        return overlap_classifier(profile['text_columns'], candidate, config.seed)
    if family == 'pairwise_overlap_xgb':
        from superagent.models.text.pairwise import overlap_xgb_classifier
        return overlap_xgb_classifier(profile['text_columns'], candidate, config)
    if family == 'pairwise_semantic_mlp_blend':
        from superagent.models.tabular import ProbabilityBlend
        semantic_weight = candidate.get('semantic_weight', .05)
        mlp_weight = candidate.get('mlp_weight', .35)
        pair_weight = 1. - semantic_weight - mlp_weight
        if pair_weight <= 0:
            raise ValueError('Blend weights must leave a positive pairwise share')
        pairwise = build_model(profile, {
            'family': 'pairwise_overlap', 'C': candidate.get('C', 1.),
            'fit_intercept': True,
            'vectorizer_options': candidate.get('vectorizer_options') or {'ngram_range': [3, 6]},
        }, config)
        semantic = build_model(profile, {
            'family': 'embedding_logistic', 'C': candidate.get('C', 1.),
            'encoder': candidate.get('encoder_semantic', 'mpnet'),
            'batch_size': candidate.get('batch_size', 64),
            'max_length': candidate.get('max_length', 256),
        }, config)
        mlp = build_model(profile, {
            'family': 'embedding_mlp', 'encoder': candidate.get('encoder_mlp', 'bge'),
            'hidden': candidate.get('hidden', 384), 'epochs': candidate.get('epochs', 40),
            'learning_rate': candidate.get('learning_rate', .0007),
            'weight_decay': candidate.get('weight_decay', .03),
            'dropout': candidate.get('dropout', .3),
            'batch_size': candidate.get('batch_size', 64),
            'max_length': candidate.get('max_length', 256),
        }, config)
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', ProbabilityBlend(
                             [pairwise, semantic, mlp],
                             weights=[pair_weight, semantic_weight, mlp_weight]))])
    if family == 'pairwise_semantic_blend':
        from superagent.models.tabular import ProbabilityBlend
        weight = candidate.get('semantic_weight', .1)
        pairwise = build_model(profile, {
            'family': 'pairwise_overlap', 'C': candidate.get('C', 1.),
            'fit_intercept': True,
            'vectorizer_options': candidate.get('vectorizer_options') or {'ngram_range': [3, 6]},
        }, config)
        semantic = build_model(profile, {
            'family': 'embedding_logistic', 'C': candidate.get('C', 1.),
            'encoder': candidate.get('encoder', 'mpnet'),
            'batch_size': candidate.get('batch_size', 64),
            'max_length': candidate.get('max_length', 256),
        }, config)
        return Pipeline([('preprocess', 'passthrough'),
                         ('classifier', ProbabilityBlend(
                             [pairwise, semantic], weights=[1 - weight, weight]))])
    common = {"random_state": config.seed}
    if family == "dummy":
        estimator = DummyClassifier(strategy="prior", **common)
    elif family == "logistic":
        estimator = LogisticRegression(C=candidate["C"], solver="lbfgs", max_iter=1200,
                                       class_weight=candidate.get("class_weight"),
                                       fit_intercept=candidate.get('fit_intercept',True), **common)
    elif family == 'linear_svm':
        estimator = CalibratedClassifierCV(LinearSVC(C=candidate['C'], dual='auto', max_iter=10000,
                                                     **common), cv=3, n_jobs=config.threads)
    elif family == 'rbf_svm':
        estimator = CalibratedClassifierCV(SVC(C=candidate['C'], cache_size=512, **common),
                                           cv=3, ensemble=False, n_jobs=config.threads)
    elif family in {"extra_trees", "random_forest"}:
        from superagent.models.tabular import BalancedExtraTrees, BalancedRandomForest

        cls = BalancedExtraTrees if family == "extra_trees" else BalancedRandomForest
        estimator = cls(n_estimators=candidate["n_estimators"], max_depth=candidate["max_depth"],
                        min_samples_leaf=candidate["min_samples_leaf"], n_jobs=config.threads,
                        class_weight=candidate.get("class_weight"), **common)
    elif family == "catboost":
        from superagent.models.tabular import NativeBoostClassifier

        estimator = NativeBoostClassifier(iterations=candidate['iterations'], depth=candidate['depth'],
                                          learning_rate=candidate['learning_rate'],
                                          l2_leaf_reg=candidate['l2_leaf_reg'], threads=config.threads,
                                          balanced=candidate.get('balanced', False), **common)
    elif family == "catboost_cross":
        from superagent.models.tabular import NativeBoostClassifier

        estimator = NativeBoostClassifier(
            iterations=candidate['iterations'], depth=candidate['depth'],
            learning_rate=candidate['learning_rate'], l2_leaf_reg=candidate['l2_leaf_reg'],
            threads=config.threads, balanced=candidate.get('balanced', False),
            max_ctr_complexity=candidate.get('max_ctr_complexity', 4),
            one_hot_max_size=candidate.get('one_hot_max_size', 2),
            random_strength=candidate.get('random_strength', 1.),
            bagging_temperature=candidate.get('bagging_temperature', 1.),
            grow_policy=candidate.get('grow_policy', 'SymmetricTree'),
            max_leaves=candidate.get('max_leaves'),
            border_count=candidate.get('border_count', 254),
            boosting_type=candidate.get('boosting_type', 'Plain'), **common)
    elif family in {'lightgbm', 'catboost_ensemble', 'tabular_blend'}:
        from superagent.models.tabular import advanced_estimator

        estimator = advanced_estimator(candidate, config)
    elif family == 'xgboost':
        from superagent.models.tabular import EncodedXGBoost
        estimator = EncodedXGBoost(iterations=candidate['iterations'], depth=candidate['depth'],
                                    learning_rate=candidate['learning_rate'], reg_lambda=candidate['reg_lambda'],
                                    threads=config.threads, balanced=candidate.get('balanced', False), **common)
    elif family == 'cross_boost_blend':
        from superagent.models.tabular import ProbabilityBlend, NativeBoostClassifier, EncodedXGBoost, NativeLeafBoost
        encoded = [Pipeline([('preprocess', preprocessor(profile, {'family': 'xgboost'}, config.image_root)),
                            ('classifier', EncodedXGBoost(iterations=candidate.get('iterations', 1600),
                                                         random_state=config.seed + 101 * index,
                                                         threads=config.threads))])
                   for index in range(candidate.get('gpu_seeds', 1))]
        weight = candidate.get('gpu_weight', .25)
        leaf = candidate.get('leaf_weight', 0.)
        estimator = ProbabilityBlend([
            *[NativeBoostClassifier(iterations=candidate.get('iterations', 1200), depth=8, learning_rate=.05,
                                    l2_leaf_reg=5, threads=config.threads,
                                    random_state=config.seed + 101 * index) for index in range(3)], *encoded,
            *([NativeLeafBoost(random_state=config.seed, threads=config.threads)] if leaf else [])],
            weights=[(1 - weight - leaf) / 3] * 3 + [weight / len(encoded)] * len(encoded)
                    + ([leaf] if leaf else []))
    elif family == "nb":
        estimator = ComplementNB(alpha=candidate["alpha"])
    elif family == "sgd":
        estimator = SGDClassifier(loss="log_loss", alpha=candidate["alpha"], max_iter=1500,
                                  tol=1e-4, **common)
    elif family in {"torch_mlp", "torch_cnn", "torch_resnet", "torch_ensemble", "torch_pretrained"}:
        from superagent.models.neural import TorchClassifier, TorchEnsemble

        default_architecture = ("cnn" if family == "torch_cnn" else
                                "pretrained_resnet18" if family == "torch_pretrained" else "mlp")
        params = dict(
            architecture=candidate.get("architecture", default_architecture),
            epochs=candidate["epochs"], learning_rate=candidate["learning_rate"],
            weight_decay=candidate["weight_decay"], image_size=candidate.get("size", 32),
            threads=config.threads, batch_size=config.neural_batch_size,
            mixed_precision=config.mixed_precision, gpu_cache_fraction=config.gpu_cache_fraction,
            depth=candidate.get("depth", 3), width=candidate.get("width", 32),
            dropout=candidate.get("dropout", 0.1), early_stopping=candidate.get("early_stopping", False),
            augment=candidate.get("augment", False), cosine_schedule=candidate.get("cosine_schedule", False),
            normalize_images=candidate.get("normalize_images", False),
            imagenet_normalize=candidate.get("imagenet_normalize", False),
            pretrained=candidate.get("pretrained", True), tta=candidate.get("tta", False),
        )
        if family != "torch_pretrained":
            params["hidden"] = candidate["hidden"]
        if family == "torch_ensemble":
            estimator = TorchEnsemble(member_params=params, size=candidate.get("ensemble_size", 3),
                                      device=config.device, **common)
        else:
            estimator = TorchClassifier(**params, device=config.device, **common)
    else:
        raise ValueError(f"Unsupported model family: {family}")
    if family in {"catboost", "catboost_cross", "lightgbm", "catboost_ensemble", "tabular_blend", "cross_boost_blend"}:
        from superagent.models.tabular import NativeCategories

        return Pipeline([("preprocess", NativeCategories(
            profile.get('tabular_columns') or profile['features'],
            profile['numeric_columns'], profile.get('datetime_columns', []))),
                         ("classifier", estimator)])
    return Pipeline([("preprocess", preprocessor(profile, candidate, config.image_root)),
                     ("classifier", estimator)])
