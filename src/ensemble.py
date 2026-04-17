import numpy as np
import optuna
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def weighted_ensemble_probs(model_probs, weights):
    ensemble_probs = np.zeros_like(model_probs[0], dtype=float)

    for probs, weight in zip(model_probs, weights):
        ensemble_probs += weight * np.array(probs)

    return ensemble_probs


def normalize_weights(weights):
    weight_sum = sum(weights)
    if weight_sum == 0:
        return [1 / len(weights)] * len(weights)
    return [w / weight_sum for w in weights]


def ensemble_predictions(probs, threshold=0.5):
    probs = np.array(probs)
    return (probs >= threshold).astype(int)


def calculate_ensemble_metrics(labels, probs, threshold=0.5):
    labels = np.array(labels)
    probs = np.array(probs)
    preds = ensemble_predictions(probs, threshold=threshold)

    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_score': f1_score(labels, preds),
        'roc_auc': roc_auc_score(labels, probs)
    }


def objective_weights_factory(model_probs, labels, model_names=None):
    labels = np.array(labels)
    model_probs = [np.array(probs) for probs in model_probs]

    if model_names is None:
        model_names = [f'model_{i}' for i in range(len(model_probs))]

    def objective(trial):
        raw_weights = [
            trial.suggest_float(f'w_{name.lower()}', 0.0, 1.0)
            for name in model_names
        ]

        weights = normalize_weights(raw_weights)
        ensemble_probs = weighted_ensemble_probs(model_probs, weights)

        return roc_auc_score(labels, ensemble_probs)

    return objective


def get_best_ensemble_weights(study, model_names):
    raw_weights = [study.best_params[f'w_{name.lower()}'] for name in model_names]
    weights = normalize_weights(raw_weights)

    return dict(zip(model_names, weights))