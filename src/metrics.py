import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


def calculate_classification_metrics(labels, preds, probs):
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc
    }


def plot_training_history(train_losses, test_losses, train_accs, test_accs, model_name='Model'):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_roc_curve(labels, probs, model_name='Model'):
    roc_auc = roc_auc_score(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f'{model_name} ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_metrics_dataframe(accuracy_dict, f1_score_dict, roc_auc_dict):
    metrics_df = pd.DataFrame({
        'Accuracy': accuracy_dict,
        'F1-score': f1_score_dict,
        'ROC AUC': roc_auc_dict
    }).T

    return metrics_df


def plot_model_comparison(metrics_df, title='Model Comparison by Metrics'):
    x = np.arange(len(metrics_df.index))
    model_names = list(metrics_df.columns)
    n_models = len(model_names)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_list = []
    for i, model_name in enumerate(model_names):
        shift = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + shift, metrics_df[model_name], width, label=model_name)
        bars_list.append(bars)

    ax.set_title(title)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index)
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    for bars in bars_list:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    plt.tight_layout()
    plt.show()