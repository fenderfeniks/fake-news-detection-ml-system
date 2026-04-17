import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from src.train import train_model, evaluate_model


def objective_factory(
    model_class,
    vocab,
    pre_trained_emb,
    train_iterator,
    val_iterator,
    device,
    n_epochs=10,
    patience=3
):
    def objective(trial):
        hidden_dim = trial.suggest_int('hidden_dim', 32, 256, step=32)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        model = model_class(
            vocab_size=len(vocab),
            embedding_dim=pre_trained_emb.shape[1],
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=n_layers,
            pre_trained_emb=pre_trained_emb,
            pad_idx=vocab['<pad>'],
            dropout=dropout
        ).to(device)

        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_roc_auc = 0
        patience_counter = 0

        for epoch in range(n_epochs):
            train_model(model, train_iterator, optimizer, criterion, device)

            _, _, val_labels, val_probs, _ = evaluate_model(
                model, val_iterator, criterion, device
            )

            val_roc_auc = roc_auc_score(val_labels, val_probs)

            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return best_val_roc_auc

    return objective