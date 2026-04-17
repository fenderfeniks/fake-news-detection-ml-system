import torch
import torch.nn as nn


class BaseSequenceModel(nn.Module):
    def __init__(
        self,
        rnn_type,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        pre_trained_emb,
        pad_idx,
        dropout=0.2
    ):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            pre_trained_emb,
            padding_idx=pad_idx
        )

        if rnn_type == 'RNN':
            self.sequence_model = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0
            )
        elif rnn_type == 'LSTM':
            self.sequence_model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0
            )
        elif rnn_type == 'GRU':
            self.sequence_model = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0
            )
        else:
            raise ValueError(f'Unsupported rnn_type: {rnn_type}')

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.rnn_type = rnn_type

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))

        if self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.sequence_model(embedded)
        else:
            output, hidden = self.sequence_model(embedded)

        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)


class RNNModel(BaseSequenceModel):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        pre_trained_emb,
        pad_idx,
        dropout=0.2
    ):
        super().__init__(
            rnn_type='RNN',
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            pre_trained_emb=pre_trained_emb,
            pad_idx=pad_idx,
            dropout=dropout
        )


class LSTMModel(BaseSequenceModel):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        pre_trained_emb,
        pad_idx,
        dropout=0.2
    ):
        super().__init__(
            rnn_type='LSTM',
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            pre_trained_emb=pre_trained_emb,
            pad_idx=pad_idx,
            dropout=dropout
        )


class GRUModel(BaseSequenceModel):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        pre_trained_emb,
        pad_idx,
        dropout=0.2
    ):
        super().__init__(
            rnn_type='GRU',
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            pre_trained_emb=pre_trained_emb,
            pad_idx=pad_idx,
            dropout=dropout
        )