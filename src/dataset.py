import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class FakeNewsDataset(Dataset):
    def __init__(self, X, y, vocab):
        self.vocab = vocab
        self.pad_idx = vocab['<pad>']
        self.unk_idx = vocab['<unk>']
        self.X = [self.tokens_to_indices(tokens) for tokens in X]
        self.y = list(y)

    def tokens_to_indices(self, tokens):
        return [self.vocab[token] if token in self.vocab else self.unk_idx for token in tokens]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_batch(batch, pad_idx):
    texts = [torch.tensor(item[0], dtype=torch.long) for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float)

    texts_padded = pad_sequence(
        texts,
        batch_first=True,
        padding_value=pad_idx
    )

    return texts_padded, labels