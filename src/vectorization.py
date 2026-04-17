from collections import Counter
import torch
from gensim.models import Word2Vec


def train_word2vec(X_train, vector_size=100, window=5, min_count=2, epochs=10, workers=4):
    w2v_model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    w2v_model.build_vocab(X_train)
    w2v_model.train(X_train, total_examples=len(X_train), epochs=epochs)
    return w2v_model


def save_word2vec(w2v_model, path):
    w2v_model.save(path)


def load_word2vec(path):
    return Word2Vec.load(path)


def build_vocab(X_train, min_freq=2):
    special_tokens = ['<pad>', '<unk>']
    counter = Counter()

    for tokens in X_train:
        counter.update(tokens)

    vocab = {token: idx for idx, token in enumerate(special_tokens)}

    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab


def tokens_to_indices(self, tokens):
    return [self.vocab[token] if token in self.vocab else self.unk_idx for token in tokens]

def build_pretrained_embedding(vocab, w2v_model):
    embedding_dim = w2v_model.vector_size
    pre_trained_emb = torch.zeros(len(vocab), embedding_dim)

    for word, idx in vocab.items():
        if word in w2v_model.wv:
            pre_trained_emb[idx] = torch.tensor(w2v_model.wv[word])

    return pre_trained_emb