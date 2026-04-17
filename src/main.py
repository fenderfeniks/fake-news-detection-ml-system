import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from src.artifacts import load_artifacts, load_config, load_model_weights

from src.models import RNNModel, LSTMModel, GRUModel
from src.inference import predict_ensemble

app = FastAPI(title='Fake News Detection API')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



vocab, pre_trained_emb = load_artifacts('artifacts')
config = load_config('artifacts')

pad_idx = vocab['<pad>']

rnn_cfg = config['rnn_model']
lstm_cfg = config['lstm_model']
gru_cfg = config['gru_model']


rnn_model = RNNModel(
    vocab_size=len(vocab),
    embedding_dim=pre_trained_emb.shape[1],
    hidden_dim=rnn_cfg['hidden_dim'],
    output_dim=rnn_cfg['output_dim'],
    n_layers=rnn_cfg['n_layers'],
    pre_trained_emb=pre_trained_emb,
    pad_idx=pad_idx,
    dropout=rnn_cfg['dropout']
).to(device)

lstm_model = LSTMModel(
    vocab_size=len(vocab),
    embedding_dim=pre_trained_emb.shape[1],
    hidden_dim=lstm_cfg['hidden_dim'],
    output_dim=lstm_cfg['output_dim'],
    n_layers=lstm_cfg['n_layers'],
    pre_trained_emb=pre_trained_emb,
    pad_idx=pad_idx,
    dropout=lstm_cfg['dropout']
).to(device)

gru_model = GRUModel(
    vocab_size=len(vocab),
    embedding_dim=pre_trained_emb.shape[1],
    hidden_dim=gru_cfg['hidden_dim'],
    output_dim=gru_cfg['output_dim'],
    n_layers=gru_cfg['n_layers'],
    pre_trained_emb=pre_trained_emb,
    pad_idx=pad_idx,
    dropout=gru_cfg['dropout']
).to(device)

rnn_model = load_model_weights(rnn_model, 'artifacts', 'rnn_model', device)
lstm_model = load_model_weights(lstm_model, 'artifacts', 'lstm_model', device)
gru_model = load_model_weights(gru_model, 'artifacts', 'gru_model', device)

rnn_model.load_state_dict(torch.load('artifacts/rnn_model.pt', map_location=device))
lstm_model.load_state_dict(torch.load('artifacts/lstm_model.pt', map_location=device))
gru_model.load_state_dict(torch.load('artifacts/gru_model.pt', map_location=device))

rnn_model.eval()
lstm_model.eval()
gru_model.eval()

models = [rnn_model, lstm_model, gru_model]
weights = [0.2, 0.3, 0.5]


class TextRequest(BaseModel):
    text: str


@app.get('/')
def root():
    return {'message': 'Fake News Detection API is running'}


@app.post('/predict')
def predict(request: TextRequest):
    result = predict_ensemble(
        text=request.text,
        models=models,
        vocab=vocab,
        device=device,
        weights=weights
    )

    return result