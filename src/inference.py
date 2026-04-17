import torch
from src.preprocessing import preprocess_text


def predict_text(model, text, vocab, device):
    model.eval()

    tokens = preprocess_text(text)
    unk_idx = vocab['<unk>']
    indices = [vocab.get(token, unk_idx) for token in tokens]

    if len(indices) == 0:
        indices = [unk_idx]

    text_tensor = torch.tensor([indices], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(text_tensor).squeeze(1)
        prob = torch.sigmoid(logits).item()
        pred = int(prob >= 0.5)

    return {
        'tokens': tokens,
        'probability': prob,
        'prediction': pred
    }


def predict_ensemble(text, models, vocab, device, weights):
    probs = []

    for model in models:
        result = predict_text(model, text, vocab, device)
        probs.append(result['probability'])

    weighted_prob = sum(prob * weight for prob, weight in zip(probs, weights))
    pred = int(weighted_prob >= 0.5)

    return {
        'model_probabilities': probs,
        'ensemble_probability': weighted_prob,
        'ensemble_prediction': pred
    }