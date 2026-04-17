import os
import torch


import os
import json
import torch


def save_artifacts(
    save_dir,
    vocab,
    pre_trained_emb,
    models_dict,
    config_dict=None
):
    os.makedirs(save_dir, exist_ok=True)

    torch.save(vocab, os.path.join(save_dir, 'vocab.pt'))
    torch.save(pre_trained_emb, os.path.join(save_dir, 'pre_trained_emb.pt'))

    for model_name, model in models_dict.items():
        path = os.path.join(save_dir, f'{model_name}.pt')
        torch.save(model.state_dict(), path)

    if config_dict is not None:
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)


def load_artifacts(load_dir):
    vocab = torch.load(os.path.join(load_dir, 'vocab.pt'))
    pre_trained_emb = torch.load(os.path.join(load_dir, 'pre_trained_emb.pt'))
    return vocab, pre_trained_emb


def load_config(load_dir):
    config_path = os.path.join(load_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model_weights(model, load_dir, model_name, device):
    path = os.path.join(load_dir, f'{model_name}.pt')
    model.load_state_dict(torch.load(path, map_location=device))
    return model