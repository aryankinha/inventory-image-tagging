"""A small CNN for 28x28 clothing images, with the switches used by the ablation study."""
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

SEEDS = (42, 1, 2)
TRAIN = {'batch_size': 128, 'max_epochs': 30, 'patience': 5, 'lr': 1e-3, 'weight_decay': 1e-4,
         'shift_pixels': 2}

FULL = {'blocks': 2, 'batch_norm': True, 'dropout': 0.4, 'augment': True}
# The ablation removes one idea at a time from the full model, plus one run with all of them removed.
CONFIGS = {
    'full': FULL,
    'no_batch_norm': {**FULL, 'batch_norm': False},
    'no_dropout': {**FULL, 'dropout': 0.0},
    # Added after the first runs showed dropout 0.4 was too strong; judged on validation data only.
    'light_dropout': {**FULL, 'dropout': 0.2},
    'no_augmentation': {**FULL, 'augment': False},
    'three_blocks': {**FULL, 'blocks': 3},
    'plain': {'blocks': 2, 'batch_norm': False, 'dropout': 0.0, 'augment': False},
}


def pick_device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def to_tensor(images):
    """uint8 (N,28,28) -> float32 (N,1,28,28) in [0,1], the same scaling as Phase 1."""
    return torch.from_numpy(images.astype(np.float32) / 255).unsqueeze(1)


class SmallCNN(nn.Module):
    """Conv blocks (conv, optional batch norm, ReLU, max pool), then a 128 -> 64 dense layer.

    The 64 numbers after the dense layers are the picture's learned features. The hybrid model
    gives them to a classical classifier instead of the final linear layer.
    """
    def __init__(self, blocks=2, batch_norm=True, dropout=0.4, augment=True):
        super().__init__()
        layers, channels_in, size = [], 1, 28
        for channels in (32, 64, 128)[:blocks]:
            layers.append(nn.Conv2d(channels_in, channels, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(channels))
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            channels_in, size = channels, size // 2
        self.features = nn.Sequential(*layers)
        self.embed = nn.Sequential(nn.Flatten(), nn.Linear(channels_in * size * size, 128), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 10))

    def forward(self, x):
        return self.head(self.embed(self.features(x)))

    def embedding(self, x):
        return self.embed(self.features(x))


def count_parameters(model, head=True):
    modules = [model] if head else [model.features, model.embed]
    return sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)


def augment(x, shift=TRAIN['shift_pixels']):
    """Random left-right flip and a random shift of up to `shift` pixels, per picture.

    Clothing keeps its category when mirrored or nudged a little. Pixels shifted in are black,
    like the dataset background.
    """
    n = len(x)
    x = torch.where(torch.rand(n, device=x.device)[:, None, None, None] < 0.5, x.flip(3), x)
    padded = F.pad(x, (shift,) * 4)
    dy = torch.randint(0, 2 * shift + 1, (n,), device=x.device)
    dx = torch.randint(0, 2 * shift + 1, (n,), device=x.device)
    steps = torch.arange(28, device=x.device)
    rows, cols = dy[:, None] + steps, dx[:, None] + steps
    cropped = padded[torch.arange(n, device=x.device)[:, None, None], 0, rows[:, :, None], cols[:, None, :]]
    return cropped.unsqueeze(1)


@torch.inference_mode()
def forward_all(model, x, device, batch=1000):
    """Return (logits, embeddings) as CPU tensors, with dropout and batch-norm updates off."""
    model.eval().to(device)
    logits, embeddings = [], []
    for start in range(0, len(x), batch):
        embedding = model.embedding(x[start:start + batch].to(device))
        logits.append(model.head(embedding).cpu())
        embeddings.append(embedding.cpu())
    return torch.cat(logits), torch.cat(embeddings)


def train_cnn(config, seed, x_train, y_train, x_val, y_val, device):
    """Train with Adam; stop when validation loss has not improved for `patience` epochs.

    Weights from the best validation-loss epoch are restored, so reported scores never come from
    a later, more overfit epoch. Returns the model (CPU), the per-epoch history and a summary.
    """
    torch.manual_seed(seed)
    model = SmallCNN(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN['lr'], weight_decay=TRAIN['weight_decay'])
    x_train, y_train = x_train.to(device), y_train.to(device)
    shuffle = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_epoch, stale, history = float('inf'), None, 0, 0, []
    start = time.perf_counter()
    for epoch in range(1, TRAIN['max_epochs'] + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=shuffle).to(device)
        total, seen = 0.0, 0
        for first in range(0, len(order), TRAIN['batch_size']):
            batch = order[first:first + TRAIN['batch_size']]
            inputs = augment(x_train[batch]) if config['augment'] else x_train[batch]
            loss = F.cross_entropy(model(inputs), y_train[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total, seen = total + loss.item() * len(batch), seen + len(batch)
        logits, _ = forward_all(model, x_val, device)
        val_loss = F.cross_entropy(logits, y_val).item()
        predicted = logits.argmax(1).numpy()
        history.append({'epoch': epoch, 'train_loss': total / seen, 'validation_loss': val_loss,
                        'validation_accuracy': float((predicted == y_val.numpy()).mean()),
                        'validation_macro_f1': f1_score(y_val.numpy(), predicted, average='macro')})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= TRAIN['patience']:
                break
    model.load_state_dict(best_state)
    model = model.cpu()
    row = history[best_epoch - 1]
    summary = {'best_epoch': best_epoch, 'epochs_run': len(history), 'early_stopped': len(history) < TRAIN['max_epochs'],
               'validation_loss': row['validation_loss'], 'validation_accuracy': row['validation_accuracy'],
               'validation_macro_f1': row['validation_macro_f1'], 'parameters': count_parameters(model),
               'seconds': time.perf_counter() - start}
    return model, pd.DataFrame(history), summary
