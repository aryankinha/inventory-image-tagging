"""Phase 2 checks. They use tiny random data, so they run in seconds and need no downloads."""
import numpy as np
import pytest

torch = pytest.importorskip('torch', reason='Phase 2 needs PyTorch: see requirements-phase2.txt')

from src.inventory_tagging_dl import cnn


def test_every_config_maps_pictures_to_ten_scores_and_64_features():
    x = torch.rand(4, 1, 28, 28)
    for config in cnn.CONFIGS.values():
        model = cnn.SmallCNN(**config).eval()
        assert model(x).shape == (4, 10)
        assert model.embedding(x).shape == (4, 64)


def test_ablation_switches_change_the_model_they_claim_to_change():
    count = {name: cnn.count_parameters(cnn.SmallCNN(**config)) for name, config in cnn.CONFIGS.items()}
    assert count['no_batch_norm'] < count['full']          # batch norm adds scale and shift weights
    assert count['three_blocks'] != count['full']
    assert count['no_dropout'] == count['no_augmentation'] == count['full']   # these add no weights
    assert cnn.count_parameters(cnn.SmallCNN(), head=False) < count['full']


def test_augmentation_keeps_shape_and_range_and_only_shifts_or_mirrors():
    torch.manual_seed(0)
    x = torch.rand(50, 1, 28, 28)
    out = cnn.augment(x)
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    still = cnn.augment(x, shift=0)                          # no shift: each picture is itself or its mirror
    assert all(torch.equal(a, b) or torch.equal(a, b.flip(2)) for a, b in zip(still, x))
    assert cnn.augment(torch.zeros(3, 1, 28, 28)).sum() == 0  # black background stays black


def test_early_stopping_restores_the_best_validation_epoch(monkeypatch):
    monkeypatch.setitem(cnn.TRAIN, 'max_epochs', 4)
    rng = np.random.default_rng(0)
    x, y = torch.rand(200, 1, 28, 28), torch.from_numpy(rng.integers(0, 10, 200))
    model, history, summary = cnn.train_cnn(cnn.CONFIGS['full'], 0, x[:150], y[:150], x[150:], y[150:], torch.device('cpu'))
    best = history.validation_loss.idxmin()
    assert summary['best_epoch'] == history.epoch[best] and len(history) <= 4
    logits, _ = cnn.forward_all(model, x[150:], torch.device('cpu'))
    assert torch.nn.functional.cross_entropy(logits, y[150:]).item() == pytest.approx(history.validation_loss.min(), rel=1e-4)


def test_same_seed_gives_the_same_model():
    x, y = torch.rand(100, 1, 28, 28), torch.randint(0, 10, (100,))
    runs = [cnn.train_cnn(cnn.CONFIGS['plain'], 7, x[:80], y[:80], x[80:], y[80:], torch.device('cpu'))[1] for _ in range(2)]
    assert np.allclose(runs[0].train_loss, runs[1].train_loss)


def test_phase2_uses_the_phase1_split():
    from src.inventory_tagging.data import ROOT
    if not (ROOT / 'data/raw/train-images-idx3-ubyte.gz').exists():
        pytest.skip('Training archives are optional on a fresh clone.')
    from src.inventory_tagging_dl import experiments
    x_train, y_train, x_val, y_val = experiments.training_data()
    assert (len(x_train), len(x_val)) == (48000, 12000)
    assert np.bincount(y_val.numpy()).tolist() == [1200] * 10
