"""Published weights must match the frozen record before they can be loaded."""
from io import BytesIO
import hashlib
import json

import pytest

from dashboard import hosting


def publication(tmp_path, monkeypatch, content=b'example model bytes'):
    monkeypatch.setattr(hosting, 'ROOT', tmp_path)
    (tmp_path / 'assets').mkdir()
    record = {'name': 'example', 'model_sha256': hashlib.sha256(content).hexdigest()}
    manifest = {'models': {'example': {'sha256': record['model_sha256'], 'bytes': len(content)}}}
    (tmp_path / 'assets/hosted_models.json').write_text(json.dumps(manifest))
    return record


def test_published_download_is_checked_then_reused(tmp_path, monkeypatch):
    content = b'example model bytes'
    record = publication(tmp_path, monkeypatch, content)
    calls = []
    def download(url, timeout):
        calls.append(url)
        return BytesIO(content)
    monkeypatch.setattr(hosting, 'urlopen', download)
    assert hosting.model_available(tmp_path, record)
    path = hosting.ensure_model(tmp_path, record)
    assert path.read_bytes() == content
    assert hosting.ensure_model(tmp_path, record) == path
    assert calls == [hosting.RELEASE_URL + 'example.joblib']


@pytest.mark.parametrize('content', [b'wrong model bytes!!', b'oversized corrupt model bytes'])
def test_bad_download_never_becomes_a_loadable_model(tmp_path, monkeypatch, content):
    record = publication(tmp_path, monkeypatch)
    monkeypatch.setattr(hosting, 'urlopen', lambda *a, **kw: BytesIO(content))
    with pytest.raises(ValueError, match='size|checksum'):
        hosting.ensure_model(tmp_path, record)
    assert list((tmp_path / 'models').iterdir()) == []


def test_reproduction_cannot_substitute_published_weights(tmp_path, monkeypatch):
    record = publication(tmp_path, monkeypatch)
    assert hosting.published_model(tmp_path / 'runs/phase1', record) is None
    assert hosting.published_model(tmp_path, dict(record, model_sha256='0' * 64)) is None
