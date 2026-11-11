"""The hosted deployment must install CPU-only PyTorch for the committed CNN."""
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def project_config():
    return tomllib.loads((ROOT / 'pyproject.toml').read_text())


def test_phase2_is_a_default_group_for_hosted_installs():
    assert 'phase2' in project_config()['tool']['uv']['default-groups']


def test_linux_torch_uses_the_cpu_index():
    config = project_config()
    indexes = {index['name']: index for index in config['tool']['uv']['index']}
    assert indexes['pytorch-cpu']['url'] == 'https://download.pytorch.org/whl/cpu'
    assert indexes['pytorch-cpu']['explicit'] is True
    sources = config['tool']['uv']['sources']['torch']
    assert any(source['index'] == 'pytorch-cpu' and "sys_platform == 'linux'" in source['marker']
               for source in sources)


def test_lock_has_cpu_torch_without_a_cuda_stack():
    lock = (ROOT / 'uv.lock').read_text()
    assert 'name = "torch"\nversion = "2.14.1+cpu"\nsource = { registry = "https://download.pytorch.org/whl/cpu" }' in lock
    for package in ('cuda-toolkit', 'nvidia-cublas', 'nvidia-cudnn-cu13', 'triton'):
        assert f'name = "{package}"' not in lock


def test_pip_exports_match_the_split():
    phase1 = (ROOT / 'requirements.txt').read_text()
    phase2 = (ROOT / 'requirements-phase2.txt').read_text()
    assert 'torch==' not in phase1
    assert '--extra-index-url https://download.pytorch.org/whl/cpu' in phase2
    assert "torch==2.14.1+cpu ; sys_platform == 'linux'" in phase2
    assert 'nvidia-' not in phase2
