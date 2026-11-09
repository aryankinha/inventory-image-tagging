"""Inference contracts, frozen-stage protection and the local Phase 2 page."""
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

pytest.importorskip('torch')
from src.inventory_tagging_dl import cnn, experiments
from src.inventory_tagging_dl.predict import load_predictor, predict
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]


def test_input_contract_rejects_wrong_scale_and_shape():
    for image in [np.zeros((1,28,28),dtype=float),np.zeros((28,28),dtype=np.uint8),np.zeros((1,29,28),dtype=np.uint8)]:
        with pytest.raises(ValueError):
            cnn.to_tensor(image)


def test_verified_checkpoint_predicts_included_image_and_has_real_probabilities():
    from PIL import Image
    image=np.asarray(Image.open(ROOT/'assets/sample.png'))[None]
    result=predict(image,load_predictor())
    assert result['categories']==['Shirt']
    assert result['elapsed_ms']>0
    assert np.isclose(np.sum(result['probabilities'][0]),1)


def test_tampered_checkpoint_is_rejected_before_loading(tmp_path):
    folder=ROOT/'reports/phase2'
    shutil.copyfile(folder/'frozen.json',tmp_path/'frozen.json')
    (tmp_path/'checkpoints').mkdir()
    tag=json.loads((folder/'frozen.json').read_text())['cnn_config']+'_seed42.pt'
    (tmp_path/'checkpoints'/tag).write_bytes(b'bad checkpoint')
    with pytest.raises(ValueError,match='checksum'):
        load_predictor(tmp_path)


@pytest.mark.parametrize('stage',['ablation','select','hybrid'])
def test_frozen_stages_cannot_rewrite_choices(tmp_path,monkeypatch,stage):
    (tmp_path/'frozen.json').write_text('{}')
    monkeypatch.setattr(experiments,'OUT',tmp_path)
    with pytest.raises(FileExistsError,match='frozen'):
        getattr(experiments,stage)()


def test_phase2_page_predicts_sample_without_retraining():
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=40)
    app.sidebar.radio[0].set_value('Phase 2').run(timeout=40)
    assert not app.exception and not app.error
    app.button(key='p2_predict').click().run(timeout=40)
    assert not app.exception and not app.error
    assert app.success[0].value=='Predicted category: Shirt'


def test_phase2_missing_reports_are_explained(tmp_path):
    from unittest.mock import patch
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=40)
    # Keep a valid Phase 1 folder while optional Phase 2 files are absent.
    (tmp_path/'reports/test').mkdir(parents=True)
    for relative in ['frozen.json','test/results.csv']:
        shutil.copyfile(ROOT/'reports'/relative,tmp_path/'reports'/relative)
    app.sidebar.text_input[0].set_value(str(tmp_path)).run(timeout=40)
    app.sidebar.radio[0].set_value('Phase 2').run(timeout=40)
    assert not app.exception
    assert any('no completed Phase 2' in item.value for item in app.info)


def test_hybrid_checksum_is_checked_before_deserialization(tmp_path):
    folder=ROOT/'reports/phase2'
    frozen=json.loads((folder/'frozen.json').read_text())
    (tmp_path/'checkpoints').mkdir()
    shutil.copyfile(folder/'frozen.json',tmp_path/'frozen.json')
    tag=f"{frozen['cnn_config']}_seed42.pt"
    shutil.copyfile(folder/'checkpoints'/tag,tmp_path/'checkpoints'/tag)
    from src.inventory_tagging.data import sha256
    (tmp_path/'inference.json').write_text(json.dumps({'frozen_sha256':sha256(tmp_path/'frozen.json'),'hybrids':{'42':{'file':'bad.joblib','sha256':'0'*64}}}))
    (tmp_path/'checkpoints/bad.joblib').write_bytes(b'not a model')
    with pytest.raises(ValueError,match='Hybrid classifier checksum'):
        load_predictor(tmp_path,'hybrid')


def test_photo_preparation_is_explicit_and_keeps_benchmark_input_unchanged():
    from io import BytesIO
    from PIL import Image
    from dashboard.services import prepare_image
    content=(ROOT/'assets/sample.png').read_bytes()
    pixels,_=prepare_image(content)
    assert np.array_equal(pixels,np.asarray(Image.open(BytesIO(content))))
    image=Image.open(BytesIO(content)).convert('RGB').resize((112,140))
    buf=BytesIO();image.save(buf,format='JPEG')
    with pytest.raises(ValueError):
        prepare_image(buf.getvalue())
    prepared,_=prepare_image(buf.getvalue(),experimental=True)
    assert prepared.dtype==np.uint8 and prepared.shape==(28,28)
    assert len(predict(prepared[None],load_predictor())['categories'])==1
