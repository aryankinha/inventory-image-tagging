"""Check navigation, included examples and any local prediction models without training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
from dashboard.services import experiment, predict_one, prepare_image, sample_gallery
from dashboard.views import PAGES


def check(work_dir, require_models=False):
    results, frozen, records = experiment(work_dir)
    samples = sample_gallery()
    if len(samples) != 10 or len({s['category'] for s in samples}) != 10:
        raise ValueError('The ten-category sample gallery is incomplete.')
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=40)
    app.sidebar.text_input[0].set_value(str(work_dir)).run(timeout=40)
    for section in PAGES:
        app.sidebar.radio[0].set_value(section).run(timeout=40)
        if app.exception or app.error:
            raise ValueError('Dashboard page failed: ' + section)
    available = [r for r in records.values() if (work_dir / 'models' / (r['name'] + '.joblib')).is_file()]
    if require_models and len(available) != len(records):
        raise ValueError('This check requires all saved models, but some are missing.')
    for record in available:
        content = (ROOT / 'assets/examples' / samples[0]['filename']).read_bytes()
        pixels, _ = prepare_image(content)
        result = predict_one(work_dir, record, pixels)
        if result['Inference ms'] <= 0:
            raise ValueError('Prediction timing failed: ' + record['name'])
    if len(available) >= 2:
        app.sidebar.radio[0].set_value('Prediction Lab').run(timeout=40)
        first, second = available[0]['name'], available[1]['name']
        app.selectbox(key='predictor').set_value(first).run()
        app.multiselect(key='compare_predictors').set_value([second]).run()
        app.button[0].click().run(timeout=40)
        if app.exception or app.error or len(app.dataframe[0].value) != 2:
            raise ValueError('Comparing two models on one image failed.')
    return {'status': 'passed', 'sections_checked': len(PAGES), 'gallery_categories': len(samples),
            'real_models_verified_and_run': len(available), 'two_model_ui_comparison': len(available) >= 2,
            'note': 'No training or downloads. Models missing from a fresh clone are reported, not substituted.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT)
    parser.add_argument('--require-models', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.work_dir.resolve(), args.require_models), indent=2))


if __name__ == '__main__':
    main()
