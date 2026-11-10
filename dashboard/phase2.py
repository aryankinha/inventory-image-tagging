"""A compact prediction page for the verified CNN and optional local hybrid."""
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from .services import ROOT, prepare_image, read_json, stamp


@st.cache_resource(max_entries=2, show_spinner=False)
def predictor(folder, kind, seed, identities):
    from src.inventory_tagging_dl.predict import load_predictor
    return load_predictor(Path(folder), kind, seed)


def show(work_dir):
    st.header('Try the neural model')
    folder = work_dir / 'reports/phase2'
    if not (folder / 'frozen.json').exists():
        st.info('This folder has no completed Phase 2 results. Use . for the included experiment.')
        return
    if importlib.util.find_spec('torch') is None:
        st.info('Neural-model prediction is unavailable in this installation.')
        return

    options = ['cnn']
    try:
        record = read_json(stamp(folder / 'inference.json'))
        hybrid = record.get('hybrids', {}).get('42', {})
        filename = hybrid.get('file', '')
        if filename and Path(filename).name == filename and (folder / 'checkpoints' / filename).is_file():
            options.append('hybrid')
    except (OSError, ValueError, TypeError, KeyError):
        pass  # The CNN works without an exported hybrid.
    if st.session_state.get('p2_model') not in options:
        st.session_state['p2_model'] = options[0]

    inputs, preview = st.columns([1.2, 1], gap='large')
    with inputs:
        uploaded = st.file_uploader('Choose a clothing image', type=['png', 'jpg', 'jpeg'], key='p2_image')
        content = uploaded.getvalue() if uploaded else (ROOT / 'assets/sample.png').read_bytes()
        kind = st.selectbox('Prediction model', options, format_func={'cnn': 'CNN', 'hybrid': 'CNN + Random Forest'}.get, key='p2_model')
        try:
            with Image.open(BytesIO(content)) as image:
                benchmark = image.format == 'PNG' and image.mode == 'L' and image.size == (28, 28)
        except (OSError, ValueError, Image.DecompressionBombError):
            st.error('Could not open this picture. Choose a valid PNG or JPEG image.')
            return
        identity = hashlib.sha256(content).hexdigest()
        if st.session_state.get('p2_picture_identity') != identity:
            for key in list(st.session_state):
                if key.startswith('p2_photo_settings_') or key in ('p2_prediction_run', 'p2_invert'):
                    st.session_state.pop(key, None)
            st.session_state['p2_photo'] = not benchmark
            st.session_state['p2_picture_identity'] = identity
        photo_options, invert = {}, False
        with st.expander('Picture settings'):
            photo = st.checkbox('Prepare an ordinary photo (experimental)', key='p2_photo')
            if photo:
                from .photo import photo_controls
                photo_options = photo_controls('p2_photo_settings')
                if not photo_options['isolate']:
                    invert = st.checkbox('Invert light and dark', key='p2_invert')
        if photo:
            st.warning('Photo predictions are experimental. Use one complete item on a plain background.')

    try:
        pixels, steps = prepare_image(content, experimental=photo, invert=invert, **photo_options)
        answer = preview.container()
        preview.image(steps[0][1] if photo else pixels, caption='Your picture' if uploaded else 'Included sample · Shirt', width=280 if photo else 180)
        frozen = read_json(stamp(folder / 'frozen.json'))
        files = [folder / 'frozen.json', folder / 'checkpoints' / f"{frozen['cnn_config']}_seed42.pt"]
        if kind == 'hybrid':
            files += [folder / 'inference.json', folder / 'checkpoints' / record['hybrids']['42']['file']]
        identities = tuple(stamp(path) for path in files)
        signature = (str(folder), identity, pixels.tobytes(), kind, identities)
        if inputs.button('Predict image', key='p2_predict', type='primary', width='stretch'):
            # Clear an earlier answer even if the next model fails to load.
            st.session_state.pop('p2_prediction_run', None)
            from src.inventory_tagging_dl.predict import predict
            with st.spinner('Making a prediction…'):
                result = predict(pixels[None], predictor(str(folder), kind, 42, identities))
            st.session_state['p2_prediction_run'] = (signature, result)
        previous = st.session_state.get('p2_prediction_run')
        if previous and previous[0] == signature:
            result = previous[1]
            answer.success('Predicted category: ' + result['categories'][0])
            answer.caption(f"Prediction time: {result['elapsed_ms']:.1f} ms")
        with st.expander('See how the picture is prepared'):
            for name, image in steps:
                st.image(image, caption=name, width=160)
            st.caption('The CNN uses this 28 × 28 grayscale input. Photo accuracy is unmeasured; the model always chooses one of ten clothing categories.')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, Image.DecompressionBombError) as error:
        st.error(f'Could not predict this image: {error}')

    with st.expander('Saved evaluation results'):
        try:
            scores = pd.read_csv(folder / 'test/results.csv').groupby('model')[['test_accuracy', 'test_macro_f1']].mean()
            st.dataframe(scores.rename(index={'cnn': 'CNN', 'hybrid': 'CNN + Random Forest'}, columns={'test_accuracy': 'Test accuracy', 'test_macro_f1': 'Macro F1'}), width='stretch')
            st.caption('Mean of three runs on Fashion-MNIST. Models were selected using validation data, before the final test.')
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            st.info('The optional saved scores are unavailable. Predictions can still use the verified checkpoint.')
