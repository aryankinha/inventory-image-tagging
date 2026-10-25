"""View the saved Phase 1 experiment and predict a benchmark-style image."""
from io import BytesIO
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
from threadpoolctl import threadpool_limits

from src.inventory_tagging.data import CLASSES, sha256
from src.inventory_tagging.features import extract

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_model(path, expected_hash):
    if sha256(path) != expected_hash:
        raise ValueError('The saved model does not match the frozen experiment.')
    return joblib.load(path)


st.set_page_config(page_title='Inventory Image Tagging', layout='wide')
st.title('Inventory Image Tagging')
st.write('Phase 1: pixels, HOG and classical clothing classifiers.')
folder = st.sidebar.text_input('Experiment folder', value='.')
work_dir = Path(folder).expanduser()
if not work_dir.is_absolute():
    work_dir = ROOT / work_dir
reports = work_dir / 'reports'

if not (reports / 'test/results.csv').is_file() or not (reports / 'frozen.json').is_file():
    st.info('Choose a completed experiment folder containing reports/test/results.csv and reports/frozen.json.')
    st.stop()

results = pd.read_csv(reports / 'test/results.csv')
frozen = json.loads((reports / 'frozen.json').read_text())
selected = results.set_index('name').loc[frozen['selected']]
st.subheader('Model selected using validation data')
st.write(frozen['selected'].replace('_', ' '))
a, b, c = st.columns(3)
a.metric('Test accuracy', f'{selected.test_accuracy:.2%}')
b.metric('Macro F1', f'{selected.test_macro_f1:.4f}')
c.metric('Training images', f'{int(selected.train_images):,}')
st.caption('These are saved Fashion-MNIST results. Opening the dashboard does not train or evaluate models.')

st.subheader('Model comparison')
st.dataframe(results, hide_index=True)
for title, filename in [('Confusion matrix', 'selected_confusion.png'),
                        ('Examples of upper-body mistakes', 'error_examples.png')]:
    figure = reports / 'figures' / filename
    if figure.is_file():
        st.subheader(title)
        st.image(str(figure))

st.subheader('Try one image')
st.write('Use an upright 28x28 grayscale PNG on a dark background. Phone-photo accuracy has not been measured.')
record = next(model for model in frozen['models'] if model['name'] == frozen['selected'])
model_path = work_dir / 'models' / f"{record['name']}.joblib"
if not model_path.is_file():
    st.info('Prediction needs the fitted model, which is excluded from Git. Run the reproduction commands in the README, then choose runs/reproduction as the experiment folder.')
else:
    uploaded = st.file_uploader('Choose a grayscale PNG', type=['png'])
    use_sample = st.checkbox('Use the included sample image', value=True)
    source = BytesIO(uploaded.getvalue()) if uploaded is not None else (ROOT / 'assets/sample.png' if use_sample else None)
    if source is not None:
        try:
            with Image.open(source) as image:
                if image.size != (28, 28) or image.mode != 'L':
                    st.error('The image must be exactly 28x28 pixels in grayscale mode (L).')
                else:
                    pixels = np.asarray(image, dtype=np.uint8).copy()
                    st.image(image, width=168)
                    if st.button('Predict category'):
                        model = load_model(str(model_path), record['model_sha256'])
                        with threadpool_limits(limits=2):
                            prediction = int(model.predict(extract(pixels[None], record['representation']))[0])
                        st.success(f'Predicted category: {CLASSES[prediction]}')
                        st.caption('The included sample is a training image for demonstrating prediction, not an additional accuracy estimate.')
        except (UnidentifiedImageError, OSError, ValueError) as error:
            st.error(f'Could not predict this image: {error}')
