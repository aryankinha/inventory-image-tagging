"""Explore the saved Phase 1 experiment without changing its frozen artifacts."""
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
import streamlit as st

from dashboard.services import ROOT, experiment, load_model
from dashboard.views import (PAGES, comparison, dataset_explorer, error_analysis,
                             how_it_works, overview, prediction_lab)

st.set_page_config(page_title='Inventory Image Tagging', page_icon='🧥', layout='wide')
st.title('Inventory Image Tagging')
st.caption('Fashion-MNIST · Phase 1 · Raw pixels, HOG and classical machine learning')
# Keep predictor choices when their widgets are temporarily hidden by navigation.
for key in ('predictor', 'compare_predictors', 'input_mode'):
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]
folder = st.sidebar.text_input('Experiment folder', value='.')
work_dir = Path(folder).expanduser()
if not work_dir.is_absolute():
    work_dir = ROOT / work_dir
work_dir = work_dir.resolve()
section = st.sidebar.radio('Explore', PAGES, key='section')
st.sidebar.caption('Read-only saved experiment. Models load only when you request a prediction.')

try:
    results, frozen, records = experiment(work_dir)
except FileNotFoundError:
    st.info('Choose a completed experiment folder containing reports/test/results.csv and reports/frozen.json.')
    st.stop()
except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as error:
    st.error(f'Could not open this experiment: {error}. Check the report files and folder path.')
    st.stop()

views = dict(zip(PAGES, [overview, prediction_lab, comparison, error_analysis, dataset_explorer, how_it_works]))
try:
    views[section](work_dir, results, frozen, records)
except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, BadZipFile) as error:
    st.error(f'Could not display this artifact: {error}. Check that the files belong to this completed experiment.')
