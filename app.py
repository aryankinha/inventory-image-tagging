"""Explore the saved Phase 1 experiment without changing its frozen artifacts."""
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
import streamlit as st

from dashboard.services import ROOT, experiment
from dashboard.views import PAGES, about, comparison, error_analysis, prediction_lab

st.set_page_config(page_title='Inventory Image Tagging', page_icon='🧥', layout='wide')
st.title('Inventory Image Tagging')
st.caption('Choose a clothing picture and find its category.')
# Keep predictor choices when their widgets are temporarily hidden by navigation.
for key in ('predictor', 'compare_predictors', 'input_mode'):
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]
section = st.sidebar.radio('Open', PAGES, key='section', format_func={'Prediction Lab': 'Predict', 'Model Comparison': 'Results', 'Error Analysis': 'Mistakes', 'About': 'About'}.get)
with st.sidebar.expander('Local setup'):
    folder = st.text_input('Results folder', value='.', help='Use . for the included results, or runs/reproduction for your own completed run.')
work_dir = Path(folder).expanduser()
if not work_dir.is_absolute():
    work_dir = ROOT / work_dir
work_dir = work_dir.resolve()


try:
    results, frozen, records = experiment(work_dir)
except FileNotFoundError:
    st.info('Choose a completed experiment folder containing reports/test/results.csv and reports/frozen.json.')
    st.stop()
except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as error:
    st.error(f'Could not open this experiment: {error}. Check the report files and folder path.')
    st.stop()

views = dict(zip(PAGES, [prediction_lab, comparison, error_analysis, about]))
try:
    views[section](work_dir, results, frozen, records)
except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, BadZipFile) as error:
    st.error(f'Could not show these saved files: {error}. Check that the files belong to this completed experiment.')
