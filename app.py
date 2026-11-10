"""Explore the saved Phase 1 experiment without changing its frozen artifacts."""
from pathlib import Path
import importlib.util
from zipfile import BadZipFile

import pandas as pd
import streamlit as st

from dashboard.services import ROOT, experiment
from dashboard.views import PAGES, about, comparison, error_analysis, prediction_lab

st.set_page_config(page_title='Inventory Image Tagging', page_icon='🧥', layout='wide')
st.html('<style>.stMainBlockContainer{max-width:1120px;padding-top:2rem}</style>')
st.title('Inventory Image Tagging')
st.caption('One picture. Ten clothing categories. Choose an image to get started.')
# Keep predictor choices when their widgets are temporarily hidden by navigation.
for key in list(st.session_state):
    if key not in ('predictor', 'compare_predictors', 'input_mode', 'p2_model', 'p2_photo', 'p2_invert') and not key.startswith(('p1_photo_', 'p2_photo_settings_')):
        continue
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]
sections = list(PAGES)
if importlib.util.find_spec('torch') is not None:
    sections.insert(1, 'Phase 2')
routes = {'predict': 'Prediction Lab', 'neural': 'Phase 2', 'results': 'Model Comparison',
          'mistakes': 'Error Analysis', 'about': 'About'}
requested = st.query_params.get('view', 'predict')
if st.session_state.get('_current_view') != requested:
    st.session_state['section'] = routes.get(requested, 'Prediction Lab')
if st.session_state.get('section') not in sections:
    st.session_state['section'] = sections[0]
section = st.sidebar.radio('Explore', sections, key='section', format_func={'Prediction Lab': 'Predict', 'Model Comparison': 'Results', 'Error Analysis': 'Mistakes', 'About': 'About', 'Phase 2': 'Neural model'}.get)
current = next(slug for slug, name in routes.items() if name == section)
if st.query_params.get('view') != current:
    st.query_params['view'] = current
st.session_state['_current_view'] = current
with st.sidebar.expander('Local setup'):
    folder = st.text_input('Results folder', value='.', help='Use . for the included results, or runs/phase1 for your own completed run.')
work_dir = Path(folder).expanduser()
if not work_dir.is_absolute():
    work_dir = ROOT / work_dir
work_dir = work_dir.resolve()


if section == 'Phase 2':
    from dashboard.phase2 import show
    try:
        show(work_dir)
    except (OSError, ValueError, KeyError, RuntimeError, pd.errors.ParserError) as error:
        st.error(f'Could not open Phase 2: {error}')
    st.stop()

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
