"""Streamlit sections for a read-only, saved-experiment exploration app."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.inventory_tagging.data import CLASSES, sha256
from .services import (ROOT, confusion_table, dataset_identity, feature_view,
                       local_dataset, predict_one, prepare_image, read_csv,
                       saved_predictions, stamp)

PAGES = ['Overview', 'Prediction Lab', 'Model Comparison', 'Error Analysis', 'Dataset Explorer', 'How It Works']
FEATURE_NAMES = {'pixels': 'Raw pixels', 'hog': 'HOG · 7×7 cells', 'hog_fine': 'HOG · 4×4 cells'}
FAMILY_NAMES = {'logistic_regression': 'Logistic Regression', 'knn': 'k-nearest neighbours',
                'random_forest': 'Random Forest', 'linear_svm': 'Linear SVM', 'rbf_svm': 'RBF SVM'}
METRICS = {'test_accuracy': 'Test accuracy', 'test_macro_f1': 'Test Macro F1',
           'validation_macro_f1': 'Validation Macro F1', 'single_ms_median': 'Saved single-image latency (ms)',
           'batch_ms_per_image': 'Saved batch latency per image (ms)', 'fit_seconds': 'Fit time (seconds)',
           'model_bytes': 'Saved model size (bytes)', 'feature_count': 'Feature count'}


def navigate(page):
    st.session_state['section'] = page


def figure_fallback(reports, name, caption=None):
    path = reports / 'figures' / name
    if path.is_file():
        st.image(str(path), caption=caption)


def overview(work_dir, results, frozen, records):
    st.header('A closer look at clothing classification')
    st.write('Compare fixed image features and classical classifiers on ten Fashion-MNIST clothing categories. Explore saved evidence, then try the trained models on the same image.')
    chosen = results.set_index('name').loc[frozen['selected']]
    st.subheader('Model selected using validation data')
    st.write(frozen['selected'].replace('_', ' '))
    cols = st.columns(3)
    cols[0].metric('Test accuracy', f'{chosen.test_accuracy:.2%}')
    cols[1].metric('Test Macro F1', f'{chosen.test_macro_f1:.4f}')
    cols[2].metric('Training images', f'{int(chosen.train_images):,}')
    st.caption('Saved official test results. Model selection used validation Macro F1, then validation accuracy, before test evaluation. This dashboard does not train or re-evaluate models.')
    st.write(f"**Shared split:** {frozen['train_count']:,} training / {frozen['validation_count']:,} validation images. The official test set has {int(chosen.get('test_images', 10000)):,} images.")
    st.write('**Image format:** 28 × 28 grayscale, 0–255 pixels. The feature extractor divides by 255 once. Clothing is light against a dark background.')
    left, right = st.columns(2)
    left.button('Open Prediction Lab', on_click=navigate, args=('Prediction Lab',), width='stretch')
    right.button('Explore classification errors', on_click=navigate, args=('Error Analysis',), width='stretch')
    st.subheader('Recorded feature and classifier combinations')
    combinations = results[['name', 'representation', 'family']].copy()
    combinations['Saved model available'] = combinations.name.map(lambda n: (work_dir / 'models' / (n + '.joblib')).is_file())
    st.dataframe(combinations, hide_index=True, width='stretch')
    st.write('**Raw pixels** preserve brightness at each of 784 positions. **HOG** summarizes edge directions in local cells. **Classical classifiers** learn decision boundaries from these fixed features; they do not learn the image representation itself.')
    st.caption('Categories: ' + ' · '.join(CLASSES))


def prediction_lab(work_dir, results, frozen, records):
    st.header('Prediction Lab')
    available = [name for name in records if (work_dir / 'models' / (name + '.joblib')).is_file()]
    if not available:
        st.info('Prediction needs the fitted model files, which are excluded from Git. Run the reproduction commands in the README, then choose runs/reproduction as the experiment folder.')
        return
    if 'predictor' not in st.session_state or st.session_state.predictor not in available:
        st.session_state.predictor = frozen['selected'] if frozen['selected'] in available else available[0]
    primary = st.selectbox('Predictor', available, key='predictor')
    st.session_state['compare_predictors'] = [name for name in st.session_state.get('compare_predictors', []) if name in available and name != primary]
    others = st.multiselect('Compare with other saved models', [n for n in available if n != primary], key='compare_predictors')
    mode = st.radio('Input mode', ['Exact benchmark input', 'Experimental photo preparation'], horizontal=True, key='input_mode')
    experimental = mode.startswith('Experimental')
    if experimental:
        st.warning('Ordinary photographs have not been benchmarked. These models were trained on small Fashion-MNIST images; predictions may be unreliable. Preparation here never changes training or evaluation data.')
    else:
        st.caption('Exact benchmark mode accepts only an upright 28×28 grayscale PNG in mode L. Pixels pass through unchanged.')
    source = st.radio('Image source', ['Included training sample', 'Upload an image'], horizontal=True)
    uploaded = st.file_uploader('Upload image', type=['png', 'jpg', 'jpeg'],
                               help='Benchmark mode requires a 28×28 grayscale PNG. Other PNG/JPEG inputs require experimental preparation.',
                               key='prediction_upload') if source == 'Upload an image' else None
    if source == 'Upload an image' and uploaded is None:
        st.info('Choose an image to see its prepared input.')
        return
    sample = ROOT / 'assets/sample.png'
    if source != 'Upload an image' and not sample.is_file():
        st.info('The included sample is missing. Choose Upload an image.')
        return
    content = uploaded.getvalue() if uploaded is not None else sample.read_bytes()
    sizing, invert, brightness, contrast = 'Pad', False, 1., 1.
    if experimental:
        a, b = st.columns(2)
        sizing = a.selectbox('Fit to 28×28', ['Pad', 'Crop', 'Stretch'])
        invert = b.checkbox('Invert light background to dark', value=False)
        brightness = a.slider('Brightness', 0.2, 2.0, 1., 0.1)
        contrast = b.slider('Contrast', 0.2, 2.0, 1., 0.1)
        st.caption('Pad preserves the whole image; Crop uses a centered square; Stretch changes the aspect ratio. Adjust inversion so the garment is light on a dark background. No automatic segmentation is applied.')
    try:
        pixels, steps = prepare_image(content, experimental, sizing, invert, brightness, contrast)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        st.error(f'Could not prepare this image: {error}')
        return
    columns = st.columns(len(steps))
    for column, (label, image) in zip(columns, steps):
        column.image(image, caption=label, width=168)
    record = records[primary]
    features, drawing = feature_view(pixels, record['representation'])
    st.subheader('What this predictor receives')
    a, b = st.columns([1, 3])
    a.image(drawing, clamp=True, caption=FEATURE_NAMES[record['representation']], width=168)
    b.write(f"28×28 uint8 → divide by 255 → {FEATURE_NAMES[record['representation']]} → {len(features):,} feature values → {FAMILY_NAMES.get(record['family'], record['family'])} → category")
    b.caption('Any fitted scaler is already inside the saved model pipeline. The dashboard adds no new scaling. The HOG image shows edge directions, not model attention or a causal explanation.')
    with st.expander('Inspect the feature vector'):
        st.line_chart(pd.DataFrame({'Feature value': features}), height=170)
    chosen = [primary, *others]
    signature = (str(work_dir), hashlib.sha256(pixels.tobytes()).hexdigest(),
                 tuple((name, stamp(work_dir / 'models' / (name + '.joblib')), records[name]['model_sha256']) for name in chosen))
    if st.button('Predict and compare', type='primary'):
        runs, failures = [], []
        with st.spinner('Verifying and running the saved predictors…'):
            for name in chosen:
                try:
                    runs.append(predict_one(work_dir, records[name], pixels))
                except Exception as error:
                    failures.append(f'{name}: {type(error).__name__}: {error}')
        st.session_state['prediction_run'] = (signature, runs, failures)
    previous = st.session_state.get('prediction_run')
    if previous and previous[0] == signature:
        for error in previous[2]:
            st.error('Could not predict: ' + error)
        runs = previous[1]
        if runs:
            st.dataframe(pd.DataFrame([{k: r[k] for k in ('Model', 'Prediction', 'Inference ms')} for r in runs]), hide_index=True, width='stretch')
            st.success(f"Predicted category: {runs[0]['Prediction']}")
            st.caption('Live inference latency includes feature extraction and prediction with two CPU threads. It excludes image decoding, model loading and alternative-class calculation. A single run is not the saved repeated timing benchmark.')
            for run in runs:
                if run['alternatives'] is not None:
                    with st.expander(run['Model'] + ' · alternative classes'):
                        st.caption(run['kind'] + '. Values are specific to this model and are not comparable between models.')
                        st.dataframe(run['alternatives'], hide_index=True)
    if source == 'Included training sample':
        st.caption('The included sample is a training image for demonstrating prediction, not an additional accuracy estimate.')


def comparison(work_dir, results, frozen, records):
    st.header('Model Comparison')
    st.caption('All results are saved. The highlighted model was selected using validation data before official test evaluation.')
    a, b, c = st.columns(3)
    reps = a.multiselect('Feature representation', list(results.representation.unique()), default=list(results.representation.unique()), format_func=lambda x: FEATURE_NAMES.get(x, x))
    families = b.multiselect('Classifier', list(results.family.unique()), default=list(results.family.unique()), format_func=lambda x: FAMILY_NAMES.get(x, x))
    metrics = [key for key in METRICS if key in results.columns and results[key].notna().any()]
    metric = c.selectbox('Comparison metric', metrics, format_func=METRICS.get)
    filtered = results[results.representation.isin(reps) & results.family.isin(families)].copy()
    if filtered.empty:
        st.info('Choose at least one matching feature representation and classifier.')
        return
    filtered['Selection'] = np.where(filtered.name == frozen['selected'], 'Selected on validation', 'Other recorded model')
    ascending = metric not in ('test_accuracy', 'test_macro_f1', 'validation_macro_f1')
    filtered = filtered.sort_values(metric, ascending=ascending)
    measured = filtered.dropna(subset=[metric])
    st.plotly_chart(px.bar(measured, x=metric, y='name', color='Selection', orientation='h',
                           labels={metric: METRICS[metric], 'name': 'Model'},
                           color_discrete_map={'Selected on validation': '#167d9a', 'Other recorded model': '#b3c8d0'}), width='stretch')
    if len(measured) != len(filtered):
        st.caption('Models without a saved value for this metric are omitted from the chart and remain in the table.')
    st.dataframe(filtered, hide_index=True, width='stretch')
    if {'single_ms_median', 'test_accuracy'}.issubset(filtered.columns):
        tradeoff = filtered.dropna(subset=['single_ms_median', 'test_accuracy'])
        tradeoff = tradeoff[tradeoff.single_ms_median > 0]
        if not tradeoff.empty:
            st.subheader('Saved accuracy and inference cost')
            st.plotly_chart(px.scatter(tradeoff, x='single_ms_median', y='test_accuracy', color='representation',
                                      symbol='Selection', hover_name='name', log_x=True,
                                      labels={'single_ms_median': 'Median single-image latency (ms, log scale)', 'test_accuracy': 'Official test accuracy'}), width='stretch')
            st.caption('Saved CPU measurements include feature extraction and pipeline scaling; 20 single-image repetitions, two threads. These timings depend on the evaluation computer.')
    if {'complexity_kind', 'complexity_count'}.issubset(filtered.columns):
        st.subheader('Capacity within each model family')
        st.dataframe(filtered[['name', 'complexity_kind', 'complexity_count']], hide_index=True)
        st.caption('Tree nodes, stored training images, support vectors and linear coefficients measure different things. Their counts are not directly equivalent. Saved model size is also available in the metric selector.')


def error_analysis(work_dir, results, frozen, records):
    st.header('Error Analysis')
    st.caption('Inspect the completed, frozen test evaluation. Use these results to describe limitations, not to choose new settings or tune on the test set.')
    reports = work_dir / 'reports'
    names = list(results.name)
    model_name = st.selectbox('Evaluation model', names, index=names.index(frozen['selected']), key='error_model')
    try:
        matrix = confusion_table(reports / 'test' / (model_name + '_confusion.csv'))
    except FileNotFoundError:
        st.info('Raw confusion counts are unavailable for this model.')
        if model_name == frozen['selected']:
            figure_fallback(reports, 'selected_confusion.png', 'Saved confusion matrix of the validation-selected model')
    except (ValueError, pd.errors.ParserError) as error:
        st.warning(f'Could not read confusion counts: {error}')
    else:
        counts = matrix.to_numpy()
        rates = np.divide(counts, counts.sum(1, keepdims=True), out=np.zeros_like(counts, dtype=float), where=counts.sum(1, keepdims=True) != 0)
        display = st.radio('Matrix values', ['Counts', 'Fraction of true category'], horizontal=True)
        heatmap = go.Figure(go.Heatmap(z=counts if display == 'Counts' else rates,
            x=list(CLASSES), y=list(CLASSES), customdata=np.stack([counts, rates], axis=-1),
            colorscale='Blues', hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{customdata[0]}<br>Fraction of true category: %{customdata[1]:.1%}<extra></extra>'))
        heatmap.update_layout(xaxis_title='Predicted category', yaxis_title='True category', yaxis_autorange='reversed', height=550)
        st.plotly_chart(heatmap, width='stretch')
        recall = pd.DataFrame({'Category': CLASSES, 'Recall': np.diag(rates)})
        st.plotly_chart(px.bar(recall, x='Category', y='Recall', range_y=[0, 1]), width='stretch')
    per_class = reports / 'test/per_class.csv'
    if per_class.is_file():
        try:
            frame = read_csv(stamp(per_class))
            st.dataframe(frame[frame.name == model_name], hide_index=True, width='stretch')
        except (ValueError, KeyError, pd.errors.ParserError) as error:
            st.warning(f'Per-class metrics are unavailable: {error}')
    st.subheader('Inspect saved classifications')
    st.caption('Shirt, T-shirt/top, Pullover and Coat share outlines and sleeves, making their boundaries especially difficult.')
    try:
        predictions = saved_predictions(stamp(reports / 'test/predictions.npz'), model_name)
    except (FileNotFoundError, ValueError, KeyError, OSError) as error:
        st.info(f'Saved per-image predictions are unavailable: {error}')
        figure_fallback(reports, 'error_examples.png', 'Saved illustrative errors for the validation-selected model')
        return
    a, b, c = st.columns(3)
    true = a.selectbox('True category', ['All', *CLASSES], key='error_true')
    predicted = b.selectbox('Predicted category', ['All', *CLASSES], key='error_predicted')
    correctness = c.selectbox('Classification', ['Incorrect', 'Correct', 'All'])
    selected = predictions.copy()
    if true != 'All':
        selected = selected[selected['True category'] == true]
    if predicted != 'All':
        selected = selected[selected['Predicted category'] == predicted]
    if correctness != 'All':
        selected = selected[selected.Correct == (correctness == 'Correct')]
    st.write(f'{len(selected):,} matching saved classifications')
    st.dataframe(selected[['test_index', 'True category', 'Predicted category', 'Correct']], hide_index=True, height=230)
    if selected.empty:
        return
    if st.checkbox('Show images from the saved evaluation (read only)', value=False):
        try:
            images, labels = local_dataset(str(work_dir), 'test', dataset_identity(work_dir, 'test'))
            ids = predictions.test_index.to_numpy()
            if ids.dtype.kind not in 'iu' or (ids < 0).any() or (ids >= len(images)).any() or not np.array_equal(labels[ids], predictions.true_id.to_numpy()):
                raise ValueError('Saved prediction indices/labels do not match the local test archive.')
            show_grid(images[selected.test_index.to_numpy()[:12]],
                      [f"#{r.test_index} · {r['True category']} → {r['Predicted category']}" for _, r in selected.head(12).iterrows()])
            st.caption('First 12 matches in original test-index order. This is an inspection of saved predictions; no model is re-evaluated.')
        except (FileNotFoundError, ValueError, OSError) as error:
            st.info(f'Local test images are unavailable or incompatible: {error}')
            figure_fallback(reports, 'error_examples.png', 'Saved illustrative upper-body errors for the validation-selected model')


def show_grid(images, captions):
    columns = st.columns(4)
    for index, (image, caption) in enumerate(zip(images, captions)):
        columns[index % 4].image(image, caption=caption, width=112)


def dataset_explorer(work_dir, results, frozen, records):
    st.header('Dataset Explorer')
    st.write('Explore only the 48,000 training IDs from the persisted shared split. Validation images and the frozen test set are excluded from this section.')
    st.caption('28×28 pixels · one grayscale channel · uint8 [0,255] → float32 [0,1] inside extract().')
    reports = work_dir / 'reports'
    counts_path = reports / 'class_counts.csv'
    if counts_path.is_file():
        try:
            counts = read_csv(stamp(counts_path))
            st.plotly_chart(px.bar(counts, x='category', y='training', labels={'category': 'Category', 'training': 'Training images'}), width='stretch')
            st.dataframe(counts, hide_index=True)
        except (KeyError, ValueError) as error:
            st.warning(f'Class-count report unavailable: {error}')
    category = st.selectbox('Training category', CLASSES)
    try:
        identities = dataset_identity(work_dir, 'train')
        expected = next(iter(records.values())).get('split_sha256')
        if expected and sha256(work_dir / 'data/split.npz') != expected:
            raise ValueError('The training split does not match the frozen experiment.')
    except (FileNotFoundError, ValueError, OSError) as error:
        st.info(f'Local training data are unavailable: {error}. Saved training visualizations are shown below. Use the README preparation commands outside this dashboard to obtain data.')
        figure_fallback(reports, 'training_samples.png', 'Saved training examples')
        figure_fallback(reports, 'class_means.png', 'Saved class means from training IDs only')
        return
    try:
        with st.spinner('Reading verified training archives…'):
            images, labels = local_dataset(str(work_dir), 'train', identities)
        class_ids = np.flatnonzero(labels == CLASSES.index(category))
        st.write(f'{len(class_ids):,} training images in {category}')
        page = st.slider('Example page', 1, max(1, (len(class_ids) + 11) // 12), 1)
        ids = class_ids[(page - 1) * 12:page * 12]
        show_grid(images[ids], [f'{category} · training position {i}' for i in ids])
        mean = images[class_ids].mean(axis=0)
        a, b = st.columns(2)
        a.plotly_chart(px.imshow(mean, zmin=0, zmax=255, color_continuous_scale='gray', labels={'color': 'Mean pixel intensity'}, title='Mean image · training category'), width='stretch')
        counts = np.bincount(images[class_ids].ravel(), minlength=256)
        b.plotly_chart(px.line(x=np.arange(256), y=counts, labels={'x': 'Original uint8 pixel intensity', 'y': 'Pixel count'}, title='Training-category pixel distribution'), width='stretch')
        representation = st.selectbox('Training sample representation', ['pixels', 'hog', 'hog_fine'], format_func=FEATURE_NAMES.get)
        vector, drawing = feature_view(images[ids[0]], representation)
        st.image(drawing, clamp=True, width=168, caption=f'{FEATURE_NAMES[representation]} · {len(vector)} feature values')
    except (ValueError, OSError, IndexError) as error:
        st.warning(f'Could not explore these training data: {error}')


def how_it_works(work_dir, results, frozen, records):
    st.header('How It Works')
    st.write('**Image → preprocessing → raw pixels or HOG → saved classifier → clothing category**')
    st.write('The benchmark starts with an upright 28×28 grayscale image. The shared extract() function converts uint8 pixels to float32 and divides by 255 exactly once. A saved model pipeline applies any training-fitted scaler internally.')
    st.subheader('Two fixed representations')
    st.write('Raw pixels flatten the image into 784 brightness values, preserving position. HOG groups local gradients into nine edge-direction bins and normalizes neighbouring blocks. The saved configurations use 7×7 cells (324 features) or finer 4×4 cells (1,296 features). Neither representation learns from other images.')
    figure_fallback(work_dir / 'reports', 'hog_visualization.png', 'Saved training images and their HOG edge visualizations')
    st.subheader('Classifiers in this experiment')
    st.dataframe(pd.DataFrame([
        ('Logistic Regression', 'Learns a linear score for each category; a fitted standard scaler is included.'),
        ('k-nearest neighbours', 'Votes using nearby stored training feature vectors, weighted by distance.'),
        ('Random Forest', 'Combines predictions from many decision trees.'),
        ('Linear SVM', 'Learns separating linear margins between categories.'),
        ('RBF SVM', 'Uses kernel similarity to form nonlinear boundaries.')], columns=['Classifier', 'How it predicts']), hide_index=True, width='stretch')
    st.write('Shirts, pullovers, coats and T-shirts often share silhouettes. At 28×28 pixels, texture, sleeves and collars can be ambiguous. Error Analysis shows the actual saved confusions rather than assuming every mistake has the same cause.')
    st.subheader('Where a small CNN would fit')
    st.write('A CNN would learn filters and spatial representations from training images, then learn the classifier jointly. Phase 2 is planned. This repository has no verified CNN checkpoint or evaluation results, so no CNN scores or predictions are shown.')
    st.subheader('Experiment integrity')
    st.write('48,000 training images fit the models; 12,000 validation images choose settings and the selected model. Those choices are frozen before the 10,000-image official test evaluation. The dashboard reads saved results and checks every loaded model against its recorded SHA-256. It never retrains, tunes, rewrites reports or modifies the split.')
