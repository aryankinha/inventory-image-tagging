"""Plain-language pages for exploring the saved Phase 1 work."""
import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.inventory_tagging.data import CLASSES, sha256
from .analysis import paired_outcomes
from .services import (ROOT, confusion_table, dataset_identity, feature_view,
                       local_dataset, predict_one, prepare_image, read_csv,
                       sample_gallery, saved_predictions, stamp, model_available, model_identity)

PAGES = ['Prediction Lab', 'Model Comparison', 'Error Analysis', 'About']
FEATURE_NAMES = {'pixels': 'Brightness values', 'hog': 'Edge directions', 'hog_fine': 'Finer edge directions'}
FAMILY_NAMES = {'logistic_regression': 'Linear scoring', 'knn': 'Closest examples',
                'random_forest': 'Many decision trees', 'linear_svm': 'Straight boundaries', 'rbf_svm': 'Curved boundaries'}
METRICS = {'test_accuracy': 'Final test accuracy', 'test_macro_f1': 'Category balance score',
           'validation_macro_f1': 'Practice-set balance score', 'single_ms_median': 'Typical prediction time (ms)',
           'batch_ms_per_image': 'Time per image when processed together (ms)', 'fit_seconds': 'Training time (seconds)',
           'model_bytes': 'Saved file size (MB)', 'feature_count': 'Number of image measurements'}


def model_label(name, records):
    record = records[name]
    return FEATURE_NAMES[record['representation']] + ' + ' + FAMILY_NAMES[record['family']]


def navigate(page):
    st.session_state['section'] = page


def figure_fallback(reports, name, caption=None):
    path = reports / 'figures' / name
    if path.is_file():
        st.image(str(path), caption=caption)


def about(work_dir, results, frozen, records):
    st.header('About this project')
    st.write('Phase 1 compares ways to recognise ten clothing categories from small black-and-white pictures.')
    chosen = results.set_index('name').loc[frozen['selected']]
    a, b = st.columns(2)
    a.metric('Saved test accuracy', f'{chosen.test_accuracy:.2%}')
    b.metric('Category balance score', f'{chosen.test_macro_f1:.4f}')
    st.caption('The model was chosen using separate validation images before the final test. These are saved test scores; the app does not retrain models.')
    with st.expander('How predictions work'):
        st.write('Picture → brightness or edge measurements → model → clothing category.')
        st.write('Brightness describes each pixel. Edge measurements describe outlines. The model connects these numbers to clothing names. Shirts, coats and pullovers can be hard to tell apart at 28×28 pixels.')
        st.caption('This app contains Phase 1 models only. A small CNN is planned for Phase 2.')
    if st.checkbox('Browse training pictures'):
        dataset_explorer(work_dir, results, frozen, records)


def prediction_lab(work_dir, results, frozen, records):
    st.header('Try a picture')
    available = [n for n in records if model_available(work_dir, records[n])]
    uploaded = st.file_uploader('Upload a clothing picture', type=['png', 'jpg', 'jpeg'], key='prediction_upload')
    sample = None
    if uploaded is None:
        try:
            samples = sample_gallery()
            sample = st.selectbox('Or choose a sample', samples, format_func=lambda s: s['category'], key='sample_category')
            content = (ROOT / 'assets/examples' / sample['filename']).read_bytes()
        except (OSError, ValueError, KeyError) as error:
            st.info(f'Samples are unavailable. Upload a picture instead. {error}')
            return
    else:
        content = uploaded.getvalue()
    # Keep exact dataset inputs unchanged. Other uploads explicitly use photo preparation.
    if uploaded is not None:
        try:
            with Image.open(BytesIO(content)) as image:
                is_dataset_image = image.format == 'PNG' and image.mode == 'L' and image.size == (28, 28)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            st.error('Could not open this picture. Choose a valid PNG or JPEG image.')
            return
    else:
        is_dataset_image = True
    identity = hashlib.sha256(content).hexdigest()
    if st.session_state.get('picture_identity') != identity:
        st.session_state['input_mode'] = 'Dataset image (unchanged)' if is_dataset_image else 'Prepare an ordinary picture'
        st.session_state['picture_identity'] = identity
    with st.expander('Picture and model settings'):
        if available:
            if st.session_state.get('predictor') not in available:
                st.session_state.predictor = frozen['selected'] if frozen['selected'] in available else available[0]
            primary = st.selectbox('Choose a model', available, key='predictor', format_func=lambda n: model_label(n, records))
            st.session_state['compare_predictors'] = [n for n in st.session_state.get('compare_predictors', []) if n in available and n != primary]
            others = st.multiselect('Compare with another model', [n for n in available if n != primary], key='compare_predictors', format_func=lambda n: model_label(n, records))
        else:
            primary, others = frozen['selected'], []
        mode = st.radio('Picture type', ['Dataset image (unchanged)', 'Prepare an ordinary picture'], key='input_mode')
        experimental = mode == 'Prepare an ordinary picture'
        sizing, invert, brightness, contrast = 'Pad', False, 1., 1.
        if experimental:
            fit = st.selectbox('Fit the picture', ['Keep the whole picture', 'Crop the centre', 'Stretch to a square'])
            sizing = {'Keep the whole picture': 'Pad', 'Crop the centre': 'Crop', 'Stretch to a square': 'Stretch'}[fit]
            invert = st.checkbox('The original background is light', value=False)
            brightness = st.slider('Brightness', 0.2, 2.0, 1., 0.1)
            contrast = st.slider('Contrast', 0.2, 2.0, 1., 0.1)
            st.caption('Aim for light clothing on a dark background.')
    if experimental:
        st.warning('Ordinary photos have not been tested for accuracy. For best results, use a small clothing image on a plain dark background.')
    try:
        pixels, steps = prepare_image(content, experimental, sizing, invert, brightness, contrast)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        st.error(f'Could not prepare this picture: {error}')
        return
    st.image(steps[-1][1], caption='Picture used for prediction', width=180)
    if sample is not None:
        st.caption('Training sample · Known category: ' + sample['category'])
    if not available:
        st.info('Prediction needs the saved model files for this results folder.')
        with st.expander('Enable local predictions'):
            st.code('bash scripts/build_models.sh', language='bash')
            st.write('Then choose runs/reproduction under Local setup in the sidebar.')
    chosen = [primary, *others]
    signature = (str(work_dir), hashlib.sha256(pixels.tobytes()).hexdigest(), tuple((n, model_identity(work_dir, records[n]), records[n]['model_sha256']) for n in chosen)) if available else None
    if st.button('Predict', type='primary', disabled=not available, width='stretch'):
        runs, failures = [], []
        with st.spinner('Making a prediction…'):
            for name in chosen:
                try:
                    runs.append(predict_one(work_dir, records[name], pixels))
                except Exception as error:
                    detail = 'This model file has changed. Restore the matching file.' if 'checksum mismatch' in str(error) else str(error)
                    failures.append(f'{model_label(name, records)}: {detail}')
        # A hosted model may have been downloaded by this click.
        signature = (str(work_dir), hashlib.sha256(pixels.tobytes()).hexdigest(), tuple((n, model_identity(work_dir, records[n]), records[n]['model_sha256']) for n in chosen))
        st.session_state['prediction_run'] = (signature, runs, failures)
    previous = st.session_state.get('prediction_run')
    if signature is not None and previous and previous[0] == signature:
        for error in previous[2]:
            st.error('Could not predict: ' + error)
        runs = previous[1]
        if runs:
            st.success(f"Predicted category: {runs[0]['Prediction']}")
            st.caption(f"Prediction time: {runs[0]['Inference ms']:.1f} ms")
            with st.expander('Prediction details'):
                table = pd.DataFrame([{'Model': model_label(r['Model'], records), 'Predicted category': r['Prediction'], 'Prediction time (ms)': r['Inference ms']} for r in runs])
                st.dataframe(table, hide_index=True, width='stretch')
                st.caption('Time includes image measurements and prediction, excluding image decoding and model download/loading.')
                for run in runs:
                    if run['alternatives'] is not None:
                        is_probability = run['kind'].startswith('Model probability')
                        st.caption(model_label(run['Model'], records))
                        st.caption('Model probability estimates; not independently calibrated.' if is_probability else 'Ranking scores, not probabilities. Higher means the model favours that category.')
                        st.dataframe(run['alternatives'].rename(columns={'Value': 'Probability estimate' if is_probability else 'Ranking score'}), hide_index=True)
    with st.expander('See how the picture is prepared'):
        for column, (label, image) in zip(st.columns(len(steps)), steps):
            column.image(image, caption=label, width=112)
        features, drawing = feature_view(pixels, records[primary]['representation'])
        st.image(drawing, clamp=True, caption=FEATURE_NAMES[records[primary]['representation']], width=168)
        st.caption(f'{len(features):,} image measurements. This is a feature view, not an attention map.')


@st.cache_data(max_entries=16, show_spinner=False)
def compare_predictions(identity, first_name, second_name):
    first = saved_predictions(identity, first_name)
    second = saved_predictions(identity, second_name)
    if not np.array_equal(first.test_index, second.test_index) or not np.array_equal(first.true_id, second.true_id):
        raise ValueError('These results do not describe the same images.')
    return paired_outcomes(first.true_id, first.predicted_id, second.predicted_id)


def paired_view(work_dir, names, frozen, records):
    if len(names) < 2:
        return
    st.subheader('Which mistakes does each model make?')
    st.caption('This compares answers already saved for the same images. It does not run another test or change which model was chosen.')
    a, b = st.columns(2)
    first = a.selectbox('First model', names, index=names.index(frozen['selected']) if frozen['selected'] in names else 0, format_func=lambda n: model_label(n, records), key='pair_first')
    alternatives = [n for n in names if n != first]
    core = [r for r in records.values() if r['family'] != 'rbf_svm' and r['name'] in alternatives]
    default = max(core, key=lambda r: (r['macro_f1'], r['accuracy']))['name'] if core else alternatives[0]
    second = b.selectbox('Second model', alternatives, index=alternatives.index(default), format_func=lambda n: model_label(n, records), key='pair_second')
    try:
        report = compare_predictions(stamp(work_dir / 'reports/test/predictions.npz'), first, second)
    except (OSError, ValueError, KeyError) as error:
        st.info(f'The saved answers needed for this comparison are unavailable: {error}')
        return
    st.dataframe(pd.DataFrame({'Outcome': ['Both were right', 'Only the first was right', 'Only the second was right', 'Both were wrong'], 'Images': [report[k] for k in ('both_right', 'first_only_right', 'second_only_right', 'both_wrong')]}), hide_index=True)
    low, high = report['interval_95']
    st.write(f"First model minus second model: **{report['accuracy_difference'] * 100:+.2f} percentage points** in final test accuracy.")
    st.caption('A positive difference favours the first model; a negative one favours the second.')
    with st.expander('How certain is this difference?'):
        st.write(f'When we repeatedly sampled the saved image pairs, the middle 95% of the accuracy differences ranged from {low * 100:+.2f} to {high * 100:+.2f} percentage points. We repeated this 2,000 times without changing the models.')
        st.caption('This describes variation in which test pictures are sampled. It does not show how much retraining with another random seed would change the models. This is an additional description of saved results, not a new model-selection experiment.')


def comparison(work_dir, results, frozen, records):
    st.header('Saved results')
    st.write('Choose what you want to compare. The chosen model was picked using separate practice images before the final test.')
    a, b, c = st.columns(3)
    reps = a.multiselect('What the model measures', list(results.representation.unique()), default=list(results.representation.unique()), format_func=FEATURE_NAMES.get)
    families = b.multiselect('Learning method', list(results.family.unique()), default=list(results.family.unique()), format_func=FAMILY_NAMES.get)
    metric = c.selectbox('What to compare', [k for k in METRICS if k in results and results[k].notna().any()], format_func=METRICS.get)
    filtered = results[results.representation.isin(reps) & results.family.isin(families)].copy()
    if filtered.empty:
        st.info('Choose at least one matching measurement and learning method.')
        return
    filtered['Model'] = filtered.name.map(lambda n: model_label(n, records))
    filtered['Choice'] = np.where(filtered.name == frozen['selected'], 'Chosen before final test', 'Other saved model')
    filtered = filtered.sort_values(metric, ascending=metric not in ('test_accuracy', 'test_macro_f1', 'validation_macro_f1'))
    display_table = filtered.copy()
    if 'model_bytes' in display_table:
        display_table['model_bytes'] = display_table.model_bytes / 1_000_000
    measured = display_table.dropna(subset=[metric])
    chart = px.bar(measured, x=metric, y='Model', color='Choice', orientation='h', labels={metric: METRICS[metric]}, color_discrete_map={'Chosen before final test': '#167d9a', 'Other saved model': '#b3c8d0'})
    st.plotly_chart(chart, width='stretch')
    if len(measured) != len(filtered):
        st.caption('Models with no saved value for this measurement are left out of the chart.')
    table_columns = list(dict.fromkeys([metric, 'test_accuracy', 'test_macro_f1']))
    st.dataframe(display_table[['Model', 'Choice', *table_columns]].rename(columns=METRICS), hide_index=True, width='stretch')
    st.caption('Accuracy and balance scores are better when larger. Times and file sizes are smaller when less work or storage is needed. Click a table heading to sort it.')
    with st.expander('More comparisons'):
        if {'single_ms_median', 'test_accuracy'}.issubset(filtered.columns):
            points = filtered.dropna(subset=['single_ms_median', 'test_accuracy'])
            points = points[points.single_ms_median > 0]
            if not points.empty:
                st.subheader('Accuracy and prediction time')
                points = points.assign(Measurements=points.representation.map(FEATURE_NAMES))
                st.plotly_chart(px.scatter(points, x='single_ms_median', y='test_accuracy', color='Measurements', symbol='Choice', hover_name='Model', log_x=True, labels={'single_ms_median': 'Typical prediction time (ms)', 'test_accuracy': 'Final test accuracy'}), width='stretch')
                st.caption('Further up means more correct answers. Further left means faster predictions. The time axis uses uneven spacing to show both fast and slow models. Timings come from the original evaluation computer.')
        with st.expander('Saved model details'):
            if {'complexity_kind', 'complexity_count'}.issubset(filtered.columns):
                capacity = filtered[['Model', 'complexity_kind', 'complexity_count']].copy()
                descriptions = {'tree nodes': 'Decision points in the trees', 'stored training images': 'Training pictures kept for comparison', 'support vectors': 'Examples kept to define the boundaries', 'linear coefficients and intercepts': 'Numbers learned for the scoring rules'}
                capacity['complexity_kind'] = capacity.complexity_kind.map(descriptions).fillna(capacity.complexity_kind)
                st.dataframe(capacity.rename(columns={'complexity_kind': 'What is counted', 'complexity_count': 'Count'}), hide_index=True)
                st.caption('These models store different things, so their counts do not mean the same thing. File size gives a common storage measurement.')
        paired_view(work_dir, list(results.name), frozen, records)

def error_analysis(work_dir, results, frozen, records):
    st.header('Where models make mistakes')
    st.write('Find out which clothes the models mix up. These are saved final-test answers; use them to explain mistakes, not to choose new settings.')
    reports = work_dir / 'reports'
    names = list(results.name)
    name = st.selectbox('Choose a model', names, index=names.index(frozen['selected']), key='error_model', format_func=lambda n: model_label(n, records))
    try:
        matrix = confusion_table(reports / 'test' / (name + '_confusion.csv'))
    except (OSError, ValueError) as error:
        st.info(f'The saved counts for this model are unavailable: {error}')
        if name == frozen['selected']:
            figure_fallback(reports, 'selected_confusion.png', 'Saved mistakes for the chosen model')
    else:
        counts = matrix.to_numpy()
        rates = np.divide(counts, counts.sum(1, keepdims=True), out=np.zeros_like(counts, dtype=float), where=counts.sum(1, keepdims=True) != 0)
        display = st.radio('Show answers as', ['Number of images', 'Share of each actual category'], horizontal=True)
        st.caption('Rows are the actual clothing category. Columns are the model’s guess. The diagonal shows correct answers. Hover over a square for its count and percentage.')
        chart = go.Figure(go.Heatmap(z=counts if display == 'Number of images' else rates, x=list(CLASSES), y=list(CLASSES), customdata=np.stack([counts, rates], axis=-1), colorscale='Blues', hovertemplate='Actual: %{y}<br>Guessed: %{x}<br>Images: %{customdata[0]}<br>Share of actual category: %{customdata[1]:.1%}<extra></extra>'))
        chart.update_layout(xaxis_title='Model guessed', yaxis_title='Actual category', yaxis_autorange='reversed', height=550)
        st.plotly_chart(chart, width='stretch')
        with st.expander('Results by category'):
            st.subheader('How many items in each category were found correctly?')
            recall = pd.DataFrame({'Category': CLASSES, 'Share found correctly': np.diag(rates)})
            chart = px.bar(recall, x='Category', y='Share found correctly', range_y=[0, 1])
            chart.update_yaxes(tickformat='.0%')
            st.plotly_chart(chart, width='stretch')
    per_class = reports / 'test/per_class.csv'
    if per_class.is_file():
        with st.expander('More scores for each category'):
            frame = read_csv(stamp(per_class))
            frame = frame[frame.name == name].drop(columns='name').rename(columns={'category': 'Category', 'precision': 'Share of guesses that were right', 'recall': 'Share of items found correctly', 'f1-score': 'Balance score', 'support': 'Images'})
            st.dataframe(frame, hide_index=True)
    st.subheader('Look at individual answers')
    st.caption('Shirts, T-shirts, pullovers and coats share similar shapes, so they can be hard to tell apart in a tiny picture.')
    try:
        predictions = saved_predictions(stamp(reports / 'test/predictions.npz'), name)
    except (OSError, ValueError, KeyError) as error:
        st.info(f'The saved answers for individual images are unavailable: {error}')
        figure_fallback(reports, 'error_examples.png', 'Saved examples of mistakes by the chosen model')
        return
    a, b, c = st.columns(3)
    true = a.selectbox('Actual category', ['All', *CLASSES], key='error_true')
    predicted = b.selectbox('Model guessed', ['All', *CLASSES], key='error_predicted')
    correctness = c.selectbox('Show answers', ['Incorrect', 'Correct', 'All'])
    selected = predictions.copy()
    if true != 'All':
        selected = selected[selected['True category'] == true]
    if predicted != 'All':
        selected = selected[selected['Predicted category'] == predicted]
    if correctness != 'All':
        selected = selected[selected.Correct == (correctness == 'Correct')]
    st.write(f'{len(selected):,} saved answers match these choices')
    st.dataframe(selected[['test_index', 'True category', 'Predicted category', 'Correct']].rename(columns={'test_index': 'Image number', 'True category': 'Actual category', 'Predicted category': 'Model guessed', 'Correct': 'Correct answer'}), hide_index=True, height=230)
    if selected.empty:
        return
    if st.checkbox('Show the pictures for these saved answers', value=False):
        try:
            images, labels = local_dataset(str(work_dir), 'test', dataset_identity(work_dir, 'test'))
            ids = predictions.test_index.to_numpy()
            if (ids >= len(images)).any() or not np.array_equal(labels[ids], predictions.true_id.to_numpy()):
                raise ValueError('The saved answers do not match the local image files.')
            show_grid(images[selected.test_index.to_numpy()[:12]], [f"#{r.test_index} · {r['True category']} → {r['Predicted category']}" for _, r in selected.head(12).iterrows()])
            st.caption('The first 12 matches in the original image order. These are saved answers, not a new evaluation.')
        except (OSError, ValueError) as error:
            st.info('The full test-image files are not on this Mac, or do not match these answers. Saved example pictures are shown below when available.')
            figure_fallback(reports, 'error_examples.png', 'Saved examples of mistakes by the chosen model')


def show_grid(images, captions):
    columns = st.columns(4)
    for index, (image, caption) in enumerate(zip(images, captions)):
        columns[index % 4].image(image, caption=caption, width=112)


def dataset_explorer(work_dir, results, frozen, records):
    st.header('Dataset Explorer')
    st.write('Browse the pictures used for training. Practice and final-test pictures are kept out of this section.')
    st.caption('Each image has 28×28 pixels and one shade of gray per pixel. The model converts brightness from 0–255 to 0–1 before making image measurements.')
    reports = work_dir / 'reports'
    counts_path = reports / 'class_counts.csv'
    if counts_path.is_file():
        counts = read_csv(stamp(counts_path))
        st.plotly_chart(px.bar(counts, x='category', y='training', labels={'category': 'Category', 'training': 'Training images'}), width='stretch')
        st.dataframe(counts.rename(columns={'label': 'Category number', 'category': 'Category', 'official_train': 'Original training set', 'training': 'Used for training', 'validation': 'Used for practice'}), hide_index=True)
    category = st.selectbox('Training category', CLASSES)
    try:
        identities = dataset_identity(work_dir, 'train')
        expected = next(iter(records.values())).get('split_sha256')
        if expected and sha256(work_dir / 'data/split.npz') != expected:
            raise ValueError('The saved training groups do not match this experiment.')
    except (OSError, ValueError):
        st.info('The full training-image files are not on this Mac. You can still browse the included examples and saved training pictures below.')
        samples = sample_gallery()
        sample = next(s for s in samples if s['category'] == category)
        st.image(str(ROOT / 'assets/examples' / sample['filename']), width=168, caption='Saved training example · ' + category)
        figure_fallback(reports, 'training_samples.png', 'Saved examples from the training set')
        figure_fallback(reports, 'class_means.png', 'Average training image for each category')
        return
    images, labels = local_dataset(str(work_dir), 'train', identities)
    class_ids = np.flatnonzero(labels == CLASSES.index(category))
    st.write(f'{len(class_ids):,} training images in {category}')
    page = st.slider('Example page', 1, max(1, (len(class_ids) + 11) // 12), 1)
    ids = class_ids[(page - 1) * 12:page * 12]
    show_grid(images[ids], [category for _ in ids])
    a, b = st.columns(2)
    a.plotly_chart(px.imshow(images[class_ids].mean(axis=0), zmin=0, zmax=255, color_continuous_scale='gray', labels={'color': 'Average brightness'}, title='Average training image'), width='stretch')
    counts = np.bincount(images[class_ids].ravel(), minlength=256)
    b.plotly_chart(px.line(x=np.arange(256), y=counts, labels={'x': 'Pixel brightness', 'y': 'Number of pixels'}, title='How bright are these pictures?'), width='stretch')
    representation = st.selectbox('Show image measurements', ['pixels', 'hog', 'hog_fine'], format_func=FEATURE_NAMES.get)
    vector, drawing = feature_view(images[ids[0]], representation)
    st.image(drawing, clamp=True, width=168, caption=f'{FEATURE_NAMES[representation]} · {len(vector)} measurements')
