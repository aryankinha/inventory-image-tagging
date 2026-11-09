"""A small local prediction page for the completed Phase 2 experiment."""
from pathlib import Path
import importlib.util
import json
import pandas as pd
import streamlit as st

from .services import ROOT, prepare_image, stamp


@st.cache_resource(max_entries=8, show_spinner=False)
def predictor(folder, kind, seed, identities):
    from src.inventory_tagging_dl.predict import load_predictor
    return load_predictor(Path(folder),kind,seed)


def show(work_dir):
    st.header('Phase 2: learned image features')
    folder=work_dir/'reports/phase2'
    if not (folder/'frozen.json').exists():
        st.info('This folder has no completed Phase 2 results. Use . for the included experiment.')
        return
    scores=pd.read_csv(folder/'test/results.csv').groupby('model')[['test_accuracy','test_macro_f1']].mean()
    st.dataframe(scores.rename(index={'cnn':'CNN','hybrid':'CNN + Random Forest'},columns={'test_accuracy':'Test accuracy','test_macro_f1':'Macro F1'}),width='stretch')
    st.caption('Mean of three runs. Choices used validation data. These scores apply to Fashion-MNIST.')
    if importlib.util.find_spec('torch') is None:
        st.info('For local Phase 2 predictions, run: bash scripts/setup_mac.sh --phase2')
        return
    options=['cnn']
    if (folder/'inference.json').exists():
        record=json.loads((folder/'inference.json').read_text())
        if all((folder/'checkpoints'/r['file']).is_file() for r in record['hybrids'].values()):
            options.append('hybrid')
    kind=st.selectbox('Prediction model',options,format_func={'cnn':'CNN','hybrid':'CNN + Random Forest'}.get,key='p2_model')
    uploaded=st.file_uploader('Choose a clothing image',type=['png','jpg','jpeg'],key='p2_image')
    photo=st.checkbox('Prepare an ordinary photo (experimental)',key='p2_photo')
    photo_options={}
    invert=False
    if photo:
        from .photo import photo_controls
        photo_options=photo_controls('p2_photo_settings')
        if not photo_options['isolate']:
            invert=st.checkbox('Invert light and dark',key='p2_invert')
    if photo:
        st.warning('Ordinary photos have not been tested for accuracy. Preview the prepared image first.')
    content=uploaded.getvalue() if uploaded else (ROOT/'assets/sample.png').read_bytes()
    try:
        pixels,steps=prepare_image(content,experimental=photo,invert=invert,**photo_options)
        st.image(pixels,caption='28 × 28 image used for prediction',width=180)
        if photo:
            with st.expander('Check the prepared picture'):
                for name,image in steps[:-1]:
                    st.image(image,caption=name,width=260)
        if st.button('Predict image',key='p2_predict',type='primary'):
            frozen=json.loads((folder/'frozen.json').read_text())
            files=[folder/'frozen.json',folder/'checkpoints'/f"{frozen['cnn_config']}_seed42.pt"]
            if kind=='hybrid':
                record=json.loads((folder/'inference.json').read_text())
                files += [folder/'inference.json',folder/'checkpoints'/record['hybrids']['42']['file']]
            from src.inventory_tagging_dl.predict import predict
            result=predict(pixels[None],predictor(str(folder),kind,42,tuple(stamp(f) for f in files)))
            st.success('Predicted category: '+result['categories'][0])
            st.caption(f"Prediction time: {result['elapsed_ms']:.2f} ms. Excludes loading and image decoding.")
    except (OSError,ValueError,KeyError,RuntimeError,EOFError) as error:
        st.error(f'Could not predict this image: {error}')
    if len(options)==1:
        st.caption('To enable the hybrid locally, prepare the training data and run: bash scripts/run_python.sh -m src.inventory_tagging_dl.cli export')
