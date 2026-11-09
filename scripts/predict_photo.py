"""Prepare one ordinary photo and predict with a frozen Phase 1 classifier."""
import argparse
import json
from pathlib import Path
import sys
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dashboard.services import experiment, prepare_image, predict_one


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--work-dir',type=Path,default=ROOT)
    parser.add_argument('--model',help='Saved model name; defaults to the validation-selected model')
    parser.add_argument('--crop',type=float,nargs=4,metavar=('LEFT','TOP','RIGHT','BOTTOM'))
    parser.add_argument('--rotation',type=int,choices=[0,90,180,270],default=0)
    parser.add_argument('--shadow-background',action='store_true',help='Describe a background containing distinct light and shadow areas')
    parser.add_argument('--keep-background',action='store_true',help='Use the previous resize-only photo mode')
    parser.add_argument('--invert',action='store_true',help='Invert in resize-only mode')
    parser.add_argument('--preview',type=Path,help='Save the exact 28 x 28 input')
    args=parser.parse_args()
    if args.keep_background and (args.crop is not None or args.rotation or args.shadow_background):
        parser.error('Crop, rotation and shadow settings require background removal.')
    if args.invert and not args.keep_background:
        parser.error('Use --invert only with --keep-background; background removal sets polarity automatically.')
    _,frozen,records=experiment(args.work_dir.resolve())
    name=args.model or frozen['selected']
    if name not in records:
        parser.error('Choose a model recorded in this experiment.')
    pixels,_=prepare_image(args.image.read_bytes(),experimental=True,invert=args.invert,
                           isolate=not args.keep_background,crop=args.crop,rotation=args.rotation,shadow_background=args.shadow_background)
    if args.preview:
        args.preview.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(pixels).save(args.preview)
    result=predict_one(args.work_dir.resolve(),records[name],pixels)
    alternatives=result.get('alternatives')
    if alternatives is not None:
        result['alternatives']=alternatives.to_dict('records')
    result['Photo scope']='Experimental photo prediction; saved benchmark accuracy does not measure this photo.'
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    try:
        main()
    except (OSError,ValueError,KeyError) as error:
        raise SystemExit(f'Could not predict this photo: {error}') from error
