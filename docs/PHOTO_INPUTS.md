# Ordinary photo inputs: Phase 1 fixes

The original photo option resized the whole scene to 28 × 28 pixels. A wall,
shadow or second shoe could occupy more of the input than the clothing item.
Fashion-MNIST contains centred grayscale items on black backgrounds, so this
was a poor match for training.

## What changed

The ordinary-photo option now estimates the background from the picture edges,
finds a single foreground region, removes its background, crops it and centres
it in a 28 × 28 grayscale picture with black padding. Dark garments become light
foreground. These are conventional image operations using existing dependencies.
There is no added neural model, retraining, label lookup or choice of preparation
based on the predicted answer.

Under **Picture and model settings**, use **Crop around one item** if another
object is present. **Turn the picture** adjusts orientation. **Background has
strong shadows** estimates several background colours instead of one; inspect
the preview because this can also remove part of a garment. If the item blends
into the background, take another picture on a contrasting surface.

Detected multiple objects, very small regions and thin fragments produce an
explanation instead of a prediction. This is a limited preparation check, not a
reliable detector of every unsuitable photograph. Check the preview yourself.
The previous resize, brightness, contrast and inversion controls remain available
by turning off **Remove a plain background**.

Exact 28 × 28 grayscale benchmark inputs bypass all these operations. The saved
model, feature extractor, model checksum, train/validation/test split and official
scores are unchanged. The same preparation adapter is shared with the local
Phase 2 page, but no Phase 2 model is needed for these Phase 1 improvements.

## Command-line use

After the normal project setup:

```bash
bash scripts/run_python.sh scripts/predict_photo.py /path/to/photo.jpg --preview runs/photo-input.png
```

Optional arguments: `--model pixels_random_forest`, `--work-dir runs/phase1`,
`--crop 0.1 0.1 0.9 0.9`, `--rotation 90`, `--shadow-background`.
Crop coordinates are fractions: left, top, right, bottom. Rotation is anticlockwise.
For the previous resize-only mode, use `--keep-background` and optionally `--invert`.
The command uses the frozen record for each model and verifies its checksum.

## Checks on internet photographs

The following records use Phase 1's validation-selected pixel Random Forest.
The five original photographs were all misclassified with resize-only input.
The new preparation produced the following results. The sneaker crop selected
one visible shoe and rotated it to face left; it was not selected by model score.

| Picture | New preparation | Phase 1 result |
| --- | --- | --- |
| White T-shirt on white surface | Automatic and cropped | Requests a clearer picture; no prediction |
| Two sneakers | Automatic | Requests a crop; no prediction |
| Two sneakers | One shoe cropped and rotated | Sneaker, correct |
| Backpack | Automatic | Bag, correct |
| Black dress | Automatic | Dress, correct |
| Work boot | Automatic | Ankle boot, correct |
| Additional black T-shirt with wall shadow | Automatic | Bag, wrong |
| Additional white sneaker with brush | Automatic | Rejects a thin fragment; no prediction |
| Additional brown handbag | Automatic | Bag, correct |

These eight photographs were used during development. They are a small manual
check, not an independent accuracy benchmark. Refusing an input is not a correct
classification. The remaining black T-shirt error shows that background removal
does not make Fashion-MNIST models reliable on arbitrary photos. More varied
photo training and independent evaluation would be a separate study.

The local `runs/photo-preparation-fix/` folder retains source links, every result,
preparation previews and the code version checked. Earlier variants are retained
there too. These files are kept outside the official benchmark reports.

## Reference checked

The [reference photo code](https://github.com/krishnaverma09/Fashion_Classifier_AML/blob/main/src/predict.py)
uses grayscale conversion, explicit inversion, resizing and padding, and explicitly
states that background removal is not attempted. Copying that implementation
would retain the original whole-scene problem. Our new foreground preparation
uses the project's existing SciPy, scikit-image and scikit-learn dependencies.
No photo performance claim is taken from the reference.

## Further checks and process fixes

Transparent dark items now receive a contrasting background based only on visible
pixels. Small isolated items are enlarged to fit the input. A new picture resets
the previous crop, rotation, shadow settings and answer. The original benchmark
mode remains unchanged. See [the next photo check](PHOTO_TEST_ROUND2.md) for eight
additional photographs, remaining errors, and the checks on both Python versions.
