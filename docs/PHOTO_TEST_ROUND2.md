# Further Phase 1 photo checks

The project instructions specify machine learning for Phase 1. Project 11 asks
for normalized 28 × 28 inputs, pixels and HOG with Logistic Regression, k-NN,
Random Forest and SVM, using the official split and validation drawn from training.
Those eight experiments and their saved scores have not changed. No new classifier,
photo training dataset, ensemble, or CNN was added in this round.

## Process fixes

- **Transparent images:** background removal now chooses a contrasting fill from
  visible pixels. A black item on transparency previously disappeared against a
  black fill. Invisible RGB values no longer influence that choice. Fully
  transparent pictures receive a clear explanation. Existing resize-only mode
  retains its explicit dark/light background option.
- **Small items:** the isolated item is now enlarged as well as reduced to fit the
  input. Previously a small crop could occupy only a few of the 28 × 28 pixels.
  Its proportions and black padding are preserved.
- **New uploads:** crop, rotation and shadow settings reset when a different
  picture is selected, and the previous prediction is cleared. A crop for the
  previous picture can no longer silently cut the new item.
- **Expectations:** the photo interface explains that a classifier always chooses
  one of the ten training categories. It does not have a learned unknown category.

Camera orientation, partially transparent items, palette transparency, unchanged
benchmark pixels, deterministic preparation and navigation were also checked.
The eight original model checksums and the 109 protected evidence files passed.

## Eight additional internet photographs

All eight Phase 1 models were exercised on each input that passed preparation.
The table shows the existing validation-selected pixel Random Forest. The manual
crop was recorded from the visible image before predictions. Settings were not
searched for a correct label. All outcomes are retained, including disagreements.

| Photo | Selected model / preparation outcome | Interpretation |
| --- | --- | --- |
| White T-shirt on hanger | Shirt; cropped version also Shirt | Wrong category. Two HOG models call the automatic input T-shirt/top; this does not justify changing the selected model. |
| T-shirt draped over hanger | Requests a clearer background | No prediction. Clothing shape is partly hidden. |
| Two formal shoes | Requests one item | No prediction. Formal shoes also have no exact named class in the ten categories. |
| Flat lay of several items | Requests one item | No single correct category for the whole scene. |
| Folded sweatpants | T-shirt/top | Wrong item type; the full trouser shape is not visible. |
| Two overlapping shirts | Sandal | Wrong. The foreground check does not catch every unsuitable scene. |
| White fur handbag | Requests a clearer picture or crop | No prediction; poor contrast and background details. |
| Sunlit T-shirt close-up | Requests space around the item | No prediction; the item is cut off at the picture edges. |

The last two photos were checked after the preparation changes were fixed. Two
photos do not establish independent accuracy. The first six are development checks.
These examples deliberately include difficult inputs; none of the refusals is
counted as a correct classification. Original backpack, dress, boot, cropped
sneaker and brown-handbag successes still reproduce, and earlier failures remain.

## Reproduction and evidence

The local `runs/photo-round2/` folder contains source URLs and download hashes,
`before.json`, `final_protocol.json`, `after.json` (including each model's actual
answer and inference time), previews and the repeatable `check.py` script.
The images are local test material, outside the official evaluation reports.
The six development-photo outcomes were unchanged by these process fixes; the
fixes address transparency, geometry and state, not the classifier's learned
category boundaries.

The full suite passed **85 tests on Python 3.12 and Python 3.11**. A separate source
copy without project model/data caches passed **30 photo/dashboard tests** using
the independent Python 3.11 environment. The saved results, artifact integrity and
Phase 2 regression checks passed. No dependency was added. These checks were completed before merging the work into main.

## Remaining limit

The application is more reliable at handling an input, but the existing
Fashion-MNIST classifiers are not reliable general clothing-photo classifiers.
Removing all real-photo category errors would require representative photo data,
a new training/validation study and independent evaluation. Relabelling errors or
choosing a model from these photo results would not be a valid fix to this study.

## Sources

- [white tee](https://www.pexels.com/photo/white-shirt-on-a-black-hanger-8148576/)
- [hanging shirt](https://www.pexels.com/photo/grayscale-photo-of-a-shirt-on-a-hanger-14459666/)
- [dress shoes](https://www.pexels.com/photo/brown-leather-dress-shoes-on-white-back-8670488/)
- [mixed outfit](https://www.pexels.com/photo/flat-lay-of-clothing-and-accessories-3927390/)
- [folded sweatpants](https://www.pexels.com/photo/stylish-assorted-sweatpants-flat-lay-31961165/)
- [multiple shirts](https://www.pexels.com/photo/shirts-on-hanger-10131161/)
- [white fur bag](https://www.pexels.com/photo/a-white-small-bag-with-fur-on-a-white-surface-11114126/)
- [sunlit tee](https://www.pexels.com/photo/minimal-white-t-shirt-in-sunlit-flat-lay-37704844/)
