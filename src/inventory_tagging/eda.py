"""Training-only data audit and figures for the notebook and report."""
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.exposure import rescale_intensity
from .data import ROOT, CLASSES, CHECKSUMS, load_images, split_indices, write_json, sha256
from .features import HOG, normalized


def save(fig, name):
    folder = ROOT / 'reports/figures'
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run():
    images, labels = load_images('train')
    train, val = split_indices(labels)
    # All content-driven exploration uses training IDs. Integrity counts also cover validation.
    counts = pd.DataFrame({'label': range(10), 'category': CLASSES,
                           'official_train': np.bincount(labels),
                           'training': np.bincount(labels[train]),
                           'validation': np.bincount(labels[val])})
    counts.to_csv(ROOT / 'reports/class_counts.csv', index=False)
    hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    audit = {'official_training_images': len(images), 'training_images': len(train),
             'validation_images': len(val), 'image_shape': [28, 28], 'dtype': str(images.dtype),
             'pixel_range': [int(images.min()), int(images.max())],
             'blank_images': int(np.sum(images.max(axis=(1, 2)) == images.min(axis=(1, 2)))),
             'exact_duplicate_images': len(images) - len(set(hashes)),
             'split_overlap': int(len(np.intersect1d(train, val))),
             'train_validation_image_overlap': len({hashes[i] for i in train} & {hashes[i] for i in val}),
             'finite_pixels': bool(np.isfinite(images).all()),
             'near_duplicate_check': 'Not established by exact hashing; product IDs are unavailable.',
             'archives': {name: {'md5': expected, 'sha256': sha256(ROOT / 'data/raw' / name)}
                          for name, expected in CHECKSUMS.items() if name.startswith('train')},
             'official_test_loaded': False}
    write_json(ROOT / 'reports/data_audit.json', audit)
    ax = counts.set_index('category')[['training', 'validation']].plot.bar(figsize=(10, 4), rot=30)
    ax.set_ylabel('Images'); ax.set_title('Shared split: balanced categories')
    save(ax.figure, 'class_counts.png')
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(3, 10, figsize=(14, 5))
    for c, name in enumerate(CLASSES):
        for row, index in enumerate(rng.choice(train[labels[train] == c], 3, replace=False)):
            axes[row, c].imshow(images[index], cmap='gray', vmin=0, vmax=255)
            axes[row, c].axis('off')
            if row == 0:
                axes[row, c].set_title(name, fontsize=9)
    save(fig, 'training_samples.png')
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for c, ax in enumerate(axes.flat):
        ax.imshow(images[train[labels[train] == c]].mean(axis=0), cmap='gray', vmin=0, vmax=255)
        ax.set_title(CLASSES[c]); ax.axis('off')
    save(fig, 'class_means.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in (0, 2, 4, 6):
        ax.hist(images[train[labels[train] == c]].mean(axis=(1, 2)) / 255,
                bins=30, histtype='step', label=CLASSES[c], density=True)
    ax.set(xlabel='Mean image intensity (0 to 1)', ylabel='Density', title='Upper-body categories: brightness overlaps')
    ax.legend(); save(fig, 'intensity_distribution.png')
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for c in range(5):
        index = train[labels[train] == (0, 2, 4, 6, 8)[c]][0]
        _, drawing = hog(normalized(images[index]), visualize=True, **HOG)
        axes[0, c].imshow(images[index], cmap='gray', vmin=0, vmax=255)
        axes[0, c].set_title(CLASSES[labels[index]])
        axes[1, c].imshow(rescale_intensity(drawing, in_range=(0, .1)), cmap='gray')
        axes[0, c].axis('off'); axes[1, c].axis('off')
    fig.suptitle('Original images (top) and HOG (bottom)')
    save(fig, 'hog_visualization.png')
    print('Training audit and five EDA figures saved. No test data loaded.')
