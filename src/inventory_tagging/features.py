"""Fixed pixel and HOG representations. Neither learns from other images."""
import numpy as np
from skimage.feature import hog

HOG = dict(orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2),
           block_norm='L2-Hys', transform_sqrt=False, feature_vector=True,
           channel_axis=None)


def normalized(images):
    images = np.asarray(images)
    if images.dtype != np.uint8:
        raise TypeError('Expected original uint8 images; do not normalize twice')
    return images.astype(np.float32) / 255.0


def extract(images, representation):
    values = normalized(images)
    if representation == 'pixels':
        return values.reshape(len(images), 784)
    if representation not in ('hog', 'hog_fine'):
        raise ValueError(f'Unknown representation: {representation}')
    settings = dict(HOG)
    if representation == 'hog_fine':
        settings['pixels_per_cell'] = (4, 4)
    return np.asarray([hog(image, **settings) for image in values], dtype=np.float32)
