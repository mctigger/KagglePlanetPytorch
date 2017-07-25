import numpy as np
from skimage.transform import AffineTransform, SimilarityTransform, PolynomialTransform, warp
from skimage.filters import gaussian
from skimage import exposure, img_as_float
from scipy.misc import imresize
import Augmentor

center_shift = 256 / 2
tf_center = SimilarityTransform(translation=-center_shift)
tf_uncenter = SimilarityTransform(translation=center_shift)


def apply_chain(chain):
    def call(x):

        for fn in chain:
            x = fn(x)

        return x

    return call


def resize(output_size):
    def call(x):
        x = imresize(x, output_size) # x[::256//output_size, ::256//output_size, :]
        return x

    return call


def random_crop(size):
    def call(x):
        height, width, d = x.shape

        r1 = np.random.random()
        r2 = np.random.random()

        left = round(r1 * (width - size))
        right = round((1 - r1) * (width - size))

        top = round(r2 * (height - size))
        bottom = round((1 - r2) * (height - size))

        crop = x[top:height-bottom, left:width-right, :]

        return crop

    return call


def center_crop(size):
    def call(x):
        height, width, d = x.shape
        a = int(0.5 * (height - size))
        b = int(0.5 * (width - size))

        top, bottom = a, (height - size) - a
        left, right = b, (height - size) - b

        crop = x[top:height-bottom, left:width-right, :]

        return crop

    return call


def crop_top_left(size):
    def call(x):
        crop = x[:size, :size, :]
        return crop

    return call


def crop_top_right(size):
    def call(x):
        crop = x[:size, x.shape[1] - size:, :]
        return crop

    return call


def crop_bottom_left(size):
    def call(x):
        crop = x[x.shape[0] - size:, :size, :]
        return crop

    return call


def crop_bottom_right(size):
    def call(x):
        crop = x[x.shape[0] - size:, x.shape[1] - size:, :]
        return crop

    return call


def image_to_array(x):
    return np.array(x)


def to_float(x):
    return img_as_float(x)


def blur(sigma=0.1):
    def call(x):
        x = gaussian(x, sigma=sigma, preserve_range=True, multichannel=True)
        return x
    return call


def random_blur(sigma=lambda: np.random.random_sample()*1):
    def call(x):
        x = gaussian(x, sigma=sigma(), preserve_range=True, multichannel=True)
        return x
    return call


def random_gamma(gamma=lambda: np.random.rand() * 0.4 + 0.8):
    def call(x):
        return exposure.adjust_gamma(x, gamma())

    return call


def random_contrast(weight=lambda: np.random.rand() * 0.3 + 0.7):
    def call(x):
        w = weight()
        return x * w + (1 - w) * exposure.rescale_intensity(x)

    return call


def augment_color(weight=0.1):
    def call(x):
        height, width, channels = x.shape

        img_rgb_col = x.reshape(height*width, channels)
        cov = np.cov(img_rgb_col.T)
        eigvals, eigvects = np.linalg.eigh(cov)
        random_eigvals = np.sqrt(eigvals) * np.random.randn(channels) * weight
        scaled_eigvects = np.dot(eigvects, random_eigvals)
        x = np.clip(x + scaled_eigvects, 0, 1)

        return x

    return call


def augment_color_deterministic(weight=0.1):
    def call(x):
        height, width, channels = x.shape

        img_rgb_col = x.reshape(height*width, channels)
        cov = np.cov(img_rgb_col.T)
        eigvals, eigvects = np.linalg.eigh(cov)
        random_eigvals = np.sqrt(eigvals) * np.array([1, 1, 1]) * weight
        scaled_eigvects = np.dot(eigvects, random_eigvals)
        x = np.clip(x + scaled_eigvects, 0, 1)

        return x

    return call


def distort():
    p = Augmentor.Pipeline()
    p.random_distortion(probability=1, grid_width=5, grid_height=5, magnitude=8)

    def call(x):
        x = p.sample_with_array(x.astype('uint8'), False)
        return x

    return call


def random_zoom_range(zoom_range=[1/1.2, 1.2]):
    def call():
        #https://github.com/benanne/kaggle-galaxies/blob/master/realtime_augmentation.py#L147
        #log_zoom_range = [np.log(z) for z in zoom_range]
        #zoom = np.exp(np.random.uniform(*log_zoom_range))

        zoom = np.random.uniform(zoom_range[0], zoom_range[1])

        return zoom, zoom

    return call


def augment(
        rotation_fn=lambda: np.random.random_integers(0, 360),
        translation_fn=lambda: (np.random.random_integers(-20, 20), np.random.random_integers(-20, 20)),
        scale_factor_fn=random_zoom_range(),
        shear_fn=lambda: np.random.random_integers(-10, 10)
):
    def call(x):
        rotation = rotation_fn()
        translation = translation_fn()
        scale = scale_factor_fn()
        shear = shear_fn()

        tf_augment = AffineTransform(scale=scale, rotation=np.deg2rad(rotation), translation=translation, shear=np.deg2rad(shear))
        tf = tf_center + tf_augment + tf_uncenter

        x = warp(x, tf, order=1, preserve_range=True, mode='symmetric')

        return x

    return call


def rotate_90(k=1):
    def call(x):
        x = np.rot90(x, k).copy()
        return x

    return call


def fliplr():
    def call(x):
        x = np.fliplr(x).copy()
        return x

    return call


def flipud():
    def call(x):
        x = np.flipud(x).copy()
        return x

    return call


def random_fliplr():
    def call(x):
        if np.random.randint(2) > 0:
            x = np.fliplr(x).copy()
        return x

    return call


def random_flipud():
    def call(x):
        if np.random.randint(2) > 0:
            x = np.flipud(x).copy()
        return x

    return call


def augment_deterministic(
        rotation=0,
        translation=0,
        scale_factor=1,
        shear=0
):
    def call(x):
        scale = scale_factor, scale_factor
        rotation_tmp = rotation

        tf_augment = AffineTransform(
            scale=scale,
            rotation=np.deg2rad(rotation_tmp),
            translation=translation,
            shear=np.deg2rad(shear)
        )
        tf = tf_center + tf_augment + tf_uncenter

        x = warp(x, tf, order=1, preserve_range=True, mode='symmetric')

        return x

    return call