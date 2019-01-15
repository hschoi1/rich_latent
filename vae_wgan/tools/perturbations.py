# code from https://github.com/hendrycks/robustness/blob/master/ImageNet-P/create_p/make_tinyimagenet_p.py
import os
import numpy as np
import math
import numbers
import cv2
from PIL import Image as PILImage
from PIL import PILLOW_VERSION
import skimage.color as skcolor
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
from io import BytesIO
import ctypes
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb


def perturb(x):
    x_ = x * 255. # input is [0,1] so transform to [0,255]
    gray = (x.shape[-1] == 1)
    if gray:
        x_ = np.concatenate([x_, x_, x_], axis=3) #grey scale to rgb scale by setting same rgb values

    perturb_tuple = (gaussian_noise, shot_noise, motion_blur, zoom_blur, snow,
                    brightness, translate, rotate, tilt, scale, speckle_noise, gaussian_blur, spatter, shear)

    perturbed_list = []
    perturb_nums = [i for i in range(14)]
    for perturb_type in perturb_nums:
        print("perturbing type: ", perturb_type)
        x_perturbed = []
        for img in x_:
            img_perturbed = perturb_tuple[perturb_type](PILImage.fromarray(img.astype('uint8')))
            severities = []
            for severity in range(30):
                if gray:   # transform rgb to grey if input x is in grey scale
                    one_severity = cv2.cvtColor(np.array(img_perturbed[severity]).astype("uint8"), cv2.COLOR_RGB2GRAY).reshape(img.shape[0],img.shape[1],1)
                else:
                    one_severity = np.array(img_perturbed[severity]).astype("uint8")
                severities.append(one_severity)
            img_perturbed = np.array(severities)/255.    # (30, img_size, img_size, 1(3))
            x_perturbed.append(img_perturbed.astype(np.float32))

        plot_perturb_severity(gray, x[0], x_perturbed, perturb_tuple[perturb_type].__name__)  # plot images of all 30 severities per perturb. type
        x_perturbed = np.array(x_perturbed)  # (num_data, 30, img_size, img_size, 1(3))
        x_perturbed = np.transpose(x_perturbed, (1, 0, 2, 3, 4))  # (30, num_data, img_size, img_size, 1(3))
        x_perturbed = list(x_perturbed)  # 30 x (num_data, img_size, img_size, 1(3))
        perturbed_list += x_perturbed

    plot_perturb_types(gray, x[0], perturb_nums, perturbed_list, perturb_tuple)  # plot different types of perturb.
    return perturbed_list  # 14*30 x  (num_data, img_size, img_size, 1(3))


def plot_perturb_types(gray, original, perturb_nums, perturb_types, perturb_names):
    f, axes = plt.subplots(4, 4)  # show original and perturbed images
    if gray:
        axes[0, 0].imshow(original[:, :, 0])  # grey scale: (imgsize,imgsize,1)
    else:
        axes[0, 0].imshow(original)  # rgb scale: (imgsize,imgsize,3)
    axes[0, 0].set_title("original")
    perturb_types_ = np.array(perturb_types).reshape(len(perturb_nums), -1, 30, original.shape[0], original.shape[1], original.shape[2])
    for index, perturb_type in zip(perturb_nums, perturb_types_[:,:,0,:,:,:]): # iterate through perturbations of  severity 1
        if gray:
            axes[(index + 1) % 4, (index + 1) // 4].imshow(
                perturb_type[0][:, :, 0])  # grey scale: (imgsize,imgsize,1)
        else:
            axes[(index + 1) % 4, (index + 1) // 4].imshow(
                perturb_type[0][:, :, :])  # rgb scale: (imgsize,imgsize,3)
        axes[(index + 1) % 4, (index + 1) // 4].set_title(perturb_names[index].__name__)
    plt.tight_layout()
    plt.savefig("img_perturbations.png")
    plt.show()


def plot_perturb_severity(gray, original, perturb_imgs,  perturb_name):
    f, axes = plt.subplots(6, 6)  # show original and perturbed images
    if gray:
        axes[0, 0].imshow(original[:, :, 0])  # grey scale: (imgsize,imgsize,1)
    else:
        axes[0, 0].imshow(original)  # rgb scale: (imgsize,imgsize,3)
    axes[0, 0].set_title("original")

    for severity in range(30):
        if gray:
            axes[(severity + 1) % 6, (severity + 1) // 6].imshow(perturb_imgs[0][severity, :, :, 0])  # grey scale: (imgsize,imgsize,1)
        else:
            axes[(severity + 1) % 6, (severity + 1) // 6].imshow(perturb_imgs[0][severity, :, :])  # rgb scale: (imgsize,imgsize,3)
        axes[(severity + 1) % 6, (severity + 1) // 6].set_title(perturb_name + str(severity))
    plt.tight_layout()
    plt.savefig("perturbations/" + str(perturb_name) + "_severities.png")
    plt.show()



# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]



# /////////////// Gaussian Noise Code ///////////////
def gaussian_noise(img):
    img_shape = img.size[0]
    x = img
    return gaussian_noise1(np.array(x)/255.)

def gaussian_noise1(x):
    perturbed = []
    for i in range(1, 31):
        z = PILImage.fromarray(np.uint8(255 *np.clip(x + 0.025 *np.random.normal(size=x.shape), 0, 1)))
        perturbed.append(z)
    return perturbed

def gaussian_noise2(x):
    perturbed = []
    for i in range(1, 31):
        z = PILImage.fromarray(np.uint8(255 *np.clip(x + 0.05 * np.random.normal(size=x.shape), 0, 1)))
        perturbed.append(z)
    return perturbed

def gaussian_noise3(x):
    perturbed = []
    for i in range(1, 31):
        z = PILImage.fromarray(np.uint8(255 *np.clip(x + 0.075 * np.random.normal(size=x.shape), 0, 1)))
        perturbed.append(z)
    return perturbed

# /////////////// End Gaussian Noise Code ///////////////

# /////////////// Shot Noise Code ///////////////

def shot_noise(img):
    img_shape = img.size[0]
    x = img
    return shot_noise1(x)

def shot_noise1(x):
    perturbed = []
    for i in range(1, 31):
        z = np.array(x, copy=True) / 255.
        z = PILImage.fromarray(
            np.uint8(255 * np.clip(np.random.poisson(z * 500) / 500., 0, 1)))
        perturbed.append(z)
    return perturbed
def shot_noise2(x):
    perturbed = []
    for i in range(1, 31):
        z = np.array(x, copy=True) / 255.
        z = PILImage.fromarray(
            np.uint8(255 * np.clip(np.random.poisson(z * 250) / 250., 0, 1)))
        perturbed.append(z)
    return perturbed
def shot_noise3(x):
    perturbed = []
    for i in range(1, 31):
        z = np.array(x, copy=True) / 255.
        z = PILImage.fromarray(
            np.uint8(255 * np.clip(np.random.poisson(z * 125) / 125., 0, 1)))
        perturbed.append(z)
    return perturbed
# /////////////// End Shot Noise Code ///////////////

# /////////////// Motion Blur Code ///////////////

def motion_blur(orig_img):
    perturbed = []
    img_shape = orig_img.size[0]
    for i in range(0,30):
        z = orig_img
        output = BytesIO()
        z.save(output, format='PNG')
        z = MotionImage(blob=output.getvalue())

        z.motion_blur(radius=14, sigma=3, angle=(i-30)*5)

        z = cv2.imdecode(np.fromstring(z.make_blob(), np.uint8),
                         cv2.IMREAD_UNCHANGED)

        if z.shape != (img_shape, img_shape):
            z = np.clip(z[..., [2, 1, 0]], 0, 255)  # BGR to RGB
        else:  # grayscale to RGB
            z = np.clip(np.array([z, z, z]).transpose((1, 2, 0)), 0, 255)

        perturbed.append(resized_center_crop(PILImage.fromarray(z), img_shape))

    return perturbed
# /////////////// End Motion Blur Code ///////////////


# /////////////// Zoom Blur Code ///////////////
def zoom_blur(img):
    img_shape = img.size[0]
    perturbed = []
    z = img
    avg = np.array(z)/255.
    for i in range(1, 31):
        z = resized_center_crop(affine(img, angle=0, translate=(0, 0),
                                            scale=1+0.004*i, shear=0, resample=PILImage.BILINEAR), img_shape)
        avg += np.array(z)/255.
        perturbed.append(PILImage.fromarray(np.uint8(255*(avg / (i + 1)))))
    return perturbed
# /////////////// End Zoom Blur Code ///////////////


# /////////////// Snow Code ///////////////
def snow(img):
    img_shape = img.size[0]
    perturbed = []
    x = img
    x = np.array(x) / 255.

    snow_layer = np.random.normal(size=(img_shape, img_shape), loc=0.05, scale=0.28)

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], 2)
    snow_layer[snow_layer < 0.5] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    output = output.getvalue()

    for i in range(0, 30):
        moving_snow = MotionImage(blob=output)
        moving_snow.motion_blur(radius=10, sigma=2.5, angle=i*4-150)

        snow_layer = cv2.imdecode(np.fromstring(moving_snow.make_blob(), np.uint8),
                                  cv2.IMREAD_UNCHANGED) / 255.
        snow_layer = snow_layer[..., np.newaxis]

        z = 0.85 * x + (1 - 0.85) * np.maximum(
            x, cv2.cvtColor(np.float32(x), cv2.COLOR_RGB2GRAY).reshape(img_shape, img_shape, 1) * 1.5 + 0.5)

        z = np.uint8(np.clip(z + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255)

        perturbed.append(PILImage.fromarray(z))
    return perturbed

# /////////////// End Snow Code ///////////////

# /////////////// Brightness Code ///////////////

def brightness_helper(_x, c=0.):
    _x = np.array(_x, copy=True) / 255.
    _x = skcolor.rgb2hsv(_x)
    _x[:, :, 2] = np.clip(_x[:, :, 2] + c, 0, 1)
    _x = skcolor.hsv2rgb(_x)

    return np.uint8(_x * 255)


def brightness(img):
    img_shape = img.size[0]
    perturbed = []
    x = img

    for i in range(0, 30):
        z = PILImage.fromarray(brightness_helper(x, c=(i - 15) * 2 / 100.))
        perturbed.append(z)
    return perturbed
# /////////////// End Brightness Code ///////////////


# /////////////// Translate Code ///////////////
def translate(img):
    img_shape = img.size[0]
    perturbed = []
    for i in range(0,30):
        z = resized_center_crop(affine(img, angle=0, translate=(i-15, 0), scale=1, shear=0), img_shape)
        perturbed.append(z)
    return perturbed
# /////////////// End Translate Code ///////////////


# /////////////// Rotate Code ///////////////
def rotate(img):
    img_shape = img.size[0]
    perturbed = []
    for i in range(0, 30):
        z =resized_center_crop(affine(img, angle=i-15, translate=(0, 0),
                                            scale=1., shear=0, resample=PILImage.BILINEAR), img_shape)
        perturbed.append(z)
    return perturbed
# /////////////// End Rotate Code ///////////////

# /////////////// Tilt Code ///////////////
def tilt(img):
    img_shape = img.size[0]
    perturbed = []
    x = np.array(img)
    h, w = x.shape[0:2]

    for i in range(0, 30):
        phi, theta = np.deg2rad(i-15), np.deg2rad(i-15)

        f = np.sqrt(w ** 2 + h ** 2)

        P1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1], [0, 0, 1]])

        RX = np.array([[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0],
                       [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]])

        RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0], [0, 1, 0, 0],
                       [np.sin(phi), 0, np.cos(phi), 0], [0, 0, 0, 1]])

        T = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                      [0, 0, 1, f], [0, 0, 0, 1]])

        P2 = np.array([[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]])

        mat = P2 @ T @ RX @ RY @ P1

        z = resized_center_crop(PILImage.fromarray(cv2.warpPerspective(x, mat, (w, h))), img_shape)

        perturbed.append(z)
    return perturbed


# /////////////// End Tilt Code ///////////////

# /////////////// Scale Code ///////////////
def scale(img):
    img_shape = img.size[0]
    perturbed = []
    for i in range(0, 30):
        z = resized_center_crop(affine(img, angle=0, translate=(0, 0),
                                            scale=(i * 2.5 + 40) / 100., shear=0, resample=PILImage.BILINEAR), img_shape)
        perturbed.append(z)
    return perturbed
# /////////////// End Scale Code ///////////////

#/////////////// Validation Data ///////////////

# /////////////// Speckle Noise Code ///////////////
def speckle_noise(img):
    img_shape = img.size[0]
    x = img
    x = np.array(x) / 255.
    return speckle_noise1(x)

def speckle_noise1(x):
    perturbed = []
    for i in range(1, 31):
        z = PILImage.fromarray(
            np.uint8(255 * np.clip(x + x * np.random.normal(size=x.shape, scale=0.05), 0, 1)))
        perturbed.append(z)
    return perturbed

def speckle_noise2(x):
    perturbed = []
    for i in range(1, 31):
        z = PILImage.fromarray(
            np.uint8(255 * np.clip(x + x * np.random.normal(size=x.shape, scale=0.1), 0, 1)))
        perturbed.append(z)
    return perturbed

def speckle_noise3(x):
    perturbed = []
    for i in range(1, 31):
        z = PILImage.fromarray(
            np.uint8(255 * np.clip(x + x * np.random.normal(size=x.shape, scale=0.15), 0, 1)))
        perturbed.append(z)
    return perturbed
# /////////////// End Speckle Noise Code ///////////////

# /////////////// Gaussian Blur Code ///////////////
def gaussian_blur(img):
    img_shape = img.size[0]
    x = img
    perturbed = []
    for i in range(0, 30):
        perturbed.append(PILImage.fromarray(
            np.uint8(255*gaussian(np.array(x, copy=True)/255.,
                                  sigma=0.35 + 0.014*i, multichannel=True, truncate=7.0))))
    return perturbed

# /////////////// End Gaussian Blur Code ///////////////

# /////////////// Spatter Code ///////////////
def spatter(img):
    img_shape = img.size[0]
    perturbed = []
    x = img
    x = cv2.cvtColor(np.array(x, dtype=np.float32) / 255., cv2.COLOR_BGR2BGRA)

    liquid_layer = np.random.normal(size=x.shape[:2], loc=0.6, scale=0.25)
    liquid_layer = gaussian(liquid_layer, sigma=1.75)
    liquid_layer[liquid_layer < 0.7] = 0

    for i in range(0, 30):

        liquid_layer_i = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer_i, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer_i * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= 0.6

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)

        z = np.uint8(cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255)

        liquid_layer = np.apply_along_axis(lambda mat:
                                           np.convolve(mat, np.array([0.2, 0.8]), mode='same'),
                                           axis=0, arr=liquid_layer)

        perturbed.append(PILImage.fromarray(z))
    return perturbed
# /////////////// End Spatter Code ///////////////

# /////////////// Shear Code ///////////////
def shear(img):
    img_shape = img.size[0]
    perturbed = []
    for i in range(0, 30):
        z = resized_center_crop(affine(img, angle=0, translate=(0, 0),
                                             scale=1., shear=i-15, resample=PILImage.BILINEAR), img_shape)
        perturbed.append(z)
    return perturbed
# /////////////// End Shear Code ///////////////



# from torch source code:  https://pytorch.org/docs/master/_modules/torchvision/transforms/functional.html

try:
    import accimage
except ImportError:
    accimage = None

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (PILImage.Image, accimage.Image))
    else:
        return isinstance(img, PILImage.Image)

def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    angle = math.radians(angle)
    shear = math.radians(shear)
    scale = 1.0 / scale

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
    matrix = [
        math.cos(angle + shear), math.sin(angle + shear), 0,
        -math.sin(angle), math.cos(angle), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]
    return matrix


def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
    """Apply affine transformation on the image keeping image center invariant

    Args:
        img (PIL Image): PIL Image to be rotated.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] == '5' else {}
    return img.transform(output_size, PILImage.AFFINE, matrix, resample, **kwargs)

def resize(img, size, interpolation=PILImage.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def center_crop(img, output_size):
    w, h = img.size
    # Case of single value provided
    if isinstance(output_size, numbers.Number):
        # Float case: constraint for fraction must be from 0 to 1.0
        if isinstance(output_size, float):
            if not 0.0 < output_size <= 1.0:
                raise ValueError("Invalid float output size. Range is (0.0, 1.0]")
            output_size = (output_size, output_size)
            th, tw = int(h * output_size[0]), int(w * output_size[1])
        elif isinstance(output_size, int):
            output_size = (output_size, output_size)
            th, tw = output_size
    # Case of tuple of values provided
    else:
        if isinstance(output_size[0], float):
            th, tw = int(h * output_size[0]), int(w * output_size[1])
        elif isinstance(output_size[0], int):
            th, tw = output_size

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img.crop((j, i, j + tw, i + th))


def resized_center_crop(img, size, scale=1.0, interpolation=PILImage.BILINEAR):
    img = center_crop(img, scale)
    img = resize(img, size, interpolation)
    return img
