"""
Define auto augmentation operators
"""

from mindspore.dataset import vision
import mindspore.dataset.transforms as c_transforms

PARAMETER_MAX = 10


def float_parameter(level, maxval):
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    return int(level * maxval / PARAMETER_MAX)


def shear_x(level):
    transforms_list = []
    v = float_parameter(level, 0.3)

    transforms_list.append(vision.RandomAffine(degrees=0, shear=(-v, -v)))
    transforms_list.append(vision.RandomAffine(degrees=0, shear=(v, v)))
    return c_transforms.RandomChoice(transforms_list)


def shear_y(level):
    transforms_list = []
    v = float_parameter(level, 0.3)

    transforms_list.append(vision.RandomAffine(degrees=0, shear=(0, 0, -v, -v)))
    transforms_list.append(vision.RandomAffine(degrees=0, shear=(0, 0, v, v)))
    return c_transforms.RandomChoice(transforms_list)


def translate_x(level):
    transforms_list = []
    v = float_parameter(level, 150 / 331)

    transforms_list.append(vision.RandomAffine(degrees=0, translate=(-v, -v)))
    transforms_list.append(vision.RandomAffine(degrees=0, translate=(v, v)))
    return c_transforms.RandomChoice(transforms_list)


def translate_y(level):
    transforms_list = []
    v = float_parameter(level, 150 / 331)

    transforms_list.append(vision.RandomAffine(degrees=0, translate=(0, 0, -v, -v)))
    transforms_list.append(vision.RandomAffine(degrees=0, translate=(0, 0, v, v)))
    return c_transforms.RandomChoice(transforms_list)


def color_impl(level):
    v = float_parameter(level, 1.8) + 0.1
    return vision.RandomColor(degrees=(v, v))


def rotate_impl(level):
    transforms_list = []
    v = int_parameter(level, 30)

    transforms_list.append(vision.RandomRotation(degrees=(-v, -v)))
    transforms_list.append(vision.RandomRotation(degrees=(v, v)))
    return c_transforms.RandomChoice(transforms_list)


def solarize_impl(level):
    level = int_parameter(level, 256)
    v = 256 - level
    return vision.RandomSolarize(threshold=(0, v))


def posterize_impl(level):
    level = int_parameter(level, 4)
    v = 4 - level
    return vision.RandomPosterize(bits=(v, v))


def contrast_impl(level):
    v = float_parameter(level, 1.8) + 0.1
    return vision.RandomColorAdjust(contrast=(v, v))


def autocontrast_impl():
    return vision.AutoContrast()


def sharpness_impl(level):
    v = float_parameter(level, 1.8) + 0.1
    return vision.RandomSharpness(degrees=(v, v))


def brightness_impl(level):
    v = float_parameter(level, 1.8) + 0.1
    return vision.RandomColorAdjust(brightness=(v, v))


# define the Auto Augmentation policy
imagenet_policy = [
    [(posterize_impl(8), 0.4), (rotate_impl(9), 0.6)],
    [(solarize_impl(5), 0.6), (autocontrast_impl(), 0.6)],
    [(vision.Equalize(), 0.8), (vision.Equalize(), 0.6)],
    [(posterize_impl(7), 0.6), (posterize_impl(6), 0.6)],

    [(vision.Equalize(), 0.4), (solarize_impl(4), 0.2)],
    [(vision.Equalize(), 0.4), (rotate_impl(8), 0.8)],
    [(solarize_impl(3), 0.6), (vision.Equalize(), 0.6)],
    [(posterize_impl(5), 0.8), (vision.Equalize(), 1.0)],
    [(rotate_impl(3), 0.2), (solarize_impl(8), 0.6)],
    [(vision.Equalize(), 0.6), (posterize_impl(6), 0.4)],

    [(rotate_impl(8), 0.8), (color_impl(0), 0.4)],
    [(rotate_impl(9), 0.4), (vision.Equalize(), 0.6)],
    [(vision.Equalize(), 0.0), (vision.Equalize(), 0.8)],
    [(vision.Invert(), 0.6), (vision.Equalize(), 1.0)],
    [(color_impl(4), 0.6), (contrast_impl(8), 1.0)],

    [(rotate_impl(8), 0.8), (color_impl(2), 1.0)],
    [(color_impl(8), 0.8), (solarize_impl(7), 0.8)],
    [(sharpness_impl(7), 0.4), (vision.Invert(), 0.6)],
    [(shear_x(5), 0.6), (vision.Equalize(), 1.0)],
    [(color_impl(0), 0.4), (vision.Equalize(), 0.6)],

    [(vision.Equalize(), 0.4), (solarize_impl(4), 0.2)],
    [(solarize_impl(5), 0.6), (autocontrast_impl(), 0.6)],
    [(vision.Invert(), 0.6), (vision.Equalize(), 1.0)],
    [(color_impl(4), 0.6), (contrast_impl(8), 1.0)],
    [(vision.Equalize(), 0.8), (vision.Equalize(), 0.6)],
]
