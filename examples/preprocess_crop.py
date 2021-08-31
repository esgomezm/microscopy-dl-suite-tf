import random
import keras_preprocessing.image


def load_and_crop_img(path, grayscale=False, color_mode='rgb', target_size=None,
                      interpolation='nearest'):
    """Wraps keras_preprocessing.image.utils.loag_img() and adds cropping.
    Cropping method enumarated in interpolation

    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation and crop methods used to resample and crop the image
            if the target size is different from that of the loaded image.
            Methods are delimited by ":" where first part is interpolation and second is crop
            e.g. "lanczos:random".
            Supported interpolation methods are "nearest", "bilinear", "bicubic", "lanczos",
            "box", "hamming" By default, "nearest" is used.
            Supported crop methods are "none", "center", "random".

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    # Decode interpolation string. Allowed Crop methods: none, center, random
    interpolation, crop = interpolation.split(":") if ":" in interpolation else (interpolation, "none")

    if crop == "none":
        return keras_preprocessing.image.utils.load_img(path,
                                                        grayscale=grayscale,
                                                        color_mode=color_mode,
                                                        target_size=target_size,
                                                        interpolation=interpolation)

    # Load original size image using Keras
    img = keras_preprocessing.image.utils.load_img(path,
                                                   grayscale=grayscale,
                                                   color_mode=color_mode,
                                                   target_size=None,
                                                   interpolation=interpolation)

    # Crop fraction of total image
    crop_fraction = 0.875
    target_width = target_size[1]
    target_height = target_size[0]

    if target_size is not None:
        if img.size != (target_width, target_height):

            if crop not in ["center", "random"]:
                raise ValueError('Invalid crop method {} specified.', crop)

            if interpolation not in keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(interpolation,
                                            ", ".join(
                                                keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS.keys())))

            # resample = keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS[interpolation]

            # width, height = img.size

            # # Resize keeping aspect ratio
            # # result shold be no smaller than the targer size, include crop fraction overhead
            # target_size_before_crop = (target_width / crop_fraction, target_height / crop_fraction)
            # ratio = max(target_size_before_crop[0] / width, target_size_before_crop[1] / height)
            # target_size_before_crop_keep_ratio = int(width * ratio), int(height * ratio)
            # img = img.resize(target_size_before_crop_keep_ratio, resample=resample)

            width, height = img.size

            if crop == "center":
                left_corner = int(round(width / 2)) - int(round(target_width / 2))
                top_corner = int(round(height / 2)) - int(round(target_height / 2))
                return img.crop((left_corner, top_corner, left_corner + target_width, top_corner + target_height))
            elif crop == "random":
                
                left_shift = random.randint(0, int((width - target_width)))
                down_shift = random.randint(0, int((height - target_height)))
                return img.crop((left_shift, down_shift, target_width + left_shift, target_height + down_shift))

    return img


# Monkey patch
keras_preprocessing.image.iterator.load_img = load_and_crop_img