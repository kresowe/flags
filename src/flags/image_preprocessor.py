import numpy as np
from numpy.typing import NDArray
import cairosvg
from PIL import Image, ImageEnhance
from matplotlib.image import imread
import math
import os
from typing import Union


def preprocess_image(file_path_in: Union[str, os.PathLike], file_path_out: Union[str, os.PathLike],
                     width: int, height: int) -> None:
    """Preprocess image file_path_in provided by user and saves it to file_path_out.
    Namely, it converts it to PNG file, then resizes to (width, height) dimensions (in pixels) and converts to RGB."""
    convert_to_png(file_path_in, file_path_out)
    resize(file_path_out, file_path_out, width, height)
    convert_to_rgb(file_path_out, file_path_out)


def convert_to_png(img_path_in: Union[str, os.PathLike], img_path_out: Union[str, os.PathLike]) -> None:
    """Converts image at img_path_in from SVG to PNG and saves it at img_path_out if it is SVG.
     If it is already converted to PNG it just saves it as img_path_out."""
    try:
        if is_svg_file(img_path_in):
            cairosvg.svg2png(url=img_path_in, write_to=img_path_out)
        else:
            with Image.open(img_path_in) as img:
                img.save(img_path_out)
    except FileNotFoundError:
        print(f'Missing file {img_path_in}')
    except Exception as e:
        print(f'Unexpected error: {e}')


def resize(img_path_in: Union[str, os.PathLike], img_path_out: Union[str, os.PathLike], width: int, height: int) -> None:
    """Resizes image img_path_in to (width, height) dimensions (numbers of pixels) and saves it as img_path_out."""
    try:
        with Image.open(img_path_in) as img:
            resized_img = img.resize((width, height))
            resized_img.save(img_path_out)
    except FileNotFoundError:
        print(f'Missing file {img_path_in}')
    except Exception as e:
        print(f'Unexpected error: {e}')


def convert_to_rgb(img_path_in: Union[str, os.PathLike], img_path_out: Union[str, os.PathLike]) -> None:
    """Converts image img_path_in to RGB if it is RGBA and saves it as img_path_out.
    If image already is RGB, it does nothing."""
    try:
        if not is_rgba(img_path_in):
            return
        with Image.open(img_path_in) as img_pil:
            img_out = img_pil.convert('RGB')
            img_out.save(img_path_out)
    except FileNotFoundError:
        print(f'Missing file {img_path_in}')
    except Exception as e:
        print(f'Unexpected error: {e}')


def change_brightness(img_path_in: Union[str, os.PathLike], img_path_out: Union[str, os.PathLike], factor: float) -> None:
    """Changes brightness of image img_path_in by factor and saves it as img_path_out."""
    try:
        with Image.open(img_path_in) as img:
            enhancer = ImageEnhance.Brightness(img)
            enhanced_img = enhancer.enhance(factor)
            enhanced_img.save(img_path_out)
    except FileNotFoundError:
        print(f'Missing file {img_path_in}')
    except Exception as e:
        print(f'Unexpected error: {e}')


def change_contrast(img_path_in: Union[str, os.PathLike], img_path_out: Union[str, os.PathLike], factor: float) -> None:
    """Changes contrast of image img_path_in by factor and saves it as img_path_out."""
    try:
        with Image.open(img_path_in) as img:
            enhancer = ImageEnhance.Contrast(img)
            enhanced_img = enhancer.enhance(factor)
            enhanced_img.save(img_path_out)
    except FileNotFoundError:
        print(f'Missing file {img_path_in}')
    except Exception as e:
        print(f'Unexpected error: {e}')


def create_data_sample_from_image(img_path_in: Union[str, os.PathLike], pixels_positions: tuple) -> NDArray:
    """Creates data sample in the form of np.array by selecting (R,G,B) from pixels determined by pixels_positions
    from image img_path_in."""
    img = imread(img_path_in)
    img_sample = img[pixels_positions[0][0], pixels_positions[0][1]]
    for i in range(1, len(pixels_positions)):
        img_sample = np.concatenate((img_sample, img[pixels_positions[i][0], pixels_positions[i][1]]))
    return img_sample


def create_data_sample_as_single(img_path_in: Union[str, os.PathLike], pixels_positions: tuple) -> NDArray:
    """Creates data sample in the form of np.array by selecting (R,G,B) from pixels determined by pixels_positions
    from image img_path_in.
    Then it reshapes it so that it can be input to predict() of machine learning model."""
    img_sample = create_data_sample_from_image(img_path_in, pixels_positions)
    return img_sample.reshape(1, -1)


def get_pixels_positions(width: int, height: int) -> tuple:
    """Returns positions of pixels that are selected to be used for creating a dataset.
    It also returns width_limits and height_limits that are typical lines where rectangles on flags have their borders."""
    width_limits = width * np.array([1 / 3, 1 / 2, 2 / 3])
    height_limits = height * np.array([1 / 3, 1 / 2, 2 / 3])
    widths = [math.floor((0 + width_limits[0]) / 2),
              math.floor((width_limits[0] + width_limits[1]) / 2),
              math.floor((width_limits[1] + width_limits[2]) / 2),
              math.floor((width_limits[2] + width) / 2)
              ]
    heights = [math.floor((0 + height_limits[0]) / 2),
               math.floor((height_limits[0] + height_limits[1]) / 2),
               math.floor((height_limits[1] + height_limits[2]) / 2),
               math.floor((height_limits[2] + height) / 2)
               ]

    pixels_positions = ([heights[0], widths[0]],
                        [heights[0], widths[3]],
                        [heights[3], widths[0]],
                        [heights[3], widths[3]],
                        [heights[1], widths[1]],
                        [heights[1], widths[2]],
                        [heights[2], widths[1]],
                        [heights[2], widths[2]],
                        )
    return pixels_positions, width_limits, height_limits


def is_svg_file(file_path: Union[str, os.PathLike]) -> bool:
    """Checks if a file is SVG."""
    return file_path.lower().endswith('.svg')


def is_rgba(file_path: Union[str, os.PathLike]) -> bool:
    """Checks if a file is RGBA image."""
    try:
        img = imread(file_path)
        return img.shape[2] > 3
    except FileNotFoundError:
        print(f'Missing file {file_path}')
    except Exception as e:
        print(f'Unexpected error: {e}')



