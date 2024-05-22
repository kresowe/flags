import numpy as np
import cairosvg
from PIL import Image, ImageEnhance
from matplotlib.image import imread
import math


def convert_to_png(img_path_in, img_path_out):
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


def resize(img_path_in, img_path_out, width, height):
    try:
        with Image.open(img_path_in) as img:
            resized_img = img.resize((width, height))
            resized_img.save(img_path_out)
    except FileNotFoundError:
        print(f'Missing file {img_path_in}')
    except Exception as e:
        print(f'Unexpected error: {e}')


def convert_to_rgb(img_path_in, img_path_out):
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


def change_brightness(img_path_in, img_path_out, factor):
    try:
        with Image.open(img_path_in) as img:
            enhancer = ImageEnhance.Brightness(img)
            enhanced_img = enhancer.enhance(factor)
            enhanced_img.save(img_path_out)
    except FileNotFoundError:
        print(f'Missing file {img_path_in}')
    except Exception as e:
        print(f'Unexpected error: {e}')


def change_contrast(img_path_in, img_path_out, factor):
    try:
        with Image.open(img_path_in) as img:
            enhancer = ImageEnhance.Contrast(img)
            enhanced_img = enhancer.enhance(factor)
            enhanced_img.save(img_path_out)
    except FileNotFoundError:
        print(f'Missing file {img_path_in}')
    except Exception as e:
        print(f'Unexpected error: {e}')


def create_data_sample_from_image(img_path_in, pixels_positions):
    img = imread(img_path_in)
    img_sample = img[pixels_positions[0][0], pixels_positions[0][1]]
    for i in range(1, len(pixels_positions)):
        img_sample = np.concatenate((img_sample, img[pixels_positions[i][0], pixels_positions[i][1]]))
    return img_sample


def create_data_sample_as_single(img_path_in, pixels_positions):
    img_sample = create_data_sample_from_image(img_path_in, pixels_positions)
    return img_sample.reshape(1, -1)


def get_pixels_positions(width, height) -> tuple[tuple, list, list]:
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


def is_svg_file(file_path):
    return file_path.lower().endswith('.svg')


def is_rgba(file_path):
    try:
        img = imread(file_path)
        return img.shape[2] > 3
    except FileNotFoundError:
        print(f'Missing file {file_path}')
    except Exception as e:
        print(f'Unexpected error: {e}')