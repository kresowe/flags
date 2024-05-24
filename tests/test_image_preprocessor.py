from matplotlib.image import imread
import pytest
import os
from src.flags.image_preprocessor import (resize, convert_to_rgb, create_data_sample_from_image,
                                          create_data_sample_as_single, get_pixels_positions, is_svg_file, is_rgba)
from src.flags.utils import remove_file_if_exists
from tests.test_final_recognize_flag import pixels


def test_resize_resizes_correctly():
    width, height = 32, 20
    file_path_in = os.path.join('tests', 'data_for_tests', '116.png')
    file_path_out = os.path.join('tests', 'data_for_tests', 'new.png')
    remove_file_if_exists(file_path_out)
    img_in = imread(file_path_in)
    assert img_in.shape[1] != width and img_in.shape[0] != height
    resize(file_path_in, file_path_out, width, height)
    img_out = imread(file_path_out)
    assert img_out.shape[1] == width and img_out.shape[0] == height


def test_rgba_is_converted_to_rgb():
    file_path_in = os.path.join('tests', 'data_for_tests', '31.png')
    file_path_out = os.path.join('tests', 'data_for_tests', 'new.png')
    remove_file_if_exists(file_path_out)
    assert is_rgba(file_path_in)
    convert_to_rgb(file_path_in, file_path_out)
    assert not is_rgba(file_path_out)


def test_created_data_sample_has_correct_shape(pixels):
    img_sample = create_data_sample_from_image(os.path.join('tests', 'data_for_tests', '140_3.png'), pixels)
    assert img_sample.shape == (24, )


def test_created_data_sample_as_single_has_correct_shape(pixels):
    img_sample = create_data_sample_as_single(os.path.join('tests', 'data_for_tests', '140_3.png'), pixels)
    assert img_sample.shape == (1, 24)


def test_get_pixels_positions_gives_correct_limits():
    pixels, width_limits, height_limits = get_pixels_positions(32, 20)
    assert width_limits[0] == pytest.approx(32/3)
    assert width_limits[1] == pytest.approx(16)
    assert width_limits[2] == pytest.approx(32 * 2 / 3)

    assert height_limits[0] == pytest.approx(20 / 3)
    assert height_limits[1] == pytest.approx(10)
    assert height_limits[2] == pytest.approx(20 * 2 / 3)


def test_get_pixels_positions_gives_correct_pixels_positions(pixels):
    assert len(pixels) == 8
    assert [3, 5] in pixels
    assert [3, 26] in pixels
    assert [16, 5] in pixels
    assert [16, 26] in pixels
    assert [8, 13] in pixels
    assert [8, 18] in pixels
    assert [11, 13] in pixels
    assert [11, 18] in pixels


def test_svg_file_is_svg():
    assert is_svg_file(os.path.join('tests/data_for_tests', '140.svg'))


def test_png_file_is_not_svg():
    assert not is_svg_file(os.path.join('tests/data_for_tests', '140_3.png'))


def test_rgba_file_is_rgba():
    assert is_rgba(os.path.join('tests/data_for_tests', '31.png'))


def test_rgb_file_is_not_rgba():
    assert not is_rgba(os.path.join('tests/data_for_tests', '140_3.png'))
