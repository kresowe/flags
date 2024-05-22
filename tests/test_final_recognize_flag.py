import pytest
import pickle
import os
from image_preprocessor import get_pixels_positions, create_data_sample_as_single
from recognize_flag import preprocess_image, remove_file_if_exists, load_countries_names


@pytest.fixture()
def clf():
    return pickle.load(open(os.path.join('tests/data_for_tests', 'clf.pkl'), 'rb'))


@pytest.fixture()
def pixels():
    pixels_positions, _, _ = get_pixels_positions(width=32, height=20)
    return pixels_positions


def pipeline(clf, pixels, file_path_in, file_path_out):
    try:
        remove_file_if_exists(file_path_out)
        preprocess_image(file_path_in, file_path_out, 32, 20)
        img_sample = create_data_sample_as_single(file_path_out, pixels)
        predicted = clf.predict(img_sample)[0]
        return predicted
    except Exception as e:
        print('Impossible to recognize flag because of error.')


def test_flag_of_poland_svg_recognized_as_flag_of_poland(clf, pixels):
    file_path_in = os.path.join('tests/data_for_tests', '140.svg')
    file_path_out = os.path.join('tests/data_for_tests', 'new.png')
    predicted = pipeline(clf, pixels, file_path_in, file_path_out)
    assert predicted == 140


def test_flag_of_poland_png_modified_recognized_as_flag_of_poland(clf, pixels):
    file_path_in = os.path.join('tests/data_for_tests', '140_3.png')
    file_path_out = os.path.join('tests/data_for_tests', 'new.png')
    predicted = pipeline(clf, pixels, file_path_in, file_path_out)
    assert predicted == 140


def test_flag_of_mongolia_recognized_as_flag_of_mongolia(clf, pixels):
    file_path_in = os.path.join('tests/data_for_tests', '116.png')
    file_path_out = os.path.join('tests/data_for_tests', 'new.png')
    predicted = pipeline(clf, pixels, file_path_in, file_path_out)
    assert predicted == 116


def test_non_existing_file_prints_message(clf, pixels, capsys):
    file_path_in = os.path.join('tests/data_for_tests', '440_3.png')
    file_path_out = os.path.join('tests/data_for_tests', 'new.png')
    predicted = pipeline(clf, pixels, file_path_in, file_path_out)
    out, err = capsys.readouterr()
    print(out, err)
    assert 'Impossible to recognize flag because of error.' in out


def test_existing_non_image_file_prints_message(clf, pixels, capsys):
    file_path_in = os.path.join('tests/data_for_tests', 'clf.pkl')
    file_path_out = os.path.join('tests/data_for_tests', 'new.png')
    predicted = pipeline(clf, pixels, file_path_in, file_path_out)
    out, err = capsys.readouterr()
    assert 'Impossible to recognize flag because of error.' in out


def test_load_countries_names_gives_correct_list():
    countries_list = load_countries_names(os.path.join('tests', 'data_for_tests', 'countries.txt'))
    assert len(countries_list) == 206
    assert countries_list[140] == 'Poland'


def test_load_countries_names_with_wrong_file_prints_message(capsys):
    countries_list = load_countries_names(os.path.join('tests', 'data_for_tests', 'countrieeees.txt'))
    out, err = capsys.readouterr()
    assert 'Missing file' in out
