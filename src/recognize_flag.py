import os
import pickle
from typing import Union
from image_preprocessor import (resize, convert_to_png, convert_to_rgb, get_pixels_positions,
                                create_data_sample_as_single)


def load_countries_names(file_path: Union[str, os.PathLike]) -> list:
    """Load countries names from file file_path into list that is returning, keeping the order.
    It is assumed that each line contains one country name."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"Missing file {file_path}")
    except Exception as e:
        print(f"Unexpected error: {e}.")


def remove_file_if_exists(file_path: Union[str, os.PathLike]) -> None:
    """Removes file file_path if it existed."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f'Error: {e}')


def preprocess_image(file_path_in: Union[str, os.PathLike], file_path_out: Union[str, os.PathLike],
                     width: int, height: int) -> None:
    """Preprocess image file_path_in provided by user and saves it to file_path_out.
    Namely, it converts it to PNG file, then resizes to (width, height) dimensions (in pixels) and converts to RGB."""
    convert_to_png(file_path_in, file_path_out)
    resize(file_path_out, file_path_out, width, height)
    convert_to_rgb(file_path_out, file_path_out)


if __name__ == '__main__':
    WIDTH = 32
    HEIGHT = 20
    DIRNAME = os.path.dirname(__file__)
    file_path_new_sample = os.path.join(DIRNAME, 'data_new', 'new_sample.png')
    file_path_clf = os.path.join(DIRNAME, 'models', 'clf.pkl')
    file_path_countries_list = os.path.join(DIRNAME, 'data', 'countries.txt')

    print("Preparing environment...")
    try:
        clf = pickle.load(open(file_path_clf, 'rb'))
    except Exception as e:
        print(f"Error {e}")
    pixels_positions, _, _ = get_pixels_positions(width=WIDTH, height=HEIGHT)
    countries_names = load_countries_names(file_path_countries_list)
    remove_file_if_exists(file_path_new_sample)

    file_path_in = input('Provide path to picture of a flag: ')
    try:
        print("Converting your picture...")
        preprocess_image(file_path_in, file_path_new_sample, WIDTH, HEIGHT)
        img_sample = create_data_sample_as_single(file_path_new_sample, pixels_positions)
        predicted = clf.predict(img_sample)[0]
        print(f"This is a flag of {countries_names[predicted]}.")
    except Exception as e:
        print('Impossible to recognize flag because of error.')
