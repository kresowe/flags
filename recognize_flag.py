import os
import pickle
from image_preprocessor import resize, convert_to_png, convert_to_rgb, get_pixels_positions, create_data_sample_as_single


def load_countries(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"Missing file {file_path}")
    except Exception as e:
        print(f"Unexpected error: {e}.")


def remove_file_if_exists(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    WIDTH = 32
    HEIGHT = 20
    file_path = os.path.join('data_new', 'new_sample.png')

    print("Preparing environment...")
    clf = pickle.load(open(os.path.join('models', 'clf.pkl'), 'rb'))
    pixels_positions = get_pixels_positions(width=WIDTH, height=HEIGHT)
    countries = load_countries(os.path.join('data', 'countries.txt'))
    remove_file_if_exists(file_path)

    file_path_in = input('Provide path to picture of a flag: ')
    try:
        print("Converting your picture...")
        convert_to_png(file_path_in, file_path)
        resize(file_path, file_path, WIDTH, HEIGHT)
        convert_to_rgb(file_path, file_path)
        img_sample = create_data_sample_as_single(file_path, pixels_positions)
        predicted = clf.predict(img_sample)[0]
        print(f"This is a flag of {countries[predicted]}.")
    except Exception as e:
        print(f'Impossible to recognize flag because of error.')
