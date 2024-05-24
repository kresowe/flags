import os
import pickle
from flags.image_preprocessor import preprocess_image, get_pixels_positions, create_data_sample_as_single
from flags.utils import load_countries_names, remove_file_if_exists


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
        pixels_positions, _, _ = get_pixels_positions(width=WIDTH, height=HEIGHT)
        countries_names = load_countries_names(file_path_countries_list)
        remove_file_if_exists(file_path_new_sample)

        file_path_in = input('Provide path to picture of a flag: ')
        print("Converting your picture...")
        preprocess_image(file_path_in, file_path_new_sample, WIDTH, HEIGHT)
        img_sample = create_data_sample_as_single(file_path_new_sample, pixels_positions)
        predicted = clf.predict(img_sample)[0]
        print(f"This is a flag of {countries_names[predicted]}.")
    except Exception as e:
        print(f'Impossible to recognize flag because of error. Error: {e}')
