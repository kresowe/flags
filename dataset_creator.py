import numpy as np
import requests
from bs4 import BeautifulSoup, element
import os
from image_preprocessor import (convert_to_rgb, resize, convert_to_png, get_pixels_positions,
                                create_data_sample_from_image, change_brightness, change_contrast)


class DatasetCreator:
    """Class used to create a dataset given the URLs of website with data.
    page_url is expected to be url of Wikipedia gallery of flags,
    img_url_start is expected to be the beginning of the url where the image of flag is located,
    path_data is a path where to save the images of flags,
    width and height are the desired dimensions of the images.
    """
    def __init__(self, page_url: str, img_url_start: str, path_data: str, width: int = 32, height: int = 20) -> None:
        self._page_url = page_url
        self._img_url_start = img_url_start
        self._path_data = path_data
        self._page_content = ''
        self._countries_numbers = list(range(206))
        self._width = width
        self._height = height
        self._X = []
        self._y = []
        self._files_per_country = 0

    def download_dataset(self) -> None:
        """Downloads images of flags and saves the list of the countries into text file."""
        if not self._get_page_content():
            return
        soup = BeautifulSoup(self._page_content, 'html.parser')
        elements = soup.find_all(class_='mw-file-description')

        counter = 0
        with open(os.path.join(self._path_data, 'countries.txt'), 'w') as txt_file_handler:
            print('Downloading data...')
            for elem in elements:
                country_name = elem['title']
                txt_file_handler.write(country_name + '\n')

                img_url = self._get_image_url(elem)
                self._download_image(img_url, counter, country_name)

                counter += 1
        print('Finished downloading data.')

    def _get_page_content(self) -> bool:
        """Saves the content of self._page_url into self._page_content"""
        response = requests.get(self._page_url)

        if not response.ok:
            print(f"No connection to {self._page_url}. Check your Internet connection.")
            return False

        self._page_content = response.content
        return True

    def _get_image_url(self, elem: element.Tag) -> str:
        """Scraps, creates and returns url of the image."""
        img_tag = [child for child in elem.children][0]
        src = img_tag['src']
        pos_start = src.find('thumb/') + len('thumb/')
        pos_end = src.find('.svg') + len('.svg')
        img_name = src[pos_start:pos_end]
        img_url = self._img_url_start + img_name
        return img_url

    def _download_image(self, img_url: str, counter: int, country_name: str) -> None:
        """Downloads image from url and saves as <counter>.svg (the original files from Wikipedia are SVG)
        in the self._path_data.
        It also adds counter to self._countries_numbers as class label for this sample."""
        response = requests.get(img_url)

        if not response.ok:
            print(f'Failed to connect to image {counter} ({country_name})')
            return

        img_data = response.content
        self._countries_numbers = []
        with open(os.path.join(self._path_data, f'{counter}.svg'), 'wb') as img_file_handler:
            img_file_handler.write(img_data)
            self._countries_numbers.append(counter)

    def preprocess_initial_images(self) -> None:
        """Preprocesses images by converting them to PNG, resizing to (self._width, self._height)
        and converting to RGB."""
        self._convert_images_to_png()
        self._resize_images()
        self._convert_images_to_rgb_if_needed()

    def _convert_images_to_png(self) -> None:
        """Converts all the downloaded images to PNG (the images downloaded from Wikipedia are in SVG).
        It saves each converted image as <country_number>.png in self._path_data."""
        print("Converting images to PNG...")
        for country_number in self._countries_numbers:
            img_path_in = os.path.join(self._path_data, f'{country_number}.svg')
            img_path_png = os.path.join(self._path_data, f'{country_number}.png')
            convert_to_png(img_path_in, img_path_png)

    def _resize_images(self) -> None:
        """Resizes all the images to (self._width, self._height). It is assumed that the images are already PNG.
        The resized image is saved as <country_number>_0.png in self._path_data"""
        print("Resizing images...")
        for country_number in self._countries_numbers:
            img_path_in = os.path.join(self._path_data, f'{country_number}.png')
            img_path_resized = os.path.join(self._path_data, f'{country_number}_0.png')
            resize(img_path_in, img_path_resized, width=self._width, height=self._height)

    def create_new_samples(self) -> None:
        """Creates new samples by modifying brightness and contrast of each downloaded and preprocessed image.
        New samples for country identified by country_number are saved as <country_number>_<counter>.png in
        self._path_data."""
        print("Creating new samples...")
        for country_number in self._countries_numbers:
            img_path_in = os.path.join(self._path_data, f'{country_number}_0.png')
            counter = 1
            counter = self._create_by_modified_brightness(img_path_in, country_number, counter)
            counter = self._create_by_modified_contrast(img_path_in, country_number, counter)
        self._files_per_country = counter

    def _create_by_modified_brightness(self, img_path_in: str, country_number: int, counter: int) -> int:
        """Creates new samples by modifying brightness of each downloaded and preprocessed image.
        For each image there are plenty of new samples because the brightness is modified by different factors.
        New samples for country identified by country_number are saved as <country_number>_<counter>.png in
        self._path_data, where <counter> starts from the 'counter' argument.
        Returns counter for naming the next sample to be created.
        """
        min_factor, max_factor, factor_step = 0.8, 1.2, 0.02
        for factor in np.arange(min_factor, max_factor, factor_step):
            img_path_out = os.path.join(self._path_data, f'{country_number}_{counter}.png')
            change_brightness(img_path_in, img_path_out, factor)
            counter += 1
        return counter

    def _create_by_modified_contrast(self, img_path_in: str, country_number: int, counter: int) -> int:
        """Creates new samples by modifying contrast of each downloaded and preprocessed image.
        For each image there are plenty of new samples because the contrast is modified by different factors.
        New samples for country identified by country_number are saved as <country_number>_<counter>.png in
        self._path_data, where <counter> starts from the 'counter' argument.
        Returns counter for naming the next sample to be created.
        """
        min_factor, max_factor, factor_step = 0.8, 1.2, 0.02
        for factor in np.arange(min_factor, max_factor, factor_step):
            img_path_out = os.path.join(self._path_data, f'{country_number}_{counter}.png')
            change_contrast(img_path_in, img_path_out, factor)
            counter += 1
        return counter

    def create_np_dataset_from_selected_pixels(self) -> None:
        """Creates dataset in the form of numpy arrays from the images.
        From each image it gets (R,G,B) of specific pixels that are determined by get_pixels_positions.
        It writes R, G, B of each of these pixels as features in self._X and country numbers as class labels in self._y.
        """
        print('Creating np dataset from selected pixels...')
        pixels_positions, _, _ = get_pixels_positions(width=self._width, height=self._height)
        img_samples = tuple()
        y = []
        for file_name in self._get_file_list():
            try:
                img_sample = create_data_sample_from_image(os.path.join(self._path_data, file_name), pixels_positions)
                img_samples += (img_sample, )
                y.append(self._get_country_number_from_file_name(file_name))
            except FileNotFoundError:
                print(f'Missing file {file_name}.')
            except Exception as e:
                print(f'Unexpected error: {e}')
        self._X = np.stack(img_samples, axis=0)
        self._y = np.array(y)

    def _get_file_list(self) -> list[str]:
        """Returns list of image files that are used as data samples."""
        return [f'{country_number}_{i}.png' for country_number in self._countries_numbers
                for i in range(self._files_per_country)]

    @staticmethod
    def _get_country_number_from_file_name(file_name: str) -> int:
        """It extracts country number from file name."""
        return int(file_name[:file_name.find('_')])

    def _convert_images_to_rgb_if_needed(self) -> None:
        """For each image it converts it to RGB in case it is RGBA. This ensures that it has 3 channels and not 4.
        Images should be already converted to PNG."""
        print("Converting images to RGB in case they are in RGBA...")
        for country_number in self._countries_numbers:
            img_path_in = os.path.join(self._path_data, f'{country_number}_0.png')
            img_path_out = os.path.join(self._path_data, f'{country_number}_0.png')
            convert_to_rgb(img_path_in, img_path_out)

    def save_compressed_dataset(self, file_name: str) -> None:
        """Saves dataset self._X, self._y as <file_name>, where file_name will be .npz"""
        print("Saving compressed dataset...")
        try:
            np.savez_compressed(os.path.join(self._path_data, file_name),
                                X=self._X, y=self._y)
            print(f"Dataset saved to {file_name}.")
        except Exception as e:
            print(f'Unexpected error: {e}')

