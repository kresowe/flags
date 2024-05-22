import numpy as np
from numpy.typing import NDArray
import requests
from bs4 import BeautifulSoup, element
import cairosvg
from PIL import Image, ImageEnhance
from matplotlib.image import imread
import os
import math
from image_preprocessor import convert_to_rgb, resize, convert_to_png, get_pixels_positions, create_data_sample_from_image


class DatasetCreator:
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

    def download_dataset(self) -> None:
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
        response = requests.get(self._page_url)

        if not response.ok:
            print(f"No connection to {self._page_url}. Check your Internet connection.")
            return False

        self._page_content = response.content
        return True

    def _get_image_url(self, elem: element.Tag) -> str:
        img_tag = [child for child in elem.children][0]
        src = img_tag['src']
        pos_start = src.find('thumb/') + len('thumb/')
        pos_end = src.find('.svg') + len('.svg')
        img_name = src[pos_start:pos_end]
        img_url = self._img_url_start + img_name
        return img_url

    def _download_image(self, img_url: str, counter: int, country_name: str) -> None:
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
        self._convert_images_to_png()
        self._resize_images()
        self._convert_images_to_rgb_if_needed()

    def _convert_images_to_png(self) -> None:
        print("Converting images to PNG...")
        for country_number in self._countries_numbers:
            img_path_in = os.path.join(self._path_data, f'{country_number}.svg')
            img_path_png = os.path.join(self._path_data, f'{country_number}.png')
            convert_to_png(img_path_in, img_path_png)
            # try:
            #     cairosvg.svg2png(url=img_path_in, write_to=img_path_png)
            # except FileNotFoundError:
            #     print(f'Missing file {img_path_in}')
            # except Exception as e:
            #     print(f'Unexpected error: {e}')

    def _resize_images(self) -> None:
        print("Resizing images...")
        for country_number in self._countries_numbers:
            img_path_in = os.path.join(self._path_data, f'{country_number}.png')
            img_path_resized = os.path.join(self._path_data, f'{country_number}_0.png')
            resize(img_path_in, img_path_resized, width=self._width, height=self._height)
            # try:
            #     with Image.open(img_path_in) as img:
            #         resized_img = img.resize((self._width, self._height))
            #         resized_img.save(img_path_resized)
            # except FileNotFoundError:
            #     print(f'Missing file {img_path_in}')
            # except Exception as e:
            #     print(f'Unexpected error: {e}')

    def create_new_samples(self) -> None:
        print("Creating new samples...")
        # for country_number in range(206):
        for country_number in self._countries_numbers:
            img_path_in = os.path.join(self._path_data, f'{country_number}_0.png')
            counter = 1
            counter = self._create_by_modified_brightness(img_path_in, country_number, counter)
            counter = self._create_by_modified_contrast(img_path_in, country_number, counter)

    def _create_by_modified_brightness(self, img_path_in: str, country_number: int, counter: int) -> int:
        for factor in np.arange(0.8, 1.2, 0.02):
            img_path_out = os.path.join(self._path_data, f'{country_number}_{counter}.png')
            try:
                with Image.open(img_path_in) as img:
                    enhancer = ImageEnhance.Brightness(img)
                    enhanced_img = enhancer.enhance(factor)
                    enhanced_img.save(img_path_out)
            except FileNotFoundError:
                print(f'Missing file {img_path_in}')
            except Exception as e:
                print(f'Unexpected error: {e}')
            counter += 1
        return counter

    def _create_by_modified_contrast(self, img_path_in: str, country_number: int, counter: int) -> int:
        for factor in np.arange(0.8, 1.2, 0.02):
            img_path_out = os.path.join(self._path_data, f'{country_number}_{counter}.png')
            try:
                with Image.open(img_path_in) as img:
                    enhancer = ImageEnhance.Contrast(img)
                    enhanced_img = enhancer.enhance(factor)
                    enhanced_img.save(img_path_out)
            except FileNotFoundError:
                print(f'Missing file {img_path_in}')
            except Exception as e:
                print(f'Unexpected error: {e}')
            counter += 1
        return counter

    def create_np_dataset_from_selected_pixels(self) -> None:
        print('Creating np dataset from selected pixels...')
        pixels_positions, _, _ = get_pixels_positions(width=self._width, height=self._height)
        img_samples = tuple()
        y = []
        for file_name in self._get_file_list():
            try:
                img_sample = create_data_sample_from_image(os.path.join(self._path_data, file_name), pixels_positions)
                # img = imread(os.path.join(self._path_data, file_name))
                # img_sample = img[pixels_positions[0][0], pixels_positions[0][1]]
                # for i in range(1, len(pixels_positions)):
                #     img_sample = np.concatenate((img_sample, img[pixels_positions[i][0], pixels_positions[i][1]]))
                img_samples += (img_sample, )
                y.append(self._get_country_from_file_name(file_name))
            except FileNotFoundError:
                print(f'Missing file {file_name}.')
            except Exception as e:
                print(f'Unexpected error: {e}')
        self._X = np.stack(img_samples, axis=0)
        self._y = np.array(y)

    # def _get_pixels_positions(self) -> tuple:
    #     width_limits = self._width * np.array([1 / 3, 1 / 2, 2 / 3])
    #     height_limits = self._height * np.array([1 / 3, 1 / 2, 2 / 3])
    #     widths = [math.floor((0 + width_limits[0]) / 2),
    #               math.floor((width_limits[0] + width_limits[1]) / 2),
    #               math.floor((width_limits[1] + width_limits[2]) / 2),
    #               math.floor((width_limits[2] + self._width) / 2)
    #               ]
    #     heights = [math.floor((0 + height_limits[0]) / 2),
    #                math.floor((height_limits[0] + height_limits[1]) / 2),
    #                math.floor((height_limits[1] + height_limits[2]) / 2),
    #                math.floor((height_limits[2] + self._height) / 2)
    #                ]
    #
    #     pixels_positions = ([heights[0], widths[0]],
    #                         [heights[0], widths[3]],
    #                         [heights[3], widths[0]],
    #                         [heights[3], widths[3]],
    #                         [heights[1], widths[1]],
    #                         [heights[1], widths[2]],
    #                         [heights[2], widths[1]],
    #                         [heights[2], widths[2]],
    #                         )
    #     return pixels_positions

    def _get_file_list(self) -> list[str]:
        return [f'{country_number}_{i}.png' for country_number in self._countries_numbers for i in range(41)]

    @staticmethod
    def _get_country_from_file_name(file_name: str) -> int:
        return int(file_name[:file_name.find('_')])

    def _convert_images_to_rgb_if_needed(self) -> None:
        print("Converting images to RGB in case they are in RGBA...")
        for country_number in self._countries_numbers:
            img_path_in = os.path.join(self._path_data, f'{country_number}_0.png')
            img_path_out = os.path.join(self._path_data, f'{country_number}_0.png')
            convert_to_rgb(img_path_in, img_path_out)
            # try:
            #     img = imread(img_path_in)
            #     if img.shape[2] > 3:
            #         with Image.open(img_path_in) as img_pil:
            #             img_out = img_pil.convert('RGB')
            #             img_out.save(img_path_out)
            # except FileNotFoundError:
            #     print(f'Missing file {img_path_in}')
            # except Exception as e:
            #     print(f'Unexpected error: {e}')

    def save_compressed_dataset(self):
        print("Saving compressed dataset...")
        file_name = 'np_from_selected_compressed.npz'
        print(self._X.shape, self._y.shape)
        try:
            np.savez_compressed(os.path.join(self._path_data, file_name),
                                X=self._X, y=self._y)
            print(f"Dataset saved to {file_name}.")
        except Exception as e:
            print(f'Unexpected error: {e}')

