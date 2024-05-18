import numpy as np
import requests
from bs4 import BeautifulSoup, element
import cairosvg
from PIL import Image, ImageEnhance
import os


class DatasetCreator:
    def __init__(self, page_url: str, img_url_start: str, path_data: str) -> None:
        self._page_url = page_url
        self._img_url_start = img_url_start
        self._path_data = path_data
        self._page_content = ''
        self._countries_numbers = []

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
        with open(os.path.join(self._path_data, f'{counter}.svg'), 'wb') as img_file_handler:
            img_file_handler.write(img_data)
            self._countries_numbers.append(counter)

    def preprocess_initial_images(self) -> None:
        self._convert_images_to_png()
        self._resize_images()

    def _convert_images_to_png(self) -> None:
        print("Converting images to PNG...")
        for country_number in self._countries_numbers:
            img_path_in = os.path.join(self._path_data, f'{country_number}.svg')
            img_path_png = os.path.join(self._path_data, f'{country_number}.png')
            cairosvg.svg2png(url=img_path_in, write_to=img_path_png)

    def _resize_images(self) -> None:
        print("Resizing images...")
        for country_number in self._countries_numbers:
            img_path_in = os.path.join(self._path_data, f'{country_number}.png')
            img_path_resized = os.path.join(self._path_data, f'{country_number}_0.png')
            with Image.open(img_path_in) as img:
                resized_img = img.resize((32, 20))
                resized_img.save(img_path_resized)

    def create_new_samples(self) -> None:
        print("Creating new samples...")
        # for country_number in range(206):
        for country_number in self._countries_numbers:
            img_path_in = os.path.join(self._path_data, f'{country_number}_0.png')
            counter = 1
            counter = self._create_by_modified_brightness(img_path_in, country_number, counter)
            counter = self._create_by_modified_brightness(img_path_in, country_number, counter)

    def _create_by_modified_brightness(self, img_path_in: str, country_number: int, counter: int) -> int:
        for factor in np.arange(0.8, 1.2, 0.02):
            img_path_out = os.path.join(self._path_data, f'{country_number}_{counter}.png')
            with Image.open(img_path_in) as img:
                enhancer = ImageEnhance.Brightness(img)
                enhanced_img = enhancer.enhance(factor)
                enhanced_img.save(img_path_out)
            counter += 1
        return counter

    def _create_by_modified_contrast(self, img_path_in: str, country_number: int, counter: int) -> int:
        for factor in np.arange(0.8, 1.2, 0.02):
            img_path_out = os.path.join(self._path_data, f'{country_number}_{counter}.png')
            with Image.open(img_path_in) as img:
                enhancer = ImageEnhance.Contrast(img)
                enhanced_img = enhancer.enhance(factor)
                enhanced_img.save(img_path_out)
            counter += 1
        return counter





