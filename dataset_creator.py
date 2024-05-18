import requests
from bs4 import BeautifulSoup, element
import os


class DatasetCreator:
    def __init__(self, page_url: str, img_url_start: str) -> None:
        self._page_url = page_url
        self._img_url_start = img_url_start
        self._page_content = ''

    def download_dataset(self) -> None:
        if not self._get_page_content():
            return
        soup = BeautifulSoup(self._page_content, 'html.parser')
        elements = soup.find_all(class_='mw-file-description')

        counter = 0
        with open(os.path.join('data', 'countries.txt'), 'w') as txt_file_handler:
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

    @staticmethod
    def _download_image(img_url: str, counter: int, country_name: str) -> None:
        response = requests.get(img_url)

        if not response.ok:
            print(f'Failed to connect to image {counter} ({country_name})')
            return

        img_data = response.content
        with open(os.path.join('data', f'{counter}.svg'), 'wb') as img_file_handler:
            img_file_handler.write(img_data)
