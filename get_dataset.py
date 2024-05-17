import requests
from bs4 import BeautifulSoup
import os

PAGE_URL = "https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags"
IMG_URL_START = "https://upload.wikimedia.org/wikipedia/commons/"

response = requests.get(PAGE_URL)
if not response.ok:
    print("Page not found. Check your Internet connection.")
else:
    soup = BeautifulSoup(response.content, 'html.parser')
    elems = soup.find_all(class_='mw-file-description')
    counter = 0
    with open(os.path.join('data', 'countries.txt'), 'w') as txt_file_handler:
        for elem in elems:
            txt_file_handler.write(elem['title'] + '\n')
            children = [child for child in elem.children]
            img_tag = children[0]
            src = img_tag['src']
            pos_start = src.find('thumb/') + len('thumb/')
            pos_end = src.find('.svg') + len('.svg')
            img_name = src[pos_start:pos_end]
            img_url = IMG_URL_START + img_name

            img_data = requests.get(img_url).content
            if not response.ok:
                print(f'{counter}', response)

            with open(os.path.join('data', f'{counter}.svg'), 'wb') as img_file_handler:
                img_file_handler.write(img_data)

            counter += 1


