from dataset_creator import DatasetCreator

if __name__ == '__main__':
    PAGE_URL = "https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags"
    IMG_URL_START = "https://upload.wikimedia.org/wikipedia/commons/"
    PATH_DATA = 'data'

    set_creator = DatasetCreator(page_url=PAGE_URL, img_url_start=IMG_URL_START, path_data=PATH_DATA)
    set_creator.download_dataset()
    set_creator.preprocess_initial_images()
    set_creator.create_new_samples()
