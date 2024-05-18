# Flags
## Goal 
The goal of this project is to create a machine learning model that can recognize the flags of the countries.

## Steps of the solution
### Creating dataset
#### Downloading data
Images of the flags of countries and the names of the countries are downloaded from https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags using `requests` and `BeautifulSoup4` libraries. 
The images are saved in the directory `data`. The names of the countries are saved in the same order to the file `countries.txt` in the same directory.
#### Preprocessing data
The downloaded images are in SVG form. 
Then, they are converted into PNGs using `cairosvg` library.
Next, each image is resized to smaller size, i.e., 32x20 pixels using `Pillow` library.
