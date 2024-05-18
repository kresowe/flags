# Flags

## Goal 

The goal of this project is to create a machine learning model that can recognize the flags of the countries.

## Steps of the solution

### Creating dataset

#### Downloading data

Images of the flags of countries and the names of the countries are downloaded from https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags using `requests` and `BeautifulSoup4` libraries. 

The images are saved in the directory `data`. The names of the countries are saved in the same order to the file `countries.txt` in the same directory.

#### Preprocessing data

1. The downloaded images are in SVG form. 
2. Then, they are converted into PNGs using `cairosvg` library.
3. Next, each image is resized to smaller size, i.e., 32x20 pixels using `Pillow` library.

#### Producing new data samples

In order to recognize the flag of each country, a machine learning classifier should choose between about 200 classes. In the beginning there is only one picture per class which is significantly too few. To address this problem, I created new data samples.

For each country, 20 new data samples were created by modifying the brightness of a picture and 20 new samples by modyfying its contrast.

## Discussion

I note that any machine learning classifier is likely not the best approach to this specific problem. The package `flagpy` [^1] identifies a flag by comparing it with a template flag for each country and chooses the one that is most similar to the given image. The similarity is calculated by the *distance function* defined, e.g., by mean square difference.  


[^1]: https://pypi.org/project/flagpy/ and https://github.com/saahilkumar/world-flag-identifier/