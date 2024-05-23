# Flags

## Goal 

The goal of this project is to create a machine learning model that can recognize the flags of the countries.

## How to run

Inference pipeline where you can identify the country by its flag is run by the terminal command:

```commandline
python3 src/recognize_flag.py 
```

When asked to provide path to picture you can use for example `tests/data_for_tests/116.png` or you can add your own picture to `data` and refer to it by `src/data/<your_file>`.

This does not require the steps listed below because it uses already trained and saved model.

The steps below illustrate how to create dataset and train model from scratch. 

In order to create dataset run:

```commandline
cd src
python3 get_dataset.py 
```

Note that this will download the pictures from  https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags and then generate new files. 

Running the command

```commandline
python3 ml_training_cross_val_manual.py 
```

performs a nested cross-validation on the dataset to find the best value of hyperparameter and estimate the model performance.

Then:

```commandline
python3 ml_training_best_model.py
```

trains the best model and estimates its performance and then trains the model on the whole available dataset.

### Docker

I created a Docker image from the inference pipeline. You can download it and use in the following way:
```commandline
docker pull kresowe/flags-mb:0.1
docker run -t -i kresowe/flags-mb:0.1
```

When asked to provide path to picture you can use for example `/flags-app/app/data/116.png`.


## Steps of the solution

### Creating dataset

#### Downloading data

Images of the flags of countries and the names of the countries are downloaded from https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags using `requests` and `BeautifulSoup4` libraries. 

The names of the countries are saved in the same order to the text file.

#### Preprocessing data

1. The downloaded images are in SVG form. Then, they are converted into PNGs using `cairosvg` library.
2. Next, each image is resized to smaller size, i.e., 32x20 pixels using `Pillow` library.
3. It turns out that some of the flags have 4 channels (RGBA) instead of 3 (RGB) as the majority. Therefore, those RGBA are converted to RGB using `Pillow` library. 

#### Producing new data samples

In order to recognize the flag of each country, a machine learning classifier should choose between about 200 classes. In the beginning there is only one picture per class which is significantly too few. To address this problem, I created new data samples.

For each country, 20 new data samples were created by modifying the brightness of a picture and 20 new samples by modyfying its contrast.

#### Creating numeric dataset from images

In this approach only 8 pixels are selected from each image. For details see "Solution idea" section. 

For each image sample the 24 feature values are taken, i.e., R, G, B of each of the 8 selected pixels. Target variable is a class label which is a true country number.

The created dataset in the form of `numpy` arrays is saved as .npz.

### Model selection, validation and training

The logistic regression was chosen as a machine learning model.

The accuracy was used as a metric. Namely, it is defined as the fraction of correct predictions in the randomly selected test / validation set. 

The nested cross-validation was used because of the relatively small amount of data w.r.t. number of classes. The model performance with different values of the regularization hyperparameter was assessed.

The model with the best performance was chosen. For this model a cross-validation was used to make a final estimation of its accuracy.

The best model was trained on the whole available dataset and then serialized for future use.

### Inference pipeline

Script `recognize_flag.py` uses the trained model. It enables user to provide a path to the image and predicts which country's flag is in the image.

## Solution idea

I realized that many of the flags consist of 2 or 3 horizontal or vertical rectangles, so it may be sufficient to extract only a small number of pixels because typically all the pixels in the rectangle have the same color.

The positions of the selected pictures and of the borders of 2 or 3 horizontal / vertical rectangles are shown in figure.

![alt text](https://github.com/kresowe/flags/blob/master/img/pixels.png?raw=true)

## Results

This simple method of selecting just 8 pixels and using logistic regression was surprisingly successful.

As seen from `ml_training_cross_val_manual.py`, very good accuracy (about 97-99%) is obtained for regularization hyperparameter C in range from 1 to 1000. Three out of four folds suggest that C = 100 gives the best accuracy. I note that for C = 1, the accuracy is similar with shorter training time however given the relatively small size of dataset these times are quite short even on my old notebook.

According to `ml_training_best_model.py`, the accuracy of the chosen model (logistic regression with C=100) is 99.3 +/- 0.1 %.

## Comments

I note that this specific problem can be solved without machine learning. The package `flagpy` [^1] identifies a flag by comparing it with a template flag for each country and chooses the one that is most similar to the given image. The similarity is calculated by the *distance function* defined, e.g., by mean square difference.   


[^1]: https://pypi.org/project/flagpy/ and https://github.com/saahilkumar/world-flag-identifier/