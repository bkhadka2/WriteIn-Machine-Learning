# DataFrame handling
import pandas as pd

# Confusion matrix function
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

# keras Models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras import models, layers, datasets
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop,Adam


# Split data with stratified cv
from sklearn.model_selection import StratifiedKFold, train_test_split

# Encoding of classifications
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical, normalize
print('Tensorflow imported')

def read_data():
    df = pd.read_csv('./handwritten_data_785.csv')
    print(df.shape)
    print(df.head())

def createX_createY():
    df = read_data()
    X_train = df.drop(labels=['0'], axis=1)
    y_train = df['0']
    X_train = X_train / 255.0 # Normalizing the data
    y_train = to_categorical(y_train, num_classes = 26)
    return [X_train, y_train]

def splitTrain_Test():
    X, y = createX_createY()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=2)
    return [X_train, X_val, y_train, y_val]

def create_model():
    model = Sequential()
    model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                    activation ='relu', input_shape = (28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation = "softmax"))
    return model

def fit_model():
    X_train, X_val, y_train, y_val = splitTrain_Test()
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model = create_model()
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    epochs = 10 
    batch_size = 250 
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,y_val), steps_per_epoch=X_train.shape[0]
    

if __name__ == '__main__':
    read_data()
    createX_createY()
    splitTrain_Test()
    create_model()
    fit_model()






