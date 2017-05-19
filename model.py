import os
import json
from keras.models import Sequential, load_model
from keras.layers import *


def create(input_dim):
    """ Create our neural net model based on the size of the input
    :param input_dim: the window size for the input, i.e. (batches, input_dim, 1) is our expected shape
    :return: the compiled model
    """
    model = Sequential()
    model.add(Conv1D(input_shape=(input_dim,1),filters=24, kernel_size=16, activation='relu', use_bias=True))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters=8, kernel_size=4, activation='relu', use_bias=True))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def classify(model, data):
    """ Classifies the data according to a trained model
    :param model: the trained model
    :param data: data to be classified/predicted
    :return: the average score
    """
    results = model.predict(data)
    avg = np.mean(results)
    return avg


def save(model, filename):
    """ Save the trained model, including weights
    :param model: the model to save
    :param filename: the file path to save the model
    """
    path, file = os.path.split(filename)
    if not os.path.exists(path):
        os.mkdir(path)
    model.save(filename, overwrite=True, include_optimizer=True)




def save_labels(labels, filename):
    with open(filename, 'w') as outfile:
        json.dump(labels, outfile)


def load_labels(filename):
    with open(filename) as data_file:
        return json.load(data_file)


def restore(filename):
    return load_model(filename)


def most_recent(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return max(files, key=os.path.getmtime)


def model_exists(directory):
    return len(os.listdir(directory)) > 0