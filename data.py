import pandas
from sklearn.preprocessing import LabelEncoder

def load_labeled_data(filename):
    """ Loads data from a csv, where the last column is the label of the data in that row
    :param filename: name of the file to load
    :return: data frames and labels in separate arrays
    """
    dataframe = pandas.read_csv(filename, header=None)
    dataset = dataframe.values
    data = dataset[:, 0:-1].astype(float)
    labels = dataset[:, -1]
    return data, labels

def load_unlabeled_data(filename):
    """Loads raw unlabeled data as floats from a csv file using pandas"""
    dataframe = pandas.read_csv(filename, header=None)
    dataset = dataframe.values
    return dataset[:, 0:].astype(float)


def encode_labels(labels):
    """Encodes an array of labels, returns the transformed labels and a dict to lookup original label"""
    encoder = LabelEncoder()
    encoder.fit(labels)
    label_dict = [(x,int(encoder.transform([x])[0])) for x in encoder.classes_]
    return encoder.transform(labels), label_dict
