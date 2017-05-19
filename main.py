import argparse
import data
import model
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold


def parse_args():
    """ Parses command line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Audio classifier')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--most_recent', action='store_true')
    parser.add_argument('data_file')
    parser.add_argument('--model')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--logdir', default="./models/")
    parser.add_argument('--kfolds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()
    return args


def train(mdl, input_data, output, folds, epochs):
    """ Trains the model on input data
    :param mdl: compiled keras model
    :param input_data: data input tuple (data,labels)
    :param output: output file name for weights
    :param folds: number of folds to perform over the data (Stratified k-fold)
    :param epochs: number of epochs to run
    """
    print("Training params:")
    print("  epochs:", epochs)

    X, Y = input_data

    kfold = StratifiedKFold(n_splits=folds, shuffle=True)  # random_state=seed

    filepath = output + "_weights.hdf5"
    callbacks_list = []

    X = np.expand_dims(X, axis=2)

    # Iterate over kfold
    for train,test in kfold.split(X, Y):
        
        mdl.fit(X[train], Y[train], validation_data=(X[test],Y[test]), epochs=epochs, batch_size=10, verbose=1, callbacks=callbacks_list)
        model.save(mdl, filepath)


def classify(mdl, input):
    """ Classifies the supplied data, prints results
    :param mdl: trained model
    :param input: data input (without labels)
    """
    print("Classification result:")

    X = np.expand_dims(input, axis=2)
    print("  expanded data to shape: " + str(np.shape(X)))   

    results = mdl.predict(X)

    print("  average: " + str(np.mean(results)))
    print("  stddev: " + str(np.std(results)))


def main():
    args = parse_args()

    print('Audio classifier')
    
    output_prefix = os.path.splitext(args.data_file)[0]

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    # create / restore model
    mdl = None

    if args.model:
        print("Restoring model..")
        print("  input:", args.model)
        mdl = model.restore(args.model)
    elif model.model_exists(args.logdir) and not args.force:
        print("Restoring last created model..")
        file = model.most_recent(args.logdir)
        print("  input:", file)
        mdl = model.restore(file)

    if not args.predict:
        # training task
        print("Loading training data..")
        print("  input:", args.data_file)

        # load data from file
        X, Y = data.load_labeled_data(args.data_file)

        w, h = np.shape(X)
        print("  size: {}x{}".format(w, h))

        # encode labels
        Y, Dict = data.encode_labels(Y)

        print("  labels: ", Dict)
        # save labels with data
        # model.save_labels(Dict, args.logdir + output_prefix + "_labels.txt")

        if mdl is None:
            print("Creating new model..")
            mdl = model.create(h)

        train(mdl, (X, Y), args.logdir + output_prefix, args.kfolds, args.epochs)

    else:
        # prediction task
        print("Loading classification data..")
        print("  input:", args.data_file)

        X = data.load_unlabeled_data(args.data_file)

        w, h = np.shape(X)
        print("  size: {}x{}".format(w, h))

        if mdl is None:
            print("Can't classify without an existing model")
            return

        classify(mdl, X)


if __name__ == "__main__":
    main()
