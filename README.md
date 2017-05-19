
#  Audio classifier

These scripts are designed to train a neural network to perform binary classification of audio samples, and subsequently use the trained network to perform classification of audio streamed through an input device, or over a socket.

### Requirements

The scripts were written in Python 3.5, package requirements for the script are contained in requirements.txt. Use `pip install -r requirements.txt` to install.

## Usage

These scripts are designed to operate on data prepared using our wav chunker, located here: [](github.com/rndakqa/wav_chunker)

### Training

To train the neural network, and save a trained model, you use the following command:

```
python main.py <input_data_file>
```

This will use default settings for all parameters (see main.py for all command line switches). The input data file must be in the format provided by the wav chunker, with two distinct categories of samples. The trained model can then be used in classification.

### Classifying

To classify (unseen) data, we use the following command:

```
python main.py <input_data_file> --predict --model <trained_model>
```

This will output the average classification, and standard deviation after passing the input data through the trained model supplied.

### Device input

A more common task is to classify data capture through an input device such as a microphone. To configure which device to use, you will have to modify the script itself. To start listening/classifying on the default input device, use the following:

```
python stream.py <trained_model>
```

If you have trained your model using non-default parameters, you can specify the additional model parameters on the command line.

### Server

To launch a server that will accept audio data streamed over an open socket, you can use:

```
python server.py <trained_model>
```

By default, the audio is assumed to be single channel, 16bit samples at 16kHz.