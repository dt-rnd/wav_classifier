import argparse
import model
import numpy as np
import pyaudio
import struct
import math


def parse_args():
    """ Parses command line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Audio stream classifier')
    parser.add_argument('model')
    parser.add_argument('--labels', nargs=2)
    parser.add_argument('--frame_size', type=int, default=2000)
    parser.add_argument('--sample_rate', type=int, default=16000)
    args = parser.parse_args()
    return args


def main():
    """
    Loads an existing model, opens audio input stream, classifies input
    """
    args = parse_args()

    print('Audio stream classifier')

    print("Restoring model: ", args.model)
    mdl = model.restore(args.model)

    if mdl is None:
        print("Can't classify data without an existing model.")
        return

    print("Opening audio input..")

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paFloat32, channels=1,
                        rate=args.sample_rate, input=True,
                        frames_per_buffer=args.frame_size)

    label_a = label_b = ""

    if args.labels is not None:
        label_a = args.labels[0]
        label_b = args.labels[1]

    while True:
        # Peel off [frame_size] bytes from the audio stream
        stream_data = stream.read(args.frame_size)

        # Unpack the binary stream and expand
        data = struct.unpack("%df" % args.frame_size, stream_data)
        data = np.expand_dims([data], axis=2)

        avg = model.classify(mdl, data)

        steps = 20
        a = int(math.ceil(avg * steps))
        b = steps - a

        print(label_a + " [" + ("." * a) + "|" + ("." * b) + "] " + label_b + " - " + str(avg), end='\r')


if __name__ == "__main__":
    main()
