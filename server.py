import struct
import numpy as np
import pyaudio
import argparse
import model
import math
from twisted.internet import reactor, protocol, endpoints
from twisted.protocols import basic


def parse_args():
    """ Parses command line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Audio socket classifier')
    parser.add_argument('model')
    parser.add_argument('--stream_out', action="store_true")
    parser.add_argument('--frame_size', type=int, default=2000)
    parser.add_argument('--sample_rate', type=int, default=16000)
    args = parser.parse_args()
    return args


def open_audio_output(sample_rate, frame_size):
    """ Opens an optional output stream to provide debug output of audio data received
    :param sample_rate: sample rate of the output audio
    :param frame_size: frames per buffer
    :return: the open audio output stream
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paFloat32, channels=1,
                        rate=sample_rate, output=True,
                        frames_per_buffer=frame_size)
    return stream


def process_data(data, frame_size, mdl, stream):
    """ Processes a chunk of data from the stream, returns the average classification
        Optionally outputs to the debug stream if provided
    :param data: the data to classify
    :param frame_size: the size of the data (in audio samples)
    :param mdl: the previously trained model
    :param stream: optional output stream for debugging
    :return: the classification
    """
    unpacked = list(struct.unpack("%dh" % frame_size, data))  # Unpack the first 2000 16bit samples

    inverse_short_max = 3.0517578125e-5
    data = [x * inverse_short_max for x in unpacked]

    if stream is not None:
        packed = struct.pack("%df" % frame_size, *data)
        stream.write(packed)

    data = np.expand_dims([data], axis=2)

    average = model.classify(mdl, data)

    return average



class VKServer(basic.LineReceiver):
    
    def __init__(self, factory):
        self.factory = factory
        self.data = bytearray()


    def connectionMade(self):
        self.factory.clients.add(self)
        print("client connected")
        print("clients are ", self.factory.clients)


    def connectionLost(self, reason):
        self.factory.clients.remove(self)
        print("client disconnected")


    def dataReceived(self, data):
        self.data.extend(data)
        single_buffer_size = (self.factory.frame_size * 2)

        while len(self.data) >= single_buffer_size :

            val = process_data(self.data[:single_buffer_size], self.factory.model, self.factory.stream)

            if val is not None:
                response = struct.pack("f", val)
                steps = 20
                a = int(math.ceil(val * steps))
                b = steps - a

                print("[" + ("." * a) + "|" + ("." * b) + "] - " + str(val))

                for c in self.factory.clients:
                    c.transport.write(response)

            self.data = self.data[single_buffer_size:]


class PubFactory(protocol.Factory):

    def __init__(self, model, args):
        self.clients = set()
        self.model = model
        self.stream = None
        self.frame_size = args.frame_size
        if args.stream_out:
            self.stream = open_audio_output(args.sample_rate, args.frame_size)

    def buildProtocol(self, addr):
        return VKServer(self)


def main():
    args = parse_args()

    print('Audio socket classifier')

    print("Restoring model: ", args.model)
    mdl = model.restore(args.model)
   
    if mdl is None:
        print("Can't classify without an existing model")
        return

    endpoints.serverFromString(reactor, "tcp:80").listen(PubFactory(mdl, args))
    reactor.run()


if __name__ == "__main__":
    main()
