import numpy as np


class NeuralNetwork:

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((9, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 - np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_in, training_outputs, training_iterations):

        for i in range(training_iterations):

            output = self.think(training_in)
            error = (training_outputs - output)
            adjustments = np.dot(training_in.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


if __name__ == '__main__':

    neural_network = NeuralNetwork()

    print('Random synaptic weights: ')
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [0, 1, 0, 1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 1, 0, 0],
                           [0, 1, 0, 1, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 1, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 1, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 1, 0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 1, 0],
                           [1, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 1, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 1, 0],
                           [0, 1, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 1, 0],
                           [1, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 1]])

    training_outputs = np.array([[0.4666666667],
                             [0.221417021],
                             [0.3400211193],
                             [0.2951737953],
                             [0.8],
                             [0.4333819006],
                             [0.344714353],
                             [0.5566065622],
                             [0.154512788],
                             [0.2181015506],
                             [0.6335877863],
                             [0.6112353],
                             [0.3574149722],
                             [0.4444444444],
                             [0.2911153119],
                             [0.1642522844],
                             [0.2857142857],
                             [0.1747172193],
                             [0.275862069],
                             [0.3333333333],
                             [0.2627815253],
                             [0.1558003013],
                             [0.2598496926],
                             [0.1793920365],
                             [0.55],
                             [0.55],
                             [0.1942487265],
                             [0.68],
                             [0.3462343096],
                             [0.4074074074],
                             [0.4159560799]])

    neural_network.train(training_inputs, training_outputs, 1000)

    print('Synaptic weights after training: ')
    print(neural_network.synaptic_weights)

    print('Input Trade Data')

    A = str(input('Craigslist: '))
    B = str(input('Offerup: '))
    C = str(input('Facebook MP: '))
    D = str(input('Microphone: '))
    E = str(input('Rackmount Gear: '))
    F = str(input('Guitar: '))
    G = str(input('Percussion: '))
    H = str(input('Keyboard: '))
    I = str(input('Guitar Pedal: '))

    print('Input Data: ', A, B, C, D, E, F, G, H, I)
    print('Output data: ')
    print(neural_network.think(np.array([A, B, C, D, E, F, G, H, I])))

