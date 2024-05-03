# Base class

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y for an input X
    def forward_propogation(self, input_data):
        raise  NotImplementedError

    # computes the input_error dE/dX for a given output_error dE/dY

    def backward_propogation(self, output_error, learning_rate):
        raise NotImplementedError