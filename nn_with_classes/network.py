class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layers to your network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict the output for a given output (could be used for test)
    def predict(self, input_data):
        # sample the dimensions first
        samples = len(input_data)
        result = []

        # run the network over all samples
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propogation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epoch, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epoch):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propogation(output)

                # compute loss (just for display)
                err += self.loss(y_train[j], output)

                # backward propogation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propogation(error, learning_rate)

            # calculate the average on all samples
            err /= samples
            print("epoch %d/%d  error=%f" % (i+1, epoch, err))


