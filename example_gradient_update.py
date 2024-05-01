# Taken from an online page
# refer: https://towardsdatascience.com/how-to-train-a-neural-network-from-scratch-952bbcdae729
import numpy as np

""" x is our input and y_true is our expected output """

x = np.array([[3,4,7], [6,2,9], [8,10,12], [9,7,4], [7,6,3]])
y_true = np.array([13.5, 15.6, 27.4, 15.6, 12.5])
lr = 0.001

""" the initial weight here is a guess and it will be updated """

w0 = np.array([20.0, -14.0, 346.0])

""" The weights will be updated 2000 times, each update we grab the gradient for each input, average all the gradients, 
    then update the weights, makes sure we move in the direction of -gradient)"""

for n in range(2000):
    gradients = [[],[],[],[],[]]
    for i in range(5):
        z = np.dot(w0, x[i])
        gradients[i] = -2 * (y_true[i]-z) * x[i]

    avg_grad = np.average(gradients,0)
    w0 -= lr * avg_grad
    print("Gradients:", gradients)
    print("Avg_grad:", avg_grad)

print(w0)