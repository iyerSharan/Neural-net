import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import  train_test_split

# Create dataset with 10,000 samples.
x, y = make_circles(n_samples=10000, noise=0.5, random_state=26)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.33, random_state=26)

# Visualize the data
fig, (train_ax, test_ax) = plt.subplots(ncols = 2, sharex = True, sharey = True, figsize = (10, 5))
train_ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train )
train_ax.set_title("Training Data")
train_ax.set_xlabel("Feature #0")
train_ax.set_ylabel("Feature #1")

test_ax.scatter(x_test[:, 0], x_test[:, 1], c = y_test)
test_ax.set_xlabel("Feature #0")
test_ax.set_title("Testing Data")
plt.show()

