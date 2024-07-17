from neural_net import train, predict, get_accuracy_value

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
sns.set_style("whitegrid")

# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.25

NN_ARCHITECTURE = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)

# the function making up the graph of a dataset
def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style("whitegrid")
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 1, cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
    if(file_name):
        plt.savefig(file_name)
        plt.close()

# make_plot(X, y, "Dataset", 'dataset.png')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

print(X_train.shape, X_test.shape)

params_values, cost_hist, accuracy_hist = train(X_train.T, y_train.reshape((y_train.shape[0], 1)).T, NN_ARCHITECTURE, 10000, 0.01)

Y_test_hat = predict(X_test.T, params_values, NN_ARCHITECTURE)

accuracy = get_accuracy_value(Y_test_hat, y_test.reshape((y_test.shape[0], 1)).T)
print("Test set accuracy: {:.2f}" .format(accuracy))

make_plot(X_test, Y_test_hat, "Dataset_predicted", 'dataset_pred.png')
make_plot(X_test, y_test, "Dataset_truth", 'dataset_truth.png')