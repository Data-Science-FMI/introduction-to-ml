import matplotlib.pyplot as plt

def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %d' % y[idx])
    plt.show()

def plot_error(model):
    plt.plot(range(len(model.error_)), model.error_)
    plt.ylim([0, 500])
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.show()

def plot_digit_dist(X, y, idx, model):
    plt.bar(range(model.n_classes), model.predict_proba(X[idx:idx + 1])[0])
    plt.xticks(range(model.n_classes), range(model.n_classes))
    plt.title('true label: %d' % y[idx:idx + 1])
    plt.show()
