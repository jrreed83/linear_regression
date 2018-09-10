from model.linear_regression import LinearRegression
from training.train1 import train as train1
from training.train2 import train as train2 
import numpy as np 
import matplotlib.pyplot as plt 

# Generate data
def generate_data(n_train, n_valid, n_test):

    w = np.array([1.0, 2.0, 3.0])
    b = 1.0

    X_train = np.random.randn(n_train, 3) 
    y_train = np.dot(X_train, w) + b 
    y_train += np.random.randn(n_train)

    X_valid = np.random.randn(n_valid, 3) 
    y_valid = np.dot(X_valid, w) + b 
    y_valid += np.random.randn(n_valid)

    X_test = np.random.randn(n_test, 3) 
    y_test = np.dot(X_test, w) + b  

    train = (X_train, y_train)
    valid = (X_valid, y_valid)
    test = (X_test, y_test)

    return train, valid, test


def main():

    # Generate Data Training Data
    training_data, valid_data, test_data = generate_data(1000, 100, 100) 

    X_train, y_train = training_data
    X_valid, y_valid = valid_data
    X_test, y_test = test_data

    # Instantiate Model
    model = LinearRegression(din = 3)

    # Run training algorithm
    history = train1( 
        model, 
        training_data = (X_train, y_train), 
        num_epochs = 100, 
        lr = 1e-3,
        validation_data=(X_valid, y_valid))

    print(model.layer.weight)
    print(model.layer.bias)
    plt.plot(history['training_loss'], 'bo-')
    plt.plot(history['validation_loss'], 'ro-')
    plt.show()
if __name__ == '__main__':
    main()