from model.linear_regression import LinearRegression
from training.train import train 
import numpy as np 
import matplotlib.pyplot as plt 

# Generate data
def generate_data(n_samples = 1000):
    w = np.array([1.0, 2.0, 3.0])
    b = 1.0
    X = 100*np.random.randn(n_samples, 3) 
    y = np.dot(X, w) + b 
    return X, y


def main():

    # Generate Data Training Data
    X_train, y_train = generate_data(1000) 

    # Generate Validation Data and normalize
    X_valid, y_valid = generate_data(100)

    # Instantiate Model
    model = LinearRegression(din = 3)

    # Run training algorithm
    history = train( 
        model, 
        training_data = (X_train, y_train), 
        num_epochs = 100, 
        lr = 1e-3,
        validation_data=(X_valid, y_valid))

    print(model.layer.weight)
    print(model.layer.bias)
    plt.plot(history['training_loss'])
    plt.plot(history['validation_loss'])
    plt.show()
if __name__ == '__main__':
    main()