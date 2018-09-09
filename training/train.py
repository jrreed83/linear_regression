import torch.nn as nn 
import torch.optim as optim 
import data.utils as utils 

def train(model, train_data, valid_data, num_epochs=10):

    # Function being minimized
    loss_fn = nn.MSELoss() 

    # Optimization algorithm being used to minimize the loss
    optimizer = optim.RMSprop(model.parameters())

    # Build the dataset and get the dataloader for the training data
    training_data = utils.load_data(X_train, y_train, batch_size=batch_size)
    
    # Main optimization loop
    for epoch in range(num_epochs):

        # Loop over all mini-batches
        for inputs, targets in training_data:

            # Compute the predicted outputs
            outputs = model(inputs)

            # Evaluate the difference between the known targets
            # and the predicted targets
            loss = loss_fn(outputs, targets)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate the accuracy of the model on the validation data
    model()


