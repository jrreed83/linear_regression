import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import data.utils as utils 

def train(model, training_data, num_epochs = 10, lr = 1e-2, batch_size = 1, validation_data = None):

    model.double()

    # Function being minimized
    loss_fn = nn.MSELoss() 

    # Optimization algorithm being used to minimize the loss
    optimizer = optim.RMSprop(model.parameters(), lr = lr)

    # Build the dataset and get the dataloader for the training data
    X_train, y_train = training_data
    train_minibatches = utils.load_data(X_train, y_train, batch_size=10)

    # Validation data
    X_valid, y_valid = validation_data
    valid_minibatches = utils.load_data(X_valid, y_valid, batch_size = len(y_valid))

    history = {
        'training_loss': [],
        'validation_loss': []
    }

    # Main optimization loop
    for epoch in range(num_epochs):
        # Loop over all mini-batches
        batch_loss = []
        for inputs, targets in train_minibatches:

            # Compute the predicted outputs
            outputs = model(inputs)
            
            # Evaluate the difference between the known targets
            # and the predicted targets
            loss = loss_fn(outputs, targets)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add the loss for this mini-batch to the array of losses
            batch_loss.append(loss.item())

        # The loss for each epoch is the average loss observed for all mini-batches
        avg_loss = torch.tensor(batch_loss).mean().item()
        
        history['training_loss'].append(avg_loss)    
        # Evaluate on the validation data
        print(f'Epoch {epoch}: {avg_loss}')
    
        # Validation loss/error
        for x_valid, y_valid in valid_minibatches:
            print(x_valid.size())
            pred = model(x_valid)
            err = F.mse_loss(pred, y_valid)
            err = err.item()
            history['validation_loss'].append(err)

    return history
    # Evaluate the accuracy of the model on the validation data


