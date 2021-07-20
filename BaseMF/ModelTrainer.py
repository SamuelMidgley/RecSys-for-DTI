import torch


def train_model(train_loader, test_loader, model, num_epochs=100):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_func = torch.nn.MSELoss()

    model.to(dev)

    train_losses = []
    test_losses = []
    for epoch in range(0, num_epochs):
        count = 0
        cum_loss = 0.
        for i, (train_batch, label_batch) in enumerate(train_loader):
            count = 1 + i
            # Predict and calculate loss for user factor and bias
            optimizer = torch.optim.SGD([model.user_biases.weight, model.user_factors.weight,
                                         model.item_biases.weight, model.item_factors.weight],
                                        lr=0.05, weight_decay=1e-5)
            prediction = model(train_batch[:, 0].to(dev), train_batch[:, 1].to(dev))
            loss = loss_func(prediction, label_batch.to(dev)).float()
            loss_item = loss.item()
            cum_loss += loss_item

            # Backpropagate
            loss.backward()

            # Update the parameters
            optimizer.step()
            optimizer.zero_grad()

        train_loss = cum_loss / count
        train_losses.append(train_loss)

        cum_loss = 0.
        count = 0
        for i, (test_batch, label_batch) in enumerate(test_loader):
            count = 1 + i
            with torch.no_grad():
                prediction = model(test_batch[:, 0].to(dev), test_batch[:, 1].to(dev))
                loss = loss_func(prediction, label_batch.to(dev))
                cum_loss += loss.item()

        test_loss = cum_loss / count
        test_losses.append(test_loss)
        if epoch % 5 == 0:
            print('epoch: ', epoch, ' avg training loss: ', train_loss, ' avg test loss: ', test_loss)
    return train_losses, test_losses, model
