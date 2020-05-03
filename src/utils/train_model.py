import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.utils.save_load_model import save_parameters


def l1_loss(model, images):
    loss = 0
    model_children = list(model.children())
    values = images

    for i in range(len(model_children)):
        values = F.leaky_relu((model_children[i](values)))
        loss += th.mean(th.abs(values))
    return loss


def train_step(model, optimizer, loss_fn, out, images, reg_param):
    """
    :param model:
    :param reg_param:
    :param optimizer: Optimizer
    :param loss_fn: Function type
    :param out: reconstructed images
    :param images: true images
    :return: loss, pixel-wise difference between input & output image
    """
    optimizer.zero_grad()
    bce_loss = loss_fn(out, images)
    sparsity_loss = l1_loss(model, images)
    loss = bce_loss + reg_param * sparsity_loss
    loss.backward()
    optimizer.step()
    return [loss, bce_loss, sparsity_loss]


def training(model, train_loader, validation_loader, num_epochs, hyperps, directories, to_print=1, save_model=True):
    """
    :param model: AutoEncoder class
    :param train_loader: Torch DataLoader object
    :param validation_loader:
    :param num_epochs: max number of epochs to train
    :param hyperps: dictionary of hyper-parameters
    :param directories:
    :param to_print: int (0, 1, 2)
    :param save_model: ok
    :return: k
    """
    th.manual_seed(0)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=hyperps['lr'],
                           weight_decay=1e-4)

    outputs = []
    train_losses = []
    val_losses = []
    best_loss = float("inf")

    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader):
            img, _ = data
            recon = model(img)
            loss = train_step(model, optimizer, criterion, recon, img, hyperps["l1"])
            train_losses.append(loss[1])

            if to_print > 1:
                print("Epoch: {}, iteration: {}, iter_loss: {:.3f}".format(epoch, idx, float(loss[0])))

        if to_print:
            print('Epoch:{}, Train loss:{:.4f}'.format(epoch + 1, float(loss[0])))
        outputs.append((epoch, img, recon), )

        validation_loss = 0
        for data in validation_loader:
            img, _ = data
            validation_loss += criterion(model(img), img)
        val_losses.append(validation_loss)
        print('     Validation loss:{:.4f}'.format(validation_loss))

        if validation_loss <= best_loss:
            best_loss = validation_loss

            if epoch == num_epochs-1:
                if save_model:
                    save_parameters(model, directories['MODEL_DIR'])

    return outputs, [train_losses, val_losses]
